// Windows' own decoders (Media Foundation's, on the GPU through Direct3D 11
// video) behind gpu_frames.h's C API, for AMD's GPUs: the hand-over only
// (Params.handover, amf_handover.h, as amf_frames.cpp's), pictures not
// scaled. gpu_frames.py opens this one first and AMF's (amf_frames.cpp) where
// it cannot.
//
// The same decoder hardware as AMF's, with less in the way: on a Radeon 780M
// it decoded two 4K HEVC streams at 128 and 149 frames a second together
// where AMF's took 110-120, with a quarter of the CPU (0.22 cores where AMF's
// took 0.83 at 90 frames a second each), and VMAF v1 with it ran 63.3 -> 71.8
// frames a second at 4K, warm. It also decodes AV1 that AMF's failed on.
//
// One thread drives the decoder (Media Foundation's synchronous decoder MFTs
// are not to be called from two): nvf_push gives it a packet and takes every
// picture it has ready, nvf_finish drains it. A picture taken is held in a
// slot (its sample kept from the decoder's pool) and queued for nvf_pop once
// it is known to be decoded: the fence reached after a 2 x 2 copy of it,
// which Direct3D 11 makes wait for the decoder. Its pictures are shareable
// textures (MF_SA_D3D11_SHARED_WITHOUT_MUTEX) Vulkan imports, but H.264's are
// layers of a texture array, which the Radeon 780M's Vulkan driver failed to
// view a layer of (an access violation in vkCreateImageView or after): each
// of those is copied by Direct3D 11 into its slot's own texture instead.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "amf_handover.h"
#include "gpu_frames.h"

namespace {

std::mutex g_mf_lock;
bool g_mf_started = false;

bool start_mf(std::string &error) {
    std::lock_guard<std::mutex> lock(g_mf_lock);
    if (g_mf_started) return true;
    if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) {
        error = "Media Foundation could not start";
        return false;
    }
    g_mf_started = true;
    return true;
}

const GUID *mf_codec(int codec) {
    switch (codec) {
    case CODEC_H264: return &MFVideoFormat_H264;
    case CODEC_HEVC: return &MFVideoFormat_HEVC;
    case CODEC_AV1: return &MFVideoFormat_AV1;
    default: return nullptr;
    }
}

std::string hr_text(const char *what, HRESULT hr) {
    char text[160];
    snprintf(text, sizeof text, "%s (0x%08lx)", what, static_cast<unsigned long>(hr));
    return text;
}

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

// A Direct3D 11 video device on the first AMD GPU (amf_frames.cpp's).
ID3D11Device *video_device(std::string &error) {
    IDXGIFactory1 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory)))) {
        error = "DirectX could not list the GPUs";
        return nullptr;
    }
    ID3D11Device *device = nullptr;
    IDXGIAdapter1 *adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; index++) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002) {
            D3D_FEATURE_LEVEL level;
            ID3D11DeviceContext *context = nullptr;
            HRESULT hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                                           nullptr, 0, D3D11_SDK_VERSION, &device, &level, &context);
            if (context) context->Release();
            if (FAILED(hr)) device = nullptr;
        }
        adapter->Release();
        if (device) break;
    }
    factory->Release();
    if (!device) error = "no AMD GPU with a Direct3D 11 video device";
    return device;
}

struct Ready {
    int slot;
    long long pts;
};

struct Decoder {
    Params params{};
    Info info{};
    std::vector<unsigned char> extradata;  // AV1's sequence header: before the first packet
    bool first_packet = true;

    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *immediate = nullptr;
    ID3D10Multithread *multithread = nullptr;
    IMFDXGIDeviceManager *manager = nullptr;
    UINT token = 0;
    IMFTransform *mft = nullptr;
    bool streaming = false;

    // The hand-over's (as amf_frames.cpp's): the textures Slots::make imports,
    // the fence a picture's 2 x 2 copy into `touch` is signalled with.
    ID3D11Texture2D *shared[2] = {nullptr, nullptr};
    ID3D11DeviceContext4 *signals = nullptr;
    ID3D11Fence *fence = nullptr;
    UINT64 fence_value = 0;
    HANDLE fence_event = nullptr;
    ID3D11Texture2D *touch = nullptr;
    struct Unconfirmed {
        int slot;
        long long pts;
        UINT64 fence;
    };
    std::deque<Unconfirmed> unconfirmed;
    // The samples of slots given back while a copy out of them not waited
    // for may still be reading them: given back to the decoder once the
    // timeline passes `copied`, by `sweeper` -- not the decoding thread,
    // which may be the one waiting for them: the decoder's ProcessOutput
    // waits for a free picture of its pool, and holding ten of its 4K HEVC
    // pictures it never returned (a Radeon 780M; asking for a larger pool,
    // MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT, did not change that).
    struct Deferred {
        IMFSample *sample;
        uint64_t copied;
    };
    std::vector<Deferred> deferred;
    // At most this many of the decoder's samples kept (in slots or deferred):
    // a picture beyond them is copied into its slot's own texture (`own`)
    // instead, so that the decoder never runs out (it let 9 of its 4K HEVC
    // pictures be kept, of any pool size asked for).
    static constexpr size_t kMostHeld = 6;
    std::thread sweeper;
    bool sweeper_stops = false;
    // A picture that is a layer of a texture array (Media Foundation's H.264
    // decoder's), which Vulkan does not read a layer of, or one beyond
    // kMostHeld: its crop copied by Direct3D 11 into its slot's own shared
    // texture (made the first time; Vulkan's source `own_source`), once the
    // copies out of the slot's last picture are done (`own_busy`, the
    // timeline's value); its sample given back at once.
    std::vector<ID3D11Texture2D *> own;
    std::vector<int> own_source;
    std::vector<uint64_t> own_busy;
    handover::Slots vk;

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<IMFSample *> slots;
    std::deque<int> free_slots;
    std::deque<Ready> ready;
    bool ended = false, aborted = false, failed = false;
    std::string error;

    void fail(const std::string &text) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!failed) error = text;
        failed = true;
        changed.notify_all();
    }

    bool stopped() {
        std::lock_guard<std::mutex> guard(mutex);
        return failed || aborted;
    }
};

// Whether the hand-over's timeline reaches `value` within `ns` nanoseconds
// (blocked, not spinning).
bool timeline_wait(uint64_t value, uint64_t ns) {
    handover::Device &g = handover::g_device;
    static PFN_vkWaitSemaphores wait =
        reinterpret_cast<PFN_vkWaitSemaphores>(g.vkGetDeviceProcAddr(g.device, "vkWaitSemaphores"));
    if (!value || handover::timeline_reached(value)) return true;
    if (!wait) {
        Sleep(1);
        return handover::timeline_reached(value);
    }
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &g.timeline;
    info.pValues = &value;
    return wait(g.device, &info, ns) == VK_SUCCESS;
}

// The deferred samples whose copies are done, given back to the decoder.
void sweep(Decoder *d) {
    std::vector<IMFSample *> back;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        for (size_t i = 0; i < d->deferred.size();) {
            if (handover::timeline_reached(d->deferred[i].copied)) {
                back.push_back(d->deferred[i].sample);
                d->deferred.erase(d->deferred.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
    }
    for (IMFSample *sample : back) sample->Release();
}

// Decoder::sweeper: the deferred samples given back as their copies finish.
void sweep_thread(Decoder *d) {
    while (true) {
        uint64_t first = 0;
        {
            std::unique_lock<std::mutex> guard(d->mutex);
            d->changed.wait(guard, [d] { return d->sweeper_stops || !d->deferred.empty(); });
            if (d->sweeper_stops) return;
            first = d->deferred.front().copied;
            for (const Decoder::Deferred &later : d->deferred) first = std::min(first, later.copied);
        }
        timeline_wait(first, 50000000ull);  // (50 ms at most: to see sweeper_stops)
        sweep(d);
    }
}

int take_slot(Decoder *d) {
    std::unique_lock<std::mutex> guard(d->mutex);
    d->changed.wait(guard, [d] { return !d->free_slots.empty() || d->aborted || d->failed; });
    if (d->aborted || d->failed) return -1;
    const int slot = d->free_slots.front();
    d->free_slots.pop_front();
    return slot;
}

bool fence_done(Decoder *d, UINT64 value) {
    if (FAILED(d->fence->SetEventOnCompletion(value, d->fence_event))) return false;
    while (WaitForSingleObject(d->fence_event, 100) == WAIT_TIMEOUT) {
        if (d->stopped()) return false;
    }
    return true;
}

// The held pictures whose decoding is done, queued for nvf_pop in order;
// `all`: every one, waiting for them.
bool confirm(Decoder *d, bool all) {
    while (!d->unconfirmed.empty()) {
        const Decoder::Unconfirmed picture = d->unconfirmed.front();
        if (d->fence->GetCompletedValue() < picture.fence) {
            if (!all) return true;
            if (!fence_done(d, picture.fence)) {
                if (!d->stopped()) d->fail("the decoder did not finish a picture");
                return false;
            }
        }
        d->unconfirmed.pop_front();
        std::lock_guard<std::mutex> guard(d->mutex);
        d->ready.push_back({picture.slot, picture.pts});
        d->info.displayed++;
        d->changed.notify_all();
    }
    return true;
}

// Waits for the hand-over's timeline to reach `value`; false when decoding
// stops first.
bool wait_timeline(Decoder *d, uint64_t value) {
    while (!timeline_wait(value, 100000000ull)) {  // 100 ms at a time
        if (d->stopped()) return false;
    }
    return true;
}

// A decoded picture that is a layer of a texture array: its crop copied into
// its slot's own texture (Decoder::own), and held there.
bool deliver_copied(Decoder *d, IMFSample *sample, ID3D11Texture2D *texture, UINT subresource, long long pts) {
    bool full;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        full = d->free_slots.empty();
    }
    const int slot = confirm(d, full) ? take_slot(d) : -1;
    if (slot < 0) {
        texture->Release();
        sample->Release();
        return false;
    }
    if (!d->own[slot]) {
        D3D11_TEXTURE2D_DESC own{};
        own.Width = static_cast<UINT>(d->params.crop_w);
        own.Height = static_cast<UINT>(d->params.crop_h);
        own.MipLevels = own.ArraySize = 1;
        own.Format = d->params.bit_depth > 8 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
        own.SampleDesc.Count = 1;
        own.Usage = D3D11_USAGE_DEFAULT;
        own.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        own.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        IDXGIResource *resource = nullptr;
        HANDLE handle = nullptr;
        std::string why;
        const VkFormat format = d->params.bit_depth > 8 ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                                                        : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        if (FAILED(d->device->CreateTexture2D(&own, nullptr, &d->own[slot]))
            || FAILED(d->own[slot]->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&resource)))
            || FAILED(resource->GetSharedHandle(&handle)) || !handle
            || (d->own_source[slot] = d->vk.source_of(handle, own.Width, own.Height, format, why)) < 0) {
            release(resource);
            texture->Release();
            sample->Release();
            d->fail("the decoder's pictures cannot be handed over" + (why.empty() ? std::string() : ": " + why));
            return false;
        }
        release(resource);
    }
    if (!wait_timeline(d, d->own_busy[slot])) {  // its last picture's copies
        texture->Release();
        sample->Release();
        return false;
    }
    const D3D11_BOX box{static_cast<UINT>(d->params.crop_x), static_cast<UINT>(d->params.crop_y), 0,
                        static_cast<UINT>(d->params.crop_x + d->params.crop_w),
                        static_cast<UINT>(d->params.crop_y + d->params.crop_h), 1};
    d->multithread->Enter();
    d->immediate->CopySubresourceRegion(d->own[slot], 0, 0, 0, 0, texture, subresource, &box);
    d->signals->Signal(d->fence, ++d->fence_value);
    d->immediate->Flush();
    d->multithread->Leave();
    texture->Release();
    sample->Release();
    d->vk.hold(slot, d->own_source[slot], 0, 0);
    d->unconfirmed.push_back({slot, pts, d->fence_value});
    return confirm(d, false);
}

// A decoded picture (the decoder's sample, now ours), held in a slot.
bool deliver(Decoder *d, IMFSample *sample) {
    LONGLONG pts = 0;
    sample->GetSampleTime(&pts);
    IMFMediaBuffer *buffer = nullptr;
    IMFDXGIBuffer *dxgi = nullptr;
    ID3D11Texture2D *texture = nullptr;
    UINT subresource = 0;
    if (FAILED(sample->GetBufferByIndex(0, &buffer))
        || FAILED(buffer->QueryInterface(__uuidof(IMFDXGIBuffer), reinterpret_cast<void **>(&dxgi)))
        || FAILED(dxgi->GetResource(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&texture)))
        || FAILED(dxgi->GetSubresourceIndex(&subresource))) {
        release(texture);
        release(dxgi);
        release(buffer);
        sample->Release();
        d->fail("the decoder's picture is not on the GPU");
        return false;
    }
    release(dxgi);
    release(buffer);
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    const DXGI_FORMAT expected = d->params.bit_depth > 8 ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    IDXGIResource *resource = nullptr;
    HANDLE handle = nullptr;
    const bool shared = desc.MipLevels == 1 && subresource < desc.ArraySize
                        && SUCCEEDED(texture->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&resource)))
                        && SUCCEEDED(resource->GetSharedHandle(&handle)) && handle;
    release(resource);
    if (desc.Format != expected || static_cast<int>(desc.Width) < d->params.width
        || static_cast<int>(desc.Height) < d->params.height || !shared) {
        char text[200];
        snprintf(text, sizeof text, "the decoder's picture cannot be handed over (%ux%u format %d, array %u, misc 0x%x)",
                 desc.Width, desc.Height, static_cast<int>(desc.Format), desc.ArraySize, desc.MiscFlags);
        texture->Release();
        sample->Release();
        d->fail(text);
        return false;
    }
    std::string why;
    const VkFormat format = d->params.bit_depth > 8 ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                                                    : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    size_t held;  // the decoder's samples kept (Decoder::kMostHeld)
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        held = d->deferred.size();
        for (IMFSample *kept : d->slots) held += kept != nullptr;
    }
    if (desc.ArraySize > 1 || held >= Decoder::kMostHeld) return deliver_copied(d, sample, texture, subresource, pts);
    const int source = d->vk.source_of(handle, desc.Width, desc.Height, format, why);
    if (source < 0) {
        texture->Release();
        sample->Release();
        d->fail("Vulkan cannot read the decoder's pictures: " + why);
        return false;
    }
    bool full;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        full = d->free_slots.empty();
    }
    const int slot = confirm(d, full) ? take_slot(d) : -1;
    if (slot < 0) {
        texture->Release();
        sample->Release();
        return false;
    }
    const int left = d->params.crop_x, top = d->params.crop_y;
    const D3D11_BOX box{static_cast<UINT>(left), static_cast<UINT>(top), 0, static_cast<UINT>(left + 2),
                        static_cast<UINT>(top + 2), 1};
    d->multithread->Enter();
    d->immediate->CopySubresourceRegion(d->touch, 0, 0, 0, 0, texture, subresource, &box);
    d->signals->Signal(d->fence, ++d->fence_value);
    d->immediate->Flush();
    d->multithread->Leave();
    texture->Release();
    d->vk.hold(slot, source, left, top);
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        d->slots[slot] = sample;
    }
    d->unconfirmed.push_back({slot, pts, d->fence_value});
    return confirm(d, false);
}

bool set_output_type(Decoder *d) {
    const GUID wanted = d->params.bit_depth > 8 ? MFVideoFormat_P010 : MFVideoFormat_NV12;
    for (DWORD i = 0;; ++i) {
        IMFMediaType *type = nullptr;
        HRESULT hr = d->mft->GetOutputAvailableType(0, i, &type);
        if (FAILED(hr)) {
            d->fail(hr_text("the decoder gives no NV12/P010 pictures", hr));
            return false;
        }
        GUID subtype{};
        type->GetGUID(MF_MT_SUBTYPE, &subtype);
        if (subtype == wanted) {
            hr = d->mft->SetOutputType(0, type, 0);
            type->Release();
            if (FAILED(hr)) {
                d->fail(hr_text("the decoder's output could not be set", hr));
                return false;
            }
            return true;
        }
        type->Release();
    }
}

// Every picture the decoder has ready (`draining`: to its end).
bool pull(Decoder *d) {
    while (!d->stopped()) {
        MFT_OUTPUT_DATA_BUFFER out{};
        DWORD status = 0;
        HRESULT hr = d->mft->ProcessOutput(0, 1, &out, &status);
        if (out.pEvents) out.pEvents->Release();
        if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return true;
        if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
            if (out.pSample) out.pSample->Release();
            if (!set_output_type(d)) return false;
            continue;
        }
        if (FAILED(hr)) {
            if (out.pSample) out.pSample->Release();
            d->fail(hr_text("the GPU's decoder failed", hr));
            return false;
        }
        if (out.pSample && !deliver(d, out.pSample)) return false;
    }
    return false;
}

bool start_handover(Decoder *d, std::string &why) {
    IDXGIDevice *dxgi = nullptr;
    IDXGIAdapter *adapter = nullptr;
    DXGI_ADAPTER_DESC desc{};
    const bool listed = SUCCEEDED(d->device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void **>(&dxgi)))
                        && SUCCEEDED(dxgi->GetAdapter(&adapter)) && SUCCEEDED(adapter->GetDesc(&desc));
    release(adapter);
    release(dxgi);
    if (!listed) {
        why = "DirectX did not say which GPU decodes";
        return false;
    }
    uint8_t luid[VK_LUID_SIZE];
    memcpy(luid, &desc.AdapterLuid, VK_LUID_SIZE);
    if (!handover::shared(luid)) {
        why = handover::g_device.error;
        return false;
    }
    const bool wide = d->params.bit_depth > 8;
    D3D11_TEXTURE2D_DESC shared{};
    shared.Width = static_cast<UINT>(d->params.crop_w);
    shared.Height = static_cast<UINT>(d->params.crop_h);
    shared.MipLevels = shared.ArraySize = 1;
    shared.Format = wide ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    shared.SampleDesc.Count = 1;
    shared.Usage = D3D11_USAGE_DEFAULT;
    shared.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    shared.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
    HANDLE textures[2] = {nullptr, nullptr};
    bool exported = true;
    for (int i = 0; i < 2 && exported; ++i) {
        IDXGIResource1 *resource = nullptr;
        exported = SUCCEEDED(d->device->CreateTexture2D(&shared, nullptr, &d->shared[i]))
                   && SUCCEEDED(d->shared[i]->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&resource)))
                   && SUCCEEDED(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                             nullptr, &textures[i]));
        release(resource);
    }
    ID3D11Device5 *device5 = nullptr;
    const bool fenced = exported
                        && SUCCEEDED(d->device->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void **>(&device5)))
                        && SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, __uuidof(ID3D11Fence),
                                                          reinterpret_cast<void **>(&d->fence)))
                        && SUCCEEDED(d->immediate->QueryInterface(__uuidof(ID3D11DeviceContext4),
                                                                  reinterpret_cast<void **>(&d->signals)))
                        && (d->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr)) != nullptr;
    release(device5);
    if (!fenced) {
        for (HANDLE texture : textures)
            if (texture) CloseHandle(texture);
        why = "Direct3D 11 cannot share its pictures, or has no fences";
        return false;
    }
    const bool made = d->vk.make(d->params, textures,
                                 wide ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                                 why);
    for (HANDLE texture : textures) CloseHandle(texture);
    if (!made) return false;
    if (!d->vk.can_hold()) {  // (reading the decoder's own textures: amf_direct.slang)
        why = "Vulkan cannot read the decoder's pictures where they are";
        return false;
    }
    D3D11_TEXTURE2D_DESC touch = shared;
    touch.Width = touch.Height = 2;
    touch.BindFlags = 0;
    touch.MiscFlags = 0;
    if (FAILED(d->device->CreateTexture2D(&touch, nullptr, &d->touch))) {
        why = "Direct3D 11 could not make a texture";
        return false;
    }
    return true;
}

// The decoder MFT for `codec` that decodes on Direct3D 11, set up for the
// video; or why not.
IMFTransform *open_mft(int codec, int bit_depth, int width, int height, IMFDXGIDeviceManager *manager, int pool,
                       std::string &why) {
    const GUID *subtype = mf_codec(codec);
    if (!subtype || (codec == CODEC_H264 && bit_depth > 8)) {
        why = "not a codec Media Foundation's decoder is asked for";
        return nullptr;
    }
    MFT_REGISTER_TYPE_INFO input{MFMediaType_Video, *subtype};
    IMFActivate **activates = nullptr;
    UINT32 count = 0;
    HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER, MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER,
                           &input, nullptr, &activates, &count);
    IMFTransform *mft = nullptr;
    for (UINT32 i = 0; SUCCEEDED(hr) && i < count; ++i) {
        IMFTransform *candidate = nullptr;
        if (!mft && SUCCEEDED(activates[i]->ActivateObject(__uuidof(IMFTransform), reinterpret_cast<void **>(&candidate)))) {
            IMFAttributes *attributes = nullptr;
            UINT32 aware = 0;
            if (SUCCEEDED(candidate->GetAttributes(&attributes)) && attributes)
                attributes->GetUINT32(MF_SA_D3D11_AWARE, &aware);
            release(attributes);
            if (aware && (!manager || SUCCEEDED(candidate->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER,
                                                                          reinterpret_cast<ULONG_PTR>(manager)))))
                mft = candidate;
            else
                candidate->Release();
        }
        activates[i]->Release();
    }
    CoTaskMemFree(activates);
    if (!mft) {
        why = "no Media Foundation decoder decodes this codec on the GPU";
        return nullptr;
    }
    IMFMediaType *type = nullptr;
    MFCreateMediaType(&type);
    type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    type->SetGUID(MF_MT_SUBTYPE, *subtype);
    MFSetAttributeSize(type, MF_MT_FRAME_SIZE, static_cast<UINT32>(width), static_cast<UINT32>(height));
    type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    hr = mft->SetInputType(0, type, 0);
    type->Release();
    if (FAILED(hr)) {
        why = hr_text("Media Foundation's decoder does not decode this video", hr);
        mft->Release();
        return nullptr;
    }
    IMFAttributes *output = nullptr;
    if (SUCCEEDED(mft->GetOutputStreamAttributes(0, &output)) && output) {
        // Its pictures shareable (KMT), for Vulkan to import, and enough of
        // them for the slots held and those deferred.
        output->SetUINT32(MF_SA_D3D11_SHARED_WITHOUT_MUTEX, TRUE);
        output->SetUINT32(MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT, static_cast<UINT32>(pool + 8));
        output->Release();
    }
    return mft;
}

void destroy(Decoder *d) {
    if (d->sweeper.joinable()) {
        {
            std::lock_guard<std::mutex> guard(d->mutex);
            d->sweeper_stops = true;
            d->changed.notify_all();
        }
        d->sweeper.join();
    }
    if (handover::g_device.device) d->vk.finish_copies();
    for (IMFSample *sample : d->slots)
        if (sample) sample->Release();
    for (const Decoder::Deferred &later : d->deferred) later.sample->Release();
    if (handover::g_device.device) d->vk.free();
    for (ID3D11Texture2D *&texture : d->own) release(texture);
    if (d->mft) {
        if (d->streaming) d->mft->ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        d->mft->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, 0);
    }
    release(d->mft);
    release(d->fence);
    release(d->signals);
    if (d->fence_event) CloseHandle(d->fence_event);
    release(d->touch);
    for (ID3D11Texture2D *&texture : d->shared) release(texture);
    release(d->manager);
    release(d->multithread);
    release(d->immediate);
    release(d->device);
    delete d;
}

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

int download_result(Decoder *d, bool copied) {
    if (copied) return 0;
    d->fail(d->vk.error);
    return NVF_ERROR;
}

}  // namespace

// --------------------------------------------------------------------- API

NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    std::string text;
    if (!start_mf(text)) {
        copy_text(error, error_size, text);
        return nullptr;
    }
    if (!params_valid(*params) || params->widen || !mf_codec(params->codec) || !params->handover || is_scaled(*params)
        || (params->crop_w & 1) || (params->crop_h & 1)) {
        copy_text(error, error_size, "Media Foundation's decoder hands its pictures over only, not scaled");
        return nullptr;
    }
    if (params->crop_x + params->crop_w > params->width || params->crop_y + params->crop_h > params->height) {
        copy_text(error, error_size, "the crop is outside the picture");
        return nullptr;
    }
    Decoder *d = new Decoder();
    d->params = *params;
    if (params->extradata && params->extradata_size > 0)
        d->extradata.assign(params->extradata, params->extradata + params->extradata_size);
    d->params.extradata = nullptr;
    d->info.frame_bytes = static_cast<long long>(frame_bytes(d->params));
    d->info.bit_depth = params->bit_depth;
    d->info.chroma_format = 1;
    d->info.display_right = params->width;
    d->info.display_bottom = params->height;
    d->device = video_device(text);
    bool ok = d->device != nullptr;
    if (ok) {
        d->device->GetImmediateContext(&d->immediate);
        ok = SUCCEEDED(d->device->QueryInterface(__uuidof(ID3D10Multithread), reinterpret_cast<void **>(&d->multithread)));
        if (ok) d->multithread->SetMultithreadProtected(TRUE);
        ok = ok && SUCCEEDED(MFCreateDXGIDeviceManager(&d->token, &d->manager))
             && SUCCEEDED(d->manager->ResetDevice(d->device, d->token));
        if (!ok) text = "Media Foundation could not use the GPU";
    }
    ok = ok && (d->mft = open_mft(params->codec, params->bit_depth, params->width, params->height, d->manager,
                                  params->pool, text)) != nullptr;
    if (ok) {
        // An output type now: the video's if the decoder offers it, else its
        // first (a 10-bit stream's P010 is offered once the decoder has seen
        // the stream, at its first stream change, where it is set).
        IMFMediaType *type = nullptr;
        const GUID wanted = params->bit_depth > 8 ? MFVideoFormat_P010 : MFVideoFormat_NV12;
        IMFMediaType *first = nullptr;
        for (DWORD i = 0; SUCCEEDED(d->mft->GetOutputAvailableType(0, i, &type)); ++i) {
            GUID subtype{};
            type->GetGUID(MF_MT_SUBTYPE, &subtype);
            if (subtype == wanted) {
                release(first);
                first = type;
                break;
            }
            if (!first)
                first = type;
            else
                type->Release();
        }
        if (first) {
            ok = SUCCEEDED(d->mft->SetOutputType(0, first, 0));
            first->Release();
            if (!ok) text = "the decoder's output could not be set";
        }
    }
    if (ok) {
        MFT_OUTPUT_STREAM_INFO stream{};
        ok = SUCCEEDED(d->mft->GetOutputStreamInfo(0, &stream))
             && (stream.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES));
        if (!ok) text = "Media Foundation's decoder does not keep its pictures on the GPU";
    }
    ok = ok && start_handover(d, text);
    if (ok) {
        d->mft->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        d->mft->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        d->streaming = true;
    }
    if (!ok) {
        copy_text(error, error_size, text);
        destroy(d);
        return nullptr;
    }
    d->slots.assign(params->pool, nullptr);
    d->own.assign(params->pool, nullptr);
    d->own_source.assign(params->pool, -1);
    d->own_busy.assign(params->pool, 0);
    for (int slot = 0; slot < params->pool; slot++) d->free_slots.push_back(slot);
    try {
        d->sweeper = std::thread(sweep_thread, d);
    } catch (...) {
        copy_text(error, error_size, "a thread could not be started");
        destroy(d);
        return nullptr;
    }
    return d;
}

NVF_API int nvf_push(void *handle, const unsigned char *data, int size, long long pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->stopped()) return d->aborted ? NVF_ABORTED : NVF_ERROR;
    sweep(d);
    const size_t before = d->first_packet ? d->extradata.size() : 0;
    d->first_packet = false;
    IMFSample *sample = nullptr;
    IMFMediaBuffer *buffer = nullptr;
    BYTE *bytes = nullptr;
    HRESULT hr = MFCreateSample(&sample);
    if (SUCCEEDED(hr)) hr = MFCreateMemoryBuffer(static_cast<DWORD>(before + size), &buffer);
    if (SUCCEEDED(hr)) hr = buffer->Lock(&bytes, nullptr, nullptr);
    if (SUCCEEDED(hr)) {
        if (before) memcpy(bytes, d->extradata.data(), before);
        memcpy(bytes + before, data, static_cast<size_t>(size));
        buffer->Unlock();
        buffer->SetCurrentLength(static_cast<DWORD>(before + size));
        hr = sample->AddBuffer(buffer);
    }
    if (SUCCEEDED(hr)) hr = sample->SetSampleTime(pts);
    release(buffer);
    if (FAILED(hr)) {
        release(sample);
        d->fail(hr_text("a packet could not be given to the decoder", hr));
        return NVF_ERROR;
    }
    while (!d->stopped()) {
        hr = d->mft->ProcessInput(0, sample, 0);
        if (hr == MF_E_NOTACCEPTING) {  // its pictures first
            if (!pull(d)) break;
            continue;
        }
        if (FAILED(hr)) {
            d->fail(hr_text("the GPU's decoder refused a packet", hr));
        } else {
            std::lock_guard<std::mutex> guard(d->mutex);
            d->info.decoded++;
        }
        break;
    }
    sample->Release();
    if (!d->stopped()) pull(d);
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped()) {
        d->mft->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
        const HRESULT hr = d->mft->ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0);
        if (FAILED(hr))
            d->fail(hr_text("finishing the video failed", hr));
        else if (pull(d))
            confirm(d, true);
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    d->ended = true;
    d->changed.notify_all();
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

NVF_API int nvf_pop(void *handle, int timeout_ms, int *slot, long long *pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::unique_lock<std::mutex> guard(d->mutex);
    auto settled = [d] { return !d->ready.empty() || d->ended || d->aborted || d->failed; };
    if (timeout_ms < 0) {
        d->changed.wait(guard, settled);
    } else if (!d->changed.wait_for(guard, std::chrono::milliseconds(timeout_ms), settled)) {
        return NVF_TIMEOUT;
    }
    if (d->aborted) return NVF_ABORTED;
    if (d->failed) return NVF_ERROR;
    if (d->ready.empty()) return NVF_END;
    *slot = d->ready.front().slot;
    *pts = d->ready.front().pts;
    d->ready.pop_front();
    return NVF_FRAME;
}

NVF_API int nvf_download(void *handle, int slot, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    return download_result(d, d->vk.download(slot, host));
}

NVF_API int nvf_copy_luma(void *handle, int slot, unsigned long long dst, long long dst_pitch) {
    Decoder *d = static_cast<Decoder *>(handle);
    return download_result(d, d->vk.copy_luma(slot, dst, dst_pitch));
}

NVF_API int nvf_copy_luma_async(void *handle, int slot, unsigned long long dst, long long dst_pitch,
                                unsigned long long *value) {
    Decoder *d = static_cast<Decoder *>(handle);
    uint64_t signalled = 0;
    const int result = download_result(d, d->vk.copy_luma(slot, dst, dst_pitch, &signalled));
    *value = signalled;
    return result;
}

NVF_API int nvf_copy_planes_async(void *handle, int slot, const unsigned long long *addresses, const long long *pitches,
                                  unsigned long long *value) {
    Decoder *d = static_cast<Decoder *>(handle);
    uint64_t signalled = 0;
    const int result = download_result(d, d->vk.copy_planes(slot, addresses, pitches, &signalled));
    *value = signalled;
    return result;
}

NVF_API int nvf_timeline(void *, void **win32) {
    handover::Device &g = handover::g_device;
    *win32 = nullptr;
    if (!g.timeline || !g.direct_pipelines[0]) return -1;
    VkSemaphoreGetWin32HandleInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    info.semaphore = g.timeline;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE exported = nullptr;
    if (g.vkGetSemaphoreWin32HandleKHR(g.device, &info, &exported) != VK_SUCCESS || !exported) return -1;
    *win32 = exported;
    return 0;
}

NVF_API int nvf_pin(void *, void *host, unsigned long long bytes) {
    return handover::pin(host, bytes);
}

NVF_API int nvf_unpin(void *, void *host) {
    return handover::unpin(host);
}

NVF_API int nvf_copy_planes(void *handle, int slot, const unsigned long long *addresses, const long long *pitches) {
    Decoder *d = static_cast<Decoder *>(handle);
    return download_result(d, d->vk.copy_planes(slot, addresses, pitches));
}

NVF_API int nvf_download_planes(void *handle, int slot, void *const *planes, const long long *pitches) {
    Decoder *d = static_cast<Decoder *>(handle);
    return download_result(d, d->vk.download_planes(slot, planes, pitches));
}

NVF_API int nvf_import_vulkan(void *, void *win32_handle, unsigned long long bytes, unsigned memory_type,
                              const unsigned char *device_uuid, const unsigned char *driver_uuid,
                              unsigned long long *address, void **memory) {
    return handover::import_memory(win32_handle, bytes, memory_type, device_uuid, driver_uuid, address, memory);
}

NVF_API void nvf_unimport(void *, void *memory) {
    if (memory) handover::unimport(memory);
}

NVF_API unsigned long long nvf_slot_pointer(void *, int) {
    return 0;
}

// The slot free again (on the caller's thread: see Decoder::deferred), its
// sample given back to the decoder once its copies are done.
NVF_API void nvf_release(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    IMFSample *sample;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        const uint64_t copied = d->vk.copy_pending(slot);
        sample = d->slots[slot];
        d->slots[slot] = nullptr;
        if (!sample) {
            d->own_busy[slot] = copied;  // (its own texture's)
        } else if (copied && !handover::timeline_reached(copied)) {
            d->deferred.push_back({sample, copied});
            sample = nullptr;
        }
        d->vk.unhold(slot);
        d->free_slots.push_back(slot);
        d->changed.notify_all();
    }
    if (sample) sample->Release();
}

NVF_API void nvf_abort(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->aborted = true;
    d->changed.notify_all();
}

NVF_API void nvf_close(void *handle) {
    if (handle) destroy(static_cast<Decoder *>(handle));
}

NVF_API int nvf_error(void *handle, char *out, int size) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    copy_text(out, size, d->error);
    return static_cast<int>(d->error.size());
}

NVF_API void nvf_info(void *handle, Info *out) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    *out = d->info;
}

NVF_API int nvf_scale_note(void *, char *out, int size) {
    copy_text(out, size, "");
    return 0;
}

NVF_API int nvf_scale_test(int, const Params *, const unsigned char *, int, unsigned char *, char *error, int error_size) {
    copy_text(error, error_size, "Media Foundation's decoder does not scale");
    return -1;
}

NVF_API int nvf_supports(int, int codec, int bit_depth, int width, int height, char *error, int error_size) {
    std::string text;
    if (!start_mf(text)) {
        copy_text(error, error_size, text);
        return 0;
    }
    IMFTransform *mft = open_mft(codec, bit_depth, width, height, nullptr, 1, text);
    if (!mft) {
        copy_text(error, error_size, text);
        return 0;
    }
    mft->Release();
    return 1;
}
