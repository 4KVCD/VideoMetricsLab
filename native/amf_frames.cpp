// amf_frames: decodes a video on an AMD GPU with AMD's Advanced Media
// Framework (AMF) and hands back its pictures cropped and laid out as the GPU
// metrics read them -- the AMD counterpart of nvdec_frames.cpp and
// vpl_frames.cpp, with the same C API (gpu_frames.h).
//
// AMF's decoder parses the stream itself: the app feeds it the packets FFmpeg
// copies out of the container (vmaf_app/core/nvdec_frames.py) with their
// timestamps, and gets the pictures back in display order. Each is brought to
// system memory by AMF (Convert to host memory), and nvf_download makes one
// CPU pass from there into the caller's buffer (Vship's): the crop's luma,
// U and V split out of their interleaved plane, 10-bit samples aligned as
// asked (convert_frame). The samples are moved, never computed.
//
// Pictures that are scaled are scaled on the GPU instead, from the decoder's
// Direct3D 11 texture, as soon as they are decoded (d3d11_scale.h's shader);
// only the scaled picture comes to system memory, into the slot's own
// buffer. The first pictures are scaled on the CPU as well and compared; if
// the GPU cannot scale them, or its picture is ever not the CPU's, the CPU
// scales the rest (deliver_scaled).
//
// amfrt64.dll comes with AMD's graphics driver (System32) and is loaded at run
// time; this library links nothing of AMD's. The decoder runs on a Direct3D
// 11 device made on the AMD GPU. The headers in native/amf are the AMF SDK's
// public ones (MIT).
//
// AMF's decoder is used from one thread only, the one calling nvf_push and
// nvf_finish: it submits packets, collects pictures, brings them to system
// memory, and gives back the pictures the caller has released. nvf_pop /
// nvf_download / nvf_release come from another thread and touch only host
// memory and the queues.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "d3d11_scale.h"
#include "gpu_frames.h"
// AMF's interfaces overload virtual methods in ways g++ warns about.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "amf/components/ComponentCaps.h"
#include "amf/components/VideoDecoderUVD.h"
#include "amf/core/Factory.h"
#pragma GCC diagnostic pop

namespace {

// ------------------------------------------------------------- the library

HMODULE g_amf = nullptr;
amf::AMFFactory *g_factory = nullptr;
std::mutex g_amf_lock;
bool g_amf_loaded = false;
std::string g_amf_error;

bool load_amf() {
    std::lock_guard<std::mutex> lock(g_amf_lock);
    if (g_amf_loaded) return true;
    if (!g_amf_error.empty()) return false;
    // System32's: the copy AMD's driver installs.
    g_amf = LoadLibraryExW(AMF_DLL_NAME, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!g_amf) {
        g_amf_error = "amfrt64.dll (AMD's media library, installed with AMD's graphics driver) is not installed";
        return false;
    }
    auto init = reinterpret_cast<AMFInit_Fn>(reinterpret_cast<void *>(GetProcAddress(g_amf, AMF_INIT_FUNCTION_NAME)));
    if (!init) {
        g_amf_error = "AMD's media library has no AMFInit";
        return false;
    }
    AMF_RESULT result = init(AMF_FULL_VERSION, &g_factory);
    if (result != AMF_OK || !g_factory) {
        g_amf_error = "AMD's media library could not start (AMF result " + std::to_string(static_cast<int>(result)) + ")";
        return false;
    }
    g_amf_loaded = true;
    return true;
}

const wchar_t *amf_codec(int codec) {
    switch (codec) {
    case CODEC_H264: return AMFVideoDecoderUVD_H264_AVC;
    case CODEC_HEVC: return AMFVideoDecoderHW_H265_HEVC;
    case CODEC_AV1: return AMFVideoDecoderHW_AV1;
    default: return nullptr;
    }
}

std::string result_text(AMF_RESULT result) {
    return "AMF result " + std::to_string(static_cast<int>(result));
}

// A Direct3D 11 device on the first AMD GPU, for AMF's decoder.
ID3D11Device *amd_device(std::string &error) {
    IDXGIFactory1 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory)))) {
        error = "DirectX could not list the GPUs";
        return nullptr;
    }
    ID3D11Device *device = nullptr;
    IDXGIAdapter1 *adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; index++) {
        DXGI_ADAPTER_DESC1 desc;
        bool amd = SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002;
        if (amd) {
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

// Each step of a decoder's shutdown, appended to the file VML_AMF_TRACE
// names, with the thread (scripts/diagnose_amf_close.py): a shutdown that
// never returns then shows which call it is in. One did on a Radeon 780M,
// in Terminate, after the decoder had failed on a video it does not decode
// (see decodes).
void trace(const char *step) {
    char path[MAX_PATH];
    DWORD length = GetEnvironmentVariableA("VML_AMF_TRACE", path, sizeof path);
    if (length == 0 || length >= sizeof path) return;
    if (FILE *file = fopen(path, "a")) {
        fprintf(file, "%llu thread %lu: %s\n", static_cast<unsigned long long>(GetTickCount64()),
                GetCurrentThreadId(), step);
        fclose(file);
    }
}

struct Session {
    ID3D11Device *device = nullptr;
    amf::AMFContext *context = nullptr;
    amf::AMFComponent *decoder = nullptr;

    void close() {
        if (decoder) {
            trace("decoder Terminate");
            decoder->Terminate();
            trace("decoder Release");
            decoder->Release();
            decoder = nullptr;
        }
        if (context) {
            trace("context Terminate");
            context->Terminate();
            trace("context Release");
            context->Release();
            context = nullptr;
        }
        if (device) {
            trace("device Release");
            device->Release();
            device = nullptr;
        }
    }
};

// Whether the decoder says it gives pictures of this depth and size (its
// capabilities' output formats and size range), or why not. Its Init does
// not say: it takes 10-bit H.264, which no AMD GPU decodes, and the pictures
// were then not the video's (a Radeon 780M, driver 32.0.31041.1004: every
// picture wrong, VMAF 69.86 for 86.43), or decoding failed part-way with
// AMF_DIRECTX_FAILED and the decoder's Terminate never returned (driver
// 32.0.21028.21). A decoder that states no capabilities is believed, but for
// H.264 above 8 bits.
bool decodes(amf::AMFComponent *decoder, int codec, int bit_depth, int width, int height, std::string &error) {
    const amf::AMF_SURFACE_FORMAT wanted = bit_depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12;
    bool format = !(codec == CODEC_H264 && bit_depth > 8), size = true;
    amf::AMFCaps *caps = nullptr;
    amf::AMFIOCaps *output = nullptr;
    if (decoder->GetCaps(&caps) == AMF_OK && caps && caps->GetOutputCaps(&output) == AMF_OK && output) {
        format = false;
        for (amf_int32 i = 0; i < output->GetNumOfFormats(); ++i) {
            amf::AMF_SURFACE_FORMAT listed = amf::AMF_SURFACE_UNKNOWN;
            amf_bool native = false;
            if (output->GetFormatAt(i, &listed, &native) == AMF_OK && listed == wanted) format = true;
        }
        amf_int32 least = 0, most = 0;
        output->GetWidthRange(&least, &most);
        if (most > 0 && (width < least || width > most)) size = false;
        output->GetHeightRange(&least, &most);
        if (most > 0 && (height < least || height > most)) size = false;
    }
    if (output) output->Release();
    if (caps) caps->Release();
    if (!format) {
        error = "AMD's GPU decoder does not decode this codec at " + std::to_string(bit_depth) + " bits";
    } else if (!size) {
        error = "AMD's GPU decoder does not decode this codec at " + std::to_string(width) + "x"
                + std::to_string(height);
    }
    return format && size;
}

// AMF's decoder for `codec` at this format and size on the AMD GPU, or the reason there is none.
bool open_session(int codec, int bit_depth, int width, int height, const std::vector<unsigned char> &extradata,
                  bool read_on_cpu, Session &s, std::string &error) {
    s.device = amd_device(error);
    if (!s.device) return false;
    AMF_RESULT result = g_factory->CreateContext(&s.context);
    if (result == AMF_OK) result = s.context->InitDX11(s.device, amf::AMF_DX11_0);
    if (result != AMF_OK) {
        error = "AMD's media library could not start on the GPU (" + result_text(result) + ")";
        s.close();
        return false;
    }
    result = g_factory->CreateComponent(s.context, amf_codec(codec), &s.decoder);
    if (result != AMF_OK) {
        error = "no AMD GPU decoder for this codec (" + result_text(result) + ")";
        s.close();
        return false;
    }
    if (!decodes(s.decoder, codec, bit_depth, width, height, error)) {
        s.close();
        return false;
    }
    // Pictures in display order, with the timestamps given; read on the CPU.
    s.decoder->SetProperty(AMF_VIDEO_DECODER_REORDER_MODE, static_cast<amf_int64>(AMF_VIDEO_DECODER_MODE_REGULAR));
    s.decoder->SetProperty(AMF_TIMESTAMP_MODE, static_cast<amf_int64>(AMF_TS_PRESENTATION));
    s.decoder->SetProperty(AMF_VIDEO_DECODER_SURFACE_CPU, read_on_cpu);
    if (!extradata.empty()) {
        amf::AMFBuffer *buffer = nullptr;
        if (s.context->AllocBuffer(amf::AMF_MEMORY_HOST, extradata.size(), &buffer) == AMF_OK) {
            memcpy(buffer->GetNative(), extradata.data(), extradata.size());
            s.decoder->SetProperty(AMF_VIDEO_DECODER_EXTRADATA, static_cast<amf::AMFInterface *>(buffer));
            buffer->Release();
        }
    }
    result = s.decoder->Init(bit_depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, width, height);
    if (result != AMF_OK) {
        error = "AMD's GPU decoder does not decode this video (" + result_text(result) + ")";
        s.close();
        return false;
    }
    return true;
}

// ----------------------------------------------------------------- decoder

struct Ready {
    int slot;
    long long pts;
};

struct Decoder {
    Params params{};
    Info info{};
    std::vector<unsigned char> extradata;
    PlaneScaler scaler;  // when the pictures are scaled (on the CPU)
    // Scaled pictures: each scaled as it is decoded, into its slot's buffer
    // (feeding thread only, but for the buffers).
    bool buffered = false, gpu_scaling = false;
    int checked = 0;  // pictures the GPU's scaling has been checked on
    gpu_scale::Scaler gpu;
    std::vector<std::vector<uint8_t>> buffers;  // slot -> scaled picture
    std::vector<uint8_t> check;
    std::string scale_note;  // why the CPU scales, when the GPU was meant to
    Session session;

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<amf::AMFSurface *> slots;  // slot -> picture in host memory, or null
    std::deque<int> free_slots;
    std::deque<int> returned;  // released by the caller; given back by the feeding thread
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

void recycle(Decoder *d) {
    std::vector<amf::AMFSurface *> back;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        while (!d->returned.empty()) {
            int slot = d->returned.front();
            d->returned.pop_front();
            if (d->slots[slot]) back.push_back(d->slots[slot]);  // a scaled picture is its buffer's
            d->slots[slot] = nullptr;
            d->free_slots.push_back(slot);
            d->changed.notify_all();
        }
    }
    for (amf::AMFSurface *surface : back) surface->Release();
}

// A free slot, once there is one; -1 when decoding has stopped.
int take_slot(Decoder *d) {
    while (true) {
        recycle(d);
        std::unique_lock<std::mutex> guard(d->mutex);
        if (d->aborted || d->failed) return -1;
        if (!d->free_slots.empty()) {
            int slot = d->free_slots.front();
            d->free_slots.pop_front();
            return slot;
        }
        d->changed.wait(guard, [d] { return !d->free_slots.empty() || !d->returned.empty() || d->aborted || d->failed; });
    }
}

// A picture in host memory, cropped (and scaled) and planes packed, into `out`.
bool host_picture(Decoder *d, amf::AMFSurface *surface, uint8_t *out) {
    amf::AMFPlane *luma = surface->GetPlaneAt(0), *chroma = surface->GetPlaneAt(1);
    if (!luma || !chroma || luma->GetHPitch() != chroma->GetHPitch()) return false;
    size_t pitch = static_cast<size_t>(luma->GetHPitch());
    size_t sample = d->params.bit_depth > 8 ? 2 : 1;
    // A plane's picture can start inside its allocation.
    const uint8_t *y = static_cast<const uint8_t *>(luma->GetNative()) + luma->GetOffsetY() * pitch
                       + luma->GetOffsetX() * sample;
    const uint8_t *uv = static_cast<const uint8_t *>(chroma->GetNative()) + chroma->GetOffsetY() * pitch
                        + chroma->GetOffsetX() * 2 * sample;
    if (is_scaled(d->params)) {  // P010: the top bits
        scale_frame(d->params, d->scaler, y, uv, pitch, true, out);
    } else {
        convert_frame(d->params, y, uv, pitch, true, out);
    }
    return true;
}

void scale_on_cpu_from_now(Decoder *d, const std::string &why) {
    d->gpu_scaling = false;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->scale_note = why;
    d->info.scaled_on_gpu = 0;
}

// AMF's name for which picture of a texture array a surface is, kept on the
// texture (AMFTextureArrayIndexGUID in AMF's samples and FFmpeg).
const GUID kTextureArrayIndex = {0x28115527, 0xe7c3, 0x4b66, {0x99, 0xd3, 0x4f, 0x2a, 0xe6, 0xb4, 0x7f, 0xaf}};

// The decoder's picture scaled on the GPU into `out`, or why not.
bool scale_on_gpu(Decoder *d, amf::AMFSurface *surface, uint8_t *out, std::string &why) {
    amf::AMFPlane *luma = surface->GetPlaneAt(0);
    if (surface->GetMemoryType() != amf::AMF_MEMORY_DX11 || !luma || !luma->GetNative()) {
        why = "AMD's decoder gave no Direct3D 11 texture";
        return false;
    }
    ID3D11Texture2D *texture = static_cast<ID3D11Texture2D *>(luma->GetNative());
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    UINT slice = 0, size = sizeof(slice);
    bool named = SUCCEEDED(texture->GetPrivateData(kTextureArrayIndex, &size, &slice)) && size == sizeof(slice);
    if (!named) slice = 0;
    if (desc.ArraySize != 1 && !named) {
        why = "AMD's decoder keeps its pictures in one texture";  // which slice is not said
        return false;
    }
    d->session.context->LockDX11();
    bool scaled = (d->gpu.started() || d->gpu.start(texture, d->params, why))
                  && d->gpu.scale(texture, slice, luma->GetOffsetX(), luma->GetOffsetY(), out, why);
    d->session.context->UnlockDX11();
    return scaled;
}

// A decoded picture scaled into a slot's buffer and queued for nvf_pop: on
// the GPU, and for the session's first pictures on the CPU too, which must
// give the same picture. False when decoding has stopped.
bool deliver_scaled(Decoder *d, amf::AMFSurface *surface, long long pts) {
    int slot = take_slot(d);
    if (slot < 0) {
        surface->Release();
        return false;
    }
    uint8_t *out = d->buffers[slot].data();
    bool on_gpu = false;
    if (d->gpu_scaling) {
        std::string why;
        on_gpu = scale_on_gpu(d, surface, out, why);
        if (!on_gpu) scale_on_cpu_from_now(d, why);
        // The tests' way to a GPU picture that is not the CPU's.
        if (on_gpu && d->params.cpu_scaling == 2) out[0] ^= 1;
    }
    if (!on_gpu || d->checked < gpu_scale::GPU_SCALE_CHECKED) {
        AMF_RESULT result = surface->Convert(amf::AMF_MEMORY_HOST);
        uint8_t *cpu = on_gpu ? d->check.data() : out;
        if (result != AMF_OK || !host_picture(d, surface, cpu)) {
            surface->Release();
            d->fail("Reading a decoded picture failed (" + result_text(result) + ")");
            return false;
        }
        if (on_gpu) {
            if (memcmp(cpu, out, d->check.size()) == 0) {
                d->checked++;
            } else {
                memcpy(out, cpu, d->check.size());
                scale_on_cpu_from_now(d, "the GPU's scaled picture is not the CPU's");
            }
        }
    }
    surface->Release();
    std::lock_guard<std::mutex> guard(d->mutex);
    d->ready.push_back({slot, pts});
    d->info.displayed++;
    d->changed.notify_all();
    return true;
}

// One picture from the decoder: checked, brought to host memory and queued
// for nvf_pop once a slot is free. False when decoding has stopped.
bool deliver(Decoder *d, amf::AMFData *data) {
    amf::AMFSurface *surface = nullptr;
    data->QueryInterface(amf::AMFSurface::IID(), reinterpret_cast<void **>(&surface));
    long long pts = data->GetPts();
    data->Release();
    if (!surface) {
        d->fail("AMD's GPU decoder gave something other than a picture");
        return false;
    }
    char text[200];
    amf::AMFPlane *luma = surface->GetPlaneAt(0);
    amf::AMF_SURFACE_FORMAT expected = d->params.bit_depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12;
    if (surface->GetFormat() != expected || !luma || luma->GetWidth() != d->params.width
        || luma->GetHeight() != d->params.height) {
        snprintf(text, sizeof text, "a picture is %dx%d, not the %dx%d video", luma ? luma->GetWidth() : 0,
                 luma ? luma->GetHeight() : 0, d->params.width, d->params.height);
        surface->Release();
        d->fail(text);
        return false;
    }
    if (surface->GetFrameType() != amf::AMF_FRAME_PROGRESSIVE) {
        surface->Release();
        d->fail("the video is interlaced");
        return false;
    }
    if (d->buffered) return deliver_scaled(d, surface, pts);
    AMF_RESULT result = surface->Convert(amf::AMF_MEMORY_HOST);
    if (result != AMF_OK) {
        surface->Release();
        d->fail("Reading a decoded picture failed (" + result_text(result) + ")");
        return false;
    }
    int slot = take_slot(d);
    if (slot < 0) {
        surface->Release();
        return false;
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    d->slots[slot] = surface;
    d->ready.push_back({slot, pts});
    d->info.displayed++;
    d->changed.notify_all();
    return true;
}

// Collects every picture the decoder has ready. `draining`: until its end.
bool collect(Decoder *d, bool draining) {
    while (!d->stopped()) {
        amf::AMFData *data = nullptr;
        AMF_RESULT result = d->session.decoder->QueryOutput(&data);
        if (result == AMF_EOF) return true;
        if (data) {
            if (!deliver(d, data)) return false;
            continue;
        }
        if (result == AMF_REPEAT || result == AMF_OK) {
            if (!draining) return true;
            Sleep(1);  // draining: the last pictures are still being decoded
            continue;
        }
        if (result == AMF_RESOLUTION_CHANGED || result == AMF_RESOLUTION_UPDATED) {
            d->fail("the video's format changes partway through");
            return false;
        }
        d->fail("AMD's GPU decoder failed (" + result_text(result) + ")");
        return false;
    }
    return false;
}

void destroy(Decoder *d) {
    trace("close: releasing the pictures held");
    for (amf::AMFSurface *surface : d->slots) {
        if (surface) surface->Release();
    }
    d->session.close();
    trace("close: freeing the decoder");
    delete d;
    trace("close: done");
}

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

}  // namespace

// --------------------------------------------------------------------- API

NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    if (!load_amf()) {
        copy_text(error, error_size, g_amf_error);
        return nullptr;
    }
    if (!params_valid(*params) || params->widen || !amf_codec(params->codec)) {
        copy_text(error, error_size, "invalid decoder parameters");
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
    if (is_scaled(d->params)) prepare_scaler(d->params, d->scaler);
    d->info.bit_depth = params->bit_depth;
    d->info.chroma_format = 1;
    d->info.display_right = params->width;
    d->info.display_bottom = params->height;
    // Scaled pictures stay on the GPU, to be scaled there.
    d->buffered = d->gpu_scaling = is_scaled(d->params) && d->params.cpu_scaling != 1;
    if (d->buffered) {
        d->buffers.assign(params->pool, std::vector<uint8_t>(frame_bytes(d->params)));
        d->check.resize(frame_bytes(d->params));
        d->info.scaled_on_gpu = 1;
    }
    std::string text;
    if (!open_session(params->codec, params->bit_depth, params->width, params->height, d->extradata, !d->buffered,
                      d->session, text)) {
        copy_text(error, error_size, text);
        destroy(d);
        return nullptr;
    }
    d->slots.assign(params->pool, nullptr);
    for (int slot = 0; slot < params->pool; slot++) d->free_slots.push_back(slot);
    return d;
}

NVF_API int nvf_push(void *handle, const unsigned char *data, int size, long long pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->stopped()) return d->aborted ? NVF_ABORTED : NVF_ERROR;
    amf::AMFBuffer *buffer = nullptr;
    AMF_RESULT result = d->session.context->AllocBuffer(amf::AMF_MEMORY_HOST, static_cast<amf_size>(size), &buffer);
    if (result != AMF_OK) {
        d->fail("Allocating a packet buffer failed (" + result_text(result) + ")");
        return NVF_ERROR;
    }
    memcpy(buffer->GetNative(), data, static_cast<size_t>(size));
    buffer->SetPts(pts);
    while (true) {
        recycle(d);
        result = d->session.decoder->SubmitInput(buffer);
        if (result == AMF_INPUT_FULL || result == AMF_DECODER_NO_FREE_SURFACES) {
            // Its queue is full: take its pictures, then try again.
            if (!collect(d, false)) break;
            Sleep(1);
            continue;
        }
        if (result == AMF_RESOLUTION_CHANGED) {
            d->fail("the video's format changes partway through");
        } else if (result != AMF_OK && result != AMF_NEED_MORE_INPUT) {
            d->fail("AMD's GPU decoder failed (" + result_text(result) + ")");
        } else {
            std::lock_guard<std::mutex> guard(d->mutex);
            d->info.decoded++;
        }
        break;
    }
    buffer->Release();
    if (!d->stopped()) collect(d, false);
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped()) {
        AMF_RESULT result = d->session.decoder->Drain();
        if (result != AMF_OK && result != AMF_INPUT_FULL) {
            d->fail("Finishing the video failed (" + result_text(result) + ")");
        } else {
            collect(d, true);
        }
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

// Copies the slot's picture, cropped (and scaled) and planes packed, into `host`.
NVF_API int nvf_download(void *handle, int slot, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->buffered) {  // scaled when it was decoded
        memcpy(host, d->buffers[slot].data(), d->buffers[slot].size());
        return 0;
    }
    amf::AMFSurface *surface;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        surface = d->slots[slot];
    }
    if (!surface) return NVF_ERROR;
    return host_picture(d, surface, static_cast<uint8_t *>(host)) ? 0 : NVF_ERROR;
}

NVF_API int nvf_copy_luma(void *, int, unsigned long long, long long) {
    return NVF_ERROR;  // pictures in system memory: there is no GPU copy to make
}

NVF_API unsigned long long nvf_slot_pointer(void *, int) {
    return 0;
}

NVF_API void nvf_release(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->returned.push_back(slot);
    d->changed.notify_all();
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

// Why the pictures are scaled on the CPU, when the GPU was to scale them.
NVF_API int nvf_scale_note(void *handle, char *out, int size) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    copy_text(out, size, d->scale_note);
    return static_cast<int>(d->scale_note.size());
}

// For the tests: one picture scaled on the CPU or on a GPU (gpu_scale::scale_test).
NVF_API int nvf_scale_test(int vendor, const Params *params, const unsigned char *picture, int on_gpu,
                           unsigned char *out, char *error, int error_size) {
    std::string text;
    int result = gpu_scale::scale_test(vendor, *params, picture, on_gpu, out, text);
    copy_text(error, error_size, text);
    return result;
}

// Whether AMD's GPU decoder decodes `codec` 4:2:0 at `bit_depth` and this size.
NVF_API int nvf_supports(int, int codec, int bit_depth, int width, int height, char *error, int error_size) {
    if (!load_amf()) {
        copy_text(error, error_size, g_amf_error);
        return 0;
    }
    if (!amf_codec(codec)) {
        copy_text(error, error_size, "not a codec AMD's decoder is asked for");
        return 0;
    }
    Session session;
    std::string text;
    bool ok = open_session(codec, bit_depth, width, height, {}, true, session, text);
    session.close();
    if (!ok) copy_text(error, error_size, text);
    return ok ? 1 : 0;
}
