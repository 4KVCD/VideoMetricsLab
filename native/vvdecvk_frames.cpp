// vvdecvk_frames.dll: VVC decoded on the GPU by vvdec-vulkan (github.com/4KVCD/vvdec-vulkan, the GPU's parse), its
// pictures handed over on the GPU without a download to host memory of the decoder's own (vvdec_handover_*):
// - into Vulkan VMAF's slots: the slots' memory is imported into the decoder's Vulkan device, and each picture's luma
//   is copied there by a compute pass that signals the decoder's timeline semaphore, which Vulkan VMAF waits for
//   (vv_commit_after);
// - into Vship's frame buffers (host memory Vship's device maps): nvf_pin imports each buffer into the decoder's
//   device, and nvf_download copies a picture's Y, U and V straight into it on the GPU, returning once they are there.
// The C API is gpu_frames.h's, as amf_frames.dll's hand-over has it.
//
// What it takes (else nvf_open refuses, and the video is decoded another way): VVC, 8 or 10 bits, its pictures not
// scaled or widened, no film grain added. nvf_download takes only buffers pinned here: the pictures have no samples
// in host memory to copy from. The decoder's calls (decode, copy, give back) run one at a time, under the decoder's
// mutex.

#include "gpu_frames.h"

#include <windows.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "vvdec/vvdec.h"

namespace {

struct Ready {
    int slot;
    long long pts;
};

// An imported buffer: what nvf_import_vulkan hands back is base (an address of its own, high bits only) plus an
// offset into it.
struct Import {
    unsigned long long base;
    void *buffer;  // vvdec_handover_import's
};
constexpr int kImportShift = 40;
constexpr unsigned long long kHandOverWaitNs = 10'000'000'000ull;  // nvf_download's wait for a picture's copy: 10 s

// A host buffer pinned for nvf_download (vvdec_handover_import_host's)
struct Pinned {
    const unsigned char *host;
    unsigned long long bytes;
    void *buffer;
};

struct Decoder {
    Params params{};
    Info info{};
    vvdecDecoder *dec = nullptr;
    vvdecAccessUnit *unit = nullptr;
    std::mutex vvdec_lock;  // vvdec's calls, one at a time (vvdec.h's hand-over)

    std::mutex mutex;  // what follows
    std::condition_variable changed;
    std::vector<vvdecFrame *> slots;  // the pictures held (null: free)
    std::deque<Ready> ready;
    std::vector<Import> imports;
    std::vector<Pinned> pinned;
    unsigned long long next_import = 1;
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

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

// The process's vvdec-vulkan settings, before its first decoder: the GPU's parse (which the hand-over needs) and the
// GPU (its Vulkan device index), unless already set. vvdec reads them once per process.
void set_environment(int device) {
    static std::once_flag once;
    std::call_once(once, [device] {
        if (!getenv("VVDEC_GPU_PARSE")) _putenv_s("VVDEC_GPU_PARSE", "2");
        if (!getenv("VVDEC_GPU_DEVICE")) _putenv_s("VVDEC_GPU_DEVICE", std::to_string(device).c_str());
    });
}

// A picture vvdec gave out, into a free slot (waiting for one), queued for nvf_pop; or given back at once when
// decoding has stopped. Called with vvdec_lock held.
void take_picture(Decoder *d, vvdecFrame *frame) {
    if (!frame) return;
    std::unique_lock<std::mutex> guard(d->mutex);
    for (;;) {
        if (d->failed || d->aborted) break;
        for (size_t i = 0; i < d->slots.size(); ++i) {
            if (!d->slots[i]) {
                d->slots[i] = frame;
                d->ready.push_back({static_cast<int>(i), frame->ctsValid ? static_cast<long long>(frame->cts) : 0});
                d->info.decoded++;
                d->info.displayed++;
                d->changed.notify_all();
                return;
            }
        }
        // The pool is full: wait for a slot, letting the copies and the slots given back (which take vvdec_lock)
        // happen meanwhile
        d->vvdec_lock.unlock();
        d->changed.wait(guard);
        guard.unlock();
        d->vvdec_lock.lock();
        guard.lock();
    }
    guard.unlock();
    vvdec_frame_unref(d->dec, frame);
}

int status(Decoder *d) {
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

void destroy(Decoder *d) {
    if (d->dec) {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);
        for (vvdecFrame *&frame : d->slots) {
            if (frame) vvdec_frame_unref(d->dec, frame);
            frame = nullptr;
        }
        for (const Import &imported : d->imports) vvdec_handover_release(d->dec, imported.buffer);
        for (const Pinned &host : d->pinned) vvdec_handover_release(d->dec, host.buffer);
        vvdec_decoder_close(d->dec);
    }
    if (d->unit) vvdec_accessUnit_free(d->unit);
    delete d;
}

// What this decoder takes, or why not.
std::string refusal(const Params &p) {
    if (!params_valid(p)) return "invalid decoder parameters";
    if (p.codec != CODEC_VVC) return "vvdec-vulkan decodes VVC only";
    if (is_scaled(p) || p.widen) return "the hand-over does not scale or widen the pictures";
    // Its chroma planes' rectangle is the crop's halved: whole chroma samples only
    if (!p.luma_only && ((p.crop_x | p.crop_y | p.crop_w | p.crop_h) & 1))
        return "the hand-over copies whole 4:2:0 pictures, cropped at even positions and sizes only";
    if (p.film_grain != 0 && p.film_grain_kinds) return "the hand-over does not add film grain";
    if (!p.handover) return "vvdec-vulkan here only hands its pictures over";
    return {};
}

}  // namespace

// --------------------------------------------------------------------- API

NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    const std::string refused = refusal(*params);
    if (!refused.empty()) {
        copy_text(error, error_size, refused);
        return nullptr;
    }
    set_environment(params->device);
    Decoder *d = new Decoder();
    d->params = *params;
    d->params.extradata = nullptr;
    d->slots.assign(static_cast<size_t>(params->pool), nullptr);
    d->info.frame_bytes = static_cast<long long>(frame_bytes(d->params));
    vvdecParams vp;
    vvdec_params_default(&vp);
    vp.threads = -1;
    vp.gpu = 1;
    vp.filmGrainSynthesis = false;  // grain would be added from host memory, which handed-over pictures do not fill
    vp.logLevel = VVDEC_SILENT;
    d->dec = vvdec_decoder_open(&vp);
    d->unit = vvdec_accessUnit_alloc();
    if (!d->dec || !d->unit || vvdec_handover_enable(d->dec) != VVDEC_OK) {
        copy_text(error, error_size, std::string("vvdec-vulkan cannot hand its pictures over on this GPU")
                                         + (d->dec ? std::string(": ") + vvdec_get_last_error(d->dec) : ""));
        destroy(d);
        return nullptr;
    }
    return d;
}

// One packet (Annex B, a whole access unit); its pictures reach nvf_pop in display order. Blocks while the pool is full.
NVF_API int nvf_push(void *handle, const unsigned char *data, int size, long long pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->stopped()) return status(d);
    std::lock_guard<std::mutex> lock(d->vvdec_lock);
    vvdec_accessUnit_alloc_payload(d->unit, size);  // keeps a payload that is large enough
    memcpy(d->unit->payload, data, static_cast<size_t>(size));
    d->unit->payloadUsedSize = size;
    d->unit->cts = static_cast<uint64_t>(pts);
    d->unit->ctsValid = true;
    vvdecFrame *frame = nullptr;
    const int result = vvdec_decode(d->dec, d->unit, &frame);
    if (result != VVDEC_OK && result != VVDEC_TRY_AGAIN && result != VVDEC_EOF && result != VVDEC_ERR_DEC_INPUT) {
        d->fail(std::string("vvdec-vulkan failed: ") + vvdec_get_last_error(d->dec));
    }
    take_picture(d, frame);
    return status(d);
}

// The end of the video: the pictures still held come out, then nvf_pop reports the end.
NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped()) {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);
        for (int result = VVDEC_OK; result == VVDEC_OK && !d->stopped();) {
            vvdecFrame *frame = nullptr;
            result = vvdec_flush(d->dec, &frame);
            take_picture(d, frame);
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

namespace {

// Waits for the copy that signals `value`, outside the decoder's lock (vvdec_handover_wait_value may run beside its
// calls), and not forever: a GPU that stops answering fails the picture. 0, or NVF_ERROR with the reason kept (the
// wait leaves the decoder's error message alone: its code says why).
int wait_for_copy(Decoder *d, uint64_t value) {
    const int waited = vvdec_handover_wait_value(d->dec, value, kHandOverWaitNs);
    if (waited == VVDEC_OK) return 0;
    d->fail(waited == VVDEC_TRY_AGAIN ? "a picture's copy on the GPU did not finish within 10 s"
                                      : "the GPU failed while copying a picture (" + std::to_string(waited) + ")");
    return NVF_ERROR;
}

}  // namespace

// The slot's picture (frame_bytes' layout: the crop's Y, then U and V) copied by the GPU into a buffer nvf_pin took,
// returning once it is there. One caller at a time (the copier thread). NVF_ERROR for a buffer not pinned here: the
// pictures have no samples in host memory to copy from.
NVF_API int nvf_download(void *handle, int slot, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    const unsigned char *at = static_cast<const unsigned char *>(host);
    const unsigned long long bytes = d->info.frame_bytes;
    void *buffer = nullptr;
    unsigned long long offset = 0;
    vvdecFrame *frame = nullptr;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        for (const Pinned &pinned : d->pinned) {
            if (at >= pinned.host && at + bytes <= pinned.host + pinned.bytes) {
                buffer = pinned.buffer;
                offset = static_cast<unsigned long long>(at - pinned.host);
            }
        }
        if (slot >= 0 && slot < static_cast<int>(d->slots.size())) frame = d->slots[static_cast<size_t>(slot)];
    }
    if (!buffer || !frame) {
        d->fail("a picture asked for in memory the decoder did not pin, or of no slot");
        return NVF_ERROR;
    }
    // As copy_luma's samples; the planes one after another, rows packed
    const Params &p = d->params;
    const int sample = p.bit_depth > 8 ? 2 : 1;
    const int shift = sample == 2 && p.shift == 0 ? 6 : 0;
    const unsigned long long luma = static_cast<unsigned long long>(p.crop_w) * p.crop_h * sample;
    const unsigned long long chroma = static_cast<unsigned long long>(p.crop_w / 2) * (p.crop_h / 2) * sample;
    const uint64_t offsets[3] = {offset, offset + luma, offset + luma + chroma};
    const uint32_t strides[3] = {static_cast<uint32_t>(p.crop_w * sample), static_cast<uint32_t>(p.crop_w / 2 * sample),
                                 static_cast<uint32_t>(p.crop_w / 2 * sample)};
    uint64_t value = 0;
    {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);
        const int result = p.luma_only ? vvdec_handover_pack(d->dec, frame, buffer, offsets[0], strides[0], sample, shift,
                                                             p.crop_x, p.crop_y, p.crop_w, p.crop_h, &value)
                                       : vvdec_handover_pack_frame(d->dec, frame, buffer, offsets, strides, sample, shift,
                                                                   p.crop_x, p.crop_y, p.crop_w, p.crop_h, &value);
        if (result != VVDEC_OK) {
            d->fail(std::string("handing a picture over failed: ") + vvdec_get_last_error(d->dec));
            return NVF_ERROR;
        }
    }
    return wait_for_copy(d, value);
}

namespace {

// The slot's luma (its crop) into imported memory at `dst`, rows `dst_pitch` bytes apart; *value: the timeline's
// value once it is there.
int copy_luma(Decoder *d, int slot, unsigned long long dst, long long dst_pitch, unsigned long long *value) {
    const unsigned long long base = dst >> kImportShift << kImportShift;
    void *buffer = nullptr;
    vvdecFrame *frame = nullptr;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        for (const Import &imported : d->imports) buffer = imported.base == base ? imported.buffer : buffer;
        if (slot >= 0 && slot < static_cast<int>(d->slots.size())) frame = d->slots[static_cast<size_t>(slot)];
    }
    if (!buffer || !frame || dst_pitch <= 0) return NVF_ERROR;
    // 8-bit samples in a byte; 10-bit ones in two, in the low bits (shift 6: as FFmpeg's yuv420p10le) or the top ones
    // (shift 0: P016)
    const int bytes = d->params.bit_depth > 8 ? 2 : 1;
    const int shift = bytes == 2 && d->params.shift == 0 ? 6 : 0;
    uint64_t signalled = 0;
    std::lock_guard<std::mutex> lock(d->vvdec_lock);
    const int result = vvdec_handover_pack(d->dec, frame, buffer, dst - base, static_cast<uint32_t>(dst_pitch), bytes, shift,
                                           d->params.crop_x, d->params.crop_y, d->params.crop_w, d->params.crop_h,
                                           &signalled);
    if (result != VVDEC_OK) {
        d->fail(std::string("handing a picture over failed: ") + vvdec_get_last_error(d->dec));
        return NVF_ERROR;
    }
    *value = signalled;
    return 0;
}

}  // namespace

// The slot's luma into imported memory, returning once it is there.
NVF_API int nvf_copy_luma(void *handle, int slot, unsigned long long dst, long long dst_pitch) {
    Decoder *d = static_cast<Decoder *>(handle);
    unsigned long long value = 0;
    if (copy_luma(d, slot, dst, dst_pitch, &value) != 0) return NVF_ERROR;
    return wait_for_copy(d, value);
}

// nvf_copy_luma, not waited for: *value is the timeline's value once the copy is done (nvf_timeline).
NVF_API int nvf_copy_luma_async(void *handle, int slot, unsigned long long dst, long long dst_pitch,
                                unsigned long long *value) {
    return copy_luma(static_cast<Decoder *>(handle), slot, dst, dst_pitch, value);
}

// The decoder's timeline semaphore, which the copies signal, as an opaque Win32 handle the caller imports and closes.
NVF_API int nvf_timeline(void *handle, void **win32) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> lock(d->vvdec_lock);
    *win32 = vvdec_handover_timeline(d->dec);
    return *win32 ? 0 : -1;
}

// Vulkan VMAF's memory (vv_export's handle of `bytes`, from `memory_type` of the device with these UUIDs): *address
// is what nvf_copy_luma is given, *memory what nvf_unimport frees. 0, or -4 for another GPU or driver.
NVF_API int nvf_import_vulkan(void *handle, void *win32_handle, unsigned long long bytes, unsigned memory_type,
                              const unsigned char *device_uuid, const unsigned char *driver_uuid,
                              unsigned long long *address, void **memory) {
    Decoder *d = static_cast<Decoder *>(handle);
    void *buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);
        uint8_t own_device[16], own_driver[16];
        if (vvdec_handover_device(d->dec, own_device, own_driver) != VVDEC_OK || memcmp(own_device, device_uuid, 16)
            || memcmp(own_driver, driver_uuid, 16)) {
            return -4;
        }
        buffer = vvdec_handover_import(d->dec, win32_handle, bytes, memory_type, device_uuid, driver_uuid);
    }
    if (!buffer) return -3;
    std::lock_guard<std::mutex> guard(d->mutex);
    const unsigned long long base = d->next_import++ << kImportShift;
    d->imports.push_back({base, buffer});
    *address = base;
    *memory = buffer;
    return 0;
}

NVF_API void nvf_unimport(void *handle, void *memory) {
    Decoder *d = static_cast<Decoder *>(handle);
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        for (auto it = d->imports.begin(); it != d->imports.end(); ++it) {
            if (it->buffer == memory) {
                d->imports.erase(it);
                break;
            }
        }
    }
    std::lock_guard<std::mutex> lock(d->vvdec_lock);
    vvdec_handover_release(d->dec, memory);
}

// Imports a host buffer (Vship's frame memory) into the decoder's device for nvf_download: `host` and `bytes` aligned
// as vvdec_handover_import_host asks. 0, or -1 when the decoder's device cannot take it.
NVF_API int nvf_pin(void *handle, void *host, unsigned long long bytes) {
    Decoder *d = static_cast<Decoder *>(handle);
    void *buffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);  // nvf_push may be decoding meanwhile
        buffer = vvdec_handover_import_host(d->dec, host, bytes);
    }
    if (!buffer) return -1;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->pinned.push_back({static_cast<const unsigned char *>(host), bytes, buffer});
    return 0;
}

NVF_API int nvf_unpin(void *handle, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    void *buffer = nullptr;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        for (auto it = d->pinned.begin(); it != d->pinned.end(); ++it) {
            if (it->host == host) {
                buffer = it->buffer;
                d->pinned.erase(it);
                break;
            }
        }
    }
    if (!buffer) return -1;
    std::lock_guard<std::mutex> lock(d->vvdec_lock);
    return vvdec_handover_release(d->dec, buffer) == VVDEC_OK ? 0 : -1;
}

NVF_API void nvf_release(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    vvdecFrame *frame = nullptr;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        if (slot < 0 || slot >= static_cast<int>(d->slots.size())) return;
        frame = d->slots[static_cast<size_t>(slot)];
        d->slots[static_cast<size_t>(slot)] = nullptr;
        d->changed.notify_all();
    }
    if (frame) {
        std::lock_guard<std::mutex> lock(d->vvdec_lock);
        vvdec_frame_unref(d->dec, frame);
    }
}

NVF_API void nvf_abort(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->aborted = true;
    d->changed.notify_all();
}

// Only once no other thread is in a call on this decoder.
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

// Whether the decoder takes this video: VVC at 8 or 10 bits (the GPU is checked when it opens).
NVF_API int nvf_supports(int, int codec, int bit_depth, int, int, char *error, int error_size) {
    if (codec != CODEC_VVC || (bit_depth != 8 && bit_depth != 10)) {
        copy_text(error, error_size, "vvdec-vulkan hands over 8- and 10-bit VVC only");
        return 0;
    }
    return 1;
}
