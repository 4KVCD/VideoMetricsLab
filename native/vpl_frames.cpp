// vpl_frames: decodes a video on an Intel GPU with Intel's Video Processing
// Library (oneVPL) and hands back its pictures cropped and laid out as the
// GPU metrics read them -- the Intel counterpart of nvdec_frames.cpp, with
// the same C API (gpu_frames.h).
//
// oneVPL parses the stream itself, as NVIDIA's decoder does: the app feeds
// it the packets FFmpeg copies out of the container (vmaf_app/core/
// nvdec_frames.py) with their timestamps, and gets the pictures back in
// display order, in system memory -- on Intel's integrated GPUs the GPU's
// own memory, filled by the GPU. nvf_download then makes the one CPU pass a
// picture costs: the crop's luma copied, U and V split out of their
// interleaved plane, 10-bit samples aligned as asked (convert_frame), straight
// into the caller's buffer (Vship's). The samples are moved, never computed:
// Intel's decoder through oneVPL gives FFmpeg's CPU decode, sample for sample
// (checked against it frame by frame).
//
// Pictures that are scaled stay on the GPU instead (video memory): each is
// scaled there by d3d11_scale.h's shader as soon as it is decoded, and only
// the scaled picture comes to system memory, into the slot's own buffer.
// The first pictures are scaled on the CPU as well and compared; if the GPU
// cannot scale them, or its picture is ever not the CPU's, the CPU scales
// the rest (deliver_scaled).
//
// libvpl.dll, the oneVPL dispatcher, comes with Intel's graphics driver
// (System32), and finds the driver's runtime; it is loaded at run time, and
// this library links nothing. The headers in native/onevpl are oneVPL's
// (MIT).
//
// oneVPL's session is used from one thread only, the one calling nvf_push and
// nvf_finish: it decodes, waits for each picture, maps it, and gives back the
// pictures the caller has released. nvf_pop / nvf_download / nvf_release come
// from another thread and touch only mapped memory and the queues.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "d3d11_scale.h"
#include "gpu_frames.h"
#include "onevpl/vpl/mfxdispatcher.h"
#include "onevpl/vpl/mfxvideo.h"

namespace {

// ------------------------------------------------------------- the library

struct Vpl {
    HMODULE dll = nullptr;
    decltype(&MFXLoad) Load;
    decltype(&MFXUnload) Unload;
    decltype(&MFXCreateConfig) CreateConfig;
    decltype(&MFXSetConfigFilterProperty) SetConfigFilterProperty;
    decltype(&MFXCreateSession) CreateSession;
    decltype(&MFXEnumImplementations) EnumImplementations;
    decltype(&MFXDispReleaseImplDescription) ReleaseImplDescription;
    decltype(&MFXClose) Close;
    decltype(&MFXVideoDECODE_DecodeHeader) DecodeHeader;
    decltype(&MFXVideoDECODE_Init) Init;
    decltype(&MFXVideoDECODE_DecodeFrameAsync) DecodeFrameAsync;
    decltype(&MFXVideoDECODE_Close) DecodeClose;
};

Vpl g_vpl;
std::mutex g_vpl_lock;
bool g_vpl_loaded = false;
std::string g_vpl_error;

template <typename T>
bool load_symbol(HMODULE module, T &target, const char *name) {
    target = reinterpret_cast<T>(reinterpret_cast<void *>(GetProcAddress(module, name)));
    if (!target) g_vpl_error = std::string("Intel's video library has no ") + name;
    return target != nullptr;
}

bool load_vpl() {
    std::lock_guard<std::mutex> lock(g_vpl_lock);
    if (g_vpl_loaded) return true;
    if (!g_vpl_error.empty()) return false;
    Vpl &v = g_vpl;
    // System32's: the copy Intel's driver installs.
    v.dll = LoadLibraryExW(L"libvpl.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!v.dll) {
        g_vpl_error = "libvpl.dll (Intel's video library, installed with Intel's graphics driver) is not installed";
        return false;
    }
    bool ok = load_symbol(v.dll, v.Load, "MFXLoad") && load_symbol(v.dll, v.Unload, "MFXUnload")
              && load_symbol(v.dll, v.CreateConfig, "MFXCreateConfig")
              && load_symbol(v.dll, v.SetConfigFilterProperty, "MFXSetConfigFilterProperty")
              && load_symbol(v.dll, v.CreateSession, "MFXCreateSession") && load_symbol(v.dll, v.Close, "MFXClose")
              && load_symbol(v.dll, v.EnumImplementations, "MFXEnumImplementations")
              && load_symbol(v.dll, v.ReleaseImplDescription, "MFXDispReleaseImplDescription")
              && load_symbol(v.dll, v.DecodeHeader, "MFXVideoDECODE_DecodeHeader")
              && load_symbol(v.dll, v.Init, "MFXVideoDECODE_Init")
              && load_symbol(v.dll, v.DecodeFrameAsync, "MFXVideoDECODE_DecodeFrameAsync")
              && load_symbol(v.dll, v.DecodeClose, "MFXVideoDECODE_Close");
    if (ok) g_vpl_loaded = true;
    return ok;
}

mfxU32 vpl_codec(int codec) {
    switch (codec) {
    case CODEC_H264: return MFX_CODEC_AVC;
    case CODEC_HEVC: return MFX_CODEC_HEVC;
    case CODEC_AV1: return MFX_CODEC_AV1;
    default: return 0;
    }
}

std::string status_text(mfxStatus status) {
    return "oneVPL status " + std::to_string(static_cast<int>(status));
}

bool set_filter(mfxLoader loader, const char *name, mfxVariantType type, mfxU32 value) {
    mfxConfig config = g_vpl.CreateConfig(loader);
    if (!config) return false;
    mfxVariant variant{};
    variant.Version.Version = MFX_VARIANT_VERSION;
    variant.Type = type;
    if (type == MFX_VARIANT_TYPE_U16) {
        variant.Data.U16 = static_cast<mfxU16>(value);
    } else {
        variant.Data.U32 = value;
    }
    return g_vpl.SetConfigFilterProperty(config, reinterpret_cast<const mfxU8 *>(name), variant) == MFX_ERR_NONE;
}

// A session on Intel's GPU decoder for `codec`, or the reason there is none.
bool open_session(mfxU32 codec, mfxLoader *loader, mfxSession *session, std::string &error) {
    *loader = g_vpl.Load();
    if (!*loader) {
        error = "Intel's video library could not start";
        return false;
    }
    bool filtered = set_filter(*loader, "mfxImplDescription.Impl", MFX_VARIANT_TYPE_U32, MFX_IMPL_TYPE_HARDWARE)
                    && set_filter(*loader, "mfxImplDescription.VendorID", MFX_VARIANT_TYPE_U32, 0x8086)
                    && set_filter(*loader, "mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
                                  MFX_VARIANT_TYPE_U32, codec);
    // The decoded pictures are copied to system memory by the GPU, where it can.
    set_filter(*loader, "DeviceCopy", MFX_VARIANT_TYPE_U16, MFX_GPUCOPY_ON);
    mfxStatus status = filtered ? g_vpl.CreateSession(*loader, 0, session) : MFX_ERR_UNSUPPORTED;
    if (status != MFX_ERR_NONE) {
        error = "no Intel GPU decoder for this codec (" + status_text(status) + ")";
        g_vpl.Unload(*loader);
        *loader = nullptr;
        *session = nullptr;
        return false;
    }
    return true;
}

// ----------------------------------------------------------------- decoder

// Timestamps travel through oneVPL unsigned, MFX_TIMESTAMP_UNKNOWN being all
// ones: moved well above zero, negative ones are safe too.
constexpr long long kPtsOffset = 1LL << 48;

struct Ready {
    int slot;
    long long pts;
};

struct Decoder {
    Params params{};
    Info info{};
    std::vector<unsigned char> extradata;
    PlaneScaler scaler;  // when the pictures are scaled (on the CPU)
    // Scaled pictures decoded to video memory: each scaled as it is decoded,
    // into its slot's buffer (feeding thread only, but for the buffers).
    bool buffered = false, gpu_scaling = false;
    int checked = 0;  // pictures the GPU's scaling has been checked on
    gpu_scale::Scaler gpu;
    std::vector<std::vector<uint8_t>> buffers;  // slot -> scaled picture
    std::vector<uint8_t> check;
    std::string scale_note;  // why the CPU scales, when the GPU was meant to
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxVideoParam video{};
    bool header = false, initialized = false;
    std::vector<mfxU8> stream;  // the bitstream buffer oneVPL reads from
    mfxBitstream bitstream{};
    // Pictures decoded and not yet waited for, oldest first (feeding thread only).
    std::deque<mfxFrameSurface1 *> in_flight;

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<mfxFrameSurface1 *> slots;  // slot -> mapped picture, or null
    std::deque<int> free_slots;
    std::deque<int> returned;  // released by the caller; given back to oneVPL by the feeding thread
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

void give_back(mfxFrameSurface1 *surface) {
    surface->FrameInterface->Unmap(surface);
    surface->FrameInterface->Release(surface);
}

// Gives the pictures the caller has released back to oneVPL (feeding thread).
void recycle(Decoder *d) {
    std::vector<mfxFrameSurface1 *> back;
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
    for (mfxFrameSurface1 *surface : back) give_back(surface);
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

void scale_on_cpu_from_now(Decoder *d, const std::string &why) {
    d->gpu_scaling = false;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->scale_note = why;
    d->info.scaled_on_gpu = 0;
}

// A picture in video memory, scaled into a slot's buffer and queued for
// nvf_pop: on the GPU, and for the session's first pictures on the CPU too,
// which must give the same picture. False when decoding has stopped.
bool deliver_scaled(Decoder *d, mfxFrameSurface1 *surface) {
    const long long pts = static_cast<long long>(surface->Data.TimeStamp) - kPtsOffset;
    int slot = take_slot(d);
    if (slot < 0) {
        surface->FrameInterface->Release(surface);
        return false;
    }
    uint8_t *out = d->buffers[slot].data();
    bool on_gpu = false;
    if (d->gpu_scaling) {
        std::string why;
        mfxHDL resource = nullptr;
        mfxResourceType type{};
        if (surface->FrameInterface->GetNativeHandle(surface, &resource, &type) != MFX_ERR_NONE || !resource
            || type != MFX_RESOURCE_DX11_TEXTURE) {
            why = "Intel's decoder gave no Direct3D 11 texture";
        } else {
            ID3D11Texture2D *texture = static_cast<ID3D11Texture2D *>(resource);
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            if (desc.ArraySize != 1) {
                why = "Intel's decoder keeps its pictures in one texture";  // which slice is not said
            } else {
                on_gpu = (d->gpu.started() || d->gpu.start(texture, d->params, why))
                         && d->gpu.scale(texture, 0, 0, 0, out, why);
            }
        }
        if (!on_gpu) scale_on_cpu_from_now(d, why);
        // The tests' way to a GPU picture that is not the CPU's.
        if (on_gpu && d->params.cpu_scaling == 2) out[0] ^= 1;
    }
    if (!on_gpu || d->checked < gpu_scale::GPU_SCALE_CHECKED) {
        mfxStatus status = surface->FrameInterface->Map(surface, MFX_MAP_READ);
        if (status != MFX_ERR_NONE) {
            surface->FrameInterface->Release(surface);
            d->fail("Reading a decoded picture failed (" + status_text(status) + ")");
            return false;
        }
        const mfxFrameData &data = surface->Data;
        size_t pitch = (static_cast<size_t>(data.PitchHigh) << 16) | data.PitchLow;
        bool msb = d->params.bit_depth > 8 && surface->Info.Shift;
        uint8_t *cpu = on_gpu ? d->check.data() : out;
        scale_frame(d->params, d->scaler, data.Y, data.UV, pitch, msb, cpu);
        surface->FrameInterface->Unmap(surface);
        if (on_gpu) {
            if (memcmp(cpu, out, d->check.size()) == 0) {
                d->checked++;
            } else {
                memcpy(out, cpu, d->check.size());
                scale_on_cpu_from_now(d, "the GPU's scaled picture is not the CPU's");
            }
        }
    }
    surface->FrameInterface->Release(surface);
    std::lock_guard<std::mutex> guard(d->mutex);
    d->ready.push_back({slot, pts});
    d->info.displayed++;
    d->changed.notify_all();
    return true;
}

// Waits for the oldest picture in flight, maps it and queues it for nvf_pop,
// once a slot is free. False when decoding has stopped.
bool deliver_oldest(Decoder *d) {
    mfxFrameSurface1 *surface = d->in_flight.front();
    d->in_flight.pop_front();
    mfxStatus status;
    do {
        status = surface->FrameInterface->Synchronize(surface, 1000);
    } while (status == MFX_WRN_IN_EXECUTION && !d->stopped());
    if (status != MFX_ERR_NONE) {
        surface->FrameInterface->Release(surface);
        if (!d->stopped()) d->fail("Intel's GPU decoder failed (" + status_text(status) + ")");
        return false;
    }
    const mfxFrameInfo &info = surface->Info;
    char text[200];
    if (surface->Data.Corrupted) {
        surface->FrameInterface->Release(surface);
        // FFmpeg would conceal the damage its own way.
        d->fail("Intel's GPU decoder found an error in the video");
        return false;
    }
    if (info.CropX != 0 || info.CropY != 0 || info.CropW != d->params.width || info.CropH != d->params.height) {
        snprintf(text, sizeof text, "a picture is %ux%u, not %dx%d", info.CropW, info.CropH, d->params.width,
                 d->params.height);
        surface->FrameInterface->Release(surface);
        d->fail(text);
        return false;
    }
    if (surface->Data.TimeStamp == static_cast<mfxU64>(MFX_TIMESTAMP_UNKNOWN)) {
        surface->FrameInterface->Release(surface);
        d->fail("Intel's GPU decoder lost a picture's timestamp");
        return false;
    }
    if (d->buffered) return deliver_scaled(d, surface);
    status = surface->FrameInterface->Map(surface, MFX_MAP_READ);
    if (status != MFX_ERR_NONE) {
        surface->FrameInterface->Release(surface);
        d->fail("Reading a decoded picture failed (" + status_text(status) + ")");
        return false;
    }
    int slot = take_slot(d);
    if (slot < 0) {
        give_back(surface);
        return false;
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    d->slots[slot] = surface;
    d->ready.push_back({slot, static_cast<long long>(surface->Data.TimeStamp) - kPtsOffset});
    d->info.displayed++;
    d->changed.notify_all();
    return true;
}

// The stream's format, once oneVPL has found its header; then the decoder.
bool start_decoder(Decoder *d) {
    mfxVideoParam &v = d->video;
    v = mfxVideoParam{};
    v.mfx.CodecId = vpl_codec(d->params.codec);
    mfxStatus status = g_vpl.DecodeHeader(d->session, &d->bitstream, &v);
    if (status == MFX_ERR_MORE_DATA) return true;  // no header yet: more packets
    if (status != MFX_ERR_NONE) {
        d->fail("Intel's GPU decoder could not read the video's header (" + status_text(status) + ")");
        return false;
    }
    const mfxFrameInfo &f = v.mfx.FrameInfo;
    int depth = f.BitDepthLuma ? f.BitDepthLuma : 8;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        d->info.coded_width = f.Width;
        d->info.coded_height = f.Height;
        d->info.display_left = f.CropX;
        d->info.display_top = f.CropY;
        d->info.display_right = f.CropX + f.CropW;
        d->info.display_bottom = f.CropY + f.CropH;
        d->info.bit_depth = depth;
        d->info.chroma_format = f.ChromaFormat;
        d->info.progressive = f.PicStruct == MFX_PICSTRUCT_PROGRESSIVE;
    }
    char text[200];
    if (f.ChromaFormat != MFX_CHROMAFORMAT_YUV420) {
        d->fail("the video is not 4:2:0");
        return false;
    }
    if (depth != d->params.bit_depth || (f.BitDepthChroma && f.BitDepthChroma != f.BitDepthLuma)) {
        snprintf(text, sizeof text, "the video decodes at %d bits, not %d", depth, d->params.bit_depth);
        d->fail(text);
        return false;
    }
    if (f.PicStruct != MFX_PICSTRUCT_PROGRESSIVE && f.PicStruct != MFX_PICSTRUCT_UNKNOWN) {
        d->fail("the video is interlaced");
        return false;
    }
    if (f.CropX != 0 || f.CropY != 0 || f.CropW != d->params.width || f.CropH != d->params.height) {
        snprintf(text, sizeof text, "the decoder's picture (%ux%u, showing %u,%u %ux%u) is not the %dx%d video",
                 f.Width, f.Height, f.CropX, f.CropY, f.CropW, f.CropH, d->params.width, d->params.height);
        d->fail(text);
        return false;
    }
    if (d->params.crop_x + d->params.crop_w > d->params.width || d->params.crop_y + d->params.crop_h > d->params.height) {
        d->fail("the crop is outside the picture");
        return false;
    }
    // Scaled pictures stay on the GPU, to be scaled there.
    d->buffered = is_scaled(d->params) && d->params.cpu_scaling != 1;
    v.IOPattern = d->buffered ? MFX_IOPATTERN_OUT_VIDEO_MEMORY : MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    v.AsyncDepth = 4;
    status = g_vpl.Init(d->session, &v);
    if (status < MFX_ERR_NONE && d->buffered) {  // no video memory: as unscaled pictures are decoded
        g_vpl.DecodeClose(d->session);
        d->buffered = false;
        d->scale_note = "Intel's decoder would not keep its pictures on the GPU (" + status_text(status) + ")";
        v.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
        status = g_vpl.Init(d->session, &v);
    }
    if (d->buffered) {
        d->gpu_scaling = true;
        d->buffers.assign(d->params.pool, std::vector<uint8_t>(frame_bytes(d->params)));
        d->check.resize(frame_bytes(d->params));
    }
    if (status < MFX_ERR_NONE) {
        d->fail("Starting Intel's GPU decoder failed (" + status_text(status) + ")");
        return false;
    }
    d->header = d->initialized = true;
    std::lock_guard<std::mutex> guard(d->mutex);
    d->info.decode_surfaces = v.AsyncDepth;
    d->info.scaled_on_gpu = d->gpu_scaling;
    return true;
}

// Decodes what the bitstream buffer holds (null: drains the decoder).
// False once decoding has stopped.
bool decode(Decoder *d, mfxBitstream *bitstream) {
    while (true) {
        if (d->stopped()) return false;
        recycle(d);
        mfxFrameSurface1 *out = nullptr;
        mfxSyncPoint sync = nullptr;
        mfxStatus status = g_vpl.DecodeFrameAsync(d->session, bitstream, nullptr, &out, &sync);
        if (status == MFX_WRN_DEVICE_BUSY) {
            // The GPU is busy: give it the pictures waited for so far, then retry.
            if (!d->in_flight.empty()) {
                if (!deliver_oldest(d)) return false;
            } else {
                Sleep(1);
            }
            continue;
        }
        if (status == MFX_WRN_VIDEO_PARAM_CHANGED) continue;  // a repeated sequence header
        if (status == MFX_ERR_MORE_DATA) return true;
        if (status == MFX_ERR_MORE_SURFACE) {
            // Every picture is held: hand one over and try again.
            if (d->in_flight.empty()) {
                d->fail("Intel's GPU decoder ran out of pictures");
                return false;
            }
            if (!deliver_oldest(d)) return false;
            continue;
        }
        if (status == MFX_ERR_INCOMPATIBLE_VIDEO_PARAM) {
            d->fail("the video's format changes partway through");
            return false;
        }
        if (status < MFX_ERR_NONE) {
            d->fail("Intel's GPU decoder failed (" + status_text(status) + ")");
            return false;
        }
        if (out && sync) {
            {
                std::lock_guard<std::mutex> guard(d->mutex);
                d->info.decoded++;
            }
            d->in_flight.push_back(out);
            // A few pictures ahead, as FFmpeg's QSV decoder keeps them.
            while (static_cast<int>(d->in_flight.size()) > 3) {
                if (!deliver_oldest(d)) return false;
            }
        }
    }
}

void destroy(Decoder *d) {
    for (mfxFrameSurface1 *surface : d->in_flight) surface->FrameInterface->Release(surface);
    for (mfxFrameSurface1 *surface : d->slots) {
        if (surface) give_back(surface);
    }
    if (d->session) {
        if (d->initialized) g_vpl.DecodeClose(d->session);
        g_vpl.Close(d->session);
    }
    if (d->loader) g_vpl.Unload(d->loader);
    delete d;
}

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

}  // namespace

// --------------------------------------------------------------------- API

NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    if (!load_vpl()) {
        copy_text(error, error_size, g_vpl_error);
        return nullptr;
    }
    if (!params_valid(*params) || params->widen || !vpl_codec(params->codec)) {
        copy_text(error, error_size, "invalid decoder parameters");
        return nullptr;
    }
    Decoder *d = new Decoder();
    d->params = *params;
    if (params->extradata && params->extradata_size > 0)
        d->extradata.assign(params->extradata, params->extradata + params->extradata_size);
    d->params.extradata = nullptr;
    d->info.frame_bytes = static_cast<long long>(frame_bytes(d->params));
    if (is_scaled(d->params)) prepare_scaler(d->params, d->scaler);
    std::string text;
    if (!open_session(vpl_codec(params->codec), &d->loader, &d->session, text)) {
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
    // What oneVPL left of the last packet, then this one (with AV1's sequence
    // header before the first, as NVIDIA's parser is given it).
    mfxBitstream &bs = d->bitstream;
    size_t left = bs.DataLength;
    if (left && bs.DataOffset) memmove(d->stream.data(), d->stream.data() + bs.DataOffset, left);
    size_t extra = d->extradata.empty() || d->header ? 0 : d->extradata.size();
    size_t need = left + extra + static_cast<size_t>(size);
    if (d->stream.size() < need) d->stream.resize(need + (need >> 1));
    if (extra) {
        memcpy(d->stream.data() + left, d->extradata.data(), extra);
        d->extradata.clear();
    }
    memcpy(d->stream.data() + left + extra, data, static_cast<size_t>(size));
    bs.Data = d->stream.data();
    bs.DataOffset = 0;
    bs.DataLength = static_cast<mfxU32>(need);
    bs.MaxLength = static_cast<mfxU32>(d->stream.size());
    bs.TimeStamp = static_cast<mfxU64>(pts + kPtsOffset);
    bs.DataFlag = MFX_BITSTREAM_COMPLETE_FRAME;
    if (!d->initialized && (!start_decoder(d) || !d->initialized)) return d->stopped() ? NVF_ERROR : 0;
    decode(d, &bs);
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped() && d->initialized && decode(d, nullptr)) {
        while (!d->in_flight.empty()) {
            if (!deliver_oldest(d)) break;
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
    mfxFrameSurface1 *surface;
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        surface = d->slots[slot];
    }
    if (!surface) return NVF_ERROR;
    const mfxFrameData &data = surface->Data;
    size_t pitch = (static_cast<size_t>(data.PitchHigh) << 16) | data.PitchLow;
    bool msb = d->params.bit_depth > 8 && surface->Info.Shift;
    if (is_scaled(d->params)) {
        scale_frame(d->params, d->scaler, data.Y, data.UV, pitch, msb, static_cast<uint8_t *>(host));
    } else {
        convert_frame(d->params, data.Y, data.UV, pitch, msb, static_cast<uint8_t *>(host));
    }
    return 0;
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

// Whether Intel's GPU decoder decodes `codec` 4:2:0 at `bit_depth` and this
// size: from the decoder's published capabilities -- a profile of the codec
// with an output format for the depth (NV12, P010) at this size. A depth no
// profile offers (10-bit H.264) is refused here, before a pass starts.
NVF_API int nvf_supports(int, int codec, int bit_depth, int width, int height, char *error, int error_size) {
    if (!load_vpl()) {
        copy_text(error, error_size, g_vpl_error);
        return 0;
    }
    mfxU32 id = vpl_codec(codec);
    if (!id) {
        copy_text(error, error_size, "not a codec Intel's decoder is asked for");
        return 0;
    }
    mfxLoader loader = g_vpl.Load();
    if (!loader) {
        copy_text(error, error_size, "Intel's video library could not start");
        return 0;
    }
    bool filtered = set_filter(loader, "mfxImplDescription.Impl", MFX_VARIANT_TYPE_U32, MFX_IMPL_TYPE_HARDWARE)
                    && set_filter(loader, "mfxImplDescription.VendorID", MFX_VARIANT_TYPE_U32, 0x8086);
    mfxImplDescription *description = nullptr;
    bool found = filtered && g_vpl.EnumImplementations(loader, 0, MFX_IMPLCAPS_IMPLDESCSTRUCTURE,
                                                       reinterpret_cast<mfxHDL *>(&description)) == MFX_ERR_NONE
                 && description;
    mfxU32 format = bit_depth > 8 ? MFX_FOURCC_P010 : MFX_FOURCC_NV12;
    bool supported = false;
    if (found) {
        const mfxDecoderDescription &decoders = description->Dec;
        for (int c = 0; c < decoders.NumCodecs && !supported; c++) {
            const auto &decoder = decoders.Codecs[c];
            if (decoder.CodecID != id) continue;
            for (int p = 0; p < decoder.NumProfiles && !supported; p++) {
                const auto &profile = decoder.Profiles[p];
                for (int m = 0; m < profile.NumMemTypes && !supported; m++) {
                    const auto &memory = profile.MemDesc[m];
                    if (static_cast<mfxU32>(width) > memory.Width.Max || static_cast<mfxU32>(height) > memory.Height.Max)
                        continue;
                    for (int f = 0; f < memory.NumColorFormats && !supported; f++)
                        supported = memory.ColorFormats[f] == format;
                }
            }
        }
        g_vpl.ReleaseImplDescription(loader, description);
    }
    g_vpl.Unload(loader);
    if (!supported) {
        char text[160];
        snprintf(text, sizeof text, "Intel's GPU decoder does not decode this codec at %d bits and %dx%d", bit_depth,
                 width, height);
        copy_text(error, error_size, found ? text : "no Intel GPU decoder");
    }
    return supported ? 1 : 0;
}
