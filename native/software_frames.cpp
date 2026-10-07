// software_frames: decodes a video on the CPU in the scoring process and
// hands back its pictures cropped and laid out as the GPU metrics read them
// -- for the videos FFmpeg would decode in software, the counterpart of
// nvdec_frames.cpp, vpl_frames.cpp and amf_frames.cpp, with the same C API
// (gpu_frames.h).
//
// Those videos used to reach the scoring process through pipes: FFmpeg
// decoded them and wrote every picture out raw -- at 4K, 25 MB a frame,
// which a Windows pipe carries at 60-80 frames a second whatever decodes
// faster -- and the scoring process read them back. Here they are decoded
// in this process by FFmpeg's own decoders -- FFmpeg 9.0.2's libavcodec and
// libavutil, built for the app with only the decoders below (LGPL;
// vmaf_app/tools/ffmpeg, from scripts/build_ffmpeg_decoders.ps1) -- and AV1
// by dav1d, the bundled GStreamer's (BSD; this FFmpeg has no dav1d wrapper,
// and FFmpeg's own AV1 decoder only drives a GPU's). The app loads them
// (vmaf_app/core/gpu_frames.py) and hands them over with nvf_set_libraries;
// this library links neither. It is compiled against FFmpeg's headers
// (native/ffmpeg) and refuses libraries of another major version; of dav1d
// it mirrors the few structures it uses, checked against its API version.
// The app feeds the packets FFmpeg copies out of the container with their
// timestamps, as for the GPU decoders.
//
// Their pictures are FFmpeg's decode's sample for sample -- film grain an
// H.264, HEVC or VVC stream asks for (H.274's, or AOM's AFGS1) added as
// FFmpeg 9 adds it, AV1's by dav1d as FFmpeg's libdav1d decoder has it.
// They are held as the decoder made them, by reference, until the caller
// downloads them: nvf_download makes the one CPU pass a picture costs -- the
// crop's planes copied, 10-bit samples aligned as asked, 8-bit ones widened
// as FFmpeg widens them, or the crop scaled with the GPU decoders' filters
// (scale_filter.h) -- straight into the caller's buffer. The app does not ask
// it to scale (gpu_frames.decoder_supports): on the caller's one thread that
// is slower than FFmpeg scaling in threads before its pipe.
//
// A damaged stream is left to FFmpeg: its decoders conceal damage, and one
// FFmpeg's concealment is not another's (on a stream with a few garbled
// packets, FFmpeg 9 lost 200 frames where 7.1 gave every one) -- the
// FFmpeg the app runs is the user's, of whatever version. libavcodec reports
// what it finds at its error level, which this library counts (the log goes
// nowhere else: ffmpeg.exe's is read by the app, but this one would print to
// the scoring process's console), and dav1d fails decoding: either stops
// decoding here, and the run is made again through FFmpeg.
//
// One thread feeds the packets (nvf_push, nvf_finish) and waits for a free
// slot when the pool is full; nvf_pop, nvf_download and nvf_release come from
// another.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/dict.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
}

#include "gpu_frames.h"

namespace {

// ------------------------------------------------------------- FFmpeg's

// libavutil's and libavcodec's functions, as the app's FFmpeg exports them.
struct Ffmpeg {
    decltype(&::avutil_version) avutil_version;
    decltype(&::av_frame_alloc) av_frame_alloc;
    decltype(&::av_frame_free) av_frame_free;
    decltype(&::av_frame_unref) av_frame_unref;
    decltype(&::av_frame_move_ref) av_frame_move_ref;
    decltype(&::av_dict_set) av_dict_set;
    decltype(&::av_dict_free) av_dict_free;
    decltype(&::av_strerror) av_strerror;
    decltype(&::av_mallocz) av_mallocz;
    decltype(&::av_log_set_callback) av_log_set_callback;
    decltype(&::avcodec_version) avcodec_version;
    decltype(&::avcodec_find_decoder_by_name) avcodec_find_decoder_by_name;
    decltype(&::avcodec_alloc_context3) avcodec_alloc_context3;
    decltype(&::avcodec_open2) avcodec_open2;
    decltype(&::avcodec_free_context) avcodec_free_context;
    decltype(&::avcodec_send_packet) avcodec_send_packet;
    decltype(&::avcodec_receive_frame) avcodec_receive_frame;
    decltype(&::av_packet_alloc) av_packet_alloc;
    decltype(&::av_packet_free) av_packet_free;
    decltype(&::av_new_packet) av_new_packet;
    decltype(&::av_packet_unref) av_packet_unref;
};

// ------------------------------------------------- dav1d's, as mirrored

// dav1d's API 7 (dav1d 1.x from 1.3 on; GStreamer 1.28 ships 1.5).
struct Dav1dUserData {
    const uint8_t *data;
    void *ref;
};
struct Dav1dDataProps {
    int64_t timestamp, duration, offset;
    size_t size;
    Dav1dUserData user_data;
};
struct Dav1dData {
    const uint8_t *data;
    size_t sz;
    void *ref;
    Dav1dDataProps m;
};
struct Dav1dPictureParameters {
    int w, h;
    int layout;  // Dav1dPixelLayout: 1 is I420
    int bpc;
};
struct Dav1dPicture {
    void *seq_hdr, *frame_hdr;
    void *data[3];
    ptrdiff_t stride[2];
    Dav1dPictureParameters p;
    Dav1dDataProps m;
    void *content_light, *mastering_display, *itut_t35;
    size_t n_itut_t35;
    uintptr_t reserved[4];
    void *frame_hdr_ref, *seq_hdr_ref, *content_light_ref, *mastering_display_ref, *itut_t35_ref;
    uintptr_t reserved_ref[4];
    void *ref;
    void *allocator_data;
};
struct Dav1dSettings {
    int n_threads, max_frame_delay, apply_grain, operating_point, all_layers;
    unsigned frame_size_limit;
    void *allocator_cookie, *alloc_picture_callback, *release_picture_callback;
    void *logger_cookie, *logger_callback;
    int strict_std_compliance, output_invisible_frames, inloop_filters, decode_frame_type;
    uint8_t reserved[16];
};
static_assert(offsetof(Dav1dPicture, p) == 56 && offsetof(Dav1dPicture, m) == 72, "Dav1dPicture");
static_assert(offsetof(Dav1dSettings, logger_callback) == 56, "Dav1dSettings");
constexpr int kDav1dI420 = 1;
constexpr unsigned kDav1dApiMajor = 7;
constexpr int kDav1dAgain = -11;  // DAV1D_ERR(EAGAIN)

struct Dav1d {
    unsigned (*version_api)();
    void (*default_settings)(Dav1dSettings *);
    int (*open)(void **, const Dav1dSettings *);
    int (*send_data)(void *, Dav1dData *);
    int (*get_picture)(void *, Dav1dPicture *);
    void (*close)(void **);
    uint8_t *(*data_create)(Dav1dData *, size_t);
    void (*data_unref)(Dav1dData *);
    void (*picture_unref)(Dav1dPicture *);
};

Ffmpeg g_av{};
Dav1d g_dav1d{};
// Errors libavcodec has reported in this process (its log's error level).
std::atomic<unsigned> g_ffmpeg_errors{0};

void count_errors(void *, int level, const char *, va_list) {
    if (level <= AV_LOG_ERROR) g_ffmpeg_errors++;
}
bool g_av_ready = false, g_dav1d_ready = false;
std::string g_libraries_error = "the decoding libraries were not loaded";
std::mutex g_libraries_lock;

template <typename T>
bool bind(HMODULE module, T &target, const char *name, std::string &missing) {
    target = reinterpret_cast<T>(reinterpret_cast<void *>(GetProcAddress(module, name)));
    if (!target && missing.empty()) missing = name;
    return target != nullptr;
}

std::string av_error(int code) {
    char text[128] = {};
    if (g_av.av_strerror(code, text, sizeof text) < 0) snprintf(text, sizeof text, "error %d", code);
    return text;
}

// --------------------------------------------------------- planar pictures

// Copies the crop of a decoded planar 4:2:0 picture -- 8-bit, or 10-bit
// samples in the low bits of 16 (yuv420p10le, as FFmpeg's decoders and
// dav1d give them) -- into `dst`, planes packed, as convert_frame does for
// the GPU decoders' NV12/P010: samples moved, never computed, but for 8-bit
// pictures widened to 10 (Params::widen), made as FFmpeg makes them: v << 2,
// the luma's top two bits repeated below when the video is full range.
void convert_planar(const Params &p, uint8_t *const plane[3], const ptrdiff_t pitch[3], uint8_t *dst) {
    const size_t w = p.crop_w, h = p.crop_h, cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
    const size_t cx = p.crop_x / 2, cy = p.crop_y / 2;
    if (p.bit_depth == 8 && p.widen) {
        uint16_t *wide = reinterpret_cast<uint16_t *>(dst);
        const bool repeat = p.widen == 2;
        for (size_t row = 0; row < h; row++) {
            const uint8_t *src = plane[0] + (p.crop_y + row) * pitch[0] + p.crop_x;
            uint16_t *o = wide + row * w;
            if (repeat) {
                for (size_t i = 0; i < w; i++) o[i] = static_cast<uint16_t>((src[i] << 2) | (src[i] >> 6));
            } else {
                for (size_t i = 0; i < w; i++) o[i] = static_cast<uint16_t>(src[i] << 2);
            }
        }
        if (p.luma_only) return;
        for (int c = 1; c <= 2; c++) {
            uint16_t *out = wide + w * h + (c - 1) * cw * ch;
            for (size_t row = 0; row < ch; row++) {
                const uint8_t *src = plane[c] + (cy + row) * pitch[c] + cx;
                uint16_t *o = out + row * cw;
                for (size_t i = 0; i < cw; i++) o[i] = static_cast<uint16_t>(src[i] << 2);
            }
        }
        return;
    }
    if (p.bit_depth == 8) {
        for (size_t row = 0; row < h; row++) memcpy(dst + row * w, plane[0] + (p.crop_y + row) * pitch[0] + p.crop_x, w);
        if (p.luma_only) return;
        for (int c = 1; c <= 2; c++) {
            uint8_t *out = dst + w * h + (c - 1) * cw * ch;
            for (size_t row = 0; row < ch; row++) memcpy(out + row * cw, plane[c] + (cy + row) * pitch[c] + cx, cw);
        }
        return;
    }
    // 10-bit: kept in the low bits (shift 6, yuv420p10le) or moved to the
    // top ones (shift 0, P016's layout, which Vship reads as 16-bit).
    const int up = p.shift ? 0 : 6;
    uint16_t *out = reinterpret_cast<uint16_t *>(dst);
    bool streamed = false;
    auto copy_plane = [&](const uint8_t *base, ptrdiff_t stride, size_t x, size_t y, size_t width, size_t height,
                          uint16_t *target) {
        for (size_t row = 0; row < height; row++) {
            const uint16_t *src = reinterpret_cast<const uint16_t *>(base + (y + row) * stride) + x;
            uint16_t *o = target + row * width;
            if (up == 0) {
                memcpy(o, src, width * 2);
            } else {
                streamed |= shift_row(src, o, width, 0, up);
            }
        }
    };
    copy_plane(plane[0], pitch[0], p.crop_x, p.crop_y, w, h, out);
    if (!p.luma_only) {
        copy_plane(plane[1], pitch[1], cx, cy, cw, ch, out + w * h);
        copy_plane(plane[2], pitch[2], cx, cy, cw, ch, out + w * h + cw * ch);
    }
    if (streamed) _mm_sfence();  // the streamed rows are written before another thread reads them
}

// convert_planar, scaled to the output size on the way, as scale_frame
// scales the GPU decoders' pictures.
void scale_planar(const Params &p, PlaneScaler &s, uint8_t *const plane[3], const ptrdiff_t pitch[3], uint8_t *dst) {
    const bool wide = p.bit_depth > 8, out_wide = wide_out(p);
    const size_t bps = wide ? 2 : 1, out_bps = out_wide ? 2 : 1;
    const int out_shift = wide && p.shift == 0 ? 6 : 0;
    const float max = out_wide ? 1023.0f : 255.0f;
    const float luma_gain = p.widen == 2 ? 1023.0f / 255.0f : p.widen ? 4.0f : 1.0f;
    const float chroma_gain = p.widen ? 4.0f : 1.0f;
    const int ow = out_width(p), oh = out_height(p), ocw = (ow + 1) / 2, och = (oh + 1) / 2;
    const int cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
    scale_plane(plane[0] + p.crop_y * pitch[0] + p.crop_x * bps, pitch[0], bps, wide, 0, p.crop_w, p.crop_h,
                s.luma_h, s.luma_v, dst, ow, oh, out_wide, luma_gain, max, out_shift, s.scratch);
    if (p.luma_only) return;
    uint8_t *u = dst + static_cast<size_t>(ow) * oh * out_bps, *v = u + static_cast<size_t>(ocw) * och * out_bps;
    for (int c = 1; c <= 2; c++) {
        scale_plane(plane[c] + (p.crop_y / 2) * pitch[c] + (p.crop_x / 2) * bps, pitch[c], bps, wide, 0, cw, ch,
                    s.chroma_h, s.chroma_v, c == 1 ? u : v, ocw, och, out_wide, chroma_gain, max, out_shift,
                    s.scratch);
    }
}

// ----------------------------------------------------------------- decoder

struct Ready {
    int slot;
    long long pts;
};

// A decoded picture held for the caller: libavcodec's frame or dav1d's picture.
struct Slot {
    AVFrame *frame = nullptr;
    Dav1dPicture picture{};
    bool held = false;
};

struct Decoder {
    Params params{};
    Info info{};
    bool av1 = false;
    std::vector<unsigned char> extradata;  // AV1: the sequence header, before the first packet
    PlaneScaler scaler;                    // when the pictures are scaled
    AVCodecContext *codec_context = nullptr;
    AVPacket *packet = nullptr;
    AVFrame *frame = nullptr;              // received into, then moved into a slot
    void *dav1d = nullptr;                 // dav1d's context
    unsigned errors_seen = 0;              // g_ffmpeg_errors when this decoder last looked

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Slot> slots;
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

// A free slot, once there is one; -1 when decoding has stopped.
int take_slot(Decoder *d) {
    std::unique_lock<std::mutex> guard(d->mutex);
    d->changed.wait(guard, [d] { return !d->free_slots.empty() || d->aborted || d->failed; });
    if (d->aborted || d->failed) return -1;
    int slot = d->free_slots.front();
    d->free_slots.pop_front();
    return slot;
}

void queue_picture(Decoder *d, int slot, long long pts) {
    std::lock_guard<std::mutex> guard(d->mutex);
    d->slots[slot].held = true;
    d->ready.push_back({slot, pts});
    d->info.decoded++;
    d->info.displayed++;
    d->changed.notify_all();
}

void unhold(Decoder *d, Slot &slot) {
    if (d->av1) {
        g_dav1d.picture_unref(&slot.picture);
    } else {
        g_av.av_frame_unref(slot.frame);
    }
    slot.held = false;
}

// Whether libavcodec has reported an error since this decoder last looked:
// in this stream or the other video's (one process scores one pair), and
// either way the run is made again through FFmpeg.
bool error_reported(Decoder *d) {
    if (g_ffmpeg_errors.load() == d->errors_seen) return false;
    d->fail("the decoder found an error in the video");
    return true;
}

// A picture libavcodec has decoded (in d->frame), checked and queued for
// nvf_pop. False when decoding has stopped.
bool deliver_frame(Decoder *d) {
    AVFrame *f = d->frame;
    char text[200];
    if (error_reported(d)) {
        g_av.av_frame_unref(f);
        return false;
    }
    if (f->width != d->params.width || f->height != d->params.height) {
        snprintf(text, sizeof text, "a picture is %dx%d, not %dx%d", f->width, f->height, d->params.width,
                 d->params.height);
        g_av.av_frame_unref(f);
        d->fail(text);
        return false;
    }
    const bool format_ok = d->params.bit_depth > 8 ? f->format == AV_PIX_FMT_YUV420P10LE
                                                   : f->format == AV_PIX_FMT_YUV420P || f->format == AV_PIX_FMT_YUVJ420P;
    if (!format_ok) {
        g_av.av_frame_unref(f);
        d->fail("a picture is not 4:2:0 at the video's depth");
        return false;
    }
    if (f->flags & AV_FRAME_FLAG_CORRUPT) {
        // FFmpeg would conceal the damage, perhaps not as this version does.
        g_av.av_frame_unref(f);
        d->fail("the decoder found an error in the video");
        return false;
    }
    const long long pts = f->pts != AV_NOPTS_VALUE ? f->pts : f->best_effort_timestamp;
    if (pts == AV_NOPTS_VALUE) {
        g_av.av_frame_unref(f);
        d->fail("the decoder lost a picture's timestamp");
        return false;
    }
    int slot = take_slot(d);
    if (slot < 0) {
        g_av.av_frame_unref(f);
        return false;
    }
    g_av.av_frame_move_ref(d->slots[slot].frame, f);
    queue_picture(d, slot, pts);
    return true;
}

// Takes every picture libavcodec has ready. False when decoding has stopped.
bool receive_frames(Decoder *d) {
    while (true) {
        if (d->stopped()) return false;
        int result = g_av.avcodec_receive_frame(d->codec_context, d->frame);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return !error_reported(d);
        if (result < 0) {
            d->fail("the decoder failed: " + av_error(result));
            return false;
        }
        if (!deliver_frame(d)) return false;
    }
}

// Sends one packet to libavcodec (null: drains it), taking pictures out
// whenever it wants room. False when decoding has stopped.
bool send_packet(Decoder *d, const AVPacket *packet) {
    while (true) {
        if (d->stopped()) return false;
        int result = g_av.avcodec_send_packet(d->codec_context, packet);
        if (result == 0) return receive_frames(d);
        if (result == AVERROR(EAGAIN)) {
            if (!receive_frames(d)) return false;
            continue;
        }
        if (result == AVERROR_EOF && !packet) return receive_frames(d);
        d->fail("the decoder failed: " + av_error(result));
        return false;
    }
}

// A picture dav1d has decoded, checked and queued. False when decoding has stopped.
bool deliver_picture(Decoder *d, Dav1dPicture &picture) {
    char text[200];
    const char *wrong = nullptr;
    if (picture.p.w != d->params.width || picture.p.h != d->params.height) {
        snprintf(text, sizeof text, "a picture is %dx%d, not %dx%d", picture.p.w, picture.p.h, d->params.width,
                 d->params.height);
        wrong = text;
    } else if (picture.p.layout != kDav1dI420 || picture.p.bpc != d->params.bit_depth) {
        wrong = "a picture is not 4:2:0 at the video's depth";
    }
    if (wrong) {
        g_dav1d.picture_unref(&picture);
        d->fail(wrong);
        return false;
    }
    int slot = take_slot(d);
    if (slot < 0) {
        g_dav1d.picture_unref(&picture);
        return false;
    }
    const long long pts = picture.m.timestamp;
    d->slots[slot].picture = picture;  // the reference moves into the slot
    picture = Dav1dPicture{};
    queue_picture(d, slot, pts);
    return true;
}

// Takes every picture dav1d has ready (at the end: every one it holds).
// False when decoding has stopped.
bool get_pictures(Decoder *d) {
    while (true) {
        if (d->stopped()) return false;
        Dav1dPicture picture{};
        int result = g_dav1d.get_picture(d->dav1d, &picture);
        if (result == kDav1dAgain) return true;
        if (result < 0) {
            d->fail("dav1d failed decoding (error " + std::to_string(result) + ")");
            return false;
        }
        if (!deliver_picture(d, picture)) return false;
    }
}

bool send_data(Decoder *d, Dav1dData &data) {
    while (data.sz > 0) {
        if (d->stopped()) {
            g_dav1d.data_unref(&data);
            return false;
        }
        int result = g_dav1d.send_data(d->dav1d, &data);
        if (result < 0 && result != kDav1dAgain) {
            g_dav1d.data_unref(&data);
            d->fail("dav1d failed decoding (error " + std::to_string(result) + ")");
            return false;
        }
        // Taken (data is empty), or dav1d wants its pictures taken first.
        if (!get_pictures(d)) {
            g_dav1d.data_unref(&data);
            return false;
        }
    }
    return true;
}

void destroy(Decoder *d) {
    for (Slot &slot : d->slots) {
        if (slot.held) unhold(d, slot);
        if (slot.frame) g_av.av_frame_free(&slot.frame);
    }
    if (d->codec_context) g_av.avcodec_free_context(&d->codec_context);
    if (d->packet) g_av.av_packet_free(&d->packet);
    if (d->frame) g_av.av_frame_free(&d->frame);
    if (d->dav1d) g_dav1d.close(&d->dav1d);
    delete d;
}

void copy_text(char *out, int size, const std::string &text) {
    if (out && size > 0) snprintf(out, size, "%s", text.c_str());
}

// libavcodec's decoders by the codec numbers of gpu_frames.h.
const char *decoder_name(int codec) {
    switch (codec) {
    case CODEC_H264: return "h264";
    case CODEC_HEVC: return "hevc";
    case CODEC_VVC: return "vvc";
    case CODEC_VP9: return "vp9";
    case CODEC_MPEG2: return "mpeg2video";
    case CODEC_FFV1: return "ffv1";
    default: return nullptr;
    }
}

// Whether `codec` is decoded here, or why not.
bool decodes(int codec, std::string &why) {
    std::lock_guard<std::mutex> lock(g_libraries_lock);
    if (codec == CODEC_AV1) {
        if (!g_dav1d_ready) why = g_libraries_error;
        return g_dav1d_ready;
    }
    if (!g_av_ready) {
        why = g_libraries_error;
        return false;
    }
    const char *name = decoder_name(codec);
    if (!name || !g_av.avcodec_find_decoder_by_name(name)) {
        why = "no software decoder for this codec";
        return false;
    }
    return true;
}

}  // namespace

// --------------------------------------------------------------------- API

// What is wrong with the libraries nvf_set_libraries was given (empty: nothing).
NVF_API int nvf_libraries_error(char *out, int size) {
    std::lock_guard<std::mutex> lock(g_libraries_lock);
    copy_text(out, size, g_libraries_error);
    return static_cast<int>(g_libraries_error.size());
}

// The libraries to decode with, loaded by the app (their module handles):
// its FFmpeg's libavutil and libavcodec, and GStreamer's dav1d (any may be
// null). 0 when both work; else what is wrong, in nvf_open's and
// nvf_supports' refusals.
NVF_API int nvf_set_libraries(void *avutil, void *avcodec, void *dav1d) {
    std::lock_guard<std::mutex> lock(g_libraries_lock);
    std::string missing, problems;
    if (avutil && avcodec) {
        HMODULE u = static_cast<HMODULE>(avutil), c = static_cast<HMODULE>(avcodec);
        Ffmpeg &a = g_av;
        bool bound = bind(u, a.avutil_version, "avutil_version", missing)
                     && bind(u, a.av_frame_alloc, "av_frame_alloc", missing)
                     && bind(u, a.av_frame_free, "av_frame_free", missing)
                     && bind(u, a.av_frame_unref, "av_frame_unref", missing)
                     && bind(u, a.av_frame_move_ref, "av_frame_move_ref", missing)
                     && bind(u, a.av_dict_set, "av_dict_set", missing) && bind(u, a.av_dict_free, "av_dict_free", missing)
                     && bind(u, a.av_strerror, "av_strerror", missing)
                     && bind(u, a.av_mallocz, "av_mallocz", missing)
                     && bind(u, a.av_log_set_callback, "av_log_set_callback", missing)
                     && bind(c, a.avcodec_version, "avcodec_version", missing)
                     && bind(c, a.avcodec_find_decoder_by_name, "avcodec_find_decoder_by_name", missing)
                     && bind(c, a.avcodec_alloc_context3, "avcodec_alloc_context3", missing)
                     && bind(c, a.avcodec_open2, "avcodec_open2", missing)
                     && bind(c, a.avcodec_free_context, "avcodec_free_context", missing)
                     && bind(c, a.avcodec_send_packet, "avcodec_send_packet", missing)
                     && bind(c, a.avcodec_receive_frame, "avcodec_receive_frame", missing)
                     && bind(c, a.av_packet_alloc, "av_packet_alloc", missing)
                     && bind(c, a.av_packet_free, "av_packet_free", missing)
                     && bind(c, a.av_new_packet, "av_new_packet", missing)
                     && bind(c, a.av_packet_unref, "av_packet_unref", missing);
        if (!bound) {
            problems = "FFmpeg's libraries have no " + missing;
        } else if (a.avutil_version() >> 16 != LIBAVUTIL_VERSION_MAJOR
                   || a.avcodec_version() >> 16 != LIBAVCODEC_VERSION_MAJOR) {
            problems = "FFmpeg's libraries are not the version this decoder was built for (libavutil "
                       + std::to_string(a.avutil_version() >> 16) + ", libavcodec "
                       + std::to_string(a.avcodec_version() >> 16) + ")";
        } else {
            g_av_ready = true;
            a.av_log_set_callback(count_errors);
        }
    } else {
        problems = "FFmpeg's libraries are not bundled";
    }
    missing.clear();
    if (dav1d) {
        HMODULE m = static_cast<HMODULE>(dav1d);
        Dav1d &v = g_dav1d;
        bool bound = bind(m, v.version_api, "dav1d_version_api", missing)
                     && bind(m, v.default_settings, "dav1d_default_settings", missing)
                     && bind(m, v.open, "dav1d_open", missing) && bind(m, v.send_data, "dav1d_send_data", missing)
                     && bind(m, v.get_picture, "dav1d_get_picture", missing) && bind(m, v.close, "dav1d_close", missing)
                     && bind(m, v.data_create, "dav1d_data_create", missing)
                     && bind(m, v.data_unref, "dav1d_data_unref", missing)
                     && bind(m, v.picture_unref, "dav1d_picture_unref", missing);
        if (!bound) {
            problems += (problems.empty() ? "" : "; ") + std::string("dav1d has no ") + missing;
        } else if (v.version_api() >> 16 != kDav1dApiMajor) {
            problems += (problems.empty() ? "" : "; ") + std::string("dav1d is not the version this decoder was made for");
        } else {
            g_dav1d_ready = true;
        }
    } else {
        problems += (problems.empty() ? "" : "; ") + std::string("dav1d is not bundled");
    }
    g_libraries_error = problems.empty() ? "" : problems;
    return problems.empty() ? 0 : NVF_ERROR;
}

NVF_API void *nvf_open(const Params *params, char *error, int error_size) {
    std::string why;
    if (!params_valid(*params) || !decodes(params->codec, why)) {
        copy_text(error, error_size, why.empty() ? "invalid decoder parameters" : why);
        return nullptr;
    }
    Decoder *d = new Decoder();
    d->params = *params;
    d->av1 = params->codec == CODEC_AV1;
    if (d->av1 && params->extradata && params->extradata_size > 0)
        d->extradata.assign(params->extradata, params->extradata + params->extradata_size);
    d->params.extradata = nullptr;
    d->info.frame_bytes = static_cast<long long>(frame_bytes(d->params));
    d->info.coded_width = d->info.display_right = params->width;
    d->info.coded_height = d->info.display_bottom = params->height;
    d->info.bit_depth = params->bit_depth;
    d->info.chroma_format = 1;
    d->info.progressive = 1;
    d->info.decode_surfaces = params->pool;
    if (is_scaled(d->params)) prepare_scaler(d->params, d->scaler);
    d->slots.resize(params->pool);
    for (int slot = 0; slot < params->pool; slot++) d->free_slots.push_back(slot);
    if (d->av1) {
        Dav1dSettings settings{};
        g_dav1d.default_settings(&settings);
        // As FFmpeg's libdav1d decoder sets it: the operating point's
        // highest spatial layer only, film grain applied; its own threads.
        settings.all_layers = 0;
        settings.n_threads = 0;
        settings.max_frame_delay = 0;
        settings.logger_callback = nullptr;  // its messages would go to the scoring process's stderr
        if (g_dav1d.open(&d->dav1d, &settings) < 0) {
            copy_text(error, error_size, "dav1d could not start");
            destroy(d);
            return nullptr;
        }
    } else {
        d->errors_seen = g_ffmpeg_errors.load();
        const AVCodec *codec = g_av.avcodec_find_decoder_by_name(decoder_name(params->codec));
        d->codec_context = g_av.avcodec_alloc_context3(codec);
        d->packet = g_av.av_packet_alloc();
        d->frame = g_av.av_frame_alloc();
        for (Slot &slot : d->slots) slot.frame = g_av.av_frame_alloc();
        bool allocated = d->codec_context && d->packet && d->frame;
        for (Slot &slot : d->slots) allocated = allocated && slot.frame;
        if (allocated && params->codec == CODEC_FFV1) {
            // FFV1's size, which its frames do not carry, and its configuration
            // (a version 3 stream's is not in its packets): the container's,
            // as ffmpeg.exe gives its decoder. Allocated as libavcodec frees
            // it with the context, padded.
            AVCodecContext *context = d->codec_context;
            context->width = params->width;
            context->height = params->height;
            if (params->extradata && params->extradata_size > 0) {
                const size_t size = static_cast<size_t>(params->extradata_size);
                context->extradata = static_cast<uint8_t *>(g_av.av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE));
                allocated = context->extradata != nullptr;
                if (allocated) {
                    memcpy(context->extradata, params->extradata, size);
                    context->extradata_size = params->extradata_size;
                }
            }
        }
        // FFmpeg's own threading, as its decode in ffmpeg.exe has it.
        AVDictionary *options = nullptr;
        g_av.av_dict_set(&options, "threads", "auto", 0);
        int result = allocated ? g_av.avcodec_open2(d->codec_context, codec, &options) : -1;
        g_av.av_dict_free(&options);
        if (result < 0) {
            copy_text(error, error_size, allocated ? "the decoder could not start: " + av_error(result)
                                                   : std::string("out of memory"));
            destroy(d);
            return nullptr;
        }
    }
    return d;
}

NVF_API int nvf_push(void *handle, const unsigned char *data, int size, long long pts) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (d->stopped()) return d->aborted ? NVF_ABORTED : NVF_ERROR;
    if (size <= 0) {
        // libavcodec takes an empty packet for the end of the stream.
        d->fail("the video has an empty packet");
        return NVF_ERROR;
    }
    if (d->av1) {
        // AV1's sequence header before the first packet, as the GPU decoders are given it.
        Dav1dData input{};
        const size_t extra = d->extradata.size();
        uint8_t *buffer = g_dav1d.data_create(&input, extra + static_cast<size_t>(size));
        if (!buffer) {
            d->fail("out of memory");
            return NVF_ERROR;
        }
        if (extra) memcpy(buffer, d->extradata.data(), extra);
        memcpy(buffer + extra, data, static_cast<size_t>(size));
        d->extradata.clear();
        input.m.timestamp = pts;
        send_data(d, input);
    } else {
        if (g_av.av_new_packet(d->packet, size) < 0) {
            d->fail("out of memory");
            return NVF_ERROR;
        }
        memcpy(d->packet->data, data, static_cast<size_t>(size));
        d->packet->pts = pts;
        d->packet->dts = AV_NOPTS_VALUE;
        send_packet(d, d->packet);
        g_av.av_packet_unref(d->packet);
    }
    std::lock_guard<std::mutex> guard(d->mutex);
    return d->aborted ? NVF_ABORTED : d->failed ? NVF_ERROR : 0;
}

NVF_API int nvf_finish(void *handle) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (!d->stopped()) {
        if (d->av1) {
            get_pictures(d);
        } else {
            send_packet(d, nullptr);
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
    if (slot < 0 || slot >= static_cast<int>(d->slots.size())) return NVF_ERROR;
    Slot &held = d->slots[slot];
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        if (!held.held) return NVF_ERROR;
    }
    uint8_t *planes[3];
    ptrdiff_t pitches[3];
    if (d->av1) {
        for (int c = 0; c < 3; c++) planes[c] = static_cast<uint8_t *>(held.picture.data[c]);
        pitches[0] = held.picture.stride[0];
        pitches[1] = pitches[2] = held.picture.stride[1];
    } else {
        for (int c = 0; c < 3; c++) {
            planes[c] = held.frame->data[c];
            pitches[c] = held.frame->linesize[c];
        }
    }
    if (is_scaled(d->params)) {
        scale_planar(d->params, d->scaler, planes, pitches, static_cast<uint8_t *>(host));
    } else {
        convert_planar(d->params, planes, pitches, static_cast<uint8_t *>(host));
    }
    return 0;
}

NVF_API int nvf_copy_luma(void *, int, unsigned long long, long long) {
    return NVF_ERROR;  // pictures in system memory: there is no GPU copy to make
}

NVF_API void nvf_release(void *handle, int slot) {
    Decoder *d = static_cast<Decoder *>(handle);
    if (slot < 0 || slot >= static_cast<int>(d->slots.size())) return;
    Slot &held = d->slots[slot];
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        if (!held.held) return;
    }
    unhold(d, held);  // the decoder's buffer goes back to its pool (thread-safe in both libraries)
    std::lock_guard<std::mutex> guard(d->mutex);
    d->free_slots.push_back(slot);
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

// Whether `codec` at `bit_depth` is decoded here: 4:2:0 at 8 or 10 bits of
// a codec the libraries decode, at any size.
NVF_API int nvf_supports(int, int codec, int bit_depth, int width, int height, char *error, int error_size) {
    std::string why;
    if (!decodes(codec, why)) {
        copy_text(error, error_size, why);
        return 0;
    }
    if ((bit_depth != 8 && bit_depth != 10) || width <= 0 || height <= 0) {
        copy_text(error, error_size, "the software decoder takes 8- and 10-bit 4:2:0 video");
        return 0;
    }
    return 1;
}
