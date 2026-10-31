// software_frames: decodes a video on the CPU in the scoring process and
// hands back its pictures cropped and laid out as the GPU metrics read them.
// It is for the videos FFmpeg would decode in software, the counterpart of
// nvdec_frames.cpp, vpl_frames.cpp and amf_frames.cpp, with the same C API
// (gpu_frames.h).
//
// Don't pipe these videos to the scoring process from FFmpeg: every picture
// goes through raw, 25 MB a frame at 4K, and a Windows pipe carries 60-80
// such frames a second however fast the decoding is. Here they are decoded
// in this process by FFmpeg's own decoders: FFmpeg 9.0.2's libavcodec and
// libavutil, built for the app with only the decoders below (LGPL;
// resources/tools/ffmpeg, from scripts/build_ffmpeg_decoders.ps1). AV1 is
// decoded by dav1d (BSD; beside them, from scripts/build_dav1d.ps1), because
// this FFmpeg has no dav1d wrapper and FFmpeg's own AV1 decoder only drives a
// GPU's; AV2, which FFmpeg does not decode, by dav2d (VideoLAN's AV2 decoder,
// dav1d's API with other names; scripts/build_dav2d.ps1), through the same
// code (DavApi). The app loads them (cpp/core/src/gpu_frames.cpp) and hands
// them over with nvf_set_libraries; this library links none. It is compiled
// against FFmpeg's headers (native/ffmpeg) and refuses libraries of another
// major version; of dav1d and dav2d it mirrors the few structures it uses
// (dav1d/dav1d_api.h, dav2d/dav2d_api.h), checked against their API
// versions. The app feeds the packets FFmpeg copies out of the container
// with their timestamps (a raw stream's made up), as for the GPU decoders;
// AV2's units, which FFmpeg cannot read, come from the app's own reader.
//
// Their pictures are the same as FFmpeg's decode, sample for sample. The
// film grain a stream asks for is as the video is scored (Params::film_grain):
// left out, its parameters exported; or added, VVC's H.274 by VVdeC (FFmpeg's
// libvvdec, of either model at any depth), AFGS1 as FFmpeg 9 adds it and
// AV1's by dav1d; or, the grain not known, as the decoders add it by
// themselves (FFmpeg's H.274 at 8 bits only). Pictures are held as the
// decoder made them, by reference, until the caller has downloaded them or
// read them where they are (nvf_slot_planes, for a picture a download would
// only copy). nvf_download makes the one CPU pass a picture costs, straight
// into the caller's buffer: it copies the crop's planes, aligns 10-bit
// samples as asked and widens 8-bit ones as FFmpeg widens them, or scales
// the crop with the GPU decoders' filters (scale_filter.h). The app asks it
// to scale only a video no other decoder may make
// (gpu_frames.decoder_supports): on the caller's one thread that is slower
// than FFmpeg scaling in threads before its pipe.
//
// A damaged stream is left to FFmpeg. Its decoders conceal damage, and one
// FFmpeg's concealment is not another's: on a stream with a few garbled
// packets, FFmpeg 9 lost 200 frames where 7.1 gave every one. The FFmpeg the
// app runs is the user's, of whatever version. libavcodec reports
// what it finds at its error level, which this library counts (the log goes
// nowhere else: ffmpeg.exe's is read by the app, but this one would print to
// the scoring process's console), and dav1d fails decoding: either stops
// decoding here, and the run is made again through FFmpeg.
//
// One thread feeds the packets (nvf_push, nvf_finish) and waits for a free
// slot when the pool is full; nvf_pop, nvf_download, nvf_slot_planes and
// nvf_release come from another.

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
#include <optional>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/dict.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
}

#include "dav1d/dav1d_api.h"
#include "dav2d/dav2d_api.h"
#include "dav2d/display_order.h"
#include "gpu_frames.h"
#include "grain/grain_decoding.h"

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

// ------------------------------------------ dav1d's and dav2d's, as mirrored

// dav1d's or dav2d's functions, as this library calls them: one API under two
// names, each over its own structures (dav1d/dav1d_api.h, dav2d/dav2d_api.h).
template <typename SettingsT, typename DataT, typename PictureT>
struct DavApi {
    using Settings = SettingsT;
    using Data = DataT;
    using Picture = PictureT;
    const char *name;  // "dav1d" or "dav2d": its functions' prefix, and for messages
    unsigned api_major;
    unsigned (*version_api)() = nullptr;
    void (*default_settings)(Settings *) = nullptr;
    int (*open)(void **, const Settings *) = nullptr;
    int (*send_data)(void *, Data *) = nullptr;
    int (*get_picture)(void *, Picture *) = nullptr;
    void (*close)(void **) = nullptr;
    uint8_t *(*data_create)(Data *, size_t) = nullptr;
    void (*data_unref)(Data *) = nullptr;
    void (*picture_unref)(Picture *) = nullptr;
};
using Dav1d = DavApi<Dav1dSettings, Dav1dData, Dav1dPicture>;
using Dav2d = DavApi<Dav2dSettings, Dav2dData, Dav2dPicture>;
static_assert(kDav1dAgain == kDav2dAgain && kDav1dI420 == kDav2dI420, "dav1d's and dav2d's codes");

Ffmpeg g_av{};
Dav1d g_dav1d{"dav1d", kDav1dApiMajor};
Dav2d g_dav2d{"dav2d", kDav2dApiMajor};
// Errors libavcodec has reported in this process (its log's error level),
// and the last one's words, for the failure it ends a decoder with.
std::atomic<unsigned> g_ffmpeg_errors{0};
std::mutex g_last_error_lock;
std::string g_last_error;

void count_errors(void *, int level, const char *format, va_list arguments) {
    if (level > AV_LOG_ERROR) return;
    char text[300];
    vsnprintf(text, sizeof text, format, arguments);
    size_t length = strlen(text);
    while (length > 0 && (text[length - 1] == '\n' || text[length - 1] == '\r')) text[--length] = '\0';
    {
        std::lock_guard<std::mutex> guard(g_last_error_lock);
        g_last_error = text;
    }
    g_ffmpeg_errors++;
}

std::string last_ffmpeg_error() {
    std::lock_guard<std::mutex> guard(g_last_error_lock);
    return g_last_error;
}
bool g_av_ready = false, g_dav1d_ready = false, g_dav2d_ready = false;
std::string g_libraries_error = "the decoding libraries were not loaded";
// Why AV2 is not decoded where dav2d is not bundled: said to an AV2 video
// alone, as a library the app may ship without (the others always come).
std::string g_dav2d_missing;
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

// Copies the crop of a decoded planar 4:2:0 picture (8-bit, or 10-bit
// samples in the low bits of 16: yuv420p10le, as FFmpeg's decoders and
// dav1d give them) into `dst`, planes packed, as convert_frame does for
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
    const int ow = out_width(p), oh = out_height(p), ocw = (ow + 1) / 2, och = (oh + 1) / 2;
    const int cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
    scale_plane(plane[0] + p.crop_y * pitch[0] + p.crop_x * bps, pitch[0], bps, wide, 0, plane_widen(p, false),
                p.crop_w, p.crop_h, s.luma_h, s.luma_v, simd_rows(oh, false), dst, ow, oh, out_shift, s.scratch);
    if (p.luma_only) return;
    uint8_t *u = dst + static_cast<size_t>(ow) * oh * out_bps, *v = u + static_cast<size_t>(ocw) * och * out_bps;
    for (int c = 1; c <= 2; c++) {
        scale_plane(plane[c] + (p.crop_y / 2) * pitch[c] + (p.crop_x / 2) * bps, pitch[c], bps, wide, 0,
                    plane_widen(p, true), cw, ch, s.chroma_h, s.chroma_v, simd_rows(oh, true), c == 1 ? u : v, ocw,
                    och, out_shift, s.scratch);
    }
}

// ----------------------------------------------------------------- decoder

struct Ready {
    int slot;
    long long pts;
};

// A decoded picture held for the caller: libavcodec's frame, dav1d's or dav2d's picture.
struct Slot {
    AVFrame *frame = nullptr;
    Dav1dPicture picture{};
    Dav2dPicture av2_picture{};
    bool held = false;
};

Dav1dPicture &held_picture(Slot &slot, const Dav1d &) { return slot.picture; }
Dav2dPicture &held_picture(Slot &slot, const Dav2d &) { return slot.av2_picture; }

// Which library decodes a decoder's video.
enum class Engine { libavcodec, dav1d, dav2d };

struct Decoder {
    Params params{};
    Info info{};
    Engine engine = Engine::libavcodec;
    std::vector<unsigned char> extradata;  // AV1: the sequence header, before the first packet
    PlaneScaler scaler;                    // when the pictures are scaled
    AVCodecContext *codec_context = nullptr;
    AVPacket *packet = nullptr;
    AVFrame *frame = nullptr;              // received into, then moved into a slot
    void *dav = nullptr;                   // dav1d's or dav2d's context
    Dav2dDisplayOrder display_order;       // AV2: the stamps of the units fed, to time its pictures by
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
    switch (d->engine) {
    case Engine::dav1d: g_dav1d.picture_unref(&slot.picture); break;
    case Engine::dav2d: g_dav2d.picture_unref(&slot.av2_picture); break;
    case Engine::libavcodec: g_av.av_frame_unref(slot.frame); break;
    }
    slot.held = false;
}

// Whether libavcodec has reported an error since this decoder last looked:
// in this stream or the other video's (one process scores one pair), and
// either way the run is made again through FFmpeg.
bool error_reported(Decoder *d) {
    if (g_ffmpeg_errors.load() == d->errors_seen) return false;
    const std::string said = last_ffmpeg_error();
    d->fail("the decoder found an error in the video" + (said.empty() ? std::string() : ": " + said));
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

// When a picture dav1d or dav2d gives is shown: dav1d's carries its own
// time; dav2d's, in display order, are timed by the units fed
// (dav2d/display_order.h). Nothing where dav2d gives more pictures than units.
bool picture_time(Decoder *, const Dav1dPicture &picture, long long &pts) {
    pts = picture.m.timestamp;
    return true;
}
bool picture_time(Decoder *d, const Dav2dPicture &, long long &pts) {
    const std::optional<long long> shown = d->display_order.shown();
    if (shown) pts = *shown;
    return shown.has_value();
}

// A picture dav1d or dav2d has decoded, checked and queued. False when decoding has stopped.
template <typename Api>
bool deliver_picture(Decoder *d, const Api &api, typename Api::Picture &picture) {
    char text[200];
    const char *wrong = nullptr;
    if (picture.p.w != d->params.width || picture.p.h != d->params.height) {
        snprintf(text, sizeof text, "a picture is %dx%d, not %dx%d", picture.p.w, picture.p.h, d->params.width,
                 d->params.height);
        wrong = text;
    } else if (picture.p.layout != kDav1dI420 || picture.p.bpc != d->params.bit_depth) {
        wrong = "a picture is not 4:2:0 at the video's depth";
    }
    long long pts = 0;
    if (!wrong && !picture_time(d, picture, pts)) wrong = "the decoder gave more pictures than packets";
    if (wrong) {
        api.picture_unref(&picture);
        d->fail(wrong);
        return false;
    }
    int slot = take_slot(d);
    if (slot < 0) {
        api.picture_unref(&picture);
        return false;
    }
    held_picture(d->slots[slot], api) = picture;  // the reference moves into the slot
    picture = typename Api::Picture{};
    queue_picture(d, slot, pts);
    return true;
}

// Takes every picture dav1d or dav2d has ready (at the end: every one it
// holds). False when decoding has stopped.
template <typename Api>
bool get_pictures(Decoder *d, const Api &api) {
    while (true) {
        if (d->stopped()) return false;
        typename Api::Picture picture{};
        int result = api.get_picture(d->dav, &picture);
        if (result == kDav1dAgain || result == kDav2dEof) return true;  // more data wanted; dav2d drained
        if (result < 0) {
            d->fail(std::string(api.name) + " failed decoding (error " + std::to_string(result) + ")");
            return false;
        }
        if (!deliver_picture(d, api, picture)) return false;
    }
}

template <typename Api>
bool send_data(Decoder *d, const Api &api, typename Api::Data &data) {
    while (data.sz > 0) {
        if (d->stopped()) {
            api.data_unref(&data);
            return false;
        }
        int result = api.send_data(d->dav, &data);
        if (result < 0 && result != kDav1dAgain) {
            api.data_unref(&data);
            d->fail(std::string(api.name) + " failed decoding (error " + std::to_string(result) + ")");
            return false;
        }
        // Taken (data is empty), or the decoder wants its pictures taken first.
        if (!get_pictures(d, api)) {
            api.data_unref(&data);
            return false;
        }
    }
    return true;
}

// Starts dav1d or dav2d as FFmpeg's libdav1d decoder starts dav1d: the
// operating point's highest spatial layer only, film grain applied unless
// the video is scored without it; its own threads. False when it cannot.
template <typename Api>
bool open_dav(Decoder *d, const Api &api, bool apply_grain) {
    typename Api::Settings settings{};
    api.default_settings(&settings);
    settings.all_layers = 0;
    settings.n_threads = 0;
    settings.max_frame_delay = 0;
    settings.apply_grain = apply_grain ? 1 : 0;
    settings.logger_callback = nullptr;  // its messages would go to the scoring process's stderr
    return api.open(&d->dav, &settings) >= 0;
}

// One packet to dav1d or dav2d (AV1's sequence header before the first, as
// the GPU decoders are given it). False when decoding has stopped.
template <typename Api>
bool push_dav(Decoder *d, const Api &api, const unsigned char *data, int size, long long pts) {
    typename Api::Data input{};
    const size_t extra = d->extradata.size();
    uint8_t *buffer = api.data_create(&input, extra + static_cast<size_t>(size));
    if (!buffer) {
        d->fail("out of memory");
        return false;
    }
    if (extra) memcpy(buffer, d->extradata.data(), extra);
    memcpy(buffer + extra, data, static_cast<size_t>(size));
    d->extradata.clear();
    input.m.timestamp = pts;
    if (d->engine == Engine::dav2d) d->display_order.fed(pts);
    return send_data(d, api, input);
}

// Binds dav1d's or dav2d's functions in `module`, by its prefix. True when
// all are there and its API is the one mirrored; else says why in `problems`.
template <typename Api>
bool bind_dav(HMODULE module, Api &api, std::string &problems) {
    const std::string prefix = std::string(api.name) + "_";
    std::string missing;
    bool bound = bind(module, api.version_api, (prefix + "version_api").c_str(), missing)
                 && bind(module, api.default_settings, (prefix + "default_settings").c_str(), missing)
                 && bind(module, api.open, (prefix + "open").c_str(), missing)
                 && bind(module, api.send_data, (prefix + "send_data").c_str(), missing)
                 && bind(module, api.get_picture, (prefix + "get_picture").c_str(), missing)
                 && bind(module, api.close, (prefix + "close").c_str(), missing)
                 && bind(module, api.data_create, (prefix + "data_create").c_str(), missing)
                 && bind(module, api.data_unref, (prefix + "data_unref").c_str(), missing)
                 && bind(module, api.picture_unref, (prefix + "picture_unref").c_str(), missing);
    std::string text;
    if (!bound) {
        text = std::string(api.name) + " has no " + missing;
    } else if (api.version_api() >> 16 != api.api_major) {
        text = std::string(api.name) + " is not the version this decoder was made for";
    }
    if (!text.empty()) problems += (problems.empty() ? "" : "; ") + text;
    return text.empty();
}

void destroy(Decoder *d) {
    for (Slot &slot : d->slots) {
        if (slot.held) unhold(d, slot);
        if (slot.frame) g_av.av_frame_free(&slot.frame);
    }
    if (d->codec_context) g_av.avcodec_free_context(&d->codec_context);
    if (d->packet) g_av.av_packet_free(&d->packet);
    if (d->frame) g_av.av_frame_free(&d->frame);
    if (d->dav && d->engine == Engine::dav1d) g_dav1d.close(&d->dav);
    if (d->dav && d->engine == Engine::dav2d) g_dav2d.close(&d->dav);
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
    if (codec == CODEC_AV1 || codec == CODEC_AV2) {
        const bool ready = codec == CODEC_AV1 ? g_dav1d_ready : g_dav2d_ready;
        if (!ready) why = codec == CODEC_AV2 && !g_dav2d_missing.empty() ? g_dav2d_missing : g_libraries_error;
        return ready;
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
// its FFmpeg's libavutil and libavcodec, dav1d and dav2d (any may be null).
// 0 when all work; else what is wrong, in nvf_open's and nvf_supports'
// refusals.
NVF_API int nvf_set_libraries(void *avutil, void *avcodec, void *dav1d, void *dav2d) {
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
    if (dav1d) {
        g_dav1d_ready = bind_dav(static_cast<HMODULE>(dav1d), g_dav1d, problems);
    } else {
        problems += (problems.empty() ? "" : "; ") + std::string("dav1d is not bundled");
    }
    g_dav2d_missing = dav2d ? "" : "dav2d is not bundled";
    if (dav2d) g_dav2d_ready = bind_dav(static_cast<HMODULE>(dav2d), g_dav2d, problems);
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
    d->engine = params->codec == CODEC_AV1 ? Engine::dav1d : params->codec == CODEC_AV2 ? Engine::dav2d
                                                                                       : Engine::libavcodec;
    if (d->engine == Engine::dav1d && params->extradata && params->extradata_size > 0)
        d->extradata.assign(params->extradata, params->extradata + params->extradata_size);
    d->params.extradata = nullptr;
    d->info.frame_bytes = static_cast<long long>(frame_bytes(d->params));
    d->info.coded_width = d->info.display_right = params->width;
    d->info.coded_height = d->info.display_bottom = params->height;
    d->info.bit_depth = params->bit_depth;
    d->info.chroma_format = 1;
    d->info.progressive = 1;
    d->info.decode_surfaces = params->pool;
    if (is_scaled(d->params) && !prepare_scaler(d->params, d->scaler)) {
        copy_text(error, error_size, "so large a downscale is left to FFmpeg, which makes it in two steps");
        destroy(d);
        return nullptr;
    }
    d->slots.resize(params->pool);
    for (int slot = 0; slot < params->pool; slot++) d->free_slots.push_back(slot);
    const GrainDecoding grain = grain_decoding(params->codec == CODEC_VVC, params->film_grain);
    if (d->engine != Engine::libavcodec) {
        const bool opened =
            d->engine == Engine::dav1d ? open_dav(d, g_dav1d, grain.apply_grain) : open_dav(d, g_dav2d, grain.apply_grain);
        if (!opened) {
            copy_text(error, error_size, std::string(d->engine == Engine::dav1d ? "dav1d" : "dav2d") + " could not start");
            destroy(d);
            return nullptr;
        }
    } else {
        d->errors_seen = g_ffmpeg_errors.load();
        const char *name = grain.decoder != nullptr ? grain.decoder : decoder_name(params->codec);
        const AVCodec *codec = g_av.avcodec_find_decoder_by_name(name);
        if (!codec) {
            copy_text(error, error_size,
                      std::string("this FFmpeg build has no ") + name + " decoder (scripts/build_ffmpeg_decoders.ps1)");
            destroy(d);
            return nullptr;
        }
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
        // A video scored without its film grain: its parameters given as the
        // frames' side data, not added (H.274's, AFGS1, and VVdeC's).
        if (allocated && grain.export_grain) {
            d->codec_context->export_side_data |= AV_CODEC_EXPORT_DATA_FILM_GRAIN;
        }
        // FFmpeg's own threading, as its decode in ffmpeg.exe has it.
        AVDictionary *options = nullptr;
        g_av.av_dict_set(&options, "threads", "auto", 0);
        if (std::string(name) == "libvvdec") {
            // One picture parsed ahead per thread, not VVdeC's 1.5 (scoring_parse_delay).
            const std::string delay = std::to_string(scoring_parse_delay(std::thread::hardware_concurrency()));
            g_av.av_dict_set(&options, "parse_delay", delay.c_str(), 0);
        }
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
    if (d->engine == Engine::dav1d) {
        push_dav(d, g_dav1d, data, size, pts);
    } else if (d->engine == Engine::dav2d) {
        push_dav(d, g_dav2d, data, size, pts);
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
        if (d->engine == Engine::dav1d) {
            get_pictures(d, g_dav1d);
        } else if (d->engine == Engine::dav2d) {
            g_dav2d.send_data(d->dav, nullptr);  // the end: dav2d gives the pictures it holds
            get_pictures(d, g_dav2d);
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

// The picture the caller holds in `slot`, as the decoder made it: each
// plane's first sample and the bytes from one row to the next. False for a
// slot that is not held.
bool held_planes(Decoder *d, int slot, uint8_t *planes[3], ptrdiff_t pitches[3]) {
    if (slot < 0 || slot >= static_cast<int>(d->slots.size())) return false;
    Slot &held = d->slots[slot];
    {
        std::lock_guard<std::mutex> guard(d->mutex);
        if (!held.held) return false;
    }
    if (d->engine != Engine::libavcodec) {
        void *const *data = d->engine == Engine::dav1d ? held.picture.data : held.av2_picture.data;
        const ptrdiff_t *stride = d->engine == Engine::dav1d ? held.picture.stride : held.av2_picture.stride;
        for (int c = 0; c < 3; c++) planes[c] = static_cast<uint8_t *>(data[c]);
        pitches[0] = stride[0];
        pitches[1] = pitches[2] = stride[1];
    } else {
        for (int c = 0; c < 3; c++) {
            planes[c] = held.frame->data[c];
            pitches[c] = held.frame->linesize[c];
        }
    }
    return true;
}

// Copies the slot's picture, cropped (and scaled) and planes packed, into `host`.
NVF_API int nvf_download(void *handle, int slot, void *host) {
    Decoder *d = static_cast<Decoder *>(handle);
    uint8_t *planes[3];
    ptrdiff_t pitches[3];
    if (!held_planes(d, slot, planes, pitches)) return NVF_ERROR;
    if (is_scaled(d->params)) {
        scale_planar(d->params, d->scaler, planes, pitches, static_cast<uint8_t *>(host));
    } else {
        convert_planar(d->params, planes, pitches, static_cast<uint8_t *>(host));
    }
    return 0;
}

// The crop of the slot's picture where the decoder holds it, for a caller
// that reads it there rather than download it: each plane's first sample and
// the bytes from one row to the next, valid until nvf_release. Only for a
// picture nvf_download would only copy (convert_planar's memcpy: neither
// scaled nor widened, 10-bit samples kept in the low bits); NVF_ERROR
// otherwise. The app's gpu_frames.cpp can_lend_planes has the same rule.
NVF_API int nvf_slot_planes(void *handle, int slot, const void **planes, long long *pitches) {
    Decoder *d = static_cast<Decoder *>(handle);
    const Params &p = d->params;
    const bool copied_as_is = !is_scaled(p) && !p.widen && (p.bit_depth == 8 || p.shift != 0);
    uint8_t *base[3];
    ptrdiff_t pitch[3];
    if (!copied_as_is || !held_planes(d, slot, base, pitch)) return NVF_ERROR;
    const ptrdiff_t sample = p.bit_depth > 8 ? 2 : 1;
    for (int c = 0; c < 3; c++) {
        const ptrdiff_t x = c ? p.crop_x / 2 : p.crop_x, y = c ? p.crop_y / 2 : p.crop_y;
        planes[c] = base[c] + y * pitch[c] + x * sample;
        pitches[c] = pitch[c];
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
