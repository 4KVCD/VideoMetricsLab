// The C API every frame decoder library of the app's exports
// (nvdec_frames.dll for NVIDIA, vpl_frames.dll for Intel, amf_frames.dll for
// AMD, software_frames.dll for what FFmpeg decodes in software), so
// vmaf_app/core/gpu_frames.py drives any of them the same way:
//
//   nvf_open(params) -> decoder, nvf_push(packet, pts) ... nvf_finish(),
//   nvf_pop() -> (slot, pts) in display order, nvf_download(slot, host) or
//   nvf_copy_luma(slot, device memory), nvf_release(slot), nvf_abort(),
//   nvf_close(), nvf_error(), nvf_info(), nvf_supports().
//
// And the CPU conversion the Intel and AMD libraries hand a picture over
// with: their decoders give NV12/P010 in system memory, and the picture
// Vship is given is planar.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>

#include "scale_filter.h"

#define NVF_API extern "C" __declspec(dllexport)

enum Status { NVF_FRAME = 1, NVF_END = 0, NVF_ERROR = -1, NVF_ABORTED = -2, NVF_TIMEOUT = 2 };

// The codecs, by NVIDIA's numbers (cudaVideoCodec), for every library; VVC
// and FFV1, which NVIDIA's decoder does not have, by the app's. The GPU
// decoders are asked for H.264, HEVC and AV1 only (gpu_frames.py).
enum Codec {
    CODEC_MPEG2 = 1, CODEC_H264 = 4, CODEC_HEVC = 8, CODEC_VP9 = 10, CODEC_AV1 = 11, CODEC_VVC = 100, CODEC_FFV1 = 101
};

struct Params {
    int device;          // GPU: CUDA device ordinal (NVIDIA); unused elsewhere
    int codec;           // Codec
    int bit_depth;       // 8 or 10: what the stream must have
    int width, height;   // the displayed size the stream must have
    int crop_x, crop_y;  // even
    int crop_w, crop_h;  // the size of the pictures handed back
    int shift;           // right shift of 16-bit samples: 0 or 6
    int luma_only;       // only the Y plane is produced
    int pool;            // pictures decoded ahead
    const unsigned char *extradata;  // AV1: the sequence header OBUs; FFV1: its configuration (may be null)
    int extradata_size;
    int out_w, out_h;    // the crop scaled to this size (0: not scaled)
    int scaler;          // Scaler (scale_filter.h)
    int widen;           // 8-bit pictures handed back as 10-bit: 1 shifted, 2 with
                         // the luma's top bits repeated (full range), as FFmpeg widens
    int cpu_scaling;     // Intel, AMD: 1 scaled on the CPU even where the GPU can (to
                         // check one against the other); 2 for the tests: the GPU's
                         // first picture is spoiled, as a driver's wrong one would be
    int handover;        // AMD: the pictures stay on the GPU, for nvf_copy_luma,
                         // nvf_download_planes and nvf_import_vulkan (amf_handover.h); not
                         // scaled. NVIDIA's always do; unused elsewhere
};

inline int out_width(const Params &p) { return p.out_w > 0 ? p.out_w : p.crop_w; }
inline int out_height(const Params &p) { return p.out_h > 0 ? p.out_h : p.crop_h; }
inline bool is_scaled(const Params &p) { return out_width(p) != p.crop_w || out_height(p) != p.crop_h; }
// 16-bit samples handed back: 10-bit pictures, or widened 8-bit ones.
inline bool wide_out(const Params &p) { return p.bit_depth > 8 || p.widen; }

struct Info {
    int coded_width, coded_height;
    int display_left, display_top, display_right, display_bottom;
    int bit_depth, chroma_format, progressive;
    int decode_surfaces;
    long long decoded, displayed, frame_bytes;
    int scaled_on_gpu;   // the pictures are being scaled on the GPU
};

inline bool params_valid(const Params &p) {
    return (p.bit_depth == 8 || p.bit_depth == 10) && !(p.crop_x & 1) && !(p.crop_y & 1) && p.crop_w > 0
           && p.crop_h > 0 && p.pool >= 1 && (p.shift == 0 || p.shift == 6) && !(p.bit_depth == 8 && p.shift)
           && p.out_w >= 0 && p.out_h >= 0 && p.scaler >= 0 && p.scaler <= 3 && p.widen >= 0 && p.widen <= 2
           && !(p.widen && p.bit_depth != 8);
}

// The bytes of one picture handed back: the (scaled) crop's luma, then U and V.
inline size_t frame_bytes(const Params &p) {
    size_t sample = wide_out(p) ? 2 : 1;
    size_t w = out_width(p), h = out_height(p);
    size_t luma = w * h * sample;
    if (p.luma_only) return luma;
    return luma + 2 * ((w + 1) / 2) * ((h + 1) / 2) * sample;
}

// One row of 16-bit samples shifted from the decoder's alignment to the one
// handed back, 8 at a time. Into a row 16-byte aligned (libvmaf's pictures
// are) by streaming stores, which do not read the destination into the cache
// first: a 4K picture (16 MB) is out of the cache before it is scored, and
// on an iGPU the CPU's reads compete with the decoder's for the same memory.
// True when it streamed: the caller fences once it is done.
inline bool shift_row(const uint16_t *src, uint16_t *dst, size_t w, int down, int up) {
    const bool stream = (reinterpret_cast<uintptr_t>(dst) & 15) == 0;
    const __m128i right = _mm_cvtsi32_si128(down), left = _mm_cvtsi32_si128(up);
    size_t i = 0;
    for (; i + 8 <= w; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
        v = _mm_sll_epi16(_mm_srl_epi16(v, right), left);
        if (stream)
            _mm_stream_si128(reinterpret_cast<__m128i *>(dst + i), v);
        else
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), v);
    }
    for (; i < w; i++) dst[i] = static_cast<uint16_t>((src[i] >> down) << up);
    return stream;
}

// Copies the crop of a decoded NV12/P010 picture -- the luma plane at `y`,
// U and V interleaved at `uv`, rows `pitch` bytes apart -- into `dst`, planes
// packed. 10-bit samples sit in the top bits of 16 when `msb`, else in the
// low ones; they are handed back in the top bits (shift 0, P016's layout,
// which Vship reads as 16-bit) or the low ones (shift 6, yuv420p10le). The
// samples are moved, never computed -- but for 8-bit pictures widened to 10
// (Params::widen), which are made as FFmpeg makes them and NVIDIA's decoder
// here does (its widen8 kernel): v << 2, in the low bits of 16, the luma's
// top two bits repeated below when the video is full range.
inline void convert_frame(const Params &p, const uint8_t *y, const uint8_t *uv, size_t pitch, bool msb,
                          uint8_t *dst) {
    const size_t w = p.crop_w, h = p.crop_h, cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
    if (p.bit_depth == 8 && p.widen) {
        uint16_t *wide = reinterpret_cast<uint16_t *>(dst);
        const bool repeat = p.widen == 2;
        for (size_t row = 0; row < h; row++) {
            const uint8_t *src = y + (p.crop_y + row) * pitch + p.crop_x;
            uint16_t *o = wide + row * w;
            if (repeat) {
                for (size_t i = 0; i < w; i++) o[i] = static_cast<uint16_t>((src[i] << 2) | (src[i] >> 6));
            } else {
                for (size_t i = 0; i < w; i++) o[i] = static_cast<uint16_t>(src[i] << 2);
            }
        }
        if (p.luma_only) return;
        uint16_t *u = wide + w * h, *v = u + cw * ch;
        for (size_t row = 0; row < ch; row++) {
            const uint8_t *src = uv + (p.crop_y / 2 + row) * pitch + p.crop_x;
            uint16_t *ur = u + row * cw, *vr = v + row * cw;
            for (size_t i = 0; i < cw; i++) {
                ur[i] = static_cast<uint16_t>(src[2 * i] << 2);
                vr[i] = static_cast<uint16_t>(src[2 * i + 1] << 2);
            }
        }
        return;
    }
    if (p.bit_depth == 8) {
        for (size_t row = 0; row < h; row++)
            memcpy(dst + row * w, y + (p.crop_y + row) * pitch + p.crop_x, w);
        if (p.luma_only) return;
        uint8_t *u = dst + w * h, *v = u + cw * ch;
        for (size_t row = 0; row < ch; row++) {
            const uint8_t *src = uv + (p.crop_y / 2 + row) * pitch + p.crop_x;
            uint8_t *ur = u + row * cw, *vr = v + row * cw;
            for (size_t i = 0; i < cw; i++) {
                ur[i] = src[2 * i];
                vr[i] = src[2 * i + 1];
            }
        }
        return;
    }
    // 16-bit samples: from the decoder's alignment to the one handed back.
    const int down = msb ? 6 : 0, up = p.shift ? 0 : 6;
    uint16_t *out = reinterpret_cast<uint16_t *>(dst);
    bool streamed = false;
    for (size_t row = 0; row < h; row++) {
        const uint16_t *src = reinterpret_cast<const uint16_t *>(y + (p.crop_y + row) * pitch) + p.crop_x;
        uint16_t *o = out + row * w;
        if (down == 6 && up == 6) {
            memcpy(o, src, w * 2);
        } else {
            streamed |= shift_row(src, o, w, down, up);
        }
    }
    if (streamed) _mm_sfence();  // the streamed rows are written before another thread reads them
    if (p.luma_only) return;
    uint16_t *u = out + w * h, *v = u + cw * ch;
    for (size_t row = 0; row < ch; row++) {
        const uint16_t *src = reinterpret_cast<const uint16_t *>(uv + (p.crop_y / 2 + row) * pitch) + p.crop_x;
        uint16_t *ur = u + row * cw, *vr = v + row * cw;
        for (size_t i = 0; i < cw; i++) {
            ur[i] = static_cast<uint16_t>((src[2 * i] >> down) << up);
            vr[i] = static_cast<uint16_t>((src[2 * i + 1] >> down) << up);
        }
    }
}

// Scaling on the CPU, for Intel's and AMD's decoders: what their GPU scaling
// (d3d11_scale.h) is checked against, and what scales where the GPU cannot.
// The filters of scale_filter.h, made once.
struct PlaneScaler {
    Filter luma_h, luma_v, chroma_h, chroma_v;
    ScaleScratch scratch;
};

inline void prepare_scaler(const Params &p, PlaneScaler &s) {
    const int ow = out_width(p), oh = out_height(p);
    s.luma_h = plane_filter(p.crop_w, ow, p.scaler);
    s.luma_v = plane_filter(p.crop_h, oh, p.scaler);
    s.chroma_h = plane_filter((p.crop_w + 1) / 2, (ow + 1) / 2, p.scaler);
    s.chroma_v = plane_filter((p.crop_h + 1) / 2, (oh + 1) / 2, p.scaler);
}

// convert_frame, scaled to the output size on the way. A widened 8-bit
// picture is the filtered value times 4 (full-range luma: to 1023 for 255),
// as NVIDIA's decoder here scales and widens.
inline void scale_frame(const Params &p, PlaneScaler &s, const uint8_t *y, const uint8_t *uv, size_t pitch, bool msb,
                        uint8_t *dst) {
    const bool wide = p.bit_depth > 8, out_wide = wide_out(p);
    const size_t bps = wide ? 2 : 1, out_bps = out_wide ? 2 : 1;
    const int in_shift = wide && msb ? 6 : 0, out_shift = wide && p.shift == 0 ? 6 : 0;
    const float max = out_wide ? 1023.0f : 255.0f;
    const float luma_gain = p.widen == 2 ? 1023.0f / 255.0f : p.widen ? 4.0f : 1.0f;
    const float chroma_gain = p.widen ? 4.0f : 1.0f;
    const int ow = out_width(p), oh = out_height(p), ocw = (ow + 1) / 2, och = (oh + 1) / 2;
    const int cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
    scale_plane(y + p.crop_y * pitch + p.crop_x * bps, pitch, bps, wide, in_shift, p.crop_w, p.crop_h, s.luma_h,
                s.luma_v, dst, ow, oh, out_wide, luma_gain, max, out_shift, s.scratch);
    if (p.luma_only) return;
    const uint8_t *chroma = uv + (p.crop_y / 2) * pitch + p.crop_x * bps;
    uint8_t *u = dst + static_cast<size_t>(ow) * oh * out_bps, *v = u + static_cast<size_t>(ocw) * och * out_bps;
    scale_plane(chroma, pitch, 2 * bps, wide, in_shift, cw, ch, s.chroma_h, s.chroma_v, u, ocw, och, out_wide,
                chroma_gain, max, out_shift, s.scratch);
    scale_plane(chroma + bps, pitch, 2 * bps, wide, in_shift, cw, ch, s.chroma_h, s.chroma_v, v, ocw, och, out_wide,
                chroma_gain, max, out_shift, s.scratch);
}
