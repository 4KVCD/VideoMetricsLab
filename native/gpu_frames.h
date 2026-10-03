// The C API every GPU frame decoder library of the app's exports
// (nvdec_frames.dll for NVIDIA, vpl_frames.dll for Intel, amf_frames.dll for
// AMD), so vmaf_app/core/nvdec_frames.py drives any of them the same way:
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

#define NVF_API extern "C" __declspec(dllexport)

enum Status { NVF_FRAME = 1, NVF_END = 0, NVF_ERROR = -1, NVF_ABORTED = -2, NVF_TIMEOUT = 2 };

// The codecs, by NVIDIA's numbers (cudaVideoCodec), for every library.
enum Codec { CODEC_H264 = 4, CODEC_HEVC = 8, CODEC_AV1 = 11 };

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
    const unsigned char *extradata;  // AV1: the sequence header OBUs (may be null)
    int extradata_size;
};

struct Info {
    int coded_width, coded_height;
    int display_left, display_top, display_right, display_bottom;
    int bit_depth, chroma_format, progressive;
    int decode_surfaces;
    long long decoded, displayed, frame_bytes;
};

inline bool params_valid(const Params &p) {
    return (p.bit_depth == 8 || p.bit_depth == 10) && !(p.crop_x & 1) && !(p.crop_y & 1) && p.crop_w > 0
           && p.crop_h > 0 && p.pool >= 1 && (p.shift == 0 || p.shift == 6) && !(p.bit_depth == 8 && p.shift);
}

// The bytes of one picture handed back: the crop's luma, then U and V.
inline size_t frame_bytes(const Params &p) {
    size_t sample = p.bit_depth > 8 ? 2 : 1;
    size_t luma = static_cast<size_t>(p.crop_w) * p.crop_h * sample;
    if (p.luma_only) return luma;
    return luma + 2 * static_cast<size_t>((p.crop_w + 1) / 2) * ((p.crop_h + 1) / 2) * sample;
}

// Copies the crop of a decoded NV12/P010 picture -- the luma plane at `y`,
// U and V interleaved at `uv`, rows `pitch` bytes apart -- into `dst`, planes
// packed. 10-bit samples sit in the top bits of 16 when `msb`, else in the
// low ones; they are handed back in the top bits (shift 0, P016's layout,
// which Vship reads as 16-bit) or the low ones (shift 6, yuv420p10le). The
// samples are moved, never computed.
inline void convert_frame(const Params &p, const uint8_t *y, const uint8_t *uv, size_t pitch, bool msb,
                          uint8_t *dst) {
    const size_t w = p.crop_w, h = p.crop_h, cw = (p.crop_w + 1) / 2, ch = (p.crop_h + 1) / 2;
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
    for (size_t row = 0; row < h; row++) {
        const uint16_t *src = reinterpret_cast<const uint16_t *>(y + (p.crop_y + row) * pitch) + p.crop_x;
        uint16_t *o = out + row * w;
        if (down == 6 && up == 6) {
            memcpy(o, src, w * 2);
        } else {
            for (size_t i = 0; i < w; i++) o[i] = static_cast<uint16_t>((src[i] >> down) << up);
        }
    }
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
