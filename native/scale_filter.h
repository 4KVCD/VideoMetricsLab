// The filters the GPU frame decoders scale pictures with -- the same for every
// GPU maker: NVIDIA's applies them on the GPU (nvdec_frames.cpp), Intel's and
// AMD's on the CPU (scale_plane below).
//
// A picture scaled here is not FFmpeg's scale filter's to the sample (the
// user decided, 2026-10-03, that the algorithm and where it runs make the same
// comparison: ComparisonRecipe.identity_dict), but it is the same filter:
// FFmpeg's bicubic (Mitchell-Netravali with B = 0, C = 0.6, swscale's
// default), bilinear, Lanczos with 3 lobes, or swscale's natural cubic spline.
// Downscaling widens the filter by the ratio, as swscale does, so nothing
// aliases. Each output sample is the weighted sum of `taps` input samples
// from `starts[i]` on, edges repeated; weights are normalised to 1.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

enum Scaler { SCALER_BILINEAR = 0, SCALER_BICUBIC = 1, SCALER_LANCZOS = 2, SCALER_SPLINE = 3 };

struct Filter {
    int taps = 0;
    std::vector<int32_t> starts;  // per output sample
    std::vector<float> weights;   // per output sample, `taps` each
};

inline double filter_radius(int scaler) {
    // swscale's spline reaches 10 samples out (its "sizeFactor" of 20).
    return scaler == SCALER_BILINEAR ? 1.0 : scaler == SCALER_LANCZOS ? 3.0 : scaler == SCALER_SPLINE ? 10.0 : 2.0;
}

// swscale's getSplineCoeff: a cubic per unit interval, continued smoothly.
inline double spline_coefficient(double a, double b, double c, double d, double distance) {
    while (distance > 1.0) {
        const double next_b = b + 2.0 * c + 3.0 * d, next_c = c + 3.0 * d, next_d = -b - 3.0 * c - 6.0 * d;
        a = 0.0;
        b = next_b;
        c = next_c;
        d = next_d;
        distance -= 1.0;
    }
    return ((d * distance + c) * distance + b) * distance + a;
}

inline double filter_kernel(int scaler, double x) {
    x = std::fabs(x);
    switch (scaler) {
    case SCALER_BILINEAR:
        return x < 1.0 ? 1.0 - x : 0.0;
    case SCALER_LANCZOS: {
        if (x < 1e-9) return 1.0;
        if (x >= 3.0) return 0.0;
        const double pi = 3.14159265358979323846;
        return 3.0 * std::sin(pi * x) * std::sin(pi * x / 3.0) / (pi * pi * x * x);
    }
    case SCALER_SPLINE: {
        const double p = -2.196152422706632;  // swscale's
        return x >= 10.0 ? 0.0 : spline_coefficient(1.0, 0.0, p, -p - 1.0, x);
    }
    default: {
        // Mitchell-Netravali; swscale's bicubic is B = 0, C = 0.6.
        const double b = 0.0, c = 0.6;
        if (x < 1.0) return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) / 6.0;
        if (x < 2.0)
            return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6.0;
        return 0.0;
    }
    }
}

// The filter from `src` samples to `dst`. `center(i)` is output sample i's
// position in input samples.
template <typename Center>
Filter make_filter(int src, int dst, int scaler, Center center) {
    Filter f;
    const double ratio = static_cast<double>(src) / dst;
    const double widen = ratio > 1.0 ? ratio : 1.0;  // downscaling: as wide as the step
    const double support = filter_radius(scaler) * widen;
    f.taps = static_cast<int>(std::ceil(2.0 * support)) + 1;
    f.starts.resize(dst);
    f.weights.assign(static_cast<size_t>(dst) * f.taps, 0.0f);
    for (int i = 0; i < dst; i++) {
        const double c = center(i);
        const int start = static_cast<int>(std::floor(c - support)) + 1;
        f.starts[i] = start;
        double sum = 0.0;
        std::vector<double> w(f.taps);
        for (int k = 0; k < f.taps; k++) {
            w[k] = filter_kernel(scaler, (start + k - c) / widen);
            sum += w[k];
        }
        for (int k = 0; k < f.taps; k++) {
            f.weights[static_cast<size_t>(i) * f.taps + k] = static_cast<float>(sum != 0.0 ? w[k] / sum : (k == 0));
        }
    }
    return f;
}

// One plane's filter: sample centres map centre to centre. 4:2:0 chroma
// planes are scaled as planes of their own, from their size to theirs, as
// FFmpeg's scale filter scales them: the pictures come out within 1 of
// FFmpeg's in every plane (measured: 4K to 1080p at 10 bits, 80 dB; 1080p
// to 4K at 8 bits, 64-67 dB). Placing chroma where H.264 sites it instead,
// left with the even luma column, put U and V up to 20 apart.
inline Filter plane_filter(int src, int dst, int scaler) {
    const double ratio = static_cast<double>(src) / dst;
    return make_filter(src, dst, scaler, [ratio](int i) { return (i + 0.5) * ratio - 0.5; });
}

// Between the passes a filtered value is capped where swscale's 15-bit
// intermediate saturates: 32767 / 128 for 8-bit samples, / 32 for 10-bit. A
// filter that rings past white (Lanczos, the spline) on a hard edge is
// clipped there by FFmpeg before its second pass: kept, it put U and V up to
// 25 apart on a synthetic test picture (2 to 3 with the cap).
constexpr float kIntermediateCap8 = 32767.0f / 128.0f, kIntermediateCap10 = 32767.0f / 32.0f;

// The CPU scaler's working planes, kept between pictures.
struct ScaleScratch {
    std::vector<float> plane, first, row;
};

// Each output row of `rows` (`height` rows of `width` floats) is the
// weighted sum of the input rows `f` names, whole rows at a time -- a loop
// the compiler vectorises.
inline void vertical(const float *rows, int width, int height, const Filter &f, int out_rows, float *out) {
    for (int y = 0; y < out_rows; y++) {
        float *acc = out + static_cast<size_t>(y) * width;
        std::fill(acc, acc + width, 0.0f);
        const float *wt = f.weights.data() + static_cast<size_t>(y) * f.taps;
        for (int k = 0; k < f.taps; k++) {
            int r = f.starts[y] + k;
            r = r < 0 ? 0 : r >= height ? height - 1 : r;
            const float w = wt[k];
            const float *src = rows + static_cast<size_t>(r) * width;
            for (int x = 0; x < width; x++) acc[x] += w * src[x];
        }
    }
}

// Each row filtered along itself, from `width` samples to `out_width`; the
// row is copied with its edges repeated first, so no tap is clamped.
inline void horizontal(const float *rows, int width, int height, const Filter &f, int out_width, float *out,
                       std::vector<float> &padded) {
    const int pad = f.taps + 1;
    padded.resize(static_cast<size_t>(width) + 2 * pad);
    for (int y = 0; y < height; y++) {
        const float *src = rows + static_cast<size_t>(y) * width;
        std::fill(padded.begin(), padded.begin() + pad, src[0]);
        std::copy(src, src + width, padded.begin() + pad);
        std::fill(padded.begin() + pad + width, padded.end(), src[width - 1]);
        float *dst = out + static_cast<size_t>(y) * out_width;
        for (int x = 0; x < out_width; x++) {
            int start = f.starts[x] + pad;
            start = start < 0 ? 0 : start + f.taps > static_cast<int>(padded.size()) ? static_cast<int>(padded.size()) - f.taps : start;
            const float *taps = padded.data() + start;
            const float *wt = f.weights.data() + static_cast<size_t>(x) * f.taps;
            float acc = 0.0f;
            for (int k = 0; k < f.taps; k++) acc += wt[k] * taps[k];
            dst[x] = acc;
        }
    }
}

// Scales one plane on the CPU: `in` samples (8-bit, or 16-bit shifted right
// by `in_shift`), `step` bytes apart in rows `pitch` bytes apart, `w` x `h`,
// to `out` (`ow` x `oh`, packed), each value times `gain`, rounded, clamped
// to `max`, then shifted left by `out_shift` -- 8- or 16-bit as `out_wide`.
// The plane is made floats once; the vertical pass, which runs a whole row
// at a time, goes first where it shrinks the picture, so the horizontal
// pass, one sample at a time, has the fewer rows (about 15 ms for a 4K
// 10-bit picture to 1080p, against 44 ms one sample at a time). The cap
// between the passes is applied after whichever comes first.
inline void scale_plane(const uint8_t *in, size_t pitch, size_t step, bool in_wide, int in_shift, int w, int h,
                        const Filter &fh, const Filter &fv, uint8_t *out, int ow, int oh, bool out_wide,
                        float gain, float max, int out_shift, ScaleScratch &scratch) {
    scratch.plane.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = in + y * pitch;
        float *dst = scratch.plane.data() + static_cast<size_t>(y) * w;
        if (in_wide) {
            for (int x = 0; x < w; x++)
                dst[x] = static_cast<float>(*reinterpret_cast<const uint16_t *>(row + x * step) >> in_shift);
        } else {
            for (int x = 0; x < w; x++) dst[x] = static_cast<float>(row[x * step]);
        }
    }
    const float cap = in_wide ? kIntermediateCap10 : kIntermediateCap8;
    const float *result;
    if (oh <= h) {  // vertical first: from h rows to oh, then each to ow
        scratch.first.resize(static_cast<size_t>(w) * oh);
        vertical(scratch.plane.data(), w, h, fv, oh, scratch.first.data());
        for (float &v : scratch.first) v = v < cap ? v : cap;
        scratch.plane.resize(static_cast<size_t>(ow) * oh);
        horizontal(scratch.first.data(), w, oh, fh, ow, scratch.plane.data(), scratch.row);
        result = scratch.plane.data();
    } else {  // horizontal first: each of h rows to ow, then to oh rows
        scratch.first.resize(static_cast<size_t>(ow) * h);
        horizontal(scratch.plane.data(), w, h, fh, ow, scratch.first.data(), scratch.row);
        for (float &v : scratch.first) v = v < cap ? v : cap;
        scratch.plane.resize(static_cast<size_t>(ow) * oh);
        vertical(scratch.first.data(), ow, h, fv, oh, scratch.plane.data());
        result = scratch.plane.data();
    }
    const size_t count = static_cast<size_t>(ow) * oh;
    for (size_t i = 0; i < count; i++) {
        float v = std::nearbyint(result[i] * gain);
        v = v < 0.0f ? 0.0f : v > max ? max : v;
        unsigned value = static_cast<unsigned>(v) << out_shift;
        if (out_wide) {
            reinterpret_cast<uint16_t *>(out)[i] = static_cast<uint16_t>(value);
        } else {
            out[i] = static_cast<uint8_t>(value);
        }
    }
}
