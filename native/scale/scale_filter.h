// The filters the GPU frame decoders scale pictures with, the same for every
// GPU maker. NVIDIA's applies them on the GPU (nvdec_frames.cpp), and so do
// Intel's and AMD's (d3d11_scale.h, a shader that gives scale_plane's picture
// below, which they check it against and fall back to).
//
// A picture scaled here is FFmpeg's scale filter's to the sample: this is
// FFmpeg 9's libswscale arithmetic (LGPL 2.1 or later; initFilter, the
// hScale and yuv2planeX functions) for a 4:2:0 picture scaled to the same
// format, as its x86 builds run it, which give the same pictures with any of
// SSE2 to AVX2. Checked against ffmpeg.exe 9.0.1 (gyan.dev) on 15 sizes
// (odd ones, width or height alone, up and down to 6x), 8 and 10 bits, all
// four scalers: identical. Its scale filter scales chroma centred whatever
// the video's chroma siting (checked on a left-sited HEVC picture), as here.
//
// Each output sample is two passes, as swscale makes them: along the row
// into a 15-bit intermediate, then down the column. The filters are
// swscale's integer weights: 14-bit along the rows, 12-bit down the columns.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

enum Scaler { SCALER_BILINEAR = 0, SCALER_BICUBIC = 1, SCALER_LANCZOS = 2, SCALER_SPLINE = 3 };

// How 8-bit samples are made 10-bit before they are scaled, as FFmpeg's
// format filter makes them in front of its scale filter: v << 2, or with the
// top bits repeated (full-range luma).
enum Widen { WIDEN_NONE = 0, WIDEN_SHIFT = 1, WIDEN_REPEAT = 2 };

// One direction of one plane: output sample i is the weighted sum of `taps`
// input samples from starts[i] on. taps 0: swscale would scale this in two
// steps (a very large downscale), which is not done here.
struct Filter {
    int taps = 0;
    std::vector<int32_t> starts;   // per output sample
    std::vector<int16_t> weights;  // per output sample, `taps` each
};

// swscale's getSplineCoeff: a cubic per unit interval, continued smoothly.
inline double spline_coefficient(double a, double b, double c, double d, double distance) {
    if (distance <= 1.0) return ((d * distance + c) * distance + b) * distance + a;
    return spline_coefficient(0.0, b + 2.0 * c + 3.0 * d, c + 3.0 * d, -b - 3.0 * c - 6.0 * d, distance - 1.0);
}

// swscale's raw weight of a tap `d` from the output sample's centre (1 << 30
// per input sample, already widened by the ratio where scaling down), in
// units of `one` (fone): Mitchell-Netravali with B = 0, C = 0.6 (swscale's
// bicubic), a tent, Lanczos with 3 lobes, or swscale's natural cubic spline.
inline int64_t raw_weight(int scaler, int64_t d, int64_t one) {
    const double distance = static_cast<double>(d) * (1.0 / (1 << 30));
    switch (scaler) {
    case SCALER_BILINEAR: {
        const int64_t weight = (1 << 30) - d;
        return weight < 0 ? 0 : weight * (one >> 30);
    }
    case SCALER_LANCZOS: {
        const double pi = 3.14159265358979323846, lobes = 3.0;
        if (distance > lobes) return 0;
        return static_cast<int64_t>(
            (d ? std::sin(distance * pi) * std::sin(distance * pi / lobes) / (distance * distance * pi * pi / lobes)
               : 1.0) *
            static_cast<double>(one));
    }
    case SCALER_SPLINE: {
        const double p = -2.196152422706632;  // swscale's
        return static_cast<int64_t>(spline_coefficient(1.0, 0.0, p, -p - 1.0, distance) * static_cast<double>(one));
    }
    default: {
        const int64_t b = 0, c = static_cast<int64_t>(0.6 * (1 << 24));
        if (d >= 1LL << 31) return 0;
        const int64_t dd = (d * d) >> 30, ddd = (dd * d) >> 30;
        int64_t weight;
        if (d < 1LL << 30)
            weight = (12 * (1 << 24) - 9 * b - 6 * c) * ddd + (-18 * (1 << 24) + 12 * b + 6 * c) * dd +
                     (6 * (1 << 24) - 2 * b) * (1 << 30);
        else
            weight = (-b - 6 * c) * ddd + (6 * b + 30 * c) * dd + (-12 * b - 48 * c) * d + (8 * b + 24 * c) * (1 << 30);
        return weight / ((1LL << 54) / one);
    }
    }
}

// The filter from `src` samples to `dst`, as swscale's initFilter makes it
// for a scale filter with no extra source or destination filter on x86:
// along the rows (`vertical` false) with 14-bit weights and taps a multiple
// of 4, down the columns with 12-bit weights and taps a multiple of 2 (or a
// single tap). Every plane is sited centred, as the scale filter sites them.
inline Filter plane_filter(int src, int dst, int scaler, bool vertical) {
    Filter out;
    if (src < 1 || dst < 1) return out;
    const int one = vertical ? 1 << 12 : 1 << 14;
    int align = vertical ? 2 : 4;
    const int64_t step = ((static_cast<int64_t>(src) << 16) + (dst >> 1)) / dst;
    int log2_ratio = 0;
    for (unsigned ratio = static_cast<unsigned>(src / dst); ratio >>= 1;) log2_ratio++;
    const int64_t fone = 1LL << (54 - std::min(log2_ratio, 8));
    const int position = 128;  // the centre, in 1/256 of a sample

    std::vector<int32_t> starts(dst);
    std::vector<int64_t> weights;
    int taps;
    if (std::llabs(step - 0x10000) < 10) {  // the same size: one tap
        taps = 1;
        weights.assign(dst, fone);
        for (int i = 0; i < dst; i++) starts[i] = i;
    } else {
        const int size_factor = scaler == SCALER_BICUBIC ? 4 : scaler == SCALER_BILINEAR ? 2 : scaler == SCALER_SPLINE ? 20 : 6;
        taps = step <= 1 << 16 ? 1 + size_factor : 1 + (size_factor * src + dst - 1) / dst;
        taps = std::max(std::min(taps, src - 2), 1);
        weights.assign(static_cast<size_t>(dst) * taps, 0);
        int64_t centre = ((position * step) >> 7) - ((position * 0x10000LL) >> 7);
        for (int i = 0; i < dst; i++) {
            int first = static_cast<int>((centre - (taps - 2) * (1LL << 16)) / (1 << 17));
            starts[i] = first;
            for (int j = 0; j < taps; j++, first++) {
                int64_t d = std::llabs(static_cast<int64_t>(first) * (1 << 17) - centre) << 13;
                if (step > 1 << 16) d = d * dst / src;
                weights[static_cast<size_t>(i) * taps + j] = raw_weight(scaler, d, fone);
            }
            centre += 2 * step;
        }
    }

    // Near-zero weights dropped from the left of each output sample's, and
    // the fewest taps that keep every sample's significant ones.
    const double cut_off = 0.002 * static_cast<double>(fone);  // swscale's SWS_MAX_REDUCE_CUTOFF
    int needed = 0;
    for (int i = dst - 1; i >= 0; i--) {
        int64_t *row = weights.data() + static_cast<size_t>(i) * taps;
        int64_t sum = 0;
        for (int j = 0; j < taps; j++) {
            sum += std::llabs(row[0]);
            if (static_cast<double>(sum) > cut_off || (i < dst - 1 && starts[i] >= starts[i + 1])) break;
            std::copy(row + 1, row + taps, row);
            row[taps - 1] = 0;
            starts[i]++;
        }
        int kept = taps;
        sum = 0;
        for (int j = taps - 1; j > 0; j--) {
            sum += std::llabs(row[j]);
            if (static_cast<double>(sum) > cut_off) break;
            kept--;
        }
        needed = std::max(needed, kept);
    }
    if (needed == 1 && align == 2) align = 1;
    const int size = (needed + align - 1) & ~(align - 1);
    if (size >= 256) return out;  // swscale's MAX_FILTER_SIZE: it would cascade
    std::vector<int64_t> sized(static_cast<size_t>(dst) * size, 0);
    for (int i = 0; i < dst; i++)
        for (int j = 0; j < size && j < taps; j++)
            sized[static_cast<size_t>(i) * size + j] = weights[static_cast<size_t>(i) * taps + j];

    // At the edges, the weights of samples outside the picture move onto
    // its first or last sample.
    for (int i = 0; i < dst; i++) {
        int64_t *row = sized.data() + static_cast<size_t>(i) * size;
        if (starts[i] < 0) {
            for (int j = 1; j < size; j++) {
                const int left = std::max(j + starts[i], 0);
                row[left] += row[j];
                row[j] = 0;
            }
            starts[i] = 0;
        }
        if (starts[i] + size > src) {
            const int shift = starts[i] + std::min(size - src, 0);
            int64_t outside = 0;
            for (int j = size - 1; j >= 0; j--) {
                if (starts[i] + j >= src) {
                    outside += row[j];
                    row[j] = 0;
                }
            }
            for (int j = size - 1; j >= 0; j--) row[j] = j < shift ? 0 : row[j - shift];
            starts[i] -= shift;
            row[src - 1 - starts[i]] += outside;
        }
    }

    // Normalised to `one`, each weight rounded with the error carried on.
    out.taps = size;
    out.starts = std::move(starts);
    out.weights.assign(static_cast<size_t>(dst) * size, 0);
    for (int i = 0; i < dst; i++) {
        const int64_t *row = sized.data() + static_cast<size_t>(i) * size;
        int64_t sum = 0, error = 0;
        for (int j = 0; j < size; j++) sum += row[j];
        sum = (sum + one / 2) / one;
        if (!sum) sum = 1;
        for (int j = 0; j < size; j++) {
            const int64_t v = row[j] + error;
            const int64_t rounded = (v >= 0 ? v + (sum >> 1) : v - (sum >> 1)) / sum;
            out.weights[static_cast<size_t>(i) * size + j] = static_cast<int16_t>(rounded);
            error = v - rounded * sum;
        }
    }
    return out;
}

// The output rows of a plane that swscale's SSE/AVX column pass makes
// (8-bit pictures): all but those it leaves to its C code, the picture's
// last two luma rows and the chroma rows made with them. `luma_rows` is the
// scaled picture's height.
inline int simd_rows(int luma_rows, bool chroma) { return chroma ? (luma_rows - 1) / 2 : luma_rows - 2; }

// One sample of the row pass: the weighted sum of 8-bit samples (or
// widened, or 10-bit ones), shifted to swscale's 15-bit intermediate.
inline int16_t row_sum(int64_t sum, bool ten_bit) {
    const int64_t value = sum >> (ten_bit ? 9 : 7);
    return static_cast<int16_t>(value < 32767 ? value : 32767);
}

// One sample of the column pass from `taps` intermediate values, as swscale
// writes it: 8-bit through its SSE/AVX code (`simd`: per-tap products' top
// 16 bits, summed in 16 bits, with its rounding) or its C code, or 10-bit.
template <typename Value>
inline int column_sample(const int16_t *weights, int taps, Value value, bool ten_bit, bool simd) {
    if (ten_bit) {
        int result;
        if (taps == 1) {
            result = (value(0) + 16) >> 5;
        } else {
            int sum = 1 << 16;
            for (int k = 0; k < taps; k++) sum += value(k) * weights[k];
            result = sum >> 17;
        }
        return std::clamp(result, 0, 1023);
    }
    int result;
    if (taps == 1) {
        result = (value(0) + 64) >> 7;
    } else if (simd) {
        int16_t sum = static_cast<int16_t>((((taps - 1) << 3) + 64) >> 4);
        for (int k = 0; k < taps; k++) sum = static_cast<int16_t>(sum + ((weights[k] * value(k)) >> 16));
        result = sum >> 3;
    } else {
        int sum = 64 << 12;
        for (int k = 0; k < taps; k++) sum += value(k) * weights[k];
        result = sum >> 19;
    }
    return std::clamp(result, 0, 255);
}

// The CPU scaler's working rows, kept between pictures.
struct ScaleScratch {
    std::vector<int16_t> rows;
};

// Scales one plane on the CPU: `in` samples (8-bit, or 16-bit shifted right
// by `in_shift`), `step` bytes apart in rows `pitch` bytes apart, `w` x `h`,
// 8-bit ones widened first as `widen` says, to `out` (`ow` x `oh`, packed),
// shifted left by `out_shift`; 8- or 16-bit samples as the scale is (10-bit
// for 16-bit or widened input). `simd` is simd_rows for the plane.
inline void scale_plane(const uint8_t *in, size_t pitch, size_t step, bool in_wide, int in_shift, int widen, int w,
                        int h, const Filter &fh, const Filter &fv, int simd, uint8_t *out, int ow, int oh,
                        int out_shift, ScaleScratch &scratch) {
    const bool ten_bit = in_wide || widen != WIDEN_NONE;
    scratch.rows.resize(static_cast<size_t>(ow) * h);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = in + y * pitch;
        int16_t *dst = scratch.rows.data() + static_cast<size_t>(y) * ow;
        for (int x = 0; x < ow; x++) {
            const int16_t *weight = fh.weights.data() + static_cast<size_t>(x) * fh.taps;
            int64_t sum = 0;
            for (int k = 0; k < fh.taps; k++) {
                const int at = std::min(fh.starts[x] + k, w - 1);
                int sample;
                if (in_wide) {
                    sample = *reinterpret_cast<const uint16_t *>(row + at * step) >> in_shift;
                } else {
                    sample = row[at * step];
                    if (widen != WIDEN_NONE) sample = (sample << 2) | (widen == WIDEN_REPEAT ? sample >> 6 : 0);
                }
                sum += static_cast<int64_t>(sample) * weight[k];
            }
            dst[x] = row_sum(sum, ten_bit);
        }
    }
    for (int y = 0; y < oh; y++) {
        const int16_t *weight = fv.weights.data() + static_cast<size_t>(y) * fv.taps;
        const int start = fv.starts[y];
        for (int x = 0; x < ow; x++) {
            auto value = [&](int k) {
                return static_cast<int>(scratch.rows[static_cast<size_t>(std::min(start + k, h - 1)) * ow + x]);
            };
            const unsigned sample = static_cast<unsigned>(column_sample(weight, fv.taps, value, ten_bit, y < simd))
                                    << out_shift;
            const size_t i = static_cast<size_t>(y) * ow + x;
            if (ten_bit) {
                reinterpret_cast<uint16_t *>(out)[i] = static_cast<uint16_t>(sample);
            } else {
                out[i] = static_cast<uint8_t>(sample);
            }
        }
    }
}
