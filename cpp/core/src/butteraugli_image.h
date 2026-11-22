// The small part of libjxl's image API that the Butteraugli port
// (butteraugli_kernels.cpp, butteraugli.cpp) is written against: float
// planes whose rows are 64-byte aligned and padded to a whole number of
// vectors, as libjxl's ImageF rows are, so that its vector loops can run past
// the last pixel of a row without leaving it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "row_pool.h"

namespace vml::butteraugli {

// Floats per row are rounded up to this (64 bytes, one AVX-512 vector), and
// one more vector is added so that unaligned loads just past the last pixel
// stay inside the row.
inline constexpr std::size_t kRowAlign = 16;

class ImageF {
public:
    ImageF() = default;
    ImageF(std::size_t xsize, std::size_t ysize) { allocate(xsize, ysize); }
    ImageF(ImageF&&) noexcept = default;
    ImageF& operator=(ImageF&&) noexcept = default;

    std::size_t xsize() const { return xsize_; }
    std::size_t ysize() const { return ysize_; }
    std::size_t PixelsPerRow() const { return stride_; }

    float* Row(std::size_t y) { return data_.get() + y * stride_; }
    const float* Row(std::size_t y) const { return data_.get() + y * stride_; }
    const float* ConstRow(std::size_t y) const { return Row(y); }

    // This size, in the plane's buffer when it is large enough (so that the
    // half-resolution comparison reuses the full one's planes), else in a new
    // one. The contents are left as they are, as libjxl's are undefined after
    // Create: every user writes a plane before reading it.
    void ensure(std::size_t xsize, std::size_t ysize) {
        if (xsize == xsize_ && ysize == ysize_) return;
        const std::size_t stride = row_floats(xsize);
        if (stride * std::max<std::size_t>(ysize, 1) > capacity_) {
            allocate(xsize, ysize);
            return;
        }
        xsize_ = xsize;
        ysize_ = ysize;
        stride_ = stride;
    }

    void zero() {
        if (data_) std::fill(data_.get(), data_.get() + stride_ * ysize_, 0.0f);
    }

private:
    struct AlignedFree {
        void operator()(float* p) const { ::operator delete[](p, std::align_val_t{64}); }
    };

    static std::size_t row_floats(std::size_t xsize) { return ((xsize + kRowAlign - 1) / kRowAlign + 1) * kRowAlign; }

    void allocate(std::size_t xsize, std::size_t ysize) {
        xsize_ = xsize;
        ysize_ = ysize;
        stride_ = row_floats(xsize);
        capacity_ = stride_ * std::max<std::size_t>(ysize, 1);
        data_.reset(static_cast<float*>(::operator new[](capacity_ * sizeof(float), std::align_val_t{64})));
        // Zeroed so that the padding never holds NaNs.
        std::fill(data_.get(), data_.get() + capacity_, 0.0f);
    }

    std::unique_ptr<float[], AlignedFree> data_;
    std::size_t xsize_ = 0;
    std::size_t ysize_ = 0;
    std::size_t stride_ = 0;
    std::size_t capacity_ = 0;  // floats in data_
};

class Image3F {
public:
    Image3F() = default;
    Image3F(std::size_t xsize, std::size_t ysize)
        : planes_{ImageF(xsize, ysize), ImageF(xsize, ysize), ImageF(xsize, ysize)} {}

    std::size_t xsize() const { return planes_[0].xsize(); }
    std::size_t ysize() const { return planes_[0].ysize(); }

    ImageF& Plane(std::size_t c) { return planes_[c]; }
    const ImageF& Plane(std::size_t c) const { return planes_[c]; }
    float* PlaneRow(std::size_t c, std::size_t y) { return planes_[c].Row(y); }
    const float* PlaneRow(std::size_t c, std::size_t y) const { return planes_[c].Row(y); }
    const float* ConstPlaneRow(std::size_t c, std::size_t y) const { return planes_[c].Row(y); }

    void ensure(std::size_t xsize, std::size_t ysize) {
        for (auto& plane : planes_) plane.ensure(xsize, ysize);
    }

    void zero() {
        for (auto& plane : planes_) plane.zero();
    }

private:
    ImageF planes_[3];
};

struct ButteraugliParams {
    float hf_asymmetry = 1.0f;     // multiplier for new HF artifacts against blurred-away features
    float xmul = 1.0f;             // multiplier for the X channel's difference
    float intensity_target = 80.0f;  // nits of an input value of 1.0
};

// One picture's frequency bands, as libjxl's PsychoImage: ultra-high and high
// frequencies for X and Y, medium and low for X, Y and B.
struct PsychoImage {
    ImageF uhf[2];
    ImageF hf[2];
    Image3F mf;
    Image3F lf;
};

// The separable blurs' plane between their two passes, kept between calls.
// The columns of one strip of separable_blur, and the rows of one strip's
// ring: two copies of the last 2 * 16 + 1 rows of the longest kernel.
inline constexpr std::size_t kBlurStripColumns = 256;
inline constexpr std::size_t kBlurRingRows = 2 * (2 * 16 + 1);

// blur's scratch: the strips' rings, and a plane for a blur in place.
struct BlurTemp {
    ImageF* plane(const ImageF& in) {
        temp.ensure(in.xsize(), in.ysize());
        return &temp;
    }
    // A ring of kBlurRingRows rows per strip of a plane `xsize` wide. Its
    // rows are ImageF rows of kBlurStripColumns, whose padding (one more
    // 64-byte line) keeps their distance off a power of two.
    ImageF* rings(std::size_t xsize) {
        const std::size_t strips = (xsize + kBlurStripColumns - 1) / kBlurStripColumns;
        ring_rows.ensure(kBlurStripColumns, strips * kBlurRingRows);
        return &ring_rows;
    }
    ImageF temp;
    ImageF ring_rows;
};

// Planes of the level's size that the masking overwrites (its inputs and
// intermediate results), lent by the engine from planes it no longer needs.
struct MaskScratch {
    ImageF* mask0;
    ImageF* mask1;
    ImageF* diff0;
    ImageF* diff1;
    ImageF* blurred0;
    ImageF* blurred1;
};

// The weights of a 5x5 separable blur, as libjxl's WeightsSeparable5: per
// direction the weights at distances 0, 1 and 2, each repeated 4 times.
struct WeightsSeparable5 {
    float horz[3 * 4];
    float vert[3 * 4];
};

// libjxl's Mirror: out-of-range coordinates reflected, the border pixel
// repeated (-1 -> 0, xsize -> xsize - 1).
inline std::int64_t Mirror(std::int64_t x, const std::int64_t xsize) {
    while (x < 0 || x >= xsize) {
        x = x < 0 ? -x - 1 : 2 * xsize - 1 - x;
    }
    return x;
}

// The functions below that take a RowPool run their passes in its threads,
// each pass's rows in bands; the results do not depend on the thread count.

// butteraugli.cpp: libjxl's Blur (a 5-tap kernel through separable5, others
// through separable_blur).
void blur(RowPool& pool, const ImageF& in, float sigma, BlurTemp* temp, ImageF* out);
// convolve_slow.cc's SlowSeparable5, for planes too narrow for separable5.
void slow_separable5(const ImageF& in, const WeightsSeparable5& weights, ImageF* out);

// butteraugli_kernels.cpp, dispatched to the CPU's best instruction set.
void separable5(RowPool& pool, const ImageF& in, const WeightsSeparable5& weights, ImageF* out);
void opsin_dynamics_image(RowPool& pool, const Image3F& rgb, const ButteraugliParams& params, Image3F* blurred,
                          BlurTemp* blur_temp, Image3F* xyb);
// Uses xyb up: its planes become ps.mf's, and it gets ps.mf's old ones.
void separate_frequencies(RowPool& pool, BlurTemp* blur_temp, Image3F& xyb, PsychoImage& ps);
// padded_diffs: scratch, resized here to hold Malta's differences with a
// margin of zeros round them.
void malta_diff_map(RowPool& pool, const ImageF& lum0, const ImageF& lum1, double w_0gt1, double w_0lt1, double norm1,
                    ImageF* padded_diffs, ImageF* block_diff_ac);
void malta_diff_map_lf(RowPool& pool, const ImageF& lum0, const ImageF& lum1, double w_0gt1, double w_0lt1, double norm1,
                       ImageF* padded_diffs, ImageF* block_diff_ac);
void l2_diff_asymmetric(RowPool& pool, const ImageF& i0, const ImageF& i1, float w_0gt1, float w_0lt1, ImageF* diffmap);
void l2_diff(RowPool& pool, const ImageF& i0, const ImageF& i1, float w, ImageF* diffmap);
void set_l2_diff(RowPool& pool, const ImageF& i0, const ImageF& i1, float w, ImageF* diffmap);
void mask_psycho_image(RowPool& pool, const PsychoImage& pi0, const PsychoImage& pi1, BlurTemp* blur_temp,
                       const MaskScratch& scratch, ImageF* mask, ImageF* diff_ac);
void combine_channels_to_diffmap(RowPool& pool, const ImageF& mask, const Image3F& block_diff_dc, const Image3F& block_diff_ac,
                                 float xmul, ImageF* result);
// libjxl's Blur for a kernel of 2 * half + 1 taps (half 3, 6, 7 or 16), in
// to out (not in), through temp's rings.
void separable_blur(RowPool& pool, const ImageF& in, const float* kernel, std::size_t half, BlurTemp* temp,
                    ImageF* out);

}  // namespace vml::butteraugli
