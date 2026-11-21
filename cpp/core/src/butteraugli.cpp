// Butteraugli computed by VideoMetricsLab itself (vml/butteraugli.h). The
// scalar parts below are ported from libjxl 0.12 (lib/jxl/butteraugli/
// butteraugli.cc, lib/jxl/convolve_slow.cc; BSD licence,
// docs/THIRD_PARTY.md) with the same operations in the same order; the
// vector kernels are in butteraugli_kernels.cpp.
#include "vml/butteraugli.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "butteraugli_image.h"

namespace vml {
namespace butteraugli {
namespace {

// The weights of the per-band differences (libjxl's, as it names them).
constexpr double wMfMalta = 37.0819870399;
constexpr double norm1Mf = 130262059.556;
constexpr double wMfMaltaX = 8246.75321353;
constexpr double norm1MfX = 1009002.70582;
constexpr double wHfMalta = 18.7237414387;
constexpr double norm1Hf = 4498534.45232;
constexpr double wHfMaltaX = 6923.99476109;
constexpr double norm1HfX = 8051.15833247;
constexpr double wUhfMalta = 1.10039032555;
constexpr double norm1Uhf = 71.7800275169;
constexpr double wUhfMaltaX = 173.5;
constexpr double norm1UhfX = 5.0;
constexpr double wmul[9] = {
    400.0, 1.50815703118, 0, 2150.0, 10.6195433239, 16.2176043152, 29.2353797994, 0.844626970982, 0.703646627719,
};

// libjxl's ComputeKernel: a Gaussian of `sigma` truncated at 2.25 sigma.
std::vector<float> compute_kernel(float sigma) {
    const float m = 2.25;
    const double scaler = -1.0 / (2.0 * sigma * sigma);
    const int diff = std::max<int>(1, static_cast<int>(m * std::fabs(sigma)));
    std::vector<float> kernel(2 * diff + 1);
    for (int i = -diff; i <= diff; ++i) {
        kernel[i + diff] = static_cast<float>(std::exp(scaler * i * i));
    }
    return kernel;
}

// convolve_slow.cc's SlowSeparablePixel for a radius of 2, mirrored borders.
float slow_separable5_pixel(const ImageF& in, int64_t x, int64_t y, const WeightsSeparable5& weights) {
    const int64_t xsize = static_cast<int64_t>(in.xsize());
    const int64_t ysize = static_cast<int64_t>(in.ysize());
    float mul = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        const float wy = weights.vert[std::abs(dy) * 4];
        const float* row = in.ConstRow(static_cast<size_t>(Mirror(y + dy, ysize)));
        for (int dx = -2; dx <= 2; ++dx) {
            const float wx = weights.horz[std::abs(dx) * 4];
            mul += row[Mirror(x + dx, xsize)] * wx * wy;
        }
    }
    return mul;
}

}  // namespace

void slow_separable5(const ImageF& in, const WeightsSeparable5& weights, ImageF* out) {
    for (size_t y = 0; y < in.ysize(); ++y) {
        float* row_out = out->Row(y);
        for (size_t x = 0; x < in.xsize(); ++x) {
            row_out[x] = slow_separable5_pixel(in, static_cast<int64_t>(x), static_cast<int64_t>(y), weights);
        }
    }
}

void blur(RowPool& pool, const ImageF& in, float sigma, BlurTemp* temp, ImageF* out) {
    const std::vector<float> kernel = compute_kernel(sigma);
    // separable_blur's lengths; separable5 takes 5 out of place.
    const size_t len = kernel.size();
    if (len != 7 && len != 13 && len != 15 && len != 33 && !(len == 5 && &in != out)) {
        throw std::logic_error("butteraugli: no convolution for this kernel length");
    }
    // separable5 cannot work in place.
    if (kernel.size() == 5 && &in != out) {
        float sum_weights = 0.0f;
        for (const float w : kernel) {
            sum_weights += w;
        }
        const float scale = 1.0f / sum_weights;
        const float w0 = kernel[2] * scale;
        const float w1 = kernel[1] * scale;
        const float w2 = kernel[0] * scale;
        const WeightsSeparable5 weights = {
            {w0, w0, w0, w0, w1, w1, w1, w1, w2, w2, w2, w2},
            {w0, w0, w0, w0, w1, w1, w1, w1, w2, w2, w2, w2},
        };
        separable5(pool, in, weights, out);
        return;
    }
    separable_blur(pool, in, kernel.data(), len / 2, temp->plane(in), out);
}

namespace {

// libjxl's SubSample2x: 2x2 means (an odd last column or row counts twice).
// VideoMetricsLab: output rows in bands, each pixel's sum and doublings in
// libjxl's order (its two input rows in turn, then the last column's, then
// the last row's doubling).
void subsample_2x(RowPool& pool, const Image3F& in, Image3F* out) {
    const size_t xs = (in.xsize() + 1) / 2;
    const size_t ys = (in.ysize() + 1) / 2;
    out->ensure(xs, ys);
    pool.run(ys, [&](size_t row_begin, size_t row_end) {
        for (size_t c = 0; c < 3; ++c) {
            for (size_t oy = row_begin; oy < row_end; ++oy) {
                float* row_out = out->PlaneRow(c, oy);
                std::fill(row_out, row_out + out->Plane(c).PixelsPerRow(), 0.0f);
                for (size_t y = 2 * oy; y < std::min(2 * oy + 2, in.ysize()); ++y) {
                    const float* row_in = in.PlaneRow(c, y);
                    for (size_t x = 0; x < in.xsize(); ++x) {
                        row_out[x / 2] += 0.25f * row_in[x];
                    }
                }
                if ((in.xsize() & 1) != 0) {
                    row_out[xs - 1] *= 2.0f;
                }
                if ((in.ysize() & 1) != 0 && oy == ys - 1) {
                    for (size_t x = 0; x < xs; ++x) {
                        row_out[x] *= 2.0f;
                    }
                }
            }
        }
    });
}

// libjxl's AddSupersampled2x: dest scaled by 1 - 0.3 w (in double), plus w
// times the half-size map's pixel.
void add_supersampled_2x(RowPool& pool, const ImageF& src, float w, ImageF& dest) {
    constexpr double kHeuristicMixingValue = 0.3;
    pool.run(dest.ysize(), [&](size_t row_begin, size_t row_end) {
        for (size_t y = row_begin; y < row_end; ++y) {
            float* row = dest.Row(y);
            const float* row_src = src.Row(y / 2);
            for (size_t x = 0; x < dest.xsize(); ++x) {
                row[x] = static_cast<float>(row[x] * (1.0 - kHeuristicMixingValue * w));
                row[x] += w * row_src[x / 2];
            }
        }
    });
}

// Image3F::zero in bands.
void zero_planes(RowPool& pool, Image3F& image) {
    pool.run(image.ysize(), [&](size_t row_begin, size_t row_end) {
        for (size_t c = 0; c < 3; ++c) {
            ImageF& plane = image.Plane(c);
            std::fill(plane.Row(row_begin), plane.Row(row_begin) + (row_end - row_begin) * plane.PixelsPerRow(), 0.0f);
        }
    });
}

// The planes of a comparison, shared by the full and the half resolution
// (ImageF::ensure fits a smaller size into a plane's buffer), and kept from
// frame to frame. Planes whose contents a step no longer needs are lent to
// later steps, so that a comparison takes about 31 full-size planes where
// libjxl's comparator takes over 50.
struct Workspace {
    Image3F rgb;      // the full-size picture being split; later the DC differences
    Image3F blurred;  // opsin's blur; later the masking's scratch
    Image3F xyb;      // opsin's result, used up by separate_frequencies; later scratch
    ImageF malta_diffs;  // Malta's differences, with a margin of zeros (malta_diff_map)
    BlurTemp blur_temp;
    PsychoImage pi0;  // the reference's bands; pi0.lf later the AC differences
    PsychoImage pi1;  // the test picture's bands; pi1.lf later the mask
};

// One picture's bands (libjxl's comparator: OpsinDynamicsImage, then
// SeparateFrequencies).
void split_picture(RowPool& pool, Workspace& work, const Image3F& rgb, const ButteraugliParams& params,
                   PsychoImage& bands) {
    const std::size_t xsize = rgb.xsize();
    const std::size_t ysize = rgb.ysize();
    work.blurred.ensure(xsize, ysize);
    work.xyb.ensure(xsize, ysize);
    opsin_dynamics_image(pool, rgb, params, &work.blurred, &work.blur_temp, &work.xyb);
    separate_frequencies(pool, &work.blur_temp, work.xyb, bands);
}

// The difference map of work.pi1 against work.pi0 (libjxl's
// DiffmapPsychoImage). Every difference lands in each pixel in libjxl's
// order; the DC differences, which are set rather than added to, come first
// so that the low-frequency bands' planes are free for the AC sums and the
// mask.
void compare_bands(RowPool& pool, Workspace& work, const ButteraugliParams& params, ImageF* diffmap) {
    const PsychoImage& pi0 = work.pi0;
    const PsychoImage& pi1 = work.pi1;
    const std::size_t xsize = pi0.hf[0].xsize();
    const std::size_t ysize = pi0.hf[0].ysize();
    const float hf_asymmetry = params.hf_asymmetry;

    Image3F& block_diff_dc = work.rgb;  // the pictures are split
    block_diff_dc.ensure(xsize, ysize);
    for (size_t c = 0; c < 3; ++c) {
        set_l2_diff(pool, pi0.lf.Plane(c), pi1.lf.Plane(c), static_cast<float>(wmul[6 + c]), &block_diff_dc.Plane(c));
    }

    Image3F& block_diff_ac = work.pi0.lf;  // used by the DC differences only
    zero_planes(pool, block_diff_ac);
    ImageF* ac[3] = {&block_diff_ac.Plane(0), &block_diff_ac.Plane(1), &block_diff_ac.Plane(2)};
    work.xyb.ensure(xsize, ysize);  // scratch from here on (separate_frequencies traded its planes)
    work.blurred.ensure(xsize, ysize);
    ImageF* diffs = &work.malta_diffs;
    malta_diff_map(pool, pi0.uhf[1], pi1.uhf[1], wUhfMalta * hf_asymmetry, wUhfMalta / hf_asymmetry, norm1Uhf, diffs,
                   ac[1]);
    malta_diff_map(pool, pi0.uhf[0], pi1.uhf[0], wUhfMaltaX * hf_asymmetry, wUhfMaltaX / hf_asymmetry, norm1UhfX,
                   diffs, ac[0]);
    malta_diff_map_lf(pool, pi0.hf[1], pi1.hf[1], wHfMalta * std::sqrt(hf_asymmetry),
                      wHfMalta / std::sqrt(hf_asymmetry), norm1Hf, diffs, ac[1]);
    malta_diff_map_lf(pool, pi0.hf[0], pi1.hf[0], wHfMaltaX * std::sqrt(hf_asymmetry),
                      wHfMaltaX / std::sqrt(hf_asymmetry), norm1HfX, diffs, ac[0]);
    malta_diff_map_lf(pool, pi0.mf.Plane(1), pi1.mf.Plane(1), wMfMalta, wMfMalta, norm1Mf, diffs, ac[1]);
    malta_diff_map_lf(pool, pi0.mf.Plane(0), pi1.mf.Plane(0), wMfMaltaX, wMfMaltaX, norm1MfX, diffs, ac[0]);
    for (size_t c = 0; c < 3; ++c) {
        if (c < 2) {  // no blue-channel error at high frequencies
            l2_diff_asymmetric(pool, pi0.hf[c], pi1.hf[c], static_cast<float>(wmul[c] * hf_asymmetry),
                               static_cast<float>(wmul[c] / hf_asymmetry), ac[c]);
        }
        l2_diff(pool, pi0.mf.Plane(c), pi1.mf.Plane(c), static_cast<float>(wmul[3 + c]), ac[c]);
    }
    ImageF& mask = work.pi1.lf.Plane(0);  // used by the DC differences only
    const MaskScratch scratch = {&work.blurred.Plane(0), &work.blurred.Plane(1), &work.blurred.Plane(2),
                                 &work.xyb.Plane(0),     &work.xyb.Plane(1),     &work.xyb.Plane(2)};
    mask_psycho_image(pool, pi0, pi1, &work.blur_temp, scratch, &mask, ac[1]);
    diffmap->ensure(xsize, ysize);
    combine_channels_to_diffmap(pool, mask, block_diff_dc, block_diff_ac, params.xmul, diffmap);
}

}  // namespace
}  // namespace butteraugli

class Butteraugli::Engine {
public:
    Engine(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
           std::size_t height, float intensity_target, std::size_t threads)
        : pool_(threads),
          readings_{source_reading, test_reading},
          test_scale_(test_scale_of(source_reading, test_reading)),
          width_(width),
          height_(height),
          // libjxl's comparator gives pictures under 8x8 an all-zero map, and
          // compares at half resolution too when that is 8x8 or more.
          compares_(width >= 8 && height >= 8),
          has_half_((width + 1) / 2 >= 8 && (height + 1) / 2 >= 8),
          map_(width * height, 0.0f) {
        params_.intensity_target = intensity_target;
    }

    // libjxl's comparator brings the test picture to the source's scale:
    // each is made linear with its own intensity target (1.0 is 10000 nits
    // for PQ, 1000 for HLG, 255 otherwise), then the test is multiplied by
    // its target over the source's (JxlButteraugliComparator::CompareWith).
    static float test_scale_of(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading) {
        return test_reading.intensity_target() / source_reading.intensity_target();
    }

    void scale_row(butteraugli::Image3F& rgb, std::size_t y) const {
        for (std::size_t c = 0; c < 3; ++c) {
            float* row = rgb.PlaneRow(c, y);
            for (std::size_t x = 0; x < width_; ++x) row[x] = test_scale_ * row[x];
        }
    }

    const std::vector<float>& distortion_map(const std::uint16_t* source, const std::uint16_t* test,
                                             std::size_t stride) {
        if (!compares_) return map_;  // all zero
        const std::uint16_t* pictures[2] = {source, test};
        butteraugli::PsychoImage* bands[2] = {&work_.pi0, &work_.pi1};
        for (int n = 0; n < 2; ++n) {
            butteraugli::Image3F& rgb = work_.rgb;
            rgb.ensure(width_, height_);
            pool_.run(height_, [&](std::size_t row_begin, std::size_t row_end) {
                for (std::size_t y = row_begin; y < row_end; ++y) {
                    readings_[n].convert_row(pictures[n] + y * stride, width_, rgb.PlaneRow(0, y),
                                         rgb.PlaneRow(1, y), rgb.PlaneRow(2, y));
                    if (n == 1 && test_scale_ != 1.0f) scale_row(rgb, y);
                }
            });
            if (has_half_) {
                butteraugli::subsample_2x(pool_, rgb, &half_rgb_[n]);
            }
            butteraugli::split_picture(pool_, work_, rgb, params_, *bands[n]);
        }
        butteraugli::compare_bands(pool_, work_, params_, &diffmap_);
        if (has_half_) {
            butteraugli::split_picture(pool_, work_, half_rgb_[0], params_, work_.pi0);
            butteraugli::split_picture(pool_, work_, half_rgb_[1], params_, work_.pi1);
            butteraugli::compare_bands(pool_, work_, params_, &half_diffmap_);
            butteraugli::add_supersampled_2x(pool_, half_diffmap_, 0.5f, diffmap_);
        }
        pool_.run(height_, [&](std::size_t row_begin, std::size_t row_end) {
            for (std::size_t y = row_begin; y < row_end; ++y) {
                std::memcpy(map_.data() + y * width_, diffmap_.Row(y), width_ * sizeof(float));
            }
        });
        return map_;
    }

    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }

private:
    RowPool pool_;
    std::array<JxlLinearRgb, 2> readings_;  // source's, test's
    float test_scale_;                      // test_scale_of; 1 for pictures of the same target
    std::size_t width_;
    std::size_t height_;
    bool compares_;
    bool has_half_;
    butteraugli::ButteraugliParams params_;
    butteraugli::Workspace work_;
    butteraugli::Image3F half_rgb_[2];  // both pictures at half resolution
    butteraugli::ImageF diffmap_;       // the full resolution's map
    butteraugli::ImageF half_diffmap_;
    std::vector<float> map_;
};

Butteraugli::Butteraugli(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                         std::size_t height, float intensity_target, std::size_t threads)
    : engine_(std::make_unique<Engine>(source_reading, test_reading, width, height, intensity_target, threads)) {}
Butteraugli::~Butteraugli() = default;
Butteraugli::Butteraugli(Butteraugli&&) noexcept = default;
Butteraugli& Butteraugli::operator=(Butteraugli&&) noexcept = default;

std::size_t Butteraugli::width() const { return engine_->width(); }
std::size_t Butteraugli::height() const { return engine_->height(); }

const std::vector<float>& Butteraugli::distortion_map(const std::uint16_t* source, const std::uint16_t* test,
                                                      std::size_t stride) {
    return engine_->distortion_map(source, test, stride);
}

}  // namespace vml
