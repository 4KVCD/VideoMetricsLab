// VideoMetricsLab's SSIMULACRA2 engine (ssimulacra2.h), a streaming port of
// libjxl 0.12's tools/ssimulacra2.cc (BSD licence, docs/THIRD_PARTY.md).
//
// libjxl computes, at each of up to 6 scales (each half the size of the one
// before, downsampled in linear sRGB): both pictures in XYB, five planes per
// channel blurred with a recursive Gaussian (the two pictures, their squares
// and their product), then an SSIM map and two edge maps whose sums give the
// scale's averages. Here every scale is fed one linear row at a time, top to
// bottom. A row's XYB, products and horizontal blurs are made at once; the
// vertical blur of row n follows when row n + 4 arrives (libjxl's filter
// reads rows n - 6 and n + 4), and the maps of row n with it. Two linear rows
// make one row of the next scale. The rings below hold just the rows still
// needed.
#include "vml/ssimulacra2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <new>

#include "ssimulacra2_kernels.h"

namespace vml {
namespace {

namespace kernels = ssimulacra2_kernels;

constexpr std::size_t kMaxScales = 6;
constexpr std::size_t kChannels = 3;
constexpr std::size_t kPictures = 2;  // the source, then the test picture
// The vertical blur's output row n reads the horizontally blurred rows n - 6
// and n + 4 (libjxl's radius of 5).
constexpr std::size_t kBlurAbove = 6;
constexpr std::size_t kBlurBelow = 4;
constexpr std::size_t kBlurredRows = kBlurAbove + 1 + kBlurBelow;
// The maps of row n need its XYB rows, made when row n + 4 arrived.
constexpr std::size_t kXybRows = kBlurBelow + 1;
constexpr std::size_t kColumnStateRows = 6;  // blur_column_step's state

// The planes blurred per channel (libjxl's mu1, mu2, sigma1_sq, sigma2_sq
// and sigma12): the edge maps use the first two, the SSIM map all five.
enum Quantity { kMu1, kMu2, kSigma11, kSigma22, kSigma12, kQuantities };

constexpr float kC2 = 0.0009f;

// The score's weights (Msssim::Score): for each channel, scale, and norm
// (mean, then 4-norm), the SSIM, ringing and blurring terms.
constexpr double kWeights[108] = {
    0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0,
    0.0, 0.0004371155730107379, 0.0, 1.1041726426657346, 0.00066284834129271, 0.00015231632783718752,
    0.0, 0.0016406437456599754, 0.0, 1.8422455520539298, 11.441172603757666, 0.0,
    0.0007989109436015163, 0.000176816438078653, 0.0, 1.8787594979546387, 10.94906990605142, 0.0,
    0.0007289346991508072, 0.9677937080626833, 0.0, 0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053,
    0.0004550992113792063, 0.0, 0.0, 0.0013648766163243398, 0.0, 0.0,
    0.0, 0.0, 0.0, 7.466890328078848, 0.0, 17.445833984131262,
    0.0006235601634041466, 0.0, 0.0, 6.683678146179332, 0.00037724407979611296, 1.027889937768264,
    225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361, 0.001237755635509985,
    176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757, 0.0004485436923833408,
    0.0, 0.0, 0.0, 34.77906344483772, 44.835625328877896, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0008680556573291698, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0005313191874358747, 0.0, 0.00016533814161379112, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833, 0.0,
    0.0020827005846636437, 0.0, 0.0, 8.826982764996862, 23.19243343998926, 0.0,
    95.1080498811086, 0.9863978034400682, 0.9834382792465353, 0.0012286405048278493, 171.2667255897307, 0.9807858872435379,
    0.0, 0.0, 0.0, 0.0005130064588990679, 0.0, 0.00010854057858411537,
};

// libjxl's opsin absorbance matrix and bias (lib/jxl/cms/opsin_params.h),
// computed in float as there.
constexpr float kM02 = 0.078f;
constexpr float kM00 = 0.30f;
constexpr float kM01 = 1.0f - kM02 - kM00;
constexpr float kM12 = 0.078f;
constexpr float kM10 = 0.23f;
constexpr float kM11 = 1.0f - kM12 - kM10;
constexpr float kM20 = 0.24342268924547819f;
constexpr float kM21 = 0.20476744424496821f;
constexpr float kM22 = 1.0f - kM20 - kM21;
constexpr float kOpsinMatrix[9] = {kM00, kM01, kM02, kM10, kM11, kM12, kM20, kM21, kM22};
constexpr float kOpsinBias = 0.0037930732552754493f;
// -cbrtf(kOpsinBias), as libjxl's tools compute it.
constexpr float kNegCbrtOpsinBias = -0x1.3f64ecp-3f;

// libjxl's ComputePremulAbsorb: the matrix scaled by the intensity target.
std::array<float, 11> opsin_constants(float intensity_target) {
    const float scale = intensity_target / 255.0f;
    std::array<float, 11> constants{};
    for (std::size_t i = 0; i < 9; ++i) constants[i] = kOpsinMatrix[i] * scale;
    constants[9] = kOpsinBias;
    constants[10] = kNegCbrtOpsinBias;
    return constants;
}

// Rows of floats padded to 16 (one AVX-512 vector, four NEON vectors), each
// starting on a 64-byte boundary.
std::size_t padded_width(std::size_t width) { return (width + 15) / 16 * 16; }

struct AlignedDelete {
    void operator()(float* rows) const { ::operator delete[](rows, std::align_val_t{64}); }
};
using AlignedRows = std::unique_ptr<float[], AlignedDelete>;

// `count` floats, zeroed: the padding past each row's width stays zero.
AlignedRows allocate_floats(std::size_t count) {
    auto* rows = static_cast<float*>(::operator new[](count * sizeof(float), std::align_val_t{64}));
    std::fill_n(rows, count, 0.0f);
    return AlignedRows(rows);
}

double quartic(double x) {
    x *= x;
    x *= x;
    return x;
}

// libjxl's SSIMMap for one row of one channel, adding to its two sums.
void add_ssim_row(const float* m1, const float* m2, const float* s11, const float* s22, const float* s12,
                  std::size_t width, double* sums) {
    double sum_mean = sums[0];
    double sum_quartic = sums[1];
    for (std::size_t x = 0; x < width; ++x) {
        const float mu1 = m1[x];
        const float mu2 = m2[x];
        const float mu11 = mu1 * mu1;
        const float mu22 = mu2 * mu2;
        const float mu12 = mu1 * mu2;
        // As in libjxl: the square in float, the subtraction in double.
        const float num_m = static_cast<float>(1.0 - (mu1 - mu2) * (mu1 - mu2));
        const float num_s = 2 * (s12[x] - mu12) + kC2;
        const float denom_s = (s11[x] - mu11) + (s22[x] - mu22) + kC2;
        double d = 1.0 - (num_m * num_s / denom_s);
        d = std::max(d, 0.0);
        sum_mean += d;
        sum_quartic += quartic(d);
    }
    sums[0] = sum_mean;
    sums[1] = sum_quartic;
}

// libjxl's EdgeDiffMap for one row of one channel, adding to its four sums.
void add_edge_row(const float* img1, const float* mu1, const float* img2, const float* mu2, std::size_t width,
                  double* sums) {
    double artifact_sum = sums[0];
    double artifact_quartic = sums[1];
    double detail_lost_sum = sums[2];
    double detail_lost_quartic = sums[3];
    for (std::size_t x = 0; x < width; ++x) {
        const double d1 = (1.0 + std::abs(img2[x] - mu2[x])) / (1.0 + std::abs(img1[x] - mu1[x])) - 1.0;
        const double artifact = std::max(d1, 0.0);
        artifact_sum += artifact;
        artifact_quartic += quartic(artifact);
        const double detail_lost = std::max(-d1, 0.0);
        detail_lost_sum += detail_lost;
        detail_lost_quartic += quartic(detail_lost);
    }
    sums[0] = artifact_sum;
    sums[1] = artifact_quartic;
    sums[2] = detail_lost_sum;
    sums[3] = detail_lost_quartic;
}

// libjxl's Downsample(in, 2, 2) for one output row: each 2x2 block's mean,
// summed in this order from 0; an odd last row or column is repeated.
void downsample_row(const float* upper, const float* lower, std::size_t in_width, float* out,
                    std::size_t out_width) {
    for (std::size_t ox = 0; ox < out_width; ++ox) {
        const std::size_t x0 = 2 * ox;
        const std::size_t x1 = std::min(2 * ox + 1, in_width - 1);
        float sum = 0.0f;
        sum += upper[x0];
        sum += upper[x1];
        sum += lower[x0];
        sum += lower[x1];
        out[ox] = sum * 0.25f;
    }
}

// Pointers to both pictures' three linear planes of one row.
struct LinearRow {
    const float* planes[kPictures][kChannels];
};

// One scale's rows and sums.
struct Scale {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t stride = 0;  // floats per row
    bool ssim_map[kChannels] = {};
    bool edge_maps[kChannels] = {};

    float* xyb[kPictures][kChannels][kXybRows] = {};               // by row % kXybRows
    float* blurred_rows[kChannels][kQuantities][kBlurredRows] = {};  // horizontal blur, by row % kBlurredRows
    float* column_state[kChannels][kQuantities] = {};
    float* blurred[kChannels][kQuantities] = {};  // the vertical blur's current output row
    float* product = nullptr;
    float* zeros = nullptr;
    float* even_row[kPictures][kChannels] = {};      // the last even linear row, for the next scale
    float* next_scale_row[kPictures][kChannels] = {};  // the next scale's row being made
    AlignedRows memory;

    std::size_t rows_in = 0;
    std::ptrdiff_t next_output = 0;  // the vertical blur's next step
    double ssim_sums[kChannels][2] = {};
    double edge_sums[kChannels][4] = {};

    bool uses(std::size_t c, std::size_t q) const {
        return q <= kMu2 ? ssim_map[c] || edge_maps[c] : ssim_map[c];
    }
};

}  // namespace

class Ssimulacra2::Engine {
public:
    Engine(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
           std::size_t height, Terms terms)
        : readings_{source_reading, test_reading}, width_(width), height_(height),
          opsin_{opsin_constants(source_reading.intensity_target()),
                 opsin_constants(test_reading.intensity_target())} {
        make_scales(terms);
        const std::size_t stride = padded_width(width);
        input_memory_ = allocate_floats(kPictures * kChannels * stride);
        for (std::size_t i = 0; i < kPictures; ++i) {
            for (std::size_t c = 0; c < kChannels; ++c) input_[i][c] = &input_memory_[(i * kChannels + c) * stride];
        }
    }

    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }
    const std::vector<ScaleAverages>& averages() const { return averages_; }

    double score(const std::uint16_t* source, const std::uint16_t* test, std::size_t stride) {
        for (Scale& scale : scales_) start(scale);
        LinearRow row;
        for (std::size_t i = 0; i < kPictures; ++i) {
            for (std::size_t c = 0; c < kChannels; ++c) row.planes[i][c] = input_[i][c];
        }
        if (!scales_.empty()) {
            for (std::size_t y = 0; y < height_; ++y) {
                readings_[0].convert_row(source + y * stride, width_, input_[0][0], input_[0][1], input_[0][2]);
                readings_[1].convert_row(test + y * stride, width_, input_[1][0], input_[1][1], input_[1][2]);
                feed(0, row);
            }
        }
        averages_.clear();
        for (Scale& scale : scales_) {
            finish(scale);
            averages_.push_back(averages_of(scale));
        }
        return score_of(averages_);
    }

private:
    // libjxl computes a scale while the scale before it (for the first two,
    // the full picture) is at least 8 by 8.
    void make_scales(Terms terms) {
        std::size_t width = width_, height = height_;
        std::size_t checked_width = width_, checked_height = height_;
        for (std::size_t s = 0; s < kMaxScales; ++s) {
            if (checked_width < 8 || checked_height < 8) break;
            Scale scale;
            scale.width = width;
            scale.height = height;
            scale.stride = padded_width(width);
            scales_.push_back(std::move(scale));
            checked_width = width;
            checked_height = height;
            width = (width + 1) / 2;
            height = (height + 1) / 2;
        }
        // The weight of each term: channel, then scale (of those computed),
        // then norm, as Msssim::Score walks them.
        for (std::size_t s = 0; s < scales_.size(); ++s) {
            Scale& scale = scales_[s];
            for (std::size_t c = 0; c < kChannels; ++c) {
                const double* weight = &kWeights[(c * scales_.size() + s) * 6];
                scale.ssim_map[c] = terms == Terms::all || weight[0] != 0.0 || weight[3] != 0.0;
                scale.edge_maps[c] = terms == Terms::all || weight[1] != 0.0 || weight[2] != 0.0 ||
                                     weight[4] != 0.0 || weight[5] != 0.0;
            }
            allocate(scale, s + 1 < scales_.size() ? &scales_[s + 1] : nullptr);
        }
    }

    // Every row a scale needs, from one allocation: XYB rings, and per plane
    // it blurs a ring of horizontal blurs, the vertical state and an output.
    static void allocate(Scale& scale, const Scale* next) {
        std::size_t rows = kPictures * kChannels * kXybRows + 2;  // + product + zeros
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (std::size_t q = 0; q < kQuantities; ++q) {
                if (scale.uses(c, q)) rows += kBlurredRows + kColumnStateRows + 1;
            }
        }
        if (next) rows += 2 * kPictures * kChannels;
        scale.memory = allocate_floats(rows * scale.stride);
        float* free_row = scale.memory.get();
        const auto take = [&](std::size_t count) {
            float* taken = free_row;
            free_row += count * scale.stride;
            return taken;
        };
        for (auto& picture : scale.xyb) {
            for (auto& channel : picture) {
                for (float*& row : channel) row = take(1);
            }
        }
        scale.product = take(1);
        scale.zeros = take(1);
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (std::size_t q = 0; q < kQuantities; ++q) {
                if (!scale.uses(c, q)) continue;
                for (float*& row : scale.blurred_rows[c][q]) row = take(1);
                scale.column_state[c][q] = take(kColumnStateRows);
                scale.blurred[c][q] = take(1);
            }
        }
        if (next) {
            for (std::size_t i = 0; i < kPictures; ++i) {
                for (std::size_t c = 0; c < kChannels; ++c) {
                    scale.even_row[i][c] = take(1);
                    scale.next_scale_row[i][c] = take(1);
                }
            }
        }
    }

    // A new picture pair: the vertical filters start from zero (libjxl
    // clears its ring buffer per column strip) and so do the sums.
    static void start(Scale& scale) {
        scale.rows_in = 0;
        scale.next_output = -static_cast<std::ptrdiff_t>(kBlurBelow);
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (std::size_t q = 0; q < kQuantities; ++q) {
                if (scale.uses(c, q)) {
                    std::fill_n(scale.column_state[c][q], kColumnStateRows * scale.stride, 0.0f);
                }
            }
            std::fill(std::begin(scale.ssim_sums[c]), std::end(scale.ssim_sums[c]), 0.0);
            std::fill(std::begin(scale.edge_sums[c]), std::end(scale.edge_sums[c]), 0.0);
        }
    }

    // One linear row of scale s: its XYB, products and horizontal blurs, the
    // vertical step it completes, and every second row the next scale's row.
    void feed(std::size_t s, const LinearRow& linear) {
        Scale& scale = scales_[s];
        const std::size_t y = scale.rows_in++;
        const std::size_t xyb_slot = y % kXybRows;
        for (std::size_t i = 0; i < kPictures; ++i) {
            kernels::positive_xyb_row(linear.planes[i][0], linear.planes[i][1], linear.planes[i][2], opsin_[i].data(),
                                      scale.xyb[i][0][xyb_slot], scale.xyb[i][1][xyb_slot],
                                      scale.xyb[i][2][xyb_slot], scale.stride);
        }
        const std::size_t blur_slot = y % kBlurredRows;
        for (std::size_t c = 0; c < kChannels; ++c) {
            if (!scale.uses(c, kMu1)) continue;
            const float* img1 = scale.xyb[0][c][xyb_slot];
            const float* img2 = scale.xyb[1][c][xyb_slot];
            kernels::blur_row(img1, scale.blurred_rows[c][kMu1][blur_slot], scale.width);
            kernels::blur_row(img2, scale.blurred_rows[c][kMu2][blur_slot], scale.width);
            if (!scale.ssim_map[c]) continue;
            const std::pair<const float*, const float*> factors[] = {{img1, img1}, {img2, img2}, {img1, img2}};
            for (std::size_t k = 0; k < 3; ++k) {
                kernels::multiply_row(factors[k].first, factors[k].second, scale.product, scale.stride);
                kernels::blur_row(scale.product, scale.blurred_rows[c][kSigma11 + k][blur_slot], scale.width);
            }
        }
        column_step(scale, static_cast<std::ptrdiff_t>(y) - static_cast<std::ptrdiff_t>(kBlurBelow));

        if (s + 1 == scales_.size()) return;
        const bool even = y % 2 == 0;
        if (even && y + 1 < scale.height) {
            for (std::size_t i = 0; i < kPictures; ++i) {
                for (std::size_t c = 0; c < kChannels; ++c) {
                    std::memcpy(scale.even_row[i][c], linear.planes[i][c], scale.width * sizeof(float));
                }
            }
            return;
        }
        // Rows y - 1 and y, or the last (even) row twice.
        const Scale& next = scales_[s + 1];
        LinearRow next_row;
        for (std::size_t i = 0; i < kPictures; ++i) {
            for (std::size_t c = 0; c < kChannels; ++c) {
                const float* upper = even ? linear.planes[i][c] : scale.even_row[i][c];
                downsample_row(upper, linear.planes[i][c], scale.width, scale.next_scale_row[i][c], next.width);
                next_row.planes[i][c] = scale.next_scale_row[i][c];
            }
        }
        feed(s + 1, next_row);
    }

    // The vertical blur's output row n (n < 0 only warms the filters up),
    // then the maps of that row. FastGaussianVertical's input for row n is
    // the blurred row n + 4 alone up to row 5, then rows n - 6 and n + 4;
    // past the bottom, a row of zeros stands in for row n + 4.
    void column_step(Scale& scale, std::ptrdiff_t n) {
        const std::size_t below = static_cast<std::size_t>(n + static_cast<std::ptrdiff_t>(kBlurBelow));
        const bool has_above = n >= static_cast<std::ptrdiff_t>(kBlurAbove);
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (std::size_t q = 0; q < kQuantities; ++q) {
                if (!scale.uses(c, q)) continue;
                float* const* ring = scale.blurred_rows[c][q];
                const float* bottom = below < scale.height ? ring[below % kBlurredRows] : scale.zeros;
                const float* top =
                    has_above ? ring[static_cast<std::size_t>(n - static_cast<std::ptrdiff_t>(kBlurAbove)) %
                                     kBlurredRows]
                              : nullptr;
                kernels::blur_column_step(top, bottom, scale.column_state[c][q], n >= 0 ? scale.blurred[c][q] : nullptr,
                                          scale.stride);
            }
        }
        scale.next_output = n + 1;
        if (n >= 0) add_map_rows(scale, static_cast<std::size_t>(n));
    }

    static void add_map_rows(Scale& scale, std::size_t n) {
        const std::size_t slot = n % kXybRows;
        for (std::size_t c = 0; c < kChannels; ++c) {
            float* const* blurred = scale.blurred[c];
            if (scale.ssim_map[c]) {
                add_ssim_row(blurred[kMu1], blurred[kMu2], blurred[kSigma11], blurred[kSigma22], blurred[kSigma12],
                             scale.width, scale.ssim_sums[c]);
            }
            if (scale.edge_maps[c]) {
                add_edge_row(scale.xyb[0][c][slot], blurred[kMu1], scale.xyb[1][c][slot], blurred[kMu2], scale.width,
                             scale.edge_sums[c]);
            }
        }
    }

    // The last rows' vertical steps, with zeros below the picture.
    void finish(Scale& scale) {
        for (std::ptrdiff_t n = scale.next_output; n < static_cast<std::ptrdiff_t>(scale.height); ++n) {
            column_step(scale, n);
        }
    }

    static ScaleAverages averages_of(const Scale& scale) {
        ScaleAverages averages;
        const double one_per_pixels = 1.0 / static_cast<double>(scale.height * scale.width);
        for (std::size_t c = 0; c < kChannels; ++c) {
            averages.ssim[c * 2] = one_per_pixels * scale.ssim_sums[c][0];
            averages.ssim[c * 2 + 1] = std::sqrt(std::sqrt(one_per_pixels * scale.ssim_sums[c][1]));
            for (std::size_t k = 0; k < 4; k += 2) {
                averages.edge[c * 4 + k] = one_per_pixels * scale.edge_sums[c][k];
                averages.edge[c * 4 + k + 1] = std::sqrt(std::sqrt(one_per_pixels * scale.edge_sums[c][k + 1]));
            }
        }
        return averages;
    }

    // Msssim::Score.
    static double score_of(const std::vector<ScaleAverages>& scales) {
        double ssim = 0.0;
        std::size_t i = 0;
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (const ScaleAverages& scale : scales) {
                for (std::size_t n = 0; n < 2; ++n) {
                    ssim += kWeights[i++] * std::abs(scale.ssim[c * 2 + n]);
                    ssim += kWeights[i++] * std::abs(scale.edge[c * 4 + n]);
                    ssim += kWeights[i++] * std::abs(scale.edge[c * 4 + n + 2]);
                }
            }
        }
        ssim = ssim * 0.9562382616834844;
        ssim = 2.326765642916932 * ssim - 0.020884521182843837 * ssim * ssim +
               6.248496625763138e-05 * ssim * ssim * ssim;
        return ssim > 0 ? 100.0 - 10.0 * std::pow(ssim, 0.6276336467831387) : 100.0;
    }

    std::array<JxlLinearRgb, kPictures> readings_;  // source's, test's
    std::size_t width_;
    std::size_t height_;
    std::array<std::array<float, 11>, kPictures> opsin_;  // each picture's
    std::vector<Scale> scales_;
    AlignedRows input_memory_;
    float* input_[kPictures][kChannels] = {};
    std::vector<ScaleAverages> averages_;
};

Ssimulacra2::Ssimulacra2(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                         std::size_t height, Terms terms)
    : engine_(std::make_unique<Engine>(source_reading, test_reading, width, height, terms)) {}
Ssimulacra2::~Ssimulacra2() = default;
Ssimulacra2::Ssimulacra2(Ssimulacra2&&) noexcept = default;
Ssimulacra2& Ssimulacra2::operator=(Ssimulacra2&&) noexcept = default;

std::size_t Ssimulacra2::width() const { return engine_->width(); }
std::size_t Ssimulacra2::height() const { return engine_->height(); }

double Ssimulacra2::score(const std::uint16_t* source, const std::uint16_t* test, std::size_t stride) {
    return engine_->score(source, test, stride);
}

const std::vector<Ssimulacra2::ScaleAverages>& Ssimulacra2::averages() const { return engine_->averages(); }

bool Ssimulacra2::fused_multiply_add() { return kernels::fused_multiply_add(); }

}  // namespace vml
