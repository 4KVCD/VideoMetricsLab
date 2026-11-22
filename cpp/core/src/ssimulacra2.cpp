// VideoMetricsLab's SSIMULACRA2 engine (ssimulacra2.h), a streaming port of
// libjxl 0.12's tools/ssimulacra2.cc (BSD licence, docs/THIRD_PARTY.md).
//
// libjxl computes, at each of up to 6 scales (each half the size of the one
// before, downsampled in linear sRGB): both pictures in XYB, five planes per
// channel blurred with a recursive Gaussian (the two pictures, their squares
// and their product), then an SSIM map and two edge maps whose sums give the
// scale's averages. Here the pair streams through every scale from top to
// bottom in steps of a few dozen rows, split between the pair's threads like
// Butteraugli's engine (butteraugli.cpp). Each scale has three stages: its
// rows (linear, then XYB, products and horizontal blurs; two linear rows make
// one row of the next scale), the vertical blur in column strips (libjxl's
// filter makes row n from rows n - 6 and n + 4, column by column), and the
// maps' sums. In a step every stage works on rows that earlier steps made,
// so its tasks run in any order on any thread; the rings below hold just the
// rows still needed. Each sum is one task per step that adds the step's rows
// in order, so every sum is added pixel by pixel in libjxl's order and the
// score does not depend on the number of threads.
#include "vml/ssimulacra2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#include "row_pool.h"
#include "ssimulacra2_kernels.h"

namespace vml {
namespace {

namespace kernels = ssimulacra2_kernels;

constexpr std::size_t kMaxScales = 6;
constexpr std::size_t kChannels = 3;
constexpr std::size_t kPictures = 2;  // the source, then the test picture
// The vertical blur's output row n reads the horizontally blurred rows n - 6
// and n + 4 (libjxl's radius of 5).
constexpr std::int64_t kBlurAbove = 6;
constexpr std::int64_t kBlurBelow = 4;
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

// ---- The schedule ----

// Each scale's stages: its rows (linear, XYB, products, horizontal blurs),
// the vertical blur in column strips, and the maps' sums.
enum StageKind { kRowStage, kColumnStage, kSumStage, kStageKinds };
constexpr std::size_t kStages = kMaxScales * kStageKinds;
constexpr std::size_t stage_of(std::size_t scale, StageKind kind) { return scale * kStageKinds + kind; }

// The vertical blur takes a scale's height + 4 steps: step m makes output row
// m - 4 (the first 4 only warm the filters up) from the horizontally blurred
// rows m (the bottom; zeros past the picture) and m - 10 (the top, from
// output row 6 on).
constexpr std::int64_t kWarmUpSteps = kBlurBelow;
constexpr std::int64_t kTopBehind = kBlurAbove + kBlurBelow;

// The rings a scale's stages write: its linear rows (read by the next scale),
// XYB rows, horizontal blurs and vertical blurs.
enum Group { kLinear, kXyb, kHorizontal, kVertical, kGroups };

// How a consumer's row reaches into its producer's rows.
enum class Reach : std::uint8_t {
    kSame,       // row y reads row y
    kSubsample,  // the next scale's row y reads rows 2y and 2y + 1 (clipped)
    kColumn,     // the vertical blur's step m reads rows m - 10 and m (clipped)
    kBlurred,    // the sums' row n read the vertical blur's step n + 4
};

struct Edge {
    std::size_t consumer;
    std::size_t producer;
    Reach reach;
    Group group;  // the producer's ring it reads
};

// The last producer row that consumer row y reads.
std::int64_t last_read(Reach reach, std::int64_t y, std::int64_t producer_rows) {
    switch (reach) {
    case Reach::kSame: return y;
    case Reach::kSubsample: return std::min(2 * y + 1, producer_rows - 1);
    case Reach::kColumn: return std::min(y, producer_rows - 1);
    case Reach::kBlurred: return y + kWarmUpSteps;
    }
    return y;
}

// The last consumer row that reads producer row j.
std::int64_t last_reader(Reach reach, std::int64_t j, std::int64_t consumer_rows) {
    switch (reach) {
    case Reach::kSame: return std::min(j, consumer_rows - 1);
    case Reach::kSubsample: return std::min(j / 2, consumer_rows - 1);
    case Reach::kColumn: return std::min(j + kTopBehind, consumer_rows - 1);
    case Reach::kBlurred: return std::clamp<std::int64_t>(j - kWarmUpSteps, 0, consumer_rows - 1);
    }
    return j;
}

// The rows one consumer row reads at once, which need slots of their own.
std::int64_t window_rows(Reach reach) {
    switch (reach) {
    case Reach::kSubsample: return 2;
    case Reach::kColumn: return kTopBehind + 1;
    default: return 1;
    }
}

// n / d rounded up, for d > 0 and any n.
std::int64_t ceil_div(std::int64_t n, std::int64_t d) { return n >= 0 ? (n + d - 1) / d : -(-n / d); }

// When each stage makes each row, as Butteraugli's engine does
// (butteraugli.cpp): a step's rows read only rows of earlier steps, so they
// can run in any order, and a ring keeps a row until its last reader's step.
// Rows are counted in full-size rows, so that every scale keeps pace with the
// one above it: scale s's row y is at 2^s * y (its `pitch`), and stage s
// makes in step k the rows at [k * band - lag, (k + 1) * band - lag).
struct Schedule {
    std::size_t scales = 0;
    std::int64_t band = 1;
    std::int64_t rows[kStages] = {};
    std::int64_t pitch[kStages] = {};
    std::int64_t lag[kStages] = {};
    std::int64_t steps = 0;
    std::size_t ring_rows[kMaxScales][kGroups] = {};

    std::int64_t step_of(std::size_t stage, std::int64_t row) const {
        return (row * pitch[stage] + lag[stage]) / band;
    }
    // The first row stage `stage` makes in step k or later.
    std::int64_t first_row(std::size_t stage, std::int64_t k) const {
        return std::clamp<std::int64_t>(ceil_div(k * band - lag[stage], pitch[stage]), 0, rows[stage]);
    }
};

// The full-size rows a step makes on `threads` threads for a picture `width`
// pixels wide (a smaller scale makes rows in proportion). One thread makes
// one row per step: its rows then stay in the cache, as in a plain streaming
// engine (64 rows took a quarter longer). Several threads take a band of
// about as many pixels as 16 rows of 4K, so that the rings stay in the
// last-level cache: on 12 threads of a Core Ultra 9 285K, 4K took 2.0x the
// CPU time at 64 rows and 1.2x at 32 or 8 rows, 12 to 24 rows being as
// fast as 16; at 1080p, 16 to 64 rows were as fast, 8 slower. A test build
// with VML_SSIMULACRA2_BAND_FROM_ENV takes it from VML_SSIMULACRA2_BAND, to
// show that every band gives the same score.
std::int64_t band_rows(std::size_t threads, std::size_t width) {
    constexpr std::size_t kBandPixels = 16 * 3840;
    const auto rows_for_width = static_cast<std::int64_t>(kBandPixels / std::max<std::size_t>(width, 1));
    const std::int64_t kBandRows = threads > 1 ? std::clamp<std::int64_t>(rows_for_width, 8, 64) : 1;
#if defined(VML_SSIMULACRA2_BAND_FROM_ENV) && defined(_WIN32)
    char* text = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&text, &length, "VML_SSIMULACRA2_BAND") == 0 && text != nullptr) {
        const long value = std::strtol(text, nullptr, 10);
        std::free(text);
        if (value >= 1 && value <= 1024) return value;
    }
#endif
    return kBandRows;
}

// ---- The scales ----

// A ring of rows, row y in slot y % rows.
struct Ring {
    float* data = nullptr;
    std::size_t rows = 0;
    std::size_t stride = 0;

    float* row(std::int64_t y) const { return data + static_cast<std::size_t>(y) % rows * stride; }
};

// One scale's rings and sums.
struct Scale {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t stride = 0;  // floats per row
    bool ssim_map[kChannels] = {};
    bool edge_maps[kChannels] = {};

    Ring linear[kPictures][kChannels];
    Ring xyb[kPictures][kChannels];
    Ring horizontal[kChannels][kQuantities];  // the horizontal blurs
    Ring vertical[kChannels][kQuantities];    // the vertical blurs, by the step that made them
    float* column_state[kChannels][kQuantities] = {};  // each column strip's 6 rows of blur_column_step's state
    float* zeros = nullptr;

    double ssim_sums[kChannels][2] = {};
    double edge_sums[kChannels][4] = {};

    bool uses(std::size_t c, std::size_t q) const {
        return q <= kMu2 ? ssim_map[c] || edge_maps[c] : ssim_map[c];
    }
    bool blurs(std::size_t c) const { return uses(c, kMu1); }
    bool any_edge_maps() const { return edge_maps[0] || edge_maps[1] || edge_maps[2]; }
    bool any_blurs() const { return blurs(0) || blurs(1) || blurs(2); }
};

// The scales libjxl computes for pictures of this size, while the scale
// before (for the first two, the full picture) is at least 8 by 8, each with
// the maps that `terms` needs.
std::vector<Scale> scales_for(std::size_t width, std::size_t height, Ssimulacra2::Terms terms) {
    std::vector<Scale> scales;
    std::size_t checked_width = width, checked_height = height;
    for (std::size_t s = 0; s < kMaxScales; ++s) {
        if (checked_width < 8 || checked_height < 8) break;
        Scale scale;
        scale.width = width;
        scale.height = height;
        scale.stride = padded_width(width);
        scales.push_back(scale);
        checked_width = width;
        checked_height = height;
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }
    // The weight of each term: channel, then scale (of those computed), then
    // norm, as Msssim::Score walks them.
    for (std::size_t s = 0; s < scales.size(); ++s) {
        for (std::size_t c = 0; c < kChannels; ++c) {
            const double* weight = &kWeights[(c * scales.size() + s) * 6];
            scales[s].ssim_map[c] = terms == Ssimulacra2::Terms::all || weight[0] != 0.0 || weight[3] != 0.0;
            scales[s].edge_maps[c] = terms == Ssimulacra2::Terms::all || weight[1] != 0.0 || weight[2] != 0.0 ||
                                     weight[4] != 0.0 || weight[5] != 0.0;
        }
    }
    return scales;
}

std::vector<Edge> make_edges(const std::vector<Scale>& scales) {
    std::vector<Edge> edges;
    for (std::size_t s = 0; s < scales.size(); ++s) {
        const std::size_t rows = stage_of(s, kRowStage);
        if (s > 0) edges.push_back({rows, stage_of(s - 1, kRowStage), Reach::kSubsample, kLinear});
        if (!scales[s].any_blurs()) continue;
        const std::size_t columns = stage_of(s, kColumnStage);
        const std::size_t sums = stage_of(s, kSumStage);
        edges.push_back({columns, rows, Reach::kColumn, kHorizontal});
        edges.push_back({sums, columns, Reach::kBlurred, kVertical});
        if (scales[s].any_edge_maps()) edges.push_back({sums, rows, Reach::kSame, kXyb});
    }
    return edges;
}

Schedule make_schedule(const std::vector<Scale>& scales, std::int64_t band) {
    Schedule schedule;
    schedule.scales = scales.size();
    for (std::size_t s = 0; s < scales.size(); ++s) {
        const auto height = static_cast<std::int64_t>(scales[s].height);
        const std::int64_t rows[kStageKinds] = {height, height + kWarmUpSteps, height};
        for (std::size_t kind = 0; kind < kStageKinds; ++kind) {
            const std::size_t stage = stage_of(s, static_cast<StageKind>(kind));
            schedule.rows[stage] = rows[kind];
            schedule.pitch[stage] = std::int64_t{1} << s;
        }
    }
    schedule.band = band;
    const std::vector<Edge> edges = make_edges(scales);
    const std::size_t stages = scales.size() * kStageKinds;

    // Lags, in the stages' order (every producer comes before its consumers):
    // the least that puts every row a consumer row reads in an earlier step.
    for (std::size_t c = 0; c < stages; ++c) {
        std::int64_t lag = 0;
        for (const Edge& e : edges) {
            if (e.consumer != c) continue;
            for (std::int64_t y = 0; y < schedule.rows[c]; ++y) {
                const std::int64_t made =
                    schedule.step_of(e.producer, last_read(e.reach, y, schedule.rows[e.producer]));
                lag = std::max(lag, (made + 1) * band - y * schedule.pitch[c]);
            }
        }
        schedule.lag[c] = lag;
    }
    for (std::size_t stage = 0; stage < stages; ++stage) {
        schedule.steps = std::max(schedule.steps, schedule.step_of(stage, schedule.rows[stage] - 1) + 1);
    }

    // Ring sizes: a producer's row j may be overwritten (by row j + rows) only
    // in a step after the last one that reads it; the rows a step makes, and
    // those one consumer row reads, need slots of their own.
    const StageKind producer_of[kGroups] = {kRowStage, kRowStage, kRowStage, kColumnStage};
    for (std::size_t s = 0; s < scales.size(); ++s) {
        for (std::size_t g = 0; g < kGroups; ++g) {
            const std::size_t p = stage_of(s, producer_of[g]);
            std::int64_t ring = ceil_div(band, schedule.pitch[p]);
            for (const Edge& e : edges) {
                if (e.producer != p || e.group != g) continue;
                ring = std::max(ring, window_rows(e.reach));
                for (std::int64_t j = 0; j < schedule.rows[p]; ++j) {
                    const std::int64_t read =
                        schedule.step_of(e.consumer, last_reader(e.reach, j, schedule.rows[e.consumer]));
                    ring = std::max(ring, ceil_div((read + 1) * band - schedule.lag[p], schedule.pitch[p]) - j);
                }
            }
            schedule.ring_rows[s][g] = static_cast<std::size_t>(std::min(ring, schedule.rows[p]));
        }
    }
    return schedule;
}

// With several threads, the vertical blur's tasks each take columns of this
// many floats (a multiple of 16, so whole vectors; the last strip of a row may
// be shorter); one thread takes whole rows.
constexpr std::size_t kStripFloats = 256;

std::size_t strip_floats(std::size_t threads) {
    constexpr std::size_t kWholeRow = std::numeric_limits<std::size_t>::max() / 2;  // wider than any row
    return threads > 1 ? kStripFloats : kWholeRow;
}
std::size_t strips_of(const Scale& scale, std::size_t strip) { return (scale.stride + strip - 1) / strip; }

// Points scale s's rings, states and zero row into `block` (counting only,
// when null), in one order; returns the floats they take.
std::size_t lay_out(Scale& scale, const Schedule& schedule, std::size_t s, float* block) {
    std::size_t used = 0;
    const auto take = [&](std::size_t rows) {
        float* taken = block ? block + used : nullptr;
        used += rows * scale.stride;
        return taken;
    };
    const auto ring = [&](Ring& r, Group g) {
        r.rows = schedule.ring_rows[s][g];
        r.stride = scale.stride;
        r.data = take(r.rows);
    };
    for (std::size_t i = 0; i < kPictures; ++i) {
        for (std::size_t c = 0; c < kChannels; ++c) {
            ring(scale.linear[i][c], kLinear);
            ring(scale.xyb[i][c], kXyb);
        }
    }
    for (std::size_t c = 0; c < kChannels; ++c) {
        for (std::size_t q = 0; q < kQuantities; ++q) {
            if (!scale.uses(c, q)) continue;
            ring(scale.horizontal[c][q], kHorizontal);
            ring(scale.vertical[c][q], kVertical);
            scale.column_state[c][q] = take(kColumnStateRows);
        }
    }
    scale.zeros = take(1);
    return used;
}

// A row for the products the horizontal blur reads, one per thread.
float* product_row(std::size_t stride) {
    thread_local AlignedRows row;
    thread_local std::size_t size = 0;
    if (size < stride) {
        row = allocate_floats(stride);
        size = stride;
    }
    return row.get();
}

// One task of a step: a row of a scale's rows, a column strip of one
// channel's vertical blurs, or one channel's SSIM or edge sums over the
// step's rows.
struct Task {
    std::uint8_t scale;
    std::uint8_t kind;    // StageKind
    std::uint16_t index;  // the column task's channel * strips + strip; the sum task's channel * 2 + (edge maps)
    std::uint32_t first;  // its rows [first, end)
    std::uint32_t end;
};

}  // namespace

class Ssimulacra2::Engine {
public:
    Engine(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
           std::size_t height, Terms terms, std::size_t threads)
        : pool_(threads), strip_(strip_floats(threads)), readings_{source_reading, test_reading}, width_(width),
          height_(height),
          opsin_{opsin_constants(source_reading.intensity_target()),
                 opsin_constants(test_reading.intensity_target())},
          scales_(scales_for(width, height, terms)) {
        if (scales_.empty()) return;
        const Schedule schedule = make_schedule(scales_, band_rows(threads, width));
        allocate(schedule);
        make_tasks(schedule);
    }

    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }
    const std::vector<ScaleAverages>& averages() const { return averages_; }

    double score(const std::uint16_t* source, const std::uint16_t* test, std::size_t stride) {
        pictures_[0] = source;
        pictures_[1] = test;
        stride_ = stride;
        for (Scale& scale : scales_) start(scale);
        run_steps();
        averages_.clear();
        for (const Scale& scale : scales_) averages_.push_back(averages_of(scale));
        return score_of(averages_);
    }

private:
    // One block for every scale's rings: zeros, or NaNs in a build with
    // VML_SSIMULACRA2_POISON, to show that no pixel reads a row before it is
    // written. The zero rows are zeros in both.
    void allocate(const Schedule& schedule) {
        std::size_t total = 0;
        for (std::size_t s = 0; s < scales_.size(); ++s) total += lay_out(scales_[s], schedule, s, nullptr);
        memory_ = allocate_floats(total);
#ifdef VML_SSIMULACRA2_POISON
        std::fill_n(memory_.get(), total, std::numeric_limits<float>::quiet_NaN());
#endif
        float* next = memory_.get();
        for (std::size_t s = 0; s < scales_.size(); ++s) {
            next += lay_out(scales_[s], schedule, s, next);
            std::fill_n(scales_[s].zeros, scales_[s].stride, 0.0f);
        }
    }

    // Every step's tasks: the sums first (each one long, as its rows must be
    // added in order), then the column strips, then the rows, which the
    // threads that end a step share; the largest scale first in each.
    void make_tasks(const Schedule& schedule) {
        for (std::int64_t k = 0; k < schedule.steps; ++k) {
            step_begin_.push_back(tasks_.size());
            for (const StageKind kind : {kSumStage, kColumnStage, kRowStage}) {
                for (std::size_t s = 0; s < scales_.size(); ++s) {
                    const std::size_t stage = stage_of(s, kind);
                    const std::int64_t first = schedule.first_row(stage, k);
                    const std::int64_t end = schedule.first_row(stage, k + 1);
                    if (first < end) add_tasks(s, kind, first, end);
                }
            }
        }
        step_begin_.push_back(tasks_.size());
    }

    void add_tasks(std::size_t s, StageKind kind, std::int64_t first, std::int64_t end) {
        const Scale& scale = scales_[s];
        const auto add = [&](std::size_t index, std::int64_t from, std::int64_t to) {
            tasks_.push_back({static_cast<std::uint8_t>(s), static_cast<std::uint8_t>(kind),
                              static_cast<std::uint16_t>(index), static_cast<std::uint32_t>(from),
                              static_cast<std::uint32_t>(to)});
        };
        switch (kind) {
        case kRowStage:
            for (std::int64_t y = first; y < end; ++y) add(0, y, y + 1);
            break;
        case kColumnStage:
            for (std::size_t c = 0; c < kChannels; ++c) {
                if (!scale.blurs(c)) continue;
                const std::size_t strips = strips_of(scale, strip_);
                for (std::size_t strip = 0; strip < strips; ++strip) add(c * strips + strip, first, end);
            }
            break;
        case kSumStage:
            for (std::size_t c = 0; c < kChannels; ++c) {
                if (scale.ssim_map[c]) add(c * 2, first, end);
                if (scale.edge_maps[c]) add(c * 2 + 1, first, end);
            }
            break;
        default: break;
        }
    }

    // A new picture pair: the vertical filters start from zero (libjxl clears
    // its ring buffer per column strip) and so do the sums.
    static void start(Scale& scale) {
        for (std::size_t c = 0; c < kChannels; ++c) {
            for (std::size_t q = 0; q < kQuantities; ++q) {
                if (scale.uses(c, q)) std::fill_n(scale.column_state[c][q], kColumnStateRows * scale.stride, 0.0f);
            }
            std::fill(std::begin(scale.ssim_sums[c]), std::end(scale.ssim_sums[c]), 0.0);
            std::fill(std::begin(scale.edge_sums[c]), std::end(scale.edge_sums[c]), 0.0);
        }
    }

    // A task that throws (a thread's first product row not allocated) must
    // not leave the pool, whose threads cannot throw: the step's other tasks
    // are skipped and the error is thrown here. The next pair starts afresh
    // (start(), and every row is written before it is read).
    void run_steps() {
        std::mutex failure_mutex;
        std::exception_ptr failure;
        std::atomic<bool> failed{false};
        for (std::size_t k = 0; k + 1 < step_begin_.size(); ++k) {
            const std::size_t begin = step_begin_[k];
            pool_.run_each(step_begin_[k + 1] - begin, [&](std::size_t i) {
                if (failed.load(std::memory_order_relaxed)) return;
                try {
                    run_task(tasks_[begin + i]);
                } catch (...) {
                    const std::lock_guard lock(failure_mutex);
                    if (!failure) failure = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            });
            if (failed.load(std::memory_order_relaxed)) std::rethrow_exception(failure);
        }
    }

    void run_task(const Task& task) {
        Scale& scale = scales_[task.scale];
        switch (task.kind) {
        case kRowStage: make_row(task.scale, task.first); break;
        case kColumnStage:
            blur_columns(scale, task.index / strips_of(scale, strip_), task.index % strips_of(scale, strip_),
                         task.first, task.end);
            break;
        case kSumStage: add_map_rows(scale, task.index / 2, task.index % 2 != 0, task.first, task.end); break;
        default: break;
        }
    }

    // Row y of scale s: its linear rows (converted from the pictures at the
    // full size, else the 2x2 means of two rows of the scale above, an odd
    // last row taken twice), XYB, products and horizontal blurs.
    void make_row(std::size_t s, std::size_t y) {
        Scale& scale = scales_[s];
        float* linear[kPictures][kChannels];
        for (std::size_t i = 0; i < kPictures; ++i) {
            for (std::size_t c = 0; c < kChannels; ++c) linear[i][c] = scale.linear[i][c].row(y);
            if (s == 0) {
                readings_[i].convert_row(pictures_[i] + y * stride_, width_, linear[i][0], linear[i][1], linear[i][2]);
                continue;
            }
            const Scale& above = scales_[s - 1];
            const std::size_t upper = 2 * y;
            const std::size_t lower = std::min(2 * y + 1, above.height - 1);
            for (std::size_t c = 0; c < kChannels; ++c) {
                downsample_row(above.linear[i][c].row(upper), above.linear[i][c].row(lower), above.width,
                               linear[i][c], scale.width);
            }
        }
        float* xyb[kPictures][kChannels];
        for (std::size_t i = 0; i < kPictures; ++i) {
            for (std::size_t c = 0; c < kChannels; ++c) xyb[i][c] = scale.xyb[i][c].row(y);
            kernels::positive_xyb_row(linear[i][0], linear[i][1], linear[i][2], opsin_[i].data(), xyb[i][0],
                                      xyb[i][1], xyb[i][2], scale.stride);
        }
        for (std::size_t c = 0; c < kChannels; ++c) {
            if (!scale.blurs(c)) continue;
            const float* img1 = xyb[0][c];
            const float* img2 = xyb[1][c];
            kernels::blur_row(img1, scale.horizontal[c][kMu1].row(y), scale.width);
            kernels::blur_row(img2, scale.horizontal[c][kMu2].row(y), scale.width);
            if (!scale.ssim_map[c]) continue;
            float* product = product_row(scale.stride);
            const std::pair<const float*, const float*> factors[] = {{img1, img1}, {img2, img2}, {img1, img2}};
            for (std::size_t k = 0; k < 3; ++k) {
                kernels::multiply_row(factors[k].first, factors[k].second, product, scale.stride);
                kernels::blur_row(product, scale.horizontal[c][kSigma11 + k].row(y), scale.width);
            }
        }
    }

    // The vertical blur's steps [first, end) of channel c's planes, in one
    // column strip. FastGaussianVertical's input for output row n = m - 4 is
    // the horizontal blur of row m alone up to row 5, then rows m - 10 and
    // m; past the bottom, a row of zeros stands in for row m.
    void blur_columns(Scale& scale, std::size_t c, std::size_t strip, std::int64_t first, std::int64_t end) const {
        const std::size_t x0 = strip * strip_;
        const std::size_t count = std::min(strip_, scale.stride - x0);
        const auto height = static_cast<std::int64_t>(scale.height);
        for (std::size_t q = 0; q < kQuantities; ++q) {
            if (!scale.uses(c, q)) continue;
            const Ring& horizontal = scale.horizontal[c][q];
            const Ring& vertical = scale.vertical[c][q];
            float* state = scale.column_state[c][q] + kColumnStateRows * x0;
            for (std::int64_t m = first; m < end; ++m) {
                const float* bottom = (m < height ? horizontal.row(m) : scale.zeros) + x0;
                const float* top = m >= kTopBehind ? horizontal.row(m - kTopBehind) + x0 : nullptr;
                float* out = m >= kWarmUpSteps ? vertical.row(m) + x0 : nullptr;
                kernels::blur_column_step(top, bottom, state, out, count);
            }
        }
    }

    // Rows [first, end) of channel c's SSIM map or edge maps, added to their
    // sums in order.
    static void add_map_rows(Scale& scale, std::size_t c, bool edge, std::int64_t first, std::int64_t end) {
        const Ring* blurred = scale.vertical[c];
        for (std::int64_t n = first; n < end; ++n) {
            const std::int64_t m = n + kWarmUpSteps;
            if (edge) {
                add_edge_row(scale.xyb[0][c].row(n), blurred[kMu1].row(m), scale.xyb[1][c].row(n),
                             blurred[kMu2].row(m), scale.width, scale.edge_sums[c]);
            } else {
                add_ssim_row(blurred[kMu1].row(m), blurred[kMu2].row(m), blurred[kSigma11].row(m),
                             blurred[kSigma22].row(m), blurred[kSigma12].row(m), scale.width, scale.ssim_sums[c]);
            }
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

    RowPool pool_;
    std::size_t strip_;  // the vertical blur's column strip, in floats (strip_floats)
    std::array<JxlLinearRgb, kPictures> readings_;  // source's, test's
    std::size_t width_;
    std::size_t height_;
    std::array<std::array<float, 11>, kPictures> opsin_;  // each picture's
    std::vector<Scale> scales_;
    AlignedRows memory_;
    std::vector<Task> tasks_;
    std::vector<std::size_t> step_begin_;  // each step's first task in tasks_, then tasks_'s size
    const std::uint16_t* pictures_[kPictures] = {};
    std::size_t stride_ = 0;
    std::vector<ScaleAverages> averages_;
};

Ssimulacra2::Ssimulacra2(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                         std::size_t height, Terms terms, std::size_t threads)
    : engine_(std::make_unique<Engine>(source_reading, test_reading, width, height, terms, threads)) {}
Ssimulacra2::~Ssimulacra2() = default;
Ssimulacra2::Ssimulacra2(Ssimulacra2&&) noexcept = default;
Ssimulacra2& Ssimulacra2::operator=(Ssimulacra2&&) noexcept = default;

std::size_t Ssimulacra2::width() const { return engine_->width(); }
std::size_t Ssimulacra2::height() const { return engine_->height(); }

std::size_t Ssimulacra2::memory_bytes(std::size_t width, std::size_t height, std::size_t threads) {
    std::vector<Scale> scales = scales_for(width, height, Terms::weighted);
    if (scales.empty()) return 0;
    const Schedule schedule = make_schedule(scales, band_rows(threads, width));
    std::size_t floats = threads * scales[0].stride;  // each thread's product row
    for (std::size_t s = 0; s < scales.size(); ++s) floats += lay_out(scales[s], schedule, s, nullptr);
    return floats * sizeof(float);
}

double Ssimulacra2::score(const std::uint16_t* source, const std::uint16_t* test, std::size_t stride) {
    return engine_->score(source, test, stride);
}

const std::vector<Ssimulacra2::ScaleAverages>& Ssimulacra2::averages() const { return engine_->averages(); }

bool Ssimulacra2::fused_multiply_add() { return kernels::fused_multiply_add(); }

}  // namespace vml
