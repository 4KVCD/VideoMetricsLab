#include "vml/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>

#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml {

namespace {

constexpr std::array<Threshold, 6> vmaf_thresholds{{
    {'>', 95.0}, {'>', 90.0}, {'>', 85.0}, {'<', 85.0}, {'<', 80.0}, {'<', 70.0}}};
constexpr std::array<Threshold, 6> psnr_thresholds{{
    {'>', 41.0}, {'>', 38.0}, {'>', 35.0}, {'<', 35.0}, {'<', 34.0}, {'<', 32.0}}};
constexpr std::array<Threshold, 6> ssim_thresholds{{
    {'>', 0.99}, {'>', 0.98}, {'>', 0.97}, {'<', 0.97}, {'<', 0.96}, {'<', 0.95}}};
constexpr std::array<Threshold, 6> xpsnr_thresholds{{
    {'>', 38.0}, {'>', 35.0}, {'>', 33.0}, {'<', 33.0}, {'<', 30.0}, {'<', 27.0}}};
constexpr std::array<Threshold, 6> ssimulacra2_thresholds{{
    {'>', 90.0}, {'>', 80.0}, {'>', 70.0}, {'<', 70.0}, {'<', 50.0}, {'<', 30.0}}};

constexpr FfmpegMetricBinding bool_option(std::string_view name) { return {name, {}}; }
constexpr FfmpegMetricBinding libvmaf_feature(std::string_view name) { return {{}, name}; }

// The smallest width and height VMAF is calculated at: the GPU's VMAF
// engine (vmaf_vulkan) refuses anything smaller, and libvmaf's CPU code
// reads past its pictures under 17 pixels (ADM) or 8 rows (VIF). One limit
// for both, so a video is scored, or refused, alike on either.
constexpr int vmaf_smallest_side = 32;

using enum MetricKind;
using enum MetricDirection;
using enum MetricAggregation;

// In the order the app shows them. Key "vmaf" is the v0.6.1 family (its
// standard and 4K models), as it always was: saved scores, settings and
// exports keep the key; only the label names the version.
const std::array<MetricDefinition, 17> registry{{
    {"vmaf", "VMAF v0.6.1", "VMAF v0.6.1", "VMAF v0.6.1", "VMAF v0.6.1", 2, "", frame, higher_is_better,
     arithmetic, 100.0, vmaf_thresholds, bool_option("compute_vmaf"), {}, vmaf_smallest_side},
    {"vmaf_neg", "VMAF NEG", "VMAF NEG", "VMAF NEG", "VMAF NEG", 2, "", frame, higher_is_better, arithmetic,
     100.0, vmaf_thresholds, bool_option("compute_vmaf_neg"), {}, vmaf_smallest_side},
    // Netflix's VMAF v1 models, calculated in the same libvmaf pass. No
    // fixed axis: the 4K/3H model scores up to 110.
    {"vmaf_v1", "VMAF v1", "VMAF v1", "VMAF v1", "VMAF v1", 2, "", frame, higher_is_better, arithmetic,
     std::nullopt, vmaf_thresholds, bool_option("compute_vmaf_v1"), {}, vmaf_smallest_side},
    // Netflix's HDR VMAF models: both videos converted to ICtCp (PQ) at the
    // model's display size first, then scored as VMAF v1 is (on the GPU, or
    // by libvmaf). PQ videos only.
    {"vmaf_hdr", "HDR VMAF", "HDR VMAF", "HDR VMAF", "HDR VMAF", 2, "", frame, higher_is_better, arithmetic,
     std::nullopt, vmaf_thresholds, bool_option("compute_vmaf_hdr"), {}, vmaf_smallest_side},
    {"psnr", "PSNR", "PSNR", "PSNR (dB)", "PSNR (dB)", 2, " dB", frame, higher_is_better, arithmetic,
     std::nullopt, psnr_thresholds, libvmaf_feature("name=psnr"), {}, 0},
    {"ssim", "SSIM", "SSIM", "SSIM", "SSIM", 4, "", frame, higher_is_better, arithmetic, std::nullopt,
     ssim_thresholds, libvmaf_feature("name=float_ssim"), {}, 11},
    // PSNR and SSIM of the three planes weighted 6:1:1 ((6 Y + Cb + Cr) / 8),
    // as JVET's common test conditions weigh PSNR and MSU's codec comparisons
    // weigh YUV-SSIM. Scored by the app's own
    // libvmaf-fast only: FFmpeg's filter has no SSIM of the chroma. Chroma
    // needs 11 samples for SSIM's window, so twice that for 4:2:0.
    {"psnr_yuv", "PSNR YUV (6:1:1)", "PSNR YUV", "PSNR YUV (dB)", "PSNR YUV (dB)", 2, " dB", frame,
     higher_is_better, arithmetic, std::nullopt, psnr_thresholds, bool_option("compute_psnr_yuv"), {}, 0},
    {"ssim_yuv", "SSIM YUV (6:1:1)", "SSIM YUV", "SSIM YUV", "SSIM YUV", 4, "", frame, higher_is_better,
     arithmetic, std::nullopt, ssim_thresholds, bool_option("compute_ssim_yuv"), {}, 22},
    // XPSNR is the one metric that reports infinity (an identical frame),
    // and is averaged its own way (stats.h).
    {"xpsnr", "XPSNR", "XPSNR", "XPSNR (dB)", "XPSNR (dB)", 2, " dB", frame, higher_is_better,
     square_mean_root_db, std::nullopt, xpsnr_thresholds, bool_option("compute_xpsnr"), {}, 0},
    // Calculated by the app's own libvmaf-fast only (libvmaf_cpu.h), off by
    // default. Their size and bit-depth limits fail the metric alone
    // (cpu_metrics_unfit), not the run, so their smallest side is 0 here.
    // MS-SSIM: SSIM at five scales (Wang et al. 2003); MS-SSIM YUV its planes
    // weighted 6:1:1 as SSIM YUV.
    {"ms_ssim", "MS-SSIM", "MS-SSIM", "MS-SSIM", "MS-SSIM", 4, "", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_ms_ssim"), {}, 0},
    {"ms_ssim_yuv", "MS-SSIM YUV (6:1:1)", "MS-SSIM YUV", "MS-SSIM YUV", "MS-SSIM YUV", 4, "", frame,
     higher_is_better, arithmetic, std::nullopt, {}, bool_option("compute_ms_ssim_yuv"), {}, 0},
    // PSNR-HVS-M (Ponomarenko et al. 2007, Xiph's code): PSNR of the
    // contrast-masked, contrast-sensitivity-weighted DCT error of 8x8 blocks,
    // 0.8 Y + 0.1 Cb + 0.1 Cr in the error. +infinity for an identical (or
    // fully masked) frame, so averaged as its error.
    {"psnr_hvs", "PSNR-HVS-M", "PSNR-HVS-M", "PSNR-HVS-M (dB)", "PSNR-HVS-M (dB)", 2, " dB", frame,
     higher_is_better, mean_error_db, std::nullopt, {}, bool_option("compute_psnr_hvs"), {}, 0},
    // CAMBI: Netflix's banding of the test video alone (0 none, about 5
    // annoying, 24 the most Netflix has seen; libvmaf clips at 1000).
    {"cambi", "CAMBI", "CAMBI", "CAMBI", "CAMBI", 3, "", frame, lower_is_better, arithmetic, std::nullopt, {},
     bool_option("compute_cambi"), {}, 0},
    // CIEDE2000: the mean colour difference ΔE00 of the frame's pixels
    // (0 identical, about 1 just noticeable).
    {"ciede2000", "CIEDE2000 (ΔE)", "CIEDE2000", "CIEDE2000 (ΔE)", "CIEDE2000 (ΔE)", 3, "", frame,
     lower_is_better, arithmetic, std::nullopt, {}, bool_option("compute_ciede2000"), {}, 0},
    {"ssimulacra2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", 2, "", frame, higher_is_better,
     arithmetic, 100.0, ssimulacra2_thresholds, std::nullopt, "perceptual", 8},
    {"butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", 4, "", frame, lower_is_better,
     arithmetic, std::nullopt, {}, std::nullopt, "perceptual", 8},
    // ColorVideoVDP: one score per video in JOD (just-objectionable
    // differences; 10 = no visible difference). GPU only.
    {"cvvdp", "CVVDP", "CVVDP", "CVVDP", "CVVDP (JOD)", 3, "", sequence, higher_is_better, arithmetic, 10.0, {},
     std::nullopt, "perceptual", 0},
}};

}  // namespace

std::string MetricDefinition::format_value(std::optional<double> value) const {
    if (!value || std::isnan(*value)) {
        return "—";
    }
    if (std::isinf(*value)) {
        return *value > 0 ? "∞" : "−∞";  // U+2212, the minus sign, as the app shows it
    }
    return format_fixed(*value, decimals);
}

std::string MetricDefinition::format_delta(std::optional<double> value) const {
    if (!value || std::isnan(*value)) {
        return "—";
    }
    if (std::isinf(*value)) {
        return *value > 0 ? "+∞" : "−∞";
    }
    return (std::signbit(*value) ? "" : "+") + format_fixed(*value, decimals);
}

std::vector<ScoreMark> MetricDefinition::best_and_worst(const std::vector<std::optional<double>>& scores) const {
    std::vector<ScoreMark> marks(scores.size(), ScoreMark::none);
    std::vector<std::size_t> order;  // the videos with a score, lowest first (equal ones in their order)
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (scores[index] && !std::isnan(*scores[index])) {
            order.push_back(index);
        }
    }
    if (order.size() < 2) {
        return marks;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&scores](std::size_t a, std::size_t b) { return *scores[a] < *scores[b]; });
    const bool lowest_best = direction == MetricDirection::lower_is_better;
    const std::string best = format_value(scores[lowest_best ? order.front() : order.back()]);
    const std::string worst = format_value(scores[lowest_best ? order.back() : order.front()]);
    if (best == worst) {
        return marks;
    }
    for (const std::size_t index : order) {
        const std::string shown = format_value(scores[index]);
        marks[index] = shown == best ? ScoreMark::best : shown == worst ? ScoreMark::worst : ScoreMark::none;
    }
    return marks;
}

std::span<const MetricDefinition> metrics() { return registry; }

const MetricDefinition* find_metric(std::string_view key) {
    for (const MetricDefinition& metric : registry) {
        if (metric.key == key) {
            return &metric;
        }
    }
    return nullptr;
}

const MetricDefinition& metric_definition(std::string_view key) {
    if (const MetricDefinition* metric = find_metric(key)) {
        return *metric;
    }
    throw std::out_of_range("unknown metric: " + std::string(key));
}

std::optional<std::string> too_small_problem(const std::vector<std::string>& keys, int width, int height) {
    // The metrics the pictures are too small for, by the size each needs,
    // largest first: "VMAF v0.6.1 and VMAF NEG (at least 32x32); SSIM (...)".
    std::map<int, std::vector<std::string_view>, std::greater<>> refused;
    for (const std::string& key : keys) {
        const MetricDefinition& metric = metric_definition(key);
        if (width < metric.smallest_side || height < metric.smallest_side) {
            refused[metric.smallest_side].push_back(metric.label);
        }
    }
    if (refused.empty()) {
        return std::nullopt;
    }
    std::vector<std::string> needs;
    for (const auto& [side, labels] : refused) {
        const std::string size = std::to_string(side);
        needs.push_back(join(labels, " and ") + " (at least " + size + "x" + size + ")");
    }
    return "The compared pictures, " + std::to_string(width) + "x" + std::to_string(height) +
           ", are too small for " + join(needs, "; ") + ".";
}

std::string metric_labels(const std::vector<std::string>& keys, std::string_view separator) {
    std::vector<std::string_view> labels;
    labels.reserve(keys.size());
    for (const std::string& key : keys) {
        labels.push_back(metric_definition(key).label);
    }
    return join(labels, separator);
}

}  // namespace vml
