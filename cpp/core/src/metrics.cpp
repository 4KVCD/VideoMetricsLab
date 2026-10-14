#include "vml/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "vml/number_format.h"

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

using enum MetricKind;
using enum MetricDirection;
using enum MetricAggregation;

// In the order the app shows them. Key "vmaf" is the v0.6.1 family (its
// standard and 4K models), as it always was: saved scores, settings and
// exports keep the key; only the label names the version.
const std::array<MetricDefinition, 9> registry{{
    {"vmaf", "VMAF v0.6.1", "VMAF v0.6.1", "VMAF v0.6.1", "VMAF v0.6.1", 2, "", frame, higher_is_better,
     arithmetic, 100.0, vmaf_thresholds, bool_option("compute_vmaf"), {}},
    {"vmaf_neg", "VMAF NEG", "VMAF NEG", "VMAF NEG", "VMAF NEG", 2, "", frame, higher_is_better, arithmetic,
     100.0, vmaf_thresholds, bool_option("compute_vmaf_neg"), {}},
    // Netflix's VMAF v1 models, calculated in the same libvmaf pass. No
    // fixed axis: the 4K/3H model scores up to 110.
    {"vmaf_v1", "VMAF v1", "VMAF v1", "VMAF v1", "VMAF v1", 2, "", frame, higher_is_better, arithmetic,
     std::nullopt, vmaf_thresholds, bool_option("compute_vmaf_v1"), {}},
    {"psnr", "PSNR", "PSNR", "PSNR (dB)", "PSNR (dB)", 2, " dB", frame, higher_is_better, arithmetic,
     std::nullopt, psnr_thresholds, libvmaf_feature("name=psnr"), {}},
    {"ssim", "SSIM", "SSIM", "SSIM", "SSIM", 4, "", frame, higher_is_better, arithmetic, std::nullopt,
     ssim_thresholds, libvmaf_feature("name=float_ssim"), {}},
    // XPSNR is the one metric that reports infinity (an identical frame),
    // and is averaged its own way (stats.h).
    {"xpsnr", "XPSNR", "XPSNR", "XPSNR (dB)", "XPSNR (dB)", 2, " dB", frame, higher_is_better,
     square_mean_root_db, std::nullopt, xpsnr_thresholds, bool_option("compute_xpsnr"), {}},
    {"ssimulacra2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", 2, "", frame, higher_is_better,
     arithmetic, 100.0, ssimulacra2_thresholds, std::nullopt, "perceptual"},
    {"butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", 4, "", frame, lower_is_better,
     arithmetic, std::nullopt, {}, std::nullopt, "perceptual"},
    // ColorVideoVDP: one score per video in JOD (just-objectionable
    // differences; 10 = no visible difference). GPU only.
    {"cvvdp", "CVVDP", "CVVDP", "CVVDP", "CVVDP (JOD)", 3, "", sequence, higher_is_better, arithmetic, 10.0, {},
     std::nullopt, "perceptual"},
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

}  // namespace vml
