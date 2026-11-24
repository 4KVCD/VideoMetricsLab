#include "vml/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>

#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/translatable.h"

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
const std::array<MetricDefinition, 29> registry{{
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
    // The planes of PSNR YUV, SSIM YUV, MS-SSIM YUV and PSNR-HVS-M, each
    // after its metric (shown with Add/remove metrics' "Y, U and V of the
    // YUV metrics"), scored in that metric's pass. No thresholds: a chroma
    // plane's PSNR runs far higher than the luma's.
    {"psnr_yuv_y", "PSNR YUV: Y", "PSNR Y", "Y", "PSNR Y (dB)", 2, " dB", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_psnr_yuv_y"), {}, 0, "psnr_yuv"},
    {"psnr_yuv_u", "PSNR YUV: U", "PSNR U", "U", "PSNR U (dB)", 2, " dB", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_psnr_yuv_u"), {}, 0, "psnr_yuv"},
    {"psnr_yuv_v", "PSNR YUV: V", "PSNR V", "V", "PSNR V (dB)", 2, " dB", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_psnr_yuv_v"), {}, 0, "psnr_yuv"},
    {"ssim_yuv", "SSIM YUV (6:1:1)", "SSIM YUV", "SSIM YUV", "SSIM YUV", 4, "", frame, higher_is_better,
     arithmetic, std::nullopt, ssim_thresholds, bool_option("compute_ssim_yuv"), {}, 22},
    {"ssim_yuv_y", "SSIM YUV: Y", "SSIM Y", "Y", "SSIM Y", 4, "", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_ssim_yuv_y"), {}, 22, "ssim_yuv"},
    {"ssim_yuv_u", "SSIM YUV: U", "SSIM U", "U", "SSIM U", 4, "", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_ssim_yuv_u"), {}, 22, "ssim_yuv"},
    {"ssim_yuv_v", "SSIM YUV: V", "SSIM V", "V", "SSIM V", 4, "", frame, higher_is_better, arithmetic,
     std::nullopt, {}, bool_option("compute_ssim_yuv_v"), {}, 22, "ssim_yuv"},
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
    {"ms_ssim_yuv_y", "MS-SSIM YUV: Y", "MS-SSIM Y", "Y", "MS-SSIM Y", 4, "", frame, higher_is_better,
     arithmetic, std::nullopt, {}, bool_option("compute_ms_ssim_yuv_y"), {}, 0, "ms_ssim_yuv"},
    {"ms_ssim_yuv_u", "MS-SSIM YUV: U", "MS-SSIM U", "U", "MS-SSIM U", 4, "", frame, higher_is_better,
     arithmetic, std::nullopt, {}, bool_option("compute_ms_ssim_yuv_u"), {}, 0, "ms_ssim_yuv"},
    {"ms_ssim_yuv_v", "MS-SSIM YUV: V", "MS-SSIM V", "V", "MS-SSIM V", 4, "", frame, higher_is_better,
     arithmetic, std::nullopt, {}, bool_option("compute_ms_ssim_yuv_v"), {}, 0, "ms_ssim_yuv"},
    // PSNR-HVS-M (Ponomarenko et al. 2007, Xiph's code): PSNR of the
    // contrast-masked, contrast-sensitivity-weighted DCT error of 8x8 blocks,
    // 0.8 Y + 0.1 Cb + 0.1 Cr in the error. +infinity for an identical (or
    // fully masked) frame, so averaged as its error.
    {"psnr_hvs", "PSNR-HVS-M", "PSNR-HVS-M", "PSNR-HVS-M (dB)", "PSNR-HVS-M (dB)", 2, " dB", frame,
     higher_is_better, mean_error_db, std::nullopt, {}, bool_option("compute_psnr_hvs"), {}, 0},
    {"psnr_hvs_y", "PSNR-HVS-M: Y", "PSNR-HVS-M Y", "Y", "PSNR-HVS-M Y (dB)", 2, " dB", frame,
     higher_is_better, mean_error_db, std::nullopt, {}, bool_option("compute_psnr_hvs_y"), {}, 0, "psnr_hvs"},
    {"psnr_hvs_u", "PSNR-HVS-M: U", "PSNR-HVS-M U", "U", "PSNR-HVS-M U (dB)", 2, " dB", frame,
     higher_is_better, mean_error_db, std::nullopt, {}, bool_option("compute_psnr_hvs_u"), {}, 0, "psnr_hvs"},
    {"psnr_hvs_v", "PSNR-HVS-M: V", "PSNR-HVS-M V", "V", "PSNR-HVS-M V (dB)", 2, " dB", frame,
     higher_is_better, mean_error_db, std::nullopt, {}, bool_option("compute_psnr_hvs_v"), {}, 0, "psnr_hvs"},
    // CAMBI: Netflix's banding of the test video alone (0 none, about 5
    // annoying, 24 the most Netflix has seen; libvmaf clips at 1000).
    {"cambi", "CAMBI", "CAMBI", "CAMBI", "CAMBI", 3, "", frame, lower_is_better, arithmetic, std::nullopt, {},
     bool_option("compute_cambi"), {}, 0},
    // CIEDE2000: the mean color difference ΔE00 of the frame's pixels
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

std::string_view scored_by(std::string_view key) {
    const MetricDefinition& metric = metric_definition(key);
    return metric.is_plane() ? metric.plane_of : metric.key;
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

namespace {

// The app's bands (the thresholds above) in words: the edges Metric Graphs'
// statistics count frames past.
constexpr const char* vmaf_ranges = N_("The app's bands: above 95 hard to tell apart, 90 to 95 very good, 85 to 90 good, 80 to 85 some loss, 70 to 80 visible loss, below 70 poor.");
constexpr const char* psnr_ranges = N_("The app's bands: above 41 dB excellent, 38 to 41 very good, 35 to 38 good, 34 to 35 some loss, 32 to 34 visible loss, below 32 poor.");
constexpr const char* ssim_ranges = N_("The app's bands: above 0.99 excellent, 0.98 to 0.99 very good, 0.97 to 0.98 good, 0.96 to 0.97 some loss, 0.95 to 0.96 visible loss, below 0.95 poor.");

// Each listed metric's guide (metric_guide), by key.
struct KeyedGuide {
    std::string_view key;
    MetricGuide guide;
};
const std::array<KeyedGuide, 17> guides{{
    {"vmaf",
     {N_("Netflix's score, trained on viewers"),
      N_("Netflix's VMAF: how viewers would rate the test video against the reference, by a model trained on their ratings. 0 to 100, higher is better."),
      vmaf_ranges, "the app's VMAF bands, vmaf_thresholds above"}},
    {"vmaf_neg",
     {N_("VMAF without credit for sharpening"),
      N_("VMAF without the credit it gives sharpening and contrast boosts (No Enhancement Gain), so it runs a few points below VMAF where an encoder or filter sharpens. 0 to 100, higher is better."),
      vmaf_ranges, "the app's VMAF bands, vmaf_thresholds above"}},
    {"vmaf_v1",
     {N_("Netflix's newer VMAF models"),
      N_("Netflix's VMAF v1 models, calculated in the same pass as VMAF v0.6.1. 0 to 100 (the 4K model up to 110), higher is better."),
      vmaf_ranges, "the app's VMAF bands, vmaf_thresholds above"}},
    {"vmaf_hdr",
     {N_("VMAF of HDR10 video"),
      N_("Netflix's HDR VMAF: both videos converted to ICtCp (PQ), then scored as VMAF v1 is. PQ (HDR10) videos only: another video fails HDR VMAF alone, with the reason. Higher is better."),
      vmaf_ranges, "the app's VMAF bands, vmaf_thresholds above"}},
    {"psnr",
     {N_("pixel error of the brightness, in dB"),
      N_("Peak signal-to-noise ratio of the luma (Y) plane: the pixels' mean squared error, in dB. Higher is better."),
      psnr_ranges, "the app's PSNR bands, psnr_thresholds above"}},
    {"ssim",
     {N_("structure kept, of the brightness"),
      N_("Structural similarity of the luma (Y) plane: how much of the pictures' local structure, contrast and brightness is kept. Up to 1, higher is better."),
      ssim_ranges, "the app's SSIM bands, ssim_thresholds above"}},
    {"psnr_yuv",
     {N_("PSNR with color, as JVET weighs it"),
      N_("(6 x Y + Cb + Cr) / 8 of the three planes' PSNR, as JVET's common test conditions weigh it. In dB, higher is better."),
      psnr_ranges, "the app's PSNR bands, psnr_thresholds above"}},
    {"ssim_yuv",
     {N_("SSIM with color, as MSU weighs it"),
      N_("(6 x Y + Cb + Cr) / 8 of the three planes' SSIM, as MSU's codec comparisons weigh YUV-SSIM. Up to 1, higher is better."),
      ssim_ranges, "the app's SSIM bands, ssim_thresholds above"}},
    {"xpsnr",
     {N_("PSNR weighted by how visible errors are"),
      N_("Extended perceptually weighted PSNR (XPSNR) of the luma (Y) plane: each block's error weighted by how visible it is. In dB, higher is better."),
      N_("The app's bands: above 38 dB excellent, 35 to 38 very good, 33 to 35 good, 30 to 33 some loss, 27 to 30 visible loss, below 27 poor."),
      "the app's XPSNR bands, xpsnr_thresholds above"}},
    {"ms_ssim",
     {N_("SSIM at five scales, as MSU reports it"),
      N_("SSIM of the luma (Y) plane at five scales (Wang et al. 2003), as MSU's codec comparisons report it. Up to 1, higher is better."),
      {}}},
    {"ms_ssim_yuv",
     {N_("MS-SSIM with color"),
      N_("(6 x Y + Cb + Cr) / 8 of the three planes' MS-SSIM, as SSIM YUV weighs them. Up to 1, higher is better."),
      {}}},
    {"psnr_hvs",
     {N_("PSNR weighted by the eye, as Xiph reports it"),
      N_("PSNR of the 8x8 blocks' error weighted by the eye's sensitivity to contrast and its masking (Ponomarenko et al. 2007), of 0.8 Y + 0.1 Cb + 0.1 Cr, as Xiph's codec tests report it. In dB, higher is better."),
      {}}},
    {"cambi",
     {N_("banding in the test video"),
      N_("Netflix's banding score of the test video alone, without the reference. Lower is better."),
      N_("Netflix's scale: 0 is no banding, about 5 starts to annoy, and 24 is the most Netflix has seen."),
      "Netflix's CAMBI scale, as the user guide and the registry's CAMBI entry give it"}},
    {"ciede2000",
     {N_("color difference"),
      N_("The mean color difference (CIEDE2000 ΔE) of the pixels. Lower is better."),
      N_("CIEDE2000's scale: 0 is identical, and about 1 is just noticeable."),
      "the CIEDE2000 (ΔE00) scale, as the user guide and the registry's CIEDE2000 entry give it"}},
    {"ssimulacra2",
     {N_("JPEG XL's perceptual score"),
      N_("SSIMULACRA2, from the JPEG XL project: a perceptual score of the structure, color and edge errors at several scales. 100 is identical, higher is better."),
      N_("The app's bands are at 90, 80, 70, 50 and 30; the JPEG XL project calls 90 very high quality (hard to tell from the original side by side), 70 high, 50 medium and 30 low."),
      "the app's SSIMULACRA2 bands, ssimulacra2_thresholds above; the words: libjxl's tools/ssimulacra2 README"}},
    {"butteraugli",
     {N_("JPEG XL's visible-difference model"),
      N_("Butteraugli, from the JPEG XL project: how visible the differences are, by a model of the eye. The app shows its 3-norm, a mean of the frame's differences that weighs the larger ones more, not their maximum. 0 is identical, lower is better."),
      {}}},
    {"cvvdp",
     {N_("visible difference on your display"),
      N_("ColorVideoVDP: how visible the difference is to a viewer of the display chosen (CVVDP display), over the whole video. One score per video, in JOD (just-objectionable differences), higher is better."),
      N_("Its authors' scale: 10 means no visible difference, and one JOD lower means 75% of viewers would pick the better video."),
      "ColorVideoVDP's JOD scale (Mantiuk et al. 2024), as the app's CVVDP score tooltip gives it"}},
}};

}  // namespace

MetricSection section_of(std::string_view key) {
    const std::string_view metric = scored_by(key);
    const bool specialized = std::find(metrics_hidden_at_first.begin(), metrics_hidden_at_first.end(), metric) !=
                             metrics_hidden_at_first.end();
    return specialized ? MetricSection::specialized : MetricSection::general_purpose;
}

std::vector<std::string_view> metrics_in(MetricSection section) {
    std::vector<std::string_view> keys;
    for (const MetricDefinition& metric : registry) {
        if (!metric.is_plane() && section_of(metric.key) == section) {
            keys.push_back(metric.key);
        }
    }
    return keys;
}

MetricGuide metric_guide(std::string_view key) {
    const std::string_view metric = scored_by(key);
    for (const KeyedGuide& item : guides) {
        if (item.key == metric) {
            return item.guide;
        }
    }
    return {};
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
