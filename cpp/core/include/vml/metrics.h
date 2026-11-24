// The metrics VideoMetricsLab supports: each one's key, labels, number
// format, direction, way of averaging and score bands. A port of
// vmaf_app/core/metrics.py; the registry's order is the order the app shows
// the metrics in.
#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vml {

// Whether a metric scores each frame or the whole video.
enum class MetricKind { frame, sequence };

// Whether a higher or a lower score is better.
enum class MetricDirection { higher_is_better, lower_is_better };

// How a metric's per-frame scores become one number for the video
// (aggregate_scores in stats.h).
// The two in decibels average the frames' errors, so an identical frame
// (+infinity) adds no error: XPSNR's -20 log10(mean(10^(-dB / 20))), as
// FFmpeg's xpsnr filter averages, and PSNR-HVS-M's -10 log10(mean(10^(-dB / 10))),
// as Daala's tool averages its error.
enum class MetricAggregation { arithmetic, square_mean_root_db, mean_error_db };

// A score band, such as "> 95": the frames scoring above (comparison '>')
// or below ('<') the value.
struct Threshold {
    char comparison;
    double value;
};

// A score's place among the test videos' (MetricDefinition::best_and_worst).
enum class ScoreMark { none, best, worst };

// How an FFmpeg metric is switched on in a run's options: by an on/off
// option, or by a libvmaf feature. One of the two is set.
struct FfmpegMetricBinding {
    std::string_view bool_option;
    std::string_view libvmaf_feature;
};

struct MetricDefinition {
    std::string_view key;           // as saved in results, settings and exports
    std::string_view label;
    std::string_view short_label;
    std::string_view table_header;
    std::string_view axis_label;
    int decimals;                   // the precision scores are shown at
    std::string_view value_suffix;  // such as " dB"
    MetricKind kind;
    MetricDirection direction;
    MetricAggregation aggregation;
    std::optional<double> fixed_y_max;  // the graph's fixed top, if it has one
    std::span<const Threshold> thresholds;
    std::optional<FfmpegMetricBinding> ffmpeg_binding;
    // The backend that calculates a metric that is not FFmpeg's.
    std::string_view backend_id;
    // The smallest width and height the metric gives a real score at (0:
    // any); smaller pictures are refused (too_small_problem). SSIM's 11x11
    // window, SSIMULACRA2's and Butteraugli's 8x8 (libjxl's own tool refuses
    // less; its library and Vship give a perfect score there).
    int smallest_side;
    // One plane of a metric of the three planes (PSNR YUV, SSIM YUV, MS-SSIM
    // YUV, PSNR-HVS-M): that metric's key; empty for every other metric. A
    // plane is scored in its metric's pass, and is shown, saved and exported
    // as a metric of its own.
    std::string_view plane_of = {};

    bool is_plane() const { return !plane_of.empty(); }

    // A score as shown: at the metric's precision, infinity as a sign, and
    // no value (or NaN) as a dash.
    std::string format_value(std::optional<double> value) const;
    // A difference between two scores as shown: with its sign, at the
    // metric's precision.
    std::string format_delta(std::optional<double> value) const;

    // Each of `scores` (one per video; none where a video has none) marked
    // best, worst or neither, by the metric's direction. Scores are compared
    // as shown (format_value), so scores that show the same number are
    // equally best or worst. With fewer than two to compare, or all showing
    // the same, there is neither.
    std::vector<ScoreMark> best_and_worst(const std::vector<std::optional<double>>& scores) const;

    // Whether a frame's score can be +infinity (an identical frame): the
    // metrics averaged in decibels of their error.
    bool can_be_infinite() const { return aggregation != MetricAggregation::arithmetic; }
};

// Every metric, in the order the app shows them.
std::span<const MetricDefinition> metrics();

// The metric with this key, or nullptr if there is none.
const MetricDefinition* find_metric(std::string_view key);

// The metric with this key. Throws std::out_of_range for an unknown key.
const MetricDefinition& metric_definition(std::string_view key);

// The metric whose pass scores `key`: a plane's metric (plane_of), or the
// metric itself. Throws std::out_of_range for an unknown key.
std::string_view scored_by(std::string_view key);

// Why pictures compared at `width` x `height` cannot be scored with the
// metrics of `keys`, naming those they are too small for and the size each
// needs (MetricDefinition::smallest_side); nothing when all can be.
std::optional<std::string> too_small_problem(const std::vector<std::string>& keys, int width, int height);

// The metrics' labels with `separator` between them: "VMAF v0.6.1, PSNR".
std::string metric_labels(const std::vector<std::string>& keys, std::string_view separator = ", ");

// The metrics most users do not need at first, in the registry's order: HDR
// VMAF (HDR10 videos only), and the weightings, color and banding metrics
// that codec comparisons such as JVET's, MSU's and Xiph's report. The one
// list behind both their place and their first state: Add/remove metrics
// and Settings > Metrics list them as Specialized, the rest as General
// purpose; new settings start with them ticked but hidden
// (Settings::hidden_metrics), so showing one is enough to calculate it.
inline constexpr std::array<std::string_view, 8> metrics_hidden_at_first{
    "vmaf_hdr", "psnr_yuv", "ssim_yuv", "ms_ssim", "ms_ssim_yuv", "psnr_hvs", "cambi", "ciede2000"};

// The two sections Add/remove metrics and Settings > Metrics list the
// metrics in.
enum class MetricSection { general_purpose, specialized };

// A metric's section: Specialized for metrics_hidden_at_first, else General
// purpose. A plane is in its metric's.
MetricSection section_of(std::string_view key);

// The metrics a section lists, in the registry's order: every metric but the
// planes, which come with their metric (Settings::show_yuv_planes).
std::vector<std::string_view> metrics_in(MetricSection section);

// What the window says of a metric, in English (N_(): the window translates
// it): a few words beside its name in Add/remove metrics and Settings >
// Metrics; then, for its tooltip, what it measures, its unit and which way
// is better; and its broad ranges only where they have a source: the app's
// own bands (its thresholds; the YUV metrics' are PSNR's and SSIM's) or the
// metric's authors' published scale, else empty. `ranges_source` names that
// source (not shown; for review and Metrics.EveryMetricFollowsTheRulesFor-
// AddingOne), empty with no ranges. A plane has its metric's.
// docs/ARCHITECTURE.md's "Adding a metric" says what a new metric needs.
struct MetricGuide {
    std::string_view brief;
    std::string_view measures;
    std::string_view ranges;
    std::string_view ranges_source = {};
};
MetricGuide metric_guide(std::string_view key);

}  // namespace vml
