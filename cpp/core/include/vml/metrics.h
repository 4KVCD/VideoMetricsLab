// The metrics VideoMetricsLab supports: each one's key, labels, number
// format, direction, way of averaging and score bands. A port of
// vmaf_app/core/metrics.py; the registry's order is the order the app shows
// the metrics in.
#pragma once

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
enum class MetricAggregation { arithmetic, square_mean_root_db };

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
};

// Every metric, in the order the app shows them.
std::span<const MetricDefinition> metrics();

// The metric with this key, or nullptr if there is none.
const MetricDefinition* find_metric(std::string_view key);

// The metric with this key. Throws std::out_of_range for an unknown key.
const MetricDefinition& metric_definition(std::string_view key);

// The metrics' labels with `separator` between them: "VMAF v0.6.1, PSNR".
std::string metric_labels(const std::vector<std::string>& keys, std::string_view separator = ", ");

}  // namespace vml
