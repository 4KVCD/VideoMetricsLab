// A comparison's result: every metric's scores, each with its provenance
// (what calculated it). A port of vmaf_app/core/metric_results.py and the
// result parts of models.py.
//
// Scores are kept as 32-bit floats, as the Python app keeps them (and as
// libvmaf and FFmpeg give them), so that they are written out digit for
// digit as the Python app writes them.
#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vml/options.h"
#include "vml/video.h"

namespace vml {

// How a metric's value was made, separate from whether a saved score can
// stand for a new one (implementation_compatibility_id says that).
struct MetricProvenance {
    std::string implementation;            // "ffmpeg/libvmaf"
    std::string implementation_version;    // "ffmpeg 9.0.1"
    std::string compute_backend;           // "cpu" or "gpu"
    std::string implementation_compatibility_id;
    // Text, as JSON, of the parameters the value depends on ({"model": ...}).
    std::map<std::string, std::string> parameters;
};

// A provenance of a metric FFmpeg calculated: its libvmaf filter, or its
// xpsnr filter for XPSNR. `version` is FFmpeg's ("9.0.1").
MetricProvenance current_ffmpeg_provenance(const std::string& key, const std::string& version,
                                           std::map<std::string, std::string> parameters = {});

// A metric with a score per frame.
struct FrameMetricResult {
    std::string key;
    std::vector<int> frame;
    std::vector<double> time;
    std::vector<float> values;
    MetricProvenance provenance;

    // The one number for the whole video, by the metric's own way of
    // averaging; nothing when no frame has a score.
    std::optional<double> aggregate() const;
};

// One score for the whole video (CVVDP), with a timeline where it has one.
struct SequenceMetricResult {
    std::string key;
    double score = 0.0;
    MetricProvenance provenance;
    std::vector<int> frame;
    std::vector<double> time;
    std::vector<float> values;

    bool has_timeline() const { return !values.empty(); }
};

// A comparison's metrics by key, in the order they were added.
class MetricResultSet {
public:
    // Adds a metric's result, or replaces the one with its key.
    void add(FrameMetricResult result);
    void add(SequenceMetricResult result);

    bool has(const std::string& key) const;
    const FrameMetricResult* frame(const std::string& key) const;
    const SequenceMetricResult* sequence(const std::string& key) const;
    const std::vector<std::string>& keys() const { return keys_; }
    bool empty() const { return keys_.empty(); }

private:
    std::vector<std::string> keys_;
    std::map<std::string, FrameMetricResult> frames_;
    std::map<std::string, SequenceMetricResult> sequences_;
};

// The frame metrics that share one frame axis: the first one present, in
// the registry's order, sets the frames, and the others with the same frame
// numbers come with it (the Python app's FrameScores view).
struct FrameScores {
    std::vector<int> frame;
    std::vector<double> time;
    std::map<std::string, std::vector<float>> metrics;
};
FrameScores frame_scores_from_results(const MetricResultSet& results);

// One comparison's result: the two videos, how they were compared, and
// every metric's scores.
struct ComparisonResult {
    std::filesystem::path source;
    std::filesystem::path distorted;
    double fps = 0.0;
    std::string model;  // VMAF v0.6.1's model= value
    std::optional<CropBox> source_crop;
    std::optional<CropBox> distorted_crop;
    VideoInfo source_info;
    VideoInfo distorted_info;
    ScaleDirection scale_direction = ScaleDirection::source_to_distorted;
    std::string scale_algorithm = "bicubic";
    long long compared_frame_count = 0;
    std::optional<std::string> model_choice;
    std::string model_v1;
    std::optional<std::string> model_choice_v1;
    MetricResultSet metric_results;

    bool has_metric(const std::string& key) const { return metric_results.has(key); }
};

}  // namespace vml
