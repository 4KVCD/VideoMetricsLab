#include "vml/results.h"

#include <algorithm>

#include "vml/metrics.h"
#include "vml/stats.h"

namespace vml {

namespace {

// The identity saved XPSNR scores are reused by: v2 since October 2026,
// weighted by the source's activity.
constexpr const char* xpsnr_compatibility_id = "ffmpeg-xpsnr-v2";

void remember_key(std::vector<std::string>& keys, const std::string& key) {
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
        keys.push_back(key);
    }
}

}  // namespace

MetricProvenance current_ffmpeg_provenance(const std::string& key, const std::string& version,
                                           std::map<std::string, std::string> parameters) {
    const bool xpsnr = key == "xpsnr";
    return MetricProvenance{xpsnr ? "ffmpeg/xpsnr" : "ffmpeg/libvmaf", "ffmpeg " + version, "cpu",
                            xpsnr ? xpsnr_compatibility_id : "ffmpeg-libvmaf-v1", std::move(parameters)};
}

std::optional<double> FrameMetricResult::aggregate() const {
    const MetricDefinition* definition = find_metric(key);
    const MetricAggregation aggregation = definition != nullptr ? definition->aggregation : MetricAggregation::arithmetic;
    const std::vector<double> scores(values.begin(), values.end());
    return aggregate_scores(scores, aggregation);
}

void MetricResultSet::add(FrameMetricResult result) {
    remember_key(keys_, result.key);
    sequences_.erase(result.key);
    const std::string key = result.key;
    frames_.insert_or_assign(key, std::move(result));
}

void MetricResultSet::add(SequenceMetricResult result) {
    remember_key(keys_, result.key);
    frames_.erase(result.key);
    const std::string key = result.key;
    sequences_.insert_or_assign(key, std::move(result));
}

bool MetricResultSet::has(const std::string& key) const { return frames_.contains(key) || sequences_.contains(key); }

const FrameMetricResult* MetricResultSet::frame(const std::string& key) const {
    const auto found = frames_.find(key);
    return found == frames_.end() ? nullptr : &found->second;
}

const SequenceMetricResult* MetricResultSet::sequence(const std::string& key) const {
    const auto found = sequences_.find(key);
    return found == sequences_.end() ? nullptr : &found->second;
}

FrameScores frame_scores_from_results(const MetricResultSet& results) {
    // Frames are matched by frame number alone, which identifies the frame;
    // the view takes the first metric's times (others' can differ by
    // rounding for the same frames).
    FrameScores scores;
    const FrameMetricResult* first = nullptr;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.kind != MetricKind::frame) {
            continue;
        }
        const FrameMetricResult* result = results.frame(std::string(metric.key));
        if (result == nullptr) {
            continue;
        }
        if (first == nullptr) {
            first = result;
            scores.frame = result->frame;
            scores.time = result->time;
        }
        if (result->frame == first->frame) {
            scores.metrics[result->key] = result->values;
        }
    }
    return scores;
}

}  // namespace vml
