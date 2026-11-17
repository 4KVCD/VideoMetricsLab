// Where a video's scores drop (issue #24): its worst frames, one per dip,
// and the stretches where scores fall past a threshold ("weak segments"),
// with the JSON, CSV and x265 --zones texts they are exported as.
//
// Frame numbers are the run's: counted from each video's first frame, also
// for a run over a range, so they are the encode's own.
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/metrics.h"
#include "vml/results.h"

namespace vml {

// Frames first to last, both included.
struct FrameInterval {
    int first = 0;
    int last = 0;
};

// One metric's scored samples: a score per frame (every frame, or every
// Nth on a subsample), or a score per second (CVVDP's timeline).
struct ScoreSamples {
    std::span<const int> frame;
    std::span<const double> time;
    std::span<const float> values;
    bool per_second = false;
};

// `key`'s samples in `results`: a frame metric's, or a whole-video metric's
// timeline; nothing when it has neither.
std::optional<ScoreSamples> score_samples(const MetricResultSet& results, const std::string& key);

// `samples`' score at `frame`: the sample on it, or for a per-second metric
// the second holding it; nothing where there is none.
std::optional<double> score_at(const ScoreSamples& samples, int frame);

// The last frame `result` compared, numbered as its scores are.
int last_compared_frame(const ComparisonResult& result);

// The worst frame of a dip.
struct WorstFrame {
    int frame = 0;
    double time = 0.0;
    double score = 0.0;
};

// The worst frames of `count` dips, worst first, by the metric's direction
// (Butteraugli's worst are its highest). A dip is grouped by time: a sample
// within max(1 s, the samples' step) of a dip already found joins it rather
// than starting another, so one flash of 12 frames is one entry. Only
// samples inside `ranges` count (empty: all); unscored (NaN) ones never do.
std::vector<WorstFrame> worst_frames(const ScoreSamples& samples, MetricDirection direction, std::size_t count,
                                     std::span<const FrameInterval> ranges = {});

// "vmaf<90": a metric is weak where its score is below (or, with '>',
// above) the threshold.
struct WeakCondition {
    std::string key;
    char comparison = '<';
    double threshold = 0.0;

    bool is_weak(double score) const;
    std::string text() const;  // "vmaf<90"
};

// "vmaf<90" or "butteraugli>2.5" read, or nothing when it is not of that
// shape. The key is not checked against the registry.
std::optional<WeakCondition> parse_weak_condition(std::string_view text);

// A condition to start from for `metric`: below its first band of worse
// frames ("< 85" for VMAF, as its statistics count them); for the two with
// no bands, Butteraugli above 2 and CVVDP below 9 JOD.
WeakCondition default_weak_condition(const MetricDefinition& metric);

// Every member after `conditions` has a default, so SegmentRules{{...}}
// leaves none uninitialized (GCC's and Clang's -Wextra warn otherwise).
struct SegmentRules {
    std::vector<WeakCondition> conditions;
    bool all = false;      // a frame is weak where every condition holds there, not any of them
    int min_frames = 1;    // shorter segments are left out (after merging)
    int merge_frames = 0;  // segments with at most this many frames between them become one
    std::vector<FrameInterval> ranges{};  // only frames inside them; empty: all
};

// A condition's metric within a segment: its scores' lowest, highest and
// mean (by the metric's own way of averaging, as its whole-video score).
struct SegmentMetric {
    std::string key;
    std::optional<double> min;
    std::optional<double> max;
    std::optional<double> mean;
};

struct WeakSegment {
    int first_frame = 0;
    int last_frame = 0;
    double start_time = 0.0;  // where its first frame starts
    double end_time = 0.0;    // where its last frame ends
    std::vector<SegmentMetric> metrics;  // in the order of the conditions

    int frames() const { return last_frame - first_frame + 1; }
};

// The stretches of `results` where the rules' conditions hold, in frame
// order. A frame sample is weak on its own frame, and the frames between
// two weak samples of a subsample are weak too; a per-second sample (CVVDP)
// covers its second, to the frame before the next sample's (the last one:
// to `last_frame`, the run's last compared frame). A metric `results` does
// not have is weak nowhere. `fps` gives the times.
std::vector<WeakSegment> weak_segments(const MetricResultSet& results, const SegmentRules& rules, double fps,
                                       int last_frame);
// The same over the whole of `result`: its scores, frame rate and last
// compared frame.
std::vector<WeakSegment> weak_segments(const ComparisonResult& result, const SegmentRules& rules);

// The segments as JSON: the rules used, then one object per segment with
// first_frame, last_frame, start_time, end_time, frames and each metric's
// min, max and mean.
nlohmann::ordered_json segments_json(const std::vector<WeakSegment>& segments, const SegmentRules& rules);

// The segments as CSV, a row per segment: first_frame, last_frame,
// start_time, end_time, frames, then <metric>_min, _max and _mean.
std::string segments_csv(const std::vector<WeakSegment>& segments, const SegmentRules& rules);

// x265's --zones value for the segments: "first,last,option" per segment
// ("120,240,b=1.5"), joined by '/'; empty when there are none.
std::string x265_zones(const std::vector<WeakSegment>& segments, const std::string& option);

// Whether `option` is a setting x265's --zones takes for a zone: b=<factor>
// (its bitrate times a positive factor) or q=<QP> (a fixed QP, 0 to 69).
bool valid_zones_option(std::string_view option);

}  // namespace vml
