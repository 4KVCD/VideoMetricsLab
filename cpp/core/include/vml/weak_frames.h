// Where a video's scores drop (issue #24): its worst frames, one per dip,
// and the stretches where scores fall past a threshold ("weak segments"),
// with the JSON and CSV texts they are exported as.
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
    // For each of the rules' conditions, in their order: the frames of the
    // segment where it holds (what made the segment weak, with "any").
    std::vector<long long> condition_frames;

    int frames() const { return last_frame - first_frame + 1; }
};

// A condition's frames: where it holds (inside the rules' ranges), and how
// many of those are in the segments found. They differ where frames were
// left out: a run shorter than the shortest segment, or, with every
// condition required, a frame where another did not hold.
struct ConditionCount {
    long long holds = 0;
    long long in_segments = 0;
};

// The rules Settings keeps (weak_segment_*): each condition as its text
// ("vmaf<90"), leaving out one unreadable or of a metric the app does not
// have; a shortest segment below 1 frame taken as 1, a gap below 0 as 0.
SegmentRules saved_segment_rules(const std::vector<std::string>& conditions, bool all, int min_frames,
                                 int merge_frames);
// The conditions' texts, for Settings.
std::vector<std::string> condition_texts(const std::vector<WeakCondition>& conditions);

// The stretches of `results` where the rules' conditions hold, in frame
// order. A frame sample is weak on its own frame, and the frames between
// two weak samples of a subsample are weak too; a per-second sample (CVVDP)
// covers its second, to the frame before the next sample's (the last one:
// to `last_frame`, the run's last compared frame). A condition on a metric
// `results` has no scores for is left out (scored_rules), whether any or
// every condition must hold: the dialog, the exports and the command line
// apply the rest. `fps` gives the times.
std::vector<WeakSegment> weak_segments(const MetricResultSet& results, const SegmentRules& rules, double fps,
                                       int last_frame);
// The same over the whole of `result`: its scores, frame rate and last
// compared frame.
std::vector<WeakSegment> weak_segments(const ComparisonResult& result, const SegmentRules& rules);

// The rules as they apply to `results`: without the conditions on a metric
// it has no scores for, which weak_segments leaves out. `left_out`, where
// given, gets those.
SegmentRules scored_rules(const MetricResultSet& results, const SegmentRules& rules,
                          std::vector<WeakCondition>* left_out = nullptr);
// For each of the rules' conditions, in their order, its frames
// (ConditionCount), `segments` being weak_segments' for the same scores and
// rules. A condition on a metric `results` does not have holds nowhere.
std::vector<ConditionCount> condition_counts(const MetricResultSet& results, const SegmentRules& rules,
                                             int last_frame, const std::vector<WeakSegment>& segments);
std::vector<ConditionCount> condition_counts(const ComparisonResult& result, const SegmentRules& rules,
                                             const std::vector<WeakSegment>& segments);

// The segments as JSON: the rules used, then one object per segment with
// first_frame, last_frame, start_time, end_time, frames and each metric's
// min, max and mean. Conditions `left_out` (scored_rules') are given after
// the rules' as "left_out", where there are any.
nlohmann::ordered_json segments_json(const std::vector<WeakSegment>& segments, const SegmentRules& rules,
                                     const std::vector<WeakCondition>& left_out = {});
// segments_json as a file of its own (<video>.<metrics>.segments.json):
// led by document_head's "app", "version", "commit" and "format" (segments-1).
nlohmann::ordered_json segments_file_json(const std::vector<WeakSegment>& segments, const SegmentRules& rules,
                                          const std::vector<WeakCondition>& left_out = {});

// What an export of segments by `rules` is named after, before ".json" or
// ".csv": its conditions' metrics, each once, in the conditions' order,
// joined by "+", then ".segments" ("vmaf+ssimulacra2.segments"). So exports
// of one video by different metrics have names of their own.
std::string segments_file_suffix(const SegmentRules& rules);

// The segments as CSV, a row per segment: first_frame, last_frame,
// start_time, end_time, frames, then <metric>_min, _max and _mean.
std::string segments_csv(const std::vector<WeakSegment>& segments, const SegmentRules& rules);

// Worst frames `worst` (worst_frames' list, for metric `key`) as JSON, led
// by document_head's keys (worst-frames-1): the metric, the number of dips
// asked for, the frame ranges looked in (empty:
// all), then "worst_frames", one object per frame, worst first: rank (1 is
// the worst), frame, time and "scores", every scored metric's score at that
// frame, in the registry's order (null where it has none).
nlohmann::ordered_json worst_frames_json(const MetricResultSet& results, const std::string& key,
                                         const std::vector<WorstFrame>& worst, std::size_t dips,
                                         const std::vector<FrameInterval>& ranges);

// The same as CSV, a row per frame: rank, frame, time, then each scored
// metric's score (empty where it has none).
std::string worst_frames_csv(const MetricResultSet& results, const std::vector<WorstFrame>& worst);

// Every metric `results` has scores of, its worst frames of `dips` dips
// (worst_frames, over the whole video), as one document: "dips", then
// "metrics", one object a metric in the registry's order: "metric" and
// "worst_frames" (worst_frames_json's). For the archive Settings' Exports
// include makes (Worst frames).
nlohmann::ordered_json worst_frames_report_json(const MetricResultSet& results, std::size_t dips);

// The same as CSV: worst_frames_csv's rows, each led by its metric
// ("metric,rank,frame,time,<every scored metric>"), a metric after another.
std::string worst_frames_report_csv(const MetricResultSet& results, std::size_t dips);

// What an export of worst frames of `key` is named after, before ".json" or
// ".csv": "<key>.worst" ("vmaf.worst").
std::string worst_frames_file_suffix(const std::string& key);

// A video in a weak segments report: its name and file, its reference's,
// and its result.
struct SegmentsReportVideo {
    std::string name;
    std::string path;
    std::string reference;
    const ComparisonResult* result = nullptr;
};

// Every video's segments by `rules`, as one JSON report, led by
// document_head's keys (segments-report-1): the rules (as
// segments_json gives them, without ranges), then "videos", each with its
// name, path, reference, segment_count, weak_frames (the segments' frames
// added up) and segments (segments_json's); a video without scores for a
// condition's metric has it in "left_out" (scored_rules'), before segments.
nlohmann::ordered_json segments_report_json(const std::vector<SegmentsReportVideo>& videos,
                                            const SegmentRules& rules);

// The same as CSV: segments_csv's rows, each led by its video's name and
// reference ("video,reference,first_frame,..."), the videos in order.
std::string segments_report_csv(const std::vector<SegmentsReportVideo>& videos, const SegmentRules& rules);

}  // namespace vml
