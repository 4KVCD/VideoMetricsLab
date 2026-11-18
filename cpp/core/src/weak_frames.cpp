#include "vml/weak_frames.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iterator>
#include <map>

#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/stats.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

bool inside(int frame, std::span<const FrameInterval> ranges) {
    return ranges.empty() || std::any_of(ranges.begin(), ranges.end(), [frame](const FrameInterval& range) {
               return frame >= range.first && frame <= range.last;
           });
}

// The conditions' metrics, each once, in the conditions' order.
std::vector<std::string> condition_keys(const SegmentRules& rules) {
    std::vector<std::string> keys;
    for (const WeakCondition& condition : rules.conditions) {
        if (std::find(keys.begin(), keys.end(), condition.key) == keys.end()) {
            keys.push_back(condition.key);
        }
    }
    return keys;
}

// The frames `begin` to `end` where `condition` holds for `samples` (1),
// as weak_segments describes.
std::vector<char> weak_mask(const ScoreSamples& samples, const WeakCondition& condition, int begin, int end) {
    std::vector<char> mask(static_cast<std::size_t>(end - begin + 1), 0);
    const auto mark = [&mask, begin, end](int from, int to) {
        for (int frame = std::max(from, begin); frame <= std::min(to, end); ++frame) {
            mask[static_cast<std::size_t>(frame - begin)] = 1;
        }
    };
    const std::size_t count = samples.values.size();
    for (std::size_t i = 0; i < count; ++i) {
        if (!condition.is_weak(samples.values[i])) {
            continue;
        }
        const bool next_exists = i + 1 < count;
        if (samples.per_second) {
            mark(samples.frame[i], next_exists ? samples.frame[i + 1] - 1 : end);
        } else if (next_exists && condition.is_weak(samples.values[i + 1])) {
            mark(samples.frame[i], samples.frame[i + 1] - 1);  // and the unscored frames up to the next
        } else {
            mark(samples.frame[i], samples.frame[i]);
        }
    }
    return mask;
}

// `key`'s scores within `segment`: the samples on its frames, and a
// per-second sample whose second overlaps it. The samples are in frame
// order, so only those from the segment's start are read (a run with
// thousands of segments stays linear).
SegmentMetric segment_metric(const ScoreSamples& samples, const std::string& key, const WeakSegment& segment) {
    auto from = samples.per_second
                    ? std::upper_bound(samples.frame.begin(), samples.frame.end(), segment.first_frame)
                    : std::lower_bound(samples.frame.begin(), samples.frame.end(), segment.first_frame);
    if (samples.per_second && from != samples.frame.begin()) {
        --from;  // the second the segment starts in
    }
    std::vector<double> scores;
    for (auto i = static_cast<std::size_t>(from - samples.frame.begin());
         i < samples.values.size() && samples.frame[i] <= segment.last_frame; ++i) {
        if (!std::isnan(samples.values[i])) {
            scores.push_back(samples.values[i]);
        }
    }
    SegmentMetric metric{key, std::nullopt, std::nullopt, std::nullopt};
    if (scores.empty()) {
        return metric;
    }
    const auto [lowest, highest] = std::minmax_element(scores.begin(), scores.end());
    metric.min = *lowest;
    metric.max = *highest;
    const MetricDefinition* definition = find_metric(key);
    const MetricAggregation aggregation =
        definition != nullptr ? definition->aggregation : MetricAggregation::arithmetic;
    metric.mean = aggregate_scores(scores, aggregation);
    return metric;
}

std::string csv_number(const std::optional<double>& value) { return value ? python_repr(*value) : std::string(); }

}  // namespace

std::optional<ScoreSamples> score_samples(const MetricResultSet& results, const std::string& key) {
    if (const FrameMetricResult* metric = results.frame(key)) {
        return ScoreSamples{metric->frame, metric->time, metric->values, false};
    }
    if (const SequenceMetricResult* metric = results.sequence(key); metric != nullptr && metric->has_timeline()) {
        return ScoreSamples{metric->frame, metric->time, metric->values, true};
    }
    return std::nullopt;
}

std::optional<double> score_at(const ScoreSamples& samples, int frame) {
    const auto after = std::upper_bound(samples.frame.begin(), samples.frame.end(), frame);
    if (after == samples.frame.begin()) {
        return std::nullopt;
    }
    const auto i = static_cast<std::size_t>(after - samples.frame.begin() - 1);
    if ((!samples.per_second && samples.frame[i] != frame) || std::isnan(samples.values[i])) {
        return std::nullopt;
    }
    return samples.values[i];
}

int last_compared_frame(const ComparisonResult& result) {
    int first = INT_MAX;
    int last = -1;
    for (const std::string& key : result.metric_results.keys()) {
        if (const std::optional<ScoreSamples> samples = score_samples(result.metric_results, key);
            samples && !samples->frame.empty()) {
            first = std::min(first, samples->frame.front());
            last = std::max(last, samples->frame.back());
        }
    }
    if (first == INT_MAX || result.compared_frame_count <= 0) {
        return last;
    }
    return std::max(last, static_cast<int>(first + result.compared_frame_count - 1));
}

std::vector<WorstFrame> worst_frames(const ScoreSamples& samples, MetricDirection direction, std::size_t count,
                                     std::span<const FrameInterval> ranges) {
    const std::size_t size = samples.values.size();
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isnan(samples.values[i]) && inside(samples.frame[i], ranges)) {
            order.push_back(i);
        }
    }
    if (order.empty() || count == 0) {
        return {};
    }
    // Worst first; equal scores in frame order (the sort is stable).
    const bool high_is_worse = direction == MetricDirection::lower_is_better;
    std::stable_sort(order.begin(), order.end(), [&samples, high_is_worse](std::size_t a, std::size_t b) {
        return high_is_worse ? samples.values[a] > samples.values[b] : samples.values[a] < samples.values[b];
    });
    // A dip's reach: 1 s, or the samples' step where it is longer. The
    // small margin keeps samples exactly one step apart within reach.
    const double step = size > 1 ? (samples.time[size - 1] - samples.time[0]) / static_cast<double>(size - 1) : 0.0;
    const double reach = std::max(1.0, step) + 1e-6;

    // The times the dips found so far cover, as disjoint start -> end
    // spans. A sample within reach of a span joins it (and merges every
    // span it reaches); one out of reach of all is the worst of a new dip.
    std::map<double, double> covered;
    std::vector<WorstFrame> found;
    for (const std::size_t i : order) {
        const double time = samples.time[i];
        double start = time;
        double end = time;
        bool joined = false;
        const auto after = covered.upper_bound(time + reach);
        while (after != covered.begin()) {
            const auto span = std::prev(after);
            if (span->second < time - reach) {
                break;  // the spans before it end earlier still
            }
            start = std::min(start, span->first);
            end = std::max(end, span->second);
            covered.erase(span);
            joined = true;
        }
        if (!joined) {
            if (found.size() == count) {
                break;
            }
            found.push_back({samples.frame[i], time, samples.values[i]});
        }
        covered.emplace(start, end);
    }
    return found;
}

bool WeakCondition::is_weak(double score) const { return comparison == '>' ? score > threshold : score < threshold; }

std::string WeakCondition::text() const { return key + comparison + format_general(threshold, 12); }

std::optional<WeakCondition> parse_weak_condition(std::string_view text) {
    const std::size_t at = text.find_first_of("<>");
    if (at == std::string_view::npos || at == 0 || at + 1 >= text.size()) {
        return std::nullopt;
    }
    // python_float, not std::from_chars: macOS's library has from_chars for a double only from macOS 26.
    const std::optional<double> threshold = python_float(text.substr(at + 1));
    if (!threshold || !std::isfinite(*threshold)) {
        return std::nullopt;
    }
    return WeakCondition{std::string(text.substr(0, at)), text[at], *threshold};
}

WeakCondition default_weak_condition(const MetricDefinition& metric) {
    const std::string key(metric.key);
    for (const Threshold& band : metric.thresholds) {
        if (band.comparison == '<') {
            return {key, '<', band.value};
        }
    }
    // Butteraugli is the one metric whose worst are its highest.
    return metric.direction == MetricDirection::lower_is_better ? WeakCondition{key, '>', 2.0}
                                                                : WeakCondition{key, '<', 9.0};
}

std::vector<WeakSegment> weak_segments(const MetricResultSet& results, const SegmentRules& rules, double fps,
                                       int last_frame) {
    // The frame axis: from the first scored frame to the run's last.
    std::vector<std::pair<const WeakCondition*, ScoreSamples>> scored;
    int begin = last_frame;
    for (const WeakCondition& condition : rules.conditions) {
        if (const std::optional<ScoreSamples> samples = score_samples(results, condition.key);
            samples && !samples->frame.empty()) {
            scored.emplace_back(&condition, *samples);
            begin = std::min(begin, samples->frame.front());
        } else if (rules.all) {
            return {};  // a metric with no scores is weak nowhere
        }
    }
    if (scored.empty() || begin > last_frame) {
        return {};
    }
    std::vector<char> weak;
    for (const auto& [condition, samples] : scored) {
        const std::vector<char> mask = weak_mask(samples, *condition, begin, last_frame);
        if (weak.empty()) {
            weak = mask;
            continue;
        }
        for (std::size_t f = 0; f < weak.size(); ++f) {
            weak[f] = rules.all ? (weak[f] & mask[f]) : (weak[f] | mask[f]);
        }
    }

    // The runs of weak frames inside the ranges, merged across short gaps.
    std::vector<WeakSegment> segments;
    for (int frame = begin; frame <= last_frame; ++frame) {
        if (!weak[static_cast<std::size_t>(frame - begin)] || !inside(frame, rules.ranges)) {
            continue;
        }
        if (!segments.empty() && frame - segments.back().last_frame - 1 <= rules.merge_frames) {
            segments.back().last_frame = frame;
        } else {
            segments.push_back(WeakSegment{frame, frame, 0.0, 0.0, {}});
        }
    }
    std::erase_if(segments, [&rules](const WeakSegment& segment) { return segment.frames() < rules.min_frames; });

    const std::vector<std::string> keys = condition_keys(rules);
    for (WeakSegment& segment : segments) {
        segment.start_time = fps > 0.0 ? segment.first_frame / fps : 0.0;
        segment.end_time = fps > 0.0 ? (segment.last_frame + 1) / fps : 0.0;
        for (const std::string& key : keys) {
            const std::optional<ScoreSamples> samples = score_samples(results, key);
            segment.metrics.push_back(samples ? segment_metric(*samples, key, segment)
                                              : SegmentMetric{key, std::nullopt, std::nullopt, std::nullopt});
        }
    }
    return segments;
}

std::vector<WeakSegment> weak_segments(const ComparisonResult& result, const SegmentRules& rules) {
    return weak_segments(result.metric_results, rules, result.fps, last_compared_frame(result));
}

Json segments_json(const std::vector<WeakSegment>& segments, const SegmentRules& rules) {
    Json conditions = Json::array();
    for (const WeakCondition& condition : rules.conditions) {
        conditions.push_back(condition.text());
    }
    Json ranges = Json::array();
    for (const FrameInterval& range : rules.ranges) {
        ranges.push_back(Json{{"first_frame", range.first}, {"last_frame", range.last}});
    }
    Json list = Json::array();
    for (const WeakSegment& segment : segments) {
        Json metrics = Json::object();
        for (const SegmentMetric& metric : segment.metrics) {
            metrics[metric.key] = Json{{"min", json_number(metric.min)},
                                       {"max", json_number(metric.max)},
                                       {"mean", json_number(metric.mean)}};
        }
        list.push_back(Json{{"first_frame", segment.first_frame},
                            {"last_frame", segment.last_frame},
                            {"start_time", segment.start_time},
                            {"end_time", segment.end_time},
                            {"frames", segment.frames()},
                            {"metrics", std::move(metrics)}});
    }
    return Json{{"conditions", std::move(conditions)},
                {"combine", rules.all ? "all" : "any"},
                {"min_frames", rules.min_frames},
                {"merge_frames", rules.merge_frames},
                {"ranges", std::move(ranges)},
                {"segments", std::move(list)}};
}

std::string segments_file_suffix(const SegmentRules& rules) {
    std::string name;
    for (const std::string& key : condition_keys(rules)) {
        name += (name.empty() ? "" : "+") + key;
    }
    return name + ".segments";
}

std::string segments_csv(const std::vector<WeakSegment>& segments, const SegmentRules& rules) {
    std::string text = "first_frame,last_frame,start_time,end_time,frames";
    for (const std::string& key : condition_keys(rules)) {
        text += "," + key + "_min," + key + "_max," + key + "_mean";
    }
    text += "\n";
    for (const WeakSegment& segment : segments) {
        text += std::to_string(segment.first_frame) + "," + std::to_string(segment.last_frame) + "," +
                python_repr(segment.start_time) + "," + python_repr(segment.end_time) + "," +
                std::to_string(segment.frames());
        for (const SegmentMetric& metric : segment.metrics) {
            text += "," + csv_number(metric.min) + "," + csv_number(metric.max) + "," + csv_number(metric.mean);
        }
        text += "\n";
    }
    return text;
}

}  // namespace vml
