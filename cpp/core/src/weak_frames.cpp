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

// Text as a CSV cell: quoted, its quotes doubled, where it holds a comma, a
// quote or a line break.
std::string csv_text(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) {
        return text;
    }
    std::string quoted = "\"";
    for (const char c : text) {
        quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    }
    return quoted + "\"";
}

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

SegmentRules saved_segment_rules(const std::vector<std::string>& conditions, bool all, int min_frames,
                                 int merge_frames) {
    SegmentRules rules;
    for (const std::string& text : conditions) {
        if (const std::optional<WeakCondition> condition = parse_weak_condition(text);
            condition && find_metric(condition->key) != nullptr) {
            rules.conditions.push_back(*condition);
        }
    }
    rules.all = all;
    rules.min_frames = std::max(min_frames, 1);
    rules.merge_frames = std::max(merge_frames, 0);
    return rules;
}

std::vector<std::string> condition_texts(const std::vector<WeakCondition>& conditions) {
    std::vector<std::string> texts;
    for (const WeakCondition& condition : conditions) {
        texts.push_back(condition.text());
    }
    return texts;
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

SegmentRules scored_rules(const MetricResultSet& results, const SegmentRules& rules,
                          std::vector<WeakCondition>* left_out) {
    SegmentRules scored = rules;
    scored.conditions.clear();
    for (const WeakCondition& condition : rules.conditions) {
        const std::optional<ScoreSamples> samples = score_samples(results, condition.key);
        if (samples && !samples->frame.empty()) {
            scored.conditions.push_back(condition);
        } else if (left_out != nullptr) {
            left_out->push_back(condition);
        }
    }
    return scored;
}

std::vector<WeakSegment> weak_segments(const MetricResultSet& results, const SegmentRules& rules, double fps,
                                       int last_frame) {
    // The frame axis: from the first scored frame to the run's last.
    std::vector<std::pair<const WeakCondition*, ScoreSamples>> scored;
    int begin = last_frame;
    for (const WeakCondition& condition : rules.conditions) {
        // One on a metric with no scores is left out, as scored_rules does.
        if (const std::optional<ScoreSamples> samples = score_samples(results, condition.key);
            samples && !samples->frame.empty()) {
            scored.emplace_back(&condition, *samples);
            begin = std::min(begin, samples->frame.front());
        }
    }
    if (scored.empty() || begin > last_frame) {
        return {};
    }
    std::vector<char> weak;
    // Each condition's own frames, kept to say which made each segment weak
    // (none for a condition on a metric without scores).
    std::vector<std::vector<char>> masks(rules.conditions.size());
    for (const auto& [condition, samples] : scored) {
        const std::vector<char> mask = weak_mask(samples, *condition, begin, last_frame);
        masks[static_cast<std::size_t>(condition - rules.conditions.data())] = mask;
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
            segments.push_back(WeakSegment{frame, frame, 0.0, 0.0, {}, {}});
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
        for (const std::vector<char>& mask : masks) {
            long long frames = 0;
            for (int frame = segment.first_frame; !mask.empty() && frame <= segment.last_frame; ++frame) {
                frames += mask[static_cast<std::size_t>(frame - begin)] && inside(frame, rules.ranges) ? 1 : 0;
            }
            segment.condition_frames.push_back(frames);
        }
    }
    return segments;
}

std::vector<WeakSegment> weak_segments(const ComparisonResult& result, const SegmentRules& rules) {
    return weak_segments(result.metric_results, rules, result.fps, last_compared_frame(result));
}

std::vector<ConditionCount> condition_counts(const MetricResultSet& results, const SegmentRules& rules,
                                             int last_frame, const std::vector<WeakSegment>& segments) {
    std::vector<ConditionCount> counts(rules.conditions.size());
    for (std::size_t c = 0; c < rules.conditions.size(); ++c) {
        const std::optional<ScoreSamples> samples = score_samples(results, rules.conditions[c].key);
        if (samples && !samples->frame.empty() && samples->frame.front() <= last_frame) {
            const int begin = samples->frame.front();
            const std::vector<char> mask = weak_mask(*samples, rules.conditions[c], begin, last_frame);
            for (int frame = begin; frame <= last_frame; ++frame) {
                counts[c].holds += mask[static_cast<std::size_t>(frame - begin)] && inside(frame, rules.ranges) ? 1 : 0;
            }
        }
        for (const WeakSegment& segment : segments) {
            if (c < segment.condition_frames.size()) {
                counts[c].in_segments += segment.condition_frames[c];
            }
        }
    }
    return counts;
}

std::vector<ConditionCount> condition_counts(const ComparisonResult& result, const SegmentRules& rules,
                                             const std::vector<WeakSegment>& segments) {
    return condition_counts(result.metric_results, rules, last_compared_frame(result), segments);
}

Json segments_json(const std::vector<WeakSegment>& segments, const SegmentRules& rules,
                   const std::vector<WeakCondition>& left_out) {
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
    Json json{{"conditions", std::move(conditions)}};
    if (!left_out.empty()) {
        json["left_out"] = condition_texts(left_out);
    }
    json["combine"] = rules.all ? "all" : "any";
    json["min_frames"] = rules.min_frames;
    json["merge_frames"] = rules.merge_frames;
    json["ranges"] = std::move(ranges);
    json["segments"] = std::move(list);
    return json;
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

namespace {

// The metrics `results` has scores of, in the registry's order, with them.
std::vector<std::pair<std::string, ScoreSamples>> scored_metrics(const MetricResultSet& results) {
    std::vector<std::pair<std::string, ScoreSamples>> scored;
    for (const MetricDefinition& definition : metrics()) {
        if (std::optional<ScoreSamples> samples = score_samples(results, std::string(definition.key))) {
            scored.emplace_back(std::string(definition.key), *samples);
        }
    }
    return scored;
}

}  // namespace

Json worst_frames_json(const MetricResultSet& results, const std::string& key, const std::vector<WorstFrame>& worst,
                       std::size_t dips, const std::vector<FrameInterval>& ranges) {
    const std::vector<std::pair<std::string, ScoreSamples>> scored = scored_metrics(results);
    Json range_list = Json::array();
    for (const FrameInterval& range : ranges) {
        range_list.push_back(Json{{"first_frame", range.first}, {"last_frame", range.last}});
    }
    Json list = Json::array();
    for (std::size_t i = 0; i < worst.size(); ++i) {
        Json scores = Json::object();
        for (const auto& [metric, samples] : scored) {
            scores[metric] = json_number(score_at(samples, worst[i].frame));
        }
        list.push_back(Json{{"rank", i + 1},
                            {"frame", worst[i].frame},
                            {"time", worst[i].time},
                            {"scores", std::move(scores)}});
    }
    return Json{{"metric", key}, {"dips", dips}, {"ranges", std::move(range_list)}, {"worst_frames", std::move(list)}};
}

std::string worst_frames_csv(const MetricResultSet& results, const std::vector<WorstFrame>& worst) {
    const std::vector<std::pair<std::string, ScoreSamples>> scored = scored_metrics(results);
    std::string text = "rank,frame,time";
    for (const auto& [metric, samples] : scored) {
        text += "," + metric;
    }
    text += "\n";
    for (std::size_t i = 0; i < worst.size(); ++i) {
        text += std::to_string(i + 1) + "," + std::to_string(worst[i].frame) + "," + python_repr(worst[i].time);
        for (const auto& [metric, samples] : scored) {
            text += "," + csv_number(score_at(samples, worst[i].frame));
        }
        text += "\n";
    }
    return text;
}

std::string worst_frames_file_suffix(const std::string& key) { return key + ".worst"; }

Json segments_report_json(const std::vector<SegmentsReportVideo>& videos, const SegmentRules& rules) {
    SegmentRules without_ranges = rules;
    without_ranges.ranges.clear();
    Json report = segments_json({}, without_ranges);
    report.erase("ranges");
    report.erase("segments");
    Json list = Json::array();
    for (const SegmentsReportVideo& video : videos) {
        const std::vector<WeakSegment> segments = weak_segments(*video.result, without_ranges);
        long long frames = 0;
        for (const WeakSegment& segment : segments) {
            frames += segment.frames();
        }
        Json entry{{"name", video.name},
                   {"path", video.path},
                   {"reference", video.reference},
                   {"segment_count", segments.size()},
                   {"weak_frames", frames}};
        std::vector<WeakCondition> left_out;
        scored_rules(video.result->metric_results, without_ranges, &left_out);
        if (!left_out.empty()) {
            entry["left_out"] = condition_texts(left_out);
        }
        entry["segments"] = segments_json(segments, without_ranges).at("segments");
        list.push_back(std::move(entry));
    }
    report["videos"] = std::move(list);
    return report;
}

std::string segments_report_csv(const std::vector<SegmentsReportVideo>& videos, const SegmentRules& rules) {
    SegmentRules without_ranges = rules;
    without_ranges.ranges.clear();
    // segments_csv's header and rows, each row led by the video's cells.
    const std::string header = segments_csv({}, without_ranges);
    std::string text = "video,reference," + header;
    for (const SegmentsReportVideo& video : videos) {
        const std::string rows = segments_csv(weak_segments(*video.result, without_ranges), without_ranges);
        const std::string lead = csv_text(video.name) + "," + csv_text(video.reference) + ",";
        for (std::size_t start = header.size(); start < rows.size();) {
            const std::size_t end = rows.find('\n', start);
            text += lead + rows.substr(start, end - start + 1);
            start = end + 1;
        }
    }
    return text;
}

}  // namespace vml
