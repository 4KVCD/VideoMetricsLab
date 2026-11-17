#include "weak_report.h"

#include <algorithm>
#include <exception>
#include <set>
#include <stdexcept>
#include <system_error>

#include "common.h"
#include "vml/export_csv.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/temp_files.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// One metric's worst dips in `video`, as a table under the metric's name:
// the frame, its time, and every metric's score there, this one's first.
void add_worst_table(std::vector<std::string>& lines, const TestVideo& video, const std::string& key,
                     std::size_t count) {
    const ComparisonResult& result = *video.result;
    const std::optional<ScoreSamples> samples = score_samples(result.metric_results, key);
    if (!samples) {
        return;
    }
    const MetricDefinition& definition = metric_definition(key);
    std::vector<std::string> keys{key};
    for (const std::string& other : video.metrics) {
        if (other != key && score_samples(result.metric_results, other)) {
            keys.push_back(other);
        }
    }
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header{"Frame", "Time"};
    for (const std::string& column : keys) {
        header.emplace_back(metric_definition(column).short_label);
    }
    rows.push_back(std::move(header));
    for (const WorstFrame& worst : worst_frames(*samples, definition.direction, count)) {
        std::vector<std::string> row{std::to_string(worst.frame), format_hms(worst.time, 3)};
        for (const std::string& column : keys) {
            row.push_back(metric_definition(column).format_value(
                score_at(*score_samples(result.metric_results, column), worst.frame)));
        }
        rows.push_back(std::move(row));
    }
    lines.push_back("");
    lines.push_back(std::string(definition.label) + (samples->per_second ? " (second by second)" : ""));
    if (rows.size() == 1) {
        lines.emplace_back("  no frame has a score");
        return;
    }
    add_table(lines, rows, false);
}

std::string joined(const std::vector<std::string>& lines) {
    std::string text;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        text += (i == 0 ? "" : "\n") + lines[i];
    }
    return text;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    write_file_whole(path, [&text](std::ostream& out) { out << text; });
}

}  // namespace

SegmentRules segment_rules(const CompareArguments& arguments) {
    SegmentRules rules;
    rules.conditions = arguments.segments;
    rules.all = arguments.segments_all;
    rules.min_frames = arguments.segment_min_frames.value_or(rules.min_frames);
    rules.merge_frames = arguments.segment_merge_frames.value_or(rules.merge_frames);
    return rules;
}

std::optional<std::string> segments_refusal(const std::vector<WeakCondition>& conditions,
                                            const std::vector<std::string>& asked) {
    for (const WeakCondition& condition : conditions) {
        if (std::find(asked.begin(), asked.end(), condition.key) == asked.end()) {
            return "--segments: " + condition.key + " is not one of the metrics compared (-m)";
        }
    }
    return std::nullopt;
}

std::string worst_text(const std::vector<TestVideo>& videos, std::size_t count) {
    std::vector<std::string> lines;
    for (const TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        lines.push_back("");
        lines.push_back(video.name() + ": each metric's " + std::to_string(count) +
                        " worst dips (a frame within 1 s of a worse one is part of its dip)");
        for (const std::string& key : video.metrics) {
            add_worst_table(lines, video, key, count);
        }
    }
    return joined(lines);
}

std::string segments_text(const std::vector<TestVideo>& videos, const SegmentRules& rules, std::size_t shown) {
    std::string conditions;
    for (const WeakCondition& condition : rules.conditions) {
        conditions += (conditions.empty() ? "" : rules.all ? " and " : " or ") + condition.text();
    }
    std::vector<std::string> lines;
    for (const TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        const std::vector<WeakSegment> segments = weak_segments(*video.result, rules);
        lines.push_back("");
        lines.push_back(video.name() + ": " + with_thousands(static_cast<long long>(segments.size())) +
                        (segments.size() == 1 ? " segment" : " segments") + " where " + conditions);
        if (segments.empty()) {
            continue;
        }
        std::vector<std::vector<std::string>> rows;
        std::vector<std::string> header{"First", "Last", "Frames", "Start", "End"};
        for (const SegmentMetric& metric : segments.front().metrics) {
            const std::string label(metric_definition(metric.key).short_label);
            header.push_back(label + " min");
            header.push_back(label + " mean");
        }
        rows.push_back(std::move(header));
        for (std::size_t i = 0; i < segments.size() && i < shown; ++i) {
            const WeakSegment& segment = segments[i];
            std::vector<std::string> row{std::to_string(segment.first_frame), std::to_string(segment.last_frame),
                                         std::to_string(segment.frames()), format_hms(segment.start_time, 3),
                                         format_hms(segment.end_time, 3)};
            for (const SegmentMetric& metric : segment.metrics) {
                const MetricDefinition& definition = metric_definition(metric.key);
                // The worst end: Butteraugli's highest.
                row.push_back(definition.format_value(
                    definition.direction == MetricDirection::lower_is_better ? metric.max : metric.min));
                row.push_back(definition.format_value(metric.mean));
            }
            rows.push_back(std::move(row));
        }
        add_table(lines, rows, false);
        if (segments.size() > shown) {
            lines.push_back("  ... and " + with_thousands(static_cast<long long>(segments.size() - shown)) +
                            " more (all in --segments-dir's files and the --json)");
        }
    }
    return joined(lines);
}

void add_segments_json(Json& document, const std::vector<TestVideo>& videos, const SegmentRules& rules) {
    Json& entries = document.at("videos");
    for (std::size_t v = 0; v < videos.size() && v < entries.size(); ++v) {
        if (videos[v].result) {
            entries[v]["weak_segments"] = segments_json(weak_segments(*videos[v].result, rules), rules);
        }
    }
}

bool write_segment_files(const std::vector<TestVideo>& videos, const SegmentRules& rules,
                         const std::filesystem::path& folder,
                         const std::function<void(const std::string&)>& message,
                         const std::function<void(const std::string&)>& failed) {
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        failed("The segments folder could not be made: " + python_os_error(error, folder));
        return false;
    }
    bool written = true;
    std::set<std::filesystem::path> reserved;
    for (const TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        const std::vector<WeakSegment> segments = weak_segments(*video.result, rules);
        const std::vector<std::filesystem::path> paths = unique_output_paths(
            folder, platform::path_text(video.path.stem()), {".segments.json", ".segments.csv"}, reserved);
        try {
            if (const std::optional<std::string> wrong = write_json_file(paths[0], segments_json(segments, rules))) {
                throw std::runtime_error(*wrong);
            }
            write_text(paths[1], text_file(segments_csv(segments, rules)));
        } catch (const std::exception& failure) {
            failed(video.name() + ": the segment files could not be written (" + failure.what() + ")");
            written = false;
            continue;
        }
        message(video.name() + ": its weak segments in " + platform::path_text(paths[0]) + " and .segments.csv");
    }
    return written;
}

}  // namespace vml::cli
