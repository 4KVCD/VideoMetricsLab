#include "weak_report.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <set>
#include <stdexcept>
#include <system_error>

#include "common.h"
#include "vml/export_csv.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/segment_presets.h"
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

std::string segments_source(const CompareArguments& arguments) {
    return arguments.segments_preset    ? "preset:" + *arguments.segments_preset
           : arguments.segments_profile ? "profile:" + *arguments.segments_profile
                                        : std::string("saved");
}

std::vector<std::string> take_saved_segments(CompareArguments& arguments, const Settings& settings,
                                             const std::vector<std::string>& asked) {
    const std::string option = "--segments " + segments_source(arguments);
    std::optional<SegmentProfile> profile;
    if (arguments.segments_preset) {
        const SegmentPreset* preset = segment_preset(*arguments.segments_preset);
        if (preset == nullptr) {
            std::string keys;
            for (const SegmentPreset& known : segment_presets()) {
                keys += (keys.empty() ? "" : ", ") + std::string(known.key);
            }
            throw std::invalid_argument(option + ": no weak segment preset has that name; choose from: " + keys);
        }
        profile = SegmentProfile{std::string(preset->key), preset->rules};
    } else {
        profile = arguments.segments_profile ? segment_profile_named(settings, *arguments.segments_profile)
                                             : segment_profile_in_use(settings);
    }
    if (!profile) {
        std::string names;
        for (const SegmentProfile& other : segment_profiles(settings)) {
            names += (names.empty() ? "" : ", ") + (other.name.empty() ? std::string("Default") : other.name);
        }
        throw std::invalid_argument(option + ": no weak segment profile has that name; the app's: " + names);
    }
    const SegmentRules& saved = profile->rules;
    // The conditions given after it: those on a metric replace all the
    // profile's on it, in the first one's place, or are added after them.
    const std::vector<WeakCondition> given = arguments.segments;
    const auto given_on = [&given](const std::string& key) {
        return std::any_of(given.begin(), given.end(), [&key](const WeakCondition& c) { return c.key == key; });
    };
    std::vector<std::string> left_out;
    std::set<std::string> replaced;
    arguments.segments.clear();
    for (const WeakCondition& condition : saved.conditions) {
        if (given_on(condition.key)) {
            if (replaced.insert(condition.key).second) {
                std::copy_if(given.begin(), given.end(), std::back_inserter(arguments.segments),
                             [&condition](const WeakCondition& c) { return c.key == condition.key; });
            }
        } else if (std::find(asked.begin(), asked.end(), condition.key) != asked.end()) {
            arguments.segments.push_back(condition);
        } else {
            left_out.push_back(condition.text());
        }
    }
    for (const WeakCondition& condition : given) {
        if (std::none_of(saved.conditions.begin(), saved.conditions.end(),
                         [&condition](const WeakCondition& c) { return c.key == condition.key; })) {
            arguments.segments.push_back(condition);
        }
    }
    if (saved.conditions.empty() && given.empty()) {
        throw std::invalid_argument(option + ": the profile has no conditions; set them in the app's Settings > "
                                    "Weak segments, or give them here (such as \"vmaf<90\")");
    }
    if (arguments.segments.empty() && given.empty()) {
        std::string texts;
        for (const std::string& text : left_out) {
            texts += (texts.empty() ? "" : ", ") + text;
        }
        throw std::invalid_argument(option + ": none of the " + (arguments.segments_preset ? "preset" : "profile") +
                                    "'s conditions (" + texts +
                                    ") is on a metric compared (-m)");
    }
    arguments.segments_all = !arguments.segments_any && (arguments.segments_all || saved.all);
    arguments.segment_min_frames = arguments.segment_min_frames.value_or(saved.min_frames);
    arguments.segment_merge_frames = arguments.segment_merge_frames.value_or(saved.merge_frames);
    return left_out;
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

bool write_segment_files(std::vector<TestVideo>& videos, const SegmentRules& rules,
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
    for (TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        const std::vector<WeakSegment> segments = weak_segments(*video.result, rules);
        // Named after the conditions' metrics, as the window names them.
        const std::string suffix = "." + segments_file_suffix(rules);
        const std::vector<std::filesystem::path> paths =
            unique_output_paths(folder, video.file_stem(), {suffix + ".json", suffix + ".csv"}, reserved);
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
        video.segment_files = paths;
        message(video.name() + ": its weak segments in " + platform::path_text(paths[0]) + " and .csv");
    }
    return written;
}

std::optional<std::string> write_segments_report(
    const std::vector<std::pair<const TestVideo*, std::filesystem::path>>& videos, const SegmentRules& rules,
    const std::filesystem::path& file) {
    std::vector<SegmentsReportVideo> report;
    for (const auto& [video, reference] : videos) {
        if (video->result) {
            report.push_back({video->name(), platform::path_text(video->path), platform::path_text(reference),
                              &*video->result});
        }
    }
    if (ascii_lower(platform::path_text(file.extension())) == ".csv") {
        try {
            write_text(file, text_file(segments_report_csv(report, rules)));
        } catch (const std::exception& failure) {
            return failure.what();
        }
        return std::nullopt;
    }
    return write_json_file(file, segments_report_json(report, rules));
}

}  // namespace vml::cli
