#include "saved.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include "common.h"
#include "compare.h"
#include "report.h"
#include "test_video.h"
#include "stills.h"
#include "weak_report.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/run_file.h"
#include "vml/settings.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// How many of a video's weak segments the tables show (compare's too).
constexpr std::size_t shown_segments = 20;
// An ID's least prefix: show's (one of 16^6 is enough to tell a few
// thousand comparisons apart); delete's longer, so a typo deletes nothing.
constexpr std::size_t show_prefix = 6;
constexpr std::size_t delete_prefix = 12;

// The saved results' folder: the settings', else the default.

// A path as two are compared: whole, its separators the system's, in the
// case the system ignores.
std::string path_as_compared(std::filesystem::path path) {
    return platform::name_as_compared(platform::path_text(path.lexically_normal().make_preferred()));
}

bool is_hex(const std::string& text) {
    return !text.empty() &&
           std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

// `keys` in the app's order of metrics.
std::vector<std::string> in_metric_order(const std::vector<std::string>& keys) {
    std::vector<std::string> ordered;
    for (const MetricDefinition& metric : metrics()) {
        if (std::find(keys.begin(), keys.end(), metric.key) != keys.end()) {
            ordered.emplace_back(metric.key);
        }
    }
    return ordered;
}

// A result read back (saved, or a run file) as a compared video, its scores
// all saved: what the reports take.
TestVideo video_of(ComparisonResult result) {
    TestVideo video;
    video.path = result.distorted;
    video.info = result.distorted_info;
    video.source = result.source_info;
    video.metrics = in_metric_order(result.metric_results.keys());
    video.save_scores = false;
    video.from_cache = true;
    video.result = std::move(result);
    return video;
}

// The videos of one reference, as compare groups them.
struct Group {
    VideoInfo source;
    std::vector<TestVideo> videos;
};

void add_to_groups(std::vector<Group>& groups, TestVideo video) {
    const std::string reference = path_as_compared(video.source.path);
    const auto found = std::find_if(groups.begin(), groups.end(), [&reference](const Group& group) {
        return path_as_compared(group.source.path) == reference;
    });
    if (found != groups.end()) {
        found->videos.push_back(std::move(video));
    } else {
        groups.push_back(Group{video.source, {}});
        groups.back().videos.push_back(std::move(video));
    }
}

// The reports of `groups`, as compare gives them for scores it found saved:
// the tables on standard output (or the JSON there for --json -), and the
// files asked for. `read_from` says where the scores came from, in the JSON.
// False when a file could not be written.
bool report_groups(SavedArguments arguments, const Settings& settings, std::vector<Group>& groups,
                   const std::string& read_from) {
    CompareArguments& output = arguments.output;
    const auto message = [&output](const std::string& text) {
        if (!output.quiet) {
            print_error(text);
        }
    };
    const auto failed = [](const std::string& text) { print_error(text); };
    std::vector<std::string> asked;
    for (const Group& group : groups) {
        for (const TestVideo& video : group.videos) {
            for (const std::string& key : video.metrics) {
                if (std::find(asked.begin(), asked.end(), key) == asked.end()) {
                    asked.push_back(key);
                }
            }
        }
    }
    asked = in_metric_order(asked);
    if (output.segments_saved) {
        take_saved_segments_of_scores(output, settings, asked, message);
    }
    if (const std::optional<std::string> refused = segments_refusal(output.segments, asked)) {
        throw UsageError(*refused);
    }
    const SegmentRules rules = segment_rules(output);
    bool written = true;
    std::string tables;
    Json batch = Json::array();
    for (Group& group : groups) {
        std::vector<TestVideo>& videos = group.videos;
        const std::vector<Summaries> summaries = summarize(videos);
        if (output.run_dir) {
            written = write_run_files(videos, python_path(*output.run_dir), message, failed) && written;
        }
        if (output.csv) {
            written = write_csvs(videos, python_path(*output.csv), message, failed) && written;
        }
        if (output.frames_json) {
            written = write_csvs(videos, python_path(*output.frames_json), message, failed, true) && written;
        }
        if (!rules.conditions.empty() && output.segments_dir) {
            written =
                write_segment_files(videos, rules, python_path(*output.segments_dir), message, failed) && written;
        }
        if (output.stills) {
            written = write_stills(videos, output.stills_from.value_or("worst"),
                                   static_cast<std::size_t>(output.stills_count.value_or(5)), rules,
                                   python_path(*output.stills), message, failed) &&
                      written;
        }
        Json document = results_document(group.source, videos, summaries, false);
        document["read_from"] = read_from;
        if (!rules.conditions.empty()) {
            add_segments_json(document, videos, rules);
        }
        batch.push_back(std::move(document));
        tables += (tables.empty() ? "" : "\n") + results_table(group.source, videos, summaries, false, std::nullopt);
        if (output.worst) {
            tables += "\n" + worst_text(videos, static_cast<std::size_t>(*output.worst));
        }
        if (!rules.conditions.empty()) {
            tables += "\n" + segments_text(videos, rules, shown_segments);
        }
    }
    if (output.segments_report) {
        std::vector<std::pair<const TestVideo*, std::filesystem::path>> all;
        for (const Group& group : groups) {
            for (const TestVideo& video : group.videos) {
                all.emplace_back(&video, group.source.path);
            }
        }
        const std::filesystem::path file = python_path(*output.segments_report);
        if (const std::optional<std::string> failure = write_segments_report(all, rules, file)) {
            failed("The weak segments report could not be written: " + *failure);
            written = false;
        } else {
            message("Weak segments of every video in " + platform::path_text(file));
        }
    }
    // One reference: its document; several: {"batch": [...]}, as compare --batch writes them.
    const Json document = batch.size() == 1 ? batch.front() : Json{{"batch", batch}};
    if (output.json_path && *output.json_path == "-") {
        return print_results(python_json_dumps(document) + "\n") && written;
    }
    written = print_results(tables + "\n") && written;
    if (output.json_path) {
        if (const std::optional<std::string> failure = write_json_file(python_path(*output.json_path), document)) {
            failed("The JSON file could not be written: " + *failure);
            return false;
        }
        message("Results as JSON in " + platform::path_text(python_path(*output.json_path)));
    }
    return written;
}

std::string metrics_text(const SavedComparison& comparison) {
    // A metric this build doesn't know (a score saved by a newer version
    // sharing the folder) by its key, as the Saved Results tab lists it.
    std::string text;
    for (const std::string& key : comparison.keys()) {
        const MetricDefinition* metric = find_metric(key);
        text += (text.empty() ? "" : ", ") + (metric != nullptr ? std::string(metric->short_label) : key);
    }
    return text;
}

}  // namespace

void take_saved_segments_of_scores(CompareArguments& output, const Settings& settings,
                                   const std::vector<std::string>& asked,
                                   const std::function<void(const std::string&)>& message) {
    std::optional<WeakCondition> fallback;
    std::vector<std::string> left_out;
    try {
        left_out = take_saved_segments(output, settings, asked, &fallback);
    } catch (const std::invalid_argument& error) {
        throw UsageError(error.what());  // as compare: the message, exit code 2
    }
    for (const std::string& condition : left_out) {
        message("note: --segments " + segments_source(output) + ": " + condition +
                " is left out: no video here has its metric");
    }
    if (fallback) {
        message("note: --segments " + segments_source(output) +
                ": the profile has no condition on the metrics here, so " + fallback->text() +
                " is taken, as the Worst Frames tab starts (a condition given after it takes its place)");
    }
}

std::string saved_id(const SavedComparison& comparison) { return platform::path_text(comparison.folder.filename()); }

std::vector<const SavedComparison*> saved_named(const std::vector<SavedComparison>& saved, const std::string& item,
                                                std::size_t min_prefix) {
    std::vector<const SavedComparison*> found;
    const std::string lower = ascii_lower(item);
    if (is_hex(lower) && lower.size() >= min_prefix) {
        for (const SavedComparison& comparison : saved) {
            if (saved_id(comparison).starts_with(lower)) {
                found.push_back(&comparison);
            }
        }
        return found;
    }
    const bool path = item.find_first_of("/\\.") != std::string::npos;
    if (!path) {
        throw UsageError(item + ": not a saved comparison's ID (at least " + std::to_string(min_prefix) +
                         " of its hex digits, saved list gives them) nor a video's path");
    }
    std::error_code ignored;
    const std::filesystem::path whole = std::filesystem::absolute(python_path(item), ignored);
    const std::string wanted = path_as_compared(whole);
    for (const SavedComparison& comparison : saved) {
        if (path_as_compared(comparison.distorted) == wanted) {
            found.push_back(&comparison);
        }
    }
    return found;
}

ComparisonResult read_comparison(const std::string& item, const std::filesystem::path& folder) {
    const std::filesystem::path path = python_path(item);
    std::error_code ignored;
    if (std::filesystem::is_regular_file(path, ignored)) {
        return load_run(path).first;
    }
    if (!is_hex(ascii_lower(item)) || item.size() < show_prefix) {
        throw std::runtime_error("no such file, nor a saved comparison's ID (at least " +
                                 std::to_string(show_prefix) + " of its hex digits, saved list gives them)");
    }
    const std::vector<SavedComparison> all = list_saved_comparisons(folder);
    const std::vector<const SavedComparison*> found = saved_named(all, item, show_prefix);
    if (found.empty()) {
        throw std::runtime_error("no saved comparison has it (saved list gives them)");
    }
    if (found.size() > 1) {
        throw std::runtime_error("more than one saved comparison's ID begins with it; give more of it");
    }
    std::optional<std::pair<ComparisonResult, std::string>> loaded = load_saved_comparison(*found.front());
    if (!loaded) {
        throw std::runtime_error("its saved scores could not be read");
    }
    return std::move(loaded->first);
}

std::string saved_list_text(const std::vector<const SavedComparison*>& saved) {
    if (saved.empty()) {
        return "No saved comparisons.\n";
    }
    std::vector<std::vector<std::string>> rows{{"ID", "Saved (UTC)", "Test video", "Reference", "Size", "Metrics"}};
    for (const SavedComparison* comparison : saved) {
        rows.push_back({saved_id(*comparison).substr(0, 12), utc_text(comparison->saved), name_of(comparison->distorted),
                        name_of(comparison->source) + (comparison->reference_found ? "" : " (not found)"),
                        file_size_text(comparison->bytes), metrics_text(*comparison)});
    }
    std::vector<std::string> lines;
    add_table(lines, rows, true);
    std::string text;
    for (const std::string& line : lines) {
        text += line + "\n";
    }
    return text + with_thousands(static_cast<long long>(saved.size())) +
           (saved.size() == 1 ? " saved comparison\n" : " saved comparisons\n");
}

Json saved_list_json(const std::vector<const SavedComparison*>& saved) {
    Json list = Json::array();
    for (const SavedComparison* comparison : saved) {
        list.push_back(Json{{"id", saved_id(*comparison)},
                            {"test", platform::path_text(comparison->distorted)},
                            {"reference", platform::path_text(comparison->source)},
                            {"reference_found", comparison->reference_found},
                            {"saved_utc", utc_text(comparison->saved)},
                            {"bytes", comparison->bytes},
                            {"metrics", comparison->keys()}});
    }
    Json json = document_head(saved_list_format);
    json["comparisons"] = std::move(list);
    return json;
}

int saved(const SavedArguments& arguments, const Settings& settings) {
    // The settings' folder, or --cache-dir's (compare's own)
    const std::filesystem::path folder = saved_scores_use(arguments.profile, settings).folder;
    const std::vector<SavedComparison> all = list_saved_comparisons(folder);
    if (arguments.action == "list") {
        std::vector<const SavedComparison*> shown;
        // As the Saved Results tab's Filter: surrounding spaces ignored, case
        // as the file system compares names.
        const std::string filter =
            arguments.filter ? platform::name_as_compared(python_strip(*arguments.filter)) : std::string();
        for (const SavedComparison& comparison : all) {
            const std::vector<std::string> texts = comparison.filter_texts();
            if (filter.empty() || std::any_of(texts.begin(), texts.end(), [&filter](const std::string& text) {
                    return platform::name_as_compared(text).find(filter) != std::string::npos;
                })) {
                shown.push_back(&comparison);
            }
        }
        const std::optional<std::string>& json = arguments.output.json_path;
        if (json && *json == "-") {
            return print_results(python_json_dumps(saved_list_json(shown)) + "\n") ? exit_ok : exit_failed;
        }
        if (!print_results(saved_list_text(shown))) {
            return exit_failed;
        }
        if (json) {
            if (const std::optional<std::string> failure = write_json_file(python_path(*json), saved_list_json(shown))) {
                print_error("The JSON file could not be written: " + *failure);
                return exit_failed;
            }
        }
        return exit_ok;
    }
    const bool deleting = arguments.action == "delete";
    // Each item's comparisons, each once, in the order named.
    std::vector<const SavedComparison*> named;
    bool missing = false;
    for (const std::string& item : arguments.items) {
        const std::vector<const SavedComparison*> found =
            saved_named(all, item, deleting ? delete_prefix : show_prefix);
        if (found.empty()) {
            print_error(item + ": no saved comparison has it (saved list gives them)");
            missing = true;
        }
        if (deleting && found.size() > 1) {
            throw UsageError(item + ": more than one saved comparison's ID begins with it; give more of it");
        }
        for (const SavedComparison* comparison : found) {
            if (std::find(named.begin(), named.end(), comparison) == named.end()) {
                named.push_back(comparison);
            }
        }
    }
    bool ok = true;
    if (deleting) {
        for (const SavedComparison* comparison : named) {
            if (delete_saved_comparison(folder, *comparison)) {
                print_results("Deleted " + saved_id(*comparison).substr(0, 12) + ": " + name_of(comparison->distorted) +
                              " against " + name_of(comparison->source) + "\n");
            } else {
                print_error(saved_id(*comparison).substr(0, 12) + ": could not be deleted (a file in use?)");
                ok = false;
            }
        }
        return missing ? exit_input : ok ? exit_ok : exit_failed;
    }
    std::vector<Group> groups;
    for (const SavedComparison* comparison : named) {
        std::optional<std::pair<ComparisonResult, std::string>> loaded = load_saved_comparison(*comparison);
        if (!loaded) {
            print_error(saved_id(*comparison).substr(0, 12) + ": its saved scores could not be read");
            ok = false;
            continue;
        }
        add_to_groups(groups, video_of(std::move(loaded->first)));
    }
    if (!groups.empty()) {
        ok = report_groups(arguments, settings, groups, "saved results") && ok;
    }
    return missing ? exit_input : ok ? exit_ok : exit_failed;
}

int report(const SavedArguments& arguments, const Settings& settings) {
    std::vector<Group> groups;
    bool unread = false;
    // Patterns (runs\*.metrics.json) as compare finds its test videos.
    for (const std::filesystem::path& path : expand_videos(arguments.items, std::nullopt)) {
        if (const std::optional<std::string> nothing = found_nothing(path)) {
            print_error(platform::path_text(path) + ": " + *nothing);
            unread = true;
            continue;
        }
        try {
            auto [result, label] = load_run(path);
            add_to_groups(groups, video_of(std::move(result)));
        } catch (const std::exception& error) {
            print_error(platform::path_text(path) + ": not read: " + error.what());
            unread = true;
        }
    }
    const bool written = groups.empty() || report_groups(arguments, settings, groups, "run files");
    return unread ? exit_input : written ? exit_ok : exit_failed;
}

}  // namespace vml::cli
