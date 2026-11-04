#include "report.h"

#include <algorithm>
#include <set>

#include "common.h"
#include "vml/build_info.h"
#include "vml/difference.h"
#include "vml/export_csv.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/results_summary.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// A video's media information on one line.
std::string media_text(const VideoInfo& info) {
    return std::to_string(info.width) + "x" + std::to_string(info.height) + ", " + format_fixed(info.fps, 3) +
           " fps, " + info.codec_name + ", " + info.pix_fmt + ", " + format_hms(info.duration);
}

// A table row for CVVDP on each of the video's other displays; why there
// is none goes to `lines`.
void add_other_display_rows(const TestVideo& video, std::vector<std::vector<std::string>>& rows,
                            std::vector<std::string>& lines) {
    const MetricDefinition& definition = metric_definition("cvvdp");
    for (const TestVideo& other : video.other_displays) {
        for (const std::string& note : other.notes) {
            lines.push_back("  note: CVVDP on " + python_string_repr(other.display) + ": " + note);
        }
        const std::optional<Json> summary = other.result ? metric_summary(*other.result, "cvvdp") : std::nullopt;
        if (!summary) {
            const std::string reason = !other.error.empty() ? other.error
                                       : !other.failures.empty() && !other.failures.front().second.empty()
                                           ? other.failures.front().second
                                           : "not calculated";
            lines.push_back("  CVVDP on " + python_string_repr(other.display) + " failed: " + reason);
            continue;
        }
        rows.push_back({std::string(definition.label) + " (" + other.display + ")",
                        definition.format_value(summary_number(summary->at("score"))), "", "", "", "",
                        ascii_upper(summary->at("computed_on").get<std::string>())});
    }
}

// CVVDP on one display, as --json lists each display's: the display, and
// its score or why there is none.
Json display_entry(const TestVideo& target) {
    const std::optional<Json> summary = target.result ? metric_summary(*target.result, "cvvdp") : std::nullopt;
    Json entry{{"display", target.display}, {"score", summary ? summary->at("score") : Json(nullptr)}};
    if (target.result) {
        entry["from_saved_scores"] = target.from_cache;
    }
    if (target.csv_file) {
        entry["csv"] = platform::path_text(*target.csv_file);
    }
    if (!target.notes.empty()) {
        entry["notes"] = target.notes;
    }
    std::string reason = target.error;
    if (reason.empty()) {
        const auto cvvdp = std::find_if(target.failures.begin(), target.failures.end(),
                                        [](const auto& failure) { return failure.first == "cvvdp"; });
        reason = cvvdp != target.failures.end() ? cvvdp->second : "";
    }
    if (reason.empty() && !target.failures.empty()) {
        reason = target.failures.front().second;
    }
    if (!reason.empty()) {
        entry["error"] = reason;
    }
    return entry;
}

// Several videos side by side: one line each, every metric's score.
// Nothing for a single video.
void add_summary(std::vector<std::string>& lines, const std::vector<TestVideo>& videos,
                 const std::vector<Summaries>& summaries) {
    std::vector<std::size_t> scored;
    for (std::size_t i = 0; i < videos.size(); ++i) {
        if (videos[i].result) {
            scored.push_back(i);
        }
    }
    if (scored.size() < 2) {
        return;
    }
    std::vector<std::string> keys;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (std::any_of(scored.begin(), scored.end(), [&](std::size_t i) { return summaries[i].contains(key); })) {
            keys.push_back(key);
        }
    }
    if (keys.empty()) {
        return;
    }
    std::vector<std::vector<std::string>> rows{{"Video"}};
    for (const std::string& key : keys) {
        rows.front().emplace_back(metric_definition(key).label);
    }
    for (const std::size_t i : scored) {
        std::vector<std::string> cells{videos[i].name()};
        for (const std::string& key : keys) {
            const auto summary = summaries[i].find(key);
            cells.push_back(summary == summaries[i].end()
                                ? ""
                                : metric_definition(key).format_value(summary_number(summary->second.at("score"))));
        }
        rows.push_back(cells);
    }
    lines.emplace_back("");
    lines.emplace_back(std::find(keys.begin(), keys.end(), "cvvdp") != keys.end()
                           ? "Summary (mean scores; CVVDP is one score for the whole video)"
                           : "Summary (mean scores)");
    add_table(lines, rows, false);
}

// --difference's table: a row per metric both videos have, and a note for
// those only one has.
void add_difference(std::vector<std::string>& lines, const std::vector<TestVideo>& videos,
                    const std::vector<Summaries>& summaries, const Json& difference, bool cancelled) {
    lines.emplace_back("");
    lines.push_back("Difference: " + videos[0].name() + " minus " + videos[1].name() +
                    ", per frame (CVVDP per second)" + (cancelled ? ", of the scores finished" : ""));
    std::vector<std::vector<std::string>> rows{{"Metric", "Score", "Mean", "Min", "Max", "Paired"}};
    for (const auto& [key, entry] : difference.items()) {
        const MetricDefinition& definition = metric_definition(key);
        const auto show = [&definition, &entry](const char* field) {
            return definition.format_delta(summary_number(entry.at(field)));
        };
        rows.push_back({std::string(definition.label), show("score"), show("mean"), show("minimum"),
                        show("maximum"), with_thousands(entry.at("paired").get<long long>())});
    }
    if (rows.size() > 1) {
        add_table(lines, rows, false);
    }
    std::vector<std::string> one_only;
    for (std::size_t v = 0; v < 2; ++v) {
        for (const std::string& key : videos[v].metrics) {
            if (summaries[v].contains(key) && !difference.contains(key)) {
                one_only.push_back(std::string(metric_definition(key).label) + " (" + videos[v].name() + ")");
            }
        }
    }
    if (!one_only.empty()) {
        std::string note = "  note: scored for one video only: ";
        for (std::size_t i = 0; i < one_only.size(); ++i) {
            note += (i == 0 ? "" : ", ") + one_only[i];
        }
        lines.push_back(note);
    }
}

// The differences of a metric's points in both results, at the first's
// points (vml/difference.h); nothing where one has no points of its own.
std::vector<double> paired_differences(const ComparisonResult& first, const ComparisonResult& second,
                                       const std::string& key) {
    if (const SequenceMetricResult* mine = first.metric_results.sequence(key)) {
        const SequenceMetricResult* other = second.metric_results.sequence(key);
        return other == nullptr ? std::vector<double>()
                                : second_difference(mine->time, mine->values, other->time, other->values);
    }
    const FrameMetricResult* mine = first.metric_results.frame(key);
    const FrameMetricResult* other = second.metric_results.frame(key);
    return mine == nullptr || other == nullptr
               ? std::vector<double>()
               : frame_difference(mine->frame, mine->values, other->frame, other->values);
}

}  // namespace

std::optional<Json> difference_metrics(const std::vector<TestVideo>& videos, const std::vector<Summaries>& summaries) {
    if (videos.size() != 2 || !videos[0].result || !videos[1].result) {
        return std::nullopt;
    }
    Json metrics = Json::object();
    for (const std::string& key : videos[0].metrics) {
        const auto ours = summaries[0].find(key);
        const auto theirs = summaries[1].find(key);
        if (ours == summaries[0].end() || theirs == summaries[1].end()) {
            continue;
        }
        const std::optional<double> first = summary_number(ours->second.at("score"));
        const std::optional<double> second = summary_number(theirs->second.at("score"));
        const std::optional<DifferenceSummary> points =
            summarize_difference(paired_differences(*videos[0].result, *videos[1].result, key));
        metrics[key] = Json{
            {"score", json_number(first && second ? std::optional(point_difference(*first, *second)) : std::nullopt)},
            {"mean", json_number(points ? std::optional(points->mean) : std::nullopt)},
            {"minimum", json_number(points ? std::optional(points->minimum) : std::nullopt)},
            {"maximum", json_number(points ? std::optional(points->maximum) : std::nullopt)},
            {"paired", points ? points->paired : 0},
        };
    }
    return metrics;
}

std::string TestVideo::name() const { return name_of(path); }

std::string TestVideo::label() const { return for_display_only ? name() + " (CVVDP on " + display + ")" : name(); }

std::vector<Summaries> summarize(const std::vector<TestVideo>& videos) {
    std::vector<Summaries> all;
    for (const TestVideo& video : videos) {
        Summaries& summaries = all.emplace_back();
        if (!video.result) {
            continue;
        }
        for (const std::string& key : video.metrics) {
            if (std::optional<Json> summary = metric_summary(*video.result, key)) {
                summaries.emplace(key, std::move(*summary));
            }
        }
    }
    return all;
}

std::string video_status(const TestVideo& video, bool cancelled) {
    std::size_t have = 0;
    for (const std::string& key : video.metrics) {
        if (video.result && video.result->has_metric(key)) {
            ++have;
        }
    }
    const bool others_scored =
        std::all_of(video.other_displays.begin(), video.other_displays.end(), [](const TestVideo& other) {
            return other.result && other.result->has_metric("cvvdp") && other.failures.empty();
        });
    if (!video.metrics.empty() && have == video.metrics.size() && video.failures.empty() && others_scored) {
        return "ok";
    }
    if (cancelled && video.error.empty() && video.failures.empty()) {
        return "cancelled";
    }
    return have > 0 ? "partial" : "failed";
}

std::string results_table(const VideoInfo& source, const std::vector<TestVideo>& videos,
                          const std::vector<Summaries>& summaries, bool cancelled,
                          const std::optional<Json>& difference) {
    std::vector<std::string> lines{"Reference: " + name_of(source.path) + " (" + media_text(source) + ")"};
    for (std::size_t v = 0; v < videos.size(); ++v) {
        const TestVideo& video = videos[v];
        lines.emplace_back("");
        const std::string status = video_status(video, cancelled);
        std::string head = video.name() + (video.info ? " (" + media_text(*video.info) + ")" : "");
        if (status != "ok") {
            head += " -- " + ascii_upper(status);
        } else if (video.from_cache) {
            head += " -- saved scores";
        }
        lines.push_back(head);
        if (!video.error.empty()) {
            lines.push_back("  " + video.error);
        }
        for (const std::string& note : video.notes) {
            lines.push_back("  note: " + note);
        }
        for (const auto& [key, reason] : video.failures) {
            const MetricDefinition* definition = find_metric(key);
            lines.push_back("  " + (definition != nullptr ? std::string(definition->label) : "metrics") +
                            " failed: " + reason);
        }
        if (!video.result) {
            continue;
        }
        std::vector<std::vector<std::string>> rows;
        for (const std::string& key : video.metrics) {
            const auto found = summaries[v].find(key);
            if (found == summaries[v].end()) {
                continue;
            }
            const Json& summary = found->second;
            const MetricDefinition& definition = metric_definition(key);
            const auto show = [&definition, &summary](const char* field) {
                return definition.format_value(summary_number(summary.at(field)));
            };
            const std::string on = ascii_upper(summary.at("computed_on").get<std::string>());
            if (definition.kind != MetricKind::frame) {  // one score for the video
                // With several displays, each CVVDP row names its own.
                const std::string label = video.other_displays.empty()
                                              ? std::string(definition.label)
                                              : std::string(definition.label) + " (" + video.display + ")";
                rows.push_back({label, show("score"), "", "", "", "", on});
                continue;
            }
            rows.push_back({std::string(definition.label), show("score"), show("median"), show("min"), show("max"),
                            show("worst_1_percent") + " (" + summary.at("worst_is").get<std::string>() + ")", on});
        }
        add_other_display_rows(video, rows, lines);
        if (!rows.empty()) {
            rows.insert(rows.begin(), {"Metric", "Mean", "Median", "Min", "Max", "1% worst", "On"});
            add_table(lines, rows, true);
            lines.push_back("  " + with_thousands(video.result->compared_frame_count) + " frames compared");
        }
    }
    add_summary(lines, videos, summaries);
    if (difference) {
        add_difference(lines, videos, summaries, *difference, cancelled);
    }
    if (cancelled) {
        lines.emplace_back("");
        lines.emplace_back("Cancelled: the scores finished before that are saved.");
    }
    std::string text;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        text += (i == 0 ? "" : "\n") + lines[i];
    }
    return text;
}

Json results_document(const VideoInfo& source, const std::vector<TestVideo>& videos,
                      const std::vector<Summaries>& summaries, bool cancelled, const std::optional<Json>& difference) {
    Json reference{{"path", platform::path_text(source.path)}};
    add_media(reference, source);
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"commit", build_commit()}, {"format", results_format},
                  {"cancelled", cancelled}, {"reference", reference}, {"videos", Json::array()}};
    for (std::size_t v = 0; v < videos.size(); ++v) {
        const TestVideo& video = videos[v];
        // Each video names its reference too, as the window's export does,
        // whose videos can have different ones.
        Json entry{{"path", platform::path_text(video.path)},
                   {"reference", platform::path_text(source.path)},
                   {"status", video_status(video, cancelled)}};
        if (video.info) {
            add_media(entry, *video.info);
        }
        if (!video.error.empty()) {
            entry["error"] = video.error;
        }
        if (!video.notes.empty()) {
            entry["notes"] = video.notes;
        }
        if (video.csv_file) {
            entry["csv"] = platform::path_text(*video.csv_file);
        }
        if (!video.failures.empty()) {
            Json failed = Json::object();
            for (const auto& [key, reason] : video.failures) {
                failed[key] = reason;
            }
            entry["failed_metrics"] = failed;
        }
        if (video.result) {
            const ComparisonResult& result = *video.result;
            entry["from_saved_scores"] = video.from_cache;
            entry["compared_frames"] = result.compared_frame_count;
            entry["vmaf_model"] = result.model;
            entry["reference_crop"] = summary_crop_json(result.source_crop);
            entry["test_crop"] = summary_crop_json(result.distorted_crop);
            Json scored = Json::object();  // in the order asked for
            for (const std::string& key : video.metrics) {
                if (const auto found = summaries[v].find(key); found != summaries[v].end()) {
                    scored[key] = found->second;
                }
            }
            entry["metrics"] = scored;
        }
        if (!video.display.empty()) {
            entry["cvvdp_display"] = video.display;
        }
        if (!video.other_displays.empty()) {
            Json displays = Json::array({display_entry(video)});
            for (const TestVideo& other : video.other_displays) {
                displays.push_back(display_entry(other));
            }
            entry["cvvdp_displays"] = displays;
        }
        document["videos"].push_back(entry);
    }
    if (difference) {
        document["difference"] = Json{{"first", platform::path_text(videos[0].path)},
                                      {"second", platform::path_text(videos[1].path)},
                                      {"metrics", *difference}};
    }
    return document;
}

namespace {

// One video's (or one further display's) scores in `folder`, under a name no
// other file has. False, and said, when the file cannot be written.
bool write_csv_of(TestVideo& video, const std::string& name, const std::filesystem::path& folder,
                  std::set<std::filesystem::path>& reserved, const std::function<void(const std::string&)>& message,
                  const std::function<void(const std::string&)>& failed) {
    const std::filesystem::path path = unique_output_path(folder, name, ".csv", reserved);
    try {
        export_csv(*video.result, path);
    } catch (const std::exception& failure) {
        failed(video.label() + ": the CSV could not be written (" + failure.what() + ")");
        return false;
    }
    video.csv_file = path;
    message(video.label() + ": per-frame scores in " + platform::path_text(path));
    if (const std::optional<std::string> note = name_taken_note(path, name, ".csv")) {
        failed(*note);
    }
    return true;
}

}  // namespace

bool write_csvs(std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                const std::function<void(const std::string&)>& message,
                const std::function<void(const std::string&)>& failed) {
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        failed("The CSV folder could not be made: " + python_os_error(error, folder));
        return false;
    }
    bool written = true;
    std::set<std::filesystem::path> reserved;
    for (TestVideo& video : videos) {
        const std::string name = platform::path_text(video.path.stem());
        if (video.result) {
            written = write_csv_of(video, name, folder, reserved, message, failed) && written;
        }
        for (TestVideo& other : video.other_displays) {
            if (other.result) {
                written = write_csv_of(other, name + "_" + other.display, folder, reserved, message, failed) && written;
            }
        }
    }
    return written;
}

bool write_difference_csv(const std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                          const std::function<void(const std::string&)>& message,
                          const std::function<void(const std::string&)>& failed) {
    if (videos.size() != 2 || !videos[0].result || !videos[1].result) {
        return true;  // nothing to write: the tables say which has no result
    }
    // write_csvs has made the folder, or said why it could not.
    std::set<std::filesystem::path> reserved;
    const std::filesystem::path path = unique_output_path(
        folder,
        difference_label(platform::path_text(videos[0].path.stem()), platform::path_text(videos[1].path.stem())),
        ".csv", reserved);
    const std::string names = videos[0].name() + " minus " + videos[1].name();
    try {
        export_difference_csv(*videos[0].result, *videos[1].result, path);
        message(names + ": per-frame difference in " + platform::path_text(path));
        return true;
    } catch (const std::exception& failure) {
        failed(names + ": the CSV could not be written (" + failure.what() + ")");
        return false;
    }
}

}  // namespace vml::cli
