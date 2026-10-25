#include "report.h"

#include <algorithm>
#include <set>

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

std::string left_justified(const std::string& text, std::size_t width) {
    const std::size_t length = python_length(text);
    return length >= width ? text : text + std::string(width - length, ' ');
}

std::string right_justified(const std::string& text, std::size_t width) {
    const std::size_t length = python_length(text);
    return length >= width ? text : std::string(width - length, ' ') + text;
}

std::string name_of(const std::filesystem::path& path) { return platform::path_text(path.filename()); }

// A video's media information on one line.
std::string media_text(const VideoInfo& info) {
    return std::to_string(info.width) + "x" + std::to_string(info.height) + ", " + format_fixed(info.fps, 3) +
           " fps, " + info.codec_name + ", " + info.pix_fmt + ", " + format_hms(info.duration);
}

// A whole number with commas between thousands ("28,800").
std::string with_thousands(long long number) {
    std::string digits = std::to_string(number < 0 ? -number : number);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return (number < 0 ? "-" : "") + digits;
}

// Table rows as lines: the first column left-justified, the others right,
// the last as it is, two spaces between, trailing spaces cut.
void add_table(std::vector<std::string>& lines, const std::vector<std::vector<std::string>>& rows,
               bool last_column_free) {
    std::vector<std::size_t> widths(rows.front().size(), 0);
    for (const auto& row : rows) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            widths[column] = std::max(widths[column], python_length(row[column]));
        }
    }
    for (const auto& row : rows) {
        std::string line = "  " + left_justified(row[0], widths[0]);
        const std::size_t justified_end = last_column_free ? row.size() - 1 : row.size();
        for (std::size_t column = 1; column < justified_end; ++column) {
            line += "  " + right_justified(row[column], widths[column]);
        }
        if (last_column_free) {
            line += "  " + row.back();
        }
        lines.emplace_back(python_rstrip(line));
    }
}

// A table row for CVVDP on each of the video's other displays; why there
// is none goes to `lines`.
void add_other_display_rows(const TestVideo& video, std::vector<std::vector<std::string>>& rows,
                            std::vector<std::string>& lines) {
    const MetricDefinition& definition = metric_definition("cvvdp");
    for (const TestVideo& other : video.other_displays) {
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

}  // namespace

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
                          const std::vector<Summaries>& summaries, bool cancelled) {
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
                      const std::vector<Summaries>& summaries, bool cancelled) {
    Json reference{{"path", platform::path_text(source.path)}};
    add_media(reference, source);
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"format", results_format},
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
            entry["reference_crop"] = crop_json(result.source_crop);
            entry["test_crop"] = crop_json(result.distorted_crop);
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
    return document;
}

bool write_csvs(const std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                const std::function<void(const std::string&)>& message,
                const std::function<void(const std::string&)>& failed) {
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        failed("The CSV folder could not be made: " + python_os_error(error, folder));
        return false;
    }
    bool written = true;
    for (const TestVideo& video : videos) {
        if (!video.result) {
            continue;
        }
        // Each name checked against the files already written, as the
        // Python app checks it (they are written one by one).
        std::set<std::filesystem::path> reserved;
        const std::filesystem::path path = unique_output_path(
            folder, safe_filename_stem(platform::path_text(video.path.stem())), ".csv", reserved);
        try {
            export_csv(*video.result, path);
            message(video.name() + ": per-frame scores in " + platform::path_text(path));
        } catch (const std::exception& failure) {
            written = false;
            failed(video.name() + ": the CSV could not be written (" + failure.what() + ")");
        }
    }
    return written;
}

}  // namespace vml::cli
