#include "report.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "vml/export_csv.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/stats.h"
#include "vml/time_format.h"

namespace vml::cli {

namespace {

using Json = nlohmann::ordered_json;

// How many characters UTF-8 text shows (Python's len() of the string, by
// which it pads columns).
std::size_t length_of(const std::string& text) {
    return static_cast<std::size_t>(std::count_if(
        text.begin(), text.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

std::string left_justified(const std::string& text, std::size_t width) {
    const std::size_t length = length_of(text);
    return length >= width ? text : text + std::string(width - length, ' ');
}

std::string right_justified(const std::string& text, std::size_t width) {
    const std::size_t length = length_of(text);
    return length >= width ? text : std::string(width - length, ' ') + text;
}

std::string without_trailing_spaces(std::string text) {
    text.erase(text.find_last_not_of(" \t\r\n") + 1);
    return text;
}

std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return text;
}

std::string name_of(const std::filesystem::path& path) { return platform::path_text(path.filename()); }

// JSON has no infinity: a metric's "identical" is the string, and no score
// (or NaN) is null.
Json json_number(const std::optional<double>& value) {
    if (!value || std::isnan(*value)) {
        return nullptr;
    }
    if (std::isinf(*value)) {
        return *value > 0 ? "inf" : "-inf";
    }
    return *value;
}

// A summary number back as a number ("inf" is text in the JSON).
std::optional<double> value_of(const Json& number) {
    if (number.is_string()) {
        return python_float(number.get<std::string>());
    }
    if (number.is_null()) {
        return std::nullopt;
    }
    return number.get<double>();
}

// One metric of a result: its score, and for a per-frame metric the
// statistics the Metric Graphs tab shows; nothing if it has none.
std::optional<Json> metric_summary(const ComparisonResult& result, const std::string& key) {
    const MetricDefinition& definition = metric_definition(key);
    if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
        return Json{{"score", json_number(sequence->score)},
                    {"computed_on", sequence->provenance.compute_backend},
                    {"implementation", sequence->provenance.implementation},
                    {"version", sequence->provenance.implementation_version}};
    }
    const FrameMetricResult* metric = result.metric_results.frame(key);
    if (metric == nullptr || metric->values.empty()) {
        return std::nullopt;
    }
    const std::vector<double> values(metric->values.begin(), metric->values.end());
    const ScoreStats stats = compute_stats(values, {}, definition.aggregation, definition.direction);
    // The worst frames are the low ones, or for Butteraugli the high ones.
    const std::string worst = definition.direction == MetricDirection::lower_is_better ? "high" : "low";
    return Json{{"score", json_number(metric->aggregate())},
                {"frames", stats.count},
                {"median", json_number(stats.median)},
                {"stdev", json_number(stats.stdev)},
                {"min", json_number(stats.minimum)},
                {"max", json_number(stats.maximum)},
                {"worst_is", worst},
                {"worst_10_percent", json_number(stats.worst_10_percent)},
                {"worst_5_percent", json_number(stats.worst_5_percent)},
                {"worst_1_percent", json_number(stats.worst_1_percent)},
                {"worst_0.1_percent", json_number(stats.worst_0_1_percent)},
                {"computed_on", metric->provenance.compute_backend},
                {"implementation", metric->provenance.implementation},
                {"version", metric->provenance.implementation_version}};
}

Json crop_json(const std::optional<CropBox>& box) {
    if (!box) {
        return nullptr;
    }
    return Json{{"x", box->x}, {"y", box->y}, {"width", box->w}, {"height", box->h}};
}

// A video's media information as --json writes it.
void add_media(Json& entry, const VideoInfo& info) {
    entry["width"] = info.width;
    entry["height"] = info.height;
    entry["fps"] = info.fps;
    entry["codec"] = info.codec_name;
    entry["pixel_format"] = info.pix_fmt;
    entry["duration"] = info.duration;
}

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
            widths[column] = std::max(widths[column], length_of(row[column]));
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
        lines.push_back(without_trailing_spaces(line));
    }
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
                                : metric_definition(key).format_value(value_of(summary->second.at("score"))));
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
    if (!video.metrics.empty() && have == video.metrics.size() && video.failures.empty()) {
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
            head += " -- " + upper(status);
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
                return definition.format_value(value_of(summary.at(field)));
            };
            const std::string on = upper(summary.at("computed_on").get<std::string>());
            if (!summary.contains("frames")) {  // one score for the video
                rows.push_back({std::string(definition.label), show("score"), "", "", "", "", on});
                continue;
            }
            rows.push_back({std::string(definition.label), show("score"), show("median"), show("min"), show("max"),
                            show("worst_1_percent") + " (" + summary.at("worst_is").get<std::string>() + ")", on});
        }
        if (!rows.empty()) {
            rows.insert(rows.begin(), {"Metric", "Mean", "Median", "Min", "Max", "1% worst", "On"});
            add_table(lines, rows, true);
            lines.push_back("  " + with_thousands(video.result->compared_frame_count) + " frames compared");
        }
    }
    add_summary(lines, videos, summaries);
    if (cancelled) {
        lines.emplace_back("");
        // The Python app adds that the scores finished before were saved;
        // this version saves none yet (compare.h).
        lines.emplace_back("Cancelled.");
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
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"format", 1}, {"cancelled", cancelled},
                  {"reference", reference}, {"videos", Json::array()}};
    for (std::size_t v = 0; v < videos.size(); ++v) {
        const TestVideo& video = videos[v];
        Json entry{{"path", platform::path_text(video.path)}, {"status", video_status(video, cancelled)}};
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
        document["videos"].push_back(entry);
    }
    return document;
}

bool write_csvs(const std::vector<TestVideo>& videos, const std::filesystem::path& folder,
                const std::function<void(const std::string&)>& message) {
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error) {
        message("The CSV folder could not be made: " + error.message());
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
            message(video.name() + ": the CSV could not be written (" + failure.what() + ")");
        }
    }
    return written;
}

}  // namespace vml::cli
