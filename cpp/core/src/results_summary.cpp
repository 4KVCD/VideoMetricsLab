#include "vml/results_summary.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "vml/film_grain_detect.h"
#include "vml/film_grain_scoring.h"
#include "vml/hardware_decoding.h"
#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/stats.h"
#include "vml/weak_frames.h"

namespace vml {

std::string csv_cell(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) {
        return text;
    }
    std::string quoted = "\"";
    for (const char c : text) {
        quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    }
    return quoted + "\"";
}

namespace {

using Json = nlohmann::ordered_json;

// How many dips' worst frames a summary lists ("worst_frames").
constexpr std::size_t summary_worst_frames = 10;

// The statistics of `samples`, a metric's frames or CVVDP's seconds
// (`unit`), added to `summary`: how many have a score, then the rest, each
// null where none has, then the worst frames of its worst dips.
void add_statistics(Json& summary, const ScoreSamples& samples, const MetricDefinition& definition,
                    const std::string& unit) {
    const std::vector<double> scores(samples.values.begin(), samples.values.end());
    const ScoreStats stats = compute_stats(scores, definition.thresholds, definition.aggregation, definition.direction);
    const auto statistic = [&stats](double value) { return stats.count == 0 ? Json(nullptr) : json_number(value); };
    summary[unit] = stats.count;
    summary["median"] = statistic(stats.median);
    summary["stdev"] = statistic(stats.stdev);
    summary["min"] = statistic(stats.minimum);
    summary["max"] = statistic(stats.maximum);
    // The worst are the low ones, or for Butteraugli the high ones.
    summary["worst_is"] = definition.direction == MetricDirection::lower_is_better ? "high" : "low";
    summary["worst_10_percent"] = statistic(stats.worst_10_percent);
    summary["worst_5_percent"] = statistic(stats.worst_5_percent);
    summary["worst_1_percent"] = statistic(stats.worst_1_percent);
    summary["worst_0.1_percent"] = statistic(stats.worst_0_1_percent);
    if (stats.identical > 0) {
        summary["identical_" + unit] = stats.identical;
    }
    if (!stats.thresholds.empty()) {
        Json bands = Json::object();
        for (const ThresholdCount& band : stats.thresholds) {
            bands[std::string(1, band.threshold.comparison) + " " + format_general(band.threshold.value)] =
                statistic(band.percentage);
        }
        summary["percent_of_" + unit] = std::move(bands);
    }
    // As three lists rather than a list of objects, so that the CSV gets a
    // cell for each (flatten).
    Json frames = Json::array();
    Json times = Json::array();
    Json worst_scores = Json::array();
    for (const WorstFrame& worst : worst_frames(samples, definition.direction, summary_worst_frames)) {
        frames.push_back(worst.frame);
        times.push_back(json_number(worst.time));
        worst_scores.push_back(json_number(worst.score));
    }
    summary["worst_frames"] =
        Json{{"frame", std::move(frames)}, {"time", std::move(times)}, {"score", std::move(worst_scores)}};
}

void add_provenance(Json& summary, const MetricProvenance& provenance) {
    summary["computed_on"] = provenance.compute_backend;
    summary["implementation"] = provenance.implementation;
    summary["version"] = provenance.implementation_version;
}

// `value`'s cells for the CSV, each named by its place in the JSON. A list
// of objects (each display's CVVDP: cvvdp_displays) gives each item's
// values columns of their own, numbered as in the JSON
// ("cvvdp_displays.1.score"); any other list is one cell, its JSON.
void flatten(const Json& value, const std::string& name, std::vector<std::pair<std::string, std::string>>& cells) {
    if (value.is_object()) {
        for (const auto& [key, item] : value.items()) {
            flatten(item, name.empty() ? key : name + "." + key, cells);
        }
        return;
    }
    if (value.is_array() && !value.empty() &&
        std::all_of(value.begin(), value.end(), [](const Json& item) { return item.is_object(); })) {
        for (std::size_t i = 0; i < value.size(); ++i) {
            flatten(value[i], name + "." + std::to_string(i), cells);
        }
        return;
    }
    cells.emplace_back(name, value.is_string() ? value.get<std::string>() : value.is_null() ? "" : value.dump());
}

}  // namespace

Json json_number(const std::optional<double>& value) {
    if (!value || std::isnan(*value)) {
        return nullptr;
    }
    if (std::isinf(*value)) {
        return *value > 0 ? "inf" : "-inf";
    }
    return *value;
}

std::optional<double> summary_number(const Json& number) {
    if (number.is_string()) {
        return python_float(number.get<std::string>());  // "inf" and "-inf"
    }
    if (number.is_null()) {
        return std::nullopt;
    }
    return number.get<double>();
}

std::optional<Json> metric_summary(const ComparisonResult& result, const std::string& key) {
    const MetricDefinition& definition = metric_definition(key);
    if (const SequenceMetricResult* sequence = result.metric_results.sequence(key)) {
        // A whole-video score (CVVDP's) is of every frame compared: it is not
        // calculated on a subsample.
        Json summary{{"score", json_number(sequence->score)}, {"frames", result.compared_frame_count}};
        if (sequence->has_timeline()) {
            Json per_second = Json::object();
            const ScoreSamples seconds{sequence->frame, sequence->time, sequence->values, true};
            add_statistics(per_second, seconds, definition, "seconds");
            summary["per_second"] = std::move(per_second);
        }
        add_provenance(summary, sequence->provenance);
        return summary;
    }
    const FrameMetricResult* metric = result.metric_results.frame(key);
    if (metric == nullptr || metric->values.empty()) {
        return std::nullopt;
    }
    Json summary{{"score", json_number(metric->aggregate())}};
    add_statistics(summary, ScoreSamples{metric->frame, metric->time, metric->values, false}, definition, "frames");
    add_provenance(summary, metric->provenance);
    return summary;
}

std::optional<CvvdpSettings> cvvdp_display_of(const SequenceMetricResult& cvvdp) {
    try {
        nlohmann::json parameters = nlohmann::json::object();
        for (const auto& [name, text] : cvvdp.provenance.parameters) {
            parameters[name] = nlohmann::json::parse(text);
        }
        return CvvdpSettings::from_parameters(parameters);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
}

Json cvvdp_display_entry(const std::string& name, const ComparisonResult* result) {
    const SequenceMetricResult* cvvdp = result != nullptr ? result->metric_results.sequence("cvvdp") : nullptr;
    const std::optional<CvvdpSettings> display = cvvdp != nullptr ? cvvdp_display_of(*cvvdp) : std::nullopt;
    const std::optional<Json> summary = result != nullptr ? metric_summary(*result, "cvvdp") : std::nullopt;
    return Json{{"display", name},
                {"score", summary ? summary->at("score") : Json(nullptr)},
                {"display_parameters", display ? Json(display->parameters()) : Json(nullptr)},
                {"cvvdp", summary ? *summary : Json(nullptr)}};
}

bool scores_of_a_range(const ComparisonResult& result) {
    for (const std::string& key : result.metric_results.keys()) {
        const FrameMetricResult* frames = result.metric_results.frame(key);
        const MetricProvenance& provenance =
            frames != nullptr ? frames->provenance : result.metric_results.sequence(key)->provenance;
        if (provenance.parameters.contains("range_frames")) {
            return true;
        }
    }
    return false;
}

Json summary_crop_json(const std::optional<CropBox>& box) {
    if (!box) {
        return nullptr;
    }
    return Json{{"x", box->x}, {"y", box->y}, {"width", box->w}, {"height", box->h}};
}

namespace {

// Text, or null where there is none.
Json text_or_null(const std::string& text) { return text.empty() ? Json(nullptr) : Json(text); }

// The chroma subsampling an FFmpeg pixel format's name says: "4:2:0" for
// yuv420p10le, nv12 and p010le; null where it does not say.
Json chroma_subsampling(const std::string& pix_fmt) {
    if (pix_fmt.find("420") != std::string::npos || pix_fmt.starts_with("nv12") || pix_fmt.starts_with("p01")) {
        return "4:2:0";
    }
    if (pix_fmt.find("422") != std::string::npos) {
        return "4:2:2";
    }
    if (pix_fmt.find("444") != std::string::npos || pix_fmt.starts_with("gbr")) {
        return "4:4:4";
    }
    if (pix_fmt.starts_with("gray")) {
        return "4:0:0";
    }
    return nullptr;
}

// What the file is, as the probe found it: its size and bitrate, the
// stream's format and colour, and what wrote and encoded it (issue #14), so
// that a results file can be read against size on its own.
void add_file_context(Json& entry, const VideoInfo& info) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(info.path, error);
    entry["file_size"] = error ? Json(nullptr) : Json(size);
    entry["container"] = text_or_null(info.format_name);
    entry["bit_rate"] = info.bit_rate > 0 ? Json(info.bit_rate) : Json(nullptr);
    entry["bit_rate_of"] = info.bit_rate <= 0          ? Json(nullptr)
                           : info.bit_rate_whole_file ? Json("file")
                           : info.bit_rate_estimated  ? Json("estimate")
                                                      : Json("video");
    entry["stream_bytes"] = info.stream_bytes > 0 ? Json(info.stream_bytes) : Json(nullptr);
    entry["profile"] = text_or_null(info.profile);
    entry["level"] = info.level > 0 ? Json(info.level) : Json(nullptr);
    entry["bit_depth"] = info.pix_fmt.empty() ? Json(nullptr) : Json(bit_depth(info.pix_fmt));
    entry["chroma_subsampling"] = chroma_subsampling(info.pix_fmt);
    // As the app reads the video: its tags, or those given (--colour) or
    // taken from the other video where it has none
    entry["colour"] = Json{{"range", text_or_null(info.color_range)},
                           {"matrix", text_or_null(info.color_space)},
                           {"transfer", text_or_null(info.color_transfer)},
                           {"primaries", text_or_null(info.color_primaries)},
                           {"chroma_location", text_or_null(info.chroma_location)}};
    entry["writing_library"] = text_or_null(info.writing_library);
    entry["encoder"] = text_or_null(info.encoder);
    entry["encoder_settings"] = text_or_null(info.encoder_settings);
}

}  // namespace

void add_media(Json& entry, const VideoInfo& info) {
    entry["width"] = info.width;
    entry["height"] = info.height;
    entry["fps"] = info.fps;
    entry["codec"] = info.codec_name;
    entry["pixel_format"] = info.pix_fmt;
    entry["duration"] = info.duration;
    add_file_context(entry, info);
    // Its film grain where its stream has some: the kinds, and how it was
    // scored (and why without, where that was no choice).
    const FilmGrainScoring grain = effective_film_grain(info);
    if (grain.scoring != GrainScoring::none) {
        Json film_grain{{"kinds", film_grain_names(grain.kinds)},
                        {"scored", grain.scoring == GrainScoring::with ? "with" : "without"}};
        if (!grain.why_without.empty()) {
            film_grain["why_without"] = grain.why_without;
        }
        entry["film_grain"] = film_grain;
    }
}

std::string results_csv(const Json& document) {
    std::vector<std::string> columns;
    std::map<std::string, std::size_t> column_of;
    std::vector<std::map<std::size_t, std::string>> rows;
    for (const Json& video : document.at("videos")) {
        std::vector<std::pair<std::string, std::string>> cells;
        flatten(video, "", cells);
        std::map<std::size_t, std::string>& row = rows.emplace_back();
        for (auto& [name, text] : cells) {
            const auto [found, added] = column_of.try_emplace(name, columns.size());
            if (added) {
                columns.push_back(name);
            }
            row[found->second] = std::move(text);
        }
    }
    const auto line = [](const auto& texts) {
        std::string text;
        for (std::size_t i = 0; i < texts.size(); ++i) {
            text += (i == 0 ? "" : ",") + csv_cell(texts[i]);
        }
        return text + "\r\n";
    };
    std::string csv = "\xEF\xBB\xBF" + line(columns);
    for (const auto& row : rows) {
        std::vector<std::string> texts(columns.size());
        for (const auto& [column, text] : row) {
            texts[column] = text;
        }
        csv += line(texts);
    }
    return csv;
}

}  // namespace vml
