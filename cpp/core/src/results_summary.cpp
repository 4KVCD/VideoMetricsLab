#include "vml/results_summary.h"

#include <cmath>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "vml/metrics.h"
#include "vml/number_format.h"
#include "vml/python_text.h"
#include "vml/stats.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// The statistics of `values`, a metric's frames or CVVDP's seconds (`unit`),
// added to `summary`: how many have a score, then the rest, each null where
// none has.
void add_statistics(Json& summary, std::span<const float> values, const MetricDefinition& definition,
                    const std::string& unit) {
    const std::vector<double> scores(values.begin(), values.end());
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
}

void add_provenance(Json& summary, const MetricProvenance& provenance) {
    summary["computed_on"] = provenance.compute_backend;
    summary["implementation"] = provenance.implementation;
    summary["version"] = provenance.implementation_version;
}

// `value`'s cells for the CSV, each named by its place in the JSON; an
// array is one cell, its JSON.
void flatten(const Json& value, const std::string& name, std::vector<std::pair<std::string, std::string>>& cells) {
    if (value.is_object()) {
        for (const auto& [key, item] : value.items()) {
            flatten(item, name.empty() ? key : name + "." + key, cells);
        }
        return;
    }
    cells.emplace_back(name, value.is_string() ? value.get<std::string>() : value.is_null() ? "" : value.dump());
}

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
            add_statistics(per_second, sequence->values, definition, "seconds");
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
    add_statistics(summary, metric->values, definition, "frames");
    add_provenance(summary, metric->provenance);
    return summary;
}

Json crop_json(const std::optional<CropBox>& box) {
    if (!box) {
        return nullptr;
    }
    return Json{{"x", box->x}, {"y", box->y}, {"width", box->w}, {"height", box->h}};
}

void add_media(Json& entry, const VideoInfo& info) {
    entry["width"] = info.width;
    entry["height"] = info.height;
    entry["fps"] = info.fps;
    entry["codec"] = info.codec_name;
    entry["pixel_format"] = info.pix_fmt;
    entry["duration"] = info.duration;
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
