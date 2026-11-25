#include "vml/score_summary.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <variant>

#include "vml/build_info.h"
#include "vml/cvvdp.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/results_summary.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// The worst frames' tails, as ScoreStats has them.
struct Tail {
    const char* key;
    const char* share;
    double ScoreStats::*value;
};
constexpr Tail tails[] = {{"worst_10", "10%", &ScoreStats::worst_10_percent},
                          {"worst_5", "5%", &ScoreStats::worst_5_percent},
                          {"worst_1", "1%", &ScoreStats::worst_1_percent},
                          {"worst_0_1", "0.1%", &ScoreStats::worst_0_1_percent}};

std::optional<double ScoreStats::*> stat_of(const std::string& key) {
    if (key == "median") {
        return &ScoreStats::median;
    }
    if (key == "stdev") {
        return &ScoreStats::stdev;
    }
    if (key == "min") {
        return &ScoreStats::minimum;
    }
    if (key == "max") {
        return &ScoreStats::maximum;
    }
    for (const Tail& tail : tails) {
        if (key == tail.key) {
            return tail.value;
        }
    }
    return std::nullopt;
}

// The columns asked for, known ones only, in their own order.
std::vector<std::string> known_columns(const std::vector<std::string>& columns) {
    std::vector<std::string> shown;
    for (const std::string& key : score_summary_column_keys()) {
        if (std::find(columns.begin(), columns.end(), key) != columns.end()) {
            shown.push_back(key);
        }
    }
    return shown;
}

// A band as Metric Graphs heads it ("> 90"), and its share as it shows it.
std::string band_label(const ThresholdCount& band) {
    return std::string(1, band.threshold.comparison) + " " + format_general(band.threshold.value);
}

// The time, in UTC, as ISO 8601 ("2026-11-25T14:02:11Z").
std::string iso_time(std::chrono::system_clock::time_point when) {
    const std::time_t seconds = std::chrono::system_clock::to_time_t(when);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    std::ostringstream text;
    text << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return text.str();
}

std::string short_label(const std::string& key) {
    const MetricDefinition* metric = find_metric(key);
    return metric != nullptr ? std::string(metric->short_label) : key;
}

// A cell of the CSV: quoted where it has a comma, a quote or a line break.
std::string csv_quoted(const std::string& text) {
    if (text.find_first_of(",\"\n\r") == std::string::npos) {
        return text;
    }
    std::string quoted = "\"";
    for (const char c : text) {
        quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    }
    return quoted + "\"";
}

}  // namespace

const std::vector<std::string>& score_summary_column_keys() {
    static const std::vector<std::string> keys{"median",  "stdev",   "min",       "max",   "worst_10",
                                               "worst_5", "worst_1", "worst_0_1", "bands", "model",
                                               "made",    "device",  "saved"};
    return keys;
}

std::string score_summary_heading(const std::string& key) {
    for (const Tail& tail : tails) {
        if (key == tail.key) {
            return std::string("Worst ") + tail.share;
        }
    }
    return key == "median"   ? "Median"
           : key == "stdev"  ? "StDev"
           : key == "min"    ? "Min"
           : key == "max"    ? "Max"
           : key == "bands"  ? "Score bands"
           : key == "model"  ? "Model or display"
           : key == "made"   ? "Made with"
           : key == "device" ? "CPU or GPU"
           : key == "saved"  ? "Saved"
                             : key;
}

std::vector<ScoreSummaryRow> score_summary(const ComparisonResult& result, const std::string& presets_json,
                                       const std::map<std::string, std::chrono::system_clock::time_point>& saved) {
    std::vector<ScoreSummaryRow> details;
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        const FrameMetricResult* frames = result.metric_results.frame(key);
        const SequenceMetricResult* sequence = frames == nullptr ? result.metric_results.sequence(key) : nullptr;
        if (frames == nullptr && sequence == nullptr) {
            continue;  // not scored, or a plane, which is in its metric's file
        }
        ScoreSummaryRow detail;
        detail.key = key;
        const MetricProvenance& provenance = frames != nullptr ? frames->provenance : sequence->provenance;
        const std::vector<float>& values = frames != nullptr ? frames->values : sequence->values;
        detail.score = frames != nullptr ? frames->aggregate() : std::optional<double>(sequence->score);
        if (!values.empty()) {
            const std::vector<double> scores(values.begin(), values.end());
            detail.stats = compute_stats(scores, metric.thresholds, metric.aggregation, metric.direction);
        }
        detail.device = provenance.compute_backend == "gpu" ? "gpu" : "cpu";
        std::string made = provenance.implementation + " " + provenance.implementation_version;
        while (!made.empty() && made.back() == ' ') {
            made.pop_back();
        }
        detail.made = made;
        if (const auto model = provenance.parameters.find("model"); model != provenance.parameters.end()) {
            std::string value = model->second;
            value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
            detail.model = model_shown(value);
        }
        if (sequence != nullptr) {
            if (const std::optional<CvvdpSettings> display = cvvdp_display_of(*sequence)) {
                const std::optional<CvvdpPreset> preset = matching_cvvdp_preset(*display, presets_json, "");
                detail.model = preset ? preset->name : "a display of its own";
            }
        }
        if (const auto when = saved.find(key); when != saved.end()) {
            detail.saved = when->second;
        }
        details.push_back(std::move(detail));
    }
    return details;
}

nlohmann::ordered_json score_summary_json(const std::vector<ScoreSummaryEntry>& entries,
                                          const std::vector<std::string>& columns) {
    const std::vector<std::string> shown = known_columns(columns);
    Json document{{"app", "VideoMetricsLab"}, {"url", project_url}, {"version", VML_VERSION},
                  {"commit", build_commit()}, {"format", score_summary_format}, {"columns", shown},
                  {"comparisons", Json::array()}};
    for (const ScoreSummaryEntry& entry : entries) {
        Json scores = Json::array();
        for (const ScoreSummaryRow& detail : entry.scores) {
            Json row{{"metric", detail.key}, {"label", short_label(detail.key)}, {"score", json_number(detail.score)}};
            for (const std::string& key : shown) {
                if (const std::optional<double ScoreStats::*> stat = stat_of(key)) {
                    row[key] = detail.stats && detail.stats->count > 0 ? json_number((*detail.stats).*(*stat))
                                                                       : Json(nullptr);
                } else if (key == "bands") {
                    Json bands = Json::array();
                    if (detail.stats && detail.stats->count > 0) {
                        for (const ThresholdCount& band : detail.stats->thresholds) {
                            bands.push_back({{"band", band_label(band)}, {"share", band.percentage}});
                        }
                    }
                    row[key] = bands;
                } else if (key == "model") {
                    row[key] = detail.model.empty() ? Json(nullptr) : Json(detail.model);
                } else if (key == "made") {
                    row[key] = detail.made;
                } else if (key == "device") {
                    row[key] = detail.device;
                } else if (key == "saved") {
                    row[key] = detail.saved ? Json(iso_time(*detail.saved)) : Json(nullptr);
                }
            }
            scores.push_back(std::move(row));
        }
        document["comparisons"].push_back({{"label", entry.label},
                                           {"test", platform::path_text(entry.test)},
                                           {"reference", platform::path_text(entry.reference)},
                                           {"scores", std::move(scores)}});
    }
    return document;
}

std::string score_summary_csv(const std::vector<ScoreSummaryEntry>& entries, const std::vector<std::string>& columns) {
    const std::vector<std::string> shown = known_columns(columns);
    std::string csv = "Comparison,Metric,Score";
    for (const std::string& key : shown) {
        csv += "," + csv_quoted(score_summary_heading(key));
    }
    csv += "\n";
    for (const ScoreSummaryEntry& entry : entries) {
        for (const ScoreSummaryRow& detail : entry.scores) {
            const MetricDefinition* metric = find_metric(detail.key);
            const auto number = [metric](std::optional<double> value) {
                return metric != nullptr ? metric->format_value(value) : (value ? format_general(*value) : "");
            };
            std::vector<std::string> cells{entry.label, short_label(detail.key), number(detail.score)};
            for (const std::string& key : shown) {
                if (const std::optional<double ScoreStats::*> stat = stat_of(key)) {
                    cells.push_back(detail.stats && detail.stats->count > 0
                                        ? number((*detail.stats).*(*stat))
                                        : std::string());
                } else if (key == "bands") {
                    std::string bands;
                    if (detail.stats && detail.stats->count > 0) {
                        for (const ThresholdCount& band : detail.stats->thresholds) {
                            bands += (bands.empty() ? "" : "; ") + band_label(band) + ": " +
                                     format_fixed(band.percentage, 1) + "%";
                        }
                    }
                    cells.push_back(bands);
                } else if (key == "model") {
                    cells.push_back(detail.model);
                } else if (key == "made") {
                    cells.push_back(detail.made);
                } else if (key == "device") {
                    cells.push_back(detail.device == "gpu" ? "GPU" : "CPU");
                } else if (key == "saved") {
                    cells.push_back(detail.saved ? iso_time(*detail.saved) : std::string());
                }
            }
            std::string line;
            for (const std::string& cell : cells) {
                line += (line.empty() ? "" : ",") + csv_quoted(cell);
            }
            csv += line + "\n";
        }
    }
    return csv;
}

}  // namespace vml
