// Score summary: a comparison's scores, one row a metric, with the statistics
// of its frames and how each score was made, the columns Saved Results'
// details panel shows (Settings::saved_details_columns). A leaner file than
// the results: the scores alone, as a table. The window saves it (the
// details panel's save icon, the export menus, Settings' Exports include)
// and the command line writes it (--score-summary), each the same way.
#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vml/results.h"
#include "vml/stats.h"

namespace vml {

// The project's home, in the JSON files that say which program wrote them.
inline constexpr const char* project_url = "https://github.com/4KVCD/VideoMetricsLab";

// score-summary-1: this file's format, in its JSON's "format".
inline constexpr const char* score_summary_format = "score-summary-1";

// The columns a score can have after its metric and score, as Settings keeps
// them, in the order shown: the statistics of its frames, as Metric Graphs'
// table has them ("median", "stdev", "min", "max", "worst_10", "worst_5",
// "worst_1", "worst_0_1", and "bands": the share in each of the metric's
// score bands), then how it was made ("model", "made", "device"), then when
// it was saved ("saved").
const std::vector<std::string>& score_summary_column_keys();
// A column's heading, in English (the CSV's, and the window's before it is
// translated).
std::string score_summary_heading(const std::string& key);

// One metric's score of a comparison.
struct ScoreSummaryRow {
    std::string key;
    std::optional<double> score;   // the whole video's, by the metric's own way of averaging
    std::optional<ScoreStats> stats;  // of its frames (seconds for CVVDP); none where it has none
    std::string device;            // "cpu" or "gpu"
    std::string made;              // the implementation and its version
    std::string model;             // VMAF's model, CVVDP's display (a preset's name); empty for none
    std::optional<std::chrono::system_clock::time_point> saved;  // where it is known
};

// `result`'s scores, in the metrics' order; `presets_json` names CVVDP's
// displays (Settings::cvvdp_presets_json); `saved` gives each metric's
// saved time where known (Saved Results'), by its key.
std::vector<ScoreSummaryRow> score_summary(
    const ComparisonResult& result, const std::string& presets_json,
    const std::map<std::string, std::chrono::system_clock::time_point>& saved = {});

// A comparison's scores, as a file holds them.
struct ScoreSummaryEntry {
    std::string label;  // the test video's name as shown
    std::filesystem::path test;
    std::filesystem::path reference;
    std::vector<ScoreSummaryRow> scores;
};

// The file: {app, url, version, commit, format, columns, comparisons: [{label,
// test, reference, scores: [{metric, label, score, <columns>}]}]}, with
// `columns` (unknown keys left out, in score_summary_column_keys' order).
nlohmann::ordered_json score_summary_json(const std::vector<ScoreSummaryEntry>& entries,
                                          const std::vector<std::string>& columns);
// The same as CSV: a row a score, headed Comparison, Metric, Score and the
// columns' headings; numbers at their metric's precision.
std::string score_summary_csv(const std::vector<ScoreSummaryEntry>& entries, const std::vector<std::string>& columns);

}  // namespace vml
