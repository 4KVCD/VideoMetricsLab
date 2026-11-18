#include "results_export.h"

#include <QAction>
#include <QMenu>

#include <algorithm>
#include <fstream>
#include <set>

#include "dialogs.h"
#include "i18n.h"
#include "window_helpers.h"
#include "vml/build_info.h"
#include "vml/frame_pairing.h"
#include "vml/metrics.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/results_summary.h"

namespace vml::app {

namespace {

using Json = nlohmann::ordered_json;

// A run's entry, as the command line's --json writes a video's.
Json run_entry(const ExportedRun& run) {
    const ComparisonResult& result = *run.result;
    Json entry{{"path", platform::path_text(result.distorted)},
               {"reference", platform::path_text(result.source)},
               {"label", run.label.toStdString()},
               {"status", run.failures.empty() ? "ok" : "partial"}};
    add_media(entry, result.distorted_info);
    if (!run.failures.empty()) {
        Json failed = Json::object();
        for (const auto& [key, why] : run.failures) {
            failed[key] = why.toStdString();
        }
        entry["failed_metrics"] = failed;
    }
    entry["compared_frames"] = result.compared_frame_count;
    entry["frame_sync"] = frame_pairing_json(result);
    entry["vmaf_model"] = result.model;
    entry["reference_crop"] = summary_crop_json(result.source_crop);
    entry["test_crop"] = summary_crop_json(result.distorted_crop);
    Json scored = Json::object();  // in the app's order
    for (const MetricDefinition& metric : metrics()) {
        const std::string key(metric.key);
        if (std::optional<Json> summary = metric_summary(result, key)) {
            scored[key] = std::move(*summary);
        }
    }
    entry["metrics"] = scored;
    if (!run.cvvdp_display.empty()) {
        entry["cvvdp_display"] = run.cvvdp_display;
    }
    const bool has_cvvdp = result.metric_results.sequence("cvvdp") != nullptr;
    if (has_cvvdp || !run.other_displays.empty()) {
        // Every display CVVDP was scored on, the first (metrics.cvvdp's) too:
        // a display that is no preset by its parameters alone (name "").
        Json displays = Json::array();
        if (has_cvvdp) {
            displays.push_back(cvvdp_display_entry(run.cvvdp_display, &result));
        }
        for (const ExportedDisplay& other : run.other_displays) {
            Json display = cvvdp_display_entry(other.name, other.result.get());
            // Its own run's times: the first display's are the video's
            Json record;
            add_video_record(record, other.facts.get(), nullptr, other.name);
            display["timings"] = record["timings"];
            displays.push_back(std::move(display));
        }
        entry["cvvdp_displays"] = displays;
    }
    add_video_record(entry, run.facts.get(), run.result.get(), run.cvvdp_display);
    return entry;
}

// The window's runs the exported videos came from, as one: the first's
// start, the last's end (none while one is still running), their times
// added up, and the latest finished one's machine (`machine` without one).
RunRecord runs_record(const std::vector<ExportedRun>& runs, const MachineFacts& machine) {
    std::set<const RunRecord*> records;
    for (const ExportedRun& run : runs) {
        if (run.record) {
            records.insert(run.record.get());
        }
    }
    RunRecord all{"gui", std::nullopt, "", "", std::nullopt, std::nullopt, static_cast<int>(records.size()),
                  machine};
    if (records.empty()) {
        return all;
    }
    const bool running = std::any_of(records.begin(), records.end(),
                                     [](const RunRecord* record) { return record->finished_utc.empty(); });
    double wall = 0.0;
    double paused = 0.0;
    std::string latest;  // the latest finished run's end
    for (const RunRecord* record : records) {
        // utc_text's times sort as text
        if (all.started_utc.empty() || record->started_utc < all.started_utc) {
            all.started_utc = record->started_utc;
        }
        if (!record->finished_utc.empty() && record->finished_utc > latest) {
            latest = record->finished_utc;
            all.machine = record->machine;
        }
        wall += record->wall_seconds.value_or(0.0);
        paused += record->paused_seconds.value_or(0.0);
    }
    if (!running) {
        all.finished_utc = latest;
        all.wall_seconds = wall;
        all.paused_seconds = paused;
    }
    return all;
}

}  // namespace

Json export_document(const std::optional<VideoInfo>& reference, const std::vector<ExportedRun>& runs,
                     const MachineFacts& machine) {
    Json source = nullptr;
    if (reference) {
        source = Json{{"path", platform::path_text(reference->path)}};
        add_media(source, *reference);
    }
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_VERSION}, {"commit", build_commit()}, {"format", results_format},
                  {"reference", source}, {"videos", Json::array()}};
    add_run_record(document, runs_record(runs, machine));
    for (const ExportedRun& run : runs) {
        document["videos"].push_back(run_entry(run));
    }
    return document;
}

ExportActions add_export_actions(QMenu& menu, bool enabled) {
    ExportActions actions{menu.addAction(app::tr("Export results as CSV...")),
                          menu.addAction(app::tr("Export results as JSON..."))};
    actions.csv->setEnabled(enabled);
    actions.json->setEnabled(enabled);
    return actions;
}

std::optional<ResultsFormat> chosen_format(const ExportActions& actions, const QAction* chosen) {
    if (chosen != nullptr && chosen == actions.csv) {
        return ResultsFormat::csv;
    }
    if (chosen != nullptr && chosen == actions.json) {
        return ResultsFormat::json;
    }
    return std::nullopt;
}

void save_results(const Dialogs& dialogs, const Json& document, const QString& name, ResultsFormat format) {
    const bool csv = format == ResultsFormat::csv;
    const QString path = dialogs.save_file(app::tr("Export results"), name + (csv ? ".csv" : ".json"),
                                           csv ? app::tr("CSV file (*.csv)") : app::tr("JSON file (*.json)"));
    if (path.isEmpty()) {
        return;
    }
    // The CSV ends its lines with CR LF itself; the JSON is written as the
    // command line writes it, a text file (CR LF on Windows).
    std::ofstream file(path_of(path), csv ? std::ios::binary : std::ios::out);
    file << (csv ? results_csv(document) : python_json_dumps(document) + "\n");
    file.close();
    if (!file) {
        dialogs.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

}  // namespace vml::app
