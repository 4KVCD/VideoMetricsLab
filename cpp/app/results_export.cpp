#include "results_export.h"

#include <QAction>
#include <QMenu>

#include <fstream>

#include "dialogs.h"
#include "i18n.h"
#include "window_helpers.h"
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
    entry["vmaf_model"] = result.model;
    entry["reference_crop"] = crop_json(result.source_crop);
    entry["test_crop"] = crop_json(result.distorted_crop);
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
    return entry;
}

}  // namespace

Json export_document(const std::optional<VideoInfo>& reference, const std::vector<ExportedRun>& runs) {
    Json source = nullptr;
    if (reference) {
        source = Json{{"path", platform::path_text(reference->path)}};
        add_media(source, *reference);
    }
    Json document{{"app", "VideoMetricsLab"}, {"version", VML_APP_VERSION}, {"format", results_format},
                  {"reference", source}, {"videos", Json::array()}};
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
