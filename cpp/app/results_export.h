// Export results (issue #8), from the Videos tab's and Metric Graphs'
// right-click menus alike: the same file, of every run the window shows, with
// each metric's score and the statistics Metric Graphs shows, in the shape
// the command line's --json writes (results_summary.h): as JSON, or as CSV
// with a row for each video. The scores, not the frames (Metric Graphs'
// Export CSV has those).
#pragma once

#include <QString>

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vml/results.h"
#include "vml/video.h"

class QAction;
class QMenu;

namespace vml::app {

struct Dialogs;

enum class ResultsFormat { csv, json };

// A run to export: its result, the name the window gives it, why each
// metric that failed on its last run did, and the preset of the display its
// CVVDP score is for (none without one).
struct ExportedRun {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::map<std::string, QString> failures{};
    std::string cvvdp_display{};
};

// The runs as one document: the reference the Videos tab compares with
// (none without one), and an entry for each run, which names its own.
nlohmann::ordered_json export_document(const std::optional<VideoInfo>& reference,
                                       const std::vector<ExportedRun>& runs);

// The menus' two entries, CSV and JSON, added to `menu`.
struct ExportActions {
    QAction* csv = nullptr;
    QAction* json = nullptr;
};
ExportActions add_export_actions(QMenu& menu, bool enabled);
// The format of the entry `chosen`, or nothing for another entry.
std::optional<ResultsFormat> chosen_format(const ExportActions& actions, const QAction* chosen);

// Asks where to save `document`, suggesting `name` with the format's
// extension, and saves it there; says why when it cannot.
void save_results(const Dialogs& dialogs, const nlohmann::ordered_json& document, const QString& name,
                  ResultsFormat format);

}  // namespace vml::app
