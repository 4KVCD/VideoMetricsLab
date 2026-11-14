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
#include "vml/run_record.h"
#include "vml/video.h"

class QAction;
class QMenu;

namespace vml::app {

struct Dialogs;

enum class ResultsFormat { csv, json };

// CVVDP on a further display of a run: the display's preset, and the result
// of the row that scored CVVDP on it (a display companion's), with how it
// was made (its run's facts).
struct ExportedDisplay {
    std::string name;
    std::shared_ptr<const ComparisonResult> result;
    std::shared_ptr<const RunFacts> facts{};
};

// A run to export: its result, the name the window gives it, why each
// metric that failed on its last run did, the preset of the display its
// CVVDP score is for (none without one), CVVDP on further displays, and how
// it was made: its run's facts and the run's record (none where unknown).
struct ExportedRun {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::map<std::string, QString> failures{};
    std::string cvvdp_display{};
    std::vector<ExportedDisplay> other_displays{};
    std::shared_ptr<const RunFacts> facts{};
    std::shared_ptr<const RunRecord> record{};
};

// The runs as one document: the reference the Videos tab compares with
// (none without one), the window's runs they came from with the machine
// (the latest run's, or `machine` without one), and an entry for each run,
// which names its own and records how it was made (run_record.h).
nlohmann::ordered_json export_document(const std::optional<VideoInfo>& reference,
                                       const std::vector<ExportedRun>& runs, const MachineFacts& machine);

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
