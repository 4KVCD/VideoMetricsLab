// Export results (issue #8), from the Videos tab's and Metric Graphs'
// right-click menus alike: the same file, of every run the window shows, with
// each metric's score and the statistics Metric Graphs shows, in the shape
// the command line's --json writes (results_summary.h): as JSON, or as CSV
// with a row for each video. The scores, not the frames (the per-frame
// scores export has those).
#pragma once

#include <QString>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "graph_bundle.h"
#include "vml/results.h"
#include "vml/run_record.h"
#include "vml/settings.h"
#include "vml/video.h"
#include "vml/weak_frames.h"

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

// What an export includes (Settings > Storage > Exports include), as the
// right-click menus say it: the kinds of `items` (exported_items), in a
// list: "results, graphs and weak segments". Empty when nothing is ticked.
QString export_contents(const std::vector<std::string>& items);

// The right-click menus' exports, named after what they include: of the
// video clicked, of all the videos (the Videos tab's run), or of the ones
// selected (Saved results). With nothing ticked, says where to tick it.
enum class ExportScope { this_video, all_videos, selected_videos };
QString export_action_text(ExportScope scope, const std::vector<std::string>& items);

// CSV text as the command line writes it: CR LF on Windows.
std::string text_lines(const std::string& text);

// A video's per-frame scores in an export ("<stem>.json", "<stem>.csv"),
// made only as they are written (made_files), on the file-write queue: a
// 2-hour video's take most of a second, which the window's thread would
// wait, and an export cancelled in its save dialog too.
struct FrameFiles {
    std::shared_ptr<const ComparisonResult> result;
    std::string stem;
    bool json = false;
    bool csv = false;
};
// `files`, then each of `frames`' files, made now.
std::vector<BundleFile> made_files(std::vector<BundleFile> files, const std::vector<FrameFiles>& frames);

// The files of an export besides its graphs, as Settings' Exports include
// (exported_items) and its formats say, named after `name`: the results
// (`document`, made only when they're included), and the weak segments by
// `rules` of the one video, or (`several`) a report of all of `videos`; each
// video's per-frame scores ("<video> - frames", named as its graphs are) in
// `frames`, made later. Weak segments need a condition in `rules`: without
// one there are none, and `segments_skipped` is set.
std::vector<BundleFile> export_files(const Settings& settings, const std::vector<BundleVideo>& videos,
                                     const std::filesystem::path& reference,
                                     const std::function<nlohmann::ordered_json()>& document,
                                     const SegmentRules& rules, bool several, const QString& name,
                                     std::vector<FrameFiles>& frames, bool& segments_skipped);

}  // namespace vml::app
