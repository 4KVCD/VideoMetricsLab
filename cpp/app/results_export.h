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
// CVVDP score is for (none without one), CVVDP on further displays, how
// it was made: its run's facts and the run's record (none where unknown),
// and for a row that scores another row's file another way (a film grain
// twin, a "Test both" companion, a further CVVDP display), that file.
struct ExportedRun {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::map<std::string, QString> failures{};
    std::string cvvdp_display{};
    std::vector<ExportedDisplay> other_displays{};
    std::shared_ptr<const RunFacts> facts{};
    std::shared_ptr<const RunRecord> record{};
    std::optional<std::filesystem::path> variant_of{};
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

// Whether a row's run is one to export (MainWindow::scored_video): each of
// the metrics `requested` (ticked) has its score from the run (`kept`) or
// failed on that run (`failed`), and at least one has its score. A metric
// ticked since the run, never calculated, holds it back; one that failed
// does not, or the scores beside it could never be exported, nor by
// Settings' after-run export.
bool settled_for_export(const std::vector<std::string>& requested, const std::vector<std::string>& kept,
                        const std::vector<std::string>& failed);

// An export's files written into `folder` (one the user chose, or Settings'
// after-run export's), replacing none there: where any of their names is
// taken, by another video's export of the same file name say, they are all
// numbered alike (unique_output_names). The paths written. Throws
// std::runtime_error on a file that could not be written.
std::vector<std::filesystem::path> write_loose_export(const std::filesystem::path& folder,
                                                      const std::vector<BundleFile>& files);

// The files of an archive of several videos laid out: each video's own
// files, named "<name> - ..." (`names`, bundle_names': its graphs, per-frame
// scores and worst frames), in a folder of its name ("<name>/..."); the files
// of all of them together where they are: those named "<stem>.<extension>"
// for a stem of `shared` (the results, the weak segments report: a test
// video may be named as the reference they are named after), and the rest
// named after no video ("All test videos - ..."). Names are matched longest
// first, so that a video "a - b" keeps its files from a video "a". No
// `names`: as they are (one video's export).
std::vector<BundleFile> in_video_folders(std::vector<BundleFile> files, const std::vector<QString>& names,
                                         const std::vector<std::string>& shared);

// The stems of an archive's files of all its videos together, named after
// `name` (export_files'): its results and its weak segments report.
std::vector<std::string> shared_stems(const QString& name);

// CSV text as the command line writes it: CR LF on Windows.
std::string text_lines(const std::string& text);

// A video's files of its frames in an export, made only as they are written
// (made_files), on the file-write queue: a 2-hour video's per-frame scores
// take most of a second, and its worst frames a tenth, which the window's
// thread would wait, and an export cancelled in its save dialog too.
// - Per-frame scores (worst_dips 0): "<stem>.json", "<stem>.csv", added
//   after the other files.
// - Worst frames of every metric, of `worst_dips` dips each: made into the
//   files listed for them where they are (json_at, csv_at), so that the
//   export's files keep their order.
struct FrameFiles {
    std::shared_ptr<const ComparisonResult> result;
    std::string stem;
    bool json = false;
    bool csv = false;
    std::size_t worst_dips = 0;
    std::size_t json_at = 0;
    std::size_t csv_at = 0;
};
// `files`, with each of `frames`' files made now.
std::vector<BundleFile> made_files(std::vector<BundleFile> files, const std::vector<FrameFiles>& frames);

// The files of an export besides its graphs, as Settings' Exports include
// (exported_items) and its formats say, named after `name`: the results
// (`document`, made only when they're included), each video's worst frames
// of every metric ("<video> - worst frames", Settings' graph_worst_frames
// dips each: listed empty, and in `frames` to be made later), and the weak
// segments by `rules` of the one video, or (`several`) a report of all of
// `videos`; each video's per-frame scores ("<video> - frames", named as its
// graphs are) in `frames`, made later. Weak segments need a condition in
// `rules`: without one there are none, and `segments_skipped` is set.
std::vector<BundleFile> export_files(const Settings& settings, const std::vector<BundleVideo>& videos,
                                     const std::filesystem::path& reference,
                                     const std::function<nlohmann::ordered_json()>& document,
                                     const SegmentRules& rules, bool several, const QString& name,
                                     std::vector<FrameFiles>& frames, bool& segments_skipped);

}  // namespace vml::app
