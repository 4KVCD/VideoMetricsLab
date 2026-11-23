// A set of videos' graphs exported at once (the Videos tab's Export >
// Graphs as ZIP, a video's ZIP from its right-click menu, Settings'
// automatic export): each test video's Metric Graphs page of every metric
// it has, all of them together when there are several, and their bitrate
// with the reference's.
// They are drawn in panels of their own that are never shown, so what the
// tabs show (the lines ticked, the page, the zoom) is left as it is.
#pragma once

#include <QColor>
#include <QSize>
#include <QString>

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "theme.h"
#include "vml/bitrate_store.h"
#include "vml/results.h"

namespace vml::app {

class BitratePanel;
class VideoColors;

// A scored test video: its result, its name in the window and its file;
// and what its bitrate is known by in an export where that is not its file
// (saved_bitrates).
struct BundleVideo {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::filesystem::path path;
    std::filesystem::path bitrate_key{};

    const std::filesystem::path& bitrate_path() const { return bitrate_key.empty() ? path : bitrate_key; }
};

// A file for the archive: its name there and its bytes.
using BundleFile = std::pair<std::string, std::string>;

// The bitrate an export draws for a file in place of what the Bitrate tab
// reads at its path: a saved comparison's file as its bitrate was saved
// (bitrate_store.h), where the file is gone or another is there now (a new
// encode under the same name). None where none was saved: nothing is drawn
// for the file then. The tab, which knows a file by its path, is not asked.
using SavedBitrates = std::map<std::filesystem::path, std::optional<SavedBitrate>>;

// Where a saved comparison's file is not the one its scores were saved for
// (gone, or another file there now: a new encode under the same name), the
// bitrate its graphs draw for it: as it was saved with them. `videos` are
// the comparisons saved in `folders` (one each), of the reference
// `reference` (the first's). `own` tells whether the file at a path is the
// one the comparison in a folder was saved for (its reference, or its test
// video); `load` reads what was saved of it. Each test video so drawn is
// given a bitrate_key of its own: another of the videos may be of the file
// now at its path, or of another one saved under it, each with its own
// bitrate (issue #134).
using OwnFile = std::function<bool(const std::filesystem::path& folder, bool reference,
                                   const std::filesystem::path& file)>;
using LoadSavedBitrate = std::function<std::optional<SavedBitrate>(
    const std::filesystem::path& folder, bool reference, const std::filesystem::path& file)>;
SavedBitrates saved_bitrates(std::vector<BundleVideo>& videos, const std::vector<std::filesystem::path>& folders,
                             const std::filesystem::path& reference, const OwnFile& own,
                             const LoadSavedBitrate& load);

// How the graphs are drawn: the session's video colours, the chart colours
// (Settings' Exported graphs) and the chart's size.
struct BundleStyle {
    VideoColors* colors = nullptr;
    theme::ChartColours colours;
    QSize size{1600, 600};
};

// The videos' names for their files, in their order: each its label; where
// several share one, each with its folder's name after it in brackets; where
// those are shared too, numbered after that ("name (folder) 2"). Names are
// shared as file names are: once file_safe, as the system compares them.
std::vector<QString> bundle_names(const std::vector<BundleVideo>& videos);

// `text` fit for a file name: the characters Windows does not allow in one
// replaced with "_".
QString file_safe(const QString& text);

// Metric Graphs' PNG files: for each video, every metric it has
// ("<name> - <metric>.png"); with `together` and more than one video, every
// metric any of them has, with all of them ("All test videos - <metric>.png").
std::vector<BundleFile> metric_graph_files(const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                           bool together);

// The files whose bitrate an export draws, for the Bitrate tab to read: the
// reference, then each video with a file of its own (not a resolution
// test's). Each is its result's file as probed, at the export's `reference`
// and the video's `path`: where the file is now, not where it was when its
// saved scores were made (a file moved, renamed or copied since). Those of
// `saved` (by bitrate_path) are not read: their saved bitrate is drawn.
std::vector<VideoInfo> bitrate_files(const std::filesystem::path& reference, const std::vector<BundleVideo>& videos,
                                     const SavedBitrates& saved = {});

// The bitrate PNG files, from what `read` (the Bitrate tab) has read: with
// more than one video, the reference with all of them ("All test videos -
// Bitrate.png") and each with the reference ("<name> - Bitrate.png"); with
// one, that one. A file of `saved` is drawn as saved instead. A file `read`
// has not read (or `saved` has no bitrate of) is left out of the graphs, and
// its name added to `missing`.
std::vector<BundleFile> bitrate_graph_files(const BitratePanel& read, const std::filesystem::path& reference,
                                            const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                            std::vector<QString>& missing, const SavedBitrates& saved = {});

}  // namespace vml::app
