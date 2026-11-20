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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "theme.h"
#include "vml/results.h"

namespace vml::app {

class BitratePanel;
class VideoColors;

// A scored test video: its result, its name in the window and its file.
struct BundleVideo {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::filesystem::path path;
};

// A file for the archive: its name there and its bytes.
using BundleFile = std::pair<std::string, std::string>;

// How the graphs are drawn: the session's video colours, the chart colours
// (Settings' Exported graphs) and the chart's size.
struct BundleStyle {
    VideoColors* colors = nullptr;
    theme::ChartColours colours;
    QSize size{1600, 600};
};

// The videos' names for their files, in their order: each its label; where
// several share one, each with its folder's name after it in brackets; where
// those are shared too, numbered after that ("name (folder) 2").
std::vector<QString> bundle_names(const std::vector<BundleVideo>& videos);

// `text` fit for a file name: the characters Windows does not allow in one
// replaced with "_".
QString file_safe(const QString& text);

// Metric Graphs' PNG files: for each video, every metric it has
// ("<name> - <metric>.png"); with `together` and more than one video, every
// metric any of them has, with all of them ("All test videos - <metric>.png").
std::vector<BundleFile> metric_graph_files(const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                           bool together);

// The bitrate PNG files, from what `read` (the Bitrate tab) has read: with
// more than one video, the reference with all of them ("All test videos -
// Bitrate.png") and each with the reference ("<name> - Bitrate.png"); with
// one, that one. A file `read` has not read is left out of the graphs, and
// its name added to `missing`.
std::vector<BundleFile> bitrate_graph_files(const BitratePanel& read, const std::filesystem::path& reference,
                                            const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                            std::vector<QString>& missing);

}  // namespace vml::app
