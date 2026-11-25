// Graph exports that wait for their bitrate: the bitrate graphs need the
// files' bitrate, which the Bitrate tab reads after each run, so an export
// made before it is read is queued here and written as soon as it is (the
// window's graph exports, main_window_exports.cpp).
#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "graph_bundle.h"
#include "results_export.h"

class QTimer;

namespace vml::app {

class BitratePanel;

// Graphs to save as one .zip file (graph_bundle.h) once the bitrate they
// need is read: of `videos`, compared with `reference`, with `extra` files
// (a video's results); `together`: the videos' graphs together as well;
// `automatic`: Settings' export after a run, which says less.
struct PendingBundle {
    std::vector<BundleVideo> videos;
    std::filesystem::path reference;
    std::filesystem::path destination;
    std::vector<BundleFile> extra;
    bool together = true;
    bool automatic = false;
    // Each file written into `destination`, a folder, rather than a .zip,
    // replacing none there (write_loose_export).
    bool loose = false;
    // The .zip replaces no file at `destination`: it takes a free name beside
    // it (Settings' after-run export, named after the video).
    bool numbered = false;
    // Saved comparisons' files drawn as their bitrate was saved
    // (saved_bitrates).
    SavedBitrates saved_bitrate{};
    // Per-frame scores, made with the files (made_files).
    std::vector<FrameFiles> frames{};
    // With several videos, the folders their own files go in, and the stems
    // of the files of all of them (in_video_folders); none: as they are.
    std::vector<QString> folders{};
    std::vector<std::string> shared{};
};

class ExportQueue : public QObject {
public:
    // What the queue needs of the window.
    struct Window {
        QObject* parent;
        BitratePanel& bitrate;  // reads the bitrate, and draws its graphs
        // The graphs' colours, asked at each write as the user may have
        // changed them.
        std::function<BundleStyle()> style;
        // Writes an export off the window's thread, the indicator saying how
        // it went (MainWindow::submit_export); `quiet`: only a failure.
        std::function<void(const QString& description, std::function<QStringList()> write, bool quiet)> submit;
        std::function<void(const QString&)> show_saving;  // the indicator, while an export waits
        std::function<void(const QString&)> show_status;  // the window's status line
    };

    explicit ExportQueue(Window window);

    // Saves `bundle` when its bitrate is read, starting to read what is not.
    void queue(PendingBundle bundle);
    // Writes the exports whose bitrate is read; with `now`, every one,
    // without the bitrate of a file not read (the window closing).
    void write_ready(bool now = false);

private:
    Window window_;
    std::vector<PendingBundle> pending_;
    QTimer* timer_ = nullptr;  // looks whether their bitrate is read, while any waits
};

// `out`, or where it is taken, a free name beside it (unique_output_names):
// for a .zip file the user did not name, of Settings' after-run export.
std::filesystem::path free_path(const std::filesystem::path& out);
// write_loose_export, saying where its files were written if numbered: the
// files written, as the indicator lists them.
QStringList write_loose(const std::filesystem::path& folder, const std::vector<BundleFile>& files);

}  // namespace vml::app
