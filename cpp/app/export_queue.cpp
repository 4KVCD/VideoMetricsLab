#include "export_queue.h"

#include <QTimer>

#include <algorithm>
#include <iterator>
#include <optional>
#include <set>
#include <utility>

#include "bitrate_panel.h"
#include "i18n.h"
#include "vml/app_log.h"
#include "vml/export_csv.h"
#include "vml/platform/system.h"
#include "vml/zip_archive.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.graph_export");

// How often a waiting export looks whether its bitrate is read.
constexpr int bundle_poll_ms = 500;

// Says where an export was written where its name was taken.
void log_numbered(const std::filesystem::path& wanted, const std::filesystem::path& written) {
    if (written != wanted) {
        logger.info(platform::path_text(wanted.filename()) + " is taken: exported as " +
                    platform::path_text(written));
    }
}

// The paths as the indicator lists them.
QStringList texts_of(const std::vector<std::filesystem::path>& paths) {
    QStringList texts;
    for (const std::filesystem::path& path : paths) {
        texts.append(text_of(path));
    }
    return texts;
}

// Whether the Bitrate tab is still reading `path` (or is about to).
bool still_reading(const BitratePanel& panel, const std::filesystem::path& path) {
    const std::optional<BitrateState> state = panel.state_of(path);
    return state == BitrateState::queued || state == BitrateState::reading || state == BitrateState::packets_read;
}

// The files whose bitrate an export draws as the Bitrate tab reads it
// (bitrate_files), by path.
std::vector<std::filesystem::path> bitrate_paths(const std::filesystem::path& reference,
                                                const std::vector<BundleVideo>& videos, const SavedBitrates& saved) {
    std::vector<std::filesystem::path> paths;
    for (const VideoInfo& file : bitrate_files(reference, videos, saved)) {
        paths.push_back(file.path);
    }
    return paths;
}

}  // namespace

std::filesystem::path free_path(const std::filesystem::path& out) {
    std::set<std::filesystem::path> reserved;
    const std::filesystem::path free =
        unique_output_names(out.parent_path(), {platform::path_text(out.filename())}, reserved).front();
    log_numbered(out, free);
    return free;
}

QStringList write_loose(const std::filesystem::path& folder, const std::vector<BundleFile>& files) {
    const std::vector<std::filesystem::path> written = write_loose_export(folder, files);
    if (!written.empty()) {
        log_numbered(folder / platform::path_from_text(files.front().first), written.front());
    }
    return texts_of(written);
}

ExportQueue::ExportQueue(Window window) : QObject(window.parent), window_(std::move(window)) {}

void ExportQueue::queue(PendingBundle bundle) {
    // What the Bitrate tab has not read, nor is reading, is read now: a
    // video loaded with saved scores, say, or one encoded again under its
    // name since the tab read it (add_files reads that again). A file it
    // could not read is not tried again. They are read where they are now,
    // the paths the export looks their bitrate up by, not where saved scores
    // say they were.
    const std::vector<VideoInfo> files = bitrate_files(bundle.reference, bundle.videos, bundle.saved_bitrate);
    const bool unread = std::any_of(files.begin(), files.end(), [this](const VideoInfo& file) {
        return window_.bitrate.wants_reading(file.path);
    });
    if (unread) {
        window_.bitrate.add_and_analyze(files);
    }
    pending_.push_back(std::move(bundle));
    write_ready();
    if (!pending_.empty()) {
        if (timer_ == nullptr) {
            timer_ = new QTimer(this);
            timer_->setInterval(bundle_poll_ms);
            connect(timer_, &QTimer::timeout, this, [this] { write_ready(); });
        }
        timer_->start();
        if (!pending_.back().automatic) {
            window_.show_saving(app::tr("The bitrate graphs are still being built; the graphs will be exported to "
                                        "{path} as soon as they're done.",
                                        {{"path", text_of(pending_.back().destination)}}));
        }
    }
}

void ExportQueue::write_ready(bool now) {
    const BundleStyle style = window_.style();
    for (auto bundle = pending_.begin(); bundle != pending_.end();) {
        const std::vector<std::filesystem::path> paths =
            bitrate_paths(bundle->reference, bundle->videos, bundle->saved_bitrate);
        if (!now && std::any_of(paths.begin(), paths.end(), [this](const auto& path) {
                return still_reading(window_.bitrate, path);
            })) {
            ++bundle;
            continue;
        }
        const PendingBundle ready = std::move(*bundle);
        bundle = pending_.erase(bundle);
        std::vector<BundleFile> files = ready.extra;
        std::vector<BundleFile> metric_graphs = metric_graph_files(ready.videos, style, ready.together);
        files.insert(files.end(), std::make_move_iterator(metric_graphs.begin()),
                     std::make_move_iterator(metric_graphs.end()));
        std::vector<BundleVideo> with_files;
        std::copy_if(ready.videos.begin(), ready.videos.end(), std::back_inserter(with_files),
                     [](const BundleVideo& video) { return !video.result->resample_target; });
        std::vector<QString> missing;
        std::vector<BundleFile> bitrate_graphs =
            bitrate_graph_files(window_.bitrate, ready.reference, with_files, style, missing, ready.saved_bitrate);
        files.insert(files.end(), std::make_move_iterator(bitrate_graphs.begin()),
                     std::make_move_iterator(bitrate_graphs.end()));
        const QString destination = text_of(ready.destination);
        // An archive of several videos: each one's files in a folder of its
        // own (save_export's folders). One the user asked for says how it
        // went on the indicator; the after-run one only where it fails.
        std::function<QStringList()> write;
        if (ready.loose) {
            write = [files = std::move(files), frames = ready.frames, out = ready.destination, folders = ready.folders,
                     shared = ready.shared] {
                return write_loose(out, in_video_folders(made_files(files, frames), folders, shared));
            };
        } else {
            write = [files = std::move(files), frames = ready.frames, out = ready.destination,
                     numbered = ready.numbered, folders = ready.folders, shared = ready.shared] {
                const std::filesystem::path written = numbered ? free_path(out) : out;
                write_zip(written, in_video_folders(made_files(files, frames), folders, shared));
                return QStringList{text_of(written)};
            };
        }
        window_.submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}), std::move(write),
                       ready.automatic);
        const QString names = QStringList(missing.begin(), missing.end()).join(", ");
        logger.info("Graphs exported to " + platform::path_text(ready.destination) +
                    (missing.empty() ? std::string() : " without the bitrate of " + names.toStdString()));
        if (!missing.empty()) {  // the files are said on the indicator; what they lack, here
            window_.show_status(app::tr("Graphs exported to {path}, without a bitrate graph for {names}: their "
                                        "bitrate couldn't be read.",
                                        {{"path", destination}, {"names", names}}));
        }
    }
    if (pending_.empty() && timer_ != nullptr) {
        timer_->stop();
    }
}

}  // namespace vml::app
