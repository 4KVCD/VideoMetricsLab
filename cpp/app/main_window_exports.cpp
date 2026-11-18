// The window's graph exports (graph_bundle.h): Export all graphs as ZIP, a
// scored video's results with or without its graphs from its right-click
// menu, and Settings' export of each video's results and graphs after its
// run. The bitrate graphs need the files' bitrate, which the Bitrate tab reads
// after each run: an export made before it is read waits for it, and says so.
#include "main_window.h"

#include <QTimer>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "bitrate_panel.h"
#include "file_writes.h"
#include "i18n.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/results_summary.h"
#include "vml/zip_archive.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.graph_export");

// How often a waiting export looks whether its bitrate is read.
constexpr int bundle_poll_ms = 500;

// Whether the Bitrate tab is still reading `path` (or is about to).
bool still_reading(const BitratePanel& panel, const std::filesystem::path& path) {
    const std::optional<BitrateState> state = panel.state_of(path);
    return state == BitrateState::queued || state == BitrateState::reading || state == BitrateState::packets_read;
}

// The files whose bitrate an export draws: its reference, then each video
// with a file of its own (not a resolution test's).
std::vector<std::filesystem::path> bitrate_paths(const std::filesystem::path& reference,
                                                const std::vector<BundleVideo>& videos) {
    std::vector<std::filesystem::path> paths{reference};
    for (const BundleVideo& video : videos) {
        if (!video.result->resample_target) {
            paths.push_back(video.path);
        }
    }
    return paths;
}

}  // namespace

std::optional<BundleVideo> MainWindow::scored_video(const RowData& row_data) const {
    const std::shared_ptr<CompletedRun>& run = row_data.completed_run;
    if (!source_info_ || !run || run->partial || row_data.cvvdp_display_of ||
        !same_path(run->result.source, source_info_->path) || !has_requested_results(row_data)) {
        return std::nullopt;
    }
    return BundleVideo{std::shared_ptr<const ComparisonResult>(run, &run->result), run->label,
                       row_data.media_path.value_or(row_data.path)};
}

std::vector<BundleVideo> MainWindow::scored_videos() const {
    std::vector<BundleVideo> videos;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (std::optional<BundleVideo> video = scored_video(*row_data)) {
            videos.push_back(std::move(*video));
        }
    }
    return videos;
}

std::optional<BundleFile> MainWindow::results_file(const RowData& row_data, ResultsFormat format) const {
    // The video's entry as Export results makes it, alone.
    const ComparisonResult* own = row_data.completed_run ? &row_data.completed_run->result : nullptr;
    for (const ExportedRun& run : exported_runs()) {
        if (run.result.get() == own) {
            const nlohmann::ordered_json document = export_document(source_info_, {run}, current_machine());
            const bool csv = format == ResultsFormat::csv;
            return BundleFile{file_safe(run.label + QStringLiteral(" - results")).toStdString() +
                                  (csv ? ".csv" : ".json"),
                              csv ? results_csv(document) : python_json_dumps(document) + "\n"};
        }
    }
    return std::nullopt;
}

void MainWindow::on_export_all_graphs() {
    const std::vector<BundleVideo> videos = scored_videos();
    if (videos.empty()) {
        dialogs_.inform(app::tr("Nothing to export"),
                        app::tr("No test video has been scored against the reference yet."));
        return;
    }
    const QString path =
        dialogs_.save_file(app::tr("Export all graphs"),
                           file_safe(text_of(source_info_->path.stem()) + QStringLiteral(" - graphs")) + ".zip",
                           app::tr("Zip archive (*.zip)"));
    if (!path.isEmpty()) {
        queue_bundle({videos, source_info_->path, path_of(path), {}, true, false});
    }
}

void MainWindow::export_video(int row, ResultsFormat format, bool with_graphs) {
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const std::optional<BundleVideo> video = scored_video(row_data);
    const std::optional<BundleFile> results = results_file(row_data, format);
    if (!video || !results) {
        return;
    }
    if (!with_graphs) {
        const bool csv = format == ResultsFormat::csv;
        const QString path = dialogs_.save_file(app::tr("Export results"), QString::fromStdString(results->first),
                                                csv ? app::tr("CSV file (*.csv)") : app::tr("JSON file (*.json)"));
        if (!path.isEmpty()) {
            file_writes_->submit(app::tr("export {name}", {{"name", name_of(path_of(path))}}),
                                 [bytes = results->second, out = path_of(path)] {
                                     std::ofstream file(out, std::ios::binary | std::ios::trunc);
                                     file << bytes;
                                     file.close();
                                     if (!file) {
                                         throw std::runtime_error(platform::path_text(out) + " could not be written");
                                     }
                                 });
        }
        return;
    }
    const QString path = dialogs_.save_file(
        app::tr("Export graphs and results"),
        file_safe(video->label + QStringLiteral(" - graphs and results")) + ".zip", app::tr("Zip archive (*.zip)"));
    if (!path.isEmpty()) {
        queue_bundle({{*video}, source_info_->path, path_of(path), {*results}, false, false});
    }
}

void MainWindow::export_automatically(const RowData& row_data) {
    if (settings_.auto_export.empty()) {
        return;
    }
    const std::optional<BundleVideo> video = scored_video(row_data);
    const std::optional<BundleFile> results = results_file(row_data, ResultsFormat::json);
    if (!video || !results) {
        return;
    }
    // In the export folder where chosen and set; else beside the test video.
    const std::filesystem::path folder = settings_.auto_export == "folder" && !settings_.export_dir.empty()
                                             ? platform::path_from_text(settings_.export_dir)
                                             : row_data.path.parent_path();
    const std::filesystem::path destination =
        folder / path_of(file_safe(video->label + QStringLiteral(" - VideoMetricsLab")) + ".zip");
    queue_bundle({{*video}, source_info_->path, destination, {*results}, false, true});
}

void MainWindow::queue_bundle(PendingBundle bundle) {
    // What the Bitrate tab has not read, nor is reading, is read now: a
    // video loaded with saved scores, say. A file it could not read is not
    // tried again.
    const std::vector<std::filesystem::path> paths = bitrate_paths(bundle.reference, bundle.videos);
    const bool unread = std::any_of(paths.begin(), paths.end(), [this](const std::filesystem::path& path) {
        const std::optional<BitrateState> state = bitrate_panel_->state_of(path);
        return !state || state == BitrateState::ready || state == BitrateState::stopped;
    });
    if (unread) {
        std::vector<VideoInfo> infos{bundle.videos.front().result->source_info};
        for (const BundleVideo& video : bundle.videos) {
            if (!video.result->resample_target) {
                infos.push_back(video.result->distorted_info);
            }
        }
        bitrate_panel_->add_and_analyze(infos);
    }
    pending_bundles_.push_back(std::move(bundle));
    write_ready_bundles();
    if (!pending_bundles_.empty()) {
        if (bundle_timer_ == nullptr) {
            bundle_timer_ = new QTimer(this);
            bundle_timer_->setInterval(bundle_poll_ms);
            connect(bundle_timer_, &QTimer::timeout, this, [this] { write_ready_bundles(); });
        }
        bundle_timer_->start();
        if (!pending_bundles_.back().automatic) {
            show_status(app::tr("The bitrate graphs are still being built; the graphs will be exported to {path} "
                                "as soon as they're done.",
                                {{"path", text_of(pending_bundles_.back().destination)}}));
        }
    }
}

void MainWindow::write_ready_bundles() {
    const BundleStyle style{video_colors_,
                            theme::export_chart_colours(settings_.export_graph_theme,
                                                        settings_.export_graph_background,
                                                        settings_.export_graph_text)};
    for (auto bundle = pending_bundles_.begin(); bundle != pending_bundles_.end();) {
        const std::vector<std::filesystem::path> paths = bitrate_paths(bundle->reference, bundle->videos);
        if (std::any_of(paths.begin(), paths.end(), [this](const auto& path) {
                return still_reading(*bitrate_panel_, path);
            })) {
            ++bundle;
            continue;
        }
        const PendingBundle ready = std::move(*bundle);
        bundle = pending_bundles_.erase(bundle);
        std::vector<BundleFile> files = ready.extra;
        std::vector<BundleFile> metric_graphs = metric_graph_files(ready.videos, style, ready.together);
        files.insert(files.end(), std::make_move_iterator(metric_graphs.begin()),
                     std::make_move_iterator(metric_graphs.end()));
        std::vector<BundleVideo> with_files;
        std::copy_if(ready.videos.begin(), ready.videos.end(), std::back_inserter(with_files),
                     [](const BundleVideo& video) { return !video.result->resample_target; });
        std::vector<QString> missing;
        std::vector<BundleFile> bitrate_graphs =
            bitrate_graph_files(*bitrate_panel_, ready.reference, with_files, style, missing);
        files.insert(files.end(), std::make_move_iterator(bitrate_graphs.begin()),
                     std::make_move_iterator(bitrate_graphs.end()));
        const QString destination = text_of(ready.destination);
        file_writes_->submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}),
                             [files = std::move(files), out = ready.destination] { write_zip(out, files); });
        logger.info("Graphs exported to " + platform::path_text(ready.destination) +
                    (missing.empty() ? std::string() : " without the bitrate of " + QStringList(missing.begin(), missing.end()).join(", ").toStdString()));
        if (!ready.automatic || !missing.empty()) {
            show_status(missing.empty()
                            ? app::tr("Graphs exported to {path}", {{"path", destination}})
                            : app::tr("Graphs exported to {path}, without a bitrate graph for {names}: their bitrate "
                                      "couldn't be read.",
                                      {{"path", destination}, {"names", QStringList(missing.begin(), missing.end()).join(", ")}}));
        }
    }
    if (pending_bundles_.empty() && bundle_timer_ != nullptr) {
        bundle_timer_->stop();
    }
}

}  // namespace vml::app
