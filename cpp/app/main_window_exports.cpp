// The window's graph exports (graph_bundle.h): Export > Graphs as ZIP, a
// scored video's results with or without its graphs from its right-click
// menu, and Settings' export of each video's results and graphs after its
// run; and the weak segments report, of every video or after each run. The bitrate graphs need the files' bitrate, which the Bitrate tab reads
// after each run: an export made before it is read waits for it, and says so.
#include "main_window.h"

#include <QMenu>
#include <QTimer>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

#include "bitrate_panel.h"
#include "file_writes.h"
#include "i18n.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/bitrate_store.h"
#include "vml/export_csv.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/results_summary.h"
#include "vml/segment_profiles.h"
#include "vml/weak_frames.h"
#include "vml/zip_archive.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.graph_export");

// How often a waiting export looks whether its bitrate is read.
constexpr int bundle_poll_ms = 500;

// Writes `bytes` to `out` whole, or throws.
void write_bytes(const std::filesystem::path& out, const std::string& bytes) {
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    file << bytes;
    file.close();
    if (!file) {
        throw std::runtime_error(platform::path_text(out) + " could not be written");
    }
}

// Says where an export was written where its name was taken.
void log_numbered(const std::filesystem::path& wanted, const std::filesystem::path& written) {
    if (written != wanted) {
        logger.info(platform::path_text(wanted.filename()) + " is taken: exported as " +
                    platform::path_text(written));
    }
}

// `out`, or where it is taken, a free name beside it (unique_output_names):
// for a .zip file the user did not name, of Settings' after-run export.
std::filesystem::path free_path(const std::filesystem::path& out) {
    std::set<std::filesystem::path> reserved;
    const std::filesystem::path free =
        unique_output_names(out.parent_path(), {platform::path_text(out.filename())}, reserved).front();
    log_numbered(out, free);
    return free;
}

// write_loose_export, saying where its files were written if numbered.
void write_loose(const std::filesystem::path& folder, const std::vector<BundleFile>& files) {
    const std::vector<std::filesystem::path> written = write_loose_export(folder, files);
    if (!written.empty()) {
        log_numbered(folder / platform::path_from_text(files.front().first), written.front());
    }
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

std::optional<BundleVideo> MainWindow::scored_video(const RowData& row_data) const {
    const std::shared_ptr<CompletedRun>& run = row_data.completed_run;
    if (!source_info_ || !run || run->partial || row_data.cvvdp_display_of ||
        !same_path(run->result.source, source_info_->path)) {
        return std::nullopt;
    }
    // Every ticked metric scored, or failed on the run (settled_for_export).
    std::vector<std::string> failed;
    for (const auto& [key, why] : row_data.metric_failures) {
        failed.push_back(key);
    }
    if (!settled_for_export(requested_metrics(row_data), reusable_metric_keys(row_data), failed)) {
        return std::nullopt;
    }
    return BundleVideo{std::shared_ptr<const ComparisonResult>(run, &run->result), run->label,
                       row_data.media_path.value_or(row_data.path)};
}

std::optional<int> MainWindow::row_of_graph(const std::string& identity) const {
    for (std::size_t row = 0; row < rows_.size(); ++row) {
        const RowData& row_data = *rows_[row];
        if (row_data.completed_run && row_data.completed_run->graph_identity == identity &&
            scored_video(row_data)) {
            return static_cast<int>(row);
        }
    }
    return std::nullopt;
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

std::optional<nlohmann::ordered_json> MainWindow::row_document(const RowData& row_data) const {
    // The video's entry as Export results makes it, alone.
    const ComparisonResult* own = row_data.completed_run ? &row_data.completed_run->result : nullptr;
    for (const ExportedRun& run : exported_runs()) {
        if (run.result.get() == own) {
            return export_document(source_info_, {run}, current_machine());
        }
    }
    return std::nullopt;
}

bool MainWindow::rows_selected() const {
    return distorted_table_->selectionModel()->hasSelection();
}

std::optional<BundleVideo> MainWindow::exportable_video(const RowData& row_data) const {
    // A finished run of the reference's, as scored_video, but whatever its
    // ticked metrics: what it has is exported, as Save selected results did.
    const std::shared_ptr<CompletedRun>& run = row_data.completed_run;
    if (!source_info_ || !run || run->partial || row_data.cvvdp_display_of ||
        !same_path(run->result.source, source_info_->path)) {
        return std::nullopt;
    }
    return BundleVideo{std::shared_ptr<const ComparisonResult>(run, &run->result), run->label,
                       row_data.media_path.value_or(row_data.path)};
}

std::vector<int> MainWindow::export_menu_rows() const {
    std::set<int> selected;
    for (const QModelIndex& index : distorted_table_->selectionModel()->selectedIndexes()) {
        selected.insert(index.row());
    }
    std::vector<int> rows;
    for (std::size_t row = 0; row < rows_.size(); ++row) {
        const bool chosen = selected.empty() || selected.contains(static_cast<int>(row));
        if (chosen && exportable_video(*rows_[row])) {
            rows.push_back(static_cast<int>(row));
        }
    }
    return rows;
}

void MainWindow::fill_videos_export_menu(QMenu& menu) {
    // The archive first, named as the right-click menu names it; then the
    // quick grabs Saved Results' Export offers too (add_quick_grabs), the
    // run file last. Of the selected rows scored, else of every scored row.
    menu.clear();
    for (QMenu* kind : menu.findChildren<QMenu*>(Qt::FindDirectChildrenOnly)) {
        delete kind;  // clear() leaves the menus addMenu made
    }
    const std::vector<int> rows = export_menu_rows();
    const bool any = !rows.empty();
    const std::vector<std::string> items = exported_items(settings_);
    const ExportScope scope = !rows_selected()     ? ExportScope::all_videos
                              : rows.size() == 1 ? ExportScope::this_video
                                                 : ExportScope::selected_videos;
    QAction* contents = menu.addAction(export_action_text(scope, items));
    // As the right-click menu's: of videos whose ticked metrics are all
    // scored, or failed on the run (scored_video).
    const bool all_scored = std::all_of(rows.begin(), rows.end(), [this](int row) {
        return scored_video(*rows_[static_cast<std::size_t>(row)]).has_value();
    });
    contents->setEnabled(any && all_scored && !export_contents(items).isEmpty());
    // Each entry takes the rows as they are when it is clicked
    // (export_menu_rows), not as the menu opened: the menu's own event loop
    // delivers a run's results while it is open, and a row's run that turned
    // partial is not exported.
    connect(contents, &QAction::triggered, this, [this] { export_rows(); });
    menu.addSeparator();
    SavedResultsPanel::add_quick_grabs(menu, any, [this](SavedResultsPanel::Export kind) { export_rows_as(kind); });
}

void MainWindow::export_rows() {
    // Of videos whose ticked metrics are all scored (or failed), as the menu
    // offered it.
    std::vector<int> rows = export_menu_rows();
    std::erase_if(rows, [this](int row) { return !scored_video(*rows_[static_cast<std::size_t>(row)]); });
    if (rows.empty() || !source_info_) {
        return;
    }
    if (!rows_selected()) {
        export_all_videos();
        return;
    }
    if (rows.size() == 1) {
        export_this_video(rows.front());
        return;
    }
    std::set<const ComparisonResult*> chosen;
    std::vector<BundleVideo> videos;
    for (const int row : rows) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        chosen.insert(&row_data.completed_run->result);
        videos.push_back(*scored_video(row_data));
    }
    std::vector<ExportedRun> runs;
    for (ExportedRun& run : exported_runs()) {
        if (chosen.contains(run.result.get())) {
            runs.push_back(std::move(run));
        }
    }
    save_export(export_plan(std::move(videos), source_info_->path,
                            [this, &runs] { return export_document(source_info_, runs, current_machine()); },
                            saved_weak_rules(), true, text_of(source_info_->path.stem())),
                std::nullopt);
}

void MainWindow::export_rows_as(SavedResultsPanel::Export kind) {
    const std::vector<int> rows = export_menu_rows();
    if (rows.empty() || !source_info_) {
        return;
    }
    std::set<const ComparisonResult*> chosen;
    std::vector<BundleVideo> videos;
    for (const int row : rows) {
        const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
        chosen.insert(&row_data.completed_run->result);
        videos.push_back(*exportable_video(row_data));
    }
    const QString name = videos.size() == 1 ? videos.front().label : text_of(source_info_->path.stem());
    switch (kind) {
    case SavedResultsPanel::Export::contents:
        export_rows();
        return;
    case SavedResultsPanel::Export::run_file: {
        std::vector<std::pair<const ComparisonResult*, QString>> runs;
        for (const int row : rows) {  // finished runs only (exportable_video)
            const CompletedRun& run = *rows_[static_cast<std::size_t>(row)]->completed_run;
            runs.emplace_back(&run.result, run.label);
        }
        save_run_files(runs);
        return;
    }
    case SavedResultsPanel::Export::results_json:
    case SavedResultsPanel::Export::results_csv: {
        // As Export results lists them (exported_runs), with what each run
        // knows of how it was made.
        std::vector<ExportedRun> runs;
        for (ExportedRun& run : exported_runs()) {
            if (chosen.contains(run.result.get())) {
                runs.push_back(std::move(run));
            }
        }
        save_results_document(export_document(source_info_, runs, current_machine()), name,
                              kind == SavedResultsPanel::Export::results_csv);
        return;
    }
    case SavedResultsPanel::Export::frames_json:
    case SavedResultsPanel::Export::frames_csv: {
        std::vector<std::pair<std::shared_ptr<const ComparisonResult>, QString>> results;
        for (const BundleVideo& video : videos) {
            results.emplace_back(video.result, video.label);
        }
        save_per_frame_files(results, kind == SavedResultsPanel::Export::frames_json);
        return;
    }
    case SavedResultsPanel::Export::graphs: {
        // Each video's graph of every metric, all of them together, and their
        // bitrate with the reference's, as one .zip file.
        const QString path = dialogs_.save_file(app::tr("Export graphs"),
                                                file_safe(name + QStringLiteral(" - graphs")) + ".zip",
                                                app::tr("Zip archive (*.zip)"));
        if (!path.isEmpty()) {
            queue_bundle({videos, source_info_->path, path_of(path), {}, true, false});
        }
        return;
    }
    case SavedResultsPanel::Export::segments: {
        const SegmentRules rules = saved_weak_rules();
        if (rules.conditions.empty()) {
            dialogs_.inform(app::tr("No weak segment conditions"),
                            app::tr("The profile picked has no conditions: set them in Settings > Weak segments, or "
                                    "pick a preset."));
            return;
        }
        std::vector<SegmentsReportVideo> report;
        for (const BundleVideo& video : videos) {
            report.push_back({video.label.toStdString(), platform::path_text(video.result->distorted),
                              platform::path_text(video.result->source), video.result.get()});
        }
        save_segments_report(report, rules, source_info_->path);
        return;
    }
    }
}

void MainWindow::save_results_document(const nlohmann::ordered_json& document, const QString& name, bool csv) {
    const QString path = dialogs_.save_file(
        app::tr("Export results"), file_safe(name + QStringLiteral(" - results")) + (csv ? ".csv" : ".json"),
        csv ? app::tr("CSV file (*.csv)") : app::tr("JSON file (*.json)"));
    if (path.isEmpty()) {
        return;
    }
    std::string bytes = csv ? results_csv(document) : python_json_dumps(document) + "\n";
    file_writes_->submit(app::tr("export {name}", {{"name", name_of(path_of(path))}}),
                         [bytes = std::move(bytes), out = path_of(path)] { write_bytes(out, bytes); });
    show_status(app::tr("Results exported to {path}", {{"path", path}}));
}

void MainWindow::save_per_frame_files(
    const std::vector<std::pair<std::shared_ptr<const ComparisonResult>, QString>>& results, bool json) {
    // A file per comparison, as the per-frame scores exports write it.
    const QString folder = dialogs_.choose_folder(app::tr("Export per-frame scores to"),
                                                  dialogs_.export_folder ? dialogs_.export_folder() : QString());
    if (folder.isEmpty()) {
        return;
    }
    std::set<std::filesystem::path> reserved;
    std::vector<std::pair<std::shared_ptr<const ComparisonResult>, std::filesystem::path>> files;
    for (const auto& [result, label] : results) {
        files.emplace_back(result, unique_output_path(path_of(folder), output_stem(label.toStdString()),
                                                      json ? ".json" : ".csv", reserved));
    }
    file_writes_->submit(app::tr("export {name}", {{"name", folder}}), [files = std::move(files), json] {
        for (const auto& [result, out] : files) {
            if (json) {
                export_per_frame_json(*result, out);
            } else {
                export_csv(*result, out);
            }
        }
    });
    show_status(app::tr("Per-frame scores exported to {path}", {{"path", folder}}));
}

MainWindow::ExportPlan MainWindow::export_plan(std::vector<BundleVideo> videos,
                                               const std::filesystem::path& reference,
                                               const std::function<nlohmann::ordered_json()>& document,
                                               const SegmentRules& rules, bool several, const QString& name) {
    const std::vector<std::string> items = exported_items(settings_);
    const bool graphs = std::find(items.begin(), items.end(), "graphs") != items.end();
    ExportPlan plan{std::move(videos), reference, {}, {}, graphs, several, name};
    bool segments_skipped = false;
    plan.files =
        export_files(settings_, plan.videos, reference, document, rules, several, name, plan.frames, segments_skipped);
    if (segments_skipped) {
        logger.info("No weak segments exported for " + name.toStdString() + ": the rules have no condition");
        show_status(app::tr("{name}'s weak segments weren't exported: no condition is set in Settings > Weak "
                            "segments.",
                            {{"name", name}}));
    }
    if (plan.videos.empty()) {
        plan.graphs = false;  // a run's results without a scored video of the reference
    }
    return plan;
}

void MainWindow::save_export(ExportPlan plan, const std::optional<std::filesystem::path>& folder) {
    if (plan.files.empty() && plan.frames.empty() && !plan.graphs) {
        if (!folder) {
            dialogs_.inform(app::tr("Nothing to export"),
                            app::tr("Nothing that Settings > Storage > Exports include ticks could be exported."));
        }
        return;
    }
    const bool automatic = folder.has_value();
    // The folders the videos' files go in (in_video_folders): their names,
    // with several videos; none for one.
    const std::vector<QString> folders =
        plan.several && plan.videos.size() > 1 ? bundle_names(plan.videos) : std::vector<QString>{};
    const std::vector<std::string> shared = shared_stems(plan.name);
    if (settings_.export_zip) {
        const QString file_name = file_safe(plan.name + QStringLiteral(" - VideoMetricsLab")) + ".zip";
        std::filesystem::path destination;
        if (folder) {
            destination = *folder / path_of(file_name);
        } else {
            const QString path = dialogs_.save_file(app::tr("Export"), file_name, app::tr("Zip archive (*.zip)"));
            if (path.isEmpty()) {
                return;
            }
            destination = path_of(path);
        }
        // The file the user named is replaced, as they were asked; the
        // after-run export's replaces none (its name is the video's file
        // name, which another video's may share).
        if (plan.graphs) {
            queue_bundle({plan.videos, plan.reference, destination, std::move(plan.files), plan.several, automatic,
                          false, automatic, std::move(plan.saved_bitrate), std::move(plan.frames), folders, shared});
            return;
        }
        file_writes_->submit(app::tr("export {name}", {{"name", name_of(destination)}}),
                             [files = std::move(plan.files), frames = std::move(plan.frames), out = destination,
                              automatic, folders, shared] {
                                 write_zip(automatic ? free_path(out) : out,
                                           in_video_folders(made_files(files, frames), folders, shared));
                             });
        if (!automatic) {
            show_status(app::tr("Exported to {path}", {{"path", text_of(destination)}}));
        }
        return;
    }
    std::filesystem::path destination;
    if (folder) {
        destination = *folder;
    } else {
        const QString chosen = dialogs_.choose_folder(app::tr("Export to"),
                                                      dialogs_.export_folder ? dialogs_.export_folder() : QString());
        if (chosen.isEmpty()) {
            return;
        }
        destination = path_of(chosen);
    }
    // Written at once with the graphs, if any, so that where a name is taken
    // they are all numbered alike (write_loose_export).
    if (plan.graphs) {
        queue_bundle({plan.videos, plan.reference, destination, std::move(plan.files), plan.several, automatic, true,
                      false, std::move(plan.saved_bitrate), std::move(plan.frames), folders, shared});
    } else {
        file_writes_->submit(app::tr("export {name}", {{"name", plan.name}}),
                             [files = std::move(plan.files), frames = std::move(plan.frames), destination,
                              folders, shared] {
                                 write_loose(destination, in_video_folders(made_files(files, frames), folders, shared));
                             });
    }
    if (!plan.graphs && !automatic) {
        show_status(app::tr("Exported to {path}", {{"path", text_of(destination)}}));
    }
}

void MainWindow::export_this_video(int row) {
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const std::optional<BundleVideo> video = scored_video(row_data);
    const std::optional<nlohmann::ordered_json> document = row_document(row_data);
    if (!video || !document) {
        return;
    }
    save_export(export_plan({*video}, source_info_->path, [&document] { return *document; }, saved_weak_rules(),
                            false, video->label),
                std::nullopt);
}

void MainWindow::export_all_videos() {
    const std::vector<ExportedRun> runs = exported_runs();
    if (runs.empty()) {
        dialogs_.inform(app::tr("Nothing to export"), app::tr("No test video has been scored against the reference yet."));
        return;
    }
    const std::filesystem::path reference = source_info_ ? source_info_->path : std::filesystem::path();
    const QString name = source_info_ ? text_of(source_info_->path.stem()) : QStringLiteral("results");
    save_export(export_plan(scored_videos(), reference,
                            [this, &runs] { return export_document(source_info_, runs, current_machine()); },
                            saved_weak_rules(), true, name),
                std::nullopt);
}

SavedBitrates MainWindow::saved_bitrates(std::vector<BundleVideo>& videos,
                                         const std::vector<LoadedSavedResult>& results) const {
    // The file at a path is read by the Bitrate tab, which knows files by
    // their path, only where it is the one the scores were saved for. Issue
    // #134: a newer encode under the same name had its bitrate drawn beside
    // the older one's scores.
    std::vector<std::filesystem::path> folders;
    for (const LoadedSavedResult& result : results) {
        folders.push_back(result.folder);
    }
    return vml::app::saved_bitrates(
        videos, folders, results.front().result->source, saved_file_at,
        [base = cache_folder()](const std::filesystem::path& folder, bool reference, const std::filesystem::path& file) {
            return load_saved_bitrate(base, saved_content_hash(folder, reference), file);
        });
}

void MainWindow::export_saved_contents(const std::vector<LoadedSavedResult>& results) {
    if (results.empty()) {
        return;
    }
    const std::filesystem::path& reference = results.front().result->source;
    const bool one_reference = std::all_of(results.begin(), results.end(), [&reference](const auto& saved) {
        return same_path(saved.result->source, reference);
    });
    const bool several = results.size() > 1;
    const std::vector<std::string> items = exported_items(settings_);
    if (several && !one_reference &&
        (std::find(items.begin(), items.end(), "graphs") != items.end() ||
         std::find(items.begin(), items.end(), "weak_segments") != items.end())) {
        dialogs_.inform(app::tr("One reference at a time"),
                        app::tr("Graphs and weak segments are exported for test videos of one reference: select "
                                "comparisons that share it."));
        return;
    }
    std::vector<BundleVideo> videos;
    std::vector<ExportedRun> runs;
    for (const LoadedSavedResult& saved : results) {
        videos.push_back({saved.result, saved.label, saved.result->distorted});
        runs.push_back({saved.result, saved.label});
    }
    const std::optional<VideoInfo> source =
        one_reference ? std::optional(results.front().result->source_info) : std::nullopt;
    const QString name = several ? text_of(reference.stem()) : results.front().label;
    ExportPlan plan = export_plan(std::move(videos), reference,
                                  [this, &runs, &source] { return export_document(source, runs, current_machine()); },
                                  saved_results_panel_->segment_rules(), several, name);
    if (plan.graphs) {
        plan.saved_bitrate = saved_bitrates(plan.videos, results);
    }
    save_export(std::move(plan), std::nullopt);
}

SegmentRules MainWindow::saved_weak_rules() const {
    return segment_profile_in_use(settings_).rules;
}

void MainWindow::export_segments_report(const SegmentRules& rules) {
    const std::vector<BundleVideo> videos = scored_videos();
    if (videos.empty()) {
        dialogs_.inform(app::tr("Nothing to export"),
                        app::tr("No test video has been scored against the reference yet."));
        return;
    }
    if (rules.conditions.empty()) {
        dialogs_.inform(app::tr("No weak segment conditions"),
                        app::tr("Set the conditions in Settings > Weak segments first: the report follows the "
                                "saved ones."));
        return;
    }
    std::vector<SegmentsReportVideo> report;
    for (const BundleVideo& video : videos) {
        report.push_back({video.label.toStdString(), platform::path_text(video.path),
                          platform::path_text(source_info_->path), video.result.get()});
    }
    save_segments_report(report, rules, source_info_->path);
}

void MainWindow::save_segments_report(const std::vector<SegmentsReportVideo>& report, const SegmentRules& rules,
                                      const std::filesystem::path& reference) {
    const QString suggested = file_safe(text_of(reference.stem()) + QStringLiteral(" - weak segments")) + "." +
                              qt_text(segments_file_suffix(rules)) + ".json";
    const QString path = dialogs_.save_file(app::tr("Export weak segments report"), suggested,
                                            app::tr("JSON file (*.json);;CSV file (*.csv)"));
    if (path.isEmpty()) {
        return;
    }
    // Made now, while the results are the window's; written on the worker.
    const bool csv = path.endsWith(QStringLiteral(".csv"), Qt::CaseInsensitive);
    std::string bytes = csv ? text_lines(segments_report_csv(report, rules))
                            : python_json_dumps(segments_report_json(report, rules)) + "\n";
    file_writes_->submit(app::tr("export {name}", {{"name", name_of(path_of(path))}}),
                         [bytes = std::move(bytes), out = path_of(path)] { write_bytes(out, bytes); });
    show_status(app::tr("Weak segments report exported to {path}", {{"path", path}}));
}

void MainWindow::export_automatically(const RowData& row_data) {
    if (!settings_.after_run_export) {
        return;
    }
    const std::optional<BundleVideo> video = scored_video(row_data);
    const std::optional<nlohmann::ordered_json> document = row_document(row_data);
    if (!video || !document) {
        return;
    }
    // In the export folder where chosen and set; else beside the test video.
    const std::filesystem::path folder = settings_.after_run_place == "folder" && !settings_.export_dir.empty()
                                             ? platform::path_from_text(settings_.export_dir)
                                             : row_data.path.parent_path();
    save_export(export_plan({*video}, source_info_->path, [&document] { return *document; }, saved_weak_rules(),
                            false, video->label),
                folder);
}

void MainWindow::queue_bundle(PendingBundle bundle) {
    // What the Bitrate tab has not read, nor is reading, is read now: a
    // video loaded with saved scores, say, or one encoded again under its
    // name since the tab read it (add_files reads that again). A file it
    // could not read is not tried again. They are read where they are now,
    // the paths the export looks their bitrate up by, not where saved scores
    // say they were.
    const std::vector<VideoInfo> files = bitrate_files(bundle.reference, bundle.videos, bundle.saved_bitrate);
    const bool unread = std::any_of(files.begin(), files.end(), [this](const VideoInfo& file) {
        return bitrate_panel_->wants_reading(file.path);
    });
    if (unread) {
        bitrate_panel_->add_and_analyze(files);
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

void MainWindow::write_ready_bundles(bool now) {
    const BundleStyle style{video_colors_,
                            theme::export_chart_colours(settings_.export_graph_theme,
                                                        settings_.export_graph_background,
                                                        settings_.export_graph_text)};
    for (auto bundle = pending_bundles_.begin(); bundle != pending_bundles_.end();) {
        const std::vector<std::filesystem::path> paths =
            bitrate_paths(bundle->reference, bundle->videos, bundle->saved_bitrate);
        if (!now && std::any_of(paths.begin(), paths.end(), [this](const auto& path) {
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
            bitrate_graph_files(*bitrate_panel_, ready.reference, with_files, style, missing, ready.saved_bitrate);
        files.insert(files.end(), std::make_move_iterator(bitrate_graphs.begin()),
                     std::make_move_iterator(bitrate_graphs.end()));
        const QString destination = text_of(ready.destination);
        // An archive of several videos: each one's files in a folder of its
        // own (save_export's folders).
        if (ready.loose) {
            file_writes_->submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}),
                                 [files = std::move(files), frames = ready.frames, out = ready.destination,
                                  folders = ready.folders, shared = ready.shared] {
                                     write_loose(out, in_video_folders(made_files(files, frames), folders, shared));
                                 });
        } else {
            file_writes_->submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}),
                                 [files = std::move(files), frames = ready.frames, out = ready.destination,
                                  numbered = ready.numbered, folders = ready.folders, shared = ready.shared] {
                                     write_zip(numbered ? free_path(out) : out,
                                               in_video_folders(made_files(files, frames), folders, shared));
                                 });
        }
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

void MainWindow::export_saved_results(SavedResultsPanel::Export kind, const std::vector<LoadedSavedResult>& results) {
    if (results.empty()) {
        return;
    }
    // One reference among them, or none (each entry names its own).
    const std::filesystem::path& first_reference = results.front().result->source;
    const bool one_reference = std::all_of(results.begin(), results.end(), [&first_reference](const auto& saved) {
        return same_path(saved.result->source, first_reference);
    });
    const QString name = results.size() == 1 ? results.front().label : text_of(first_reference.stem());
    switch (kind) {
    case SavedResultsPanel::Export::contents:
        export_saved_contents(results);
        return;
    case SavedResultsPanel::Export::run_file: {
        // As the Videos table's: a saved comparison is a run's result.
        std::vector<std::pair<const ComparisonResult*, QString>> runs;
        for (const LoadedSavedResult& saved : results) {
            runs.emplace_back(saved.result.get(), saved.label);
        }
        save_run_files(runs);
        return;
    }
    case SavedResultsPanel::Export::results_json:
    case SavedResultsPanel::Export::results_csv: {
        std::vector<ExportedRun> runs;
        for (const LoadedSavedResult& saved : results) {
            runs.push_back({saved.result, saved.label});
        }
        save_results_document(
            export_document(one_reference ? std::optional(results.front().result->source_info) : std::nullopt, runs,
                            current_machine()),
            name, kind == SavedResultsPanel::Export::results_csv);
        return;
    }
    case SavedResultsPanel::Export::frames_json:
    case SavedResultsPanel::Export::frames_csv: {
        std::vector<std::pair<std::shared_ptr<const ComparisonResult>, QString>> files;
        for (const LoadedSavedResult& saved : results) {
            files.emplace_back(saved.result, saved.label);
        }
        save_per_frame_files(files, kind == SavedResultsPanel::Export::frames_json);
        return;
    }
    case SavedResultsPanel::Export::graphs: {
        if (!one_reference) {
            dialogs_.inform(app::tr("One reference at a time"),
                            app::tr("Graphs are exported for test videos of one reference: select comparisons that "
                                    "share it."));
            return;
        }
        std::vector<BundleVideo> videos;
        for (const LoadedSavedResult& saved : results) {
            videos.push_back({saved.result, saved.label, saved.result->distorted});
        }
        const QString path = dialogs_.save_file(app::tr("Export graphs"),
                                                file_safe(name + QStringLiteral(" - graphs")) + ".zip",
                                                app::tr("Zip archive (*.zip)"));
        if (path.isEmpty()) {
            return;
        }
        SavedBitrates saved = saved_bitrates(videos, results);
        queue_bundle({videos, first_reference, path_of(path), {}, true, false, false, false, std::move(saved)});
        return;
    }
    case SavedResultsPanel::Export::segments: {
        const SegmentRules rules = saved_results_panel_->segment_rules();
        if (rules.conditions.empty()) {
            dialogs_.inform(app::tr("No weak segment conditions"),
                            app::tr("The profile picked has no conditions: set them in Settings > Weak segments, or "
                                    "pick a preset."));
            return;
        }
        std::vector<SegmentsReportVideo> report;
        for (const LoadedSavedResult& saved : results) {
            report.push_back({saved.label.toStdString(), platform::path_text(saved.result->distorted),
                              platform::path_text(saved.result->source), saved.result.get()});
        }
        save_segments_report(report, rules, first_reference);
        return;
    }
    }
}

}  // namespace vml::app
