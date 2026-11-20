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

// Each of `files` written into `folder` under its own name, or throws.
void write_loose(const std::filesystem::path& folder, const std::vector<BundleFile>& files) {
    for (const auto& [name, bytes] : files) {
        write_bytes(folder / platform::path_from_text(name), bytes);
    }
}

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
    // The shared export first, named as the right-click menu names it; then
    // each kind on its own, as Saved Results' Export offers them, and the
    // run file the app reads back (Load analysis results). Of the selected
    // rows scored, else of every scored row.
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
    // As the right-click menu's: of videos whose ticked metrics are all scored.
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
    const auto add_formats = [this, &menu, any](const QString& title, SavedResultsPanel::Export json,
                                                       SavedResultsPanel::Export csv) {
        QMenu* kind = menu.addMenu(title);
        kind->setEnabled(any);
        connect(kind->addAction(app::tr("As JSON...")), &QAction::triggered, this,
                [this, json] { export_rows_as(json); });
        connect(kind->addAction(app::tr("As CSV...")), &QAction::triggered, this,
                [this, csv] { export_rows_as(csv); });
    };
    add_formats(app::tr("Results"), SavedResultsPanel::Export::results_json, SavedResultsPanel::Export::results_csv);
    add_formats(app::tr("Per-frame scores"), SavedResultsPanel::Export::frames_json,
                SavedResultsPanel::Export::frames_csv);
    QAction* graphs = menu.addAction(app::tr("Graphs as ZIP..."));
    graphs->setEnabled(any);
    connect(graphs, &QAction::triggered, this,
            [this] { export_rows_as(SavedResultsPanel::Export::graphs); });
    QAction* segments = menu.addAction(app::tr("Export weak segments..."));
    segments->setEnabled(any);
    connect(segments, &QAction::triggered, this,
            [this] { export_rows_as(SavedResultsPanel::Export::segments); });
    menu.addSeparator();
    QAction* run_files = menu.addAction(app::tr("Run file (.metrics.json)..."));
    run_files->setEnabled(any);
    connect(run_files, &QAction::triggered, this, [this] {
        std::vector<std::shared_ptr<CompletedRun>> runs;
        for (const int row : export_menu_rows()) {  // finished runs only (exportable_video)
            runs.push_back(rows_[static_cast<std::size_t>(row)]->completed_run);
        }
        save_run_files(runs);
    });
}

void MainWindow::export_rows() {
    // Of videos whose ticked metrics are all scored, as the menu offered it.
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
    ExportPlan plan{std::move(videos), reference, {}, graphs, several, name};
    bool segments_skipped = false;
    plan.files = export_files(settings_, plan.videos, reference, document, rules, several, name, segments_skipped);
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
    if (plan.files.empty() && !plan.graphs) {
        if (!folder) {
            dialogs_.inform(app::tr("Nothing to export"),
                            app::tr("Nothing that Settings > Storage > Exports include ticks could be exported."));
        }
        return;
    }
    const bool automatic = folder.has_value();
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
        if (plan.graphs) {
            queue_bundle({plan.videos, plan.reference, destination, std::move(plan.files), plan.several, automatic});
            return;
        }
        file_writes_->submit(app::tr("export {name}", {{"name", name_of(destination)}}),
                             [files = std::move(plan.files), out = destination] { write_zip(out, files); });
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
    if (!plan.files.empty()) {
        file_writes_->submit(app::tr("export {name}", {{"name", plan.name}}),
                             [files = std::move(plan.files), destination] { write_loose(destination, files); });
    }
    if (plan.graphs) {
        queue_bundle({plan.videos, plan.reference, destination, {}, plan.several, automatic, true});
    } else if (!automatic) {
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

void MainWindow::add_saved_bitrates(const std::vector<LoadedSavedResult>& results) {
    // A file no longer there has its bitrate as it was saved, if it was.
    const std::filesystem::path base = cache_folder();
    const auto add_saved_bitrate = [&](const std::filesystem::path& file, const std::filesystem::path& folder,
                                       bool reference) {
        std::error_code error;
        const std::optional<BitrateState> state = bitrate_panel_->state_of(file);
        if (std::filesystem::exists(file, error) || state == BitrateState::complete ||
            state == BitrateState::queued || state == BitrateState::reading || state == BitrateState::packets_read) {
            return;
        }
        if (const auto saved = load_saved_bitrate(base, saved_content_hash(folder, reference), file)) {
            bitrate_panel_->add_saved(*saved);
        }
    };
    add_saved_bitrate(results.front().result->source, results.front().folder, true);
    for (const LoadedSavedResult& saved : results) {
        if (!saved.result->resample_target) {
            add_saved_bitrate(saved.result->distorted, saved.folder, false);
        }
    }
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
    if (std::find(items.begin(), items.end(), "graphs") != items.end()) {
        add_saved_bitrates(results);
    }
    const std::optional<VideoInfo> source =
        one_reference ? std::optional(results.front().result->source_info) : std::nullopt;
    const QString name = several ? text_of(reference.stem()) : results.front().label;
    save_export(export_plan(std::move(videos), reference,
                            [this, &runs, &source] { return export_document(source, runs, current_machine()); },
                            saved_results_panel_->segment_rules(), several, name),
                std::nullopt);
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
        if (ready.loose) {
            file_writes_->submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}),
                                 [files = std::move(files), out = ready.destination] { write_loose(out, files); });
        } else {
            file_writes_->submit(app::tr("export {name}", {{"name", name_of(ready.destination)}}),
                                 [files = std::move(files), out = ready.destination] { write_zip(out, files); });
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
        add_saved_bitrates(results);
        queue_bundle({videos, first_reference, path_of(path), {}, true, false});
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
