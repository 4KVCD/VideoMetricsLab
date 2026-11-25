// The window's graph exports (graph_bundle.h): Export > Graphs as ZIP, a
// scored video's results with or without its graphs from its right-click
// menu, and Settings' export of each video's results and graphs after its
// run; and the weak segments report, of every video or after each run. The bitrate graphs need the files' bitrate, which the Bitrate tab reads
// after each run: an export made before it is read waits for it, and says so.
#include "main_window.h"

#include <QHBoxLayout>
#include <QMenu>
#include <QPushButton>
#include <QTimer>

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

#include "bitrate_panel.h"
#include "export_indicator.h"
#include "file_writes.h"
#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "settings_panel.h"
#include "worst_frames_tab.h"
#include "bd_rate_panel.h"
#include "quick_save.h"
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

// The paths as the indicator lists them.
QStringList texts_of(const std::vector<std::filesystem::path>& paths) {
    QStringList texts;
    for (const std::filesystem::path& path : paths) {
        texts.append(text_of(path));
    }
    return texts;
}

// write_loose_export, saying where its files were written if numbered: the
// files written.
QStringList write_loose(const std::filesystem::path& folder, const std::vector<BundleFile>& files) {
    const std::vector<std::filesystem::path> written = write_loose_export(folder, files);
    if (!written.empty()) {
        log_numbered(folder / platform::path_from_text(files.front().first), written.front());
    }
    return texts_of(written);
}

// An export that cannot be made: said on the indicator, as one that failed,
// not in a window.
ExportOutcome refused(const QString& why) { return {{}, why, why}; }

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

void MainWindow::build_tab_corner() {
    // The tab bar's right end, the same place in both interfaces: the
    // indicator, then the Export button (the archive's panel). The title
    // buttons join them there where the window has no title bar of its own
    // (VML Classic's progress button; make_title_buttons).
    if (tab_corner_ != nullptr) {
        return;
    }
    tab_corner_ = new QWidget;
    auto* corner = new QHBoxLayout(tab_corner_);
    corner->setContentsMargins(0, 0, 0, 0);
    corner->setSpacing(6);
    // VML Modern: a pixel above its buttons. Qt gives the corner the tab
    // bar's height and puts its bottom on the tab pane's top, which Modern's
    // style sheet sets a pixel into the tab bar, so the corner's first row
    // lies above the tab widget and is not drawn: the Export button's rounded
    // top border was cut off. Room at the right for its border too.
    theme::modern_margins(tab_corner_, QMargins(0, 1, 2, 1));
    // The indicator and Export, in a row of their own: in VML Classic two
    // pixels above the tab pane, which the title buttons there (the progress
    // button) are not, being placed as when they were the corner alone.
    auto* exports = new QWidget;
    auto* layout = new QHBoxLayout(exports);
    layout->setContentsMargins(0, 0, 0, 2);
    theme::modern_margins(exports, QMargins(0, 0, 0, 0));
    layout->setSpacing(6);
    corner->addWidget(exports);
    export_indicator_ = new ExportIndicator;  // Details: a tooltip beside it, no window
    layout->addWidget(export_indicator_);
    tab_export_button_ = new QPushButton(app::tr("Export"));
    tab_export_button_->setObjectName("tabExportButton");
    tab_export_button_->setIcon(export_icon(tab_export_button_));
    tab_export_button_->setIconSize(QSize(20, 16));  // the glyph, 16 square, then room before the word
    tab_export_button_->setToolTip(app::tr("Export everything for these videos: results, per-frame scores, graphs, "
                                           "worst frames and weak segments, as Settings > Storage > Exports include "
                                           "says."));
    connect(tab_export_button_, &QPushButton::clicked, this, [this] {
        build_export_panel();
        export_panel_->open_under(tab_export_button_);
    });
    layout->addWidget(tab_export_button_);
    tabs_->setCornerWidget(tab_corner_, Qt::TopRightCorner);
    tab_corner_->show();
}

void MainWindow::build_export_panel() {
    if (export_panel_ != nullptr) {
        return;
    }
    export_panel_ = new ExportPanel(settings_, this);
    export_panel_->videos = [this] { return export_panel_videos(); };
    export_panel_->rule_choices = [this] { return saved_results_panel_->segment_rule_choices(); };
    export_panel_->rule_choice = [this] { return saved_results_panel_->segment_rule_choice(); };
    export_panel_->pick_rules = [this](const QString& choice) { saved_results_panel_->pick_segment_rules(choice); };
    export_panel_->folder = [this] { return dialogs_.export_folder(); };
    export_panel_->change_folder = [this] {
        // A dialog closes the panel (a popup): open again once it is answered.
        export_panel_->hide();
        const QString chosen = dialogs_.choose_folder(app::tr("Export to"), dialogs_.export_folder());
        if (!chosen.isEmpty()) {
            settings_.export_dir = platform::path_text(path_of(chosen));
            save_settings();
            settings_panel_->show_export_choices();
        }
        export_panel_->open_under(tab_export_button_);
    };
    export_panel_->choices_changed = [this] {
        // The panel's ticks are Settings' Exports include itself.
        save_settings();
        settings_panel_->show_export_choices();
    };
    export_panel_->tab_save = [this] { return tab_save(); };
    export_panel_->export_videos = [this](bool selection) { export_from_panel(selection); };
}

ExportPanel::Videos MainWindow::export_panel_videos() const {
    // The open tab's selection where it lists videos (Videos, Metric Graphs'
    // statistics, Saved Results), else none; and every scored video.
    ExportPanel::Videos videos;
    const QString tab = tabs_->tabText(tabs_->currentIndex()).remove(QLatin1Char('&'));
    const auto selection_text = [&tab](const std::vector<QString>& names) {
        return names.size() == 1 ? app::tr("The selected video on {tab}: {name}", {{"tab", tab}, {"name", names.front()}})
                                 : app::tr("The selected videos on {tab} ({count})",
                                           {{"tab", tab}, {"count", static_cast<long long>(names.size())}});
    };
    const auto scored_names = [this](const std::vector<int>& rows) {
        std::vector<QString> names;
        for (const int row : rows) {
            if (const std::optional<BundleVideo> video = scored_video(*rows_[static_cast<std::size_t>(row)])) {
                names.push_back(video->label);
            }
        }
        return names;
    };
    switch (tabs_->currentIndex()) {
    case tab_videos:
        if (rows_selected()) {
            const std::vector<QString> names = scored_names(export_menu_rows());
            videos.selection = selection_text(names.empty() ? std::vector<QString>{QString()} : names);
            if (names.empty()) {
                videos.selection = app::tr("The selected videos on {tab}", {{"tab", tab}});
                videos.selection_refused = nothing_scored_text();
            }
        }
        break;
    case tab_graph: {
        std::vector<int> rows;
        for (const std::string& identity : graph_panel_->selected_identities()) {
            if (const std::optional<int> row = row_of_graph(identity)) {
                rows.push_back(*row);
            }
        }
        if (!graph_panel_->selected_identities().empty()) {
            const std::vector<QString> names = scored_names(rows);
            videos.selection = names.empty() ? app::tr("The selected videos on {tab}", {{"tab", tab}})
                                             : selection_text(names);
            if (names.empty()) {
                // A saved run's line: its comparison is exported from Saved Results.
                videos.selection_refused = nothing_scored_text();
            }
        }
        break;
    }
    case tab_saved_results: {
        const std::vector<LoadedSavedResult> chosen = saved_results_panel_->selected_results();
        if (!chosen.empty()) {
            std::vector<QString> names;
            for (const LoadedSavedResult& saved : chosen) {
                names.push_back(saved.label);
            }
            videos.selection = selection_text(names);
            const std::vector<std::string> items = exported_items(settings_);
            const bool needs_one_reference = std::find(items.begin(), items.end(), "graphs") != items.end() ||
                                             std::find(items.begin(), items.end(), "weak_segments") != items.end();
            const bool one_reference = std::all_of(chosen.begin(), chosen.end(), [&chosen](const auto& saved) {
                return same_path(saved.result->source, chosen.front().result->source);
            });
            if (needs_one_reference && !one_reference) {
                videos.selection_refused = one_reference_text();
            }
        }
        break;
    }
    default:
        break;
    }
    const auto all = static_cast<long long>(scored_videos().size());
    videos.all = app::tr("All scored videos ({count})", {{"count", all}});
    if (all == 0) {
        videos.all_refused = nothing_scored_text();
    }
    return videos;
}

SaveButton* MainWindow::tab_save() const {
    // The open tab's own one-click save, its main card's save icon.
    switch (tabs_->currentIndex()) {
    case tab_videos:
        return videos_save_;
    case tab_graph:
        return graph_panel_->stats_save();
    case tab_worst_frames:
        return worst_tab_->worst_save();
    case tab_frame_compare:
        return frame_compare_panel_->save_still_button;
    case tab_bitrate:
        return bitrate_panel_->graph_save;
    case tab_bd_rate:
        return bd_rate_panel_->table_save;
    case tab_saved_results:
        return saved_results_panel_->save_button();
    default:
        return nullptr;
    }
}

void MainWindow::export_from_panel(bool selection) {
    // Into the export folder, as the panel says, without a dialog; its
    // names numbered where taken, as every export's.
    const std::filesystem::path folder = path_of(dialogs_.export_folder());
    std::error_code ignored;
    std::filesystem::create_directories(folder, ignored);
    if (!selection) {
        export_all_videos(folder);
        return;
    }
    switch (tabs_->currentIndex()) {
    case tab_videos:
        export_rows(folder);
        return;
    case tab_graph: {
        std::vector<int> rows;
        for (const std::string& identity : graph_panel_->selected_identities()) {
            if (const std::optional<int> row = row_of_graph(identity)) {
                rows.push_back(*row);
            }
        }
        export_rows_of(rows, folder);
        return;
    }
    case tab_saved_results:
        export_saved_contents(saved_results_panel_->selected_results(), folder);
        return;
    default:
        export_all_videos(folder);
        return;
    }
}

void MainWindow::report_export(const ExportOutcome& outcome) {
    if (!outcome.error.isEmpty()) {
        logger.info("Export failed: " + (outcome.details.isEmpty() ? outcome.error : outcome.details).toStdString());
    }
    export_indicator_->show_outcome(outcome);
}

void MainWindow::submit_export(const QString& description, std::function<QStringList()> write, bool quiet) {
    if (!quiet) {
        export_indicator_->show_saving();
    }
    auto written = std::make_shared<QStringList>();
    file_writes_->submit(
        description, [write = std::move(write), written] { *written = write(); },
        [this, written, description, quiet](const QString& error) {
            if (error.isEmpty()) {
                if (!quiet) {
                    report_export({*written, {}, {}});
                }
                return;
            }
            report_export({{},
                           tr_message(error),
                           app::tr("Could not {description}: {error}",
                                   {{"description", description}, {"error", tr_message(error)}})});
        });
}

QuickSaves MainWindow::window_quick_saves() {
    return {dialogs_.export_folder, [this] { return quick_formats(settings_); },
            [this, written = written_by(*file_writes_)](std::function<ExportOutcome()> save, ExportReport done) {
                export_indicator_->show_saving();
                written(std::move(save), std::move(done));
            }};
}

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

void MainWindow::fill_videos_quick_menu(QMenu& menu) {
    // The quick grabs Saved Results' save icon offers too (add_quick_grabs),
    // the run file last: of the selected rows scored, else of every scored
    // row, as they are when one is clicked (export_menu_rows: the menu's own
    // event loop delivers a run's results while it is open). The archive is
    // the tab bar's Export panel's, and the table's right-click's.
    SavedResultsPanel::add_quick_grabs(
        menu, !export_menu_rows().empty(), [this](SavedResultsPanel::Export kind) { export_rows_as(kind); },
        videos_refused());
}

void MainWindow::quick_save_results(nlohmann::ordered_json document, const QString& name) {
    // Named as Export results names it; JSON and CSV numbered alike. Their
    // text is made as they are written, off the window's thread: a 2-hour
    // film's frames, each video's.
    const auto shared = std::make_shared<const nlohmann::ordered_json>(std::move(document));
    std::vector<QuickFile> files;
    for (const std::string& format : quick_formats(settings_)) {
        const bool csv = format == "csv";
        files.push_back({file_safe(name + QStringLiteral(" - results")) + (csv ? ".csv" : ".json"),
                         [shared, csv](const QString& path) {
                             return write_text_file(path, csv ? results_csv(*shared) : python_json_dumps(*shared) + "\n");
                         }});
    }
    window_quick_saves().save(dialogs_.export_folder(), std::move(files),
                              [this](const ExportOutcome& outcome) { report_export(outcome); });
}

QString MainWindow::rows_save_text() const {
    const std::vector<int> rows = export_menu_rows();
    if (rows_selected() && rows.size() == 1) {
        return app::tr("the selected video's results ({name})",
                       {{"name", rows_[static_cast<std::size_t>(rows.front())]->completed_run->label}});
    }
    const auto count = static_cast<long long>(rows.size());
    return rows_selected() ? app::tr("the selected videos' results ({count})", {{"count", count}})
                           : app::tr("every scored video's results ({count})", {{"count", count}});
}

void MainWindow::save_rows_results() {
    // As Results as JSON / CSV in the Export menu (export_rows_as), at one click.
    const std::vector<int> rows = export_menu_rows();
    if (rows.empty() || !source_info_) {
        return;
    }
    std::set<const ComparisonResult*> chosen;
    for (const int row : rows) {
        chosen.insert(&rows_[static_cast<std::size_t>(row)]->completed_run->result);
    }
    std::vector<ExportedRun> runs;
    for (ExportedRun& run : exported_runs()) {
        if (chosen.contains(run.result.get())) {
            runs.push_back(std::move(run));
        }
    }
    const QString name = rows.size() == 1 ? rows_[static_cast<std::size_t>(rows.front())]->completed_run->label
                                          : text_of(source_info_->path.stem());
    quick_save_results(export_document(source_info_, runs, current_machine()), name);
}

std::map<SavedResultsPanel::Export, QString> MainWindow::videos_refused() const {
    std::map<SavedResultsPanel::Export, QString> refused;
    if (saved_weak_rules().conditions.empty()) {
        refused[SavedResultsPanel::Export::segments] = no_conditions_text();
    }
    return refused;
}

void MainWindow::export_rows(const std::optional<std::filesystem::path>& folder) {
    if (!rows_selected()) {
        export_all_videos(folder);
        return;
    }
    export_rows_of(export_menu_rows(), folder);
}

void MainWindow::export_rows_of(std::vector<int> rows, const std::optional<std::filesystem::path>& folder) {
    // Of videos whose ticked metrics are all scored (or failed), as the menu
    // offered it.
    std::erase_if(rows, [this](int row) { return !scored_video(*rows_[static_cast<std::size_t>(row)]); });
    if (rows.empty() || !source_info_) {
        return;
    }
    if (rows.size() == 1) {
        export_this_video(rows.front(), folder);
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
                            archive_rules(), true, text_of(source_info_->path.stem())),
                folder);
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
            report_export(refused(no_conditions_text()));
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
    submit_export(app::tr("export {name}", {{"name", name_of(path_of(path))}}),
                  [bytes = std::move(bytes), out = path_of(path)] {
                      write_bytes(out, bytes);
                      return QStringList{text_of(out)};
                  });
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
    submit_export(app::tr("export {name}", {{"name", folder}}), [files = std::move(files), json] {
        QStringList written;
        for (const auto& [result, out] : files) {
            if (json) {
                export_per_frame_json(*result, out);
            } else {
                export_csv(*result, out);
            }
            written.append(text_of(out));
        }
        return written;
    });
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

void MainWindow::save_export(ExportPlan plan, const std::optional<std::filesystem::path>& folder, bool automatic) {
    if (plan.files.empty() && plan.frames.empty() && !plan.graphs) {
        if (!automatic) {
            report_export(
                refused(app::tr("Nothing that Settings > Storage > Exports include ticks could be exported.")));
        }
        return;
    }
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
        // The file the user named is replaced, as they were asked; one
        // written into a folder replaces none (the after-run export's is
        // named after the video's file, which another video's may share).
        const bool numbered = folder.has_value();
        if (plan.graphs) {
            queue_bundle({plan.videos, plan.reference, destination, std::move(plan.files), plan.several, automatic,
                          false, numbered, std::move(plan.saved_bitrate), std::move(plan.frames), folders, shared});
            return;
        }
        const QString description = app::tr("export {name}", {{"name", name_of(destination)}});
        if (automatic) {  // after a run: said only where it fails, on the indicator
            submit_export(
                description,
                [files = std::move(plan.files), frames = std::move(plan.frames), out = destination, folders, shared] {
                    const std::filesystem::path written = free_path(out);
                    write_zip(written, in_video_folders(made_files(files, frames), folders, shared));
                    return QStringList{text_of(written)};
                },
                true);
            return;
        }
        submit_export(description, [files = std::move(plan.files), frames = std::move(plan.frames), out = destination,
                                    numbered, folders, shared] {
            const std::filesystem::path written = numbered ? free_path(out) : out;
            write_zip(written, in_video_folders(made_files(files, frames), folders, shared));
            return QStringList{text_of(written)};
        });
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
    } else if (automatic) {
        submit_export(
            app::tr("export {name}", {{"name", plan.name}}),
            [files = std::move(plan.files), frames = std::move(plan.frames), destination, folders, shared] {
                return write_loose(destination, in_video_folders(made_files(files, frames), folders, shared));
            },
            true);
    } else {
        submit_export(app::tr("export {name}", {{"name", plan.name}}),
                      [files = std::move(plan.files), frames = std::move(plan.frames), destination, folders, shared] {
                          return write_loose(destination, in_video_folders(made_files(files, frames), folders, shared));
                      });
    }
}

void MainWindow::export_this_video(int row, const std::optional<std::filesystem::path>& folder) {
    const RowData& row_data = *rows_[static_cast<std::size_t>(row)];
    const std::optional<BundleVideo> video = scored_video(row_data);
    const std::optional<nlohmann::ordered_json> document = row_document(row_data);
    if (!video || !document) {
        return;
    }
    save_export(export_plan({*video}, source_info_->path, [&document] { return *document; }, archive_rules(), false,
                            video->label),
                folder);
}

void MainWindow::export_all_videos(const std::optional<std::filesystem::path>& folder) {
    const std::vector<ExportedRun> runs = exported_runs();
    if (runs.empty()) {
        report_export(refused(nothing_scored_text()));
        return;
    }
    const std::filesystem::path reference = source_info_ ? source_info_->path : std::filesystem::path();
    const QString name = source_info_ ? text_of(source_info_->path.stem()) : QStringLiteral("results");
    save_export(export_plan(scored_videos(), reference,
                            [this, &runs] { return export_document(source_info_, runs, current_machine()); },
                            archive_rules(), true, name),
                folder);
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

void MainWindow::export_saved_contents(const std::vector<LoadedSavedResult>& results,
                                       const std::optional<std::filesystem::path>& folder) {
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
        report_export(refused(one_reference_text()));
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
    save_export(std::move(plan), folder);
}

SegmentRules MainWindow::saved_weak_rules() const {
    return segment_profile_in_use(settings_).rules;
}

SegmentRules MainWindow::archive_rules() const {
    // The Export panel's choice, kept by Saved Results (whose Export menu
    // had the picker): the profile in use until another is picked.
    return saved_results_panel_->segment_rules();
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
    submit_export(app::tr("export {name}", {{"name", name_of(path_of(path))}}),
                  [bytes = std::move(bytes), out = path_of(path)] {
                      write_bytes(out, bytes);
                      return QStringList{text_of(out)};
                  });
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
                folder, true);
}

void MainWindow::export_when_whole(const RowData& row_data) {
    if (!settings_.after_run_export) {
        return;
    }
    // The row whose entry holds `row_data`'s scores: its own, or the video's
    // it scores another display for.
    std::shared_ptr<RowData> owner;
    for (const std::shared_ptr<RowData>& candidate : rows_) {
        if (row_data.cvvdp_display_of ? !candidate->cvvdp_display_of && candidate->path == *row_data.cvvdp_display_of
                                      : candidate.get() == &row_data) {
            owner = candidate;
            break;
        }
    }
    // Its own video run and finished in this run (not a result left from
    // an earlier one): else its own end exports it, or nothing does.
    const auto owner_job = std::find(job_rows_.begin(), job_rows_.end(), owner);
    if (!owner || owner_job == job_rows_.end() ||
        !finished_jobs_.contains(static_cast<int>(owner_job - job_rows_.begin())) || !owner->completed_run ||
        owner->completed_run->partial || owner->completed_run->record != run_record_) {
        return;
    }
    // Each display row of its in this run, still in the table, over:
    // finished or failed (mark_job_over).
    bool whole = true;
    for (std::size_t job = 0; job < job_rows_.size(); ++job) {
        const RowData& job_row = *job_rows_[job];
        if (job_row.cvvdp_display_of && *job_row.cvvdp_display_of == owner->path && row_index_of(&job_row) &&
            !finished_jobs_.contains(static_cast<int>(job))) {
            whole = false;
        }
    }
    std::erase(auto_export_waiting_, owner);
    if (whole) {
        export_automatically(*owner);
    } else {
        auto_export_waiting_.push_back(owner);
    }
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
            export_indicator_->show_saving(
                app::tr("The bitrate graphs are still being built; the graphs will be exported to {path} "
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
        const QString description = app::tr("export {name}", {{"name", name_of(ready.destination)}});
        submit_export(description, std::move(write), ready.automatic);
        logger.info("Graphs exported to " + platform::path_text(ready.destination) +
                    (missing.empty() ? std::string() : " without the bitrate of " + QStringList(missing.begin(), missing.end()).join(", ").toStdString()));
        if (!missing.empty()) {  // the files are said on the indicator; what they lack, here
            show_status(app::tr("Graphs exported to {path}, without a bitrate graph for {names}: their bitrate "
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
            report_export(refused(one_reference_text()));
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
            report_export(refused(no_conditions_text()));
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
