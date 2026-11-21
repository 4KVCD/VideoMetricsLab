// The main window (the Python app's vmaf_app/ui/main_window.py): the user
// picks a reference video and one or more test videos, sets the options for
// each, runs the metrics, and browses and compares the results.
//
// The test videos are a table: each row shows the file, its media
// information and its scores. Each row has its own VmafOptions (crop
// handling, model, GPU decoding and so on), not one setting shared by all.
//
// Besides this file's: the Options panel (main_window_options.cpp); runs,
// the ticked videos' metrics calculated with their progress and the scores
// shown in the table (main_window_run.cpp); the saved scores, looked up for
// each row, saved as a run goes and forgotten by Recalculate
// (main_window_saved_scores.cpp); saved analysis results
// (main_window_run_files.cpp); and the Metric Graphs and Video Compare tabs'
// glue (main_window_graph.cpp, main_window_compare.cpp).
#pragma once

#include <QFutureWatcher>
#include <QMainWindow>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <atomic>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "dialogs.h"
#include "graph_bundle.h"
#include "probe_worker.h"
#include "results_export.h"
#include "saved_results_panel.h"
#include "row_state.h"
#include "worst_frame_list.h"
#include "vml/analysis_request.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_tools.h"
#include "vml/job_runner.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/run_line.h"
#include "vml/run_record.h"
#include "vml/settings.h"
#include "vml/update_check.h"
#include "vml/video.h"
#include "vml/weak_frames.h"

class QCheckBox;
class QGroupBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QMenu;
class QMessageBox;
class QPushButton;
class QTableWidgetItem;
class QTimer;
class QVBoxLayout;

namespace vml::app {

class CheckableHeaderView;
class ElidedLabel;
class CvvdpDisplayDialog;
class FfmpegDownload;
class FileWrites;
class StreamRecords;
class BdRatePanel;
class BitratePanel;
class FrameComparePanel;
struct FrameComparisonEntry;
class FillColumnTable;
class GraphPanel;
class Notifier;
class OptionsPanel;
class QuietTabWidget;
class RunWorker;
class SettingsPanel;
class VideoColors;

// The test-video table's columns, by their stable physical index (the
// Python app's COL_* numbers): a new column goes last, so that every other
// keeps its index. The order they are shown in comes from metric_columns(),
// which puts VMAF v1 after VMAF NEG.
enum Column : int {
    col_check,
    col_path,
    col_info,
    col_black_bars,
    col_scaling,
    col_bitrate,
    col_psnr,
    col_ssim,
    col_vmaf,
    col_xpsnr,
    col_vmaf_neg,
    col_ssimulacra2,
    col_butteraugli,
    col_cvvdp,
    col_vmaf_v1,
    column_count,
};

// A table column bound to one metric.
struct MetricColumn {
    int column;
    const char* key;
};

// The metric columns, in the order the table shows them.
const std::vector<MetricColumn>& metric_columns();

// The metric column at `column`, or nothing for another column.
const MetricColumn* metric_column_of(int column);

// The window's tabs.
enum Tab : int { tab_videos, tab_graph, tab_frame_compare, tab_bitrate, tab_bd_rate, tab_saved_results, tab_settings };

// A run's own line on Metric Graphs: a name no other run has (the Python
// app's graph_identity, an object of its own).
std::string new_graph_identity();

// A row's own entry in Video Compare (the Python app's frame_identity).
std::string new_frame_identity();

// A run's result on a row (the Python app's CompletedRun).
struct CompletedRun {
    ComparisonResult result;
    // A video's result so far, during its run; its line is updated in place
    // when the rest arrives.
    bool partial = false;
    // Its line on Metric Graphs, kept by a run that replaces it to update
    // the same line (its colour, whether it is hidden or removed stay).
    std::string graph_identity = new_graph_identity();
    // Its name in saved runs and the graphs: the test video's, or the one a
    // saved run or saved scores were given.
    QString label;
    // Each metric column's score, and the frames of its scores that were
    // identical to the reference with how many it has: worked out once, as
    // a long video's take a while and its row is drawn again and again.
    std::map<int, std::optional<double>> scores;
    std::map<int, std::pair<long long, long long>> identical;
    // How it was made, for Export results (run_record.h): its run's facts
    // and the run's record; none for saved scores (whose facts are the
    // row's request as it is exported: saved_facts) or a saved run loaded.
    std::shared_ptr<const RunFacts> facts;
    std::shared_ptr<const RunRecord> record;
    bool from_saved_scores = false;
};

// One row of the test-video table: the video, its options and its results.
struct RowData {
    std::filesystem::path path;
    // The file actually decoded, when `path` is a made-up stand-in (a "Test
    // both directions" companion row, a resolution test): the saved scores
    // are looked up under it.
    std::optional<std::filesystem::path> media_path;
    std::optional<VideoInfo> video_info;
    // Whether its frame rate was asked for as it was added (a raw stream that
    // stores none): after a Cancel, only Calculate metrics asks again.
    bool frame_rate_asked = false;
    // Whether its length was compared with the reference's (warned about
    // once, as it was added or the reference chosen).
    bool length_checked = false;
    VmafOptions options;
    // The range before the Range unit was last switched, restored where the
    // range is switched back with no edit (switch_range_unit).
    std::optional<ComparedRange> range_before_unit_switch;
    // The metrics not run by FFmpeg that the row has ticked.
    std::set<std::string> extra_metric_keys;
    // The display CVVDP models for this row: part of what a CVVDP score
    // means, so part of its saved scores' identity.
    CvvdpSettings cvvdp;
    // CVVDP's scores saved for this comparison with other displays, for the
    // CVVDP cell's tooltip while the row's own display has none.
    std::vector<std::pair<CvvdpSettings, double>> cvvdp_elsewhere;
    // The CVVDP preset the row was given, if any: only tells apart presets
    // with the same display.
    std::string cvvdp_preset;
    // Whether the row's display follows the reference, SDR or HDR (the
    // Automatic default: automatic_cvvdp_preset), until it is given a
    // display of its own.
    bool cvvdp_automatic = false;
    std::optional<RowState> analysis_status;
    // An FFmpeg error, or how many frames a loaded result holds: shown on
    // the file name, the one cell about the whole row.
    QString status_detail;
    // Why each metric failed on the last run, by key, for its red cell.
    std::map<std::string, QString> metric_failures;
    // The scores of the row's last run; nothing before one.
    std::shared_ptr<CompletedRun> completed_run;
    // Its entry in Video Compare while it has no run.
    std::string frame_identity = new_frame_identity();
    // True only for a "Test both" companion row, whose scale direction was
    // set on purpose to the direction the row stands for.
    bool scale_direction_pinned = false;
    // For a row added to score a test video with the other film grain choice
    // (synthetic_film_grain_variant_path): the path of the video's own row.
    // Its choice is held, as its name says it, and so is the video's own
    // row's while the twin is in the table (film_grain_twinned): the two
    // always compare with and without.
    std::optional<std::filesystem::path> film_grain_twin_of;
    // For a row added to score CVVDP alone on another display
    // (synthetic_cvvdp_display_variant_path): the path of the row it scores
    // the display for, whose entry lists it in Export results.
    std::optional<std::filesystem::path> cvvdp_display_of;
    // The metrics (keys) a Recalculate forgot, by the recipe they were
    // forgotten for (its identity's JSON), where another recipe's saved
    // scores would answer them (load_result): Auto's for Source only, the
    // whole videos' for a range. Never taken from those for that recipe, even
    // once the new scores are saved (its own folder answers first then;
    // emptied at the save, the set would let the others back in while the
    // save is still being written); another range of the row still takes
    // them. Cleared when the reference changes (forget_recalculations_unless).
    // Not kept when the app closes: after a restart the other recipes' scores
    // answer them again, which are the same scores.
    std::map<std::string, std::set<std::string>> forgotten_metrics;

    // The file the saved scores are looked up under: the real file when the
    // row's own path is a made-up stand-in.
    const std::filesystem::path& identity_path() const { return media_path ? *media_path : path; }
};

// How the window keeps the PC from sleeping while a run goes on
// (platform::keep_system_awake). The tests put a stand-in in its place, so
// that no test holds the real request.
void set_keep_awake(std::function<bool(bool awake)> keep_awake);

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow();
    // Waits for the videos still being read, and ends a run: a thread must
    // not outlive its window.
    ~MainWindow() override;

    // Asks GitHub for the latest release, without holding the window, and
    // offers it when it is newer and not skipped. Called once, when the app
    // starts (main.cpp): the window never calls it itself, so building a
    // window never uses the network.
    void check_for_updates();

protected:
    void changeEvent(QEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    // ------------------------------------------------------------ the window
    void build_ui();
    QWidget* build_files_panel();
    // build_files_panel's parts.
    void add_reference_row(QVBoxLayout* layout);
    QHBoxLayout* build_table_heading();
    void build_test_table();
    std::vector<QWidget*> add_file_buttons(QVBoxLayout* layout);
    QWidget* build_run_panel();
    void theme_changed();
    void apply_theme_colours();
    void close_when_idle();
    std::vector<QThread*> live_workers() const;
    void store_open_videos();
    void open_videos_changed();
    void save_open_videos();
    void restore_open_videos();
    void reopen_videos(const QString& reference, const QStringList& tests);
    void show_status(const QString& text);
    void save_settings();

    // ---------------------------------------------------------- the settings
    VmafOptions options_from_settings() const;
    std::set<std::string> extra_metrics_from_settings() const;
    // Whether Settings' CVVDP default is Automatic: no preset chosen, or the
    // one chosen gone.
    bool cvvdp_default_automatic() const;
    // The preset a new row starts with: Settings' default, or with Automatic
    // the reference's display (the SDR one before a reference is chosen).
    CvvdpPreset new_row_cvvdp_preset() const;
    // Gives every row whose display follows the reference (cvvdp_automatic)
    // the display of the reference now chosen.
    void follow_reference_cvvdp();
    // Takes the Settings tab's defaults for newly added videos. Rows already
    // in the table keep what they were given.
    void on_default_settings_changed();

    // ------------------------------------------------------- Video Compare
    // Makes Video Compare list every row it can show, scored or not.
    void sync_frame_compare();
    // A row's entry in Video Compare; nothing while it cannot be shown.
    std::optional<FrameComparisonEntry> frame_comparison_entry(const RowData& row_data) const;
    // Keeps the black bars found in the saved scores' folder now named.
    void use_crop_store();

    // ------------------------------------------------------- the update check
    // GitHub's answer (`body`), or why it could not be asked (`error`).
    void on_update_answer(const std::string& body, const QString& error);
    // Offers a newer release, unless the user chose to skip it.
    void offer_update(const Release& release);

    // ------------------------------------------------------ FFmpeg's presence
    bool check_ffmpeg(bool prompt);
    bool on_locate_ffmpeg();
    // Downloads FFmpeg into the app's own folder and uses it (Windows).
    void on_download_ffmpeg();
    void on_ffmpeg_downloaded(const std::filesystem::path& folder);
    // Hides the FFmpeg banner and says which ffmpeg is used now.
    void ffmpeg_found(const ToolsStatus& status, const std::filesystem::path& directory);

    // ------------------------------------------------------------ the source
    void on_browse_source();
    void choose_source(const std::filesystem::path& path);
    QString shown_source_path() const;
    void on_source_path_entered();
    QString video_dialog_dir(const QString& kind) const;
    void remember_video_dir(const QString& kind, const std::filesystem::path& path);
    void start_source_probe(const std::filesystem::path& path);
    void on_source_probed(int generation, const Probed& probed);
    void forget_recalculations_unless(const std::filesystem::path& reference);
    void apply_source_info(const std::filesystem::path& path, const VideoInfo& info);
    void check_rows_against_new_source(const std::filesystem::path& path);
    void invalidate_completed_results(const std::vector<int>& rows);
    void show_reading(const QString& text);
    void show_ready();
    void on_source_probe_finished(int generation, ProbeWorker* worker);
    void retire(ProbeWorker* worker);
    std::vector<QString> remove_rows_owned_by_the_previous_source();

    // ----------------------------------------------------- the test videos
    int add_table_row(const std::filesystem::path& path);
    void set_row_metrics(int row);
    // set_row_metrics's three kinds of metric cell.
    void show_score_cell(QTableWidgetItem& item, const RowData& row_data, CompletedRun& run,
                         const MetricColumn& metric, double score);
    void show_unavailable_cell(QTableWidgetItem& item, const QString& reason);
    void show_tick_cell(QTableWidgetItem& item, const RowData& row_data, const MetricColumn& metric,
                        bool enabled);
    // A row's detail for its scores: how many frames were scored, and which
    // metrics ("{count} scored frames; metrics: VMAF, PSNR").
    QString scores_detail(const ComparisonResult& result) const;
    RowState row_state(const RowData& row_data) const;
    void refresh_row_state(int row);
    void set_row_black_bars(int row, bool probe_failed = false);
    std::pair<QString, QString> resize_mismatch(int row) const;
    void fit_columns(std::initializer_list<int> columns);
    void fit_pending_columns();
    void set_row_scaling(int row);
    void set_row_info(int row, const std::optional<VideoInfo>& info, const QString& error = QString());
    void set_row_status(int row, RowState state, const QString& detail = QString());
    void on_add_distorted();
    void on_files_dropped_on_table(const QStringList& paths);
    void add_test_videos(const std::vector<std::filesystem::path>& paths);
    void on_add_resample_test();
    void add_resample_test(const ResampleTarget& target);
    void set_resample_row_info(int row, const ResampleTarget& target);
    void start_media_probe(const std::vector<std::filesystem::path>& paths);
    void on_probed(const Probed& probed);
    void on_probe_finished(ProbeWorker* worker);
    // `worker` (none: a lane that ended without saying so) has read every
    // video's details; its rates may still be coming.
    void on_lane_probed(ProbeWorker* worker);
    void on_rate_estimated(const RateEstimated& estimated);
    void on_source_lane_probed(int generation, ProbeWorker* worker);
    // The current rate of `info`'s file, as the reference or a row has it
    // now, into `info`; unchanged when no row or reference is that file.
    void take_current_rate(VideoInfo& info) const;
    void take_current_rates(ComparisonResult& result) const;
    void set_row_bitrate(int row, const VideoInfo& info);
    // Asks the frame rate of the videos just added that have none (raw streams
    // that store none), and of the reference.
    void ask_new_frame_rates();
    // Warns, once for each, of the test videos just added (or all of them,
    // after a new reference) that do not last as long as the reference.
    void warn_about_lengths();
    // The same for the reference (`reference`) and `videos`, and enters the
    // answers. False once one is cancelled.
    bool ask_frame_rates(bool reference, const std::vector<std::filesystem::path>& videos);
    void enter_frame_rate(const std::filesystem::path& path, double fps);
    void enter_reference_frame_rate(const std::filesystem::path& path, double fps);
    void apply_row_order(const std::vector<int>& order);
    void on_rows_dragged(const QList<int>& rows, int target);
    void on_header_clicked(int column);
    void clear_sort_indicator();
    std::optional<int> row_index_of_path(const std::filesystem::path& path) const;
    void on_remove_distorted();
    void on_remove_all_distorted();
    void on_table_selection_changed();
    void on_highlight_toggled(bool on);
    void schedule_best_worst();
    void apply_best_worst();

    // ----------------------------------------------------------- the metrics
    bool default_metric_ticked(int column) const;
    static bool metric_enabled(const VmafOptions& options, int column);
    static void set_metric_option(VmafOptions& options, int column, bool checked);
    static std::vector<std::string> selected_metrics(const RowData& row_data);
    std::vector<std::string> requested_metrics(const RowData& row_data) const;
    // The facts of `row_data`'s saved scores, shown as `shown`.
    std::shared_ptr<const RunFacts> saved_facts(const RowData& row_data, const ComparisonResult& shown) const;
    static std::optional<QString> metric_unavailable_reason(const RowData& row_data, const std::string& key);
    static bool row_metric_enabled(const RowData& row_data, int column);
    void apply_metric_visibility();
    void show_metrics_menu(const QPoint& position);
    void on_metric_visibility_toggled(const std::string& key, bool shown, QMenu* menu);
    void apply_metric_selection(const std::vector<int>& rows, int column, bool checked, bool set_default);
    void on_table_item_changed(QTableWidgetItem* item);
    void on_metric_column_toggled(int column, bool checked);

    // ------------------------------------------------------ the saved scores
    std::filesystem::path cache_folder() const;
    AnalysisRequest analysis_request_of(const RowData& row_data) const;
    // Whether a lookup started (none with saved scores off, no reference, or
    // the window closing).
    bool start_cache_lookup(const std::vector<std::filesystem::path>& paths);
    // The lookup of the rows only their whole test stream can tell for a
    // remux (stream_check_paths_), after the others; and its end.
    void start_stream_check();
    void cancel_stream_check();
    void on_stream_check_finished(ProbeWorker* worker);
    // A stream being read said on the status line (only while no run uses
    // it), and taken down once done if it is still what the line says.
    void show_stream_status(const QString& text);
    void end_stream_status();
    // The stream fingerprinter with `ffmpeg` and `ffprobe`, that reads
    // nothing while Settings' "Recognize test videos remuxed" is off.
    void use_stream_fingerprinter(const std::string& ffmpeg, const std::string& ffprobe);
    void sync_remux_setting();
    void reload_saved_scores(const std::vector<int>& rows);
    std::vector<std::filesystem::path> cancel_cache_lookup();
    void on_cache_lookup_finished(ProbeWorker* worker);
    void on_saved_found(int generation, const SavedScoresFound& found);
    void on_other_displays_found(int generation, const OtherDisplaysFound& found);
    void show_table_menu(const QPoint& position);
    void recompute_rows(const std::vector<int>& rows, bool forget_saved = true);
    void on_saved_results_cleared();
    void drop_metric_results(int row, const std::vector<std::string>& keys);
    QString cvvdp_elsewhere_note(const RowData& row_data) const;

    // ------------------------------------- the Options panel (main_window_options.cpp)
    void write_panel_options(const VmafOptions& options);
    VmafOptions read_panel_options() const;
    void on_panel_field_edited(const QString& field);
    void on_model_changed(int index);
    void on_scale_direction_combo_changed(int index);
    int add_companion_row(int row, const std::filesystem::path& path, const std::function<void(RowData&)>& differ);
    void add_opposite_scale_direction_rows(const std::vector<int>& rows);
    void apply_cvvdp(const std::function<CvvdpSettings(const CvvdpSettings&)>& change,
                     const std::optional<std::vector<int>>& rows = std::nullopt,
                     const std::optional<std::string>& preset_name = std::nullopt,
                     std::optional<bool> automatic = std::nullopt);
    void on_cvvdp_preset_chosen(int index);
    void on_cvvdp_resize_toggled(bool checked);
    std::set<std::string> cvvdp_preset_names() const;
    void on_cvvdp_add_preset();
    void on_cvvdp_edit_display();
    void save_cvvdp_preset(const CvvdpDisplayDialog& dialog, const CvvdpSettings& settings,
                           const std::optional<std::string>& replacing, bool apply_to_rows = true);
    void on_cvvdp_delete_preset();
    // "Also score CVVDP on...": offers the presets, and for the one chosen
    // adds a row to each of `rows` scoring CVVDP alone on it.
    void on_cvvdp_other_display_requested();
    void add_cvvdp_display_rows(const std::vector<int>& rows, const CvvdpPreset& preset);
    // The film grain choice (VideoInfo::with_film_grain), main_window_options.cpp.
    std::vector<std::filesystem::path> choose_film_grain(const std::vector<int>& rows, bool reference, bool with);
    std::vector<std::filesystem::path> apply_film_grain_default();
    // Offers, for each of `paths` already in the table whose video has film
    // grain to choose, to add it again with the other choice.
    void offer_film_grain_twins(const std::vector<std::filesystem::path>& paths);
    void add_film_grain_twin(int row);
    // Whether `row_data`'s film grain choice is held: a twin row, or the row a
    // twin in the table was made from.
    bool film_grain_twinned(const RowData& row_data) const;
    // Fills a Manual row's crops, where none is entered yet, from the boxes
    // its last run cut; whether it filled any.
    bool fill_manual_crops(RowData& row_data);
    void on_film_grain_chosen(bool with);
    void show_panel_film_grain();
    // The reference's line under its path: its media info, bitrate, length,
    // and its film grain in the tooltip.
    void show_source_label();

    // ------------------------------------------------------- Metric Graphs
    void on_tab_changed(int index);
    void sync_graph();
    void open_or_update_graph(const std::vector<std::shared_ptr<CompletedRun>>& runs, bool show = true);
    // Video Compare at `frame` of the pair `identity` (a graph line's), or
    // why it cannot show it; with Worst frames' `list` to step through, if
    // opened from it.
    void show_in_video_compare(const std::string& identity, int frame, const WorstFrameList& list);
    void on_show_graph_clicked();

    // -------------------------------------- saved runs (main_window_run_files.cpp)
    void on_load_saved_run();
    void load_saved_run(const std::filesystem::path& path);
    void adopt_source_from_run(const ComparisonResult& result);
    void read_details_of_loaded_run(const ComparisonResult& result);
    void take_read_details_of_file(const Probed& probed);
    bool offer_to_switch_source(const ComparisonResult& result);
    bool offer_to_take_reference_film_grain(const ComparisonResult& result);
    std::vector<std::shared_ptr<CompletedRun>> selected_runs() const;
    void on_save_selected();
    void submit_save(const ComparisonResult& result, const std::filesystem::path& path, const QString& label);
    std::vector<ExportedRun> exported_runs() const;

    // ------------------------------------- graph exports (main_window_exports.cpp)
    // Graphs to save as one .zip file (graph_bundle.h) once the bitrate they
    // need is read: of `videos`, compared with `reference`, with `extra`
    // files (a video's results); `together`: the videos' graphs together as
    // well; `automatic`: Settings' export after a run, which says less.
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
        // The .zip replaces no file at `destination`: it takes a free name
        // beside it (Settings' after-run export, named after the video).
        bool numbered = false;
        // Saved comparisons' files drawn as their bitrate was saved
        // (saved_bitrates).
        SavedBitrates saved_bitrate;
        // Per-frame scores, made with the files (made_files).
        std::vector<FrameFiles> frames;
    };
    // The scored test videos of the reference the Videos tab compares with,
    // in its order; a row's, when it is one of them.
    std::vector<BundleVideo> scored_videos() const;
    std::optional<BundleVideo> scored_video(const RowData& row_data) const;
    // The row whose scored video (scored_video) is Metric Graphs' line
    // `identity`; none for a line of a saved run loaded, or of a row whose
    // result is not one to export.
    std::optional<int> row_of_graph(const std::string& identity) const;
    // Every scored test video's weak segments (scored_videos) by `rules`
    // (the saved ones, or the Weak segments dialog's), in one JSON or CSV
    // file the user names.
    void export_segments_report(const SegmentRules& rules);
    // `videos`' weak segments by `rules` in one JSON or CSV file the user
    // names, suggested after `reference` (export_segments_report, and the
    // Saved results tab's).
    void save_segments_report(const std::vector<SegmentsReportVideo>& videos, const SegmentRules& rules,
                              const std::filesystem::path& reference);
    // The Saved results tab's exports of the comparisons it read.
    void export_saved_results(SavedResultsPanel::Export kind, const std::vector<LoadedSavedResult>& results);
    // The rules of the weak segment profile in use (Settings > Weak segments).
    SegmentRules saved_weak_rules() const;
    void on_export_all_graphs();
    // What an export holds, by Settings' Exports include: for one video, its
    // results (alone in the document) and its weak segments; for several,
    // the run's results document and their weak segments report; and the
    // graphs, of each (and all together, with several). Named after `name`.
    struct ExportPlan {
        std::vector<BundleVideo> videos;
        std::filesystem::path reference;
        std::vector<BundleFile> files;
        std::vector<FrameFiles> frames;  // made as they are written
        bool graphs = false;
        bool several = false;
        QString name;
        SavedBitrates saved_bitrate;  // PendingBundle's
    };
    ExportPlan export_plan(std::vector<BundleVideo> videos, const std::filesystem::path& reference,
                           const std::function<nlohmann::ordered_json()>& document, const SegmentRules& rules,
                           bool several, const QString& name);
    // Saves `plan` as Settings' Exports include say (a .zip file, or loose
    // files): asked where, or into `folder` after a run.
    void save_export(ExportPlan plan, const std::optional<std::filesystem::path>& folder);
    // The right-click menus' exports: the video of `row`, and all the
    // videos (the Videos tab's run, every video Export results lists).
    void export_this_video(int row);
    void export_all_videos();
    // The Saved results tab's: the comparisons selected.
    void export_saved_contents(const std::vector<LoadedSavedResult>& results);
    // The bitrate the Saved results tab's exports draw for the selected
    // comparisons' files that are not those their scores were saved for
    // (gone, or another file there now): as it was saved with them.
    SavedBitrates saved_bitrates(const std::vector<LoadedSavedResult>& results) const;
    // Settings > Storage > After each run, for a video just scored: what
    // Exports include, next to it or in the export folder.
    void export_automatically(const RowData& row_data);
    // The document Export results makes with `row_data`'s video alone.
    std::optional<nlohmann::ordered_json> row_document(const RowData& row_data) const;
    // Saves `bundle` when its bitrate is read, starting to read what is not.
    void queue_bundle(PendingBundle bundle);
    void write_ready_bundles();

    // ------------------------------------------------------------ the scores
    std::vector<std::string> reusable_metric_keys(const RowData& row_data) const;
    MetricResultSet reusable_results(const RowData& row_data) const;
    bool has_requested_results(const RowData& row_data) const;
    static bool backend_matches(const RowData& row_data, const std::string& key, const MetricProvenance& provenance);
    static QString other_scale_note();
    static std::optional<VshipBuild> other_vship_build(const RowData& row_data, const MetricProvenance& provenance);
    static std::optional<double> score_of(CompletedRun& run, int column);
    static QString identical_frames_note(CompletedRun& run, int column);
    QString cvvdp_note(const RowData& row_data) const;
    static QString crop_detail(const QString& label, const VideoInfo& info, const std::optional<CropBox>& crop);

    // ---------------------------------------------------- the run (main_window_run.cpp)
    // The status line during a run says how much is done; this holds
    // something else there: "Paused" before the counts, or a message alone
    // ("Cancelling...").
    enum class RunHold { none, paused, message };

    std::vector<int> checked_rows() const;
    void on_run_clicked();
    // on_run_clicked's parts.
    bool ready_to_run(const std::vector<int>& checked);
    void show_already_scored(const std::vector<int>& checked, long long count);
    std::optional<std::vector<VmafJob>> plan_jobs(const std::vector<int>& rows,
                                                  std::vector<std::shared_ptr<RowData>>& job_rows);
    std::variant<VmafJob, QString> job_for(const RowData& row_data) const;
    void begin_run(const std::vector<int>& checked, std::vector<std::shared_ptr<RowData>> job_rows,
                   long long already_scored);
    void start_run_worker(std::vector<VmafJob> jobs);
    bool confirm_skipping(const std::vector<std::pair<std::shared_ptr<RowData>, QString>>& skipped, bool others);
    bool confirm_long_cpu_perceptual(const std::vector<std::shared_ptr<RowData>>& job_rows);
    // Starts waiting for the GPU checks the run needs, unless they are done:
    // true while they are not, the run then starting by itself.
    bool wait_for_gpu_checks(const std::vector<int>& checked);
    void stop_waiting_for_gpu_checks();
    // The checked rows' and the reference's film grain choices, made before
    // the run: Settings' default, or asked. False when the run does not start
    // now: the question was cancelled, or the saved scores of rows whose
    // choice changed are looked up first (the run then starts by itself).
    bool settle_film_grain(const std::vector<int>& checked);
    // The frame rates the checked rows and the reference still lack, asked for
    // before the run. False when the question was cancelled: the run does not start.
    bool settle_frame_rates(const std::vector<int>& checked);
    bool wait_for_saved_scores();
    void stop_waiting_for_saved_scores();
    // While a run waits for something before it starts: Run off, Cancel on,
    // and the status line saying what (put back after, unless it changed).
    void show_waiting_to_run(const QString& text);
    void stop_showing_waiting_to_run(const QString& text);
    void refresh_metric_cells();
    void on_vship_checked();
    void set_run_ui_active(bool active);
    int parallel_jobs() const;
    void on_parallel_jobs_changed(bool on);
    void on_cancel_clicked();
    void on_pause_clicked();
    void on_job_started(int index);
    void redraw_job_lines();
    void set_job_line(int index, const QString& text, const QString& tooltip = QString());
    void arrange_job_lines();
    void mark_job_over(int index);
    QString job_label(int index) const;
    void update_run_status();
    std::optional<double> run_elapsed() const;
    QString run_end_message() const;
    QString run_summary(int running, int total) const;
    void pause_free_rates(int index, std::vector<TaskSnapshot>& tasks);
    void on_task_progress(int index, std::vector<TaskSnapshot> tasks);
    void note_fallbacks(int index, const std::vector<TaskSnapshot>& tasks);
    QString with_fallbacks(int index, const QString& detail) const;
    int fallback_count() const;
    void on_run_tick();
    void render_job_progress(int index);
    std::optional<int> row_index_of(const RowData* row_data) const;
    void on_result_updated(int index, const ComparisonResult& result);
    std::optional<int> on_job_finished(int index, const ComparisonResult& result, bool final = true);
    // The run's record, completed as the run ends (run_record_).
    void complete_run_record();
    // This machine, with the FFmpeg the window found.
    MachineFacts current_machine() const;
    // on_job_finished's parts.
    void save_job_result(std::size_t job, const RowData& row_data, const ComparisonResult& result,
                         const QString& label);
    std::shared_ptr<CompletedRun> make_completed_run(const RowData& row_data, const ComparisonResult& result,
                                                     const QString& label, bool final);
    void on_job_partially_failed(int index, const ComparisonResult& result, const QString& message,
                                 const QString& stderr_tail, const MetricFailures& failures);
    void on_job_failed(int index, const QString& message, const QString& stderr_tail);
    void on_all_finished();
    // Sends the run's notification, where Settings > Notifications asks for
    // one (on_all_finished), and says how it went in the status bar and log.
    void notify_run_end();
    void on_run_thread_finished(RunWorker* worker);

    Settings settings_;
    std::optional<VideoInfo> source_info_;
    // The rows, in the table's order. Held by pointer: work that outlives a
    // reorder (a probe, a run) finds its row by identity or by path, never
    // by position.
    std::vector<std::shared_ptr<RowData>> rows_;
    std::vector<ProbeWorker*> probe_workers_;
    // The videos whose saved scores are looked up again once their probe
    // lane ends: reading them changed what the scores are filed under.
    std::vector<std::filesystem::path> probed_lookups_;
    ProbeWorker* source_probe_worker_ = nullptr;
    int source_probe_generation_ = 0;
    std::optional<std::filesystem::path> source_reading_;  // the reference being read, until it is
    std::set<int> columns_to_fit_;
    QStringList reference_drop_;  // the files a drag over the reference box carries
    QString reading_text_;        // the status line's "Reading..." while it shows
    bool run_active_ = false;
    bool defaults_changed_in_run_ = false;  // Settings' defaults changed during a run: applied as it ends
    bool closing_ = false;
    bool asking_frame_rates_ = false;           // a frame rate question is open
    bool reference_frame_rate_asked_ = false;   // RowData::frame_rate_asked, for the reference
    // Whether Vship-fast's metrics were last found to run on 5.1.2 here
    // (on_vship_checked).
    bool vship51_in_use_ = false;
    // The window, for the thread that looks for the videos open last time
    // (restore_open_videos), which may outlive it: nullptr once it is gone.
    struct OpenVideosLookup {
        std::mutex mutex;
        MainWindow* window = nullptr;
    };
    std::shared_ptr<OpenVideosLookup> open_videos_lookup_;
    // From restore_open_videos until reopen_videos: the table does not hold
    // the videos open last time yet, so they are not saved over.
    bool open_videos_restoring_ = false;
    // save_open_videos once the window is next idle: one write for every
    // change to the videos that lands in one go.
    QTimer* open_videos_timer_ = nullptr;
    bool syncing_panel_ = false;
    bool syncing_table_ = false;
    bool theme_refresh_due_ = true;  // until the window is built
    std::optional<std::tuple<bool, unsigned, unsigned>> theme_applied_;
    std::optional<int> sort_column_;
    Qt::SortOrder sort_order_ = Qt::AscendingOrder;

    // What a newly added row starts with: each row's own settings are edited
    // in the Options panel.
    VmafOptions default_options_;
    std::set<std::string> default_extra_metric_keys_;
    std::set<std::string> hidden_metrics_;

    // The Options panel, the rows it edits (the selected ones, top to
    // bottom), and the GPUs found here, which it offers.
    OptionsPanel* options_panel_ = nullptr;
    std::vector<int> panel_target_rows_;
    std::vector<GpuVendor> detected_gpu_vendors_;
    // Opens the CVVDP display editor and waits for it (exec); the tests
    // answer it in its place, as a user would.
    std::function<int(CvvdpDisplayDialog&)> run_cvvdp_dialog_;
    // The other dialogs that wait for the user, which the tests answer in
    // the user's place.
    Dialogs dialogs_;

    // The lookup of saved scores under way: the rows it was asked about, and
    // a number each lookup raises, with which the window refuses the answers
    // of one it replaced (they were for the reference and settings then).
    ProbeWorker* cache_worker_ = nullptr;
    std::vector<std::filesystem::path> cache_lookup_paths_;
    // Rows the lookup could tell for a remux of a saved encode only by
    // reading their whole test stream, and the lookup that does so once the
    // others are in (start_stream_check): nothing waits for it, and a run
    // cancels it.
    std::vector<std::filesystem::path> stream_check_paths_;
    ProbeWorker* stream_check_worker_ = nullptr;
    std::vector<std::filesystem::path> stream_checking_paths_;  // the rows it reads
    // Settings' "Recognize test videos remuxed", as the lookups' and the
    // records' threads read it (sync_remux_setting).
    std::shared_ptr<std::atomic<bool>> recognise_remuxes_ = std::make_shared<std::atomic<bool>>(false);
    // Reads a test video's stream with the app's FFmpeg: the saved scores'
    // stream fingerprinter while the setting is on, none while it is off.
    std::function<std::optional<std::string>(const std::filesystem::path&, int, const CancelToken&)> stream_reader_;
    // The status line's text about a stream being read, to be taken down
    // when the reading ends if nothing replaced it.
    QString stream_status_;
    int cache_generation_ = 0;
    FileWrites* file_writes_ = nullptr;
    // Test streams read after their scores are saved: shared with the writes
    // that queue them, which can outlive the window's other members.
    std::shared_ptr<StreamRecords> stream_records_;

    // The run under way, or the last one, by its jobs' indexes. Its jobs'
    // rows are held as rows, not positions, so that a row moved or removed
    // during the run never takes another's result; with the options and
    // display each job started with.
    RunWorker* worker_ = nullptr;
    std::vector<std::shared_ptr<RowData>> job_rows_;
    std::vector<VmafOptions> job_options_;
    std::vector<CvvdpSettings> job_cvvdp_;
    std::vector<int> running_jobs_;  // in the order they started
    std::set<int> finished_jobs_;
    int run_failed_count_ = 0;
    Notifier* notifier_ = nullptr;  // made at the first run-end notification
    int run_partial_count_ = 0;  // videos with some metrics failed, the rest scored
    int run_skipped_ = 0;        // ticked videos left out, as they had every score
    bool run_was_cancelled_ = false;
    // The run's clock, in seconds: started, paused in all, paused since.
    std::optional<double> run_started_at_;
    double paused_total_ = 0;
    std::optional<double> paused_since_;
    // The run's record, completed as it ends, and each job's facts
    // (RunWorker::job_facts), for the CompletedRuns it makes.
    std::shared_ptr<RunRecord> run_record_;
    std::vector<std::shared_ptr<const RunFacts>> job_facts_;
    RunHold run_hold_ = RunHold::none;
    QString run_hold_message_;
    // Each video's line: the latest state of its tasks, the line's text and
    // tooltip, which of the lines shows it, what it shows apart from its
    // figures, and whether its figures are due on the next tick.
    std::map<int, std::vector<TaskSnapshot>> job_task_progress_;
    std::map<int, std::pair<QString, QString>> job_line_text_;
    std::map<int, int> job_line_slot_;
    std::map<int, LineShape> job_line_shape_;
    std::set<int> job_lines_due_;
    // Each video's GPU fallbacks (a GPU route that failed, and where its
    // metrics went instead), as the engine says them (fallback_note): on its
    // row once it is over, and counted in the run's end message.
    std::map<int, std::vector<std::string>> job_fallbacks_;
    // Each running task's recent progress on the run's clock, by job, task
    // and pass: (seconds, frames done).
    std::map<std::tuple<int, std::string, long long>, std::deque<std::pair<double, long long>>> rate_samples_;
    QTimer* run_tick_timer_ = nullptr;
    // apply_best_worst once the window is next idle: one pass over the table
    // for every change that lands in one go (a run's results, the saved
    // scores', rows removed).
    QTimer* best_worst_timer_ = nullptr;
    // Seconds on a steady clock; a clock of their own in the tests.
    std::function<double()> clock_;

    QLabel* ffmpeg_banner_ = nullptr;
    QPushButton* locate_ffmpeg_button_ = nullptr;
    QPushButton* download_ffmpeg_button_ = nullptr;  // Windows only
    FfmpegDownload* ffmpeg_download_ = nullptr;  // while one runs
    QuietTabWidget* tabs_ = nullptr;
    QGroupBox* files_box_ = nullptr;
    QLineEdit* source_edit_ = nullptr;
    QPushButton* source_browse_button_ = nullptr;
    QLabel* source_info_label_ = nullptr;
    QPushButton* metrics_button_ = nullptr;
    FillColumnTable* distorted_table_ = nullptr;
    CheckableHeaderView* metric_header_ = nullptr;
    QPushButton* remove_all_button_ = nullptr;
    QCheckBox* highlight_check_ = nullptr;
    std::vector<QWidget*> file_action_widgets_;
    QPushButton* run_button_ = nullptr;
    QPushButton* pause_button_ = nullptr;
    QPushButton* cancel_button_ = nullptr;
    QCheckBox* parallel_jobs_check_ = nullptr;
    QPushButton* load_button_ = nullptr;
    // The videos' colours, shared by the graphs, and Metric Graphs.
    VideoColors* video_colors_ = nullptr;
    GraphPanel* graph_panel_ = nullptr;
    FrameComparePanel* frame_compare_panel_ = nullptr;
    BitratePanel* bitrate_panel_ = nullptr;
    BdRatePanel* bd_rate_panel_ = nullptr;
    // The newer release offered, while its message is open.
    QPointer<QMessageBox> update_box_;
    // The GPU checks a run is waiting for (wait_for_gpu_checks), while it is,
    // and what the status line said before it waited for something.
    QFutureWatcher<void>* gpu_check_watcher_ = nullptr;
    QString status_before_waiting_;
    // Whether a run asked for waits for the saved scores of rows whose film
    // grain choice changed (settle_film_grain).
    bool run_waits_for_saved_scores_ = false;
    // The Settings tab, and the latest check of FFmpeg, which it shows.
    SavedResultsPanel* saved_results_panel_ = nullptr;
    SettingsPanel* settings_panel_ = nullptr;
    ToolsStatus tools_status_;
    // The rows ticked when the run started, those already scored too: their
    // results go to the graph when it ends.
    std::vector<std::shared_ptr<RowData>> checked_rows_for_run_;
    QPushButton* save_button_ = nullptr;
    QPushButton* show_graph_button_ = nullptr;
    QPushButton* export_graphs_button_ = nullptr;
    // Graph exports waiting for their bitrate (queue_bundle), and the timer
    // that looks whether it is read.
    std::vector<PendingBundle> pending_bundles_;
    QTimer* bundle_timer_ = nullptr;
    QLabel* status_label_ = nullptr;
    std::vector<ElidedLabel*> job_progress_labels_;
};

}  // namespace vml::app
