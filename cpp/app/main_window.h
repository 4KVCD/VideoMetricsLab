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

#include <QMainWindow>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <array>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "probe_worker.h"
#include "row_state.h"
#include "vml/analysis_request.h"
#include "vml/cvvdp.h"
#include "vml/ffmpeg_tools.h"
#include "vml/job_runner.h"
#include "vml/options.h"
#include "vml/results.h"
#include "vml/settings.h"
#include "vml/update_check.h"
#include "vml/video.h"

class QCheckBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QMessageBox;
class QPushButton;
class QTableWidgetItem;
class QTimer;

namespace vml::app {

class CheckableHeaderView;
class ElidedLabel;
class CvvdpDisplayDialog;
class FileWrites;
class BitratePanel;
class FrameComparePanel;
struct FrameComparisonEntry;
class FillColumnTable;
class GraphPanel;
class OptionsPanel;
class QuietTabWidget;
class RunWorker;
class SettingsPanel;
class VideoColors;

// The test-video table's columns, by their stable physical index (the
// Python app's COL_* numbers). VMAF v1 was added last, so every older column
// keeps its index; it is shown after VMAF NEG.
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

// The window's tabs.
enum Tab : int { tab_videos, tab_graph, tab_frame_compare, tab_bitrate, tab_settings };

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
};

// One row of the test-video table: the video, its options and its results.
struct RowData {
    std::filesystem::path path;
    // The file actually decoded, when `path` is a made-up stand-in (a "Test
    // both directions" companion row, a resolution test): the saved scores
    // are looked up under it.
    std::optional<std::filesystem::path> media_path;
    std::optional<VideoInfo> video_info;
    VmafOptions options;
    // The metrics not run by FFmpeg that the row has ticked.
    std::set<std::string> extra_metric_keys;
    // Whether each Vship metric runs on the GPU or the CPU: where it runs,
    // never a score or its saved scores' identity.
    std::map<std::string, std::string> metric_backends{{"ssimulacra2", "gpu"}, {"butteraugli", "gpu"}};
    // The display CVVDP models for this row: part of what a CVVDP score
    // means, so part of its saved scores' identity.
    CvvdpSettings cvvdp;
    // CVVDP's scores saved for this comparison with other displays, for the
    // CVVDP cell's tooltip while the row's own display has none.
    std::vector<std::pair<CvvdpSettings, double>> cvvdp_elsewhere;
    // The CVVDP preset the row was given, if any: only tells apart presets
    // with the same display.
    std::string cvvdp_preset;
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
    friend class WindowTest;  // the window's tests (tests/app)

    // ------------------------------------------------------------ the window
    void build_ui();
    QWidget* build_files_panel();
    QWidget* build_run_panel();
    void theme_changed();
    void apply_theme_colours();
    void close_when_idle();
    std::vector<QThread*> live_workers() const;
    void store_open_videos();
    void restore_open_videos();
    void reopen_videos(const QString& reference, const QStringList& tests);
    void show_status(const QString& text);
    void save_settings();

    // ---------------------------------------------------------- the settings
    VmafOptions options_from_settings() const;
    std::set<std::string> extra_metrics_from_settings() const;
    CvvdpSettings cvvdp_from_settings() const;
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

    // ------------------------------------------------------------ the source
    void on_browse_source();
    void choose_source(const std::filesystem::path& path);
    QString shown_source_path() const;
    void on_source_path_entered();
    QString video_dialog_dir(const QString& kind) const;
    void remember_video_dir(const QString& kind, const std::filesystem::path& path);
    void start_source_probe(const std::filesystem::path& path);
    void on_source_probed(int generation, const Probed& probed);
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
    void start_cache_lookup(const std::vector<std::filesystem::path>& paths);
    void reload_saved_scores(const std::vector<int>& rows);
    std::vector<std::filesystem::path> cancel_cache_lookup();
    void on_cache_lookup_finished(ProbeWorker* worker);
    void on_saved_found(int generation, const SavedScoresFound& found);
    void on_other_displays_found(int generation, const OtherDisplaysFound& found);
    void show_table_menu(const QPoint& position);
    void recompute_rows(const std::vector<int>& rows);
    void drop_metric_results(int row, const std::vector<std::string>& keys);
    QString cvvdp_elsewhere_note(const RowData& row_data) const;

    // ------------------------------------- the Options panel (main_window_options.cpp)
    void write_panel_options(const VmafOptions& options);
    VmafOptions read_panel_options() const;
    void on_panel_field_edited(const QString& field);
    void on_metric_backend_changed(const QString& metric_key);
    void on_model_changed(int index);
    void on_scale_direction_combo_changed(int index);
    void add_opposite_scale_direction_rows(const std::vector<int>& rows);
    void apply_cvvdp(const std::function<CvvdpSettings(const CvvdpSettings&)>& change,
                     const std::optional<std::vector<int>>& rows = std::nullopt,
                     const std::optional<std::string>& preset_name = std::nullopt);
    void on_cvvdp_preset_chosen(int index);
    void on_cvvdp_resize_toggled(bool checked);
    std::set<std::string> cvvdp_preset_names() const;
    void on_cvvdp_add_preset();
    void on_cvvdp_edit_display();
    void save_cvvdp_preset(const CvvdpDisplayDialog& dialog, const CvvdpSettings& settings,
                           const std::optional<std::string>& replacing, bool apply_to_rows = true);
    void on_cvvdp_delete_preset();

    // ------------------------------------------------------- Metric Graphs
    void on_tab_changed(int index);
    void sync_graph();
    void open_or_update_graph(const std::vector<std::shared_ptr<CompletedRun>>& runs, bool show = true);
    void on_show_graph_clicked();

    // -------------------------------------- saved runs (main_window_run_files.cpp)
    void on_load_saved_run();
    void load_saved_run(const std::filesystem::path& path);
    void adopt_source_from_run(const ComparisonResult& result);
    bool offer_to_switch_source(const ComparisonResult& result);
    std::vector<std::shared_ptr<CompletedRun>> selected_runs() const;
    void on_save_selected();
    void submit_save(const ComparisonResult& result, const std::filesystem::path& path, const QString& label);

    // ------------------------------------------------------------ the scores
    std::vector<std::string> reusable_metric_keys(const RowData& row_data) const;
    MetricResultSet reusable_results(const RowData& row_data) const;
    bool has_requested_results(const RowData& row_data) const;
    static bool backend_matches(const RowData& row_data, const std::string& key, const MetricProvenance& provenance);
    static QString backend_note(const RowData& row_data, const std::string& key, const FrameMetricResult* metric);
    static std::optional<double> score_of(CompletedRun& run, int column);
    static QString identical_frames_note(CompletedRun& run, int column);
    QString cvvdp_note(const RowData& row_data) const;
    static QString crop_detail(const QString& label, const VideoInfo& info, const std::optional<CropBox>& crop);

    // ---------------------------------------------------- the run (main_window_run.cpp)
    // The status line during a run says how much is done; this holds
    // something else there: "Paused" before the counts, or a message alone
    // ("Cancelling...").
    enum class RunHold { none, paused, message };
    // What a video's line shows apart from its figures, which redraws it at
    // once when it changes: each task's state, pass, wait, and metrics moved
    // to the CPU or done.
    using LineShape = std::vector<std::tuple<std::string, std::optional<std::array<long long, 3>>, std::string,
                                             std::vector<std::string>, std::vector<std::string>>>;

    std::vector<int> checked_rows() const;
    void on_run_clicked();
    bool confirm_skipping(const std::vector<std::pair<std::shared_ptr<RowData>, QString>>& skipped, bool others);
    bool confirm_long_cpu_perceptual(const std::vector<std::shared_ptr<RowData>>& job_rows);
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
    void on_run_tick();
    void render_job_progress(int index);
    std::optional<int> row_index_of(const RowData* row_data) const;
    void on_result_updated(int index, const ComparisonResult& result);
    std::optional<int> on_job_finished(int index, const ComparisonResult& result, bool final = true);
    void on_job_partially_failed(int index, const ComparisonResult& result, const QString& message,
                                 const QString& stderr_tail, const MetricFailures& failures);
    void on_job_failed(int index, const QString& message, const QString& stderr_tail);
    void on_all_finished();
    void on_run_thread_finished(RunWorker* worker);

    Settings settings_;
    std::optional<VideoInfo> source_info_;
    // The rows, in the table's order. Held by pointer: work that outlives a
    // reorder (a probe, a run) finds its row by identity or by path, never
    // by position.
    std::vector<std::shared_ptr<RowData>> rows_;
    std::vector<ProbeWorker*> probe_workers_;
    ProbeWorker* source_probe_worker_ = nullptr;
    int source_probe_generation_ = 0;
    std::optional<std::filesystem::path> source_reading_;  // the reference being read, until it is
    std::set<int> columns_to_fit_;
    QStringList reference_drop_;  // the files a drag over the reference box carries
    QString reading_text_;        // the status line's "Reading..." while it shows
    bool run_active_ = false;
    bool closing_ = false;
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
    std::map<std::string, std::string> default_metric_backends_;
    CvvdpSettings default_cvvdp_;
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
    int cache_generation_ = 0;
    FileWrites* file_writes_ = nullptr;

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
    int run_partial_count_ = 0;  // videos with some metrics failed, the rest scored
    int run_skipped_ = 0;        // ticked videos left out, as they had every score
    bool run_was_cancelled_ = false;
    // The run's clock, in seconds: started, paused in all, paused since.
    std::optional<double> run_started_at_;
    double paused_total_ = 0;
    std::optional<double> paused_since_;
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
    // The newer release offered, while its message is open.
    QPointer<QMessageBox> update_box_;
    // The Settings tab, and the latest check of FFmpeg, which it shows.
    SettingsPanel* settings_panel_ = nullptr;
    ToolsStatus tools_status_;
    // The rows ticked when the run started, those already scored too: their
    // results go to the graph when it ends.
    std::vector<std::shared_ptr<RowData>> checked_rows_for_run_;
    QPushButton* save_button_ = nullptr;
    QPushButton* show_graph_button_ = nullptr;
    QLabel* status_label_ = nullptr;
    std::vector<ElidedLabel*> job_progress_labels_;
};

}  // namespace vml::app
