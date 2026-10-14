// The main window (the Python app's vmaf_app/ui/main_window.py): the user
// picks a reference video and one or more test videos, sets the options for
// each, runs the metrics, and browses and compares the results.
//
// The test videos are a table: each row shows the file, its media
// information and its scores. Each row has its own VmafOptions (crop
// handling, model, GPU decoding and so on), not one setting shared by all.
//
// Ported so far: the window's frame, its tabs, and the Videos tab's files
// panel (the reference, the test-video table and its buttons). The other
// tabs, the Options panel and runs are still the Python app's alone.
#pragma once

#include <QMainWindow>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "probe_worker.h"
#include "row_state.h"
#include "vml/cvvdp.h"
#include "vml/options.h"
#include "vml/settings.h"
#include "vml/video.h"

class QCheckBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QPushButton;
class QTableWidgetItem;
class QTimer;

namespace vml::app {

class CheckableHeaderView;
class ElidedLabel;
class FillColumnTable;
class QuietTabWidget;

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
    // The CVVDP preset the row was given, if any: only tells apart presets
    // with the same display.
    std::string cvvdp_preset;
    std::optional<RowState> analysis_status;
    // An FFmpeg error, or how many frames a loaded result holds: shown on
    // the file name, the one cell about the whole row.
    QString status_detail;
    // Why each metric failed on the last run, by key, for its red cell.
    std::map<std::string, QString> metric_failures;
    // True only for a "Test both" companion row, whose scale direction was
    // set on purpose to the direction the row stands for.
    bool scale_direction_pinned = false;

    // The file the saved scores are looked up under: the real file when the
    // row's own path is a made-up stand-in.
    const std::filesystem::path& identity_path() const { return media_path ? *media_path : path; }
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow();
    // Waits for the videos still being read: a thread must not outlive its
    // window.
    ~MainWindow() override;

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
    void show_reading(const QString& text);
    void show_ready();
    void on_source_probe_finished(int generation, ProbeWorker* worker);
    void retire(ProbeWorker* worker);

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
    QPushButton* save_button_ = nullptr;
    QPushButton* show_graph_button_ = nullptr;
    QLabel* status_label_ = nullptr;
    std::vector<ElidedLabel*> job_progress_labels_;
};

}  // namespace vml::app
