// The Metric Graphs tab (the Python app's vmaf_app/ui/graph_panel.py): each
// metric's score over time, for several runs at once.
//
// Several runs (encodes compared with the same or different references) are
// drawn as lines of their own colours, with one readout for the same moment
// in all of them, and a table comparing them. Each metric has a page of its
// own, as their scales differ (0-100, decibels, 0-1); the table of lines and
// statistics above them is shared, as it is the same set of runs.
#pragma once

#include <QColor>
#include <QFont>
#include <QPixmap>
#include <QString>
#include <QWidget>

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "vml/metrics.h"
#include "vml/results.h"
#include "vml/stats.h"

class QLabel;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QTableWidgetItem;
class QTabWidget;

namespace vml::app {

class ChartWidget;
class FileWrites;
class VideoColors;

// Where an XPSNR frame identical to its reference (infinity) is drawn: above
// every real score, or a perfect frame would dip below merely good ones. For
// drawing only: the scores, statistics, readout and exports keep infinity.
inline constexpr double xpsnr_infinity_plot_db = 123.0;

// One run on the graph: its result, name, colour, whether it is shown, and
// the figures its table row shows.
struct SeriesEntry {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    QColor color;
    bool visible = true;
    std::string identity;
    // Each metric's figure for the whole run (its frames' mean, or the
    // whole-video score), by key; nothing where the run has none. Kept here,
    // as the table shows them all at once while pages are built when first
    // shown.
    std::map<std::string, std::optional<double>> means;
    // Frames scoring +inf, by metric, where there are any.
    std::map<std::string, long long> identical;
};

// One run's line on one metric's page: its statistics and points (the
// result's own arrays, or ones made for the line, which `owner` keeps).
struct MetricCurve {
    ScoreStats stats;
    std::span<const float> values;
    std::span<const int> frames;
    std::span<const double> times;
    std::shared_ptr<const void> owner;
    QString label;
    bool visible = true;
    // A per-second line (CVVDP): each second's start; each point is drawn at
    // the middle of its second. Empty for a per-frame line.
    std::vector<double> starts;
    // The run's other frame metrics with any score, in the app's order, for
    // the readout after the "|": with the same frame numbers as this line
    // (the usual case, read by index) or not (a subsampled one, by frame).
    struct Other {
        const MetricDefinition* metric = nullptr;
        std::span<const int> frames;
        std::span<const float> values;
        bool same_frames = false;
    };
    std::vector<Other> others;
};

// One metric's page: its chart, a note when no line has its data, and the
// readout of the frame under the pointer.
class MetricPage : public QWidget {
    Q_OBJECT

public:
    explicit MetricPage(const MetricDefinition& metric, QWidget* parent = nullptr);

    const MetricDefinition& definition() const { return *metric_; }
    ChartWidget& chart() { return *chart_; }

    // The run's line on this page, from its data now, or none when it has
    // none for this metric.
    void set_curve(int series_id, const SeriesEntry& entry);
    void remove_curves(const std::vector<int>& series_ids);
    // Gives the lines the new ids `renumbered` names (old to new), the
    // panel's order (GraphPanel::apply_order).
    void renumber(const std::map<int, int>& renumbered);
    void set_visible(int series_id, bool visible);
    const MetricCurve* curve(int series_id) const;
    const std::map<int, MetricCurve>& curves() const { return curves_; }

    // The pointer at time `x`, value `y`: the frames to report, and the
    // crosshair on the moment chosen.
    void on_hover(double x, double y, const std::map<int, SeriesEntry>& entries);
    // Every shown line at exactly `frame`; whether any has it.
    bool show_frame(int frame, const std::map<int, SeriesEntry>& entries);
    // The frame numbers the shown lines span, for the Go to frame box.
    std::pair<int, int> frame_range() const;

private:
    friend class GraphPanelTest;  // the graph's tests (tests/app)

    void set_timeline_curve(int series_id, const SeriesEntry& entry);
    void update_no_data_label();
    void fit_hover_label();
    std::vector<QString> visible_labels() const;
    QString readout_prefix(const QString& label, int frame, double time) const;
    std::vector<const MetricDefinition*> extra_columns() const;
    void refresh_readout_layout();
    QString readout_row(const QString& prefix, std::optional<double> value,
                        const std::map<std::string, std::optional<double>>& others) const;
    static std::map<std::string, std::optional<double>> other_values(const MetricCurve& curve, int frame,
                                                                      std::size_t index);
    QString readout_missing_frame(const QString& label, int frame) const;
    void set_hover_text(const QString& text);
    std::size_t find_hover_index(const MetricCurve& curve, double x, double y, double half_window) const;
    double find_shared_hover_time(const std::vector<const MetricCurve*>& shown, double x, double y,
                                  double x_per_pixel) const;

    const MetricDefinition* metric_;
    // Hover snaps to the worst frame nearby: a dip for VMAF, a peak for
    // Butteraugli. The search works on sign * value, so lowest means worst.
    float sign_;
    QString placeholder_;
    std::map<int, MetricCurve> curves_;  // by series, only lines with data
    QString hover_text_;
    // The readout's monospaced font, which the label takes from the first
    // hover on; the placeholder before it is in the window's.
    QFont readout_font_;
    bool showing_readout_ = false;
    // The readout's columns, worked out when the lines change, not at each
    // move of the pointer.
    int series_width_ = 0;
    int main_width_ = 0;
    std::vector<std::pair<const MetricDefinition*, int>> columns_;
    ChartWidget* chart_ = nullptr;
    QLabel* no_data_label_ = nullptr;
    QLabel* hover_label_ = nullptr;
};

// The tab: one page per metric, and the table of lines and statistics.
class GraphPanel : public QWidget {
    Q_OBJECT

public:
    // `colors` are shared with the Bitrate tab, so a video has one colour.
    explicit GraphPanel(VideoColors* colors, QWidget* parent = nullptr);
    // Waits for the exports left: one cut short would leave half a file.
    ~GraphPanel() override;

    // A run to show: its line is updated when one with `identity` is shown
    // already. `restore`: a line the user removed (identity remembered as
    // removed) comes back; otherwise it stays away.
    struct Run {
        std::shared_ptr<const ComparisonResult> result;
        QString label;  // empty: the test video's name
        std::string identity;  // empty: the test video's path
        bool restore = true;
    };
    void add_run(const Run& run);
    // add_run for each, with the table and the frame range brought up to
    // date once.
    void add_runs(const std::vector<Run>& runs);

    // Removes lines: of a test video, of a row (its identity), or several at
    // once, each by identity or, for one not found, by the path beside it.
    bool remove_by_path(const std::filesystem::path& distorted);
    bool remove_by_identity(const std::string& identity);
    void remove_by_identities(const std::vector<std::string>& identities,
                              const std::vector<std::filesystem::path>& paths = {});
    // Removes lines by series; `suppress`: the user removed them, so a later
    // add without restore leaves them away.
    void remove_runs(const std::vector<int>& series_ids, bool suppress = true);
    // Shows or hides a line on every page; its row stays, to show it again.
    void set_series_visible(int series_id, bool visible);

    // Lists the lines in the order of `identities`, the Videos table's; lines
    // not among them (results added here) follow, in the order they came.
    // They were listed in the order their runs finished.
    void set_order(std::vector<std::string> identities);

    // The metric to show when a run has it (Settings.graph_metric).
    void set_preferred_metric(const std::string& key);
    // Whether any run is shown.
    bool empty() const { return entries_.empty(); }

    // The window's theme changed: the table's colours are set per cell.
    void refresh_theme();
    // Waits until the exports queued are written, for at most `timeout`.
    bool wait_until_file_writes_idle(std::chrono::milliseconds timeout);

    // The chart of the metric shown, with a title, a legend keyed by the
    // lines' colours and the statistics: what an exported PNG holds.
    QPixmap render_export_image();

    // The dialogs that wait for the user, which the tests answer.
    Dialogs dialogs;

signals:
    // The metric the user chose to look at.
    void metric_changed(const QString& key);

private:
    friend class GraphPanelTest;  // the graph's tests (tests/app)
    friend class WindowTest;      // and the window's

    MetricPage* build_page(const MetricDefinition& metric);
    void on_tab_changed(int index);
    void update_metric_hint();
    void select_available_metric();
    QWidget* build_action_bar();
    void on_frame_requested(int frame);
    void refresh_frame_range();
    void cap_panel_heights();
    const MetricDefinition& current_metric() const;
    void setup_stats_table();
    QTableWidgetItem* series_name_item(const SeriesEntry& entry) const;
    void choose_video_color(int row);
    void on_video_color_changed(const QString& path_key, const QColor& color);
    std::optional<int> series_id_at_row(int row) const;
    void on_stats_cell_clicked(int row, int column);
    void on_stats_header_clicked(int column);
    void show_metric_detail(int index);
    void on_stats_item_changed(QTableWidgetItem* item);
    bool add_series(const Run& run);
    void series_changed();
    bool apply_order();
    void set_visible(int series_id, bool visible);
    void refresh_stats_table();
    std::pair<std::vector<QString>, std::vector<std::tuple<QString, QColor, std::vector<QString>>>> export_table();
    void on_add_saved_run();
    void on_export_png();
    void on_export_csv();
    void on_file_write_failed(const QString& description, const QString& error);
    void on_file_writes_idle();

    VideoColors* colors_;
    // By series: ids rise as runs are added, and are given again in the
    // order wanted (apply_order).
    std::map<int, SeriesEntry> entries_;
    // The lines' identities in the order wanted, the Videos table's.
    std::vector<std::string> order_;
    std::set<std::string> suppressed_identities_;
    int next_id_ = 0;
    std::string preferred_metric_ = "vmaf";
    bool selecting_available_metric_ = false;
    bool populating_stats_ = false;  // the table is being filled: its edits are not the user's
    FileWrites* file_writes_ = nullptr;
    std::optional<QString> export_destination_;

    QLabel* stats_hint_ = nullptr;
    QTableWidget* stats_table_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    std::map<std::string, MetricPage*> pages_;
    QPushButton* export_csv_button_ = nullptr;
    QSpinBox* frame_spin_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* metric_hint_ = nullptr;
};

}  // namespace vml::app
