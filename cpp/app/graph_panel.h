// The Metric Graphs tab (the Python app's vmaf_app/ui/graph_panel.py): each
// metric's score over time, for several runs at once.
//
// Several runs (encodes compared with the same or different references) are
// drawn as lines of their own colours, with one readout for the same frame
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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "export_indicator.h"
#include "quick_save.h"
#include "dialogs.h"
#include "results_export.h"
#include "theme.h"
#include "video_names.h"
#include "vml/metrics.h"
#include "vml/results.h"
#include "vml/stats.h"
#include "vml/segment_profiles.h"
#include "vml/weak_frames.h"
#include "widgets.h"
#include "worst_frame_list.h"

class QCheckBox;
class QLabel;
class QPushButton;
class QStyledItemDelegate;
class QTableWidget;
class QTableWidgetItem;
class QTabWidget;

namespace vml::app {

class ChartWidget;
class TabPills;
class FileWrites;
class FrameControls;
class VideoColors;

// Where a frame scored +infinity (XPSNR's or PSNR-HVS-M's identical frame:
// MetricDefinition::can_be_infinite) is drawn: above every real score, or a
// perfect frame would dip below merely good ones. For drawing only: the
// scores, statistics, readout and exports keep infinity.
inline constexpr double infinity_plot_db = 123.0;

// One run on the graph: its result, name, colour, whether it is shown, and
// the figures its table row shows.
struct SeriesEntry {
    std::shared_ptr<const ComparisonResult> result;
    // The line's whole name: the video's, and " (without film grain)" where
    // it was scored so. Exports, file names and tooltips use it.
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
    // The video's name alone (label without the film grain note), and the
    // label as the window shows it: the name shortened with the other lines'
    // (video_names.h). Empty until the panel sets it.
    QString name{};
    QString shown{};
};

// The label the window shows for the line (SeriesEntry::shown, else its label).
inline const QString& shown_label(const SeriesEntry& entry) { return entry.shown.isEmpty() ? entry.label : entry.shown; }

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
    // The line's name as shown changed (GraphPanel::refresh_shown_names).
    void set_label(int series_id, const QString& label);
    const MetricCurve* curve(int series_id) const;
    const std::map<int, MetricCurve>& curves() const { return curves_; }

    // The pointer at time `x`, value `y`: the frames to report, and the
    // crosshair on the moment chosen.
    void on_hover(double x, double y, const std::map<int, SeriesEntry>& entries);
    // Each shown line's frame that the readout reports for the pointer at
    // time `x`, value `y` (the graph's right-click: that frame in Video
    // Compare); none when no line is shown.
    std::vector<std::pair<const SeriesEntry*, int>> frames_at(double x, double y,
                                                              const std::map<int, SeriesEntry>& entries) const;
    // Every shown line at exactly `frame`; whether any has it.
    bool show_frame(int frame, const std::map<int, SeriesEntry>& entries);
    // The readout written again for the point it shows (the last hover or
    // Go to frame), for the lines shown now: after a line is shown or
    // hidden. The crosshair and the view stay where they are.
    void refresh_readout(const std::map<int, SeriesEntry>& entries);
    // The frame numbers the shown lines span, for the Go to frame box.
    std::pair<int, int> frame_range() const;

    // Draws the difference of the two shown lines, the first (in the
    // table's order) minus the second, in a band under them, while exactly
    // two are shown (difference_possible).
    void set_difference(bool on);
    bool difference_possible() const { return difference_pair().has_value(); }
    // The series ids of the two shown lines, in the table's order, when
    // exactly two are shown: the pair the band draws.
    std::optional<std::pair<int, int>> difference_pair() const;

    // Shades frames `frames.first` to `frames.second` (nothing: none), as
    // the first shown line (in the table's order) with a point among them
    // places them in time.
    void set_selected_frames(std::optional<std::pair<int, int>> frames);
    // The frame at time `time` on the first shown line, for a range dragged
    // across the chart: its nearest point's, or within a per-second line's
    // second, the frame that far into it. Nothing when no line is shown.
    std::optional<int> frame_at(double time) const;
    // The first shown line's frames in time, for the time boxes
    // (FrameControls::Timeline): a frame's time, and the frame at a time,
    // the first at it or after or the last at it or before, within half a
    // millisecond. A per-second line (CVVDP) places a frame between its
    // seconds' first frames.
    std::optional<double> time_of_frame(int frame) const;
    std::optional<int> frame_at_time(double seconds, bool at_or_after) const;

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    friend class GraphPanelTest;  // the graph's tests (tests/app)

    bool erase_curves(const std::vector<int>& series_ids);
    void set_frame_curve(int series_id, const SeriesEntry& entry);
    void set_timeline_curve(int series_id, const SeriesEntry& entry);
    void refresh_difference();
    void refresh_highlight();
    void update_no_data_label();
    void fit_hover_label();
    void fit_readout_margins();
    std::vector<QString> visible_labels() const;
    QString readout_prefix(const QString& label, int frame, double time) const;
    std::vector<const MetricDefinition*> extra_columns() const;
    void refresh_readout_layout();
    std::vector<std::size_t> readout_breaks() const;
    QString wrapped(const QString& text) const;
    QString readout_row(const QString& prefix, std::optional<double> value,
                        const std::map<std::string, std::optional<double>>& others) const;
    static std::map<std::string, std::optional<double>> other_values(const MetricCurve& curve, int frame,
                                                                      std::size_t index);
    QString readout_missing_frame(const QString& label, int frame) const;
    void set_hover_text(const QString& text);
    void hover_readout(double x, double y, const std::map<int, SeriesEntry>& entries, bool place_cursor);
    bool frame_readout(int frame, const std::map<int, SeriesEntry>& entries, bool place_cursor);
    std::size_t find_hover_index(const MetricCurve& curve, double x, double y, double half_window) const;
    // A line's point.
    struct HoverPoint {
        const MetricCurve* curve = nullptr;
        std::size_t index = 0;
    };
    std::optional<HoverPoint> find_shared_hover_point(const std::vector<const MetricCurve*>& shown, double x, double y,
                                                      double x_per_pixel) const;
    // A line's row in the readout: its run, line and point.
    using ReadoutPick = std::tuple<const SeriesEntry*, const MetricCurve*, std::size_t>;
    // The shown lines' rows for the pointer at time `x`, value `y`, in the
    // table's order (none when no line is shown), and where the crosshair
    // goes: nothing for the first row's point.
    struct ReadoutPicks {
        std::vector<ReadoutPick> picks;
        std::optional<double> cursor_time;
    };
    ReadoutPicks readout_picks(double x, double y, const std::map<int, SeriesEntry>& entries) const;
    std::optional<QString> delta_row(const ReadoutPick& first, const ReadoutPick& second) const;

    const MetricDefinition* metric_;
    // Hover snaps to the worst frame nearby: a dip for VMAF, a peak for
    // Butteraugli. The search works on sign * value, so lowest means worst.
    float sign_;
    QString placeholder_;
    std::map<int, MetricCurve> curves_;  // by series, only lines with data
    bool difference_on_ = false;
    std::optional<std::pair<int, int>> selected_frames_;
    QString hover_text_;
    // What the readout shows, for refresh_readout: nothing (the hint), the
    // pointer's place, or a frame asked for.
    struct HoverAt {
        double x = 0;
        double y = 0;
    };
    struct FrameAt {
        int frame = 0;
    };
    std::variant<std::monostate, HoverAt, FrameAt> readout_of_;
    // The readout's monospaced font, which the label takes from the first
    // hover on; the placeholder before it is in the window's.
    QFont readout_font_;
    bool showing_readout_ = false;
    // The readout's columns, worked out when the lines change, not at each
    // move of the pointer.
    int series_width_ = 0;
    int main_width_ = 0;
    std::vector<std::pair<const MetricDefinition*, int>> columns_;
    // The first of columns_ on each of a line's rows (readout_breaks): {0}
    // while they fit on one.
    std::vector<std::size_t> breaks_{0};
    ChartWidget* chart_ = nullptr;
    QLabel* no_data_label_ = nullptr;
    ReadoutLabel* hover_label_ = nullptr;
};

// The tab: one page per metric a line has, and the table of lines and
// statistics.
// The line of a run file on Metric Graphs, by the file it was read from:
// one line for it however it is imported, on Metric Graphs or the Videos
// tab, and however often.
std::string saved_run_identity(const std::filesystem::path& file);

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
        // Given by runs() only: the line's name as the window shows it (the
        // Video names setting: SeriesEntry::shown). add_run ignores it.
        QString shown{};
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
    void set_order(std::vector<std::string> identities);
    // How the lines' names are shown (Settings > Window > Video names).
    void set_video_name_rule(const VideoNameRule& rule);
    // The lines' runs, in the table's order.
    std::vector<Run> runs() const;

    // The metric to show when a run has it (Settings.graph_metric).
    void set_preferred_metric(const std::string& key);
    // The frames highlighted (Highlight range), the Worst Frames tab's too;
    // nothing: none.
    std::optional<std::pair<int, int>> highlighted_frames() const { return selected_frames(); }
    // Whether any run is shown.
    bool empty() const { return entries_.empty(); }

    // The window's theme changed: the table's colours are set per cell.
    void refresh_theme();
    // Waits until the exports queued are written, for at most `timeout`.
    bool wait_until_file_writes_idle(std::chrono::milliseconds timeout);

    // The chart of the metric shown, with a title, a legend keyed by the
    // lines' colours and the statistics: what an exported PNG holds. Titled
    // `title`, or export_title() where it is empty.
    QPixmap render_export_image(const QString& title = {});
    // render_export_image of metric `key`'s page, with only the lines whose
    // identities are `identities` drawn (all of them when empty), the chart
    // drawn at `size`. The page shown, the lines shown and the chart's size
    // are put back after. For an export drawn apart from the tab
    // (graph_bundle.h).
    QPixmap render_export_image(const std::string& key, const std::vector<std::string>& identities, QSize size);
    // The metrics any line has scores of, in the app's order: those with a
    // tab and a column in the table, and those the export covers.
    std::vector<std::string> scored_metrics() const;

    // The dialogs that wait for the user, which the tests answer.
    Dialogs dialogs;
    // Where its exports say how they went: the window's indicator
    // (export_indicator.h). Not set: nowhere.
    ExportReport export_report;
    // The export folder and formats its save icons save in (quick_save.h).
    QuickSaves quick_saves;
    // The colours Export graph as PNG draws in (Settings' Exported graphs), asked at
    // each export; the light chart colours when not set.
    std::function<theme::ChartColours()> export_colours;
    // The statistics' right-click archive, as the Videos table's: what it
    // includes (exported_items of the Settings), and whether a line's video
    // (its identity) can be exported on its own: the window's, asked as the
    // menu opens. Not set: no such export offered.
    std::function<std::vector<std::string>()> export_items;
    std::function<bool(const std::string& identity)> video_exportable;
    // The identities of the lines selected in the statistics table, in its
    // order (the tab bar's Export panel's "the selected videos").
    std::vector<std::string> selected_identities() const;
    // The statistics' save icon, the tab's own one-click save.
    SaveButton* stats_save() const { return stats_save_; }
    // The identities of the Videos tab's rows' lines: those removed with
    // their row still there are what Show removed videos brings back.
    std::function<std::vector<std::string>()> row_identities;
    // Show removed videos enabled where one of row_identities was removed
    // here. Called as lines are added or removed, so at each visit to the
    // tab too (the window's sync_graph adds every row's).
    void refresh_restore_button();
    // Whether the user removed the line of `identity` (a row's): the row's
    // send mark on the Videos tab brings it back.
    bool removed(const std::string& identity) const { return suppressed_identities_.contains(identity); }

signals:
    // The metric the user chose to look at.
    void metric_changed(const QString& key);
    // The statistics' right-click menu, as the Videos table's: a line's
    // video's export, of what Settings' Exports include.
    void video_export_requested(const std::string& identity);
    // The graph's right-click: frame `frame` of a line's video in Video
    // Compare.
    void video_compare_requested(const std::string& identity, int frame);
    // Show removed videos: the rows' lines removed here, to add again with
    // restore (the window has their results).
    void restore_removed_requested();
    // Lines were added or removed (at refresh_restore_button): the Videos
    // tab's send marks are drawn again.
    void lines_changed();

private:
    friend class GraphPanelTest;  // the graph's tests (tests/app)

    MetricPage* build_page(const MetricDefinition& metric);
    void on_tab_changed(int index);
    // The tab shown built, in place of its stand-in, when first shown.
    void build_current_page();
    void update_metric_hint();
    // The tabs (and the table's mean columns) brought up to date with the
    // metrics the lines have, and the preferred metric shown if a line has
    // it, else the first that one has; whether the tabs changed.
    bool select_available_metric();
    void set_metric_tabs(std::vector<const MetricDefinition*> shown, const MetricDefinition& current);
    // The table's columns: the series, a mean per shown metric, the
    // selected metric's statistics, the remove button.
    int mean_column_end() const { return 1 + static_cast<int>(shown_metrics_.size()); }
    bool is_mean_column(int column) const { return column >= 1 && column < mean_column_end(); }
    QWidget* build_action_bar();
    void on_frame_requested(int frame);
    void on_range_dragged(MetricPage* page, double start, double now, bool finished);
    // The frames whose statistics the table gives and the pages shade: the
    // range's two ends in order, or nothing for the whole video.
    std::optional<std::pair<int, int>> selected_frames() const;
    // The range changed (FrameControls::range_changed): every page shades it;
    // once `finished`, the table gives its statistics.
    void on_range_changed(bool finished);
    void refresh_page_controls();
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
    // Every line's SeriesEntry::shown, from the lines there now, and the
    // pages' lines named so.
    void refresh_shown_names();
    std::pair<std::vector<QString>, std::vector<std::tuple<QString, QColor, std::vector<QString>>>> export_table();
    void on_add_saved_run();
    void on_export_png();
    // Difference as CSV..., while the band shows one.
    void on_export_difference();
    // Statistics as CSV... / as JSON...: the statistics table as shown (the
    // page's metric, the lines drawn, the frame range), each cell's text.
    void export_statistics(bool json);
    // The name the page's exports suggest: the one test video shown, or the
    // reference when the lines are of several.
    QString export_stem() const;
    // The page in view's difference pair (MetricPage::difference_pair).
    std::optional<std::pair<int, int>> shown_difference_pair() const;
    void show_stats_menu(const QPoint& position);
    // The tab's export items (docs/ARCHITECTURE.md, "Exports"): the
    // archive of the table's `row`'s video (a line's, or none) and of all
    // the videos, then the quick grabs of what the page shows. The
    // statistics' right-click menu has them all but the graph; the graph's,
    // the graph and the difference; the Export menu, all but a row's video.
    struct StatsMenu {
        std::string identity;  // the row's line's, if any
        QAction* this_video = nullptr;
        QAction* graph = nullptr;
        QAction* statistics_csv = nullptr;
        QAction* statistics_json = nullptr;
        QAction* difference = nullptr;  // while the band shows a difference
    };
    void fill_stats_menu(QMenu& menu, int row, StatsMenu& items, bool graph = false) const;
    // The quick grabs alone: the graph (`graph`), the statistics
    // (`statistics`), and the difference while the band shows one.
    void add_quick_grabs(QMenu& menu, StatsMenu& items, bool graph, bool statistics) const;
    // Each of `items` doing what it says.
    void connect_menu(const StatsMenu& items);
    // The graph's right-click at `position` on `page`'s chart: the frame
    // there in Video Compare, inside the plotting area, then the quick grabs.
    void fill_graph_menu(QMenu& menu, MetricPage& page, const QPoint& position, StatsMenu& items);
    void show_graph_menu(MetricPage& page, const QPoint& position);
    // The statistics' file name without extension, and their text as JSON
    // or CSV (export_statistics' and the save icon's).
    QString statistics_stem() const;
    std::string statistics_text(bool json);
    // The save icons' one-click saves, into the export folder.
    void save_statistics();
    void save_graph();
    // Whether the page shown draws a graph, and its PNG's name.
    bool shown_graph() const;
    QString png_name() const;
    // The exported image's own title: the metric shown, with its notes.
    QString export_title() const;
    // The difference of the lines `pair`: its file's name, and its writing
    // to `path`, said with the files `with` once written.
    QString difference_name(const std::pair<int, int>& pair) const;
    void write_difference(const std::pair<int, int>& pair, const QString& path, const QStringList& with);

    VideoColors* colors_;
    // By series: ids rise as runs are added, and are given again in the
    // order wanted (apply_order).
    std::map<int, SeriesEntry> entries_;
    VideoNameRule name_rule_;
    // The lines' identities in the order wanted, the Videos table's.
    std::vector<std::string> order_;
    std::set<std::string> suppressed_identities_;
    int next_id_ = 0;
    std::string preferred_metric_ = "vmaf";
    bool selecting_available_metric_ = false;
    bool populating_stats_ = false;  // the table is being filled: its edits are not the user's
    FileWrites* file_writes_ = nullptr;
    // The save icons of the statistics' card and the graph's (issue #224).
    SaveButton* stats_save_ = nullptr;
    SaveButton* graph_save_ = nullptr;

    QLabel* stats_hint_ = nullptr;
    QTableWidget* stats_table_ = nullptr;
    QStyledItemDelegate* remove_marks_ = nullptr;  // the remove column's, moved to it as columns change
    QTabWidget* tabs_ = nullptr;
    TabPills* metric_pills_ = nullptr;  // tabs_' tabs, in rows that wrap (its own bar hidden)
    std::map<std::string, MetricPage*> pages_;
    // The metrics with a tab and a mean column, in the app's order: those
    // some line has scores of (scored_metrics), or while none has any, the
    // preferred one alone. Never empty.
    std::vector<const MetricDefinition*> shown_metrics_;
    // Go to frame and the frame range (issue #17), shared with the Bitrate tab.
    FrameControls* frame_controls_ = nullptr;
    // The Difference box: on for every page (not saved), drawn where a page
    // shows exactly two lines.
    bool difference_on_ = false;
    QCheckBox* difference_box_ = nullptr;
    QPushButton* restore_button_ = nullptr;  // Show removed videos
    QLabel* metric_hint_ = nullptr;
};

}  // namespace vml::app
