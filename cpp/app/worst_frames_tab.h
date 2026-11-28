// The Worst Frames tab: a workbench for a video's worst frames and weak
// segments (issue #24), in place of Metric Graphs' side panel and its Weak
// segments dialog. On the left what the lists are made from: the videos,
// each with its weak segments and its worst score, and under them the
// conditions (as the dialog had them, with each condition's frames). On the
// right what they find: the metric shown, a graph of the video with its weak
// segments shaded and its worst frames marked, and the two lists, side by
// side where they fit and one above the other where they don't. Under the
// conditions, Details: the frame under the pointer on the graph (every
// metric's score, why it is weak), else the row selected in either list.
#pragma once

#include <filesystem>
#include <QColor>
#include <QString>
#include <QWidget>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "export_indicator.h"
#include "quick_save.h"
#include "dialogs.h"
#include "segment_rules_editor.h"
#include "theme.h"
#include "worst_frame_list.h"
#include "vml/metrics.h"
#include "vml/results.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/weak_frames.h"

class QBoxLayout;
class QComboBox;
class QLabel;
class QMenu;
class QPushButton;
class QSpinBox;
class QSplitter;
class QTabBar;
class QTableWidget;
class QToolButton;
class QVBoxLayout;

namespace vml::app {

class ChartWidget;
class ElidedLabel;
class TabPills;

// A video the tab can show: a graph line's run.
struct WeakFramesVideo {
    std::string identity;  // the line's (SeriesEntry::identity), Video Compare's pair
    QString label;  // its whole name: the graph's title, tooltips and exports
    std::shared_ptr<const ComparisonResult> result;
    QString shown{};  // its name in the videos list, as Metric Graphs shows it; empty: label
};

// How many dips Worst frames lists: one of these, or a number of the user's
// own (Custom...) up to most_worst_frames.
inline constexpr std::array<int, 3> worst_frame_counts{10, 25, 50};
inline constexpr int most_worst_frames = 999;

class WorstFramesTab : public QWidget {
    Q_OBJECT

public:
    // `dialogs` asks where to export and a new profile's name.
    explicit WorstFramesTab(Dialogs dialogs, QWidget* parent = nullptr);

    // Settings' Exported graphs, read at each export of the graph (light
    // where not set).
    std::function<theme::ChartColours()> export_colours;
    // Where its exports say how they went: the window's indicator
    // (export_indicator.h). Not set: nowhere.
    ExportReport export_report;
    // The export folder and formats its save buttons save in (quick_save.h).
    QuickSaves quick_saves;
    // The worst frames list's save button, the tab's own one-click save.
    SaveButton* worst_save() const { return saves_.size() > 1 ? saves_[1] : nullptr; }

    // The videos to choose from (the graph's lines, in their order). The
    // video chosen stays chosen while it is among them; else the first is.
    void set_videos(std::vector<WeakFramesVideo> videos);
    // Only frames within `range` (Metric Graphs' highlighted frames; nothing: all).
    void set_range(std::optional<std::pair<int, int>> range);
    // The weak segment profiles (Settings > Weak segments) and the one in
    // use. Changed in Settings, the profile picked here stays picked where it
    // is still there, with the conditions shown kept.
    void set_segment_profiles(std::vector<SegmentProfile> profiles, const std::string& in_use);
    // How many dips Worst frames lists: 10, 25, 50, or another number up to
    // most_worst_frames, shown as Custom... with its number (none: 10).
    void set_worst_count(int count);

    // The rules shown, with the range.
    SegmentRules rules() const;
    // What Save to profile saves: rules() without the range, and the
    // profile's conditions on metrics this video has none of, left as they
    // were (a preset's, once one is loaded: it replaces them all).
    SegmentRules rules_to_save() const;
    const WeakFramesVideo* chosen_video() const;
    // The metric shown. Not named metric(): that would hide QWidget's virtual metric(PaintDeviceMetric), which
    // clang's -Woverloaded-virtual turns into an error on macOS.
    const MetricDefinition* shown_metric() const { return metric_; }
    const std::vector<WeakSegment>& segments() const { return segments_; }
    // Worst frames' list as shown, for Video Compare to step through.
    WorstFrameList shown_list() const;
    // Video Compare stepped to `frame` of a Worst frames list (`identity`'s,
    // of `metric`, as shown_list names them): where it is the list opened
    // from here, that row is the one selected when the tab is next looked at.
    void keep_worst_frame(const std::string& identity, const QString& metric, int frame);

    // For the tests.
    SegmentRulesEditor* editor() const { return editor_; }
    QTableWidget* videos_table() const { return videos_table_; }
    ElidedLabel* video_title() const { return video_title_; }
    QTableWidget* worst_table() const { return worst_table_; }
    QTableWidget* segments_table() const { return segments_table_; }
    QTabBar* metric_bar() const { return metric_bar_; }
    QComboBox* profile_combo() const { return profile_combo_; }
    QComboBox* count_combo() const { return count_combo_; }
    QToolButton* legend_button() const { return legend_button_; }
    QSpinBox* count_spin() const { return count_spin_; }
    QPushButton* save_button() const { return save_button_; }
    QPushButton* save_new_button() const { return save_new_button_; }
    QPushButton* preset_button() const { return preset_button_; }
    QLabel* conditions_note() const { return conditions_note_; }
    ChartWidget* chart() const { return chart_; }
    QLabel* details() const { return details_; }
    // Whether the two lists are side by side (else one above the other).
    bool lists_side_by_side() const;

signals:
    // A row double-clicked: that frame of that video, with Worst frames'
    // list to step through (none for a segment).
    void open_requested(const std::string& identity, int frame, const vml::app::WorstFrameList& list);
    // Save to profile, or Save as new profile: the profile's name ("" for
    // Default) and rules_to_save(), for Settings.
    void rules_saved(const std::string& profile, const vml::SegmentRules& rules);
    void count_changed(int count);

protected:
    // The work area resized (by the window or the divider): the lists arranged for its width.
    bool eventFilter(QObject* watched, QEvent* event) override;
    // The theme changed: the legend's ⓘ drawn in its text colour.
    void changeEvent(QEvent* event) override;

private:
    QWidget* build_toolbar();
    // The quick grabs (what the tab shows: the video, the metric, the
    // conditions, the dips and the range), each list's and the graph's: the
    // same items on a right-click on its card's save button and on the card.
    void fill_worst_menu(QMenu* menu);
    void fill_graph_menu(QMenu* menu);
    void fill_segments_menu(QMenu* menu);
    // A card's save button (quick_save.h), and its menu on a right-click on
    // `view`, filled by `fill` each time.
    enum class Card { graph, worst, segments };
    SaveButton* save_button(Card card);
    std::vector<SaveButton*> saves_;  // every card's, refreshed with the tab
    void offer_menu(QWidget* view, const std::function<void(QMenu*)>& fill);
    QWidget* build_conditions();
    QWidget* build_lists();
    // Details, under the conditions: the frame under the pointer, else the
    // row selected in either list.
    QWidget* build_details();
    // The pointer on the graph at `time`: the frame under it, or a worst
    // frame marked within a few pixels of it; the line on that frame.
    void hover_at(double time);
    // The pointer left the graph: the line back on the worst frame selected,
    // if one is.
    void hover_left();
    // Details for what is shown now (hover_frame_, else the selection).
    void show_details();
    // `frame`'s lines: its time, where it ranks among the worst frames,
    // whether it is weak and why, and every metric's score, the one
    // graphed first. Empty when the video has no such frame.
    QString frame_details(int frame) const;
    QString segment_details(const WeakSegment& segment) const;
    // The video chosen changed: its metrics in the bar, and the profile's
    // conditions on them.
    void choose_video(int index);
    void choose_metric(const std::string& key);
    void fill_metric_bar();
    void reorder_metrics();
    // Everything that follows the rules: the videos' counts, the graph, the
    // two lists, the notes.
    void refresh();
    void refresh_videos_table();
    void refresh_chart();
    void fill_worst();
    void fill_segments();
    void show_condition_counts();
    void show_saved_state();
    void start_from_profile();
    std::string profile_name() const;
    void load_preset(const SegmentPreset& preset);
    // What each card shows: whether it shows something to save, its file's
    // name (without extension for a list's), and a list's text in a format.
    bool graph_shown() const;
    bool worst_shown() const;
    bool segments_shown() const;
    QString graph_name() const;
    // The exported graph's own title, and the graph titled `title` (empty: that one).
    QString graph_title() const;
    QPixmap graph_image(const QString& title = {});
    QString worst_stem() const;
    std::string worst_text(const std::string& format) const;
    QString segments_stem() const;
    std::string segments_text(const std::string& format) const;
    // A list's quick save: a file of `text` in each of Settings' Quick saves
    // formats, named `stem` and the format.
    std::vector<QuickFile> quick_texts(const QString& stem,
                                       std::string (WorstFramesTab::*text)(const std::string&) const) const;
    enum class ExportKind { json, csv };
    void export_segments(ExportKind kind);
    void export_worst(ExportKind kind);
    void export_graph();
    // Writes `text` at `path`, the Save dialog's answer (nothing for Cancel),
    // and says how it went (export_report).
    void write_export(const QString& path, const std::string& text);
    // A text file as the command line writes it (CR LF on Windows): the
    // error, empty when written.
    static QString write_text(const QString& path, const std::string& text);
    // The lists side by side where both fit at their natural widths, else
    // one above the other. Only compares the widths kept by fit_list with
    // the room: called on every resize of the work area.
    void arrange_lists();
    // `table`'s columns fitted to what it shows now, its natural width kept
    // in `width`, then arrange_lists: where its contents change (fill_worst,
    // fill_segments), not on a resize, which leaves them as they are.
    void fit_list(QTableWidget* table, int& width);
    // The videos list as tall as its rows, up to five: most score one to
    // three videos at a time.
    void fit_videos_height();
    std::optional<int> worst_frame_of_row(int row) const;
    // The segment a Weak segments row shows, wherever a sort has put it.
    std::optional<std::size_t> segment_of_row(int row) const;
    // A list sorted by a column (a click on its heading): kept by heading
    // while the list is filled again, as long as that column is there.
    struct ListSort {
        QString heading;
        Qt::SortOrder order = Qt::AscendingOrder;
    };
    std::optional<ListSort> worst_sort_;
    std::optional<ListSort> segments_sort_;
    // The list put in `sort`'s order, after it is filled in its own.
    void sort_list(QTableWidget* table, std::optional<ListSort>& sort);
    // A click on a list's heading: ascending, descending, then its own order.
    void header_clicked(QTableWidget* table, std::optional<ListSort>& sort, int column);
    void open_selected();
    // The row last selected in either list, selected again (and scrolled
    // to) after the lists are filled again: they are on every refresh, as
    // when the tab is opened, so a row opened in Video Compare was lost on
    // the way back.
    void select_kept();

    Dialogs dialogs_;
    std::vector<WeakFramesVideo> videos_;
    std::string chosen_identity_;  // the video shown
    std::filesystem::path chosen_file_;       // its test video
    std::filesystem::path chosen_reference_;  // and its reference
    // The video the user chose: shown again once it is back among the
    // videos (its line is taken away while it is calculated again, and comes
    // back as a new line of the same file).
    std::string wanted_identity_;
    std::filesystem::path wanted_file_;
    std::filesystem::path wanted_reference_;
    std::vector<std::string> keys_;  // the chosen video's metrics with scores, in the app's order
    const MetricDefinition* metric_ = nullptr;
    std::optional<std::pair<int, int>> range_;
    int count_ = worst_frame_counts.front();
    std::vector<SegmentProfile> profiles_;
    std::string profile_in_use_;
    SegmentRules saved_;
    std::vector<WeakCondition> unshown_;
    bool preset_loaded_ = false;
    std::vector<WeakSegment> segments_;
    std::vector<WorstFrame> worst_;
    // The graph's dots on the worst frames, one per point (ChartSeries::marks).
    std::vector<std::uint8_t> marks_;
    std::vector<QColor> mark_colors_;
    bool building_ = true;
    // The row last selected: its video, its list (Worst frames' with its
    // metric), and its frame (a segment's first). Kept while the lists are
    // filled again (filling_).
    struct Kept {
        std::string identity;
        bool segment = false;
        QString metric;  // Worst frames': its metric's label, as shown_list names it
        int frame = 0;
    };
    std::optional<Kept> kept_;
    bool filling_ = false;

    QTableWidget* videos_table_ = nullptr;
    QTabBar* metric_bar_ = nullptr;  // hidden: metric_pills_ shows its tabs
    TabPills* metric_pills_ = nullptr;
    QComboBox* profile_combo_ = nullptr;
    QPushButton* preset_button_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QPushButton* save_new_button_ = nullptr;
    SegmentRulesEditor* editor_ = nullptr;
    QWidget* editor_holder_ = nullptr;
    QLabel* saved_note_ = nullptr;
    QLabel* summary_ = nullptr;
    QLabel* conditions_note_ = nullptr;
    QVBoxLayout* chart_layout_ = nullptr;
    ElidedLabel* video_title_ = nullptr;  // the video shown, above its graph
    QWidget* inputs_ = nullptr;  // the left column: the videos, the conditions
    QSplitter* split_ = nullptr;
    ChartWidget* chart_ = nullptr;
    QBoxLayout* lists_layout_ = nullptr;
    int worst_width_ = 0;     // the Worst frames table's natural width (fit_list)
    int segments_width_ = 0;  // the Weak segments table's
    QWidget* worst_card_ = nullptr;
    QWidget* segments_card_ = nullptr;
    QComboBox* count_combo_ = nullptr;
    QSpinBox* count_spin_ = nullptr;  // Custom...'s number, shown with it
    QToolButton* legend_button_ = nullptr;  // the graph's ⓘ
    // What the graph shows, under its ⓘ: the metric shown's line, dots and shading.
    void open_legend();
    // The dips listed changed (the drop-down or the number box): the list
    // and the graph's dots made again, Settings told.
    void apply_count(int count);
    QTableWidget* worst_table_ = nullptr;
    QLabel* worst_note_ = nullptr;
    QTableWidget* segments_table_ = nullptr;
    QLabel* empty_note_ = nullptr;
    QWidget* work_ = nullptr;
    QLabel* details_hint_ = nullptr;  // until there is a frame or a row to tell of
    QLabel* details_ = nullptr;       // monospaced, so that its columns line up
    std::optional<int> hover_frame_;  // the frame under the pointer, while it is on the graph
};

}  // namespace vml::app
