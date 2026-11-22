// The Worst Frames tab: a workbench for a video's worst frames and weak
// segments (issue #24), in place of Metric Graphs' side panel and its Weak
// segments dialog. On the left what the lists are made from: the videos,
// each with its weak segments and its worst score, and under them the
// conditions (as the dialog had them, with each condition's frames). On the
// right what they find: the metric shown, a graph of the video with its weak
// segments shaded and its worst frames marked, and the two lists, side by
// side where they fit and one above the other where they don't.
#pragma once

#include <QColor>
#include <QString>
#include <QWidget>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "segment_rules_editor.h"
#include "worst_frame_list.h"
#include "vml/metrics.h"
#include "vml/results.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/weak_frames.h"

class QBoxLayout;
class QComboBox;
class QLabel;
class QPushButton;
class QSplitter;
class QTabBar;
class QTableWidget;
class QVBoxLayout;

namespace vml::app {

class ChartWidget;
class ElidedLabel;

// A video the tab can show: a graph line's run.
struct WeakFramesVideo {
    std::string identity;  // the line's (SeriesEntry::identity), Video Compare's pair
    QString label;
    std::shared_ptr<const ComparisonResult> result;
};

// How many dips Worst frames can list.
inline constexpr std::array<int, 3> worst_frame_counts{10, 25, 50};

class WorstFramesTab : public QWidget {
    Q_OBJECT

public:
    // `dialogs` asks where to export and a new profile's name.
    explicit WorstFramesTab(Dialogs dialogs, QWidget* parent = nullptr);

    // The videos to choose from (the graph's lines, in their order). The
    // video chosen stays chosen while it is among them; else the first is.
    void set_videos(std::vector<WeakFramesVideo> videos);
    // Shows `identity`'s video on metric `key` (Metric Graphs' Worst frames
    // button), within frames `range` (nothing: all).
    void show_video(const std::string& identity, const std::string& key, std::optional<std::pair<int, int>> range);
    // Only frames within `range` (Metric Graphs' highlighted frames; nothing: all).
    void set_range(std::optional<std::pair<int, int>> range);
    // The weak segment profiles (Settings > Weak segments) and the one in
    // use. Changed in Settings, the profile picked here stays picked where it
    // is still there, with the conditions shown kept.
    void set_segment_profiles(std::vector<SegmentProfile> profiles, const std::string& in_use);
    // How many dips Worst frames lists: 10, 25 or 50 (another: 10).
    void set_worst_count(int count);

    // The rules shown, with the range.
    SegmentRules rules() const;
    // What Save to profile saves: rules() without the range, and the
    // profile's conditions on metrics this video has none of, left as they
    // were (a preset's, once one is loaded: it replaces them all).
    SegmentRules rules_to_save() const;
    const WeakFramesVideo* chosen_video() const;
    const MetricDefinition* metric() const { return metric_; }
    const std::vector<WeakSegment>& segments() const { return segments_; }
    // Worst frames' list as shown, for Video Compare to step through.
    WorstFrameList shown_list() const;

    // For the tests.
    SegmentRulesEditor* editor() const { return editor_; }
    QTableWidget* videos_table() const { return videos_table_; }
    ElidedLabel* video_title() const { return video_title_; }
    QTableWidget* worst_table() const { return worst_table_; }
    QTableWidget* segments_table() const { return segments_table_; }
    QTabBar* metric_bar() const { return metric_bar_; }
    QComboBox* profile_combo() const { return profile_combo_; }
    QPushButton* save_button() const { return save_button_; }
    QPushButton* save_new_button() const { return save_new_button_; }
    QPushButton* preset_button() const { return preset_button_; }
    QLabel* conditions_note() const { return conditions_note_; }
    ChartWidget* chart() const { return chart_; }
    // Whether the two lists are side by side (else one above the other).
    bool lists_side_by_side() const;

signals:
    // A row double-clicked, or Open in Video Compare: that frame of that
    // video, with Worst frames' list to step through (none for a segment).
    void open_requested(const std::string& identity, int frame, const vml::app::WorstFrameList& list);
    // Save to profile, or Save as new profile: the profile's name ("" for
    // Default) and rules_to_save(), for Settings.
    void rules_saved(const std::string& profile, const vml::SegmentRules& rules);
    // Export all videos...: the report of every scored test video, by the
    // rules shown (without the range).
    void report_requested(const vml::SegmentRules& rules);
    void count_changed(int count);

protected:
    // The work area resized (by the window or the divider): the lists arranged for its width.
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QWidget* build_toolbar();
    QWidget* build_conditions();
    QWidget* build_lists();
    // The video chosen changed: its metrics in the bar, and the profile's
    // conditions on them.
    void choose_video(int index);
    void choose_metric(const std::string& key);
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
    enum class ExportKind { json, csv };
    void export_segments(ExportKind kind);
    // The lists side by side where both fit at their natural widths, else
    // one above the other.
    void arrange_lists();
    // The videos list as tall as its rows, up to five: most score one to
    // three videos at a time.
    void fit_videos_height();
    std::optional<int> worst_frame_of_row(int row) const;
    void open_selected();

    Dialogs dialogs_;
    std::vector<WeakFramesVideo> videos_;
    std::string chosen_identity_;  // the video shown
    // The video the user chose: shown again once it is back among the
    // videos (its line is taken away while it is calculated again).
    std::string wanted_identity_;
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

    QTableWidget* videos_table_ = nullptr;
    QTabBar* metric_bar_ = nullptr;
    QComboBox* profile_combo_ = nullptr;
    QPushButton* preset_button_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QPushButton* save_new_button_ = nullptr;
    QPushButton* open_button_ = nullptr;
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
    QWidget* worst_card_ = nullptr;
    QWidget* segments_card_ = nullptr;
    QComboBox* count_combo_ = nullptr;
    QTableWidget* worst_table_ = nullptr;
    QLabel* worst_note_ = nullptr;
    QTableWidget* segments_table_ = nullptr;
    QLabel* empty_note_ = nullptr;
    QWidget* work_ = nullptr;
};

}  // namespace vml::app
