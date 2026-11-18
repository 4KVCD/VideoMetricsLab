// Metric Graphs' Worst frames (issue #24): the worst frame of each of a
// video's worst dips on the metric shown, and the Weak segments dialog that
// lists and exports the stretches where scores fall past a threshold.
#pragma once

#include <QDialog>
#include <QString>
#include <QWidget>

#include <array>
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
#include "vml/weak_frames.h"

class QComboBox;
class QLabel;
class QPushButton;
class QTableWidget;
class QVBoxLayout;

namespace vml::app {

// A video the panel and the dialog can show: a graph line's run.
struct WeakFramesVideo {
    std::string identity;  // the line's (SeriesEntry::identity), Video Compare's pair
    QString label;
    std::shared_ptr<const ComparisonResult> result;
};

// How many dips the list can show.
inline constexpr std::array<int, 3> worst_frame_counts{10, 25, 50};

class WorstFramesPanel : public QWidget {
    Q_OBJECT

public:
    explicit WorstFramesPanel(QWidget* parent = nullptr);

    // What to list: `metric`'s dips in one of `videos` (the one chosen,
    // whenever it is among them; else the first), within frames `range`
    // (nothing: all).
    void set_view(const MetricDefinition& metric, std::vector<WeakFramesVideo> videos,
                  std::optional<std::pair<int, int>> range);
    // 10, 25 or 50 (another count is taken as 10).
    void set_count(int count);
    int count() const { return count_; }
    const WeakFramesVideo* chosen_video() const;
    // The list as shown, for Video Compare to step through.
    WorstFrameList shown_list() const;

    // For the tests.
    QTableWidget* table() const { return table_; }
    QPushButton* segments_button() const { return segments_button_; }

signals:
    // A row clicked: its frame, to mark on the graph.
    void frame_clicked(int frame);
    // A row double-clicked, or Open in Video Compare: that frame of that video.
    void open_requested(const std::string& identity, int frame);
    void count_changed(int count);
    void segments_requested();

private:
    void refresh();
    std::optional<int> frame_of_row(int row) const;

    const MetricDefinition* metric_ = nullptr;
    std::vector<WeakFramesVideo> videos_;
    std::optional<std::pair<int, int>> range_;
    int count_ = worst_frame_counts.front();
    std::string chosen_identity_;
    QLabel* title_ = nullptr;
    QComboBox* video_combo_ = nullptr;
    QComboBox* count_combo_ = nullptr;
    QTableWidget* table_ = nullptr;
    QLabel* note_ = nullptr;
    QPushButton* open_button_ = nullptr;
    QPushButton* segments_button_ = nullptr;
};

// The stretches of one video where its scores fall past thresholds, as a
// list that follows the settings, and exported as JSON or CSV
// (weak_frames.h).
class WeakSegmentsDialog : public QDialog {
    Q_OBJECT

public:
    // Starts with the `saved` rules (Settings > Weak segments): the saved
    // conditions on the metrics the video has, else one on `metric`, from
    // its first band of worse frames. What is changed here holds for this
    // dialog until Save as default. Only frames within `range` (nothing:
    // all). `dialogs` asks where to export.
    WeakSegmentsDialog(WeakFramesVideo video, const MetricDefinition& metric,
                       std::optional<std::pair<int, int>> range, const SegmentRules& saved, Dialogs dialogs,
                       QWidget* parent = nullptr);

    // The rules shown, with the range.
    SegmentRules rules() const;
    // What Save as default saves: rules() without the range, and the saved
    // conditions on metrics this video has none of, left as they were.
    SegmentRules rules_to_save() const;
    const std::vector<WeakSegment>& segments() const { return segments_; }

    // For the tests.
    SegmentRulesEditor* editor() const { return editor_; }
    QPushButton* save_button() const { return save_button_; }
    QTableWidget* table() const { return table_; }

signals:
    // A segment double-clicked: its first frame, in Video Compare.
    void open_requested(const std::string& identity, int frame);
    // Save as default: rules_to_save(), for Settings.
    void rules_saved(const vml::SegmentRules& rules);
    // Export all videos...: the report of every scored test video, by the
    // rules shown (without the range).
    void report_requested(const vml::SegmentRules& rules);

private:
    void refresh();
    // Save as default enabled, and the note beside it, as the rules shown
    // differ from those saved or not.
    void show_saved_state();
    void fill_table();
    enum class ExportKind { json, csv };
    void export_segments(ExportKind kind);

    WeakFramesVideo video_;
    std::optional<std::pair<int, int>> range_;
    Dialogs dialogs_;
    std::vector<std::string> keys_;  // the metrics the video has scores of, in the app's order
    SegmentRules saved_;  // as saved in Settings (since the dialog opened, or last saved here)
    std::vector<WeakSegment> segments_;
    SegmentRulesEditor* editor_ = nullptr;
    QLabel* saved_note_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QLabel* summary_ = nullptr;
    QTableWidget* table_ = nullptr;
};

}  // namespace vml::app
