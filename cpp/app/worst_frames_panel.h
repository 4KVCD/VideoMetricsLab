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
#include "worst_frame_list.h"
#include "vml/metrics.h"
#include "vml/results.h"
#include "vml/weak_frames.h"

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
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

    // What to list: `metric`'s dips in one of `videos` (the one chosen
    // before, while it is among them), within frames `range` (nothing: all).
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
// list that follows the settings, and exported as JSON, CSV or an x265
// --zones value (weak_frames.h).
class WeakSegmentsDialog : public QDialog {
    Q_OBJECT

public:
    // Starts with one condition, on `metric`; only frames within `range`
    // (nothing: all). `dialogs` asks where to export.
    WeakSegmentsDialog(WeakFramesVideo video, const MetricDefinition& metric,
                       std::optional<std::pair<int, int>> range, Dialogs dialogs, QWidget* parent = nullptr);

    SegmentRules rules() const;
    const std::vector<WeakSegment>& segments() const { return segments_; }

    // For the tests: adds a condition, as Add condition does.
    void add_condition(const WeakCondition& condition);
    QTableWidget* table() const { return table_; }

signals:
    // A segment double-clicked: its first frame, in Video Compare.
    void open_requested(const std::string& identity, int frame);

private:
    // A condition's row: the metric, "below" or "above" (its direction),
    // the threshold and its Remove button.
    struct ConditionRow {
        QWidget* row = nullptr;
        QComboBox* metric = nullptr;
        QLabel* comparison = nullptr;
        QDoubleSpinBox* threshold = nullptr;
    };
    void set_condition_metric(ConditionRow& row, const WeakCondition& condition);
    void remove_condition(QWidget* row);
    void refresh();
    void fill_table();
    enum class ExportKind { json, csv, zones };
    void export_segments(ExportKind kind);

    WeakFramesVideo video_;
    std::optional<std::pair<int, int>> range_;
    Dialogs dialogs_;
    std::vector<std::string> keys_;  // the metrics the video has scores of, in the app's order
    std::vector<ConditionRow> conditions_;
    std::vector<WeakSegment> segments_;
    QVBoxLayout* condition_layout_ = nullptr;
    QComboBox* combine_combo_ = nullptr;
    QSpinBox* min_frames_spin_ = nullptr;
    QSpinBox* merge_frames_spin_ = nullptr;
    QLineEdit* zones_edit_ = nullptr;
    QLabel* summary_ = nullptr;
    QTableWidget* table_ = nullptr;
};

}  // namespace vml::app
