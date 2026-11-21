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
#include "vml/segment_profiles.h"
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
    // Starts with profile `in_use` of `profiles` (Settings > Weak
    // segments): its conditions on the metrics the video has, else one on
    // `metric`, from its first band of worse frames. Another profile can be
    // picked to start from. What is changed here holds for this dialog until
    // Save to profile (the profile picked) or Save as new profile. Only
    // frames within `range` (nothing: all). `dialogs` asks where to export
    // and a new profile's name.
    WeakSegmentsDialog(WeakFramesVideo video, const MetricDefinition& metric,
                       std::optional<std::pair<int, int>> range, std::vector<SegmentProfile> profiles,
                       const std::string& in_use, Dialogs dialogs, QWidget* parent = nullptr);

    // The profiles as Settings has them now (changed there while this is
    // open): the one picked stays picked where it is still there, else the
    // profile in use is, with the rules shown kept. Save as new profile
    // checks names against these, and Save to profile saves to one of them.
    void set_profiles(std::vector<SegmentProfile> profiles, const std::string& in_use);

    // The rules shown, with the range.
    SegmentRules rules() const;
    // What Save to profile saves: rules() without the range, and the
    // profile's conditions on metrics this video has none of, left as they
    // were.
    SegmentRules rules_to_save() const;
    const std::vector<WeakSegment>& segments() const { return segments_; }

    // For the tests.
    SegmentRulesEditor* editor() const { return editor_; }
    QPushButton* save_button() const { return save_button_; }
    QPushButton* save_new_button() const { return save_new_button_; }
    QComboBox* profile_combo() const { return profile_combo_; }
    QTableWidget* table() const { return table_; }

signals:
    // A segment double-clicked: its first frame, in Video Compare.
    void open_requested(const std::string& identity, int frame);
    // Save to profile, or Save as new profile: the profile's name ("" for
    // Default) and rules_to_save(), for Settings.
    void rules_saved(const std::string& profile, const vml::SegmentRules& rules);
    // Export all videos...: the report of every scored test video, by the
    // rules shown (without the range).
    void report_requested(const vml::SegmentRules& rules);

private:
    void refresh();
    // Shows the profile picked: its rules, as saved.
    void start_from_profile(const MetricDefinition& metric);
    std::string profile_name() const;
    // Save to profile enabled, and the note beside it, as the rules shown
    // differ from the profile's or not.
    void show_saved_state();
    void fill_table();
    enum class ExportKind { json, csv };
    void export_segments(ExportKind kind);

    WeakFramesVideo video_;
    std::optional<std::pair<int, int>> range_;
    Dialogs dialogs_;
    std::vector<std::string> keys_;  // the metrics the video has scores of, in the app's order
    std::vector<SegmentProfile> profiles_;  // as saved in Settings (set_profiles), or saved here
    SegmentRules saved_;  // the profile picked's rules, as saved
    std::vector<WeakSegment> segments_;
    SegmentRulesEditor* editor_ = nullptr;
    QLabel* saved_note_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QPushButton* save_new_button_ = nullptr;
    QComboBox* profile_combo_ = nullptr;
    QLabel* summary_ = nullptr;
    QTableWidget* table_ = nullptr;
};

}  // namespace vml::app
