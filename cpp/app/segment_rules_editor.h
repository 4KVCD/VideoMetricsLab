// Weak segments' rules, edited (weak_frames.h): its conditions (a metric,
// "is below" or "is above" as the metric's direction says, a threshold and
// Remove), Add condition, whether any or every condition must hold, the
// shortest segment and the gap joined across. Metric Graphs' Weak segments
// dialog and Settings > Weak segments both edit with it.
#pragma once

#include <QWidget>

#include <string>
#include <vector>

#include "vml/weak_frames.h"

class QComboBox;
class QDoubleSpinBox;
class QFormLayout;
class QLabel;
class QSpinBox;
class QVBoxLayout;

namespace vml::app {

class SegmentRulesEditor : public QWidget {
    Q_OBJECT

public:
    // Conditions can be on the metrics `keys` (the app's order). With
    // `at_least_one`, the last condition cannot be removed (the dialog:
    // without one nothing is weak); without, the editor can be empty (no
    // rules saved).
    SegmentRulesEditor(std::vector<std::string> keys, bool at_least_one, QWidget* parent = nullptr);

    // Shows `rules` (their conditions on `keys`, the others left out),
    // without changed().
    void set_rules(const SegmentRules& rules);
    // The rules shown, without a range.
    SegmentRules rules() const;
    // The conditions a metric picked for a condition starts from (its
    // threshold); a metric with none starts from default_weak_condition.
    void set_remembered(std::vector<WeakCondition> conditions) { remembered_ = std::move(conditions); }
    // Adds a condition, as Add condition does.
    void add_condition(const WeakCondition& condition);
    int condition_count() const { return static_cast<int>(conditions_.size()); }
    // A further row under the options, aligned with them (Settings' export
    // after each run).
    void add_option_row(const QString& label, QWidget* field);

signals:
    // A condition or option changed.
    void changed();

private:
    struct ConditionRow {
        QWidget* row = nullptr;
        QComboBox* metric = nullptr;
        QLabel* comparison = nullptr;
        QDoubleSpinBox* threshold = nullptr;
    };
    void add_row(const WeakCondition& condition);
    void set_condition_metric(ConditionRow& row, const WeakCondition& condition);
    void remove_condition(QWidget* row);
    WeakCondition starting_condition(const std::string& key) const;
    void edited();

    std::vector<std::string> keys_;
    bool at_least_one_;
    std::vector<WeakCondition> remembered_;
    std::vector<ConditionRow> conditions_;
    QVBoxLayout* condition_layout_ = nullptr;
    QLabel* none_ = nullptr;  // shown while there is no condition
    QComboBox* combine_combo_ = nullptr;
    QSpinBox* min_frames_spin_ = nullptr;
    QSpinBox* merge_frames_spin_ = nullptr;
    QFormLayout* form_ = nullptr;
    bool quiet_ = false;  // set_rules: no changed()
};

}  // namespace vml::app
