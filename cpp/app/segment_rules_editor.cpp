#include "segment_rules_editor.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScopedValueRollback>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"

namespace vml::app {

int threshold_decimals(double value, int usual) {
    constexpr int most = 6;
    for (int decimals = std::max(usual, 0); decimals < most; ++decimals) {
        const double scale = std::pow(10.0, decimals);
        if (std::abs(std::round(value * scale) / scale - value) <= 1e-9 * std::max(1.0, std::abs(value))) {
            return decimals;
        }
    }
    return most;
}

SegmentRulesEditor::SegmentRulesEditor(std::vector<std::string> keys, bool at_least_one, QWidget* parent)
    : QWidget(parent), keys_(std::move(keys)), at_least_one_(at_least_one) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(new QLabel(app::tr("A frame is weak where:")));
    condition_layout_ = new QVBoxLayout;
    layout->addLayout(condition_layout_);
    none_ = new QLabel(app::tr("No condition yet: the Worst Frames tab starts from the metric shown."));
    theme::style(none_, "color: {muted};");
    none_->setVisible(!at_least_one_);
    layout->addWidget(none_);
    auto* add_button = new QPushButton(app::tr("Add condition"));
    add_button->setEnabled(!keys_.empty());
    connect(add_button, &QPushButton::clicked, this, [this] {
        // The first metric no condition names yet, or the first.
        std::string key = keys_.front();
        for (const std::string& candidate : keys_) {
            if (std::none_of(conditions_.begin(), conditions_.end(), [&candidate](const ConditionRow& row) {
                    return row.metric->currentData().toString().toStdString() == candidate;
                })) {
                key = candidate;
                break;
            }
        }
        add_condition(starting_condition(key));
    });
    auto* add_row = new QHBoxLayout;
    add_row->addWidget(add_button);
    add_row->addStretch(1);
    layout->addLayout(add_row);

    auto* form = new QFormLayout;
    form_ = form;
    form->setContentsMargins(0, 0, 0, 0);
    // The fields as wide as they need, not the window's width.
    form->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);
    combine_combo_ = new QComboBox;
    combine_combo_->addItem(app::tr("any condition holds"));
    combine_combo_->addItem(app::tr("every condition holds"));
    connect(combine_combo_, &QComboBox::currentIndexChanged, this, [this] { edited(); });
    form->addRow(app::tr("Weak where:"), combine_combo_);
    min_frames_spin_ = new QSpinBox;
    min_frames_spin_->setRange(1, 1'000'000);
    min_frames_spin_->setMinimumWidth(120);
    connect(min_frames_spin_, &QSpinBox::valueChanged, this, [this] { edited(); });
    form->addRow(app::tr("Shortest segment (frames):"), min_frames_spin_);
    merge_frames_spin_ = new QSpinBox;
    merge_frames_spin_->setRange(0, 1'000'000);
    merge_frames_spin_->setMinimumWidth(120);
    connect(merge_frames_spin_, &QSpinBox::valueChanged, this, [this] { edited(); });
    form->addRow(app::tr("Join segments up to this many frames apart:"), merge_frames_spin_);
    layout->addLayout(form);
}

void SegmentRulesEditor::add_option_row(const QString& label, QWidget* field) { form_->addRow(label, field); }

void SegmentRulesEditor::wrap_long_options() { form_->setRowWrapPolicy(QFormLayout::WrapLongRows); }

void SegmentRulesEditor::set_rules(const SegmentRules& rules) {
    const QScopedValueRollback quiet(quiet_, true);
    for (const ConditionRow& row : conditions_) {
        delete row.row;
    }
    conditions_.clear();
    for (const WeakCondition& condition : rules.conditions) {
        if (std::find(keys_.begin(), keys_.end(), condition.key) != keys_.end()) {
            add_row(condition);
        }
    }
    combine_combo_->setCurrentIndex(rules.all ? 1 : 0);
    min_frames_spin_->setValue(rules.min_frames);
    merge_frames_spin_->setValue(rules.merge_frames);
    none_->setVisible(conditions_.empty() && !at_least_one_);
}

SegmentRules SegmentRulesEditor::rules() const {
    SegmentRules rules;
    for (const ConditionRow& row : conditions_) {
        const std::string key = row.metric->currentData().toString().toStdString();
        rules.conditions.push_back(
            {key, row.comparison->currentData().toString() == ">" ? '>' : '<', row.threshold->value()});
    }
    rules.all = combine_combo_->currentIndex() == 1;
    rules.min_frames = min_frames_spin_->value();
    rules.merge_frames = merge_frames_spin_->value();
    return rules;
}

void SegmentRulesEditor::add_condition(const WeakCondition& condition) {
    add_row(condition);
    edited();
}

void SegmentRulesEditor::add_row(const WeakCondition& condition) {
    ConditionRow entry;
    entry.row = new QWidget;
    auto* row_layout = new QHBoxLayout(entry.row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    entry.metric = new QComboBox;
    for (const std::string& key : keys_) {
        entry.metric->addItem(qt_text(metric_definition(key).label), QString::fromStdString(key));
    }
    row_layout->addWidget(entry.metric);
    // Below for most metrics; above for Butteraugli, whose worst are its
    // highest, or to find what is better than it need be (vmaf>98).
    entry.comparison = new QComboBox;
    entry.comparison->addItem(app::tr("is below"), QStringLiteral("<"));
    entry.comparison->addItem(app::tr("is above"), QStringLiteral(">"));
    row_layout->addWidget(entry.comparison);
    entry.threshold = new QDoubleSpinBox;
    entry.threshold->setRange(-1'000'000.0, 1'000'000.0);
    row_layout->addWidget(entry.threshold);
    auto* remove = new QPushButton(app::tr("Remove"));
    row_layout->addWidget(remove);
    row_layout->addStretch(1);
    condition_layout_->addWidget(entry.row);

    QWidget* row = entry.row;
    connect(remove, &QPushButton::clicked, this, [this, row] { remove_condition(row); });
    connect(entry.metric, &QComboBox::currentIndexChanged, this, [this, row] {
        for (ConditionRow& found : conditions_) {
            if (found.row == row) {
                const std::string key = found.metric->currentData().toString().toStdString();
                set_condition_metric(found, starting_condition(key));
            }
        }
        edited();
    });
    connect(entry.comparison, &QComboBox::currentIndexChanged, this, [this] { edited(); });
    connect(entry.threshold, &QDoubleSpinBox::valueChanged, this, [this] { edited(); });
    set_condition_metric(entry, condition);
    conditions_.push_back(entry);
    none_->setVisible(false);
}

void SegmentRulesEditor::set_condition_metric(ConditionRow& row, const WeakCondition& condition) {
    const MetricDefinition& definition = metric_definition(condition.key);
    {
        const QSignalBlocker blocker(row.metric);
        row.metric->setCurrentIndex(row.metric->findData(QString::fromStdString(condition.key)));
    }
    // As the condition says: a metric picked anew starts from its own
    // direction (starting_condition), a saved text keeps its own.
    {
        const QSignalBlocker blocker(row.comparison);
        row.comparison->setCurrentIndex(condition.comparison == '>' ? 1 : 0);
    }
    const QSignalBlocker blocker(row.threshold);
    // The metric's usual decimals, or more where the value has them (one
    // written by hand, vmaf<80.125), up to 6: a box rounds what it is given,
    // and a value rounded would be saved so at the next edit.
    row.threshold->setDecimals(threshold_decimals(condition.threshold, definition.decimals));
    // A step of about a hundredth of the scale: 1 for VMAF's 0-100, 0.01
    // for SSIM's 0-1.
    const double magnitude = std::abs(condition.threshold);
    row.threshold->setSingleStep(magnitude >= 10.0 ? 1.0 : magnitude >= 1.0 ? 0.1 : 0.01);
    row.threshold->setValue(condition.threshold);
}

void SegmentRulesEditor::remove_condition(QWidget* row) {
    if (at_least_one_ && conditions_.size() <= 1) {
        return;  // one condition at least: without, nothing is weak
    }
    std::erase_if(conditions_, [row](const ConditionRow& entry) { return entry.row == row; });
    row->deleteLater();
    none_->setVisible(conditions_.empty() && !at_least_one_);
    edited();
}

WeakCondition SegmentRulesEditor::starting_condition(const std::string& key) const {
    const auto found = std::find_if(remembered_.begin(), remembered_.end(),
                                    [&key](const WeakCondition& condition) { return condition.key == key; });
    return found != remembered_.end() ? *found : default_weak_condition(metric_definition(key));
}

void SegmentRulesEditor::edited() {
    if (!quiet_) {
        emit changed();
    }
}

}  // namespace vml::app
