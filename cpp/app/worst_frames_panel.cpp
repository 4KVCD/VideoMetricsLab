#include "worst_frames_panel.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <fstream>

#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/python_json.h"
#include "vml/time_format.h"

namespace vml::app {

namespace {

// How many segments the dialog lists; its exports have them all.
constexpr int listed_segments = 1000;

QTableWidgetItem* cell(const QString& text, bool number = true) {
    auto* item = new QTableWidgetItem(text);
    item->setTextAlignment(number ? (Qt::AlignRight | Qt::AlignVCenter) : (Qt::AlignLeft | Qt::AlignVCenter));
    return item;
}

QTableWidget* make_table() {
    auto* table = new QTableWidget;
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    return table;
}

QString time_text(double seconds) { return qt_text(format_hms(seconds, 3)); }

QString range_note(const std::optional<std::pair<int, int>>& range) {
    return range ? app::tr("Only the selected frames, {first} to {last}.",
                           {{"first", range->first}, {"last", range->second}})
                 : QString();
}

}  // namespace

// ------------------------------------------------------------ the list

WorstFramesPanel::WorstFramesPanel(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 0, 0, 0);
    title_ = new QLabel;
    QFont bold = title_->font();
    bold.setBold(true);
    title_->setFont(bold);
    layout->addWidget(title_);

    auto* choices = new QHBoxLayout;
    video_combo_ = new QComboBox;
    video_combo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    video_combo_->setMinimumContentsLength(12);
    connect(video_combo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index >= 0 && static_cast<std::size_t>(index) < videos_.size()) {
            chosen_identity_ = videos_[static_cast<std::size_t>(index)].identity;
            refresh();
        }
    });
    choices->addWidget(video_combo_, 1);
    count_combo_ = new QComboBox;
    for (const int count : worst_frame_counts) {
        count_combo_->addItem(ntr("{count} dip", "{count} dips", count), count);
    }
    connect(count_combo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index >= 0) {
            count_ = count_combo_->itemData(index).toInt();
            refresh();
            emit count_changed(count_);
        }
    });
    choices->addWidget(count_combo_);
    layout->addLayout(choices);

    table_ = make_table();
    connect(table_, &QTableWidget::cellClicked, this, [this](int row, int) {
        if (const std::optional<int> frame = frame_of_row(row)) {
            emit frame_clicked(*frame);
        }
    });
    connect(table_, &QTableWidget::cellDoubleClicked, this, [this](int row, int) {
        const WeakFramesVideo* video = chosen_video();
        if (const std::optional<int> frame = frame_of_row(row); frame && video != nullptr) {
            emit open_requested(video->identity, *frame);
        }
    });
    connect(table_, &QTableWidget::itemSelectionChanged, this,
            [this] { open_button_->setEnabled(!table_->selectedItems().isEmpty()); });
    layout->addWidget(table_, 1);

    note_ = new QLabel;
    note_->setWordWrap(true);
    theme::style(note_, "color: {muted};");
    layout->addWidget(note_);

    auto* buttons = new QHBoxLayout;
    open_button_ = new QPushButton(app::tr("Open in Video Compare"));
    open_button_->setEnabled(false);
    connect(open_button_, &QPushButton::clicked, this, [this] {
        const QList<QTableWidgetItem*> selected = table_->selectedItems();
        const WeakFramesVideo* video = chosen_video();
        if (selected.isEmpty() || video == nullptr) {
            return;
        }
        if (const std::optional<int> frame = frame_of_row(selected.front()->row())) {
            emit open_requested(video->identity, *frame);
        }
    });
    buttons->addWidget(open_button_);
    segments_button_ = new QPushButton(app::tr("Weak segments..."));
    connect(segments_button_, &QPushButton::clicked, this, [this] { emit segments_requested(); });
    buttons->addWidget(segments_button_);
    buttons->addStretch(1);
    layout->addLayout(buttons);
}

void WorstFramesPanel::set_view(const MetricDefinition& metric, std::vector<WeakFramesVideo> videos,
                                std::optional<std::pair<int, int>> range) {
    metric_ = &metric;
    videos_ = std::move(videos);
    range_ = range;
    const auto chosen = std::find_if(videos_.begin(), videos_.end(),
                                     [this](const WeakFramesVideo& video) { return video.identity == chosen_identity_; });
    const int index = videos_.empty() ? -1 : chosen == videos_.end() ? 0 : static_cast<int>(chosen - videos_.begin());
    {
        const QSignalBlocker blocker(video_combo_);
        video_combo_->clear();
        for (const WeakFramesVideo& video : videos_) {
            video_combo_->addItem(video.label);
        }
        video_combo_->setCurrentIndex(index);
    }
    chosen_identity_ = index >= 0 ? videos_[static_cast<std::size_t>(index)].identity : std::string();
    refresh();
}

void WorstFramesPanel::set_count(int count) {
    const auto found = std::find(worst_frame_counts.begin(), worst_frame_counts.end(), count);
    const int index = found == worst_frame_counts.end() ? 0 : static_cast<int>(found - worst_frame_counts.begin());
    const QSignalBlocker blocker(count_combo_);
    count_combo_->setCurrentIndex(index);
    count_ = worst_frame_counts[static_cast<std::size_t>(index)];
    refresh();
}

const WeakFramesVideo* WorstFramesPanel::chosen_video() const {
    const int index = video_combo_->currentIndex();
    return index >= 0 && static_cast<std::size_t>(index) < videos_.size() ? &videos_[static_cast<std::size_t>(index)]
                                                                         : nullptr;
}

WorstFrameList WorstFramesPanel::shown_list() const {
    WorstFrameList list;
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr) {
        return list;
    }
    list.identity = video->identity;
    list.metric = qt_text(metric_->label);
    for (int row = 0; row < table_->rowCount(); ++row) {
        const std::optional<int> frame = frame_of_row(row);
        const QTableWidgetItem* score = table_->item(row, 2);  // the metric listed
        if (frame && score != nullptr) {
            list.stops.push_back({*frame, score->text()});
        }
    }
    return list;
}

std::optional<int> WorstFramesPanel::frame_of_row(int row) const {
    const QTableWidgetItem* item = table_->item(row, 0);
    return item != nullptr ? std::optional(item->data(Qt::UserRole).toInt()) : std::nullopt;
}

void WorstFramesPanel::refresh() {
    table_->clearContents();
    table_->setRowCount(0);
    open_button_->setEnabled(false);
    if (metric_ == nullptr) {
        return;
    }
    title_->setText(app::tr("Worst frames: {metric}", {{"metric", qt_text(metric_->label)}}));
    const WeakFramesVideo* video = chosen_video();
    segments_button_->setEnabled(video != nullptr);
    video_combo_->setEnabled(videos_.size() > 1);
    if (video == nullptr) {
        table_->setColumnCount(0);
        note_->setText(app::tr("No video shown has {metric} scores.", {{"metric", qt_text(metric_->label)}}));
        return;
    }
    const MetricResultSet& results = video->result->metric_results;
    const std::optional<ScoreSamples> samples = score_samples(results, std::string(metric_->key));
    // The metric listed, then the run's others with scores, in the app's order.
    std::vector<std::pair<const MetricDefinition*, ScoreSamples>> columns{{metric_, *samples}};
    for (const MetricDefinition& other : metrics()) {
        if (&other != metric_) {
            if (const std::optional<ScoreSamples> found = score_samples(results, std::string(other.key))) {
                columns.emplace_back(&other, *found);
            }
        }
    }
    QStringList headers{app::tr("Frame"), app::tr("Time")};
    for (const auto& [definition, column_samples] : columns) {
        headers.append(qt_text(definition->short_label));
    }
    table_->setColumnCount(static_cast<int>(headers.size()));
    table_->setHorizontalHeaderLabels(headers);

    std::vector<FrameInterval> ranges;
    if (range_) {
        ranges.push_back({range_->first, range_->second});
    }
    const std::vector<WorstFrame> worst =
        worst_frames(*samples, metric_->direction, static_cast<std::size_t>(count_), ranges);
    table_->setRowCount(static_cast<int>(worst.size()));
    for (std::size_t row = 0; row < worst.size(); ++row) {
        const int r = static_cast<int>(row);
        QTableWidgetItem* frame = cell(QString::number(worst[row].frame));
        frame->setData(Qt::UserRole, worst[row].frame);
        table_->setItem(r, 0, frame);
        table_->setItem(r, 1, cell(time_text(worst[row].time)));
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const auto& [definition, column_samples] = columns[c];
            table_->setItem(r, static_cast<int>(c) + 2,
                            cell(qt_text(definition->format_value(score_at(column_samples, worst[row].frame)))));
        }
    }
    QString note = samples->per_second
                       ? app::tr("CVVDP scores each second: a row is its second's first frame, and a dip is the "
                                 "seconds within 1 s of a worse one.")
                       : app::tr("The worst frame of each dip, worst first: a frame within 1 s of a worse one is part "
                                 "of its dip.");
    note += " " + app::tr("Click a row to mark it on the graph; double-click it to open it in Video Compare.");
    if (range_) {
        note += " " + range_note(range_);
    }
    note_->setText(note);
}

// ---------------------------------------------------------- the dialog

WeakSegmentsDialog::WeakSegmentsDialog(WeakFramesVideo video, const MetricDefinition& metric,
                                       std::optional<std::pair<int, int>> range, Dialogs dialogs, QWidget* parent)
    : QDialog(parent), video_(std::move(video)), range_(range), dialogs_(std::move(dialogs)) {
    setWindowTitle(app::tr("Weak segments: {name}", {{"name", video_.label}}));
    for (const MetricDefinition& definition : metrics()) {
        if (score_samples(video_.result->metric_results, std::string(definition.key))) {
            keys_.emplace_back(definition.key);
        }
    }
    auto* layout = new QVBoxLayout(this);
    layout->addWidget(new QLabel(app::tr("A frame is weak where:")));
    condition_layout_ = new QVBoxLayout;
    layout->addLayout(condition_layout_);
    auto* add_button = new QPushButton(app::tr("Add condition"));
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
        add_condition(default_weak_condition(metric_definition(key)));
    });
    auto* add_row = new QHBoxLayout;
    add_row->addWidget(add_button);
    add_row->addStretch(1);
    layout->addLayout(add_row);

    auto* form = new QFormLayout;
    combine_combo_ = new QComboBox;
    combine_combo_->addItem(app::tr("any condition holds"));
    combine_combo_->addItem(app::tr("every condition holds"));
    connect(combine_combo_, &QComboBox::currentIndexChanged, this, [this] { refresh(); });
    form->addRow(app::tr("Weak where:"), combine_combo_);
    min_frames_spin_ = new QSpinBox;
    min_frames_spin_->setRange(1, 1'000'000);
    connect(min_frames_spin_, &QSpinBox::valueChanged, this, [this] { refresh(); });
    form->addRow(app::tr("Shortest segment (frames):"), min_frames_spin_);
    merge_frames_spin_ = new QSpinBox;
    merge_frames_spin_->setRange(0, 1'000'000);
    connect(merge_frames_spin_, &QSpinBox::valueChanged, this, [this] { refresh(); });
    form->addRow(app::tr("Join segments up to this many frames apart:"), merge_frames_spin_);
    layout->addLayout(form);

    summary_ = new QLabel;
    summary_->setWordWrap(true);
    layout->addWidget(summary_);
    table_ = make_table();
    table_->setMinimumHeight(220);
    connect(table_, &QTableWidget::cellDoubleClicked, this, [this](int row, int) {
        if (row >= 0 && static_cast<std::size_t>(row) < segments_.size()) {
            emit open_requested(video_.identity, segments_[static_cast<std::size_t>(row)].first_frame);
        }
    });
    layout->addWidget(table_, 1);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close);
    const auto add_export = [this, buttons](const QString& text, ExportKind kind) {
        QPushButton* button = buttons->addButton(text, QDialogButtonBox::ActionRole);
        connect(button, &QPushButton::clicked, this, [this, kind] { export_segments(kind); });
    };
    add_export(app::tr("Export JSON..."), ExportKind::json);
    add_export(app::tr("Export CSV..."), ExportKind::csv);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);

    add_condition(default_weak_condition(metric));
    resize(720, 560);
}

void WeakSegmentsDialog::add_condition(const WeakCondition& condition) {
    ConditionRow entry;
    entry.row = new QWidget;
    auto* row_layout = new QHBoxLayout(entry.row);
    row_layout->setContentsMargins(0, 0, 0, 0);
    entry.metric = new QComboBox;
    for (const std::string& key : keys_) {
        entry.metric->addItem(qt_text(metric_definition(key).label), QString::fromStdString(key));
    }
    row_layout->addWidget(entry.metric);
    entry.comparison = new QLabel;
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
                set_condition_metric(found, default_weak_condition(metric_definition(key)));
            }
        }
        refresh();
    });
    connect(entry.threshold, &QDoubleSpinBox::valueChanged, this, [this] { refresh(); });
    set_condition_metric(entry, condition);
    conditions_.push_back(entry);
    refresh();
}

void WeakSegmentsDialog::set_condition_metric(ConditionRow& row, const WeakCondition& condition) {
    const MetricDefinition& definition = metric_definition(condition.key);
    {
        const QSignalBlocker blocker(row.metric);
        row.metric->setCurrentIndex(row.metric->findData(QString::fromStdString(condition.key)));
    }
    row.comparison->setText(condition.comparison == '>' ? app::tr("is above") : app::tr("is below"));
    const QSignalBlocker blocker(row.threshold);
    row.threshold->setDecimals(definition.decimals);
    // A step of about a hundredth of the scale: 1 for VMAF's 0-100, 0.01
    // for SSIM's 0-1.
    const double magnitude = std::abs(condition.threshold);
    row.threshold->setSingleStep(magnitude >= 10.0 ? 1.0 : magnitude >= 1.0 ? 0.1 : 0.01);
    row.threshold->setValue(condition.threshold);
}

void WeakSegmentsDialog::remove_condition(QWidget* row) {
    if (conditions_.size() <= 1) {
        return;  // one condition at least: without, nothing is weak
    }
    std::erase_if(conditions_, [row](const ConditionRow& entry) { return entry.row == row; });
    row->deleteLater();
    refresh();
}

SegmentRules WeakSegmentsDialog::rules() const {
    SegmentRules rules;
    for (const ConditionRow& row : conditions_) {
        const std::string key = row.metric->currentData().toString().toStdString();
        const MetricDefinition& definition = metric_definition(key);
        rules.conditions.push_back(
            {key, definition.direction == MetricDirection::lower_is_better ? '>' : '<', row.threshold->value()});
    }
    rules.all = combine_combo_->currentIndex() == 1;
    rules.min_frames = min_frames_spin_->value();
    rules.merge_frames = merge_frames_spin_->value();
    if (range_) {
        rules.ranges.push_back({range_->first, range_->second});
    }
    return rules;
}

void WeakSegmentsDialog::refresh() {
    if (combine_combo_ == nullptr || min_frames_spin_ == nullptr || merge_frames_spin_ == nullptr) {
        return;  // still being built
    }
    segments_ = weak_segments(*video_.result, rules());
    long long frames = 0;
    for (const WeakSegment& segment : segments_) {
        frames += segment.frames();
    }
    // Two plurals, so one segment of one frame reads "1 frame".
    QString summary = ntr("{count} segment", "{count} segments", static_cast<long long>(segments_.size())) + ", " +
                      ntr("{count} frame in all.", "{count} frames in all.", frames);
    if (segments_.size() > static_cast<std::size_t>(listed_segments)) {
        summary += " " + app::tr("The first {count} are listed; the exports have them all.",
                                 {{"count", listed_segments}});
    }
    if (!segments_.empty()) {
        summary += " " + app::tr("Double-click one to open its first frame in Video Compare.");
    }
    if (range_) {
        summary += " " + range_note(range_);
    }
    summary_->setText(summary);
    fill_table();
}

void WeakSegmentsDialog::fill_table() {
    const SegmentRules current = rules();
    QStringList headers{app::tr("First"), app::tr("Last"), app::tr("Frames"), app::tr("Start"), app::tr("End")};
    std::vector<std::string> keys;
    for (const WeakCondition& condition : current.conditions) {
        if (std::find(keys.begin(), keys.end(), condition.key) == keys.end()) {
            keys.push_back(condition.key);
            const QString label = qt_text(metric_definition(condition.key).short_label);
            headers.append(app::tr("{metric} worst", {{"metric", label}}));
            headers.append(app::tr("{metric} mean", {{"metric", label}}));
        }
    }
    table_->clearContents();
    table_->setColumnCount(static_cast<int>(headers.size()));
    table_->setHorizontalHeaderLabels(headers);
    const int rows = static_cast<int>(std::min(segments_.size(), static_cast<std::size_t>(listed_segments)));
    table_->setRowCount(rows);
    for (int r = 0; r < rows; ++r) {
        const WeakSegment& segment = segments_[static_cast<std::size_t>(r)];
        table_->setItem(r, 0, cell(QString::number(segment.first_frame)));
        table_->setItem(r, 1, cell(QString::number(segment.last_frame)));
        table_->setItem(r, 2, cell(QString::number(segment.frames())));
        table_->setItem(r, 3, cell(time_text(segment.start_time)));
        table_->setItem(r, 4, cell(time_text(segment.end_time)));
        int column = 5;
        for (const SegmentMetric& metric : segment.metrics) {
            const MetricDefinition& definition = metric_definition(metric.key);
            // The worst end: Butteraugli's highest.
            const std::optional<double> worst =
                definition.direction == MetricDirection::lower_is_better ? metric.max : metric.min;
            table_->setItem(r, column++, cell(qt_text(definition.format_value(worst))));
            table_->setItem(r, column++, cell(qt_text(definition.format_value(metric.mean))));
        }
    }
}

void WeakSegmentsDialog::export_segments(ExportKind kind) {
    const SegmentRules current = rules();
    QString suffix;
    QString filter;
    std::string text;
    switch (kind) {
    case ExportKind::json:
        suffix = QStringLiteral(".segments.json");
        filter = app::tr("JSON file (*.json)");
        text = python_json_dumps(segments_json(segments_, current)) + "\n";
        break;
    case ExportKind::csv:
        suffix = QStringLiteral(".segments.csv");
        filter = app::tr("CSV file (*.csv)");
        text = segments_csv(segments_, current);
        break;
    }
    const QString path =
        dialogs_.save_file(app::tr("Export weak segments"), suggested_file_name(video_.label, suffix), filter);
    if (path.isEmpty()) {
        return;
    }
    // A text file, as the command line writes them: CR LF on Windows.
    std::ofstream file(path_of(path));
    file << text;
    file.close();
    if (!file) {
        dialogs_.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

}  // namespace vml::app
