#include "worst_frames_panel.h"

#include <QAbstractItemView>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QSignalBlocker>
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
#include "vml/python_text.h"
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
    segments_button_ = new QPushButton(app::tr("Weak segments"));
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
    // The video chosen stays chosen while it is not shown (its scores are
    // cleared to be calculated again), and is shown again once it is back;
    // meanwhile the first video is shown.
    if (chosen_identity_.empty() && index >= 0) {
        chosen_identity_ = videos_[static_cast<std::size_t>(index)].identity;
    }
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

namespace {

// The conditions on none of the metrics `keys`.
std::vector<WeakCondition> conditions_not_on(const std::vector<WeakCondition>& conditions,
                                             const std::vector<std::string>& keys) {
    std::vector<WeakCondition> left;
    for (const WeakCondition& condition : conditions) {
        if (std::find(keys.begin(), keys.end(), condition.key) == keys.end()) {
            left.push_back(condition);
        }
    }
    return left;
}

}  // namespace

WeakSegmentsDialog::WeakSegmentsDialog(WeakFramesVideo video, const MetricDefinition& metric,
                                       std::optional<std::pair<int, int>> range,
                                       std::vector<SegmentProfile> profiles, const std::string& in_use,
                                       Dialogs dialogs, QWidget* parent)
    : QDialog(parent), video_(std::move(video)), range_(range), dialogs_(std::move(dialogs)),
      profiles_(std::move(profiles)) {
    if (profiles_.empty()) {
        profiles_.push_back({"", SegmentRules{}});
    }
    setWindowTitle(app::tr("Weak segments: {name}", {{"name", video_.label}}));
    for (const MetricDefinition& definition : metrics()) {
        if (score_samples(video_.result->metric_results, std::string(definition.key))) {
            keys_.emplace_back(definition.key);
        }
    }
    auto* layout = new QVBoxLayout(this);
    auto* profile_row = new QHBoxLayout;
    profile_row->addWidget(new QLabel(app::tr("Profile:")));
    profile_combo_ = new QComboBox;
    for (const SegmentProfile& profile : profiles_) {
        profile_combo_->addItem(profile.name.empty() ? app::tr("Default") : qt_text(profile.name),
                                qt_text(profile.name));
    }
    const int start = profile_combo_->findData(qt_text(in_use));
    profile_combo_->setCurrentIndex(start >= 0 ? start : 0);
    profile_row->addWidget(profile_combo_);
    // Settings' From preset, into the conditions shown here: unsaved, as an
    // edit is.
    preset_button_ = new QPushButton(app::tr("From preset"));
    preset_button_->setToolTip(app::tr("Loads a starting point for a kind of video into these conditions, "
                                       "replacing them, to adjust as needed. Save to profile keeps them."));
    auto* presets_menu = new QMenu(preset_button_);
    presets_menu->setToolTipsVisible(true);
    for (const SegmentPreset& preset : segment_presets()) {
        QAction* action = presets_menu->addAction(app::tr(preset.name));
        // One with no condition on this video's metrics would show none.
        const bool usable = std::ranges::any_of(preset.rules.conditions, [this](const WeakCondition& condition) {
            return std::find(keys_.begin(), keys_.end(), condition.key) != keys_.end();
        });
        action->setEnabled(usable);
        action->setToolTip(usable ? app::tr(preset.about)
                                  : app::tr(preset.about) + " " +
                                        app::tr("This video has no scores of the metrics it uses."));
        connect(action, &QAction::triggered, this, [this, &preset] { load_preset(preset); });  // the registry's
    }
    preset_button_->setMenu(presets_menu);
    profile_row->addWidget(preset_button_);
    profile_row->addStretch(1);
    layout->addLayout(profile_row);

    editor_ = new SegmentRulesEditor(keys_, true);
    connect(editor_, &SegmentRulesEditor::changed, this, [this] { refresh(); });
    layout->addWidget(editor_);
    connect(profile_combo_, &QComboBox::currentIndexChanged, this, [this, definition = &metric] {
        // The registry's: it outlives the dialog.
        start_from_profile(*definition);
        refresh();
    });

    // What is changed here holds for this dialog; Save to profile keeps it
    // in the profile picked, in Settings.
    auto* save_row = new QHBoxLayout;
    saved_note_ = new QLabel;
    saved_note_->setWordWrap(true);
    theme::style(saved_note_, "color: {muted};");
    save_row->addWidget(saved_note_, 1);
    save_button_ = new QPushButton(app::tr("Save to profile"));
    save_button_->setToolTip(app::tr("Keeps these conditions and options in the profile picked above, in Settings "
                                     "> Weak segments."));
    connect(save_button_, &QPushButton::clicked, this, [this] {
        saved_ = rules_to_save();
        const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                          [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
        if (profile != profiles_.end()) {
            profile->rules = saved_;
        }
        emit rules_saved(profile_name(), saved_);
        show_saved_state();
    });
    save_row->addWidget(save_button_);
    save_new_button_ = new QPushButton(app::tr("Save as new profile"));
    connect(save_new_button_, &QPushButton::clicked, this, [this] {
        const std::optional<QString> name =
            dialogs_.ask_text(app::tr("New weak segment profile"), app::tr("Profile name:"), QString());
        if (!name) {
            return;
        }
        const std::string text = std::string(python_strip(name->toStdString()));
        if (!segment_profile_name_free(profiles_, text)) {
            dialogs_.alert(app::tr("Profile not saved"),
                           app::tr("A profile needs a name of its own, without commas, and not Default or another "
                               "profile's."));
            return;
        }
        saved_ = rules_to_save();
        profiles_.push_back({text, saved_});
        {
            const QSignalBlocker blocker(profile_combo_);  // shown as it is, not started again
            profile_combo_->addItem(qt_text(text), qt_text(text));
            profile_combo_->setCurrentIndex(profile_combo_->count() - 1);
        }
        // Picked before the window hands the profiles back (set_profiles).
        emit rules_saved(text, saved_);
        show_saved_state();
    });
    save_row->addWidget(save_new_button_);
    layout->addLayout(save_row);

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
    add_export(app::tr("Export JSON"), ExportKind::json);
    add_export(app::tr("Export CSV"), ExportKind::csv);
    QPushButton* report = buttons->addButton(app::tr("Export all videos"), QDialogButtonBox::ActionRole);
    report->setToolTip(app::tr("Every scored test video's weak segments by these conditions, in one JSON or CSV file."));
    connect(report, &QPushButton::clicked, this, [this] { emit report_requested(editor_->rules()); });
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);

    start_from_profile(metric);
    refresh();
    resize(760, 620);
}

void WeakSegmentsDialog::set_profiles(std::vector<SegmentProfile> profiles, const std::string& in_use) {
    profiles_ = std::move(profiles);
    if (profiles_.empty()) {
        profiles_.push_back({"", SegmentRules{}});
    }
    {
        const QSignalBlocker blocker(profile_combo_);  // what is shown stays, unsaved changes too
        const QString picked = profile_combo_->currentData().toString();
        profile_combo_->clear();
        for (const SegmentProfile& profile : profiles_) {
            profile_combo_->addItem(profile.name.empty() ? app::tr("Default") : qt_text(profile.name),
                                    qt_text(profile.name));
        }
        // Names are the same in any case, as Settings takes them.
        int index = profile_combo_->findData(picked, Qt::UserRole, Qt::MatchFixedString);
        if (index < 0) {
            index = profile_combo_->findData(qt_text(in_use), Qt::UserRole, Qt::MatchFixedString);
        }
        profile_combo_->setCurrentIndex(std::max(index, 0));
    }
    const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                      [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
    saved_ = profile != profiles_.end() ? profile->rules : SegmentRules{};
    saved_.ranges.clear();
    editor_->set_remembered(saved_.conditions);
    show_saved_state();
}

std::string WeakSegmentsDialog::profile_name() const { return profile_combo_->currentData().toString().toStdString(); }

void WeakSegmentsDialog::start_from_profile(const MetricDefinition& metric) {
    const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                      [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
    saved_ = profile != profiles_.end() ? profile->rules : SegmentRules{};
    saved_.ranges.clear();
    unshown_ = conditions_not_on(saved_.conditions, keys_);
    editor_->set_remembered(saved_.conditions);
    editor_->set_rules(saved_);
    if (editor_->condition_count() == 0) {
        // None in the profile on this video's metrics: one on the metric shown.
        SegmentRules start = saved_;
        start.conditions = {default_weak_condition(metric)};
        editor_->set_rules(start);
    }
}

void WeakSegmentsDialog::load_preset(const SegmentPreset& preset) {
    // As Settings loads one into a profile: every rule replaced, the
    // conditions on metrics this video has none of too (kept for Save to
    // profile). The menu offers only presets with a condition shown here.
    unshown_ = conditions_not_on(preset.rules.conditions, keys_);
    editor_->set_remembered(preset.rules.conditions);
    editor_->set_rules(preset.rules);
    refresh();
}

SegmentRules WeakSegmentsDialog::rules() const {
    SegmentRules rules = editor_->rules();
    if (range_) {
        rules.ranges.push_back({range_->first, range_->second});
    }
    return rules;
}

SegmentRules WeakSegmentsDialog::rules_to_save() const {
    SegmentRules saving = editor_->rules();
    saving.conditions.insert(saving.conditions.end(), unshown_.begin(), unshown_.end());
    return saving;
}

void WeakSegmentsDialog::show_saved_state() {
    const SegmentRules shown = rules_to_save();
    // The same conditions in another order are the same rules: the saved ones
    // this video has no metric for come last here, wherever they were.
    std::vector<std::string> shown_texts = condition_texts(shown.conditions);
    std::vector<std::string> saved_texts = condition_texts(saved_.conditions);
    std::ranges::sort(shown_texts);
    std::ranges::sort(saved_texts);
    const bool differ = shown_texts != saved_texts ||
                        shown.all != saved_.all || shown.min_frames != saved_.min_frames ||
                        shown.merge_frames != saved_.merge_frames;
    save_button_->setEnabled(differ);
    const QString name = profile_combo_->currentText();
    saved_note_->setText(differ ? app::tr("Changed here only: Save to profile keeps these in {profile}.",
                                          {{"profile", name}})
                                : app::tr("As saved in profile {profile}.", {{"profile", name}}));
}

void WeakSegmentsDialog::refresh() {
    if (summary_ == nullptr || table_ == nullptr || save_button_ == nullptr) {
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
    show_saved_state();
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
    // Named after the metrics it covers: an export by another metric does not
    // take the last one's name.
    QString suffix = "." + qt_text(segments_file_suffix(current));
    QString filter;
    std::string text;
    switch (kind) {
    case ExportKind::json:
        suffix += QStringLiteral(".json");
        filter = app::tr("JSON file (*.json)");
        text = python_json_dumps(segments_json(segments_, current)) + "\n";
        break;
    case ExportKind::csv:
        suffix += QStringLiteral(".csv");
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
