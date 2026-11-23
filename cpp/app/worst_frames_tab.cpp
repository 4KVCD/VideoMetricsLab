#include "worst_frames_tab.h"

#include <QAbstractItemView>
#include <QBoxLayout>
#include <QComboBox>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QEvent>
#include <QFontMetrics>
#include <QScopeGuard>
#include <QScrollBar>
#include <QSizePolicy>
#include <QSplitter>
#include <QSignalBlocker>
#include <QTabBar>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <set>
#include <cmath>
#include <fstream>
#include <iterator>
#include <span>

#include "chart.h"
#include "export_image.h"
#include "i18n.h"
#include "theme.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/time_format.h"

namespace vml::app {

namespace {

// How many segments the list shows; the exports have them all.
constexpr int listed_segments = 1000;

// A condition as the editor's rows read it: "VMAF v0.6.1 < 85".
QString condition_label(const WeakCondition& condition) {
    const MetricDefinition& metric = metric_definition(condition.key);
    return qt_text(metric.short_label) + (condition.comparison == '<' ? QStringLiteral(" < ") : QStringLiteral(" > ")) +
           qt_text(metric.format_value(condition.threshold));
}

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
    // One line a row: wrapped, a long name kept only its first line, cut at
    // its end whatever the elide mode.
    table->setWordWrap(false);
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    return table;
}

QString time_text(double seconds) { return qt_text(format_hms(seconds, 3)); }

// How near (in pixels) a worst frame's dot must be for the pointer to take it
// rather than the frame under it: a dip's worst frame is easy to land on.
constexpr int snap_pixels = 6;

// Rows of (label, value), the values lined up after the longest label: for
// the monospaced Details.
QStringList aligned(const std::vector<std::pair<QString, QString>>& rows) {
    qsizetype width = 0;
    for (const auto& [label, value] : rows) {
        width = std::max(width, label.size());
    }
    QStringList lines;
    for (const auto& [label, value] : rows) {
        lines << label.leftJustified(width + 2) + value;
    }
    return lines;
}

// The metrics `result` has scores of that the tab lists, in the app's order.
std::vector<std::string> metric_keys(const ComparisonResult& result) {
    std::vector<std::string> keys;
    for (const MetricDefinition& definition : metrics()) {
        if (score_samples(result.metric_results, std::string(definition.key))) {
            keys.emplace_back(definition.key);
        }
    }
    return keys;
}

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

// A table's width with all its columns: what it needs not to scroll sideways.
int natural_width(const QTableWidget* table) {
    return table->horizontalHeader()->length() + 2 * table->frameWidth() + table->verticalScrollBar()->sizeHint().width();
}

// The heading of a segment's score beside its mean: its worst, as for most
// conditions, else the end its conditions look for (vmaf>98: the highest).
QString end_heading(const MetricDefinition& metric, SegmentEnd end) {
    const QString label = qt_text(metric.short_label);
    return is_worst_end(end, metric)       ? app::tr("{metric} worst", {{"metric", label}})
           : end == SegmentEnd::highest ? app::tr("{metric} highest", {{"metric", label}})
                                        : app::tr("{metric} lowest", {{"metric", label}});
}

// `samples`' worst score within `ranges` (all: none given), by `direction`.
std::optional<double> worst_score(const ScoreSamples& samples, MetricDirection direction,
                                  const std::vector<FrameInterval>& ranges) {
    std::optional<double> worst;
    for (std::size_t i = 0; i < samples.values.size(); ++i) {
        const int frame = samples.frame[i];
        if (!ranges.empty() && std::none_of(ranges.begin(), ranges.end(), [frame](const FrameInterval& range) {
                return frame >= range.first && frame <= range.last;
            })) {
            continue;
        }
        const double value = samples.values[i];
        if (!std::isfinite(value)) {
            continue;
        }
        if (!worst || (direction == MetricDirection::lower_is_better ? value > *worst : value < *worst)) {
            worst = value;
        }
    }
    return worst;
}

}  // namespace

// ------------------------------------------------------------------ building

WorstFramesTab::WorstFramesTab(Dialogs dialogs, QWidget* parent) : QWidget(parent), dialogs_(std::move(dialogs)) {
    auto* root = new QHBoxLayout(this);

    // The videos, each with its weak segments and its worst score.
    QBoxLayout* side = nullptr;
    QWidget* videos_card = theme::container("card", false, &side);
    videos_card->setObjectName("worstVideos");
    auto* videos_title = new QLabel(app::tr("Videos"));
    theme::set_role(videos_title, "title");
    side->addWidget(videos_title);
    videos_table_ = make_table();
    videos_table_->setColumnCount(3);
    videos_table_->setHorizontalHeaderLabels({app::tr("Video"), app::tr("Segments"), app::tr("Worst")});
    videos_table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    // Each video as Metric Graphs names it (Settings > General > Window >
    // Video names). A name still too long is cut in its middle: its end (the
    // CRF, the encoder) tells encodes apart. The whole name is in the tooltip.
    videos_table_->setTextElideMode(Qt::ElideMiddle);
    connect(videos_table_, &QTableWidget::currentCellChanged, this, [this](int row, int, int, int) {
        if (!building_ && row >= 0 && static_cast<std::size_t>(row) < videos_.size()) {
            wanted_identity_ = videos_[static_cast<std::size_t>(row)].identity;
            wanted_file_ = videos_[static_cast<std::size_t>(row)].result->distorted;
            wanted_reference_ = videos_[static_cast<std::size_t>(row)].result->source;
            choose_video(row);
        }
    });
    side->addWidget(videos_table_);

    // On the left what the lists are made from: the videos (most score one
    // to three at a time: the list only as tall as its rows) and, under
    // them, the conditions. On the right what they find.
    inputs_ = new QWidget;
    auto* inputs = new QVBoxLayout(inputs_);
    inputs->setContentsMargins(0, 0, 0, 0);
    inputs->addWidget(videos_card);
    inputs->addWidget(build_conditions());
    inputs->addWidget(build_details());
    inputs->addStretch(1);

    // The work: the metric, the graph and the two lists.
    work_ = new QWidget;
    auto* work = new QVBoxLayout(work_);
    work->setContentsMargins(0, 0, 0, 0);
    work->addWidget(build_toolbar());
    QBoxLayout* graph = nullptr;
    QWidget* graph_card = theme::container("card", false, &graph);
    graph_card->setObjectName("worstGraph");
    chart_layout_ = static_cast<QVBoxLayout*>(graph);
    video_title_ = new ElidedLabel;
    video_title_->set_elide_mode(Qt::ElideMiddle);
    theme::set_role(video_title_, "title");
    video_title_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    auto* graph_head = new QHBoxLayout;
    graph_head->addWidget(video_title_, 1);
    graph_head->addWidget(export_button([this](QMenu* menu) { fill_graph_menu(menu); }));
    chart_layout_->addLayout(graph_head);
    // The graph as tall as the lists: with the conditions on the left, the
    // height they had is the graph's.
    work->addWidget(graph_card, 1);
    work->addWidget(build_lists(), 1);
    // The inputs beside the work, the divider between them dragged to give
    // long names room.
    split_ = new QSplitter(Qt::Horizontal);
    split_->setChildrenCollapsible(false);
    split_->addWidget(inputs_);
    split_->addWidget(work_);
    split_->setStretchFactor(1, 1);
    split_->setSizes({380, 1000});
    work_->installEventFilter(this);
    root->addWidget(split_, 1);

    empty_note_ = new QLabel(app::tr("No video has scores yet: calculate metrics on the Videos tab, or open saved "
                                     "results in Metric Graphs."));
    empty_note_->setWordWrap(true);
    empty_note_->setAlignment(Qt::AlignCenter);
    theme::style(empty_note_, "color: {muted};");
    root->addWidget(empty_note_, 1);

    building_ = false;
    refresh();
}

QWidget* WorstFramesTab::build_toolbar() {
    QBoxLayout* row = nullptr;
    QWidget* bar = theme::container("toolbar", true, &row);
    bar->setObjectName("worstToolbar");
    row->addWidget(new QLabel(app::tr("Metric")));
    metric_bar_ = new QTabBar;
    metric_bar_->setDrawBase(false);
    metric_bar_->setExpanding(false);
    theme::set_role(metric_bar_, "segmented");
    connect(metric_bar_, &QTabBar::currentChanged, this, [this](int index) {
        if (!building_ && index >= 0) {
            choose_metric(metric_bar_->tabData(index).toString().toStdString());
        }
    });
    // The bar takes the whole row, its tabs at its left: given only its
    // size hint, it showed scroll arrows once the selected (semibold) tab
    // had widened it. Export is in the Weak segments card, what it exports:
    // here it took the room of a metric, and at 1280 px ten (HDR VMAF's
    // with the others) needed scroll arrows.
    row->addWidget(metric_bar_, 1);
    return bar;
}

QPushButton* WorstFramesTab::export_button(const std::function<void(QMenu*)>& fill) {
    // The Export menu of what its card shows, filled as it opens: the items
    // usable then, as the right-click menu has them.
    auto* button = new QPushButton(app::tr("Export"));
    auto* menu = new QMenu(button);
    menu->setToolTipsVisible(true);
    connect(menu, &QMenu::aboutToShow, this, [menu, fill] {
        menu->clear();
        fill(menu);
    });
    button->setMenu(menu);
    return button;
}

void WorstFramesTab::offer_menu(QWidget* view, const std::function<void(QMenu*)>& fill) {
    // A list's or the graph's right-click: its card's Export items.
    view->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(view, &QWidget::customContextMenuRequested, this, [view, fill](const QPoint& at) {
        QMenu menu;
        menu.setToolTipsVisible(true);
        fill(&menu);
        const auto* table = qobject_cast<const QAbstractScrollArea*>(view);
        menu.exec((table != nullptr ? table->viewport() : view)->mapToGlobal(at));
    });
}

void WorstFramesTab::fill_worst_menu(QMenu* menu) {
    // The dips listed: frame, time and every metric's score at each.
    const bool any = !worst_.empty();
    QAction* csv = menu->addAction(app::tr("Worst frames as CSV..."));
    connect(csv, &QAction::triggered, this, [this] { export_worst(ExportKind::csv); });
    QAction* json = menu->addAction(app::tr("Worst frames as JSON..."));
    connect(json, &QAction::triggered, this, [this] { export_worst(ExportKind::json); });
    csv->setEnabled(any);
    json->setEnabled(any);
}

void WorstFramesTab::fill_graph_menu(QMenu* menu) {
    QAction* png = menu->addAction(app::tr("Graph as PNG..."));
    connect(png, &QAction::triggered, this, [this] { export_graph(); });
    png->setEnabled(chart_ != nullptr && chosen_video() != nullptr);
}

void WorstFramesTab::fill_segments_menu(QMenu* menu) {
    // This video's segments by the conditions shown, or every scored test
    // video's in one file.
    const bool video = chosen_video() != nullptr && editor_ != nullptr;
    QAction* json = menu->addAction(app::tr("Weak segments as JSON..."));
    connect(json, &QAction::triggered, this, [this] { export_segments(ExportKind::json); });
    QAction* csv = menu->addAction(app::tr("Weak segments as CSV..."));
    connect(csv, &QAction::triggered, this, [this] { export_segments(ExportKind::csv); });
    QAction* report = menu->addAction(app::tr("All videos' weak segments as JSON or CSV..."));
    report->setToolTip(app::tr("Every scored test video's weak segments by these conditions, in one JSON or CSV file."));
    connect(report, &QAction::triggered, this, [this] {
        SegmentRules rules = editor_ != nullptr ? editor_->rules() : SegmentRules{};
        emit report_requested(rules);
    });
    json->setEnabled(video);
    csv->setEnabled(video);
    report->setEnabled(video);
}

QWidget* WorstFramesTab::build_conditions() {
    // In the left column, one thing under another: the profile and saving
    // to one, the conditions (the editor, made again for each video's
    // metrics: choose_video), then what they find and why.
    QBoxLayout* column = nullptr;
    QWidget* card = theme::container("card", false, &column);
    card->setObjectName("worstConditions");
    auto* title = new QLabel(app::tr("Conditions"));
    theme::set_role(title, "title");
    column->addWidget(title);
    // The profile the conditions start from, and saving them to one.
    auto* profile_row = new QHBoxLayout;
    profile_row->addWidget(new QLabel(app::tr("Profile:")));
    profile_combo_ = new QComboBox;
    connect(profile_combo_, &QComboBox::currentIndexChanged, this, [this] {
        if (!building_) {
            start_from_profile();
            refresh();
        }
    });
    profile_row->addWidget(profile_combo_);
    preset_button_ = new QPushButton(app::tr("From preset"));
    preset_button_->setToolTip(app::tr("Loads a starting point for a kind of video into these conditions, "
                                       "replacing them, to adjust as needed. Save to profile keeps them."));
    preset_button_->setMenu(new QMenu(preset_button_));
    profile_row->addWidget(preset_button_);
    save_button_ = new QPushButton(app::tr("Save to profile"));
    save_button_->setToolTip(app::tr("Keeps these conditions and options in the profile picked, in Settings > Weak "
                                     "segments."));
    connect(save_button_, &QPushButton::clicked, this, [this] {
        saved_ = rules_to_save();
        preset_loaded_ = false;  // the profile has the preset's conditions now
        const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                          [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
        if (profile != profiles_.end()) {
            profile->rules = saved_;
        }
        emit rules_saved(profile_name(), saved_);
        show_saved_state();
    });
    profile_row->addStretch(1);
    column->addLayout(profile_row);
    auto* save_row = new QHBoxLayout;
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
        preset_loaded_ = false;
        profiles_.push_back({text, saved_});
        {
            const QSignalBlocker blocker(profile_combo_);  // shown as it is, not started again
            profile_combo_->addItem(qt_text(text), qt_text(text));
            profile_combo_->setCurrentIndex(profile_combo_->count() - 1);
        }
        emit rules_saved(text, saved_);  // the window hands the profiles back (set_segment_profiles)
        show_saved_state();
    });
    save_row->addWidget(save_new_button_);
    save_row->addStretch(1);
    column->addLayout(save_row);
    editor_holder_ = new QWidget;
    auto* holder = new QVBoxLayout(editor_holder_);
    holder->setContentsMargins(0, 0, 0, 0);
    column->addWidget(editor_holder_);
    saved_note_ = new QLabel;
    saved_note_->setWordWrap(true);
    theme::style(saved_note_, "color: {muted};");
    column->addWidget(saved_note_);
    summary_ = new QLabel;
    summary_->setWordWrap(true);
    column->addWidget(summary_);
    conditions_note_ = new QLabel;
    conditions_note_->setWordWrap(true);
    conditions_note_->setObjectName("conditionsNote");
    theme::style(conditions_note_, "color: {muted};");
    column->addWidget(conditions_note_);
    return card;
}

QWidget* WorstFramesTab::build_details() {
    // What the pointer is on, in the column's room under the conditions: a
    // frame's every score and why it is weak, as Metric Graphs' readout
    // tells its frames.
    QBoxLayout* column = nullptr;
    QWidget* card = theme::container("card", false, &column);
    card->setObjectName("worstDetails");
    auto* title = new QLabel(app::tr("Details"));
    theme::set_role(title, "title");
    column->addWidget(title);
    // Under the readout, what to do next: before anything is chosen, how to
    // choose; with a row chosen, that a double-click opens it in Video
    // Compare (the lists' only way there).
    details_hint_ = new QLabel;
    details_hint_->setWordWrap(true);
    theme::style(details_hint_, "color: {muted};");
    details_ = new QLabel;
    details_->setObjectName("worstDetailsText");
    details_->setTextFormat(Qt::PlainText);
    QFont font{QString::fromLatin1(number_font_family)};
    font.setStyleHint(QFont::Monospace);
    details_->setFont(font);
    details_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    // As wide as the column whatever it says (a line too long is cut, the
    // column not widened), and never less than two lines tall: on a short
    // window the frame and the score graphed, the rest cut.
    details_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    details_->setMinimumHeight(2 * QFontMetrics(font).lineSpacing());
    details_->hide();
    column->addWidget(details_);
    column->addWidget(details_hint_);
    return card;
}

QWidget* WorstFramesTab::build_lists() {
    auto* lists = new QWidget;
    lists_layout_ = new QBoxLayout(QBoxLayout::LeftToRight, lists);
    lists_layout_->setContentsMargins(0, 0, 0, 0);

    QBoxLayout* worst = nullptr;
    worst_card_ = theme::container("card", false, &worst);
    worst_card_->setObjectName("worstList");
    auto* worst_head = new QHBoxLayout;
    auto* worst_title = new QLabel(app::tr("Worst frames"));
    theme::set_role(worst_title, "title");
    worst_head->addWidget(worst_title);
    worst_head->addStretch(1);
    count_combo_ = new QComboBox;
    for (const int count : worst_frame_counts) {
        count_combo_->addItem(ntr("{count} dip", "{count} dips", count), count);
    }
    connect(count_combo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index >= 0 && !building_) {
            count_ = count_combo_->itemData(index).toInt();
            fill_worst();
            refresh_chart();
            show_details();  // the list made again: a row selected is gone
            emit count_changed(count_);
        }
    });
    worst_head->addWidget(count_combo_);
    worst_head->addWidget(export_button([this](QMenu* menu) { fill_worst_menu(menu); }));
    worst->addLayout(worst_head);
    worst_table_ = make_table();
    offer_menu(worst_table_, [this](QMenu* menu) { fill_worst_menu(menu); });
    connect(worst_table_, &QTableWidget::itemSelectionChanged, this, [this] {
        const QList<QTableWidgetItem*> selected = worst_table_->selectedItems();
        if (selected.isEmpty()) {
            return;
        }
        {
            const QSignalBlocker blocker(segments_table_);
            segments_table_->clearSelection();
        }
        if (const std::optional<int> frame = worst_frame_of_row(selected.front()->row()); frame && chart_ != nullptr) {
            chart_->set_highlight(std::nullopt);
            const auto found = std::find_if(worst_.begin(), worst_.end(),
                                             [&](const WorstFrame& w) { return w.frame == *frame; });
            if (found != worst_.end()) {
                chart_->bring_into_view(found->time);
                chart_->set_cursor_time(found->time);
            }
        }
        show_details();
    });
    connect(worst_table_, &QTableWidget::cellDoubleClicked, this, [this](int, int) { open_selected(); });
    worst->addWidget(worst_table_, 1);
    worst_note_ = new QLabel;
    worst_note_->setWordWrap(true);
    theme::style(worst_note_, "color: {muted};");
    worst->addWidget(worst_note_);
    lists_layout_->addWidget(worst_card_, 2);

    QBoxLayout* segments = nullptr;
    segments_card_ = theme::container("card", false, &segments);
    segments_card_->setObjectName("segmentsList");
    auto* segments_head = new QHBoxLayout;
    auto* segments_title = new QLabel(app::tr("Weak segments"));
    theme::set_role(segments_title, "title");
    segments_head->addWidget(segments_title);
    segments_head->addStretch(1);
    segments_head->addWidget(export_button([this](QMenu* menu) { fill_segments_menu(menu); }));
    segments->addLayout(segments_head);
    segments_table_ = make_table();
    offer_menu(segments_table_, [this](QMenu* menu) { fill_segments_menu(menu); });
    connect(segments_table_, &QTableWidget::itemSelectionChanged, this, [this] {
        const QList<QTableWidgetItem*> selected = segments_table_->selectedItems();
        if (selected.isEmpty()) {
            return;
        }
        {
            const QSignalBlocker blocker(worst_table_);
            worst_table_->clearSelection();
        }
        const int row = selected.front()->row();
        if (row >= 0 && static_cast<std::size_t>(row) < segments_.size() && chart_ != nullptr) {
            const WeakSegment& segment = segments_[static_cast<std::size_t>(row)];
            chart_->set_highlight(std::pair(segment.start_time, segment.end_time));
            chart_->set_cursor_time(std::nullopt);
            chart_->bring_into_view(segment.start_time);
        }
        show_details();
    });
    connect(segments_table_, &QTableWidget::cellDoubleClicked, this, [this](int, int) { open_selected(); });
    segments->addWidget(segments_table_, 1);
    lists_layout_->addWidget(segments_card_, 3);
    return lists;
}

// --------------------------------------------------------------- the inputs

void WorstFramesTab::set_videos(std::vector<WeakFramesVideo> videos) {
    // Only those with scores of a metric the tab can list.
    std::erase_if(videos, [](const WeakFramesVideo& video) {
        return video.result == nullptr || std::none_of(metrics().begin(), metrics().end(), [&](const MetricDefinition& m) {
                   return score_samples(video.result->metric_results, std::string(m.key)).has_value();
               });
    });
    std::set<std::string> listed;  // the lines listed before
    for (const WeakFramesVideo& video : videos_) {
        listed.insert(video.identity);
    }
    videos_ = std::move(videos);
    // A video scored again is a new line, a new identity, of the same file:
    // found by its test video and reference where its identity is gone, if
    // one line alone, new since, has them. Its CVVDP display rows share both,
    // as does the file scored against the same reference twice: lines listed
    // before are not it, and with two new ones which it is is not known, so
    // none is taken for it (#221's review: a sibling was).
    const auto find = [this, &listed](const std::string& identity, const std::filesystem::path& file,
                                      const std::filesystem::path& reference) {
        const auto by_identity = std::find_if(videos_.begin(), videos_.end(),
                                              [&](const WeakFramesVideo& video) { return video.identity == identity; });
        if (by_identity != videos_.end() || file.empty()) {
            return by_identity;
        }
        const auto alike = [&](const WeakFramesVideo& video) {
            return video.result->distorted == file && video.result->source == reference &&
                   !listed.contains(video.identity);
        };
        const auto first = std::find_if(videos_.begin(), videos_.end(), alike);
        return first != videos_.end() && std::count_if(videos_.begin(), videos_.end(), alike) == 1 ? first
                                                                                                    : videos_.end();
    };
    const auto wanted = find(wanted_identity_, wanted_file_, wanted_reference_);
    const int index = videos_.empty() ? -1 : wanted == videos_.end() ? 0 : static_cast<int>(wanted - videos_.begin());
    split_->setVisible(!videos_.empty());
    empty_note_->setVisible(videos_.empty());
    {
        const QSignalBlocker blocker(videos_table_);
        videos_table_->setRowCount(static_cast<int>(videos_.size()));
        for (std::size_t i = 0; i < videos_.size(); ++i) {
            // As Metric Graphs names it (the Video names setting), the
            // whole name in the tooltip.
            auto* name = cell(videos_[i].shown.isEmpty() ? videos_[i].label : videos_[i].shown, false);
            name->setToolTip(videos_[i].label);
            videos_table_->setItem(static_cast<int>(i), 0, name);
        }
    }
    fit_videos_height();
    if (index >= 0) {
        const WeakFramesVideo& video = videos_[static_cast<std::size_t>(index)];
        const bool same =
            editor_ != nullptr && find(chosen_identity_, chosen_file_, chosen_reference_) - videos_.begin() == index;
        {
            const QSignalBlocker blocker(videos_table_);
            videos_table_->setCurrentCell(index, 0);
        }
        if (same && video.identity != wanted_identity_ && wanted_file_ == video.result->distorted &&
            wanted_reference_ == video.result->source) {
            wanted_identity_ = video.identity;  // the chosen video, scored again
        }
        if (same && metric_keys(*video.result) == keys_) {
            chosen_identity_ = video.identity;
            refresh();  // its scores may be new: the rules shown stay
        } else {
            choose_video(index);
        }
    } else {
        chosen_identity_.clear();
        metric_ = nullptr;
        refresh();
    }
}

void WorstFramesTab::set_range(std::optional<std::pair<int, int>> range) {
    if (range != range_) {
        range_ = range;
        refresh();
    }
}

void WorstFramesTab::set_segment_profiles(std::vector<SegmentProfile> profiles, const std::string& in_use) {
    profiles_ = std::move(profiles);
    profile_in_use_ = in_use;
    if (profiles_.empty()) {
        profiles_.push_back({"", SegmentRules{}});
    }
    bool started = false;
    {
        const QSignalBlocker blocker(profile_combo_);  // what is shown stays, unsaved changes too
        const QString picked = profile_combo_->currentData().toString();
        const bool first = profile_combo_->count() == 0;
        profile_combo_->clear();
        for (const SegmentProfile& profile : profiles_) {
            profile_combo_->addItem(profile.name.empty() ? app::tr("Default") : qt_text(profile.name),
                                    qt_text(profile.name));
        }
        // Names are the same in any case, as Settings takes them. The
        // profile picked gone (deleted in Settings): the one in use is picked,
        // with the conditions shown kept.
        int index = first ? -1 : profile_combo_->findData(picked, Qt::UserRole, Qt::MatchFixedString);
        if (index < 0) {
            index = profile_combo_->findData(qt_text(in_use), Qt::UserRole, Qt::MatchFixedString);
        }
        started = first;
        profile_combo_->setCurrentIndex(std::max(index, 0));
    }
    if (started && editor_ != nullptr) {
        start_from_profile();
        refresh();
        return;
    }
    const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                      [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
    saved_ = profile != profiles_.end() ? profile->rules : SegmentRules{};
    saved_.ranges.clear();
    // Save to profile keeps the profile's conditions on metrics this video
    // lacks as Settings has them now; a preset loaded keeps its own.
    if (!preset_loaded_ && editor_ != nullptr) {
        unshown_ = conditions_not_on(saved_.conditions, keys_);
        editor_->set_remembered(saved_.conditions);
    }
    show_saved_state();
}

void WorstFramesTab::set_worst_count(int count) {
    const auto found = std::find(worst_frame_counts.begin(), worst_frame_counts.end(), count);
    const int index = found == worst_frame_counts.end() ? 0 : static_cast<int>(found - worst_frame_counts.begin());
    {
        const QSignalBlocker blocker(count_combo_);
        count_combo_->setCurrentIndex(index);
    }
    count_ = worst_frame_counts[static_cast<std::size_t>(index)];
    fill_worst();
    refresh_chart();
    show_details();  // the list made again: a row selected is gone
}

// ---------------------------------------------------------------- choosing

void WorstFramesTab::choose_video(int index) {
    if (index < 0 || static_cast<std::size_t>(index) >= videos_.size()) {
        return;
    }
    const WeakFramesVideo& video = videos_[static_cast<std::size_t>(index)];
    chosen_identity_ = video.identity;
    chosen_file_ = video.result->distorted;
    chosen_reference_ = video.result->source;
    if (wanted_identity_.empty()) {
        wanted_identity_ = video.identity;
        wanted_file_ = video.result->distorted;
        wanted_reference_ = video.result->source;
    }
    keys_ = metric_keys(*video.result);
    // Room for a frame's lines (its time, its rank, why it is weak, a blank,
    // a score per metric), so that Details does not grow as the pointer
    // moves from one frame to another.
    details_->setMinimumHeight(static_cast<int>(keys_.size() + 5) * details_->fontMetrics().lineSpacing());
    // The metric shown stays where this video has it.
    const std::string previous = metric_ != nullptr ? std::string(metric_->key) : std::string();
    const std::string key =
        std::find(keys_.begin(), keys_.end(), previous) != keys_.end() ? previous : keys_.empty() ? "" : keys_.front();
    metric_ = key.empty() ? nullptr : &metric_definition(key);
    {
        const QSignalBlocker blocker(metric_bar_);
        while (metric_bar_->count() > 0) {
            metric_bar_->removeTab(0);
        }
        for (const std::string& k : keys_) {
            const int tab = metric_bar_->addTab(qt_text(metric_definition(k).short_label));
            metric_bar_->setTabData(tab, qt_text(k));
            if (k == key) {
                metric_bar_->setCurrentIndex(tab);
            }
        }
    }
    // The conditions edited here, on this video's metrics; those shown stay
    // where they are on its metrics too.
    const SegmentRules shown = editor_ != nullptr ? editor_->rules() : SegmentRules{};
    delete editor_;
    editor_ = new SegmentRulesEditor(keys_, true);
    editor_->wrap_long_options();
    connect(editor_, &SegmentRulesEditor::changed, this, [this] { refresh(); });
    editor_holder_->layout()->addWidget(editor_);
    // From preset: those with a condition on this video's metrics.
    QMenu* presets = preset_button_->menu();
    presets->clear();
    presets->setToolTipsVisible(true);
    for (const SegmentPreset& preset : segment_presets()) {
        QAction* action = presets->addAction(app::tr(preset.name));
        const bool usable = std::ranges::any_of(preset.rules.conditions, [this](const WeakCondition& condition) {
            return std::find(keys_.begin(), keys_.end(), condition.key) != keys_.end();
        });
        action->setEnabled(usable);
        action->setToolTip(usable ? app::tr(preset.about)
                                  : app::tr(preset.about) + " " +
                                        app::tr("This video has no scores of the metrics it uses."));
        connect(action, &QAction::triggered, this, [this, &preset] { load_preset(preset); });  // the registry's
    }
    const bool keep = !shown.conditions.empty() &&
                      std::ranges::all_of(shown.conditions, [this](const WeakCondition& condition) {
                          return std::find(keys_.begin(), keys_.end(), condition.key) != keys_.end();
                      });
    if (keep) {
        // With the conditions set aside for the video before (on metrics it
        // lacked) that are on this one's: judged by them as the exports and
        // the command line judge it, and each saved once.
        SegmentRules taken = shown;
        std::vector<WeakCondition> still_unshown;
        for (const WeakCondition& condition : unshown_) {
            if (std::find(keys_.begin(), keys_.end(), condition.key) != keys_.end()) {
                taken.conditions.push_back(condition);
            } else {
                still_unshown.push_back(condition);
            }
        }
        unshown_ = std::move(still_unshown);
        editor_->set_remembered(saved_.conditions);
        editor_->set_rules(taken);
    } else {
        start_from_profile();
    }
    refresh();
}

void WorstFramesTab::choose_metric(const std::string& key) {
    if (std::find(keys_.begin(), keys_.end(), key) == keys_.end()) {
        return;
    }
    metric_ = &metric_definition(key);
    {
        const QSignalBlocker blocker(metric_bar_);
        for (int tab = 0; tab < metric_bar_->count(); ++tab) {
            if (metric_bar_->tabData(tab).toString().toStdString() == key) {
                metric_bar_->setCurrentIndex(tab);
            }
        }
    }
    refresh();
}

std::string WorstFramesTab::profile_name() const { return profile_combo_->currentData().toString().toStdString(); }

void WorstFramesTab::start_from_profile() {
    if (editor_ == nullptr) {
        return;
    }
    const auto profile = std::find_if(profiles_.begin(), profiles_.end(),
                                      [name = profile_name()](const SegmentProfile& p) { return p.name == name; });
    saved_ = profile != profiles_.end() ? profile->rules : SegmentRules{};
    saved_.ranges.clear();
    unshown_ = conditions_not_on(saved_.conditions, keys_);
    preset_loaded_ = false;
    editor_->set_remembered(saved_.conditions);
    editor_->set_rules(saved_);
    if (editor_->condition_count() == 0 && metric_ != nullptr) {
        // None in the profile on this video's metrics: one on the metric shown
        // (starting_metric, as exports and the command line choose it).
        if (const MetricDefinition* metric = starting_metric(keys_, metric_->key)) {
            SegmentRules start = saved_;
            start.conditions = {default_weak_condition(*metric)};
            editor_->set_rules(start);
        }
    }
}

void WorstFramesTab::load_preset(const SegmentPreset& preset) {
    // As Settings loads one into a profile: every rule replaced, the
    // conditions on metrics this video has none of too (kept for Save to
    // profile). The menu offers only presets with a condition shown here.
    unshown_ = conditions_not_on(preset.rules.conditions, keys_);
    preset_loaded_ = true;
    editor_->set_remembered(preset.rules.conditions);
    editor_->set_rules(preset.rules);
    refresh();
}

SegmentRules WorstFramesTab::rules() const {
    SegmentRules rules = editor_ != nullptr ? editor_->rules() : SegmentRules{};
    if (range_) {
        rules.ranges.push_back({range_->first, range_->second});
    }
    return rules;
}

SegmentRules WorstFramesTab::rules_to_save() const {
    SegmentRules saving = editor_ != nullptr ? editor_->rules() : SegmentRules{};
    saving.conditions.insert(saving.conditions.end(), unshown_.begin(), unshown_.end());
    return saving;
}

const WeakFramesVideo* WorstFramesTab::chosen_video() const {
    const auto found = std::find_if(videos_.begin(), videos_.end(),
                                    [this](const WeakFramesVideo& video) { return video.identity == chosen_identity_; });
    return found == videos_.end() ? nullptr : &*found;
}

bool WorstFramesTab::lists_side_by_side() const { return lists_layout_->direction() == QBoxLayout::LeftToRight; }

// --------------------------------------------------------------- refreshing

void WorstFramesTab::refresh() {
    if (building_) {
        return;
    }
    const WeakFramesVideo* video = chosen_video();
    segments_ = video != nullptr && editor_ != nullptr ? weak_segments(*video->result, rules()) : std::vector<WeakSegment>{};
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
    if (range_) {
        summary += " " + app::tr("Only the selected frames, {first} to {last}.",
                                 {{"first", range_->first}, {"last", range_->second}});
    }
    summary_->setText(video != nullptr ? summary : QString());
    show_condition_counts();
    show_saved_state();
    fill_worst();
    fill_segments();
    refresh_videos_table();
    refresh_chart();
    show_details();
}

void WorstFramesTab::refresh_videos_table() {
    // Each video's segments by the rules shown, and its worst score on the
    // metric shown: which encode is weakest, at a glance.
    const SegmentRules current = rules();
    std::vector<FrameInterval> ranges = current.ranges;
    const QSignalBlocker blocker(videos_table_);
    const WeakFramesVideo* chosen = chosen_video();
    for (std::size_t i = 0; i < videos_.size(); ++i) {
        const WeakFramesVideo& video = videos_[i];
        const int row = static_cast<int>(i);
        // The chosen video's are segments_, found by refresh with these rules
        // a moment ago: not found again (about 3.5 ms for a two-hour film).
        const std::size_t count = editor_ == nullptr  ? 0
                                  : &video == chosen ? segments_.size()
                                                     : weak_segments(*video.result, current).size();
        videos_table_->setItem(row, 1, cell(QString::number(count)));
        QString worst;
        if (metric_ != nullptr) {
            if (const std::optional<ScoreSamples> samples =
                    score_samples(video.result->metric_results, std::string(metric_->key))) {
                if (const std::optional<double> score = worst_score(*samples, metric_->direction, ranges)) {
                    worst = qt_text(metric_->format_value(*score));
                }
            }
        }
        videos_table_->setItem(row, 2, cell(worst));
    }
}

void WorstFramesTab::refresh_chart() {
    // The chart is the metric's: its scale (VMAF's top of 100) and, for
    // one where lower is better, upside down, as Metric Graphs draws it.
    const WeakFramesVideo* video = chosen_video();
    if (chart_ != nullptr && (metric_ == nullptr || chart_->property("metric").toString() != qt_text(metric_->key))) {
        delete chart_;
        chart_ = nullptr;
        hover_frame_.reset();
    }
    if (video != nullptr) {
        video_title_->set_text(video->label);
    } else {
        video_title_->clear_text();
    }
    if (metric_ == nullptr || video == nullptr) {
        return;
    }
    if (chart_ == nullptr) {
        const bool inverted = metric_->direction == MetricDirection::lower_is_better;
        const QString axis = qt_text(metric_->axis_label);
        chart_ = new ChartWidget(inverted ? app::tr("{axis_label} (lower is better)", {{"axis_label", axis}}) : axis,
                                 metric_->fixed_y_max, inverted);
        chart_->setProperty("metric", qt_text(metric_->key));
        chart_->setMinimumHeight(100);  // room for the lists at 1280x800 without the page scrolling
        chart_layout_->addWidget(chart_, 1);  // the card's height to the graph, not around its title
        connect(chart_, &ChartWidget::hovered, this, [this](double time, double) { hover_at(time); });
        connect(chart_, &ChartWidget::left, this, [this] { hover_left(); });
        offer_menu(chart_, [this](QMenu* menu) { fill_graph_menu(menu); });
    }
    const std::optional<ScoreSamples> samples = score_samples(video->result->metric_results, std::string(metric_->key));
    if (!samples) {
        chart_->clear();
        return;
    }
    // A dot on each worst frame listed.
    marks_.assign(samples->values.size(), 0);
    for (const WorstFrame& worst : worst_) {
        const auto at = std::lower_bound(samples->frame.begin(), samples->frame.end(), worst.frame);
        if (at != samples->frame.end() && *at == worst.frame) {
            marks_[static_cast<std::size_t>(at - samples->frame.begin())] = 1;
        }
    }
    mark_colors_ = {QColor(), palette().color(QPalette::Highlight)};
    ChartSeries series{samples->time, samples->values, video->result, palette().color(QPalette::Link), true};
    series.marks = marks_;
    series.mark_colors = mark_colors_;
    chart_->set_series(0, series);
    std::vector<std::pair<double, double>> shaded;
    shaded.reserve(segments_.size());
    for (const WeakSegment& segment : segments_) {
        shaded.emplace_back(segment.start_time, segment.end_time);
    }
    chart_->set_shaded(std::move(shaded));
    chart_->set_highlight(std::nullopt);
    chart_->set_cursor_time(std::nullopt);
}

void WorstFramesTab::fill_worst() {
    // Its columns fitted to what it shows, on every way out, emptied too.
    const auto fitted = qScopeGuard([this] { fit_list(worst_table_, worst_width_); });
    worst_table_->clearContents();
    worst_table_->setRowCount(0);
    worst_.clear();
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr) {
        worst_table_->setColumnCount(0);
        worst_note_->clear();
        return;
    }
    const MetricResultSet& results = video->result->metric_results;
    const std::optional<ScoreSamples> samples = score_samples(results, std::string(metric_->key));
    if (!samples) {
        return;
    }
    // The metric listed, then the conditions' others: what fits beside the
    // segments. Every metric's score at the frame is in the row's tooltip.
    std::vector<std::pair<const MetricDefinition*, ScoreSamples>> columns{{metric_, *samples}};
    std::vector<std::pair<const MetricDefinition*, ScoreSamples>> all;
    for (const MetricDefinition& other : metrics()) {
        if (const std::optional<ScoreSamples> found = score_samples(results, std::string(other.key))) {
            all.emplace_back(&other, *found);
        }
    }
    if (editor_ != nullptr) {
        for (const WeakCondition& condition : editor_->rules().conditions) {
            const MetricDefinition* definition = &metric_definition(condition.key);
            const bool listed = std::any_of(columns.begin(), columns.end(),
                                            [definition](const auto& column) { return column.first == definition; });
            if (!listed) {
                if (const std::optional<ScoreSamples> found = score_samples(results, condition.key)) {
                    columns.emplace_back(definition, *found);
                }
            }
        }
    }
    QStringList headers{app::tr("Frame"), app::tr("Time")};
    for (const auto& [definition, column_samples] : columns) {
        headers.append(qt_text(definition->short_label));
    }
    worst_table_->setColumnCount(static_cast<int>(headers.size()));
    worst_table_->setHorizontalHeaderLabels(headers);

    std::vector<FrameInterval> ranges;
    if (range_) {
        ranges.push_back({range_->first, range_->second});
    }
    worst_ = worst_frames(*samples, metric_->direction, static_cast<std::size_t>(count_), ranges);
    worst_table_->setRowCount(static_cast<int>(worst_.size()));
    for (std::size_t row = 0; row < worst_.size(); ++row) {
        const int r = static_cast<int>(row);
        QStringList every;
        for (const auto& [definition, column_samples] : all) {
            every << qt_text(definition->short_label) + ": " +
                         qt_text(definition->format_value(score_at(column_samples, worst_[row].frame)));
        }
        const QString tooltip = every.join(QLatin1Char('\n'));
        QTableWidgetItem* frame = cell(QString::number(worst_[row].frame));
        frame->setData(Qt::UserRole, worst_[row].frame);
        frame->setToolTip(tooltip);
        worst_table_->setItem(r, 0, frame);
        QTableWidgetItem* time = cell(time_text(worst_[row].time));
        time->setToolTip(tooltip);
        worst_table_->setItem(r, 1, time);
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const auto& [definition, column_samples] = columns[c];
            QTableWidgetItem* score =
                cell(qt_text(definition->format_value(score_at(column_samples, worst_[row].frame))));
            score->setToolTip(tooltip);
            worst_table_->setItem(r, static_cast<int>(c) + 2, score);
        }
    }
    worst_note_->setText(samples->per_second
                             ? app::tr("CVVDP scores each second: a row is its second's first frame, and a dip is the "
                                       "seconds within 1 s of a worse one.")
                             : app::tr("The worst frame of each dip, worst first: a frame within 1 s of a worse one is "
                                       "part of its dip."));
}

void WorstFramesTab::fill_segments() {
    const SegmentRules current = rules();
    // As the dialog's table, without End (Last and Frames give it): it fits
    // beside Worst frames.
    QStringList headers{app::tr("First"), app::tr("Last"), app::tr("Frames"), app::tr("Start")};
    // Which conditions held in each segment, where there are several and any
    // of them makes a frame weak: with every one required, each segment's
    // frames hold them all.
    const bool weak_by = current.conditions.size() > 1 && !current.all;
    if (weak_by) {
        headers.append(app::tr("Weak by"));
    }
    std::vector<std::string> keys;
    for (const WeakCondition& condition : current.conditions) {
        if (std::find(keys.begin(), keys.end(), condition.key) == keys.end()) {
            keys.push_back(condition.key);
            const MetricDefinition& definition = metric_definition(condition.key);
            headers.append(end_heading(definition, segment_end(current, definition)));
            headers.append(app::tr("{metric} mean", {{"metric", qt_text(definition.short_label)}}));
        }
    }
    segments_table_->clearContents();
    segments_table_->setColumnCount(static_cast<int>(headers.size()));
    segments_table_->setHorizontalHeaderLabels(headers);
    const int rows = static_cast<int>(std::min(segments_.size(), static_cast<std::size_t>(listed_segments)));
    segments_table_->setRowCount(rows);
    for (int r = 0; r < rows; ++r) {
        const WeakSegment& segment = segments_[static_cast<std::size_t>(r)];
        segments_table_->setItem(r, 0, cell(QString::number(segment.first_frame)));
        segments_table_->setItem(r, 1, cell(QString::number(segment.last_frame)));
        segments_table_->setItem(r, 2, cell(QString::number(segment.frames())));
        QTableWidgetItem* start = cell(time_text(segment.start_time));
        start->setToolTip(app::tr("{start} to {end}", {{"start", time_text(segment.start_time)},
                                                         {"end", time_text(segment.end_time)}}));
        segments_table_->setItem(r, 3, start);
        int column = 4;
        if (weak_by) {
            QStringList held;
            QStringList frames;
            for (std::size_t c = 0; c < current.conditions.size() && c < segment.condition_frames.size(); ++c) {
                const QString condition = condition_label(current.conditions[c]);
                if (segment.condition_frames[c] > 0) {
                    held << condition;
                }
                frames << ntr("{condition}: {count} frame", "{condition}: {count} frames",
                              segment.condition_frames[c], {{"condition", condition}});
            }
            QTableWidgetItem* item = cell(held.join(QStringLiteral(", ")), false);
            item->setToolTip(frames.join(QLatin1Char('\n')));
            segments_table_->setItem(r, column++, item);
        }
        for (const SegmentMetric& metric : segment.metrics) {
            const MetricDefinition& definition = metric_definition(metric.key);
            const std::optional<double> end = segment_end_value(metric, segment_end(current, definition));
            segments_table_->setItem(r, column++, cell(qt_text(definition.format_value(end))));
            segments_table_->setItem(r, column++, cell(qt_text(definition.format_value(metric.mean))));
        }
    }
    fit_list(segments_table_, segments_width_);
}

void WorstFramesTab::show_condition_counts() {
    // Each condition's frames, and how many are in the segments: a condition
    // whose frames are all left out (a single frame under a shortest segment
    // of 6, say) made none of the segments listed.
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || editor_ == nullptr) {
        conditions_note_->clear();
        return;
    }
    const SegmentRules current = rules();
    const std::vector<ConditionCount> counts = condition_counts(*video->result, current, segments_);
    QStringList lines;
    bool left_out = false;
    for (std::size_t c = 0; c < counts.size(); ++c) {
        const QString condition = condition_label(current.conditions[c]);
        const ConditionCount& count = counts[c];
        if (count.holds == 0) {
            lines << app::tr("{condition}: no frame", {{"condition", condition}});
        } else if (count.in_segments == 0) {
            lines << ntr("{condition}: {count} frame, in no segment", "{condition}: {count} frames, in no segment",
                         count.holds, {{"condition", condition}});
        } else if (count.in_segments == count.holds) {
            lines << ntr("{condition}: {count} frame, in a segment", "{condition}: {count} frames, all in segments",
                         count.holds, {{"condition", condition}});
        } else {
            lines << ntr("{condition}: {count} frame, {in} of them in segments",
                         "{condition}: {count} frames, {in} of them in segments", count.holds,
                         {{"condition", condition}, {"in", count.in_segments}});
        }
        left_out = left_out || count.in_segments < count.holds;
    }
    QStringList why;
    if (left_out && current.min_frames > 1) {
        why << ntr("Runs shorter than {count} frame are left out.", "Runs shorter than {count} frames are left out.",
                   current.min_frames);
    }
    if (left_out && current.all && current.conditions.size() > 1) {
        why << app::tr("A frame is weak only where every condition holds on it.");
    }
    if (!why.isEmpty()) {
        lines << why.join(QLatin1Char(' '));
    }
    conditions_note_->setText(lines.join(QLatin1Char('\n')));
}

void WorstFramesTab::show_saved_state() {
    if (editor_ == nullptr) {
        save_button_->setEnabled(false);
        saved_note_->clear();
        return;
    }
    const SegmentRules shown = rules_to_save();
    std::vector<std::string> shown_texts = condition_texts(shown.conditions);
    std::vector<std::string> saved_texts = condition_texts(saved_.conditions);
    std::ranges::sort(shown_texts);
    std::ranges::sort(saved_texts);
    const bool differ = shown_texts != saved_texts || shown.all != saved_.all ||
                        shown.min_frames != saved_.min_frames || shown.merge_frames != saved_.merge_frames;
    save_button_->setEnabled(differ);
    const QString name = profile_combo_->currentText();
    saved_note_->setText(differ ? app::tr("Changed here only: Save to profile keeps these in {profile}.",
                                          {{"profile", name}})
                                : app::tr("As saved in profile {profile}.", {{"profile", name}}));
}

// ----------------------------------------------------------------- details

void WorstFramesTab::hover_at(double time) {
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr || chart_ == nullptr) {
        return;
    }
    const std::optional<ScoreSamples> samples = score_samples(video->result->metric_results, std::string(metric_->key));
    if (!samples || samples->time.empty()) {
        return;
    }
    // The frame nearest in time, unless a worst frame's dot is within a few
    // pixels: that one, which is what the dot is for.
    const std::span<const double> times = samples->time;
    auto i = static_cast<std::size_t>(std::lower_bound(times.begin(), times.end(), time) - times.begin());
    if (i == times.size() || (i > 0 && time - times[i - 1] <= times[i] - time)) {
        --i;
    }
    int frame = samples->frame[i];
    double at = times[i];
    const int pointer = chart_->pixel_for_time(time);
    int nearest = snap_pixels + 1;
    for (const WorstFrame& worst : worst_) {
        const int distance = std::abs(chart_->pixel_for_time(worst.time) - pointer);
        if (distance < nearest) {
            nearest = distance;
            frame = worst.frame;
            at = worst.time;
        }
    }
    chart_->set_cursor_time(at);
    if (frame != hover_frame_) {  // a move within one frame's pixels tells the same
        hover_frame_ = frame;
        show_details();
    }
}

void WorstFramesTab::hover_left() {
    hover_frame_.reset();
    if (chart_ != nullptr) {
        // Back on the worst frame selected, as selecting it put it; none for
        // a segment, whose shading shows it.
        std::optional<double> selected;
        if (const QList<QTableWidgetItem*> rows = worst_table_->selectedItems(); !rows.isEmpty()) {
            if (const std::optional<int> frame = worst_frame_of_row(rows.front()->row())) {
                const auto found = std::find_if(worst_.begin(), worst_.end(),
                                                [&](const WorstFrame& w) { return w.frame == *frame; });
                if (found != worst_.end()) {
                    selected = found->time;
                }
            }
        }
        chart_->set_cursor_time(selected);
    }
    show_details();
}

void WorstFramesTab::show_details() {
    if (details_ == nullptr) {
        return;
    }
    QString text;
    if (hover_frame_) {
        text = frame_details(*hover_frame_);
    }
    const bool hovering = !text.isEmpty();
    if (!hovering) {
        if (const QList<QTableWidgetItem*> rows = worst_table_->selectedItems(); !rows.isEmpty()) {
            if (const std::optional<int> frame = worst_frame_of_row(rows.front()->row())) {
                text = frame_details(*frame);
            }
        } else if (const QList<QTableWidgetItem*> picked = segments_table_->selectedItems(); !picked.isEmpty()) {
            const int row = picked.front()->row();
            if (row >= 0 && static_cast<std::size_t>(row) < segments_.size()) {
                text = segment_details(segments_[static_cast<std::size_t>(row)]);
            }
        }
    }
    details_->setText(text);
    details_->setVisible(!text.isEmpty());
    details_hint_->setText(text.isEmpty() ? app::tr("Point at the graph for a frame's scores, or select a row in the "
                                                    "lists. Double-click a row to open it in Video Compare.")
                                          : app::tr("Double-click the row to open it in Video Compare."));
    details_hint_->setVisible(!hovering);
}

QString WorstFramesTab::frame_details(int frame) const {
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr) {
        return {};
    }
    const MetricResultSet& results = video->result->metric_results;
    const std::optional<ScoreSamples> graphed = score_samples(results, std::string(metric_->key));
    if (!graphed) {
        return {};
    }
    const auto at = std::lower_bound(graphed->frame.begin(), graphed->frame.end(), frame);
    if (at == graphed->frame.end() || *at != frame) {
        return {};
    }
    const double time = graphed->time[static_cast<std::size_t>(at - graphed->frame.begin())];
    QStringList lines{app::tr("Frame {frame}", {{"frame", frame}}) + QStringLiteral("   ") + time_text(time)};
    const auto worst =
        std::find_if(worst_.begin(), worst_.end(), [frame](const WorstFrame& w) { return w.frame == frame; });
    if (worst != worst_.end()) {
        lines << app::tr("Worst frame {rank} of {count}", {{"rank", static_cast<int>(worst - worst_.begin()) + 1},
                                                            {"count", static_cast<int>(worst_.size())}});
    }
    // Its segment (they are in order and apart), and the conditions that
    // hold on it: one may hold on a frame no segment has (a run too short).
    const auto after = std::upper_bound(segments_.begin(), segments_.end(), frame,
                                        [](int f, const WeakSegment& segment) { return f < segment.first_frame; });
    if (after != segments_.begin() && std::prev(after)->last_frame >= frame) {
        lines << app::tr("In weak segment {first} to {last}",
                         {{"first", std::prev(after)->first_frame}, {"last", std::prev(after)->last_frame}});
    }
    QStringList held;
    if (editor_ != nullptr) {
        for (const WeakCondition& condition : editor_->rules().conditions) {
            if (const std::optional<ScoreSamples> samples = score_samples(results, condition.key)) {
                if (const std::optional<double> value = score_at(*samples, frame); value && condition.is_weak(*value)) {
                    held << condition_label(condition);
                }
            }
        }
    }
    lines << (held.isEmpty() ? app::tr("No condition holds")
                             : app::tr("Weak by {conditions}", {{"conditions", held.join(QStringLiteral(", "))}}));
    // Every metric's score, the one graphed first.
    std::vector<std::string> keys{std::string(metric_->key)};
    for (const std::string& key : keys_) {
        if (key != keys.front()) {
            keys.push_back(key);
        }
    }
    std::vector<std::pair<QString, QString>> scores;
    for (const std::string& key : keys) {
        const MetricDefinition& definition = metric_definition(key);
        const std::optional<ScoreSamples> samples = score_samples(results, key);
        scores.emplace_back(qt_text(definition.short_label),
                            qt_text(definition.format_value(samples ? score_at(*samples, frame) : std::nullopt)));
    }
    lines << QString() << aligned(scores);
    return lines.join(QLatin1Char('\n'));
}

QString WorstFramesTab::segment_details(const WeakSegment& segment) const {
    QStringList lines{
        app::tr("Segment {first} to {last}", {{"first", segment.first_frame}, {"last", segment.last_frame}}),
        app::tr("{start} to {end}", {{"start", time_text(segment.start_time)}, {"end", time_text(segment.end_time)}})};
    const SegmentRules current = rules();
    QStringList held;
    for (std::size_t c = 0; c < current.conditions.size() && c < segment.condition_frames.size(); ++c) {
        if (segment.condition_frames[c] > 0) {
            held << condition_label(current.conditions[c]);
        }
    }
    if (!held.isEmpty()) {
        lines << app::tr("Weak by {conditions}", {{"conditions", held.join(QStringLiteral(", "))}});
    }
    std::vector<std::pair<QString, QString>> rows{{app::tr("Frames"), QString::number(segment.frames())}};
    for (const SegmentMetric& metric : segment.metrics) {
        const MetricDefinition& definition = metric_definition(metric.key);
        const QString label = qt_text(definition.short_label);
        const SegmentEnd end = segment_end(current, definition);
        rows.emplace_back(end_heading(definition, end), qt_text(definition.format_value(segment_end_value(metric, end))));
        rows.emplace_back(app::tr("{metric} mean", {{"metric", label}}), qt_text(definition.format_value(metric.mean)));
    }
    lines << QString() << aligned(rows);
    return lines.join(QLatin1Char('\n'));
}

// -------------------------------------------------------------- the actions

std::optional<int> WorstFramesTab::worst_frame_of_row(int row) const {
    const QTableWidgetItem* item = worst_table_->item(row, 0);
    return item != nullptr ? std::optional(item->data(Qt::UserRole).toInt()) : std::nullopt;
}

WorstFrameList WorstFramesTab::shown_list() const {
    WorstFrameList list;
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr) {
        return list;
    }
    list.identity = video->identity;
    list.metric = qt_text(metric_->label);
    for (int row = 0; row < worst_table_->rowCount(); ++row) {
        const std::optional<int> frame = worst_frame_of_row(row);
        const QTableWidgetItem* score = worst_table_->item(row, 2);  // the metric listed
        if (frame && score != nullptr) {
            list.stops.push_back({*frame, score->text()});
        }
    }
    return list;
}

void WorstFramesTab::open_selected() {
    // The row selected in either list: a worst frame, with the list to step
    // through; a segment's first frame.
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr) {
        return;
    }
    if (const QList<QTableWidgetItem*> worst = worst_table_->selectedItems(); !worst.isEmpty()) {
        if (const std::optional<int> frame = worst_frame_of_row(worst.front()->row())) {
            emit open_requested(video->identity, *frame, shown_list());
        }
        return;
    }
    if (const QList<QTableWidgetItem*> segment = segments_table_->selectedItems(); !segment.isEmpty()) {
        const int row = segment.front()->row();
        if (row >= 0 && static_cast<std::size_t>(row) < segments_.size()) {
            emit open_requested(video->identity, segments_[static_cast<std::size_t>(row)].first_frame, {});
        }
    }
}

void WorstFramesTab::export_segments(ExportKind kind) {
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr) {
        return;
    }
    const SegmentRules current = rules();
    // Named after the metrics it covers.
    QString suffix = "." + qt_text(segments_file_suffix(current));
    QString filter;
    std::string text;
    switch (kind) {
    case ExportKind::json:
        suffix += QStringLiteral(".json");
        filter = app::tr("JSON file (*.json)");
        text = python_json_dumps(segments_file_json(*video->result, segments_, current)) + "\n";
        break;
    case ExportKind::csv:
        suffix += QStringLiteral(".csv");
        filter = app::tr("CSV file (*.csv)");
        text = segments_csv(segments_, current);
        break;
    }
    const QString path =
        dialogs_.save_file(app::tr("Export weak segments"), suggested_file_name(video->label, suffix), filter);
    write_export(path, text);
}

void WorstFramesTab::export_worst(ExportKind kind) {
    // The dips listed (the metric, the count and the range shown), each with
    // every metric's score.
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr || worst_.empty()) {
        return;
    }
    const std::string key(metric_->key);
    QString suffix = "." + qt_text(worst_frames_file_suffix(key));
    QString filter;
    std::string text;
    switch (kind) {
    case ExportKind::json: {
        std::vector<FrameInterval> ranges;
        if (range_) {
            ranges.push_back({range_->first, range_->second});
        }
        suffix += QStringLiteral(".json");
        filter = app::tr("JSON file (*.json)");
        text = python_json_dumps(worst_frames_json(video->result->metric_results, key, worst_,
                                                   static_cast<std::size_t>(count_), ranges)) +
               "\n";
        break;
    }
    case ExportKind::csv:
        suffix += QStringLiteral(".csv");
        filter = app::tr("CSV file (*.csv)");
        text = worst_frames_csv(video->result->metric_results, worst_);
        break;
    }
    const QString path =
        dialogs_.save_file(app::tr("Export worst frames"), suggested_file_name(video->label, suffix), filter);
    write_export(path, text);
}

void WorstFramesTab::export_graph() {
    // The graph as shown (the weak segments shaded, the worst frames
    // dotted), titled as Metric Graphs titles its own, with the video's name.
    const WeakFramesVideo* video = chosen_video();
    if (video == nullptr || metric_ == nullptr || chart_ == nullptr) {
        return;
    }
    QString title = app::tr("{label} vs time", {{"label", qt_text(metric_->label)}}) + QStringLiteral(" — ") +
                    video->label;
    if (range_) {
        title += app::tr(" — frames {first} to {last}", {{"first", range_->first}, {"last", range_->second}});
    }
    const QString suggested = suggested_file_name(video->label + "_" + qt_text(metric_->key), ".png");
    const QString path = dialogs_.save_file(app::tr("Export graph"), suggested, app::tr("PNG image (*.png)"));
    if (path.isEmpty()) {
        return;
    }
    const QPixmap image = export_image(*chart_, title, ExportTable{}, font(),
                                       export_colours ? export_colours() : theme::chart_colours(false));
    if (!image.save(path)) {
        dialogs_.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

void WorstFramesTab::write_export(const QString& path, const std::string& text) {
    if (path.isEmpty()) {
        return;  // Cancel
    }
    // A text file, as the command line writes them: CR LF on Windows.
    std::ofstream file(path_of(path));
    file << text;
    file.close();
    if (!file) {
        dialogs_.alert(app::tr("Export failed"), app::tr("Could not write {path}", {{"path", path}}));
    }
}

// ------------------------------------------------------------------ layout

void WorstFramesTab::fit_videos_height() { fit_table_height(videos_table_, 5); }

void WorstFramesTab::arrange_lists() {
    // Side by side where both tables fit without scrolling sideways (with
    // their cards' room), else one above the other: never a scroll bar
    // across the page, nor text made smaller.
    if (lists_layout_ == nullptr || work_ == nullptr) {
        return;
    }
    const int room = work_->width();
    const int padding = 60;  // two cards' margins and the gap between them
    const int needed = worst_width_ + segments_width_ + padding;
    const QBoxLayout::Direction direction = needed <= room ? QBoxLayout::LeftToRight : QBoxLayout::TopToBottom;
    if (lists_layout_->direction() != direction) {
        lists_layout_->setDirection(direction);
    }
}

void WorstFramesTab::fit_list(QTableWidget* table, int& width) {
    // Measuring every cell: 27 ms for a 1000-row list (the segments listed at
    // most) on a Core Ultra 9 285K, so not on every resize (#180).
    table->horizontalHeader()->resizeSections(QHeaderView::ResizeToContents);
    width = natural_width(table);
    arrange_lists();
}

bool WorstFramesTab::eventFilter(QObject* watched, QEvent* event) {
    if (watched == work_ && event->type() == QEvent::Resize) {
        arrange_lists();
    }
    return QWidget::eventFilter(watched, event);
}

}  // namespace vml::app
