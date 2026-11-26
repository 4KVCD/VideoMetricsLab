#include "saved_results_panel.h"

#include <QAction>
#include <QButtonGroup>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QEvent>
#include <QFont>
#include <QFrame>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QGroupBox>
#include <QIcon>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelection>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMouseEvent>
#include <QPixmap>
#include <QPushButton>
#include <QRadioButton>
#include <QScopedValueRollback>
#include <QScrollArea>
#include <QStandardItemModel>
#include <QResizeEvent>
#include <QStyle>
#include <QTableWidget>
#include <QToolButton>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <chrono>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>

#include "i18n.h"
#include "metric_order.h"
#include "results_export.h"
#include "saved_deletion.h"
#include "stay_open_menu.h"
#include "theme.h"
#include "widgets.h"
#include "window_helpers.h"
#include "vml/cvvdp.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/results_summary.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"
#include "vml/stats.h"

namespace vml::app {

namespace {

enum Column : int { col_test, col_reference, col_saved, col_metrics, col_size, column_count };

// The columns' keys in Settings, in their order.
constexpr const char* column_keys[column_count] = {"test", "reference", "saved", "metrics", "size"};

QString time_text(std::chrono::system_clock::time_point when) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count();
    return QLocale().toString(QDateTime::fromSecsSinceEpoch(seconds), QLocale::ShortFormat);
}

// A saved score's one number for the whole video, at its metric's
// precision: a frame metric's average by its own way of averaging, CVVDP's
// score; "—" for none.
QString score_text(const std::string& key, const std::variant<FrameMetricResult, SequenceMetricResult>& score) {
    const std::optional<double> value =
        std::visit([](const auto& metric) -> std::optional<double> {
            if constexpr (std::is_same_v<std::decay_t<decltype(metric)>, SequenceMetricResult>) {
                return metric.score;
            } else {
                return metric.aggregate();
            }
        }, score);
    if (const MetricDefinition* metric = find_metric(key); metric != nullptr) {
        return qt_text(metric->format_value(value));
    }
    return value ? QString::number(*value) : QStringLiteral("—");
}

QString metric_label(const std::string& key) {
    const MetricDefinition* metric = find_metric(key);
    return metric != nullptr ? qt_text(metric->short_label) : qt_text(key);
}

// A table cell that sorts by `order` rather than its text.
class SortItem : public QTableWidgetItem {
public:
    SortItem(const QString& text, double order) : QTableWidgetItem(text) { setData(Qt::UserRole, order); }
    bool operator<(const QTableWidgetItem& other) const override {
        return data(Qt::UserRole).toDouble() < other.data(Qt::UserRole).toDouble();
    }
};

}  // namespace

SavedResultsPanel::SavedResultsPanel(const Settings& settings, const Dialogs& dialogs, FileWrites& file_writes,
                                     QWidget* parent)
    : QWidget(parent),
      settings_(settings),
      dialogs_(dialogs),
      file_writes_(file_writes),
      reading_(new QFutureWatcher<std::vector<SavedComparison>>(this)) {
    connect(&file_writes_, &FileWrites::became_idle, this, [this] { on_writes_idle(); });
    details_reading_ = new QFutureWatcher<ReadScores>(this);
    connect(details_reading_, &QFutureWatcher<ReadScores>::finished, this, [this] {
        ReadScores read = details_reading_->result();
        if (read.complete && read.listing == listing_) {
            details_read_ = std::move(read);
        }
        show_details();  // if it is still the one selected; else read again
    });
    connect(reading_, &QFutureWatcher<std::vector<SavedComparison>>::finished, this,
            [this] { show_list(reading_->result()); });
    connect(&metric_order_signal(), &MetricOrderSignal::changed, this, [this] { reorder_metrics(); });
    auto* layout = new QVBoxLayout(this);
    auto* note = new QLabel(app::tr("Every comparison the app has saved scores of. Open them in Metric Graphs, or "
                                    "export their results, per-frame scores, graphs and weak segments, without the "
                                    "videos being added again."));
    note->setWordWrap(true);
    theme::style(note, "color: {muted};");
    layout->addWidget(note);
    auto* top = new QHBoxLayout;
    folder_label_ = new QLabel;
    top->addWidget(folder_label_, 1);
    filter_ = new QLineEdit;
    filter_->setPlaceholderText(app::tr("Filter by file name"));
    filter_->setClearButtonEnabled(true);
    filter_->setMinimumWidth(240);
    connect(filter_, &QLineEdit::textChanged, this, [this] { apply_filter(); });
    top->addWidget(filter_);
    auto* refresh_button = new QPushButton(app::tr("Refresh"));
    connect(refresh_button, &QPushButton::clicked, this, [this] { refresh(); });
    top->addWidget(refresh_button);
    // The list's save icon, at its top right (issue #224).
    save_button_ = new SaveButton;
    save_button_->why_not = [this] {
        return selected().empty()                                   ? app::tr("Select the comparisons to save.")
               : !quick_formats || quick_formats().empty() ? no_format_text()
                                                                    : QString();
    };
    save_button_->describe = [this] {
        const std::vector<const SavedComparison*> chosen = selected();
        const QString what = chosen.size() == 1
                                 ? app::tr("the selected comparison's results ({name})",
                                           {{"name", text_of(chosen.front()->distorted.filename())}})
                                 : app::tr("the selected comparisons' results ({count})",
                                           {{"count", static_cast<long long>(chosen.size())}});
        return quick_save_tooltip(what, quick_formats ? quick_formats() : std::vector<std::string>{});
    };
    save_button_->save = [this] {
        if (save_results) {
            save_results(selected_results());
        }
    };
    save_button_->save_as = [this](QMenu& menu) { add_quick_items(menu); };
    top->addWidget(save_button_);
    layout->addLayout(top);

    table_ = new QTableWidget(0, column_count);
    table_->setHorizontalHeaderLabels(
        {app::tr("Test video"), app::tr("Reference"), app::tr("Saved"), app::tr("Metrics"), app::tr("Size")});
    table_->verticalHeader()->setVisible(false);
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    // Every column resized by dragging, the last one shown filling the rest.
    // Until one is, they are fitted to the tab as it grows and shrinks: the
    // saved time, metrics and size as wide as their text, the test video and
    // reference sharing what is left.
    QHeaderView* header = table_->horizontalHeader();
    header->setSectionResizeMode(QHeaderView::Interactive);
    header->setStretchLastSection(true);
    widths_.assign(column_count, 0);
    const std::vector<int>& widths = settings.saved_results_column_widths;
    if (widths.size() == static_cast<std::size_t>(column_count) &&
        std::all_of(widths.begin(), widths.end(), [](int width) { return width > 0; })) {
        widths_ = widths;
        own_widths_ = true;
        for (int column = 0; column < column_count; ++column) {
            header->resizeSection(column, widths_[static_cast<std::size_t>(column)]);
        }
    }
    const std::vector<std::string>& hidden = settings.saved_results_hidden_columns;
    for (int column = col_reference; column < column_count; ++column) {
        table_->setColumnHidden(column, std::find(hidden.begin(), hidden.end(), column_keys[column]) != hidden.end());
    }
    // A quick second click comes as a double-click, which the heading takes
    // for no click (Windows sends no second press), and Qt's sorting does
    // not turn the arrow for it: turned here as Qt does for a click, the
    // same column the other way, another ascending. The table sorts on the
    // arrow's change. A double-click on a column's edge is
    // sectionHandleDoubleClicked, which fits the column.
    connect(header, &QHeaderView::sectionDoubleClicked, this, [header](int column) {
        const bool reverse =
            header->sortIndicatorSection() == column && header->sortIndicatorOrder() == Qt::AscendingOrder;
        header->setSortIndicator(column, reverse ? Qt::DescendingOrder : Qt::AscendingOrder);
    });
    header->viewport()->installEventFilter(this);
    table_->viewport()->installEventFilter(this);
    connect(header, &QHeaderView::sectionResized, this,
            [this] { resized_by_hand_ = resized_by_hand_ || (header_pressed_ && !fitting_); });
    header->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(header, &QWidget::customContextMenuRequested, this, [this, header](const QPoint& position) {
        auto* menu = new StayOpenMenu(this);
        menu->setAttribute(Qt::WA_DeleteOnClose);
        fill_column_menu(*menu);
        menu->popup(header->viewport()->mapToGlobal(position));
    });
    table_->setMinimumHeight(200);
    connect(table_, &QTableWidget::itemSelectionChanged, this, [this] {
        show_details();
        update_buttons();
    });
    connect(table_, &QTableWidget::cellDoubleClicked, this, [this] { emit open_requested(selected_results()); });
    table_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(table_, &QWidget::customContextMenuRequested, this, [this](const QPoint& position) {
        QMenu menu(this);
        fill_context_menu(menu);
        if (!menu.isEmpty()) {
            menu.exec(table_->viewport()->mapToGlobal(position));
        }
    });

    // The selected comparison's scores, in a column on the table's left
    // (as Worst Frames' Videos, Conditions and Details): in VML Modern a card
    // a score, tiles_a_row of them a row, in Classic a row each; scrolled
    // where they are more than it holds. Its ⓘ button opens every detail
    // of them (open_details).
    details_ = new QGroupBox(app::tr("Saved scores"));
    details_->setObjectName(QStringLiteral("savedScoresColumn"));
    details_->setFixedWidth(scores_column_width());
    auto* details_layout = new QVBoxLayout(details_);
    auto* details_head = new QHBoxLayout;
    details_name_ = new ElidedLabel;
    theme::style(details_name_, "color: {muted};");
    details_head->addWidget(details_name_, 1);
    details_button_ = new QToolButton;
    details_button_->setObjectName(QStringLiteral("savedScoresDetailsButton"));
    details_button_->setAutoRaise(true);
    details_button_->setToolTip(app::tr("Every saved score of this comparison: its statistics, when it was saved and "
                                        "how it was made"));
    details_button_->setAccessibleName(app::tr("Details"));
    draw_details_icon();
    PopupPanel::open_on_press(details_button_, [this] { open_details(); });
    details_head->addWidget(details_button_);
    details_layout->addLayout(details_head);
    auto* scroll = new QScrollArea;
    details_scroll_ = scroll;
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* grid_holder = new QWidget;
    // The cards at the top, the column's spare height under them.
    auto* holder_layout = new QVBoxLayout(grid_holder);
    holder_layout->setContentsMargins(0, 0, 0, 0);
    details_grid_ = new QGridLayout;
    holder_layout->addLayout(details_grid_);
    holder_layout->addStretch(1);
    details_grid_->setContentsMargins(0, 0, 0, 0);
    scroll->setWidget(grid_holder);
    scroll->viewport()->setAutoFillBackground(false);  // the box's own background
    grid_holder->setAutoFillBackground(false);
    details_layout->addWidget(scroll, 1);
    auto* middle = new QHBoxLayout;
    middle->addWidget(details_);
    middle->addWidget(table_, 1);
    layout->addLayout(middle, 1);

    auto* buttons = new QHBoxLayout;
    open_button_ = new QPushButton(app::tr("Open in Metric Graphs"));
    theme::set_role(open_button_, "primary");  // the tab's own action (VML Modern)
    connect(open_button_, &QPushButton::clicked, this, [this] { emit open_requested(selected_results()); });
    buttons->addWidget(open_button_);
    buttons->addStretch(1);
    delete_button_ = new QPushButton(app::tr("Delete..."));
    theme::set_role(delete_button_, "danger");
    connect(delete_button_, &QPushButton::clicked, this, [this] { delete_selected(); });
    buttons->addWidget(delete_button_);
    layout->addLayout(buttons);

    show_details();
    update_buttons();
}

SavedResultsPanel::~SavedResultsPanel() {
    if (listing_stopped_) {
        *listing_stopped_ = true;
    }
    reading_->waitForFinished();
    stop_details_read();
    details_reading_->waitForFinished();
}

void SavedResultsPanel::clear_layout(QLayout* layout) {
    // Every widget in it, those of layouts within it too (Classic's rows
    // are one each): deleting a layout leaves its widgets where they were,
    // drawn under the next comparison's.
    while (QLayoutItem* item = layout->takeAt(0)) {
        if (QLayout* inner = item->layout()) {
            clear_layout(inner);
        }
        delete item->widget();
        delete item;
    }
}

std::filesystem::path SavedResultsPanel::base() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

void SavedResultsPanel::refresh() {
    folder_label_->setText(app::tr("Reading {directory}...", {{"directory", text_of(base())}}));
    // A listing still running is stopped: this one takes its place.
    if (listing_stopped_) {
        *listing_stopped_ = true;
    }
    listing_stopped_ = std::make_shared<std::atomic<bool>>(false);
    reading_->setFuture(QtConcurrent::run([folder = base(), stopped = listing_stopped_] {
        return list_saved_comparisons(folder, [&stopped] { return stopped->load(); });
    }));
}

// The Metrics column's text: the comparison's metrics in the order shown.
QString SavedResultsPanel::metrics_text(const SavedComparison& saved) {
    QStringList metrics;
    for (const std::string& key : in_order_shown(saved.keys())) {
        metrics.append(metric_label(key));
    }
    return metrics.join(", ");
}

// The Metrics column and the scores shown in the metrics' new order.
void SavedResultsPanel::reorder_metrics() {
    const QSignalBlocker blocker(table_);
    const bool sorting = table_->isSortingEnabled();
    table_->setSortingEnabled(false);  // a row sorted elsewhere as its text changed would be passed over
    for (int row = 0; row < table_->rowCount(); ++row) {
        QTableWidgetItem* test = table_->item(row, col_test);
        QTableWidgetItem* metrics = table_->item(row, col_metrics);
        if (test != nullptr && metrics != nullptr) {
            metrics->setText(metrics_text(list_.at(test->data(Qt::UserRole).toUInt())));
        }
    }
    table_->setSortingEnabled(sorting);
    show_details();
}

void SavedResultsPanel::show_list(std::vector<SavedComparison> list) {
    list_ = std::move(list);
    details_read_ = {};  // its files may have changed since: read again
    emit listed();
    ++listing_;
    stop_details_read();
    folder_label_->setText(ntr("{count} saved comparison in {directory}", "{count} saved comparisons in {directory}",
                               static_cast<long long>(list_.size()), {{"directory", text_of(base())}}));
    const QSignalBlocker blocker(table_);
    table_->setSortingEnabled(false);
    table_->clearContents();
    table_->setRowCount(static_cast<int>(list_.size()));
    for (int row = 0; row < static_cast<int>(list_.size()); ++row) {
        const SavedComparison& saved = list_[static_cast<std::size_t>(row)];
        auto* test = new QTableWidgetItem(qt_text(saved.label));
        test->setToolTip(text_of(saved.distorted));
        test->setData(Qt::UserRole, row);  // its place in list_, whatever the sort
        table_->setItem(row, col_test, test);
        auto* reference = new QTableWidgetItem(name_of(saved.source));
        reference->setToolTip(text_of(saved.source));
        if (!saved.reference_found) {
            // Its scores are the only ones there are: said where it is seen.
            reference->setText(app::tr("{name} (not found)", {{"name", name_of(saved.source)}}));
            reference->setToolTip(app::tr("{path}\n\nNot found there: without it, these scores could not be "
                                          "calculated again.",
                                          {{"path", text_of(saved.source)}}));
            QFont font = reference->font();
            font.setItalic(true);
            reference->setFont(font);
        }
        table_->setItem(row, col_reference, reference);
        table_->setItem(row, col_saved,
                        new SortItem(time_text(saved.saved),
                                     static_cast<double>(saved.saved.time_since_epoch().count())));
        table_->setItem(row, col_metrics, new QTableWidgetItem(metrics_text(saved)));
        table_->setItem(row, col_size, new SortItem(saved_size_text(saved.bytes), static_cast<double>(saved.bytes)));
    }
    table_->setSortingEnabled(true);
    fit_columns();
    apply_filter();
    show_details();
    update_buttons();
}

bool SavedResultsPanel::eventFilter(QObject* watched, QEvent* event) {
    if (watched == table_->horizontalHeader()->viewport()) {
        // A column dragged wider or narrower, or double-clicked to fit its
        // text: the widths are the user's from now on, saved as the header
        // is let go (not at each step of the drag).
        // The left button only: a right-click's release goes to its menu.
        if ((event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonDblClick) &&
            static_cast<QMouseEvent*>(event)->button() == Qt::LeftButton) {
            header_pressed_ = true;
        } else if (event->type() == QEvent::MouseButtonRelease) {
            header_pressed_ = false;
            if (std::exchange(resized_by_hand_, false)) {
                own_widths_ = true;
                keep_widths();
                emit columns_changed();
            }
        }
    } else if (watched == table_->viewport() && event->type() == QEvent::Resize) {
        fit_columns();
    }
    return QWidget::eventFilter(watched, event);
}

void SavedResultsPanel::fit_columns() {
    if (own_widths_ || fitting_) {
        return;
    }
    const QScopedValueRollback<bool> fitting(fitting_, true);
    QHeaderView* header = table_->horizontalHeader();
    const QAbstractItemView* view = table_;  // its sizeHintForColumn is public, QTableView's protected
    int used = 0;
    for (const int column : {int(col_saved), int(col_metrics), int(col_size)}) {
        if (!table_->isColumnHidden(column)) {
            const int width = std::max(header->sectionSizeHint(column), view->sizeHintForColumn(column));
            header->resizeSection(column, width);
            used += width;
        }
    }
    const int names = table_->isColumnHidden(col_reference) ? 1 : 2;
    const int width = std::max((table_->viewport()->width() - used) / names, header->sectionSizeHint(col_reference));
    header->resizeSection(col_test, width);
    header->resizeSection(col_reference, width);
}

void SavedResultsPanel::keep_widths() {
    const QHeaderView* header = table_->horizontalHeader();
    for (int column = 0; column < column_count; ++column) {
        int& width = widths_[static_cast<std::size_t>(column)];
        if (!table_->isColumnHidden(column)) {
            width = header->sectionSize(column);
        } else if (width <= 0) {
            // Hidden since before the widths were the user's.
            width = std::max(header->sectionSizeHint(column), header->defaultSectionSize());
        }
    }
}

std::vector<int> SavedResultsPanel::column_widths() const {
    return own_widths_ ? widths_ : std::vector<int>{};
}

std::vector<std::string> SavedResultsPanel::hidden_columns() const {
    std::vector<std::string> hidden;
    for (int column = col_reference; column < column_count; ++column) {
        if (table_->isColumnHidden(column)) {
            hidden.emplace_back(column_keys[column]);
        }
    }
    return hidden;
}

void SavedResultsPanel::show_column(int column, bool shown) {
    if (column == col_test || table_->isColumnHidden(column) != shown) {
        return;
    }
    if (own_widths_) {
        keep_widths();  // the column's width, for when it is shown again
    }
    table_->setColumnHidden(column, !shown);
    if (shown && own_widths_) {
        table_->horizontalHeader()->resizeSection(column, widths_[static_cast<std::size_t>(column)]);
    }
    fit_columns();
    emit columns_changed();
}

void SavedResultsPanel::fill_column_menu(QMenu& menu) {
    menu.setToolTipsVisible(true);
    for (int column = 0; column < column_count; ++column) {
        QAction* action = menu.addAction(table_->horizontalHeaderItem(column)->text());
        action->setCheckable(true);
        action->setChecked(!table_->isColumnHidden(column));
        if (column == col_test) {
            // A row is its test video: without it, the rows could not be told apart.
            action->setEnabled(false);
            action->setToolTip(app::tr("The test video's column is always shown."));
            continue;
        }
        connect(action, &QAction::toggled, this, [this, column](bool checked) { show_column(column, checked); });
    }
    menu.addSeparator();
    QAction* reset = menu.addAction(app::tr("Reset column widths"));
    reset->setToolTip(app::tr("Fit the columns to the tab again, as they grow and shrink with it."));
    reset->setEnabled(own_widths_);
    connect(reset, &QAction::triggered, this, [this] {
        own_widths_ = false;
        widths_.assign(column_count, 0);
        fit_columns();
        emit columns_changed();
    });
}

void SavedResultsPanel::apply_filter() {
    const QString wanted = filter_->text().trimmed();
    for (int row = 0; row < table_->rowCount(); ++row) {
        // By the files' names and paths, not the cells' tooltips: a reference
        // not found has a sentence there ("Not found there: ...").
        const int place = table_->item(row, col_test)->data(Qt::UserRole).toInt();
        const SavedComparison& saved = list_[static_cast<std::size_t>(place)];
        bool shown = wanted.isEmpty();
        for (const std::string& text : saved.filter_texts()) {
            shown = shown || qt_text(text).contains(wanted, Qt::CaseInsensitive);
        }
        table_->setRowHidden(row, !shown);
    }
    // A row hidden is no longer selected: what is chosen is what is seen, and
    // the box and the buttons follow (itemSelectionChanged), as for a click.
    QItemSelection hidden;
    for (const QModelIndex& index : table_->selectionModel()->selectedRows()) {
        if (table_->isRowHidden(index.row())) {
            hidden.select(index, index.siblingAtColumn(table_->columnCount() - 1));
        }
    }
    if (!hidden.isEmpty()) {
        table_->selectionModel()->select(hidden, QItemSelectionModel::Deselect | QItemSelectionModel::Rows);
    }
}

std::vector<const SavedComparison*> SavedResultsPanel::selected() const {
    std::set<int> rows;
    for (const QModelIndex& index : table_->selectionModel()->selectedRows()) {
        if (!table_->isRowHidden(index.row())) {
            rows.insert(index.row());
        }
    }
    std::vector<const SavedComparison*> chosen;
    for (const int row : rows) {
        const int place = table_->item(row, col_test)->data(Qt::UserRole).toInt();
        chosen.push_back(&list_[static_cast<std::size_t>(place)]);
    }
    return chosen;
}

void SavedResultsPanel::show_details() {
    clear_layout(details_grid_);
    // Cards alike across the column, or Classic's rows (a metric's label,
    // then its score and how it was made).
    const bool tiles = showing_tiles();
    for (int column = 0; column < tiles_a_row; ++column) {
        details_grid_->setColumnStretch(column, tiles || column == 1 ? 1 : 0);
    }
    const std::vector<const SavedComparison*> chosen = selected();
    const bool one = chosen.size() == 1;
    details_name_->clear_text();
    details_button_->setEnabled(false);  // until its scores are read
    if (!one) {
        auto* none = new QLabel(chosen.empty() ? app::tr("Select a comparison to see its saved scores.")
                                               : app::tr("Select one comparison to see its saved scores."));
        none->setWordWrap(true);
        // As wide as the column, wrapped: aligned, it kept its one line's width.
        details_grid_->addWidget(none, 0, 0, 1, -1);
        return;
    }
    const SavedComparison& saved = *chosen.front();
    details_name_->set_text(qt_text(saved.label), qt_text(saved.label));
    if (details_read_.folder == saved.folder) {
        details_button_->setEnabled(true);
        fill_details(saved);
        return;
    }
    // Read on a thread of their own: each file is read whole, its per-frame
    // scores too, which for a long film takes a while. Issue #133: every
    // row clicked was read on the window's thread.
    details_grid_->addWidget(new QLabel(app::tr("Reading...")), 0, 0, 1, -1, Qt::AlignLeft | Qt::AlignTop);
    if (details_reading_->isRunning() && details_stop_ && !*details_stop_ && details_reading_folder_ == saved.folder) {
        return;  // being read already, for this listing (show_list stops the reads of an earlier one)
    }
    stop_details_read();
    details_stop_ = std::make_shared<std::atomic<bool>>(false);
    details_reading_folder_ = saved.folder;
    details_reading_->setFuture(QtConcurrent::run([saved, listing = listing_, stop = details_stop_] {
        ReadScores read{saved.folder, {}, {}, listing};
        for (const SavedScoreFile& file : saved.scores) {
            if (*stop) {
                return read;  // not complete: not used
            }
            read.scores[file.file] = read_saved_score(file.file);
            if (const std::optional<SavedScore>& score = read.scores[file.file]) {
                if (const MetricDefinition* metric = find_metric(file.key)) {
                    const std::vector<float>& values = std::visit(
                        [](const auto& result) -> const std::vector<float>& { return result.values; }, score->metric);
                    const std::vector<double> scores(values.begin(), values.end());
                    read.stats[file.file] =
                        compute_stats(scores, metric->thresholds, metric->aggregation, metric->direction);
                }
            }
        }
        read.complete = true;
        return read;
    }));
}

void SavedResultsPanel::stop_details_read() {
    if (details_stop_) {
        *details_stop_ = true;
    }
}

void SavedResultsPanel::wait_for_details() {
    details_reading_->waitForFinished();
    QCoreApplication::sendPostedEvents(details_reading_);
}

void SavedResultsPanel::fill_details(const SavedComparison& saved) { show_scores(saved); }

bool SavedResultsPanel::showing_tiles() const {
    return tiles_.value_or(theme::current_interface() == theme::Interface::modern);
}

void SavedResultsPanel::set_tiles(bool tiles) {
    tiles_ = tiles;
    show_details();
}

const SavedScore* SavedResultsPanel::read_score(const std::filesystem::path& file) const {
    const auto read = details_read_.scores.find(file);
    return read != details_read_.scores.end() && read->second ? &*read->second : nullptr;
}

QString SavedResultsPanel::unusable_note(const std::filesystem::path& file) const {
    const SavedScore* score = read_score(file);
    if (score == nullptr) {
        return app::tr("can't be read");
    }
    // As the Videos tab has it: not used, but scored again.
    return score->outdated ? app::tr("out of date: scored again when compared") : QString();
}

std::optional<std::filesystem::path> SavedResultsPanel::used_file(const SavedComparison& saved, const std::string& key,
                                                                  ScoreReads* reads) const {
    if (const auto picked = picked_.find(saved.folder); picked != picked_.end()) {
        if (const auto file = picked->second.find(key);
            file != picked->second.end() && unusable_note(file->second).isEmpty()) {
            return file->second;
        }
    }
    // By Settings > Saved Results' rule (pick_saved_score), among the files
    // read: those of the comparison shown, or else read now (only where the
    // metric has several, whose choice needs how each was made).
    std::vector<SavedScoreChoice> choices;
    std::optional<std::filesystem::path> first;
    for (const SavedScoreFile& file : saved.scores) {
        if (file.key != key) {
            continue;
        }
        first = first.value_or(file.file);
        choices.push_back({file.file, file.saved, std::nullopt});
    }
    if (choices.size() < 2) {
        return first;  // nothing to choose; usable or not, it's the one there is
    }
    const bool shown = details_read_.folder == saved.folder;
    for (SavedScoreChoice& choice : choices) {
        std::optional<SavedScore> read;
        const SavedScore* score = shown ? read_score(choice.file) : nullptr;
        if (!shown && reads != nullptr) {
            const auto [kept, first_read] = reads->try_emplace(choice.file);
            if (first_read) {
                kept->second = read_saved_score(choice.file);
            }
            score = kept->second ? &*kept->second : nullptr;
        } else if (!shown) {
            read = read_saved_score(choice.file);
            score = read ? &*read : nullptr;
        }
        if (score != nullptr && !score->outdated) {
            choice.provenance =
                std::visit([](const auto& metric) { return metric.provenance; }, score->metric);
        }
    }
    const SavedScoreRule rule{settings_.saved_score_pick, settings_.saved_score_device, settings_.saved_score_engine};
    std::function<bool(const SavedScoreChoice&)> videos;
    if (videos_would_use) {
        videos = [this, &saved, &key](const SavedScoreChoice& choice) { return videos_would_use(saved, key, choice); };
    }
    return pick_saved_score(choices, rule, videos).value_or(*first);
}

std::optional<std::filesystem::path> SavedResultsPanel::used_score(const std::string& key) const {
    const std::vector<const SavedComparison*> chosen = selected();
    if (chosen.size() != 1 || details_read_.folder != chosen.front()->folder) {
        return std::nullopt;
    }
    return used_file(*chosen.front(), key);
}

void SavedResultsPanel::pick_score(const std::string& key, const std::filesystem::path& file) {
    const std::vector<const SavedComparison*> chosen = selected();
    if (chosen.size() != 1) {
        return;
    }
    picked_[chosen.front()->folder][key] = file;
    show_details();
}

SavedResultsPanel::Made SavedResultsPanel::made_of(
    const std::variant<FrameMetricResult, SequenceMetricResult>& score) const {
    Made made;
    const MetricProvenance& provenance =
        std::visit([](const auto& metric) -> const MetricProvenance& { return metric.provenance; }, score);
    made.device = provenance.compute_backend == "gpu" ? QStringLiteral("GPU") : QStringLiteral("CPU");
    made.made = qt_text(provenance.implementation + " " + provenance.implementation_version).trimmed();
    if (const auto model = provenance.parameters.find("model"); model != provenance.parameters.end()) {
        // The model's own name for a bundled model, not its file's path
        // ("path=C:/.../vmaf_v1.0.16_3d0h.json").
        made.model = qt_text(model_shown(qt_text(model->second).remove(QLatin1Char('"')).toStdString()));
    }
    if (const auto* cvvdp = std::get_if<SequenceMetricResult>(&score)) {
        if (const std::optional<CvvdpSettings> display = cvvdp_display_of(*cvvdp)) {
            const std::optional<CvvdpPreset> preset = matching_cvvdp_preset(*display, settings_.cvvdp_presets_json, "");
            made.model = preset ? app::tr(qt_text(preset->name)) : app::tr("a display of its own");
        }
    }
    return made;
}

std::vector<SavedScoreRow> SavedResultsPanel::detail_rows(const SavedComparison& saved) const {
    std::vector<SavedScoreRow> rows;
    for (const std::string& key : in_order_shown(saved.keys())) {
        const std::optional<std::filesystem::path> used = used_file(saved, key);
        const int versions = static_cast<int>(std::count_if(
            saved.scores.begin(), saved.scores.end(), [&key](const SavedScoreFile& file) { return file.key == key; }));
        for (const SavedScoreFile& file : saved.scores) {
            if (file.key != key) {
                continue;
            }
            SavedScoreRow row;
            row.key = key;
            row.file = file.file;
            row.saved = time_text(file.saved);
            row.note = unusable_note(file.file);
            row.used = used && *used == file.file;
            row.versions = versions;
            row.score = QStringLiteral("\u2014");
            if (const SavedScore* score = read_score(file.file)) {
                row.score = score_text(key, score->metric);
                const Made made = made_of(score->metric);
                row.device = made.device;
                row.made = made.made;
                row.model = made.model;
                if (const auto stats = details_read_.stats.find(file.file); stats != details_read_.stats.end()) {
                    row.stats = stats->second;
                }
            }
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

SavedScoresPanel* SavedResultsPanel::open_details() {
    const std::vector<const SavedComparison*> chosen = selected();
    if (chosen.size() != 1 || details_read_.folder != chosen.front()->folder) {
        return nullptr;
    }
    const SavedComparison& saved = *chosen.front();
    auto* panel = new SavedScoresPanel(app::tr("Score summary of {name}", {{"name", qt_text(saved.label)}}),
                                       detail_rows(saved), settings_.saved_details_columns, this);
    panel->setAttribute(Qt::WA_DeleteOnClose);
    panel->set_anchor(details_button_);
    panel->picked = [this, folder = saved.folder](const std::string& key, const std::filesystem::path& file) {
        picked_[folder][key] = file;
        show_details();  // the cards follow; the panel stays open
    };
    connect(panel, &SavedScoresPanel::settings_requested, this, [this] { emit scores_settings_requested(); });
    // Its save icon: this score summary at one click, or as JSON or CSV
    // where asked (Save as...).
    SaveButton* save = panel->save_button();
    save->why_not = [this] {
        return !quick_formats || quick_formats().empty() ? no_format_text() : QString();
    };
    save->describe = [this] {
        return quick_save_tooltip(app::tr("this score summary"),
                                  quick_formats ? quick_formats() : std::vector<std::string>{});
    };
    save->save = [this] {
        if (save_score_summary) {
            save_score_summary(selected_results());
        }
    };
    save->save_as = [this](QMenu& menu) {
        for (const auto& [text, kind] : {std::pair(app::tr("Score summary as JSON..."), Export::score_summary_json),
                                         std::pair(app::tr("Score summary as CSV..."), Export::score_summary_csv)}) {
            connect(menu.addAction(text), &QAction::triggered, this,
                    [this, kind] { emit export_requested(kind, selected_results()); });
        }
    };
    panel->show_under(details_button_);
    return panel;
}

void SavedResultsPanel::show_scores(const SavedComparison& saved) {
    // A card (VML Modern) or a row (Classic) for each metric's score used,
    // or for each score saved; with several of a metric, a radio button on
    // each picks the one used.
    const bool tiles = showing_tiles();
    const bool each = settings_.saved_score_versions == "each";
    int placed = 0;
    for (const std::string& key : in_order_shown(saved.keys())) {
        const std::optional<std::filesystem::path> used = used_file(saved, key);
        std::vector<const SavedScoreFile*> files;
        for (const SavedScoreFile& file : saved.scores) {
            if (file.key == key && (each || (used && file.file == *used))) {
                files.push_back(&file);
            }
        }
        const int versions = static_cast<int>(std::count_if(
            saved.scores.begin(), saved.scores.end(), [&key](const SavedScoreFile& file) { return file.key == key; }));
        const bool picks = each && files.size() > 1;
        QButtonGroup* group = nullptr;  // the metric's radio buttons', owned by its first: gone with the cards
        for (const SavedScoreFile* file : files) {
            const SavedScore* score = read_score(file->file);
            const QString value = score != nullptr ? score_text(key, score->metric) : QStringLiteral("\u2014");
            const QString note = unusable_note(file->file);
            QString line = note.isEmpty() && score != nullptr ? made_of(score->metric).device : note;
            QString tip;
            QString badge;  // "2 saved", drawn to catch the eye
            if (each && files.size() > 1) {
                line += QStringLiteral(" \u00B7 ") + time_text(file->saved);
            } else if (versions > 1) {
                badge = ntr("{count} saved", "{count} saved", versions);
                tip = ntr("{count} score of this metric is saved: choose the one used in the details, opened by "
                          "the button above.",
                          "{count} scores of this metric are saved: choose the one used in the details, opened by "
                          "the button above.",
                          versions);
            }
            QWidget* name = nullptr;
            const QString label = metric_label(key) + (tiles ? QString() : QStringLiteral(":"));
            if (picks) {
                auto* pick = new QRadioButton(label);
                pick->setObjectName("pick_" + QString::fromStdString(key));
                pick->setChecked(used && *used == file->file);
                pick->setEnabled(note.isEmpty());
                pick->setToolTip(app::tr("Use this score"));
                group = group != nullptr ? group : new QButtonGroup(pick);
                group->addButton(pick);
                connect(pick, &QRadioButton::toggled, this,
                        [this, key, folder = saved.folder, path = file->file](bool on) {
                            if (on) {
                                // Shown again once this button's own signal
                                // is done: showing them deletes it.
                                picked_[folder][key] = path;
                                QMetaObject::invokeMethod(this, [this] { show_details(); }, Qt::QueuedConnection);
                            }
                        });
                name = pick;
            } else {
                name = new QLabel(label);
            }
            if (tiles) {
                details_grid_->addWidget(make_tile(key, name, value, line, badge, tip), placed / tiles_a_row,
                                         placed % tiles_a_row);
            } else {
                auto* shown = new QLabel(value + QStringLiteral(" \u00B7 ") + line);
                details_grid_->addWidget(name, placed, 0);
                auto* value_row = new QHBoxLayout;
                value_row->addWidget(shown);
                if (!badge.isEmpty()) {
                    value_row->addWidget(saved_badge(key, badge, tip));
                }
                value_row->addStretch(1);
                details_grid_->addLayout(value_row, placed, 1);
            }
            ++placed;
        }
    }
}

QLabel* SavedResultsPanel::saved_badge(const std::string& key, const QString& text, const QString& tip) {
    // How many scores of a metric are saved, bold in the link's colour: a
    // choice to make, where a muted line went unseen (the user, 2026-11-25).
    auto* badge = new QLabel(text);
    badge->setObjectName("saved_" + QString::fromStdString(key));
    badge->setToolTip(tip);
    theme::style(badge, "color: {link}; font-weight: bold;");
    return badge;
}

QWidget* SavedResultsPanel::make_tile(const std::string& key, QWidget* name, const QString& score,
                                      const QString& line, const QString& badge, const QString& badge_tip) const {
    // A frame of its own (a card's tile in VML Modern); its name small and
    // muted, its score large, as Metric Graphs' table shows it, then CPU or
    // GPU and how many are saved.
    auto* tile = new QFrame;
    tile->setObjectName("tile_" + QString::fromStdString(key));
    tile->setFrameShape(QFrame::StyledPanel);
    theme::set_role(tile, "tile");
    tile->setFixedHeight(tile_height());
    auto* tile_layout = new QVBoxLayout(tile);
    tile_layout->setContentsMargins(8, 5, 8, 5);
    tile_layout->setSpacing(1);
    if (qobject_cast<QLabel*>(name) != nullptr) {
        theme::style(name, "color: {muted};");
    }
    auto* name_row = new QHBoxLayout;
    name_row->setSpacing(4);
    name_row->addWidget(name);
    name_row->addStretch(1);
    if (!badge.isEmpty()) {
        name_row->addWidget(saved_badge(key, badge, badge_tip));
    }
    tile_layout->addLayout(name_row);
    auto* score_label = new QLabel(score);
    score_label->setObjectName("score_" + QString::fromStdString(key));
    QFont large = score_label->font();
    large.setPointSizeF(large.pointSizeF() * 1.4);
    large.setWeight(QFont::DemiBold);
    score_label->setFont(large);
    tile_layout->addWidget(score_label);
    auto* made = new ElidedLabel;
    made->setObjectName("made_" + QString::fromStdString(key));
    made->set_text(line);
    theme::style(made, "color: {muted};");
    tile_layout->addWidget(made);
    tile_layout->addStretch(1);
    return tile;
}

int SavedResultsPanel::tile_height() const {
    // Its name (a radio button's height, where it picks), its score large,
    // and one line, with the frame's margins.
    const QFontMetrics text = fontMetrics();
    QFont large = font();
    large.setPointSizeF(large.pointSizeF() * 1.4);
    const int name = std::max(text.height(), style()->pixelMetric(QStyle::PM_ExclusiveIndicatorHeight) + 2);
    return name + QFontMetrics(large).height() + text.height() + 10 + 2 * 1 + 4;
}

int SavedResultsPanel::scores_column_width() const {
    // Three cards a row, each as wide as the widest metric's name or a long
    // score large; Classic's rows fit in the same.
    const QFontMetrics text = fontMetrics();
    QFont large = font();
    large.setPointSizeF(large.pointSizeF() * 1.4);
    large.setWeight(QFont::DemiBold);
    int card = QFontMetrics(large).horizontalAdvance(QStringLiteral("100.0000"));
    const int radio = style()->pixelMetric(QStyle::PM_ExclusiveIndicatorWidth) +
                      style()->pixelMetric(QStyle::PM_RadioButtonLabelSpacing);  // a name after its radio button
    for (const MetricDefinition& metric : metrics()) {
        card = std::max(card, text.horizontalAdvance(qt_text(metric.short_label)) + radio);
    }
    card += 16 + 2;
    const int spacing = std::max(style()->pixelMetric(QStyle::PM_LayoutHorizontalSpacing), 6);
    return tiles_a_row * card + (tiles_a_row - 1) * spacing + 2 * 12 + style()->pixelMetric(QStyle::PM_ScrollBarExtent);
}

void SavedResultsPanel::scores_settings_changed() { show_details(); }

void SavedResultsPanel::draw_details_icon() {
    // The ⓘ (info_icon, as Worst Frames' graph legend has it) in the text's colour.
    details_button_->setIcon(info_icon(palette().color(QPalette::WindowText), devicePixelRatioF()));
}

void SavedResultsPanel::changeEvent(QEvent* event) {
    QWidget::changeEvent(event);
    if ((event->type() == QEvent::PaletteChange || event->type() == QEvent::StyleChange) &&
        details_button_ != nullptr) {
        draw_details_icon();  // the theme's text colour
    }
}

std::vector<LoadedSavedResult> SavedResultsPanel::selected_results() const {
    std::vector<LoadedSavedResult> results;
    for (const SavedComparison* saved : selected()) {
        const auto picked = picked_.find(saved->folder);
        // The scores the details or the choice among several have read, where
        // they have: each file read once (on the window's thread, as before).
        ScoreReads reads;
        const auto read = [this, saved, &reads](const std::filesystem::path& file) -> std::optional<SavedScore> {
            if (details_read_.folder == saved->folder) {
                if (const auto found = details_read_.scores.find(file); found != details_read_.scores.end()) {
                    return found->second;
                }
            }
            if (const auto kept = reads.find(file); kept != reads.end()) {
                return std::move(kept->second);  // read once (readable or not), and needed no more
            }
            return read_saved_score(file);
        };
        // Each metric's score by the rule (used_file), the one picked first.
        std::map<std::string, std::filesystem::path> chosen =
            picked != picked_.end() ? picked->second : std::map<std::string, std::filesystem::path>{};
        for (const std::string& key : saved->keys()) {
            if (!chosen.contains(key)) {
                if (const std::optional<std::filesystem::path> used = used_file(*saved, key, &reads)) {
                    chosen[key] = *used;
                }
            }
        }
        const std::optional<std::pair<ComparisonResult, std::string>> loaded =
            load_saved_comparison(*saved, chosen, read);
        if (loaded) {
            // When each score used was saved: the one picked, else its newest.
            std::map<std::string, std::chrono::system_clock::time_point> times;
            for (const std::string& key : saved->keys()) {
                const std::optional<std::filesystem::path> used = used_file(*saved, key);
                for (const SavedScoreFile& file : saved->scores) {
                    if (file.key == key && used && file.file == *used) {
                        times[key] = file.saved;
                    }
                }
            }
            results.push_back({std::make_shared<const ComparisonResult>(loaded->first), qt_text(loaded->second),
                               saved->folder, std::move(times)});
        }
    }
    return results;
}

std::vector<std::pair<QString, QString>> SavedResultsPanel::segment_rule_choices() const {
    std::vector<std::pair<QString, QString>> choices;
    const SegmentProfile in_use = segment_profile_in_use(settings_);
    choices.emplace_back(QStringLiteral("in-use"),
                         app::tr("Profile in use ({name})",
                                 {{"name", in_use.name.empty() ? app::tr("Default") : qt_text(in_use.name)}}));
    for (const SegmentProfile& profile : segment_profiles(settings_)) {
        choices.emplace_back(QStringLiteral("profile:") + qt_text(profile.name),
                             app::tr("Profile: {name}",
                                     {{"name", profile.name.empty() ? app::tr("Default") : qt_text(profile.name)}}));
    }
    for (const SegmentPreset& preset : segment_presets()) {
        choices.emplace_back(QStringLiteral("preset:") + qt_text(std::string(preset.key)),
                             app::tr("Preset: {name}", {{"name", app::tr(preset.name)}}));
    }
    return choices;
}

void SavedResultsPanel::pick_segment_rules(const QString& choice) {
    rules_choice_ = choice;
}

SegmentRules SavedResultsPanel::segment_rules() const {
    // The rules picked, while they are still there; else the profile in use.
    const std::string choice = rules_choice_.toStdString();
    if (choice.starts_with("preset:")) {
        if (const SegmentPreset* preset = segment_preset(choice.substr(7))) {
            return preset->rules;
        }
    } else if (choice.starts_with("profile:")) {
        if (const std::optional<SegmentProfile> profile = segment_profile_named(settings_, choice.substr(8))) {
            return profile->rules;
        }
    }
    return segment_profile_in_use(settings_).rules;
}

namespace {

// A path as locked_by_run compares them: made normal by its text alone,
// with the system's separators, in one case on Windows (as same_path
// compares). No disk is asked: a selection of many comparisons, some on a
// share, is checked at each click.
QString run_key(const std::filesystem::path& path) {
    QString text = text_of(path.lexically_normal().make_preferred());
#ifdef _WIN32
    text = text.toCaseFolded();
#endif
    return text;
}

// A pair as locked_by_run compares them: its reference's key, then its
// test video's.
QString pair_key(const std::filesystem::path& reference, const std::filesystem::path& test) {
    return run_key(reference) + QLatin1Char('\n') + run_key(test);
}

}  // namespace

void SavedResultsPanel::set_run_active(
    bool active, const std::vector<std::pair<std::filesystem::path, std::filesystem::path>>& pairs) {
    run_active_ = active;
    run_pairs_.clear();
    run_folders_.clear();
    run_folders_known_ = false;
    if (active) {
        for (const auto& [reference, test] : pairs) {
            run_pairs_.insert(pair_key(reference, test));
        }
    }
    update_buttons();
}

void SavedResultsPanel::add_run_folders(const std::vector<std::filesystem::path>& folders, bool all) {
    if (!run_active_) {
        return;  // a run's, told after it ended
    }
    for (const std::filesystem::path& folder : folders) {
        run_folders_.insert(run_key(folder));
    }
    run_folders_known_ = run_folders_known_ || all;
    update_buttons();
}

bool SavedResultsPanel::locked_by_run(const SavedComparison& saved) const {
    return run_active_ && (!run_folders_known_ || run_pairs_.contains(pair_key(saved.source, saved.distorted)) ||
                           run_folders_.contains(run_key(saved.folder)));
}

void SavedResultsPanel::update_buttons() {
    const std::vector<const SavedComparison*> chosen = selected();
    open_button_->setEnabled(!chosen.empty());
    save_button_->refresh();
    // Locked where any selected comparison is one the run uses (the others
    // can be deleted while it runs), and while a Delete is queued.
    const bool locked = std::any_of(chosen.begin(), chosen.end(),
                                    [this](const SavedComparison* saved) { return locked_by_run(*saved); });
    delete_button_->setEnabled(!chosen.empty() && !locked && !deleting_);
    delete_button_->setToolTip(locked ? app::tr("Locked while metrics are calculated: the run saves its "
                                                "scores there.")
                                      : QString());
}

void SavedResultsPanel::add_quick_grabs(QMenu& menu, bool enabled, const std::function<void(Export)>& chosen,
                                        const std::map<Export, QString>& refused) {
    menu.setToolTipsVisible(true);
    const auto add = [&menu, enabled, &chosen, &refused](const QString& text, Export kind) {
        QAction* action = menu.addAction(text);
        const auto why = refused.find(kind);
        action->setEnabled(enabled && why == refused.end());
        if (enabled && why != refused.end()) {
            action->setToolTip(why->second);
        }
        QObject::connect(action, &QAction::triggered, &menu, [chosen, kind] { chosen(kind); });
        return action;
    };
    add(app::tr("Results as JSON..."), Export::results_json);
    add(app::tr("Results as CSV..."), Export::results_csv);
    add(app::tr("Per-frame scores as JSON..."), Export::frames_json);
    add(app::tr("Per-frame scores as CSV..."), Export::frames_csv);
    add(app::tr("Score summary as JSON..."), Export::score_summary_json);
    add(app::tr("Score summary as CSV..."), Export::score_summary_csv);
    QAction* graphs = add(app::tr("Graphs as ZIP..."), Export::graphs);
    if (!enabled || !refused.contains(Export::graphs)) {  // not refused: what it holds
        graphs->setToolTip(app::tr("Each video's graph of every metric, and with several of one reference every "
                                   "metric with all of them; the bitrate graphs where the videos are still there."));
    }
    add(app::tr("Weak segments as JSON or CSV..."), Export::segments);
    menu.addSeparator();
    add(app::tr("Run file (.metrics.json)..."), Export::run_file);
}

void SavedResultsPanel::add_quick_items(QMenu& menu) {
    // The quick grabs of the selected comparisons, each with the Save
    // dialog; those that cannot be made greyed, saying why.
    std::map<Export, QString> refused;
    if (several_references()) {
        refused[Export::graphs] = one_reference_text();
        refused[Export::segments] = one_reference_text();
    } else if (segment_rules().conditions.empty()) {
        refused[Export::segments] = no_conditions_text();
    }
    add_quick_grabs(menu, !selected().empty(),
                    [this](Export kind) { emit export_requested(kind, selected_results()); }, refused);
}

void SavedResultsPanel::add_export_items(QMenu& menu) {
    // The selection's archive first, the one archive item outside the tab
    // bar's Export panel (whose picker sets the weak segment rules); then
    // the quick grabs, for a one-off.
    const QAction* contents = fill_menu(menu);
    if (contents == nullptr) {
        return;
    }
    connect(contents, &QAction::triggered, this,
            [this] { emit export_requested(Export::contents, selected_results()); });
    menu.addSeparator();
    add_quick_items(menu);
}

QString SavedResultsPanel::segment_rule_choice() const {
    for (const auto& [choice, text] : segment_rule_choices()) {
        if (choice == rules_choice_) {
            return choice;
        }
    }
    return QStringLiteral("in-use");  // the one picked is gone (a profile deleted)
}

QAction* SavedResultsPanel::fill_menu(QMenu& menu) const {
    const std::size_t count = selected().size();
    if (count == 0) {
        return nullptr;
    }
    const std::vector<std::string> items = exported_items(settings_);
    QAction* contents =
        menu.addAction(export_action_text(count == 1 ? ExportScope::this_video : ExportScope::selected_videos, items));
    const bool needs_one_reference = std::find(items.begin(), items.end(), "graphs") != items.end() ||
                                     std::find(items.begin(), items.end(), "weak_segments") != items.end();
    const bool refused = needs_one_reference && several_references();
    contents->setEnabled(!export_contents(items).isEmpty() && !refused);
    if (refused) {
        menu.setToolTipsVisible(true);
        contents->setToolTip(one_reference_text());
    }
    return contents;
}

bool SavedResultsPanel::several_references() const {
    const std::vector<const SavedComparison*> chosen = selected();
    return std::any_of(chosen.begin(), chosen.end(), [&chosen](const SavedComparison* saved) {
        return !same_path(saved->source, chosen.front()->source);
    });
}

void SavedResultsPanel::fill_context_menu(QMenu& menu) {
    // Open in Metric Graphs, as the button under the box and a double-click,
    // so that it is beside the rows; then the export; then Delete..., as the
    // button under the box: the same question, locked when it is, and why.
    if (selected().empty()) {
        return;
    }
    connect(menu.addAction(app::tr("Open in Metric Graphs")), &QAction::triggered, this,
            [this] { emit open_requested(selected_results()); });
    menu.addSeparator();
    add_export_items(menu);
    menu.addSeparator();
    QAction* remove = menu.addAction(app::tr("Delete..."));
    remove->setEnabled(delete_button_->isEnabled());
    remove->setToolTip(delete_button_->toolTip());
    menu.setToolTipsVisible(true);
    connect(remove, &QAction::triggered, this, [this] { delete_selected(); });
}

void SavedResultsPanel::delete_selected() {
    // Copies: the question waits in an event loop of its own, in which a
    // listing that finishes replaces list_ (show_list).
    std::vector<SavedComparison> chosen;
    for (const SavedComparison* saved : selected()) {
        chosen.push_back(*saved);
    }
    // Asked again after the question: a run may have started meanwhile.
    const auto locked = [this, &chosen] {
        return deleting_ || std::any_of(chosen.begin(), chosen.end(),
                                        [this](const SavedComparison& saved) { return locked_by_run(saved); });
    };
    if (chosen.empty() || locked()) {
        return;
    }
    // The question every deletion of saved results asks (saved_deletion.h),
    // Settings' Delete... too: what goes, and what can't be scored again.
    if (dialogs_.message(saved_deletion_message(saved_deletion(chosen, base()))).button != 0 || locked()) {
        return;
    }
    // Through the window's queue of writes, after what is queued there, as
    // Settings' Delete...: a run's saves still queued would otherwise bring a
    // folder deleted back, or leave one without its context.json, which the
    // tab can neither list nor delete (issue #133).
    deleting_ = std::make_shared<Deleting>();
    update_buttons();
    file_writes_.submit(app::tr("delete saved results"),
                        [folder = base(), chosen = std::move(chosen), deleting = deleting_] {
                            for (const SavedComparison& saved : chosen) {
                                if (!delete_saved_comparison(folder, saved)) {
                                    ++deleting->failed;
                                }
                            }
                            deleting->done = true;
                        });
}

void SavedResultsPanel::on_writes_idle() {
    // Once the Delete has run: a write queued before it, finished as it was
    // queued, says the queue is idle too, too early.
    if (!deleting_ || !deleting_->done) {
        return;
    }
    const int failed = deleting_->failed;
    deleting_.reset();
    update_buttons();
    if (failed > 0) {
        dialogs_.alert(app::tr("Not all deleted"),
                       ntr("{count} comparison's saved scores could not be deleted: a file of theirs may be open.",
                           "{count} comparisons' saved scores could not be deleted: a file of theirs may be open.",
                           failed));
    }
    emit deleted();
    refresh();
}

}  // namespace vml::app
