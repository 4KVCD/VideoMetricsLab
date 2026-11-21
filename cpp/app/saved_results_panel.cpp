#include "saved_results_panel.h"

#include <QAction>
#include <QActionGroup>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QEvent>
#include <QFont>
#include <QFrame>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QScopedValueRollback>
#include <QScrollArea>
#include <QStandardItemModel>
#include <QStyle>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <chrono>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>

#include "i18n.h"
#include "results_export.h"
#include "stay_open_menu.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/cvvdp.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/results_summary.h"
#include "vml/segment_presets.h"
#include "vml/segment_profiles.h"

namespace vml::app {

namespace {

enum Column : int { col_test, col_reference, col_saved, col_metrics, col_size, column_count };

// The columns' keys in Settings, in their order.
constexpr const char* column_keys[column_count] = {"test", "reference", "saved", "metrics", "size"};

QString time_text(std::chrono::system_clock::time_point when) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count();
    return QLocale().toString(QDateTime::fromSecsSinceEpoch(seconds), QLocale::ShortFormat);
}

QString size_text(std::uintmax_t bytes) {
    // A comparison's scores are often under a megabyte.
    return bytes < 1048576 ? app::tr("{size:.0f} KB", {{"size", std::max(1.0, static_cast<double>(bytes) / 1024.0)}})
                           : app::tr("{size:.1f} MB", {{"size", static_cast<double>(bytes) / 1048576.0}});
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
    table_->setMinimumHeight(260);
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
    layout->addWidget(table_, 1);

    // The selected comparison's scores, two metrics a row, in a box as tall as
    // every metric needs whatever is selected: were it to grow when a row is
    // clicked, the table would shrink under the pointer and scroll to keep
    // the row in view.
    details_ = new QGroupBox(app::tr("Saved scores"));
    auto* details_layout = new QVBoxLayout(details_);
    auto* scroll = new QScrollArea;
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    auto* grid_holder = new QWidget;
    // The rows at the top, the box's spare height under them: a comparison
    // of few metrics would otherwise have its rows spread down the box.
    auto* holder_layout = new QVBoxLayout(grid_holder);
    holder_layout->setContentsMargins(0, 0, 0, 0);
    details_grid_ = new QGridLayout;
    holder_layout->addLayout(details_grid_);
    holder_layout->addStretch(1);
    details_grid_->setContentsMargins(0, 0, 0, 0);
    details_grid_->setColumnMinimumWidth(2, 24);  // between the two metrics of a row
    details_grid_->setColumnStretch(5, 1);
    scroll->setWidget(grid_holder);
    scroll->viewport()->setAutoFillBackground(false);  // the box's own background
    grid_holder->setAutoFillBackground(false);
    details_layout->addWidget(scroll);
    const int rows = (static_cast<int>(metrics().size()) + 1) / 2;
    const int row_height = QComboBox().sizeHint().height();
    const int spacing = details_grid_->verticalSpacing() >= 0 ? details_grid_->verticalSpacing()
                                                               : style()->pixelMetric(QStyle::PM_LayoutVerticalSpacing);
    scroll->setFixedHeight(rows * row_height + (rows - 1) * std::max(spacing, 0) + 2);
    details_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    layout->addWidget(details_);

    auto* buttons = new QHBoxLayout;
    open_button_ = new QPushButton(app::tr("Open in Metric Graphs"));
    connect(open_button_, &QPushButton::clicked, this, [this] { emit open_requested(selected_results()); });
    buttons->addWidget(open_button_);
    // One Export button: its menu is made as it opens (fill_export_menu).
    export_button_ = new QPushButton(app::tr("Export"));
    export_menu_ = new QMenu(export_button_);
    export_menu_->setToolTipsVisible(true);
    connect(export_menu_, &QMenu::aboutToShow, this, [this] { fill_export_menu(*export_menu_); });
    export_button_->setMenu(export_menu_);
    buttons->addWidget(export_button_);
    buttons->addStretch(1);
    delete_button_ = new QPushButton(app::tr("Delete..."));
    connect(delete_button_, &QPushButton::clicked, this, [this] { delete_selected(); });
    buttons->addWidget(delete_button_);
    layout->addLayout(buttons);

    show_details();
    update_buttons();
}

SavedResultsPanel::~SavedResultsPanel() {
    reading_->waitForFinished();
    stop_details_read();
    details_reading_->waitForFinished();
}

std::filesystem::path SavedResultsPanel::base() const {
    return settings_.cache_dir_path().value_or(default_cache_dir());
}

void SavedResultsPanel::refresh() {
    folder_label_->setText(app::tr("Reading {directory}...", {{"directory", text_of(base())}}));
    reading_->setFuture(QtConcurrent::run([folder = base()] { return list_saved_comparisons(folder); }));
}

void SavedResultsPanel::show_list(std::vector<SavedComparison> list) {
    list_ = std::move(list);
    details_read_ = {};  // its files may have changed since: read again
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
        QStringList metrics;
        for (const std::string& key : saved.keys()) {
            metrics.append(metric_label(key));
        }
        table_->setItem(row, col_metrics, new QTableWidgetItem(metrics.join(", ")));
        table_->setItem(row, col_size, new SortItem(size_text(saved.bytes), static_cast<double>(saved.bytes)));
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
        const bool shown = wanted.isEmpty() ||
                           table_->item(row, col_test)->text().contains(wanted, Qt::CaseInsensitive) ||
                           table_->item(row, col_test)->toolTip().contains(wanted, Qt::CaseInsensitive) ||
                           table_->item(row, col_reference)->toolTip().contains(wanted, Qt::CaseInsensitive);
        table_->setRowHidden(row, !shown);
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
    while (QLayoutItem* item = details_grid_->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    score_combos_.clear();
    const std::vector<const SavedComparison*> chosen = selected();
    if (chosen.size() != 1) {
        details_->setTitle(app::tr("Saved scores"));
        details_grid_->addWidget(new QLabel(chosen.empty() ? app::tr("Select a comparison to see its saved scores.")
                                                           : app::tr("Select one comparison to see its saved scores.")),
                                 0, 0, 1, -1, Qt::AlignLeft | Qt::AlignTop);
        return;
    }
    const SavedComparison& saved = *chosen.front();
    details_->setTitle(app::tr("Saved scores of {name}", {{"name", qt_text(saved.label)}}));
    if (details_read_.folder == saved.folder) {
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
        ReadScores read{saved.folder, {}, listing};
        for (const SavedScoreFile& file : saved.scores) {
            if (*stop) {
                return read;  // not complete: not used
            }
            read.scores[file.file] = read_saved_score(file.file);
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

void SavedResultsPanel::fill_details(const SavedComparison& saved) {
    // Two metrics a row: a metric's label, then its score or the list to pick
    // one from.
    int placed = 0;
    const auto add_row = [this, &placed](const QString& label, QWidget* field) {
        const int row = placed / 2;
        const int column = (placed % 2) * 3;
        details_grid_->addWidget(new QLabel(label), row, column);
        details_grid_->addWidget(field, row, column + 1, Qt::AlignLeft);
        ++placed;
    };
    for (const std::string& key : saved.keys()) {
        auto* combo = new QComboBox;
        int first_usable = -1;  // the score used unless another is picked
        for (const SavedScoreFile& file : saved.scores) {
            if (file.key != key) {
                continue;
            }
            bool usable = true;
            // The score, as Metric Graphs' table shows it, then how it was
            // made, as its file records it.
            QStringList made{time_text(file.saved)};
            const auto read = details_read_.scores.find(file.file);
            const SavedScore* score =
                read != details_read_.scores.end() && read->second ? &*read->second : nullptr;
            if (score != nullptr) {
                const MetricProvenance& provenance = std::visit(
                    [](const auto& metric) -> const MetricProvenance& { return metric.provenance; }, score->metric);
                made.prepend(score_text(key, score->metric));
                made.append(provenance.compute_backend == "gpu" ? QStringLiteral("GPU") : QStringLiteral("CPU"));
                made.append(qt_text(provenance.implementation + " " + provenance.implementation_version).trimmed());
                if (const auto model = provenance.parameters.find("model"); model != provenance.parameters.end()) {
                    // The model's own name for a bundled model, not its file's
                    // path ("path=C:/.../vmaf_v1.0.16_3d0h.json").
                    made.append(qt_text(model_shown(qt_text(model->second).remove(QLatin1Char('"')).toStdString())));
                }
                if (const auto* cvvdp = std::get_if<SequenceMetricResult>(&score->metric)) {
                    if (const std::optional<CvvdpSettings> display = cvvdp_display_of(*cvvdp)) {
                        const std::optional<CvvdpPreset> preset =
                            matching_cvvdp_preset(*display, settings_.cvvdp_presets_json, "");
                        made.append(preset ? app::tr(qt_text(preset->name)) : app::tr("a display of its own"));
                    }
                }
                if (score->outdated) {
                    // As the Videos tab has it: not used, but scored again.
                    made.append(app::tr("out of date: scored again when compared"));
                    usable = false;
                }
            } else {
                made.append(app::tr("can't be read"));
                usable = false;
            }
            combo->addItem(made.join(QStringLiteral(" · ")), text_of(file.file));
            if (!usable) {
                if (auto* items = qobject_cast<QStandardItemModel*>(combo->model())) {
                    items->item(combo->count() - 1)->setEnabled(false);
                }
            } else if (first_usable < 0) {
                first_usable = combo->count() - 1;
            }
        }
        combo->setCurrentIndex(std::max(first_usable, 0));
        if (const auto picked = picked_.find(saved.folder); picked != picked_.end()) {
            if (const auto file = picked->second.find(key); file != picked->second.end()) {
                if (const int index = combo->findData(text_of(file->second)); index >= 0) {
                    combo->setCurrentIndex(index);
                }
            }
        }
        if (combo->count() == 1) {
            // Nothing to pick: its one score, as text (a disabled list is hard to read).
            add_row(metric_label(key) + ":", new QLabel(combo->itemText(0)));
            delete combo;
            continue;
        }
        combo->setToolTip(app::tr("Several scores of this metric are saved: the newest is used unless another is "
                                  "picked."));
        connect(combo, &QComboBox::currentIndexChanged, this, [this, combo, folder = saved.folder, key] {
            picked_[folder][key] = path_of(combo->currentData().toString());
        });
        add_row(metric_label(key) + ":", combo);
        score_combos_[key] = combo;
    }
}

QComboBox* SavedResultsPanel::score_choice(const std::string& key) const {
    const auto found = score_combos_.find(key);
    return found != score_combos_.end() ? found->second : nullptr;
}

std::vector<LoadedSavedResult> SavedResultsPanel::selected_results() const {
    std::vector<LoadedSavedResult> results;
    for (const SavedComparison* saved : selected()) {
        const auto picked = picked_.find(saved->folder);
        // The scores the box has read, where it has: not read again.
        const auto read = [this, saved](const std::filesystem::path& file) -> std::optional<SavedScore> {
            if (details_read_.folder == saved->folder) {
                if (const auto found = details_read_.scores.find(file); found != details_read_.scores.end()) {
                    return found->second;
                }
            }
            return read_saved_score(file);
        };
        const std::optional<std::pair<ComparisonResult, std::string>> loaded = load_saved_comparison(
            *saved, picked != picked_.end() ? picked->second : std::map<std::string, std::filesystem::path>{},
            read);
        if (loaded) {
            results.push_back({std::make_shared<const ComparisonResult>(loaded->first), qt_text(loaded->second),
                               saved->folder});
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

void SavedResultsPanel::add_run_folders(const std::vector<std::filesystem::path>& folders) {
    if (!run_active_) {
        return;  // a run's, told after it ended
    }
    for (const std::filesystem::path& folder : folders) {
        run_folders_.insert(run_key(folder));
    }
    run_folders_known_ = true;
    update_buttons();
}

bool SavedResultsPanel::locked_by_run(const SavedComparison& saved) const {
    return run_active_ && (!run_folders_known_ || run_pairs_.contains(pair_key(saved.source, saved.distorted)) ||
                           run_folders_.contains(run_key(saved.folder)));
}

void SavedResultsPanel::update_buttons() {
    const std::vector<const SavedComparison*> chosen = selected();
    open_button_->setEnabled(!chosen.empty());
    export_button_->setEnabled(!chosen.empty());
    // Locked where any selected comparison is one the run uses (the others
    // can be deleted while it runs), and while a Delete is queued.
    const bool locked = std::any_of(chosen.begin(), chosen.end(),
                                    [this](const SavedComparison* saved) { return locked_by_run(*saved); });
    delete_button_->setEnabled(!chosen.empty() && !locked && !deleting_);
    delete_button_->setToolTip(locked ? app::tr("Locked while metrics are calculated: the run saves its "
                                                "scores there.")
                                      : QString());
}

void SavedResultsPanel::fill_export_menu(QMenu& menu) {
    // The right-click menu's export first; then each kind on its own, for a
    // one-off: results and per-frame scores as JSON or CSV, the graphs as a
    // .zip, and the weak segments, by the rules picked under them.
    menu.clear();
    for (QMenu* kind : menu.findChildren<QMenu*>(Qt::FindDirectChildrenOnly)) {
        delete kind;  // clear() leaves the menus addMenu made
    }
    if (const QAction* contents = fill_menu(menu)) {
        connect(contents, &QAction::triggered, this,
                [this] { emit export_requested(Export::contents, selected_results()); });
    }
    if (menu.isEmpty()) {
        return;
    }
    menu.addSeparator();
    const auto add_formats = [this, &menu](const QString& title, Export json, Export csv) {
        QMenu* kind = menu.addMenu(title);
        connect(kind->addAction(app::tr("As JSON...")), &QAction::triggered, this,
                [this, json] { emit export_requested(json, selected_results()); });
        connect(kind->addAction(app::tr("As CSV...")), &QAction::triggered, this,
                [this, csv] { emit export_requested(csv, selected_results()); });
    };
    add_formats(app::tr("Results"), Export::results_json, Export::results_csv);
    add_formats(app::tr("Per-frame scores"), Export::frames_json, Export::frames_csv);
    QAction* graphs = menu.addAction(app::tr("Graphs as ZIP..."));
    graphs->setToolTip(app::tr("Each video's graph of every metric, and with several of one reference every metric "
                               "with all of them; the bitrate graphs where the videos are still there."));
    connect(graphs, &QAction::triggered, this, [this] { emit export_requested(Export::graphs, selected_results()); });
    QMenu* segments = menu.addMenu(app::tr("Weak segments"));
    segments->setToolTipsVisible(true);
    connect(segments->addAction(app::tr("Export weak segments...")), &QAction::triggered, this,
            [this] { emit export_requested(Export::segments, selected_results()); });
    segments->addSeparator();
    // The rules, one picked: this export's and the first entry's.
    auto* rules = new QActionGroup(segments);
    const QString picked = segment_rule_choice();
    for (const auto& [choice, text] : segment_rule_choices()) {
        QAction* action = segments->addAction(text);
        action->setCheckable(true);
        action->setChecked(choice == picked);
        action->setToolTip(app::tr("The rules the weak segments are found by: they are not saved with the scores, "
                                   "but found from them now."));
        rules->addAction(action);
        connect(action, &QAction::triggered, this, [this, choice] { pick_segment_rules(choice); });
    }
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
    QAction* contents = menu.addAction(export_action_text(
        count == 1 ? ExportScope::this_video : ExportScope::selected_videos, exported_items(settings_)));
    contents->setEnabled(!export_contents(exported_items(settings_)).isEmpty());
    return contents;
}

void SavedResultsPanel::fill_context_menu(QMenu& menu) {
    // Open in Metric Graphs, as the button under the box and a double-click,
    // so that it is beside the rows; then the export.
    if (selected().empty()) {
        return;
    }
    connect(menu.addAction(app::tr("Open in Metric Graphs")), &QAction::triggered, this,
            [this] { emit open_requested(selected_results()); });
    menu.addSeparator();
    if (const QAction* contents = fill_menu(menu)) {
        connect(contents, &QAction::triggered, this,
                [this] { emit export_requested(Export::contents, selected_results()); });
    }
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
    QString question =
        ntr("Delete the saved scores of {count} comparison? They can't be brought back: the videos are scored "
            "again the next time they are compared.",
            "Delete the saved scores of {count} comparisons? They can't be brought back: the videos are scored "
            "again the next time they are compared.",
            static_cast<long long>(chosen.size()));
    if (std::any_of(chosen.begin(), chosen.end(), [](const SavedComparison& saved) { return !saved.reference_found; })) {
        question += "\n\n" + app::tr("Some of them could not be scored again: their reference video is no longer "
                                      "where it was (marked \"not found\").");
    }
    if (!dialogs_.ask_no_default(app::tr("Delete saved results"), question) || locked()) {
        return;
    }
    // Through the window's queue of writes, after what is queued there, as
    // Settings' Clear: a run's saves still queued would otherwise bring a
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
