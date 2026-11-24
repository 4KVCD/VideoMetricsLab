#include "widgets.h"

#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QHelpEvent>
#include <QToolTip>
#include <QTextDocument>
#include <QApplication>
#include <QComboBox>
#include <QContextMenuEvent>
#include <QCoreApplication>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QFocusEvent>
#include <QFrame>
#include <QIcon>
#include <QImage>
#include <QItemSelectionModel>
#include <QKeyEvent>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPaintEvent>
#include <QPixmap>
#include <QPen>
#include <QStyleOptionSlider>
#include <QResizeEvent>
#include <QScrollBar>
#include <QShowEvent>
#include <QSizePolicy>
#include <QStyle>
#include <QStyleOption>
#include <QStyleOptionButton>
#include <QStyleOptionHeader>
#include <QTabBar>
#include <QTableView>
#include <QUrl>
#include <QWheelEvent>

#include <algorithm>
#include <numeric>
#include <utility>

#include "i18n.h"

namespace vml::app {

namespace {

constexpr int indicator_margin = 4;
// The fill column's narrowest, in characters: with many columns shown the
// table scrolls sideways rather than squeeze it further (a file name of 24
// characters stays readable).
constexpr int min_fill_characters = 24;

}  // namespace

// ----------------------------------------------------------- CheckableHeaderView

CheckableHeaderView::CheckableHeaderView(const QMap<int, bool>& checkable, QWidget* parent)
    : QHeaderView(Qt::Horizontal, parent), checked_(checkable) {
    setSectionsClickable(true);
}

bool CheckableHeaderView::is_checked(int section) const { return checked_.value(section, false); }

void CheckableHeaderView::set_checked(int section, bool value) {
    if (checked_.contains(section) && checked_.value(section) != value) {
        checked_[section] = value;
        updateSection(section);
    }
}

QRect CheckableHeaderView::indicator_rect(const QRect& rect) const {
    const int size = style()->pixelMetric(QStyle::PM_IndicatorWidth, nullptr, this);
    return {rect.x() + indicator_margin, rect.y() + (rect.height() - size) / 2, size, size};
}

void CheckableHeaderView::paintSection(QPainter* painter, const QRect& rect, int logical_index) const {
    painter->save();
    QHeaderView::paintSection(painter, rect, logical_index);
    painter->restore();
    if (!checked_.contains(logical_index)) {
        return;
    }
    QStyleOptionButton option;
    option.rect = indicator_rect(rect);
    option.state = QStyle::State_Enabled | (checked_.value(logical_index) ? QStyle::State_On : QStyle::State_Off);
    style()->drawPrimitive(QStyle::PE_IndicatorCheckBox, &option, painter, this);
}

QSize CheckableHeaderView::sectionSizeFromContents(int logical_index) const {
    QSize size = QHeaderView::sectionSizeFromContents(logical_index);
    if (checked_.contains(logical_index)) {
        // Qt's size hint knows only the centred text, not the checkbox painted
        // over its left edge: room on both sides, so that at its default width
        // the label cannot overlap the checkbox.
        const int indicator = style()->pixelMetric(QStyle::PM_IndicatorWidth, nullptr, this);
        size.setWidth(size.width() + 2 * (indicator + indicator_margin));
    }
    return size;
}

QRect CheckableHeaderView::section_indicator_rect(int section) const {
    return indicator_rect(QRect(sectionViewportPosition(section), 0, sectionSize(section), height()));
}

void CheckableHeaderView::mousePressEvent(QMouseEvent* event) {
    const QPoint position = event->position().toPoint();
    const int index = logicalIndexAt(position);
    if (checked_.contains(index) && section_indicator_rect(index).contains(position)) {
        checked_[index] = !checked_.value(index);
        updateSection(index);
        emit sectionToggled(index, checked_.value(index));
        return;
    }
    // Anywhere else in the section is an ordinary header click: dragging the
    // edge to resize the column, or sorting. A click anywhere in the section
    // must not toggle: every near miss of the checkbox would quietly change
    // what the next run calculates.
    QHeaderView::mousePressEvent(event);
}

// ------------------------------------------------------------- FrozenColumnsView

// freeze_leading_columns' view: the table's leading columns again, over the
// table's own, in the strip of those columns (frozen_strip). It is a child of
// the table's viewport, not a sibling over it: Qt scrolls a widget by moving
// its pixels (a blit) only while no sibling lies over it, and otherwise
// redraws only what no opaque sibling covers, which for a sibling as large as
// the viewport is nothing: the cells on screen would go stale. Its heading is
// a FrozenHeader of its own, over the table's header.
//
// It only shows: whatever is done over it (presses, drags, drops, the wheel,
// a context menu, a tooltip) is passed on to the table, at the same place in
// the table's viewport, and the table then finds the cell drawn here
// (indexAt, visualRect). One view then holds the selection's anchor, the
// pressed cell and the drag, wherever the user clicks. Its own hover stays
// its own, for the cells it draws; a mouse move passed on brings the table's.
class FrozenColumnsView : public QTableView {
public:
    explicit FrozenColumnsView(FillColumnTable& table) : QTableView(table.viewport()), table_(table) {}

protected:
    bool viewportEvent(QEvent* event) override {
        switch (event->type()) {
            case QEvent::MouseButtonPress:
            case QEvent::MouseButtonRelease:
            case QEvent::MouseButtonDblClick:
            case QEvent::MouseMove: {
                const auto* mouse = static_cast<QMouseEvent*>(event);
                QMouseEvent moved(mouse->type(), to_table(mouse->position()), mouse->scenePosition(),
                                  mouse->globalPosition(), mouse->button(), mouse->buttons(), mouse->modifiers(),
                                  mouse->pointingDevice());
                moved.setTimestamp(mouse->timestamp());  // double-clicks are told by it
                return pass_on(event, moved);
            }
            case QEvent::Wheel: {
                const auto* wheel = static_cast<QWheelEvent*>(event);
                QWheelEvent moved(to_table(wheel->position()), wheel->globalPosition(), wheel->pixelDelta(),
                                  wheel->angleDelta(), wheel->buttons(), wheel->modifiers(), wheel->phase(),
                                  wheel->inverted(), wheel->source(), wheel->pointingDevice());
                moved.setTimestamp(wheel->timestamp());
                return pass_on(event, moved);
            }
            case QEvent::ContextMenu: {
                const auto* menu = static_cast<QContextMenuEvent*>(event);
                QContextMenuEvent moved(menu->reason(), to_table(menu->pos()), menu->globalPos(), menu->modifiers());
                return pass_on(event, moved);
            }
            case QEvent::ToolTip:
            case QEvent::QueryWhatsThis:
            case QEvent::WhatsThis: {
                const auto* help = static_cast<QHelpEvent*>(event);
                QHelpEvent moved(help->type(), to_table(help->pos()), help->globalPos());
                return pass_on(event, moved);
            }
            case QEvent::DragEnter: {
                auto* drag = static_cast<QDragEnterEvent*>(event);
                QDragEnterEvent moved(to_table(drag->position()), drag->possibleActions(), drag->mimeData(),
                                      drag->buttons(), drag->modifiers());
                return pass_on_drop(drag, moved);
            }
            case QEvent::DragMove: {
                auto* drag = static_cast<QDragMoveEvent*>(event);
                QDragMoveEvent moved(to_table(drag->position()), drag->possibleActions(), drag->mimeData(),
                                     drag->buttons(), drag->modifiers());
                return pass_on_drop(drag, moved);
            }
            case QEvent::Drop: {
                auto* drop = static_cast<QDropEvent*>(event);
                QDropEvent moved(to_table(drop->position()), drop->possibleActions(), drop->mimeData(),
                                 drop->buttons(), drop->modifiers());
                return pass_on_drop(drop, moved);
            }
            case QEvent::DragLeave:  // no position
                QCoreApplication::sendEvent(table_.viewport(), event);
                return true;
            default:
                return QTableView::viewportEvent(event);
        }
    }

    void paintEvent(QPaintEvent* event) override {
        QTableView::paintEvent(event);
        table_.paint_drop_line(viewport());
    }

private:
    // A point on this view's viewport, on the table's.
    QPointF to_table(QPointF position) const { return position + viewport()->mapTo(table_.viewport(), QPoint()); }
    QPoint to_table(QPoint position) const { return position + viewport()->mapTo(table_.viewport(), QPoint()); }

    // Sends `moved`, the table's copy of `event`, to the table's viewport,
    // and gives `event` its answer.
    bool pass_on(QEvent* event, QEvent& moved) {
        QCoreApplication::sendEvent(table_.viewport(), &moved);
        event->setAccepted(moved.isAccepted());
        return true;
    }
    bool pass_on_drop(QDropEvent* event, QDropEvent& moved) {
        pass_on(event, moved);
        event->setDropAction(moved.dropAction());  // the drag shows it, and the drop reports it
        return true;
    }

    FillColumnTable& table_;
};

namespace {

// The frozen columns' heading, over the table's header at frozen_'s strip.
// It draws the sort arrow of the table's own header (which the sorting sets).
// A child of the table, like the header itself, not of the header's
// viewport: that one is scrolled outside of scrollContentsBy too.
class FrozenHeader : public QHeaderView {
public:
    FrozenHeader(const QTableView& table, QWidget* parent) : QHeaderView(Qt::Horizontal, parent), table_(table) {}

protected:
    // Some styles (Windows 11's) draw header sections see-through, over the
    // table's base; the table's own scrolled heading lies under this one, so
    // fill the base first or its text shows through.
    void paintEvent(QPaintEvent* event) override {
        {
            QPainter painter(viewport());
            painter.fillRect(event->rect(), table_.palette().brush(QPalette::Base));
        }
        QHeaderView::paintEvent(event);
    }

    void initStyleOptionForIndex(QStyleOptionHeader* option, int logical_index) const override {
        QHeaderView::initStyleOptionForIndex(option, logical_index);
        const QHeaderView* header = table_.horizontalHeader();
        if (!header->isSortIndicatorShown() || header->sortIndicatorSection() != logical_index) {
            option->sortIndicator = QStyleOptionHeader::None;
        } else {
            // As QHeaderView draws its own.
            option->sortIndicator = header->sortIndicatorOrder() == Qt::AscendingOrder ? QStyleOptionHeader::SortDown
                                                                                         : QStyleOptionHeader::SortUp;
        }
    }

private:
    const QTableView& table_;
};

}  // namespace

// --------------------------------------------------------------- FillColumnTable

FillColumnTable::FillColumnTable(int rows, int columns, int fill_column, std::vector<int> other_columns,
                                 QWidget* parent)
    : QTableWidget(rows, columns, parent), fill_column_(fill_column), other_columns_(std::move(other_columns)) {
    connect(horizontalHeader(), &QHeaderView::sectionResized, this, &FillColumnTable::on_section_resized);
}

void FillColumnTable::select_row_if_none_is(int row) {
    if (row < 0 || row >= rowCount() || selectionModel()->hasSelection()) {
        return;
    }
    // The current cell in the first column shown, where the keyboard goes on.
    int column = 0;
    while (column < columnCount() - 1 && isColumnHidden(column)) {
        ++column;
    }
    selectionModel()->setCurrentIndex(model()->index(row, column),
                                      QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
}

void FillColumnTable::keep_a_row_selected() {
    keeps_a_row_ = true;
    connect(selectionModel(), &QItemSelectionModel::selectionChanged, this, [this] {
        const QModelIndexList rows = selectionModel()->selectedRows();
        if (!rows.isEmpty()) {
            last_selected_row_ = currentRow() >= 0 && selectionModel()->isRowSelected(currentRow(), {})
                                     ? currentRow()
                                     : rows.front().row();
            return;
        }
        // Once control is back in the event loop, not here: rows are removed
        // one by one (removeRow) while their owner's own list of them is
        // still being changed, and a row selected now would be read from it.
        if (!reselect_pending_) {
            reselect_pending_ = true;
            QMetaObject::invokeMethod(this, [this] {
                reselect_pending_ = false;
                if (rowCount() > 0) {
                    select_row_if_none_is(std::clamp(last_selected_row_, 0, rowCount() - 1));
                }
            }, Qt::QueuedConnection);
        }
    });
}

void FillColumnTable::mousePressEvent(QMouseEvent* event) {
    // Where there is no row: nothing, rather than no row selected (Qt clears
    // the selection there), and no drag to select from there.
    if (keeps_a_row_ && rowCount() > 0 && !indexAt(event->position().toPoint()).isValid()) {
        event->accept();
        return;
    }
    QTableWidget::mousePressEvent(event);
}

void FillColumnTable::setHorizontalHeader(QHeaderView* header) {
    // A new header does not have the connection the constructor made to the
    // old one; without it, the fill column would stop following the other
    // columns' resizing.
    QTableWidget::setHorizontalHeader(header);
    connect(header, &QHeaderView::sectionResized, this, &FillColumnTable::on_section_resized);
    if (frozen_ != nullptr) {
        header->viewport()->installEventFilter(this);  // eventFilter: the sort arrow
        frozen_header_->raise();                       // over the new header too
        place_frozen_view();
    }
}

void FillColumnTable::resizeEvent(QResizeEvent* event) {
    QTableWidget::resizeEvent(event);
    recalculate_fill_column();
}

QStringList FillColumnTable::dropped_files(const QMimeData* mime) {
    QStringList files;
    if (mime == nullptr || !mime->hasUrls()) {
        return files;
    }
    for (const QUrl& url : mime->urls()) {
        if (!url.isLocalFile()) {
            continue;
        }
        const QString path = url.toLocalFile();
        if (!path.isEmpty() && QFileInfo(path).isFile()) {
            files.append(path);
        }
    }
    return files;
}

FillColumnTable::DropKind FillColumnTable::drop_kind(QDropEvent* event) const {
    if (!drops_allowed) {
        return DropKind::none;
    }
    if (event->source() == this) {
        return selectionModel()->selectedRows().isEmpty() ? DropKind::none : DropKind::rows;
    }
    return dropped_files(event->mimeData()).isEmpty() ? DropKind::none : DropKind::files;
}

int FillColumnTable::insertion_row(QDropEvent* event) const {
    const int y = event->position().toPoint().y();
    const int row = rowAt(y);
    if (row < 0) {
        return y > 0 ? rowCount() : 0;
    }
    const int middle = rowViewportPosition(row) + rowHeight(row) / 2;
    return y < middle ? row : row + 1;
}

void FillColumnTable::set_drop_line(std::optional<int> row) {
    if (row != drop_line_) {
        drop_line_ = row;
        viewport()->update();
        if (frozen_ != nullptr) {
            frozen_->viewport()->update();  // the line runs under frozen_ too
        }
    }
}

void FillColumnTable::dragEnterEvent(QDragEnterEvent* event) {
    drag_kind_ = drop_kind(event);
    if (drag_kind_ == DropKind::none) {
        event->ignore();
        return;
    }
    event->setDropAction(drag_kind_ == DropKind::rows ? Qt::MoveAction : Qt::CopyAction);
    event->accept();
}

void FillColumnTable::dragMoveEvent(QDragMoveEvent* event) {
    const DropKind kind = drops_allowed ? drag_kind_ : DropKind::none;
    if (kind == DropKind::none) {
        set_drop_line(std::nullopt);
        event->ignore();
        return;
    }
    set_drop_line(kind == DropKind::rows ? std::optional(insertion_row(event)) : std::nullopt);
    event->setDropAction(kind == DropKind::rows ? Qt::MoveAction : Qt::CopyAction);
    event->accept();
}

void FillColumnTable::dragLeaveEvent(QDragLeaveEvent* event) {
    drag_kind_ = DropKind::none;
    set_drop_line(std::nullopt);
    QTableWidget::dragLeaveEvent(event);
}

void FillColumnTable::dropEvent(QDropEvent* event) {
    set_drop_line(std::nullopt);
    drag_kind_ = DropKind::none;
    const DropKind kind = drop_kind(event);
    if (kind == DropKind::none) {
        event->ignore();
        return;
    }
    if (kind == DropKind::rows) {
        QList<int> rows;
        for (const QModelIndex& index : selectionModel()->selectedRows()) {
            rows.append(index.row());
        }
        std::sort(rows.begin(), rows.end());
        const int target = insertion_row(event);
        // Reported as a copy (a drag from this table offers both), because on
        // a move, Qt's drag would delete the dragged rows itself. A drag that
        // offers only a move is turned down.
        event->setDropAction(Qt::CopyAction);
        if (event->dropAction() != Qt::CopyAction) {
            event->ignore();
            return;
        }
        event->accept();
        emit rowsMoved(rows, target);
        return;
    }
    const QStringList files = dropped_files(event->mimeData());
    event->setDropAction(Qt::CopyAction);
    event->accept();
    // Said once the drop has returned: what the files start may ask a
    // question ("Already added"), and one opened while the system's drag
    // and drop is still running opened at the window's corner, not centred
    // over it (issue #202).
    QMetaObject::invokeMethod(this, [this, files] { emit filesDropped(files); }, Qt::QueuedConnection);
}

void FillColumnTable::paintEvent(QPaintEvent* event) {
    QTableWidget::paintEvent(event);
    paint_drop_line(viewport());
}

void FillColumnTable::paint_drop_line(QWidget* port) const {
    if (!drop_line_) {
        return;
    }
    int y = 0;
    if (*drop_line_ < rowCount()) {
        y = rowViewportPosition(*drop_line_);
    } else if (rowCount() > 0) {
        const int last = rowCount() - 1;
        y = rowViewportPosition(last) + rowHeight(last) - 1;
    }
    QPainter painter(port);
    painter.setPen(QPen(palette().highlight().color(), 2));
    painter.drawLine(0, y, port->width(), y);
}

// frozen_ draws the current cell's focus frame while the table has the focus.
void FillColumnTable::focusInEvent(QFocusEvent* event) {
    QTableWidget::focusInEvent(event);
    if (frozen_ != nullptr) {
        frozen_->viewport()->update();
    }
}

void FillColumnTable::focusOutEvent(QFocusEvent* event) {
    QTableWidget::focusOutEvent(event);
    if (frozen_ != nullptr) {
        frozen_->viewport()->update();
    }
}

void FillColumnTable::on_section_resized(int logical_index, int /*old_size*/, int new_size) {
    // A frozen column's width, the fill column's included, is frozen_'s and
    // its heading's too, and moves the edge of its strip.
    if (frozen_ != nullptr && is_frozen(logical_index)) {
        frozen_->setColumnWidth(logical_index, new_size);
        if (frozen_header_->sectionSize(logical_index) != new_size) {
            frozen_header_->resizeSection(logical_index, new_size);
        }
        place_frozen_view();
    }
    if (recalculating_) {
        return;
    }
    if (logical_index == fill_column_) {
        fill_column_user_width_ = new_size;  // a manual drag of the fill column's own edge
        return;
    }
    recalculate_fill_column();
}

void FillColumnTable::recalculate_fill_column() {
    if (recalculating_) {
        return;
    }
    const int others = std::accumulate(other_columns_.begin(), other_columns_.end(), 0,
                                       [this](int total, int column) { return total + columnWidth(column); });
    const int narrowest = fontMetrics().averageCharWidth() * min_fill_characters;
    const int natural = std::max(narrowest, viewport()->width() - others);
    const int target = fill_column_user_width_ ? std::max(natural, *fill_column_user_width_) : natural;
    if (target == columnWidth(fill_column_)) {
        return;
    }
    recalculating_ = true;
    setColumnWidth(fill_column_, target);
    recalculating_ = false;
}

void FillColumnTable::freeze_leading_columns(int count) {
    if (frozen_ != nullptr || count <= 0) {
        return;  // once
    }
    frozen_count_ = count;
    auto* view = new FrozenColumnsView(*this);
    frozen_ = view;
    view->setObjectName(QStringLiteral("frozenColumns"));  // VML Modern draws it without the table's border
    view->horizontalHeader()->hide();                       // frozen_header_ is its heading
    // Opaque, as the viewport under it is: Qt then moves it by moving its
    // pixels too (scrollContentsBy).
    view->setAutoFillBackground(true);
    // The table's rows, selection and cell painting.
    view->setModel(model());
    QItemSelectionModel* own_selection = view->selectionModel();
    view->setSelectionModel(selectionModel());
    delete own_selection;
    view->setItemDelegate(itemDelegate());
    view->setSelectionBehavior(selectionBehavior());
    view->setSelectionMode(selectionMode());
    view->setEditTriggers(NoEditTriggers);  // the table edits, if anything
    view->setWordWrap(wordWrap());
    view->setTextElideMode(textElideMode());
    view->setAlternatingRowColors(alternatingRowColors());
    view->setShowGrid(showGrid());
    view->setGridStyle(gridStyle());
    view->setVerticalScrollMode(verticalScrollMode());
    view->setAutoScroll(false);  // only the table scrolls; frozen_ follows it
    view->setFrameShape(QFrame::NoFrame);
    view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    view->verticalHeader()->hide();
    // Keys go to the table; the current cell's focus frame is drawn while it
    // has the focus.
    view->setFocusPolicy(Qt::NoFocus);
    view->setFocusProxy(this);
    // Drops and hovering reach frozen_'s viewport first (FrozenColumnsView).
    view->setAcceptDrops(true);
    view->viewport()->setAcceptDrops(true);
    view->viewport()->setAttribute(Qt::WA_Hover, viewport()->testAttribute(Qt::WA_Hover));
    view->viewport()->setMouseTracking(viewport()->hasMouseTracking());

    QHeaderView* header = horizontalHeader();
    auto* frozen_header = new FrozenHeader(*this, this);
    frozen_header_ = frozen_header;
    frozen_header->setObjectName(QStringLiteral("frozenHeader"));
    frozen_header->setModel(model());
    QItemSelectionModel* header_selection = frozen_header->selectionModel();
    frozen_header->setSelectionModel(selectionModel());  // for highlightSections
    delete header_selection;
    frozen_header->setSectionResizeMode(QHeaderView::Interactive);
    frozen_header->setStretchLastSection(false);
    frozen_header->setSectionsClickable(header->sectionsClickable());
    frozen_header->setHighlightSections(header->highlightSections());
    frozen_header->setDefaultAlignment(header->defaultAlignment());
    frozen_header->setMinimumSectionSize(header->minimumSectionSize());
    frozen_header->setContextMenuPolicy(header->contextMenuPolicy());
    // In frozen_ and its heading, the frozen columns in the table's order,
    // the others hidden.
    for (QHeaderView* columns : {view->horizontalHeader(), static_cast<QHeaderView*>(frozen_header)}) {
        for (int visual = 0; visual < count && visual < header->count(); ++visual) {
            columns->moveSection(columns->visualIndex(header->logicalIndex(visual)), visual);
        }
        for (int column = 0; column < columnCount(); ++column) {
            columns->setSectionHidden(column, !is_frozen(column) || isColumnHidden(column));
            if (is_frozen(column)) {
                columns->resizeSection(column, columnWidth(column));
            }
        }
    }

    // A frozen column resized on its heading is resized in the table
    // (on_section_resized brings frozen_ and the heading along).
    connect(frozen_header, &QHeaderView::sectionResized, this, [this](int column, int, int size) {
        if (columnWidth(column) != size) {
            setColumnWidth(column, size);
        }
    });
    // A click on a frozen heading sorts, and a right-click opens the
    // header's menu, as on the table's own header.
    connect(frozen_header, &QHeaderView::sectionClicked, this,
            [this](int column) { emit horizontalHeader()->sectionClicked(column); });
    connect(frozen_header, &QWidget::customContextMenuRequested, this, [this, frozen_header](const QPoint& position) {
        QHeaderView* own = horizontalHeader();
        emit own->customContextMenuRequested(own->mapFromGlobal(frozen_header->mapToGlobal(position)));
    });
    // The rows scroll together: the table's scroll bar leads, frozen_'s
    // follows it, also when its range changes later than the table's.
    connect(verticalScrollBar(), &QAbstractSlider::valueChanged, view->verticalScrollBar(),
            &QAbstractSlider::setValue);
    connect(view->verticalScrollBar(), &QAbstractSlider::rangeChanged, this,
            [this] { frozen_->verticalScrollBar()->setValue(verticalScrollBar()->value()); });
    connect(verticalHeader(), &QHeaderView::sectionResized, this,
            [this](int row, int, int size) { frozen_->setRowHeight(row, size); });
    // The sort arrow, which frozen_header_ copies, is checked as the
    // table's header paints (eventFilter).
    header->viewport()->installEventFilter(this);

    setHorizontalScrollMode(ScrollPerPixel);  // a column slides under frozen_, not jumps
    view->show();
    frozen_header->show();
    place_frozen_view();
}

bool FillColumnTable::is_frozen(int logical_column) const {
    const int visual = horizontalHeader()->visualIndex(logical_column);
    return visual >= 0 && visual < frozen_count_;
}

QRect FillColumnTable::frozen_strip() const {
    if (frozen_ == nullptr) {
        return {};
    }
    int width = 0;
    for (int visual = 0; visual < frozen_count_; ++visual) {
        const int column = horizontalHeader()->logicalIndex(visual);
        if (column >= 0 && !isColumnHidden(column)) {
            width += columnWidth(column);
        }
    }
    // At most three quarters of the table: a File name dragged wider than
    // the window still leaves the other columns room to scroll into view.
    const int port_width = viewport()->width();
    width = std::min(width, port_width * 3 / 4);
    return {isRightToLeft() ? port_width - width : 0, 0, width, viewport()->height()};
}

void FillColumnTable::place_frozen_view() {
    if (frozen_ == nullptr) {
        return;
    }
    const QRect strip = frozen_strip();
    QHeaderView* header = horizontalHeader();
    frozen_->setVisible(!strip.isEmpty());
    frozen_header_->setVisible(!strip.isEmpty() && !header->isHidden());
    if (strip.isEmpty()) {
        return;
    }
    // Only moves what is not in place (after a scroll, scrollContentsBy).
    frozen_->setGeometry(strip);
    const QRect heading(viewport()->x() + strip.x(), header->y(), strip.width(), header->height());
    frozen_header_->setGeometry(heading);
    // Rows as high as the table's, which the style and font set.
    QHeaderView* rows = verticalHeader();
    QHeaderView* frozen_rows = frozen_->verticalHeader();
    if (frozen_rows->minimumSectionSize() != rows->minimumSectionSize()) {
        frozen_rows->setMinimumSectionSize(rows->minimumSectionSize());
    }
    if (frozen_rows->defaultSectionSize() != rows->defaultSectionSize()) {
        frozen_rows->setDefaultSectionSize(rows->defaultSectionSize());
    }
}

QPoint FillColumnTable::frozen_offset() const {
    return frozen_->viewport()->mapTo(viewport(), QPoint());
}

QModelIndex FillColumnTable::indexAt(const QPoint& position) const {
    if (frozen_ != nullptr && !frozen_->isHidden() && frozen_strip().contains(position)) {
        return frozen_->indexAt(position - frozen_offset());
    }
    return QTableWidget::indexAt(position);
}

QRect FillColumnTable::visualRect(const QModelIndex& index) const {
    if (frozen_ != nullptr && !frozen_->isHidden() && index.isValid() && is_frozen(index.column())) {
        return frozen_->visualRect(index).translated(frozen_offset());
    }
    return QTableWidget::visualRect(index);
}

void FillColumnTable::updateGeometries() {
    QTableWidget::updateGeometries();
    place_frozen_view();
}

void FillColumnTable::scrollContentsBy(int dx, int dy) {
    // The viewport's scroll moves its children with it, frozen_ too: back to
    // the strip, once the scroll has returned (frozen_'s move event comes
    // while the viewport is still being scrolled).
    QTableWidget::scrollContentsBy(dx, dy);
    place_frozen_view();
}

void FillColumnTable::scrollTo(const QModelIndex& index, ScrollHint hint) {
    QScrollBar* bar = horizontalScrollBar();
    const int before = bar->value();
    QTableWidget::scrollTo(index, hint);
    if (frozen_ == nullptr || frozen_->isHidden() || !index.isValid()) {
        return;
    }
    if (is_frozen(index.column())) {
        bar->setValue(before);  // always in view: only the rows scroll
        return;
    }
    // A cell under frozen_ is brought out to its edge. Positions counted
    // from the leading edge, as the scroll bar's value is, either way round.
    const int position = horizontalHeader()->sectionPosition(index.column());
    const int strip_width = frozen_strip().width();
    if (position - bar->value() < strip_width) {
        bar->setValue(position - strip_width);
    }
}

bool FillColumnTable::eventFilter(QObject* watched, QEvent* event) {
    if (frozen_ != nullptr && event->type() == QEvent::Paint && watched == horizontalHeader()->viewport()) {
        const QHeaderView* header = horizontalHeader();
        const std::tuple sort{header->isSortIndicatorShown(), header->sortIndicatorSection(),
                              header->sortIndicatorOrder()};
        if (sort != frozen_sort_) {
            frozen_sort_ = sort;
            frozen_header_->viewport()->update();
        }
    }
    return QTableWidget::eventFilter(watched, event);
}

// ------------------------------------------------------------------ ElidedLabel

BandSlider::BandSlider(Qt::Orientation orientation, QWidget* parent) : QSlider(orientation, parent) {}

void BandSlider::set_band(std::optional<std::pair<int, int>> band) {
    if (band == band_) {
        return;
    }
    band_ = band;
    update();
}

QRect BandSlider::band_rect() const {
    if (!band_ || maximum() <= minimum() || orientation() != Qt::Horizontal) {
        return {};
    }
    QStyleOptionSlider option;
    initStyleOption(&option);
    const QRect groove = style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderGroove, this);
    const QRect handle = style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, this);
    // The handle's centre at a value, as the style places it.
    const int span = groove.width() - handle.width();
    const auto centre = [&](int value) {
        return groove.left() + handle.width() / 2 +
               QStyle::sliderPositionFromValue(minimum(), maximum(), std::clamp(value, minimum(), maximum()), span,
                                               option.upsideDown);
    };
    int left = centre(std::min(band_->first, band_->second));
    int right = centre(std::max(band_->first, band_->second));
    if (right - left < min_band_px) {
        const int middle = (left + right) / 2;
        left = middle - min_band_px / 2;
        right = left + min_band_px;
    }
    const int height = std::max(handle.height(), groove.height());
    return {left, rect().center().y() - height / 2, right - left, height};
}

void BandSlider::paintEvent(QPaintEvent* event) {
    if (!band_) {
        QSlider::paintEvent(event);
        return;
    }
    // The groove, the band over it, then the handle over both.
    QPainter painter(this);
    QStyleOptionSlider option;
    initStyleOption(&option);
    option.subControls = QStyle::SC_SliderGroove;
    style()->drawComplexControl(QStyle::CC_Slider, &option, &painter, this);
    const QColor accent = palette().color(QPalette::Accent);
    QColor fill = accent;
    fill.setAlphaF(isEnabled() ? 0.28F : 0.12F);
    QColor edge = accent;
    edge.setAlphaF(isEnabled() ? 0.6F : 0.25F);
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(edge, 1.0));
    painter.setBrush(fill);
    painter.drawRoundedRect(QRectF(band_rect()).adjusted(0.5, 0.5, -0.5, -0.5), 3, 3);
    painter.restore();
    initStyleOption(&option);
    option.subControls = QStyle::SC_SliderHandle;
    style()->drawComplexControl(QStyle::CC_Slider, &option, &painter, this);
}

ElidedLabel::ElidedLabel(QWidget* parent) : QLabel(parent) {
    setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
}

void ElidedLabel::set_text(const QString& text, const QString& tooltip) {
    full_ = text;
    shorter_.clear();
    tooltip_ = tooltip;
    fit();
}

void ElidedLabel::set_text_or_shorter(const QString& text, const QString& shorter) {
    full_ = text;
    shorter_ = shorter == text ? QString() : shorter;
    tooltip_.clear();
    fit();
}

void ElidedLabel::clear_text() {
    full_.clear();
    shorter_.clear();
    tooltip_.clear();
    QLabel::clear();
    setToolTip(QString());
}

QSize ElidedLabel::minimumSizeHint() const { return {0, QLabel::minimumSizeHint().height()}; }

void ElidedLabel::set_elide_mode(Qt::TextElideMode mode) {
    mode_ = mode;
    fit();
}

void ElidedLabel::resizeEvent(QResizeEvent* event) {
    QLabel::resizeEvent(event);
    fit();
}

void ElidedLabel::showEvent(QShowEvent* event) {
    QLabel::showEvent(event);
    fit();
}

void ElidedLabel::fit() {
    const int width = contentsRect().width();
    // Not cut until shown and laid out: before that the width is a
    // placeholder, not the room the line will have.
    // The whole text where it fits, else the shorter one, cut if need be.
    QString shown = full_;
    if (isVisible() && width > 0 && fontMetrics().horizontalAdvance(full_) > width) {
        shown = fontMetrics().elidedText(shorter_.isEmpty() ? full_ : shorter_, mode_, width);
    }
    QLabel::setText(shown);
    const bool cut = shown != full_;
    setToolTip(cut ? full_ + (tooltip_.isEmpty() ? QString() : "\n\n" + tooltip_) : tooltip_);
}

// ---------------------------------------------------------------- ReadoutLabel

namespace {

// A text's line count and its longest line's length in characters.
std::pair<int, qsizetype> text_extent(const QString& text) {
    int lines = 1;
    qsizetype longest = 0;
    qsizetype start = 0;
    for (qsizetype at = 0; at <= text.size(); ++at) {
        if (at == text.size() || text[at] == QLatin1Char('\n')) {
            longest = std::max(longest, at - start);
            start = at + 1;
            lines += at < text.size() ? 1 : 0;
        }
    }
    return {lines, longest};
}

}  // namespace

ReadoutLabel::ReadoutLabel(const QString& text, QWidget* parent) : QWidget(parent), text_(text) {
    // The tab's colour, which the pane it sits on is drawn in, and a label's
    // text colour.
    setBackgroundRole(QPalette::Base);
    setForegroundRole(QPalette::WindowText);
    setAutoFillBackground(true);
    right_to_left_ = text_.isRightToLeft();
}

void ReadoutLabel::set_text(const QString& text) {
    if (text == text_) {
        return;
    }
    text_ = text;
    right_to_left_ = text_.isRightToLeft();
    text_size_.reset();
    // The text is not measured here: as QLabel measures it, a readout takes
    // 0.13 ms, and painting it measures it again. It is laid out again only
    // when it has more lines, or a longer one, than the text it was last
    // sized for (sizeHint): a readout's lines keep their lengths as the
    // values in them change.
    const auto [lines, longest] = text_extent(text_);
    if (lines > sized_for_.first || longest > sized_for_.second) {
        updateGeometry();
    }
    update(contentsRect());  // the text's room; its indent moves to the right for right-to-left text
}

void ReadoutLabel::set_indent(int indent, int trailing) {
    indent_ = indent;
    trailing_ = trailing;
    remeasure();
}

void ReadoutLabel::reserve_lines(int lines) {
    if (lines != reserved_lines_) {
        reserved_lines_ = lines;
        updateGeometry();
    }
}

QSize ReadoutLabel::sizeHint() const {
    // As QLabel sizes plain text that does not wrap (QLabelPrivate::
    // sizeForWidth): the text, the indent and the room after the text, the
    // margins. Measured only here, when the layout asks.
    if (!text_size_) {
        text_size_ = measure(text_);
        sized_for_ = text_extent(text_);
    }
    int height = text_size_->height();
    if (reserved_lines_ > 0) {
        QStringList lines;
        for (int line = 0; line < reserved_lines_; ++line) {
            lines.append(QStringLiteral("x"));
        }
        height = std::max(height, measure(lines.join('\n')).height());
    }
    const QMargins margins = contentsMargins();
    return QSize(text_size_->width() + indent_ + trailing_ + margins.left() + margins.right(),
                 height + indent_ + margins.top() + margins.bottom())
        .expandedTo(minimumSize());
}

QSize ReadoutLabel::minimumSizeHint() const {
    // As tall as the text, but no wider than its indent and margins: a long
    // line (a long file name in it) is cut at the edge, not made the least
    // width of the page. As QLabel's, it was the text's whole width, so a
    // readout over a long-named video widened the Metric Graphs tab past
    // the window: the tab scrolled sideways, and Worst frames' divider could
    // not be dragged, the chart's side being at its least width already.
    const QMargins margins = contentsMargins();
    return QSize(indent_ + trailing_ + margins.left() + margins.right(), sizeHint().height());
}

void ReadoutLabel::paintEvent(QPaintEvent* /*event*/) {
    // As QLabel paints plain text; the background is filled already.
    QPainter painter(this);
    QStyleOption option;
    option.initFrom(this);
    const int flags = alignment() | (right_to_left_ ? Qt::TextForceRightToLeft : Qt::TextForceLeftToRight);
    style()->drawItemText(&painter, text_rect(), flags, option.palette, isEnabled(), text_, foregroundRole());
}

void ReadoutLabel::changeEvent(QEvent* event) {
    if (event->type() == QEvent::FontChange) {
        remeasure();
    }
    QWidget::changeEvent(event);
}

QSize ReadoutLabel::measure(const QString& text) const {
    // As QLabel measures it: laid out within 2000 pixels, less the indent,
    // the room after the text and the margins.
    const QMargins margins = contentsMargins();
    const int width = 2000 - (indent_ + trailing_ + margins.left() + margins.right());
    return fontMetrics().boundingRect(0, 0, width, 2000, alignment(), text).size();
}

int ReadoutLabel::alignment() const {
    // At the left and top, or at the right for right-to-left text.
    return static_cast<int>(QStyle::visualAlignment(right_to_left_ ? Qt::RightToLeft : Qt::LeftToRight,
                                                    Qt::AlignLeft | Qt::AlignTop));
}

QRect ReadoutLabel::text_rect() const {
    // The room for the text: the contents, less the indent on the side the
    // text starts and at the top (QLabelPrivate::documentRect), and the room
    // kept on the side it ends.
    QRect rect = contentsRect();
    if (right_to_left_) {
        rect.setRight(rect.right() - indent_);
        rect.setLeft(rect.left() + trailing_);
    } else {
        rect.setLeft(rect.left() + indent_);
        rect.setRight(rect.right() - trailing_);
    }
    rect.setTop(rect.top() + indent_);
    return rect;
}

void ReadoutLabel::remeasure() {
    text_size_.reset();
    sized_for_ = {0, 0};
    updateGeometry();
    update();
}

// --------------------------------------------------------------- QuietTabWidget

void keep_focus_on_tab_click(QTabWidget* tabs) {
    // Before the tab changes (tabBarClicked comes first): a focus in the page
    // shown now is cleared, so the page coming has none to take over.
    QObject::connect(tabs->tabBar(), &QTabBar::tabBarClicked, tabs, [tabs] {
        QWidget* focus = QApplication::focusWidget();
        if (focus != nullptr && tabs->currentWidget() != nullptr && tabs->currentWidget()->isAncestorOf(focus)) {
            focus->clearFocus();
        }
    });
}

QuietTabWidget::QuietTabWidget(QWidget* parent) : QTabWidget(parent) {
    keep_focus_on_tab_click(this);
#ifdef Q_OS_MACOS
    // Control+Tab reaches the window, not the focused control, and only as
    // a ShortcutOverride (no KeyPress follows): seen before either takes it.
    qApp->installEventFilter(this);
#endif
}

bool QuietTabWidget::event(QEvent* event) {
    if (event->type() == QEvent::LayoutRequest) {
        const auto corner_size = [this](Qt::Corner corner) -> std::optional<QSize> {
            if (QWidget* widget = cornerWidget(corner)) {
                return widget->sizeHint();
            }
            return std::nullopt;
        };
        const Shape shape{tabBar()->sizeHint(), sizeHint(), minimumSizeHint(), corner_size(Qt::TopLeftCorner),
                          corner_size(Qt::TopRightCorner)};
        if (laid_out_for_ == shape) {
            return true;
        }
        laid_out_for_ = shape;
    }
    return QTabWidget::event(event);
}

#ifdef Q_OS_MACOS
namespace {

// Whether `tabs` has a tab to go to: another one enabled and shown.
bool has_another_tab(const QuietTabWidget& tabs) {
    for (int index = 0; index < tabs.count(); ++index) {
        if (index != tabs.currentIndex() && tabs.isTabEnabled(index) && tabs.isTabVisible(index)) {
            return true;
        }
    }
    return false;
}

// The tabs Control+Tab switches, as Ctrl+Tab's on Windows: the innermost
// with a tab to go to holding the focus, in its own window (not the window
// a dialog was opened from); with the focus elsewhere in the active window
// (a click on the Mac focuses no button), the window's outermost. None while
// a popup is open (a combo box's list), as with Qt's own.
QuietTabWidget* tabs_for_keys() {
    QWidget* const window = QApplication::activeWindow();
    if (window == nullptr || QApplication::activePopupWidget() != nullptr) {
        return nullptr;
    }
    for (QWidget* widget = QApplication::focusWidget(); widget != nullptr && !widget->isWindow();
         widget = widget->parentWidget()) {
        auto* tabs = qobject_cast<QuietTabWidget*>(widget);
        if (tabs != nullptr && has_another_tab(*tabs)) {
            return tabs;
        }
    }
    for (QuietTabWidget* tabs : window->findChildren<QuietTabWidget*>()) {
        QWidget* above = tabs->parentWidget();
        while (above != nullptr && above != window && qobject_cast<QuietTabWidget*>(above) == nullptr) {
            above = above->parentWidget();
        }
        if (above == window && tabs->window() == window && tabs->isVisible() && has_another_tab(*tabs)) {
            return tabs;
        }
    }
    return nullptr;
}

}  // namespace

bool QuietTabWidget::eventFilter(QObject* watched, QEvent* event) {
    if (event->type() != QEvent::ShortcutOverride && event->type() != QEvent::KeyPress) {
        return QTabWidget::eventFilter(watched, event);
    }
    const auto* key = static_cast<QKeyEvent*>(event);
    const Qt::KeyboardModifiers held = key->modifiers();
    if ((key->key() != Qt::Key_Tab && key->key() != Qt::Key_Backtab) || !held.testFlag(Qt::MetaModifier) ||
        held.testAnyFlags(Qt::ControlModifier | Qt::AltModifier) || tabs_for_keys() != this) {
        return QTabWidget::eventFilter(watched, event);
    }
    // Switched at the override; a KeyPress after it (a test's) is only taken,
    // or it would move the focus as Tab.
    if (event->type() == QEvent::ShortcutOverride) {
        const bool back = key->key() == Qt::Key_Backtab || held.testFlag(Qt::ShiftModifier);
        const int tabs = count();
        for (int step = 1; step < tabs; ++step) {
            const int index = (currentIndex() + (back ? tabs - step : step)) % tabs;
            if (isTabEnabled(index) && isTabVisible(index)) {
                setCurrentIndex(index);
                break;
            }
        }
    }
    event->accept();
    return true;
}
#endif

// ---------------------------------------------------------------- ScrollingPage

ScrollingPage::ScrollingPage(QWidget* page) {
    setWidget(page);
    setWidgetResizable(true);
    setFrameShape(QFrame::NoFrame);
    // Not a stop of its own when Tab moves through the controls.
    setFocusPolicy(Qt::NoFocus);
    // Transparent, so the tab's own pane shows behind the page: setWidget
    // makes the page paint itself in the window colour.
    page->setAutoFillBackground(false);
    viewport()->setAutoFillBackground(false);
    connect(qApp, &QApplication::focusChanged, this, &ScrollingPage::show_focus);
}

void ScrollingPage::show_focus(QWidget* /*old*/, QWidget* now) {
    // Scrolls a control of the page that takes the focus into view. Qt does
    // this itself only when Tab moves between the page's own controls, not
    // when the focus comes in from the tab bar. The whole control is shown,
    // with no margin, so a click near an edge does not make the page jump.
    QWidget* page = widget();
    if (now != nullptr && page != nullptr && page->isAncestorOf(now)) {
        const QPoint centre = now->mapTo(page, now->rect().center());
        ensureVisible(centre.x(), centre.y(), now->width() / 2, now->height() / 2);
    }
}

void MarkedCellDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const {
    // A colour or a brush, as the table was given it; a clear one is none.
    const QVariant background = index.data(Qt::BackgroundRole);
    const QBrush brush = background.value<QBrush>();
    const QColor mark = background.typeId() == QMetaType::QColor ? background.value<QColor>()
                        : brush.style() == Qt::NoBrush             ? QColor(0, 0, 0, 0)
                                                                   : brush.color();
    if (!(option.state & QStyle::State_Selected) || mark.alpha() == 0) {
        QStyledItemDelegate::paint(painter, option, index);
        return;
    }
    // The cell as unselected (its own background and text colours), its mark
    // untouched, then the selection's colour as a band along its top and
    // bottom edges, which runs on into the row's other selected cells. A tint
    // over the whole cell turned green grey and red purple.
    QStyleOptionViewItem unselected(option);
    unselected.state &= ~QStyle::State_Selected;
    QStyledItemDelegate::paint(painter, unselected, index);
    // The selection's colour as the row's other selected cells show it: a
    // blank selected item painted by the table's own style over its
    // background, as the view paints a cell (the row's panel, then the
    // item), a few pixels of it. VML Modern's style sheet lays a
    // translucent accent tint in each, which over the mark itself would
    // blend with the green or red.
    QStyleOptionViewItem blank(option);
    blank.text.clear();
    blank.icon = QIcon();
    blank.features = QStyleOptionViewItem::None;
    blank.backgroundBrush = QBrush();
    blank.rect = QRect(0, 0, 8, 8);
    QImage probe(8, 8, QImage::Format_ARGB32_Premultiplied);
    probe.fill(option.palette.color(QPalette::Active, QPalette::Base));
    {
        QPainter probe_painter(&probe);
        const QStyle* style = option.widget != nullptr ? option.widget->style() : QApplication::style();
        style->drawPrimitive(QStyle::PE_PanelItemViewRow, &blank, &probe_painter, option.widget);
        style->drawControl(QStyle::CE_ItemViewItem, &blank, &probe_painter, option.widget);
    }
    const QColor band = probe.pixelColor(4, 4);
    const int thick = std::max(2, option.rect.height() / 8);
    painter->fillRect(QRect(option.rect.left(), option.rect.top(), option.rect.width(), thick), band);
    painter->fillRect(QRect(option.rect.left(), option.rect.bottom() - thick + 1, option.rect.width(), thick), band);
}

QFont symbol_font(const QFont& font) {
#ifdef _WIN32
    QFont symbols(QStringLiteral("Segoe UI Symbol"));
    if (font.pointSizeF() > 0) {
        symbols.setPointSizeF(font.pointSizeF());
    } else {
        symbols.setPixelSize(font.pixelSize());
    }
    symbols.setWeight(font.weight());
    return symbols;
#else
    return font;
#endif
}

namespace {

// The play triangle or the pause bars, `size` pixels square, in `colour`, at
// the screen's pixel ratio.
QPixmap media_symbol(bool pause, int size, qreal ratio, const QColor& colour) {
    QPixmap pixmap(QSize(size, size) * ratio);
    pixmap.setDevicePixelRatio(ratio);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    painter.setBrush(colour);
    const qreal s = size;
    if (pause) {
        const qreal bar = s * 0.3;  // two bars, a bar's width apart, as the glyph is drawn
        painter.drawRoundedRect(QRectF(s * 0.12, 0, bar, s), s * 0.06, s * 0.06);
        painter.drawRoundedRect(QRectF(s * 0.88 - bar, 0, bar, s), s * 0.06, s * 0.06);
    } else {
        QPainterPath triangle;
        triangle.moveTo(s * 0.1, 0);
        triangle.lineTo(s * 0.98, s * 0.5);
        triangle.lineTo(s * 0.1, s);
        triangle.closeSubpath();
        painter.drawPath(triangle);
    }
    return pixmap;
}

}  // namespace

PlayPauseButton::PlayPauseButton(QWidget* parent) : QPushButton(parent) { refresh(); }

void PlayPauseButton::set_playing(bool playing) {
    playing_ = playing;
    refresh();
}

void PlayPauseButton::changeEvent(QEvent* event) {
    QPushButton::changeEvent(event);
    // The colours come from the palette, which the theme and a style sheet
    // (a role's text colour) set; and the font sets the size.
    if (event->type() == QEvent::PaletteChange || event->type() == QEvent::StyleChange ||
        event->type() == QEvent::FontChange) {
        refresh();
    }
}

bool PlayPauseButton::event(QEvent* event) {
    // The symbol is drawn for one screen's scale: drawn again for a screen
    // of another (the window moved to it, or its scaling changed), else it
    // is blurred there.
    if (event->type() == QEvent::DevicePixelRatioChange) {
        refresh();
    }
    return QPushButton::event(event);
}

void PlayPauseButton::refresh() {
    setText(playing_ ? app::tr("Pause") : app::tr("Play"));
    // As tall as a capital letter of the text beside it, filled top to bottom.
    const int size = std::max(8, fontMetrics().capHeight() + 1);
    const qreal ratio = devicePixelRatioF();
    QIcon icon;
    icon.addPixmap(media_symbol(playing_, size, ratio, palette().color(QPalette::Active, QPalette::ButtonText)),
                   QIcon::Normal);
    icon.addPixmap(media_symbol(playing_, size, ratio, palette().color(QPalette::Disabled, QPalette::ButtonText)),
                   QIcon::Disabled);
    setIcon(icon);
    setIconSize(QSize(size, size));
}

namespace {

bool is_value_box(const QObject* object) {
    return qobject_cast<const QAbstractSpinBox*>(object) != nullptr || qobject_cast<const QComboBox*>(object) != nullptr;
}

}  // namespace

namespace {

// The wheel passing over a box does not give it the focus either
// (Qt::WheelFocus, theirs and their line edits' by default), where the arrow
// keys pressed next would change it. Qt gives that focus before any event
// filter sees the wheel event, so it is taken off beforehand.
void no_wheel_focus(QWidget* widget) {
    if (widget->focusPolicy() == Qt::WheelFocus &&
        (is_value_box(widget) || is_value_box(widget->parentWidget()))) {
        widget->setFocusPolicy(Qt::StrongFocus);
    }
}

}  // namespace

WheelPastValueBoxes::WheelPastValueBoxes(QWidget* page) : QObject(page), page_(page) {
    for (QWidget* widget : page->findChildren<QWidget*>()) {  // the boxes already made
        no_wheel_focus(widget);
    }
    QCoreApplication::instance()->installEventFilter(this);  // Qt takes it off when it is deleted
}

bool WheelPastValueBoxes::eventFilter(QObject* watched, QEvent* event) {
    // The page's ancestry is looked up only for boxes.
    if (event->type() == QEvent::Wheel && is_value_box(watched) &&
        page_->isAncestorOf(static_cast<QWidget*>(watched))) {
        event->ignore();
        return true;
    }
    if (event->type() == QEvent::Polish) {  // a box added later, before it is first shown
        auto* widget = static_cast<QWidget*>(watched);  // only widgets are polished
        if (page_->isAncestorOf(widget)) {
            no_wheel_focus(widget);
        }
    }
    return false;
}

// A plain line of a tooltip longer than this many characters is wrapped.
constexpr int tool_tip_line = 70;

void fit_table_height(QTableWidget* table, int rows) {
    // The header's hint, not its height: before the table is first shown its
    // header has not been laid out yet.
    const QHeaderView* header = table->horizontalHeader();
    int height = (header->isHidden() ? 0 : header->sizeHint().height()) + 2 * table->frameWidth();
    for (int row = 0; row < std::min(rows, table->rowCount()); ++row) {
        height += table->rowHeight(row);
    }
    table->setFixedHeight(height);
}

QString wrapped_tool_tip(const QString& text) {
    if (text.size() <= tool_tip_line || Qt::mightBeRichText(text)) {
        return text;
    }
    return Qt::convertFromPlainText(text, Qt::WhiteSpaceNormal);
}

WrapLongToolTips::WrapLongToolTips(QObject* parent) : QObject(parent) {
    QCoreApplication::instance()->installEventFilter(this);  // Qt takes it off when it is deleted
}

bool WrapLongToolTips::eventFilter(QObject* watched, QEvent* event) {
    if (event->type() == QEvent::ToolTipChange) {
        // A widget's tooltip, as it is set: set again wrapped (which comes
        // back here as rich text, and is left).
        if (auto* widget = qobject_cast<QWidget*>(watched)) {
            const QString tip = widget->toolTip();
            const QString wrapped = wrapped_tool_tip(tip);
            if (wrapped != tip) {
                widget->setToolTip(wrapped);
            }
        }
        return false;
    }
    if (event->type() == QEvent::ToolTip) {
        // An item's, as it is shown: the view's own showing of it, wrapped.
        auto* viewport = qobject_cast<QWidget*>(watched);
        auto* view = viewport == nullptr ? nullptr : qobject_cast<QAbstractItemView*>(viewport->parentWidget());
        if (view == nullptr || view->viewport() != viewport) {
            return false;
        }
        const auto* help = static_cast<QHelpEvent*>(event);
        const QModelIndex index = view->indexAt(help->pos());
        const QString tip = index.isValid() ? index.data(Qt::ToolTipRole).toString() : QString();
        const QString wrapped = wrapped_tool_tip(tip);
        if (wrapped == tip) {
            return false;  // the view shows it as it would
        }
        QToolTip::showText(help->globalPos(), wrapped, viewport, view->visualRect(index));
        return true;
    }
    return false;
}

}  // namespace vml::app
