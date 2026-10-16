#include "widgets.h"

#include <QApplication>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QFrame>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QShowEvent>
#include <QSizePolicy>
#include <QStyle>
#include <QStyleOption>
#include <QStyleOptionButton>
#include <QTabBar>
#include <QUrl>

#include <algorithm>
#include <numeric>
#include <utility>

namespace vml::app {

namespace {

constexpr int indicator_margin = 4;
constexpr int min_fill_width = 60;

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

// --------------------------------------------------------------- FillColumnTable

FillColumnTable::FillColumnTable(int rows, int columns, int fill_column, std::vector<int> other_columns,
                                 QWidget* parent)
    : QTableWidget(rows, columns, parent), fill_column_(fill_column), other_columns_(std::move(other_columns)) {
    connect(horizontalHeader(), &QHeaderView::sectionResized, this, &FillColumnTable::on_section_resized);
}

void FillColumnTable::setHorizontalHeader(QHeaderView* header) {
    // A new header does not have the connection the constructor made to the
    // old one; without it, the fill column would stop following the other
    // columns' resizing.
    QTableWidget::setHorizontalHeader(header);
    connect(header, &QHeaderView::sectionResized, this, &FillColumnTable::on_section_resized);
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
    emit filesDropped(files);
}

void FillColumnTable::paintEvent(QPaintEvent* event) {
    QTableWidget::paintEvent(event);
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
    QPainter painter(viewport());
    painter.setPen(QPen(palette().highlight().color(), 2));
    painter.drawLine(0, y, viewport()->width(), y);
}

void FillColumnTable::on_section_resized(int logical_index, int /*old_size*/, int new_size) {
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
    const int natural = std::max(min_fill_width, viewport()->width() - others);
    const int target = fill_column_user_width_ ? std::max(natural, *fill_column_user_width_) : natural;
    if (target == columnWidth(fill_column_)) {
        return;
    }
    recalculating_ = true;
    setColumnWidth(fill_column_, target);
    recalculating_ = false;
}

// ------------------------------------------------------------------ ElidedLabel

ElidedLabel::ElidedLabel(QWidget* parent) : QLabel(parent) {
    setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
}

void ElidedLabel::set_text(const QString& text, const QString& tooltip) {
    full_ = text;
    tooltip_ = tooltip;
    fit();
}

void ElidedLabel::clear_text() {
    full_.clear();
    tooltip_.clear();
    QLabel::clear();
    setToolTip(QString());
}

QSize ElidedLabel::minimumSizeHint() const { return {0, QLabel::minimumSizeHint().height()}; }

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
    const QString shown =
        !isVisible() || width <= 0 ? full_ : fontMetrics().elidedText(full_, Qt::ElideRight, width);
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

QSize ReadoutLabel::minimumSizeHint() const { return sizeHint(); }  // as QLabel's, for text that does not wrap

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

QuietTabWidget::QuietTabWidget(QWidget* parent) : QTabWidget(parent) {}

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

}  // namespace vml::app
