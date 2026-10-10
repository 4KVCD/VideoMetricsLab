"""Reusable Qt widgets that know nothing of the app's data or columns.

Each exists because Qt's own widget is slightly wrong for how the app uses
it. Keeping them here rather than in main_window.py keeps that module about
the app's window, and these about Qt.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QTableWidget,
    QTabWidget,
)

_INDICATOR_MARGIN = 4
_MIN_FILL_WIDTH = 60


class CheckableHeaderView(QHeaderView):
    """A horizontal header where chosen sections have a checkbox, as
    FFMetrics' PSNR, SSIM and XPSNR columns do, so that which metrics are
    calculated is set right above the column the results go in, rather than
    in a separate options panel.

    `checkable` maps a section's index to its first checked state. Sections
    not in it are drawn and behave as ordinary headers.
    """

    sectionToggled = Signal(int, bool)

    def __init__(self, checkable: dict[int, bool], parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._checked = dict(checkable)
        self.setSectionsClickable(True)

    def is_checked(self, section: int) -> bool:
        """Whether a section's checkbox is checked."""
        return self._checked.get(section, False)

    def set_checked(self, section: int, value: bool) -> None:
        """Sets a checkbox without sending sectionToggled, for bringing the
        header in line with a change made elsewhere, which must not loop
        back."""
        if section in self._checked and self._checked[section] != value:
            self._checked[section] = value
            self.updateSection(section)

    def _indicator_rect(self, rect) -> QRect:
        """Where the checkbox goes in a section's rectangle: at its left, centred up and down."""
        size = self.style().pixelMetric(QStyle.PM_IndicatorWidth, None, self)
        return QRect(
            rect.x() + _INDICATOR_MARGIN, rect.y() + (rect.height() - size) // 2, size, size,
        )

    def paintSection(self, painter, rect, logicalIndex: int) -> None:
        """Draws the section as usual, and its checkbox over it."""
        painter.save()
        super().paintSection(painter, rect, logicalIndex)
        painter.restore()
        if logicalIndex not in self._checked:
            return
        opt = QStyleOptionButton()
        opt.rect = self._indicator_rect(rect)
        opt.state = QStyle.State_Enabled | (
            QStyle.State_On if self._checked[logicalIndex] else QStyle.State_Off
        )
        self.style().drawPrimitive(QStyle.PE_IndicatorCheckBox, opt, painter, self)

    def sectionSizeFromContents(self, logicalIndex):
        """The section's size, with room for its checkbox."""
        size = super().sectionSizeFromContents(logicalIndex)
        if logicalIndex in self._checked:
            # Qt's size hint knows only the centred text, not the checkbox
            # painted over its left edge. Room is added on both sides, so that
            # at its default width the label cannot overlap the checkbox.
            indicator = self.style().pixelMetric(QStyle.PM_IndicatorWidth, None, self)
            size.setWidth(size.width() + 2 * (indicator + _INDICATOR_MARGIN))
        return size

    def section_indicator_rect(self, section: int) -> QRect:
        """Where this section's checkbox is drawn, in viewport coordinates.

        It is the same geometry paintSection uses, so what can be clicked is
        exactly what is seen.
        """
        return self._indicator_rect(QRect(
            self.sectionViewportPosition(section), 0,
            self.sectionSize(section), self.height(),
        ))

    def mousePressEvent(self, event) -> None:
        """Toggles a checkbox clicked on; any other click is an ordinary header click."""
        pos = event.position().toPoint()
        index = self.logicalIndexAt(pos)
        if index in self._checked and self.section_indicator_rect(index).contains(pos):
            self._checked[index] = not self._checked[index]
            self.updateSection(index)
            self.sectionToggled.emit(index, self._checked[index])
            return
        # Anywhere else in the section is an ordinary header click: dragging
        # the edge to resize the column, or sorting. Don't toggle on a click
        # anywhere in the section: the column could then not be resized
        # without turning the metric on or off, and every near miss of the
        # checkbox would quietly change what the next run calculates.
        super().mousePressEvent(event)


class FillColumnTable(QTableWidget):
    """A QTableWidget where one column (`fill_column`) always grows to fill
    the space the others leave, while the user can still resize it by
    dragging. Qt's own Stretch mode fills the space too, but stops the
    column being dragged at all, which does not suit the one column users
    most want to resize (Path).
    """

    #: Files dropped on the table from outside it (a file manager): their paths.
    filesDropped = Signal(list)
    #: Rows dragged within the table: the rows moved, in order, and the row
    #: they go before (rowCount() for the end). The table moves nothing
    #: itself: what a row is lives with its owner.
    rowsMoved = Signal(list, int)

    def __init__(self, rows: int, cols: int, fill_column: int, other_columns: list[int], parent=None):
        super().__init__(rows, cols, parent)
        # Drops are taken by dragEnterEvent and friends below, not by Qt's
        # item-view handling, which would write the dragged cells over the
        # ones dropped on.
        self.drops_allowed = True
        self._drop_line: int | None = None  # the row a dragged row would go before, while dragging
        # What the drag over the table carries, found once when it enters. A
        # drag move comes with every mouse move, and finding it again then
        # would look at every dragged file on disk each time (slow on a
        # network drive).
        self._drag_kind: str | None = None
        self._fill_column = fill_column
        self._other_columns = other_columns
        self._recalculating = False
        # Remembers a manual drag of the fill column past the width that
        # just fills the space. Without it, any later resize (the window's,
        # say) would set the column straight back to that width, quietly
        # undoing the drag instead of letting it grow past the room there is,
        # with a horizontal scrollbar.
        self._fill_column_user_width: int | None = None
        self.horizontalHeader().sectionResized.connect(self._on_section_resized)

    def setHorizontalHeader(self, header) -> None:
        """Puts in another header, and connects it as __init__ connected the
        first."""
        # A new header (the checkable metric header, say) does not have the
        # connection __init__ made to the old one. Without connecting it
        # again, the fill column would stop following the other columns'
        # resizing.
        super().setHorizontalHeader(header)
        header.sectionResized.connect(self._on_section_resized)

    def resizeEvent(self, event) -> None:
        """Fits the fill column to the new width."""
        super().resizeEvent(event)
        self._recalculate_fill_column()

    # ------------------------------------------------------------ drag and drop
    @staticmethod
    def dropped_files(mime) -> list[str]:
        """The local files (not folders) a drag carries."""
        if not mime.hasUrls():
            return []
        paths = (url.toLocalFile() for url in mime.urls() if url.isLocalFile())
        return [path for path in paths if path and Path(path).is_file()]

    def _drop_kind(self, event) -> str | None:
        """"rows" for this table's own rows, "files" for files from outside, else None."""
        if not self.drops_allowed:
            return None
        if event.source() is self:
            return "rows" if self.selectionModel().selectedRows() else None
        return "files" if self.dropped_files(event.mimeData()) else None

    def _insertion_row(self, event) -> int:
        """The row a drop at the event's position goes before: the nearer
        boundary of the row under it, or the end below the last."""
        y = event.position().toPoint().y()
        row = self.rowAt(y)
        if row < 0:
            return self.rowCount() if y > 0 else 0
        middle = self.rowViewportPosition(row) + self.rowHeight(row) // 2
        return row if y < middle else row + 1

    def _set_drop_line(self, row: int | None) -> None:
        """Shows the line where dragged rows would go (None hides it)."""
        if row != self._drop_line:
            self._drop_line = row
            self.viewport().update()

    def dragEnterEvent(self, event) -> None:
        """Accepts a drag of the table's own rows or of files, and notes which."""
        kind = self._drag_kind = self._drop_kind(event)
        if kind is None:
            event.ignore()
            return
        event.setDropAction(Qt.MoveAction if kind == "rows" else Qt.CopyAction)
        event.accept()

    def dragMoveEvent(self, event) -> None:
        """Shows where dragged rows would go."""
        kind = self._drag_kind if self.drops_allowed else None
        if kind is None:
            self._set_drop_line(None)
            event.ignore()
            return
        self._set_drop_line(self._insertion_row(event) if kind == "rows" else None)
        event.setDropAction(Qt.MoveAction if kind == "rows" else Qt.CopyAction)
        event.accept()

    def dragLeaveEvent(self, event) -> None:
        """Hides the drop line."""
        self._drag_kind = None
        self._set_drop_line(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        """Reports dropped rows (rowsMoved) or files (filesDropped)."""
        self._set_drop_line(None)
        self._drag_kind = None
        kind = self._drop_kind(event)
        if kind is None:
            event.ignore()
            return
        if kind == "rows":
            rows = sorted(index.row() for index in self.selectionModel().selectedRows())
            target = self._insertion_row(event)
            # Reported as a copy (a drag from this table offers both),
            # because on a move, Qt's drag would delete the dragged rows
            # itself. A drag that offers only a move is turned down.
            event.setDropAction(Qt.CopyAction)
            if event.dropAction() != Qt.CopyAction:
                event.ignore()
                return
            event.accept()
            self.rowsMoved.emit(rows, target)
            return
        files = self.dropped_files(event.mimeData())
        event.setDropAction(Qt.CopyAction)
        event.accept()
        self.filesDropped.emit(files)

    def paintEvent(self, event) -> None:
        """Draws the table, and the drop line while rows are dragged."""
        super().paintEvent(event)
        if self._drop_line is None:
            return
        if self._drop_line < self.rowCount():
            y = self.rowViewportPosition(self._drop_line)
        elif self.rowCount():
            last = self.rowCount() - 1
            y = self.rowViewportPosition(last) + self.rowHeight(last) - 1
        else:
            y = 0
        painter = QPainter(self.viewport())
        painter.setPen(QPen(self.palette().highlight().color(), 2))
        painter.drawLine(0, y, self.viewport().width(), y)
        painter.end()

    def _on_section_resized(self, logical_index: int, old_size: int, new_size: int) -> None:
        """Remembers a drag of the fill column, or fits it after another column was resized."""
        if self._recalculating:
            return
        if logical_index == self._fill_column:
            self._fill_column_user_width = new_size  # a manual drag of the fill column's own edge
            return
        self._recalculate_fill_column()

    def _recalculate_fill_column(self) -> None:
        """Sets the fill column to the space the others leave, or to the width the user dragged it to if that is wider."""
        if self._recalculating:
            return
        other_total = sum(self.columnWidth(c) for c in self._other_columns)
        natural = max(_MIN_FILL_WIDTH, self.viewport().width() - other_total)
        user_width = self._fill_column_user_width
        target = natural if user_width is None else max(natural, user_width)
        if target == self.columnWidth(self._fill_column):
            return
        self._recalculating = True
        try:
            self.setColumnWidth(self._fill_column, target)
        finally:
            self._recalculating = False


class ElidedLabel(QLabel):
    """One line of text cut to the width it is given, ending in "\u2026",
    with the whole text in the tooltip when it is cut.

    A plain QLabel asks for its text's full width, so one long line (a run
    line with a long file name, every task's figures and the decode plan)
    would set the window's minimum width, forcing it wider than the screen
    on a smaller display. This one asks for no width at all.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self._tooltip = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def set_text(self, text: str, tooltip: str = "") -> None:
        """Sets the text, and a tooltip to show with it."""
        self._full, self._tooltip = text, tooltip
        self._fit()

    def full_text(self) -> str:
        """The whole text, before cutting."""
        return self._full

    def clear(self) -> None:
        """Clears the text and the tooltip."""
        self._full = self._tooltip = ""
        super().clear()
        self.setToolTip("")

    def minimumSizeHint(self) -> QSize:
        """Asks for no width at all, and the usual height."""
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        """Cuts the text again for the new width."""
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event) -> None:
        """Cuts the text to the width it is shown at."""
        super().showEvent(event)
        self._fit()

    def _fit(self) -> None:
        """Shows the text cut to the width there is, with the whole text in the tooltip when it is cut."""
        width = self.contentsRect().width()
        # Not cut until shown and laid out: before that the width is a
        # placeholder, not the room the line will have.
        shown = self._full if not self.isVisible() or width <= 0 else \
            self.fontMetrics().elidedText(self._full, Qt.ElideRight, width)
        super().setText(shown)
        cut = shown != self._full
        self.setToolTip(self._full + (f"\n\n{self._tooltip}" if self._tooltip else "") if cut else self._tooltip)


class QuietTabWidget(QTabWidget):
    """A QTabWidget that lays itself out again, and repaints, only when
    that can change something.

    Qt's QTabWidget lays itself out again at every LayoutRequest, and
    repaints the whole of itself with it (QTabWidget::setUpLayout ends in
    update()). A label on one of its pages sends one up whenever its text
    changes: setText invalidates the layouts above it, and each, once
    redone, passes the request on whether its size hints changed or not.
    So a frame's time on Video Compare would repaint every control on the
    tab at every frame of playback, 112 times a second, and a run's
    once-a-second lines would repaint the whole Videos tab twice a second.

    Its own layout depends on its size (resizeEvent lays it out anyway) and
    on its tab bar's and corner widgets' sizes, and what is above it depends
    on its size hints. A request that changes none of them is answered here,
    without the layout and repaint."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._laid_out_for = None

    def event(self, event) -> bool:
        """Answers a layout request that would change nothing, without laying out and repainting."""
        if event.type() == QEvent.LayoutRequest:
            corners = (self.cornerWidget(Qt.TopLeftCorner), self.cornerWidget(Qt.TopRightCorner))
            shape = (self.tabBar().sizeHint(), self.sizeHint(), self.minimumSizeHint(),
                     *(None if corner is None else corner.sizeHint() for corner in corners))
            if shape == self._laid_out_for:
                return True
            self._laid_out_for = shape
        return super().event(event)

