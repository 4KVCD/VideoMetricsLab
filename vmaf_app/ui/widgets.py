"""Reusable Qt widgets with no knowledge of this app's data or columns.

Both of these exist because Qt's stock behaviour is subtly wrong for the
distorted-files table; keeping them here rather than in main_window.py keeps
that module about *this app's* window and these about Qt.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtWidgets import QHeaderView, QStyle, QStyleOptionButton, QTableWidget

_INDICATOR_MARGIN = 4
_MIN_FILL_WIDTH = 60


class CheckableHeaderView(QHeaderView):
    """A horizontal header where chosen sections carry a checkbox, the way
    FFMetrics' PSNR/SSIM/XPSNR columns do -- so which metrics get computed is
    set right above the column the results land in, instead of hidden away in
    a separate options panel.

    `checkable` maps section index -> initial checked state; sections absent
    from it are drawn and behave as ordinary headers.
    """

    sectionToggled = Signal(int, bool)

    def __init__(self, checkable: dict[int, bool], parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._checked = dict(checkable)
        self.setSectionsClickable(True)

    def is_checked(self, section: int) -> bool:
        return self._checked.get(section, False)

    def set_checked(self, section: int, value: bool) -> None:
        """Sets a checkbox without emitting sectionToggled -- for syncing the
        header to state that changed elsewhere, which must not loop back."""
        if section in self._checked and self._checked[section] != value:
            self._checked[section] = value
            self.updateSection(section)

    def _indicator_rect(self, rect) -> QRect:
        size = self.style().pixelMetric(QStyle.PM_IndicatorWidth, None, self)
        return QRect(
            rect.x() + _INDICATOR_MARGIN, rect.y() + (rect.height() - size) // 2, size, size,
        )

    def paintSection(self, painter, rect, logicalIndex: int) -> None:
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

    def section_indicator_rect(self, section: int) -> QRect:
        """Where this section's checkbox is drawn, in viewport coordinates.

        Same geometry paintSection uses, so what is clickable is exactly
        what is visible.
        """
        return self._indicator_rect(QRect(
            self.sectionViewportPosition(section), 0,
            self.sectionSize(section), self.height(),
        ))

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()
        index = self.logicalIndexAt(pos)
        if index in self._checked and self.section_indicator_rect(index).contains(pos):
            self._checked[index] = not self._checked[index]
            self.updateSection(index)
            self.sectionToggled.emit(index, self._checked[index])
            return
        # Anywhere else in the section is an ordinary header click: dragging
        # the boundary to resize the column, or sorting. Toggling on the
        # whole section meant the column could not be resized without also
        # turning the metric on and off, and made every near-miss of the
        # checkbox silently change what the next run would compute.
        super().mousePressEvent(event)


class FillColumnTable(QTableWidget):
    """A QTableWidget where one column (`fill_column`) always expands to
    fill whatever space is left over after the others, while STILL being
    drag-resizable by the user -- Qt's own Stretch resize mode fills leftover
    space too, but disables dragging for that column entirely, which doesn't
    work when that's the one column users most want to resize (Path).
    """

    def __init__(self, rows: int, cols: int, fill_column: int, other_columns: list[int], parent=None):
        super().__init__(rows, cols, parent)
        self._fill_column = fill_column
        self._other_columns = other_columns
        self._recalculating = False
        # Remembers a manual drag of the fill column past its "natural fill"
        # width. Without this, any later resizeEvent (e.g. the window itself
        # being resized) called _recalculate_fill_column() unconditionally
        # and clamped the column straight back down to the leftover-space
        # width, silently undoing the drag instead of letting it grow past
        # the available room and produce a horizontal scrollbar.
        self._fill_column_user_width: int | None = None
        self.horizontalHeader().sectionResized.connect(self._on_section_resized)

    def setHorizontalHeader(self, header) -> None:
        # Swapping in a different header (e.g. the checkable metric header)
        # drops the connection __init__ made to the *old* one, which
        # silently disables the fill-column behaviour entirely -- the column
        # simply stops responding to any other column being resized.
        super().setHorizontalHeader(header)
        header.sectionResized.connect(self._on_section_resized)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._recalculate_fill_column()

    def _on_section_resized(self, logical_index: int, old_size: int, new_size: int) -> None:
        if self._recalculating:
            return
        if logical_index == self._fill_column:
            self._fill_column_user_width = new_size  # a manual drag of the fill column's own edge
            return
        self._recalculate_fill_column()

    def _recalculate_fill_column(self) -> None:
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
