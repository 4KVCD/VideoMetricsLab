"""Session-only video colors shared by metric and bitrate plots.

This is presentation state, not result/cache metadata. Keep reservations
after a row is removed so re-adding its video restores the same color.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QDialogButtonBox, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from vmaf_app.i18n import tr

PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)


class VideoColors(QObject):
    """Window-owned presentation state, changed only on the UI thread."""

    changed = Signal(str, str)  # normalized path, color

    def __init__(self) -> None:
        super().__init__()
        self._colors: dict[str, str] = {}
        self._defaults: dict[str, str] = {}

    @staticmethod
    def path_key(path: Path) -> str:
        try:
            full_path = str(Path(path).resolve())
        except OSError:
            full_path = os.path.abspath(path)
        # Windows paths are case-insensitive; Linux paths need not be.
        return os.path.normcase(full_path)

    def for_path(self, path: Path) -> str:
        key = self.path_key(path)
        if key not in self._colors:
            self._colors[key] = PALETTE[len(self._colors) % len(PALETTE)]
            self._defaults[key] = self._colors[key]
        return self._colors[key]

    def set_color(self, path: Path, color: QColor) -> None:
        """Notify both tabs only when an accepted, opaque color changes."""
        if not color.isValid():
            return
        current = self.for_path(path)
        value = color.name()
        if current.lower() != value.lower():
            key = self.path_key(path)
            self._colors[key] = value
            self.changed.emit(key, value)

    def choose(self, path: Path, parent) -> None:
        """Preview the selection in the dialog; apply only after OK."""
        dialog = QColorDialog(QColor(self.for_path(path)), parent)
        dialog.setWindowTitle(tr("Choose video color"))
        dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        buttons = dialog.findChild(QDialogButtonBox)
        if buttons is not None:
            reset = buttons.addButton(tr("Reset color"), QDialogButtonBox.ResetRole)
            reset.clicked.connect(lambda: dialog.setCurrentColor(QColor(self._defaults[self.path_key(path)])))
        if dialog.exec() == QColorDialog.Accepted:
            self.set_color(path, dialog.currentColor())
        dialog.deleteLater()


class ColorSwatchDelegate(QStyledItemDelegate):
    """Only the painted color square opens the picker, not the checkbox/name.

    Qt supplies the decoration rectangle, so hit testing follows the current
    style, display scaling, scrolling and right-to-left layout.
    """

    clicked = Signal(int)

    def swatch_rect(self, option, index):
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        return styled.widget.style().subElementRect(QStyle.SE_ItemViewItemDecoration, styled, styled.widget)

    def editorEvent(self, event, model, option, index):
        if (event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick)
                and event.button() == Qt.LeftButton
                and self.swatch_rect(option, index).contains(event.position().toPoint())):
            if event.type() == QEvent.MouseButtonRelease:
                self.clicked.emit(index.row())
            return True
        return super().editorEvent(event, model, option, index)
