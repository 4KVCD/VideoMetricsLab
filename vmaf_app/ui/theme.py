"""The window's colours, for a light or a dark theme.

Qt's windows11 style draws the window in Windows' light or dark colours (or
the ones Settings > Window > Theme asks for: apply_theme), and every widget
that takes its colours from the palette follows. What the app colours itself
-- a muted hint, a failed score, a chart -- is named here once, with a value
for each theme, rather than as a fixed colour that is right on one theme
only: dark grey hints and black scores on a dark window, white charts in it
(issue #5).

`color(name)` is the colour for the theme in use. A widget styled with
`style(widget, "color: {muted};")` is styled again when the theme changes
(`refresh`), as the palette's own colours are. Painted things (the charts,
the table's cells) ask for their colours when they draw.

Files the app writes -- an exported graph -- keep the light colours whatever
the window shows: they are read elsewhere, and on paper.
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication

_log = logging.getLogger(__name__)

#: Settings.theme's values: "" follows Windows.
THEMES = ("", "light", "dark")

#: name -> (light, dark)
_COLOURS: dict[str, tuple[str, str]] = {
    "muted": ("#666666", "#a6a6a6"),        # hints and secondary text
    "faint": ("#888888", "#8c8c8c"),        # n/a cells
    "dimmed": ("#999999", "#7a7a7a"),       # a row's cells while it is not the one shown
    "run_line": ("#444444", "#c8c8c8"),     # the per-video progress lines
    "failed": ("#a03030", "#ff8a80"),
    "stale": ("#8a6d00", "#e6c35c"),
    "good": ("#207020", "#81c784"),
    "error_border": ("#cc3333", "#ef6a6a"),
    "banner_background": ("#fff3cd", "#4d4012"),
    "banner_border": ("#ffe08a", "#8f7623"),
    "banner_text": ("#3d3000", "#fff0c2"),
    "link": ("#2a5db0", "#8ab4f8"),         # a graph table heading that sorts
    "link_selected": ("#12327a", "#c2d7ff"),
    # Behind the graph statistics' means, and the selected metric's.
    "mean_tint": ("#f4f6fa", "#262a31"),
    "mean_selected": ("#cfe0fa", "#27406a"),
}


@dataclass(frozen=True)
class ChartColours:
    background: QColor
    text: QColor
    axis: QColor
    grid: QColor
    crosshair: QColor


_CHART = {
    False: ChartColours(QColor("white"), QColor(40, 40, 40), QColor(90, 90, 90), QColor(0, 0, 0, 40),
                        QColor(120, 120, 120)),
    True: ChartColours(QColor(30, 30, 30), QColor(220, 220, 220), QColor(150, 150, 150),
                       QColor(255, 255, 255, 38), QColor(170, 170, 170)),
}


def is_dark() -> bool:
    """Whether the window is drawn in dark colours now."""
    app = QGuiApplication.instance()
    if app is None:
        return False
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return app.palette().window().color().lightness() < 128


def color(name: str, dark: bool | None = None) -> QColor:
    light_value, dark_value = _COLOURS[name]
    return QColor(dark_value if (is_dark() if dark is None else dark) else light_value)


def readable(series_colour: str) -> QColor:
    """A series' colour as text: lighter on a dark window, where the
    palette's darker blues and reds read poorly; as it is on a light one."""
    colour = QColor(series_colour)
    return colour.lighter(150) if is_dark() else colour


def chart_colours(dark: bool | None = None) -> ChartColours:
    return _CHART[is_dark() if dark is None else dark]


def _values() -> dict[str, str]:
    dark = is_dark()
    return {name: (pair[1] if dark else pair[0]) for name, pair in _COLOURS.items()}


_styled: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def style(widget, template: str) -> None:
    """widget.setStyleSheet(template) with {name}s filled in from this
    theme's colours, done again by refresh() when the theme changes."""
    _styled[widget] = template
    widget.setStyleSheet(template.format(**_values()))


def foreground(item, colour: QColor | None) -> None:
    """A table cell's text colour; None for the table's own, which follows
    the palette as the theme changes. A colour copied from the palette
    would stay that theme's: white text on a light table after a switch."""
    item.setData(Qt.ForegroundRole, colour)


def refresh() -> None:
    """Restyles every widget style() styled, for the theme now in use, and
    applies every other style sheet again: Qt resolves a sheet's palette()
    colours when the sheet is applied, so a sheet kept from before a switch
    keeps the old theme's (the graph's metric tabs stayed dark)."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    widgets = QApplication.allWidgets()
    # A palette left from before the switch follows the application's
    # again: a widget whose child has a style sheet can miss Qt's update
    # (the graph's metric tabs' QTabWidget kept the dark one, without a
    # palette set on it, and the pages under it stayed dark). The sheets
    # below set what they need anew. The app sets no palette of its own.
    window = QApplication.palette().window().color()
    for widget in widgets:
        if widget.testAttribute(Qt.WA_SetPalette) or widget.palette().window().color() != window:
            widget.setPalette(QPalette())
    values = _values()
    for widget, template in list(_styled.items()):
        try:
            widget.setStyleSheet(template.format(**values))
        except RuntimeError:  # its Qt object is gone
            _styled.pop(widget, None)
    for widget in widgets:
        if widget not in _styled and widget.styleSheet():
            widget.setStyleSheet(widget.styleSheet())


def apply_theme(theme: str) -> None:
    """Asks Qt for Settings.theme's colours: "light", "dark", or "" for
    Windows' own. Takes effect at once; QStyleHints.colorSchemeChanged
    follows, which the window answers by redrawing what it colours itself."""
    app = QGuiApplication.instance()
    if app is None:
        return
    scheme = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}.get(theme, Qt.ColorScheme.Unknown)
    app.styleHints().setColorScheme(scheme)
