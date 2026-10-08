"""Issue #5: the window in dark colours -- what the app colours itself
follows the theme, and Settings can choose light or dark."""
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core.settings import Settings
from vmaf_app.ui import theme
from vmaf_app.ui.main_window import COL_VMAF, MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dark(monkeypatch):
    """The theme as the window would see it, switchable: tests set it
    rather than the platform's colour scheme, which they would leak."""
    state = {"dark": True}
    monkeypatch.setattr(theme, "is_dark", lambda: state["dark"])
    return state


def test_each_colour_has_a_light_and_a_dark_value(dark):
    muted_dark = theme.color("muted")
    dark["dark"] = False
    assert theme.color("muted") != muted_dark
    assert theme.color("muted").lightness() < muted_dark.lightness()  # darker text on a light window


def test_the_theme_setting_is_applied_at_once_and_kept(qapp, monkeypatch):
    applied = []
    monkeypatch.setattr(theme, "apply_theme", applied.append)
    win = MainWindow()
    try:
        assert win.settings_theme.currentData() == ""  # same as Windows by default
        win.settings_theme.setCurrentIndex(win.settings_theme.findData("dark"))
        assert applied == ["dark"]
        assert Settings.load().theme == "dark"
        win.settings_theme.setCurrentIndex(win.settings_theme.findData(""))
        assert applied == ["dark", ""]
    finally:
        win.close()


def test_a_switch_recolours_every_coloured_cell_and_leaves_the_rest(qapp, dark):
    win = MainWindow()
    try:
        row = win._add_table_row(Path("a.mkv"))
        win._on_probed(Path("a.mkv"), None, "could not read it")  # "Probe failed", in the failed colour
        from vmaf_app.ui.main_window import COL_INFO
        info = win.distorted_table.item(row, COL_INFO)
        assert info.foreground().color() == theme.color("failed")
        plain = win.distorted_table.item(row, COL_VMAF)
        dark["dark"] = False
        win._apply_theme_colours()
        assert info.foreground().color() == theme.color("failed")  # now the light value
        assert plain.data(Qt.ForegroundRole) is None
    finally:
        win.close()
