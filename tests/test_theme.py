"""Issue #5: the window in dark colours -- what the app colours itself
follows the theme, and Settings can choose light or dark."""
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from tests.factories import fake_completed_run, fake_video_info
from vmaf_app.core.settings import Settings
from vmaf_app.ui import theme
from vmaf_app.ui.main_window import COL_PATH, COL_VMAF, MainWindow


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


def test_a_styled_widget_is_styled_again_when_the_theme_changes(qapp, dark):
    label = QLabel()
    theme.style(label, "color: {muted};")
    assert theme.color("muted").name() in label.styleSheet()
    dark["dark"] = False
    theme.refresh()
    assert label.styleSheet() == f"color: {theme.color('muted').name()};"


def test_a_deleted_widget_is_let_go(qapp, dark):
    label = QLabel()
    theme.style(label, "color: {muted};")
    label.deleteLater()
    QApplication.processEvents()
    theme.refresh()  # does not raise


def test_series_text_is_lightened_on_a_dark_window(dark):
    blue = "#4C72B0"
    assert theme.readable(blue).lightness() > theme.QColor(blue).lightness()
    dark["dark"] = False
    assert theme.readable(blue).name().lower() == blue.lower()


def test_a_score_is_in_the_table_s_text_colour_not_black(qapp, dark):
    """The issue's first picture: black scores on a dark table."""
    win = MainWindow()
    try:
        row = win._add_table_row(Path("a.mkv"))
        win._rows[row].video_info = fake_video_info("a.mkv")
        win._rows[row].completed_run = fake_completed_run("a.mkv")
        win._set_row_metrics(row)
        cell = win.distorted_table.item(row, COL_VMAF)
        # No colour of its own: the table's, which follows a theme switch.
        # One copied from the palette stayed white on a table switched to light.
        assert cell.text() and cell.data(Qt.ForegroundRole) is None
        assert win.distorted_table.item(row, COL_PATH).data(Qt.ForegroundRole) is None
    finally:
        win.close()


def test_a_theme_change_recolours_what_the_window_colours_itself(qapp, dark):
    win = MainWindow()
    try:
        row = win._add_table_row(Path("a.mkv"))
        win._rows[row].analysis_status = None
        from vmaf_app.ui.row_state import RowState
        win._set_row_status(row, RowState.FAILED, "broken")
        failed_dark = win.distorted_table.item(row, COL_PATH).foreground().color()
        source_style = win.source_info_label.styleSheet()
        dark["dark"] = False
        win._apply_theme_colours()
        assert win.distorted_table.item(row, COL_PATH).foreground().color() != failed_dark
        assert win.source_info_label.styleSheet() != source_style
    finally:
        win.close()


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


def test_the_graph_statistics_are_tinted_for_the_theme(qapp, dark):
    win = MainWindow()
    try:
        name = "a.mkv"
        run = fake_completed_run(name)
        win.graph_panel.add_run(run.result, name)
        table = win.graph_panel.stats_table
        tinted = [table.item(0, col) for col in range(table.columnCount())
                  if table.item(0, col) is not None and table.item(0, col).background().style() != Qt.NoBrush]
        assert tinted and all(item.background().color().lightness() < 80 for item in tinted)
        dark["dark"] = False
        win.graph_panel.refresh_theme()
        tinted = [table.item(0, col) for col in range(table.columnCount())
                  if table.item(0, col) is not None and table.item(0, col).background().style() != Qt.NoBrush]
        assert all(item.background().color().lightness() > 180 for item in tinted)
    finally:
        win.close()


def test_every_colour_value_is_its_own():
    """A switch recolours a cell by its colour's value (theme.recolour): two
    names sharing a value could not be told apart."""
    values = [value.lower() for pair in theme._COLOURS.values() for value in pair]
    assert len(values) == len(set(values))


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
