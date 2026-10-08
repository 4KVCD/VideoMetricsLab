"""Issue #4: adding and arranging videos by hand -- files dropped on the
table or the reference box, a reference path pasted in, rows dragged into
order or sorted by a column, the file dialogs' folder, and the videos
reopened at the next start when asked for."""
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

from tests.factories import fake_run_result, fake_video_info
from vmaf_app.core.models import ResampleTarget, synthetic_resample_distorted_path
from vmaf_app.core.settings import Settings
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import COL_PATH, COL_VMAF, CompletedRun, MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, monkeypatch):
    """A window that probes nothing: added rows stay as added, a chosen
    reference is recorded instead of read."""
    probed: list[Path] = []
    monkeypatch.setattr(MainWindow, "_start_media_probe", lambda self, paths: None)
    monkeypatch.setattr(MainWindow, "_start_cache_lookup", lambda self, paths: None)
    monkeypatch.setattr(MainWindow, "_start_source_probe", lambda self, path: probed.append(path))
    window = MainWindow()
    window.probed_sources = probed
    yield window
    window.close()


def _names(win) -> list[str]:
    shown = [win.distorted_table.item(row, COL_PATH).text() for row in range(win.distorted_table.rowCount())]
    assert shown == [row.path.name for row in win._rows]  # the cells and the rows behind them move together
    return shown


def _add(win, *names: str) -> None:
    for name in names:
        row = win._add_table_row(Path(name))
        win._rows[row].video_info = fake_video_info(name)


def _selected(win) -> list[int]:
    return sorted(index.row() for index in win.distorted_table.selectionModel().selectedRows())


def _drop(widget, paths, *, target=None) -> QDropEvent:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    (target or widget.dropEvent)(event)
    return event


# ------------------------------------------------------------------ dragging rows


def test_dragged_rows_move_with_their_data_and_stay_selected(win):
    _add(win, "a.mkv", "b.mkv", "c.mkv", "d.mkv")
    win.distorted_table.setRangeSelected(QTableWidgetSelectionRange(1, 0, 2, win.distorted_table.columnCount() - 1),
                                         True)
    win._on_rows_dragged([1, 2], 4)
    assert _names(win) == ["a.mkv", "d.mkv", "b.mkv", "c.mkv"]
    assert _selected(win) == [2, 3]
    assert win._panel_target_rows == [2, 3]  # the options panel edits the same videos as before

    win._on_rows_dragged([3], 0)
    assert _names(win) == ["c.mkv", "a.mkv", "d.mkv", "b.mkv"]


def test_a_result_still_reaches_its_row_after_the_rows_moved(win):
    _add(win, "a.mkv", "b.mkv")
    b = win._rows[1]
    win._on_rows_dragged([1], 0)
    assert win._row_index_of(b) == 0
    assert win._row_index_of_path(Path("b.mkv")) == 0


# ------------------------------------------------------------------ files dropped on the table


def test_files_dropped_on_the_table_are_added_in_order_once(win, tmp_path):
    files = [tmp_path / name for name in ("x.mkv", "y.mkv")]
    for path in files:
        path.write_bytes(b"")
    _drop(win.distorted_table, [*files, files[0], tmp_path])  # again, and a folder
    assert _names(win) == ["x.mkv", "y.mkv"]
    assert win._settings.last_video_dir == str(tmp_path)
    _drop(win.distorted_table, files)
    assert _names(win) == ["x.mkv", "y.mkv"]  # already there


# ------------------------------------------------------------------ sorting


def test_a_metric_sorts_by_its_score_with_unscored_rows_last_either_way(win):
    _add(win, "a.mkv", "b.mkv", "c.mkv", "d.mkv")
    for row, score in ((0, 80.0), (2, 95.0), (3, 60.0)):  # b has no score
        name = win._rows[row].path.name
        win._rows[row].completed_run = CompletedRun(fake_run_result(name, vmaf=score), name)
    win._on_header_clicked(COL_VMAF)
    assert _names(win) == ["d.mkv", "a.mkv", "c.mkv", "b.mkv"]
    win._on_header_clicked(COL_VMAF)
    assert _names(win) == ["c.mkv", "a.mkv", "d.mkv", "b.mkv"]


# ------------------------------------------------------------------ videos reopened at the next start


def _close_with(win, tmp_path, remember: bool):
    reference = tmp_path / "ref.mkv"
    tests = [tmp_path / "a.mkv", tmp_path / "b.mkv"]
    for path in (reference, *tests):
        path.write_bytes(b"")
    win.settings_remember_videos.setChecked(remember)
    win._source_info = fake_video_info(reference)
    for path in tests:
        _add(win, str(path))
    resample = win._add_table_row(synthetic_resample_distorted_path(reference, ResampleTarget(1280, 720)))
    win._rows[resample].options.resample_test = ResampleTarget(1280, 720)
    win.close()
    return reference, tests


class _InlineThread:
    """threading.Thread run at start(), in the caller's thread: no waiting."""

    def __init__(self, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def test_the_videos_open_at_closing_are_reopened_those_still_there(win, tmp_path, monkeypatch):
    reference, tests = _close_with(win, tmp_path, remember=True)
    saved = Settings.load()
    assert saved.remembered_reference == str(reference)
    assert saved.remembered_tests == [str(path) for path in tests]  # not the resolution test
    tests[0].unlink()

    win.probed_sources.clear()
    threads = []
    monkeypatch.setattr(main_window_module.threading, "Thread",
                        lambda target, **kw: threads.append(target) or _InlineThread(target))
    reopened = MainWindow()
    try:
        assert threads  # looked for off the window's thread (a network drive can take long)
        assert [row.path for row in reopened._rows] == [tests[1]]
        assert win.probed_sources == [reference]
    finally:
        reopened.close()
