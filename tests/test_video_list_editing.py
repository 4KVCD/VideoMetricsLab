"""Issue #4: adding and arranging videos by hand -- files dropped on the
table or the reference box, a reference path pasted in, rows dragged into
order or sorted by a column, the file dialogs' folder, and the videos
reopened at the next start when asked for."""
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

from tests.factories import fake_completed_run, fake_run_result, fake_video_info
from vmaf_app.core.models import ResampleTarget, synthetic_resample_distorted_path
from vmaf_app.core.settings import Settings
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import COL_BITRATE, COL_PATH, COL_VMAF, CompletedRun, MainWindow


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


def test_a_row_dragged_within_the_table_lands_before_the_nearer_row_boundary(win):
    """FillColumnTable's own drop: the selected rows and the row they go
    before, from where the drop is; reported as a copy, or Qt's drag would
    delete the dragged rows itself."""
    _add(win, "a.mkv", "b.mkv", "c.mkv")
    table = win.distorted_table
    win.resize(1280, 800)
    win.show()
    QApplication.processEvents()
    table.selectRow(0)
    moved = []
    table.rowsMoved.connect(lambda rows, target: moved.append((rows, target)))
    last = table.rowViewportPosition(2) + table.rowHeight(2) - 2  # the lower half of the last row
    mime = table.model().mimeData([table.model().index(0, COL_PATH)])

    def drop(actions):
        event = QDropEvent(QPointF(5, last), actions, mime, Qt.LeftButton, Qt.NoModifier)
        event.source = lambda: table  # a drag from the table itself
        table.dropEvent(event)
        return event

    # A drag from the table offers a copy, so the drop can be reported as one.
    assert table.model().supportedDragActions() & Qt.CopyAction
    assert table.defaultDropAction() in (Qt.IgnoreAction, Qt.CopyAction)
    only_move = drop(Qt.MoveAction)
    assert not only_move.isAccepted() and moved == []
    assert table.rowCount() == 3

    event = drop(Qt.CopyAction | Qt.MoveAction)  # what QAbstractItemView.startDrag offers
    assert moved == [([0], 3)]
    assert event.isAccepted() and event.dropAction() == Qt.CopyAction
    assert _names(win) == ["b.mkv", "c.mkv", "a.mkv"]


def test_a_row_dropped_where_it_already_is_changes_nothing(win):
    _add(win, "a.mkv", "b.mkv")
    win._on_rows_dragged([0], 1)
    win._on_rows_dragged([1], 1)
    assert _names(win) == ["a.mkv", "b.mkv"]


def test_a_result_still_reaches_its_row_after_the_rows_moved(win):
    _add(win, "a.mkv", "b.mkv")
    b = win._rows[1]
    win._on_rows_dragged([1], 0)
    assert win._row_index_of(b) == 0
    assert win._row_index_of_path(Path("b.mkv")) == 0


def test_rows_cannot_be_moved_during_a_run(win):
    _add(win, "a.mkv", "b.mkv")
    win._set_run_ui_active(True)
    try:
        assert not win.distorted_table.drops_allowed
        win._on_rows_dragged([1], 0)
        assert _names(win) == ["a.mkv", "b.mkv"]
    finally:
        win._set_run_ui_active(False)
    assert win.distorted_table.drops_allowed


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


def test_a_drop_of_no_files_is_refused(win, tmp_path):
    event = _drop(win.distorted_table, [tmp_path])  # a folder only
    assert not event.isAccepted()
    assert win._rows == []


def test_nothing_is_dropped_on_the_table_during_a_run(win, tmp_path):
    path = tmp_path / "x.mkv"
    path.write_bytes(b"")
    win._set_run_ui_active(True)
    try:
        event = _drop(win.distorted_table, [path])
        assert not event.isAccepted()
        assert win._rows == []
    finally:
        win._set_run_ui_active(False)


# ------------------------------------------------------------------ sorting


def test_a_heading_click_sorts_by_name_naturally_and_again_reverses(win):
    _add(win, "v10.mkv", "V2.mkv", "v1.mkv")
    win._on_header_clicked(COL_PATH)
    assert _names(win) == ["v1.mkv", "V2.mkv", "v10.mkv"]
    header = win.distorted_table.horizontalHeader()
    assert header.isSortIndicatorShown() and header.sortIndicatorSection() == COL_PATH
    win._on_header_clicked(COL_PATH)
    assert _names(win) == ["v10.mkv", "V2.mkv", "v1.mkv"]
    assert header.sortIndicatorOrder() == Qt.DescendingOrder


def test_a_metric_sorts_by_its_score_with_unscored_rows_last_either_way(win):
    _add(win, "a.mkv", "b.mkv", "c.mkv", "d.mkv")
    for row, score in ((0, 80.0), (2, 95.0), (3, 60.0)):  # b has no score
        name = win._rows[row].path.name
        win._rows[row].completed_run = CompletedRun(fake_run_result(name, vmaf=score), name)
    win._on_header_clicked(COL_VMAF)
    assert _names(win) == ["d.mkv", "a.mkv", "c.mkv", "b.mkv"]
    win._on_header_clicked(COL_VMAF)
    assert _names(win) == ["c.mkv", "a.mkv", "d.mkv", "b.mkv"]


def test_bitrate_sorts_as_a_number(win):
    _add(win, "a.mkv", "b.mkv", "c.mkv")
    for row, rate in ((0, 900_000), (1, 12_000_000), (2, 2_000_000)):
        win._rows[row].video_info = replace(win._rows[row].video_info, bit_rate=rate)
    win._on_header_clicked(COL_BITRATE)
    assert _names(win) == ["a.mkv", "c.mkv", "b.mkv"]


def test_an_added_or_dragged_row_takes_the_sort_arrow_away(win):
    _add(win, "b.mkv", "a.mkv")
    header = win.distorted_table.horizontalHeader()
    win._on_header_clicked(COL_PATH)
    _add(win, "0.mkv")
    assert not header.isSortIndicatorShown()
    win._on_header_clicked(COL_PATH)
    win._on_rows_dragged([0], 3)
    assert not header.isSortIndicatorShown()
    win._on_header_clicked(COL_PATH)  # a fresh sort starts ascending again
    assert _names(win) == ["0.mkv", "a.mkv", "b.mkv"]


def test_nothing_is_sorted_during_a_run(win):
    _add(win, "b.mkv", "a.mkv")
    win._set_run_ui_active(True)
    try:
        win._on_header_clicked(COL_PATH)
    finally:
        win._set_run_ui_active(False)
    assert _names(win) == ["b.mkv", "a.mkv"]


def test_a_metric_tick_box_in_the_heading_does_not_sort(win):
    """The box is the metric's on/off switch: a click on it must not also
    reorder the table (CheckableHeaderView takes it before sectionClicked)."""
    _add(win, "b.mkv", "a.mkv")
    header = win.metric_header
    rect = header.section_indicator_rect(COL_VMAF)
    from PySide6.QtTest import QTest
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    assert _names(win) == ["b.mkv", "a.mkv"]


# ------------------------------------------------------------------ the reference box


def test_a_pasted_reference_path_is_read_without_its_quotes(win, tmp_path):
    path = tmp_path / "ref.mkv"
    path.write_bytes(b"")
    win.source_edit.setText(f'"{path}"')
    win._on_source_path_entered()
    win._on_source_path_entered()  # Enter, then leaving the box: read once
    assert win.probed_sources == [path]


def test_a_path_to_nothing_entered_says_so_and_is_taken_back(win, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(win.source_edit, "hasFocus", lambda: True)  # Enter, in the box
    win.source_edit.setText(str(tmp_path / "missing.mkv"))
    win._on_source_path_entered()
    assert win.probed_sources == []
    assert warnings and "missing.mkv" in warnings[0]
    assert win.source_edit.text() == ""


def test_leaving_the_box_half_typed_keeps_the_text_without_a_warning(win, tmp_path, monkeypatch):
    """Switching to another program (to copy the rest of a path) also ends
    the edit; that is no reason to complain or to throw the text away."""
    warnings = []
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(win.source_edit, "hasFocus", lambda: False)
    win.source_edit.setText(str(tmp_path / "half"))
    win._on_source_path_entered()
    assert warnings == [] and win.probed_sources == []
    assert win.source_edit.text() == str(tmp_path / "half")


def test_the_reference_written_another_way_is_not_read_again(win, tmp_path):
    """Read again, the reference would lose its resolution tests."""
    path = tmp_path / "Ref.mkv"
    path.write_bytes(b"")
    win._source_info = replace(fake_video_info(path), path=path)
    win.source_edit.setText(str(path).upper().replace("\\", "/"))
    win._on_source_path_entered()
    assert win.probed_sources == []
    assert win.source_edit.text() == str(path)


def test_a_drag_over_the_table_looks_at_its_files_once(win, tmp_path, monkeypatch):
    """A drag move comes with every mouse move; the files are looked at on
    disk when the drag enters (slow on a network drive), not each time."""
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent

    from vmaf_app.ui.widgets import FillColumnTable
    path = tmp_path / "x.mkv"
    path.write_bytes(b"")
    looked = []
    real = FillColumnTable.dropped_files
    monkeypatch.setattr(FillColumnTable, "dropped_files", staticmethod(lambda mime: looked.append(1) or real(mime)))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    table = win.distorted_table
    table.dragEnterEvent(QDragEnterEvent(QPointF(5, 5).toPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    for y in range(5, 50, 5):
        move = QDragMoveEvent(QPointF(5, y).toPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        table.dragMoveEvent(move)
        assert move.isAccepted()
    assert len(looked) == 1


def test_the_reference_box_cannot_be_typed_in_during_a_run(win):
    win._set_run_ui_active(True)
    try:
        assert win.source_edit.isReadOnly()
    finally:
        win._set_run_ui_active(False)
    assert not win.source_edit.isReadOnly()


def test_one_file_dropped_on_the_reference_box_becomes_the_reference(win, tmp_path):
    files = [tmp_path / "ref.mkv", tmp_path / "other.mkv"]
    for path in files:
        path.write_bytes(b"")
    two = _drop(win.source_edit, files, target=lambda e: win.eventFilter(win.source_edit, e))
    assert not two.isAccepted() and win.probed_sources == []
    one = _drop(win.source_edit, files[:1], target=lambda e: win.eventFilter(win.source_edit, e))
    assert one.isAccepted() and win.probed_sources == [files[0]]
    assert win.source_edit.text() == ""  # shown once it is read, not before


# ------------------------------------------------------------------ the file dialogs' folder


def test_the_file_dialogs_open_where_a_video_last_came_from(win, tmp_path, monkeypatch):
    asked = []

    def dialog(parent, title, folder=""):
        asked.append(folder)
        return [str(tmp_path / "v.mkv")], ""

    monkeypatch.setattr(main_window_module.QFileDialog, "getOpenFileNames", dialog)
    win._on_add_distorted()
    win._on_add_distorted()
    assert asked == ["", str(tmp_path)]
    assert Settings.load().last_video_dir == str(tmp_path)  # kept for the next start
    win._settings.last_video_dir = str(tmp_path / "gone")
    assert win._video_dialog_dir() == ""


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


def test_videos_are_not_kept_unless_asked_for(win, tmp_path):
    assert not win.settings_remember_videos.isChecked()  # off by default
    _close_with(win, tmp_path, remember=False)
    saved = Settings.load()
    assert saved.remembered_reference == "" and saved.remembered_tests == []


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


def test_videos_added_before_the_remembered_ones_are_found_win(win, tmp_path, monkeypatch):
    """The remembered videos come later, from their thread; a video the user
    added meanwhile is not joined by last session's."""
    _close_with(win, tmp_path, remember=True)
    later = []
    monkeypatch.setattr(main_window_module.threading, "Thread", lambda target, **kw: _Deferred(target, later))
    reopened = MainWindow()
    try:
        _add(reopened, "mine.mkv")
        later[0]()
        assert [row.path.name for row in reopened._rows] == ["mine.mkv"]
    finally:
        reopened.close()


class _Deferred:
    def __init__(self, target, queue):
        self._target, self._queue = target, queue

    def start(self):
        self._queue.append(self._target)


def test_turning_it_off_forgets_the_videos(win, tmp_path):
    _close_with(win, tmp_path, remember=True)
    again = MainWindow()
    try:
        again.settings_remember_videos.setChecked(False)
        saved = Settings.load()
        assert not saved.remember_videos and saved.remembered_tests == []
    finally:
        again.close()


def test_sorting_reads_the_score_the_cell_shows(win):
    """Through _metric_mean, as _set_row_metrics does."""
    _add(win, "a.mkv")
    win._rows[0].completed_run = fake_completed_run("a.mkv")
    assert win._sort_key(0, COL_VMAF) == pytest.approx(90.0)
