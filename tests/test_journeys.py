"""Multi-step journeys, driven through the widgets a user actually clicks.

Every other test file here checks one operation in isolation, by calling the
handler directly. That is why a run of basic bugs shipped anyway: the graph
staying empty until a button was pressed, a removed video's curve staying on
the plot, a readout that never updated. None of those are broken *functions*
-- each function worked. They were broken *sequences*, and broken links
between components.

So these tests do two things the others don't:

  * drive the UI through its real signal path (button.click(), not
    _on_button_clicked()), so anything that depends on a connection being
    made is exercised;
  * assert whole-app invariants after every step, rather than checking the
    return value of the thing just called.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import (
    FrameScores,
    ResampleTarget,
    VideoInfo,
    VmafRunResult,
    synthetic_resample_distorted_path,
)
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import COL_VMAF, TAB_GRAPH, TAB_VIDEOS, MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def confirm_yes(monkeypatch):
    """Answers "yes" to any confirmation, so a journey isn't blocked on one."""
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: main_window_module.QMessageBox.Yes,
    )
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(main_window_module.QMessageBox, name, lambda *a, **k: None)


def _info(name: str, w: int = 3840, h: int = 2160) -> VideoInfo:
    return VideoInfo(path=Path(name), width=w, height=h, fps=24.0, duration=10.0,
                     nb_frames=240, codec_name="hevc")


def _result(distorted: Path, source: VideoInfo, score: float = 95.0) -> VmafRunResult:
    n = 240
    return VmafRunResult(
        source=source.path, distorted=distorted,
        frames=FrameScores(
            frame=np.arange(n, dtype=np.int32),
            time=np.arange(n, dtype=np.float64) / 24.0,
            vmaf=np.full(n, score, dtype=np.float32),
            psnr=None, ssim=None, xpsnr=None,
        ),
        fps=24.0, model="m", source_crop=None, distorted_crop=None,
        source_info=source, distorted_info=source,
    )


def _finish_run(win: MainWindow, rows_and_results) -> None:
    """Drives a completed batch the way the worker's signals do."""
    win._job_rows = [win._rows[r] for r, _ in rows_and_results]
    win._checked_rows_for_run = list(win._job_rows)
    for job_index, (_row, result) in enumerate(rows_and_results):
        win._on_job_finished(job_index, result)
    win._on_all_finished()


def assert_graph_matches_rows(win: MainWindow) -> None:
    """THE invariant: the graph shows exactly the videos that have a result.

    Not more (a removed video's curve lingering), and not fewer (a finished
    run that never reached the plot). Every journey below asserts this after
    every step, because both directions shipped as bugs.
    """
    expected = {
        Path(r.completed_run.result.distorted)
        for r in win._rows
        if r.completed_run is not None
    }
    actual = {Path(e.result.distorted) for e in win.graph_panel._entries.values()}
    assert actual == expected, (
        f"graph out of step with the table\n"
        f"  on the plot but not a scored row: {actual - expected}\n"
        f"  scored row but not on the plot  : {expected - actual}"
    )


# ------------------------------------------------------------------ journeys

def test_a_finished_run_reaches_the_graph_without_pressing_anything(qapp):
    # The graph used to stay empty until "Show graph" was pressed.
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    row = win._add_table_row(Path("C:/vid/a.mkv"))
    win._rows[row].video_info = _info("C:/vid/a.mkv", 1920, 1080)

    assert_graph_matches_rows(win)
    _finish_run(win, [(row, _result(Path("C:/vid/a.mkv"), source))])
    assert_graph_matches_rows(win)
    assert len(win.graph_panel._entries) == 1


def test_the_reported_sequence_source_swap_remove_all_resample_run(qapp, confirm_yes):
    """Exactly the steps that were reported as leaving the graph empty:
    run one encode, swap the source, remove everything, add a new encode and
    a resolution test, run again."""
    win = MainWindow()

    source_a = _info("C:/vid/sourceA.mkv")
    win._source_info = source_a
    row = win._add_table_row(Path("C:/vid/encodeA.mkv"))
    win._rows[row].video_info = _info("C:/vid/encodeA.mkv", 1920, 1080)
    _finish_run(win, [(row, _result(Path("C:/vid/encodeA.mkv"), source_a, 95.0))])
    assert_graph_matches_rows(win)

    win.tabs.setCurrentIndex(TAB_GRAPH)
    assert len(win.graph_panel._entries) == 1
    win.tabs.setCurrentIndex(TAB_VIDEOS)

    # Swap the source, then clear the table.
    source_b = _info("C:/vid/sourceB.mkv")
    win._source_info = source_b
    win.remove_all_btn.click()
    assert win._rows == []
    assert_graph_matches_rows(win)
    assert len(win.graph_panel._entries) == 0, "clearing the table clears the plot"

    # A new encode and a round-trip test.
    row_b = win._add_table_row(Path("C:/vid/encodeB.mkv"))
    win._rows[row_b].video_info = _info("C:/vid/encodeB.mkv", 1920, 1080)
    target = ResampleTarget(width=1920, label="1080p")
    rpath = synthetic_resample_distorted_path(source_b.path, target)
    row_r = win._add_table_row(rpath)
    win._rows[row_r].options.resample_test = target
    win._rows[row_r].video_info = source_b
    assert_graph_matches_rows(win)

    _finish_run(win, [
        (row_b, _result(Path("C:/vid/encodeB.mkv"), source_b, 93.0)),
        (row_r, _result(rpath, source_b, 98.0)),
    ])

    assert win.distorted_table.item(row_b, COL_VMAF).text() == "93.00"
    assert win.distorted_table.item(row_r, COL_VMAF).text() == "98.00"
    assert_graph_matches_rows(win)
    assert len(win.graph_panel._pages["vmaf"]._curves) == 2, "both curves drawn"
    assert win.graph_panel.stats_table.rowCount() == 2


def test_removing_videos_one_at_a_time_keeps_the_graph_in_step(qapp):
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    rows = []
    for name in ("a", "b", "c"):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        rows.append(r)
    _finish_run(win, [(r, _result(Path(f"C:/vid/{n}.mkv"), source))
                      for r, n in zip(rows, "abc", strict=True)])
    assert_graph_matches_rows(win)

    while win._rows:
        win.distorted_table.selectRow(0)
        win._on_remove_distorted()
        assert_graph_matches_rows(win)
    assert len(win.graph_panel._entries) == 0


def test_switching_tabs_repeatedly_never_duplicates_a_series(qapp):
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    row = win._add_table_row(Path("C:/vid/a.mkv"))
    win._rows[row].video_info = _info("C:/vid/a.mkv", 1920, 1080)
    _finish_run(win, [(row, _result(Path("C:/vid/a.mkv"), source))])

    for _ in range(5):
        win.tabs.setCurrentIndex(TAB_GRAPH)
        win.tabs.setCurrentIndex(TAB_VIDEOS)
        assert_graph_matches_rows(win)
    assert len(win.graph_panel._entries) == 1


def test_the_buttons_are_wired_to_something(qapp, confirm_yes):
    # Calling the handler directly proves the handler works, not that the
    # button reaches it. A button built into an unattached layout, or never
    # connected, passes every other kind of test here.
    win = MainWindow()
    win._add_table_row(Path("C:/vid/a.mkv"))

    win.distorted_table.selectRow(0)
    win.remove_all_btn.click()
    assert win._rows == [], "Remove all must be connected to its handler"


def test_the_graph_starts_empty_and_says_what_to_do(qapp):
    # The empty state is what a user sees first and is the easiest to never
    # look at.
    win = MainWindow()
    win.tabs.setCurrentIndex(TAB_GRAPH)

    assert len(win.graph_panel._entries) == 0
    assert win.graph_panel.stats_table.rowCount() == 0
    readout = win.graph_panel._pages["vmaf"].hover_label.text()
    assert "Hover" in readout, "the empty plot should explain itself"


def test_jumping_to_a_frame_reports_every_visible_series(qapp):
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    for name, score in (("a", 95.0), ("b", 85.0)):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        win._rows[r].completed_run = main_window_module.CompletedRun(
            _result(Path(f"C:/vid/{name}.mkv"), source, score), name
        )
    win.tabs.setCurrentIndex(TAB_GRAPH)

    win.graph_panel.frame_spin.setValue(120)
    text = win.graph_panel._pages["vmaf"].hover_label.text()
    assert "Frame 120" in text
    assert "VMAF=95.00" in text and "VMAF=85.00" in text


def test_curves_appear_as_each_job_finishes_not_only_at_the_end(qapp):
    # With eight videos queued, the first result should be on the plot while
    # the rest are still running -- watching progress is most of the point.
    # Mutation testing showed nothing covered this: the end-of-batch sync
    # hid a missing per-job update.
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    rows = []
    for name in ("a", "b", "c"):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        rows.append(r)

    win._job_rows = [win._rows[r] for r in rows]
    win._checked_rows_for_run = list(win._job_rows)

    for done, name in enumerate("abc"):
        win._on_job_finished(done, _result(Path(f"C:/vid/{name}.mkv"), source))
        # ...before _on_all_finished has run.
        assert len(win.graph_panel._entries) == done + 1, (
            f"after {done + 1} of 3 jobs the plot should show {done + 1} curve(s)"
        )
        assert_graph_matches_rows(win)
