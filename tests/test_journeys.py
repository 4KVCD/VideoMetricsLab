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
    ComparisonResult,
    FrameScores,
    VideoInfo,
)
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import (
    TAB_FRAME_COMPARE,
    TAB_VIDEOS,
    MainWindow,
)


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


def _result(distorted: Path, source: VideoInfo, score: float = 95.0) -> ComparisonResult:
    n = 240
    return ComparisonResult(
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


def assert_frame_compare_matches_rows(win: MainWindow) -> None:
    """Frame Compare shows every row it can render -- scored or not.

    Frames are comparable without metrics, so a row that has merely been
    probed still gets an entry; it is identified by the row itself, while a
    scored row keeps the identity its result already carries.
    """
    expected = []
    for row in win._rows:
        if row.completed_run is not None:
            expected.append(row.completed_run.graph_identity)
        elif win._source_info is not None and (
            row.options.resample_test is not None or row.video_info is not None
        ):
            expected.append(row.frame_identity)
    actual = [entry.identity for entry in win.frame_compare_panel._entries]
    assert actual == expected


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


def test_frame_compare_tracks_completed_removed_and_invalidated_rows(qapp):
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    rows = []
    for name in ("a", "b"):
        row = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[row].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        rows.append(row)

    _finish_run(win, [
        (rows[0], _result(Path("C:/vid/a.mkv"), source, 95.0)),
        (rows[1], _result(Path("C:/vid/b.mkv"), source, 85.0)),
    ])
    assert_frame_compare_matches_rows(win)

    # Opening the tab synchronizes it without creating duplicates. Avoid
    # showing the test's deliberately nonexistent media by replacing only
    # the decode-triggering hook; the real tab and currentChanged signal run.
    win.frame_compare_panel._show_or_request = lambda: None
    win.tabs.setCurrentIndex(TAB_FRAME_COMPARE)
    assert_frame_compare_matches_rows(win)

    win.tabs.setCurrentIndex(TAB_VIDEOS)
    win.distorted_table.selectRow(0)
    win._on_remove_distorted()
    assert_frame_compare_matches_rows(win)
    assert len(win.frame_compare_panel._entries) == 1

    win._invalidate_completed_results([0])
    assert_frame_compare_matches_rows(win)
    # Losing a score does not make the frames incomparable: the row stays,
    # now identified by itself and carrying no scores.
    assert len(win.frame_compare_panel._entries) == 1
    entry = win.frame_compare_panel._entries[0]
    assert entry.scores is None
    assert entry.identity is win._rows[0].frame_identity


def test_the_buttons_are_wired_to_something(qapp, confirm_yes):
    # Calling the handler directly proves the handler works, not that the
    # button reaches it. A button built into an unattached layout, or never
    # connected, passes every other kind of test here.
    win = MainWindow()
    win._add_table_row(Path("C:/vid/a.mkv"))

    win.distorted_table.selectRow(0)
    win.remove_all_btn.click()
    assert win._rows == [], "Remove all must be connected to its handler"


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


# ------------------------------------------------------------------ found by looking


def test_a_new_source_clears_scores_that_belonged_to_the_old_one(qapp, monkeypatch):
    # A score is for a (source, distorted) pair. Keeping the old numbers
    # against a new source would show a comparison that was never made.
    win = MainWindow()
    source_a = _info("C:/vid/sourceA.mkv")
    win._source_info = source_a
    r = win._add_table_row(Path("C:/vid/a.mkv"))
    win._rows[r].video_info = _info("C:/vid/a.mkv", 1920, 1080)
    _finish_run(win, [(r, _result(Path("C:/vid/a.mkv"), source_a))])
    assert win._rows[r].completed_run is not None
    assert_graph_matches_rows(win)

    monkeypatch.setattr(win, "_start_cache_lookup", lambda *a, **k: None)
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: ("C:/vid/sourceB.mkv", "")),
    )
    monkeypatch.setattr(
        win, "_start_source_probe",
        lambda p: win._apply_source_info(p, _info(str(p))),
    )
    win._on_browse_source()

    assert win._rows[r].completed_run is None, "the old source's score must not stand"
    assert_graph_matches_rows(win)
    assert not win.graph_panel._entries, "the old source's curve must not stand either"
