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
from vmaf_app.ui.main_window import (
    COL_VMAF,
    TAB_FRAME_COMPARE,
    TAB_GRAPH,
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

    win._invalidate_completed_result(0)
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


# ------------------------------------------------------------------ found by looking

def test_the_delta_uses_each_metrics_own_precision(qapp):
    # Every metric's delta was formatted to 2dp. SSIM's entire range is 0-1,
    # so every real SSIM difference rendered as "+0.00" -- the number was
    # there, and useless. Only visible by reading the output.
    from vmaf_app.ui.graph_panel import METRICS

    by_key = {m.key: m for m in METRICS}
    assert by_key["vmaf"].format_delta(1.234) == "+1.23"
    assert by_key["psnr"].format_delta(-0.5) == "-0.50"
    assert by_key["ssim"].format_delta(0.0004) == "+0.0004", "SSIM needs its 4dp"
    assert by_key["ssim"].format_delta(-0.0012) == "-0.0012"


def test_re_adding_a_video_keeps_its_colour(qapp):
    # add_run runs again for the same video on every tab switch and as each
    # job finishes. Taking the next palette entry each time walked four
    # videos out of blue/orange/green/red and into brown/pink/grey -- which
    # only showed up by looking at the plot.
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    row = win._add_table_row(Path("C:/vid/a.mkv"))
    win._rows[row].video_info = _info("C:/vid/a.mkv", 1920, 1080)
    _finish_run(win, [(row, _result(Path("C:/vid/a.mkv"), source))])

    first_colour = next(iter(win.graph_panel._entries.values())).color
    for _ in range(6):
        win.tabs.setCurrentIndex(TAB_GRAPH)
        win.tabs.setCurrentIndex(TAB_VIDEOS)
    assert next(iter(win.graph_panel._entries.values())).color == first_colour


def test_four_videos_get_the_first_four_palette_colours(qapp):
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    rows = []
    for name in ("a", "b", "c", "d"):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        rows.append(r)
    _finish_run(win, [(r, _result(Path(f"C:/vid/{n}.mkv"), source))
                      for r, n in zip(rows, "abcd", strict=True)])
    win.tabs.setCurrentIndex(TAB_GRAPH)

    from vmaf_app.ui.graph_panel import _PALETTE

    colours = [e.color for e in win.graph_panel._entries.values()]
    assert colours == _PALETTE[:4], f"expected the first four palette colours, got {colours}"


def test_a_probed_row_does_not_stay_greyed_out(qapp):
    # Rows show a grey "Reading..." while the probe runs; the real media
    # info must come back in normal text or a loaded row looks disabled.
    from vmaf_app.ui.main_window import COL_INFO

    win = MainWindow()
    row = win._add_table_row(Path("C:/vid/a.mkv"))
    win._set_row_status(row, "Reading...")
    assert win._row_state(win._rows[row]) == "Reading..."

    win._set_row_info(row, _info("C:/vid/a.mkv", 1920, 1080))
    after = win.distorted_table.item(row, COL_INFO).foreground().color()
    assert after == win.distorted_table.palette().text().color()
    assert win._row_state(win._rows[row]) == "Not calculated"


def test_the_readout_fits_every_line_it_prints(qapp):
    # The last series' line was cut off: the stylesheet padding comes out of
    # the fixed height, and the allowance did not cover it.
    from PySide6.QtGui import QFontMetrics

    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source
    rows = []
    for name in ("a", "b", "c", "d"):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)
        rows.append(r)
    _finish_run(win, [(r, _result(Path(f"C:/vid/{n}.mkv"), source))
                      for r, n in zip(rows, "abcd", strict=True)])
    win.tabs.setCurrentIndex(TAB_GRAPH)

    win.graph_panel.frame_spin.setValue(100)
    page = win.graph_panel._pages["vmaf"]
    fm = QFontMetrics(page.hover_label.font())
    lines = page.hover_label.text().splitlines()
    assert len(lines) == 5, "one header line plus four series"
    needed = len(lines) * fm.lineSpacing()
    assert needed <= page.hover_label.height(), (
        f"{len(lines)} lines need {needed}px but the readout is "
        f"{page.hover_label.height()}px -- the last line is cut off"
    )


def test_changing_the_source_does_not_reload_caches_on_the_ui_thread(qapp, monkeypatch):
    # A cached result is a multi-MB JSON parse. Re-checking every row inline
    # when the source changed froze the window for hundreds of ms per row --
    # seconds with a handful of feature-length videos loaded.
    win = MainWindow()
    for name in ("a", "b", "c", "d"):
        r = win._add_table_row(Path(f"C:/vid/{name}.mkv"))
        win._rows[r].video_info = _info(f"C:/vid/{name}.mkv", 1920, 1080)

    loaded_inline = []
    monkeypatch.setattr(
        main_window_module, "result_cache",
        type("Spy", (), {
            "load_cached": staticmethod(lambda *a: loaded_inline.append(a)),
            "store": staticmethod(lambda *a: None),
            "clear": staticmethod(lambda *a: None),
            "cache_dir": staticmethod(lambda: Path(".")),
            "set_cache_dir_override": staticmethod(lambda *a: None),
        })(),
    )
    started = []
    monkeypatch.setattr(win, "_start_cache_lookup", started.append)

    win._source_info = _info("C:/vid/newsource.mkv")
    win._reload_cached_for_all_rows()

    assert not loaded_inline, "cached results must not be parsed on the UI thread"
    assert started, "the reload should be handed to the worker"
    paths = started[0]
    assert len(paths) == 4
    # A source change only invalidates cached SCORES; the distorted files
    # themselves have not changed, so this must not re-probe them.
    assert win._cache_worker is None or not win._cache_worker.isRunning()


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


def test_saving_two_runs_with_the_same_label_keeps_both_files(qapp, tmp_path, monkeypatch):
    """Two rows can carry the same label -- the same basename from two
    folders is the everyday case -- and saving them into one folder used to
    write both to <label>.vmafrun.json, losing the first."""
    win = MainWindow()
    source = _info("C:/vid/source.mkv")
    win._source_info = source

    rows = []
    for folder, score in (("a", 95.0), ("b", 70.0)):
        path = Path(f"C:/vid/{folder}/movie.mkv")
        row = win._add_table_row(path)
        win._rows[row].video_info = _info(str(path), 1920, 1080)
        rows.append((row, _result(path, source, score)))
    _finish_run(win, rows)
    win.distorted_table.selectAll()

    monkeypatch.setattr(
        main_window_module.QFileDialog, "getExistingDirectory",
        lambda *a, **k: str(tmp_path),
    )
    win._on_save_selected()
    assert win._file_writes.wait_until_idle(10.0), "the save never finished"

    saved = sorted(p.name for p in tmp_path.glob("*.vmafrun.json"))
    assert saved == ["movie.vmafrun.json", "movie_2.vmafrun.json"], (
        "one run overwrote the other"
    )
