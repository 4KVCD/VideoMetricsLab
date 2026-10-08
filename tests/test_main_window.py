import json
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

from tests.factories import decode_plan
from tests.factories import (
    fake_completed_run as _fake_completed_run,
)
from tests.factories import (
    fake_video_info as _fake_video_info,
)
from vmaf_app.core import result_cache
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    displayable_metric_specs,
)
from vmaf_app.core.models import (
    ComparisonResult,
    CropBox,
    CropMode,
    FrameScore,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    synthetic_resample_distorted_path,
    synthetic_scale_direction_variant_path,
)
from vmaf_app.core.settings import Settings
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui import probe_worker as probe_worker_module
from vmaf_app.ui.main_window import (
    COL_BLACK_BARS,
    COL_INFO,
    COL_PATH,
    COL_PSNR,
    COL_SCALING,
    COL_SSIM,
    COL_VMAF,
    COL_XPSNR,
    TAB_GRAPH,
    TAB_SETTINGS,
    CompletedRun,
    MainWindow,
)
from vmaf_app.ui.row_state import RowState


def _cache_request(options):
    return analysis_request_from_vmaf_options(options)


def _cache_key(source, distorted, options):
    return result_cache.cache_key(source, distorted, _cache_request(options))


def _load_cached(source, distorted, options, directory=None):
    return result_cache.load_cached(
        source, distorted, _cache_request(options), directory,
        displayable_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(
        source, distorted, result, label, _cache_request(options), directory
    )


def _clear_cached(source, distorted, options, directory=None):
    # The ticked metrics only, as the window clears them for a recalculation.
    return result_cache.clear(source, distorted, _cache_request(options), directory)

@pytest.fixture
def clock(monkeypatch):
    """The window's clock, standing at 1000 s until a test moves it."""
    class Clock:
        now = 1000.0

        def monotonic(self):
            return self.now

        def __getattr__(self, name):  # the rest of the time module
            return getattr(time, name)

    fake = Clock()
    monkeypatch.setattr(main_window_module, "time", fake)
    return fake


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _ask_for_vmaf_only(win, row: int) -> None:
    """Narrows a row to VMAF, matching what _fake_completed_run provides.

    New rows request all four metrics, so a row holding a VMAF-only fixture
    result is genuinely still incomplete and will be queued again. Tests
    about *scored vs unscored* rows have to ask for what the fixture
    actually contains, or they are testing the fixture's gaps instead.
    """
    options = win._rows[row].options
    options.extra_features = []
    options.compute_xpsnr = False
    options.compute_vmaf = True
    win._set_row_metrics(row)


# ------------------------------------------------------------------ row state


def test_run_clicked_only_queues_unscored_rows(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")

    scored_row = win._add_table_row(Path("scored.mp4"))
    win._rows[scored_row].video_info = _fake_video_info("scored.mp4")
    win._rows[scored_row].completed_run = _fake_completed_run("scored.mp4")
    _ask_for_vmaf_only(win, scored_row)

    unscored_row = win._add_table_row(Path("tests/fixtures/distorted.mp4"))
    win._rows[unscored_row].video_info = _fake_video_info("tests/fixtures/distorted.mp4")

    win._on_run_clicked()

    assert win._worker is not None
    assert win._job_rows == [win._rows[unscored_row]]  # jobs track RowData identity, not row index
    assert "(1 already scored, not recalculated)" in win.status_label.text()

    win._worker.cancel()
    win._worker.wait(5000)


# ------------------------------------------------------------------ per-video settings panel


def test_editing_with_multiple_rows_selected_applies_to_all_of_them(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))
    row_c = win._add_table_row(Path("c.mp4"))

    win.distorted_table.setRangeSelected(
        QTableWidgetSelectionRange(row_a, 0, row_b, win.distorted_table.columnCount() - 1), True,
    )
    win.crop_combo.setCurrentIndex(1)

    assert win._rows[row_a].options.crop_mode == CropMode.NONE
    assert win._rows[row_b].options.crop_mode == CropMode.NONE
    assert win._rows[row_c].options.crop_mode == CropMode.AUTO


def test_multi_row_edit_preserves_each_rows_unrelated_settings(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))
    win._rows[row_a].options.n_threads = 2
    win._rows[row_a].options.gpu_decode = False
    win._rows[row_b].options.n_threads = 11
    win._rows[row_b].options.gpu_decode = True
    win.distorted_table.setRangeSelected(
        QTableWidgetSelectionRange(
            row_a, 0, row_b, win.distorted_table.columnCount() - 1
        ),
        True,
    )

    win.crop_combo.setCurrentIndex(1)

    assert win._rows[row_a].options.crop_mode == CropMode.NONE
    assert win._rows[row_b].options.crop_mode == CropMode.NONE
    assert win._rows[row_a].options.n_threads == 2
    assert win._rows[row_b].options.n_threads == 11
    assert win._rows[row_a].options.gpu_decode is False
    assert win._rows[row_b].options.gpu_decode is True


def test_selecting_a_row_loads_its_own_settings_into_the_panel(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))

    win.distorted_table.selectRow(row_a)
    win.crop_combo.setCurrentIndex(1)  # row_a -> None
    win.distorted_table.selectRow(row_b)
    win.crop_combo.setCurrentIndex(0)  # row_b -> Auto (already default, but exercises the write path)

    win.distorted_table.selectRow(row_a)
    assert win.crop_combo.currentIndex() == 1  # panel reflects row_a's own stored setting, not row_b's

    win.distorted_table.selectRow(row_b)
    assert win.crop_combo.currentIndex() == 0


# ------------------------------------------------------------------ reopening the graph window


def test_show_graph_adds_every_scored_row_and_switches_to_the_tab(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].completed_run = _fake_completed_run(name)

    assert len(win.graph_panel._entries) == 0
    win._on_show_graph_clicked()

    assert win.tabs.currentIndex() == TAB_GRAPH
    assert len(win.graph_panel._entries) == 2


# ------------------------------------------------------------------ column resizing


def test_black_bars_column_answers_yes_or_no_about_the_test_video(qapp):
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 3840, 2160)
    distorted = _fake_video_info_res("encode.mp4", 1920, 804)
    win._source_info = source
    row = win._add_table_row(distorted.path)
    result = ComparisonResult(
        source=source.path,
        distorted=distorted.path,
        frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)],
        fps=30.0,
        model="version=vmaf_4k_v0.6.1",
        source_crop=CropBox(w=3840, h=1608, x=0, y=276),
        distorted_crop=CropBox(w=1920, h=804, x=0, y=0),
        source_info=source,
        distorted_info=distorted,
    )
    win._rows[row].completed_run = CompletedRun(result, "encode")

    win._set_row_metrics(row)

    # The cell answers only for the test video, which here is already cropped.
    item = win.distorted_table.item(row, COL_BLACK_BARS)
    assert item.text() == "No"
    # The reference's bars, and every pixel count, are on hover.
    assert "Reference: black bars cropped off -- top 276 px, bottom 276 px." in item.toolTip()
    assert "Compared at 3840x1608 instead of 3840x2160." in item.toolTip()
    assert "Test video: no black bars. Compared in full at 1920x804." in item.toolTip()

    win._rows[row].options.crop_mode = CropMode.NONE
    win._invalidate_completed_result(row)
    assert win.distorted_table.item(row, COL_BLACK_BARS).text() == "Off"


def test_resize_mismatch_note_reflects_the_actual_scale_direction_used(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))
    tag = win.distorted_table.item(row, COL_SCALING).text()
    tip = win.distorted_table.item(row, COL_SCALING).toolTip()
    assert tag == "↓ source"
    assert "Reference downscaled" in tip
    assert "3840x2160" in tip and "1920x1080" in tip
    # and it must NOT bloat the Media info column any more
    assert "downscaled" not in win.distorted_table.item(row, COL_INFO).text()

    # A completed run recorded as the *other* direction should override the
    # row's current (unrelated) settings when describing what happened.
    frames = [FrameScore(frame=0, time=0.0, vmaf=90.0)]
    result = ComparisonResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"), frames=frames, fps=30.0,
        model="m", source_crop=None, distorted_crop=None,
        source_info=win._source_info, distorted_info=_fake_video_info_res("a.mp4", 1920, 1080),
        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
    )
    win._rows[row].completed_run = CompletedRun(result, "a")
    win._set_row_info(row, result.distorted_info)
    assert win.distorted_table.item(row, COL_SCALING).text() == "Test upscaled to source"


# ------------------------------------------------------------------ persistent result cache

def test_loaded_saved_run_shows_every_metric_present_in_the_file(qapp, monkeypatch):
    win = MainWindow()
    info = _fake_video_info("saved.mp4")
    result = ComparisonResult(
        source=Path("source.mp4"), distorted=Path("saved.mp4"),
        frames=[
            FrameScore(0, 0.0, 90.0, psnr=42.0, ssim=0.9876, xpsnr=39.0),
            FrameScore(1, 1 / 30, 92.0, psnr=44.0, ssim=0.9890, xpsnr=41.0),
        ],
        fps=30.0, model="version=vmaf_v0.6.1",
        source_crop=CropBox(w=1920, h=1080, x=0, y=0),
        distorted_crop=CropBox(w=1920, h=1080, x=0, y=0),
        source_info=info, distorted_info=info,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        lambda *a, **kw: ("saved.metrics.json", ""),
    )
    monkeypatch.setattr(main_window_module, "load_run", lambda _path: (result, "saved"))

    win._on_load_saved_run()

    assert win.distorted_table.item(0, COL_PSNR).text() == "43.00"
    assert win.distorted_table.item(0, COL_SSIM).text() == "0.9883"
    # 39.94, not the arithmetic 40.00: XPSNR aggregates as a square-mean-root
    # (ffmpeg's own sequence average), which leans towards the worse frame.
    assert win.distorted_table.item(0, COL_XPSNR).text() == "39.94"
    assert win.distorted_table.item(0, COL_BLACK_BARS).text() == "No"


def test_clear_cache_never_deletes_unrelated_json_files(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache

    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    app_result = tmp_path / "v2" / "comparison" / "context.json"
    app_result.parent.mkdir(parents=True)
    app_result.write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "family_budget.json"
    unrelated.write_text("important", encoding="utf-8")
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **kw: main_window_module.QMessageBox.Yes,
    )

    win = MainWindow()
    win._on_clear_cache()
    assert win._file_writes.wait_until_idle(10.0)

    assert not app_result.exists()
    assert unrelated.read_text(encoding="utf-8") == "important"


def test_adding_a_row_picks_up_a_cached_result(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source

    cached_result = _fake_completed_run(str(distorted)).result
    cached_result.source = source
    cached_result.distorted = distorted
    cached_result.source_crop = CropBox(w=1920, h=1080, x=0, y=0)
    cached_result.distorted_crop = CropBox(w=1920, h=1080, x=0, y=0)
    _store_cached(
        source, distorted, cached_result, label="cached-label",
        options=win._rows[0].options if win._rows else win._default_options,
    )

    row = win._add_table_row(distorted)
    win._set_row_info(row, win._rows[row].video_info or _fake_video_info(str(distorted)))
    applied = win._try_load_cached_result(row)

    assert applied is True
    assert win._rows[row].completed_run is not None
    assert win._rows[row].completed_run.label == "cached-label"
    assert win.distorted_table.item(row, COL_BLACK_BARS).text() == "No"


def test_finishing_a_job_persists_to_cache(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._job_rows = [win._rows[row]]

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    win._on_job_finished(0, result)
    # The cache write runs on a background thread now, so the assertion has
    # to wait for it rather than assuming it happened inline.
    assert win._file_writes.wait_until_idle(10.0)

    assert _load_cached(source, distorted, win._rows[row].options) is not None


def test_finishing_an_old_job_cannot_attach_or_cache_it_under_a_new_source(
    qapp, tmp_path, monkeypatch
):
    from vmaf_app.core import result_cache

    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    old_source = tmp_path / "old" / "old-source.mp4"
    new_source = tmp_path / "new" / "new-source.mp4"
    distorted = tmp_path / "distorted.mp4"
    for path, data in ((old_source, b"old"), (new_source, b"new"), (distorted, b"dist")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    win = MainWindow()
    win._source_info = _fake_video_info(str(new_source))
    row = win._add_table_row(distorted)
    win._job_rows = [win._rows[row]]
    result = _fake_completed_run(str(distorted)).result
    result.source = old_source
    result.distorted = distorted

    win._on_job_finished(0, result)
    assert win._file_writes.wait_until_idle(10.0)

    assert win._rows[row].completed_run is None
    assert _load_cached(old_source, distorted, win._rows[row].options) is not None
    assert _load_cached(new_source, distorted, win._rows[row].options) is None


def test_recompute_clears_row_and_deletes_cache_entry(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._rows[row].completed_run = _fake_completed_run(str(distorted))
    _store_cached(
        source, distorted, win._rows[row].completed_run.result, label="x",
        options=win._rows[row].options,
    )

    win._recompute_rows([row])
    assert win._file_writes.wait_until_idle(10.0)

    assert win._rows[row].completed_run is None
    assert _load_cached(source, distorted, win._rows[row].options) is None


# ------------------------------------------------------------------ resolution round-trip test row


def test_add_resample_test_creates_a_row_with_the_chosen_target(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("1080p", True))

    win._on_add_resample_test()

    assert len(win._rows) == 1
    row_data = win._rows[0]
    assert row_data.options.resample_test == ResampleTarget(width=1920, label="1080p")
    assert row_data.video_info is win._source_info
    assert row_data.path == synthetic_resample_distorted_path(
        Path("source.mp4"), ResampleTarget(width=1920, label="1080p")
    )


def test_run_clicked_builds_a_job_for_a_resample_row_without_probing(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("480p", True))
    win._on_add_resample_test()

    win._on_run_clicked()

    assert win._worker is not None
    assert len(win._worker.scheduler.jobs) == 1
    assert win._worker.scheduler.jobs[0].options.resample_test == ResampleTarget(width=854, label="480p")

    win._worker.cancel()
    win._worker.wait(5000)


# ------------------------------------------------------------------ opposite scale-direction rows

def _fake_video_info_res(name: str, width: int, height: int) -> VideoInfo:
    return VideoInfo(
        path=Path(name), width=width, height=height, fps=30.0, duration=5.0,
        nb_frames=150, codec_name="h264",
    )


def test_add_opposite_scale_direction_adds_a_row_with_the_flipped_direction(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)

    win._add_opposite_scale_direction_rows([row])

    assert len(win._rows) == 2
    new_row = win._rows[1]
    assert new_row.options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE
    assert new_row.video_info is win._rows[0].video_info
    assert new_row.path == synthetic_scale_direction_variant_path(Path("a.mp4"), ScaleDirection.DISTORTED_TO_SOURCE)
    # The original row's own direction/options are untouched.
    assert win._rows[0].options.scale_direction == ScaleDirection.SOURCE_TO_DISTORTED


def test_run_clicked_gives_the_opposite_direction_row_a_distinct_result_identity(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    win._add_opposite_scale_direction_rows([row])

    win._on_run_clicked()

    assert win._worker is not None
    jobs_by_direction = {j.options.scale_direction: j for j in win._worker.scheduler.jobs}
    assert jobs_by_direction[ScaleDirection.SOURCE_TO_DISTORTED].result_distorted_path == Path("a.mp4")
    assert jobs_by_direction[ScaleDirection.DISTORTED_TO_SOURCE].result_distorted_path == synthetic_scale_direction_variant_path(
        Path("a.mp4"), ScaleDirection.DISTORTED_TO_SOURCE
    )

    win._worker.cancel()
    win._worker.wait(5000)


# ------------------------------------------------------------------ rows removed mid-run

def test_removing_a_row_mid_run_still_lands_results_on_the_right_row(qapp):
    # Regression test: jobs used to be tracked by table row *index*. Removing
    # a row mid-run shifts every later index down by one, so a finishing job
    # wrote its result onto the wrong file's row -- or crashed with
    # IndexError once the shifted index ran past the end of the list.
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        r = win._add_table_row(Path(name))
        win._rows[r].video_info = _fake_video_info(name)
    win._job_rows = list(win._rows)
    win._checked_rows_for_run = list(win._rows)

    win.distorted_table.selectRow(0)
    win._on_remove_distorted()  # drop "a.mp4" while the run is in flight

    win._on_job_finished(2, _fake_completed_run("c.mp4").result)  # job 2 == c.mp4's row

    by_name = {rd.path.name: rd for rd in win._rows}
    assert by_name["c.mp4"].completed_run is not None  # landed on the right row
    assert by_name["b.mp4"].completed_run is None      # and not on its neighbour


def test_closing_mid_run_cancels_the_worker(qapp, monkeypatch):
    # Closing the window while ffmpeg is running used to leave the
    # subprocess alive in the background and tear down a live QThread.
    win = MainWindow()
    cancelled = []

    class FakeWorker:
        def isRunning(self):
            return True

        def cancel(self):
            cancelled.append(True)

        def wait(self, ms):
            return True

    win._worker = FakeWorker()
    win.close()

    assert cancelled == [True]


# ------------------------------------------------------------------ fps / ETA display

def test_job_progress_shows_fps_and_file_eta(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._job_rows = [win._rows[row]]
    win._on_job_started(0, "a")

    win._on_job_progress(0, current=1000, total=3000, fps=100.0)

    # A file's own rate and remaining time belong on that file's line.
    # 2000 frames remaining at 100fps -> 20s
    assert "100.0 fps" in win.job_progress_labels[0].text()
    assert "0:00:20 remaining" in win.job_progress_labels[0].text()
    # Live progress belongs in the status bar above, not the VMAF column --
    # that's reserved for the final score (or "Failed").
    cell = win.distorted_table.item(row, COL_VMAF)
    assert cell.text() == ""
    assert cell.checkState() == Qt.Checked  # still just "will be calculated"


def test_cancelled_run_does_not_claim_done(qapp):
    win = MainWindow()
    win._on_run_cancelled()

    win._on_all_finished()

    assert win.status_label.text() == "Cancelled."


def test_failed_run_reports_failure_instead_of_done(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("bad.mp4"))
    win._job_rows = [win._rows[row]]

    win._on_job_failed(0, "ffmpeg failed", "details")
    win._on_all_finished()

    assert win.status_label.text() == "Finished: 1 video failed (hover over its name for why)."
    assert "Done" not in win.status_label.text()


def test_live_run_disables_inputs_that_can_change_the_jobs(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    win._set_run_ui_active(True)

    assert all(not w.isEnabled() for w in win._file_action_widgets)
    assert not win.options_box.isEnabled()
    assert not win.tabs.isTabEnabled(TAB_SETTINGS)
    assert not win.run_btn.isEnabled()
    assert win.pause_btn.isEnabled()
    assert win.cancel_btn.isEnabled()

    win._set_run_ui_active(False)
    assert all(w.isEnabled() for w in win._file_action_widgets)
    assert win.options_box.isEnabled()
    assert win.tabs.isTabEnabled(TAB_SETTINGS)


# ------------------------------------------------------------------ metric columns


def test_clicking_a_metric_tick_box_selects_that_metric_for_the_row(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))

    # Clicking a row that is not selected applies to that row alone.
    win.distorted_table.item(0, COL_PSNR).setCheckState(Qt.Unchecked)
    assert "psnr" not in win._rows[0].options.requested_metrics()
    assert "psnr" in win._rows[1].options.requested_metrics()
    # ...and does not become the default for files added later.
    assert "psnr" in win._default_options.requested_metrics()

    # Clicking one of several selected rows applies to all of them.
    win.distorted_table.selectAll()
    win.distorted_table.item(1, COL_XPSNR).setCheckState(Qt.Unchecked)
    assert all("xpsnr" not in r.options.requested_metrics() for r in win._rows)


def test_ticking_a_metric_column_header_enables_it_for_every_row(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))

    win._on_metric_column_toggled(COL_PSNR, False)

    assert all("name=psnr" not in r.options.extra_features for r in win._rows)
    assert all(
        win.distorted_table.item(r, COL_PSNR).checkState() == Qt.Unchecked for r in range(2)
    )
    # The header is a statement about the table, so new files inherit it.
    assert "psnr" not in win._default_options.requested_metrics()

    win._on_metric_column_toggled(COL_PSNR, True)
    assert all("name=psnr" in r.options.extra_features for r in win._rows)
    assert win.distorted_table.item(0, COL_PSNR).checkState() == Qt.Checked


def test_adding_a_metric_retains_scores_and_marks_result_partial(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win.graph_panel.add_run(win._rows[row].completed_run.result, "a")

    win._on_metric_column_toggled(COL_PSNR, True)

    assert win._rows[row].completed_run is not None
    assert not win._has_requested_results(win._rows[row])
    assert win._row_state(win._rows[row]) == "Partially calculated"
    assert win.distorted_table.item(row, COL_VMAF).text() == "90.00"
    assert win.graph_panel._entries


def test_changing_a_calculation_option_marks_an_existing_result_stale(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    win.subsample_spin.setValue(5)

    assert win._rows[row].completed_run is None


@pytest.mark.parametrize("control", ["gpu", "threads", "scaling algorithm"])
def test_execution_only_option_change_keeps_an_existing_result(qapp, control):
    """Which GPU, how many threads, and how frames are scaled change how a
    comparison is made, not what it is: its scores stay."""
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    completed = _fake_completed_run("a.mp4")
    win._rows[row].completed_run = completed
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    if control == "gpu":
        win.gpu_checkbox.setChecked(not win.gpu_checkbox.isChecked())
    elif control == "threads":
        win.threads_spin.setValue(7)
    else:
        win.scale_algo_combo.setCurrentText("lanczos")
        assert win._rows[row].options.scale_algorithm == "lanczos"

    assert win._rows[row].completed_run is completed


def test_selecting_a_slow_source_does_not_block_the_ui(qapp, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    on_ui_thread = []

    def slow_probe(path, process_handle=None):
        on_ui_thread.append(threading.current_thread() is threading.main_thread())
        started.set()
        if not on_ui_thread[-1]:
            release.wait(10.0)  # a slow probe, where it does not hold up the window
        return _fake_video_info(str(path))

    monkeypatch.setattr(probe_worker_module, "probe_video", slow_probe)
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: ("slow-source.mp4", "")),
    )
    win = MainWindow()

    win._on_browse_source()

    assert started.wait(5.0)
    assert on_ui_thread == [False], "the source was probed on the UI thread"
    assert win._source_info is None

    release.set()
    deadline = time.monotonic() + 10.0
    while win._source_probe_worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert win._source_info is not None
    assert win.source_edit.text() == "slow-source.mp4"


def test_metric_columns_show_each_metrics_own_mean(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._on_metric_column_toggled(COL_PSNR, True)
    win._on_metric_column_toggled(COL_SSIM, True)
    win._on_metric_column_toggled(COL_XPSNR, True)

    info = _fake_video_info("a.mp4")
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0, psnr=42.0, ssim=0.95, xpsnr=38.0) for i in range(4)]
    result = ComparisonResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"), frames=frames, fps=30.0,
        model="m", source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )
    win._rows[row].completed_run = CompletedRun(result, "a")
    win._set_row_metrics(row)

    assert win.distorted_table.item(row, COL_VMAF).text() == "90.00"
    assert win.distorted_table.item(row, COL_PSNR).text() == "42.00"
    assert win.distorted_table.item(row, COL_SSIM).text() == "0.9500"  # SSIM needs more decimals to be useful
    assert win.distorted_table.item(row, COL_XPSNR).text() == "38.00"


def test_vmaf_compute_is_each_videos_choice_and_keeps_its_scores(qapp, monkeypatch):
    """Performance > VMAF compute, beside SSIMULACRA2's and
    Butteraugli's: per video, what new videos and the next session start
    with, and execution only -- the GPU's VMAF agrees with the CPU's to
    within a thousandth of a point, so a video keeps the scores it has."""
    from vmaf_app.core.models import GpuVendor

    monkeypatch.setattr(main_window_module, "detected_gpu_vendors", lambda: [GpuVendor.NVIDIA])
    win = MainWindow()
    combo = win.vmaf_backend_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["GPU", "CPU"]
    assert combo.currentText() == "GPU" and combo.isEnabledTo(win.options_box)
    assert "VMAF v1: its detail and motion features on the GPU" in combo.toolTip()

    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    info = _fake_video_info("a.mp4")
    result = ComparisonResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"),
        frames=[FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(4)], fps=30.0,
        model="m", source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )
    win._rows[row].completed_run = CompletedRun(result, "a")
    win._panel_target_rows = [row]
    combo.setCurrentIndex(1)

    assert win._rows[row].options.vmaf_on_gpu is False
    assert win._rows[row].completed_run is not None  # its scores stay
    assert win._rows[win._add_table_row(Path("b.mp4"))].options.vmaf_on_gpu is False
    assert Settings.load().default_vmaf_on_gpu is False
    win.close()
    win = MainWindow()  # the next session
    assert win._default_options.vmaf_on_gpu is False
    win.close()


# ------------------------------------------------------------------ ffmpeg/ffprobe startup check

def test_missing_tools_show_an_actionable_banner(qapp, monkeypatch):
    # The "Locate ffmpeg" button used to be built into a layout that was
    # never attached to anything, so the warning banner had no way to act on.
    from vmaf_app.core.ffmpeg_locate import ToolsStatus, ToolStatus

    broken = ToolsStatus(
        ffmpeg=ToolStatus("ffmpeg", "ffmpeg.exe", False, None, "not found"),
        ffprobe=ToolStatus("ffprobe", "ffprobe.exe", False, None, "not found"),
    )
    monkeypatch.setattr(main_window_module, "check_tools", lambda: broken)

    win = MainWindow()

    # Both the banner and its button must be real, laid-out children -- the
    # button previously existed only as a local never added to any layout.
    assert win._locate_ffmpeg_btn.parent() is not None
    assert win._ffmpeg_banner.parent() is not None
    assert win._locate_ffmpeg_btn.isVisibleTo(win) is True
    assert win._ffmpeg_banner.isVisibleTo(win) is True
    assert "ffmpeg" in win._ffmpeg_banner.text()
    assert "ffprobe" in win._ffmpeg_banner.text()


# ------------------------------------------------------------------ graph stays in step


def test_removing_a_video_removes_its_curve(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].completed_run = _fake_completed_run(name)
    win.tabs.setCurrentIndex(TAB_GRAPH)
    assert len(win.graph_panel._entries) == 2

    win.distorted_table.selectRow(0)
    win._on_remove_distorted()
    assert len(win._rows) == 1
    assert len(win.graph_panel._entries) == 1, "the removed video's curve must go too"


def test_remove_all_clears_the_table_and_the_graph(qapp, monkeypatch):
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: main_window_module.QMessageBox.Yes,
    )
    win = MainWindow()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].completed_run = _fake_completed_run(name)
    win.tabs.setCurrentIndex(TAB_GRAPH)
    assert len(win.graph_panel._entries) == 3

    win._on_remove_all_distorted()
    assert win._rows == []
    assert win.distorted_table.rowCount() == 0
    assert len(win.graph_panel._entries) == 0


# ------------------------------- queued cache operations pin their directory

def _two_cache_dirs(tmp_path):
    a, b = tmp_path / "cache_a", tmp_path / "cache_b"
    a.mkdir()
    b.mkdir()
    return a, b


def _block_writes(win):
    """Holds the write queue open so a setting can change mid-flight."""
    import threading
    release = threading.Event()
    win._file_writes.submit("blocker", release.wait)
    return release


def test_clearing_the_cache_deletes_the_folder_the_dialog_named(qapp, tmp_path, monkeypatch):
    # The confirmation dialog names a folder. Deleting a different one than
    # the user was shown is not something to leave to timing.
    from vmaf_app.core import result_cache

    folder_a, folder_b = _two_cache_dirs(tmp_path)
    (folder_a / "v2" / "one").mkdir(parents=True)
    (folder_b / "v2" / "two").mkdir(parents=True)
    (folder_a / "v2" / "one" / "context.json").write_text("{}", encoding="utf-8")
    (folder_b / "v2" / "two" / "context.json").write_text("{}", encoding="utf-8")

    win = MainWindow()
    result_cache.set_cache_dir_override(folder_a)
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: main_window_module.QMessageBox.Yes,
    )
    release = _block_writes(win)

    win._on_clear_cache()
    result_cache.set_cache_dir_override(folder_b)
    release.set()
    assert win._file_writes.wait_until_idle(10.0)

    assert not list(folder_a.glob("v2/*/context.json")), "the named folder was not cleared"
    assert list(folder_b.glob("v2/*/context.json")), "an unnamed folder was cleared instead"


# --------------- resolution tests belong to the source they were added for

def _add_resolution_test(win, monkeypatch, label="1440p"):
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getItem", lambda *a, **k: (label, True)
    )
    win._on_add_resample_test()


def test_changing_the_source_removes_its_resolution_tests(qapp, tmp_path, monkeypatch):
    """A resolution test downscales and re-upscales THE SOURCE -- it has no
    distorted file of its own. Its synthetic path, media info, description
    and identity all come from the source selected when it was added, so a
    2560-wide "downscale" test added for a 3840-wide master becomes an
    UPSCALE against a 1280-wide one: a measurement the test never meant to
    make, on a row still describing the old source.
    """
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    for path in (big, small):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    _add_resolution_test(win, monkeypatch, "1440p")
    assert len(win._rows) == 1
    assert win._rows[0].options.resample_test.width == 2560

    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **k: None)
    win._apply_source_info(small, _fake_video_info_res(str(small), 1280, 720))

    assert win._rows == [], "a 2560 downscale test survived a move to a 1280 source"


# --------------------- a loaded run belongs to the source it was measured on

def _saved_run_file(tmp_path, source, distorted, name="run.metrics.json"):
    from vmaf_app.core.run_io import save_run

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    result.source_info = _fake_video_info(str(source))
    result.source_info.path = source
    path = tmp_path / name
    save_run(result, path, label=distorted.stem)
    return path


def _load_saved(win, monkeypatch, run_file):
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(run_file), "")),
    )
    win._on_load_saved_run()


def test_loading_a_run_with_no_source_selected_adopts_its_source(qapp, tmp_path, monkeypatch):
    """A result belongs to a (source, distorted) PAIR. With nothing to
    contradict, the window takes the run's own reference so the two cannot
    disagree about what was compared."""
    source = tmp_path / "sourceA.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)
    run_file = _saved_run_file(tmp_path, source, distorted)

    win = MainWindow()
    assert win._source_info is None
    _load_saved(win, monkeypatch, run_file)

    assert win._source_info is not None
    assert win._same_source(win._source_info.path, source)
    assert str(source) in win.source_edit.text()
    assert win._rows[0].completed_run is not None


def test_a_run_from_a_different_source_is_not_silently_shown_under_this_one(
    qapp, tmp_path, monkeypatch
):
    # The reported defect: the source field stayed on B while the row
    # displayed A's score underneath it.
    source_a = tmp_path / "sourceA.mp4"
    source_b = tmp_path / "sourceB.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source_a, source_b, distorted):
        path.write_bytes(b"x" * 100)
    run_file = _saved_run_file(tmp_path, source_a, distorted)

    win = MainWindow()
    win._apply_source_info(source_b, _fake_video_info(str(source_b)))
    asked = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: asked.append(a) or main_window_module.QMessageBox.No,
    )
    _load_saved(win, monkeypatch, run_file)

    assert asked, "the mismatch was not raised with the user"
    assert win._rows == [], "the run was added under the wrong source anyway"
    assert win._same_source(win._source_info.path, source_b), "the source changed uninvited"


# --------------- Frame Compare works before any metric has been calculated

def test_frame_compare_offers_a_probed_row_with_no_scores(qapp, tmp_path):
    """A source and a distorted video that have merely been read are enough
    to compare frames; requiring a finished run first made the tab useless
    for deciding whether a comparison is worth running at all."""
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mkv", 3840, 2160)
    row = win._add_table_row(Path("encode.mkv"))
    win._rows[row].video_info = _fake_video_info_res("encode.mkv", 1920, 1080)

    win._sync_frame_compare()

    entries = win.frame_compare_panel._entries
    assert len(entries) == 1
    assert entries[0].scores is None, "a row with no run must not claim scores"
    assert entries[0].comparison.source_info.width == 3840
    assert entries[0].comparison.distorted_info.width == 1920


# ------------------------------ progress and status with two videos running

def test_each_running_video_gets_its_own_progress_line(qapp):
    """Two videos at once need two readable lines, not one bar flickering
    between them."""
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)

    win._on_job_started(0, "a")
    win._on_job_started(1, "b")

    assert win.job_progress_labels[0].isVisibleTo(win)
    assert win.job_progress_labels[1].isVisibleTo(win)

    win._on_job_progress(0, current=250, total=1000, fps=25.0)
    win._on_job_progress(1, current=750, total=1000, fps=50.0)

    # The percentage is stated, not drawn.
    assert "a — 25%" in win.job_progress_labels[0].text()
    assert "b — 75%" in win.job_progress_labels[1].text()
    assert "25.0 fps" in win.job_progress_labels[0].text()


# ------------------------------------------------ long CPU perceptual metrics

def _long_row(win, name, *, minutes, backend="cpu"):
    win._source_info = VideoInfo(path=Path("source.mkv"), width=3840, height=2160, fps=24.0,
                                 duration=minutes * 60, nb_frames=minutes * 60 * 24, codec_name="hevc")
    row = win._add_table_row(Path(name))
    win._rows[row].video_info = VideoInfo(path=Path(name), width=3840, height=2160, fps=24.0,
                                          duration=minutes * 60, nb_frames=minutes * 60 * 24,
                                          codec_name="hevc")
    win._rows[row].extra_metric_keys.add("ssimulacra2")
    win._rows[row].metric_backends["ssimulacra2"] = backend
    return row


def _answer_warning(monkeypatch, answer):
    shown = []

    def warning(_parent, title, text, *args):
        shown.append((title, text))
        return answer

    monkeypatch.setattr(main_window_module.QMessageBox, "warning", warning)
    return shown


def test_cpu_perceptual_on_a_long_video_asks_first_and_no_means_no_run(qapp, monkeypatch):
    win = MainWindow()
    _long_row(win, "film.mkv", minutes=105)
    shown = _answer_warning(monkeypatch, main_window_module.QMessageBox.No)

    win._on_run_clicked()

    assert win._worker is None
    (_title, text), = shown
    assert "not recommended" in text and "film.mkv" in text and "SSIMULACRA2" in text
    assert "days of scoring" in text, "a 105-minute 4K film takes days on the CPU"
    win.close()


# ---------------------------------------------------------- metrics picker

def test_hiding_a_metric_hides_its_column_and_leaves_it_out_of_runs(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    assert "xpsnr" in win._requested_metrics(win._rows[row])

    win._on_metric_visibility_toggled("xpsnr", False)

    assert win.distorted_table.isColumnHidden(COL_XPSNR)
    assert "xpsnr" not in win._requested_metrics(win._rows[row])
    # The row's own tick is kept, so showing the metric restores the choice.
    assert "xpsnr" in win._selected_metrics(win._rows[row])
    assert win._settings.hidden_metrics == ["xpsnr"]
    assert json.loads(Settings.path().read_text(encoding="utf-8"))["hidden_metrics"] == ["xpsnr"]

    win._on_metric_visibility_toggled("xpsnr", True)

    assert not win.distorted_table.isColumnHidden(COL_XPSNR)
    assert "xpsnr" in win._requested_metrics(win._rows[row])
    win.close()


def test_a_row_set_to_cpu_does_not_show_a_cached_gpu_score(qapp, tmp_path, monkeypatch):
    """The app's own lookups carry each row's GPU/CPU choice."""
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet

    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    source, distorted = tmp_path / "source.mp4", tmp_path / "test.mp4"
    source.write_bytes(b"s" * 100)
    distorted.write_bytes(b"d" * 50)
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    rd = win._rows[row]
    win._set_row_info(row, _fake_video_info(str(distorted)))
    for key in ("vmaf", "psnr", "ssim", "xpsnr"):
        rd.options.set_metric_enabled(key, False)
    rd.extra_metric_keys = {"ssimulacra2"}
    rd.metric_backends["ssimulacra2"] = "cpu"
    gpu = MetricProvenance("Vship/ssimulacra2", "5.1.1", "gpu", "ssimulacra2-vship-gpu-v1")
    info = rd.video_info
    result = ComparisonResult(
        source=source, distorted=distorted, frames=[], fps=30.0, model="",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
        compared_frame_count=1,
        metric_results=MetricResultSet([FrameMetricResult("ssimulacra2", [0], [0.0], [44.47], gpu)]),
    )
    result_cache.store(source, distorted, result, "gpu run",
                       analysis_request_from_vmaf_options(rd.options, ("ssimulacra2",)))

    assert not win._try_load_cached_result(row)

    rd.metric_backends["ssimulacra2"] = "gpu"
    assert win._try_load_cached_result(row)
    win.close()


def test_highlight_best_worst_colours_each_metrics_best_and_worst_score(qapp):
    """A user's request, as FFMetrics colours its table: ticked, each
    metric's best score is green and its worst red among the test videos --
    by the metric's direction (Butteraugli's lowest is its best), only where
    two or more have one, and not a score marked as the other
    implementation's. Off by default, and kept in the settings."""
    from tests.factories import fake_run_result
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
    from vmaf_app.ui import theme

    vmaf_column, butteraugli_column = COL_VMAF, main_window_module.COL_BUTTERAUGLI
    win = MainWindow()
    assert not win.highlight_check.isChecked()
    gpu = MetricProvenance("Vship/butteraugli", "Vship 5.1.2", "gpu", "butteraugli-vship-gpu-v1")
    for name, vmaf, butteraugli in (("a.mp4", 90.0, 1.2), ("b.mp4", 95.0, 0.9), ("c.mp4", 85.0, 2.0)):
        row = win._add_table_row(Path(name))
        row_data = win._rows[row]
        row_data.video_info = _fake_video_info(name)
        result = fake_run_result(name, vmaf=vmaf)
        result.merge_metric_results(MetricResultSet([FrameMetricResult("butteraugli", [0], [0.0], [butteraugli], gpu)]))
        row_data.completed_run = CompletedRun(result, name)
        row_data.metric_backends["butteraugli"] = "gpu"
        win._set_row_metrics(row)

    def marks(column):
        names = {theme.color("best").name(): "best", theme.color("worst").name(): "worst"}
        qapp.processEvents()  # the coalesced pass (_schedule_best_worst)
        brushes = [win.distorted_table.item(row, column).background() for row in range(win.distorted_table.rowCount())]
        return [names.get(brush.color().name()) if brush.color().alpha() else None for brush in brushes]

    assert marks(vmaf_column) == [None, None, None]
    win.highlight_check.setChecked(True)
    assert Settings.load().highlight_best_worst
    assert marks(vmaf_column) == [None, "best", "worst"]
    assert marks(butteraugli_column) == [None, "best", "worst"]
    # b.mp4 set to CPU: its GPU score shows "(GPU)", on another scale.
    win._rows[1].metric_backends["butteraugli"] = "cpu"
    win._set_row_metrics(1)
    assert marks(butteraugli_column) == ["best", None, "worst"]
    # The best video removed: the next is the best of the two left.
    win.distorted_table.selectRow(1)
    win._on_remove_distorted()
    assert marks(vmaf_column) == ["best", "worst"]
    # One video left: nothing to compare.
    win.distorted_table.selectRow(1)
    win._on_remove_distorted()
    assert marks(vmaf_column) == [None]
    win.highlight_check.setChecked(False)
    assert not Settings.load().highlight_best_worst
    win.close()


def test_a_partly_failed_job_shows_and_caches_the_metrics_that_finished(qapp, tmp_path, monkeypatch):
    """VMAF finished, SSIMULACRA2 failed: the row shows VMAF, its
    SSIMULACRA2 cell says Failed with the reason on the row, VMAF is cached,
    and the run counts the video as failed."""
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    rd = win._rows[row]
    rd.extra_metric_keys.add("ssimulacra2")
    win._job_rows = [rd]
    win._run_failed_count = 0
    result = _fake_completed_run(str(distorted)).result
    result.source, result.distorted = source, distorted

    win._on_job_partially_failed(0, result, "SSIMULACRA2 failed: unsupported input", "tail")

    assert win._file_writes.wait_until_idle(10.0)
    assert rd.completed_run is not None and rd.completed_run.result.has_metric("vmaf")
    assert win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2).text() == "Failed"
    assert win.distorted_table.item(row, main_window_module.COL_VMAF).text() != "Failed"
    assert rd.analysis_status == "Partly failed"
    assert "SSIMULACRA2 failed: unsupported input" in rd.status_detail
    assert win._run_partial_count == 1 and win._run_failed_count == 0
    assert _load_cached(source, distorted, rd.options) is not None
    win.close()


def test_the_end_of_a_run_says_how_long_it_took_and_what_failed(qapp, clock):
    """It said "Done." with the time gone, or counted a video with one
    failed metric among scored ones as a failed video."""
    win = MainWindow()
    win._run_started_at = clock.now - 3725
    win._run_failed_count, win._run_partial_count = 1, 2
    win._on_all_finished()
    assert win.status_label.text() == (
        "Finished in 1:02:05: 1 video failed, 2 with some metrics failed (hover over their names for why).")
    win._run_failed_count = win._run_partial_count = 0
    win._on_all_finished()
    assert win.status_label.text() == "Done in 1:02:05."
    win._run_was_cancelled = True
    win._on_all_finished()
    assert win.status_label.text() == "Cancelled after 1:02:05."
    win.close()


def _half(backend, keys, *, state="running", decode="", current=20, total=100, fps=10.0, phase=None,
          passes=None, cpu_keys=(), done_keys=(), lane=None):
    """A half's snapshot as the worker sends it (VmafWorker.task_progress)."""
    return {"backend": backend, "metric_keys": keys, "current": current, "total": total, "fps": fps,
            "state": state, "phase": phase, "waiting_for": None, "step": "",
            "decode": decode_plan(decode) if decode else None,
            "lane": lane or ("cpu" if backend in ("ffmpeg", "perceptual_cpu") else "gpu"),
            "passes": passes or (keys,), "cpu_keys": cpu_keys, "done_keys": done_keys}


def test_the_pc_is_kept_awake_exactly_while_a_run_is_active(qapp, monkeypatch):
    """A run can take hours; a PC set to sleep after some idle time slept
    under it, and Modern Standby suspends desktop apps once asleep."""
    from vmaf_app.ui import main_window as main_window_module

    calls = []
    monkeypatch.setattr(main_window_module, "keep_system_awake", lambda awake: calls.append(awake) or True)
    win = MainWindow()
    assert calls == []
    win._set_run_ui_active(True)
    assert calls == [True]
    win._on_all_finished()  # the run's end, however it ended
    assert calls == [True, False]
    win.close()


def test_a_failed_metrics_cell_says_why_it_failed(qapp, tmp_path, monkeypatch):
    """Every red Failed cell said "This metric failed on the last run. Untick
    to skip it." -- the reason was only on the file name's tooltip."""
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    # A GPU Vship can use: without one, CVVDP's cell is "n/a" (GPU only),
    # not Failed -- as on the CI runner, where this test failed.
    monkeypatch.setattr(main_window_module.perceptual_vship, "detect_vship_device", lambda: (_CUDA_GPU, ""))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    rd = win._rows[row]
    rd.extra_metric_keys.update({"ssimulacra2", "cvvdp"})
    win._job_rows = [rd]
    result = _fake_completed_run(str(distorted)).result
    result.source, result.distorted = source, distorted

    win._on_job_partially_failed(
        0, result, "SSIMULACRA2 failed: A\nCVVDP failed: B", "",
        {"ssimulacra2": "Vship GPU calculation failed: CUDA error 999 (unknown error)",
         "cvvdp": "CVVDP handler failed: out of memory"})

    ssimulacra2 = win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2)
    cvvdp = win.distorted_table.item(row, main_window_module.COL_CVVDP)
    assert ssimulacra2.text() == "Failed" and cvvdp.text() == "Failed"
    assert ssimulacra2.toolTip().startswith(
        "Failed on the last run: Vship GPU calculation failed: CUDA error 999 (unknown error)")
    assert cvvdp.toolTip().startswith("Failed on the last run: CVVDP handler failed: out of memory")
    assert "Log files" in cvvdp.toolTip()
    # A video that failed as a whole: each of its failed cells gives the video's reason.
    win._job_rows = [rd]
    win._on_job_failed(0, "The two videos are different shapes after cropping", "stderr lines")
    assert win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2).toolTip().startswith(
        "Failed on the last run: The two videos are different shapes after cropping\n\n")
    assert win._file_writes.wait_until_idle(10.0)
    win.close()


def test_a_videos_result_so_far_is_shown_saved_and_graphed_during_its_run(qapp, tmp_path, monkeypatch):
    """A Butteraugli score done hours before the video's VMAF was neither
    shown nor saved until the whole video was done -- closing the app or a
    crash in between lost it."""
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "distorted.mp4"
    distorted.write_bytes(b"d" * 500)
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    rd = win._rows[row]
    win._job_rows = [rd]
    win._on_job_started(0, "distorted")
    rd.analysis_status = RowState.CALCULATING
    so_far = _fake_completed_run(str(distorted)).result
    so_far.source, so_far.distorted = source, distorted

    win._on_result_updated(0, so_far)

    assert win.distorted_table.item(row, main_window_module.COL_VMAF).text() != ""  # shown at once
    assert rd.analysis_status == "Calculating"  # the video is still going
    assert rd.completed_run.partial
    assert win._file_writes.wait_until_idle(10.0)
    assert _load_cached(source, distorted, rd.options) is not None  # saved at once
    assert len(win.graph_panel._entries) == 1
    series = next(iter(win.graph_panel._entries.values()))
    series.color = "#123456"  # as the user left it

    final = _fake_completed_run(str(distorted)).result
    final.source, final.distorted = source, distorted
    win._on_job_finished(0, final)
    assert len(win.graph_panel._entries) == 1  # the same series, updated in place
    assert next(iter(win.graph_panel._entries.values())).color == "#123456"
    assert not rd.completed_run.partial and rd.analysis_status != "Calculating"
    win._on_result_updated(0, so_far)  # a late piece after the video's own result: ignored
    assert rd.completed_run.result is final
    assert win._file_writes.wait_until_idle(10.0)
    win.close()


def test_building_the_window_never_checks_for_updates(qapp, monkeypatch):
    """Only the app's startup asks GitHub -- tests build hundreds of windows."""
    from vmaf_app.core import update_check

    monkeypatch.setattr(update_check, "latest_release", lambda: pytest.fail("the window asked GitHub"))
    win = MainWindow()
    qapp.processEvents()
    win.close()


_CUDA_GPU = main_window_module.perceptual_vship.VshipDevice("cuda", "NVIDIA GPU", 0, "5.1.1", None)


def test_the_window_works_in_another_language(qapp, tmp_path, monkeypatch):
    """Every text in a made-up language -- each "[[English]]" -- through the
    window's run lines, summaries and tooltips: a placeholder a translation
    cannot fill raises here, not in front of someone running Japanese."""
    import json

    from scripts.i18n_catalog import keys
    from vmaf_app import i18n

    strings, plurals = keys()
    (tmp_path / "de.json").write_text(json.dumps({
        "strings": {key: f"[[{key}]]" for key in strings},
        "plurals": {key: [f"[[{key}]]", f"[[{plural}]]"] for key, plural in plurals.items()},
    }), encoding="utf-8")
    monkeypatch.setattr(i18n, "TRANSLATIONS_DIR", tmp_path)
    assert i18n.set_language("de") == "de"
    try:
        win = MainWindow()
        assert win.tabs.tabText(0) == "[[Videos]]"
        for name in ("a.mkv", "b.mkv"):
            win._add_table_row(Path(name))
        win._job_rows = list(win._rows)
        win._on_job_started(0, "a")
        win._on_task_progress(0, [
            _half("ffmpeg", ("vmaf",), current=20, total=100, fps=5.0, decode="source cuda, distorted cpu"),
            _half("perceptual", ("ssimulacra2", "butteraugli"), current=150, total=200, fps=30.0, phase=(2, 2, 100),
                  passes=(("ssimulacra2",), ("butteraugli",)), done_keys=("ssimulacra2",),
                  decode="source cuda, distorted cuda"),
        ])
        win._on_job_status(0, "Vship GPU unavailable (No GPU that Vship can use was found.); "
                              "using CPU reference metrics…")
        line = win.job_progress_labels[0].text()
        assert "[[{kind} {numbers} of {count}: {labels}]]" not in line  # filled in, not raw
        assert "[[Decoder: Source: {source}, test video: {test}]]".replace("{source}", "GPU") not in line
        assert "[[" in line
        win._run_failed_count, win._run_partial_count = 1, 2
        win._update_run_status()
        assert win.status_label.text().startswith("[[")
        assert win._run_end_message().startswith("[[")
        win._set_row_status(0, RowState.FAILED, "Frame rates do not match (23.976 vs 24.000 fps).")
        win._refresh_row_state(0)
        assert "[[Failed]]" in win.distorted_table.item(0, COL_PATH).toolTip()
        assert "[[Frame rates do not match ({source} vs {test} fps).]]" not in \
            win.distorted_table.item(0, COL_PATH).toolTip()
        win._set_run_ui_active(False)
        win.close()
    finally:
        i18n.set_language("en")
