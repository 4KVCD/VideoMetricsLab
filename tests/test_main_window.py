import json
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QHeaderView, QTableWidgetSelectionRange

from vmaf_app.core.models import (
    CropBox,
    CropMode,
    FrameScore,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafRunResult,
    synthetic_resample_distorted_path,
    synthetic_scale_direction_variant_path,
)
from vmaf_app.core.settings import Settings
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui import probe_worker as probe_worker_module
from vmaf_app.ui.main_window import (
    COL_BITRATE,
    COL_BLACK_BARS,
    COL_CHECK,
    COL_INFO,
    COL_PATH,
    COL_PSNR,
    COL_SCALING,
    COL_SSIM,
    COL_VMAF,
    COL_XPSNR,
    TAB_BITRATE,
    TAB_FRAME_COMPARE,
    TAB_GRAPH,
    TAB_SETTINGS,
    TAB_VIDEOS,
    CompletedRun,
    MainWindow,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_video_info(name: str) -> VideoInfo:
    return VideoInfo(
        path=Path(name), width=1920, height=1080, fps=30.0, duration=5.0,
        nb_frames=150, codec_name="h264",
    )


def _fake_completed_run(name: str) -> CompletedRun:
    info = _fake_video_info(name)
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    result = VmafRunResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )
    return CompletedRun(result, name)


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


@pytest.mark.parametrize("score", [40.0, 80.0, 90.0, 98.0])
def test_vmaf_results_have_no_quality_background(qapp, score):
    win = MainWindow()
    try:
        row = win._add_table_row(Path("test.mp4"))
        run = _fake_completed_run("test.mp4")
        run.result.frames.vmaf[:] = score
        win._rows[row].completed_run = run
        win._set_row_metrics(row)
        item = win.distorted_table.item(row, COL_VMAF)
        assert item.text() == f"{score:.2f}"
        assert item.background().color().alpha() == 0
        assert "colour bands" not in item.toolTip()
    finally:
        win.close()


def test_cached_scores_do_not_replace_fresh_hdr_preview_metadata(qapp):
    from dataclasses import replace

    from vmaf_app.core.frame_extract import hdr_kind

    win = MainWindow()
    win._source_info = replace(_fake_video_info("source.mp4"), color_transfer="smpte2084")
    row = win._add_table_row(Path("encode.mp4"))
    fresh = replace(_fake_video_info("encode.mp4"), color_transfer="smpte2084", color_primaries="bt2020")
    win._rows[row].video_info = fresh
    old = _fake_completed_run("encode.mp4").result
    old_frames = old.frames
    win._on_cached_found(Path("encode.mp4"), old, "encode")
    preview = win._frame_comparison_entry(win._rows[row]).comparison
    assert win._rows[row].video_info is fresh
    assert hdr_kind(preview.source_info) == "HDR10 / PQ"
    assert hdr_kind(preview.distorted_info) == "HDR10 / PQ"
    assert win._rows[row].completed_run.result.frames is old_frames
    assert old.distorted_info.color_transfer == ""  # cached score artifact unchanged
    win.close()


# ------------------------------------------------------------------ defaults


def test_a_new_video_asks_for_all_four_metrics(qapp):
    """All four share one decode pass, so asking for one is not cheaper.

    Measured on a 10s 1080p pair: VMAF alone 2.81s, VMAF+PSNR+SSIM 2.80s,
    all four 3.58s -- against 2.2s to fetch XPSNR in a second run later.
    """
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))

    assert win._rows[row].options.requested_metrics() == ("vmaf", "psnr", "ssim", "xpsnr")


def test_an_older_settings_file_is_upgraded_to_all_four_once(tmp_path, monkeypatch):
    """The new default has to reach installs that already have a settings file.

    Those files record the previous default explicitly, so without an upgrade
    step the change would apply to nobody who has already run the app.
    """
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(Settings, "path", staticmethod(lambda: settings_file))
    settings_file.write_text(
        json.dumps({
            "default_compute_psnr": False,
            "default_compute_ssim": False,
            "default_compute_xpsnr": False,
            "parallel_jobs": 2,
        }),
        encoding="utf-8",
    )

    upgraded = Settings.load()
    assert upgraded.default_compute_psnr
    assert upgraded.default_compute_ssim
    assert upgraded.default_compute_xpsnr
    assert upgraded.default_compute_vmaf
    assert upgraded.parallel_jobs == 2  # untouched settings survive

    # Written back, so it happens once rather than at every launch...
    assert json.loads(settings_file.read_text(encoding="utf-8"))["settings_version"] == 1

    # ...and a later choice to turn one off is then respected, not undone.
    upgraded.default_compute_xpsnr = False
    upgraded.save()
    assert not Settings.load().default_compute_xpsnr


# ------------------------------------------------------------------ row state


def test_no_column_starts_narrower_than_its_own_heading(qapp):
    """"Video bitrate" was set to 65px and rendered as "'ideo bitrat".

    The starting widths are chosen for the values, which are shorter than
    several of the headings, and nothing widened a column until a row arrived
    with content to measure -- so an empty table showed a clipped heading.
    """
    win = MainWindow()
    header = win.distorted_table.horizontalHeader()

    for column in range(win.distorted_table.columnCount()):
        if column == COL_PATH:
            continue  # the fill column stretches; it starts at its minimum
        assert win.distorted_table.columnWidth(column) >= header.sectionSizeHint(column), (
            win.distorted_table.horizontalHeaderItem(column).text()
        )


def test_the_table_has_no_status_column(qapp):
    """The metric cells already answer it.

    An unticked box means the metric was not asked for, a ticked empty one
    means it has not been measured, and a number means it has -- so a Status
    column reading "Not calculated" / "Partially calculated" / "Complete"
    restated what the same row showed a few columns to its left.
    """
    win = MainWindow()
    headers = [
        win.distorted_table.horizontalHeaderItem(c).text()
        for c in range(win.distorted_table.columnCount())
    ]
    assert "Status" not in headers
    assert win.distorted_table.columnCount() == 11


def test_metrics_have_consistent_visual_order(qapp):
    win = MainWindow()
    header = win.distorted_table.horizontalHeader()
    columns = main_window_module._METRIC_COLUMN_SET
    labels = [win.distorted_table.horizontalHeaderItem(header.logicalIndex(i)).text().strip()
              for i in range(header.count()) if header.logicalIndex(i) in columns]
    assert labels == ["VMAF", "VMAF NEG", "PSNR (dB)", "SSIM", "XPSNR (dB)"]
    win.settings_default_vmaf_neg.setChecked(True)
    assert win._options_from_settings().compute_vmaf_neg


def test_a_failed_row_is_marked_on_its_name_with_the_error_on_hover(qapp):
    # Failure is the one state the metric cells cannot fully carry: they can
    # say "Failed", but not why.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._job_rows = [win._rows[row]]

    win._on_job_failed(0, "ffmpeg exited with 1", "Invalid data found")

    name = win.distorted_table.item(row, COL_PATH)
    assert "Failed" in name.toolTip()
    assert "Invalid data found" in name.toolTip()
    assert str(Path("a.mp4")) in name.toolTip()  # the path is still there
    assert name.foreground().color().name() == "#a03030"


def test_a_result_finished_for_other_settings_is_marked_rather_than_silent(qapp):
    # It completed and is cached, but is deliberately not displayed, so
    # without a mark the row would look as though nothing had happened.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))

    win._set_row_status(
        row, "Finished with the previous settings; change them back to see the result."
    )

    name = win.distorted_table.item(row, COL_PATH)
    assert "Finished with the previous settings" in name.toolTip()
    assert name.foreground().color().name() == "#8a6d00"


def test_a_row_returning_to_normal_loses_its_mark(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_status(row, "Failed", "some error")
    assert win.distorted_table.item(row, COL_PATH).foreground().color().name() == "#a03030"

    win._set_row_status(row, "")

    name = win.distorted_table.item(row, COL_PATH)
    assert name.foreground().color() == win.distorted_table.palette().text().color()
    assert "some error" not in name.toolTip()


def test_run_clicked_skips_rows_that_already_have_a_score(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")

    for name in ("a.mp4", "b.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].video_info = _fake_video_info(name)
        win._rows[row].completed_run = _fake_completed_run(name)
        _ask_for_vmaf_only(win, row)
        win._set_row_vmaf_text(row, "90.00", bold=True)

    win._on_run_clicked()

    assert win._worker is None  # nothing needed running, so no worker was ever started
    assert "already have all requested metrics" in win.status_label.text()
    assert win.graph_panel is not None
    assert len(win.graph_panel._entries) == 2


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
    assert "Skipping 1 already-scored" in win.status_label.text()

    win._worker.cancel()
    win._worker.wait(5000)


# ------------------------------------------------------------------ resolve_model / clone_options


# ------------------------------------------------------------------ per-video settings panel

def test_new_rows_start_with_a_copy_of_last_edited_settings(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    win.distorted_table.selectRow(row_a)
    win.crop_combo.setCurrentIndex(1)  # "None (use full frame)"

    row_b = win._add_table_row(Path("b.mp4"))

    assert win._rows[row_a].options.crop_mode == CropMode.NONE
    assert win._rows[row_b].options.crop_mode == CropMode.NONE
    # independent copies, not the same object
    win._rows[row_a].options.crop_mode = CropMode.AUTO
    assert win._rows[row_b].options.crop_mode == CropMode.NONE


def test_editing_panel_only_applies_to_selected_rows(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))

    win.distorted_table.selectRow(row_a)
    win.crop_combo.setCurrentIndex(1)  # None -- should only affect row_a

    assert win._rows[row_a].options.crop_mode == CropMode.NONE
    assert win._rows[row_b].options.crop_mode == CropMode.AUTO  # untouched, still the default


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


def test_scale_direction_and_xpsnr_panel_round_trip(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win.distorted_table.selectRow(row)

    win.scale_direction_combo.setCurrentIndex(1)  # "Scale distorted up to match source"
    win.metric_header.set_checked(COL_XPSNR, True)
    win._on_metric_column_toggled(COL_XPSNR, True)

    assert win._rows[row].options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE
    assert win._rows[row].options.compute_xpsnr is True

    # deselect and reselect -- panel should reflect the row's stored choice, not reset to default
    win.distorted_table.clearSelection()
    win.distorted_table.selectRow(row)
    assert win.scale_direction_combo.currentIndex() == 1
    assert win.metric_header.is_checked(COL_XPSNR) is True


def test_panel_disabled_when_nothing_selected(qapp):
    win = MainWindow()
    win._add_table_row(Path("a.mp4"))
    assert win.options_box.isEnabled() is False  # nothing selected yet by default

    win.distorted_table.selectRow(0)
    assert win.options_box.isEnabled() is True

    win.distorted_table.clearSelection()
    assert win.options_box.isEnabled() is False


# ------------------------------------------------------------------ reopening the graph window


def test_switching_away_from_the_graph_tab_and_back_keeps_its_series(qapp):
    # The graph used to be a separate window that could be closed and
    # reopened; as a tab its contents simply persist.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")

    win.distorted_table.selectRow(row)
    win._on_show_graph_clicked()
    assert win.tabs.currentIndex() == TAB_GRAPH
    assert len(win.graph_panel._entries) == 1

    win.tabs.setCurrentIndex(TAB_VIDEOS)
    win._on_show_graph_clicked()
    assert win.tabs.currentIndex() == TAB_GRAPH
    assert len(win.graph_panel._entries) == 1  # untouched, no duplicate/re-add


def test_show_graph_adds_every_scored_row_and_switches_to_the_tab(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].completed_run = _fake_completed_run(name)

    assert len(win.graph_panel._entries) == 0
    win._on_show_graph_clicked()

    assert win.tabs.currentIndex() == TAB_GRAPH
    assert len(win.graph_panel._entries) == 2


def test_show_graph_clicked_again_after_more_rows_finish_shows_all_of_them(qapp):
    # Regression test: with the graph window already open showing 2 of 8
    # queued videos, finishing 2 more and clicking "Show graph window" again
    # used to still show only the original 2 -- it re-showed the existing
    # window without syncing in anything newly completed.
    win = MainWindow()
    rows = [win._add_table_row(Path(f"{c}.mp4")) for c in "ab"]
    for row, name in zip(rows, "ab", strict=True):
        win._rows[row].completed_run = _fake_completed_run(name)

    win._on_show_graph_clicked()
    assert len(win.graph_panel._entries) == 2

    more_rows = [win._add_table_row(Path(f"{c}.mp4")) for c in "cd"]
    for row, name in zip(more_rows, "cd", strict=True):
        win._rows[row].completed_run = _fake_completed_run(name)

    win._on_show_graph_clicked()
    assert len(win.graph_panel._entries) == 4


# ------------------------------------------------------------------ column resizing

def test_distorted_table_columns_are_user_resizable(qapp):
    win = MainWindow()
    header = win.distorted_table.horizontalHeader()
    # All columns, including PATH, are Interactive -- PATH additionally
    # auto-fills leftover space via FillColumnTable (see tests below), but
    # unlike Qt's built-in Stretch mode that doesn't disable manual dragging.
    for col in (
        COL_CHECK, COL_PATH, COL_INFO, COL_BLACK_BARS, COL_SCALING,
        COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR,
    ):
        assert header.sectionResizeMode(col) == QHeaderView.Interactive
    assert header.stretchLastSection() is False


def test_path_column_itself_can_be_manually_resized(qapp):
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    win._add_table_row(Path("a.mp4"))
    qapp.processEvents()

    win.distorted_table.setColumnWidth(COL_PATH, 500)
    assert win.distorted_table.columnWidth(COL_PATH) == 500


def test_path_column_fills_leftover_space_by_default(qapp):
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    win._add_table_row(Path("a.mp4"))
    qapp.processEvents()

    other_columns_width = sum(
        win.distorted_table.columnWidth(c)
        for c in (
            COL_CHECK, COL_INFO, COL_BLACK_BARS, COL_SCALING, COL_BITRATE,
            COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR, main_window_module.COL_VMAF_NEG,
        )
    )
    viewport = win.distorted_table.viewport().width()
    assert win.distorted_table.columnWidth(COL_PATH) >= viewport - other_columns_width - 2


def test_path_column_manually_widened_past_available_room_does_not_snap_back(qapp):
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    win._add_table_row(Path("a.mp4"))
    qapp.processEvents()

    viewport = win.distorted_table.viewport().width()
    oversized = viewport + 400  # deliberately wider than the table can show at once
    win.distorted_table.setColumnWidth(COL_PATH, oversized)
    qapp.processEvents()
    assert win.distorted_table.columnWidth(COL_PATH) == oversized

    # A later resize (e.g. the window itself) used to unconditionally clamp
    # the fill column back down to "leftover space", silently undoing the
    # manual drag instead of letting it stay oversized with a scrollbar.
    win.resize(1300, 820)
    qapp.processEvents()
    assert win.distorted_table.columnWidth(COL_PATH) == oversized


def test_path_column_shrinks_when_another_column_is_widened(qapp):
    win = MainWindow()
    win.resize(1600, 800)
    win.show()
    win._add_table_row(Path("a.mp4"))
    qapp.processEvents()

    before = win.distorted_table.columnWidth(COL_PATH)
    win.distorted_table.setColumnWidth(COL_INFO, win.distorted_table.columnWidth(COL_INFO) + 100)
    qapp.processEvents()
    after = win.distorted_table.columnWidth(COL_PATH)

    assert after < before


def test_info_bitrate_vmaf_columns_stay_snug_to_their_content(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("x.mkv"))
    before = win.distorted_table.columnWidth(COL_VMAF)

    win._set_row_vmaf_text(row, "Frame 151056/151056")
    after_progress = win.distorted_table.columnWidth(COL_VMAF)
    assert after_progress > before  # widened to fit the longer progress text

    win._set_row_vmaf_text(row, "94.76", bold=True)
    after_score = win.distorted_table.columnWidth(COL_VMAF)
    assert after_score < after_progress  # shrinks back down once the final score lands


def test_file_name_column_header_and_shows_just_the_name(qapp):
    win = MainWindow()
    header_labels = [
        win.distorted_table.horizontalHeaderItem(c).text() if win.distorted_table.horizontalHeaderItem(c) else ""
        for c in range(win.distorted_table.columnCount())
    ]
    assert "File name" in header_labels
    assert "Path to file" not in header_labels

    row = win._add_table_row(Path("C:/videos/some_encode.mp4"))
    item = win.distorted_table.item(row, COL_PATH)
    assert item.text() == "some_encode.mp4"
    # The tooltip leads with the full path, then the row's state -- which is
    # where the Status column's information went when it was removed.
    assert item.toolTip().startswith(str(Path("C:/videos/some_encode.mp4")))
    assert "Not calculated" in item.toolTip()


def test_black_bars_column_distinguishes_pending_from_disabled(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))

    assert win.distorted_table.horizontalHeaderItem(COL_BLACK_BARS).text() == "Black bars"
    assert win.distorted_table.item(row, COL_BLACK_BARS).text() == "Pending"
    assert "detected when this row is run" in win.distorted_table.item(
        row, COL_BLACK_BARS
    ).toolTip()

    win.distorted_table.selectRow(row)
    win.crop_combo.setCurrentIndex(1)

    assert win.distorted_table.item(row, COL_BLACK_BARS).text() == "Off"
    assert "full frames" in win.distorted_table.item(row, COL_BLACK_BARS).toolTip()


def test_black_bars_column_answers_yes_or_no_about_the_test_video(qapp):
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 3840, 2160)
    distorted = _fake_video_info_res("encode.mp4", 1920, 804)
    win._source_info = source
    row = win._add_table_row(distorted.path)
    result = VmafRunResult(
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


def test_black_bars_column_says_yes_when_the_test_video_is_letterboxed(qapp):
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 1920, 1080)
    distorted = _fake_video_info_res("encode.mp4", 1920, 1080)
    row = win._add_table_row(distorted.path)
    result = VmafRunResult(
        source=source.path,
        distorted=distorted.path,
        frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)],
        fps=30.0,
        model="version=vmaf_v0.6.1",
        source_crop=CropBox(w=1920, h=1080, x=0, y=0),
        distorted_crop=CropBox(w=1920, h=1080, x=0, y=0),
        source_info=source,
        distorted_info=distorted,
    )
    win._rows[row].completed_run = CompletedRun(result, "encode")

    win._set_row_metrics(row)

    item = win.distorted_table.item(row, COL_BLACK_BARS)
    assert item.text() == "No"
    assert "Reference: no black bars." in item.toolTip()
    assert "Test video: no black bars." in item.toolTip()


def test_black_bars_column_says_yes_for_a_letterboxed_test_video(qapp):
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 1920, 1080)
    distorted = _fake_video_info_res("encode.mp4", 1920, 1080)
    row = win._add_table_row(distorted.path)
    result = VmafRunResult(
        source=source.path,
        distorted=distorted.path,
        frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)],
        fps=30.0,
        model="version=vmaf_v0.6.1",
        source_crop=CropBox(w=1920, h=1080, x=0, y=0),
        distorted_crop=CropBox(w=1920, h=804, x=0, y=138),
        source_info=source,
        distorted_info=distorted,
    )
    win._rows[row].completed_run = CompletedRun(result, "encode")

    win._set_row_metrics(row)

    item = win.distorted_table.item(row, COL_BLACK_BARS)
    assert item.text() == "Yes"
    assert "Test video: black bars cropped off -- top 138 px, bottom 138 px." in item.toolTip()
    assert "Compared at 1920x804 instead of 1920x1080." in item.toolTip()


def test_black_bars_column_for_a_resolution_test_answers_for_the_reference(qapp):
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("source [downscale-1080p-upscale].mp4"))
    win._rows[row].options.resample_test = ResampleTarget(width=1920, label="1080p")
    result = VmafRunResult(
        source=source.path,
        distorted=win._rows[row].path,
        frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)],
        fps=30.0,
        model="version=vmaf_4k_v0.6.1",
        source_crop=CropBox(w=3840, h=1608, x=0, y=276),
        distorted_crop=CropBox(w=3840, h=1608, x=0, y=276),
        source_info=source,
        distorted_info=source,
    )
    win._rows[row].completed_run = CompletedRun(result, "1080p")

    win._set_row_metrics(row)

    # A resolution test has no separate file, so the reference's own bars
    # are the only answer there is to give.
    item = win.distorted_table.item(row, COL_BLACK_BARS)
    assert item.text() == "Yes"
    assert "Reference:" in item.toolTip()
    assert "Test video:" not in item.toolTip()


def test_black_bars_column_is_unknown_when_media_probe_fails(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("broken.mp4"))

    win._set_row_info(row, None, error="could not read video")

    item = win.distorted_table.item(row, COL_BLACK_BARS)
    assert item.text() == "Unknown"
    assert "could not be read" in item.toolTip()


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
    result = VmafRunResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"), frames=frames, fps=30.0,
        model="m", source_crop=None, distorted_crop=None,
        source_info=win._source_info, distorted_info=_fake_video_info_res("a.mp4", 1920, 1080),
        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
    )
    win._rows[row].completed_run = CompletedRun(result, "a")
    win._set_row_info(row, result.distorted_info)
    assert win.distorted_table.item(row, COL_SCALING).text() == "Test upscaled to source"


def test_resize_mismatch_note_absent_when_resolutions_match(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 1920, 1080)
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))
    assert win.distorted_table.item(row, COL_SCALING).text() == ""


def test_letterboxed_reference_of_the_same_width_is_not_called_a_downscale(qapp):
    """1920x1080 with bars against a 1920x804 encode is not a resize.

    The two carry the same picture; one of them still has its bars on. This
    column read the stored heights literally and announced a downscale that
    the run never performs -- the bars come off first, and then the two are
    the same size.
    """
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 1920, 1080)
    row = win._add_table_row(Path("a.mp4"))

    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 804))

    item = win.distorted_table.item(row, COL_SCALING)
    assert item.text() == "Pending"
    assert "black bars" in item.toolTip()
    assert "downscaled" not in item.toolTip()

    # Once the run has actually cropped them, they are the same size and the
    # column says so rather than staying on the fence.
    source = _fake_video_info_res("source.mp4", 1920, 1080)
    distorted = _fake_video_info_res("a.mp4", 1920, 804)
    win._rows[row].completed_run = CompletedRun(
        VmafRunResult(
            source=source.path, distorted=distorted.path,
            frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)], fps=30.0, model="m",
            source_crop=CropBox(w=1920, h=804, x=0, y=138),
            distorted_crop=CropBox(w=1920, h=804, x=0, y=0),
            source_info=source, distorted_info=distorted,
        ),
        "a",
    )
    win._set_row_metrics(row)

    item = win.distorted_table.item(row, COL_SCALING)
    assert item.text() == ""
    assert "1920x804" in item.toolTip()


def test_a_genuine_resolution_difference_still_names_the_direction(qapp):
    # Both dimensions differ, so no amount of black-bar cropping makes these
    # the same picture: this really is a downscale and must be reported.
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))

    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))

    assert win.distorted_table.item(row, COL_SCALING).text() == "\u2193 source"


def test_scaling_note_is_measured_after_the_crops_a_run_applied(qapp):
    # A 4K letterboxed master against a 1080p letterboxed encode is still a
    # downscale, but of the cropped pictures -- the tooltip must not quote
    # heights that include bars neither side was compared with.
    win = MainWindow()
    source = _fake_video_info_res("source.mp4", 3840, 2160)
    distorted = _fake_video_info_res("a.mp4", 1920, 1080)
    win._source_info = source
    row = win._add_table_row(distorted.path)
    win._rows[row].completed_run = CompletedRun(
        VmafRunResult(
            source=source.path, distorted=distorted.path,
            frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)], fps=30.0, model="m",
            source_crop=CropBox(w=3840, h=1608, x=0, y=276),
            distorted_crop=CropBox(w=1920, h=804, x=0, y=138),
            source_info=source, distorted_info=distorted,
        ),
        "a",
    )

    win._set_row_metrics(row)

    item = win.distorted_table.item(row, COL_SCALING)
    assert item.text() == "\u2193 source"
    assert "3840x1608 -> 1920x804 (after black bars)" in item.toolTip()


def test_resize_mismatch_note_for_test_both_row_ignores_a_stale_cached_direction(qapp):
    # Regression test: a "Test both" companion row whose cached result was
    # written before scale_direction was persisted (or any other reason its
    # recorded direction is stale/wrong) must still show the *correct* note
    # -- the row's own pinned direction is known for certain by construction
    # (see _add_opposite_scale_direction_rows), unlike a cached result's own
    # possibly-defaulted-on-load value.
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))

    win._add_opposite_scale_direction_rows([row])
    companion_row = row + 1
    assert win._rows[companion_row].scale_direction_pinned is True
    assert win._rows[companion_row].options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE

    # Simulate a stale cached/loaded result whose recorded direction
    # defaulted to SOURCE_TO_DISTORTED (e.g. loaded from a file saved before
    # scale_direction existed) -- the *wrong* direction for this row.
    frames = [FrameScore(frame=0, time=0.0, vmaf=90.0)]
    stale_result = VmafRunResult(
        source=Path("source.mp4"), distorted=win._rows[companion_row].path, frames=frames, fps=30.0,
        model="m", source_crop=None, distorted_crop=None,
        source_info=win._source_info, distorted_info=_fake_video_info_res("a.mp4", 1920, 1080),
        scale_direction=ScaleDirection.SOURCE_TO_DISTORTED,  # stale/wrong for this row
    )
    win._rows[companion_row].completed_run = CompletedRun(stale_result, "a")
    win._set_row_info(companion_row, stale_result.distorted_info)

    assert win.distorted_table.item(companion_row, COL_SCALING).text() == "Test upscaled to source"


def test_no_horizontal_scrollbar_at_default_with_a_typical_row(qapp):
    win = MainWindow()
    # 1280 is the default window width, but the offscreen platform these
    # tests run under renders at roughly twice a real display's font size,
    # so the fixed columns alone can exceed it there and would fail for a
    # reason no user could ever hit. Widened to whatever this font needs,
    # which still proves the point: the fill column absorbs the remainder
    # rather than a scrollbar appearing.
    fixed = sum(
        win.distorted_table.columnWidth(c)
        for c in range(win.distorted_table.columnCount())
        if c != COL_PATH
    )
    win.resize(max(1280, fixed + 400), 800)
    win.show()

    row = win._add_table_row(Path(
        r"E:\Video encodings\The.Beekeeper.2024.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.HYBRID.REMUX-FraMeSToR.mkv"
    ))
    info = VideoInfo(
        path=Path("x.mkv"), width=3840, height=2160, fps=23.976, duration=6300.0,
        nb_frames=151056, codec_name="hevc", bit_rate=69_800_000,
    )
    win._set_row_info(row, info)
    win._set_row_vmaf_text(row, "94.76", bold=True)
    qapp.processEvents()

    total_width = sum(
        win.distorted_table.columnWidth(c)
        for c in (
            COL_CHECK, COL_PATH, COL_INFO, COL_BLACK_BARS, COL_SCALING,
            COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR,
        )
    )
    assert total_width <= win.distorted_table.viewport().width()


# ------------------------------------------------------------------ persistent result cache

def test_loaded_saved_run_shows_every_metric_present_in_the_file(qapp, monkeypatch):
    win = MainWindow()
    info = _fake_video_info("saved.mp4")
    result = VmafRunResult(
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
        lambda *a, **kw: ("saved.vmafrun.json", ""),
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
    app_result = tmp_path / "abc.vmafrun.json"
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
    result_cache.store(
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

    assert result_cache.load_cached(source, distorted, win._rows[row].options) is not None


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
    assert result_cache.load_cached(old_source, distorted, win._rows[row].options) is not None
    assert result_cache.load_cached(new_source, distorted, win._rows[row].options) is None


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
    result_cache.store(
        source, distorted, win._rows[row].completed_run.result, label="x",
        options=win._rows[row].options,
    )

    win._recompute_rows([row])
    assert win._file_writes.wait_until_idle(10.0)

    assert win._rows[row].completed_run is None
    assert result_cache.load_cached(source, distorted, win._rows[row].options) is None


# ------------------------------------------------------------------ resolution round-trip test row

def test_add_resample_test_requires_a_source_selected_first(qapp, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("1080p", True)
    )
    # QMessageBox.warning() is a real modal dialog -- unmocked, it blocks
    # forever in a headless test run waiting for a click that never comes.
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *a, **kw: None)

    win._on_add_resample_test()

    assert len(win._rows) == 0


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


def test_resolution_test_does_not_offer_targets_larger_than_the_source(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 1280, 720)
    offered = []
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getItem",
        lambda _parent, _title, _prompt, labels, *_a, **_kw:
        (offered.extend(labels) or ("480p", True)),
    )

    win._on_add_resample_test()

    assert offered == ["480p"]
    assert win._rows[0].options.resample_test.width == 854


def test_add_resample_test_cancelled_dialog_adds_nothing(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("1080p", False))

    win._on_add_resample_test()

    assert len(win._rows) == 0


def test_add_resample_test_same_target_twice_does_not_duplicate(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("1080p", True))
    # The second call hits the "already added" QMessageBox.information -- a
    # real modal dialog that blocks forever in a headless test run.
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    win._on_add_resample_test()
    win._on_add_resample_test()

    assert len(win._rows) == 1


def test_run_clicked_builds_a_job_for_a_resample_row_without_probing(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("480p", True))
    win._on_add_resample_test()

    win._on_run_clicked()

    assert win._worker is not None
    assert len(win._worker._jobs) == 1
    assert win._worker._jobs[0].options.resample_test == ResampleTarget(width=854, label="480p")

    win._worker.cancel()
    win._worker.wait(5000)


# ------------------------------------------------------------------ opposite scale-direction rows

def _fake_video_info_res(name: str, width: int, height: int) -> VideoInfo:
    return VideoInfo(
        path=Path(name), width=width, height=height, fps=30.0, duration=5.0,
        nb_frames=150, codec_name="h264",
    )


def test_add_opposite_scale_direction_requires_a_source_selected_first(qapp, monkeypatch):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *a, **kw: None)

    win._add_opposite_scale_direction_rows([row])

    assert len(win._rows) == 1


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


def test_add_opposite_scale_direction_flips_from_whatever_direction_the_row_already_has(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    win._rows[row].options.scale_direction = ScaleDirection.DISTORTED_TO_SOURCE

    win._add_opposite_scale_direction_rows([row])

    assert win._rows[1].options.scale_direction == ScaleDirection.SOURCE_TO_DISTORTED


def test_add_opposite_scale_direction_skipped_when_resolutions_already_match(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 1920, 1080)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    win._add_opposite_scale_direction_rows([row])

    assert len(win._rows) == 1


def test_add_opposite_scale_direction_skipped_for_a_resample_test_row(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    monkeypatch.setattr(main_window_module.QInputDialog, "getItem", lambda *a, **kw: ("1080p", True))
    win._on_add_resample_test()
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    win._add_opposite_scale_direction_rows([0])

    assert len(win._rows) == 1


def test_add_opposite_scale_direction_twice_does_not_duplicate(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    # The second call finds nothing left to add and hits the informational
    # QMessageBox -- a real modal dialog that blocks forever headless.
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **kw: None)

    win._add_opposite_scale_direction_rows([row])
    win._add_opposite_scale_direction_rows([row])

    assert len(win._rows) == 2


def test_resolution_mismatch_labels_describe_scaling(qapp):
    win = MainWindow()
    try:
        assert win.scale_direction_combo.itemText(0) == "Source downscaled to test"
        assert win.scale_direction_combo.itemText(1) == "Test upscaled to source"
    finally:
        win.close()


def test_test_video_table_omits_black_bar_column(qapp):
    from vmaf_app.ui.main_window import COL_BLACK_BARS

    win = MainWindow()
    try:
        assert win.distorted_table.isColumnHidden(COL_BLACK_BARS)
        visible_headers = [win.distorted_table.horizontalHeaderItem(col).text()
                           for col in range(win.distorted_table.columnCount())
                           if not win.distorted_table.isColumnHidden(col)]
        assert "Black bars" not in visible_headers
        assert "Scaling" in visible_headers
    finally:
        win.close()


def test_scale_direction_combo_test_both_adds_a_row_and_reverts_the_combo(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    win.distorted_table.setRangeSelected(QTableWidgetSelectionRange(row, 0, row, win.distorted_table.columnCount() - 1), True)
    win._on_table_selection_changed()
    assert win._panel_target_rows == [row]

    win.scale_direction_combo.setCurrentIndex(2)  # "Test both"

    assert len(win._rows) == 2
    assert win._rows[1].options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE
    # The combo reverts to reflect the (unchanged) original row's own direction,
    # rather than sticking on the action item.
    assert win.scale_direction_combo.currentIndex() == 0
    assert win._rows[row].options.scale_direction == ScaleDirection.SOURCE_TO_DISTORTED


def test_run_clicked_gives_the_opposite_direction_row_a_distinct_result_identity(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info_res("a.mp4", 1920, 1080)
    win._add_opposite_scale_direction_rows([row])

    win._on_run_clicked()

    assert win._worker is not None
    jobs_by_direction = {j.options.scale_direction: j for j in win._worker._jobs}
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
    win._job_total_frames = [10, 10, 10]
    win._checked_rows_for_run = list(win._rows)

    win.distorted_table.selectRow(0)
    win._on_remove_distorted()  # drop "a.mp4" while the run is in flight

    win._on_job_finished(2, _fake_completed_run("c.mp4").result)  # job 2 == c.mp4's row

    by_name = {rd.path.name: rd for rd in win._rows}
    assert by_name["c.mp4"].completed_run is not None  # landed on the right row
    assert by_name["b.mp4"].completed_run is None      # and not on its neighbour


def test_jobs_for_removed_rows_are_dropped_without_crashing(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].video_info = _fake_video_info("a.mp4")
    win._job_rows = list(win._rows)
    win._job_total_frames = [10]
    win._checked_rows_for_run = list(win._rows)

    win.distorted_table.selectRow(0)
    win._on_remove_distorted()

    # Both the success and failure paths must no-op rather than raise.
    win._on_job_finished(0, _fake_completed_run("a.mp4").result)
    win._on_job_failed(0, "boom", "")
    win._on_all_finished()

    assert len(win._rows) == 0


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


def test_closing_waits_for_the_graph_export_queue(qapp, monkeypatch):
    win = MainWindow()
    waited = []
    monkeypatch.setattr(
        win.graph_panel, "wait_until_file_writes_idle",
        lambda *a: waited.append(True) or True,
    )

    event = QCloseEvent()
    win.closeEvent(event)

    assert waited == [True]
    assert event.isAccepted()


def test_close_is_refused_if_a_file_write_does_not_finish(qapp, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(win._file_writes, "wait_until_idle", lambda *a: True)
    monkeypatch.setattr(win.graph_panel, "wait_until_file_writes_idle", lambda *a: False)

    event = QCloseEvent()
    win.closeEvent(event)

    assert not event.isAccepted()
    assert "Finishing up" in win.status_label.text()


# ------------------------------------------------------------------ fps / ETA display

def test_job_progress_shows_fps_and_file_eta(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._job_rows = [win._rows[row]]
    win._job_total_frames = [3000]  # a single 3000-frame job
    win._on_job_started(0, "a")

    win._on_job_progress(0, current=1000, total=3000, fps=100.0)

    # A file's own rate and remaining time belong on that file's line.
    # 2000 frames remaining at 100fps -> 20s
    assert "100.0 fps" in win.job_progress_labels[0].text()
    assert "0:00:20 left" in win.job_progress_labels[0].text()
    # Live progress belongs in the status bar above, not the VMAF column --
    # that's reserved for the final score (or "Failed").
    cell = win.distorted_table.item(row, COL_VMAF)
    assert cell.text() == ""
    assert cell.checkState() == Qt.Checked  # still just "will be calculated"


def test_job_progress_queue_eta_accounts_for_other_queued_jobs(qapp):
    """The ETA counts what jobs have actually reported.

    It used to assume every earlier job was finished, which was only true
    while jobs ran strictly one after another. With several in flight an
    earlier index says nothing about whether that job is done.
    """
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))
    win._job_rows = [win._rows[row_a], win._rows[row_b]]
    win._job_total_frames = [1000, 4000]
    win._on_job_started(1, "b")

    win._mark_job_over(0)  # job 0 really has finished: all 1000 frames
    win._on_job_progress(1, current=2000, total=4000, fps=50.0)

    # Only job 1's remaining 2000 frames are left, at 50fps -> 40s.
    assert "Queue ETA: 0:00:40" in win.status_label.text()


def test_queue_progress_does_not_assume_earlier_jobs_have_finished(qapp):
    # The parallel case: job 1 is running while job 0 still is too, so job
    # 0's frames must not be counted as done.
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 4000]
    win._on_job_started(1, "b")

    win._on_job_progress(1, current=2000, total=4000, fps=50.0)

    # 2000 frames still to do on job 1 and 1000 untouched on job 0, at 50fps.
    assert "Queue ETA: 0:01:00" in win.status_label.text()


def test_two_running_jobs_drain_the_queue_at_their_combined_rate(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 1000]
    win._on_job_started(0, "a")
    win._on_job_started(1, "b")

    win._on_job_progress(0, current=500, total=1000, fps=25.0)
    win._on_job_progress(1, current=500, total=1000, fps=25.0)

    # Each has 500 frames left at 25fps, and they run side by side, so the
    # queue ends in 20s rather than the 40s one after another would take.
    assert win.status_label.text() == "Running 2 of 2 together   ·   Queue ETA: 0:00:20"


def test_a_finished_job_stops_counting_towards_the_combined_rate(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 1000]
    win._on_job_started(0, "a")
    win._on_job_started(1, "b")

    win._on_job_progress(0, current=1000, total=1000, fps=25.0)
    win._mark_job_over(0)
    win._on_job_progress(1, current=0, total=1000, fps=25.0)

    # 1000 frames left at 25fps = 40s. Leaving the finished job's rate in
    # would have claimed 20s and the estimate would never be met.
    assert "Queue ETA: 0:00:40" in win.status_label.text()


def test_the_status_line_does_not_repeat_the_names_below_it(qapp):
    """Every running video has its own line directly underneath carrying its
    name, so listing the names here said the same thing twice -- and with two
    long file names it was the longest line on screen for no information."""
    win = MainWindow()
    for name in ("encode-a.mp4", "encode-b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 1000]

    win._on_job_started(0, "encode-a")
    assert win.status_label.text().startswith("Running 1 of 2")

    win._on_job_started(1, "encode-b")
    text = win.status_label.text()
    assert text.startswith("Running 2 of 2 together")
    assert "encode-a" not in text and "encode-b" not in text
    # The names are on the lines below, where they are not duplicated.
    assert "encode-a" in win.job_progress_labels[0].text()
    assert "encode-b" in win.job_progress_labels[1].text()

    win._mark_job_over(0)
    assert win.status_label.text().startswith("Running 2 of 2")


def test_job_progress_with_zero_fps_promises_no_time(qapp):
    # No rate yet means no basis for an estimate, and a made-up one is worse
    # than none.
    win = MainWindow()
    win._job_rows = [win._rows[win._add_table_row(Path("a.mp4"))]]
    win._job_total_frames = [3000]
    win._on_job_started(0, "a")

    win._on_job_progress(0, current=5, total=3000, fps=0.0)

    assert "left" not in win.job_progress_labels[0].text()
    assert win.status_label.text() == "Running 1 of 1   ·   Queue ETA: calculating..."


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

    assert "1 failed" in win.status_label.text()
    assert "Done" not in win.status_label.text()


def test_estimate_total_frames_used_when_building_jobs(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    win._source_info.duration = 10.0
    win._source_info.nb_frames = 300
    row = win._add_table_row(Path("distorted.mp4"))
    win._rows[row].video_info = VideoInfo(
        path=Path("distorted.mp4"), width=1920, height=1080, fps=30.0, duration=10.0,
        nb_frames=300, codec_name="h264",
    )

    win._on_run_clicked()

    assert win._job_total_frames == [300]

    win._worker.cancel()
    win._worker.wait(5000)


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


def test_the_video_table_stays_scrollable_during_a_run(qapp):
    """Disabling the whole Videos box took the table's scrollbar with it.

    A long queue is exactly what someone wants to scroll through while it
    runs -- to see which videos are left, or to read a failure that scrolled
    off. Freezing the controls that can change the jobs does not require
    freezing the view of them.
    """
    win = MainWindow()
    for i in range(30):
        win._add_table_row(Path(f"encode-{i}.mp4"))
    # Small enough that 30 rows genuinely overflow it; an unshown table is
    # laid out at its full height and has nothing to scroll.
    win.distorted_table.setFixedHeight(120)
    qapp.processEvents()  # the scroll range is computed during layout

    win._set_run_ui_active(True)

    assert win.distorted_table.isEnabled()
    scrollbar = win.distorted_table.verticalScrollBar()
    assert scrollbar.isEnabled()
    assert scrollbar.maximum() > 0, "nothing to scroll -- the test proves nothing"
    # And it really scrolls, rather than merely reporting itself enabled.
    scrollbar.setValue(scrollbar.maximum())
    assert scrollbar.value() > 0


def test_a_running_row_cannot_have_its_metrics_reticked(qapp):
    # The table stays live, so the guards on what it can change have to hold.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    before = win._rows[row].options.requested_metrics()
    win._set_run_ui_active(True)

    win.distorted_table.item(row, COL_PSNR).setCheckState(Qt.Unchecked)

    assert win._rows[row].options.requested_metrics() == before
    # ...and the box is put back rather than left contradicting the row.
    assert win.distorted_table.item(row, COL_PSNR).checkState() == Qt.Checked


def test_the_recompute_menu_is_suppressed_during_a_run(qapp):
    # Its only entry deletes cached results and re-queues rows, which would
    # fight the run currently using them.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win.distorted_table.selectRow(row)
    calls = []
    win._recompute_rows = lambda rows: calls.append(rows)

    win._set_run_ui_active(True)
    win._on_table_context_menu(win.distorted_table.viewport().rect().center())

    assert calls == []


# ------------------------------------------------------------------ metric columns

def test_metric_columns_are_tick_boxes_until_a_score_exists(qapp):
    """An unmeasured metric offers the choice; a measured one gives the answer.

    "N/A" and "Pending" said the same two things in words, in a cell that
    could not be acted on -- turning the metric on meant finding a separate
    panel and matching it up with the right row.
    """
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))

    # All four are requested by default -- they share one decode pass.
    for col in (COL_PSNR, COL_SSIM, COL_XPSNR, COL_VMAF):
        item = win.distorted_table.item(row, col)
        assert item.text() == ""
        assert item.checkState() == Qt.Checked
        assert item.flags() & Qt.ItemIsUserCheckable
    # And the column headers agree with the rows rather than stating their own
    # fixed set.
    for col in (COL_PSNR, COL_SSIM, COL_XPSNR, COL_VMAF):
        assert win.metric_header.is_checked(col)

    # Unticking one leaves it unticked and empty, not "N/A".
    win.distorted_table.item(row, COL_SSIM).setCheckState(Qt.Unchecked)
    assert "ssim" not in win._rows[row].options.requested_metrics()
    assert win.distorted_table.item(row, COL_SSIM).text() == ""


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


def test_a_measured_metric_shows_its_score_with_no_tick_box(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win._set_row_metrics(row)

    item = win.distorted_table.item(row, COL_VMAF)
    assert item.text() not in ("", "Pending")
    assert not item.flags() & Qt.ItemIsUserCheckable
    assert item.data(Qt.CheckStateRole) is None  # no indicator drawn at all


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


@pytest.mark.parametrize("control", ["gpu", "threads"])
def test_execution_only_option_change_keeps_an_existing_result(qapp, control):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    completed = _fake_completed_run("a.mp4")
    win._rows[row].completed_run = completed
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    if control == "gpu":
        win.gpu_checkbox.setChecked(not win.gpu_checkbox.isChecked())
    else:
        win.threads_spin.setValue(7)

    assert win._rows[row].completed_run is completed


def test_score_option_change_rechecks_cache_for_the_new_combination(qapp, monkeypatch):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    lookups = []
    monkeypatch.setattr(win, "_start_cache_lookup", lookups.append)
    win.subsample_spin.setValue(2)

    assert lookups == [[Path("a.mp4")]]


def test_replacing_a_slow_probe_keeps_the_old_thread_alive_and_ignores_it(qapp, monkeypatch):
    class FakeSignal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args):
            for callback in self.callbacks:
                callback(*args)

    class FakeProbeWorker:
        instances = []

        def __init__(self, *args, **kwargs):
            self.probed = FakeSignal()
            self.cached_found = FakeSignal()
            self.finished_all = FakeSignal()
            self.cancelled = False
            self.running = False
            self.deleted = False
            self.instances.append(self)

        def start(self):
            self.running = True

        def isRunning(self):
            return self.running

        def cancel(self):
            self.cancelled = True

        def wait(self, _ms):
            raise AssertionError("the UI must not block for an arbitrary timeout")

        def deleteLater(self):
            self.deleted = True

    monkeypatch.setattr(main_window_module, "ProbeWorker", FakeProbeWorker)
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))

    win._source_info = _fake_video_info("source.mp4")
    win._start_cache_lookup([Path("a.mp4")])
    old = FakeProbeWorker.instances[-1]
    win._start_cache_lookup([Path("a.mp4")])

    assert old.cancelled
    assert old in win._probe_workers
    assert win._rows[row].video_info is None


def test_finished_probe_is_not_reused_after_qt_deletes_it(qapp, monkeypatch):
    class FinishedWorker:
        def __init__(self):
            self.deleted = False

        def deleteLater(self):
            self.deleted = True

    win = MainWindow()
    worker = FinishedWorker()
    win._cache_worker = worker
    win._probe_workers.append(worker)

    win._on_cache_lookup_finished(win._cache_generation, worker)

    assert win._cache_worker is None
    assert worker not in win._probe_workers
    assert worker.deleted


def test_selecting_a_slow_source_does_not_block_the_ui(qapp, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_probe(path, process_handle=None):
        started.set()
        release.wait(10.0)
        return _fake_video_info(str(path))

    monkeypatch.setattr(probe_worker_module, "probe_video", slow_probe)
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: ("slow-source.mp4", "")),
    )
    win = MainWindow()

    before = time.monotonic()
    win._on_browse_source()
    elapsed = time.monotonic() - before

    assert elapsed < 1.0
    assert started.wait(5.0)
    assert win._source_info is None

    release.set()
    deadline = time.monotonic() + 10.0
    while win._source_probe_worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert win._source_info is not None
    assert win.source_edit.text() == "slow-source.mp4"


def test_cache_result_is_rejected_if_options_changed_while_it_loaded(qapp, tmp_path):
    from vmaf_app.core import result_cache

    source = tmp_path / "source.mp4"
    distorted = tmp_path / "distorted.mp4"
    source.write_bytes(b"source")
    distorted.write_bytes(b"distorted")

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    old_key = result_cache.cache_key(source, distorted, win._rows[row].options)

    # This is exactly what can happen while ProbeWorker is parsing a large
    # cached JSON file: the row remains editable before its signal arrives.
    win._rows[row].options.n_subsample = 2
    cached_result = _fake_completed_run(str(distorted)).result
    cached_result.source = source
    cached_result.distorted = distorted
    win._on_cached_if_current(
        win._cache_generation, distorted, cached_result, "old settings", old_key
    )

    assert win._rows[row].completed_run is None


def test_metric_columns_show_each_metrics_own_mean(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._on_metric_column_toggled(COL_PSNR, True)
    win._on_metric_column_toggled(COL_SSIM, True)
    win._on_metric_column_toggled(COL_XPSNR, True)

    info = _fake_video_info("a.mp4")
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0, psnr=42.0, ssim=0.95, xpsnr=38.0) for i in range(4)]
    result = VmafRunResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"), frames=frames, fps=30.0,
        model="m", source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )
    win._rows[row].completed_run = CompletedRun(result, "a")
    win._set_row_metrics(row)

    assert win.distorted_table.item(row, COL_VMAF).text() == "90.00"
    assert win.distorted_table.item(row, COL_PSNR).text() == "42.00"
    assert win.distorted_table.item(row, COL_SSIM).text() == "0.9500"  # SSIM needs more decimals to be useful
    assert win.distorted_table.item(row, COL_XPSNR).text() == "38.00"


def test_options_panel_sits_below_the_file_table_not_beside_it(qapp):
    win = MainWindow()
    files_y = win.distorted_table.mapTo(win, win.distorted_table.rect().topLeft()).y()
    options_y = win.options_box.mapTo(win, win.options_box.rect().topLeft()).y()
    assert options_y > files_y


def test_metric_toggle_does_not_clobber_other_per_row_settings(qapp):
    # The global default used to be rebuilt from row 0's options, so toggling
    # a metric column pushed that one row's unrelated model/crop/GPU choices
    # onto every future row.
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    win._rows[row_a].options.crop_mode = CropMode.NONE
    win._rows[row_a].options.n_threads = 7
    default_crop_before = win._default_options.crop_mode

    win._on_metric_column_toggled(COL_PSNR, True)

    assert win._default_options.crop_mode == default_crop_before  # untouched
    assert win._default_options.n_threads != 7
    assert "name=psnr" in win._default_options.extra_features  # but the metric did apply
    assert win._rows[row_a].options.crop_mode == CropMode.NONE  # row keeps its own settings
    assert win._rows[row_a].options.n_threads == 7


def test_rows_added_after_a_metric_toggle_inherit_it(qapp):
    win = MainWindow()
    win._on_metric_column_toggled(COL_XPSNR, True)

    row = win._add_table_row(Path("later.mp4"))

    assert win._rows[row].options.compute_xpsnr is True
    assert win.distorted_table.item(row, COL_XPSNR).text() != "N/A"


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


def test_too_old_ffmpeg_is_reported_in_the_banner(qapp, monkeypatch):
    from vmaf_app.core.ffmpeg_locate import ToolsStatus, ToolStatus

    old = ToolsStatus(
        ffmpeg=ToolStatus("ffmpeg", "ffmpeg.exe", True, (6, 1, 1)),
        ffprobe=ToolStatus("ffprobe", "ffprobe.exe", True, (6, 1, 1)),
    )
    monkeypatch.setattr(main_window_module, "check_tools", lambda: old)

    win = MainWindow()

    assert "too old" in win._ffmpeg_banner.text()
    assert win._check_ffmpeg() is False


def test_healthy_tools_leave_the_banner_hidden(qapp, monkeypatch):
    from vmaf_app.core.ffmpeg_locate import ToolsStatus, ToolStatus

    good = ToolsStatus(
        ffmpeg=ToolStatus("ffmpeg", "ffmpeg.exe", True, (9, 0, 1)),
        ffprobe=ToolStatus("ffprobe", "ffprobe.exe", True, (9, 0, 1)),
    )
    monkeypatch.setattr(main_window_module, "check_tools", lambda: good)

    win = MainWindow()

    assert win._check_ffmpeg() is True
    assert win._ffmpeg_banner.isVisible() is False


# ------------------------------------------------------------------ tabs

def test_the_window_has_videos_graph_frame_compare_and_settings_tabs(qapp):
    win = MainWindow()
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Videos", "Metric Graphs", "Video Compare", "Bitrate Viewer", "Settings"]


def test_frame_compare_is_a_tab_between_graph_and_settings(qapp):
    win = MainWindow()

    assert win.tabs.widget(TAB_FRAME_COMPARE) is win.frame_compare_panel


def test_bitrate_viewer_is_an_independent_tab(qapp):
    win = MainWindow()

    assert win.tabs.widget(TAB_BITRATE) is win.bitrate_panel


def test_finishing_metrics_automatically_analyzes_both_physical_videos(
    qapp, monkeypatch
):
    source = Path("source.mp4")
    distorted = Path("distorted.mp4")
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    row = win._add_table_row(distorted)
    win._job_rows = [win._rows[row]]
    monkeypatch.setattr(win._file_writes, "submit", lambda *args: None)
    captured = []
    monkeypatch.setattr(
        win.bitrate_panel, "add_and_analyze", lambda infos: captured.append(infos)
    )
    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    result.source_info = _fake_video_info(str(source))
    result.distorted_info = _fake_video_info(str(distorted))

    win._on_job_finished(0, result)

    assert [[info.path for info in infos] for infos in captured] == [
        [source, distorted]
    ]


def test_resolution_metric_run_only_analyzes_its_one_physical_video(qapp, monkeypatch):
    source = Path("source.mp4")
    target = ResampleTarget(width=1920, label="1080p")
    synthetic = synthetic_resample_distorted_path(source, target)
    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    row = win._add_table_row(synthetic)
    win._job_rows = [win._rows[row]]
    monkeypatch.setattr(win._file_writes, "submit", lambda *args: None)
    captured = []
    monkeypatch.setattr(
        win.bitrate_panel, "add_and_analyze", lambda infos: captured.append(infos)
    )
    result = _fake_completed_run(str(synthetic)).result
    result.source = source
    result.distorted = synthetic
    result.source_info = _fake_video_info(str(source))
    result.resample_target = target

    win._on_job_finished(0, result)

    assert [[info.path for info in infos] for infos in captured] == [[source]]


def test_frame_preview_color_mode_is_remembered(qapp):
    from vmaf_app.core.frame_extract import PreviewColorMode
    from vmaf_app.core.settings import Settings

    win = MainWindow()
    combo = win.frame_compare_panel.color_mode_combo
    combo.setCurrentIndex(combo.findData(PreviewColorMode.UNMANAGED.value))

    assert Settings.load().frame_preview_color_mode == PreviewColorMode.UNMANAGED.value
    assert win.tabs.tabText(TAB_FRAME_COMPARE) == "Video Compare"


def test_the_graph_is_a_tab_not_a_separate_window(qapp):
    # It used to be a top-level window with its own taskbar button.
    win = MainWindow()
    assert win.tabs.widget(TAB_GRAPH) is win.graph_panel
    assert not win.graph_panel.isWindow()


def test_compare_selected_switches_to_the_graph_tab(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win.distorted_table.selectRow(row)

    assert win.tabs.currentIndex() == TAB_VIDEOS
    win._on_show_graph_clicked()
    assert win.tabs.currentIndex() == TAB_GRAPH


def test_show_graph_with_nothing_scored_stays_on_the_videos_tab(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information",
        lambda *a, **k: shown.append(a),
    )
    win = MainWindow()
    win._add_table_row(Path("a.mp4"))  # added but never run

    win._on_show_graph_clicked()
    assert win.tabs.currentIndex() == TAB_VIDEOS
    assert shown, "should say why there is nothing to show"


# ------------------------------------------------------------------ settings

def test_settings_defaults_seed_newly_added_rows(qapp):
    # The Settings tab sets the starting point only; each row's own options
    # are edited in the Videos tab afterwards.
    win = MainWindow()
    win.settings_default_psnr.setChecked(True)
    win.settings_default_xpsnr.setChecked(True)

    row = win._add_table_row(Path("a.mp4"))
    options = win._rows[row].options
    assert "name=psnr" in options.extra_features
    assert options.compute_xpsnr is True


def test_editing_settings_does_not_retarget_existing_rows(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    before = list(win._rows[row].options.extra_features)

    win.settings_default_ssim.setChecked(True)
    assert win._rows[row].options.extra_features == before, "existing rows keep their own settings"


def test_the_settings_tab_reports_the_tools_it_found(qapp):
    win = MainWindow()
    assert win.settings_ffmpeg_status.text(), "the ffmpeg status should say something"
    assert "saved result" in win.settings_cache_summary.text()


# ------------------------------------------------------------------ graph stays in step

def test_opening_the_graph_tab_shows_completed_rows_without_pressing_anything(qapp):
    # The graph used to stay empty until "Show graph" was pressed -- a
    # leftover from when it was a window that had to be opened.
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")

    assert len(win.graph_panel._entries) == 0
    win.tabs.setCurrentIndex(TAB_GRAPH)
    assert len(win.graph_panel._entries) == 1


def test_graph_remove_button_is_not_undone_by_switching_tabs(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win._sync_graph()
    sid = next(iter(win.graph_panel._entries))

    win.graph_panel.remove_run(sid)
    win.tabs.setCurrentIndex(TAB_VIDEOS)
    win.tabs.setCurrentIndex(TAB_GRAPH)

    assert not win.graph_panel._entries


def test_separate_runs_of_the_same_distorted_path_can_be_compared(qapp):
    win = MainWindow()
    first = _fake_completed_run("same.mp4")
    second = _fake_completed_run("same.mp4")
    first.label = "same — model 1"
    second.label = "same — model 2"
    second.result.model = "version=vmaf_v0.6.1neg"

    win._open_or_update_graph([first, second])

    assert len(win.graph_panel._entries) == 2
    assert {entry.label for entry in win.graph_panel._entries.values()} == {
        "same — model 1", "same — model 2"
    }


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


def test_remove_all_can_be_declined(qapp, monkeypatch):
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: main_window_module.QMessageBox.No,
    )
    win = MainWindow()
    win._add_table_row(Path("a.mp4"))
    win._on_remove_all_distorted()
    assert len(win._rows) == 1


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


def test_a_queued_store_lands_in_the_folder_that_was_configured(qapp, tmp_path, monkeypatch):
    # Queued writes run later, on another thread. Resolving the cache folder
    # inside the task reads whatever the setting says by then, so a result
    # computed while folder A was configured was written into folder B.
    from vmaf_app.core import result_cache

    folder_a, folder_b = _two_cache_dirs(tmp_path)
    source = tmp_path / "source.mp4"
    distorted = tmp_path / "distorted.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._job_rows = [win._rows[row]]

    result_cache.set_cache_dir_override(folder_a)
    release = _block_writes(win)

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    win._on_job_finished(0, result)

    # The user changes the cache folder before the queue drains.
    result_cache.set_cache_dir_override(folder_b)
    release.set()
    assert win._file_writes.wait_until_idle(10.0)

    assert list(folder_a.glob("*.vmafrun.json")), "the result was written to the wrong folder"
    assert not list(folder_b.glob("*.vmafrun.json"))


def test_clearing_the_cache_deletes_the_folder_the_dialog_named(qapp, tmp_path, monkeypatch):
    # The confirmation dialog names a folder. Deleting a different one than
    # the user was shown is not something to leave to timing.
    from vmaf_app.core import result_cache

    folder_a, folder_b = _two_cache_dirs(tmp_path)
    (folder_a / "one.vmafrun.json").write_text("{}", encoding="utf-8")
    (folder_b / "two.vmafrun.json").write_text("{}", encoding="utf-8")

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

    assert not list(folder_a.glob("*.vmafrun.json")), "the named folder was not cleared"
    assert list(folder_b.glob("*.vmafrun.json")), "an unnamed folder was cleared instead"


def test_a_queued_recompute_deletes_from_the_folder_it_was_asked_about(qapp, tmp_path):
    from vmaf_app.core import result_cache

    folder_a, folder_b = _two_cache_dirs(tmp_path)
    source = tmp_path / "source.mp4"
    distorted = tmp_path / "distorted.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    options = win._rows[row].options

    result_cache.set_cache_dir_override(folder_a)
    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    result_cache.store(source, distorted, result, "d", options, folder_a)
    result_cache.store(source, distorted, result, "d", options, folder_b)

    release = _block_writes(win)
    win._recompute_rows([row])
    result_cache.set_cache_dir_override(folder_b)
    release.set()
    assert win._file_writes.wait_until_idle(10.0)

    assert result_cache.load_cached(source, distorted, options, folder_a) is None
    assert result_cache.load_cached(source, distorted, options, folder_b) is not None


# ------------------------- cache identity of synthetic ("test both") rows

def _real_pair(tmp_path, distorted_bytes=b"d" * 500):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 1000)
    distorted = tmp_path / "movie.mp4"
    distorted.write_bytes(distorted_bytes)
    return source, distorted


def _companion_row(win, tmp_path):
    """A 'Test both scaling directions' companion row for the one real file."""
    source, distorted = _real_pair(tmp_path)
    win._source_info = _fake_video_info_res(str(source), 3840, 2160)
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._rows[row].video_info = _fake_video_info_res(str(distorted), 1920, 1080)
    win.distorted_table.selectRow(row)
    win._add_opposite_scale_direction_rows([row])
    assert len(win._rows) == 2, "the companion row was not created"
    return source, distorted, win._rows[1]


def test_a_companion_rows_identity_follows_the_file_it_actually_decodes(qapp, tmp_path):
    # The companion carries a synthetic path that does not exist, so
    # _file_identity records size and mtime as -1 for it: nothing about the
    # real video reaches the key.
    from vmaf_app.core import result_cache

    win = MainWindow()
    source, distorted, companion = _companion_row(win, tmp_path)

    assert companion.path != distorted, "the companion should have its own row identity"
    assert companion.identity_path == distorted

    before = result_cache.cache_key(source, companion.identity_path, companion.options)
    distorted.write_bytes(b"REPLACED" * 200)  # different content, different size
    after = result_cache.cache_key(source, companion.identity_path, companion.options)

    assert before != after, "replacing the real video left the companion's key unchanged"


def test_the_two_scale_directions_still_have_separate_keys(qapp, tmp_path):
    from vmaf_app.core import result_cache

    win = MainWindow()
    source, _distorted, companion = _companion_row(win, tmp_path)
    original = win._rows[0]

    assert original.identity_path == companion.identity_path, "same physical file"
    assert original.options.scale_direction != companion.options.scale_direction
    assert result_cache.cache_key(source, original.identity_path, original.options) != \
        result_cache.cache_key(source, companion.identity_path, companion.options)


def test_the_companion_keeps_its_own_graph_identity(qapp, tmp_path):
    # Sharing a cache key would be wrong; sharing a row/series identity would
    # make the two directions overwrite each other on the plot.
    win = MainWindow()
    _source, distorted, companion = _companion_row(win, tmp_path)

    assert companion.path != win._rows[0].path
    assert "upscale-distorted-to-source" in companion.path.name
    assert companion.path != distorted


def test_a_resolution_test_row_follows_the_source_file(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache

    source = tmp_path / "master.mkv"
    source.write_bytes(b"s" * 1000)

    win = MainWindow()
    win._source_info = _fake_video_info_res(str(source), 3840, 2160)
    win._source_info.path = source
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getItem", lambda *a, **k: ("1080p", True)
    )
    win._on_add_resample_test()
    assert len(win._rows) == 1
    row_data = win._rows[0]

    assert row_data.identity_path == source
    before = result_cache.cache_key(source, row_data.identity_path, row_data.options)
    source.write_bytes(b"REPLACED" * 400)
    after = result_cache.cache_key(source, row_data.identity_path, row_data.options)

    assert before != after, "replacing the source left the resolution test's key unchanged"


def test_a_stale_companion_result_is_not_loaded_after_the_file_changes(qapp, tmp_path):
    from vmaf_app.core import result_cache

    win = MainWindow()
    source, distorted, companion = _companion_row(win, tmp_path)

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = companion.path
    result_cache.store(
        source, companion.identity_path, result, "movie", companion.options
    )
    assert win._try_load_cached_result(1), "the freshly stored result should load"

    win._rows[1].completed_run = None
    distorted.write_bytes(b"REPLACED" * 200)

    assert not win._try_load_cached_result(1), "a stale score loaded for replaced content"



# ------------------- media probing must survive unrelated background work

def _blocking_probe(monkeypatch, release):
    """Makes probe_video block until `release` is set, per path."""
    from vmaf_app.core import ffprobe
    from vmaf_app.ui import probe_worker as probe_worker_module

    probed = []

    def slow_probe(path, process_handle=None):
        probed.append(Path(path))
        release.wait(10.0)
        return _fake_video_info(str(path))

    monkeypatch.setattr(probe_worker_module, "probe_video", slow_probe)
    monkeypatch.setattr(ffprobe, "probe_video", slow_probe)
    return probed


def _pump_until(predicate, seconds=10.0):
    """Pumps the event loop until `predicate` holds, or gives up.

    Waiting for the worker threads to *exit* is not enough: probed/
    cached_found cross thread boundaries, so Qt queues them and they are
    only delivered by the event loop afterwards. Waiting on the effect
    rather than on the thread is what makes this deterministic.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return predicate()


def _wait_for_probes(win, seconds=10.0):
    """Every row has media info and no worker is left running."""
    return _pump_until(
        lambda: not any(w.isRunning() for w in win._probe_workers)
        and all(rd.video_info is not None for rd in win._rows),
        seconds,
    )


def test_a_cache_lookup_does_not_abandon_a_running_media_probe(qapp, tmp_path, monkeypatch):
    """The reported stuck-row bug.

    Media probing and cache lookups shared one worker slot and one
    generation counter, so starting a lookup cancelled the probe AND
    invalidated its results -- and the replacement never probed, because a
    cache lookup does not read media info. Rows stayed on "Reading..."
    forever with nothing outstanding to fill them.
    """
    import threading

    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 100)
    paths = []
    for name in ("a.mp4", "b.mp4"):
        path = tmp_path / name
        path.write_bytes(b"d" * 100)
        paths.append(path)

    release = threading.Event()
    probed = _blocking_probe(monkeypatch, release)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    for path in paths:
        win._add_table_row(path)
    win._start_media_probe(paths)

    # Anything that triggers a cache lookup while the probe is still going:
    # changing a score option, adding files, a source probe finishing.
    win._start_cache_lookup(paths)
    win._reload_cached_for_all_rows()

    release.set()
    assert _wait_for_probes(win), "workers never finished or a row was left unprobed"

    assert probed == paths, "a media probe was abandoned part-way"
    for row_data in win._rows:
        assert row_data.video_info is not None, (
            f"{row_data.path.name} was left stuck with no media info"
        )


def test_a_second_batch_of_files_does_not_strand_the_first(qapp, tmp_path, monkeypatch):
    import threading

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    for path in (first, second):
        path.write_bytes(b"d" * 100)

    release = threading.Event()
    probed = _blocking_probe(monkeypatch, release)

    win = MainWindow()
    win._add_table_row(first)
    win._start_media_probe([first])
    win._add_table_row(second)
    win._start_media_probe([second])

    release.set()
    assert _wait_for_probes(win)

    assert sorted(p.name for p in probed) == ["first.mp4", "second.mp4"]
    assert all(rd.video_info is not None for rd in win._rows)


def test_a_media_probe_result_is_applied_even_after_a_later_cache_lookup(qapp, tmp_path, monkeypatch):
    # A probe describes the FILE, so its answer stays true no matter what
    # else the window did meanwhile. It used to be discarded on generation
    # mismatch, which is what made the row unrecoverable.
    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 100)
    distorted = tmp_path / "a.mp4"
    distorted.write_bytes(b"d" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)

    win._start_cache_lookup([distorted])  # bumps the cache generation
    win._on_probed(distorted, _fake_video_info(str(distorted)), "")

    assert win._rows[row].video_info is not None


def test_recompute_cancels_the_cache_lane_but_not_media_probing(qapp, tmp_path, monkeypatch):
    # Recompute must stop an in-flight cache read from restoring the very
    # result being discarded -- without stranding a media probe.
    import threading

    source = tmp_path / "source.mp4"
    source.write_bytes(b"s" * 100)
    distorted = tmp_path / "a.mp4"
    distorted.write_bytes(b"d" * 100)

    release = threading.Event()
    probed = _blocking_probe(monkeypatch, release)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._start_media_probe([distorted])
    before = win._cache_generation

    win._recompute_rows([row])

    assert win._cache_generation > before, "an in-flight cache read stays valid"
    release.set()
    assert _wait_for_probes(win)
    assert probed == [distorted], "the media probe was cancelled by a recompute"
    assert win._rows[row].video_info is not None



# ---------------- background completions must not unlock a live run

def _start_fake_run(win, row):
    """Puts the window into the state a live VMAF job leaves it in."""
    from vmaf_app.core.models import clone_options

    win._job_rows = [win._rows[row]]
    win._job_cache_options = [clone_options(win._rows[row].options)]
    win._checked_rows_for_run = [win._rows[row]]
    win._set_run_ui_active(True)
    win.status_label.setText("Running ffmpeg...")


def test_a_probe_finishing_mid_run_does_not_re_enable_the_options(qapp, tmp_path):
    """The reported defect: _on_table_selection_changed enables the panel
    from the selection alone, and background completions call it. A probe
    landing during a run therefore unlocked every score-changing control,
    and whatever was changed there was attached to a result computed under
    the previous settings.
    """
    source = tmp_path / "source.mp4"
    distorted = tmp_path / "a.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win.distorted_table.selectRow(row)
    _start_fake_run(win, row)
    assert not win.options_box.isEnabled()

    win._on_probe_finished()

    assert not win.options_box.isEnabled(), "a probe unlocked the options mid-run"
    assert all(not w.isEnabled() for w in win._file_action_widgets)
    assert not win.run_btn.isEnabled()


def test_a_probe_finishing_mid_run_does_not_report_ready(qapp, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(tmp_path / "a.mp4")
    _start_fake_run(win, row)

    win._on_probe_finished()

    assert win.status_label.text() != "Ready.", "a live job was reported as finished"


def test_a_source_probe_finishing_mid_run_does_not_report_ready(qapp, tmp_path):
    win = MainWindow()
    row = win._add_table_row(tmp_path / "a.mp4")
    _start_fake_run(win, row)

    class _Finished:
        def deleteLater(self):
            pass

    win._on_source_probe_finished(win._source_probe_generation, _Finished())

    assert win.status_label.text() != "Ready."


def test_the_options_unlock_again_once_the_run_ends(qapp, tmp_path):
    source = tmp_path / "source.mp4"
    distorted = tmp_path / "a.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win.distorted_table.selectRow(row)
    _start_fake_run(win, row)

    win._set_run_ui_active(False)
    win._on_probe_finished()

    assert win.options_box.isEnabled()
    assert win.status_label.text() == "Ready."


def test_a_result_is_not_shown_under_options_it_was_not_computed_with(qapp, tmp_path):
    """Defence in depth for the same bug. Even if something re-enables the
    panel, a finished job must not be labelled with settings changed after
    it was launched."""
    from vmaf_app.core import result_cache
    from vmaf_app.core.models import clone_options

    source = tmp_path / "source.mp4"
    distorted = tmp_path / "a.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    _start_fake_run(win, row)
    launched_with = clone_options(win._rows[row].options)

    # The user changes a score-affecting setting while ffmpeg runs.
    win._rows[row].options.n_subsample = 7

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    win._on_job_finished(0, result)
    assert win._file_writes.wait_until_idle(10.0)

    assert win._rows[row].completed_run is None, (
        "a result was attached to a row whose settings had changed"
    )
    # It is still cached under the settings it really used, so going back to
    # them brings it straight back rather than forcing a recomputation.
    assert result_cache.load_cached(source, distorted, launched_with) is not None
    assert result_cache.load_cached(source, distorted, win._rows[row].options) is None


def test_an_unchanged_row_still_receives_its_result(qapp, tmp_path):
    source = tmp_path / "source.mp4"
    distorted = tmp_path / "a.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    _start_fake_run(win, row)

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    win._on_job_finished(0, result)

    assert win._rows[row].completed_run is not None



# ------------------------------- shutdown must not outrun its own threads

class _LiveWorker:
    """A worker that reports itself running until it is told to stop."""

    def __init__(self):
        self.running = True
        self.cancelled = False
        self.waited_ms = []

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def wait(self, ms=None):
        self.waited_ms.append(ms)
        return False  # never finishes within the wait

    def deleteLater(self):
        pass


def test_closing_does_not_accept_while_a_probe_is_still_running(qapp):
    """The reported defect: closeEvent waited a flat five seconds per worker
    and then closed anyway, destroying the widgets those threads were still
    posting into and leaving an ffprobe orphaned."""
    win = MainWindow()
    worker = _LiveWorker()
    win._probe_workers.append(worker)

    event = QCloseEvent()
    win.closeEvent(event)

    assert worker.cancelled, "the probe was never asked to stop"
    assert not event.isAccepted(), "the window closed with a thread still alive"


def test_closing_does_not_block_the_ui_thread_for_seconds(qapp):
    import time

    win = MainWindow()
    win._probe_workers.append(_LiveWorker())

    began = time.monotonic()
    win.closeEvent(QCloseEvent())
    elapsed = time.monotonic() - began

    assert elapsed < 1.0, f"closing blocked the UI thread for {elapsed:.1f}s"


def test_a_second_close_does_not_cancel_twice_but_still_refuses(qapp):
    win = MainWindow()
    worker = _LiveWorker()
    win._probe_workers.append(worker)

    win.closeEvent(QCloseEvent())
    second = QCloseEvent()
    win.closeEvent(second)

    assert not second.isAccepted()
    assert win._closing


def test_the_window_closes_once_the_workers_have_finished(qapp):
    win = MainWindow()
    worker = _LiveWorker()
    win._probe_workers.append(worker)

    win.closeEvent(QCloseEvent())
    assert win._closing

    worker.running = False  # the cancelled probe exits
    event = QCloseEvent()
    win.closeEvent(event)

    assert event.isAccepted(), "the window refused to close with nothing left running"


def test_a_running_vmaf_job_also_holds_the_close(qapp):
    win = MainWindow()
    win._worker = _LiveWorker()

    event = QCloseEvent()
    win.closeEvent(event)

    assert win._worker.cancelled
    assert not event.isAccepted()


def test_closing_with_nothing_running_still_closes_immediately(qapp):
    win = MainWindow()

    event = QCloseEvent()
    win.closeEvent(event)

    assert event.isAccepted()



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


def test_the_user_is_told_which_resolution_tests_were_removed(qapp, tmp_path, monkeypatch):
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    for path in (big, small):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    _add_resolution_test(win, monkeypatch, "1440p")

    messages = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information",
        lambda parent, title, text, *a, **k: messages.append((title, text)),
    )
    win._apply_source_info(small, _fake_video_info_res(str(small), 1280, 720))

    assert messages, "rows vanished with no explanation"
    title, text = messages[0]
    assert "Resolution test" in title
    assert "big" in text, "the message should name what was removed"


def test_ordinary_distorted_rows_survive_a_source_change(qapp, tmp_path, monkeypatch):
    # They are compared against the source, not derived from it, so they
    # remain meaningful -- only their scores are invalidated.
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    encode = tmp_path / "encode.mkv"
    for path in (big, small, encode):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    row = win._add_table_row(encode)
    win._rows[row].video_info = _fake_video_info_res(str(encode), 1920, 1080)
    _add_resolution_test(win, monkeypatch, "1440p")
    assert len(win._rows) == 2

    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **k: None)
    win._apply_source_info(small, _fake_video_info_res(str(small), 1280, 720))

    assert [rd.path for rd in win._rows] == [encode]


def test_a_removed_resolution_tests_curve_goes_with_it(qapp, tmp_path, monkeypatch):
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    for path in (big, small):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    _add_resolution_test(win, monkeypatch, "1440p")
    row = 0
    result = _fake_completed_run(str(win._rows[row].path)).result
    win._rows[row].completed_run = CompletedRun(result, "test")
    win.graph_panel.add_run(
        result, "test", identity=win._rows[row].completed_run.graph_identity
    )
    assert len(win.graph_panel._entries) == 1

    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **k: None)
    win._apply_source_info(small, _fake_video_info_res(str(small), 1280, 720))

    assert len(win.graph_panel._entries) == 0, "a removed row left its curve behind"


def test_no_message_when_there_were_no_resolution_tests(qapp, tmp_path, monkeypatch):
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    for path in (big, small):
        path.write_bytes(b"x" * 100)

    messages = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information",
        lambda *a, **k: messages.append(a),
    )
    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    win._apply_source_info(small, _fake_video_info_res(str(small), 1280, 720))

    assert messages == []


def test_a_new_test_for_the_new_source_is_not_a_duplicate(qapp, tmp_path, monkeypatch):
    # With the stale row gone, adding the same target for the new source
    # cannot collide with a leftover row under the old source's path.
    big = tmp_path / "big.mkv"
    small = tmp_path / "small.mkv"
    for path in (big, small):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(big, _fake_video_info_res(str(big), 3840, 2160))
    _add_resolution_test(win, monkeypatch, "1080p")

    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *a, **k: None)
    win._apply_source_info(small, _fake_video_info_res(str(small), 1920, 1080))
    _add_resolution_test(win, monkeypatch, "720p")

    assert len(win._rows) == 1
    assert win._rows[0].options.resample_test.width == 1280
    assert "small" in win._rows[0].path.name



# --------------------- a loaded run belongs to the source it was measured on

def _saved_run_file(tmp_path, source, distorted, name="run.vmafrun.json"):
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


def test_accepting_the_prompt_switches_the_window_to_the_runs_source(
    qapp, tmp_path, monkeypatch
):
    source_a = tmp_path / "sourceA.mp4"
    source_b = tmp_path / "sourceB.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source_a, source_b, distorted):
        path.write_bytes(b"x" * 100)
    run_file = _saved_run_file(tmp_path, source_a, distorted)

    win = MainWindow()
    win._apply_source_info(source_b, _fake_video_info(str(source_b)))
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question",
        lambda *a, **k: main_window_module.QMessageBox.Yes,
    )
    _load_saved(win, monkeypatch, run_file)

    assert win._same_source(win._source_info.path, source_a)
    assert len(win._rows) == 1
    assert win._rows[0].completed_run is not None


def test_selecting_the_matching_source_keeps_a_loaded_run(qapp, tmp_path, monkeypatch):
    """Reproduction B: loading a run for source A and then selecting A threw
    the run away, because a source change invalidated every row
    indiscriminately. Selecting the very reference a result was measured
    against confirms it -- it cannot invalidate it."""
    source = tmp_path / "sourceA.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)
    run_file = _saved_run_file(tmp_path, source, distorted)

    win = MainWindow()
    _load_saved(win, monkeypatch, run_file)
    assert win._rows[0].completed_run is not None

    win._apply_source_info(source, _fake_video_info(str(source)))

    assert win._rows[0].completed_run is not None, "selecting its own source wiped the run"


def test_selecting_a_different_source_still_invalidates_a_loaded_run(
    qapp, tmp_path, monkeypatch
):
    source_a = tmp_path / "sourceA.mp4"
    source_b = tmp_path / "sourceB.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source_a, source_b, distorted):
        path.write_bytes(b"x" * 100)
    run_file = _saved_run_file(tmp_path, source_a, distorted)

    win = MainWindow()
    _load_saved(win, monkeypatch, run_file)
    win._apply_source_info(source_b, _fake_video_info(str(source_b)))

    assert win._rows[0].completed_run is None


def test_the_same_source_written_two_ways_is_recognised(qapp, tmp_path):
    # A saved run records whatever path was used when it ran. A textual
    # comparison would call one file two different sources.
    win = MainWindow()
    source = tmp_path / "sourceA.mp4"
    source.write_bytes(b"x" * 100)

    assert win._same_source(source, Path(str(source).replace("\\", "/")))
    assert win._same_source(source, tmp_path / "." / "sourceA.mp4")
    assert not win._same_source(source, tmp_path / "other.mp4")
    assert not win._same_source(None, source)



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


def test_a_row_without_a_source_cannot_be_compared(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("encode.mkv"))
    win._rows[row].video_info = _fake_video_info_res("encode.mkv", 1920, 1080)

    win._sync_frame_compare()

    assert win.frame_compare_panel._entries == []


def test_a_row_still_being_read_is_not_offered(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mkv", 3840, 2160)
    win._add_table_row(Path("encode.mkv"))  # video_info still None

    win._sync_frame_compare()

    assert win.frame_compare_panel._entries == []


def test_an_unscored_row_uses_the_rows_own_scaling_settings(qapp):
    from vmaf_app.core.models import ScaleDirection

    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mkv", 3840, 2160)
    row = win._add_table_row(Path("encode.mkv"))
    win._rows[row].video_info = _fake_video_info_res("encode.mkv", 1920, 1080)
    win._rows[row].options.scale_direction = ScaleDirection.DISTORTED_TO_SOURCE
    win._rows[row].options.scale_algorithm = "lanczos"

    win._sync_frame_compare()
    comparison = win.frame_compare_panel._entries[0].comparison

    assert comparison.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE
    assert comparison.scale_algorithm == "lanczos"
    # Upscaling the distorted side means both are compared at the source's
    # size, exactly as a run would.
    from vmaf_app.core.frame_extract import comparison_dimensions
    assert comparison_dimensions(comparison) == (3840, 2160)


def test_an_unscored_rows_timeline_stops_at_the_shorter_input(qapp):
    # The same bound a run uses, so the slider cannot offer frames that no
    # comparison would ever produce.
    win = MainWindow()
    source = _fake_video_info_res("source.mkv", 1920, 1080)
    source.nb_frames = 300
    win._source_info = source
    row = win._add_table_row(Path("encode.mkv"))
    distorted = _fake_video_info_res("encode.mkv", 1920, 1080)
    distorted.nb_frames = 250
    win._rows[row].video_info = distorted

    win._sync_frame_compare()

    assert win.frame_compare_panel._entries[0].comparison.frame_count == 250


@pytest.mark.parametrize(
    ("crop_mode", "pending"),
    [(CropMode.AUTO, True), (CropMode.NONE, False), (CropMode.MANUAL, False)],
)
def test_only_auto_crop_is_reported_as_pending(qapp, crop_mode, pending):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mkv", 1920, 1080)
    row = win._add_table_row(Path("encode.mkv"))
    win._rows[row].video_info = _fake_video_info_res("encode.mkv", 1920, 1080)
    win._rows[row].options.crop_mode = crop_mode

    win._sync_frame_compare()

    assert win.frame_compare_panel._entries[0].comparison.auto_crop_pending is pending


def test_a_manual_crop_is_applied_to_an_unscored_preview(qapp):
    from vmaf_app.core.models import CropBox

    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mkv", 1920, 1080)
    row = win._add_table_row(Path("encode.mkv"))
    win._rows[row].video_info = _fake_video_info_res("encode.mkv", 1920, 1080)
    win._rows[row].options.crop_mode = CropMode.MANUAL
    win._rows[row].options.manual_source_crop = CropBox(w=1920, h=816, x=0, y=132)
    win._rows[row].options.manual_distorted_crop = CropBox(w=1920, h=816, x=0, y=132)

    win._sync_frame_compare()
    comparison = win.frame_compare_panel._entries[0].comparison

    # Manual crops are known without running anything, so they are exact.
    assert comparison.source_crop.h == 816
    assert comparison.distorted_crop.h == 816


def test_a_resolution_test_row_can_be_previewed_before_it_runs(qapp, tmp_path, monkeypatch):
    source = tmp_path / "master.mkv"
    source.write_bytes(b"x" * 100)

    win = MainWindow()
    win._apply_source_info(source, _fake_video_info_res(str(source), 3840, 2160))
    monkeypatch.setattr(
        main_window_module.QInputDialog, "getItem", lambda *a, **k: ("1080p", True)
    )
    win._on_add_resample_test()

    win._sync_frame_compare()
    entries = win.frame_compare_panel._entries

    assert len(entries) == 1
    # Both sides decode the source; the "distorted" one is synthesised.
    assert entries[0].comparison.resample_target.width == 1920
    assert entries[0].comparison.distorted_info.path == source


def test_running_a_row_upgrades_its_entry_to_the_real_geometry(qapp, tmp_path):
    from vmaf_app.core.frame_extract import FrameComparison
    from vmaf_app.core.models import CropBox

    source = tmp_path / "source.mp4"
    distorted = tmp_path / "encode.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x" * 100)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    win._rows[row].video_info = _fake_video_info(str(distorted))
    win._sync_frame_compare()
    assert win.frame_compare_panel._entries[0].scores is None

    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted
    result.source_crop = CropBox(w=1920, h=816, x=0, y=132)
    win._rows[row].completed_run = CompletedRun(result, "encode")
    win._sync_frame_compare()

    entry = win.frame_compare_panel._entries[0]
    assert entry.scores is not None
    # The detected crop now comes from the run rather than being pending.
    from dataclasses import replace
    assert entry.comparison == replace(
        FrameComparison.from_result(result), source_info=win._source_info,
        distorted_info=win._rows[row].video_info,
    )
    assert entry.comparison.auto_crop_pending is False



# ------------------------------ progress and status with two videos running

def test_each_running_video_gets_its_own_progress_line(qapp):
    """Two videos at once need two readable lines, not one bar flickering
    between them."""
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 1000]

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


def test_a_finished_video_frees_its_progress_line_for_the_next(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100, 100, 100]

    win._on_job_started(0, "a")
    win._on_job_started(1, "b")
    slot_of_a = win._job_line_slot[0]
    win._mark_job_over(0)
    win._on_job_started(2, "c")

    assert win._job_line_slot[2] == slot_of_a, "the freed line was not reused"
    assert 0 not in win._job_line_slot


def test_a_phase_message_is_attached_to_the_video_it_came_from(qapp):
    """With two running, whichever lane spoke last used to own the single
    status line -- so "Detecting black bars..." appeared with no way to tell
    which file it referred to, and it erased what the line was for."""
    win = MainWindow()
    for name in ("encode-a.mp4", "encode-b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100, 100]
    win._on_job_started(0, "encode-a")
    win._on_job_started(1, "encode-b")

    win._on_job_status(1, "Detecting black bars in source...")

    assert "Detecting black bars" in win.job_progress_labels[1].text()
    assert "encode-b" in win.job_progress_labels[1].text()
    # And it does not take over the shared line.
    assert win.status_label.text().startswith("Running 2 of 2 together")


@pytest.mark.parametrize("job_count", [1, 2])
def test_decode_status_survives_progress_and_tracks_fallback(qapp, job_count):
    win = MainWindow()
    for i in range(job_count):
        win._add_table_row(Path(f"encode-{i}.mp4"))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100] * job_count
    for i in range(job_count):
        win._on_job_started(i, f"encode-{i}")
        win._on_job_status(i, "Running ffmpeg (GPU decode: source cuda, distorted cuda)...")
        win._on_job_progress(i, current=20, total=100, fps=10.0)
        text = win.job_progress_labels[win._job_line_slot[i]].text()
        assert "Decode: source cuda, test cuda" in text
        assert "20%" in text and "10.0 fps" in text

    for plan, expected in (
        ("source cuda, distorted cpu", "Decode: source cuda, test CPU"),
        ("off", "Decode: source CPU, test CPU"),
    ):
        win._on_job_status(0, f"GPU decode failed, retrying (GPU decode: {plan})...")
        win._on_job_progress(0, current=30, total=100, fps=5.0)
        assert expected in win.job_progress_labels[win._job_line_slot[0]].text()
        if job_count == 2:
            assert "source cuda, test cuda" in win.job_progress_labels[win._job_line_slot[1]].text()

    slot = win._job_line_slot[0]
    win._mark_job_over(0)
    assert 0 not in win._job_decode_status
    row = win._add_table_row(Path("next.mp4"))
    win._job_rows.append(win._rows[row])
    win._job_total_frames.append(100)
    win._on_job_started(job_count, "next")
    win._on_job_progress(job_count, current=1, total=100, fps=1.0)
    assert "Decode:" not in win.job_progress_labels[slot].text()


def test_a_single_jobs_phase_also_stays_on_its_own_line(qapp):
    # Even alone, the phase belongs to the video rather than to the run: the
    # shared line is what says when the queue ends.
    win = MainWindow()
    win._add_table_row(Path("a.mp4"))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100]
    win._on_job_started(0, "a")

    win._on_job_status(0, "Running ffmpeg (GPU decode: off)...")

    assert "GPU decode" in win.job_progress_labels[0].text()
    assert win.status_label.text().startswith("Running 1 of 1")


# ------------------------------------------------- queue ETA is a makespan

def test_queue_eta_is_bounded_by_the_slowest_remaining_video(qapp):
    """Dividing remaining frames by the summed frame rate assumes every lane
    stays busy to the same instant. With 10 seconds left on one video and
    1000 on the other it claimed about 505s, when the queue plainly cannot
    end before the 1000-second one does."""
    win = MainWindow()
    for name in ("quick.mp4", "slow.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win.parallel_jobs_check.setChecked(True)
    win._job_total_frames = [10, 1000]

    # Both running at 1 fps: 10s left on one, 1000s on the other.
    win._on_job_progress(0, current=0, total=10, fps=1.0)
    win._on_job_progress(1, current=0, total=1000, fps=1.0)

    assert win._queue_eta_seconds() == pytest.approx(1000.0)


def test_queue_eta_schedules_waiting_videos_onto_free_lanes(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win.parallel_jobs_check.setChecked(True)
    win._job_total_frames = [100, 100, 100, 100]

    # Two running at 1 fps with 100s each; two more queued at the same rate.
    win._on_job_progress(0, current=0, total=100, fps=1.0)
    win._on_job_progress(1, current=0, total=100, fps=1.0)

    # Each lane takes one of the queued videos: 100s now + 100s after.
    assert win._queue_eta_seconds() == pytest.approx(200.0)


def test_queue_eta_is_unknown_until_something_reports_a_rate(qapp):
    win = MainWindow()
    win._add_table_row(Path("a.mp4"))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100]

    assert win._queue_eta_seconds() is None


def test_the_parallel_control_stays_usable_during_a_run(qapp):
    # The Settings tab is locked during a run, which is why this lives on the
    # Videos tab -- being unable to reach it mid-run was the whole complaint.
    win = MainWindow()
    win._set_run_ui_active(True)

    assert win.parallel_jobs_check.isEnabled()
    assert not win.run_btn.isEnabled()
    assert not win.options_box.isEnabled()


def test_changing_the_control_reaches_a_running_worker(qapp):
    class FakeWorker:
        def __init__(self):
            self.applied = []

        def isRunning(self):
            return True

        def set_parallel_jobs(self, count):
            self.applied.append(count)

    win = MainWindow()
    win._worker = FakeWorker()

    win.parallel_jobs_check.setChecked(True)

    assert win._worker.applied == [2]
    assert win._settings.parallel_jobs == 2



def test_the_queue_eta_rides_on_the_status_line(qapp):
    """There is one shared line, not two. It says how much is running and
    when the queue ends, and nothing that changes several times a second."""
    win = MainWindow()
    for name in ("encode-a.mp4", "encode-b.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000, 1000]
    win._on_job_started(0, "encode-a")
    win._on_job_started(1, "encode-b")

    win._on_job_progress(0, current=500, total=1000, fps=25.0)
    first = win.status_label.text()
    win._on_job_progress(1, current=500, total=1000, fps=25.0)
    second = win.status_label.text()

    for text in (first, second):
        assert text.startswith("Running 2 of 2 together")
        assert "Queue ETA:" in text
        assert "fps" not in text
        assert "encode-a" not in text and "encode-b" not in text
    assert second == "Running 2 of 2 together   ·   Queue ETA: 0:00:20"
    assert not hasattr(win, "progress_detail_label")


def test_there_are_no_progress_bars_at_all(qapp):
    # Each running video states its own percentage, and the queue line states
    # the ETA. Nothing is left for a bar to add.
    win = MainWindow()
    assert not hasattr(win, "progress_bar")
    assert not hasattr(win, "job_progress_bars")



def test_a_running_video_states_its_percentage_in_words(qapp):
    """A bar shows roughly how far along a video is; the number says exactly,
    in the same line that already carries the name, rate and time left."""
    win = MainWindow()
    win._add_table_row(Path("encode.mp4"))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000]
    win._on_job_started(0, "encode")

    win._on_job_progress(0, current=333, total=1000, fps=20.0)

    text = win.job_progress_labels[0].text()
    assert text == "encode — 33%   ·   20.0 fps   ·   0:00:33 left"


def test_a_video_with_no_rate_yet_shows_only_its_percentage(qapp):
    win = MainWindow()
    win._add_table_row(Path("encode.mp4"))
    win._job_rows = list(win._rows)
    win._job_total_frames = [1000]
    win._on_job_started(0, "encode")

    win._on_job_progress(0, current=5, total=1000, fps=0.0)

    assert win.job_progress_labels[0].text() == "encode — 0%"


def test_a_finished_video_leaves_its_line(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        win._add_table_row(Path(name))
    win._job_rows = list(win._rows)
    win._job_total_frames = [100, 100, 100]
    win._on_job_started(0, "a")
    win._on_job_started(1, "b")

    win._mark_job_over(0)

    assert not win.job_progress_labels[0].isVisible()
    assert 0 not in win._job_line_slot
