from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QHeaderView, QTableWidgetSelectionRange

from vmaf_app.core.models import (
    CropMode,
    FrameScore,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafRunResult,
    synthetic_resample_distorted_path,
    synthetic_scale_direction_variant_path,
)
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import (
    COL_BITRATE,
    COL_CHECK,
    COL_INFO,
    COL_PATH,
    COL_PSNR,
    COL_SCALING,
    COL_SSIM,
    COL_VMAF,
    COL_XPSNR,
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


def test_run_clicked_skips_rows_that_already_have_a_score(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")

    for name in ("a.mp4", "b.mp4"):
        row = win._add_table_row(Path(name))
        win._rows[row].video_info = _fake_video_info(name)
        win._rows[row].completed_run = _fake_completed_run(name)
        win._set_row_vmaf_text(row, "90.00", bold=True)

    win._on_run_clicked()

    assert win._worker is None  # nothing needed running, so no worker was ever started
    assert "already have a VMAF score" in win.status_label.text()
    assert win.graph_panel is not None
    assert len(win.graph_panel._entries) == 2


def test_run_clicked_only_queues_unscored_rows(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")

    scored_row = win._add_table_row(Path("scored.mp4"))
    win._rows[scored_row].video_info = _fake_video_info("scored.mp4")
    win._rows[scored_row].completed_run = _fake_completed_run("scored.mp4")

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
    win._rows[row_a].options.gpu_decode_source = False
    win._rows[row_b].options.n_threads = 11
    win._rows[row_b].options.gpu_decode_source = True
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
    assert win._rows[row_a].options.gpu_decode_source is False
    assert win._rows[row_b].options.gpu_decode_source is True


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
    win._on_compare_selected()
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
    for col in (COL_CHECK, COL_PATH, COL_INFO, COL_SCALING, COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR):
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
        for c in (COL_CHECK, COL_INFO, COL_SCALING, COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR)
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
    win.resize(1280, 800)
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
    assert item.toolTip() == str(Path("C:/videos/some_encode.mp4"))


def test_resize_mismatch_note_reflects_the_actual_scale_direction_used(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 3840, 2160)
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))
    tag = win.distorted_table.item(row, COL_SCALING).text()
    tip = win.distorted_table.item(row, COL_SCALING).toolTip()
    assert tag == "↓ source"
    assert "Source downscaled" in tip
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
    assert win.distorted_table.item(row, COL_SCALING).text() == "↑ distorted"


def test_resize_mismatch_note_absent_when_resolutions_match(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info_res("source.mp4", 1920, 1080)
    row = win._add_table_row(Path("a.mp4"))
    win._set_row_info(row, _fake_video_info_res("a.mp4", 1920, 1080))
    assert win.distorted_table.item(row, COL_SCALING).text() == ""


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

    assert win.distorted_table.item(companion_row, COL_SCALING).text() == "↑ distorted"


def test_no_horizontal_scrollbar_at_default_with_a_typical_row(qapp):
    win = MainWindow()
    win.resize(1280, 800)
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
        for c in (COL_CHECK, COL_PATH, COL_INFO, COL_SCALING, COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR)
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
        fps=30.0, model="version=vmaf_v0.6.1", source_crop=None,
        distorted_crop=None, source_info=info, distorted_info=info,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName",
        lambda *a, **kw: ("saved.vmafrun.json", ""),
    )
    monkeypatch.setattr(main_window_module, "load_run", lambda _path: (result, "saved"))

    win._on_load_saved_run()

    assert win.distorted_table.item(0, COL_PSNR).text() == "43.00"
    assert win.distorted_table.item(0, COL_SSIM).text() == "0.9883"
    assert win.distorted_table.item(0, COL_XPSNR).text() == "40.00"


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

    assert not app_result.exists()
    assert unrelated.read_text(encoding="utf-8") == "important"


def test_adding_a_row_picks_up_a_cached_result(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "_cache_dir", lambda: tmp_path)

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


def test_finishing_a_job_persists_to_cache(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "_cache_dir", lambda: tmp_path)

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

    assert result_cache.load_cached(source, distorted, win._rows[row].options) is not None


def test_finishing_an_old_job_cannot_attach_or_cache_it_under_a_new_source(
    qapp, tmp_path, monkeypatch
):
    from vmaf_app.core import result_cache

    monkeypatch.setattr(result_cache, "_cache_dir", lambda: tmp_path)
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

    assert win._rows[row].completed_run is None
    assert result_cache.load_cached(old_source, distorted, win._rows[row].options) is not None
    assert result_cache.load_cached(new_source, distorted, win._rows[row].options) is None


def test_recompute_clears_row_and_deletes_cache_entry(qapp, tmp_path, monkeypatch):
    from vmaf_app.core import result_cache
    monkeypatch.setattr(result_cache, "_cache_dir", lambda: tmp_path)

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

    # probe_video must never be called for a resample row -- there's no real
    # distorted file on disk at its synthetic path to probe.
    monkeypatch.setattr(
        main_window_module, "probe_video",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("probe_video should not be called")),
    )

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


# ------------------------------------------------------------------ fps / ETA display

def test_job_progress_shows_fps_and_file_eta(qapp):
    win = MainWindow()
    row = win._add_table_row(Path("a.mp4"))
    win._job_rows = [win._rows[row]]
    win._job_total_frames = [3000]  # a single 3000-frame job

    win._on_job_progress(0, current=1000, total=3000, fps=100.0)

    # 2000 frames remaining at 100fps -> 20s
    assert "100.0 fps" in win.progress_detail_label.text()
    assert "File ETA: 0:00:20" in win.progress_detail_label.text()
    # Live progress belongs in the status bar above, not the VMAF column --
    # that's reserved for the final score (or "Failed").
    assert win.distorted_table.item(row, COL_VMAF).text() == ""


def test_job_progress_queue_eta_accounts_for_other_queued_jobs(qapp):
    win = MainWindow()
    row_a = win._add_table_row(Path("a.mp4"))
    row_b = win._add_table_row(Path("b.mp4"))
    win._job_rows = [win._rows[row_a], win._rows[row_b]]
    win._job_total_frames = [1000, 4000]  # job 0 already fully done (1000 frames), job 1 in progress

    # Now on job index 1 (the second job), 2000/4000 frames in, at 50fps.
    win._on_job_progress(1, current=2000, total=4000, fps=50.0)

    # queue_total = 5000, queue_done = 1000 (job 0, all of it) + 2000 (job 1 so far) = 3000
    # queue_remaining = 2000 frames at 50fps = 40s
    assert "Queue ETA: 0:00:40" in win.progress_detail_label.text()
    # File ETA: 2000 frames remaining in job 1 at 50fps = 40s (coincidentally also 40s here)
    assert "File ETA: 0:00:40" in win.progress_detail_label.text()
    assert "(file 2 of 2)" in win.progress_detail_label.text()


def test_job_progress_with_zero_fps_shows_no_eta(qapp):
    win = MainWindow()
    win._job_rows = [win._rows[win._add_table_row(Path("a.mp4"))]]
    win._job_total_frames = [3000]

    win._on_job_progress(0, current=5, total=3000, fps=0.0)

    assert "ETA" not in win.progress_detail_label.text()
    assert "file 1 of 1" in win.progress_detail_label.text()


def test_cancelled_run_does_not_claim_done_or_force_100_percent(qapp):
    win = MainWindow()
    win.progress_bar.setValue(37)
    win._on_run_cancelled()

    win._on_all_finished()

    assert win.status_label.text() == "Cancelled."
    assert win.progress_bar.value() == 37


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

    assert not win.files_box.isEnabled()
    assert not win.options_box.isEnabled()
    assert not win.tabs.isTabEnabled(TAB_SETTINGS)
    assert not win.run_btn.isEnabled()
    assert win.pause_btn.isEnabled()
    assert win.cancel_btn.isEnabled()

    win._set_run_ui_active(False)
    assert win.files_box.isEnabled()
    assert win.options_box.isEnabled()
    assert win.tabs.isTabEnabled(TAB_SETTINGS)


# ------------------------------------------------------------------ metric columns

def test_metric_columns_show_na_until_enabled_and_computed(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))

    # PSNR/SSIM/XPSNR are off by default -> "N/A"; VMAF is always computed.
    assert win.distorted_table.item(row, COL_PSNR).text() == "N/A"
    assert win.distorted_table.item(row, COL_SSIM).text() == "N/A"
    assert win.distorted_table.item(row, COL_XPSNR).text() == "N/A"
    assert win.distorted_table.item(row, COL_VMAF).text() == ""  # enabled, just not run yet


def test_ticking_a_metric_column_header_enables_it_for_every_row(qapp):
    win = MainWindow()
    for name in ("a.mp4", "b.mp4"):
        win._add_table_row(Path(name))

    win._on_metric_column_toggled(COL_PSNR, True)

    assert all("name=psnr" in r.options.extra_features for r in win._rows)
    assert all(win.distorted_table.item(r, COL_PSNR).text() != "N/A" for r in range(2))

    win._on_metric_column_toggled(COL_PSNR, False)
    assert all("name=psnr" not in r.options.extra_features for r in win._rows)
    assert win.distorted_table.item(0, COL_PSNR).text() == "N/A"


def test_changing_a_metric_marks_an_existing_result_stale_and_runnable(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win.graph_panel.add_run(win._rows[row].completed_run.result, "a")

    win._on_metric_column_toggled(COL_PSNR, True)

    assert win._rows[row].completed_run is None
    assert win.distorted_table.item(row, COL_VMAF).text() == ""
    assert not win.graph_panel._entries


def test_changing_a_calculation_option_marks_an_existing_result_stale(qapp):
    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("a.mp4"))
    win._rows[row].completed_run = _fake_completed_run("a.mp4")
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()

    win.subsample_spin.setValue(5)

    assert win._rows[row].completed_run is None


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

    win._start_probe([Path("a.mp4")])
    old = FakeProbeWorker.instances[-1]
    win._start_probe([Path("a.mp4")])

    assert old.cancelled
    assert old in win._probe_workers
    stale_info = _fake_video_info("stale.mp4")
    old.probed.emit(Path("a.mp4"), stale_info, "")
    assert win._rows[row].video_info is None


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

def test_the_window_has_videos_graph_and_settings_tabs(qapp):
    win = MainWindow()
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Videos", "Graph", "Settings"]


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
    win._on_compare_selected()
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
