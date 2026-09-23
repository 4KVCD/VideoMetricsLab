"""Metric-neutral journeys, including real FFmpeg (no mocked scores)."""
import itertools
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app.core import result_cache
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    supplemental_metric_specs,
)
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropMode, FrameScores, ResampleTarget, VmafOptions
from vmaf_app.core.run_io import export_csv, load_run, save_run
from vmaf_app.core.settings import Settings
from vmaf_app.core.vmaf_runner import VmafRunError, run_resample_test, run_vmaf
from vmaf_app.ui.main_window import (
    COL_PSNR,
    COL_SSIM,
    COL_VMAF,
    COL_XPSNR,
    MainWindow,
)


def click_metric(win, row, column):
    """Ticks/unticks a metric in the table, the way a click on the cell does."""
    item = win.distorted_table.item(row, column)
    item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)



def _cache_request(options):
    return analysis_request_from_vmaf_options(options)


def _cache_key(source, distorted, options):
    return result_cache.cache_key(source, distorted, _cache_request(options))


def _load_cached(source, distorted, options, directory=None):
    return result_cache.load_cached(
        source, distorted, _cache_request(options), directory,
        supplemental_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(
        source, distorted, result, label, _cache_request(options), directory
    )


def _clear_cached(source, distorted, options, directory=None):
    return result_cache.clear(
        source, distorted, _cache_request(options), directory,
        supplemental_metric_specs(options),
    )

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def real_pair(tmp_path_factory):
    folder = tmp_path_factory.mktemp("independent-metrics")
    source, test = folder / "reference.mkv", folder / "test.mkv"
    exe = ffmpeg_path()
    subprocess.run([exe, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=192x108:rate=24:duration=0.5", "-c:v", "ffv1", str(source)],
                   check=True, capture_output=True, timeout=30)
    subprocess.run([exe, "-nostdin", "-v", "error", "-i", str(source),
                    "-c:v", "libx264", "-crf", "35", "-an", str(test)],
                   check=True, capture_output=True, timeout=30)
    return probe_video(source), probe_video(test)


def options_for(names):
    return VmafOptions(
        compute_vmaf="vmaf" in names, compute_xpsnr="xpsnr" in names,
        extra_features=[feature for metric, feature in (("psnr", "name=psnr"), ("ssim", "name=float_ssim")) if metric in names],
        crop_mode=CropMode.NONE, gpu_decode=False, n_threads=2,
    )


COMBINATIONS = [names for n in range(1, 5) for names in itertools.combinations(("vmaf", "psnr", "ssim", "xpsnr"), n)]


def test_subsampled_cache_cannot_replace_full_frame_xpsnr(real_pair):
    mixed = options_for(("vmaf", "xpsnr"))
    mixed.n_subsample = 3
    alone = options_for(("xpsnr",))
    alone.n_subsample = 3
    source, distorted = real_pair
    sampled = run_vmaf(source, distorted, mixed)
    full = run_vmaf(source, distorted, alone)
    assert len(sampled.frames) == 4
    assert len(full.frames) == 12
    _store_cached(source.path, distorted.path, sampled, "sampled", mixed)
    assert _load_cached(source.path, distorted.path, alone) is None
    _clear_cached(source.path, distorted.path, alone)
    _store_cached(source.path, distorted.path, full, "full", alone)
    # The subsampled run stored first is still a valid answer for the
    # subsampled request (it was never cleared); the full-frame XPSNR must
    # not stand in for it.
    reloaded, _label = _load_cached(source.path, distorted.path, mixed)
    assert len(reloaded.metric("xpsnr").frame) == 4
    assert len(reloaded.metric("vmaf").frame) == 4


@pytest.mark.parametrize("names", COMBINATIONS)
def test_real_metric_combinations_and_portable_roundtrip(real_pair, tmp_path, names):
    options = options_for(names)
    if "vmaf" not in names:
        # Disabling VMAF must also bypass a stale/missing custom model.
        options.model_choice = "__custom__"
        options.model = "path=does-not-exist.json"
        options.custom_model_path = "does-not-exist.json"
    result = run_vmaf(*real_pair, options)
    assert len(result.frames) == 12
    assert tuple(m for m in ("vmaf", "psnr", "ssim", "xpsnr") if result.frames.has(m)) == names
    assert all(np.isfinite(result.frames.values(m)).all() for m in names)
    assert bool(result.model) == ("vmaf" in names)
    path = tmp_path / "portable.metrics.json"
    save_run(result, path)
    loaded, _ = load_run(path)
    np.testing.assert_array_equal(loaded.frames.frame, result.frames.frame)
    np.testing.assert_allclose(loaded.frames.time, result.frames.time, atol=5e-7)
    for name in names:
        np.testing.assert_array_equal(loaded.frames.values(name), result.frames.values(name))
    assert loaded.frames.has("vmaf") == ("vmaf" in names)
    assert loaded.frames[:2].nbytes() > 0
    export_csv(loaded, tmp_path / "metrics.csv")
    _store_cached(result.source, result.distorted, result, "test", options)
    cached = _load_cached(result.source, result.distorted, options)
    assert cached is not None and cached[0].frames == loaded.frames


def test_feature_only_scores_match_existing_definitions(real_pair):
    all_scores = run_vmaf(*real_pair, options_for(("vmaf", "psnr", "ssim", "xpsnr"))).frames
    for name in ("psnr", "ssim", "xpsnr"):
        alone = run_vmaf(*real_pair, options_for((name,))).frames
        np.testing.assert_allclose(alone.values(name), all_scores.values(name), rtol=1e-6)


def test_non_vmaf_resample_and_subsample(real_pair):
    options = options_for(("psnr", "ssim"))
    options.n_subsample = 3
    options.resample_test = ResampleTarget(width=128, label="custom")
    result = run_resample_test(real_pair[0], options)
    np.testing.assert_array_equal(result.frames.frame, [0, 3, 6, 9])
    assert result.frames.vmaf is None
    options = options_for(("xpsnr",))
    options.n_subsample = 3  # libvmaf-only setting must not subsample XPSNR.
    assert len(run_vmaf(*real_pair, options).frames) == 12


def test_empty_selection_is_rejected(real_pair):
    with pytest.raises(VmafRunError, match="at least one metric"):
        run_vmaf(*real_pair, options_for(()))


def test_missing_vmaf_arrays_have_normal_sequence_semantics():
    scores = FrameScores(np.arange(2), np.arange(2) / 24, None, psnr=[30, 40])
    assert scores[0].vmaf is None
    assert FrameScores.from_frames(list(scores)) == scores
    assert scores[:1].vmaf is None
    assert scores.nbytes() == 2 * (4 + 8 + 4)


def test_select_calculate_load_graph_and_compare_without_vmaf(qapp, real_pair, tmp_path, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(win.bitrate_panel, "add_and_analyze", lambda *_: None)
    monkeypatch.setattr(win, "_reload_cached_for_rows", lambda *_: None)
    source, test = real_pair
    win._source_info = source
    row = win._add_table_row(test.path)
    win._set_row_info(row, test)
    win.distorted_table.selectRow(row)
    # Narrow the default (all four) down to PSNR alone.
    for column in (COL_VMAF, COL_SSIM, COL_XPSNR):
        click_metric(win, row, column)
    assert win._rows[row].options.requested_metrics() == ("psnr",)
    assert not win.model_combo.isEnabled()
    win._rows[row].options = options_for(("psnr",))
    result = run_vmaf(source, test, win._rows[row].options)
    win._job_rows = [win._rows[row]]
    win._on_job_started(0, "test")
    assert win._row_state(win._rows[row]) == "Calculating"
    win._on_job_finished(0, result)
    assert win._row_state(win._rows[row]) == "Complete"
    assert win._has_requested_results(win._rows[row])
    assert win.graph_panel._current_metric().key == "psnr"
    assert "PSNR" in win.frame_compare_panel.detail_label.text()
    assert "VMAF" not in win.frame_compare_panel.detail_label.text()
    # Adding a requested metric preserves the measured score, not a stale blank.
    click_metric(win, row, COL_SSIM)
    assert win._row_state(win._rows[row]) == "Partially calculated"
    assert not win._has_requested_results(win._rows[row])
    assert win._rows[row].completed_run.result.frames.has("psnr")
    assert len(win.graph_panel._entries) == 1
    # Exercise the real Calculate button, QThread and completion signals
    # for the second run; it must replace, not duplicate, the graph series.
    win.run_btn.click()
    deadline = time.monotonic() + 15
    while win._run_active and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not win._run_active, "real metric worker did not finish"
    assert win._has_requested_results(win._rows[row])
    assert win._rows[row].completed_run.result.frames.has("ssim")
    assert len(win.graph_panel._entries) == 1
    path = tmp_path / "results.metrics.json"
    save_run(result, path)
    monkeypatch.setattr("vmaf_app.ui.main_window.QFileDialog.getOpenFileName", lambda *_: (str(path), ""))
    win._on_load_saved_run()
    assert win._rows[-1].options.requested_metrics() == ("psnr",)
    assert win._row_state(win._rows[-1]) == "Complete"
    win.close()


def test_settings_save_failure_is_reported_without_row_access(qapp, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(win._settings, "save", lambda: "Cannot save settings")
    # The preview-setting path reports errors in the global status line,
    # not a nonexistent row in the video's status column.
    win._on_frame_color_mode_changed("display_aware")
    assert win.status_label.text() == "Cannot save settings"
    win.close()


def test_metric_scope_mixed_selection_and_graph_preference(qapp, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(win, "_reload_cached_for_rows", lambda *_: None)
    for name in ("a.mkv", "b.mkv"):
        win._add_table_row(Path(name))
    win.distorted_table.selectRow(0)
    click_metric(win, 0, COL_PSNR)
    assert "psnr" not in win._rows[0].options.requested_metrics()
    assert "psnr" in win._rows[1].options.requested_metrics()
    # A row added later does not inherit an untick made on one existing row.
    assert "psnr" in win._default_options.requested_metrics()
    # Unticking the still-ticked row while both are selected applies to both.
    win.distorted_table.selectAll()
    click_metric(win, 1, COL_PSNR)
    assert all("psnr" not in rd.options.requested_metrics() for rd in win._rows)
    win.graph_panel.tabs.setCurrentIndex(3)
    assert Settings.load().graph_metric == "ssim"
    assert "not calculated" in win.graph_panel.metric_hint.text()
    win.close()
