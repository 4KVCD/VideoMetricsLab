import json
import shutil
import subprocess

import numpy as np
import pytest

from vmaf_app.core.models import FrameScores, VmafOptions
from vmaf_app.core.run_io import _frames_to_rows, _rows_to_frames
from vmaf_app.core.vmaf_runner import _build_libvmaf_opts, _parse_log


def test_neg_arrays_roundtrip_slice_and_iteration():
    frames = FrameScores(np.arange(2), np.arange(2), np.array([90, 91]),
                         vmaf_neg=np.array([80, 81]))
    assert _rows_to_frames(_frames_to_rows(frames)) == frames
    assert frames[1].vmaf_neg == 81
    assert frames[:1].vmaf_neg.tolist() == [80]
    assert frames.with_values("psnr", None).vmaf_neg.tolist() == [80, 81]


@pytest.mark.parametrize("standard", [True, False])
def test_real_ffmpeg_neg_is_independent(tmp_path, standard):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    log = tmp_path / "metrics.json"
    options = VmafOptions(compute_vmaf=standard, compute_vmaf_neg=True, n_threads=2)
    tail = ":".join(_build_libvmaf_opts(options, log))
    result = subprocess.run([
        ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=192x108:rate=4:duration=1",
        "-filter_complex", "[0:v]split=2[a][b];[a][b]libvmaf=" + tail,
        "-f", "null", "-",
    ], cwd=tmp_path, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    frames = _parse_log(log, 4)
    assert len(frames) == 4
    assert frames.has("vmaf") == standard
    assert frames.has("vmaf_neg")
    assert np.isfinite(frames.vmaf_neg).all()


def test_legacy_neg_run_loads_only_in_neg_column(tmp_path):
    from tests.test_run_io import _sample_result
    from vmaf_app.core.run_io import load_run, save_run
    result = _sample_result()
    target = tmp_path / "old.json"
    save_run(result, target)
    payload = json.loads(target.read_text())
    payload["model"] = "version=vmaf_v0.6.1neg"
    for row in payload["frames"]:
        del row[6:]
    target.write_text(json.dumps(payload))
    restored, _ = load_run(target)
    assert restored.frames.vmaf is None
    np.testing.assert_array_equal(restored.frames.vmaf_neg, result.frames.vmaf)


def test_ui_columns_select_and_display_independent_scores():
    from PySide6.QtWidgets import QApplication

    from tests.test_main_window import _fake_completed_run
    from vmaf_app.ui.main_window import COL_VMAF, COL_VMAF_NEG, MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    options = VmafOptions()
    win._set_metric_option(options, COL_VMAF_NEG, True)
    assert options.compute_vmaf and options.compute_vmaf_neg
    win._set_metric_option(options, COL_VMAF, False)
    assert options.requested_metrics() == ("vmaf_neg",)
    run = _fake_completed_run("test.mp4")
    run.result.frames = run.result.frames.with_values("vmaf_neg", np.full(len(run.result.frames), 75))
    assert win._metric_mean(run, COL_VMAF) == 90
    assert win._metric_mean(run, COL_VMAF_NEG) == 75
    win.close()
    app.processEvents()


def test_legacy_neg_cache_is_found_without_rewriting(tmp_path):
    from tests.test_run_io import _sample_result
    from vmaf_app.core import result_cache
    from vmaf_app.core.run_io import save_run
    options = VmafOptions(compute_vmaf=False, compute_vmaf_neg=True)
    legacy = VmafOptions()
    legacy.model = legacy.model_choice = "version=vmaf_v0.6.1neg"
    result = _sample_result()
    target = result_cache._cache_path(result.source, result.distorted, legacy, tmp_path)
    result.model = legacy.model
    save_run(result, target)
    original = target.read_bytes()
    found = result_cache.load_cached(result.source, result.distorted, options, tmp_path)
    assert found is not None
    assert found[0].frames.vmaf is None
    assert found[0].frames.vmaf_neg is not None
    assert target.read_bytes() == original


@pytest.mark.parametrize("standard", [True, False])
def test_full_runner_returns_requested_neg_scores(tmp_path, standard):
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.models import CropMode
    from vmaf_app.core.vmaf_runner import run_vmaf

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    path = tmp_path / "fixture.mkv"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=192x108:rate=4:duration=1", "-c:v", "ffv1", str(path)],
                   check=True, capture_output=True, timeout=30)
    info = probe_video(path)
    options = VmafOptions(compute_vmaf=standard, compute_vmaf_neg=True,
                          gpu_decode=False, crop_mode=CropMode.NONE, n_threads=2)
    result = run_vmaf(info, info, options)
    assert len(result.frames) == 4
    assert result.frames.has("vmaf") == standard
    assert result.frames.has("vmaf_neg")
