import shutil
import subprocess

import numpy as np
import pytest

from tests.factories import fake_completed_run as _fake_completed_run
from vmaf_app.core.models import FrameScores, VmafOptions
from vmaf_app.core.vmaf_runner import _build_libvmaf_opts, _parse_log


def test_neg_arrays_roundtrip_slice_and_iteration():
    frames = FrameScores(np.arange(2), np.arange(2), np.array([90, 91]),
                         vmaf_neg=np.array([80, 81]))
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


def test_ui_columns_select_and_display_independent_scores():
    from PySide6.QtWidgets import QApplication

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
