import shutil
import subprocess

import numpy as np
import pytest

from vmaf_app.core.models import VmafOptions
from vmaf_app.core.vmaf_runner import _build_libvmaf_opts, _parse_log


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
