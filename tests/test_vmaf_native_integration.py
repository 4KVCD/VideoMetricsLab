"""Opt-in actual GPU/decoder qualification; never writes user cache/settings.

VML_TEST_NATIVE_GPU=1 .venv/Scripts/python.exe -m pytest -n 0
    tests/test_vmaf_native_integration.py
"""
import os
import subprocess
import threading
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from vmaf_app.core import vmaf_native
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropMode, VideoInfo, VmafOptions

pytestmark = pytest.mark.skipif(os.environ.get("VML_TEST_NATIVE_GPU") != "1", reason="requires actual NVIDIA GPU")


@pytest.mark.parametrize("depth,test_fps,subsample,models", (
    (8, 10, 1, {"vmaf": "vmaf_v0.6.1"}),
    (10, 12, 1, {"vmaf": "vmaf_4k_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}),
    (10, 6, 2, {"vmaf_neg": "vmaf_v0.6.1neg"}),
))
@pytest.mark.parametrize("hardware", (False, True))
def test_native_matches_pipe_scores_frame_by_frame(tmp_path, monkeypatch, depth, test_fps, subsample, models, hardware):
    ffmpeg = str(ffmpeg_path())
    pix = "yuv420p" if depth == 8 else "yuv420p10le"
    source, test = tmp_path / "reference.mkv", tmp_path / "測試 video.mkv"
    for path, fps, blur in ((source, 10, False), (test, test_fps, True)):
        # Moving content, unequal rates, a real degradation and fractional
        # duration exercise matching, end-of-stream, motion and subsampling.
        codec = "libx264" if depth == 8 else "libx265"
        encoding = ["-c:v", codec, "-preset", "ultrafast"] if hardware else ["-c:v", "ffv1"]
        command = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=192x128:rate={fps}",
                   "-t", "1.8", "-vf", f"{'boxblur=1,' if blur else ''}format={pix}", *encoding, str(path)]
        subprocess.run(command, check=True, capture_output=True)
    ref_info = VideoInfo(source, 192, 128, 10, 1.8, 18, "ffv1", pix_fmt=pix)
    test_info = VideoInfo(test, 192, 128, test_fps, 1.8, round(1.8 * test_fps), "ffv1", pix_fmt=pix)
    options = VmafOptions(crop_mode=CropMode.NONE, duration_limit=1.05, n_subsample=subsample,
                          compute_vmaf="vmaf" in models, compute_vmaf_neg="vmaf_neg" in models)
    plan = vr._GpuPlan(models, 192, 128, depth, replace(options, compute_vmaf=False, compute_vmaf_neg=False))
    hw = HwAccelPlan(source="cuda", distorted="cuda") if hardware else HwAccelPlan()
    args = (plan, ref_info, test_info, options, None, None, "version=vmaf_v0.6.1", hw, 18)
    original = vmaf_native.eligible
    assert original(ref_info, test_info, None, None, 192, 128, depth, False, HwAccelPlan())
    # Native failures must fail THIS test, not pass via the legacy fallback.
    with monkeypatch.context() as native_patch:
        native_patch.setattr(vr, "_execute_run", lambda *a, **k: pytest.fail("Native helper failed"))
        actual = vr._score_on_gpu(*args)
    monkeypatch.setattr(vmaf_native, "eligible", lambda *a: False)
    expected = vr._score_on_gpu(*args)
    np.testing.assert_array_equal(actual.frame, expected.frame)
    for key in models:
        np.testing.assert_allclose(actual.values(key), expected.values(key), atol=0.00005, rtol=0)


def test_unicode_runtime_and_output_folder(tmp_path, monkeypatch):
    from vmaf_app.core import vmaf_cuda

    folder = tmp_path / "測試 runtime"
    folder.mkdir()
    # NTFS hardlinks avoid copying 155 MB and remain independently unlinkable.
    for name in vmaf_native._FILES:
        (folder / name).hardlink_to(vmaf_native.RUNTIME / name)
    monkeypatch.setattr(vmaf_native, "RUNTIME", folder)
    monkeypatch.setattr(vmaf_native, "EXECUTABLE", folder / "vmaf_native.exe")
    monkeypatch.setattr(vmaf_native, "KERNEL", folder / "vmaf_prepare.ptx")
    real_temp = vr.tempfile.TemporaryDirectory
    monkeypatch.setattr(vr.tempfile, "TemporaryDirectory", partial(real_temp, dir=folder))
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "GPU enabled for integration test"))
    video = folder / "測試 input.mkv"
    subprocess.run([str(ffmpeg_path()), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=192x128:rate=10", "-t", "0.5", "-c:v", "ffv1", str(video)],
                   check=True, capture_output=True)
    info = VideoInfo(video, 192, 128, 10, 0.5, 5, "ffv1", pix_fmt="yuv420p")
    # Exercise the task's command/cwd/parse contract, not just executable
    # startup. No user settings/cache writes are involved.
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 192, 128, 8, VmafOptions(compute_vmaf=False))
    args = (plan, info, info, VmafOptions(crop_mode=CropMode.NONE), None, None, "version=vmaf_v0.6.1", HwAccelPlan(), 5)
    with monkeypatch.context() as guarded:
        guarded.setattr(vr, "_execute_run", lambda *a, **k: pytest.fail("Native fallback hides Unicode failure"))
        scores = vr._score_on_gpu(*args)
    assert len(scores) == 5
    monkeypatch.setattr(vmaf_native, "eligible", lambda *a: False)
    expected = vr._score_on_gpu(*args)
    np.testing.assert_array_equal(scores.vmaf, expected.vmaf)
    # The production process boundary also returns genuine GPU scores.
    isolated = vr._run_on_gpu(*args, on_progress=None, on_status=None, cancel_event=None, process_handle=None)
    assert isolated is not None
    np.testing.assert_array_equal(scores.vmaf, isolated.vmaf)


def test_cancel_reaps_native_process_and_removes_temporary_run(tmp_path, monkeypatch):
    from vmaf_app.core.process_control import ProcessHandle

    video = tmp_path / "cancel.mkv"
    subprocess.run([str(ffmpeg_path()), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=192x128:rate=100", "-t", "5", "-c:v", "ffv1", str(video)],
                   check=True, capture_output=True)
    info = VideoInfo(video, 192, 128, 100, 5, 500, "ffv1", pix_fmt="yuv420p")
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 192, 128, 8, VmafOptions(compute_vmaf=False))
    cancel = threading.Event()
    handle = ProcessHandle()
    attached = []
    attach = handle.attach

    def record(pid):
        attached.append(pid)
        attach(pid)

    monkeypatch.setattr(handle, "attach", record)
    real_temp = vr.tempfile.TemporaryDirectory
    monkeypatch.setattr(vr.tempfile, "TemporaryDirectory", partial(real_temp, dir=tmp_path))
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: pytest.fail("Cancellation must not retry"))
    with pytest.raises(vr.Cancelled):
        vr._score_on_gpu(plan, info, info, VmafOptions(crop_mode=CropMode.NONE), None, None,
                         "version=vmaf_v0.6.1", HwAccelPlan(), 500,
                         on_progress=lambda *a: cancel.set(), cancel_event=cancel, process_handle=handle)
    import psutil
    assert len(attached) == 1
    assert not psutil.pid_exists(attached[0])
    assert not handle._pids
    assert not list(tmp_path.glob("vmaf_native_run_*"))
