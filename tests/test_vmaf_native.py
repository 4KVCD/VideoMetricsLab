"""Native helper qualification and runner lifecycle, without a GPU."""
import json
import subprocess
from dataclasses import replace

import pytest

from vmaf_app.core import vmaf_native as native
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropBox, CropMode, VideoInfo, VmafOptions


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native, "RUNTIME", tmp_path)
    for name in native._FILES:
        (tmp_path / name).touch()
    ref = tmp_path / "reference.mkv"
    test = tmp_path / "test.mkv"
    ref.touch(); test.touch()
    return (VideoInfo(ref, 3840, 2160, 24, 10, 240, "hevc", pix_fmt="yuv420p10le"),
            VideoInfo(test, 3840, 1608, 24, 10, 240, "hevc", pix_fmt="yuv420p10le"),
            CropBox(3840, 1608, 0, 276), None)


def allowed(pair, **kwargs):
    options = dict(width=3840, height=1608, depth=10, cpu_output=False,
                   hw=HwAccelPlan(source="cuda", distorted="cuda"))
    options.update(kwargs)
    return native.eligible(*pair, **options)


def test_native_crop_only_gpu_and_software_decode_qualify(pair):
    assert allowed(pair)
    assert allowed(pair, hw=HwAccelPlan(source="cuda"))  # VVC test stays on CPU.
    assert allowed(pair, hw=HwAccelPlan())  # GPU scoring without GPU decode.


@pytest.mark.parametrize("change", ("resize", "depth", "format", "odd", "outside", "missing", "runtime", "cpu", "qsv"))
def test_unqualified_comparisons_keep_existing_path(pair, change):
    ref, test, rc, tc = pair
    if change == "resize": test = replace(test, width=1920, height=804)
    elif change == "depth": test = replace(test, pix_fmt="yuv420p")
    elif change == "format": test = replace(test, pix_fmt="yuv444p10le")
    elif change == "odd": rc = CropBox(3840, 1608, 0, 275)
    elif change == "outside": rc = CropBox(3840, 1608, 0, 556)
    elif change == "missing": test.path.unlink()
    elif change == "runtime": (native.RUNTIME / "swscale-10.dll").unlink()
    assert not allowed((ref, test, rc, tc), cpu_output=change == "cpu",
                       hw=HwAccelPlan(source="qsv") if change == "qsv" else HwAccelPlan(source="cuda"))


def test_command_preserves_models_crop_subsample_and_unicode(pair, tmp_path):
    ref, test, rc, tc = pair
    test = replace(test, path=tmp_path / "測試 clip.mkv")
    cmd = native.command(ref, test, rc, tc, 10, {"vmaf": "vmaf_4k_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"},
                         3, 2.25, HwAccelPlan(source="cuda"), tmp_path / "scores.json")
    def value(name): return cmd[cmd.index(name) + 1]
    assert value("--test") == str(test.path.resolve())
    assert value("--reference-crop") == "3840:1608:0:276"
    assert value("--test-crop") == "3840:1608:0:0"
    assert value("--reference-decode") == "cuda" and value("--test-decode") == "cpu"
    assert value("--duration") == "2.250" and value("--subsample") == "3"
    assert value("--output") == "scores.json"
    assert value("--wait") == "blocking"
    assert [cmd[i+1] for i, token in enumerate(cmd) if token == "--model"] == [
        "vmaf=vmaf_4k_v0.6.1", "vmaf_neg=vmaf_v0.6.1neg"]


def test_benchmark_can_override_blocking_default(pair, tmp_path):
    ref, test, rc, tc = pair
    cmd = native.command(ref, test, rc, tc, 10, {"vmaf": "vmaf_v0.6.1"}, 1, 0,
                         HwAccelPlan(), tmp_path / "scores.json", wait="auto")
    assert cmd[cmd.index("--wait") + 1] == "auto"


def test_native_runner_discards_failed_log_keeps_decode_ladder_and_status(pair, tmp_path, monkeypatch):
    source, test, rc, tc = pair
    opts = VmafOptions(crop_mode=CropMode.NONE, duration_limit=2.0)
    plan = vr._GpuPlan({"vmaf": "vmaf_4k_v0.6.1"}, 3840, 1608, 10, replace(opts, compute_vmaf=False))
    calls, messages = [], []
    handle = object()

    def run(cmd, total, progress, cancel, cwd, process_handle):
        assert process_handle is handle
        log = cwd / cmd[cmd.index("--output")+1]
        assert not log.exists()  # No partial log may survive a retry.
        calls.append(cmd)
        log.write_text(json.dumps({"frames": [{"frameNum": 0, "metrics": {"vmaf": 92}}]}))
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 1 else 0, "", "decoder failed")

    monkeypatch.setattr(vr, "_run_ffmpeg", run)
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: pytest.fail("Native success must not decode again"))
    result = vr._score_on_gpu(plan, source, test, opts, rc, tc, "version=vmaf_4k_v0.6.1",
                              HwAccelPlan(source="cuda", distorted="cuda"), 48,
                              on_status=messages.append, process_handle=handle)
    assert result.vmaf.tolist() == [92]
    assert len(calls) == 2
    assert calls[0][calls[0].index("--duration")+1] == "2.042"
    assert calls[1][calls[1].index("--test-decode")+1] == "cpu"
    assert "(GPU decode: source cuda, distorted cpu)" in messages[-1]


def test_native_failure_retains_pipe_path_and_cancel_never_retries(pair, monkeypatch):
    source, test, rc, tc = pair
    opts = VmafOptions(crop_mode=CropMode.NONE)
    plan = vr._GpuPlan({"vmaf": "vmaf_4k_v0.6.1"}, 3840, 1608, 10, replace(opts, compute_vmaf=False))
    sentinel = object()
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: sentinel)
    monkeypatch.setattr(vr, "_run_ffmpeg", lambda cmd, *a: subprocess.CompletedProcess(cmd, 1, "", "no decoder"))
    args = (plan, source, test, opts, rc, tc, "version=vmaf_4k_v0.6.1", HwAccelPlan(), 240)
    assert vr._score_on_gpu(*args) is sentinel

    def cancelled(*a): raise vr.Cancelled("Cancelled")
    monkeypatch.setattr(vr, "_run_ffmpeg", cancelled)
    with pytest.raises(vr.Cancelled): vr._score_on_gpu(*args)
