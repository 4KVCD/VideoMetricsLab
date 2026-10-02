"""VMAF on the GPU (vmaf_cuda), as far as it can be tested without a GPU --
GitHub's runner has none. The scores themselves were compared on an RTX
5090 (see vmaf_cuda's docstring)."""
import faulthandler
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmaf_app.core import vmaf_cuda
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropMode, FrameScores, ResampleTarget, VideoInfo, VmafOptions
from vmaf_app.ui import worker as worker_module


def _info(path: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_only_vmaf_and_neg_with_a_built_in_model_go_to_the_gpu():
    assert vmaf_cuda.gpu_models(True, True, "version=vmaf_v0.6.1") == {"vmaf": "vmaf_v0.6.1",
                                                                       "vmaf_neg": "vmaf_v0.6.1neg"}
    assert vmaf_cuda.gpu_models(True, False, "version=vmaf_4k_v0.6.1") == {"vmaf": "vmaf_4k_v0.6.1"}
    assert vmaf_cuda.gpu_models(False, True, "") == {"vmaf_neg": "vmaf_v0.6.1neg"}
    assert vmaf_cuda.gpu_models(True, True, "path=my_model.json") is None  # a custom model: the CPU
    assert vmaf_cuda.gpu_models(False, False, "version=vmaf_v0.6.1") is None


def test_the_setting_and_the_probe_decide(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1") == {"vmaf": "vmaf_v0.6.1"}
    monkeypatch.setattr(vmaf_cuda, "_enabled", False)
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1") is None
    monkeypatch.setattr(vmaf_cuda, "_enabled", True)
    monkeypatch.setattr(vmaf_cuda, "_probed", (False, "no NVIDIA GPU"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1") is None


def test_the_graph_gives_the_compared_frames_to_the_gpu_and_to_ffmpegs_filters():
    """VMAF v1, PSNR, SSIM and XPSNR stay in FFmpeg and get the same frames
    through a split; with nothing left for FFmpeg, the frames go straight out."""
    rest = VmafOptions(compute_vmaf=False, compute_vmaf_neg=False, compute_xpsnr=True, extra_features=["name=psnr"])
    graph = vr._build_filtergraph(_info("d.mkv"), _info("s.mkv"), rest, None, None, HwAccelPlan(),
                                  Path("vmaf_log.json"), xpsnr_log_path=Path("xpsnr.txt"), gpu_vmaf=True)
    assert "[main]split=2[main_cpu][vmaf_dist];[ref]split=2[ref_cpu][vmaf_ref]" in graph
    assert "[ref_cpu]split=2[ref_xpsnr][ref_vmaf]" in graph and "[main_cpu][ref_xpsnr]xpsnr=" in graph
    assert graph.endswith("[cpu_out]") and "model=''" in graph  # no VMAF model left in FFmpeg
    alone = vr._build_filtergraph(_info("d.mkv"), _info("s.mkv"), VmafOptions(compute_vmaf=False), None, None,
                                  HwAccelPlan(), Path("vmaf_log.json"), gpu_vmaf=True)
    assert alone.endswith("[main]null[vmaf_dist];[ref]null[vmaf_ref]")


def test_each_output_is_mapped_and_the_raw_ones_pass_every_frame_through():
    raw = ["-map", "[vmaf_dist]", "D", "-map", "[vmaf_ref]", "R"]
    args = vr._build_ffmpeg_output_args("G", 30.0, raw, cpu_output=True)
    assert args == ["-lavfi", "G", "-progress", "pipe:1", "-nostats",
                    "-map", "[cpu_out]", "-t", "30.000", "-f", "null", "-", *raw]
    assert vr._build_ffmpeg_output_args("G", 30.0, raw, cpu_output=False) == [
        "-lavfi", "G", "-progress", "pipe:1", "-nostats", *raw]
    attempt = object.__new__(vmaf_cuda.GpuAttempt)
    attempt.distorted, attempt.reference = SimpleNamespace(path="D"), SimpleNamespace(path="R")
    assert attempt.output_args(30.5) == [
        "-map", "[vmaf_dist]", "-fps_mode", "passthrough", "-t", "30.500", "-f", "rawvideo", "D",
        "-map", "[vmaf_ref]", "-fps_mode", "passthrough", "-t", "30.500", "-f", "rawvideo", "R"]


def test_the_gpus_scores_join_ffmpegs_and_must_cover_the_same_frames():
    """A frame one side lacked was kept as NaN: a VMAF with holes in it. The
    GPU result is refused instead, and the run is made again on the CPU."""
    ffmpeg = FrameScores(np.array([0, 1, 2]), np.array([0.0, 0.1, 0.2]), psnr=np.array([40, 41, 42], dtype=np.float32))
    gpu = (np.array([0, 1, 2], dtype=np.int32), {"vmaf": np.array([90.0, 91.0, 92.0])})
    joined = vr._with_gpu_scores(ffmpeg, gpu, 10.0)
    assert joined.values("vmaf").tolist() == [90.0, 91.0, 92.0]
    assert joined.values("psnr").tolist() == [40.0, 41.0, 42.0]
    alone = vr._with_gpu_scores(None, (np.array([0, 2], dtype=np.int32), {"vmaf_neg": np.array([80.0, 81.0])}), 10.0)
    assert alone.frame.tolist() == [0, 2] and alone.time.tolist() == [0.0, 0.2]
    assert alone.values("vmaf_neg").tolist() == [80.0, 81.0]
    with pytest.raises(vmaf_cuda.VmafGpuError, match="not the same"):
        vr._with_gpu_scores(ffmpeg, (np.array([0, 1], dtype=np.int32), {"vmaf": np.array([90.0, 91.0])}), 10.0)
    with pytest.raises(vmaf_cuda.VmafGpuError, match="no frames"):
        vr._with_gpu_scores(ffmpeg, (np.array([], dtype=np.int32), {"vmaf": np.array([])}), 10.0)


def test_a_duration_limit_gives_the_raw_outputs_one_frame_more(monkeypatch):
    """FFmpeg's libvmaf filter scores the first frame at or past the limit
    before the null output stops there: 721 frames for 30 s at 23.976 fps.
    The raw outputs stopped one frame earlier, so GPU and CPU runs of one
    video scored different frames."""
    seen = {}

    class Attempt:
        def __init__(self, *args):
            seen["args"] = args

        def output_args(self, limit):
            seen["limit"] = limit
            return ["RAW"]

        def finish(self, succeeded):
            return np.array([0, 1], dtype=np.int32), {"vmaf": np.array([90.0, 91.0])}

    monkeypatch.setattr(vmaf_cuda, "GpuAttempt", Attempt)
    monkeypatch.setattr(vr, "_run_ffmpeg", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "", ""))
    commands = []
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 64, 48, 8, VmafOptions(compute_vmaf=False))
    frames = vr._execute_run(
        lambda *args: commands.append(args) or ["ffmpeg"], options=VmafOptions(duration_limit=30.0), fps=24.0,
        total_frames=10, hwaccel=HwAccelPlan(), tmp_prefix="vmaf_test_", on_progress=None, on_status=None,
        cancel_event=None, process_handle=None, gpu=plan)
    assert seen["limit"] == pytest.approx(30 + 1 / 24)
    assert seen["args"] == (64, 48, 8, {"vmaf": "vmaf_v0.6.1"}, 1)
    assert commands[0][-1] == ["RAW"]
    assert frames.values("vmaf").tolist() == [90.0, 91.0]


def _crashing_gpu_run(*_args, **_kwargs):
    faulthandler._sigsegv()  # an access violation, as in libvmaf or the NVIDIA driver


def test_a_crash_in_libvmaf_ends_its_own_process_and_the_cpu_calculates_vmaf(monkeypatch, caplog):
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    monkeypatch.setattr(vr, "_score_on_gpu", _crashing_gpu_run)
    monkeypatch.setattr(vr, "_execute_run",
                        lambda *a, **k: FrameScores(np.array([0]), np.array([0.0]), vmaf=np.array([93.0])))
    statuses = []
    with caplog.at_level(logging.ERROR):
        result = vr.run_vmaf(_info("s.mkv"), _info("d.mkv"), VmafOptions(crop_mode=CropMode.NONE, gpu_decode=False),
                             on_status=statuses.append)
    assert result.frames.values("vmaf").tolist() == [93.0]
    assert result.metric_results.get("vmaf").provenance.compute_backend == "cpu"
    assert any("VMAF on the GPU failed" in status and "libvmaf crashed" in status for status in statuses)
    assert "libvmaf crashed" in caplog.text


def test_ffmpegs_half_queues_for_the_gpu_when_it_scores_vmaf_there(monkeypatch):
    """One GPU pass at a time: VMAF on the GPU never runs beside another
    video's GPU metrics. Only for the half's own metrics: a saved VMAF is
    not scored again, so the half's PSNR stays on the CPU queue."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))

    def pool(keys, options=None):
        job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), options or VmafOptions(), label="d",
                                    metric_keys=keys)
        run = worker_module._JobRun(worker_module.VmafWorker([job]), 0, job)
        [task] = [task for task in run.plan.tasks if task.backend_id == "ffmpeg"]
        return run.pool_of(task)

    assert pool(("vmaf", "psnr")) == "gpu"
    assert pool(("psnr", "ssim")) == "cpu"
    assert pool(("vmaf",), VmafOptions(resample_test=ResampleTarget(width=1920, label="1080p"))) == "cpu"
    monkeypatch.setattr(vmaf_cuda, "_enabled", False)
    assert pool(("vmaf", "psnr")) == "cpu"


def test_a_pipe_ffmpeg_never_opened_does_not_hold_the_run():
    """An output that gets no frame is never opened: its reader waited for
    FFmpeg forever, and the run with it."""
    reader = vmaf_cuda._PipeReader("test", 4)
    reader.start()
    reader.release_if_unconnected()  # FFmpeg has ended without opening it
    reader.join(5)
    assert not reader.is_alive() and reader.frames.get(timeout=1) is None


def test_frames_left_in_a_pipe_after_ffmpeg_ended_are_still_read():
    reader = vmaf_cuda._PipeReader("test", 4)
    reader.start()
    with open(reader.path, "wb") as ffmpeg:
        ffmpeg.write(b"abcdefgh")  # two frames, and FFmpeg is gone before they are read
    reader.release_if_unconnected()
    reader.join(5)
    frames = []
    while (frame := reader.frames.get(timeout=1)) is not None:
        frames.append(bytes(frame))
    assert frames == [b"abcd", b"efgh"]


def test_the_setting_applies_from_the_next_run(monkeypatch):
    """Its tooltip says so; read afresh for each video, switching it changed
    the run in progress."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d", metric_keys=("vmaf",))
    worker = worker_module.VmafWorker([job])  # the run starts with it on
    monkeypatch.setattr(vmaf_cuda, "_enabled", False)  # and it is switched off during the run
    run = worker_module._JobRun(worker, 0, job)
    [task] = [task for task in run.plan.tasks if task.backend_id == "ffmpeg"]
    assert run.pool_of(task) == "gpu"
    assert worker_module.VmafWorker([job]).gpu_vmaf is False  # the next run
