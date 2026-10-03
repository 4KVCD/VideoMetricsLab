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


@pytest.fixture(autouse=True)
def _fakes_are_not_cut_short(monkeypatch):
    """The fakes here produce a few frames for videos whose lengths promise
    more: frame_coverage would rightly call them cut short. Tested in
    test_frame_coverage.py."""
    monkeypatch.setattr(vr, "short_comparison", lambda *a, **k: None)


def _info(path: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_only_vmaf_and_neg_with_a_built_in_model_go_to_the_gpu():
    assert vmaf_cuda.gpu_models(True, True, "version=vmaf_v0.6.1") == {"vmaf": "vmaf_v0.6.1",
                                                                       "vmaf_neg": "vmaf_v0.6.1neg"}
    assert vmaf_cuda.gpu_models(True, False, "version=vmaf_4k_v0.6.1") == {"vmaf": "vmaf_4k_v0.6.1"}
    assert vmaf_cuda.gpu_models(False, True, "") == {"vmaf_neg": "vmaf_v0.6.1neg"}
    assert vmaf_cuda.gpu_models(True, True, "path=my_model.json") is None  # a custom model: the CPU
    assert vmaf_cuda.gpu_models(False, False, "version=vmaf_v0.6.1") is None


def test_the_videos_choice_and_the_probe_decide(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1") == {"vmaf": "vmaf_v0.6.1"}
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", enabled=False) is None  # set to CPU
    monkeypatch.setattr(vmaf_cuda, "_probed", (False, "no NVIDIA GPU"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1") is None


def test_the_graph_gives_the_gpu_libvmafs_frame_pairs_and_ffmpegs_filters_the_same_frames():
    """VMAF v1, PSNR, SSIM and XPSNR stay in FFmpeg and get the same frames
    through a split. The GPU's pairs come through overlay's frame sync with
    libvmaf's options -- by position, they were not always libvmaf's pairs."""
    rest = VmafOptions(compute_vmaf=False, compute_vmaf_neg=False, compute_xpsnr=True, extra_features=["name=psnr"])
    graph = vr._build_filtergraph(_info("d.mkv"), _info("s.mkv"), rest, None, None, HwAccelPlan(),
                                  Path("vmaf_log.json"), xpsnr_log_path=Path("xpsnr.txt"), gpu_vmaf=True)
    assert "[main]split=2[main_cpu][main_gpu];[ref]split=2[ref_cpu][ref_gpu]" in graph
    assert ("[main_gpu]pad=3840:1080[vmaf_canvas];[vmaf_canvas][ref_gpu]overlay=x=1920:y=0:eval=init:"
            "format=yuv420:shortest=1:repeatlast=0:ts_sync_mode=nearest,split=2[vmaf_left][vmaf_right];"
            "[vmaf_left]crop=1920:1080:0:0[vmaf_dist];[vmaf_right]crop=1920:1080:1920:0[vmaf_ref]") in graph
    assert "[ref_cpu]split=2[ref_xpsnr][ref_vmaf]" in graph and "[main_cpu][ref_xpsnr]xpsnr=" in graph
    assert graph.endswith("[cpu_out]") and "model=''" in graph  # no VMAF model left in FFmpeg
    alone = vr._build_filtergraph(_info("d.mkv"), _info("s.mkv"), VmafOptions(compute_vmaf=False), None, None,
                                  HwAccelPlan(), Path("vmaf_log.json"), gpu_vmaf=True)
    assert "[main]pad=3840:1080[vmaf_canvas];[vmaf_canvas][ref]overlay=" in alone
    assert alone.endswith("[vmaf_right]crop=1920:1080:1920:0[vmaf_ref]")


def test_an_odd_width_puts_the_source_on_a_chroma_sample():
    stage = vr._gpu_pairs_stage("yuv420p10le", 1365, 768, "main", "ref")
    assert "pad=2731:768" in stage and "overlay=x=1366:" in stage and "format=yuv420p10:" in stage
    assert stage.endswith("[vmaf_right]crop=1365:768:1366:0[vmaf_ref]")


def test_a_12_bit_comparison_is_scored_on_the_cpu(monkeypatch):
    """overlay, which gives the GPU libvmaf's frame pairs, holds 8 and 10 bits."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", bit_depth=10) == {"vmaf": "vmaf_v0.6.1"}
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", bit_depth=12) is None


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


def _halves(keys, options=None, cached=None):
    job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), options or VmafOptions(), label="d",
                                metric_keys=keys)
    if cached is not None:
        job.cached_result, job.cached_metrics = object(), cached
    run = worker_module._JobRun(worker_module.VmafWorker([job]), 0, job)
    return [(task.backend_id, task.metric_keys, run.pool_of(task)) for task in run.plan.tasks]


def test_vmaf_on_the_gpu_is_a_half_of_its_own_in_the_gpus_queue(monkeypatch):
    """In FFmpeg's run with VMAF v1, PSNR, SSIM and XPSNR it went at their
    pace, and SSIMULACRA2, Butteraugli and CVVDP waited for all of them.
    Now FFmpeg's half keeps the CPU's metrics, in the CPU's queue, and VMAF
    and NEG go ahead of Vship's metrics in the GPU's."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    gpu_vmaf = worker_module.GPU_VMAF
    assert _halves(("vmaf", "vmaf_neg", "psnr", "ssimulacra2")) == [
        ("ffmpeg", ("psnr",), "cpu"), (gpu_vmaf, ("vmaf", "vmaf_neg"), "gpu"), ("perceptual", ("ssimulacra2",), "gpu")]
    assert _halves(("vmaf",)) == [(gpu_vmaf, ("vmaf",), "gpu")]
    assert _halves(("psnr", "ssim")) == [("ffmpeg", ("psnr", "ssim"), "cpu")]
    # On the CPU: a resolution test, a video set to CPU.
    assert _halves(("vmaf",), VmafOptions(resample_test=ResampleTarget(width=1920, label="1080p"))) == [
        ("ffmpeg", ("vmaf",), "cpu")]
    assert _halves(("vmaf", "psnr"), VmafOptions(vmaf_on_gpu=False)) == [("ffmpeg", ("vmaf", "psnr"), "cpu")]


def test_a_saved_part_of_ffmpegs_metrics_is_not_calculated_again(monkeypatch):
    """The planner keeps FFmpeg's metrics together; split, the part whose
    metrics are all saved is left out."""
    from vmaf_app.core.metric_results import MetricResultSet

    class Saved(MetricResultSet):
        def __init__(self, *keys):
            super().__init__()
            self.keys = keys

        def has(self, key):
            return key in self.keys

        def __bool__(self):
            return True

    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert _halves(("vmaf", "psnr"), cached=Saved("psnr")) == [(worker_module.GPU_VMAF, ("vmaf",), "gpu")]
    assert _halves(("vmaf", "psnr"), cached=Saved("vmaf")) == [("ffmpeg", ("psnr",), "cpu")]


def test_vmaf_from_its_own_run_joins_ffmpegs_other_metrics(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    calls = []

    def run_vmaf(source, distorted, options, **_kwargs):
        calls.append((options.compute_vmaf, options.compute_vmaf_neg, options.metric_enabled("psnr")))
        frames = np.arange(4)
        values = {"vmaf": np.full(4, 93.0)} if options.compute_vmaf else {"psnr": np.full(4, 41.0)}
        return vr.ComparisonResult(
            source=source.path, distorted=distorted.path, frames=FrameScores(frames, frames / 24.0, **values),
            fps=24.0, model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
            source_info=source, distorted_info=distorted)

    monkeypatch.setattr(worker_module, "run_vmaf", run_vmaf)
    job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(extra_features=["name=psnr"]),
                                label="d", metric_keys=("vmaf", "psnr"))
    worker = worker_module.VmafWorker([job])
    finished = []
    worker.job_finished.connect(lambda _index, result: finished.append(result))
    worker.run()
    assert sorted(calls) == [(False, False, True), (True, False, False)]  # each on its own
    [result] = finished
    assert result.frames.values("vmaf").tolist() == [93.0] * 4 and result.frames.values("psnr").tolist() == [41.0] * 4


def test_the_result_keeps_vmafs_model_when_vmaf_had_a_run_of_its_own(monkeypatch):
    """FFmpeg's run without VMAF has model "": its result was the base, so
    the video's result -- shown, saved and reopened -- lost VMAF's model;
    and with a saved part of FFmpeg's metrics not calculated again, the
    saved run's models were lost the same way."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    model = "version=vmaf_4k_v0.6.1"

    def run_vmaf(source, distorted, options, **_kwargs):
        frames = np.arange(4)
        values = {"vmaf": np.full(4, 93.0)} if options.compute_vmaf else {"psnr": np.full(4, 41.0)}
        return vr.ComparisonResult(
            source=source.path, distorted=distorted.path, frames=FrameScores(frames, frames / 24.0, **values),
            fps=24.0, model=model if options.compute_vmaf else "", source_crop=None, distorted_crop=None,
            source_info=source, distorted_info=distorted)

    monkeypatch.setattr(worker_module, "run_vmaf", run_vmaf)

    def result_of(keys, saved=None):
        job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(extra_features=["name=psnr"]),
                                    label="d", metric_keys=keys)
        if saved is not None:
            job.cached_result, job.cached_metrics = saved, saved.metric_results
        worker = worker_module.VmafWorker([job])
        finished = []
        worker.job_finished.connect(lambda _index, result: finished.append(result))
        worker.run()
        [result] = finished
        return result

    both = result_of(("vmaf", "psnr"))
    assert both.model == model and both.has_metric("vmaf") and both.has_metric("psnr")
    # VMAF saved: FFmpeg's half calculates PSNR alone, and VMAF's model is the saved run's.
    saved = run_vmaf(_info("s.mp4"), _info("d.mp4"), VmafOptions())
    assert result_of(("vmaf", "psnr"), saved).model == model


def test_the_run_line_is_told_when_vmaf_is_on_the_gpu(monkeypatch):
    """VMAF and NEG while the GPU scores them; none once a failure hands
    them to FFmpeg's libvmaf. FFmpeg's half has none."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    job = worker_module.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d",
                                metric_keys=("vmaf", "vmaf_neg", "psnr"))
    run = worker_module._JobRun(worker_module.VmafWorker([job]), 0, job)

    def gpu_keys():
        with run.lock:
            return {task["backend"]: task["gpu_keys"] for task in run.task_snapshots()}

    gpu_vmaf = worker_module.GPU_VMAF
    assert gpu_keys() == {"ffmpeg": (), gpu_vmaf: ("vmaf", "vmaf_neg")}
    run.report_status(gpu_vmaf, "VMAF on the GPU failed (libvmaf crashed); calculating it on the CPU…")
    assert gpu_keys() == {"ffmpeg": (), gpu_vmaf: ()}


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


def test_a_video_set_to_cpu_has_its_vmaf_calculated_by_ffmpeg(monkeypatch):
    """Each video's own choice (Performance > VMAF v0.6.1 and NEG compute),
    taken with its options when the run starts."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    monkeypatch.setattr(vr, "_run_on_gpu", lambda *a, **k: pytest.fail("set to CPU, but scored on the GPU"))
    monkeypatch.setattr(vr, "_execute_run",
                        lambda *a, **k: FrameScores(np.array([0]), np.array([0.0]), vmaf=np.array([93.0])))
    result = vr.run_vmaf(_info("s.mkv"), _info("d.mkv"),
                         VmafOptions(crop_mode=CropMode.NONE, gpu_decode=False, vmaf_on_gpu=False))
    assert result.metric_results.get("vmaf").provenance.compute_backend == "cpu"
