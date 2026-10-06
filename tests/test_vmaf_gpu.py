"""VMAF on the GPU (vmaf_cuda), as far as it can be tested without a GPU --
GitHub's runner has none. The scores themselves were compared on an RTX
5090 (see vmaf_cuda's docstring)."""
import ctypes
import faulthandler
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from tests.factories import status
from vmaf_app.core import gpu_frames, job_runner, vmaf_cuda
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


def test_the_graph_gives_the_gpu_libvmafs_frame_pairs_and_nothing_else():
    """The GPU's pairs come through overlay's frame sync with libvmaf's
    options -- by position, they were not always libvmaf's pairs. FFmpeg's
    own filters score nothing in a GPU run: VMAF and NEG are all it has."""
    graph = vr._build_filtergraph(_info("d.mkv"), _info("s.mkv"), VmafOptions(compute_vmaf_neg=True), None, None,
                                  HwAccelPlan(), Path("vmaf_log.json"), gpu_vmaf=True)
    assert ("[main]pad=3840:1080[vmaf_canvas];[vmaf_canvas][ref]overlay=x=1920:y=0:eval=init:"
            "format=yuv420:shortest=1:repeatlast=0:ts_sync_mode=nearest,split=2[vmaf_left][vmaf_right];"
            "[vmaf_left]crop=1920:1080:0:0[vmaf_dist];[vmaf_right]crop=1920:1080:1920:0[vmaf_ref]") in graph
    assert graph.endswith("[vmaf_right]crop=1920:1080:1920:0[vmaf_ref]")
    assert "libvmaf" not in graph and "xpsnr" not in graph


def test_a_comparison_at_an_odd_size_is_scored_on_the_cpu(monkeypatch):
    """The GPU's pairs cross FFmpeg's overlay on one 4:2:0 canvas, which pad
    cannot make odd-sized: the attempt failed in FFmpeg ("VMAF on the GPU
    failed") and VMAF was calculated again on the CPU, every time."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    model = "version=vmaf_v0.6.1"
    assert vmaf_cuda.scores_on_gpu(True, False, model, size=(1366, 768)) == {"vmaf": "vmaf_v0.6.1"}
    assert vmaf_cuda.scores_on_gpu(True, False, model, size=None) == {"vmaf": "vmaf_v0.6.1"}
    for size in ((1365, 768), (1366, 767), (853, 479)):
        assert vmaf_cuda.scores_on_gpu(True, False, model, size=size) is None
    with pytest.raises(ValueError, match="even size"):
        vr._gpu_pairs_stage("yuv420p10le", 1365, 768, "main", "ref")
    stage = vr._gpu_pairs_stage("yuv420p10le", 1366, 768, "main", "ref")
    assert "pad=2732:768" in stage and "overlay=x=1366:" in stage and "format=yuv420p10:" in stage


def test_an_odd_sized_video_with_black_bars_off_has_its_vmaf_in_ffmpegs_half(monkeypatch):
    """Known before the run only with black bars off; cut, the pictures are
    even-sized. No GPU attempt is made, and the run line shows VMAF as a
    CPU metric from the start."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))

    def halves(width, height, crop_mode):
        job = job_runner.VmafJob(_info("s.mp4", width, height), _info("d.mp4", width, height),
                                 VmafOptions(crop_mode=crop_mode), label="d", metric_keys=("vmaf", "psnr"))
        run = job_runner.JobRun(job_runner.JobScheduler([job]), 0, job)
        return [(task.backend_id, task.metric_keys) for task in run.plan.tasks]

    assert halves(853, 479, CropMode.NONE) == [("ffmpeg", ("vmaf", "psnr"))]
    assert halves(854, 480, CropMode.NONE) == [("ffmpeg", ("psnr",)), (job_runner.GPU_VMAF, ("vmaf",))]
    assert halves(853, 479, CropMode.AUTO) == [("ffmpeg", ("psnr",)), (job_runner.GPU_VMAF, ("vmaf",))]


def test_a_12_bit_comparison_is_scored_on_the_cpu(monkeypatch):
    """overlay, which gives the GPU libvmaf's frame pairs, holds 8 and 10 bits."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", bit_depth=10) == {"vmaf": "vmaf_v0.6.1"}
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", bit_depth=12) is None


def test_each_output_is_mapped_and_the_raw_ones_pass_every_frame_through():
    raw = ["-map", "[vmaf_dist]", "D", "-map", "[vmaf_ref]", "R"]
    assert vr._build_ffmpeg_output_args("G", 30.0, raw) == [
        "-lavfi", "G", "-progress", "pipe:1", "-nostats", *raw]
    assert vr._build_ffmpeg_output_args("G", 30.0) == [
        "-lavfi", "G", "-progress", "pipe:1", "-nostats", "-t", "30.000", "-f", "null", "-"]
    attempt = object.__new__(vmaf_cuda.GpuAttempt)
    attempt.paired = True  # by FFmpeg; test_vmaf_gpu_streams.py for the pairing in the app
    attempt.distorted, attempt.reference = SimpleNamespace(path="D"), SimpleNamespace(path="R")
    assert attempt.output_args(30.5) == [
        "-map", "[vmaf_dist]", "-fps_mode", "passthrough", "-t", "30.500", "-f", "rawvideo", "D",
        "-map", "[vmaf_ref]", "-fps_mode", "passthrough", "-t", "30.500", "-f", "rawvideo", "R"]


def test_the_gpus_scores_are_the_runs_frame_scores():
    scores = vr._gpu_frame_scores((np.array([0, 2], dtype=np.int32), {"vmaf_neg": np.array([80.0, 81.0])}), 10.0)
    assert scores.frame.tolist() == [0, 2] and scores.time.tolist() == [0.0, 0.2]
    assert scores.values("vmaf_neg").tolist() == [80.0, 81.0]
    with pytest.raises(vmaf_cuda.VmafGpuError, match="no frames"):
        vr._gpu_frame_scores((np.array([], dtype=np.int32), {"vmaf": np.array([])}), 10.0)


def test_vmaf_with_other_ffmpeg_metrics_in_one_call_is_calculated_by_ffmpeg(monkeypatch):
    """The window gives VMAF and NEG a run of their own on the GPU. Asked
    for beside PSNR in one call, VMAF stays in FFmpeg with it: the run that
    fed both to the GPU and FFmpeg's filters at once is gone."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    monkeypatch.setattr(vr, "_run_on_gpu", lambda *a, **k: pytest.fail("VMAF beside PSNR went to the GPU"))
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: FrameScores(
        np.array([0]), np.array([0.0]), vmaf=np.array([93.0]), psnr=np.array([40.0])))
    result = vr.run_vmaf(_info("s.mkv"), _info("d.mkv"),
                         VmafOptions(crop_mode=CropMode.NONE, gpu_decode=False, extra_features=["name=psnr"]))
    assert result.metric_results.get("vmaf").provenance.compute_backend == "cpu"


def test_a_duration_limit_gives_the_raw_outputs_one_frame_more(monkeypatch):
    """FFmpeg's libvmaf filter scores the first frame at or past the limit
    before the null output stops there: 721 frames for 30 s at 23.976 fps.
    The raw outputs stopped one frame earlier, so GPU and CPU runs of one
    video scored different frames."""
    seen = {}

    class Attempt:
        def __init__(self, *args, **_kwargs):
            seen["args"] = args

        def output_args(self, limit):
            seen["limit"] = limit
            return ["RAW"]

        def finish(self, succeeded):
            return np.array([0, 1], dtype=np.int32), {"vmaf": np.array([90.0, 91.0])}

    monkeypatch.setattr(vmaf_cuda, "GpuAttempt", Attempt)
    monkeypatch.setattr(vr, "_run_ffmpeg", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "", ""))
    commands = []
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 64, 48, 8)
    frames = vr._execute_run(
        lambda *args: commands.append(args) or ["ffmpeg"], options=VmafOptions(duration_limit=30.0), fps=24.0,
        total_frames=10, hwaccel=HwAccelPlan(), tmp_prefix="vmaf_test_", on_progress=None, on_status=None,
        cancel_event=None, process_handle=None, gpu=plan)
    assert seen["limit"] == pytest.approx(30 + 1 / 24)
    assert seen["args"] == (64, 48, 8, {"vmaf": "vmaf_v0.6.1"}, 1, "cuda", None)
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
    job = job_runner.VmafJob(_info("s.mp4"), _info("d.mp4"), options or VmafOptions(), label="d",
                                metric_keys=keys)
    if cached is not None:
        job.cached_result, job.cached_metrics = object(), cached
    run = job_runner.JobRun(job_runner.JobScheduler([job]), 0, job)
    return [(task.backend_id, task.metric_keys, run.pool_of(task)) for task in run.plan.tasks]


def test_vmaf_on_the_gpu_is_a_half_of_its_own_in_the_gpus_queue(monkeypatch):
    """In FFmpeg's run with VMAF v1, PSNR, SSIM and XPSNR it went at their
    pace, and SSIMULACRA2, Butteraugli and CVVDP waited for all of them.
    Now FFmpeg's half keeps the CPU's metrics, in the CPU's queue, and VMAF
    and NEG go ahead of Vship's metrics in the GPU's."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    gpu_vmaf = job_runner.GPU_VMAF
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
    assert _halves(("vmaf", "psnr"), cached=Saved("psnr")) == [(job_runner.GPU_VMAF, ("vmaf",), "gpu")]
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

    monkeypatch.setattr(job_runner, "run_vmaf", run_vmaf)
    job = job_runner.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(extra_features=["name=psnr"]),
                                label="d", metric_keys=("vmaf", "psnr"))
    worker = worker_module.VmafWorker([job])
    finished = []
    # Direct: the job finishes on whichever lane ends last, a CPU lane's
    # thread included, where a queued call would wait for an event loop.
    worker.job_finished.connect(lambda _index, result: finished.append(result), Qt.ConnectionType.DirectConnection)
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

    monkeypatch.setattr(job_runner, "run_vmaf", run_vmaf)

    def result_of(keys, saved=None):
        job = job_runner.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(extra_features=["name=psnr"]),
                                    label="d", metric_keys=keys)
        if saved is not None:
            job.cached_result, job.cached_metrics = saved, saved.metric_results
        worker = worker_module.VmafWorker([job])
        finished = []
        worker.job_finished.connect(lambda _index, result: finished.append(result),
                                    Qt.ConnectionType.DirectConnection)
        worker.run()
        [result] = finished
        return result

    both = result_of(("vmaf", "psnr"))
    assert both.model == model and both.has_metric("vmaf") and both.has_metric("psnr")
    # VMAF saved: FFmpeg's half calculates PSNR alone, and VMAF's model is the saved run's.
    saved = run_vmaf(_info("s.mp4"), _info("d.mp4"), VmafOptions())
    assert result_of(("vmaf", "psnr"), saved).model == model


def test_the_run_line_is_told_when_vmaf_is_on_the_gpu(monkeypatch):
    """VMAF and NEG in the GPU's queue, on the GPU; once a failure hands them
    to FFmpeg's libvmaf, the CPU's (cpu_keys), with its figures from 0."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    job = job_runner.VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d",
                                metric_keys=("vmaf", "vmaf_neg", "psnr"))
    run = job_runner.JobRun(job_runner.JobScheduler([job]), 0, job)

    def halves():
        with run.lock:
            return {task["backend"]: (task["lane"], task["cpu_keys"], task["current"])
                    for task in run.task_snapshots()}

    gpu_vmaf = job_runner.GPU_VMAF
    run.report_progress(gpu_vmaf, 40, 100, 50.0)
    assert halves() == {"ffmpeg": ("cpu", (), 0), gpu_vmaf: ("gpu", (), 40)}
    run.report_status(gpu_vmaf, status("VMAF on the GPU failed (libvmaf crashed); calculating it on the CPU…"))
    assert halves() == {"ffmpeg": ("cpu", (), 0), gpu_vmaf: ("gpu", ("vmaf", "vmaf_neg"), 0)}


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


def test_frames_written_before_the_reader_connected_are_still_read():
    """FFmpeg can open the pipe, write and close it before the reader's
    thread connects -- a short run on a busy machine. Windows then answers
    ConnectNamedPipe with ERROR_NO_DATA, which was taken for an error: the
    frames, still in the pipe, were lost."""
    reader = vmaf_cuda._PipeReader("test", 4)
    with open(reader.path, "wb") as ffmpeg:
        ffmpeg.write(b"abcdefgh")
    reader.start()
    reader.join(5)
    frames = []
    while (frame := reader.frames.get(timeout=1)) is not None:
        frames.append(bytes(frame))
    assert reader.error is None
    assert frames == [b"abcd", b"efgh"]


def test_the_pipe_reader_keeps_its_names_off_the_threads():
    """It kept its pipe as _handle, the name Thread keeps its own handle
    under since Python 3.13: start() then failed with "'handle' must be a
    _ThreadHandle", on 3.13 only -- the pipe reader never started there."""
    import threading

    reader = vmaf_cuda._PipeReader("names", 4)
    try:
        own = set(vars(reader)) - set(vars(threading.Thread()))
        assert "_pipe" in own
        assert not own & {"_handle", "_os_thread_handle", "_started", "_target", "_tstate_lock"}
    finally:
        reader.stop()
        vmaf_cuda._winapi.CloseHandle(reader._pipe)


def test_a_video_set_to_cpu_has_its_vmaf_calculated_by_ffmpeg(monkeypatch):
    """Each video's own choice (Performance > VMAF compute),
    taken with its options when the run starts."""
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    monkeypatch.setattr(vr, "_run_on_gpu", lambda *a, **k: pytest.fail("set to CPU, but scored on the GPU"))
    monkeypatch.setattr(vr, "_execute_run",
                        lambda *a, **k: FrameScores(np.array([0]), np.array([0.0]), vmaf=np.array([93.0])))
    result = vr.run_vmaf(_info("s.mkv"), _info("d.mkv"),
                         VmafOptions(crop_mode=CropMode.NONE, gpu_decode=False, vmaf_on_gpu=False))
    assert result.metric_results.get("vmaf").provenance.compute_backend == "cpu"


# ------------------------------------------- frames uploaded to the GPU

class _FakeCuda:
    """CUDA's driver API as _HostUpload uses it, in system memory: what is
    allocated, made current and waited for is recorded, and a copy is made
    as cuMemcpy2D makes it. `failing`: the call that returns an error."""

    def __init__(self, failing: str = ""):
        self.failing = failing
        self.calls: list[str] = []
        self.current: list[int] = []      # the contexts pushed, innermost last
        self.host: dict[int, object] = {}  # page-locked memory not yet freed
        self.streams: set[int] = set()
        self.retained = 0
        self.pending = 0                  # copies started and not waited for

    def __getattr__(self, name: str):
        def call(*arguments):
            self.calls.append(name)
            if name == self.failing:
                return 2
            return getattr(self, "_" + name)(*arguments)
        return call

    def _cuInit(self, _flags):
        return 0

    def _cuDeviceGet(self, device, ordinal):
        device._obj.value = ordinal
        return 0

    def _cuDevicePrimaryCtxRetain(self, context, _device):
        self.retained += 1
        context._obj.value = 0xC0DE
        return 0

    def _cuDevicePrimaryCtxRelease_v2(self, _device):
        self.retained -= 1
        return 0

    def _cuCtxPushCurrent_v2(self, context):
        self.current.append(context.value)
        return 0

    def _cuCtxPopCurrent_v2(self, _context):
        self.current.pop()
        return 0

    def _cuMemHostAlloc(self, pointer, size, flags):
        assert self.current and flags == 1
        block = ctypes.create_string_buffer(size)
        pointer._obj.value = ctypes.addressof(block)
        self.host[pointer._obj.value] = block
        return 0

    def _cuMemFreeHost(self, pointer):
        assert self.current
        del self.host[pointer.value]
        return 0

    def _cuStreamCreate(self, stream, flags):
        assert self.current and flags == 1  # non-blocking: not waited for with libvmaf's work
        stream._obj.value = 0x5700 + len(self.streams)
        self.streams.add(stream._obj.value)
        return 0

    def _cuStreamDestroy_v2(self, stream):
        assert self.current
        self.streams.remove(stream.value)
        return 0

    def _cuMemcpy2DAsync_v2(self, copy, stream):
        copy = copy._obj
        assert self.current and stream.value in self.streams
        assert (copy.srcMemoryType, copy.dstMemoryType) == (1, 2) and copy.srcHost in self.host
        for row in range(copy.Height):
            ctypes.memmove(copy.dstDevice + row * copy.dstPitch, copy.srcHost + row * copy.srcPitch,
                           copy.WidthInBytes)
        self.pending += 1
        return 0

    def _cuStreamSynchronize(self, stream):
        assert self.current and stream.value in self.streams
        self.pending = 0
        return 0


class _FakeLibvmaf:
    """libvmaf as GpuScorer uses it, its pictures "in GPU memory" blocks of
    system memory: rows a multiple of 64 bytes apart, with a guard behind."""

    def __init__(self, width: int, height: int, sample: int, cuda: _FakeCuda | None = None):
        self._cuda = cuda
        self.pitch = (width * sample + 63) // 64 * 64
        self.rows = height
        self.blocks: list[np.ndarray] = []
        self.read: list[int] = []
        self.unreferenced = 0
        self.closed = False

    def __getattr__(self, name: str):
        return lambda *_arguments: 0

    def vmaf_cuda_fetch_preallocated_picture(self, _context, picture):
        block = np.zeros(self.pitch * self.rows + 4096, dtype=np.uint8)
        block[self.pitch * self.rows:] = 0xAA
        self.blocks.append(block)
        picture._obj.data[0] = block.ctypes.data
        picture._obj.stride[0] = self.pitch
        return 0

    def vmaf_read_pictures(self, _context, reference, _distorted, index):
        if reference is not None:
            # libvmaf reads the pictures from here on: the uploads are done.
            assert self._cuda is None or self._cuda.pending == 0
            self.read.append(index)
        return 0

    def vmaf_picture_unref(self, _picture):
        self.unreferenced += 1
        return 0

    def vmaf_close(self, _context):
        self.closed = True
        return 0


def _scorer(monkeypatch, width: int, height: int, bit_depth: int, failing: str = "", **options):
    cuda = _FakeCuda(failing)
    lib = _FakeLibvmaf(width, height, 1 if bit_depth <= 8 else 2, cuda)
    monkeypatch.setattr(vmaf_cuda, "_load", lambda: lib)
    monkeypatch.setattr(vmaf_cuda, "_cuda", lambda: cuda)
    try:
        scorer = vmaf_cuda.GpuScorer(width, height, bit_depth, {"vmaf": "vmaf_v0.6.1"}, **options)
    except vmaf_cuda.VmafGpuError:
        scorer = None
    if scorer is not None:
        scorer._context = ctypes.c_void_p(1)  # as vmaf_init leaves it: close() closes libvmaf
    return scorer, lib, cuda


@pytest.mark.parametrize("width, height, bit_depth", [
    (641, 361, 8),    # odd: FFmpeg's frame has chroma rows libvmaf's picture has no place for
    (641, 361, 10),
    (1365, 767, 10),
    (1920, 1080, 8),
    (1920, 1080, 10),
])
def test_a_frames_luma_is_uploaded_into_libvmafs_picture_and_nothing_else(monkeypatch, width, height, bit_depth):
    """VMAF reads the luma alone. Each row goes to its place in a picture
    whose rows are further apart than the frame's, and nothing behind it."""
    scorer, lib, cuda = _scorer(monkeypatch, width, height, bit_depth)
    sample = 1 if bit_depth <= 8 else 2
    assert scorer.frame_bytes == (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * sample
    rng = np.random.default_rng(1)
    frames = [rng.integers(1, 256, scorer.frame_bytes, dtype=np.uint8) for _ in range(4)]

    scorer.add(bytearray(frames[0].tobytes()), bytearray(frames[1].tobytes()))
    scorer.add(frames[2].tobytes(), bytearray(frames[3].tobytes()))  # bytes too

    assert lib.read == [0, 1] and len(lib.blocks) == 4  # reference, distorted, twice
    row_bytes = width * sample
    for frame, block in zip(frames, lib.blocks, strict=True):
        picture = block[:lib.pitch * height].reshape(height, lib.pitch)
        assert np.array_equal(picture[:, :row_bytes], frame[:row_bytes * height].reshape(height, row_bytes))
        assert not picture[:, row_bytes:].any()        # each row's padding is left alone
        assert np.all(block[lib.pitch * height:] == 0xAA)  # and nothing is written behind the picture
    assert not cuda.current  # libvmaf's context is left as it was found
    assert cuda.pending == 0
    scorer.close()


def test_the_uploads_memory_is_allocated_once_and_given_back(monkeypatch):
    scorer, lib, cuda = _scorer(monkeypatch, 64, 36, 10)
    assert len(cuda.host) == 2 and len(cuda.streams) == 1 and cuda.retained == 1
    frame = bytearray(scorer.frame_bytes)
    for _ in range(5):
        scorer.add(frame, frame)
    assert cuda.calls.count("cuMemHostAlloc") == 2 and cuda.calls.count("cuStreamCreate") == 1
    assert cuda.calls.count("cuStreamSynchronize") == 5  # once a pair: both sides' copies run together

    scorer.close()
    scorer.close()

    assert not cuda.host and not cuda.streams and cuda.retained == 0 and not cuda.current
    assert lib.closed


def test_a_scorer_of_pictures_in_gpu_memory_allocates_nothing_for_uploads(monkeypatch):
    scorer, _lib, cuda = _scorer(monkeypatch, 64, 36, 8, on_device=True)
    assert not cuda.calls
    filled = []
    scorer.add_on_device(lambda address, pitch: filled.append(("reference", pitch)),
                         lambda address, pitch: filled.append(("distorted", pitch)))
    assert filled == [("reference", 64), ("distorted", 64)]
    with pytest.raises(vmaf_cuda.VmafGpuError, match="GPU memory only"):
        scorer.add(bytearray(scorer.frame_bytes), bytearray(scorer.frame_bytes))
    scorer.close()


@pytest.mark.parametrize("on_device", [(True, False), (False, True), (False, False), (True, True)])
def test_a_pair_from_two_decoders_is_copied_on_the_gpu_or_uploaded_side_by_side(monkeypatch, on_device):
    """NVIDIA's decoder copies its pictures into libvmaf's on the GPU; the
    software decoder's are written into the upload's page-locked memory and
    uploaded. A pair can be one of each (an HEVC source on NVIDIA's decoder,
    a VVC test video on the CPU)."""
    width, height = 64, 36
    scorer, lib, cuda = _scorer(monkeypatch, width, height, 10, on_device=all(on_device))
    row_bytes = width * 2
    rng = np.random.default_rng(3)
    frames = [rng.integers(1, 256, row_bytes * height, dtype=np.uint8) for _ in range(2)]

    def side(index: int):
        frame = frames[index]
        if on_device[index]:
            def on_gpu(address, pitch):
                for row in range(height):
                    ctypes.memmove(address + row * pitch, frame.ctypes.data + row * row_bytes, row_bytes)
            return True, on_gpu
        return False, lambda address: ctypes.memmove(address, frame.ctypes.data, row_bytes * height)

    scorer.add_sides(side(0), side(1))

    assert lib.read == [0] and len(lib.blocks) == 2
    for frame, block in zip(frames, lib.blocks, strict=True):
        picture = block[:lib.pitch * height].reshape(height, lib.pitch)
        assert np.array_equal(picture[:, :row_bytes], frame.reshape(height, row_bytes))
        assert np.all(block[lib.pitch * height:] == 0xAA)
    assert not cuda.current and cuda.pending == 0
    if all(on_device):
        assert not cuda.calls  # nothing allocated for uploads
    scorer.close()


def test_a_scorer_of_pictures_in_gpu_memory_refuses_one_to_upload(monkeypatch):
    scorer, _lib, _cuda = _scorer(monkeypatch, 64, 36, 8, on_device=True)
    with pytest.raises(vmaf_cuda.VmafGpuError, match="GPU memory only"):
        scorer.add_sides((True, lambda address, pitch: None), (False, lambda address: None))
    scorer.close()


@pytest.mark.parametrize("failing", ["cuInit", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuCtxPushCurrent_v2",
                                     "cuStreamCreate", "cuMemHostAlloc"])
def test_an_upload_that_cannot_be_set_up_fails_the_gpu_and_leaves_nothing_behind(monkeypatch, failing):
    """VmafGpuError: the CPU then calculates VMAF."""
    scorer, _lib, cuda = _scorer(monkeypatch, 64, 36, 8, failing=failing)
    assert scorer is None
    assert not cuda.host and not cuda.streams and cuda.retained == 0 and not cuda.current


@pytest.mark.parametrize("short", ["reference", "distorted"])
def test_a_frame_shorter_than_its_luma_is_refused_and_the_pictures_given_back(monkeypatch, short):
    scorer, lib, cuda = _scorer(monkeypatch, 64, 36, 8)
    whole, cut = bytearray(scorer.frame_bytes), bytearray(64 * 36 - 1)
    with pytest.raises(vmaf_cuda.VmafGpuError, match="luma"):
        scorer.add(cut if short == "reference" else whole, cut if short == "distorted" else whole)
    assert lib.unreferenced == 2 and not lib.read
    assert cuda.pending == 0 and not cuda.current  # the copy under way was waited for
    scorer.add(whole, whole)  # and the scorer still works
    assert lib.read == [0]
    scorer.close()


def test_a_failed_copy_is_a_gpu_failure(monkeypatch):
    scorer, lib, cuda = _scorer(monkeypatch, 64, 36, 8)
    cuda.failing = "cuMemcpy2DAsync_v2"
    with pytest.raises(vmaf_cuda.VmafGpuError, match="CUDA error 2"):
        scorer.add(bytearray(scorer.frame_bytes), bytearray(scorer.frame_bytes))
    assert lib.unreferenced == 2 and not cuda.current
    scorer.close()
    assert not cuda.host and cuda.retained == 0


# ------------------------------------- videos decoded in libvmaf's process

#: The real one: the suite's conftest makes FFmpeg decode everywhere else.
_real_score_decoded_on_gpu = vr._score_decoded_on_gpu


def _gpu_plan() -> vr._GpuPlan:
    return vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 1920, 1080, 8)


def _decoded_frames() -> FrameScores:
    return FrameScores(np.array([0, 1]), np.array([0.0, 1 / 24]), vmaf=np.array([90.0, 91.0]))


def test_vmaf_alone_with_nvidia_decoding_both_videos_decodes_them_in_libvmafs_process(monkeypatch):
    decoded = _decoded_frames()
    monkeypatch.setattr(vr, "_score_decoded_on_gpu", lambda *a, **k: decoded)
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: pytest.fail("FFmpeg decoded the videos"))
    frames = vr._score_on_gpu(_gpu_plan(), _info("s.mkv"), _info("d.mkv"), VmafOptions(), None, None,
                              "version=vmaf_v0.6.1", HwAccelPlan("cuda", "cuda"), 2)
    assert frames is decoded


@pytest.mark.parametrize("hwaccel", [HwAccelPlan("cuda", None), HwAccelPlan(None, "cuda"), HwAccelPlan("qsv", "qsv")])
def test_ffmpeg_decodes_when_nvidia_does_not_decode_both(monkeypatch, hwaccel):
    monkeypatch.setattr(vr, "_score_decoded_on_gpu", lambda *a, **k: pytest.fail("decoded in libvmaf's process"))
    by_ffmpeg = _decoded_frames()
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: by_ffmpeg)
    frames = vr._score_on_gpu(_gpu_plan(), _info("s.mkv"), _info("d.mkv"), VmafOptions(), None, None,
                              "version=vmaf_v0.6.1", hwaccel, 2)
    assert frames is by_ffmpeg


@pytest.mark.parametrize("error", [gpu_frames.GpuDecodeUnavailableError("a video is scaled"),
                                   gpu_frames.GpuDecodeFailedError("the GPU's decoder found an error in the video")])
def test_when_decoding_in_libvmafs_process_is_refused_or_fails_ffmpeg_decodes(monkeypatch, error):
    def refuse(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(vr, "_score_decoded_on_gpu", refuse)
    by_ffmpeg = _decoded_frames()
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: by_ffmpeg)
    statuses = []
    frames = vr._score_on_gpu(_gpu_plan(), _info("s.mkv"), _info("d.mkv"), VmafOptions(), None, None,
                              "version=vmaf_v0.6.1", HwAccelPlan("cuda", "cuda"), 2, on_status=statuses.append)
    assert frames is by_ffmpeg
    failed = [status for status in statuses if status.startswith("GPU decoding failed")]
    assert failed == ([f"GPU decoding failed ({error}); decoding through FFmpeg instead…"]
                      if isinstance(error, gpu_frames.GpuDecodeFailedError) else [])


def test_decoded_in_libvmafs_process_the_limit_is_that_of_ffmpegs_raw_outputs(monkeypatch):
    """One frame more than the limit, as _execute_run gives FFmpeg (see
    test_a_duration_limit_gives_the_raw_outputs_one_frame_more), as the text
    FFmpeg would read: 30 s at 24 fps is -t 30.042."""
    seen = {}

    def score(*_args, **kwargs):
        seen.update(kwargs)
        return np.array([0, 1], dtype=np.int32), {"vmaf": np.array([90.0, 91.0])}

    monkeypatch.setattr(vmaf_cuda, "score_decoded", score)
    frames = _real_score_decoded_on_gpu(_gpu_plan(), _info("s.mkv"), _info("d.mkv"),
                                        VmafOptions(duration_limit=30.0), None, None, HwAccelPlan("cuda", "cuda"), 721)
    assert seen["duration_limit"] == "30.042"
    assert seen["scale_algorithm"] == "bicubic"  # the row's, for a video scaled on the GPU
    assert (seen["width"], seen["height"], seen["bit_depth"], seen["n_subsample"]) == (1920, 1080, 8, 1)
    assert frames.values("vmaf").tolist() == [90.0, 91.0]
    frames = _real_score_decoded_on_gpu(_gpu_plan(), _info("s.mkv"), _info("d.mkv"), VmafOptions(), None, None,
                                        HwAccelPlan("cuda", "cuda"), 721)
    assert seen["duration_limit"] is None


def test_any_probe_failure_means_vmaf_is_calculated_on_the_cpu(monkeypatch):
    """Only a crash was caught: anything else escaped the probe, left it
    unanswered, and failed every video's setup."""
    from vmaf_app.core import gpu, isolated

    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [gpu.GpuVendor.NVIDIA])
    monkeypatch.setattr(isolated, "run_isolated", lambda *a, **k: (_ for _ in ()).throw(OSError("pipe broke")))
    monkeypatch.setattr(vmaf_cuda, "_probed", None)
    available, text = vmaf_cuda.gpu_vmaf_available()
    assert not available and "pipe broke" in text


def test_a_failed_probe_is_made_again_a_while_later(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vmaf_cuda.time, "monotonic", lambda: clock[0])
    # (backend, Vulkan's GPU, text), as _probe_once answers for the GPU backend setting
    answers = iter([(None, None, "driver restarting"), ("cuda", None, "libvmaf 3.0")])
    monkeypatch.setattr(vmaf_cuda, "_probe_once", lambda preference: next(answers))
    monkeypatch.setattr(vmaf_cuda, "_probed", None)
    assert vmaf_cuda.gpu_vmaf_available() == (False, "driver restarting")
    clock[0] += vmaf_cuda.FAILED_PROBE_RETRY_SECONDS - 1
    vmaf_cuda.forget_failed_probe()
    assert vmaf_cuda.gpu_vmaf_available() == (False, "driver restarting")
    clock[0] += 1
    vmaf_cuda.forget_failed_probe()
    assert vmaf_cuda.gpu_vmaf_available() == (True, "libvmaf 3.0")
    vmaf_cuda.forget_failed_probe()  # a working probe is kept
    assert vmaf_cuda.gpu_vmaf_available() == (True, "libvmaf 3.0")


# ------------------------------------------------ paths given to libvmaf

def test_a_path_goes_to_libvmaf_in_the_ansi_code_page_not_utf8(tmp_path):
    """libvmaf opens files with fopen(): UTF-8 bytes of "modèle" named a
    file that is not there, and VMAF v1 on the GPU failed for every user with
    an accented letter in their profile's name."""
    assert vmaf_cuda.path_bytes(Path("C:/models/vmaf.json")) == b"C:\\models\\vmaf.json"
    accented = "C:\\Users\\J\u00f6rg\\mod\u00e8le.json"
    try:
        expected = accented.encode("mbcs", errors="strict")
    except UnicodeEncodeError:
        pytest.skip("this PC's code page has no accented letters")
    assert vmaf_cuda.path_bytes(accented) == expected != accented.encode("utf-8")


class _ShortPaths:
    """kernel32's GetShortPathNameW, answering `short` (None: it fails)."""

    def __init__(self, short: str | None):
        self.short, self.asked = short, []

    def GetShortPathNameW(self, path, buffer, size):
        self.asked.append(path)
        if self.short is None:
            return 0
        buffer.value = self.short
        return len(self.short)


def _unencodable() -> str:
    for text in ("C:\\\u6a21\u578b\\vmaf.json", "C:\\mod\u00e8le\\vmaf.json",
                 "C:\\\u043c\u043e\u0434\u0435\u043b\u044c\\vmaf.json"):
        try:
            text.encode("mbcs", errors="strict")
        except UnicodeEncodeError:
            return text
    pytest.skip("this PC's code page has every letter tried")


def test_a_name_outside_the_code_page_goes_by_its_short_path(monkeypatch):
    path = _unencodable()
    kernel = _ShortPaths("C:\\6A21~1\\vmaf.json")
    monkeypatch.setattr(vmaf_cuda.ctypes, "windll", SimpleNamespace(kernel32=kernel))
    assert vmaf_cuda.path_bytes(path) == b"C:\\6A21~1\\vmaf.json"
    assert kernel.asked == [path]


@pytest.mark.parametrize("short", [None, "same"])
def test_a_name_with_no_short_path_is_a_gpu_failure_with_the_reason(monkeypatch, short):
    """A drive without 8.3 names gives the long name back: the CPU then
    calculates, as for any other failure of the GPU's."""
    path = _unencodable()
    monkeypatch.setattr(vmaf_cuda.ctypes, "windll",
                        SimpleNamespace(kernel32=_ShortPaths(path if short == "same" else None)))
    with pytest.raises(vmaf_cuda.VmafGpuError, match="code page"):
        vmaf_cuda.path_bytes(path)
