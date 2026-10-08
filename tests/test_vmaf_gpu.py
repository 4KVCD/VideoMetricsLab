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

from vmaf_app.core import gpu_frames, job_runner, vmaf_cuda
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropMode, FrameScores, VideoInfo, VmafOptions
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


def test_a_frames_luma_is_uploaded_into_libvmafs_picture_and_nothing_else(subtests):
    """VMAF reads the luma alone. Each row goes to its place in a picture
    whose rows are further apart than the frame's, and nothing behind it."""
    def check(width, height, bit_depth, monkeypatch):
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

    for width, height, bit_depth in [
        (641, 361, 8),    # odd: FFmpeg's frame has chroma rows libvmaf's picture has no place for
        (641, 361, 10),
        (1365, 767, 10),
        (1920, 1080, 8),
        (1920, 1080, 10),
    ]:
        with subtests.test(width=width, height=height, bit_depth=bit_depth), pytest.MonkeyPatch.context() as case_patch:
            check(width, height, bit_depth, case_patch)


def test_a_pair_from_two_decoders_is_copied_on_the_gpu_or_uploaded_side_by_side(subtests):
    """NVIDIA's decoder copies its pictures into libvmaf's on the GPU; the
    software decoder's are written into the upload's page-locked memory and
    uploaded. A pair can be one of each (an HEVC source on NVIDIA's decoder,
    a VVC test video on the CPU)."""
    def check(on_device, monkeypatch):
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

    for on_device in [(True, False), (False, True), (False, False), (True, True)]:
        with subtests.test(on_device=on_device), pytest.MonkeyPatch.context() as case_patch:
            check(on_device, case_patch)


# ------------------------------------- videos decoded in libvmaf's process


def _gpu_plan() -> vr._GpuPlan:
    return vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 1920, 1080, 8)


def _decoded_frames() -> FrameScores:
    return FrameScores(np.array([0, 1]), np.array([0.0, 1 / 24]), vmaf=np.array([90.0, 91.0]))


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


def test_any_probe_failure_means_vmaf_is_calculated_on_the_cpu(monkeypatch):
    """Only a crash was caught: anything else escaped the probe, left it
    unanswered, and failed every video's setup."""
    from vmaf_app.core import gpu, isolated

    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [gpu.GpuVendor.NVIDIA])
    monkeypatch.setattr(isolated, "run_isolated", lambda *a, **k: (_ for _ in ()).throw(OSError("pipe broke")))
    monkeypatch.setattr(vmaf_cuda, "_probed", None)
    available, text = vmaf_cuda.gpu_vmaf_available()
    assert not available and "pipe broke" in text
