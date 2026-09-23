from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmaf_app.core import perceptual_cpu
from vmaf_app.core import perceptual_vship as vship
from vmaf_app.core.analysis_request import AnalysisRequest
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import CropMode, GpuVendor, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualTaskOutput


def _info(path: str, *, pix_fmt: str = "yuv420p") -> VideoInfo:
    return VideoInfo(Path(path), 64, 48, 24.0, 1.0, 24, "h264", pix_fmt=pix_fmt)


def _request() -> AnalysisRequest:
    return analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE), ("ssimulacra2", "butteraugli"),
    )


def _cpu_output() -> PerceptualTaskOutput:
    provenance = MetricProvenance("test", "1", "cpu", "test-cpu-v1")
    metrics = MetricResultSet([
        FrameMetricResult("ssimulacra2", [0], [0.0], [90.0], provenance),
        FrameMetricResult("butteraugli", [0], [0.0], [0.2], provenance),
    ])
    return PerceptualTaskOutput(metrics, None, None, 1)


def _single_metric_output(key: str, value: float, backend: str) -> PerceptualTaskOutput:
    provenance = MetricProvenance(backend, "1", backend, f"{key}-{backend}-v1")
    results = MetricResultSet([
        FrameMetricResult(key, [0], [0.0], [value], provenance),
    ])
    return PerceptualTaskOutput(results, None, None, 1)


def test_vship_device_info_matches_c_api_layout():
    assert ctypes.sizeof(vship._DeviceInfo) == 304
    assert [name for name, _kind in vship._DeviceInfo._fields_] == [
        "name", "VRAMSize", "integrated", "MultiProcessorCount", "WarpSize",
        "vulkanFeatureMatrix",
    ]


@pytest.mark.parametrize(("pixel_format", "family", "sample"), [
    ("yuv420p", 0, vship._VSHIP_ENUMS[8]),
    ("yuv420p10le", 0, vship._VSHIP_ENUMS[10]),
    ("nv12", 0, vship._VSHIP_ENUMS[8]),
    ("p010le", 0, vship._VSHIP_ENUMS[10]),
    ("gbrp10le", 1, vship._VSHIP_ENUMS[10]),
])
def test_vship_maps_common_ffmpeg_pixel_formats(pixel_format, family, sample):
    image = vship._image_format(_info("video.mkv", pix_fmt=pixel_format))
    assert image.family == family
    assert image.sample == sample


def test_unsupported_pixel_format_is_a_cpu_fallback_condition():
    with pytest.raises(vship.VshipUnavailableError, match="does not support"):
        vship._image_format(_info("video.mkv", pix_fmt="yuv411p10le"))


def test_no_supported_gpu_falls_back_to_cpu(monkeypatch):
    source, test = _info("source.mkv"), _info("test.mkv")
    expected = _cpu_output()
    statuses = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (None, "no supported GPU"))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", lambda *args, **kwargs: expected)

    actual = vship.apply_vship_cpu_fallback(
        source, test, _request(), _request().metrics, on_status=statuses.append,
    )

    assert actual is expected
    assert any("no supported GPU" in status and "CPU" in status for status in statuses)


def test_cpu_selection_skips_gpu_detection(monkeypatch):
    source, test = _info("source.mkv"), _info("test.mkv")
    request = analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE), ("ssimulacra2", "butteraugli"),
        {"ssimulacra2": "cpu", "butteraugli": "cpu"},
    )
    expected = _cpu_output()
    calls = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: pytest.fail("CPU mode must not probe Vship"))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", lambda *args, **kwargs: calls.append(args[3]) or expected)

    actual = vship.apply_vship_cpu_fallback(source, test, request, request.metrics)

    assert actual is expected
    assert [spec.key for spec in calls[0]] == ["ssimulacra2", "butteraugli"]


def test_mixed_backend_selection_runs_each_metric_on_selected_backend(monkeypatch):
    source, test = _info("source.mkv"), _info("test.mkv")
    request = analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE), ("ssimulacra2", "butteraugli"),
        {"ssimulacra2": "gpu", "butteraugli": "cpu"},
    )
    device = vship.VshipDevice("nvidia", "test GPU", 0, "4.0.2", None)
    crops = (None, None)
    routed = {"gpu": [], "cpu": []}
    progress = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: crops)

    def run_gpu(_source, _test, _request, specs, _device, *_crops, on_progress=None, **_kwargs):
        routed["gpu"].extend(spec.key for spec in specs)
        on_progress(1, 1, 10.0)
        return _single_metric_output("ssimulacra2", 91.0, "gpu")

    def run_cpu(_source, _test, _request, specs, *, resolved_crops=None, on_progress=None, **_kwargs):
        routed["cpu"].extend(spec.key for spec in specs)
        assert resolved_crops == crops
        on_progress(1, 1, 8.0)
        return _single_metric_output("butteraugli", 0.2, "cpu")

    monkeypatch.setattr(vship, "run_vship_task", run_gpu)
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", run_cpu)

    actual = vship.apply_vship_cpu_fallback(
        source, test, request, request.metrics,
        on_progress=lambda cur, total, fps: progress.append((cur, total, fps)),
    )

    assert routed == {"gpu": ["ssimulacra2"], "cpu": ["butteraugli"]}
    assert actual.metrics.keys() == ("ssimulacra2", "butteraugli")
    assert actual.metrics.get("ssimulacra2").provenance.compute_backend == "gpu"
    assert actual.metrics.get("butteraugli").provenance.compute_backend == "cpu"
    assert progress == [(1, 2, 10.0), (2, 2, 8.0)]


def test_vship_processing_error_falls_back_without_repeating_crop_detection(monkeypatch):
    source, test = _info("source.mkv"), _info("test.mkv")
    request = _request()
    expected = _cpu_output()
    crop_calls = []
    crops = (None, None)
    device = vship.VshipDevice("nvidia", "test GPU", 0, "4.0.2", None)
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: crop_calls.append(args) or crops)
    monkeypatch.setattr(vship, "run_vship_task", lambda *args, **kwargs: (_ for _ in ()).throw(
        vship.VshipUnavailableError("GPU compute unavailable"),
    ))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", lambda *args, **kwargs: expected)

    actual = vship.apply_vship_cpu_fallback(source, test, request, request.metrics)

    assert actual is expected
    assert len(crop_calls) == 1


def test_cancellation_does_not_start_cpu_fallback(monkeypatch):
    source, test = _info("source.mkv"), _info("test.mkv")
    request = _request()
    device = vship.VshipDevice("nvidia", "test GPU", 0, "4.0.2", None)
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *args, **kwargs: (_ for _ in ()).throw(
        PerceptualCancelled("cancelled"),
    ))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", lambda *args, **kwargs: pytest.fail(
        "CPU fallback must not run after cancellation",
    ))

    with pytest.raises(PerceptualCancelled):
        vship.apply_vship_cpu_fallback(source, test, request, request.metrics)


# ------------------------------------------------------------ frame transport
#
# The GPU is faked; everything else is real. FFmpeg is replaced by a Python
# child that writes raw frames, and those frames travel through the real
# reader threads, the pinned-buffer ring and the scoring lanes.

def _frames_command(count: int, frame_bytes: int, *, exit_code: int = 0, partial: bool = False) -> list[str]:
    """A child writing `count` raw frames whose first byte is the frame index."""
    script = (
        "import sys\n"
        f"n, size, partial, code = {count}, {frame_bytes}, {partial}, {exit_code}\n"
        "out = sys.stdout.buffer\n"
        "for i in range(n):\n"
        "    out.write(bytes([i % 256]) + bytes(size - 1))\n"
        "if partial:\n"
        "    out.write(bytes(size // 2))\n"
        "out.flush()\n"
        "sys.exit(code)\n"
    )
    return [sys.executable, "-c", script]


class _FakePinned:
    """Ordinary memory standing in for Vship's pinned allocation."""

    def __init__(self, _lib, size):
        self.array = (ctypes.c_uint8 * size)()
        self.address = ctypes.c_void_p(ctypes.addressof(self.array))

    # The real plane arithmetic, captured before the tests swap the class out.
    planes = vship._PinnedBuffer.planes

    def close(self):
        pass


def _fake_device():
    lib = SimpleNamespace(Vship_SetDevice=lambda _gpu: 0, Vship_SSIMU2Free=lambda _h: 0,
                          Vship_ButteraugliFree=lambda _h: 0)
    return vship.VshipDevice("nvidia", "fake GPU", 0, "5.1.1", SimpleNamespace(library=lib))


def _hevc(name: str) -> VideoInfo:
    return VideoInfo(Path(name), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt="yuv420p10le")


_FRAME_BYTES = vship._image_format(_hevc("x.mkv")).frame_layout(64, 48)[0]


def _both(command):
    return {"source": [command], "test": [command]}


def _run(monkeypatch, *, children, metrics=("ssimulacra2", "butteraugli"), gpu_decode=False,
         source=None, test=None, cancel_after=None, hwaccel=lambda _vendor, _codec: None):
    """run_vship_task where each spawned 'FFmpeg' is the next child for its input.

    `children` maps "source"/"test" to the commands that input's successive
    starts run: the two readers start concurrently, so which spawns first is
    a race, and children are matched to inputs by path rather than by order.

    The fake score is the frame index read back out of the pinned buffer the
    lane was handed, so a mixed-up slot or pairing shows up in the values.
    """
    monkeypatch.setattr(vship, "_PinnedBuffer", _FakePinned)
    monkeypatch.setattr(vship, "_init_handler", lambda *_args: vship._Handler())
    monkeypatch.setattr(vship, "pick_hwaccel", hwaccel)
    queues = {side: list(commands) for side, commands in children.items()}
    spawned: dict[str, list[list[str]]] = {"source": [], "test": []}
    source_path = str((source or _hevc("source.mkv")).path.resolve())

    def fake_spawn(command):
        side = "source" if source_path in command else "test"
        spawned[side].append(command)
        process = vship.proc_util.popen(queues[side].pop(0), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        return process, process.stdout

    cancel, scored = threading.Event(), []

    def fake_compute(_device, key, _handler, source_planes, test_planes, *_strides):
        index = source_planes[0][0]
        assert test_planes[0][0] == index, "a lane paired frames from different positions"
        scored.append(index)
        if cancel_after is not None and len(scored) >= cancel_after:
            cancel.set()
        return float(index) + (0.5 if key == "butteraugli" else 0.0)

    monkeypatch.setattr(vship, "_spawn_raw_ffmpeg", fake_spawn)
    monkeypatch.setattr(vship, "_compute_metric", fake_compute)
    request = analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE, gpu_decode=gpu_decode), metrics)
    output = vship.run_vship_task(source or _hevc("source.mkv"), test or _hevc("test.mkv"), request,
                                  request.metrics, _fake_device(), None, None, cancel_event=cancel)
    return output, spawned


def test_frames_arrive_in_order_through_the_ring_and_both_lanes(monkeypatch):
    """Many more frames than ring slots, two lanes per metric: every score
    lands at its own frame index and the slots are recycled, not exhausted."""
    count = vship._RING_SLOTS * 7 + 3
    output, _spawned = _run(monkeypatch, children=_both(_frames_command(count, _FRAME_BYTES)))

    assert output.compared_frame_count == count
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    assert list(output.metrics.get("butteraugli").values) == [i + 0.5 for i in range(count)]
    assert list(output.metrics.get("ssimulacra2").frame) == list(range(count))


def test_hardware_decode_refused_before_any_frame_is_retried_in_software(monkeypatch):
    """Hardware decode can refuse a stream (an unsupported profile, say). The
    same pictures are then decoded in software instead of abandoning the GPU run."""
    output, spawned = _run(
        monkeypatch, gpu_decode=True, metrics=("ssimulacra2",), hwaccel=lambda _v, _c: "cuda",
        children={
            "source": [_frames_command(0, _FRAME_BYTES, exit_code=1),  # hardware: refused
                       _frames_command(5, _FRAME_BYTES)],             # software retry
            "test": [_frames_command(5, _FRAME_BYTES)],
        },
    )

    assert list(output.metrics.get("ssimulacra2").values) == [0.0, 1.0, 2.0, 3.0, 4.0]
    hardware, software = spawned["source"]
    assert "-hwaccel" in hardware and "hwdownload" in " ".join(hardware)
    assert "-hwaccel" not in software and "hwdownload" not in " ".join(software)
    assert len(spawned["test"]) == 1, "the input that decoded fine was not restarted"


def test_a_truncated_frame_is_an_error_not_a_short_result(monkeypatch):
    children = {"source": [_frames_command(3, _FRAME_BYTES, partial=True)],
                "test": [_frames_command(4, _FRAME_BYTES)]}
    with pytest.raises(vship.VshipUnavailableError, match="partway"):
        _run(monkeypatch, metrics=("ssimulacra2",), children=children)


def test_different_frame_counts_are_reported(monkeypatch):
    children = {"source": [_frames_command(4, _FRAME_BYTES)], "test": [_frames_command(6, _FRAME_BYTES)]}
    with pytest.raises(vship.VshipUnavailableError, match="different frame counts"):
        _run(monkeypatch, metrics=("ssimulacra2",), children=children)


def test_cancelling_mid_run_stops_cleanly(monkeypatch):
    before = threading.active_count()
    with pytest.raises(PerceptualCancelled):
        _run(monkeypatch, metrics=("ssimulacra2",), cancel_after=10,
             children=_both(_frames_command(2000, _FRAME_BYTES)))
    assert threading.active_count() <= before, "a reader or lane thread outlived the task"


def test_vvc_is_decoded_in_software_while_the_other_input_uses_the_gpu(monkeypatch):
    """Per input and per codec: the GPU has no VVC decoder, so VVC goes to
    FFmpeg's software decoder while the HEVC reference keeps NVDEC."""
    test = VideoInfo(Path("test.mkv"), 64, 48, 24.0, 1.0, 24, "vvc", pix_fmt="yuv420p10le")
    _output, spawned = _run(
        monkeypatch, gpu_decode=True, metrics=("ssimulacra2",), test=test,
        hwaccel=lambda _vendor, codec: None if codec == "vvc" else "cuda",
        children=_both(_frames_command(2, _FRAME_BYTES)),
    )
    (source_cmd,), (test_cmd,) = spawned["source"], spawned["test"]
    assert source_cmd[source_cmd.index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel" not in test_cmd


def test_gpu_decode_off_decodes_every_input_in_software(monkeypatch):
    def hwaccel(vendor, _codec):
        return None if vendor is GpuVendor.NONE else "cuda"

    _output, spawned = _run(monkeypatch, gpu_decode=False, metrics=("ssimulacra2",), hwaccel=hwaccel,
                            children=_both(_frames_command(2, _FRAME_BYTES)))
    assert all("-hwaccel" not in command for commands in spawned.values() for command in commands)


def test_the_large_pipe_carries_raw_frames_intact():
    """The real pipe and a real child process: bytes arrive whole and in order."""
    frame_bytes = 1_000_003  # deliberately not a power of two
    process, reader = vship._spawn_raw_ffmpeg(_frames_command(4, frame_bytes))
    view = memoryview(bytearray(frame_bytes))
    firsts = []
    while vship._read_exact(reader, view) == frame_bytes:
        firsts.append(view[0])
    reader.close()
    assert process.wait() == 0
    assert firsts == [0, 1, 2, 3]


def test_scores_are_packed_by_frame_index_across_chunk_boundaries():
    """Lanes finish out of order and a long video spans several chunks."""
    scores = vship._ScoreArray()
    count = vship._ScoreArray._CHUNK * 2 + 5
    for index in range(count):
        scores.reserve(index)
    for index in reversed(range(count)):  # completion order must not matter
        scores[index] = index * 0.5
    values = scores.values(count)
    assert values.dtype == np.float32 and len(values) == count
    assert values[0] == 0.0 and values[-1] == (count - 1) * 0.5
    assert values[vship._ScoreArray._CHUNK] == vship._ScoreArray._CHUNK * 0.5
