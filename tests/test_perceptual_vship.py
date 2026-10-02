from __future__ import annotations

import ctypes
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmaf_app.core import perceptual_cpu
from vmaf_app.core import perceptual_vship as vship
from vmaf_app.core.analysis_request import AnalysisRequest
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_cache import VSHIP_COLOR_TAGS
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


@pytest.mark.parametrize(("struct", "size", "offsets"), [
    # Vship_Colorspace_t: four int64, seven 4-byte fields, the 16-byte crop.
    (vship._Colorspace, 88, {"sample": 32, "subsampling": 40, "YUVMatrix": 56, "crop": 68}),
    (vship._InitSsimulacra2, 192, {"structType": 0, "src_colorspace": 8, "dis_colorspace": 96, "gpu_id": 184}),
    (vship._InitButteraugli, 200, {"dis_colorspace": 96, "Qnorm": 184, "intensity_multiplier": 188, "gpu_id": 192}),
    (vship._InitCvvdp, 216, {"fps": 184, "resizeToDisplay": 188, "model_key_cstr": 192,
                             "model_config_json_cstr": 200, "gpu_id": 208}),
    (vship._ScoreSsimulacra2, 16, {"structType": 0, "score": 8}),
    (vship._ScoreButteraugli, 48, {"normQ": 8, "norm3": 16, "norminf": 24, "dstp": 32, "dststride": 40}),
    (vship._ScoreCvvdp, 32, {"score": 8, "dstp": 16, "dststride": 24}),
])
def test_vship_51_structs_match_the_c_layout(struct, size, offsets):
    """The init and score structs of VshipAPI.h (Vship 5.1.1), as the x64 C
    compiler lays them out: a wrong offset would hand Vship garbage."""
    assert ctypes.sizeof(struct) == size
    assert {name: getattr(struct, name).offset for name in offsets} == offsets


def test_only_a_vship_with_the_51_api_is_used():
    """4.x has no Vship_InitHandler; its per-metric functions are only a
    deprecated layer in 5.1, and this module no longer calls them."""
    assert vship._has_api(SimpleNamespace(**dict.fromkeys(vship._API_FUNCTIONS)))
    older = {name: None for name in vship._API_FUNCTIONS if name not in {
        "Vship_InitHandler", "Vship_ComputeHandler", "Vship_FreeHandler", "Vship_PinnedMalloc2"}}
    older |= {"Vship_SSIMU2Init": None, "Vship_ComputeSSIMU2": None, "Vship_PinnedMalloc": None}
    assert not vship._has_api(SimpleNamespace(**older))


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


def test_a_pixel_format_the_gpu_metrics_cannot_read_is_a_cpu_fallback_condition():
    with pytest.raises(vship.VshipUnavailableError, match="cannot read the decoded pixel format pal8"):
        vship._image_format(_info("video.mkv", pix_fmt="pal8"))


@pytest.mark.parametrize(("pixel_format", "piped", "sample", "shifts", "rgb"), [
    # Any subsampling at any depth: Vship takes two shifts and a sample
    # type. The "Vship supports these layouts" table allowed 4:4:0 only up
    # to 12 bits and 4:1:0 / 4:1:1 only at 8 -- FFmpeg's formats, not Vship's.
    ("yuv440p12le", "yuv440p12le", 12, (0, 1), False),
    ("yuv410p", "yuv410p", 8, (2, 1), False),
    ("yuv444p14le", "yuv444p14le", 14, (0, 0), False),
    ("yuv422p16be", "yuv422p16le", 16, (1, 0), False),
    ("yuva420p10le", "yuv420p10le", 10, (1, 1), False),  # alpha dropped
    ("gray10le", "yuv420p10le", 10, (1, 1), False),  # monochrome: neutral chroma
    ("p012le", "yuv420p12le", 12, (1, 1), False),
    ("nv24", "yuv444p", 8, (0, 0), False),
    ("gbrap12le", "gbrp12le", 12, (0, 0), True),
    ("bgr48le", "gbrp16le", 16, (0, 0), True),
])
def test_every_ffmpeg_layout_reaches_vship_as_a_planar_one(pixel_format, piped, sample, shifts, rgb):
    image = vship._image_format(_info("video.mkv", pix_fmt=pixel_format))
    assert image.pixel_format == piped
    assert image.sample == vship._VSHIP_ENUMS[sample]
    assert (image.subw, image.subh) == shifts and (image.family == 1) == rgb


def _colorspace(width=1920, height=1080, pix_fmt="yuv420p10le", **tags):
    info = VideoInfo(Path("video.mkv"), width, height, 24.0, 1.0, 24, "hevc", pix_fmt=pix_fmt, **tags)
    return vship._vship_colorspace(info, vship._image_format(info), width, height)


@pytest.mark.parametrize(("tags", "matrix", "transfer", "primaries"), [
    # As ffprobe names them, to VshipColor.h's values -- FFVship's mapping.
    ({"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}, 1, 1, 1),
    ({"color_space": "smpte170m", "color_transfer": "smpte170m", "color_primaries": "smpte170m"}, 6, 6, 6),
    ({"color_space": "bt470bg", "color_transfer": "bt470bg", "color_primaries": "bt470bg"}, 5, 5, 5),
    ({"color_space": "bt709", "color_transfer": "bt470m", "color_primaries": "bt470m"}, 1, 4, 4),
    ({"color_space": "bt709", "color_transfer": "smpte240m", "color_primaries": "smpte240m"}, 1, 7, 7),
    # BT.2020 SDR: its 10- and 12-bit transfer tags are the BT.709 curve.
    ({"color_space": "bt2020nc", "color_transfer": "bt2020-10", "color_primaries": "bt2020"}, 9, 1, 9),
    ({"color_space": "bt2020c", "color_transfer": "bt2020-12", "color_primaries": "bt2020"}, 10, 1, 9),
    ({"color_space": "bt2020nc", "color_transfer": "smpte2084", "color_primaries": "bt2020"}, 9, 16, 9),
    ({"color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020"}, 9, 18, 9),
    ({"color_space": "ictcp", "color_transfer": "smpte2084", "color_primaries": "bt2020"}, 14, 16, 9),
    ({"color_space": "ycgco", "color_transfer": "iec61966-2-1", "color_primaries": "smpte432"}, 8, 13, 12),
    ({"color_space": "ycgco-re", "color_transfer": "linear", "color_primaries": "bt709"}, 16, 8, 1),
    ({"color_space": "ycgco-ro", "color_transfer": "smpte428", "color_primaries": "bt709"}, 17, 17, 1),
    # Untagged, guessed as FFVship does.
    ({}, 1, 1, 1),
    ({"color_space": "ictcp"}, 14, 16, 9),
    ({"color_space": "bt2020nc"}, 9, 16, 9),
])
def test_color_tags_map_to_vships_values(tags, matrix, transfer, primaries):
    color = _colorspace(**tags)
    assert (color.YUVMatrix, color.transferFunction, color.primaries) == (matrix, transfer, primaries)


def test_untagged_video_is_guessed_as_ffvship_guesses_it():
    sd = _colorspace(width=720, height=576)
    assert (sd.YUVMatrix, sd.transferFunction, sd.primaries, sd.range) == (5, 5, 5, 0)
    rgb = _colorspace(pix_fmt="gbrp")  # sRGB, full range
    assert (rgb.YUVMatrix, rgb.transferFunction, rgb.primaries, rgb.range) == (0, 13, 1, 1)
    assert _colorspace(pix_fmt="yuvj420p").range == 1


@pytest.mark.parametrize(("tag", "value"), [
    ("color_space", "smpte240m"), ("color_space", "fcc"), ("color_transfer", "log100"),
    ("color_primaries", "film"), ("chroma_location", "bottomleft"),
])
def test_a_tag_vship_has_no_value_for_is_refused(tag, value):
    with pytest.raises(vship.VshipUnavailableError, match="Vship does not support"):
        _colorspace(**{tag: value})


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
    device = vship.VshipDevice("cuda", "test GPU", 0, "5.1.1", None)
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
    device = vship.VshipDevice("cuda", "test GPU", 0, "5.1.1", None)
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
    device = vship.VshipDevice("cuda", "test GPU", 0, "5.1.1", None)
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
    return [sys.executable, "-S", "-c", script]


class _FakePinned:
    """Ordinary memory standing in for Vship's pinned allocation."""

    def __init__(self, _lib, size, _gpu_id=0):
        self.array = (ctypes.c_uint8 * size)()
        self.address = ctypes.c_void_p(ctypes.addressof(self.array))

    # The real plane arithmetic, captured before the tests swap the class out.
    planes = vship._PinnedBuffer.planes

    def close(self):
        pass


def _fake_device():
    lib = SimpleNamespace(Vship_FreeHandler=lambda _handle: 0)
    return vship.VshipDevice("cuda","fake GPU", 0, "5.1.1", SimpleNamespace(library=lib))


def _hevc(name: str) -> VideoInfo:
    return VideoInfo(Path(name), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt="yuv420p10le")


_FRAME_BYTES = vship._image_format(_hevc("x.mkv")).frame_layout(64, 48)[0]


def _both(command):
    return {"source": [command], "test": [command]}


def _run(monkeypatch, *, children, metrics=("ssimulacra2", "butteraugli"), gpu_decode=False,
         source=None, test=None, cancel_after=None, hwaccel=lambda _vendor, _codec: None,
         inspect=None, fail=None, on_status=None, together=False, on_progress=None):
    """run_vship_task where each spawned 'FFmpeg' is the next child for its input.

    `children` maps "source"/"test" to the commands that input's successive
    starts run: the two readers start concurrently, so which spawns first is
    a race, and children are matched to inputs by path rather than by order.

    The fake score is the frame index read back out of the pinned buffer the
    lane was handed, so a mixed-up slot or pairing shows up in the values.
    """
    monkeypatch.setattr(vship, "_PinnedBuffer", _FakePinned)
    monkeypatch.setattr(vship, "_init_handler", lambda *_args: vship._Handle())
    monkeypatch.setattr(vship, "pick_hwaccel", hwaccel)
    queues = {side: list(commands) for side, commands in children.items()}
    spawned: dict[str, list[list[str]]] = {"source": [], "test": []}
    source_path = str((source or _hevc("source.mkv")).path.resolve())

    def fake_spawn(command):
        side = "source" if source_path in command else "test"
        spawned[side].append(command)
        # The last command is reused: each metric has a pass (and a decode) of its own.
        queue = queues[side]
        process = vship.proc_util.popen(queue.pop(0) if len(queue) > 1 else queue[0], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        return process, process.stdout

    cancel, scored = threading.Event(), []

    def fake_compute(_device, key, _handler, source_planes, test_planes, *_strides):
        index = source_planes[0][0]
        assert test_planes[0][0] == index, "a lane paired frames from different positions"
        scored.append(index)
        if inspect is not None:
            inspect(index, source_planes, test_planes)
        if fail is not None and (fail(key, index) if callable(fail) else key == fail[0] and index >= fail[1]):
            raise vship.VshipUnavailableError(f"Vship {key} failed: out of memory")
        if cancel_after is not None and len(scored) >= cancel_after:
            cancel.set()
        return float(index) + (0.5 if key == "butteraugli" else 0.0)

    monkeypatch.setattr(vship, "_spawn_raw_ffmpeg", fake_spawn)
    monkeypatch.setattr(vship, "_compute_metric", fake_compute)
    request = analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE, gpu_decode=gpu_decode), metrics)
    output = vship.run_vship_task(source or _hevc("source.mkv"), test or _hevc("test.mkv"), request,
                                  request.metrics, _fake_device(), None, None, cancel_event=cancel,
                                  on_status=on_status, together=together, on_progress=on_progress)
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
    same pictures are then decoded in software instead of abandoning the GPU run.

    The pass says where each video is decoded, and again when one falls
    back: the window's "Decoder: ..." for a video with only GPU metrics
    comes from these messages."""
    statuses = []
    output, spawned = _run(
        monkeypatch, gpu_decode=True, metrics=("ssimulacra2",), hwaccel=lambda _v, _c: "cuda",
        on_status=statuses.append,
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
    assert statuses == [
        "Vship GPU (fake GPU): calculating SSIMULACRA2 (GPU decode: source cuda, distorted cuda)…",
        "GPU decode failed for the source, decoding it in software (GPU decode: source cpu, distorted cuda)…",
    ]


def test_a_pass_with_gpu_decode_off_says_so(monkeypatch):
    statuses = []
    _run(monkeypatch, metrics=("ssimulacra2",), children=_both(_frames_command(2, _FRAME_BYTES)),
         on_status=statuses.append)
    assert statuses == ["Vship GPU (fake GPU): calculating SSIMULACRA2 (GPU decode: off)…"]


def test_a_truncated_frame_is_an_error_not_a_short_result(monkeypatch):
    children = {"source": [_frames_command(3, _FRAME_BYTES, partial=True)],
                "test": [_frames_command(4, _FRAME_BYTES)]}
    with pytest.raises(vship.VshipUnavailableError, match="partway"):
        _run(monkeypatch, metrics=("ssimulacra2",), children=children)


@pytest.mark.parametrize(("source_frames", "test_frames"), [(4, 6), (6, 4)])
def test_different_frame_counts_compare_the_frames_both_have(monkeypatch, source_frames, test_frames):
    """libvmaf compares the overlap of two inputs of different lengths
    (shortest=1). Vship refused such a pair, which failed the whole job and
    took VMAF with it; it now scores the frames both inputs have."""
    children = {"source": [_frames_command(source_frames, _FRAME_BYTES)],
                "test": [_frames_command(test_frames, _FRAME_BYTES)]}
    output, _spawned = _run(monkeypatch, metrics=("ssimulacra2",), children=children)
    assert output.compared_frame_count == 4
    assert list(output.metrics.get("ssimulacra2").values) == [0.0, 1.0, 2.0, 3.0]


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



def test_only_one_vship_pass_runs_at_a_time(monkeypatch):
    """Two parallel jobs reaching their GPU pass together take turns; the
    second says it is waiting rather than looking stuck."""
    running = 0
    peak = 0
    lock = threading.Lock()

    def pass_(*_args, **_kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.1)
        with lock:
            running -= 1
        return "done"

    monkeypatch.setattr(vship, "_run_vship_pass", pass_)
    statuses = []
    results = []

    def job():
        results.append(vship.run_vship_task(None, None, None, (), None, None, None,
                                            on_status=statuses.append))

    threads = [threading.Thread(target=job) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["done", "done"]
    assert peak == 1
    assert any("Waiting for the GPU" in status for status in statuses)


def test_cancel_while_waiting_for_the_gpu(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    vship._gpu_pass.acquire()
    try:
        with pytest.raises(PerceptualCancelled):
            vship.run_vship_task(None, None, None, (), None, None, None, cancel_event=cancel)
    finally:
        vship._gpu_pass.release()


@pytest.mark.parametrize(("pix_fmt", "color_range", "hwaccel", "expected"), [
    ("yuv420p10le", "tv", "cuda", ("yuv420p16le", "p010le")),
    ("yuv420p10le", "", "d3d11va", ("yuv420p16le", "p010le")),
    ("yuv420p", "tv", "cuda", ("yuv420p", "nv12")),
    ("yuv420p", "pc", "cuda", ("yuv420p", "nv12")),       # 8-bit is exact in either range
    ("yuv420p10le", "pc", "cuda", None),                   # full range: 16-bit would scale differently
    ("yuv420p10le", "tv", None, None),                     # software decode is planar already
    ("yuv420p12le", "tv", "cuda", None),
    ("yuv422p10le", "tv", "cuda", None),
])
def test_which_hardware_decoded_formats_cross_the_pipe_as_is(pix_fmt, color_range, hwaccel, expected):
    info = VideoInfo(Path("x.mkv"), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt=pix_fmt, color_range=color_range)
    result = vship._passthrough_format(info, hwaccel)
    assert (None if result is None else (result[0].pixel_format, result[1])) == expected
    if result is not None and result[1] == "p010le":
        assert result[0].sample == vship._VSHIP_ENUMS[16]


def _interleaved_command(count: int, width: int, height: int, sample_bytes: int) -> list[str]:
    """A child writing NV12/P010-layout frames: luma whose first byte is the
    frame index, then U/V pairs holding 100 + index and 200 + index."""
    script = (
        "import struct, sys\n"
        f"n, w, h, b = {count}, {width}, {height}, {sample_bytes}\n"
        "fmt = '<HH' if b == 2 else 'BB'\n"
        "out = sys.stdout.buffer\n"
        "for i in range(n):\n"
        "    out.write(bytes([i]) + bytes(w * h * b - 1))\n"
        "    out.write(struct.pack(fmt, 100 + i, 200 + i) * ((w // 2) * (h // 2)))\n"
        "out.flush()\n"
    )
    return [sys.executable, "-S", "-c", script]


@pytest.mark.parametrize(("pix_fmt", "sample_type", "sample_bytes", "piped"), [
    ("yuv420p10le", ctypes.c_uint16, 2, "p010le"),
    ("yuv420p", ctypes.c_uint8, 1, "nv12"),
])
def test_interleaved_chroma_is_split_into_the_u_and_v_planes(monkeypatch, pix_fmt, sample_type, sample_bytes, piped):
    """NVDEC's NV12/P010 frame is piped as-is and only its U/V pairs are
    separated in the app. Every chroma sample must land in its own plane, for
    every slot of the ring."""
    chroma_samples = 32 * 24
    checked = []

    def inspect(index, source_planes, test_planes):
        for planes in (source_planes, test_planes):
            u = ctypes.cast(planes[1], ctypes.POINTER(sample_type))
            v = ctypes.cast(planes[2], ctypes.POINTER(sample_type))
            assert [u[0], u[chroma_samples - 1]] == [100 + index] * 2
            assert [v[0], v[chroma_samples - 1]] == [200 + index] * 2
        checked.append(index)

    count = vship._RING_SLOTS * 2 + 1
    info = VideoInfo(Path("source.mkv"), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt=pix_fmt, color_range="tv")
    test = VideoInfo(Path("test.mkv"), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt=pix_fmt, color_range="tv")
    output, spawned = _run(
        monkeypatch, gpu_decode=True, metrics=("ssimulacra2",), source=info, test=test,
        hwaccel=lambda _v, _c: "cuda", inspect=inspect,
        children=_both(_interleaved_command(count, 64, 48, sample_bytes)),
    )

    assert sorted(checked) == list(range(count))
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    command = spawned["source"][0]
    assert command[command.index("-pix_fmt") + 1] == piped
    assert command[command.index("-vf") + 1].endswith(f"format={piped}")


def test_a_scoring_failure_on_the_first_frames_ends_the_run(monkeypatch):
    """A lane that fails keeps the pinned slots of the frames it was given.
    When every lane failed on its first frames the ring filled with slots
    nobody would release: both readers waited for a free slot and the
    dispatcher waited for a frame from them, so the job hung on
    "calculating" instead of failing over to the CPU."""
    def inspect(_index, *_planes):
        raise RuntimeError("simulated GPU fault")

    outcome: list[BaseException] = []

    def run():
        try:
            _run(monkeypatch, inspect=inspect,
                 children=_both(_frames_command(vship._RING_SLOTS * 4, _FRAME_BYTES)))
        except BaseException as error:  # the error is the result
            outcome.append(error)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=20)
    assert not runner.is_alive(), "the run deadlocked after its scoring lanes failed"
    assert isinstance(outcome[0], vship.VshipUnavailableError)
    assert "simulated GPU fault" in str(outcome[0])



def test_a_gpu_failure_on_a_long_video_is_not_retried_on_the_cpu(monkeypatch):
    """Nobody agreed to a CPU run of a film: the fallback would write every
    frame out as PNG and score for days. The metric fails with the reason;
    the worker keeps the video's other metrics."""
    source = VideoInfo(Path("source.mkv"), 3840, 2160, 24.0, 7200.0, 172800, "hevc", pix_fmt="yuv420p10le")
    test = VideoInfo(Path("test.mkv"), 3840, 2160, 24.0, 7200.0, 172800, "hevc", pix_fmt="yuv420p10le")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: (_ for _ in ()).throw(
        vship.VshipUnavailableError("CUDA error: an illegal memory access was encountered")))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: pytest.fail("a long video must not fall back to the CPU"))

    with pytest.raises(perceptual_cpu.PerceptualRunError, match="not retried on the CPU") as raised:
        vship.apply_vship_cpu_fallback(source, test, _request(), _request().metrics)
    assert "illegal memory access" in str(raised.value)
    assert "Choose CPU for SSIMULACRA2 and Butteraugli" in str(raised.value)



# ------------------------------------------------------------------ CVVDP

class _FakeCvvdp:
    """Vship's CVVDP pooling, in Python: a running sum of q^2 since the last
    score reset, reported as JOD -- the first frame after a reset as
    JOD(q * image_int). Each frame's q comes from its index, read out of the
    pinned buffer, so frames scored out of order or mispaired show up."""

    def __init__(self, fail_at=None):
        self.squares, self.count, self.order, self.resets = 0.0, 0, [], []
        self.fail_at = fail_at
        self.freed = False

    @staticmethod
    def quality(index):
        return 0.05 + 0.37 * (index % 7)  # both sides of the 0.1 linear/power switch

    def install(self, monkeypatch):
        monkeypatch.setattr(vship, "_init_cvvdp", lambda *_args: vship._Handle())
        monkeypatch.setattr(vship, "_compute_cvvdp", self.compute)
        monkeypatch.setattr(vship, "_reset_cvvdp_score", lambda *_args: self.reset())
        monkeypatch.setattr(vship, "_free_cvvdp", lambda *_args: setattr(self, "freed", True))
        return self

    def reset(self):
        self.resets.append(self.order[-1] + 1)
        self.squares, self.count = 0.0, 0

    def compute(self, _device, _handler, source_planes, test_planes, *_strides):
        index = source_planes[0][0]
        assert test_planes[0][0] == index
        if self.fail_at is not None and index == self.fail_at:
            raise vship.VshipUnavailableError("Vship CVVDP failed: out of memory")
        self.order.append(index)
        q = self.quality(index)
        self.squares += q * q
        self.count += 1
        if self.count == 1:
            return vship._jod_from_quality(q * vship._IMAGE_INT)
        return vship._jod_from_quality(math.sqrt(self.squares / self.count))


def _whole_video_jod(count):
    """What Vship reports for `count` frames scored without any reset."""
    if count == 1:
        return vship._jod_from_quality(_FakeCvvdp.quality(0) * vship._IMAGE_INT)
    squares = sum(_FakeCvvdp.quality(i) ** 2 for i in range(count))
    return vship._jod_from_quality(math.sqrt(squares / count))


@pytest.mark.parametrize("count", [1, 5, 24, 25, 49, 24 * 3 + 7])
def test_cvvdp_scores_every_frame_in_order_with_a_jod_per_second(monkeypatch, count):
    """One handler sees every frame in order (CVVDP is temporal); its score
    is reset at each second, and the overall JOD pooled from the seconds is
    what an unreset handler reports -- including a last second of one frame
    (49 frames at 24 fps), which Vship pools differently."""
    fake = _FakeCvvdp().install(monkeypatch)
    output, _ = _run(monkeypatch, metrics=("cvvdp",), children=_both(_frames_command(count, _FRAME_BYTES)))

    result = output.metrics.sequence("cvvdp")
    assert fake.order == list(range(count)) and fake.freed
    assert fake.resets == list(range(24, count, 24))
    assert result.score == pytest.approx(_whole_video_jod(count), abs=1e-9)
    assert list(result.frame) == list(range(0, count, 24))
    np.testing.assert_allclose(result.time, np.arange(0, count, 24) / 24.0)
    assert len(result.values) == math.ceil(count / 24)
    assert result.provenance.compute_backend == "gpu"
    assert result.provenance.implementation_compatibility_id == "cvvdp-vship-gpu-v1"
    assert output.failures == {}


def test_every_gpu_score_records_how_the_color_tags_were_read(monkeypatch):
    """The cache tells a score made since the tags follow FFVship 5.1.1 from
    one of v1.2's by this (metric_cache.VSHIP_COLOR_TAGS)."""
    _FakeCvvdp().install(monkeypatch)
    output, _ = _run(monkeypatch, metrics=("ssimulacra2", "butteraugli", "cvvdp"),
                     children=_both(_frames_command(3, _FRAME_BYTES)))
    for key in ("ssimulacra2", "butteraugli", "cvvdp"):
        assert output.metrics.get(key).provenance.parameters["color_tags"] == VSHIP_COLOR_TAGS


def test_each_metric_gets_a_pass_and_a_decode_of_its_own(monkeypatch):
    """One metric at a time keeps the GPU memory to the largest single
    metric's: all three in one pass needed 7.3 GB at 4K."""
    fake = _FakeCvvdp().install(monkeypatch)
    count = vship._RING_SLOTS * 5 + 2
    output, spawned = _run(monkeypatch, metrics=("ssimulacra2", "cvvdp"),
                           children=_both(_frames_command(count, _FRAME_BYTES)))
    assert len(spawned["source"]) == 2 and len(spawned["test"]) == 2
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    assert fake.order == list(range(count))
    assert output.metrics.sequence("cvvdp").score == pytest.approx(_whole_video_jod(count), abs=1e-9)


def test_together_every_metric_shares_one_pass_and_one_decode_with_the_same_scores(monkeypatch):
    """Settings > GPU metrics: decoding 4K VVC on the CPU once per metric
    tripled the decoding; together, each video is decoded once."""
    count = vship._RING_SLOTS * 5 + 2
    metrics = ("ssimulacra2", "butteraugli", "cvvdp")
    _FakeCvvdp().install(monkeypatch)
    apart, spawned_apart = _run(monkeypatch, metrics=metrics, children=_both(_frames_command(count, _FRAME_BYTES)))
    fake = _FakeCvvdp().install(monkeypatch)
    statuses = []
    together, spawned = _run(monkeypatch, metrics=metrics, children=_both(_frames_command(count, _FRAME_BYTES)),
                             together=True, on_status=statuses.append)
    assert len(spawned_apart["source"]) == 3
    assert len(spawned["source"]) == 1 and len(spawned["test"]) == 1
    for key in ("ssimulacra2", "butteraugli"):
        assert list(together.metrics.get(key).values) == list(apart.metrics.get(key).values)
    assert together.metrics.sequence("cvvdp").score == apart.metrics.sequence("cvvdp").score
    assert fake.order == list(range(count)) and together.failures == {}
    assert statuses == ["Vship GPU (fake GPU): calculating SSIMULACRA2, Butteraugli, CVVDP (GPU decode: off)…"]


def test_a_metric_that_fails_in_the_shared_pass_is_calculated_in_a_pass_of_its_own(monkeypatch):
    """Out of GPU memory with all three at once, say: the metric is not
    lost, it is calculated again alone."""
    statuses, progress = [], []
    # Butteraugli runs out of memory beside SSIMULACRA2, and not alone.
    alone = lambda: any("pass of its own" in status for status in statuses)
    count = vship._RING_SLOTS * 3
    output, spawned = _run(monkeypatch, metrics=("ssimulacra2", "butteraugli"), together=True,
                           fail=lambda key, _index: key == "butteraugli" and not alone(),
                           children=_both(_frames_command(count, _FRAME_BYTES)), on_status=statuses.append,
                           on_progress=lambda current, total, _fps: progress.append((current, total)))
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    assert list(output.metrics.get("butteraugli").values) == [i + 0.5 for i in range(count)]
    assert output.failures == {}
    assert len(spawned["source"]) == 2
    # The retry is the second pass of two: the shared one is done.
    assert statuses[1:3] == ["Butteraugli failed in the shared GPU pass; calculating it in a pass of its own…",
                             "GPU metric 2/2: Butteraugli"]
    # One count across both passes: the retry's frames come after the shared
    # pass's, rather than starting again from 0.
    currents = [current for current, _total in progress]
    assert currents == sorted(currents) and progress[-1] == (2 * count, 2 * count)
    retry = currents.index(count + 1)
    assert progress[retry - 1] == (count, count) and progress[retry][1] > count


def test_a_metric_that_fails_on_its_own_too_is_reported_with_the_others_kept(monkeypatch):
    count = vship._RING_SLOTS * 3
    output, spawned = _run(monkeypatch, metrics=("ssimulacra2", "butteraugli"), fail=("butteraugli", 2),
                           together=True, children=_both(_frames_command(count, _FRAME_BYTES)))
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    assert not output.metrics.has("butteraugli") and "out of memory" in output.failures["butteraugli"]
    assert len(spawned["source"]) == 2


def test_with_frame_subsampling_cvvdp_keeps_a_pass_of_its_own():
    """CVVDP scores every frame; SSIMULACRA2 and Butteraugli then score
    every n-th, so they cannot share its pass."""
    specs = analysis_request_from_vmaf_options(
        VmafOptions(n_subsample=3), ("ssimulacra2", "butteraugli", "cvvdp")).metrics
    keys = lambda passes: [tuple(spec.key for spec in group) for group in passes]
    assert keys(vship.vship_passes(specs)) == [("ssimulacra2",), ("butteraugli",), ("cvvdp",)]
    assert keys(vship.vship_passes(specs, together=True)) == [("ssimulacra2", "butteraugli"), ("cvvdp",)]
    full = analysis_request_from_vmaf_options(VmafOptions(), ("ssimulacra2", "butteraugli", "cvvdp")).metrics
    assert keys(vship.vship_passes(full, together=True)) == [("ssimulacra2", "butteraugli", "cvvdp")]


def test_a_cvvdp_failure_keeps_ssimulacra2_and_frees_the_ring(monkeypatch):
    """CVVDP (GPU only, and the largest VRAM user) failing part-way must not
    take SSIMULACRA2 down with it, nor hold ring slots so the pass hangs."""
    _FakeCvvdp(fail_at=3).install(monkeypatch)
    count = vship._RING_SLOTS * 6
    outcome = []
    runner = threading.Thread(target=lambda: outcome.append(_run(
        monkeypatch, metrics=("ssimulacra2", "cvvdp"),
        children=_both(_frames_command(count, _FRAME_BYTES)))[0]), daemon=True)
    runner.start()
    runner.join(timeout=20)
    assert not runner.is_alive(), "the pass hung after CVVDP failed"
    output = outcome[0]
    assert list(output.metrics.get("ssimulacra2").values) == [float(i) for i in range(count)]
    assert not output.metrics.has("cvvdp")
    assert "out of memory" in output.failures["cvvdp"]


def test_cvvdp_alone_failing_fails_the_pass(monkeypatch):
    _FakeCvvdp(fail_at=0).install(monkeypatch)
    with pytest.raises(vship.VshipUnavailableError, match="out of memory"):
        _run(monkeypatch, metrics=("cvvdp",), children=_both(_frames_command(10, _FRAME_BYTES)))


def test_pooling_one_window_gives_back_its_own_jod():
    for jod in (9.99, 9.5, 7.25, 3.0):
        assert vship.pool_cvvdp_windows([(0, 24, jod)]) == pytest.approx(jod, abs=1e-9)



def test_pooling_a_final_one_frame_second_matches_vship():
    """Measured with the real Vship 5.1.1 on the first 49 frames (23.976 fps)
    of the Beekeeper AV1 encode: one handler never reset gave 9.9112921; the
    per-second JODs below (rounded to 3 decimals) are what the reset handler
    gave. Vship reports a single frame as JOD(q * IMAGE_INT), so the last
    second must be pooled differently -- pooled like the others it gives 9.921."""
    windows = [(0, 24, 10.0), (24, 24, 9.898), (48, 1, 9.802)]
    assert vship.pool_cvvdp_windows(windows) == pytest.approx(9.9112921, abs=5e-4)

def test_subsampled_ssimulacra2_and_cvvdp_get_a_pass_each(monkeypatch):
    request = analysis_request_from_vmaf_options(
        VmafOptions(crop_mode=CropMode.NONE, n_subsample=3), ("ssimulacra2", "cvvdp"))
    passes, progress = [], []

    def pass_(_s, _t, _r, specs, *_a, on_progress=None, **_k):
        passes.append([(spec.key, spec.coverage.step) for spec in specs])
        on_progress(10, 10, 1.0)
        key = specs[0].key
        provenance = MetricProvenance("t", "1", "gpu", "t")
        metric = (vship.SequenceMetricResult(key, 9.0, provenance) if key == "cvvdp"
                  else FrameMetricResult(key, [0], [0.0], [80.0], provenance))
        return PerceptualTaskOutput(MetricResultSet([metric]), None, None, 10)

    monkeypatch.setattr(vship, "_run_vship_pass", pass_)
    output = vship.run_vship_task(_hevc("s.mkv"), _hevc("t.mkv"), request, request.metrics,
                                  _fake_device(), None, None,
                                  on_progress=lambda *args: progress.append(args))
    assert passes == [[("ssimulacra2", 3)], [("cvvdp", 1)]]
    assert output.metrics.keys() == ("ssimulacra2", "cvvdp")
    assert progress == [(10, 20, 1.0), (20, 20, 1.0)]


def _cvvdp_request(*keys, backends=None):
    return analysis_request_from_vmaf_options(VmafOptions(crop_mode=CropMode.NONE), keys, backends)


def test_without_a_gpu_cvvdp_fails_and_ssimulacra2_runs_on_the_cpu(monkeypatch):
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    cpu_keys = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (None, "no supported GPU"))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: cpu_keys.extend(s.key for s in a[3]) or _single_metric_output(
                            "ssimulacra2", 80.0, "cpu"))
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert cpu_keys == ["ssimulacra2"]
    assert output.metrics.has("ssimulacra2")
    assert "no supported GPU" in output.failures["cvvdp"]


def test_without_a_gpu_cvvdp_alone_is_an_error(monkeypatch):
    request = _cvvdp_request("cvvdp")
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (None, "no supported GPU"))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", lambda *a, **k: pytest.fail("no CPU CVVDP"))
    with pytest.raises(perceptual_cpu.PerceptualRunError, match="CVVDP needs a GPU that Vship can use"):
        vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)


def test_cvvdp_has_no_backend_choice():
    with pytest.raises(ValueError, match="cvvdp"):
        _cvvdp_request("cvvdp", backends={"cvvdp": "cpu"})


def test_a_gpu_failure_retries_the_others_on_the_cpu_and_reports_cvvdp(monkeypatch):
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    cpu_keys = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: (_ for _ in ()).throw(
        vship.VshipUnavailableError("CUDA error: out of memory")))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: cpu_keys.extend(s.key for s in a[3]) or _single_metric_output(
                            "ssimulacra2", 80.0, "cpu"))
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert cpu_keys == ["ssimulacra2"]
    assert "out of memory" in output.failures["cvvdp"]


def test_cvvdp_on_the_gpu_beside_butteraugli_on_the_cpu(monkeypatch):
    request = _cvvdp_request("butteraugli", "cvvdp", backends={"butteraugli": "cpu"})
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    cvvdp = vship.SequenceMetricResult("cvvdp", 9.1, MetricProvenance("t", "1", "gpu", "t"))
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: PerceptualTaskOutput(
        MetricResultSet([cvvdp]), None, None, 1))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: _single_metric_output("butteraugli", 0.3, "cpu"))
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert output.metrics.keys() == ("butteraugli", "cvvdp")


def test_cvvdp_alone_failing_to_start_stops_the_pass_at_once(monkeypatch):
    """With nothing else to score, a CVVDP handler that failed to start used
    to be reported only after the whole video had been decoded."""
    def cannot_start(*_args):
        raise vship.VshipUnavailableError("Could not initialize Vship CVVDP: out of memory")

    _FakeCvvdp().install(monkeypatch)
    monkeypatch.setattr(vship, "_init_cvvdp", cannot_start)
    slow = [sys.executable, "-S", "-c",
            f"import sys, time\nfor i in range(600):\n    sys.stdout.buffer.write(bytes({_FRAME_BYTES})); "
            "sys.stdout.buffer.flush(); time.sleep(0.01)\n"]
    started = time.monotonic()
    with pytest.raises(vship.VshipUnavailableError, match="out of memory"):
        _run(monkeypatch, metrics=("cvvdp",), children=_both(slow))
    assert time.monotonic() - started < 4, "the pass decoded the rest of the video first"


def test_a_ssimulacra2_failure_keeps_butteraugli_and_cvvdp(monkeypatch):
    """A SSIMULACRA2 or Butteraugli lane failing (out of VRAM on a smaller
    card, say) ended the whole pass: CVVDP was lost and reported as
    needing a GPU, and Butteraugli had to be redone. Now only SSIMULACRA2
    is reported failed; the rest of the pass finishes, without hanging."""
    fake = _FakeCvvdp().install(monkeypatch)
    count = vship._RING_SLOTS * 6
    outcome = []
    runner = threading.Thread(target=lambda: outcome.append(_run(
        monkeypatch, metrics=("ssimulacra2", "butteraugli", "cvvdp"), fail=("ssimulacra2", 3),
        children=_both(_frames_command(count, _FRAME_BYTES)))[0]), daemon=True)
    runner.start()
    runner.join(timeout=20)
    assert not runner.is_alive(), "the pass hung after a lane failed"
    output = outcome[0]
    assert not output.metrics.has("ssimulacra2")
    assert "out of memory" in output.failures["ssimulacra2"]
    assert list(output.metrics.get("butteraugli").values) == [i + 0.5 for i in range(count)]
    assert output.metrics.sequence("cvvdp").score == pytest.approx(_whole_video_jod(count), abs=1e-9)
    assert fake.order == list(range(count))


def test_every_metric_failing_still_fails_the_pass(monkeypatch):
    _FakeCvvdp(fail_at=2).install(monkeypatch)
    with pytest.raises(vship.VshipUnavailableError):
        _run(monkeypatch, metrics=("ssimulacra2", "cvvdp"), fail=("ssimulacra2", 1),
             children=_both(_frames_command(40, _FRAME_BYTES)))


@pytest.mark.parametrize("long_video", [False, True])
def test_a_metric_that_failed_on_the_gpu_is_retried_on_the_cpu_only_for_short_videos(monkeypatch, long_video):
    seconds = 7200.0 if long_video else 60.0
    source = VideoInfo(Path("source.mkv"), 64, 48, 24.0, seconds, int(seconds * 24), "h264", pix_fmt="yuv420p")
    test = VideoInfo(Path("test.mkv"), 64, 48, 24.0, seconds, int(seconds * 24), "h264", pix_fmt="yuv420p")
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    cvvdp = vship.SequenceMetricResult("cvvdp", 9.4, MetricProvenance("t", "1", "gpu", "t"))
    cpu_keys = []
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: PerceptualTaskOutput(
        MetricResultSet([cvvdp]), None, None, 1, {"ssimulacra2": "Vship SSIMULACRA2 failed: out of memory"}))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: cpu_keys.extend(s.key for s in a[3]) or _single_metric_output(
                            "ssimulacra2", 80.0, "cpu"))
    output = vship.apply_vship_cpu_fallback(source, test, request, request.metrics)
    assert output.metrics.has("cvvdp")
    if long_video:
        assert cpu_keys == [] and "not retried on the CPU" in output.failures["ssimulacra2"]
        assert not output.metrics.has("ssimulacra2")
    else:
        assert cpu_keys == ["ssimulacra2"] and output.failures == {}
        assert output.metrics.get("ssimulacra2").provenance.compute_backend == "cpu"


@pytest.mark.parametrize("cpu_problem", ["tool missing", "frame count"])
def test_a_cpu_side_failure_keeps_the_finished_gpu_scores(monkeypatch, cpu_problem):
    """CVVDP on the GPU beside Butteraugli set to CPU: a missing libjxl tool
    (or differing frame counts) after the GPU pass raised out of the
    fallback and threw away the finished CVVDP score."""
    request = _cvvdp_request("butteraugli", "cvvdp", backends={"butteraugli": "cpu"})
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    cvvdp = vship.SequenceMetricResult("cvvdp", 9.1, MetricProvenance("t", "1", "gpu", "t"))
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: PerceptualTaskOutput(
        MetricResultSet([cvvdp]), None, None, 1))

    def cpu(*_args, **_kwargs):
        if cpu_problem == "tool missing":
            raise perceptual_cpu.PerceptualRunError("butteraugli_main is not installed")
        output = _single_metric_output("butteraugli", 0.3, "cpu")
        return PerceptualTaskOutput(output.metrics, None, None, 2)

    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", cpu)
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert output.metrics.keys() == ("cvvdp",)
    assert ("not installed" if cpu_problem == "tool missing" else "frames on the CPU") in output.failures["butteraugli"]


def test_cvvdp_failure_messages_name_the_real_cause(monkeypatch):
    """A problem with the videos was reported as CVVDP needing a GPU, and on
    a long video CVVDP's failure was dropped from the message altogether."""
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", lambda *a, **k: (_ for _ in ()).throw(
        perceptual_cpu.PerceptualRunError("Frame rates do not match: 24 vs 25 fps")))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: _single_metric_output("ssimulacra2", 80.0, "cpu"))
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert output.failures["cvvdp"] == "CVVDP could not be calculated: Frame rates do not match: 24 vs 25 fps"

    film = VideoInfo(Path("film.mkv"), 64, 48, 24.0, 7200.0, 172800, "h264", pix_fmt="yuv420p")
    with pytest.raises(perceptual_cpu.PerceptualRunError) as raised:
        vship.apply_vship_cpu_fallback(film, film, request, request.metrics)
    assert "not retried on the CPU" in str(raised.value)
    assert "CVVDP could not be calculated: Frame rates do not match" in str(raised.value)


@pytest.mark.parametrize("fps", [0.0, float("nan")])
def test_cvvdp_refuses_a_video_without_a_frame_rate(fps):
    """A 0 fps video was scored as if it ran at 1 fps."""
    device = vship.VshipDevice("cuda","GPU", 0, "5.1.1", SimpleNamespace(library=SimpleNamespace()))
    with pytest.raises(vship.VshipUnavailableError, match="frame rate"):
        vship._init_cvvdp(device, None, None, None, fps)


@pytest.mark.parametrize("long_video", [False, True])
def test_when_every_gpu_pass_fails_cvvdp_keeps_its_own_reason(monkeypatch, long_video):
    """With one pass per metric, all failing raised only the first error:
    CVVDP was reported with SSIMULACRA2's out-of-memory, and its own reason
    appeared nowhere."""
    seconds = 7200.0 if long_video else 60.0
    source = VideoInfo(Path("s.mkv"), 64, 48, 24.0, seconds, int(seconds * 24), "h264", pix_fmt="yuv420p")
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    reasons = {"ssimulacra2": "Could not initialize Vship ssimulacra2: out of memory",
               "cvvdp": "Could not initialize Vship CVVDP: out of VRAM"}

    def one_pass(_s, _t, _r, specs, *_a, **_k):
        raise vship.VshipUnavailableError(reasons[specs[0].key])

    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "_run_vship_pass", one_pass)
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *a, **k: _single_metric_output("ssimulacra2", 80.0, "cpu"))
    if long_video:
        with pytest.raises(perceptual_cpu.PerceptualRunError) as raised:
            vship.apply_vship_cpu_fallback(source, source, request, request.metrics)
        assert "CVVDP could not be calculated: Could not initialize Vship CVVDP: out of VRAM" in str(raised.value)
    else:
        output = vship.apply_vship_cpu_fallback(source, source, request, request.metrics)
        assert output.failures["cvvdp"] == "CVVDP could not be calculated: Could not initialize Vship CVVDP: out of VRAM"
        assert output.metrics.has("ssimulacra2")


def _gpu_failed_ssimulacra2(monkeypatch, cpu):
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    cvvdp = vship.SequenceMetricResult("cvvdp", 9.4, MetricProvenance("t", "1", "gpu", "t"))

    def gpu(*_a, on_progress=None, **_k):
        for done in (5, 50, 100):
            if on_progress is not None:
                on_progress(done, 100, 10.0)
        return PerceptualTaskOutput(MetricResultSet([cvvdp]), None, None, 100,
                                    {"ssimulacra2": "Vship SSIMULACRA2 failed: out of memory"})

    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *args: (None, None))
    monkeypatch.setattr(vship, "run_vship_task", gpu)
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", cpu)
    return request


def test_a_cpu_retry_after_a_gpu_failure_is_announced_and_its_progress_runs_0_to_100(monkeypatch):
    """The retry was mapped onto the second half of a progress bar the GPU
    had already filled: 100% fell back to 55%."""
    def cpu(*_a, on_progress=None, **_k):
        for done in (10, 60, 100):
            on_progress(done, 100, 2.0)
        output = _single_metric_output("ssimulacra2", 80.0, "cpu")
        return PerceptualTaskOutput(output.metrics, None, None, 100)

    request = _gpu_failed_ssimulacra2(monkeypatch, cpu)
    progress, statuses = [], []
    output = vship.apply_vship_cpu_fallback(
        _info("s.mkv"), _info("t.mkv"), request, request.metrics,
        on_progress=lambda cur, total, fps: progress.append(round(100 * cur / total)),
        on_status=statuses.append)
    assert progress == [5, 50, 100, 10, 60, 100]
    assert "SSIMULACRA2 failed on the GPU; calculating it on the CPU" in statuses[-1]
    assert output.metrics.has("ssimulacra2") and output.failures == {}


def test_a_cpu_retry_that_fails_too_keeps_the_gpu_reason(monkeypatch):
    def cpu(*_a, **_k):
        raise perceptual_cpu.PerceptualRunError("ssimulacra2 is not installed.")

    request = _gpu_failed_ssimulacra2(monkeypatch, cpu)
    output = vship.apply_vship_cpu_fallback(_info("s.mkv"), _info("t.mkv"), request, request.metrics)
    assert output.metrics.has("cvvdp")
    assert output.failures["ssimulacra2"] == (
        "GPU scoring failed (Vship SSIMULACRA2 failed: out of memory); "
        "the CPU retry failed too: ssimulacra2 is not installed.")


def test_pinned_memory_is_freed_when_allocation_fails_part_way(monkeypatch):
    """A failed allocation left the buffers already allocated -- in its own
    ring, and the reference's whole ring when the test video's failed --
    pinned for the rest of the session."""
    allocated, freed = [], []

    class _Pinned(_FakePinned):
        def __init__(self, lib, size, gpu_id=0):
            if len(allocated) == vship._RING_SLOTS + 2:  # the test video's 3rd buffer
                raise vship.VshipUnavailableError("Could not allocate Vship pinned frame memory")
            super().__init__(lib, size, gpu_id)
            allocated.append(self)

        def close(self):
            freed.append(self)

    monkeypatch.setattr(vship, "_PinnedBuffer", _Pinned)
    monkeypatch.setattr(vship, "_init_handler", lambda *_args: vship._Handle())
    request = analysis_request_from_vmaf_options(VmafOptions(crop_mode=CropMode.NONE), ("ssimulacra2",))
    with pytest.raises(vship.VshipUnavailableError, match="pinned frame memory"):
        vship.run_vship_task(_hevc("source.mkv"), _hevc("test.mkv"), request, request.metrics,
                             _fake_device(), None, None)
    assert len(allocated) == vship._RING_SLOTS + 2
    assert sorted(map(id, freed)) == sorted(map(id, allocated)), "pinned buffers were left allocated"


def test_cvvdp_gets_its_display_as_json_text_with_the_gpu_in_the_init_struct():
    """Vship 5.1 parses a display config starting with "{" itself. A file
    path failed under a Windows user name outside the ANSI code page, and
    the file had to be written, found and deleted around every init."""
    import json

    from vmaf_app.core.cvvdp import VSHIP_MODEL_KEY, CvvdpSettings

    seen = []

    def init(handle, argument):
        struct = ctypes.cast(argument, ctypes.POINTER(vship._InitCvvdp)).contents
        seen.append((struct.structType, struct.gpu_id, struct.fps, struct.resizeToDisplay,
                     struct.model_key_cstr, struct.model_config_json_cstr))
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 1234
        return 0

    device = vship.VshipDevice("vulkan", "GPU", 2, "5.1.1",
                               SimpleNamespace(library=SimpleNamespace(Vship_InitHandler=init)))
    handle = vship._init_cvvdp(device, vship._Colorspace(), vship._Colorspace(), CvvdpSettings(), 23.976)
    assert handle.value == 1234
    [(struct_type, gpu_id, fps, resize, key, config)] = seen
    assert (struct_type, gpu_id, resize, key) == (vship._INIT_CVVDP, 2, False, VSHIP_MODEL_KEY.encode())
    assert fps == pytest.approx(23.976)
    assert config.startswith(b"{") and VSHIP_MODEL_KEY in json.loads(config)


def _counting_probe(monkeypatch, *results, delay=0.0):
    calls = []

    def probe():
        calls.append(threading.current_thread().name)
        time.sleep(delay)
        return results[min(len(calls), len(results)) - 1]

    monkeypatch.setattr(vship, "_probed", None)
    monkeypatch.setattr(vship, "_probe_vship_device", probe)
    return calls


def test_the_gpu_probe_runs_in_the_background_once(monkeypatch):
    """The probe ran on the UI thread when the first video was added; a
    caller arriving while the background probe runs waits for it."""
    calls = _counting_probe(monkeypatch, (None, "no GPU"), delay=0.2)
    vship.start_vship_probe()
    time.sleep(0.05)
    assert vship.detect_vship_device() == (None, "no GPU")
    assert vship.detect_vship_device() == (None, "no GPU")
    assert calls == ["vship-probe"]


def test_a_failed_gpu_probe_is_redone_after_a_while_but_a_found_gpu_is_kept(monkeypatch):
    """A failure was cached for the whole session."""
    device = _fake_device()
    calls = _counting_probe(monkeypatch, (None, "driver restarting"), (device, ""))
    assert vship.detect_vship_device()[0] is None
    vship.forget_failed_vship_probe()  # too soon: kept
    assert vship.detect_vship_device()[0] is None and len(calls) == 1
    (result, when) = vship._probed
    monkeypatch.setattr(vship, "_probed", (result, when - vship.FAILED_PROBE_RETRY_SECONDS))
    vship.forget_failed_vship_probe()
    assert vship.detect_vship_device()[0] is device and len(calls) == 2
    monkeypatch.setattr(vship, "_probed", (vship._probed[0], 0.0))
    vship.forget_failed_vship_probe()
    assert vship.detect_vship_device()[0] is device and len(calls) == 2


def test_a_passs_rate_is_timed_from_its_first_frame_not_its_start():
    """The pass's start-up was counted in its rate: "1.7 fps, 0:05:39 left"
    at the start of a pass over an 8-second clip."""
    clock = iter([10.0, 10.2, 10.6, 11.0])
    rate = vship._PassRate(clock=lambda: next(clock))
    assert rate.frames_per_second(1) == 0.0  # the first frame pair: nothing to time yet
    assert rate.frames_per_second(5) == 0.0  # 0.2 s after it: too soon to say
    assert rate.frames_per_second(13) == pytest.approx(20.0)  # 12 more frames in 0.6 s
    assert rate.frames_per_second(21) == pytest.approx(20.0)  # 20 in 1.0 s


def test_a_sampled_passs_rate_is_in_the_videos_frames_like_its_progress():
    """Every 5th frame compared: the progress counts the video's frames, and
    the rate counted pairs, so the time left was 5 times too long."""
    clock = iter([10.0, 11.0])
    rate = vship._PassRate(5, clock=lambda: next(clock))
    rate.frames_per_second(1)
    assert rate.frames_per_second(21) == pytest.approx(100.0)  # 20 pairs = 100 frames in 1 s


def test_a_failed_gpu_pass_is_logged(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="vmaf_app")
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)

    def one_pass(_s, _t, _r, specs, *_a, **_k):
        raise vship.VshipUnavailableError(f"Could not initialize Vship {specs[0].key}: out of memory")

    monkeypatch.setattr(vship, "_run_vship_pass", one_pass)
    source = VideoInfo(Path("s.mkv"), 64, 48, 24.0, 60.0, 1440, "h264", pix_fmt="yuv420p")
    with pytest.raises(vship.VshipUnavailableError):
        vship.run_vship_task(source, source, request, request.metrics, device, None, None)
    assert "GPU metric 1/2 (SSIMULACRA2) failed: Could not initialize Vship ssimulacra2: out of memory" in caplog.text
    assert "GPU metric 2/2 (CVVDP) failed: Could not initialize Vship cvvdp: out of memory" in caplog.text


def test_each_gpu_metrics_pass_is_handed_on_as_it_finishes(monkeypatch):
    """The pass loop had each metric's scores the moment its pass ended, and
    kept them until every pass was done."""
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    device = vship.VshipDevice("cuda","test GPU", 0, "5.1.1", None)
    done = []

    def one_pass(_s, _t, _r, specs, *_a, **_k):
        assert [output.metrics.keys() for output in done] == [("ssimulacra2",)] * (specs[0].key == "cvvdp")
        return _single_metric_output(specs[0].key, 80.0, "gpu")

    monkeypatch.setattr(vship, "_run_vship_pass", one_pass)
    source = VideoInfo(Path("s.mkv"), 64, 48, 24.0, 60.0, 1440, "h264", pix_fmt="yuv420p")
    vship.run_vship_task(source, source, request, request.metrics, device, None, None, on_pass_done=done.append)
    assert [output.metrics.keys() for output in done] == [("ssimulacra2",), ("cvvdp",)]



# ------------------------------------------------------------------ backends

def test_auto_tries_cuda_then_hip_then_vulkan_and_a_choice_goes_first():
    """Auto: the fastest build whose scores agree with the reference on each
    GPU; a chosen build is tried first, and the others still follow if it
    cannot run here."""
    assert vship.DEFAULT_VSHIP_BACKEND == "auto"
    assert vship._probe_order("auto") == ("cuda", "hip", "vulkan")
    assert vship._probe_order("vulkan") == ("vulkan", "cuda", "hip")
    assert vship._probe_order("hip") == ("hip", "cuda", "vulkan")


@pytest.mark.skipif(sys.platform != "win32", reason="Vship is only bundled for Windows")
def test_vships_vulkan_build_is_not_loaded_where_vulkan_has_no_gpu(monkeypatch):
    """Where the Vulkan loader is installed but no driver answers, loading
    Vship's Vulkan build fails (WinError 1114) and the process then crashes
    as it exits: every test run on GitHub's runner ended in exit code 1."""
    loaded = []

    def cdll(path):
        loaded.append(Path(path).parent.name)
        raise OSError("not loaded in this test")

    monkeypatch.setattr(vship, "_backend", "vulkan")
    monkeypatch.setattr(vship, "_vulkan_unavailable", lambda: "no GPU with a Vulkan driver was found")
    monkeypatch.setattr(vship.ctypes, "CDLL", cdll)
    device, reason = vship._probe_vship_device()
    assert device is None and "vulkan" not in loaded and loaded == ["nvidia", "amd"]
    assert reason.startswith("Vulkan: no GPU with a Vulkan driver was found; ")


def test_choosing_another_backend_forgets_the_probe(monkeypatch):
    monkeypatch.setattr(vship, "_backend", "auto")
    monkeypatch.setattr(vship, "_probed", ((None, "no GPU"), 0.0))
    vship.set_vship_backend("auto")
    assert vship._probed is not None  # the same choice: kept
    vship.set_vship_backend("vulkan")
    assert vship._probed is None and vship.vship_backend() == "vulkan"
    vship.set_vship_backend("no such build")
    assert vship.vship_backend() == "auto"


def test_vulkan_ssimulacra2_is_not_trusted_but_its_other_metrics_are():
    """Vship 5.1.1's Vulkan SSIMULACRA2 read 62.9 where CUDA read 45.5 and
    libjxl 47.4 on the same 4K frames; its Butteraugli and CVVDP agree."""
    vulkan = vship.VshipDevice("vulkan", "GPU", 0, "5.1.1", None)
    cuda = vship.VshipDevice("cuda", "GPU", 0, "5.1.1", None)
    assert not vship.scores_correctly(vulkan, "ssimulacra2")
    assert vship.scores_correctly(vulkan, "butteraugli") and vship.scores_correctly(vulkan, "cvvdp")
    assert vship.scores_correctly(cuda, "ssimulacra2")


def test_on_vulkan_ssimulacra2_is_calculated_on_the_cpu_and_butteraugli_on_the_gpu(monkeypatch):
    device = vship.VshipDevice("vulkan", "Intel Arc", 0, "5.1.1", None)
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(vship, "forget_failed_vship_probe", lambda: None)
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *_args: (None, None))
    on_gpu, on_cpu = [], []

    def gpu(_s, _d, _request, specs, used, *_crops, **_kwargs):
        on_gpu.append((used.backend, [spec.key for spec in specs]))
        return _single_metric_output("butteraugli", 1.5, "gpu")

    def cpu(_s, _d, _request, specs, **_kwargs):
        on_cpu.append([spec.key for spec in specs])
        return _single_metric_output("ssimulacra2", 80.0, "cpu")

    monkeypatch.setattr(vship, "run_vship_task", gpu)
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task", cpu)
    request = _request()  # SSIMULACRA2 and Butteraugli, both set to GPU
    output = vship.apply_vship_cpu_fallback(_info("a.mkv"), _info("b.mkv"), request, request.metrics)
    assert on_gpu == [("vulkan", ["butteraugli"])] and on_cpu == [["ssimulacra2"]]
    assert output.metrics.get("ssimulacra2").provenance.compute_backend == "cpu"
    assert not output.failures
