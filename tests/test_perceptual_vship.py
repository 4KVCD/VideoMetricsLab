from __future__ import annotations

import ctypes
import faulthandler
import logging
import math
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.factories import STDLIB_PYTHON
from vmaf_app.core import gpu, gpu_frames, perceptual_cpu
from vmaf_app.core import perceptual_vship as vship
from vmaf_app.core.analysis_request import AnalysisRequest
from vmaf_app.core.ffmpeg_locate import ffmpeg_path, ffprobe_path
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import CropMode, GpuVendor, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualTaskOutput


@pytest.fixture(autouse=True)
def _fakes_are_not_cut_short(monkeypatch):
    """The fakes here produce a few frames for videos whose lengths promise
    more: frame_coverage would rightly call them cut short. Tested in
    test_frame_coverage.py."""
    monkeypatch.setattr(vship, "short_comparison", lambda *a, **k: None)


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


def test_vship_51_structs_match_the_c_layout(subtests):
    """The init and score structs of VshipAPI.h (Vship 5.1.1), as the x64 C
    compiler lays them out: a wrong offset would hand Vship garbage."""
    def check(struct, size, offsets):
        assert ctypes.sizeof(struct) == size
        assert {name: getattr(struct, name).offset for name in offsets} == offsets

    for struct, size, offsets in [
        # Vship_Colorspace_t: four int64, seven 4-byte fields, the 16-byte crop.
        (vship._Colorspace, 88, {"sample": 32, "subsampling": 40, "YUVMatrix": 56, "crop": 68}),
        (vship._InitSsimulacra2, 192, {"structType": 0, "src_colorspace": 8, "dis_colorspace": 96, "gpu_id": 184}),
        (vship._InitButteraugli, 200, {"dis_colorspace": 96, "Qnorm": 184, "intensity_multiplier": 188, "gpu_id": 192}),
        (vship._InitCvvdp, 216, {"fps": 184, "resizeToDisplay": 188, "model_key_cstr": 192,
                                 "model_config_json_cstr": 200, "gpu_id": 208}),
        (vship._ScoreSsimulacra2, 16, {"structType": 0, "score": 8}),
        (vship._ScoreButteraugli, 48, {"normQ": 8, "norm3": 16, "norminf": 24, "dstp": 32, "dststride": 40}),
        (vship._ScoreCvvdp, 32, {"score": 8, "dstp": 16, "dststride": 24}),
    ]:
        with subtests.test(struct=struct, size=size, offsets=offsets):
            check(struct, size, offsets)


def test_every_ffmpeg_layout_reaches_vship_as_a_planar_one(subtests):
    def check(pixel_format, piped, sample, shifts, rgb):
        image = vship._image_format(_info("video.mkv", pix_fmt=pixel_format), "5.1.2")
        assert image.pixel_format == piped
        assert image.sample == vship._VSHIP_ENUMS[sample]
        assert (image.subw, image.subh) == shifts and (image.family == 1) == rgb

    for pixel_format, piped, sample, shifts, rgb in [
        # Any subsampling at any depth: Vship takes two shifts and a sample
        # type. The "Vship supports these layouts" table allowed 4:4:0 only up
        # to 12 bits and 4:1:0 / 4:1:1 only at 8 -- FFmpeg's formats, not Vship's.
        ("yuv440p12le", "yuv440p12le", 12, (0, 1), False),
        ("yuv410p", "yuv410p", 8, (2, 2), False),
        ("yuv444p14le", "yuv444p14le", 14, (0, 0), False),
        ("yuv422p16be", "yuv422p16le", 16, (1, 0), False),
        ("yuva420p10le", "yuv420p10le", 10, (1, 1), False),  # alpha dropped
        ("gray10le", "yuv420p10le", 10, (1, 1), False),  # monochrome: neutral chroma
        ("p012le", "yuv420p12le", 12, (1, 1), False),
        ("nv24", "yuv444p", 8, (0, 0), False),
        ("gbrap12le", "gbrp12le", 12, (0, 0), True),
        ("bgr48le", "gbrp16le", 16, (0, 0), True),
    ]:
        with subtests.test(pixel_format=pixel_format, piped=piped, sample=sample, shifts=shifts, rgb=rgb):
            check(pixel_format, piped, sample, shifts, rgb)


def _colorspace(width=1920, height=1080, pix_fmt="yuv420p10le", **tags):
    info = VideoInfo(Path("video.mkv"), width, height, 24.0, 1.0, 24, "hevc", pix_fmt=pix_fmt, **tags)
    return vship._vship_colorspace(info, vship._image_format(info), width, height)


def test_color_tags_map_to_vships_values(subtests):
    def check(tags, matrix, transfer, primaries):
        color = _colorspace(**tags)
        assert (color.YUVMatrix, color.transferFunction, color.primaries) == (matrix, transfer, primaries)

    for tags, matrix, transfer, primaries in [
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
    ]:
        with subtests.test(tags=tags, matrix=matrix, transfer=transfer, primaries=primaries):
            check(tags, matrix, transfer, primaries)


def test_untagged_video_is_guessed_as_ffvship_guesses_it():
    sd = _colorspace(width=720, height=576)
    assert (sd.YUVMatrix, sd.transferFunction, sd.primaries, sd.range) == (5, 5, 5, 0)
    rgb = _colorspace(pix_fmt="gbrp")  # sRGB, full range
    assert (rgb.YUVMatrix, rgb.transferFunction, rgb.primaries, rgb.range) == (0, 13, 1, 1)
    assert _colorspace(pix_fmt="yuvj420p").range == 1


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
        on_cpu=lambda keys: progress.append("cpu: " + ", ".join(keys)),
    )

    assert routed == {"gpu": ["ssimulacra2"], "cpu": ["butteraugli"]}
    assert actual.metrics.keys() == ("ssimulacra2", "butteraugli")
    assert actual.metrics.get("ssimulacra2").provenance.compute_backend == "gpu"
    assert actual.metrics.get("butteraugli").provenance.compute_backend == "cpu"
    # Each stage's own figures: the CPU's start from 0 once it is told which
    # metrics it takes (on_cpu), and the run line shows them as CPU metrics.
    assert progress == [(1, 1, 10.0), "cpu: butteraugli", (1, 1, 8.0)]


def _crashing_pass(*_args, **_kwargs):
    faulthandler._sigsegv()  # an access violation, as in Vship or the GPU driver


def test_a_crash_in_vship_ends_its_own_process_and_the_cpu_takes_over(monkeypatch, caplog):
    """A crash in Vship, or in the GPU driver under it, ended the app with
    every video's progress, with no message."""
    device = vship.VshipDevice("vulkan", "GPU", 0, "5.1.2", None, GpuVendor.NVIDIA)  # no library: isolated
    monkeypatch.setattr(vship, "detect_vship_device", lambda: (device, ""))
    monkeypatch.setattr(vship, "forget_failed_vship_probe", lambda: None)
    monkeypatch.setattr(vship, "_score_vship_pass", _crashing_pass)
    monkeypatch.setattr(perceptual_cpu, "_resolve_crops", lambda *_args: (None, None))
    monkeypatch.setattr(perceptual_cpu, "run_perceptual_task",
                        lambda *_a, **_k: _single_metric_output("ssimulacra2", 80.0, "cpu"))
    request = analysis_request_from_vmaf_options(VmafOptions(crop_mode=CropMode.NONE), ("ssimulacra2",))
    with caplog.at_level(logging.ERROR):
        output = vship.apply_vship_cpu_fallback(_info("a.mkv"), _info("b.mkv"), request, request.metrics)
    assert output.metrics.get("ssimulacra2").provenance.compute_backend == "cpu"
    assert "Vship crashed" in caplog.text


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

def _frames_command(count: int, frame_bytes: int, *, exit_code: int = 0, partial: bool = False,
                    numbers: list[int] | None = None) -> list[str]:
    """A child writing `count` raw frames whose first byte is the frame index
    -- or `numbers[index]`, the frame of the video it stands for."""
    numbers = list(range(count)) if numbers is None else numbers
    script = (
        "import sys\n"
        f"n, size, partial, code, numbers = {count}, {frame_bytes}, {partial}, {exit_code}, {numbers}\n"
        "out = sys.stdout.buffer\n"
        "for i in range(n):\n"
        "    out.write(bytes([numbers[i] % 256]) + bytes(size - 1))\n"
        "if partial:\n"
        "    out.write(bytes(size // 2))\n"
        "out.flush()\n"
        "sys.exit(code)\n"
    )
    return [STDLIB_PYTHON, "-S", "-c", script]


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
         inspect=None, fail=None, on_status=None, together=False, on_progress=None, options=None, device=None,
         timestamps=None, positions=True):
    """run_vship_task where each spawned 'FFmpeg' is the next child for its input.

    `children` maps "source"/"test" to the commands that input's successive
    starts run: the two readers start concurrently, so which spawns first is
    a race, and children are matched to inputs by path rather than by order.
    Each child's frames are stamped as FFmpeg stamps them (-stats_enc_pre):
    the n-th at `timestamps[side](n)` ms, 42 ms apart unless given.

    The fake score is the frame index read back out of the pinned buffer the
    lane was handed, so a mixed-up slot or pairing shows up in the values.
    `positions`: the source frame of each pair must be the test frame's
    number -- off for videos whose frames do not line up.
    """
    stamps = {"source": [lambda n: n * 42], "test": [lambda n: n * 42]}
    for side, given in (timestamps or {}).items():
        stamps[side] = given if isinstance(given, list) else [given]
    monkeypatch.setattr(vship, "_PinnedBuffer", _FakePinned)
    monkeypatch.setattr(vship, "_init_handler", lambda *_args: vship._Handle())
    monkeypatch.setattr(gpu, "pick_hwaccel", hwaccel)  # what gpu.pick_decode asks
    # The stand-ins write bare frames: every layout goes as raw video here.
    # The YUV4MPEG stream has tests of its own, below.
    monkeypatch.setattr(vship, "_piped_as", lambda pixel_format: ["-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"])
    queues = {side: list(commands) for side, commands in children.items()}
    spawned: dict[str, list[list[str]]] = {"source": [], "test": []}
    source_path = str((source or _hevc("source.mkv")).path.resolve())

    def fake_spawn(command):
        side = "source" if source_path in command else "test"
        spawned[side].append(command)
        # Written whole before the frames: FFmpeg writes each frame's line
        # before the frame.
        stamps_path = Path(command[command.index("-stats_enc_pre") + 1])
        stamped = stamps[side][min(len(spawned[side]), len(stamps[side])) - 1]  # one per start, the last reused
        stamps_path.write_text("".join(f"{stamped(n)} 1/1000\n" for n in range(5000)))
        # The last command is reused: each metric has a pass (and a decode) of its own.
        queue = queues[side]
        process = vship.proc_util.popen(queue.pop(0) if len(queue) > 1 else queue[0], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        return process, process.stdout

    cancel, scored = threading.Event(), []

    def fake_compute(_device, key, _handler, source_planes, test_planes, *_strides):
        index = test_planes[0][0]
        if positions:
            assert source_planes[0][0] == index, "a lane paired frames from different positions"
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
        VmafOptions(crop_mode=CropMode.NONE, gpu_decode=gpu_decode, **(options or {})), metrics)
    output = vship.run_vship_task(source or _hevc("source.mkv"), test or _hevc("test.mkv"), request,
                                  request.metrics, device or _fake_device(), None, None, cancel_event=cancel,
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


# ------------------------------------- videos decoded in the scoring process

class _FakeDecoder:
    """gpu_frames.GpuFrameStream's part in a pass: `count` pictures, each
    stamped pts(number), whose download writes the picture's number into the
    slot's first byte (the fake score reads it back)."""

    def __init__(self, count, *, pts=lambda number: number * 42, fail_at=None):
        self.count, self.pts, self.fail_at = count, pts, fail_at
        self.time_base = Fraction(1, 1000)
        self.number = 0
        self.held: dict[int, int] = {}
        self.released, self.verified, self.closed = 0, 0, False

    def start(self):
        pass

    def next(self, _timeout_ms=100):
        if self.number == self.fail_at:
            raise gpu_frames.GpuDecodeFailedError("the GPU's decoder found an error in the video")
        if self.number >= self.count:
            self.verified += 1
            return None
        slot = self.number % 4
        assert slot not in self.held, "a slot was handed out again before it was released"
        self.held[slot] = self.number
        self.number += 1
        return slot, self.pts(self.number - 1)

    def download(self, slot, address):
        ctypes.c_uint8.from_address(address).value = self.held[slot] % 256

    def release(self, slot):
        del self.held[slot]
        self.released += 1

    def verify(self):
        self.verified += 1

    def abort(self):
        pass

    def close(self):
        self.closed = True


#: The real one: the suite's conftest makes FFmpeg decode everywhere else.
_real_native_decoder = vship._native_decoder


def _decoded_here(monkeypatch, **kwargs) -> dict[str, _FakeDecoder]:
    """Each input decoded by a _FakeDecoder(**kwargs) of its own."""
    made: dict[str, _FakeDecoder] = {}

    def native(info, *_args):
        made[info.path.stem] = _FakeDecoder(**kwargs)
        return made[info.path.stem]

    monkeypatch.setattr(vship, "_native_decoder", native)
    return made


def test_a_gpu_decoded_video_gives_every_step_th_picture_up_to_the_limit(monkeypatch):
    """FFmpeg's chain: select every 3rd picture, then -t 0.5 (trim): with
    pictures 42 ms apart, pictures 0, 3, 6 and 9 -- 12 is at 504 ms. The
    source gives every picture, for each test picture's pair to be found,
    and is stopped when the pass ends."""
    made = _decoded_here(monkeypatch, count=40)
    output, _spawned = _run(monkeypatch, children=_both(_frames_command(0, _FRAME_BYTES)), metrics=("ssimulacra2",),
                            gpu_decode=True, hwaccel=lambda _vendor, _codec: "cuda",
                            options={"n_subsample": 3, "duration_limit": 0.5})
    result = output.metrics.get("ssimulacra2")
    assert list(result.values) == [0.0, 3.0, 6.0, 9.0]
    assert list(result.frame) == [0, 3, 6, 9]
    assert made["test"].verified  # the pictures up to the limit were checked
    assert made["source"].number > 13  # past picture 12, at 504 ms: the source is not cut at the limit
    assert all(d.closed and not d.held for d in made.values())


def _pairs_seen(monkeypatch, **kwargs) -> list[tuple[int, int]]:
    """(test frame, source frame) of each pair of the pass's scores, by the
    frames' first bytes: a pass made again scores its pairs again."""
    seen = []

    def inspect(_index, source_planes, test_planes):
        seen.append((test_planes[0][0], source_planes[0][0]))

    output, _spawned = _run(monkeypatch, metrics=("ssimulacra2",), inspect=inspect, positions=False, **kwargs)
    return sorted(seen[-len(output.metrics.get("ssimulacra2").values):])


def test_frames_are_paired_by_timestamp_as_libvmaf_pairs_them(monkeypatch):
    """The test video is the source with its sixth frame dropped, the
    others' times kept. Paired by position, every frame after the gap was
    compared with the source's next one."""
    test_frames = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    pairs = _pairs_seen(
        monkeypatch,
        children={"source": [_frames_command(12, _FRAME_BYTES)],
                  "test": [_frames_command(11, _FRAME_BYTES, numbers=test_frames)]},
        timestamps={"test": lambda n: test_frames[n] * 42 if n < len(test_frames) else 9999 + n})

    assert pairs == [(frame, frame) for frame in test_frames]


class _FakeProcess:
    def __init__(self, ended=False):
        self.ended = ended

    def poll(self):
        return 0 if self.ended else None


@pytest.mark.parametrize(("step", "limit"), [(1, None), (3, "1.500000")])
def test_ffmpeg_gives_each_piped_frames_own_timestamp(tmp_path, monkeypatch, step, limit):
    """Through real FFmpeg: the timestamp read for each piped frame is the
    one ffprobe gives that frame, from the first frame's."""
    monkeypatch.setattr(vship, "_PinnedBuffer", _FakePinned)
    path = _numbered_clip(tmp_path / "clip.mkv", 60, "N*40+mod(N*7\\,13)", "1/1000")
    chain = (f"select=not(mod(n\\,{step})),setpts=PTS-STARTPTS,format=yuv420p" if step > 1
             else "setpts=PTS-STARTPTS,format=yuv420p")
    command = [ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", chain,
               *(["-t", limit] if limit else []), "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
               "-f", "rawvideo", "pipe:1"]
    stream = vship._FrameStream(None, _W * _H * 3 // 2, [command], None, "test video")
    stream.start()
    stamps = []
    try:
        while (slot := stream.next(None)) != vship._EOF:
            stamps.append((stream.buffers[slot].array[0] - 3, stream.pts[slot]))
            stream.release(slot)
    finally:
        stream.close()

    out = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    probed = [int(line.strip(",")) for line in out.split() if line.strip(",")]
    assert stream.time_base == Fraction(1, 1000)
    assert stamps and all(pts == probed[number] - probed[0] for number, pts in stamps)
    assert [number for number, _pts in stamps] == _ffmpeg_piped(path, step, limit)


@pytest.mark.parametrize(("pixel_format", "shift"), [("yuv420p", 0), ("yuv420p16le", 0), ("yuv420p10le", 6)])
def test_the_gpu_decoder_gives_the_layout_vship_is_told(monkeypatch, pixel_format, shift):
    plans = []

    def stream(info, plan, *_args, **_kwargs):
        plans.append(plan)
        return "decoder"

    monkeypatch.setattr(vship.gpu_frames, "GpuFrameStream", stream)
    monkeypatch.setattr(vship.gpu_frames, "decoder_supports", lambda *_args: (True, ""))
    info = VideoInfo(Path("v.mkv"), 64, 48, 24.0, 1.0, 24, "hevc",
                     pix_fmt="yuv420p" if pixel_format == "yuv420p" else "yuv420p10le")
    image = vship._ImageFormat(pixel_format, 0, vship._VSHIP_ENUMS[8 if pixel_format == "yuv420p" else 16], 1, 1)
    frame_bytes = image.frame_layout(64, 48)[0]
    assert _real_native_decoder(info, None, (64, 48), image, frame_bytes, 0, None, "source") == "decoder"
    assert plans[0].shift == shift


# ------------------------------------------------ the frames as YUV4MPEG


def _stream_frames(command: list[str], frame_bytes: int) -> tuple[list[bytes], list[int]]:
    stream = vship._FrameStream(None, frame_bytes, [command], None, "test video")
    stream.start()
    frames, stamps = [], []
    try:
        while (slot := stream.next(None)) != vship._EOF:
            frames.append(bytes(stream.buffers[slot].array))
            stamps.append(stream.pts[slot])
            stream.release(slot)
    finally:
        stream.close()
    return frames, stamps


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le", "yuv444p16le"])
def test_ffmpegs_yuv4mpeg_frames_are_its_raw_frames_with_the_same_timestamps(tmp_path, monkeypatch, pixel_format):
    """Through real FFmpeg and the real pipe: the frames and timestamps the
    scoring gets are the ones the raw video gave it."""
    monkeypatch.setattr(vship, "_PinnedBuffer", _FakePinned)
    path = _numbered_clip(tmp_path / "clip.mkv", 30, "N*40+mod(N*7\\,13)", "1/1000")
    samples = _W * _H * {"yuv420p": 3, "yuv420p10le": 3, "yuv444p16le": 6}[pixel_format] // 2
    frame_bytes = samples * (1 if pixel_format == "yuv420p" else 2)
    head = [ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-vf", "setpts=PTS-STARTPTS", "-fps_mode", "passthrough"]
    piped = vship._piped_as(pixel_format)
    assert "yuv4mpegpipe" in piped
    raw = _stream_frames([*head, "-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"], frame_bytes)
    assert len(raw[0]) == 30
    assert _stream_frames([*head, *piped], frame_bytes) == raw


# ----------------- which pictures a video decoded in the scoring process gives

_W, _H = 64, 32


def _numbered_clip(path: Path, count: int, pts_expression: str, time_base: str = "1/1000") -> Path:
    """`count` frames whose luma is their number + 3, coded losslessly."""
    container = ["-video_track_timescale", time_base.split("/")[1]] if path.suffix == ".mp4" else []
    subprocess.run([
        ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", f"nullsrc=s={_W}x{_H}:r=24:d={count / 24 + 1}",
        "-vf", f"geq=lum='mod(N\\,250)+3':cb=128:cr=128,settb={time_base},setpts='{pts_expression}'",
        "-frames:v", str(count), "-fps_mode", "passthrough", "-enc_time_base", "filter",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", *container, str(path),
    ], check=True)
    return path


def _ffmpeg_piped(path: Path, step: int, limit: str | None) -> list[int]:
    """The frames the Vship command (perceptual_vship) pipes, by number."""
    chain = (f"select=not(mod(n\\,{step})),setpts=PTS-STARTPTS,format=yuv420p" if step > 1
             else "setpts=PTS-STARTPTS,format=yuv420p")
    raw = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", chain,
                          *(["-t", limit] if limit else []), "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                          "-f", "rawvideo", "pipe:1"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, _W * _H * 3 // 2)
    return (frames[:, 0].astype(int) - 3).tolist()


def _selected(path: Path, step: int, limit: str | None) -> list[int]:
    out = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=time_base:frame=pts", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    num, den = next(line for line in out if "/" in line).split("/")
    stamps = [int(line.strip(",")) for line in out if "/" not in line and line.strip(",")]
    selection = vship._FrameSelection(step, limit)
    kept = []
    for number, pts in enumerate(stamps):
        taken = selection.take(pts, Fraction(int(num), int(den)))
        if taken is None:
            break
        if taken:
            kept.append(number)
    return kept


def test_the_selection_is_the_ffmpeg_chains(tmp_path_factory, subtests):
    def check(name, clip, step, limit, tmp_path):
        count, expression, *rest = clip
        time_base = rest[0] if rest else "1/1000"
        suffix = rest[1] if len(rest) > 1 else "mkv"
        path = _numbered_clip(tmp_path / f"clip.{suffix}", count, expression, time_base)
        expected = _ffmpeg_piped(path, step, limit)
        assert expected
        assert _selected(path, step, limit) == expected

    for name, clip, step, limit in [
        ("every frame", (48, "N*42"), 1, None),
        ("every third", (48, "N*42"), 3, None),
        ("every seventh, a limit", (60, "N*42"), 7, "1.500000"),
        ("a limit on a frame", (48, "N*42"), 1, "1.008000"),
        ("a limit just past a frame", (48, "N*42"), 1, "1.008001"),
        ("a limit between frames", (48, "N*42"), 2, "1.000000"),
        ("variable rate", (50, "N*40+mod(N*7\\,13)"), 3, "1.300000"),
        ("mp4, 1/90000", (48, "N*3754", "1/90000", "mp4"), 2, "1.250000"),
    ]:
        with subtests.test(name=name, clip=clip, step=step, limit=limit):
            check(name, clip, step, limit, tmp_path_factory.mktemp("case"))


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
    return [STDLIB_PYTHON, "-S", "-c", script]


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


def test_cvvdp_scores_every_frame_in_order_with_a_jod_per_second(subtests):
    """One handler sees every frame in order (CVVDP is temporal); its score
    is reset at each second, and the overall JOD pooled from the seconds is
    what an unreset handler reports -- including a last second of one frame
    (49 frames at 24 fps), which Vship pools differently."""
    def check(count, monkeypatch):
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

    for count in [1, 5, 24, 25, 49, 24 * 3 + 7]:
        with subtests.test(count=count), pytest.MonkeyPatch.context() as case_patch:
            check(count, case_patch)


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


def test_pooling_a_final_one_frame_second_matches_vship():
    """Measured with the real Vship 5.1.1 on the first 49 frames (23.976 fps)
    of the Beekeeper AV1 encode: one handler never reset gave 9.9112921; the
    per-second JODs below (rounded to 3 decimals) are what the reset handler
    gave. Vship reports a single frame as JOD(q * IMAGE_INT), so the last
    second must be pooled differently -- pooled like the others it gives 9.921."""
    windows = [(0, 24, 10.0), (24, 24, 9.898), (48, 1, 9.802)]
    assert vship.pool_cvvdp_windows(windows) == pytest.approx(9.9112921, abs=5e-4)


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


def _sized(width: int, height: int) -> VideoInfo:
    return VideoInfo(Path(f"{width}x{height}.mkv"), width, height, 24.0, 60.0, 1440, "hevc", pix_fmt="yuv420p10le")


def test_cvvdp_above_1080p_is_not_started_on_an_intel_gpu(monkeypatch):
    """Vship's Vulkan CVVDP hung the 285K's Intel GPU at 4K (Windows reset
    it, the screen went black): it is refused before any GPU work, with the
    reason, and SSIMULACRA2 beside it still runs."""
    request = _cvvdp_request("ssimulacra2", "cvvdp")
    intel = vship.VshipDevice("vulkan", "Intel(R) Graphics", 0, "5.1.2", None, GpuVendor.INTEL)
    passes = []

    def one_pass(_s, _t, _r, specs, *_a, **_k):
        passes.append(tuple(spec.key for spec in specs))
        return _single_metric_output(specs[0].key, 80.0, "gpu")

    monkeypatch.setattr(vship, "_run_vship_pass", one_pass)
    output = vship.run_vship_task(_sized(3840, 2160), _sized(3840, 2160), request, request.metrics, intel, None, None)
    assert passes == [("ssimulacra2",)]
    assert output.metrics.keys() == ("ssimulacra2",)
    assert output.failures == {"cvvdp": "on Intel GPUs above 1920x1080 it is not calculated, as Vship's Vulkan "
                                        "build hangs the GPU (this comparison is 3840x2160)"}

    only = _cvvdp_request("cvvdp")
    with pytest.raises(vship.VshipPassesFailedError) as raised:  # alone: nothing to run
        vship.run_vship_task(_sized(3840, 2160), _sized(3840, 2160), only, only.metrics, intel, None, None)
    assert set(raised.value.failures) == {"cvvdp"}
    assert passes == [("ssimulacra2",)]


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
@pytest.mark.parametrize(("backend", "expected"), [("vulkan", GpuVendor.INTEL), ("cuda", GpuVendor.NVIDIA)])
def test_the_probe_knows_who_made_the_gpu(monkeypatch, backend, expected):
    """CUDA runs on NVIDIA and HIP on AMD; for Vulkan the loader says, since
    the name need not (NVIDIA's Quadro cards)."""
    asked = []

    def fake_lib(_path):
        def count(pointer):
            pointer._obj.value = 1
            return 0

        def info(pointer, _gpu_id):
            pointer._obj.name = b"Some GPU"
            return 0

        functions = {name: (lambda *_args: 0) for name in vship._API_FUNCTIONS}
        functions.update(Vship_GetDeviceCount=count, Vship_GetDeviceInfo=info,
                         Vship_GetVersion=lambda: SimpleNamespace(major=5, minor=1, minorMinor=2))
        return SimpleNamespace(**{name: _Callable(f) for name, f in functions.items()})

    monkeypatch.setattr(vship, "_backend", backend)
    monkeypatch.setattr(vship, "_vulkan_unavailable", lambda: None)
    monkeypatch.setattr(vship, "_vulkan_vendor", lambda name: asked.append(name) or GpuVendor.INTEL)
    monkeypatch.setattr(vship.ctypes, "CDLL", fake_lib)
    device, reason = vship._probe_vship_device()
    assert device is not None, reason
    assert (device.backend, device.vendor) == (backend, expected)
    assert asked == (["Some GPU"] if backend == "vulkan" else [])


class _Callable:
    """A function _configure_api can set argtypes and restype on."""

    def __init__(self, function):
        self._function = function

    def __call__(self, *args):
        return self._function(*args)


#: Vship's Vulkan SSIMULACRA2 before its shader was patched: 62.9 where CUDA
#: read 45.5 on the same 4K frames on an NVIDIA GPU, 45.50 on an Intel GPU.
_UNPATCHED_VULKAN = frozenset({("vulkan", GpuVendor.NVIDIA, "ssimulacra2")})


def test_a_metric_a_build_scores_wrongly_is_not_trusted_on_that_makers_gpu(monkeypatch):
    """Only that metric, on that build and maker's GPU; a GPU whose maker
    could not be told counts as affected."""
    monkeypatch.setattr(vship, "SCORED_WRONGLY", _UNPATCHED_VULKAN)
    nvidia = vship.VshipDevice("vulkan", "NVIDIA GeForce RTX 5090", 0, "5.1.2", None, GpuVendor.NVIDIA)
    intel = vship.VshipDevice("vulkan", "Intel(R) Graphics", 1, "5.1.2", None, GpuVendor.INTEL)
    unknown = vship.VshipDevice("vulkan", "GPU", 0, "5.1.2", None, None)
    cuda = vship.VshipDevice("cuda", "GPU", 0, "5.1.1", None, GpuVendor.NVIDIA)
    assert not vship.scores_correctly(nvidia, "ssimulacra2") and not vship.scores_correctly(unknown, "ssimulacra2")
    assert vship.scores_correctly(intel, "ssimulacra2") and vship.scores_correctly(cuda, "ssimulacra2")
    assert all(vship.scores_correctly(device, key)
               for device in (nvidia, intel, unknown) for key in ("butteraugli", "cvvdp"))


def test_an_odd_sized_video_is_decoded_in_software_from_the_start(monkeypatch):
    """FFmpeg's NVIDIA decode of an odd-sized video gives a padded picture,
    its chroma a row out for an odd height: SSIMULACRA2 20.4 for 45.1 on an
    854x479 AV1 pair. The even-sized test video keeps the GPU."""
    source = VideoInfo(Path("source.mkv"), 63, 47, 24.0, 1.0, 24, "av1", pix_fmt="yuv420p")
    statuses = []  # (the source is scaled to the test video's 64x48: its frames are that size)
    _output, spawned = _run(
        monkeypatch, gpu_decode=True, metrics=("ssimulacra2",), source=source, on_status=statuses.append,
        hwaccel=lambda _vendor, _codec: "cuda",
        children={"source": [_frames_command(2, vship._image_format(source).frame_layout(64, 48)[0])],
                  "test": [_frames_command(2, _FRAME_BYTES)]},
    )
    (source_cmd,), (test_cmd,) = spawned["source"], spawned["test"]
    assert "-hwaccel" not in source_cmd and "hwdownload" not in " ".join(source_cmd)
    assert test_cmd[test_cmd.index("-hwaccel") + 1] == "cuda"
    assert any("(GPU decode: source cpu, distorted cuda)" in status for status in statuses)
