"""gpu_frames: decoding on the GPU in the scoring process for the GPU
metrics. The plan and arithmetic are checked everywhere; the decoded
pictures against FFmpeg's decode with each GPU maker's decoder the PC has
(NVIDIA's, Intel's, AMD's)."""
from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import gpu_frames as nv
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropBox, VideoInfo


def _info(**overrides) -> VideoInfo:
    fields = {"path": Path("x.mkv"), "width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0,
              "nb_frames": 240, "codec_name": "hevc", "pix_fmt": "yuv420p10le"}
    fields.update(overrides)
    return VideoInfo(**fields)


# ------------------------------------------------------------------ the plan

def test_a_plan_is_the_whole_picture_without_a_crop():
    plan = nv.plan_decode(_info(), None, shift=6, luma_only=True)
    assert (plan.codec, plan.bit_depth, plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h, plan.shift) == (
        "hevc", 10, 0, 0, 1920, 1080, 6)
    assert plan.frame_bytes == 1920 * 1080 * 2


def test_a_crop_is_rounded_as_ffmpegs_crop_filter_rounds_it():
    """vf_crop on 4:2:0: left and top to the even sample at or before them,
    width and height down to even (checked against FFmpeg 9: crop=1917:1077:3:1
    gives 1916x1076)."""
    plan = nv.plan_decode(_info(), CropBox(1917, 1077, 3, 1))
    assert (plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h) == (2, 0, 1916, 1076)


def test_eight_bit_has_no_shift_and_packs_three_planes():
    plan = nv.plan_decode(_info(pix_fmt="yuv420p", codec_name="h264"), CropBox(1918, 1078, 2, 2), shift=6)
    assert plan.shift == 0 and plan.bytes_per_sample == 1
    assert plan.frame_bytes == 1918 * 1078 + 2 * 959 * 539


@pytest.mark.parametrize(("field", "value"), [("codec_name", "vvc"), ("codec_name", "vp9"),
                                              ("pix_fmt", "yuv422p10le"), ("pix_fmt", "yuv420p12le"),
                                              ("pix_fmt", "yuv444p"), ("width", 0),
                                              # An odd size: refused here, not once the pass has started.
                                              ("width", 1919), ("height", 1079)])
def test_what_the_gpu_decoders_do_not_decode_is_left_to_ffmpeg(field, value):
    with pytest.raises(nv.GpuDecodeUnavailableError):
        nv.plan_decode(_info(**{field: value}), None)


def test_a_scaled_plan_hands_back_the_size_it_is_scaled_to():
    plan = nv.plan_decode(_info(width=3840, height=2160), None, shift=6, size=(1920, 1080), algorithm="lanczos")
    assert plan.scaled and plan.output_size == (1920, 1080) and plan.scaler == "lanczos"
    assert plan.frame_bytes == (1920 * 1080 + 2 * 960 * 540) * 2
    unscaled = nv.plan_decode(_info(), None, size=(1920, 1080))
    assert not unscaled.scaled and unscaled.output_size == (1920, 1080)
    assert nv.plan_decode(_info(), None, algorithm="nonsense").scaler == "bicubic"


def test_eight_bit_widened_to_ten_hands_back_16_bit_samples():
    plan = nv.plan_decode(_info(pix_fmt="yuv420p", codec_name="h264"), None, luma_only=True,
                          widen=nv.WIDEN_SHIFT)
    assert plan.bytes_per_sample == 2 and plan.frame_bytes == 1920 * 1080 * 2
    with pytest.raises(nv.GpuDecodeUnavailableError):
        nv.plan_decode(_info(), None, widen=nv.WIDEN_SHIFT)  # 10-bit: nothing to widen


def test_a_crop_outside_the_picture_is_refused():
    with pytest.raises(nv.GpuDecodeUnavailableError):
        nv.plan_decode(_info(), CropBox(1920, 1000, 0, 100))


# ------------------------------------------------------------ the arithmetic

@pytest.mark.parametrize(("value", "source", "target", "expected"), [
    (1, Fraction(1, 1000), Fraction(1, 90000), 90),
    (41, Fraction(1, 1000), Fraction(1001, 24000), 1),        # 0.98 -> 1
    (1001, Fraction(1, 24000), Fraction(1, 1000), 42),        # 41.708 -> 42
    (3, Fraction(1, 2), Fraction(1, 1), 2),                   # 1.5: halves away from zero
    (-3, Fraction(1, 2), Fraction(1, 1), -2),
    (5, Fraction(1, 2), Fraction(1, 1), 3),
])
def test_rescale_rounds_as_av_rescale_q(value, source, target, expected):
    assert nv.rescale(value, source, target) == expected


def test_a_limit_is_read_to_the_microsecond_and_held_in_the_time_base():
    assert nv.duration_in("30.042", Fraction(1, 1000)) == 30042
    assert nv.duration_in("1.000000", Fraction(1, 90000)) == 90000
    assert nv.duration_in("0.0005", Fraction(1, 1000)) == 1   # 0.5 ms rounds up, as trim's av_rescale_q
    assert nv.duration_in("10.000500", Fraction(1001, 24000)) == 240


# ----------------------------------------- decoding, on the PC's GPUs

#: Each GPU maker's decoder: its tests run where the PC has one.
BACKENDS = pytest.mark.parametrize("backend", ["nvidia", "intel", "amd"])


def _gpu_decodes(plan: nv.DecodePlan, backend: str = "nvidia") -> bool:
    if not nv.LIBRARIES[backend].is_file():
        return False
    return nv.decoder_supports(0, plan, backend)[0]


def _need(plan: nv.DecodePlan, backend: str) -> None:
    if not _gpu_decodes(plan, backend):
        pytest.skip(f"no {backend} GPU decoder for this on this PC")


def _clip(path: Path, codec: str, pix_fmt: str, extra: list[str] | None = None, seconds: float = 2.0) -> Path:
    encoder = {"h264": ["-c:v", "libx264", "-preset", "veryfast", "-bf", "3"],
               "hevc": ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error"]}[codec]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s=640x360:r=24000/1001:d={seconds}", "-pix_fmt", pix_fmt, *encoder,
                    *(extra or []), str(path)], check=True)
    return path


def _decode(info: VideoInfo, plan: nv.DecodePlan, backend: str = "nvidia") -> tuple[list[str], list[int]]:
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    sums, stamps = [], []
    try:
        stream.start()
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            sums.append(hashlib.md5(out).hexdigest())
            stamps.append(item[1])
    finally:
        stream.close()
    return sums, stamps


def _ffmpeg_decode(path: Path, plan: nv.DecodePlan) -> list[str]:
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    out = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split(",")[5].strip() for line in out.splitlines() if line and not line.startswith("#")]


@pytest.mark.parametrize(("codec", "pix_fmt", "crop"), [
    ("h264", "yuv420p", None),
    ("h264", "yuv420p", CropBox(600, 300, 20, 30)),
    ("hevc", "yuv420p10le", CropBox(638, 358, 2, 2)),
])
@BACKENDS
def test_pictures_are_ffmpegs_decode(tmp_path, backend, codec, pix_fmt, crop):
    path = _clip(tmp_path / "clip.mkv", codec, pix_fmt)
    info = probe_video(path)
    plan = nv.plan_decode(info, crop, shift=6)
    _need(plan, backend)
    sums, stamps = _decode(info, plan, backend)
    assert len(sums) == round(2.0 * 24000 / 1001)
    assert sums == _ffmpeg_decode(path, plan)
    assert stamps == sorted(stamps)


@BACKENDS
def test_ten_bit_kept_in_the_top_bits_is_the_shifted_picture_times_64(tmp_path, backend):
    path = _clip(tmp_path / "clip.mkv", "hevc", "yuv420p10le", seconds=0.5)
    info = probe_video(path)
    shifted, kept = nv.plan_decode(info, None, shift=6), nv.plan_decode(info, None, shift=0)
    _need(shifted, backend)
    pictures = []
    for plan in (shifted, kept):
        stream = nv.GpuFrameStream(info, plan, backend=backend)
        out = np.empty(plan.frame_bytes // 2, dtype=np.uint16)
        try:
            stream.start()
            item = None
            while item is None:
                try:
                    item = stream.next(1000)
                except TimeoutError:
                    continue
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
        finally:
            stream.close()
        pictures.append(out)
    assert np.array_equal(pictures[1], pictures[0] << 6)


@BACKENDS
def test_frames_an_mp4_edit_list_cuts_off_are_not_handed_out(tmp_path, backend):
    """A copy cut out of an MP4 starts at a keyframe before the cut, and its
    edit list marks the frames before the cut discard: FFmpeg decodes them
    (the frames after refer to them) and drops them."""
    whole = _clip(tmp_path / "whole.mp4", "h264", "yuv420p", ["-g", "48"], seconds=4.0)
    cut = tmp_path / "cut.mp4"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-ss", "1.3", "-i", str(whole), "-c", "copy",
                    str(cut)], check=True)
    info = probe_video(cut)
    plan = nv.plan_decode(info, None)
    _need(plan, backend)
    sums, _stamps = _decode(info, plan, backend)
    expected = _ffmpeg_decode(cut, plan)
    assert sums == expected


@BACKENDS
def test_a_stream_that_does_not_start_with_a_keyframe_fails(tmp_path, backend):
    whole = _clip(tmp_path / "whole.mkv", "h264", "yuv420p", seconds=1.0)
    headless = tmp_path / "headless.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(whole), "-c", "copy",
                    "-bsf:v", "noise=drop=eq(n\\,0)", str(headless)], check=True)
    info = probe_video(headless)
    plan = nv.plan_decode(info, None)
    _need(plan, backend)
    with pytest.raises(nv.GpuDecodeFailedError):
        _decode(info, plan, backend)


@BACKENDS
def test_a_stream_without_timestamps_fails(tmp_path, backend):
    """A raw H.264 stream with B-frames: FFmpeg copies packets without
    timestamps (and says so), so which picture is which cannot be told."""
    raw = _clip(tmp_path / "clip.264", "h264", "yuv420p", seconds=1.0)
    info = probe_video(raw)
    plan = nv.plan_decode(info, None)
    _need(plan, backend)
    with pytest.raises(nv.GpuDecodeFailedError):
        _decode(info, plan, backend)


def _ffmpeg_frames(path: Path, chain: str, depth: int) -> np.ndarray:
    raw = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", chain,
                          "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint16 if depth > 8 else np.uint8)


def _frames(info: VideoInfo, plan: nv.DecodePlan, backend: str) -> np.ndarray:
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    dtype = np.uint16 if plan.bytes_per_sample == 2 else np.uint8
    pictures = []
    try:
        stream.start()
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            out = np.empty(plan.frame_bytes // plan.bytes_per_sample, dtype=dtype)
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            pictures.append(out)
    finally:
        stream.close()
    return np.concatenate(pictures)


@pytest.mark.parametrize(("pix_fmt", "size", "algorithm"), [
    ("yuv420p10le", (320, 180), "bicubic"),
    ("yuv420p", (1280, 720), "lanczos"),
    ("yuv420p", (426, 240), "bilinear"),
    ("yuv420p10le", (960, 540), "spline"),
])
@BACKENDS
def test_scaled_pictures_are_ffmpegs_but_for_rounding(tmp_path, backend, pix_fmt, size, algorithm):
    """Not FFmpeg's scale filter's to the sample (a comparison scaled any way
    is the same comparison), but the same filter: on this synthetic picture's
    hard edges and odd sizes, where they differ most -- the CPU scaler of
    Intel's and AMD's decoders filters down the columns first where that is
    faster, so its cap falls after the other pass -- all but 1 in 200
    samples within 1, none more than 6 (on film, every sample within 1). A
    wrong plane, siting or filter is tens to hundreds off."""
    codec = "hevc" if pix_fmt == "yuv420p10le" else "h264"
    path = _clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=0.5)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, shift=6, size=size, algorithm=algorithm)
    _need(plan, backend)
    ours = _frames(info, plan, backend).astype(np.int32)
    depth = 10 if pix_fmt == "yuv420p10le" else 8
    fmt = "yuv420p10le" if depth > 8 else "yuv420p"
    want = _ffmpeg_frames(path, f"scale={size[0]}:{size[1]}:flags={algorithm},format={fmt}", depth).astype(np.int32)
    assert ours.shape == want.shape
    difference = np.abs(ours - want)
    assert difference.max() <= 6
    assert np.mean(difference > 1) < 0.005


@BACKENDS
def test_widened_eight_bit_is_ffmpegs_conversion_to_ten(tmp_path, backend):
    """Widening is exact: v << 2, as FFmpeg converts limited-range 8-bit."""
    path = _clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, widen=nv.WIDEN_SHIFT)
    _need(plan, backend)
    if backend != "nvidia":
        pytest.skip("only NVIDIA's decoder widens (for VMAF on the GPU)")
    ours = _frames(info, plan, backend)
    assert np.array_equal(ours, _ffmpeg_frames(path, "format=yuv420p10le", 10))


# ------------------------------------------------- scaling on the GPU (Intel, AMD)

#: PCI vendor ids of the GPUs the scaling shader is checked on; the last is
#: Windows' software device, which every PC has.
_SCALE_GPUS = {"nvidia": 0x10DE, "intel": 0x8086, "amd": 0x1002, "software": 0x1414}


def _test_picture(width: int, height: int, depth: int) -> bytes:
    """NV12/P010 noise with hard edges, where a filter rings: a wrong weight,
    order of summing, cap or rounding shows."""
    rng = np.random.default_rng(7)
    top = (1 << depth) - 1
    luma = rng.integers(0, top + 1, (height, width))
    luma[: height // 3, : width // 2] = top
    luma[height // 3: height // 2, width // 3:] = 0
    chroma = rng.integers(0, top + 1, (height // 2, width))
    chroma[: height // 5] = top
    planes = np.concatenate([luma, chroma])
    return (planes << 6).astype("<u2").tobytes() if depth > 8 else planes.astype(np.uint8).tobytes()


@pytest.mark.parametrize("gpu", list(_SCALE_GPUS))
@pytest.mark.parametrize(("depth", "size", "to", "algorithm", "shift", "luma_only", "crop"), [
    (10, (1280, 720), (640, 360), "bicubic", 6, False, None),
    (10, (1280, 720), (640, 360), "lanczos", 0, False, None),  # the samples kept in the top bits
    (8, (640, 360), (1280, 720), "bicubic", 0, False, None),  # up: along the rows first
    (8, (1280, 720), (427, 241), "spline", 0, False, None),  # odd output: its chroma is rounded up
    (10, (1280, 720), (854, 480), "bilinear", 6, True, None),
    (10, (1280, 720), (640, 300), "bicubic", 6, False, (0, 60, 1280, 600)),  # cropped: black bars cut
    (8, (640, 360), (320, 640), "lanczos", 0, False, None),  # narrower and taller at once
])
def test_the_scaling_shader_gives_the_cpu_scalers_picture(gpu, depth, size, to, algorithm, shift, luma_only, crop):
    """Sample for sample: the same weights, summed in the same order."""
    if not nv.available("intel"):
        pytest.skip("the decoders are not built")
    x, y, w, h = crop or (0, 0, *size)
    plan = nv.DecodePlan("hevc" if depth > 8 else "h264", depth, size[0], size[1], x, y, w, h, shift, luma_only,
                         to[0], to[1], algorithm)
    picture = _test_picture(size[0], size[1], depth)
    on_gpu = nv.scale_picture(plan, picture, _SCALE_GPUS[gpu])
    if on_gpu is None:
        pytest.skip(f"no {gpu} GPU with Direct3D 11 here")
    assert on_gpu == nv.scale_picture(plan, picture, None)


@pytest.mark.parametrize("backend", ["intel", "amd"])
@pytest.mark.parametrize(("pix_fmt", "size", "algorithm"), [
    ("yuv420p10le", (320, 180), "bicubic"),
    ("yuv420p", (1920, 1080), "lanczos"),
])
def test_decoded_pictures_are_scaled_on_the_gpu_and_are_the_cpus(tmp_path, backend, pix_fmt, size, algorithm):
    from dataclasses import replace

    codec = "hevc" if pix_fmt == "yuv420p10le" else "h264"
    info = probe_video(_clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=0.5))
    plan = nv.plan_decode(info, None, shift=6, size=size, algorithm=algorithm)
    _need(plan, backend)
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    try:
        stream.start()
        while stream.stats().displayed == 0:  # where the first picture was scaled
            with contextlib.suppress(TimeoutError):
                if (item := stream.next(1000)) is None:
                    break
                stream.release(item[0])
        assert stream.scale_note() == ""
        assert stream.stats().scaled_on_gpu == 1
    finally:
        stream.close()
    assert np.array_equal(_frames(info, plan, backend), _frames(info, replace(plan, cpu_scaling=True), backend))


@pytest.mark.parametrize("backend", ["intel", "amd"])
def test_a_gpu_picture_that_is_not_the_cpus_hands_the_scaling_to_the_cpu(tmp_path, backend):
    """A driver whose shader arithmetic differed would change the scores:
    the decoders scale their first pictures on the CPU as well, and at the
    first that differs the CPU's is the one handed out, from then on."""
    from dataclasses import replace

    info = probe_video(_clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5))
    plan = nv.plan_decode(info, None, size=(320, 180))
    _need(plan, backend)
    spoiled = replace(plan, cpu_scaling=2)  # the GPU's first picture is made wrong
    stream = nv.GpuFrameStream(info, spoiled, backend=backend)
    try:
        stream.start()
        while (item := _next(stream)) is not None:
            stream.release(item[0])
        assert stream.scale_note() == "the GPU's scaled picture is not the CPU's"
        assert stream.stats().scaled_on_gpu == 0
    finally:
        stream.close()
    assert np.array_equal(_frames(info, spoiled, backend), _frames(info, replace(plan, cpu_scaling=1), backend))


def _next(stream):
    while True:
        with contextlib.suppress(TimeoutError):
            return stream.next(1000)


class _Library:
    """A decoder library whose nvf_close returns when `closes` says so."""

    def __init__(self, closes: threading.Event) -> None:
        self.closes, self.closed = closes, []

    def nvf_abort(self, handle) -> None:
        pass

    def nvf_close(self, handle) -> None:
        self.closes.wait()
        self.closed.append(handle)


def _open_stream(library) -> nv.GpuFrameStream:
    """A stream as close() finds it: open, its feeding thread ended."""
    stream = object.__new__(nv.GpuFrameStream)
    stream.info, stream.backend, stream._lib, stream._handle = _info(), "nvidia", library, 7
    stream._closing = False
    stream._reader = type("Reader", (), {"close": lambda self: None})()
    stream._feeder = threading.Thread(target=lambda: None)
    return stream


def test_a_decoder_that_never_closes_is_left_open_and_the_run_goes_on(monkeypatch, caplog):
    """AMD's library has been seen never to return from closing a decoder:
    the run hung at its end, scores and all."""
    never = threading.Event()
    library = _Library(never)
    monkeypatch.setattr(nv, "_CLOSE_SECONDS", 0.01)
    monkeypatch.setattr(nv, "_stuck_closes", 0)
    stream = _open_stream(library)
    try:
        stream.close()  # returns, though the library's close has not
        assert nv.stuck_decoders() == 1
        assert library.closed == [] and stream._handle is None
        assert "GPU decoder did not close" in caplog.text
        stream.close()  # safe to call twice
        assert nv.stuck_decoders() == 1
    finally:
        never.set()


def test_a_decoder_that_closes_is_not_counted_as_stuck(monkeypatch):
    done = threading.Event()
    done.set()
    library = _Library(done)
    monkeypatch.setattr(nv, "_stuck_closes", 0)
    stream = _open_stream(library)
    stream.close()
    assert library.closed == [7] and nv.stuck_decoders() == 0


def _stream_fed(*timestamps):
    """An GpuFrameStream's picture bookkeeping alone, with these packets fed."""
    import heapq
    import threading

    stream = nv.GpuFrameStream.__new__(nv.GpuFrameStream)
    stream._waiting, stream._fed_count, stream._fed_lock = [], 0, threading.Lock()
    stream._discard, stream._shown_count, stream._last_shown, stream._finished = set(), 0, None, False
    for pts in timestamps:
        heapq.heappush(stream._waiting, pts)
        stream._fed_count += 1
    return stream


def test_a_picture_the_decoder_drops_fails_at_the_next_one():
    """A dropped picture failed the run only at its end, a whole pass later."""
    stream = _stream_fed(0, 3000, 1000, 2000)
    stream._take(0)
    with pytest.raises(nv.GpuDecodeFailedError, match="1 pictures for 2 packets"):
        stream._take(2000)


def test_a_picture_the_packets_do_not_have_fails_at_once():
    stream = _stream_fed(0, 2000)
    stream._take(0)
    with pytest.raises(nv.GpuDecodeFailedError, match="packets do not have"):
        stream._take(1000)


def test_pictures_in_order_pass_and_the_end_counts_them():
    stream = _stream_fed(0, 3000, 1000, 2000)
    stream._discard.add(0)
    assert [stream._take(pts) for pts in (0, 1000, 2000)] == [True, False, False]
    stream._finished = True
    with pytest.raises(nv.GpuDecodeFailedError, match="3 pictures for 4 packets"):
        stream.verify()
    stream._take(3000)
    stream.verify()
    with pytest.raises(nv.GpuDecodeFailedError, match="out of order"):
        stream._take(2500)
