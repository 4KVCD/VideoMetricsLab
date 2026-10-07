"""gpu_frames: decoding on the GPU in the scoring process for the GPU
metrics. The plan and arithmetic are checked everywhere; the decoded
pictures against FFmpeg's decode with each GPU maker's decoder the PC has
(NVIDIA's, Intel's, AMD's)."""
from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading
from dataclasses import replace
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
def test_ten_bit_h264_is_refused_or_decoded_as_ffmpeg_decodes_it(tmp_path, backend):
    """Few GPUs decode H.264 above 8 bits, and a decoder must say so. AMD's
    library started on it all the same (its Init takes anything): on a
    Radeon 780M every picture was wrong, or decoding failed part-way and the
    decoder's shutdown never returned. Its stated capabilities are asked
    now."""
    eight = nv.plan_decode(probe_video(_clip(tmp_path / "eight.mkv", "h264", "yuv420p", seconds=0.5)), None, shift=6)
    _need(eight, backend)  # the PC has this maker's GPU
    path = _clip(tmp_path / "ten.mkv", "h264", "yuv420p10le", seconds=0.5)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, shift=6)
    supported, reason = nv.decoder_supports(0, plan, backend)
    if not supported:
        assert reason
        return
    assert backend != "amd", "no AMD GPU decodes 10-bit H.264"
    assert _decode(info, plan, backend)[0] == _ffmpeg_decode(path, plan)


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
    ours = _frames(info, plan, backend)
    assert np.array_equal(ours, _ffmpeg_frames(path, "format=yuv420p10le", 10))


@BACKENDS
def test_widened_full_range_luma_has_its_top_bits_repeated(tmp_path, backend):
    """255 becomes 1023, as FFmpeg widens full-range video; the chroma is
    shifted as ever."""
    path = _clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, widen=nv.WIDEN_REPEAT)
    _need(plan, backend)
    ours = _frames(info, plan, backend).reshape(-1, 640 * 360 * 3 // 2)
    narrow = _ffmpeg_frames(path, "format=yuv420p", 8).astype(np.uint16).reshape(ours.shape)
    luma = 640 * 360
    assert np.array_equal(ours[:, :luma], (narrow[:, :luma] << 2) | (narrow[:, :luma] >> 6))
    assert np.array_equal(ours[:, luma:], narrow[:, luma:] << 2)


@pytest.mark.parametrize("backend", ["intel", "amd"])
@pytest.mark.parametrize("widen", [nv.WIDEN_SHIFT, nv.WIDEN_REPEAT])
@pytest.mark.parametrize("to", [(640, 360), (1920, 1080)])
def test_a_scaled_eight_bit_picture_is_widened_as_it_is_scaled(backend, widen, to):
    """Intel's and AMD's decoders scale a widened picture on the CPU: the
    filtered value times 4 (full-range luma: 1023 for 255), 16-bit samples
    of 10 bits -- the scaled 8-bit picture's, to the rounding. No GPU is
    asked."""
    if not nv.LIBRARIES[backend].is_file():
        pytest.skip(f"{nv.LIBRARIES[backend].name} is not built")
    plain = nv.DecodePlan("h264", 8, 1280, 720, 0, 0, 1280, 720, 0, False, to[0], to[1], "bicubic")
    picture = _test_picture(1280, 720, 8)
    narrow = np.frombuffer(nv.scale_picture(plain, picture, None, backend), dtype=np.uint8).astype(np.int32)
    wide_plan = replace(plain, widen=widen, shift=6)
    assert wide_plan.frame_bytes == 2 * plain.frame_bytes
    wide = np.frombuffer(nv.scale_picture(wide_plan, picture, None, backend), dtype=np.uint16).astype(np.int32)
    assert wide.shape == narrow.shape and wide.max() <= 1023
    luma = to[0] * to[1]
    inside = (narrow > 0) & (narrow < 255)  # at the ends the 8-bit picture was clamped first
    gain = np.full(narrow.shape, 4.0)
    if widen == nv.WIDEN_REPEAT:
        gain[:luma] = 1023 / 255
    assert np.abs(wide - narrow * gain)[inside].max() <= 3
    assert wide[:luma][narrow[:luma] == 255].min() >= 1018 and wide[narrow == 0].max() <= 2  # 254.5 and 0.5, times 4


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


# ------------------------------------- AMD's hand-over (native/amf_handover.h)

@pytest.fixture(params=["windows", "amf"])
def amd_decoder(request, monkeypatch):
    """AMD's hand-over through Windows' own decoder (native/mf_frames.cpp),
    which GpuFrameStream opens first, or through AMF's (native/amf_frames.cpp),
    where Windows' does not open."""
    if request.param == "amf":
        monkeypatch.setattr(nv.GpuFrameStream, "_open_media_foundation", lambda *args: None)
    else:
        opened = nv.GpuFrameStream._open_media_foundation

        def windows_only(*args):
            if (handle := opened(*args)) is None:
                pytest.skip("Windows' decoder does not decode this on this PC")
            return handle

        monkeypatch.setattr(nv.GpuFrameStream, "_open_media_foundation", windows_only)
    return request.param


def _handed_over(info: VideoInfo, plan: nv.DecodePlan, planes: bool, pin: bool) -> list[str]:
    """Each picture's MD5, from AMD's decoder handing over: downloaded whole,
    or plane by plane into rows longer than the picture's (as libvmaf's
    pictures are), into memory pinned for the GPU or not."""
    stream = nv.GpuFrameStream(info, plan, backend="amd", handover=True)
    width, height = plan.output_size
    sample = plan.bytes_per_sample
    widths, heights = (width, (width + 1) // 2, (width + 1) // 2), (height, (height + 1) // 2, (height + 1) // 2)
    pitches = tuple(w * sample + 64 for w in widths)
    count = 1 if plan.luma_only else 3
    rows = [np.zeros((h, pitch), np.uint8) for h, pitch in zip(heights[:count], pitches[:count], strict=True)]
    whole = np.empty(plan.frame_bytes, np.uint8)
    sums = []
    try:
        if not stream.handover:
            pytest.skip("AMD's decoder does not hand over on this PC")
        if pin:
            assert all(stream.pin(plane.ctypes.data, plane.nbytes) for plane in rows)
        stream.start()
        while (item := _next(stream)) is not None:
            if planes:
                addresses = [plane.ctypes.data for plane in rows] + [None] * (3 - count)
                stream.download_planes(item[0], tuple(addresses), pitches)
                picture = np.concatenate([plane[:, :w * sample].ravel() for plane, w in zip(rows, widths[:count], strict=True)])
            else:
                stream.download(item[0], whole.ctypes.data)
                picture = whole
            stream.release(item[0])
            sums.append(hashlib.md5(picture).hexdigest())
    finally:
        if pin:
            for plane in rows:
                stream.unpin(plane.ctypes.data)
        stream.close()
    return sums


@pytest.mark.parametrize(("codec", "pix_fmt", "crop", "luma_only"), [
    ("h264", "yuv420p", None, False),
    ("h264", "yuv420p", CropBox(600, 300, 20, 30), False),
    ("hevc", "yuv420p10le", CropBox(638, 358, 2, 2), False),
    ("hevc", "yuv420p10le", None, True),
    ("h264", "yuv420p", CropBox(600, 300, 20, 30), True),
])
@pytest.mark.parametrize(("planes", "pin"), [(False, False), (True, False), (True, True)])
def test_pictures_amd_hands_over_are_ffmpegs_decode(tmp_path, amd_decoder, codec, pix_fmt, crop, luma_only, planes,
                                                    pin):
    path = _clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=1.0)
    info = probe_video(path)
    plan = nv.plan_decode(info, crop, shift=6, luma_only=luma_only)
    _need(plan, "amd")
    if luma_only:  # the luma of FFmpeg's pictures
        expected = [hashlib.md5(picture[:plan.frame_bytes]).hexdigest() for picture in _ffmpeg_pictures(path, plan)]
    else:
        expected = _ffmpeg_decode(path, plan)
    assert _handed_over(info, plan, planes, pin) == expected


def _ffmpeg_pictures(path: Path, plan: nv.DecodePlan) -> list[bytes]:
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    raw = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    size = (plan.crop_w * plan.crop_h + 2 * ((plan.crop_w + 1) // 2) * ((plan.crop_h + 1) // 2)) * plan.bytes_per_sample
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def test_ten_bit_amd_hands_over_kept_in_the_top_bits_is_the_shifted_picture_times_64(tmp_path, amd_decoder):
    info = probe_video(_clip(tmp_path / "clip.mkv", "hevc", "yuv420p10le", seconds=0.5))
    shifted, kept = nv.plan_decode(info, None, shift=6), nv.plan_decode(info, None, shift=0)
    _need(shifted, "amd")
    pictures = []
    for plan in (shifted, kept):
        stream = nv.GpuFrameStream(info, plan, backend="amd", handover=True)
        out = np.empty(plan.frame_bytes // 2, dtype=np.uint16)
        try:
            if not stream.handover:
                pytest.skip("AMD's decoder does not hand over on this PC")
            stream.start()
            item = _next(stream)
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
        finally:
            stream.close()
        pictures.append(out)
    assert np.array_equal(pictures[1], pictures[0] << 6)


def test_amd_hands_over_the_pictures_before_a_closed_gops_idr_as_ffmpeg_decodes_them(tmp_path, amd_decoder):
    """AMF's own decoder on Vulkan, which the hand-over first ran on, decodes
    HEVC's last B-pictures before each IDR wrong (a Radeon 780M, driver
    32.0.31041.1004; a UHD Blu-ray's too), on a Vulkan device AMF makes
    itself as well, where its Direct3D 11 decoder and FFmpeg's Vulkan
    decoding are right. The hand-over takes Direct3D 11's: a short closed
    GOP has those pictures every 12 frames."""
    path = tmp_path / "gop.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=s=640x360:r=24000/1001:d=2", "-pix_fmt", "yuv420p10le", "-c:v", "libx265",
                    "-preset", "ultrafast", "-x265-params", "keyint=12:min-keyint=12:no-open-gop=1:bframes=3:log-level=error",
                    str(path)], check=True)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, shift=6)
    _need(plan, "amd")
    assert _handed_over(info, plan, True, True) == _ffmpeg_decode(path, plan)


def test_pictures_amd_hands_over_and_an_edit_list_discards_are_ffmpegs(tmp_path, amd_decoder):
    """The pictures an edit list discards are decoded and handed over, then
    handed back unread: the ones after them are still FFmpeg's."""
    whole = _clip(tmp_path / "whole.mp4", "hevc", "yuv420p10le", ["-g", "48"], seconds=4.0)
    cut = tmp_path / "cut.mp4"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-ss", "1.3", "-i", str(whole), "-c", "copy",
                    str(cut)], check=True)
    info = probe_video(cut)
    plan = nv.plan_decode(info, None, shift=6)
    _need(plan, "amd")
    assert _handed_over(info, plan, True, True) == _ffmpeg_decode(cut, plan)


def test_two_amd_decoders_handing_over_at_once_give_ffmpegs_pictures(tmp_path, amd_decoder):
    """Scoring decodes both videos at once, and the decoders share the
    hand-over's Vulkan device, its queue and what is imported and pinned."""
    paths = [_clip(tmp_path / "a.mkv", "hevc", "yuv420p10le"), _clip(tmp_path / "b.mkv", "h264", "yuv420p")]
    infos = [probe_video(path) for path in paths]
    plans = [nv.plan_decode(info, None, shift=6) for info in infos]
    for plan in plans:
        _need(plan, "amd")
    expected = [_ffmpeg_decode(path, plan) for path, plan in zip(paths, plans, strict=True)]
    for _round in range(3):
        results: list[list[str] | BaseException] = [[], []]

        def decode(index: int, results: list = results) -> None:
            try:
                results[index] = _handed_over(infos[index], plans[index], True, True)
            except BaseException as error:  # reported below
                results[index] = error

        threads = [threading.Thread(target=decode, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for result in results:
            if isinstance(result, BaseException):
                raise result
        assert results == expected


def test_windows_decoder_whose_pictures_are_kept_many_at_once_gives_ffmpegs_pictures(tmp_path):
    """Windows' decoder waits for a free picture of its pool inside
    ProcessOutput when too many are kept -- it let nine of its 4K HEVC
    pictures be, of any pool asked for -- so native/mf_frames.cpp copies the
    pictures beyond a few into its own textures. Here all but one of 24 slots
    are kept at once."""
    path = _clip(tmp_path / "clip.mkv", "hevc", "yuv420p10le", seconds=2.0)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, shift=6)
    _need(plan, "amd")
    stream = nv.GpuFrameStream(info, plan, pool=24, backend="amd", handover=True)
    whole, kept, sums = np.empty(plan.frame_bytes, np.uint8), [], []
    try:
        if not stream.handover or stream._lib is not nv._libraries.get(nv.MEDIA_FOUNDATION):
            pytest.skip("Windows' decoder does not hand over on this PC")
        stream.start()
        while True:
            try:
                item = stream.next(20000)
            except TimeoutError:
                pytest.fail(f"the decoder gave no picture after {len(sums)}, {len(kept)} kept")
            if item is None:
                break
            stream.download(item[0], whole.ctypes.data)
            sums.append(hashlib.md5(whole).hexdigest())
            kept.append(item[0])
            while len(kept) > 23:
                stream.release(kept.pop(0))
    finally:
        stream.close()
    assert sums == _ffmpeg_decode(path, plan)


def test_amd_hands_over_only_unscaled_pictures_of_their_own_depth():
    plan = nv.plan_decode(_info(), None, shift=6)
    assert nv.can_hand_over(plan, "amd") and nv.can_hand_over(plan, "nvidia")
    assert not nv.can_hand_over(plan, "intel")
    assert not nv.can_hand_over(nv.plan_decode(_info(), None, size=(1280, 720)), "amd")
    eight = nv.plan_decode(_info(pix_fmt="yuv420p"), None, widen=nv.WIDEN_SHIFT)
    assert not nv.can_hand_over(eight, "amd") and nv.can_hand_over(eight, "nvidia")


class _OpeningLibrary:
    """A decoder library that records what nvf_open is asked, and refuses
    the hand-over when told to."""

    def __init__(self, hands_over: bool) -> None:
        self.hands_over, self.asked, self.imports = hands_over, [], []

    def nvf_open(self, params, error, size):
        handover = params._obj.handover
        self.asked.append(handover)
        if handover and not self.hands_over:
            error.value = b"the GPU's Vulkan driver has no VK_EXT_external_memory_host"
            return None
        return 7

    def nvf_import_vulkan(self, handle, win32, size, memory_type, device, driver, address, memory):
        self.imports.append((win32, size, memory_type, device, driver))
        address._obj.value, memory._obj.value = 1 << 40, 1 << 40
        return 0


def _amd_stream(monkeypatch, library, plan=None, windows=None) -> nv.GpuFrameStream:
    """A stream of AMD's decoder from fake libraries: AMF's `library`, and
    Windows' own decoder's `windows` (None: not bundled)."""

    def load(backend):
        if backend != nv.MEDIA_FOUNDATION:
            return library
        if windows is None:
            raise nv.GpuDecodeUnavailableError("the GPU frame decoder (mf_frames.dll) is not bundled")
        return windows

    monkeypatch.setattr(nv, "_load", load)
    monkeypatch.setattr(nv, "_PacketReader", lambda *args: type("Reader", (), {"close": lambda self: None})())
    return nv.GpuFrameStream(_info(), plan or nv.plan_decode(_info(), None, shift=6), backend="amd", handover=True)


def test_amd_hands_over_through_windows_own_decoder_first(monkeypatch):
    amf, windows = _OpeningLibrary(hands_over=True), _OpeningLibrary(hands_over=True)
    stream = _amd_stream(monkeypatch, amf, windows=windows)
    assert stream.handover and stream._lib is windows
    assert windows.asked == [1] and amf.asked == []


def test_amd_hands_over_through_amf_where_windows_own_decoder_does_not_open(monkeypatch, caplog):
    """Windows' decoders of HEVC and AV1 come with its video extensions,
    which a PC may not have."""
    caplog.set_level("INFO", logger=nv.__name__)
    amf, windows = _OpeningLibrary(hands_over=True), _OpeningLibrary(hands_over=False)
    stream = _amd_stream(monkeypatch, amf, windows=windows)
    assert stream.handover and stream._lib is amf
    assert windows.asked == [1] and amf.asked == [1]
    assert "decoded by AMF" in caplog.text


def test_windows_own_decoder_is_not_asked_for_pictures_amd_does_not_hand_over(monkeypatch):
    amf, windows = _OpeningLibrary(hands_over=True), _OpeningLibrary(hands_over=True)
    stream = _amd_stream(monkeypatch, amf, nv.plan_decode(_info(), None, shift=6, size=(1280, 720)), windows=windows)
    assert not stream.handover and stream._lib is amf
    assert windows.asked == [] and amf.asked == [0]


def test_amd_hands_over_when_its_vulkan_starts(monkeypatch):
    library = _OpeningLibrary(hands_over=True)
    stream = _amd_stream(monkeypatch, library)
    assert stream.handover and library.asked == [1]


def test_amd_decodes_as_before_when_its_vulkan_cannot_hand_over(monkeypatch, caplog):
    caplog.set_level("INFO", logger=nv.__name__)
    library = _OpeningLibrary(hands_over=False)
    stream = _amd_stream(monkeypatch, library)
    assert not stream.handover and library.asked == [1, 0]
    assert "VK_EXT_external_memory_host" in caplog.text
    assert not stream.pin(0x1000, 4096)
    assert stream.import_memory(5, 1 << 20) is None


def test_amd_is_not_asked_to_hand_over_scaled_pictures(monkeypatch):
    library = _OpeningLibrary(hands_over=True)
    stream = _amd_stream(monkeypatch, library, nv.plan_decode(_info(), None, shift=6, size=(1280, 720)))
    assert not stream.handover and library.asked == [0]


def test_amd_imports_vulkan_memory_only_knowing_its_gpu_and_driver(monkeypatch):
    """AMD's decoder imports Vulkan VMAF's memory into its own Vulkan device,
    which Vulkan allows only from the same GPU and driver: without them it
    imports nothing, and the frames go through system memory."""
    from types import SimpleNamespace

    library = _OpeningLibrary(hands_over=True)
    stream = _amd_stream(monkeypatch, library)
    assert stream.import_memory(5, 1 << 20) is None and library.imports == []
    exporter = SimpleNamespace(device_uuid=b"d" * 16, driver_uuid=b"r" * 16, memory_type=3)
    assert stream.import_memory(5, 1 << 20, exporter) == (1 << 40, 1 << 40)
    assert library.imports == [(5, 1 << 20, 3, b"d" * 16, b"r" * 16)]


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
    stream._closing, stream._finished = False, False
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
