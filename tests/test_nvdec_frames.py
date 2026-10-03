"""nvdec_frames: decoding on the GPU in the scoring process for the GPU
metrics. The plan and arithmetic are checked everywhere; the decoded
pictures against FFmpeg's decode with each GPU maker's decoder the PC has
(NVIDIA's, Intel's, AMD's)."""
from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import nvdec_frames as nv
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
                                              ("pix_fmt", "yuv444p"), ("width", 0)])
def test_what_nvdec_frames_does_not_decode_is_left_to_ffmpeg(field, value):
    with pytest.raises(nv.NvdecUnavailableError):
        nv.plan_decode(_info(**{field: value}), None)


def test_a_crop_outside_the_picture_is_refused():
    with pytest.raises(nv.NvdecUnavailableError):
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
    stream = nv.NvdecStream(info, plan, backend=backend)
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
        stream = nv.NvdecStream(info, plan, backend=backend)
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
    with pytest.raises(nv.NvdecFailedError):
        _decode(info, plan, backend)


@BACKENDS
def test_a_stream_without_timestamps_fails(tmp_path, backend):
    """A raw H.264 stream with B-frames: FFmpeg copies packets without
    timestamps (and says so), so which picture is which cannot be told."""
    raw = _clip(tmp_path / "clip.264", "h264", "yuv420p", seconds=1.0)
    info = probe_video(raw)
    plan = nv.plan_decode(info, None)
    _need(plan, backend)
    with pytest.raises(nv.NvdecFailedError):
        _decode(info, plan, backend)
