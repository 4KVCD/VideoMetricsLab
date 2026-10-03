"""frame_sync.frame_pairs against FFmpeg's own frame sync: the overlay
vmaf_runner._gpu_pairs_stage pairs the two videos with (libvmaf's options),
on small videos whose frames carry their numbers in their pixels, with
timestamps of many kinds -- the same rate, a millisecond off, other rates,
gaps, ties, one video longer, other time bases, a -t limit. And, on a PC
with an NVIDIA GPU, VMAF from frames decoded and paired here against VMAF
from FFmpeg's."""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests.test_nvdec_frames import _clip as _coded_clip
from tests.test_nvdec_frames import _gpu_decodes
from vmaf_app.core import nvdec_frames as nv
from vmaf_app.core.ffmpeg_locate import ffmpeg_path, ffprobe_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.frame_sync import frame_pairs
from vmaf_app.core.nvdec_frames import duration_in
from vmaf_app.core.vmaf_runner import _gpu_pairs_stage

W, H = 64, 32


def _clip(path: Path, count: int, pts_expression: str, time_base: str = "1/1000") -> Path:
    """`count` frames whose luma is their number + 3, stamped by
    `pts_expression` (in `time_base`), coded losslessly."""
    container = ["-video_track_timescale", time_base.split("/")[1]] if path.suffix == ".mp4" else []
    subprocess.run([
        ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
        "-i", f"nullsrc=s={W}x{H}:r=24:d={count / 24 + 1}",
        "-vf", f"geq=lum='mod(N\\,250)+3':cb=128:cr=128,settb={time_base},setpts='{pts_expression}'",
        "-frames:v", str(count), "-fps_mode", "passthrough", "-enc_time_base", "filter",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", *container, str(path),
    ], check=True)
    return path


def _timestamps(path: Path) -> tuple[Fraction, list[int]]:
    out = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=time_base:frame=pts", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    base = next(line for line in out if "/" in line)
    num, den = base.split("/")
    return Fraction(int(num), int(den)), [int(line.strip(",")) for line in out if "/" not in line and line.strip(",")]


def _ffmpeg_pairs(main: Path, ref: Path, tmp: Path, limit: str | None) -> list[tuple[int, int]]:
    graph = ("[0:v]format=yuv420p,setpts=PTS-STARTPTS[main];[1:v]format=yuv420p,setpts=PTS-STARTPTS[ref];"
             + _gpu_pairs_stage("yuv420p", W, H, "main", "ref"))
    outputs = []
    for label, name in (("vmaf_dist", "d.raw"), ("vmaf_ref", "r.raw")):
        outputs += ["-map", f"[{label}]", "-fps_mode", "passthrough", *(["-t", limit] if limit else []),
                    "-f", "rawvideo", str(tmp / name)]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(main), "-i", str(ref),
                    "-lavfi", graph, *outputs], check=True)
    size = W * H * 3 // 2
    dist = np.fromfile(tmp / "d.raw", dtype=np.uint8).reshape(-1, size)[:, 0].astype(int) - 3
    source = np.fromfile(tmp / "r.raw", dtype=np.uint8).reshape(-1, size)[:, 0].astype(int) - 3
    assert len(dist) == len(source)
    return list(zip(dist.tolist(), source.tolist(), strict=True))


def _our_pairs(main: Path, ref: Path, limit: str | None) -> list[tuple[int, int]]:
    main_base, main_pts = _timestamps(main)
    ref_base, ref_pts = _timestamps(ref)
    main_frames = iter(enumerate(main_pts))
    ref_frames = iter(enumerate(ref_pts))
    stop = duration_in(limit, main_base) if limit else None
    pairs = []
    for test, source, when in frame_pairs(lambda: next(main_frames, None), lambda: next(ref_frames, None),
                                          main_base, ref_base, lambda _f: None, lambda _f: None):
        if stop is not None and when >= stop:
            break
        assert source is not None
        pairs.append((test, source))
    return pairs


# (test video: frames, pts, time base, container), (source: ...), -t
CASES = {
    "same rate": ((48, "N*42", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), None),
    "test video shorter": ((47, "N*42", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), None),
    "source shorter": ((48, "N*42", "1/1000", "mkv"), (46, "N*42", "1/1000", "mkv"), None),
    "test frames a millisecond early": (
        (48, "N*42-if(eq(mod(N\\,3)\\,1)\\,1\\,0)", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), None),
    "test frames a millisecond late": (
        (48, "N*42+if(eq(mod(N\\,4)\\,1)\\,1\\,0)", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), None),
    "24 against 25 fps": ((48, "N*1000/24", "1/1000", "mkv"), (50, "N*40", "1/1000", "mkv"), None),
    "25 against 24 fps": ((50, "N*40", "1/1000", "mkv"), (48, "N*1000/24", "1/1000", "mkv"), None),
    "30 against 60 fps": ((30, "N*1000/30", "1/1000", "mkv"), (60, "N*1000/60", "1/1000", "mkv"), None),
    "60 against 30 fps": ((60, "N*1000/60", "1/1000", "mkv"), (30, "N*1000/30", "1/1000", "mkv"), None),
    "ties halfway": ((30, "N*40+20", "1/1000", "mkv"), (31, "N*40", "1/1000", "mkv"), None),
    "test video has gaps": (
        (40, "N*42+if(gt(N\\,10)\\,84\\,0)+if(gt(N\\,25)\\,42\\,0)", "1/1000", "mkv"),
        (48, "N*42", "1/1000", "mkv"), None),
    "source has gaps": (
        (48, "N*42", "1/1000", "mkv"), (44, "N*42+if(gt(N\\,5)\\,126\\,0)", "1/1000", "mkv"), None),
    "variable rate both": (
        (45, "N*40+mod(N*7\\,13)", "1/1000", "mkv"), (45, "N*41+mod(N*5\\,11)", "1/1000", "mkv"), None),
    "mp4 source, other time base": (
        (48, "N*42", "1/1000", "mkv"), (48, "N*3754", "1/90000", "mp4")),
    "mp4 test video, 1/24000": ((48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), None),
    # vmaf_runner's case: timestamps a fraction of a millisecond apart, from
    # two containers' clocks -- the GPU once compared 240 pairs to the CPU's 239.
    "23.976 fps from two clocks": (
        (240, "N*1001", "1/24000", "mp4"), (240, "round(N*1001/24)", "1/1000", "mkv"), None),
    "23.976 fps from two clocks, source first": (
        (240, "round(N*1001/24)", "1/1000", "mkv"), (240, "N*1001", "1/24000", "mp4"), None),
    "limit on a frame": ((48, "N*42", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), "1.008"),
    "limit between frames": ((48, "N*42", "1/1000", "mkv"), (48, "N*42", "1/1000", "mkv"), "1.000"),
    "limit, other time base": ((48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), "1.043"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_pairs_are_ffmpegs(tmp_path, name):
    test_spec, source_spec, *rest = CASES[name]
    limit = rest[0] if rest else None
    main = _clip(tmp_path / f"main.{test_spec[3]}", *test_spec[:3])
    ref = _clip(tmp_path / f"ref.{source_spec[3]}", *source_spec[:3])
    expected = _ffmpeg_pairs(main, ref, tmp_path, limit)
    assert expected, "FFmpeg paired nothing"
    assert _our_pairs(main, ref, limit) == expected


def test_frames_are_released_once_no_pair_needs_them():
    """Each frame is released exactly once, and only after the pair after
    its last pair is asked for."""
    released = []
    main = iter([("m0", 0), ("m1", 42), ("m2", 83), ("m3", 125)])
    ref = iter([("r0", 0), ("r1", 30), ("r2", 50), ("r3", 84)])
    pairs = frame_pairs(lambda: next(main, None), lambda: next(ref, None), Fraction(1, 1000), Fraction(1, 1000),
                        released.append, released.append)
    seen = []
    for test, source, _when in pairs:
        assert test not in released and source not in released
        seen.append((test, source))
    # r1 is passed over (r2 is nearer m1); m3 comes after the source's end,
    # stamped a tick after r3, and ends the comparison (shortest=1).
    assert seen == [("m0", "r0"), ("m1", "r2"), ("m2", "r3")]
    assert sorted(released) == ["m0", "m1", "m2", "m3", "r0", "r1", "r2", "r3"]


def test_vmaf_from_pictures_decoded_here_is_vmaf_from_ffmpegs(tmp_path):
    """vmaf_cuda.score_decoded against libvmaf on the GPU given FFmpeg's
    decoded frames, as before: the same scores, frame for frame."""
    from vmaf_app.core import vmaf_cuda

    if not vmaf_cuda.LIBRARY_PATH.is_file():
        pytest.skip("libvmaf with CUDA is not bundled")
    source = _coded_clip(tmp_path / "source.mkv", "h264", "yuv420p", seconds=1.0)
    test = tmp_path / "test.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(source), "-c:v", "libx264",
                    "-crf", "38", "-bf", "2", str(test)], check=True)
    source_info, test_info = probe_video(source), probe_video(test)
    if not _gpu_decodes(nv.plan_decode(source_info, None)):
        pytest.skip("no NVIDIA GPU decoder for this")
    models = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
    frames, scores = vmaf_cuda.score_decoded(source_info, test_info, None, None, width=640, height=360, bit_depth=8,
                                             models=models, n_subsample=1, duration_limit=None, total_frames=24)

    def raw(path):
        return subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo",
                               "-pix_fmt", "yuv420p", "pipe:1"], capture_output=True, check=True).stdout

    scorer = vmaf_cuda.GpuScorer(640, 360, 8, models)
    try:
        size = scorer.frame_bytes
        a, b = raw(source), raw(test)
        for at in range(0, min(len(a), len(b)), size):
            scorer.add(bytearray(a[at:at + size]), bytearray(b[at:at + size]))
        want_frames, want = scorer.finish()
    finally:
        scorer.close()
    assert frames.tolist() == want_frames.tolist() and len(frames) == 24
    for key in models:
        assert scores[key].tolist() == want[key].tolist()
