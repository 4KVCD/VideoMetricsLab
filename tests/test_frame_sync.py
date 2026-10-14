"""frame_sync's pairing -- each video's frame N with the other's -- against
FFmpeg: on small videos whose frames carry their numbers in their pixels,
with timestamps of many kinds (the same rate, a millisecond off, other
rates, gaps, ties, one video longer, other time bases), FFmpeg's overlay
pairs the two videos' numbered frames (numbering_filters) N with N, as
frame_pairs does, and an output's -t keeps the frames frames_within counts.
And, on a PC with an NVIDIA GPU, VMAF from frames decoded and paired here
against VMAF from FFmpeg's."""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_gpu_frames import _clip as _coded_clip
from tests.test_gpu_frames import _gpu_decodes
from vmaf_app.core import gpu_frames as nv
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.frame_sync import duration_args, frame_pairs, frame_rate, frames_within, numbering_filters
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


def _ffmpeg_pairs(main: Path, ref: Path, tmp: Path, rate: Fraction, limit: str | None) -> list[tuple[int, int]]:
    """The frame numbers of the pairs FFmpeg's overlay makes of the two
    videos' numbered frames, cut at `limit` as every output is (duration_args)."""
    numbering = ",".join(numbering_filters(rate))
    graph = (f"[0:v]format=yuv420p,{numbering}[main];[1:v]format=yuv420p,{numbering}[ref];"
             + _gpu_pairs_stage("yuv420p", W, H, "main", "ref"))
    outputs = []
    for label, name in (("vmaf_dist", "d.raw"), ("vmaf_ref", "r.raw")):
        outputs += ["-map", f"[{label}]", "-fps_mode", "passthrough", *duration_args(limit),
                    "-f", "rawvideo", str(tmp / name)]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(main), "-i", str(ref),
                    "-lavfi", graph, *outputs], check=True)
    size = W * H * 3 // 2
    dist = np.fromfile(tmp / "d.raw", dtype=np.uint8).reshape(-1, size)[:, 0].astype(int) - 3
    source = np.fromfile(tmp / "r.raw", dtype=np.uint8).reshape(-1, size)[:, 0].astype(int) - 3
    assert len(dist) == len(source)
    return list(zip(dist.tolist(), source.tolist(), strict=True))


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
    "30 against 60 fps": ((30, "N*1000/30", "1/1000", "mkv"), (60, "N*1000/60", "1/1000", "mkv"), None),
    "ties halfway": ((30, "N*40+20", "1/1000", "mkv"), (31, "N*40", "1/1000", "mkv"), None),
    "test video has gaps": (
        (40, "N*42+if(gt(N\\,10)\\,84\\,0)+if(gt(N\\,25)\\,42\\,0)", "1/1000", "mkv"),
        (48, "N*42", "1/1000", "mkv"), None),
    "source has gaps": (
        (48, "N*42", "1/1000", "mkv"), (44, "N*42+if(gt(N\\,5)\\,126\\,0)", "1/1000", "mkv"), None),
    "variable rate both": (
        (45, "N*40+mod(N*7\\,13)", "1/1000", "mkv"), (45, "N*41+mod(N*5\\,11)", "1/1000", "mkv"), None),
    "source starts later": ((48, "N*42", "1/1000", "mkv"), (48, "500+N*42", "1/1000", "mkv"), None),
    "mp4 test video, 1/24000": ((48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), None),
    "23.976 fps from two clocks": (
        (240, "N*1001", "1/24000", "mp4"), (240, "round(N*1001/24)", "1/1000", "mkv"), None),
    "limit on a frame": ((48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), "1.001"),
    "limit between frames": ((48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), "1.000"),
    "limit between frames at a whole rate": ((60, "N*1000/24", "1/1000", "mkv"), (60, "N*40", "1/1000", "mkv"),
                                              "1.010"),
    "limit a frame on, as the GPU paths give it": (
        (48, "N*1001", "1/24000", "mp4"), (48, "N*42", "1/1000", "mkv"), "1.043"),
}


def test_frames_are_paired_by_number_whatever_their_timestamps(tmp_path_factory, subtests):
    folder = tmp_path_factory.mktemp("clips")
    made: dict[tuple, Path] = {}

    def clip(spec: tuple) -> Path:
        """Each clip the cases share, made once."""
        if spec not in made:
            made[spec] = _clip(folder / f"{len(made)}.{spec[3]}", *spec[:3])
        return made[spec]

    def check(name, tmp_path):
        test_spec, source_spec, limit = CASES[name]
        main, ref = clip(test_spec), clip(source_spec)
        rate = frame_rate(probe_video(main))
        kept = frames_within(limit, rate)
        count = min(test_spec[0], source_spec[0], *([kept] if kept is not None else []))
        expected = [(number, number) for number in range(count)]
        assert _ffmpeg_pairs(main, ref, tmp_path, rate, limit) == expected
        test_frames, source_frames = iter(range(test_spec[0])), iter(range(source_spec[0]))
        ours = frame_pairs(lambda: next(test_frames, None), lambda: next(source_frames, None),
                           lambda _frame: None, lambda _frame: None, kept)
        assert list(ours) == expected

    for name in CASES:
        with subtests.test(name=name):
            check(name, tmp_path_factory.mktemp("case"))


def test_frames_are_numbered_at_the_test_videos_exact_frame_rate(subtests):
    for fps, rate in ((23.976023976023978, Fraction(24000, 1001)), (29.97002997002997, Fraction(30000, 1001)),
                      (119.88011988011988, Fraction(120000, 1001)), (25.0, Fraction(25)), (0.0, Fraction(25))):
        with subtests.test(fps=fps):
            assert frame_rate(SimpleNamespace(fps=fps)) == rate
    # 30 s at 23.976 fps keeps frames 0 to 719 (719 * 1001 / 24000 = 29.99 s).
    assert frames_within("30.000", Fraction(24000, 1001)) == 720
    assert frames_within(None, Fraction(25)) is None


def test_frames_are_released_once_the_next_pair_is_asked_for():
    """Each frame is released exactly once: a pair's once the next pair is
    asked for, a test frame with no source frame left at once, and the last
    pair's when the pairing ends; past a limit, no frame is taken."""
    released = []
    main = iter(["m0", "m1", "m2", "m3"])
    ref = iter(["r0", "r1", "r2"])
    pairs = frame_pairs(lambda: next(main, None), lambda: next(ref, None), released.append, released.append)
    seen = []
    for test, source in pairs:
        assert test not in released and source not in released
        seen.append((test, source))
    assert seen == [("m0", "r0"), ("m1", "r1"), ("m2", "r2")]
    assert sorted(released) == ["m0", "m1", "m2", "m3", "r0", "r1", "r2"]

    released.clear()
    main, ref = iter(["m0", "m1", "m2"]), iter(["r0", "r1", "r2"])
    pairs = frame_pairs(lambda: next(main, None), lambda: next(ref, None), released.append, released.append, 2)
    assert list(pairs) == [("m0", "r0"), ("m1", "r1")]
    assert sorted(released) == ["m0", "m1", "r0", "r1"]
    assert next(main) == "m2"


def test_vmaf_from_pictures_decoded_here_is_vmaf_from_ffmpegs(tmp_path):
    """vmaf_gpu.score_decoded against libvmaf on the GPU given FFmpeg's
    decoded frames, as before: the same scores, frame for frame."""
    from vmaf_app.core import libvmaf, vmaf_gpu

    if not libvmaf.LIBRARY_PATH.is_file():
        pytest.skip("libvmaf with CUDA is not bundled")
    source = _coded_clip(tmp_path / "source.mkv", "h264", "yuv420p", seconds=1.0)
    test = tmp_path / "test.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(source), "-c:v", "libx264",
                    "-crf", "38", "-bf", "2", str(test)], check=True)
    source_info, test_info = probe_video(source), probe_video(test)
    if not _gpu_decodes(nv.plan_decode(source_info, None)):
        pytest.skip("no NVIDIA GPU decoder for this")
    models = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
    frames, scores = vmaf_gpu.score_decoded(source_info, test_info, None, None, width=640, height=360, bit_depth=8,
                                             models=models, n_subsample=1, duration_limit=None, total_frames=24)

    def raw(path):
        return subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo",
                               "-pix_fmt", "yuv420p", "pipe:1"], capture_output=True, check=True).stdout

    scorer = vmaf_gpu.GpuScorer(640, 360, 8, models)
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
