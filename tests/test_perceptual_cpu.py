from __future__ import annotations

import sys
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import perceptual_cpu
from vmaf_app.core.analysis_request import AnalysisRequest, ExecutionPreferences, FrameCoverage, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.models import CropMode, GpuVendor, ScaleDirection, VideoInfo
from vmaf_app.core.perceptual_cpu import parse_score, run_perceptual_task


@pytest.fixture(autouse=True)
def _fakes_are_not_cut_short(monkeypatch):
    """The fakes here produce a few frames for videos whose lengths promise
    more: frame_coverage would rightly call them cut short. Tested in
    test_frame_coverage.py."""
    monkeypatch.setattr(perceptual_cpu, "short_comparison", lambda *a, **k: None)


def _info(path: str) -> VideoInfo:
    return VideoInfo(Path(path), 64, 48, 24.0, 1.0, 24, "h264")


def _request(*keys: str) -> AnalysisRequest:
    return AnalysisRequest(
        recipe=ComparisonRecipe(CropMode.NONE, "bicubic", ScaleDirection.SOURCE_TO_DISTORTED, 0.0, None),
        metrics=tuple(MetricRequestSpec(key, "perceptual", (), FrameCoverage("full"), f"{key}-reference-cli-v1") for key in keys),
        execution=ExecutionPreferences(False, GpuVendor.NONE, 1),
    )


@pytest.mark.parametrize(("text", "expected"), [
    ("SSIMULACRA2: 91.75", 91.75),
    ("butteraugli score = 0.2345", 0.2345),
])
def test_perceptual_output_parser_uses_final_scalar(text, expected):
    assert parse_score("ssimulacra2", text) == expected


def test_backend_returns_independent_frame_results_without_real_tools(tmp_path, monkeypatch):
    request = _request("ssimulacra2", "butteraugli")
    references = [tmp_path / f"r-{i}.png" for i in range(3)]
    tests = [tmp_path / f"t-{i}.png" for i in range(3)]
    for picture in (*references, *tests):
        picture.write_bytes(bytes([0x89]) + b"PNG")  # their colours are described before scoring
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.find_metric_executable", lambda key: key)
    def fake_pairs(*_args, **_kwargs):
        yield from zip(references, tests, strict=True)

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu._png_pairs", fake_pairs)
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu._tool_version", lambda executable: "test-tool 1")
    monkeypatch.setattr(
        "vmaf_app.core.perceptual_cpu._run_metric",
        lambda executable, key, reference, test, *_args: 90.0 if key == "ssimulacra2" else 0.25,
    )
    output = run_perceptual_task(_info("source.mp4"), _info("test.mp4"), request, request.metrics)
    assert output.metrics.keys() == ("ssimulacra2", "butteraugli")
    assert np.array_equal(output.metrics.frame("ssimulacra2").frame, np.array([0, 1, 2]))
    assert output.metrics.frame("butteraugli").aggregate == pytest.approx(0.25)
    assert output.metrics.frame("ssimulacra2").provenance.compute_backend == "cpu"


# ------------------------------------------ several frame pairs at a time


def _parallel_task(tmp_path, monkeypatch, count, workers, run_metric, keys=("ssimulacra2",), pairs=None, **kwargs):
    """run_perceptual_task on `count` fake pairs with `workers` at a time."""
    request = _request(*keys)
    references = [tmp_path / f"r-{i}.png" for i in range(count)]
    tests = [tmp_path / f"t-{i}.png" for i in range(count)]
    for picture in (*references, *tests):
        picture.write_bytes(bytes([0x89]) + b"PNG")

    def fake_pairs(*_args, **_kwargs):
        yield from zip(references, tests, strict=True)

    monkeypatch.setattr(perceptual_cpu, "find_metric_executable", lambda key: key)
    monkeypatch.setattr(perceptual_cpu, "_png_pairs", pairs or fake_pairs)
    monkeypatch.setattr(perceptual_cpu, "_tool_version", lambda executable: "test-tool 1")
    if workers is not None:
        monkeypatch.setattr(perceptual_cpu, "scoring_workers", lambda *a, **k: workers)
    monkeypatch.setattr(perceptual_cpu, "_run_metric", run_metric)
    return run_perceptual_task(_info("source.mp4"), _info("test.mp4"), request, request.metrics, **kwargs)


def _pair_number(picture: Path) -> int:
    return int(picture.stem.split("-")[1])


def test_scores_stay_in_the_pairs_order_when_later_pairs_finish_first(tmp_path, monkeypatch):
    """Each pair waits for the one after it to finish: only pairs scored
    at the same time can end at all, and they end last to first."""
    done = [threading.Event() for _ in range(5)]
    done[4].set()
    progress = []

    def run_metric(executable, key, reference, test, *_args):
        number = _pair_number(reference)
        assert done[number + 1].wait(30), "the next pair is not being scored at the same time"
        if key == "butteraugli":  # the pair's last tool
            done[number].set()
        return float(number) if key == "ssimulacra2" else number / 10

    output = _parallel_task(tmp_path, monkeypatch, 4, 4, run_metric, keys=("ssimulacra2", "butteraugli"),
                            on_progress=lambda current, total, fps: progress.append(current))

    assert output.metrics.frame("ssimulacra2").values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert output.metrics.frame("butteraugli").values.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert output.metrics.frame("ssimulacra2").frame.tolist() == [0, 1, 2, 3]
    assert progress == [1, 2, 3, 4, 4]  # by how many are scored, then the end
    assert not list(tmp_path.glob("*.png"))  # each pair deleted once scored


def test_cancel_ends_every_pair_being_scored(tmp_path, monkeypatch):
    cancel = threading.Event()
    both = threading.Barrier(2)
    ended = []

    def run_metric(executable, key, reference, test, process_handle, cancel_event, hdr):
        both.wait(30)
        cancel.set()  # the user's Cancel, with two pairs in progress
        assert cancel_event.is_set()
        ended.append(_pair_number(reference))
        raise perceptual_cpu.PerceptualCancelled("Cancelled by user")

    with pytest.raises(perceptual_cpu.PerceptualCancelled):
        _parallel_task(tmp_path, monkeypatch, 4, 2, run_metric, cancel_event=cancel)

    assert sorted(ended) == [0, 1]  # the pairs after them were never started


def test_cpu_frame_extraction_applies_duration_limit_to_both_outputs(tmp_path, monkeypatch):
    from dataclasses import replace

    from vmaf_app.core.perceptual_cpu import _png_pairs

    request = _request()
    recipe = replace(request.recipe, duration_limit=1.0)
    command = []

    class FinishedProcess:
        pid = 123
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(args, **_kwargs):
        command.extend(args)
        (tmp_path / "test-00000001.png").touch()
        (tmp_path / "reference-00000001.png").touch()
        return FinishedProcess()

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.proc_util.popen", fake_popen)

    list(_png_pairs(
        _info("source.mp4"), _info("test.mp4"), recipe, None, None, 1,
        tmp_path, None, None,
    ))

    assert command.count("-t") == 2
    for pattern in ("test-%08d.png", "reference-%08d.png"):
        output_index = next(i for i, value in enumerate(command) if value.endswith(pattern))
        assert command[output_index - 8:output_index] == [
            "-t", "1.000", "-fps_mode", "passthrough", "-pix_fmt", "rgb48le", "-atomic_writing", "1",
        ]


def test_cpu_extraction_of_different_lengths_keeps_the_frames_both_have(tmp_path, monkeypatch):
    """Each FFmpeg output runs to its own input's end. Unequal counts were
    rejected as "unmatched frame pairs" after the whole video had been
    extracted; the overlap is now scored, as libvmaf does. (The extra images
    go with the task's temporary folder.)"""
    from vmaf_app.core.perceptual_cpu import _png_pairs

    class FinishedProcess:
        pid = 123
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(args, **_kwargs):
        for index in range(1, 4):
            (tmp_path / f"reference-{index:08d}.png").touch()
        for index in range(1, 6):
            (tmp_path / f"test-{index:08d}.png").touch()
        return FinishedProcess()

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.proc_util.popen", fake_popen)
    pairs = list(_png_pairs(
        _info("source.mp4"), _info("test.mp4"), _request().recipe, None, None, 1, tmp_path, None, None,
    ))
    assert [(r.name, t.name) for r, t in pairs] == [
        (f"reference-{i:08d}.png", f"test-{i:08d}.png") for i in range(1, 4)
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="elsewhere, a file held open does not keep others from reading it")
def test_a_pair_is_handed_out_once_ffmpeg_has_let_go_of_its_pictures(tmp_path, monkeypatch):
    """On Windows, FFmpeg can still hold a picture for a moment after giving
    it its name, and a picture opened then could not be read: the task
    failed. The stand-in FFmpeg holds the first pair's second picture, with
    delete access as FFmpeg does, until it has been asked three times
    whether it is still running."""
    import ctypes
    from ctypes import wintypes

    from vmaf_app.core.perceptual_cpu import _png_pairs

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    held = []

    class Running:
        pid = 123  # never used: the process helpers below are stand-ins too
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 3:
                kernel32.CloseHandle(held.pop())
            return None if self.polls <= 3 else 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(args, **_kwargs):
        (tmp_path / "reference-00000001.png").write_bytes(b"source picture")
        (tmp_path / "test-00000001.png").write_bytes(b"test picture")
        delete, share_all, open_existing = 0x10000, 0x7, 3
        handle = kernel32.CreateFileW(str(tmp_path / "test-00000001.png"), delete, share_all, None, open_existing, 0,
                                      None)
        assert handle != wintypes.HANDLE(-1).value
        held.append(handle)
        return Running()

    monkeypatch.setattr(perceptual_cpu.proc_util, "popen", fake_popen)
    monkeypatch.setattr(perceptual_cpu.proc_util, "process_root", lambda pid: [])
    monkeypatch.setattr(perceptual_cpu.proc_util, "process_tree", lambda pid: [])
    monkeypatch.setattr(perceptual_cpu.proc_util, "terminate", lambda process: None)
    pairs = _png_pairs(_info("source.mp4"), _info("test.mp4"), _request().recipe, None, None, 1, tmp_path, None, None)
    try:
        _, test = next(pairs)
        assert not held  # handed out once FFmpeg had let go
        assert test.read_bytes() == b"test picture"
    finally:
        pairs.close()
        for handle in held:
            kernel32.CloseHandle(handle)


def test_pictures_are_converted_with_the_matrix_vship_reads_the_video_with():
    """FFmpeg converts an untagged video with BT.601's matrix; Vship takes an
    HD one as BT.709. The CPU tools' pictures are converted as Vship reads
    the video, before they become RGB."""
    from vmaf_app.core.perceptual_cpu import _image_filtergraph

    hd = VideoInfo(Path("hd.mkv"), 1920, 1080, 24.0, 1.0, 24, "h264", pix_fmt="yuv420p")
    full = VideoInfo(Path("j.mkv"), 640, 480, 24.0, 1.0, 24, "mjpeg", pix_fmt="yuvj420p")
    graph = _image_filtergraph(full, hd, _request().recipe, None, None, 1)
    distorted, reference = graph.split(";")
    assert "setparams=colorspace=bt709:range=tv,scale,format=rgb48le" in distorted
    assert "setparams=colorspace=bt470bg:range=pc,scale,format=rgb48le" in reference


def test_butteraugli_is_vships_3_norm_on_vships_display(tmp_path, monkeypatch):
    """The tool's own "3-norm" (the mean of the 3-, 6- and 12-norms, at 80
    nits) read about 1.8 times Vship's 3-norm (at 203) on the same frames."""
    import numpy as np

    from vmaf_app.core import perceptual_cpu

    seen = []

    class Done:
        pid, returncode = 7, 0

        def communicate(self, timeout=None):
            return "3-norm: 9.99\n", ""

    def popen(command, **_kwargs):
        seen.append(command)
        path = Path(command[command.index("--rawdistmap") + 1])
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4")
        path.write_bytes(b"Pf\n2 2\n-1.0\n" + values.tobytes())
        return Done()

    monkeypatch.setattr(perceptual_cpu.proc_util, "popen", popen)
    reference, test = tmp_path / "r.png", tmp_path / "t.png"
    score = perceptual_cpu._run_metric("butteraugli_main", "butteraugli", reference, test)
    assert score == pytest.approx((np.mean([1, 8, 27, 64])) ** (1 / 3))
    assert seen[0][seen[0].index("--intensity_target") + 1] == "203"
    assert not (tmp_path / "t-distortion.pfm").exists()
    perceptual_cpu._run_metric("butteraugli_main", "butteraugli", reference, test, hdr=True)
    assert "--intensity_target" not in seen[1]  # an HDR picture's brightness is its own


@pytest.mark.parametrize("step", [1, 3])
def test_the_images_are_each_videos_frame_n_cut_at_the_limit(tmp_path, step):
    """Through real FFmpeg, the app's command: two videos whose frame N is
    the same picture (a grey of its own), the source at 20 fps, the test
    video at 23.976 with a millisecond of jitter. Each pair is the two
    videos' frame N (frame_sync), every step-th, and the 1.5 s limit keeps
    the same frames of both: the test video's 36, not the source's 30."""
    import subprocess
    from dataclasses import replace

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.frame_sync import frame_rate, frames_within

    run = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"]
    pictures = "geq=lum='16+4*N':cb=128:cr=128,format=yuv420p"
    subprocess.run([*run, "-i", "nullsrc=s=64x48:r=20:d=3", "-vf", pictures, "-c:v", "ffv1",
                    str(tmp_path / "source.mkv")], check=True)
    subprocess.run([*run, "-i", "nullsrc=s=64x48:r=24000/1001:d=2", "-vf",
                    f"{pictures},settb=1/1000,setpts='round(N*1001/24)+mod(N\\,2)'", "-fps_mode", "passthrough",
                    "-enc_time_base", "filter", "-c:v", "ffv1", str(tmp_path / "test.mkv")], check=True)
    source, test = probe_video(tmp_path / "source.mkv"), probe_video(tmp_path / "test.mkv")
    assert (frame_rate(source), frame_rate(test)) == (20, Fraction(24000, 1001))
    recipe = replace(_request().recipe, duration_limit=1.5)
    images = tmp_path / "images"
    images.mkdir()

    pairs = []
    for reference, distorted in perceptual_cpu._png_pairs(source, test, recipe, None, None, step, images, None, None):
        pairs.append((reference.read_bytes(), distorted.read_bytes()))
        reference.unlink()
        distorted.unlink()

    kept = frames_within("1.500", frame_rate(test))
    assert kept == 36
    assert len(pairs) == len(range(0, kept, step))
    assert all(reference == distorted for reference, distorted in pairs)
    assert len({reference for reference, _ in pairs}) == len(pairs)  # each pair a different picture


def _frame_hashes(inputs: list[str], graph: str) -> dict[str, list[str]]:
    import subprocess
    import tempfile

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path

    with tempfile.TemporaryDirectory() as directory:
        files = {label: Path(directory) / f"{label}.txt" for label in ("distorted", "reference")}
        command = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", *inputs, "-filter_complex", graph]
        for label, path in files.items():
            command += ["-map", f"[{label}]", "-fps_mode", "passthrough", "-f", "framemd5", str(path)]
        subprocess.run(command, capture_output=True, check=True)
        return {label: [line.rsplit(",", 1)[-1].strip() for line in path.read_text().splitlines()
                        if line and not line.startswith("#")] for label, path in files.items()}


@pytest.mark.parametrize("step, pairs", [(1, 23), (3, 8)])
def test_pairing_by_timestamp_a_frame_dropped_from_the_test_video_pairs_the_rest_by_time(
        tmp_path, monkeypatch, step, pairs):
    """With VML_FRAME_PAIRING=timestamp, as before: the test video is the
    source with its sixth frame dropped, the others' times kept. Paired by
    position, every frame after the gap was compared with the source's next
    one; by time, as libvmaf pairs them, each frame is compared with itself."""
    import subprocess

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.perceptual_cpu import _image_filtergraph

    monkeypatch.setenv("VML_FRAME_PAIRING", "timestamp")
    run = [ffmpeg_path(), "-hide_banner", "-loglevel", "error"]
    subprocess.run([*run, "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24:duration=1", "-vf", "format=yuv420p10le",
                    "-c:v", "ffv1", str(tmp_path / "source.mkv")], check=True)
    subprocess.run([*run, "-i", str(tmp_path / "source.mkv"), "-vf", "select='not(eq(n,5))'", "-fps_mode",
                    "passthrough", "-c:v", "ffv1", str(tmp_path / "test.mkv")], check=True)
    source, test = probe_video(tmp_path / "source.mkv"), probe_video(tmp_path / "test.mkv")
    graph = _image_filtergraph(source, test, _request().recipe, None, None, step)
    assert "blend=all_mode=or:shortest=1:repeatlast=0:ts_sync_mode=nearest" in graph

    hashes = _frame_hashes(["-i", str(test.path), "-i", str(source.path)], graph)

    assert len(hashes["distorted"]) == len(hashes["reference"]) == pairs
    assert hashes["distorted"] == hashes["reference"]
