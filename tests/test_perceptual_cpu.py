from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.factories import STDLIB_PYTHON
from vmaf_app.core import perceptual_cpu
from vmaf_app.core.analysis_request import AnalysisRequest, ExecutionPreferences, FrameCoverage, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_cache import load_metric, load_metrics, store_metric
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.metrics import METRIC_BY_KEY, MetricDirection, MetricKind
from vmaf_app.core.models import CropMode, GpuVendor, ScaleDirection, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualRunError, parse_score, run_perceptual_task


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


def test_perceptual_metrics_are_registered_without_ffmpeg_bindings():
    assert METRIC_BY_KEY["ssimulacra2"].kind is MetricKind.FRAME
    assert METRIC_BY_KEY["ssimulacra2"].direction is MetricDirection.HIGHER_IS_BETTER
    assert METRIC_BY_KEY["butteraugli"].direction is MetricDirection.LOWER_IS_BETTER
    assert METRIC_BY_KEY["ssimulacra2"].ffmpeg_binding is None
    assert METRIC_BY_KEY["butteraugli"].ffmpeg_binding is None


def test_mixed_request_groups_perceptual_metrics_separately():
    options = VmafOptions(extra_features=["name=psnr"], compute_vmaf=True)
    request = analysis_request_from_vmaf_options(options, ("vmaf", "psnr", "ssimulacra2", "butteraugli"))
    plan = build_execution_plan(request)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("ffmpeg", ("vmaf", "psnr")),
        ("perceptual", ("ssimulacra2", "butteraugli")),
    ]


@pytest.mark.parametrize(("text", "expected"), [
    ("SSIMULACRA2: 91.75", 91.75),
    ("butteraugli score = 0.2345", 0.2345),
])
def test_perceptual_output_parser_uses_final_scalar(text, expected):
    assert parse_score("ssimulacra2", text) == expected


def test_perceptual_output_parser_rejects_malformed_output():
    with pytest.raises(PerceptualRunError, match="numeric score"):
        parse_score("butteraugli", "comparison failed")


def test_backend_returns_independent_frame_results_without_real_tools(tmp_path, monkeypatch):
    request = _request("ssimulacra2", "butteraugli")
    references = [tmp_path / f"r-{i}.png" for i in range(3)]
    tests = [tmp_path / f"t-{i}.png" for i in range(3)]
    for picture in (*references, *tests):
        picture.write_bytes(bytes([0x89]) + b"PNG")  # their colours are described before scoring
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.find_metric_executable", lambda key: key)
    def fake_pairs(*_args):
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


def test_auto_crop_detection_uses_full_video_not_score_duration(monkeypatch):
    from dataclasses import replace

    from vmaf_app.core.perceptual_cpu import _resolve_crops

    request = _request()
    recipe = replace(request.recipe, crop_mode=CropMode.AUTO, duration_limit=1.0)
    calls = []

    def fake_detect(info, **kwargs):
        calls.append((info.path.name, kwargs))
        return None

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.detect_crop", fake_detect)

    _resolve_crops(_info("source.mp4"), _info("test.mp4"), recipe, None, None, None)

    assert sorted(name for name, _kwargs in calls) == ["source.mp4", "test.mp4"]
    assert all("duration_limit" not in kwargs for _name, kwargs in calls)


def test_cached_backend_does_not_suppress_missing_backend():
    options = VmafOptions(compute_vmaf=True)
    request = analysis_request_from_vmaf_options(options, ("vmaf", "ssimulacra2"))
    cached = MetricResultSet([
        FrameMetricResult("vmaf", np.array([0]), np.array([0.0]), np.array([90.0]),
                          MetricProvenance("test", "1", "cpu", "test")),
    ])
    plan = build_execution_plan(request, cached)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("perceptual", ("ssimulacra2",)),
    ]


def test_perceptual_cache_entries_are_independent_and_coverage_specific(tmp_path):
    full, sampled = (
        MetricRequestSpec("ssimulacra2", "perceptual", (), FrameCoverage(mode, step), "ssimulacra2-reference-cli-v1")
        for mode, step in (("full", 1), ("sampled", 2))
    )
    butter = MetricRequestSpec("butteraugli", "perceptual", (), FrameCoverage("full", 1), "butteraugli-reference-cli-v1")
    provenance = MetricProvenance("test", "1", "cpu", "ssimulacra2-reference-cli-v1")
    store_metric(tmp_path, FrameMetricResult("ssimulacra2", [0], [0.0], [90.0], provenance), full)
    store_metric(tmp_path, FrameMetricResult("butteraugli", [0], [0.0], [0.2], provenance), butter)
    loaded = load_metrics(tmp_path, (full, sampled, butter))
    assert loaded.has("ssimulacra2") and loaded.has("butteraugli")
    # The sampled identity has a different filename/key and cannot borrow a
    # full-coverage score accidentally.
    assert len(list(tmp_path.glob("ssimulacra2_*.npz"))) == 1
    assert load_metric(tmp_path, sampled) is None


def test_ui_selects_cpu_metric_without_extending_vmaf_options(tmp_path):
    from PySide6.QtWidgets import QApplication

    from vmaf_app.ui.main_window import COL_SSIMULACRA2, MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    row = window._add_table_row(tmp_path / "test.mp4")
    window._apply_metric_selection([row], COL_SSIMULACRA2, True, set_default=False)
    assert "ssimulacra2" in window._requested_metrics(window._rows[row])
    assert not hasattr(window._rows[row].options, "compute_ssimulacra2")



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


def _waiting_tool(tmp_path) -> tuple[str, Path, Path]:
    """A stand-in for ssimulacra2 that prints its score once the file
    "finish" exists beside it, and never otherwise. Returned as
    (executable, "reference", "test") for _run_metric's argument order.

    Run by the real interpreter, not a virtual environment's python.exe:
    that is a launcher starting the real one as its child, and a pause
    landing while it creates that child makes Windows refuse the creation
    ("Unable to create process ... Access is denied", exit code 101) -- 19
    of 240 paused starts under a parallel run's load. The real tools are
    single processes."""

    script = tmp_path / "tool.py"
    script.write_text("import os, time\n"
                      f"while not os.path.exists({str(tmp_path / 'finish')!r}):\n"
                      "    time.sleep(0.01)\n"
                      "print('score: 42.5')\n", encoding="utf-8")
    return STDLIB_PYTHON, script, tmp_path / "unused.png"


def _attached(handle, worker) -> int:
    """The tool's process, once `handle` holds it: attach suspends a new
    process of a paused job under the same lock."""
    import time

    for _ in range(3000):  # a bound against a hang, not a timing assertion
        with handle._lock:
            if handle._pids:
                return next(iter(handle._pids))
        assert worker.is_alive(), "the tool ended before it was attached"
        time.sleep(0.01)
    raise AssertionError("the tool was never attached to the job's handle")


def test_pause_suspends_a_cpu_tool_and_resume_lets_it_finish(tmp_path):
    """The tools used to run outside the job's pause handle: Pause left them
    scoring while the app said Paused."""
    import threading

    import psutil

    from vmaf_app.core.perceptual_cpu import _run_metric
    from vmaf_app.core.process_control import ProcessHandle

    executable, script, other = _waiting_tool(tmp_path)
    handle = ProcessHandle()
    handle.pause()
    out = []
    worker = threading.Thread(target=lambda: out.append(_run_metric(executable, "ssimulacra2", script, other, handle)))
    worker.start()
    pid = _attached(handle, worker)
    (tmp_path / "finish").touch()  # would end it, were it running
    assert psutil.Process(pid).status() == psutil.STATUS_STOPPED, "the tool ran on while the job was paused"
    handle.resume()
    worker.join(30.0)
    assert out == [42.5]


def test_a_tool_that_never_finishes_a_frame_times_out(tmp_path, monkeypatch):
    import itertools
    import time
    from types import SimpleNamespace

    from vmaf_app.core import perceptual_cpu
    from vmaf_app.core.process_control import ProcessHandle

    clock = itertools.count(0.0, 100.0)  # each look at the clock, 100 s later
    monkeypatch.setattr(perceptual_cpu, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=time.sleep))
    never, script, other = _waiting_tool(tmp_path)
    with pytest.raises(perceptual_cpu.PerceptualRunError, match="did not finish a frame within 120 s"):
        perceptual_cpu._run_metric(never, "ssimulacra2", script, other, ProcessHandle())

def test_one_sequence_running_far_ahead_does_not_stall_the_extraction(tmp_path, monkeypatch):
    """The two image sequences come from two decoders; a fast one can run
    well ahead (a 4K AV1 test ran 24 frames ahead of its HEVC reference).
    Throttling on either side alone suspended FFmpeg while the side the
    scorer was waiting for still lagged: a deadlock. A real child process
    stands in for FFmpeg: all test images first, then the references
    slowly, each written atomically."""
    import subprocess
    import threading

    from vmaf_app.core import perceptual_cpu

    monkeypatch.setattr(perceptual_cpu, "_BACKLOG_PAIRS", 6)
    writer = (
        "import os, sys, time\n"
        "d = sys.argv[1]\n"
        "def put(name):\n"
        "    open(os.path.join(d, name + '.tmp'), 'wb').close()\n"
        "    os.replace(os.path.join(d, name + '.tmp'), os.path.join(d, name))\n"
        "for i in range(1, 41): put(f'test-{i:08d}.png')\n"
        "for i in range(1, 41):\n"
        "    put(f'reference-{i:08d}.png'); time.sleep(0.005)\n"
    )
    monkeypatch.setattr(perceptual_cpu.proc_util, "popen",
                        lambda _cmd, **kwargs: subprocess.Popen([STDLIB_PYTHON, "-S", "-c", writer, str(tmp_path)], **kwargs))
    pairs = []

    def consume():
        for reference, test in perceptual_cpu._png_pairs(
            _info("source.mp4"), _info("test.mp4"), _request().recipe, None, None, 1, tmp_path, None, None,
        ):
            pairs.append(reference.name)
            reference.unlink()
            test.unlink()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(30)
    assert not consumer.is_alive(), f"the extraction stalled after {len(pairs)} pairs"
    assert len(pairs) == 40


def test_a_saved_perceptual_metric_is_not_recalculated_beside_a_new_one():
    """Ticking CVVDP on a video whose SSIMULACRA2 was already scored on the
    CPU recalculated SSIMULACRA2 too (days for a film, with no warning):
    the three perceptual metrics ran as one all-or-nothing group."""
    from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet

    request = analysis_request_from_vmaf_options(
        VmafOptions(), ("vmaf", "ssimulacra2", "butteraugli", "cvvdp"), {"ssimulacra2": "cpu"})
    provenance = MetricProvenance("libjxl", "0.12", "cpu", "ssimulacra2-libjxl-cpu-v1")
    cached = MetricResultSet([FrameMetricResult(key, [0], [0.0], [1.0], provenance)
                              for key in ("vmaf", "ssimulacra2")])
    plan = build_execution_plan(request, cached)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("perceptual", ("butteraugli", "cvvdp")),
    ]
    assert [spec.key for spec in plan.tasks[0].requested_specs] == ["butteraugli", "cvvdp"]



def test_a_failed_frame_extraction_says_what_ffmpeg_said(tmp_path, monkeypatch):
    """FFmpeg's errors went nowhere, so a failed extraction had no reason."""
    from vmaf_app.core.perceptual_cpu import PerceptualRunError, _png_pairs

    class FailedProcess:
        pid = 123
        returncode = 1

        @staticmethod
        def poll():
            return 1

    def fake_popen(args, stderr=None, **_kwargs):
        stderr.write(b"[matroska] Invalid EBML number, skipping\nError opening input file\n")
        return FailedProcess()

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.proc_util.popen", fake_popen)
    with pytest.raises(PerceptualRunError) as raised:
        list(_png_pairs(_info("source.mp4"), _info("test.mp4"), _request().recipe, None, None, 1,
                        tmp_path, None, None))
    assert "could not prepare" in str(raised.value)
    assert "Error opening input file" in raised.value.stderr_tail


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


def _rgb_frames(command_inputs: list[str], graph: str, label: str) -> bytes:
    import subprocess

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path

    command = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", *command_inputs, "-filter_complex", graph,
               "-map", f"[{label}]", "-f", "rawvideo", "-"]
    for other in {"distorted", "reference"} - {label}:
        if f"[{other}]" in graph:
            command += ["-map", f"[{other}]", "-f", "null", "-"]
    return subprocess.run(command, capture_output=True, check=True).stdout


def test_a_scaled_untagged_hd_source_is_converted_with_bt709(tmp_path):
    """Converted while FFmpeg scaled it -- before the tags were set -- an
    untagged HD source scaled to its encode's size was made RGB with
    BT.601's matrix. Scaled in its own format, it is converted after."""
    import subprocess

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.perceptual_cpu import _image_filtergraph

    clips = {"source.mkv": "96x656", "test.mkv": "48x328"}
    for name, size in clips.items():
        subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size={size}:rate=24:duration=0.125", "-vf", "format=yuv420p",
                        "-c:v", "ffv1", str(tmp_path / name)], check=True)
    source, test = probe_video(tmp_path / "source.mkv"), probe_video(tmp_path / "test.mkv")
    inputs = ["-i", str(test.path), "-i", str(source.path)]
    graph = _image_filtergraph(source, test, _request().recipe, None, None, 1)
    made = _rgb_frames(inputs, graph, "reference")

    def converted(matrix: str) -> bytes:
        return _rgb_frames(inputs, "[0:V:0]nullsink;[1:V:0]scale=48:328:flags=bicubic,format=yuv420p,"
                                   f"scale=in_color_matrix={matrix}:in_range=tv,format=rgb48le[reference]",
                           "reference")

    assert made == converted("bt709")
    assert made != converted("bt601")


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
def test_a_frame_dropped_from_the_test_video_pairs_the_rest_by_time(tmp_path, step, pairs):
    """The test video is the source with its sixth frame dropped, the others'
    times kept. Paired by position, every frame after the gap was compared
    with the source's next one; by time, as libvmaf pairs them, each frame
    is compared with itself."""
    import subprocess

    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.perceptual_cpu import _image_filtergraph

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


@pytest.mark.parametrize("size, pix_fmt", [("63x48", "yuv420p"), ("64x47", "yuv420p10le"), ("63x47", "yuv422p"),
                                           ("63x47", "yuv444p"), ("64x48", "yuv420p")])
def test_videos_of_an_odd_size_are_paired_with_their_pictures_unchanged(tmp_path, size, pix_fmt):
    """The pairing's clock was padded to the pictures' size, and FFmpeg's pad
    gives a subsampled picture an even one: at an odd width or height blend
    refused its two inputs ("size 852x480 do not match ... 853x480") and the
    CPU tools failed. The pictures are the ones pairing by position gives."""
    import re
    import subprocess

    from vmaf_app.core import perceptual_cpu
    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video

    run = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           "testsrc2=size=128x96:rate=24:duration=0.25"]
    width, height = size.split("x")
    for name, noise in (("source.mkv", ""), ("test.mkv", "noise=alls=3:allf=t,")):
        subprocess.run([*run, "-vf", f"{noise}scale={width}:{height},format={pix_fmt}", "-c:v", "ffv1",
                        str(tmp_path / name)], check=True)
    source, test = probe_video(tmp_path / "source.mkv"), probe_video(tmp_path / "test.mkv")
    assert (source.width, source.height) == (int(width), int(height))
    inputs = ["-i", str(test.path), "-i", str(source.path)]
    graph = perceptual_cpu._image_filtergraph(source, test, _request().recipe, None, None, 1)
    assert "blend=" in graph

    paired = _frame_hashes(inputs, graph)

    by_position = pytest.MonkeyPatch()
    by_position.setattr(perceptual_cpu, "_BLEND_FORMAT", re.compile("never"))
    try:
        unpaired = _frame_hashes(inputs, perceptual_cpu._image_filtergraph(source, test, _request().recipe, None,
                                                                           None, 1))
    finally:
        by_position.undo()
    assert len(paired["reference"]) == 6
    assert paired == unpaired


@pytest.mark.parametrize("change", [{"pix_fmt": "nv12"}, {"pix_fmt": "yuvj420p"}, {"color_space": "reserved"}])
def test_a_source_blend_cannot_carry_unchanged_is_paired_by_position(change):
    from dataclasses import replace

    from vmaf_app.core.perceptual_cpu import _image_filtergraph

    source = replace(VideoInfo(Path("source.mkv"), 1920, 1080, 24.0, 1.0, 24, "h264", pix_fmt="yuv420p"), **change)
    test = VideoInfo(Path("test.mkv"), 1920, 1080, 24.0, 1.0, 24, "h264", pix_fmt="yuv420p")
    graph = _image_filtergraph(source, test, _request().recipe, None, None, 1)

    assert "blend" not in graph
    assert graph.count(";") == 1  # a chain each
