from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core.analysis_request import AnalysisRequest, ExecutionPreferences, FrameCoverage, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_cache import load_metric, load_metrics, store_metric
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.metrics import METRIC_BY_KEY, MetricDirection, MetricKind
from vmaf_app.core.models import CropMode, GpuVendor, ScaleDirection, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualRunError, parse_score, run_perceptual_task


def _info(path: str) -> VideoInfo:
    return VideoInfo(Path(path), 64, 48, 24.0, 1.0, 24, "h264")


def _request(*keys: str) -> AnalysisRequest:
    return AnalysisRequest(
        recipe=ComparisonRecipe(CropMode.NONE, None, None, "bicubic", ScaleDirection.SOURCE_TO_DISTORTED, 0.0, None),
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


def _slow_tool(tmp_path, seconds: float) -> tuple[str, Path, Path]:
    """A stand-in for ssimulacra2: sleeps, then prints a score. Returned as
    (executable, "reference", "test") for _run_metric's argument order."""
    import sys

    script = tmp_path / "tool.py"
    script.write_text(f"import time\ntime.sleep({seconds})\nprint('score: 42.5')\n", encoding="utf-8")
    return sys.executable, script, tmp_path / "unused.png"


def test_pause_suspends_a_cpu_tool_and_resume_lets_it_finish(tmp_path):
    """The tools used to run outside the job's pause handle: Pause left them
    scoring while the app said Paused."""
    import threading

    from vmaf_app.core.perceptual_cpu import _run_metric
    from vmaf_app.core.process_control import ProcessHandle

    executable, script, other = _slow_tool(tmp_path, 0.5)
    handle = ProcessHandle()
    handle.pause()
    out = []
    worker = threading.Thread(target=lambda: out.append(_run_metric(executable, "ssimulacra2", script, other, handle)))
    worker.start()
    worker.join(2.0)
    assert worker.is_alive(), "the tool ran on while the job was paused"
    handle.resume()
    worker.join(10.0)
    assert out == [42.5]


def test_cancel_ends_a_cpu_tool_at_once(tmp_path):
    import threading
    import time

    from vmaf_app.core.perceptual_cpu import PerceptualCancelled, _run_metric
    from vmaf_app.core.process_control import ProcessHandle

    executable, script, other = _slow_tool(tmp_path, 30)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(PerceptualCancelled):
        _run_metric(executable, "ssimulacra2", script, other, ProcessHandle(), cancel)
    assert time.monotonic() - started < 5


def test_time_paused_does_not_count_towards_the_tool_timeout(tmp_path, monkeypatch):
    import threading

    from vmaf_app.core import perceptual_cpu
    from vmaf_app.core.process_control import ProcessHandle

    monkeypatch.setattr(perceptual_cpu, "_TOOL_TIMEOUT_SECONDS", 1.0)
    executable, script, other = _slow_tool(tmp_path, 0.3)
    handle = ProcessHandle()
    handle.pause()
    threading.Timer(2.0, handle.resume).start()  # paused for twice the timeout
    assert perceptual_cpu._run_metric(executable, "ssimulacra2", script, other, handle) == 42.5

    slow, script, other = _slow_tool(tmp_path, 5)
    with pytest.raises(perceptual_cpu.PerceptualRunError, match="did not finish a frame"):
        perceptual_cpu._run_metric(slow, "ssimulacra2", script, other, ProcessHandle())


def test_cpu_scoring_streams_with_a_small_backlog_and_live_progress(tmp_path, monkeypatch):
    """With real FFmpeg: frames are scored while extraction is still going,
    FFmpeg is held to a small backlog instead of writing the whole video
    out first, and progress moves from the first pair."""
    import subprocess
    import tempfile
    import threading
    import time

    from vmaf_app.core import perceptual_cpu
    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    from vmaf_app.core.ffprobe import probe_video

    exe = ffmpeg_path()
    clip = tmp_path / "clip.mkv"
    subprocess.run([exe, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24",
                    "-frames:v", "240", "-c:v", "ffv1", str(clip)], check=True, capture_output=True, timeout=120)
    info = probe_video(clip)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(work))
    monkeypatch.setattr(perceptual_cpu, "_BACKLOG_PAIRS", 6)
    monkeypatch.setattr(perceptual_cpu, "find_metric_executable", lambda key: key)
    monkeypatch.setattr(perceptual_cpu, "_tool_version", lambda executable: "test")

    def slow_score(*_args):
        time.sleep(0.02)
        return 50.0

    monkeypatch.setattr(perceptual_cpu, "_run_metric", slow_score)
    most_images, stop = [0], threading.Event()

    def watch():
        while not stop.is_set():
            most_images[0] = max(most_images[0], sum(1 for _ in work.rglob("*.png")))
            time.sleep(0.005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    progress = []
    request = _request("ssimulacra2")
    try:
        output = perceptual_cpu.run_perceptual_task(
            info, info, request, request.metrics, resolved_crops=(None, None),
            on_progress=lambda done, total, rate: progress.append((done, total, rate)),
        )
    finally:
        stop.set()
        watcher.join()

    assert output.compared_frame_count == 240
    assert len(output.metrics.get("ssimulacra2").values) == 240
    # The backlog limit (6 pairs) plus the frames FFmpeg already has in its
    # queues when it resumes -- traced at 7-10 at 720p, arriving within
    # ~10 ms, before the next check can suspend it again. Bounded by that,
    # not by the video's length: writing everything first meant 480 images.
    assert most_images[0] <= 2 * (6 + 20), most_images[0]
    assert [done for done, _total, _rate in progress[:3]] == [1, 2, 3]
    assert progress[0][1] == 240 and progress[0][2] > 0


def test_one_sequence_running_far_ahead_does_not_stall_the_extraction(tmp_path, monkeypatch):
    """The two image sequences come from two decoders; a fast one can run
    well ahead (a 4K AV1 test ran 24 frames ahead of its HEVC reference).
    Throttling on either side alone suspended FFmpeg while the side the
    scorer was waiting for still lagged: a deadlock. A real child process
    stands in for FFmpeg: all test images first, then the references
    slowly, each written atomically."""
    import subprocess
    import sys
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
        "    put(f'reference-{i:08d}.png'); time.sleep(0.02)\n"
    )
    monkeypatch.setattr(perceptual_cpu.proc_util, "popen",
                        lambda _cmd, **kwargs: subprocess.Popen([sys.executable, "-c", writer, str(tmp_path)], **kwargs))
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


def test_the_backlog_holds_when_ffmpeg_is_started_through_a_launcher(tmp_path, monkeypatch):
    """GitHub's runner installs FFmpeg with Chocolatey, whose ffmpeg.exe is a
    launcher that starts the real one as a child. Suspending the launcher
    left FFmpeg writing: the runner saw 364 images waiting where the limit
    is 52 (444 here). The throttle now suspends the whole process tree."""
    import sys

    from vmaf_app.core import proc as proc_util

    real = proc_util.popen
    launcher = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"
    monkeypatch.setattr(proc_util, "popen",
                        lambda command, **kwargs: real([sys.executable, "-c", launcher, *command], **kwargs))
    test_cpu_scoring_streams_with_a_small_backlog_and_live_progress(tmp_path, monkeypatch)


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

