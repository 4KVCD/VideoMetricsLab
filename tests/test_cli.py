"""The command line (vmaf_app.cli): the app's runs without its window. The
scheduler is the real one; FFmpeg's and Vship's halves are faked, as in
test_worker.py."""
from __future__ import annotations

import gc
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmaf_app import __version__, cli
from vmaf_app.core import job_runner, perceptual_vship, result_cache
from vmaf_app.core.cvvdp import default_settings
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_cache import CPU_COLOR_TAGS
from vmaf_app.core.metric_results import (
    FrameMetricResult,
    MetricProvenance,
    MetricResultSet,
    current_ffmpeg_provenance,
    results_from_frame_scores,
)
from vmaf_app.core.models import ComparisonResult, FrameScore, FrameScores, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualRunError, PerceptualTaskOutput
from vmaf_app.core.settings import Settings
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError


def _info(path: Path, fps: float = 30.0) -> VideoInfo:
    return VideoInfo(path=Path(path), width=1920, height=1080, fps=fps, duration=5.0, nb_frames=150,
                     codec_name="h264", pix_fmt="yuv420p")


def _result(source: VideoInfo, distorted: VideoInfo, options: VmafOptions, vmaf: float = 90.0) -> ComparisonResult:
    """A run's result as the runner makes it, each score with its provenance."""
    frames = FrameScores.from_frames([FrameScore(frame=i, time=i / 30.0, vmaf=vmaf + i if options.compute_vmaf else None,
                                                 psnr=40.0 + i, ssim=0.99, xpsnr=35.0) for i in range(10)])
    return ComparisonResult(
        source=source.path, distorted=distorted.path, frames=frames, fps=30.0, model=options.model,
        source_crop=None, distorted_crop=None, source_info=source, distorted_info=distorted,
        compared_frame_count=10,
        metric_results=results_from_frame_scores(
            frames, {key: current_ffmpeg_provenance(key, "9.0") for key in frames.metric_keys}),
    )


def _perceptual(key: str = "ssimulacra2", backend: str = "gpu") -> PerceptualTaskOutput:
    provenance = (MetricProvenance(f"Vship/{key}", "5.1.2", "gpu", f"{key}-vship-gpu-v1") if backend == "gpu" else
                  MetricProvenance(key, "", "cpu", f"{key}-libjxl-cpu-v1", {"color_tags": CPU_COLOR_TAGS}))
    metric = FrameMetricResult(key, [0, 1], [0.0, 1 / 30], [87.0, 85.0], provenance)
    return PerceptualTaskOutput(MetricResultSet([metric]), None, None, 2)


@pytest.fixture
def videos(tmp_path, monkeypatch):
    """A reference and two test files, with FFmpeg found and both halves of
    a run faked. `calls` lists the halves that ran, by video."""
    # A window an earlier test in this process left for the garbage
    # collector is freed here, on the main thread. Freed during a run it
    # went on the run's own thread, where Qt waits for the main thread to
    # close the window -- which is waiting for the run: the whole suite
    # hung in this file (seen on the AMD PC, 2 runs of 2). The command line
    # itself has no Qt.
    gc.collect()
    paths = SimpleNamespace(reference=tmp_path / "reference.mkv", a=tmp_path / "a.mkv", b=tmp_path / "b.mkv",
                            calls=[], folder=tmp_path)
    for path in (paths.reference, paths.a, paths.b):
        path.write_bytes(b"video")
    monkeypatch.setattr(cli, "probe_video", lambda path: _info(path))
    monkeypatch.setattr(cli, "check_tools", lambda: SimpleNamespace(
        ok=True, problems=[], ffmpeg=SimpleNamespace(version=(9, 0, 1))))
    monkeypatch.setattr(perceptual_vship, "detect_vship_device", lambda: (None, "no GPU in the tests"))

    def ffmpeg(source, distorted, options, *args, **kwargs):
        paths.calls.append(("ffmpeg", distorted.path.name))
        return _result(source, distorted, options)

    def vship(source, distorted, request, specs, **kwargs):
        paths.calls.append(("vship", distorted.path.name))
        return _perceptual()

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
    return paths


def _run(*arguments) -> tuple[int, str, str]:
    """The compare command's exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    args = cli.build_parser().parse_args(["compare", *[str(argument) for argument in arguments]])
    return cli.compare(args, out=out, err=err), out.getvalue(), err.getvalue()


def _row(table: str, label: str) -> list[str]:
    """A metric's cells in the first video's table."""
    line = next(line for line in table.splitlines() if line.strip().startswith(label + "  "))
    return line.strip()[len(label):].split()


# ------------------------------------------------------------------ arguments

def test_metrics_are_listed_by_key_in_the_apps_order_or_all():
    assert cli._metric_list("psnr, VMAF") == ("vmaf", "psnr")
    assert cli._metric_list("all") == ("vmaf", "vmaf_neg", "vmaf_v1", "psnr", "ssim", "xpsnr", "ssimulacra2",
                                       "butteraugli", "cvvdp")


def test_an_unknown_metric_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["compare", "r.mkv", "t.mkv", "--metrics", "vmaf,psnrr"])
    assert exit_info.value.code == cli.EXIT_USAGE
    assert "unknown metric psnrr" in capsys.readouterr().err


def test_no_command_prints_the_help_and_version_prints_the_version(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "compare" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"VideoMetricsLab {__version__}"


def test_metrics_left_out_are_the_ones_a_video_added_to_the_window_starts_with():
    """The saved settings', less the metrics its Metrics... picker hides."""
    settings = Settings.load()  # the suite's: VMAF, PSNR, SSIM, XPSNR
    assert cli._default_metrics(settings) == ("vmaf", "psnr", "ssim", "xpsnr")
    settings.default_compute_ssimulacra2 = True
    settings.hidden_metrics = ["xpsnr"]
    assert cli._default_metrics(settings) == ("vmaf", "psnr", "ssim", "ssimulacra2")


def test_the_options_are_the_rows_the_window_would_hold(videos):
    args = cli.build_parser().parse_args([
        "compare", str(videos.reference), str(videos.a), "--metrics", "vmaf_neg,xpsnr", "--model", "4k",
        "--black-bars", "none", "--scale-to", "reference", "--scaler", "lanczos", "--duration", "2.5",
        "--subsample", "3", "--threads", "8", "--vmaf-on", "cpu", "--no-gpu-decode"])
    options = cli._row_options(args, Settings.load(), args.metrics)
    assert options.requested_metrics() == ("vmaf_neg", "xpsnr")
    assert (options.model_choice, options.crop_mode.value, options.scale_direction.value, options.scale_algorithm,
            options.duration_limit, options.n_subsample, options.n_threads, options.vmaf_on_gpu,
            options.gpu_decode) == ("version=vmaf_4k_v0.6.1", "none", "distorted_to_source", "lanczos", 2.5, 3, 8,
                                    False, False)


# ------------------------------------------------------------------- a run

def test_a_run_prints_each_videos_scores_and_exits_0(videos):
    code, out, err = _run(videos.reference, videos.a, videos.b)

    assert code == cli.EXIT_OK
    assert videos.calls == [("ffmpeg", "a.mkv"), ("ffmpeg", "b.mkv")]
    assert "Reference: reference.mkv (1920x1080, 30.000 fps, h264, yuv420p, 0:00:05)" in out
    assert _row(out, "VMAF v0.6.1") == ["94.50", "94.50", "90.00", "99.00", "90.09", "(low)", "CPU"]
    assert _row(out, "SSIM") == ["0.9900", "0.9900", "0.9900", "0.9900", "0.9900", "(low)", "CPU"]
    assert "10 frames compared" in out
    assert "a.mkv: started" in err and "b.mkv: done" in err


def test_scores_are_saved_where_the_window_looks_and_not_calculated_twice(videos):
    """The cache is asked with the request the window makes for a row."""
    assert _run(videos.reference, videos.a)[0] == cli.EXIT_OK

    settings = Settings.load()
    window_request = analysis_request_from_vmaf_options(
        VmafOptions(extra_features=settings.default_extra_features(), compute_xpsnr=True),
        ("vmaf", "psnr", "ssim", "xpsnr"), {"ssimulacra2": "gpu", "butteraugli": "gpu"},
        default_settings(settings.cvvdp_presets, settings.cvvdp_default_preset))
    found = result_cache.load_cached(videos.reference, videos.a, window_request)
    assert found is not None and found[0].has_metric("vmaf") and found[0].has_metric("xpsnr")

    videos.calls.clear()
    code, out, err = _run(videos.reference, videos.a)
    assert (code, videos.calls) == (cli.EXIT_OK, [])
    assert "a.mkv: every score is saved; nothing to calculate" in err
    assert "-- saved scores" in out and "94.50" in out

    assert _run(videos.reference, videos.a, "--recalculate")[0] == cli.EXIT_OK
    assert videos.calls == [("ffmpeg", "a.mkv")]


def test_only_the_metrics_not_saved_are_calculated(videos):
    _run(videos.reference, videos.a, "--metrics", "vmaf,psnr")
    videos.calls.clear()
    code, out, _err = _run(videos.reference, videos.a, "--metrics", "vmaf,psnr,ssimulacra2")
    assert code == cli.EXIT_OK
    assert videos.calls == [("vship", "a.mkv")]  # FFmpeg's half was saved
    assert "SSIMULACRA2" in out and "VMAF v0.6.1" in out


def test_a_score_made_on_the_gpu_does_not_answer_a_cpu_choice(videos):
    """The GPU's and the CPU's SSIMULACRA2 differ: metric_cache.made_as_chosen."""
    _run(videos.reference, videos.a, "--metrics", "ssimulacra2")
    videos.calls.clear()
    _run(videos.reference, videos.a, "--metrics", "ssimulacra2", "--ssimulacra2-on", "cpu")
    assert videos.calls == [("vship", "a.mkv")]


def test_a_saved_xpsnr_from_v1_4_is_calculated_again(videos, monkeypatch):
    """v1.4's XPSNR was weighted by the encode, v1.5's by the reference: a
    saved v1.4 score answers no request now (metric_cache.reusable, the
    window's rule too)."""
    for compatibility, kept in (("ffmpeg-xpsnr-v1", False), ("ffmpeg-xpsnr-v2", True)):
        def load_cached(source, distorted, request, *args, compatibility=compatibility, **kwargs):
            saved = _result(_info(source), _info(distorted), VmafOptions(compute_vmaf=True, compute_xpsnr=True))
            saved.merge_metric_results(MetricResultSet([FrameMetricResult(
                "xpsnr", [0], [0.0], [35.0], MetricProvenance("ffmpeg/xpsnr", "ffmpeg 8.0", "cpu", compatibility))]))
            return saved, "saved"

        monkeypatch.setattr(cli.result_cache, "load_cached", load_cached)
        args = cli.build_parser().parse_args(["compare", str(videos.reference), str(videos.a), "-m", "vmaf,xpsnr"])
        (video,), jobs = cli.prepare(args, Settings.load(), _info(videos.reference))
        if kept:
            assert video.from_cache and not jobs, compatibility
        else:
            (job,) = jobs
            assert job.cached_metrics.keys() == ("vmaf",) and "xpsnr" in job.metric_keys


def test_json_has_every_videos_status_and_statistics(videos):
    target = videos.folder / "out" / "results.json"
    code, out, _err = _run(videos.reference, videos.a, "--json", target)

    assert code == cli.EXIT_OK
    assert _row(out, "VMAF v0.6.1")[0] == "94.50"  # a file is as well as the table
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["format"] == 1 and document["cancelled"] is False
    assert document["reference"]["path"] == str(videos.reference)
    (video,) = document["videos"]
    assert (video["status"], video["compared_frames"], video["from_saved_scores"]) == ("ok", 10, False)
    assert list(video["metrics"]) == ["vmaf", "psnr", "ssim", "xpsnr"]
    assert video["metrics"]["vmaf"] == {
        "score": 94.5, "frames": 10, "median": 94.5, "stdev": pytest.approx(2.8723, abs=1e-4), "min": 90.0,
        "max": 99.0, "worst_is": "low", "worst_10_percent": pytest.approx(90.9), "worst_5_percent": pytest.approx(90.45),
        "worst_1_percent": pytest.approx(90.09), "worst_0.1_percent": pytest.approx(90.009),
        "computed_on": "cpu", "implementation": "ffmpeg/libvmaf", "version": "ffmpeg 9.0",
    }

    code, out, _err = _run(videos.reference, videos.a, "--json", "-")
    assert json.loads(out)["videos"][0]["from_saved_scores"] is True  # "-": the JSON alone


def test_several_videos_end_with_a_summary_side_by_side(videos, monkeypatch):
    monkeypatch.setattr(job_runner, "run_vmaf", lambda source, distorted, options, *a, **k: _result(
        source, distorted, options, vmaf=80.0 if distorted.path.name == "b.mkv" else 90.0))
    _code, out, err = _run(videos.reference, videos.a, videos.b, "-m", "vmaf,psnr")
    summary = out[out.index("Summary"):].splitlines()
    assert [line.split() for line in summary[1:]] == [
        ["Video", "VMAF", "v0.6.1", "PSNR"], ["a.mkv", "94.50", "44.50"], ["b.mkv", "84.50", "44.50"]]
    assert "Finished after 0:00:0" in err

    _code, out, _err = _run(videos.reference, videos.a)
    assert "Summary" not in out  # one video: its own table says it all


def test_a_pattern_is_every_file_it_finds_but_the_reference(videos):
    """Windows hands a program "*.mkv" as it is."""
    (videos.folder / "notes [draft].mkv").write_bytes(b"video")
    found = cli.expand_tests([str(videos.folder / "*.mkv")], videos.reference)
    assert [path.name for path in found] == ["a.mkv", "b.mkv", "notes [draft].mkv"]  # name order, no reference
    # A file that exists is itself, brackets and all; one named twice is compared once.
    named = cli.expand_tests([str(videos.folder / "notes [draft].mkv"), str(videos.b), str(videos.folder / "?.mkv")],
                             videos.reference)
    assert [path.name for path in named] == ["notes [draft].mkv", "b.mkv", "a.mkv"]
    # The reference named outright is compared (with itself, if that is what was asked).
    assert cli.expand_tests([str(videos.reference)], videos.reference) == [videos.reference]
    # A pattern that finds nothing is reported as a video that could not be read.
    assert cli.expand_tests([str(videos.folder / "*.mp4")], videos.reference) == [videos.folder / "*.mp4"]

    code, _out, _err = _run(videos.reference, videos.folder / "?.mkv", "-q")
    assert code == cli.EXIT_OK
    assert videos.calls == [("ffmpeg", "a.mkv"), ("ffmpeg", "b.mkv")]


def test_patterns_that_find_only_the_reference_are_a_wrong_command(videos):
    """Nothing to compare: the run printed the reference alone and exited
    0, which a script takes for every video compared."""
    code, out, err = _run(videos.reference, videos.folder / "reference*.mkv")
    assert code == cli.EXIT_USAGE
    assert "No test videos" in err and out == ""
    assert videos.calls == []


def test_cpu_puts_every_metric_with_a_choice_on_the_cpu_unless_one_says_otherwise(videos):
    parse = cli.build_parser().parse_args
    settings = Settings.load()
    args = parse(["compare", "r.mkv", "t.mkv", "--cpu"])
    assert cli._backends(args, settings) == {"ssimulacra2": "cpu", "butteraugli": "cpu"}
    assert cli._row_options(args, settings, ("vmaf",)).vmaf_on_gpu is False
    args = parse(["compare", "r.mkv", "t.mkv", "--cpu", "--butteraugli-on", "gpu", "--vmaf-on", "gpu"])
    assert cli._backends(args, settings) == {"ssimulacra2": "cpu", "butteraugli": "gpu"}
    assert cli._row_options(args, settings, ("vmaf",)).vmaf_on_gpu is True


def test_per_frame_scores_are_written_as_the_windows_csv(videos):
    code, _out, err = _run(videos.reference, videos.a, videos.b, "--csv", videos.folder / "csv")
    assert code == cli.EXIT_OK
    written = sorted(path.name for path in (videos.folder / "csv").iterdir())
    assert written == ["a.csv", "b.csv"]
    header, first = (videos.folder / "csv" / "a.csv").read_text(encoding="utf-8").splitlines()[:2]
    assert header.startswith("frame,time_s,vmaf,vmaf_neg,psnr,ssim,xpsnr")
    assert first.startswith("0,0.000")
    assert "a.mkv: per-frame scores in" in err


# ------------------------------------------------------------------ failures

def test_a_video_that_fails_fails_the_run_and_the_others_are_scored(videos, monkeypatch):
    def ffmpeg(source, distorted, options, *args, **kwargs):
        if distorted.path.name == "a.mkv":
            raise VmafRunError("FFmpeg could not decode it.", "tail")
        return _result(source, distorted, options)

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    code, out, err = _run(videos.reference, videos.a, videos.b, "--json", "-")

    assert code == cli.EXIT_FAILED
    first, second = json.loads(out)["videos"]
    assert (first["status"], first["error"]) == ("failed", "FFmpeg could not decode it.")
    assert "metrics" not in first
    assert second["status"] == "ok"
    assert "a.mkv: FAILED: FFmpeg could not decode it." in err


def test_a_metric_that_fails_leaves_the_others_and_exits_1(videos, monkeypatch):
    def vship(*args, **kwargs):
        raise PerceptualRunError("Vship ran out of GPU memory.", "tail")

    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
    code, out, _err = _run(videos.reference, videos.a, "--metrics", "vmaf,ssimulacra2")

    assert code == cli.EXIT_FAILED
    assert "a.mkv (1920x1080, 30.000 fps, h264, yuv420p, 0:00:05) -- PARTIAL" in out
    assert "SSIMULACRA2 failed: Vship ran out of GPU memory." in out
    assert _row(out, "VMAF v0.6.1")[0] == "94.50"
    # What did finish is saved.
    videos.calls.clear()
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", lambda *a, **k: _perceptual())
    assert _run(videos.reference, videos.a, "--metrics", "vmaf,ssimulacra2")[0] == cli.EXIT_OK
    assert videos.calls == []  # run_vmaf: nothing; the faked Vship is not in `calls`


def test_a_pair_that_cannot_be_compared_is_named_and_the_rest_run(videos, monkeypatch):
    monkeypatch.setattr(cli, "probe_video", lambda path: _info(path, fps=24.0 if path.name == "a.mkv" else 30.0))
    code, out, err = _run(videos.reference, videos.a, videos.b)

    assert code == cli.EXIT_FAILED
    assert videos.calls == [("ffmpeg", "b.mkv")]
    assert "a.mkv: not compared:" in err
    assert "a.mkv (1920x1080, 24.000 fps, h264, yuv420p, 0:00:05) -- FAILED" in out


def test_a_file_that_cannot_be_read_is_named(videos, monkeypatch):
    from vmaf_app.core.ffprobe import ProbeError

    def probe(path):
        if path.name == "a.mkv":
            raise ProbeError("no video stream")
        return _info(path)

    monkeypatch.setattr(cli, "probe_video", probe)
    code, out, _err = _run(videos.reference, videos.a, videos.b)
    assert code == cli.EXIT_FAILED
    assert "a.mkv -- FAILED\n  it could not be read: no video stream" in out
    assert videos.calls == [("ffmpeg", "b.mkv")]


def test_a_reference_that_cannot_be_read_or_no_ffmpeg_is_a_usage_error(videos, monkeypatch):
    from vmaf_app.core.ffprobe import ProbeError

    def probe(path):
        raise ProbeError("no such file")

    monkeypatch.setattr(cli, "probe_video", probe)
    code, _out, err = _run(videos.reference, videos.a)
    assert (code, err) == (cli.EXIT_USAGE, "The reference could not be read: no such file\n")
    assert videos.calls == []

    monkeypatch.setattr(cli, "check_tools", lambda: SimpleNamespace(ok=False, problems=["ffmpeg.exe could not be run."]))
    code, _out, err = _run(videos.reference, videos.a)
    assert code == cli.EXIT_USAGE and "ffmpeg.exe could not be run." in err


def test_cvvdp_is_left_out_with_the_reason_where_it_cannot_be_calculated(videos, monkeypatch):
    code, out, _err = _run(videos.reference, videos.a, "--metrics", "vmaf,cvvdp")
    assert code == cli.EXIT_OK  # the metrics that can be calculated were
    assert "note: CVVDP is calculated on the GPU only, and no GPU that Vship can use was found" in out

    monkeypatch.setattr(perceptual_vship, "detect_vship_device", lambda: (object(), ""))
    _code, out, _err = _run(videos.reference, videos.a, "--metrics", "vmaf,cvvdp", "--subsample", "2")
    assert "note: CVVDP needs every frame: it is not calculated with --subsample above 1" in out

    code, out, _err = _run(videos.reference, videos.a, "--metrics", "cvvdp", "--subsample", "2")
    assert code == cli.EXIT_FAILED
    assert "none of the metrics asked for can be calculated for it" in out


def test_a_long_cpu_run_is_not_refused(videos, monkeypatch):
    """The window asks before SSIMULACRA2 on the CPU for a video over ten
    minutes; the command line runs what it was told to."""
    monkeypatch.setattr(cli, "probe_video", lambda path: VideoInfo(
        path=Path(path), width=1920, height=1080, fps=30.0, duration=3600.0, nb_frames=108000, codec_name="h264",
        pix_fmt="yuv420p"))
    code, _out, _err = _run(videos.reference, videos.a, "--metrics", "ssimulacra2", "--ssimulacra2-on", "cpu")
    assert code == cli.EXIT_OK
    assert videos.calls == [("vship", "a.mkv")]


# ------------------------------------------------------------------- cancel

def test_cancel_ends_the_run_and_exits_130_with_what_finished_saved(videos, monkeypatch):
    started = threading.Event()

    def ffmpeg(source, distorted, options, *args, cancel_event=None, **kwargs):
        if distorted.path.name == "a.mkv":
            return _result(source, distorted, options)
        started.set()
        assert cancel_event.wait(30)  # runs until the run is cancelled
        raise Cancelled("Cancelled by user")

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    real_run = cli.run

    def run_and_cancel(videos_, jobs, settings, parallel, progress, install_cancel=None):
        def install(cancel):
            def press():
                assert started.wait(30)
                cancel()  # Ctrl+C, once the second video is being calculated
            threading.Thread(target=press, daemon=True).start()
        return real_run(videos_, jobs, settings, parallel, progress, install)

    monkeypatch.setattr(cli, "run", run_and_cancel)
    code, out, _err = _run(videos.reference, videos.a, videos.b, "--json", "-")

    document = json.loads(out)
    assert code == cli.EXIT_CANCELLED and document["cancelled"] is True
    assert [video["status"] for video in document["videos"]] == ["ok", "cancelled"]
    videos.calls.clear()
    monkeypatch.setattr(cli, "run", real_run)
    monkeypatch.setattr(job_runner, "run_vmaf", lambda s, d, o, *a, **k: videos.calls.append(d.path.name) or _result(s, d, o))
    assert _run(videos.reference, videos.a, videos.b)[0] == cli.EXIT_OK
    assert videos.calls == ["b.mkv"]  # a.mkv's scores were saved before the cancel


# ----------------------------------------------------------------- progress

class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _snapshots(percent: int) -> list[dict]:
    return [{"backend": "ffmpeg", "metric_keys": ("vmaf",), "lane": "cpu", "passes": (("vmaf",),), "cpu_keys": (),
             "done_keys": (), "current": percent, "total": 100, "fps": 25.0, "state": "running",
             "waiting_for": None, "step": "", "decode": "", "phase": None}]


def test_progress_not_on_a_terminal_is_a_line_every_ten_seconds():
    now = [100.0]
    stream = io.StringIO()
    video = cli.Video(Path("a.mkv"), job_index=0)
    progress = cli.Progress([video], quiet=False, stream=stream, clock=lambda: now[0])
    for seconds, percent in ((0, 1), (3, 20), (9.9, 60), (10, 61), (12, 80), (20, 99)):
        now[0] = 100.0 + seconds
        progress.tasks(0, _snapshots(percent))
    progress.message("a.mkv: done")
    lines = stream.getvalue().splitlines()
    assert [line.split("%")[0][-4:].strip() for line in lines[:-1]] == ["1.0", "61.0", "99.0"]
    assert lines[0].startswith("a.mkv: CPU metric") and lines[-1] == "a.mkv: done"


def test_progress_on_a_terminal_is_one_line_redrawn_and_messages_stay():
    stream = _Terminal()
    progress = cli.Progress([cli.Video(Path("a.mkv"), job_index=0)], quiet=False, stream=stream)
    progress.tasks(0, _snapshots(10))
    progress.message("a.mkv: started")
    progress.tasks(0, _snapshots(20))
    progress.done()
    text = stream.getvalue()
    assert text.count("\n") == 1 and "a.mkv: started\n" in text  # only the message ends a line
    assert "10.0%" in text and "20.0%" in text
    assert text.endswith("\r")  # the redrawn line is wiped at the end


def test_quiet_prints_no_progress():
    stream = io.StringIO()
    progress = cli.Progress([cli.Video(Path("a.mkv"), job_index=0)], quiet=True, stream=stream)
    progress.tasks(0, _snapshots(10))
    progress.message("a.mkv: started")
    progress.done()
    assert stream.getvalue() == ""


# ------------------------------------------------------------------ devices

def test_devices_says_what_calculates_each_metric(videos, monkeypatch, capsys):
    from vmaf_app.core import vmaf_cuda, vmaf_v1_gpu

    monkeypatch.setattr(vmaf_cuda, "gpu_vmaf_available", lambda: (False, "no GPU in the tests"))
    monkeypatch.setattr(vmaf_v1_gpu, "available", lambda: (False, "no GPU in the tests"))
    assert cli.main(["devices"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert f"VideoMetricsLab {__version__}" in out and "FFmpeg: 9.0.1" in out
    assert "CVVDP: not available (no GPU in the tests)" in out
    assert "SSIMULACRA2: CPU (libjxl)" in out
    assert "VMAF v0.6.1, VMAF NEG: CPU (no GPU in the tests)" in out
    assert "VMAF v1: CPU (no GPU in the tests)" in out
    assert "PSNR, SSIM, XPSNR: CPU (FFmpeg)" in out


def test_the_command_line_needs_no_qt():
    """It is the app without its window: importing it must not load one."""
    script = "import sys, vmaf_app.cli; print(any(name.startswith('PySide6') for name in sys.modules))"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                          cwd=Path(__file__).resolve().parents[1])
    assert done.stdout.strip() == "False"
