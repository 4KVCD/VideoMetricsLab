"""VideoMetricsLab from a terminal: the app's comparisons without its window.

    VideoMetricsLab-cli compare REFERENCE TEST [TEST ...] [options]
    VideoMetricsLab-cli devices
    VideoMetricsLab-cli --version

(From source: python -m vmaf_app.cli ...)

`compare` runs what the Videos tab's Calculate metrics runs, with the same
scheduler (core.job_runner), on the same terms: the app's saved settings are
the defaults (FFmpeg's folder, the cache folder, the GPU backend, the metrics
a new video starts with), every option here overrides one for this run, and
nothing is written to the settings. Scores are read from and saved to the
app's own cache, so a comparison made here opens in the window already
calculated, and one made there is not calculated again here.

Progress goes to stderr, results to stdout: each video's table and, for
several, a summary side by side; --json FILE adds the same as JSON, and
--json - prints only that. Exit code 0
when every video got every metric asked for, 1 when one failed or lost a
metric, 2 for a wrong command or a missing FFmpeg, 130 when cancelled with
Ctrl+C. The text is English whatever the window's language: scripts read it.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import logging
import math
import multiprocessing
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from vmaf_app import APP_NAME, __version__
from vmaf_app.core import perceptual_vship, result_cache, vmaf_cuda
from vmaf_app.core.cvvdp import default_settings
from vmaf_app.core.ffmpeg_locate import check_tools, format_version
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.ffprobe import ProbeError, probe_video
from vmaf_app.core.geometry import analysis_dimensions
from vmaf_app.core.job_runner import MAX_PARALLEL_JOBS, JobScheduler, RunEvents, VmafJob
from vmaf_app.core.metric_results import MetricResultSet
from vmaf_app.core.metrics import METRIC_BY_KEY, METRICS, MetricDirection, metric_definition
from vmaf_app.core.model_select import (
    AUTO_MODEL_CHOICE,
    CUSTOM_MODEL_CHOICE,
    DEFAULT_MODEL,
    UHD_MODEL,
    resolve_model,
)
from vmaf_app.core.models import ComparisonResult, CropMode, ScaleDirection, VideoInfo, VmafOptions
from vmaf_app.core.run_io import export_csv, safe_filename_stem, unique_output_path
from vmaf_app.core.settings import Settings
from vmaf_app.core.stats import compute_stats
from vmaf_app.core.time_format import format_hms
from vmaf_app.core.vmaf_runner import VmafRunError, validate_video_pair
from vmaf_app.ui.run_line import metric_line

PROGRAM = "VideoMetricsLab-cli"
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_CANCELLED = 0, 1, 2, 130
#: The metrics that are not FFmpeg's, chosen beside VmafOptions.
_PERCEPTUAL = tuple(metric.key for metric in METRICS if metric.backend_id == "perceptual")
_MODELS = {"auto": AUTO_MODEL_CHOICE, "standard": DEFAULT_MODEL, "4k": UHD_MODEL}
_SCALE_TO = {"test": ScaleDirection.SOURCE_TO_DISTORTED, "reference": ScaleDirection.DISTORTED_TO_SOURCE}
#: Seconds between a video's progress lines where they cannot be redrawn.
_PROGRESS_EVERY = 10.0


# ------------------------------------------------------------------ arguments

def _metric_list(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "all":
        return tuple(metric.key for metric in METRICS)
    keys = tuple(dict.fromkeys(part.strip().lower() for part in text.split(",") if part.strip()))
    unknown = [key for key in keys if key not in METRIC_BY_KEY]
    if unknown or not keys:
        raise argparse.ArgumentTypeError(
            f"unknown metric {', '.join(unknown) or text}; choose from: all, {', '.join(METRIC_BY_KEY)}")
    return tuple(metric.key for metric in METRICS if metric.key in keys)


def _positive(kind):
    def parse(text: str):
        value = kind(text)
        if not value > 0 or (kind is float and not math.isfinite(value)):
            raise argparse.ArgumentTypeError("must be above 0")
        return value
    return parse


_EXAMPLES = f"""examples:
  {PROGRAM} compare reference.mkv encode.mkv
      the app's default metrics for one encode
  {PROGRAM} compare reference.mkv encodes\\*.mkv -m vmaf,ssimulacra2
      two metrics for every .mkv in a folder, with a summary side by side
  {PROGRAM} compare reference.mkv a.mkv b.mkv -m all --csv results
      every metric, and each video's per-frame scores in results\\a.csv, b.csv
  {PROGRAM} compare reference.mkv encode.mkv --duration 30 --cpu
      the first 30 seconds only, everything calculated on the CPU
  {PROGRAM} compare reference.mkv encode.mkv -m vmaf --json - -q
      only JSON on the output, for a script to read
  {PROGRAM} devices
      what would calculate each metric on this PC
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM, description=f"{APP_NAME} {__version__}: video quality metrics from the command line.",
        epilog=_EXAMPLES + f"\n{PROGRAM} compare --help lists every option.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    compare = commands.add_parser(
        "compare", help="calculate metrics for test videos against a reference",
        description="Calculates metrics for each TEST video against REFERENCE. Options left out take the app's "
                    "saved settings; scores already saved by the app or an earlier run are not calculated again.",
        epilog=_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    compare.add_argument("reference", type=Path, metavar="REFERENCE")
    compare.add_argument("tests", nargs="+", metavar="TEST",
                         help="a test video, or a pattern such as encodes\\*.mkv (the reference is left out of "
                              "what a pattern finds)")
    compare.add_argument("-m", "--metrics", type=_metric_list, metavar="LIST",
                         help=f"comma-separated: {', '.join(METRIC_BY_KEY)}; or all (default: the app's)")
    compare.add_argument("--model", default="auto", metavar="MODEL",
                         help="VMAF v0.6.1's model: auto (4K model for a 4K comparison), standard, 4k, "
                              "or a model .json file (default: auto)")
    compare.add_argument("--black-bars", choices=["auto", "none"], default="auto",
                         help="auto: detect and cut black bars; none: compare the whole pictures (default: auto)")
    compare.add_argument("--scale-to", choices=list(_SCALE_TO), default="test",
                         help="which video's size the other is scaled to when they differ (default: test)")
    compare.add_argument("--scaler", choices=["bicubic", "bilinear", "lanczos", "spline"], default="bicubic",
                         help="scaling algorithm (default: bicubic)")
    compare.add_argument("--duration", type=_positive(float), metavar="SECONDS",
                         help="compare only the first SECONDS of each video")
    compare.add_argument("--subsample", type=_positive(int), default=1, metavar="N",
                         help="score every N-th frame (default: 1, every frame; CVVDP needs every frame)")
    compare.add_argument("--threads", type=_positive(int), metavar="N", help="libvmaf threads (default: automatic)")
    for key, label in (("vmaf", "VMAF v0.6.1 and VMAF NEG"), ("ssimulacra2", "SSIMULACRA2"),
                       ("butteraugli", "Butteraugli")):
        compare.add_argument(f"--{key}-on", choices=["gpu", "cpu"], dest=f"{key}_on",
                             help=f"where {label} is calculated (default: the app's)")
    compare.add_argument("--cpu", action="store_true",
                         help="calculate on the CPU every metric that can be (CVVDP is GPU only); the same as "
                              "--vmaf-on cpu --ssimulacra2-on cpu --butteraugli-on cpu")
    compare.add_argument("--no-gpu-decode", action="store_true", help="decode the videos on the CPU")
    compare.add_argument("--parallel", type=int, choices=range(1, MAX_PARALLEL_JOBS + 1), metavar="N",
                         help=f"videos calculated at once on the CPU, 1-{MAX_PARALLEL_JOBS} (default: the app's)")
    compare.add_argument("--recalculate", action="store_true",
                         help="calculate every metric again instead of using saved scores")
    compare.add_argument("--csv", type=Path, metavar="FOLDER",
                         help="write each video's per-frame scores to FOLDER as <video>.csv")
    compare.add_argument("--json", type=Path, metavar="FILE", dest="json_path",
                         help="also write the results as JSON to FILE; - prints the JSON instead of the tables")
    compare.add_argument("-q", "--quiet", action="store_true", help="no progress, only the results")
    compare.add_argument("-v", "--verbose", action="store_true", help="the app's log as well")

    commands.add_parser("devices", help="show FFmpeg and which GPU each metric would use",
                        description="Shows FFmpeg's version and what would calculate each metric on this PC.")
    return parser


# ----------------------------------------------------------------- the videos

@dataclass
class Video:
    """One test video of the run: what the window keeps in a row."""

    path: Path
    info: VideoInfo | None = None
    options: VmafOptions | None = None  # as asked: what the cache is keyed by
    metrics: tuple[str, ...] = ()
    request: object = None
    notes: list[str] = field(default_factory=list)  # metrics left out, and why
    error: str = ""  # why it was not compared at all
    result: ComparisonResult | None = None
    failures: dict[str, str] = field(default_factory=dict)  # metric -> why it failed
    from_cache: bool = False  # nothing had to be calculated
    job_index: int | None = None


def _where(args: argparse.Namespace, key: str) -> str | None:
    """Where the command says `key` is calculated: its own option, else
    --cpu; None when it says nothing."""
    return getattr(args, f"{key}_on") or ("cpu" if args.cpu else None)


def expand_tests(patterns: list[str], reference: Path) -> list[Path]:
    """The test videos named: each file as given, and what each pattern
    finds, in name order -- Windows hands a program "*.mkv" as it is. A
    pattern often takes in the reference too (reference.mkv *.mkv): it is
    left out of what patterns find, as is a file already listed."""
    found: list[Path] = []

    def same(a: Path, b: Path) -> bool:
        return a.resolve() == b.resolve()

    for text in patterns:
        if Path(text).exists() or not glob.has_magic(text):
            if not any(same(Path(text), path) for path in found):
                found.append(Path(text))
            continue
        matches = sorted(Path(match) for match in glob.glob(text) if Path(match).is_file())
        if not matches:
            found.append(Path(text))  # named in the results as not readable
        for path in matches:
            if not same(path, reference) and not any(same(path, other) for other in found):
                found.append(path)
    return found


def _row_options(args: argparse.Namespace, settings: Settings, metrics: tuple[str, ...]) -> VmafOptions:
    """The options the window's row would hold for these choices."""
    vmaf_on = _where(args, "vmaf")
    options = VmafOptions(
        gpu_decode=settings.default_gpu_decode and not args.no_gpu_decode,
        vmaf_on_gpu=settings.default_vmaf_on_gpu if vmaf_on is None else vmaf_on == "gpu",
        n_threads=args.threads or 0,
        n_subsample=args.subsample,
        scale_algorithm=args.scaler,
        scale_direction=_SCALE_TO[args.scale_to],
        duration_limit=args.duration or 0.0,
        crop_mode=CropMode.AUTO if args.black_bars == "auto" else CropMode.NONE,
        compute_vmaf=False,
    )
    if args.model in _MODELS:
        options.model_choice = _MODELS[args.model]
    else:
        options.model_choice, options.custom_model_path = CUSTOM_MODEL_CHOICE, str(Path(args.model).resolve())
    for key in metrics:
        if metric_definition(key).ffmpeg_binding is not None:
            options.set_metric_enabled(key, True)
    return options


def _default_metrics(settings: Settings) -> tuple[str, ...]:
    """What a video added to the window starts with ticked, less the
    metrics its Metrics... picker hides (they are not calculated there)."""
    options = VmafOptions(
        extra_features=settings.default_extra_features(), compute_xpsnr=settings.default_compute_xpsnr,
        compute_vmaf=settings.default_compute_vmaf, compute_vmaf_neg=settings.default_compute_vmaf_neg,
        compute_vmaf_v1=settings.default_compute_vmaf_v1,
    )
    chosen = set(options.requested_metrics())
    chosen |= {key for key in _PERCEPTUAL if getattr(settings, f"default_compute_{key}", False) is True}
    known = {metric.key for metric in METRICS}
    hidden = {key for key in settings.hidden_metrics if key in known}
    if len(hidden) < len(known):  # a settings file that hides everything hides nothing
        chosen -= hidden
    return tuple(metric.key for metric in METRICS if metric.key in chosen)


def _backends(args: argparse.Namespace, settings: Settings) -> dict[str, str]:
    chosen = {}
    for key in ("ssimulacra2", "butteraugli"):
        asked = _where(args, key)
        saved = getattr(settings, f"default_{key}_backend", "gpu")
        chosen[key] = asked or ("cpu" if saved == "cpu" else "gpu")
    return chosen


def _reusable(video: Video, saved: ComparisonResult, backends: dict[str, str]) -> MetricResultSet:
    """The saved scores a run keeps: MainWindow._reusable_results' rule. A
    SSIMULACRA2 or Butteraugli score counts only when it was made where it
    is asked for now -- the GPU's and the CPU's differ -- but a CPU score
    answers a GPU choice on a PC whose GPU cannot calculate it."""
    reusable = MetricResultSet()
    for key in video.metrics:
        sequence = saved.sequence_metric(key)
        if sequence is not None:
            if np.isfinite(sequence.score):
                reusable.add(sequence)
            continue
        metric = saved.frame_metric(key)
        if metric is None or not np.any(~np.isnan(metric.values)):
            continue
        choice, produced = backends.get(key), metric.provenance.compute_backend
        if choice == "cpu" and produced != "cpu":
            continue
        if choice == "gpu" and produced == "cpu" and perceptual_vship.gpu_can_score(key):
            continue
        reusable.add(metric)
    return reusable


def prepare(args: argparse.Namespace, settings: Settings, source: VideoInfo) -> tuple[list[Video], list[VmafJob]]:
    """Every test video, and the jobs for the ones that can be compared."""
    asked = args.metrics or _default_metrics(settings)
    backends = _backends(args, settings)
    cvvdp = default_settings(settings.cvvdp_presets, settings.cvvdp_default_preset)
    videos, jobs = [], []
    for path in expand_tests(args.tests, args.reference):
        video = Video(path)
        videos.append(video)
        try:
            video.info = probe_video(path)
        except (ProbeError, OSError) as error:
            video.error = f"it could not be read: {error}"
            continue
        metrics = list(asked)
        if "cvvdp" in metrics:
            reason = None
            if args.subsample > 1:
                reason = "CVVDP needs every frame: it is not calculated with --subsample above 1"
            elif perceptual_vship.detect_vship_device()[0] is None:
                reason = "CVVDP is calculated on the GPU only, and no GPU that Vship can use was found"
            if reason:
                metrics.remove("cvvdp")
                video.notes.append(reason)
        video.metrics = tuple(metrics)
        if not video.metrics:
            video.error = "none of the metrics asked for can be calculated for it"
            continue
        video.options = _row_options(args, settings, video.metrics)
        try:
            validate_video_pair(source, video.info, video.options)
            size = analysis_dimensions(source, video.info, video.options, None, None)
            model = resolve_model(video.options, *size) if video.options.compute_vmaf else ""
        except (ValueError, VmafRunError) as error:
            video.error = str(error)
            continue
        # The request the window makes for a row: what the cache is asked
        # with and stored under, so the two find each other's scores.
        video.request = analysis_request_from_vmaf_options(video.options, video.metrics, backends, cvvdp)
        saved = None
        if not args.recalculate and settings.use_cache:
            found = result_cache.load_cached(source.path, path, video.request)
            saved = found[0] if found else None
        reusable = _reusable(video, saved, backends) if saved is not None else MetricResultSet()
        if all(reusable.has(key) for key in video.metrics):
            # Everything asked for is saved: nothing to run, as the window
            # runs nothing for a row that has its scores.
            video.result, video.from_cache = saved, True
            continue
        video.job_index = len(jobs)
        jobs.append(VmafJob(
            source, video.info, replace(video.options, model=model), label=path.stem,
            result_distorted_path=path, metric_keys=video.metrics, metric_backends=dict(backends), cvvdp=cvvdp,
            cached_result=saved if reusable else None, cached_metrics=reusable if reusable else None,
        ))
    return videos, jobs


# ------------------------------------------------------------------- progress

class Progress:
    """The run's progress on stderr: one line a video, as the window's run
    line words it -- redrawn in place on a terminal, else repeated every
    few seconds."""

    def __init__(self, videos: list[Video], *, quiet: bool, stream=None, clock=time.monotonic) -> None:
        self._names = {video.job_index: video.path.name for video in videos if video.job_index is not None}
        self._quiet, self._stream, self._clock = quiet, stream or sys.stderr, clock
        self._redraw = bool(getattr(self._stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._last: dict[int, float] = {}
        self._shown = ""  # the line being redrawn

    def _write(self, text: str, *, keep: bool) -> None:
        if self._quiet:
            return
        with contextlib.suppress(OSError, ValueError):
            if self._redraw:
                wipe = "\r" + " " * len(self._shown) + "\r"
                if keep:
                    self._stream.write(wipe + text + "\n" + self._shown)
                else:
                    self._shown = text[:118]
                    self._stream.write(wipe + self._shown)
            elif keep:
                self._stream.write(text + "\n")
            self._stream.flush()

    def message(self, text: str) -> None:
        with self._lock:
            self._write(text, keep=True)

    def tasks(self, index: int, snapshots: list) -> None:
        lines = metric_line(snapshots, False)[0]
        if not lines:
            return
        text = f"{self._names.get(index, index)}: {'; '.join(lines)}"
        with self._lock:
            if self._redraw:
                self._write(text, keep=False)
            elif self._clock() - self._last.get(index, -_PROGRESS_EVERY) >= _PROGRESS_EVERY:
                self._last[index] = self._clock()
                self._write(text, keep=True)

    def done(self) -> None:
        with self._lock:
            if self._redraw and self._shown and not self._quiet:
                with contextlib.suppress(OSError, ValueError):
                    self._stream.write("\r" + " " * len(self._shown) + "\r")
                    self._stream.flush()
            self._shown = ""


# ---------------------------------------------------------------------- a run

def run(videos: list[Video], jobs: list[VmafJob], settings: Settings, parallel: int, progress: Progress,
        install_cancel=None) -> bool:
    """Runs the jobs, saving each result as the window does. True when the
    run was cancelled. `install_cancel` is given the function that cancels
    it (Ctrl+C)."""
    by_job = {video.job_index: video for video in videos if video.job_index is not None}
    cancelled = threading.Event()

    def save(index: int, result: ComparisonResult) -> None:
        video = by_job[index]
        video.result = result
        try:
            result_cache.store(result.source, video.path, result, video.path.stem, video.request,
                               result_cache.cache_dir())
        except OSError as error:
            progress.message(f"{video.path.name}: the scores could not be saved ({error})")

    def finished(index: int, result: ComparisonResult) -> None:
        save(index, result)
        progress.message(f"{by_job[index].path.name}: done")

    def failed(index: int, message: str, _stderr_tail: str) -> None:
        by_job[index].error = message
        progress.message(f"{by_job[index].path.name}: FAILED: {message}")

    def partly_failed(index: int, result: ComparisonResult, message: str, _stderr_tail: str, failures: dict) -> None:
        save(index, result)
        by_job[index].failures = dict(failures) or {"": message}
        progress.message(f"{by_job[index].path.name}: some metrics FAILED: {message}")

    events = RunEvents(
        job_started=lambda index, _label: progress.message(f"{by_job[index].path.name}: started"),
        task_progress=progress.tasks,
        job_finished=finished,
        job_failed=failed,
        job_partially_failed=partly_failed,
        result_updated=save,  # each half as it finishes, like the window: kept if the rest is lost
        cancelled=cancelled.set,
    )
    scheduler = JobScheduler(jobs, parallel, events, gpu_metrics_together=settings.gpu_metrics_together)
    if install_cancel is not None:
        install_cancel(scheduler.cancel)
    # On a thread of its own, the main one only waiting for it in short
    # steps: Python runs a signal's handler between the main thread's
    # steps, and a run on the main thread sat in waits of its own -- Ctrl+C
    # was answered 6 seconds later, when a frame pair happened to finish.
    failure: list[BaseException] = []

    def work() -> None:
        try:
            scheduler.run()
        except BaseException as error:
            failure.append(error)

    runner = threading.Thread(target=work, name="cli-run")
    runner.start()
    while runner.is_alive():
        runner.join(0.1)
    progress.done()
    if failure:
        raise failure[0]
    return cancelled.is_set()


# -------------------------------------------------------------------- results

def _number(value: float | None) -> float | str | None:
    """JSON has no infinity: a metric's "identical" is the string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def metric_summary(result: ComparisonResult, key: str) -> dict | None:
    """One metric of a result: its score, and for a per-frame metric the
    statistics the Metric Graphs tab shows."""
    definition = metric_definition(key)
    sequence = result.sequence_metric(key)
    if sequence is not None:
        return {"score": _number(float(sequence.score)), "computed_on": sequence.provenance.compute_backend,
                "implementation": sequence.provenance.implementation,
                "version": sequence.provenance.implementation_version}
    metric = result.frame_metric(key)
    if metric is None or not len(metric.values):
        return None
    stats = compute_stats(metric.values, [], definition.aggregation, definition.direction)
    # The worst frames are the low ones, or for Butteraugli the high ones.
    worst = "high" if definition.direction is MetricDirection.LOWER_IS_BETTER else "low"
    return {
        "score": _number(metric.aggregate), "frames": stats.count, "median": _number(stats.median),
        "stdev": _number(stats.stdev), "min": _number(stats.minimum), "max": _number(stats.maximum),
        "worst_is": worst, "worst_10_percent": _number(stats.percentile_10),
        "worst_5_percent": _number(stats.percentile_5), "worst_1_percent": _number(stats.percentile_1),
        "worst_0.1_percent": _number(stats.percentile_0_1),
        "computed_on": metric.provenance.compute_backend, "implementation": metric.provenance.implementation,
        "version": metric.provenance.implementation_version,
    }


def _crop(box) -> dict | None:
    return None if box is None else {"x": box.x, "y": box.y, "width": box.w, "height": box.h}


def _media(info: VideoInfo) -> dict:
    return {"width": info.width, "height": info.height, "fps": info.fps, "codec": info.codec_name,
            "pixel_format": info.pix_fmt, "duration": info.duration}


def results_document(source: VideoInfo, videos: list[Video], cancelled: bool) -> dict:
    """The run as --json writes it."""
    document = {"app": APP_NAME, "version": __version__, "format": 1, "cancelled": cancelled,
                "reference": {"path": str(source.path), **_media(source)}, "videos": []}
    for video in videos:
        entry: dict = {"path": str(video.path), "status": video_status(video, cancelled)}
        if video.info is not None:
            entry.update(_media(video.info))
        if video.error:
            entry["error"] = video.error
        if video.notes:
            entry["notes"] = list(video.notes)
        if video.failures:
            entry["failed_metrics"] = dict(video.failures)
        result = video.result
        if result is not None:
            entry["from_saved_scores"] = video.from_cache
            entry["compared_frames"] = result.compared_frame_count
            entry["vmaf_model"] = result.model
            entry["reference_crop"], entry["test_crop"] = _crop(result.source_crop), _crop(result.distorted_crop)
            entry["metrics"] = {key: summary for key in video.metrics
                                if (summary := metric_summary(result, key)) is not None}
        document["videos"].append(entry)
    return document


def video_status(video: Video, cancelled: bool) -> str:
    """ok: every metric asked for; partial: some; failed: none; cancelled:
    the run was stopped before it had them all."""
    have = [key for key in video.metrics if video.result is not None and video.result.has_metric(key)]
    if video.metrics and len(have) == len(video.metrics) and not video.failures:
        return "ok"
    if cancelled and not video.error and not video.failures:
        return "cancelled"
    return "partial" if have else "failed"


def results_table(source: VideoInfo, videos: list[Video], cancelled: bool) -> str:
    lines = [f"Reference: {source.path.name} ({_media_text(source)})"]
    for video in videos:
        lines.append("")
        status = video_status(video, cancelled)
        head = video.path.name + (f" ({_media_text(video.info)})" if video.info is not None else "")
        if status != "ok":
            head += f" -- {status.upper()}"
        elif video.from_cache:
            head += " -- saved scores"
        lines.append(head)
        if video.error:
            lines.append(f"  {video.error}")
        for note in video.notes:
            lines.append(f"  note: {note}")
        for key, reason in video.failures.items():
            label = metric_definition(key).label if key in METRIC_BY_KEY else "metrics"
            lines.append(f"  {label} failed: {reason}")
        if video.result is None:
            continue
        rows = []
        for key in video.metrics:
            summary = metric_summary(video.result, key)
            if summary is None:
                continue
            definition = metric_definition(key)
            show = definition.format_value
            if "frames" not in summary:  # one score for the video
                rows.append((definition.label, show(_value(summary["score"])), "", "", "", "",
                             summary["computed_on"].upper()))
                continue
            rows.append((definition.label, show(_value(summary["score"])), show(_value(summary["median"])),
                         show(_value(summary["min"])), show(_value(summary["max"])),
                         f"{show(_value(summary['worst_1_percent']))} ({summary['worst_is']})",
                         summary["computed_on"].upper()))
        if rows:
            header = ("Metric", "Mean", "Median", "Min", "Max", "1% worst", "On")
            widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
            for row in (header, *rows):
                cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:-1], widths[1:-1],
                                                                                              strict=True)]
                lines.append("  " + "  ".join([*cells, row[-1]]).rstrip())
            lines.append(f"  {video.result.compared_frame_count:,} frames compared")
    lines += summary_table(videos)
    if cancelled:
        lines += ["", "Cancelled: the scores finished before that are saved."]
    return "\n".join(lines)


def summary_table(videos: list[Video]) -> list[str]:
    """Several videos side by side: one line each, every metric's score for
    the video (the mean over its frames). Nothing for a single video."""
    scored = [video for video in videos if video.result is not None]
    if len(scored) < 2:
        return []
    keys = [metric.key for metric in METRICS if any(
        key == metric.key and metric_summary(video.result, key) is not None
        for video in scored for key in video.metrics)]
    if not keys:
        return []
    rows = [("Video", *(metric_definition(key).label for key in keys))]
    for video in scored:
        cells = [video.path.name]
        for key in keys:
            summary = metric_summary(video.result, key) if key in video.metrics else None
            cells.append("" if summary is None else metric_definition(key).format_value(_value(summary["score"])))
        rows.append(tuple(cells))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = ["", "Summary (mean scores; CVVDP is one score for the whole video)" if "cvvdp" in keys
             else "Summary (mean scores)"]
    for row in rows:
        cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  " + "  ".join(cells).rstrip())
    return lines


def _value(number) -> float | None:
    if isinstance(number, str):
        return float(number)
    return number


def _media_text(info: VideoInfo) -> str:
    return (f"{info.width}x{info.height}, {info.fps:.3f} fps, {info.codec_name}, {info.pix_fmt}, "
            f"{format_hms(info.duration)}")


def write_csvs(videos: list[Video], folder: Path, progress: Progress) -> bool:
    """Each result's per-frame scores, as the window's Export CSV writes
    them. False if one could not be written."""
    written = True
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        progress.message(f"The CSV folder could not be made: {error}")
        return False
    for video in videos:
        if video.result is None:
            continue
        path = unique_output_path(folder, safe_filename_stem(video.path.stem), ".csv")
        try:
            export_csv(video.result, path)
            progress.message(f"{video.path.name}: per-frame scores in {path}")
        except OSError as error:
            written = False
            progress.message(f"{video.path.name}: the CSV could not be written ({error})")
    return written


# ------------------------------------------------------------------- commands

def _apply_settings(settings: Settings) -> None:
    """What the window does with them as it starts; nothing is saved."""
    result_cache.set_cache_dir_override(settings.cache_dir_path())
    perceptual_vship.set_vship_backend(settings.gpu_backend)
    vmaf_cuda.set_gpu_backend(settings.gpu_backend)


def _ffmpeg_problem() -> str | None:
    status = check_tools()
    if status.ok:
        return None
    return ("FFmpeg cannot be used:\n  " + "\n  ".join(status.problems)
            + "\nPut ffmpeg and ffprobe on PATH, or set FFmpeg's folder in the app's Settings tab.")


def compare(args: argparse.Namespace, *, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    settings = Settings.load()
    _apply_settings(settings)
    if (problem := _ffmpeg_problem()) is not None:
        print(problem, file=err)
        return EXIT_USAGE
    try:
        source = probe_video(args.reference)
    except (ProbeError, OSError) as error:
        print(f"The reference could not be read: {error}", file=err)
        return EXIT_USAGE
    videos, jobs = prepare(args, settings, source)
    progress = Progress(videos, quiet=args.quiet, stream=err)
    for video in videos:
        if video.error:
            progress.message(f"{video.path.name}: not compared: {video.error}")
        for note in video.notes:
            progress.message(f"{video.path.name}: {note}")
        if video.from_cache:
            progress.message(f"{video.path.name}: every score is saved; nothing to calculate")
    cancelled = False
    started = time.monotonic()
    if jobs:
        parallel = args.parallel or settings.parallel_jobs

        def on_interrupt(cancel) -> None:
            def handler(_signal, _frame) -> None:
                progress.message("Cancelling...")
                cancel()
            # Ctrl+C, and Ctrl+Break or a closed console window.
            for name in ("SIGINT", "SIGBREAK"):
                with contextlib.suppress(ValueError, AttributeError):  # not the main thread (tests); no SIGBREAK
                    signal.signal(getattr(signal, name), handler)

        cancelled = run(videos, jobs, settings, parallel, progress, on_interrupt)
        progress.message(f"{'Cancelled' if cancelled else 'Finished'} after "
                         f"{format_hms(time.monotonic() - started)}")
    document = results_document(source, videos, cancelled)
    written = True
    if args.csv is not None:
        written = write_csvs(videos, args.csv, progress)
    # --json - is for a program to read: the JSON alone. A file is as well
    # as the tables.
    json_only = args.json_path is not None and str(args.json_path) == "-"
    if json_only:
        print(json.dumps(document, indent=2), file=out)
    else:
        print(results_table(source, videos, cancelled), file=out)
        if args.json_path is not None:
            try:
                args.json_path.parent.mkdir(parents=True, exist_ok=True)
                args.json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                progress.message(f"Results as JSON in {args.json_path}")
            except OSError as error:
                print(f"The JSON file could not be written: {error}", file=err)
                written = False
    if cancelled:
        return EXIT_CANCELLED
    every_ok = all(entry["status"] == "ok" for entry in document["videos"])
    return EXIT_OK if every_ok and written else EXIT_FAILED


def devices(_args: argparse.Namespace, *, out=None, err=None) -> int:
    out = out or sys.stdout
    settings = Settings.load()
    _apply_settings(settings)
    lines = [f"{APP_NAME} {__version__}"]
    status = check_tools()
    if status.ok:
        lines.append(f"FFmpeg: {format_version(status.ffmpeg.version)}")
    else:
        lines += [f"FFmpeg: NOT USABLE: {problem}" for problem in status.problems]
    lines.append(f"GPU backend setting: {settings.gpu_backend}")
    device, reason = perceptual_vship.detect_vship_device()
    if device is not None:
        vship = f"GPU: {device.name} (Vship {device.version}, {perceptual_vship.backend_label(device.backend)})"
        lines.append(f"CVVDP: {vship}")
        for key in ("ssimulacra2", "butteraugli"):
            where = vship if perceptual_vship.scores_correctly(device, key) else \
                "CPU (libjxl): this Vship build scores it wrongly on this GPU"
            lines.append(f"{metric_definition(key).label}: {where}")
    else:
        lines.append(f"CVVDP: not available ({reason})")
        lines += [f"{metric_definition(key).label}: CPU (libjxl)" for key in ("ssimulacra2", "butteraugli")]
    available, why = vmaf_cuda.gpu_vmaf_available()
    if available:
        backend, gpu = vmaf_cuda.gpu_vmaf_backend()
        lines.append("VMAF v0.6.1, VMAF NEG: GPU ("
                     + ("libvmaf CUDA" if backend == "cuda" else f"Vulkan, GPU {gpu}") + ")")
    else:
        lines.append(f"VMAF v0.6.1, VMAF NEG: CPU ({why})")
    lines.append("VMAF v1, PSNR, SSIM, XPSNR: CPU (FFmpeg)")
    print("\n".join(lines), file=out)
    return EXIT_OK if status.ok else EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # a character the console lacks is not an error
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.ERROR,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        return {"compare": compare, "devices": devices}[args.command](args)
    except KeyboardInterrupt:  # Ctrl+C before the run had started
        return EXIT_CANCELLED


if __name__ == "__main__":
    # The GPU metrics run in processes of their own (core.isolated): in the
    # packaged build this program started again, which runs that work here.
    multiprocessing.freeze_support()
    sys.exit(main())
