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
calculated, and one made there is not calculated again here. The one
exception is --colour: the cache knows a video by its file, not by the
colours it was read with, so those scores are neither read nor saved.

Progress goes to stderr, results to stdout: each video's table and, for
several, a summary side by side; --json FILE adds the same as JSON, and
--json - prints only that. Exit code 0
when every video got every metric asked for that it can have (one it
cannot, such as CVVDP with --subsample above 1, is left out with a note, as
the window leaves it out), 1 when one failed or lost a metric, 2 for a
wrong command, a missing FFmpeg or no test video, 130 when cancelled with
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

from vmaf_app import APP_NAME, __version__
from vmaf_app.core import libvmaf, metric_cache, perceptual_vship, result_cache, temp_files, vmaf_gpu
from vmaf_app.core.colour import UNTAGGED
from vmaf_app.core.cvvdp import DEFAULT_PRESET, CvvdpSettings, preset_named
from vmaf_app.core.cvvdp import presets as cvvdp_presets
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
#: The colours --colour gives a video with no colour tags: its transfer,
#: primaries and matrix, as ffprobe names them.
_ASSUMED_COLOURS = {
    "pq-bt2020": ("smpte2084", "bt2020", "bt2020nc"),
    "hlg-bt2020": ("arib-std-b67", "bt2020", "bt2020nc"),
    "bt709": ("bt709", "bt709", "bt709"),
}
#: The metrics whose scores depend on how a video's colours are read.
_COLOUR_METRICS = frozenset({"cvvdp", "ssimulacra2", "butteraugli"})
#: The transfers of HDR video: PQ and HLG.
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
#: The HDR display a note suggests for CVVDP on HDR video.
_SUGGESTED_HDR_DISPLAY = "30-inch 4K HDR monitor, dim room"


# ------------------------------------------------------------------ arguments

def _metric_list(text: str) -> tuple[str, ...]:
    """The metric keys named in a --metrics value, or every metric for "all"."""
    if text.strip().lower() == "all":
        return tuple(metric.key for metric in METRICS)
    keys = tuple(dict.fromkeys(part.strip().lower() for part in text.split(",") if part.strip()))
    unknown = [key for key in keys if key not in METRIC_BY_KEY]
    if unknown or not keys:
        raise argparse.ArgumentTypeError(
            f"unknown metric {', '.join(unknown) or text}; choose from: all, {', '.join(METRIC_BY_KEY)}")
    return tuple(metric.key for metric in METRICS if metric.key in keys)


def _positive(kind):
    """An argument type that reads a number of `kind` and requires it to be above 0."""
    def parse(text: str):
        """Reads `text` as a number above 0."""
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
  {PROGRAM} compare reference.y4m encode.y4m -m cvvdp --colour pq-bt2020 --frames 33
      HDR10 video in files with no colour tags, the first 33 frames
  {PROGRAM} compare reference.mkv encode.mkv -m cvvdp --cvvdp-display "30-inch 4K HDR monitor, dim room"
      CVVDP on a display other than the app's default (repeat it for several)
  {PROGRAM} devices
      what would calculate each metric on this PC
"""


def build_parser() -> argparse.ArgumentParser:
    """The command line's parser, with its compare and devices commands."""
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
    length = compare.add_mutually_exclusive_group()
    length.add_argument("--duration", type=_positive(float), metavar="SECONDS",
                        help="compare only the first SECONDS of each video")
    length.add_argument("--frames", type=_positive(int), metavar="N",
                        help="compare only the first N frames of each video (a short encode of a longer "
                             "reference, say)")
    compare.add_argument("--colour", "--color", choices=list(_ASSUMED_COLOURS), dest="colour",
                         help="the colours of a video with no colour tags (y4m, say): pq-bt2020 (HDR10), "
                              "hlg-bt2020 or bt709 (SDR). A tagged video keeps its own. Without it such a "
                              "video is taken as SDR, which changes CVVDP, SSIMULACRA2 and Butteraugli. These "
                              "scores are neither read from nor saved to the saved scores")
    compare.add_argument("--cvvdp-display", action="append", metavar="PRESET", dest="cvvdp_displays",
                         help="the display CVVDP models, by preset name (default: the app's). Repeat it for a "
                              "CVVDP score on each display; each extra one decodes the videos again. "
                              f"Presets: {', '.join(repr(preset.name) for preset in cvvdp_presets([]))}, "
                              "or one saved in the app")
    compare.add_argument("--subsample", type=_positive(int), default=1, metavar="N",
                         help="score every N-th frame (default: 1, every frame; CVVDP needs every frame)")
    compare.add_argument("--threads", type=_positive(int), metavar="N", help="libvmaf threads (default: automatic)")
    for key, label in (("vmaf", "VMAF v0.6.1, VMAF NEG and VMAF v1"), ("ssimulacra2", "SSIMULACRA2"),
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
    # What the reader should know: metrics left out and why, colours guessed.
    notes: list[str] = field(default_factory=list)
    error: str = ""  # why it was not compared at all
    result: ComparisonResult | None = None
    failures: dict[str, str] = field(default_factory=dict)  # metric -> why it failed
    from_cache: bool = False  # nothing had to be calculated
    job_index: int | None = None
    # False for scores read with --colour: the cache knows a video by its
    # file, not by the colours it was read with.
    save_scores: bool = True
    display: str = ""  # the preset CVVDP models, when CVVDP is asked for
    # CVVDP on each further --cvvdp-display: a Video of its own for each,
    # with CVVDP alone, run and saved as a separate comparison.
    other_displays: list[Video] = field(default_factory=list)
    for_display_only: bool = False  # one of another Video's other_displays

    @property
    def label(self) -> str:
        """How progress lines name it."""
        return f"{self.path.name} (CVVDP on {self.display})" if self.for_display_only else self.path.name


def has_colour_tags(info: VideoInfo) -> bool:
    """Whether the video says its transfer or its primaries."""
    return (info.color_transfer.casefold() not in UNTAGGED
            or info.color_primaries.casefold() not in UNTAGGED)


def with_assumed_colour(info: VideoInfo, colour: str | None) -> VideoInfo:
    """`info` with the colours --colour gives (_ASSUMED_COLOURS), when it
    has no colour tags of its own. A matrix it does say is kept."""
    if colour is None or has_colour_tags(info):
        return info
    transfer, primaries, matrix = _ASSUMED_COLOURS[colour]
    if info.color_space.casefold() not in UNTAGGED:
        matrix = info.color_space
    return replace(info, color_transfer=transfer, color_primaries=primaries, color_space=matrix)


def guessed_colour_note(info: VideoInfo, which: str) -> str:
    """The note for a video read without colour tags: what the app took its
    colours to be (colour.video_colour's guess)."""
    if info.color_space.casefold() in UNTAGGED:
        guess = "BT.709 SDR, as it is over 650 lines" if info.height > 650 else "BT.601 SDR, as it is 650 lines or fewer"
    else:
        guess = f"from its colour matrix, {info.color_space}"
    return (f"{which} has no colour tags, so its colours were guessed ({guess}). CVVDP, SSIMULACRA2 and "
            "Butteraugli depend on them: --colour says what they are, such as pq-bt2020 for HDR10.")


def _where(args: argparse.Namespace, key: str) -> str | None:
    """Where the command says `key` is calculated: its own option, else
    --cpu; None when it says nothing."""
    return getattr(args, f"{key}_on") or ("cpu" if args.cpu else None)


def expand_tests(patterns: list[str], reference: Path) -> list[Path]:
    """The test videos named: each file as given, and what each pattern
    finds, in name order. Windows passes a pattern such as "*.mkv" to a
    program as it is, so the program must expand it. A pattern often takes
    in the reference too (reference.mkv *.mkv), so the reference is left out
    of what patterns find, as is a file already listed."""
    found: list[Path] = []

    def same(a: Path, b: Path) -> bool:
        """Whether two paths name the same file."""
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


def _duration_limit(args: argparse.Namespace, test: VideoInfo) -> float:
    """The duration limit, in seconds, of --duration or --frames; 0 for none.

    N frames are the frames that start before N/fps, numbered at the test
    video's frame rate as every metric numbers them (frame_sync). The limit
    is half a frame short of that, so that writing it to the millisecond
    (-t) can neither drop the last frame nor take one more. Raises
    ValueError for --frames on a video without a frame rate."""
    if args.frames is None:
        return args.duration or 0.0
    if test.fps <= 0:
        raise ValueError("its frame rate is unknown, so --frames cannot count its frames; use --duration")
    return (args.frames - 0.5) / test.fps


def _row_options(args: argparse.Namespace, settings: Settings, metrics: tuple[str, ...],
                 test: VideoInfo) -> VmafOptions:
    """The options the window's row would hold for these choices, for the
    test video `test`."""
    vmaf_on = _where(args, "vmaf")
    options = VmafOptions(
        gpu_decode=settings.default_gpu_decode and not args.no_gpu_decode,
        vmaf_on_gpu=settings.default_vmaf_on_gpu if vmaf_on is None else vmaf_on == "gpu",
        n_threads=args.threads or 0,
        n_subsample=args.subsample,
        scale_algorithm=args.scaler,
        scale_direction=_SCALE_TO[args.scale_to],
        duration_limit=_duration_limit(args, test),
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
    """Where SSIMULACRA2 and Butteraugli run: as asked on the command line,
    else as saved in Settings."""
    chosen = {}
    for key in ("ssimulacra2", "butteraugli"):
        asked = _where(args, key)
        saved = getattr(settings, f"default_{key}_backend", "gpu")
        chosen[key] = asked or ("cpu" if saved == "cpu" else "gpu")
    return chosen


def cvvdp_displays(args: argparse.Namespace, settings: Settings) -> list[tuple[str, CvvdpSettings]]:
    """The CVVDP displays to score on, by preset name, each with its
    settings: those --cvvdp-display names, else the app's default. Raises
    ValueError for a name that is no preset."""
    if not args.cvvdp_displays:
        preset = preset_named(settings.cvvdp_default_preset, settings.cvvdp_presets) or DEFAULT_PRESET
        return [(preset.name, CvvdpSettings(preset.settings.display))]
    displays = []
    for name in dict.fromkeys(args.cvvdp_displays):  # each once, in the order given
        preset = preset_named(name, settings.cvvdp_presets)
        if preset is None:
            names = ", ".join(repr(preset.name) for preset in cvvdp_presets(settings.cvvdp_presets))
            raise ValueError(f"No CVVDP display preset is called {name!r}. The presets: {names}.")
        # A preset is a display; scaling to fill it starts off, as for a
        # video added to the window (cvvdp.default_settings).
        displays.append((preset.name, CvvdpSettings(preset.settings.display)))
    return displays


def _durations_hint(error: str) -> str:
    """`error`, with how the command line compares the opening both videos
    have when their lengths differ (geometry.pair_problem's message)."""
    if error.startswith("Durations do not match"):
        return f"{error} Here: --frames N or --duration SECONDS."
    return error


def _colour_notes(args: argparse.Namespace, source: VideoInfo, video: Video,
                  displays: list[tuple[str, CvvdpSettings]]) -> list[str]:
    """What the reader should know about how the colours were read: a
    video's guessed, and HDR video scored on an SDR display."""
    notes = []
    if args.colour is None and _COLOUR_METRICS & set(video.metrics):
        for info, which in ((source, "the reference"), (video.info, "the test video")):
            if not has_colour_tags(info):
                notes.append(guessed_colour_note(info, which))
    name, settings = displays[0]
    if ("cvvdp" in video.metrics and not args.cvvdp_displays and not settings.display.hdr
            and source.color_transfer.casefold() in _HDR_TRANSFERS):
        notes.append(f"the video is HDR, and CVVDP modelled the app's default display, {name!r}, which is SDR. "
                     f"--cvvdp-display {_SUGGESTED_HDR_DISPLAY!r} (or another HDR preset) models an HDR one.")
    return notes


def prepare(args: argparse.Namespace, settings: Settings, source: VideoInfo,
            displays: list[tuple[str, CvvdpSettings]]) -> tuple[list[Video], list[VmafJob]]:
    """Every test video, and the jobs for the ones that can be compared.
    `source` is the reference as the run reads it (with --colour applied);
    CVVDP models the first of `displays`, and each further one in a job of
    its own."""
    asked = args.metrics or _default_metrics(settings)
    backends = _backends(args, settings)
    # The cache knows a video by its file, not by the colours it was read
    # with, so --colour scores are neither read nor saved (save_scores).
    use_saved = not args.recalculate and settings.use_cache and args.colour is None
    videos: list[Video] = []
    jobs: list[VmafJob] = []
    for path in expand_tests(args.tests, args.reference):
        video = Video(path, save_scores=args.colour is None)
        videos.append(video)
        try:
            video.info = with_assumed_colour(probe_video(path), args.colour)
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
        video.notes += _colour_notes(args, source, video, displays)
        try:
            video.options = _row_options(args, settings, video.metrics, video.info)
            validate_video_pair(source, video.info, video.options)
            size = analysis_dimensions(source, video.info, video.options, None, None)
            model = resolve_model(video.options, *size) if video.options.compute_vmaf else ""
        except (ValueError, VmafRunError) as error:
            video.error = _durations_hint(str(error))
            continue
        name, cvvdp = displays[0]
        if "cvvdp" in video.metrics:
            video.display = name
        _queue(video, source, model, cvvdp, backends, use_saved, jobs)
        if "cvvdp" not in video.metrics:
            continue
        for name, cvvdp in displays[1:]:
            other = Video(path, info=video.info, metrics=("cvvdp",), display=name, for_display_only=True,
                          save_scores=video.save_scores)
            other.options = _row_options(args, settings, other.metrics, video.info)
            video.other_displays.append(other)
            _queue(other, source, "", cvvdp, backends, use_saved, jobs)
    return videos, jobs


def _queue(video: Video, source: VideoInfo, model: str, cvvdp: CvvdpSettings, backends: dict[str, str],
           use_saved: bool, jobs: list[VmafJob]) -> None:
    """Adds the job that calculates what `video` asks for, less what is
    saved; with everything saved, takes the saved result and adds none."""
    # The request the window makes for a row: what the cache is asked
    # with and stored under, so the two find each other's scores.
    video.request = analysis_request_from_vmaf_options(video.options, video.metrics, backends, cvvdp)
    saved = None
    if use_saved:
        found = result_cache.load_cached(source.path, video.path, video.request)
        saved = found[0] if found else None
    # The saved scores the run keeps, by the window's rule.
    reusable = (metric_cache.reusable(saved, video.request.metrics, backends, perceptual_vship.gpu_can_score,
                                      metric_cache.video_infos(source, video.info))
                if saved is not None else MetricResultSet())
    if all(reusable.has(key) for key in video.metrics):
        # Everything asked for is saved: nothing to run, as the window
        # runs nothing for a row that has its scores.
        video.result, video.from_cache = saved, True
        return
    video.job_index = len(jobs)
    jobs.append(VmafJob(
        source, video.info, replace(video.options, model=model), label=video.path.stem,
        result_distorted_path=video.path, metric_keys=video.metrics, metric_backends=dict(backends), cvvdp=cvvdp,
        cached_result=saved if reusable else None, cached_metrics=reusable if reusable else None,
    ))


def every_target(videos: list[Video]) -> list[Video]:
    """Each video, followed by its other CVVDP displays."""
    return [target for video in videos for target in (video, *video.other_displays)]


# ------------------------------------------------------------------- progress

class Progress:
    """The run's progress on stderr: one line a video, worded as in the
    window's run line. On a terminal the line is redrawn in place; elsewhere
    it is repeated every few seconds."""

    def __init__(self, videos: list[Video], *, quiet: bool, stream=None, clock=time.monotonic) -> None:
        self._names = {video.job_index: video.label for video in every_target(videos) if video.job_index is not None}
        self._quiet, self._stream, self._clock = quiet, stream or sys.stderr, clock
        self._redraw = bool(getattr(self._stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._last: dict[int, float] = {}
        self._shown = ""  # the line being redrawn

    def _write(self, text: str, *, keep: bool) -> None:
        """Writes a progress line, kept or to be redrawn in place."""
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
        """Writes a line that stays."""
        with self._lock:
            self._write(text, keep=True)

    def tasks(self, index: int, snapshots: list) -> None:
        """Shows a video's progress line."""
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
        """Wipes the line being redrawn, at the end of the run."""
        with self._lock:
            if self._redraw and self._shown and not self._quiet:
                with contextlib.suppress(OSError, ValueError):
                    self._stream.write("\r" + " " * len(self._shown) + "\r")
                    self._stream.flush()
            self._shown = ""


# ---------------------------------------------------------------------- a run

def run(videos: list[Video], jobs: list[VmafJob], settings: Settings, parallel: int, progress: Progress,
        install_cancel=None) -> bool:
    """Runs the jobs, saving each result as the window does unless the
    video's save_scores is False. True when the run was cancelled.
    `install_cancel` is given the function that cancels it (Ctrl+C)."""
    by_job = {video.job_index: video for video in every_target(videos) if video.job_index is not None}
    cancelled = threading.Event()
    # A video's CVVDP on other displays is saved in the same comparison
    # folder, and on Windows two jobs replacing its context.json at once
    # can fail (PermissionError), losing a score: one save at a time.
    saving = threading.Lock()

    def save(index: int, result: ComparisonResult) -> None:
        """Keeps a video's result so far, and saves it to the cache."""
        video = by_job[index]
        video.result = result
        if not video.save_scores:
            return
        try:
            with saving:
                result_cache.store(result.source, video.path, result, video.path.stem, video.request,
                                   result_cache.cache_dir())
        except OSError as error:
            progress.message(f"{video.label}: the scores could not be saved ({error})")

    def finished(index: int, result: ComparisonResult) -> None:
        """Saves a finished video's result and says it is done."""
        save(index, result)
        progress.message(f"{by_job[index].label}: done")

    def failed(index: int, message: str, _stderr_tail: str) -> None:
        """Records why a video failed, and says so."""
        by_job[index].error = message
        progress.message(f"{by_job[index].label}: FAILED: {message}")

    def partly_failed(index: int, result: ComparisonResult, message: str, _stderr_tail: str, failures: dict) -> None:
        """Saves a video's scored metrics and records why the others failed."""
        save(index, result)
        by_job[index].failures = dict(failures) or {"": message}
        progress.message(f"{by_job[index].label}: some metrics FAILED: {message}")

    events = RunEvents(
        job_started=lambda index, _label: progress.message(f"{by_job[index].label}: started"),
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
    # The run goes on a thread of its own, and the main thread only waits
    # for it in short steps. Don't run it on the main thread: Python runs a
    # signal's handler between the main thread's steps, and the run waits
    # in long steps of its own, so Ctrl+C was answered 6 seconds late.
    failure: list[BaseException] = []

    def work() -> None:
        """Runs the jobs, keeping any error for the main thread."""
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
    """A crop box as --json writes it, or None."""
    return None if box is None else {"x": box.x, "y": box.y, "width": box.w, "height": box.h}


def _media(info: VideoInfo) -> dict:
    """A video's media information as --json writes it."""
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
        if video.display:
            entry["cvvdp_display"] = video.display
        if video.other_displays:
            entry["cvvdp_displays"] = [_display_entry(target) for target in (video, *video.other_displays)]
        document["videos"].append(entry)
    return document


def _display_entry(target: Video) -> dict:
    """CVVDP on one display, as --json lists each display's: the display,
    and its score or why there is none."""
    summary = metric_summary(target.result, "cvvdp") if target.result is not None else None
    entry: dict = {"display": target.display, "score": None if summary is None else summary["score"]}
    if target.result is not None:
        entry["from_saved_scores"] = target.from_cache
    if reason := target.error or target.failures.get("cvvdp") or next(iter(target.failures.values()), ""):
        entry["error"] = reason
    return entry


def video_status(video: Video, cancelled: bool) -> str:
    """ok: every metric asked for; partial: some; failed: none; cancelled:
    the run was stopped before it had them all."""
    have = [key for key in video.metrics if video.result is not None and video.result.has_metric(key)]
    others_scored = all(other.result is not None and other.result.has_metric("cvvdp") and not other.failures
                        for other in video.other_displays)
    if video.metrics and len(have) == len(video.metrics) and not video.failures and others_scored:
        return "ok"
    if cancelled and not video.error and not video.failures:
        return "cancelled"
    return "partial" if have else "failed"


def results_table(source: VideoInfo, videos: list[Video], cancelled: bool) -> str:
    """The run as plain-text tables, printed unless the command says "--json -"."""
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
                # With several displays, each CVVDP row names its own.
                label = f"{definition.label} ({video.display})" if video.other_displays else definition.label
                rows.append((label, show(_value(summary["score"])), "", "", "", "", summary["computed_on"].upper()))
                continue
            rows.append((definition.label, show(_value(summary["score"])), show(_value(summary["median"])),
                         show(_value(summary["min"])), show(_value(summary["max"])),
                         f"{show(_value(summary['worst_1_percent']))} ({summary['worst_is']})",
                         summary["computed_on"].upper()))
        rows += _other_display_rows(video, lines)
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


def _other_display_rows(video: Video, lines: list[str]) -> list[tuple[str, ...]]:
    """A table row for CVVDP on each of the video's other displays; why
    there is none goes to `lines`."""
    rows = []
    definition = metric_definition("cvvdp")
    for other in video.other_displays:
        summary = metric_summary(other.result, "cvvdp") if other.result is not None else None
        if summary is None:
            reason = other.error or next(iter(other.failures.values()), "") or "not calculated"
            lines.append(f"  CVVDP on {other.display!r} failed: {reason}")
            continue
        rows.append((f"{definition.label} ({other.display})", definition.format_value(_value(summary["score"])),
                     "", "", "", "", summary["computed_on"].upper()))
    return rows


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
    """`number` as a number, when it is given as text."""
    if isinstance(number, str):
        return float(number)
    return number


def _media_text(info: VideoInfo) -> str:
    """A video's media information on one line."""
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
    vmaf_gpu.set_gpu_backend(settings.gpu_backend)


def _ffmpeg_problem() -> str | None:
    """Why FFmpeg cannot be used, or None when it can."""
    status = check_tools()
    if status.ok:
        return None
    return ("FFmpeg cannot be used:\n  " + "\n  ".join(status.problems)
            + "\nPut ffmpeg and ffprobe on PATH, or set FFmpeg's folder in the app's Settings tab.")


def compare(args: argparse.Namespace, *, out=None, err=None) -> int:
    """The compare command: compares the test videos with the reference; returns the exit code."""
    out, err = out or sys.stdout, err or sys.stderr
    settings = Settings.load()
    _apply_settings(settings)
    if (problem := _ffmpeg_problem()) is not None:
        print(problem, file=err)
        return EXIT_USAGE
    try:
        displays = cvvdp_displays(args, settings)
    except ValueError as error:
        print(error, file=err)
        return EXIT_USAGE
    try:
        source = with_assumed_colour(probe_video(args.reference), args.colour)
    except (ProbeError, OSError) as error:
        print(f"The reference could not be read: {error}", file=err)
        return EXIT_USAGE
    videos, jobs = prepare(args, settings, source, displays)
    if not videos:
        print("No test videos: the patterns find only the reference.", file=err)
        return EXIT_USAGE
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
        # What runs ended outright left behind (End task, a crash) goes
        # while this one starts.
        temp_files.start_removing_leftovers()
        parallel = args.parallel or settings.parallel_jobs

        def on_interrupt(cancel) -> None:
            """Makes Ctrl+C, Ctrl+Break and a closed console cancel the run."""
            def handler(_signal, _frame) -> None:
                """Cancels the run."""
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
    # "--json -" is for a program to read, so it prints the JSON alone.
    # "--json FILE" writes the file, and the tables are printed as well.
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
    """The devices command: prints the FFmpeg version and where each metric
    runs; returns the exit code."""
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
    available, why = vmaf_gpu.gpu_vmaf_available()
    if available:
        backend, gpu = vmaf_gpu.gpu_vmaf_backend()
        lines.append("VMAF v0.6.1, VMAF NEG: GPU ("
                     + ("libvmaf CUDA" if backend == "cuda" else f"Vulkan, GPU {gpu}") + ")")
    else:
        lines.append(f"VMAF v0.6.1, VMAF NEG: CPU ({why})")
    from vmaf_app.core import vmaf_runner, vmaf_v1_gpu

    with_gpu, what = vmaf_v1_gpu.available()
    lines.append(f"VMAF v1: {'GPU (' + what + ')' if with_gpu else 'CPU (' + what + ')'}")
    if vmaf_runner.cpu_metrics_in_app():
        lines.append(f"PSNR, SSIM, XPSNR: CPU (libvmaf-fast {libvmaf.LIBVMAF_FAST_VERSION} in the app; "
                     "FFmpeg beside VMAF on the CPU)")
    else:
        lines.append("PSNR, SSIM, XPSNR: CPU (FFmpeg)")
    print("\n".join(lines), file=out)
    return EXIT_OK if status.ok else EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    """Runs the command line; returns the exit code."""
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
