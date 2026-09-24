"""Standalone CPU perceptual-metric backend.

SSIMULACRA2 and Butteraugli are reference command-line tools for *images*,
not video filters.  This adapter gives them lossless PNG pairs from one
FFmpeg pass per comparison, scoring each pair as soon as it is written and
holding FFmpeg to a small backlog (see _png_pairs). The temporary directory
exists only for the lifetime of the task.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vmaf_app.core import proc as proc_util
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.crop_detect import CropDetectCancelled, detect_crop, detect_pair
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import CropBox, CropMode, ScaleDirection, VideoInfo
from vmaf_app.core.process_control import ProcessHandle

BACKEND_ID = "perceptual"
_NUMBER = re.compile(r"(?<![\w.])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


class PerceptualRunError(RuntimeError):
    """A recoverable failure from the standalone perceptual metric backend."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class PerceptualCancelled(RuntimeError):  # noqa: N818 - mirrors the established runner exception
    """The caller cancelled a perceptual metric task."""


@dataclass(frozen=True, slots=True)
class PerceptualTaskOutput:
    metrics: MetricResultSet
    source_crop: CropBox | None
    distorted_crop: CropBox | None
    compared_frame_count: int
    #: Metrics that could not be scored while the others were, by key, with
    #: the reason -- CVVDP, which only runs on the GPU, failing beside
    #: SSIMULACRA2/Butteraugli. The worker reports these as a partial failure.
    failures: dict[str, str] = field(default_factory=dict)


def find_metric_executable(metric: str) -> str | None:
    """Find the bundled or explicitly configured reference CLI.

    Environment overrides keep this optional tooling out of app settings and
    make CI/fake executable tests deterministic: ``SSIMULACRA2_PATH`` and
    ``BUTTERAUGLI_PATH``.  Packaged builds ship the official static libjxl
    tools, so a normal user does not need to install either executable.
    """
    if metric not in {"ssimulacra2", "butteraugli"}:
        raise KeyError(metric)
    override = os.environ.get(f"{metric.upper()}_PATH", "").strip()
    if override:
        return override if Path(override).exists() else None
    bundled_names = {
        "ssimulacra2": ("ssimulacra2.exe", "ssimulacra2"),
        # libjxl names the CLI butteraugli_main; accept the short name for
        # user-provided installations as well.
        "butteraugli": ("butteraugli_main.exe", "butteraugli.exe", "butteraugli_main", "butteraugli"),
    }[metric]
    bundled_dir = Path(__file__).resolve().parents[1] / "tools" / "libjxl"
    for name in bundled_names:
        candidate = bundled_dir / name
        if candidate.is_file():
            return str(candidate)
    for name in bundled_names:
        on_path = shutil.which(name)
        if on_path:
            return on_path
    return None


def _tool_version(executable: str) -> str:
    for arg in ("--version", "-version", "-V"):
        try:
            completed = proc_util.run([executable, arg], capture_output=True, text=True, timeout=5)
        except OSError:
            return "unknown"
        if completed.returncode == 0:
            line = (completed.stdout or completed.stderr or "").strip().splitlines()
            return line[0][:160] if line else "unknown"
    return "unknown"


def _implementation_version(executable: str) -> str:
    """Prefer the bundled release manifest when a CLI omits its version flag."""
    version = _tool_version(executable)
    bundled_dir = (Path(__file__).resolve().parents[1] / "tools" / "libjxl").resolve()
    try:
        is_bundled = Path(executable).resolve().parent == bundled_dir
    except OSError:
        is_bundled = False
    if version == "unknown" and is_bundled:
        manifest = bundled_dir / "LIBJXL-VERSION.txt"
        if manifest.is_file():
            return manifest.read_text(encoding="utf-8").splitlines()[0].strip()
    return version


def _resolve_crops(
    source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe,
    cancel_event: threading.Event | None, process_handle: ProcessHandle | None,
    on_status: Callable[[str], None] | None,
) -> tuple[CropBox | None, CropBox | None]:
    if recipe.crop_mode is CropMode.NONE:
        return None, None
    if recipe.crop_mode is CropMode.MANUAL:
        return recipe.manual_source_crop, recipe.manual_distorted_crop
    try:
        if on_status:
            on_status("Detecting black bars for perceptual metrics…")
        # Crop detection samples representative windows across the whole
        # file. A short score-duration limit may land entirely in a dark
        # intro and must not define the crop used for the comparison.
        return detect_pair(
            lambda: detect_crop(source, cancel_event=cancel_event, process_handle=process_handle),
            lambda: detect_crop(distorted, cancel_event=cancel_event, process_handle=process_handle),
        )
    except CropDetectCancelled as exc:
        raise PerceptualCancelled("Cancelled by user") from exc


#: Longer than this, SSIMULACRA2/Butteraugli run on the CPU only when the
#: user has agreed to it: the CPU tools score one still-image pair at a time,
#: 1.1 s (SSIMULACRA2) and 2.0 s (Butteraugli) for a 3840x1608 pair with the
#: bundled libjxl 0.12.0 tools -- days for a film. The Videos tab asks before such a run
#: (MainWindow._confirm_long_cpu_perceptual) and the GPU path does not fall
#: back to the CPU on its own past it (perceptual_vship.apply_vship_cpu_fallback).
LONG_CPU_RUN_SECONDS = 10 * 60


def compared_seconds(source: VideoInfo, distorted: VideoInfo, duration_limit: float) -> float:
    """How much video a comparison scores: the shorter input, capped by the
    duration limit when one is set."""
    seconds = min(source.duration, distorted.duration)
    if duration_limit > 0:
        seconds = min(seconds, duration_limit)
    return seconds


def _content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
    return (crop.w, crop.h) if crop else (info.width, info.height)


def _validate_pair(source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe) -> None:
    if source.is_variable_frame_rate or distorted.is_variable_frame_rate:
        raise PerceptualRunError("Variable-frame-rate video is not supported safely yet.")
    fps_tolerance = max(0.01, max(source.fps, distorted.fps) * 0.001)
    if abs(source.fps - distorted.fps) > fps_tolerance:
        raise PerceptualRunError(f"Frame rates do not match ({source.fps:.3f} vs {distorted.fps:.3f} fps).")
    if recipe.duration_limit > 0 and min(source.duration, distorted.duration) + 0.1 < recipe.duration_limit:
        raise PerceptualRunError("The duration limit extends beyond the end of one of the videos.")


def _crop_filter(crop: CropBox | None, info: VideoInfo) -> list[str]:
    if crop is None or crop.is_noop(info.width, info.height):
        return []
    return [crop.as_filter()]


def _image_filtergraph(
    source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe,
    source_crop: CropBox | None, distorted_crop: CropBox | None, step: int,
) -> str:
    """Return two lossless, identically sized RGB48 image streams.

    RGB48 avoids JPEG and 8-bit intermediate losses.  The command uses a
    single decoder pass for each input, then both requested image metrics use
    each resulting pair, so adding Butteraugli to SSIMULACRA2 does not decode
    the videos again.
    """
    source_size = _content_size(source, source_crop)
    distorted_size = _content_size(distorted, distorted_crop)
    if source_size != distorted_size:
        if recipe.scale_direction is ScaleDirection.DISTORTED_TO_SOURCE:
            distorted_target, source_target = source_size, None
        else:
            distorted_target, source_target = None, distorted_size
    else:
        distorted_target = source_target = None

    def chain(input_label: str, output_label: str, crop: CropBox | None, info: VideoInfo, target: tuple[int, int] | None) -> str:
        ops = _crop_filter(crop, info)
        if target is not None:
            ops.append(f"scale={target[0]}:{target[1]}:flags={recipe.scale_algorithm}")
        # select keeps a generic sampling axis distinct from FFmpeg/libvmaf.
        if step > 1:
            ops.append(f"select=not(mod(n\\,{step}))")
        ops += ["setpts=PTS-STARTPTS", "format=rgb48le"]
        return f"[{input_label}]{','.join(ops)}[{output_label}]"

    return ";".join((
        chain("0:v", "distorted", distorted_crop, distorted, distorted_target),
        chain("1:v", "reference", source_crop, source, source_target),
    ))


#: Frame pairs FFmpeg may write ahead of the scoring before it is paused.
#: Extraction runs far faster than the tools score (1-2 s a 4K pair), so
#: unthrottled it wrote the whole video out first: about 10 MB of 16-bit PNG
#: per 4K frame, terabytes for a film, and no progress until it finished.
#: 24 pairs of 4K is about 0.5 GB. FFmpeg resumes once half are scored.
_BACKLOG_PAIRS = 24


def _png_pairs(
    source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe,
    source_crop: CropBox | None, distorted_crop: CropBox | None, step: int,
    directory: Path, cancel_event: threading.Event | None,
    process_handle: ProcessHandle | None,
) -> Iterator[tuple[Path, Path]]:
    """Yield (reference, test) image pairs while FFmpeg is still writing them.

    One FFmpeg decodes both videos and writes lossless 16-bit RGB PNGs.
    Each image is written under a temporary name and renamed when complete
    (-atomic_writing), so an image that exists is whole. The caller scores
    and deletes each pair as it arrives.

    FFmpeg is suspended while it has _BACKLOG_PAIRS complete pairs ready
    ahead of the caller and resumed at half that, checked every 10 ms on a
    thread of its own (a check made only as each pair was handed out let
    FFmpeg run far ahead while one slow pair was being scored). Only
    complete pairs count: the two sequences come from two decoders and one
    can run well ahead of the other -- a 4K AV1 test ran 24 frames ahead of
    its HEVC reference -- and suspending on the leading side alone stopped
    the lagging side too, so the pair the caller was waiting for never came.
    While the caller waits, FFmpeg is always running. The suspend nests
    with the user's Pause (Windows counts suspends), so neither can undo
    the other.

    Each output runs to its own input's end; the comparison is the frames
    both have (libvmaf's shortest=1), so the pairs end when either sequence
    does.
    """
    graph = _image_filtergraph(source, distorted, recipe, source_crop, distorted_crop, step)
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y", "-i", str(distorted.path.resolve()),
           "-i", str(source.path.resolve()), "-filter_complex", graph]
    output_args = ["-fps_mode", "passthrough", "-pix_fmt", "rgb48le", "-atomic_writing", "1"]
    if recipe.duration_limit > 0:
        output_args = ["-t", f"{recipe.duration_limit:.3f}", *output_args]
    cmd += ["-map", "[distorted]", *output_args, str(directory / "test-%08d.png")]
    cmd += ["-map", "[reference]", *output_args, str(directory / "reference-%08d.png")]
    if cancel_event is not None and cancel_event.is_set():
        raise PerceptualCancelled("Cancelled by user")
    # A long FFmpeg extraction can write enough diagnostics to fill a pipe.
    # We do not need its progress stream here, so inherit neither pipe and
    # avoid deadlocking a feature-length task before it creates a frame.
    process = proc_util.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    if process_handle is not None:
        process_handle.attach(process.pid)
    throttled = False
    throttle_lock = threading.Lock()
    consumed = [1]  # the pair the caller is on; read by the regulator
    stop = threading.Event()

    # FFmpeg and anything under it -- the real ffmpeg.exe when the one
    # started is a launcher (see proc.process_tree). Listed once, as soon
    # as the first pair exists (so the real FFmpeg does too), and kept:
    # listing takes 13-40 ms on Windows. Listed at the moment FFmpeg had to
    # stop, that delay let it write 75-86 images against a limit of 52.
    tree: list = []

    def list_tree() -> None:
        if tree:
            return
        found = proc_util.process_tree(process.pid)
        with throttle_lock:
            if not tree:
                tree.extend(found)

    def throttle(on: bool) -> None:
        nonlocal throttled
        if on:
            list_tree()
        with throttle_lock:
            if on == throttled:
                return
            proc_util.signal_processes(tree, "suspend" if on else "resume")
            throttled = on

    def pair(number: int) -> tuple[Path, Path]:
        return directory / f"reference-{number:08d}.png", directory / f"test-{number:08d}.png"

    def complete(number: int) -> bool:
        reference, test = pair(number)
        return reference.exists() and test.exists()

    def regulate() -> None:
        while not stop.wait(0.01):
            if not tree and complete(1):
                list_tree()
            index = consumed[0]
            if complete(index + _BACKLOG_PAIRS):
                throttle(True)
            elif not complete(index + _BACKLOG_PAIRS // 2):
                throttle(False)

    regulator = threading.Thread(target=regulate, name="png-backlog", daemon=True)
    regulator.start()
    try:
        index = 1
        while True:
            consumed[0] = index
            reference, test = pair(index)
            while not (reference.exists() and test.exists()):
                if cancel_event is not None and cancel_event.is_set():
                    raise PerceptualCancelled("Cancelled by user")
                code = process.poll()
                if code is not None:
                    if reference.exists() and test.exists():
                        break  # written just before FFmpeg exited
                    if code != 0:
                        raise PerceptualRunError("FFmpeg could not prepare lossless perceptual-metric frames.")
                    if index == 1:
                        raise PerceptualRunError("FFmpeg produced no frame pairs for perceptual metrics.")
                    return  # the shorter input has ended
                throttle(False)  # the caller is waiting: FFmpeg must run
                time.sleep(0.02)
            yield reference, test
            index += 1
    finally:
        stop.set()
        regulator.join()
        throttle(False)
        if process.poll() is None:
            proc_util.terminate(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        if process_handle is not None:
            process_handle.detach(process.pid)


def parse_score(metric: str, output: str) -> float:
    """Extract the scalar reference-tool score from normal text output."""
    values = _NUMBER.findall(output)
    if not values:
        raise PerceptualRunError(f"{metric} did not produce a numeric score: {output.strip()[:300]}")
    try:
        return float(values[-1])
    except ValueError as exc:
        raise PerceptualRunError(f"Could not parse {metric} score.") from exc


#: Longest one tool may run on one frame pair, counting only time the job is
#: not paused. The bundled tools take 1-2 s for a 4K pair.
_TOOL_TIMEOUT_SECONDS = 120.0


def _run_metric(
    executable: str, metric: str, reference: Path, test: Path,
    process_handle: ProcessHandle | None = None,
    cancel_event: threading.Event | None = None,
) -> float:
    """Score one frame pair with a still-image tool.

    The tool process is attached to the job's handle like FFmpeg is, so
    Pause suspends it and Cancel ends it at once. It used to run outside the
    handle: pausing a CPU run left the tools scoring while the app said
    "Paused", and Cancel waited for the frame in progress.
    """
    try:
        process = proc_util.popen(
            [executable, str(reference), str(test)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as exc:
        raise PerceptualRunError(f"Could not run {metric}: {exc}") from exc
    if process_handle is not None:
        process_handle.attach(process.pid)
    try:
        running, last = 0.0, time.monotonic()
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if not (process_handle is not None and process_handle.is_pause_requested):
                    running += now - last
                last = now
                if cancel_event is not None and cancel_event.is_set():
                    proc_util.kill(process)
                    process.communicate()
                    raise PerceptualCancelled("Cancelled by user") from None
                if running > _TOOL_TIMEOUT_SECONDS:
                    proc_util.kill(process)
                    process.communicate()
                    raise PerceptualRunError(
                        f"{metric} did not finish a frame within {_TOOL_TIMEOUT_SECONDS:.0f} s."
                    ) from None
    finally:
        if process_handle is not None:
            process_handle.detach(process.pid)
    combined = (stdout or "") + "\n" + (stderr or "")
    if process.returncode != 0:
        raise PerceptualRunError(f"{metric} failed for a frame.", combined[-2000:])
    return parse_score(metric, combined)


def run_perceptual_task(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    resolved_crops: tuple[CropBox | None, CropBox | None] | None = None,
) -> PerceptualTaskOutput:
    """Execute the CPU reference implementation and return independent frame results."""
    if not specs or any(spec.backend_id != BACKEND_ID for spec in specs):
        raise ValueError("perceptual task requires perceptual metric specs")
    if request.recipe.resample_test is not None:
        raise PerceptualRunError("Perceptual CPU metrics do not support resolution round-trip tests yet.")
    _validate_pair(source, distorted, request.recipe)
    executables: dict[str, str] = {}
    for spec in specs:
        executable = find_metric_executable(spec.key)
        if executable is None:
            raise PerceptualRunError(
                f"{spec.key} is not installed. Install the reference {spec.key} executable "
                f"or set {spec.key.upper()}_PATH."
            )
        executables[spec.key] = executable
    steps = {spec.coverage.step if spec.coverage is not None else 1 for spec in specs}
    if len(steps) != 1:
        raise PerceptualRunError("Perceptual metrics in one task must use the same frame coverage.")
    step = steps.pop()
    source_crop, distorted_crop = resolved_crops or _resolve_crops(
        source, distorted, request.recipe, cancel_event, process_handle, on_status
    )
    expected = min(source.estimated_frame_count, distorted.estimated_frame_count)
    if request.recipe.duration_limit > 0:
        expected = min(expected, max(1, math.ceil(request.recipe.duration_limit * source.fps)))
    total_units = max(1, math.ceil(expected / step)) * step
    if on_status:
        on_status("Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted…")
    started = time.perf_counter()
    values: dict[str, list[float]] = {spec.key: [] for spec in specs}
    total = 0
    with tempfile.TemporaryDirectory(prefix="videometricslab-perceptual-") as temp:
        pairs = _png_pairs(
            source, distorted, request.recipe, source_crop, distorted_crop, step,
            Path(temp), cancel_event, process_handle,
        )
        try:
            for reference, test in pairs:
                if cancel_event is not None and cancel_event.is_set():
                    raise PerceptualCancelled("Cancelled by user")
                try:
                    for spec in specs:
                        values[spec.key].append(_run_metric(
                            executables[spec.key], spec.key, reference, test, process_handle, cancel_event,
                        ))
                finally:
                    # Each pair is deleted once every selected tool has
                    # scored it; with the extraction held to a small backlog
                    # the folder never holds more than a few dozen images.
                    reference.unlink(missing_ok=True)
                    test.unlink(missing_ok=True)
                total += 1
                if on_progress:
                    done = total * step
                    rate = done / max(time.perf_counter() - started, 1e-6)
                    # The estimate can run short of the real length; the
                    # total grows with the work rather than passing 100%.
                    on_progress(done, max(total_units, done + step), rate)
        finally:
            pairs.close()  # stops FFmpeg if the scoring ended early
    if on_progress:
        on_progress(total * step, total * step, total * step / max(time.perf_counter() - started, 1e-6))
    frame = np.arange(total, dtype=np.int32) * step
    times = frame.astype(np.float64) / max(source.fps, 1.0)
    results = MetricResultSet()
    for spec in specs:
        version = _implementation_version(executables[spec.key])
        compatibility = f"{spec.key}-libjxl-cpu-v1"
        results.add(FrameMetricResult(
            spec.key, frame, times, np.asarray(values[spec.key], dtype=np.float32),
            MetricProvenance(
                implementation=spec.key,
                implementation_version=version,
                compute_backend="cpu",
                implementation_compatibility_id=compatibility,
                parameters={"intermediate": "png/rgb48le", "coverage_step": step},
            ),
        ))
    return PerceptualTaskOutput(results, source_crop, distorted_crop, total * step)
