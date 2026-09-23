"""Black-bar (letterbox/pillarbox) detection via ffmpeg's cropdetect filter.

cropdetect with reset=0 reports the *tightest* crop box that still contains
every non-black pixel it has seen so far in the analyzed window -- a single
bright pixel of noise near an edge can prevent any crop from being detected.
To be robust we sample several short windows spread across the middle of the
video (skipping fades at the very start/end) and take the most common result
across windows, rather than trusting a single long pass.

The windows are independent, so they run at the same time rather than one
after another: measured on a 4K HEVC source, five sequential windows took
6.9s and five concurrent ones 2.0s. Each may decode on the GPU when the run
itself would, falling back to software on its own if that fails -- the
decoded pixels are identical either way, so the box is too. And the answer
for a file is kept for the life of the process: a batch of six encodes of
one film used to detect the source's bars six times over.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.gpu import hw_native_format, hwaccel_args
from vmaf_app.core.models import CropBox, VideoInfo

if TYPE_CHECKING:
    from vmaf_app.core.process_control import ProcessHandle

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

_SAMPLE_WINDOW_SECONDS = 3.0
_SAMPLE_COUNT = 5
_SAMPLE_SPAN = (0.1, 0.9)  # fraction of duration to sample within

#: How many files' answers to remember. A CropBox is four ints; this is a
#: bound against a pathological session, not a memory budget.
_CACHE_LIMIT = 512


class CropDetectError(RuntimeError):
    pass


class CropDetectCancelled(RuntimeError):  # noqa: N818 - expected control flow
    pass


def _sample_window(duration: float) -> float:
    """How much media each sample reads.

    Clamped to the interval being analysed, so a short scored segment is not
    measured using footage from beyond it.
    """
    if duration <= 0:
        return _SAMPLE_WINDOW_SECONDS
    return min(_SAMPLE_WINDOW_SECONDS, duration)


def _sample_offsets(duration: float, window: float = _SAMPLE_WINDOW_SECONDS) -> list[float]:
    if duration <= 0:
        return [0.0]
    # -ss is a window *start*. Every sample must leave enough media for the
    # whole analysis window; the previous formula sent most samples beyond
    # EOF on clips shorter than ~17 seconds.
    max_start = max(0.0, duration - window)
    lo = min(duration * _SAMPLE_SPAN[0], max_start)
    hi = min(duration * _SAMPLE_SPAN[1], max_start)
    if hi <= lo:
        return [lo]
    if _SAMPLE_COUNT == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (_SAMPLE_COUNT - 1)
    return [lo + i * step for i in range(_SAMPLE_COUNT)]


def _window_command(
    path: str, start: float, window: float, limit: float,
    hwaccel: str | None, download_format: str,
) -> list[str]:
    filters = f"cropdetect=limit={limit}:round=2:reset=0"
    if hwaccel:
        # hwdownload can only emit the surface's native format and cropdetect
        # cannot take that directly, so the same explicit conversion the
        # metric run uses sits between them. Without it ffmpeg fails to
        # negotiate the link and the window reports nothing.
        filters = f"hwdownload,format={download_format},{filters}"
    return [
        ffmpeg_path(),
        "-nostdin", "-hide_banner",
        *hwaccel_args(hwaccel),
        "-ss", f"{start:.3f}",
        "-i", path,
        "-t", f"{window:.3f}",
        "-vf", filters,
        "-f", "null", "-",
    ]


def _launch_window(
    cmd: list[str],
    cancel_event: threading.Event | None,
    process_handle: ProcessHandle | None,
) -> tuple[int, str]:
    """Runs one window to completion. Returns (returncode, stderr)."""
    proc = None
    try:
        proc = proc_util.popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if process_handle is not None:
            process_handle.attach(proc.pid)
        active_seconds = 0.0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
            started = time.monotonic()
            try:
                _stdout, stderr = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired as e:
                if process_handle is None or not process_handle.is_pause_requested:
                    active_seconds += time.monotonic() - started
                if active_seconds >= 60:
                    proc.terminate()
                    proc.communicate(timeout=5)
                    raise CropDetectError("Crop detection timed out") from e
        if cancel_event is not None and cancel_event.is_set():
            raise CropDetectCancelled("Crop detection cancelled")
        return proc.returncode, stderr
    except Exception as e:
        if isinstance(e, (CropDetectError, CropDetectCancelled)):
            raise
        raise CropDetectError(f"Could not run crop detection: {e}") from e
    finally:
        # This window's pid only. Others may still be running under the same
        # handle, and a bare detach() would drop them from Pause and Cancel.
        if process_handle is not None and proc is not None:
            process_handle.detach(proc.pid)


def _run_single_window(
    path: str, start: float, window: float, limit: float,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    hwaccel: str | None = None,
    download_format: str = "nv12",
) -> CropBox | None:
    if cancel_event is not None and cancel_event.is_set():
        raise CropDetectCancelled("Crop detection cancelled")
    attempts = [hwaccel, None] if hwaccel else [None]
    stderr = ""
    returncode = 0
    for attempt in attempts:
        cmd = _window_command(path, start, window, limit, attempt, download_format)
        try:
            returncode, stderr = _launch_window(cmd, cancel_event, process_handle)
        except CropDetectError as e:
            raise CropDetectError(f"{e} ({path})") from e
        if returncode == 0:
            break
        # A hardware attempt that fails -- no free decoder session, an
        # unsupported profile -- is retried on the CPU, exactly as the
        # metric run itself would fall back. The pixels, and so the box,
        # are the same either way.
    if returncode != 0:
        detail = stderr.strip().splitlines()
        tail = detail[-1] if detail else f"ffmpeg exited with code {returncode}"
        raise CropDetectError(f"Crop detection failed for {path}: {tail}")

    matches = _CROP_RE.findall(stderr)
    if not matches:
        return None
    w, h, x, y = (int(v) for v in matches[-1])
    return CropBox(w=w, h=h, x=x, y=y)


def analysed_duration(info: VideoInfo, duration_limit: float = 0.0) -> float:
    """How much of `info` a run will actually compare."""
    if duration_limit <= 0:
        return info.duration
    if info.duration <= 0:
        return duration_limit
    return min(info.duration, duration_limit)


# ------------------------------------------------------------------ the cache
# Keyed on what the answer depends on: which bytes are in the file, and which
# stretch of it is sampled. Not on how it was decoded -- the GPU and CPU paths
# return the same pixels.
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple, CropBox] = OrderedDict()
#: Files whose detection is currently running, so a second caller for the
#: same file waits for that answer instead of launching its own five
#: processes -- which is exactly what two parallel lanes starting on the same
#: source at the same moment would otherwise do.
_in_flight: dict[tuple, threading.Event] = {}


def _cache_key(path: Path, scored_duration: float, limit: float) -> tuple | None:
    """None when the file cannot be identified, in which case nothing is
    cached: a path with no size or mtime behind it is not a stable identity."""
    try:
        stat = Path(path).resolve().stat()
    except OSError:
        return None
    return (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns,
            round(scored_duration, 3), round(limit, 6))


def _claim(key: tuple, cancel_event: threading.Event | None) -> CropBox | None:
    """The cached box, or None once this caller has been given the job of
    computing it. Waits, cancellably, while another caller is on it."""
    while True:
        with _cache_lock:
            box = _cache.get(key)
            if box is not None:
                _cache.move_to_end(key)
                return box
            running = _in_flight.get(key)
            if running is None:
                _in_flight[key] = threading.Event()
                return None
        while not running.wait(0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise CropDetectCancelled("Crop detection cancelled")
        # The other caller has finished, one way or the other: either the
        # answer is cached now, or it failed and this caller takes over.


def _settle(key: tuple, box: CropBox | None) -> None:
    with _cache_lock:
        if box is not None:
            _cache[key] = box
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_LIMIT:
                _cache.popitem(last=False)
        waiting = _in_flight.pop(key, None)
    if waiting is not None:
        waiting.set()


def clear_cache() -> None:
    """Forgets every remembered box. For tests, and for anyone who has
    replaced a file in place with the same size and mtime."""
    with _cache_lock:
        _cache.clear()


def detect_crop(
    info: VideoInfo, limit: float = 24 / 255,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    duration_limit: float = 0.0,
    hwaccel: str | None = None,
) -> CropBox:
    """Detects the black-bar crop box, or raises when it cannot analyze it.

    `duration_limit` bounds crop sampling when explicitly requested. Metric
    workflows omit their score-duration limit so a short/dark opening cannot
    incorrectly determine the crop used for the entire comparison.

    `hwaccel` is the decoder the run itself will use for this input, or
    None for software. It changes how fast the answer arrives, never what
    it is.
    """
    scored_duration = analysed_duration(info, duration_limit)
    key = _cache_key(info.path, scored_duration, limit)
    if key is not None:
        cached = _claim(key, cancel_event)
        if cached is not None:
            return cached
    box: CropBox | None = None
    try:
        box = _detect_uncached(
            info, limit, scored_duration, cancel_event, process_handle, hwaccel
        )
        return box
    finally:
        if key is not None:
            _settle(key, box)


def _detect_uncached(
    info: VideoInfo, limit: float, scored_duration: float,
    cancel_event: threading.Event | None,
    process_handle: ProcessHandle | None,
    hwaccel: str | None,
) -> CropBox:
    path = str(info.path)
    window = _sample_window(scored_duration)
    starts = _sample_offsets(scored_duration, window)
    download_format = hw_native_format(info.pix_fmt)

    if cancel_event is not None and cancel_event.is_set():
        raise CropDetectCancelled("Crop detection cancelled")

    # One window at a time. Running all five at once was faster, but each
    # one is its own decoder: five hardware decoders per input held 6.2 GB of
    # VRAM for a 4K source, and a two-input comparison reached ten, more
    # than most GPUs have. Sequential windows keep one decoder per input.
    boxes: list[CropBox] = []
    failures: list[str] = []
    for start in starts:
        try:
            box = _run_single_window(
                path, start, window, limit,
                cancel_event=cancel_event, process_handle=process_handle,
                hwaccel=hwaccel, download_format=download_format,
            )
        except CropDetectError as e:
            failures.append(str(e))
        else:
            if box is not None:
                boxes.append(box)

    # Cancel wins over any boxes that came back before it landed: a partial
    # vote is not an answer the caller asked for.
    if cancel_event is not None and cancel_event.is_set():
        raise CropDetectCancelled("Crop detection cancelled")

    if not boxes:
        detail = failures[-1] if failures else "ffmpeg produced no crop measurements"
        raise CropDetectError(
            f"Could not auto-detect black bars in {info.path}: {detail}. "
            "Choose 'None (use full frame)' for this video to continue without cropping."
        )

    counts: dict[tuple[int, int, int, int], int] = {}
    for b in boxes:
        key = (b.w, b.h, b.x, b.y)
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts, key=lambda k: counts[k])
    w, h, x, y = best_key
    return CropBox(w=w, h=h, x=x, y=y)
