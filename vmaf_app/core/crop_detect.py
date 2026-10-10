"""Detects black bars (letterbox and pillarbox) with FFmpeg's cropdetect
filter. The main entry points are detect_crop, detect_pair and
common_picture.

cropdetect with reset=0 reports the tightest box that still holds every
non-black pixel it has seen so far, so a single bright pixel of noise near
an edge can keep any crop from being found. So several short windows are
sampled across the middle of the video (skipping fades at the very start
and end), and the most common box wins, rather than trusting one long pass.

A video's windows run one after another, so each video uses one decoder at
a time; the two videos of a comparison are detected at the same time
(detect_pair). A window decodes on the GPU when the run itself would, and
falls back to software if that fails; the decoded pixels are the same
either way, so the box is too. The answer for a file is kept, in this
process and between sessions (see set_store), so a batch of encodes of one
film detects the source's bars once.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, ffmpeg_path
from vmaf_app.core.gpu import hw_native_format, hwaccel_args
from vmaf_app.core.models import CropBox, VideoInfo

if TYPE_CHECKING:
    from vmaf_app.core.process_control import ProcessHandle

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

_SAMPLE_WINDOW_SECONDS = 3.0
_SAMPLE_COUNT = 5
_SAMPLE_SPAN = (0.1, 0.9)  # fraction of duration to sample within

#: How many detection decoders may run at once in the whole app, across
#: every job. One video's windows already run in turn, so this limits the
#: rest (two jobs in parallel, each detecting both its videos) to two, which
#: is 2.5 GB of GPU memory for 4K on the GPU decoder, however many jobs run.
_MAX_DETECTION_DECODERS = 2
_decoder_slots = threading.BoundedSemaphore(_MAX_DETECTION_DECODERS)

#: How many files' answers to remember. A CropBox is four integers, so this
#: is a bound against an extreme session, not a memory budget.
_CACHE_LIMIT = 512


_log = logging.getLogger(__name__)


class CropDetectError(RuntimeError):
    """Black bars could not be detected."""


class CropDetectCancelled(RuntimeError):  # noqa: N818 - expected control flow
    """Black-bar detection was cancelled."""


def _sample_window(duration: float) -> float:
    """How many seconds of video each sample reads: at most the part being
    compared, so that a short compared part is not measured with footage
    from beyond it."""
    if duration <= 0:
        return _SAMPLE_WINDOW_SECONDS
    return min(_SAMPLE_WINDOW_SECONDS, duration)


def _sample_offsets(duration: float, window: float = _SAMPLE_WINDOW_SECONDS) -> list[float]:
    """The start times of the sample windows, spread over the middle of the
    video."""
    if duration <= 0:
        return [0.0]
    # -ss is a window's start, so every sample must leave enough video for
    # the whole window; otherwise, on clips shorter than about 17 seconds,
    # most samples would start past the end.
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
    """The FFmpeg command for one sample window."""
    filters = f"cropdetect=limit={limit}:round=2:reset=0"
    if hwaccel:
        # hwdownload can only give the surface's own format, which cropdetect
        # cannot take as it is, so the same conversion as in the metric run
        # sits between them. Without it, FFmpeg cannot link the two filters
        # and the window reports nothing.
        filters = f"hwdownload,format={download_format},{filters}"
    return [
        ffmpeg_path(),
        "-nostdin", "-hide_banner",
        *hwaccel_args(hwaccel),
        "-ss", f"{start:.3f}",
        "-i", path,
        "-map", f"0:{VIDEO_STREAM}",
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
    # Waits for a decoder slot, still answering Cancel while it waits.
    while not _decoder_slots.acquire(timeout=0.1):
        if cancel_event is not None and cancel_event.is_set():
            raise CropDetectCancelled("Crop detection cancelled")
    proc = None
    try:
        # UTF-8, not the Windows code page: FFmpeg's error output starts with
        # the file's path and tags, and a curly quote (”) in either is a byte
        # that cp1252 cannot decode, which would make the output None.
        proc = proc_util.popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        if process_handle is not None:
            process_handle.attach(proc.pid)
        active_seconds = 0.0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                proc_util.terminate(proc)
            started = time.monotonic()
            try:
                _stdout, stderr = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired as e:
                if process_handle is None or not process_handle.is_pause_requested:
                    active_seconds += time.monotonic() - started
                if active_seconds >= 60:
                    proc_util.terminate(proc)
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
        # Detach this window's process only. Others may still be running
        # under the same handle, and a bare detach() would drop them from
        # Pause and Cancel.
        if process_handle is not None and proc is not None:
            process_handle.detach(proc.pid)
        _decoder_slots.release()


def _run_single_window(
    path: str, start: float, window: float, limit: float,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    hwaccel: str | None = None,
    download_format: str = "nv12",
) -> CropBox | None:
    """Runs one sample window and returns the box it found, or None."""
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
        # A hardware attempt that fails (no free decoder session, an
        # unsupported profile) is retried on the CPU, as the metric run
        # itself would fall back. The pixels, and so the box, are the same
        # either way.
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
    """How many seconds of `info` a run will compare."""
    if duration_limit <= 0:
        return info.duration
    if info.duration <= 0:
        return duration_limit
    return min(info.duration, duration_limit)


# ------------------------------------------------------------------ the cache
# The cache is keyed on what the answer depends on: which bytes are in the
# file, and which stretch of it is sampled. It is not keyed on how the video
# was decoded, because the GPU and the CPU give the same pixels.
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple, CropBox] = OrderedDict()
#: The files whose detection is running now, so that a second caller for the
#: same file waits for that answer instead of starting its own processes, as
#: two lanes starting on the same source at the same moment would.
_in_flight: dict[tuple, threading.Event] = {}


def _cache_key(path: Path, scored_duration: float, limit: float) -> tuple | None:
    """The cache key for a file, or None when the file cannot be identified,
    in which case nothing is cached: a path without a size and time behind
    it is not a stable identity."""
    try:
        stat = Path(path).resolve().stat()
    except OSError:
        return None
    return (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns,
            round(scored_duration, 3), round(limit, 6))


#: Answers are kept between sessions in a JSON file in the results cache's
#: folder (result_cache registers it), read once per file. Otherwise each
#: new session would detect each film's bars again: 4-6 s of decoding a 4K
#: pair before a run or Video Compare has them, with the preview shown
#: uncropped and its playback rebuilt once they come. Raise _STORE_VERSION
#: when the detection itself changes what it finds, so that older answers
#: are ignored.
_STORE_VERSION = 1
_store_of: Callable[[], Path] | None = None
#: The store file whose answers are in the cache already.
_store_read: Path | None = None


def set_store(path_of: Callable[[], Path] | None) -> None:
    """Keeps answers between sessions in the file that `path_of` returns when
    called (the results cache's folder can move), or, with None, not at all."""
    global _store_of, _store_read
    with _cache_lock:
        _store_of, _store_read = path_of, None


def _store_path() -> Path | None:
    """The store file now, or None if there is none."""
    if _store_of is None:
        return None
    try:
        return Path(_store_of())
    except OSError:
        return None


def _read_store() -> None:
    """Reads the store's answers into the cache, once per store file. Called
    with _cache_lock held."""
    global _store_read
    path = _store_path()
    if path is None or path == _store_read:
        return
    _store_read = path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict) or data.get("version") != _STORE_VERSION:
        return
    for item in data.get("boxes", ()):
        try:
            key, (w, h, x, y) = tuple(item[0]), item[1]
            _cache.setdefault(key, CropBox(w=int(w), h=int(h), x=int(x), y=int(y)))
        except (TypeError, ValueError, IndexError):
            continue
    while len(_cache) > _CACHE_LIMIT:
        _cache.popitem(last=False)


def _write_store() -> None:
    """Writes the cache's answers to the store, replacing it whole. Called
    with _cache_lock held. At worst, two processes writing at once lose one
    process's newest answer, which is then detected again."""
    path = _store_path()
    if path is None:
        return
    data = {"version": _STORE_VERSION,
            "boxes": [[list(key), [box.w, box.h, box.x, box.y]] for key, box in _cache.items()]}
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(data), encoding="utf-8")
        os.replace(partial, path)
    except OSError as error:
        _log.warning("Could not keep black-bar answers in %s: %s", path, error)


def _claim(key: tuple, cancel_event: threading.Event | None) -> CropBox | None:
    """Returns the cached box, or None once this caller has been given the
    job of detecting it. While another caller is on it, it waits, and can
    be cancelled meanwhile."""
    while True:
        with _cache_lock:
            _read_store()
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
    """Stores a detected box (if any), and wakes the callers waiting for it."""
    with _cache_lock:
        if box is not None:
            _cache[key] = box
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_LIMIT:
                _cache.popitem(last=False)
            _write_store()
        waiting = _in_flight.pop(key, None)
    if waiting is not None:
        waiting.set()


def clear_cache() -> None:
    """Forgets every box remembered in this process; the store is read again
    when next needed. It is for tests, and (with forget_store) for a file
    replaced by another with the same size and modification time."""
    global _store_read
    with _cache_lock:
        _cache.clear()
        _store_read = None


def forget_store() -> None:
    """Deletes the kept answers and forgets them here, so that clearing the
    results cache in Settings starts afresh."""
    global _store_read
    with _cache_lock:
        _cache.clear()
        path = _store_path()
        _store_read = None
        if path is not None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                _log.warning("Could not delete %s: %s", path, error)


#: Two pictures are the same shape when their sizes differ by one factor,
#: across and down, to within this.
_SAME_SHAPE = 0.005


def common_picture(
    source: VideoInfo, distorted: VideoInfo,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
) -> tuple[CropBox | None, CropBox | None]:
    """Returns the two videos' black-bar boxes, made to cover the same
    picture.

    Each video's bars are found on their own, and an encode's soft bar edge
    can put its box a row or two from the source's (1920x800 against
    1920x802). The two would then be compared at different sizes, so one
    would be scaled to the other, needlessly. Where the two videos show the
    same picture at one size or two (their sizes one factor apart, across
    and down), each box becomes the picture both show: the boxes' overlap,
    in each video's own pixels, with its edges on even samples (4:2:0).
    Boxes that already agree are kept, and so is any pair that is not the
    same picture, or where a box is missing."""
    if source_crop is None or distorted_crop is None:
        return source_crop, distorted_crop
    if min(source.width, source.height, distorted.width, distorted.height) <= 0:
        return source_crop, distorted_crop
    across, down = source.width / distorted.width, source.height / distorted.height
    if abs(across - down) > _SAME_SHAPE * max(across, down):
        return source_crop, distorted_crop

    def edges(box: CropBox, info: VideoInfo) -> tuple[float, float, float, float]:
        """A box's edges as fractions of the frame: left, top, right, bottom."""
        return (box.x / info.width, box.y / info.height,
                (box.x + box.w) / info.width, (box.y + box.h) / info.height)

    (sl, st, sr, sb), (dl, dt, dr, db) = edges(source_crop, source), edges(distorted_crop, distorted)
    left, top, right, bottom = max(sl, dl), max(st, dt), min(sr, dr), min(sb, db)
    if right <= left or bottom <= top:
        return source_crop, distorted_crop

    def box(info: VideoInfo) -> CropBox:
        """The shared picture in this video's pixels."""
        # Rounded inward, so the box never takes in a row of the other's bars.
        x = math.ceil(left * info.width - 1e-6)
        y = math.ceil(top * info.height - 1e-6)
        x, y = x + (x & 1), y + (y & 1)
        w = math.floor(right * info.width + 1e-6) - x
        h = math.floor(bottom * info.height + 1e-6) - y
        return CropBox(w=w - (w & 1), h=h - (h & 1), x=x, y=y)

    shared_source, shared_distorted = box(source), box(distorted)
    if shared_source.w <= 0 or shared_source.h <= 0 or shared_distorted.w <= 0 or shared_distorted.h <= 0:
        return source_crop, distorted_crop
    if (shared_source, shared_distorted) != (source_crop, distorted_crop):
        _log.info("Black bars: the picture both videos show is %dx%d at %d,%d of the source and %dx%d at %d,%d "
                  "of the test video (detected: %dx%d at %d,%d and %dx%d at %d,%d)",
                  shared_source.w, shared_source.h, shared_source.x, shared_source.y,
                  shared_distorted.w, shared_distorted.h, shared_distorted.x, shared_distorted.y,
                  source_crop.w, source_crop.h, source_crop.x, source_crop.y,
                  distorted_crop.w, distorted_crop.h, distorted_crop.x, distorted_crop.y)
    return shared_source, shared_distorted


def detect_pair(
    detect_source: Callable[[], CropBox | None],
    detect_distorted: Callable[[], CropBox | None],
) -> tuple[CropBox | None, CropBox | None]:
    """Detects the black bars of a comparison's two videos at the same time.

    Each video's windows run in turn, so this is two decoders (exactly the
    app-wide limit), in about half the time of one video after the other.
    Both are waited for before anything is raised, so no detection outlives
    the call. A cancel is reported before an error, and the source's error
    before the test video's.
    """
    results: list[CropBox | None] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def run(index: int, detect: Callable[[], CropBox | None]) -> None:
        """One video's detection thread."""
        try:
            results[index] = detect()
        except BaseException as error:  # re-raised on the calling thread
            errors[index] = error

    threads = [
        threading.Thread(target=run, args=(0, detect_source), name="crop-reference", daemon=True),
        threading.Thread(target=run, args=(1, detect_distorted), name="crop-test", daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for error in errors:
        if isinstance(error, CropDetectCancelled):
            raise error
    for error in errors:
        if error is not None:
            raise error
    return results[0], results[1]


def detect_crop(
    info: VideoInfo, limit: float = 24 / 255,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    duration_limit: float = 0.0,
    hwaccel: str | None = None,
) -> CropBox:
    """Returns the box that cuts the video's black bars, or raises
    CropDetectError when it cannot be found.

    `duration_limit` limits the sampling, when asked for. The metric runs
    leave out their duration limit, so that a short or dark opening cannot
    decide the crop for the whole comparison.

    `hwaccel` is the decoder the run itself will use for this video, or None
    for software. It changes how fast the answer arrives, never what it is.
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
        _log.info("Black bars in %s: picture %dx%d at %d,%d of %dx%d", info.path.name, box.w, box.h, box.x, box.y,
                  info.width, info.height)
        return box
    except CropDetectError as error:
        _log.warning("Black-bar detection failed for %s: %s", info.path.name, error)
        raise
    finally:
        if key is not None:
            _settle(key, box)


def _detect_uncached(
    info: VideoInfo, limit: float, scored_duration: float,
    cancel_event: threading.Event | None,
    process_handle: ProcessHandle | None,
    hwaccel: str | None,
) -> CropBox:
    """detect_crop's work, without the cache: samples the windows and returns
    the winning box."""
    path = str(info.path)
    window = _sample_window(scored_duration)
    starts = _sample_offsets(scored_duration, window)
    download_format = hw_native_format(info.pix_fmt)

    if cancel_event is not None and cancel_event.is_set():
        raise CropDetectCancelled("Crop detection cancelled")

    # One window at a time. Running all five at once is faster, but each is
    # its own decoder: five hardware decoders per video held 6.2 GB of GPU
    # memory for a 4K source, and a comparison of two videos reached ten
    # decoders, more than most GPUs have. One window at a time keeps one
    # decoder per video.
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

    # A cancel wins over any boxes that came back before it: a partial vote
    # is not an answer the caller asked for.
    if cancel_event is not None and cancel_event.is_set():
        raise CropDetectCancelled("Crop detection cancelled")

    if not boxes and not failures:
        # FFmpeg decoded nothing where the video says it has pictures. That
        # is not a question of black bars, and turning cropping off would
        # only score the pictures it does have (see frame_coverage).
        raise CropDetectError(
            f"Could not auto-detect black bars in {info.path}: FFmpeg found no pictures in the parts of it "
            "that were sampled. The file may be damaged or cut short."
        )
    if not boxes:
        raise CropDetectError(
            f"Could not auto-detect black bars in {info.path}: {failures[-1]}. "
            "Choose 'None (use full frame)' for this video to continue without cropping."
        )

    counts: dict[tuple[int, int, int, int], int] = {}
    for b in boxes:
        key = (b.w, b.h, b.x, b.y)
        counts[key] = counts.get(key, 0) + 1
    # The most common box wins, and between equally common ones the largest.
    # A box from a dark stretch is too tight (dark picture reads as bar), so
    # a tie must not go to whichever window came first, which could cut off
    # some of the picture.
    best_key = max(counts, key=lambda k: (counts[k], k[0] * k[1]))
    w, h, x, y = best_key
    return CropBox(w=w, h=h, x=x, y=y)
