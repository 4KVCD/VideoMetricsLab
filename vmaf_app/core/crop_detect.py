"""Black-bar (letterbox/pillarbox) detection via ffmpeg's cropdetect filter.

cropdetect with reset=0 reports the *tightest* crop box that still contains
every non-black pixel it has seen so far in the analyzed window -- a single
bright pixel of noise near an edge can prevent any crop from being detected.
To be robust we sample several short windows spread across the middle of the
video (skipping fades at the very start/end) and take the most common result
across windows, rather than trusting a single long pass.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import CropBox, VideoInfo

if TYPE_CHECKING:
    from vmaf_app.core.process_control import ProcessHandle

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

_SAMPLE_WINDOW_SECONDS = 3.0
_SAMPLE_COUNT = 5
_SAMPLE_SPAN = (0.1, 0.9)  # fraction of duration to sample within


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


def _run_single_window(
    path: str, start: float, window: float, limit: float,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> CropBox | None:
    cmd = [
        ffmpeg_path(),
        "-nostdin", "-hide_banner",
        "-ss", f"{start:.3f}",
        "-i", path,
        "-t", f"{window:.3f}",
        "-vf", f"cropdetect=limit={limit}:round=2:reset=0",
        "-f", "null", "-",
    ]
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
                    raise CropDetectError(
                        f"Crop detection timed out for {path}"
                    ) from e
        if cancel_event is not None and cancel_event.is_set():
            raise CropDetectCancelled("Crop detection cancelled")
    except Exception as e:
        if isinstance(e, (CropDetectError, CropDetectCancelled)):
            raise
        raise CropDetectError(
            f"Could not run crop detection for {path}: {e}"
        ) from e
    finally:
        if process_handle is not None and proc is not None:
            process_handle.detach()

    if proc.returncode != 0:
        detail = stderr.strip().splitlines()
        tail = detail[-1] if detail else f"ffmpeg exited with code {proc.returncode}"
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


def detect_crop(
    info: VideoInfo, limit: float = 24 / 255,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    duration_limit: float = 0.0,
) -> CropBox:
    """Detects the black-bar crop box, or raises when it cannot analyze it.

    `duration_limit` is the run's own limit. Every sample is taken from
    inside the stretch that will actually be scored: a film that is
    full-frame for its opening seconds and letterboxed afterwards would
    otherwise be measured on footage the comparison never looks at, and the
    detected bars cropped away from content that is really there.
    """
    path = str(info.path)
    boxes: list[CropBox] = []
    failures: list[str] = []
    scored_duration = analysed_duration(info, duration_limit)
    window = _sample_window(scored_duration)
    for start in _sample_offsets(scored_duration, window):
        if cancel_event is not None and cancel_event.is_set():
            raise CropDetectCancelled("Crop detection cancelled")
        try:
            box = _run_single_window(
                path, start, window, limit,
                cancel_event=cancel_event, process_handle=process_handle,
            )
        except CropDetectError as e:
            failures.append(str(e))
            continue
        if box is not None:
            boxes.append(box)

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
