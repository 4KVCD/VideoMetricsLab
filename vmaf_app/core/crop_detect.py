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

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import CropBox, VideoInfo

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

_SAMPLE_WINDOW_SECONDS = 3.0
_SAMPLE_COUNT = 5
_SAMPLE_SPAN = (0.1, 0.9)  # fraction of duration to sample within


class CropDetectError(RuntimeError):
    pass


def _sample_offsets(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    lo = duration * _SAMPLE_SPAN[0]
    hi = duration * _SAMPLE_SPAN[1]
    hi = max(hi, lo + _SAMPLE_WINDOW_SECONDS)
    if _SAMPLE_COUNT == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (_SAMPLE_COUNT - 1)
    return [lo + i * step for i in range(_SAMPLE_COUNT)]


def _run_single_window(path: str, start: float, window: float, limit: float) -> CropBox | None:
    cmd = [
        ffmpeg_path(),
        "-nostdin", "-hide_banner",
        "-ss", f"{start:.3f}",
        "-i", path,
        "-t", f"{window:.3f}",
        "-vf", f"cropdetect=limit={limit}:round=2:reset=0",
        "-f", "null", "-",
    ]
    try:
        proc = proc_util.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:
        return None

    matches = _CROP_RE.findall(proc.stderr)
    if not matches:
        return None
    w, h, x, y = (int(v) for v in matches[-1])
    return CropBox(w=w, h=h, x=x, y=y)


def detect_crop(info: VideoInfo, limit: float = 24 / 255) -> CropBox:
    """Detects the black-bar crop box for a video. Returns a no-op crop
    (full frame) if detection fails or finds nothing to crop."""
    path = str(info.path)
    boxes: list[CropBox] = []
    for start in _sample_offsets(info.duration):
        box = _run_single_window(path, start, _SAMPLE_WINDOW_SECONDS, limit)
        if box is not None:
            boxes.append(box)

    if not boxes:
        return CropBox(w=info.width, h=info.height, x=0, y=0)

    counts: dict[tuple[int, int, int, int], int] = {}
    for b in boxes:
        key = (b.w, b.h, b.x, b.y)
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts, key=lambda k: counts[k])
    w, h, x, y = best_key
    return CropBox(w=w, h=h, x=x, y=y)
