"""Checks that a comparison covered the frames its two videos say they have.

Every scorer stops when the shorter video's pictures end: libvmaf's frame
sync (shortest=1), Vship's readers and the CPU tools' readers alike. That is
right for two files whose lengths differ by a frame or two. But it would also
score a file whose pictures end long before its stated length: a truncated
10-second encode with five frames of picture would get VMAF 98, as if for
the whole video. Before a run, vmaf_runner.validate_video_pair checks that
the two stated lengths agree, so frames missing at the end mean a file was
cut short.
"""
from __future__ import annotations

import math

from vmaf_app.core.time_format import format_hms


def short_comparison(expected: int, compared: int, fps: float, step: int = 1) -> str | None:
    """Returns why `compared` frames are too few to stand for the video, or
    None when they cover it.

    `expected` is the number of frames the videos' lengths promise. The
    comparison covers the video if it is short by at most half a second (and
    at least three frames are allowed) plus one `step` of frame subsampling."""
    slack = max(3, math.ceil(0.5 * fps)) + max(1, step)
    if expected <= 0 or compared >= expected - slack:
        return None
    return (f"Only {compared} of the {expected} frames expected could be compared: one of the videos ends after "
            f"{format_hms(compared / max(fps, 1.0), decimals=1)}, though its file says it is longer. It may have "
            "been cut short: an encode that stopped early, or a copy that did not finish.")
