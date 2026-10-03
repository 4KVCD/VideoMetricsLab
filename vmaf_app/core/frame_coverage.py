"""Whether a comparison covered the frames its two videos say they have.

Every scorer stops where the shorter input's pictures end -- libvmaf's frame
sync (shortest=1), Vship's and the CPU tools' readers alike -- which is right
for two files a frame or two apart. It also scored a file whose pictures end
long before its length says: a truncated 10-second encode, five frames of
picture, was given VMAF 98 as if for the whole video. The two lengths are
checked to agree before a run (vmaf_runner.validate_video_pair), so frames
missing at the end mean a file was cut short.
"""
from __future__ import annotations

import math

from vmaf_app.core.time_format import format_hms


def short_comparison(expected: int, compared: int, fps: float, step: int = 1) -> str | None:
    """Why `compared` frames, out of the `expected` ones the videos' lengths
    promise, are too few to stand for the video -- or None when they cover
    it, give or take half a second and one `step` of frame subsampling."""
    slack = max(3, math.ceil(0.5 * fps)) + max(1, step)
    if expected <= 0 or compared >= expected - slack:
        return None
    return (f"Only {compared} of the {expected} frames expected could be compared: one of the videos ends after "
            f"{format_hms(compared / max(fps, 1.0), decimals=1)}, though its file says it is longer. It may have "
            "been cut short: an encode that stopped early, or a copy that did not finish.")
