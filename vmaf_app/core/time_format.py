"""Formatting durations as H:M:S instead of raw seconds, used throughout the
UI (graph axis/hover, video duration displays) per the user's preference."""
from __future__ import annotations


def format_hms(seconds: float, *, decimals: int = 0) -> str:
    """Formats a duration in seconds as H:MM:SS, or H:MM:SS.sss when
    decimals > 0. Always shows hours (even 0) for a consistent width.

    With decimals the whole value is rounded first, in whole units of the
    last decimal, and only then split into hours, minutes and seconds: the
    seconds rounded on their own came out as "0:00:60.0" for 59.96 s and
    "0:59:60.00" for 3599.999 s. Without decimals the seconds are cut, as
    they always were."""
    seconds = max(0.0, seconds)
    if decimals > 0:
        scale = 10 ** decimals
        whole, fraction = divmod(round(seconds * scale), scale)
    else:
        whole, fraction = int(seconds), 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if decimals > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:0{decimals}d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"
