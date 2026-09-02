"""Formatting durations as H:M:S instead of raw seconds, used throughout the
UI (graph axis/hover, video duration displays) per the user's preference."""
from __future__ import annotations


def format_hms(seconds: float, *, decimals: int = 0) -> str:
    """Formats a duration in seconds as H:MM:SS, or H:MM:SS.sss when
    decimals > 0. Always shows hours (even 0) for a consistent width."""
    seconds = max(0.0, seconds)
    total_whole_seconds = int(seconds)
    hours, remainder = divmod(total_whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if decimals > 0:
        frac_secs = secs + (seconds - total_whole_seconds)
        secs_str = f"{frac_secs:0{3 + decimals}.{decimals}f}"
    else:
        secs_str = f"{secs:02d}"

    return f"{hours}:{minutes:02d}:{secs_str}"
