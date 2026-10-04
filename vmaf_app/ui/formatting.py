"""How probe results and scores are rendered as text/colour in the table.

Pure presentation: every function here takes data and returns something to
display, with no widget state involved, so the wording and thresholds are in
one place instead of scattered through the row-populating code.
"""
from __future__ import annotations

from PySide6.QtGui import QColor

from vmaf_app.core.models import VideoInfo
from vmaf_app.i18n import tr

# Shown wherever a value genuinely doesn't exist, as opposed to being 0.
NOT_COMPUTED = "N/A"

# VMAF is conventionally read in bands rather than as a raw number: ~95+ is
# visually transparent, 90-95 is good, below that is where artefacts show.
_GOOD_VMAF = 95
_FAIR_VMAF = 90

_COLOUR_GOOD = QColor(198, 239, 206)  # green
_COLOUR_FAIR = QColor(255, 235, 156)  # amber
_COLOUR_POOR = QColor(255, 199, 206)  # pink


def media_info_string(info: VideoInfo) -> str:
    """The one-line summary shown in the Media info column."""
    return ", ".join([f"{info.width}x{info.height}", f"{info.fps:.2f}fps", info.codec_name])


def bitrate_string(info: VideoInfo) -> str:
    """Bitrate at a human scale: Mb/s once it's over a megabit, else kb/s.
    "≈" when it is the whole file's, the soundtrack included
    (bitrate_note says so)."""
    if not info.bit_rate:
        return NOT_COMPUTED
    approximate = "≈" if info.bit_rate_whole_file else ""
    if info.bit_rate >= 1_000_000:
        return f"{approximate}{info.bit_rate / 1_000_000:.1f} Mb/s"
    return f"{approximate}{info.bit_rate // 1000} kb/s"


def bitrate_note(info: VideoInfo) -> str:
    """The Bitrate cell's tooltip: what the figure is when it is not the
    video's own."""
    if info.bit_rate and info.bit_rate_whole_file:
        return tr("The whole file's bitrate, soundtrack included: the file does not record the video's own.")
    return ""


def vmaf_band_colour(mean: float) -> QColor:
    """A soft background tint by VMAF band, so a column of encodes can be
    scanned at a glance the way FFMetrics' highlighted scores can be."""
    if mean >= _GOOD_VMAF:
        return _COLOUR_GOOD
    if mean >= _FAIR_VMAF:
        return _COLOUR_FAIR
    return _COLOUR_POOR
