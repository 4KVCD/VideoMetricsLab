"""How probe results, CVVDP presets and displays are rendered as text in the
window.

Pure presentation: every function here takes data and returns something to
display, with no widget state involved, so the wording is in one place
instead of scattered through the row-populating code.
"""
from __future__ import annotations

from vmaf_app.core.models import VideoInfo
from vmaf_app.i18n import tr

# Shown wherever a value genuinely doesn't exist, as opposed to being 0.
NOT_COMPUTED = "N/A"


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
        return approximate + tr("{rate:.1f} Mb/s", rate=info.bit_rate / 1_000_000)
    return approximate + tr("{rate} kb/s", rate=info.bit_rate // 1000)


def bitrate_note(info: VideoInfo) -> str:
    """The Bitrate cell's tooltip: what the figure is when it is not the
    video's own."""
    if info.bit_rate and info.bit_rate_whole_file:
        return tr("The whole file's bitrate, soundtrack included: the file does not record the video's own.")
    return ""


def preset_label(preset) -> str:
    """A CVVDP preset's name as shown: a built-in one's translated, the
    user's own as they wrote it. The name itself stays the identifier."""
    return tr(preset.name) if preset.builtin else preset.name


def display_description(display) -> str:
    """A CVVDP display in one line, for tooltips, in the window's language:
    '30" 3840x2160 SDR, 200 nits, 250 lux, 0.75 m (2.0 x screen height)'.
    CvvdpDisplay.describe() gives the same line in English, for the log."""
    return tr('{diagonal:g}" {width}x{height} {range}, {nits:g} nits, {lux:g} lux, {distance:.2f} m '
              '({heights:.1f} x screen height)', diagonal=display.diagonal_inches, width=display.width,
              height=display.height, range="HDR" if display.hdr else "SDR", nits=display.peak_luminance,
              lux=display.ambient_lux, distance=display.viewing_distance_m, heights=display.distance_in_heights)
