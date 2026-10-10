"""The geometry of a comparison: the picture each video gives once its crop
is applied, the size the two are compared at, their shapes on screen, and
whether two videos can be compared at all.

Every part of the app that needs these (the runner, the CPU tools, Vship,
frame extraction and the window) asks this module, so they cannot disagree.
"""
from __future__ import annotations

from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo, VmafOptions


def content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
    """The size of the picture that reaches the comparison: the crop's, or
    the whole video's."""
    return (crop.w, crop.h) if crop is not None else (info.width, info.height)


def compared_dimensions(
    source_info: VideoInfo, distorted_info: VideoInfo, scale_direction: ScaleDirection,
    source_crop: CropBox | None = None, distorted_crop: CropBox | None = None,
) -> tuple[int, int]:
    """The size frames are compared at.

    One video is scaled to the other's size before the metrics see them, so
    the compared size need not be either input's own resolution: a 1080p
    encode measured with "upscale distorted to source" against a 4K master
    is compared at 4K. Cropping changes the size too."""
    distorted_size = content_size(distorted_info, distorted_crop)
    source_size = content_size(source_info, source_crop)
    if source_size == distorted_size:
        return distorted_size
    if scale_direction == ScaleDirection.DISTORTED_TO_SOURCE:
        return source_size
    return distorted_size


def analysis_dimensions(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None = None, distorted_crop: CropBox | None = None,
) -> tuple[int, int]:
    """compared_dimensions for a row's options. The runner's filter graph
    uses it too, so the two cannot disagree about what a run does."""
    return compared_dimensions(source_info, distorted_info, options.scale_direction, source_crop, distorted_crop)


def resample_analysis_dimensions(source_info: VideoInfo, source_crop: CropBox | None = None) -> tuple[int, int]:
    """The size a resolution round-trip test compares at: the source's own
    (cropped) size, because the downscale is undone before the comparison."""
    return content_size(source_info, source_crop)


def sar_fraction(sar: str) -> tuple[int, int]:
    """A sample aspect ratio ("4:3") as a fraction; an unknown or unset one
    means square pixels."""
    if sar in {"", "N/A", "0:1"}:
        return 1, 1
    try:
        num, den = (int(part) for part in sar.split(":", 1))
    except ValueError:
        return 1, 1
    if num <= 0 or den <= 0:
        return 1, 1
    return num, den


def display_aspect_ratio(info: VideoInfo, crop: CropBox | None = None) -> float:
    """The shape of the picture as displayed, after cropping.

    The stored width and height alone are not the shape: non-square pixels
    stretch them, and a crop changes them. This shape is what has to match
    between two videos, not the raw SAR: 1920x1080 with SAR 1:1 and
    1440x1080 with SAR 4:3 are the same 16:9 picture stored two ways.
    """
    width, height = content_size(info, crop)
    if height <= 0:
        return 0.0
    num, den = sar_fraction(info.sar)
    return (width * num) / (height * den)


_CODEC_LABELS = {"h264": "H.264", "mpeg2video": "MPEG-2", "mpeg4": "MPEG-4", "mjpeg": "MJPEG"}


def pair_problem(source_info: VideoInfo, distorted_info: VideoInfo, duration_limit: float) -> str | None:
    """Why two videos cannot be compared (their timelines do not match), or
    None. Each runner raises this as its own error."""
    problem = _timeline_problem(source_info, distorted_info, duration_limit)
    raw = next((info for info in (source_info, distorted_info) if info.is_raw_stream), None)
    if problem is None or raw is None:
        return problem
    # A raw stream's frame rate and length are FFmpeg's guesses (an average
    # of FFmpeg's default 25 fps, a nominal rate from the stream or the
    # parser's time base), so the message says how to give it a real frame
    # rate rather than suggesting a frame rate conversion. It names
    # mkvmerge, not FFmpeg: FFmpeg cannot stream-copy a raw stream with
    # B-frames into Matroska ("unknown timestamp"), and into MP4 it makes up
    # timestamps that put the frames out of order (a CRF 22 encode scored
    # VMAF 57 instead of 94). mkvmerge's file scores as the encode does.
    return (f"{raw.path.name} is a raw {_CODEC_LABELS.get(raw.codec_name, raw.codec_name.upper())} "
            "stream with no timestamps, so its frame rate is unknown. Put it in a container with its "
            "frame rate first, e.g. mkvmerge -o video.mkv --default-duration 0:120fps <file> (MKVToolNix), "
            "with the video's own frame rate.")


def untimed_pair_problem(source_info: VideoInfo, distorted_info: VideoInfo) -> str | None:
    """Video Compare's check: refuses a raw stream whose frame rate (FFmpeg's
    guess) disagrees with the other video's, as pair_problem does, or
    returns None. Video Compare shows other mismatched pairs anyway, but it
    places a raw stream's frames by that guessed rate, so it would show the
    wrong frame and then stop with "Decoder produced no frames"."""
    if not (source_info.is_raw_stream or distorted_info.is_raw_stream):
        return None
    return pair_problem(source_info, distorted_info, 0.0)


def _timeline_problem(source_info: VideoInfo, distorted_info: VideoInfo, duration_limit: float) -> str | None:
    """Why the two videos' frame rates or durations do not allow a
    comparison, or None."""
    if source_info.is_variable_frame_rate or distorted_info.is_variable_frame_rate:
        return ("Variable-frame-rate video is not supported safely yet. Convert both videos "
                "to the same constant frame rate before comparing them.")
    # The tolerance is half the gap between a whole frame rate and its NTSC
    # counterpart (24 and 23.976 are a thousandth apart), so those two are
    # told apart, while the rates read from two containers of one video
    # (24000/1001 against 2997/125, 0.00002 fps apart) still match. When
    # frames were paired by their times, the same pictures flagged 24 and
    # 23.976 fps drifted a frame apart every 42 seconds, and VMAF fell from
    # 98.8 to about 30 within 20 seconds.
    fps_tolerance = max(0.005, max(source_info.fps, distorted_info.fps) * 0.0005)
    if abs(source_info.fps - distorted_info.fps) > fps_tolerance:
        return f"Frame rates do not match ({source_info.fps:.3f} vs {distorted_info.fps:.3f} fps)."
    if source_info.duration > 0 and distorted_info.duration > 0:
        if duration_limit <= 0:
            frame_duration = 1.0 / max(source_info.fps, distorted_info.fps, 1.0)
            if abs(source_info.duration - distorted_info.duration) > max(0.1, 2 * frame_duration):
                return (f"Durations do not match ({source_info.duration:.3f} vs "
                        f"{distorted_info.duration:.3f} seconds). Set a duration limit within "
                        "both files if comparing only their common opening segment.")
        elif min(source_info.duration, distorted_info.duration) + 0.1 < duration_limit:
            return "The duration limit extends beyond the end of one of the videos."
    return None
