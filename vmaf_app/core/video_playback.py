"""ffmpeg commands for frame-locked source/distorted video playback."""
from __future__ import annotations

from pathlib import Path

from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorSettings,
    comparison_dimensions,
    frame_filter,
    frame_input_path,
)
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.vmaf_runner import _hw_native_format


def playback_dimensions(
    comparison: FrameComparison,
    maximum: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Largest even preview size that fits the physical display, if supplied.

    There is deliberately no built-in 1920-pixel budget.  Native GPU playback
    does not materialize this image in Python at all, and the FFmpeg fallback
    is limited by the actual monitor rather than an arbitrary resolution.
    """
    width, height = comparison_dimensions(comparison)
    if width <= 0 or height <= 0:
        raise ValueError("comparison has no usable display dimensions")
    if maximum is None:
        return max(2, width // 2 * 2), max(2, height // 2 * 2)
    max_width, max_height = maximum
    scale = min(1.0, max_width / width, max_height / height)
    out_w = max(2, int(width * scale) // 2 * 2)
    out_h = max(2, int(height * scale) // 2 * 2)
    return out_w, out_h


def _hwaccel_args(hwaccel: str | None) -> list[str]:
    if not hwaccel:
        return []
    return ["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel]


def _input_args(
    path: Path,
    timestamp: float,
    hwaccel: str | None,
    realtime: bool,
) -> list[str]:
    args: list[str] = []
    if realtime:
        args += ["-readrate", "1"]
    args += ["-ss", f"{timestamp:.9f}"]
    args += _hwaccel_args(hwaccel)
    args += ["-i", str(path.resolve())]
    return args


def _side_chain(
    comparison: FrameComparison,
    side: str,
    input_index: int,
    hwaccel: str | None,
    color_settings: PreviewColorSettings,
    output_size: tuple[int, int],
) -> str:
    prefix = ""
    if hwaccel:
        info = (
            comparison.source_info if side == "source"
            else comparison.distorted_info
        )
        prefix = f"hwdownload,format={_hw_native_format(info.pix_fmt)},"
    filters = frame_filter(
        comparison, side, color_settings, output_size=output_size
    )
    # Both streams are converted to the comparison's declared frame rate and
    # rebased to frame zero. hstack then emits one indivisible pair per tick:
    # the UI can flip halves without consulting two independent media clocks.
    fps = f"{comparison.fps:.12g}"
    return (
        f"[{input_index}:v]{prefix}{filters},fps={fps},"
        f"setpts=N/({fps}*TB)[{side}]"
    )


def build_video_pair_command(
    comparison: FrameComparison,
    start_frame: int,
    color_settings: PreviewColorSettings,
    hwaccel: HwAccelPlan,
    output_size: tuple[int, int],
    *,
    realtime: bool,
) -> list[str]:
    """Decode a stream of horizontally packed, frame-exact comparison pairs."""
    if comparison.fps <= 0:
        raise ValueError("comparison has no usable frame rate")
    if start_frame < 0:
        raise ValueError("start frame must be non-negative")
    timestamp = max(0.0, (start_frame - 0.125) / comparison.fps)
    source = frame_input_path(comparison, "source")
    distorted = frame_input_path(comparison, "distorted")
    source_chain = _side_chain(
        comparison, "source", 0, hwaccel.source,
        color_settings, output_size,
    )
    distorted_chain = _side_chain(
        comparison, "distorted", 1, hwaccel.distorted,
        color_settings, output_size,
    )
    graph = ";".join([
        source_chain,
        distorted_chain,
        "[source][distorted]hstack=inputs=2:shortest=1[out]",
    ])
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    cmd += _input_args(source, timestamp, hwaccel.source, realtime)
    cmd += _input_args(distorted, timestamp, hwaccel.distorted, realtime)
    cmd += [
        "-filter_complex", graph,
        "-map", "[out]",
        "-an", "-sn", "-dn",
        "-pix_fmt", "rgb24",
        "-fps_mode", "passthrough",
    ]
    if not realtime:
        cmd += ["-frames:v", "1"]
    cmd += ["-f", "rawvideo", "pipe:1"]
    return cmd


def ffplay_path() -> Path | None:
    candidate = Path(ffmpeg_path()).with_name(
        "ffplay.exe" if Path(ffmpeg_path()).suffix.lower() == ".exe" else "ffplay"
    )
    return candidate if candidate.is_file() else None


def build_audio_command(comparison: FrameComparison, start_frame: int) -> list[str] | None:
    """Play only the distorted audio through ffplay's native audio output."""
    player = ffplay_path()
    if player is None or comparison.fps <= 0:
        return None
    timestamp = max(0.0, start_frame / comparison.fps)
    return [
        str(player), "-nodisp", "-autoexit", "-loglevel", "error",
        "-ss", f"{timestamp:.9f}",
        "-i", str(frame_input_path(comparison, "distorted").resolve()),
        "-vn", "-sn",
    ]
