"""Extract one display frame using the same geometry as a VMAF run."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo, VmafRunResult
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.vmaf_runner import analysis_pix_fmt

FrameSide = Literal["source", "distorted"]


class FrameExtractError(RuntimeError):
    """A requested preview frame could not be decoded."""


class FrameExtractCancelledError(FrameExtractError):
    """Extraction was deliberately cancelled by the UI."""


def _content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
    return (crop.w, crop.h) if crop is not None else (info.width, info.height)


def comparison_dimensions(result: VmafRunResult) -> tuple[int, int]:
    """Dimensions of the pictures that reached the metric filter."""
    source_size = _content_size(result.source_info, result.source_crop)
    if result.resample_target is not None:
        return source_size
    distorted_size = _content_size(result.distorted_info, result.distorted_crop)
    if source_size == distorted_size:
        return source_size
    if result.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE:
        return source_size
    return distorted_size


def frame_input_path(result: VmafRunResult, side: FrameSide) -> Path:
    """Physical file to decode, including synthetic resolution tests."""
    if side == "source" or result.resample_target is not None:
        return result.source_info.path
    return result.distorted_info.path


def frame_filter(result: VmafRunResult, side: FrameSide) -> str:
    """The crop/scale chain for one side of a completed comparison."""
    if side not in {"source", "distorted"}:
        raise ValueError(f"unknown frame side: {side}")

    info = result.source_info if side == "source" else result.distorted_info
    crop = result.source_crop if side == "source" else result.distorted_crop
    output_w, output_h = comparison_dimensions(result)
    ops: list[str] = []

    crop_filter = (
        crop.as_filter()
        if crop is not None and not crop.is_noop(info.width, info.height)
        else None
    )
    pixel_formats = [result.source_info.pix_fmt]
    if result.resample_target is None:
        pixel_formats.append(result.distorted_info.pix_fmt)
    analysis_format = analysis_pix_fmt(*pixel_formats)
    # Match the scorer's ordering: normal distorted frames crop before the
    # common format conversion; source and resolution-test branches convert
    # first. This matters slightly when the inputs have different bit depths.
    if side == "distorted" and result.resample_target is None:
        if crop_filter:
            ops.append(crop_filter)
        ops.append(f"format={analysis_format}")
    else:
        ops.append(f"format={analysis_format}")
        if crop_filter:
            ops.append(crop_filter)

    content_w, content_h = _content_size(info, crop)
    if result.resample_target is not None and side == "distorted":
        target_w = result.resample_target.width
        target_h = max(2, round(target_w * content_h / content_w / 2) * 2)
        ops.append(
            f"scale={target_w}:{target_h}:flags={result.scale_algorithm}"
        )
        ops.append(
            f"scale={output_w}:{output_h}:flags={result.scale_algorithm}"
        )
    elif (content_w, content_h) != (output_w, output_h):
        ops.append(
            f"scale={output_w}:{output_h}:flags={result.scale_algorithm}"
        )

    # PNG has square pixels and no useful video SAR. Resetting it after the
    # geometry operations ensures source/distorted previews occupy the exact
    # same canvas when the input used anamorphic storage.
    ops.extend(["setsar=1", "format=rgb24"])
    return ",".join(ops)


def build_frame_command(
    result: VmafRunResult, side: FrameSide, frame: int
) -> list[str]:
    if frame < 0:
        raise ValueError("frame number must be non-negative")
    if result.fps <= 0:
        raise FrameExtractError("The run does not contain a usable frame rate.")

    # Accurate input seeking returns the first frame at or after the target.
    # Aim one eighth of a frame before the desired PTS: an exact/rounded-up
    # boundary can otherwise advance to the next frame, while this remains
    # far beyond the previous frame's PTS even at fractional frame rates.
    timestamp = max(0.0, (frame - 0.125) / result.fps)
    return [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{timestamp:.9f}",
        "-i", str(frame_input_path(result, side).resolve()),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", frame_filter(result, side),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-c:v", "png",
        "pipe:1",
    ]


def extract_frame_png(
    result: VmafRunResult,
    side: FrameSide,
    frame: int,
    process_handle: ProcessHandle | None = None,
) -> bytes:
    path = frame_input_path(result, side)
    if not path.is_file():
        raise FrameExtractError(f"Video file is missing: {path}")

    handle = process_handle or ProcessHandle()
    process = proc_util.popen(
        build_frame_command(result, side, frame),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    handle.attach(process.pid)
    try:
        stdout, stderr = process.communicate()
    finally:
        handle.detach()

    if process.returncode != 0:
        if handle.was_terminated:
            raise FrameExtractCancelledError("Frame extraction was cancelled.")
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise FrameExtractError(
            detail[-2000:] or f"ffmpeg exited with code {process.returncode}."
        )
    if not stdout:
        raise FrameExtractError(
            f"No frame was returned at frame {frame}; it may be beyond the end of the video."
        )
    return stdout
