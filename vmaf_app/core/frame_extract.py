"""Extract one display frame using the same geometry a VMAF run would."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import (
    CropBox,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafRunResult,
)
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.vmaf_runner import analysis_pix_fmt, display_aspect_ratio

FrameSide = Literal["source", "distorted"]


class PreviewColorMode(str, Enum):
    """How HDR video is converted for the SDR QWidget preview surface."""

    DISPLAY_AWARE = "display_aware"
    HDR_TO_SDR = "hdr_to_sdr"
    UNMANAGED = "unmanaged"


@dataclass(frozen=True, slots=True)
class PreviewColorSettings:
    mode: PreviewColorMode = PreviewColorMode.DISPLAY_AWARE
    display_hdr_enabled: bool | None = None
    display_sdr_white_nits: float | None = None

    @property
    def target_nits(self) -> float:
        """The SDR diffuse-white target used by zscale.

        Windows' SDR-content brightness only applies while HDR is enabled.
        Reject implausible driver values instead of feeding them to ffmpeg.
        """
        value = self.display_sdr_white_nits
        if (
            self.mode == PreviewColorMode.DISPLAY_AWARE
            and self.display_hdr_enabled is True
            and value is not None
            and 40 <= value <= 1000
        ):
            return float(value)
        return 100.0

    @property
    def cache_token(self) -> tuple[str, bool | None, float]:
        return (self.mode.value, self.display_hdr_enabled, round(self.target_nits, 3))


class FrameExtractError(RuntimeError):
    """A requested preview frame could not be decoded."""


class FrameExtractCancelledError(FrameExtractError):
    """Extraction was deliberately cancelled by the UI."""


@dataclass(frozen=True)
class FrameComparison:
    """Everything needed to render one pair of comparable frames.

    This is the preprocessing recipe -- which files, cropped how, scaled
    which way -- and deliberately NOT the scores. Looking at two frames side
    by side is useful on its own, and requiring a finished VMAF run before
    the pictures could be shown made the tab unavailable exactly when it is
    most wanted: while deciding whether a comparison is worth running at all.

    A run produces one of these (from_result); so does a source/distorted
    pair that has only been probed.
    """

    source_info: VideoInfo
    distorted_info: VideoInfo
    source_crop: CropBox | None = None
    distorted_crop: CropBox | None = None
    scale_direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED
    scale_algorithm: str = "bicubic"
    resample_target: ResampleTarget | None = None
    fps: float = 0.0
    frame_count: int = 0
    # True when auto-crop is selected but has not run yet, so these frames
    # are shown uncropped while a real run would remove black bars. The
    # preview is honest about being a preview rather than pretending to be
    # the scored geometry.
    auto_crop_pending: bool = False

    @classmethod
    def from_result(cls, result: VmafRunResult) -> FrameComparison:
        """The exact geometry a finished run actually used."""
        return cls(
            source_info=result.source_info,
            distorted_info=result.distorted_info,
            source_crop=result.source_crop,
            distorted_crop=result.distorted_crop,
            scale_direction=result.scale_direction,
            scale_algorithm=result.scale_algorithm,
            resample_target=result.resample_target,
            fps=result.fps,
            frame_count=result.compared_frame_count,
        )


def _content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
    return (crop.w, crop.h) if crop is not None else (info.width, info.height)


def comparison_dimensions(comparison: FrameComparison) -> tuple[int, int]:
    """Dimensions of the pictures the metric filter would see."""
    source_size = _content_size(comparison.source_info, comparison.source_crop)
    if comparison.resample_target is not None:
        return source_size
    distorted_size = _content_size(comparison.distorted_info, comparison.distorted_crop)
    if source_size == distorted_size:
        return source_size
    if comparison.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE:
        return source_size
    return distorted_size


def frame_input_path(comparison: FrameComparison, side: FrameSide) -> Path:
    """Physical file to decode, including synthetic resolution tests."""
    if side == "source" or comparison.resample_target is not None:
        return comparison.source_info.path
    return comparison.distorted_info.path


def frame_video_info(comparison: FrameComparison, side: FrameSide) -> VideoInfo:
    if side == "source" or comparison.resample_target is not None:
        return comparison.source_info
    return comparison.distorted_info


def hdr_kind(info: VideoInfo) -> str | None:
    """Return the declared HDR transfer family, without guessing from depth."""
    transfer = info.color_transfer.strip().casefold()
    if transfer in {"smpte2084", "smpte-st-2084"}:
        return "HDR10 / PQ"
    if transfer in {"arib-std-b67", "hlg"}:
        return "HLG"
    return None


def _tone_map_filter(info: VideoInfo, settings: PreviewColorSettings) -> list[str]:
    kind = hdr_kind(info)
    if settings.mode == PreviewColorMode.UNMANAGED:
        return ["format=rgb24"]
    if settings.mode == PreviewColorMode.DISPLAY_AWARE and kind is None:
        return ["format=rgb24"]

    # The explicit mode doubles as a recovery path for HDR files whose
    # container lost its colour tags.  Its UI label says that it assumes PQ;
    # Auto never guesses based on bit depth because 10-bit SDR is common.
    input_options: list[str] = []
    if kind is None:
        input_options = ["pin=bt2020", "tin=smpte2084", "min=bt2020nc", "rin=tv"]
    else:
        # Broken remuxes sometimes retain the PQ/HLG transfer tag but lose
        # primaries, matrix, or range.  zscale refuses an unspecified
        # conversion path, so use the standard BT.2020 YUV HDR defaults for
        # only those missing pieces while preserving every declared value.
        missing = {"", "unknown", "unspecified", "reserved"}
        primaries = (
            "bt2020" if info.color_primaries.casefold() in missing
            else info.color_primaries
        )
        matrix = (
            "bt2020nc" if info.color_space.casefold() in missing
            else info.color_space
        )
        color_range = (
            info.color_range if info.color_range in {"tv", "pc", "limited", "full"}
            else "tv"
        )
        input_options = [
            f"pin={primaries}", f"tin={info.color_transfer}",
            f"min={matrix}", f"rin={color_range}",
        ]

    linear = [*input_options, "t=linear", f"npl={settings.target_nits:g}"]
    return [
        f"zscale={':'.join(linear)}",
        "format=gbrpf32le",
        "tonemap=mobius:desat=2",
        (
            "zscale=p=bt709:t=bt709:m=bt709:r=tv:"
            "dither=error_diffusion"
        ),
        "format=rgb24",
    ]


def _would_distort(
    comparison: FrameComparison, info: VideoInfo, crop: CropBox | None,
    output_w: int, output_h: int,
) -> bool:
    """Whether scaling this side onto the output box changes its shape.

    Only consulted for previews whose crops are not settled yet. Once a run
    has measured them the two sides agree by construction, and a pair that
    still disagrees afterwards is refused rather than scored.
    """
    if not comparison.auto_crop_pending or output_h <= 0:
        return False
    source_shape = display_aspect_ratio(info, crop)
    target_shape = output_w / output_h
    if source_shape <= 0 or target_shape <= 0:
        return False
    return abs(source_shape - target_shape) > 0.01 * max(source_shape, target_shape)


def frame_filter(
    comparison: FrameComparison,
    side: FrameSide,
    color_settings: PreviewColorSettings | None = None,
    output_size: tuple[int, int] | None = None,
) -> str:
    """The crop/scale chain for one side of a comparison."""
    if side not in {"source", "distorted"}:
        raise ValueError(f"unknown frame side: {side}")

    info = comparison.source_info if side == "source" else comparison.distorted_info
    crop = comparison.source_crop if side == "source" else comparison.distorted_crop
    output_w, output_h = output_size or comparison_dimensions(comparison)
    ops: list[str] = []

    crop_filter = (
        crop.as_filter()
        if crop is not None and not crop.is_noop(info.width, info.height)
        else None
    )
    pixel_formats = [comparison.source_info.pix_fmt]
    if comparison.resample_target is None:
        pixel_formats.append(comparison.distorted_info.pix_fmt)
    analysis_format = analysis_pix_fmt(*pixel_formats)
    # Match the scorer's ordering: normal distorted frames crop before the
    # common format conversion; source and resolution-test branches convert
    # first. This matters slightly when the inputs have different bit depths.
    if side == "distorted" and comparison.resample_target is None:
        if crop_filter:
            ops.append(crop_filter)
        ops.append(f"format={analysis_format}")
    else:
        ops.append(f"format={analysis_format}")
        if crop_filter:
            ops.append(crop_filter)

    content_w, content_h = _content_size(info, crop)
    if comparison.resample_target is not None and side == "distorted":
        target_w = comparison.resample_target.width
        target_h = max(2, round(target_w * content_h / content_w / 2) * 2)
        ops.append(
            f"scale={target_w}:{target_h}:flags={comparison.scale_algorithm}"
        )
        ops.append(
            f"scale={output_w}:{output_h}:flags={comparison.scale_algorithm}"
        )
    elif (content_w, content_h) != (output_w, output_h):
        if _would_distort(comparison, info, crop, output_w, output_h):
            # Fitting rather than stretching. This only arises while
            # auto-crop is still pending: a letterboxed source is 16:9 until
            # its bars come off, and scaling it straight onto the shape of an
            # already-cropped 2.35:1 encode squashes the picture. A run would
            # never compare these two as they are -- validation refuses the
            # pair -- so a preview must not imply that it would. Padding
            # keeps both sides on one canvas for flipping between them while
            # every proportion stays true.
            ops.append(
                f"scale={output_w}:{output_h}:flags={comparison.scale_algorithm}"
                ":force_original_aspect_ratio=decrease"
            )
            ops.append(f"pad={output_w}:{output_h}:(ow-iw)/2:(oh-ih)/2")
        else:
            ops.append(
                f"scale={output_w}:{output_h}:flags={comparison.scale_algorithm}"
            )

    # PNG has square pixels and no useful video SAR. Resetting it after the
    # geometry operations ensures source/distorted previews occupy the exact
    # same canvas when the input used anamorphic storage.
    ops.append("setsar=1")
    ops.extend(_tone_map_filter(frame_video_info(comparison, side), color_settings or PreviewColorSettings()))
    return ",".join(ops)


def build_frame_command(
    comparison: FrameComparison,
    side: FrameSide,
    frame: int,
    color_settings: PreviewColorSettings | None = None,
) -> list[str]:
    if frame < 0:
        raise ValueError("frame number must be non-negative")
    if comparison.fps <= 0:
        raise FrameExtractError("This video has no usable frame rate.")

    # Accurate input seeking returns the first frame at or after the target.
    # Aim one eighth of a frame before the desired PTS: an exact/rounded-up
    # boundary can otherwise advance to the next frame, while this remains
    # far beyond the previous frame's PTS even at fractional frame rates.
    timestamp = max(0.0, (frame - 0.125) / comparison.fps)
    return [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{timestamp:.9f}",
        "-i", str(frame_input_path(comparison, side).resolve()),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", frame_filter(comparison, side, color_settings),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-c:v", "png",
        "pipe:1",
    ]


def extract_frame_png(
    comparison: FrameComparison,
    side: FrameSide,
    frame: int,
    process_handle: ProcessHandle | None = None,
    color_settings: PreviewColorSettings | None = None,
) -> bytes:
    path = frame_input_path(comparison, side)
    if not path.is_file():
        raise FrameExtractError(f"Video file is missing: {path}")

    handle = process_handle or ProcessHandle()
    process = proc_util.popen(
        build_frame_command(comparison, side, frame, color_settings),
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
