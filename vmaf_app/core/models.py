"""Shared data structures used across the core pipeline and UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CropMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    NONE = "none"


class GpuVendor(str, Enum):
    AUTO = "auto"
    NVIDIA = "nvidia"
    INTEL = "intel"
    AMD = "amd"
    NONE = "none"


class ScaleDirection(str, Enum):
    # Default: scale the source to match the distorted video's resolution
    # (whichever direction that means) -- evaluates quality at the
    # resolution actually delivered/viewed.
    SOURCE_TO_DISTORTED = "source_to_distorted"
    # Scale the distorted video UP to match the source's resolution instead
    # -- evaluates quality as if the distorted video were upscaled back to
    # the source's native resolution for playback.
    DISTORTED_TO_SOURCE = "distorted_to_source"


@dataclass
class CropBox:
    """A crop rectangle in source-pixel coordinates, ffmpeg crop=w:h:x:y order."""
    w: int
    h: int
    x: int
    y: int

    def as_filter(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"

    def is_noop(self, frame_w: int, frame_h: int) -> bool:
        return self.w == frame_w and self.h == frame_h and self.x == 0 and self.y == 0


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    codec_name: str
    sar: str = "1:1"
    pix_fmt: str = ""
    bit_rate: int = 0  # bits per second, 0 if unknown

    @property
    def estimated_frame_count(self) -> int:
        if self.nb_frames > 0:
            return self.nb_frames
        return max(1, round(self.duration * self.fps))


@dataclass
class ResampleTarget:
    """Describes a "downscale then upscale back" resolution round-trip test:
    the source is scaled down to `width` (height derived to preserve the
    source's own aspect ratio), then scaled back up to the source's original
    resolution, and compared against the untouched source. Named by
    horizontal resolution (not the conventional vertical-resolution meaning
    of "1080p" etc.) so it's well-defined for any source aspect ratio, not
    just 16:9.
    """
    width: int
    label: str  # e.g. "1080p", for display


RESAMPLE_TARGET_CHOICES: list[ResampleTarget] = [
    ResampleTarget(width=2560, label="1440p"),
    ResampleTarget(width=1920, label="1080p"),
    ResampleTarget(width=1280, label="720p"),
    ResampleTarget(width=854, label="480p"),
]


def synthetic_scale_direction_variant_path(distorted_path: Path, direction: ScaleDirection) -> Path:
    """A well-formed but non-existent Path standing in for a second run of the
    *same* distorted file against the *other* ScaleDirection, so both
    directions can be added as separate rows and compared side by side (in
    the table and the comparison graph) without one silently colliding with
    -- and overwriting the cache entry or graph series of -- the other, the
    same way synthetic_resample_distorted_path does for resample tests.
    """
    tag = "upscale-distorted-to-source" if direction == ScaleDirection.DISTORTED_TO_SOURCE else "downscale-source-to-distorted"
    return distorted_path.with_name(f"{distorted_path.stem} [{tag}]{distorted_path.suffix}")


def synthetic_resample_distorted_path(source_path: Path, target: ResampleTarget) -> Path:
    """A well-formed but non-existent Path standing in for the "distorted"
    file in a resolution round-trip test, since there isn't a real second
    file -- both branches are derived from the same source at run time. It
    exists so a resample test gets a stable, resolution-specific identity
    for caching, dedup in the comparison graph, and display, the same way a
    real distorted file's path would -- carrying the target resolution in
    the name is what keeps different targets (e.g. 1080p vs 720p) of the
    same source from colliding as if they were the same result.
    """
    return source_path.with_name(f"{source_path.stem} [downscale-{target.label}-upscale]{source_path.suffix}")


@dataclass
class VmafOptions:
    model: str = "version=vmaf_v0.6.1"  # the resolved ffmpeg model= value actually used to run
    # model_choice/custom_model_path capture the *UI selection* driving `model` (e.g. "__auto__"
    # picks the 4K model based on the distorted video's resolution at run time) -- kept alongside
    # the resolved value so per-video settings in the UI can be edited and re-resolved later.
    model_choice: str = "__auto__"
    custom_model_path: str | None = None
    extra_features: list[str] = field(default_factory=list)  # e.g. ["name=psnr", "name=float_ssim"]
    n_threads: int = 0  # 0 = let libvmaf decide (all cores)
    n_subsample: int = 1
    scale_algorithm: str = "bicubic"
    scale_direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED
    compute_xpsnr: bool = False

    duration_limit: float = 0.0  # seconds; 0 = no limit, process the full video

    gpu_decode_source: bool = True
    gpu_vendor: GpuVendor = GpuVendor.AUTO

    crop_mode: CropMode = CropMode.AUTO
    manual_source_crop: CropBox | None = None
    manual_distorted_crop: CropBox | None = None

    # When set, this row is a resolution round-trip test (see ResampleTarget)
    # instead of a normal comparison against a second, already-encoded file.
    resample_test: ResampleTarget | None = None


@dataclass(slots=True)
class FrameScore:
    frame: int
    time: float
    vmaf: float
    psnr: float | None = None
    ssim: float | None = None
    xpsnr: float | None = None


@dataclass
class VmafRunResult:
    source: Path
    distorted: Path
    frames: list[FrameScore]
    fps: float
    model: str
    source_crop: CropBox | None
    distorted_crop: CropBox | None
    source_info: VideoInfo
    distorted_info: VideoInfo
    raw_log_path: Path | None = None
    # Which direction resolution mismatches were resolved in for this run --
    # recorded (not just derived from the row's *current* options) so a
    # reloaded/cached result always reflects what actually produced these
    # scores, even if the row's own settings were changed since.
    scale_direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED
