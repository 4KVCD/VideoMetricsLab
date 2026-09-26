"""Shared data structures used across the core pipeline and UI."""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import numpy as np

from vmaf_app.core.metric_results import (
    MetricResultSet,
    frame_scores_from_results,
    results_from_frame_scores,
)
from vmaf_app.core.metrics import FRAME_METRICS, metric_definition


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
    # ffprobe's r_frame_rate. A meaningful difference from average fps is a
    # practical warning that frame-number/fps timestamps are unsafe (VFR).
    nominal_fps: float = 0.0
    # Stream colour tags are needed to distinguish a 10-bit SDR encode from
    # PQ/HLG HDR.  Empty means the container did not declare the value.
    color_range: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    chroma_location: str = ""

    @property
    def estimated_frame_count(self) -> int:
        if self.nb_frames > 0:
            return self.nb_frames
        return max(1, round(self.duration * self.fps))

    @property
    def is_variable_frame_rate(self) -> bool:
        if self.nominal_fps <= 0 or self.fps <= 0:
            return False
        return abs(self.nominal_fps - self.fps) > max(0.01, self.fps * 0.001)


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
    compute_vmaf: bool = True
    compute_vmaf_neg: bool = False
    # VMAF v1, a column of its own beside VMAF v0.6.1 ("vmaf"), with its own
    # model: model_choice_v1 is the UI selection ("__auto__" or a bundled v1
    # model), model_v1 what a run resolved it to.
    compute_vmaf_v1: bool = False
    model_choice_v1: str = "__auto__"
    model_v1: str = ""

    duration_limit: float = 0.0  # seconds; 0 = no limit, process the full video

    # Applies to BOTH inputs; each is planned separately (see gpu.plan_hwaccel)
    # and each falls back to software decode on its own.
    gpu_decode: bool = True
    gpu_vendor: GpuVendor = GpuVendor.AUTO

    crop_mode: CropMode = CropMode.AUTO
    manual_source_crop: CropBox | None = None
    manual_distorted_crop: CropBox | None = None

    # When set, this row is a resolution round-trip test (see ResampleTarget)
    # instead of a normal comparison against a second, already-encoded file.
    resample_test: ResampleTarget | None = None

    def requested_metrics(self) -> tuple[str, ...]:
        return tuple(
            metric.key for metric in FRAME_METRICS
            if metric.ffmpeg_binding is not None and self.metric_enabled(metric.key)
        )

    def metric_enabled(self, metric: str) -> bool:
        """Whether an established metric is requested by this options object."""
        binding = metric_definition(metric).ffmpeg_binding
        if binding is None:
            return False
        if binding.bool_option is not None:
            return bool(getattr(self, binding.bool_option))
        return binding.libvmaf_feature in self.extra_features

    def set_metric_enabled(self, metric: str, enabled: bool) -> None:
        """Enable or disable one metric without disturbing custom features."""
        binding = metric_definition(metric).ffmpeg_binding
        if binding is None:
            raise ValueError(f"metric {metric!r} is not provided by the FFmpeg options backend")
        if binding.bool_option is not None:
            setattr(self, binding.bool_option, enabled)
            return
        feature = binding.libvmaf_feature
        assert feature is not None
        if enabled:
            if feature not in self.extra_features:
                self.extra_features.append(feature)
        else:
            self.extra_features = [item for item in self.extra_features if item != feature]

def clone_options(opts: VmafOptions) -> VmafOptions:
    """A real copy, not a shared reference -- each row needs its own
    VmafOptions instance so editing one row can never bleed into another.

    `extra_features` is the reason this exists rather than a bare
    dataclasses.replace(): it's a mutable list, and a shallow copy would
    leave every row appending into the same one.
    """
    return replace(opts, extra_features=list(opts.extra_features))


@dataclass(frozen=True, slots=True)
class FrameScore:
    """A single frame's scores. This is a read-only *view* type -- convenient
    to pass around and read, but never how a whole run is stored (see
    FrameScores): a feature-length run is hundreds of thousands of frames,
    and one Python object per frame costs ~180 bytes against 28 in packed
    arrays.

    Frozen deliberately: indexing a FrameScores builds one of these on the
    fly, so assigning to it would update a throwaway object and silently
    lose the write. Better to raise than to quietly do nothing.
    """
    frame: int
    time: float
    vmaf: float | None
    psnr: float | None = None
    ssim: float | None = None
    xpsnr: float | None = None
    vmaf_neg: float | None = None


class FrameScores:
    """Per-frame scores for a whole run, stored as packed arrays rather than
    one object per frame (structure-of-arrays).

    Behaves like a sequence of FrameScore -- len(), indexing and iteration
    all work -- so readability at call sites is unchanged, but indexing
    builds a throwaway view rather than retaining an object per frame. Hot
    paths (plotting, stats, hover) should read the arrays directly instead:
    `scores.vmaf` rather than `[f.vmaf for f in scores]`.

    An optional metric that wasn't computed is None rather than an array of
    NaN, so "not computed" stays distinguishable from a real 0.0 score --
    both of which genuinely occur.
    """

    __slots__ = ("_metrics", "frame", "time")

    def __init__(
        self,
        frame: np.ndarray,
        time: np.ndarray,
        vmaf: np.ndarray | None = None,
        psnr: np.ndarray | None = None,
        ssim: np.ndarray | None = None,
        xpsnr: np.ndarray | None = None,
        vmaf_neg: np.ndarray | None = None,
        *,
        metrics: Mapping[str, object] | None = None,
    ) -> None:
        self.frame = np.asarray(frame, dtype=np.int32)
        # float64 for time: bisect during hover needs to stay exact across a
        # multi-hour run, where float32 only has ~0.001s of resolution.
        self.time = np.asarray(time, dtype=np.float64)
        self._metrics: dict[str, np.ndarray] = {}
        metric_arrays = {
            "vmaf": vmaf, "psnr": psnr, "ssim": ssim,
            "xpsnr": xpsnr, "vmaf_neg": vmaf_neg,
        }
        for key, values in metric_arrays.items():
            if values is not None:
                self._metrics[key] = np.asarray(values, dtype=np.float32)
        if metrics is not None:
            for key, values in metrics.items():
                if values is None:
                    continue
                self._metrics[key] = np.asarray(values, dtype=np.float32)

    def _metric_values(self, metric: str) -> np.ndarray | None:
        return self._metrics.get(metric)

    @property
    def vmaf(self) -> np.ndarray | None:
        return self._metric_values("vmaf")

    @property
    def vmaf_neg(self) -> np.ndarray | None:
        return self._metric_values("vmaf_neg")

    @property
    def psnr(self) -> np.ndarray | None:
        return self._metric_values("psnr")

    @property
    def ssim(self) -> np.ndarray | None:
        return self._metric_values("ssim")

    @property
    def xpsnr(self) -> np.ndarray | None:
        return self._metric_values("xpsnr")

    @classmethod
    def empty(cls) -> FrameScores:
        i32, f64 = np.int32, np.float64
        return cls(np.empty(0, i32), np.empty(0, f64))

    @classmethod
    def from_frames(cls, frames: Sequence[FrameScore]) -> FrameScores:
        """Pack a sequence of per-frame view objects into arrays."""
        if not frames:
            return cls.empty()

        def column(attr: str) -> np.ndarray | None:
            values = [getattr(f, attr) for f in frames]
            if all(v is None for v in values):
                return None
            return np.array([np.nan if v is None else v for v in values], dtype=np.float32)

        return cls(
            frame=np.array([f.frame for f in frames], dtype=np.int32),
            time=np.array([f.time for f in frames], dtype=np.float64),
            vmaf=column("vmaf"),
            psnr=column("psnr"), ssim=column("ssim"), xpsnr=column("xpsnr"),
            vmaf_neg=column("vmaf_neg"),
        )

    def values(self, metric: str) -> np.ndarray | None:
        """The array for a metric by name, or None if it wasn't computed."""
        return self._metrics.get(metric)

    def has(self, metric: str) -> bool:
        return self.values(metric) is not None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return tuple(self._metrics)

    def __len__(self) -> int:
        return int(self.frame.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int | slice) -> FrameScore | FrameScores:
        if isinstance(index, slice):
            def sliced(arr: np.ndarray | None) -> np.ndarray | None:
                return None if arr is None else arr[index]

            return FrameScores(self.frame[index], self.time[index],
                               metrics={key: sliced(values) for key, values in self._metrics.items()})

        def optional(arr: np.ndarray | None) -> float | None:
            if arr is None:
                return None
            value = float(arr[index])
            # NaN is how "no value for this particular frame" is stored
            # inside an otherwise-present column.
            return None if math.isnan(value) else value

        return FrameScore(
            frame=int(self.frame[index]),
            time=float(self.time[index]),
            vmaf=optional(self.vmaf),
            psnr=optional(self.psnr), ssim=optional(self.ssim), xpsnr=optional(self.xpsnr),
            vmaf_neg=optional(self.vmaf_neg),
        )

    def with_values(self, metric: str, values: np.ndarray | None) -> FrameScores:
        """A copy with one metric's column replaced -- the supported way to
        change scores, since the per-frame views are read-only."""
        columns = dict(self._metrics)
        if values is None:
            columns.pop(metric, None)
        else:
            columns[metric] = values
        return FrameScores(self.frame, self.time, metrics=columns)

    def __iter__(self) -> Iterator[FrameScore]:
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameScores):
            return NotImplemented
        if not (np.array_equal(self.frame, other.frame) and np.array_equal(self.time, other.time)):
            return False
        if set(self._metrics) != set(other._metrics):
            return False
        for metric in self._metrics:
            a, b = self.values(metric), other.values(metric)
            if (a is None) != (b is None):
                return False
            if a is not None and not np.array_equal(a, b, equal_nan=True):
                return False
        return True

    def nbytes(self) -> int:
        total = self.frame.nbytes + self.time.nbytes
        total += sum(arr.nbytes for arr in self._metrics.values())
        return total


@dataclass
class ComparisonResult:
    source: Path
    distorted: Path
    frames: FrameScores
    fps: float
    model: str
    source_crop: CropBox | None
    distorted_crop: CropBox | None
    source_info: VideoInfo
    distorted_info: VideoInfo
    # Which direction resolution mismatches were resolved in for this run --
    # recorded (not just derived from the row's *current* options) so a
    # reloaded/cached result always reflects what actually produced these
    # scores, even if the row's own settings were changed since.
    scale_direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED
    # Frame Compare needs the exact preprocessing recipe that produced the
    # scored pictures.
    scale_algorithm: str = "bicubic"
    resample_target: ResampleTarget | None = None
    compared_frame_count: int = 0
    # The UI choice that produced ``model`` (for example a bundled VMAF v1
    # model). Programmatically-created results may leave it unset.
    model_choice: str | None = None
    # The same for VMAF v1 (a bundled model's path, and the UI choice).
    model_v1: str = ""
    model_choice_v1: str | None = None
    # Generic results are authoritative. ``frames`` is the shared-axis view
    # consumed by the current UI and established frame-oriented tools.
    metric_results: MetricResultSet = field(default_factory=MetricResultSet)

    def __post_init__(self) -> None:
        # Accept a plain list of FrameScore and pack it so storage always has
        # exactly one representation downstream.
        if not isinstance(self.frames, FrameScores):
            self.frames = FrameScores.from_frames(self.frames)
        if not self.metric_results:
            self.metric_results = results_from_frame_scores(self.frames)
        else:
            compatible = frame_scores_from_results(self.metric_results)
            # Generic-only construction supplies FrameScores.empty(); rebuild
            # the UI view when the available frame metrics share one axis.
            if not self.frames and compatible:
                self.frames = compatible
        if self.compared_frame_count <= 0 and len(self.frames):
            self.compared_frame_count = int(self.frames.frame[-1]) + 1

    def metric(self, key: str):
        return self.metric_results.get(key)

    def has_metric(self, key: str) -> bool:
        return self.metric_results.has(key)

    def frame_metric(self, key: str):
        return self.metric_results.frame(key)

    def sequence_metric(self, key: str):
        return self.metric_results.sequence(key)

    def merge_metric_results(self, incoming: MetricResultSet) -> None:
        """Single merge point for result adapters; UI never merges arrays."""
        from vmaf_app.core.metric_results import merge_metric_results

        self.metric_results = merge_metric_results(self.metric_results, incoming)
        compatible = frame_scores_from_results(self.metric_results)
        if compatible:
            self.frames = compatible
