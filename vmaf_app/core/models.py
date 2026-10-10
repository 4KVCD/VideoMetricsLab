"""The data the core and the window share: a video's details (VideoInfo), a
row's settings (VmafOptions), black-bar crops (CropBox), and a comparison's
result (ComparisonResult, with each frame's scores in FrameScores)."""
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
    """Whether black bars are detected and cut (AUTO) or kept (NONE)."""

    AUTO = "auto"
    NONE = "none"


class GpuVendor(str, Enum):
    """The GPU maker whose decoder to use: AUTO picks one found on this PC,
    and NONE decodes in software."""

    AUTO = "auto"
    NVIDIA = "nvidia"
    INTEL = "intel"
    AMD = "amd"
    NONE = "none"


class ScaleDirection(str, Enum):
    """Which video is scaled when the two sizes differ."""

    # The default: scale the source to the test video's size (up or down),
    # which judges quality at the size actually delivered.
    SOURCE_TO_DISTORTED = "source_to_distorted"
    # Scale the test video up to the source's size instead, which judges
    # quality as if the test video were played back at the source's size.
    DISTORTED_TO_SOURCE = "distorted_to_source"


@dataclass
class CropBox:
    """A crop rectangle in the video's own pixels, in FFmpeg's crop=w:h:x:y
    order."""

    w: int
    h: int
    x: int
    y: int

    def as_filter(self) -> str:
        """The FFmpeg crop filter."""
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"

    def is_noop(self, frame_w: int, frame_h: int) -> bool:
        """Whether the crop keeps the whole frame of this size."""
        return self.w == frame_w and self.h == frame_h and self.x == 0 and self.y == 0


def crop_to_dict(crop: CropBox | None) -> dict | None:
    """A crop as run files and the metric cache's context.json save it."""
    return None if crop is None else {"w": crop.w, "h": crop.h, "x": crop.x, "y": crop.y}


def crop_from_dict(data: Mapping | None) -> CropBox | None:
    """A crop read back from crop_to_dict's form."""
    return None if data is None else CropBox(**data)


#: FFmpeg's demuxers of bare elementary streams, which carry no timestamps.
RAW_STREAM_FORMATS = frozenset({
    "h264", "hevc", "vvc", "evc", "av1", "obu", "mpegvideo", "m4v", "h263", "vc1", "dirac",
    "cavsvideo", "avs2", "avs3", "mjpeg",
})


@dataclass(slots=True)
class VideoInfo:
    """A video file's details, as ffprobe reports them (core.ffprobe)."""

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
    # The file's bitrate, soundtrack included, where the video's own is not
    # recorded (a Matroska file without statistics tags).
    bit_rate_whole_file: bool = False
    # ffprobe's r_frame_rate. A real difference from the average frame rate
    # warns that the frame rate varies, so frame number / fps is not a safe
    # time.
    nominal_fps: float = 0.0
    # ffprobe's avg_frame_rate as it gives it: 0 where it gives none, and
    # `fps` is then the nominal rate (vmaf_runner._xpsnr_frame_rate).
    average_fps: float = 0.0
    # The stream's colour tags, which tell a 10-bit SDR encode from PQ or HLG
    # HDR. Empty means the file did not state the value.
    color_range: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    chroma_location: str = ""
    # ffprobe's format_name: the container, or the raw stream's demuxer
    # ("hevc", "h264", ...). It is not saved with results, because a raw
    # stream is refused before it has any (is_raw_stream).
    format_name: str = ""
    # How long after the file's start (its earliest stream's, which FFmpeg's
    # -ss counts from) the video's first frame comes, in seconds. An MP4
    # whose soundtrack starts at 0 can have its video start frames later
    # (frame_extract.seek_seconds). It is not saved with results: it is read
    # again with the file, and is no part of what the video is.
    start_offset: float = field(default=0.0, compare=False)

    @property
    def estimated_frame_count(self) -> int:
        """The frame count ffprobe reports, or else duration times fps."""
        if self.nb_frames > 0:
            return self.nb_frames
        return max(1, round(self.duration * self.fps))

    @property
    def is_raw_stream(self) -> bool:
        """A bare elementary stream (a .hevc or .264 file), not a container:
        it has no timestamps, so FFmpeg times its frames at 25 a second
        unless told otherwise, and its frame rate cannot be known."""
        return self.format_name in RAW_STREAM_FORMATS

    @property
    def is_variable_frame_rate(self) -> bool:
        """Whether the nominal and the average frame rate differ by more
        than rounding."""
        if self.nominal_fps <= 0 or self.fps <= 0:
            return False
        return abs(self.nominal_fps - self.fps) > max(0.01, self.fps * 0.001)


def video_info_to_dict(info: VideoInfo) -> dict:
    """A video's details as run files (run_io) and the metric cache's
    context.json save them. Both use this one form, so a field added to
    VideoInfo is saved by both or by neither (format_name and start_offset
    are not saved)."""
    return {
        "path": str(info.path), "width": info.width, "height": info.height,
        "fps": info.fps, "duration": info.duration, "nb_frames": info.nb_frames,
        "codec_name": info.codec_name, "sar": info.sar, "pix_fmt": info.pix_fmt,
        "bit_rate": info.bit_rate, "bit_rate_whole_file": info.bit_rate_whole_file,
        "nominal_fps": info.nominal_fps, "average_fps": info.average_fps,
        "color_range": info.color_range, "color_space": info.color_space,
        "color_transfer": info.color_transfer, "color_primaries": info.color_primaries,
        "chroma_location": info.chroma_location,
    }


def video_info_from_dict(data: Mapping) -> VideoInfo:
    """A video's details read back from video_info_to_dict's form. A field
    missing from older files loads as its default."""
    return VideoInfo(path=Path(data["path"]), width=data["width"], height=data["height"],
                     fps=data["fps"], duration=data["duration"], nb_frames=data["nb_frames"],
                     codec_name=data["codec_name"], sar=data.get("sar", "1:1"),
                     pix_fmt=data.get("pix_fmt", ""), bit_rate=data.get("bit_rate", 0),
                     bit_rate_whole_file=bool(data.get("bit_rate_whole_file", False)),
                     nominal_fps=data.get("nominal_fps", 0.0), average_fps=data.get("average_fps", 0.0),
                     color_range=data.get("color_range", ""), color_space=data.get("color_space", ""),
                     color_transfer=data.get("color_transfer", ""),
                     color_primaries=data.get("color_primaries", ""),
                     chroma_location=data.get("chroma_location", ""))


@dataclass
class ResampleTarget:
    """A resolution test: the source is scaled down to `width` (the height
    keeping the source's aspect ratio), scaled back up to its own size, and
    compared with the untouched source. It is named by width rather than
    the usual height ("1080p"), so that it is well defined for any aspect
    ratio, not just 16:9.
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
    """Returns a made-up path that stands for a second run of the same test
    video with the other ScaleDirection. Both directions can then be rows of
    their own, side by side in the table and the graph, without one
    overwriting the other's cache entry or graph line (as
    synthetic_resample_distorted_path does for resolution tests).
    """
    tag = "upscale-distorted-to-source" if direction == ScaleDirection.DISTORTED_TO_SOURCE else "downscale-source-to-distorted"
    return distorted_path.with_name(f"{distorted_path.stem} [{tag}]{distorted_path.suffix}")


def synthetic_resample_distorted_path(source_path: Path, target: ResampleTarget) -> Path:
    """Returns a made-up path that stands for the test video of a resolution
    test, which has no second file: both videos are made from the source
    during the run. It gives the test a stable identity for the cache, the
    graph and the table, as a real test video's path would. The target
    resolution is in the name, so that different targets of the same
    source (1080p and 720p, say) are not taken for the same result.
    """
    return source_path.with_name(f"{source_path.stem} [downscale-{target.label}-upscale]{source_path.suffix}")


@dataclass
class VmafOptions:
    """A row's settings: which metrics to calculate, and how."""

    model: str = "version=vmaf_v0.6.1"  # the FFmpeg model= value a run actually uses
    # model_choice and custom_model_path are what the user chose, from which
    # `model` is worked out ("__auto__" picks the 4K model when the videos
    # are compared at 4K). They are kept beside it so that the row's
    # settings can be edited and worked out again later.
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
    # model: model_choice_v1 is what the user chose ("__auto__" or a bundled
    # v1 model), and model_v1 what a run worked it out to be.
    compute_vmaf_v1: bool = False
    model_choice_v1: str = "__auto__"
    model_v1: str = ""

    duration_limit: float = 0.0  # seconds; 0 = no limit, process the full video

    # Applies to both videos. Each is planned separately (see
    # gpu.plan_hwaccel), and each falls back to software decoding on its own.
    gpu_decode: bool = True
    gpu_vendor: GpuVendor = GpuVendor.AUTO
    # Calculate VMAF v0.6.1 and VMAF NEG on the GPU (vmaf_cuda, vmaf_vulkan),
    # and VMAF v1 with it (vmaf_v1_gpu), when a GPU can be used; else on the
    # CPU. Like gpu_decode, this changes only how scores are calculated: the
    # two agree to within a thousandth of a point per frame, so it is not
    # part of the cache key, and a saved score is kept either way.
    vmaf_on_gpu: bool = True

    crop_mode: CropMode = CropMode.AUTO

    # When set, this row is a resolution test (see ResampleTarget) instead of
    # a comparison with a second, already encoded file.
    resample_test: ResampleTarget | None = None

    def requested_metrics(self) -> tuple[str, ...]:
        """The keys of the FFmpeg metrics these options ask for."""
        return tuple(
            metric.key for metric in FRAME_METRICS
            if metric.ffmpeg_binding is not None and self.metric_enabled(metric.key)
        )

    def metric_enabled(self, metric: str) -> bool:
        """Whether these options ask for an FFmpeg metric."""
        binding = metric_definition(metric).ffmpeg_binding
        if binding is None:
            return False
        if binding.bool_option is not None:
            return bool(getattr(self, binding.bool_option))
        return binding.libvmaf_feature in self.extra_features

    def set_metric_enabled(self, metric: str, enabled: bool) -> None:
        """Turns one FFmpeg metric on or off, leaving other features alone."""
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
    """Returns a real copy of the options, so that editing one row can never
    change another.

    A plain dataclasses.replace() is not enough, because `extra_features`
    is a list: a shallow copy would leave every row adding to the same one.
    """
    return replace(opts, extra_features=list(opts.extra_features))


@dataclass(frozen=True, slots=True)
class FrameScore:
    """One frame's scores. It is a read-only view, handy to pass around and
    read, but never how a whole run is stored (see FrameScores): a
    feature-length run has hundreds of thousands of frames, and one Python
    object per frame takes about 180 bytes, against 28 in packed arrays.

    It is frozen on purpose: indexing a FrameScores builds one of these on
    the fly, so assigning to it would change a throwaway object and lose the
    change without a word. Raising an error is better.
    """
    frame: int
    time: float
    vmaf: float | None
    psnr: float | None = None
    ssim: float | None = None
    xpsnr: float | None = None
    vmaf_neg: float | None = None


class FrameScores:
    """A whole run's scores for each frame, stored as packed arrays (one per
    metric) rather than one object per frame.

    It behaves like a sequence of FrameScore (len(), indexing and iteration
    all work), but indexing builds a throwaway view rather than keeping an
    object per frame. Code that runs often (graphs, statistics, hovering)
    should read the arrays directly: `scores.vmaf` rather than
    `[f.vmaf for f in scores]`.

    A metric that was not calculated is None rather than an array of NaN,
    so that "not calculated" stays distinct from a real score of 0.0, and
    both do occur.
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
        # Times are float64, because the search when hovering must stay exact
        # across a run of several hours, where float32 resolves only about
        # a millisecond.
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
        """A metric's array, or None if it was not calculated."""
        return self._metrics.get(metric)

    @property
    def vmaf(self) -> np.ndarray | None:
        """VMAF's scores, or None."""
        return self._metric_values("vmaf")

    @property
    def vmaf_neg(self) -> np.ndarray | None:
        """VMAF NEG's scores, or None."""
        return self._metric_values("vmaf_neg")

    @property
    def psnr(self) -> np.ndarray | None:
        """PSNR's scores, or None."""
        return self._metric_values("psnr")

    @property
    def ssim(self) -> np.ndarray | None:
        """SSIM's scores, or None."""
        return self._metric_values("ssim")

    @property
    def xpsnr(self) -> np.ndarray | None:
        """XPSNR's scores, or None."""
        return self._metric_values("xpsnr")

    @classmethod
    def empty(cls) -> FrameScores:
        """Scores for no frames."""
        i32, f64 = np.int32, np.float64
        return cls(np.empty(0, i32), np.empty(0, f64))

    @classmethod
    def from_frames(cls, frames: Sequence[FrameScore]) -> FrameScores:
        """Packs a sequence of FrameScore into arrays."""
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
        """The array for a metric by name, or None if it was not calculated."""
        return self._metrics.get(metric)

    def has(self, metric: str) -> bool:
        """Whether the metric was calculated."""
        return self.values(metric) is not None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        """The keys of the metrics calculated."""
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
            # NaN stores "no value for this frame" in a metric's column.
            return None if math.isnan(value) else value

        return FrameScore(
            frame=int(self.frame[index]),
            time=float(self.time[index]),
            vmaf=optional(self.vmaf),
            psnr=optional(self.psnr), ssim=optional(self.ssim), xpsnr=optional(self.xpsnr),
            vmaf_neg=optional(self.vmaf_neg),
        )

    def with_values(self, metric: str, values: np.ndarray | None) -> FrameScores:
        """Returns a copy with one metric's column replaced (or removed, for
        None). This is the way to change scores, since the per-frame views
        are read-only."""
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
        """The memory the arrays take, in bytes."""
        total = self.frame.nbytes + self.time.nbytes
        total += sum(arr.nbytes for arr in self._metrics.values())
        return total


@dataclass
class ComparisonResult:
    """One comparison's result: the two videos, how they were compared, and
    every metric's scores."""

    source: Path
    distorted: Path
    frames: FrameScores
    fps: float
    model: str
    source_crop: CropBox | None
    distorted_crop: CropBox | None
    source_info: VideoInfo
    distorted_info: VideoInfo
    # The way the sizes were matched in this run. It is recorded, rather than
    # taken from the row's current settings, so that a reloaded or saved
    # result always says what produced these scores, even if the row's
    # settings changed since.
    scale_direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED
    # Frame Compare needs exactly how the scored pictures were made.
    scale_algorithm: str = "bicubic"
    resample_target: ResampleTarget | None = None
    compared_frame_count: int = 0
    # What the user chose that gave ``model`` (a bundled VMAF v1 model, say).
    # Results made in code may leave it unset.
    model_choice: str | None = None
    # The same for VMAF v1 (a bundled model's path, and the user's choice).
    model_v1: str = ""
    model_choice_v1: str | None = None
    # The scores themselves. ``frames`` is a view of the frame metrics that
    # share one frame axis, which the window and the frame tools read.
    metric_results: MetricResultSet = field(default_factory=MetricResultSet)

    def __post_init__(self) -> None:
        # A plain list of FrameScore is packed, so that there is only one
        # form from here on.
        if not isinstance(self.frames, FrameScores):
            self.frames = FrameScores.from_frames(self.frames)
        if not self.metric_results:
            self.metric_results = results_from_frame_scores(self.frames)
        else:
            compatible = frame_scores_from_results(self.metric_results)
            # A result made from metric_results alone gets FrameScores.empty();
            # the view is rebuilt when the frame metrics share one axis.
            if not self.frames and compatible:
                self.frames = compatible
        if self.compared_frame_count <= 0 and len(self.frames):
            self.compared_frame_count = int(self.frames.frame[-1]) + 1

    def metric(self, key: str):
        """A metric's result, or None."""
        return self.metric_results.get(key)

    def has_metric(self, key: str) -> bool:
        """Whether the result has the metric."""
        return self.metric_results.has(key)

    def frame_metric(self, key: str):
        """A per-frame metric's result, or None."""
        return self.metric_results.frame(key)

    def sequence_metric(self, key: str):
        """A whole-video metric's result (CVVDP), or None."""
        return self.metric_results.sequence(key)

    def merge_metric_results(self, incoming: MetricResultSet) -> None:
        """Adds or replaces metrics' results, and rebuilds `frames`. This is
        the one place results are merged; the window never merges arrays."""
        from vmaf_app.core.metric_results import merge_metric_results

        self.metric_results = merge_metric_results(self.metric_results, incoming)
        compatible = frame_scores_from_results(self.metric_results)
        if compatible:
            self.frames = compatible
