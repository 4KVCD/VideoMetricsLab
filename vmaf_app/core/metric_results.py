"""Generic metric results, independent of any particular execution backend."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from vmaf_app.core.metrics import FRAME_METRICS, METRIC_BY_KEY, MetricAggregation

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class MetricProvenance:
    """How a metric value was produced, separate from cache compatibility."""

    implementation: str
    implementation_version: str
    compute_backend: str
    implementation_compatibility_id: str
    parameters: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Avoid retaining a caller-owned mutable mapping in an otherwise
        # immutable value object. JSON serialization validates the allowed
        # shape early without permitting arbitrary Python values in cache data.
        import json

        copied = dict(self.parameters)
        json.dumps(copied, allow_nan=False)
        object.__setattr__(self, "parameters", copied)


UNSPECIFIED_PROVENANCE = MetricProvenance(
    implementation="unknown", implementation_version="", compute_backend="unknown",
    implementation_compatibility_id="unversioned",
)


def provenance_to_dict(provenance: MetricProvenance) -> dict[str, JSONValue]:
    """Return the JSON-ready representation shared by caches and portable files."""
    return {
        "implementation": provenance.implementation,
        "implementation_version": provenance.implementation_version,
        "compute_backend": provenance.compute_backend,
        "implementation_compatibility_id": provenance.implementation_compatibility_id,
        "parameters": dict(provenance.parameters),
    }


def provenance_from_dict(data: Mapping[str, object]) -> MetricProvenance:
    """Rebuild provenance from trusted JSON-decoded data."""
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TypeError("metric provenance parameters must be an object")
    return MetricProvenance(
        implementation=str(data["implementation"]),
        implementation_version=str(data["implementation_version"]),
        compute_backend=str(data["compute_backend"]),
        implementation_compatibility_id=str(data["implementation_compatibility_id"]),
        parameters=parameters,
    )


def current_ffmpeg_provenance(key: str, version: str, parameters: dict[str, JSONValue] | None = None) -> MetricProvenance:
    """Provenance for a newly computed current FFmpeg metric.

    Decode hardware is intentionally not reported as metric compute hardware:
    libvmaf and the current xpsnr filter perform their measurement on CPU.
    """
    return MetricProvenance(
        implementation="ffmpeg/xpsnr" if key == "xpsnr" else "ffmpeg/libvmaf",
        implementation_version=f"ffmpeg {version}",
        compute_backend="cpu",
        implementation_compatibility_id=(
            "ffmpeg-xpsnr-v1" if key == "xpsnr" else "ffmpeg-libvmaf-v1"
        ),
        parameters=parameters or {},
    )


@dataclass(slots=True)
class FrameMetricResult:
    key: str
    frame: np.ndarray
    time: np.ndarray
    values: np.ndarray
    provenance: MetricProvenance

    def __post_init__(self) -> None:
        self.frame = np.asarray(self.frame, dtype=np.int32)
        self.time = np.asarray(self.time, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float32)
        if not (len(self.frame) == len(self.time) == len(self.values)):
            raise ValueError("frame, time, and metric values must have equal lengths")

    @property
    def aggregate(self) -> float | None:
        from vmaf_app.core.stats import aggregate_scores

        definition = METRIC_BY_KEY.get(self.key)
        aggregation = (
            definition.aggregation if definition is not None
            else MetricAggregation.ARITHMETIC
        )
        return aggregate_scores(self.values, aggregation)


@dataclass(slots=True)
class SequenceMetricResult:
    """One score for the whole comparison.

    `frame`/`time`/`values` optionally carry a timeline -- for CVVDP, the JOD
    of each one-second window, at the window's first frame. The timeline is
    for finding bad stretches; `score` is the metric (it is not the mean of
    the timeline).
    """

    key: str
    score: float
    provenance: MetricProvenance
    frame: np.ndarray | None = None
    time: np.ndarray | None = None
    values: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if self.values is not None:
            self.frame = np.asarray(self.frame, dtype=np.int32)
            self.time = np.asarray(self.time, dtype=np.float64)
            self.values = np.asarray(self.values, dtype=np.float32)
            if not (len(self.frame) == len(self.time) == len(self.values)):
                raise ValueError("timeline frame, time, and values must have equal lengths")

    @property
    def has_timeline(self) -> bool:
        return self.values is not None and len(self.values) > 0

    @property
    def aggregate(self) -> float:
        return self.score


MetricResult = FrameMetricResult | SequenceMetricResult


class MetricResultSet:
    """Independent metric outputs; frame metrics need not share an axis."""

    def __init__(self, results: Iterable[MetricResult] = ()) -> None:
        self._results: dict[str, MetricResult] = {}
        for result in results:
            self.add(result)

    def add(self, result: MetricResult) -> None:
        self._results[result.key] = result

    def get(self, key: str) -> MetricResult | None:
        return self._results.get(key)

    def has(self, key: str) -> bool:
        return key in self._results

    def frame(self, key: str) -> FrameMetricResult | None:
        result = self.get(key)
        return result if isinstance(result, FrameMetricResult) else None

    def sequence(self, key: str) -> SequenceMetricResult | None:
        result = self.get(key)
        return result if isinstance(result, SequenceMetricResult) else None

    def keys(self) -> tuple[str, ...]:
        return tuple(self._results)

    def __iter__(self):
        return iter(self._results)

    def __bool__(self) -> bool:
        return bool(self._results)

    def copy(self) -> MetricResultSet:
        return MetricResultSet(self._results.values())


def results_from_frame_scores(frames, provenance_by_key: dict[str, MetricProvenance] | None = None) -> MetricResultSet:
    """Adapt a shared-axis ``FrameScores`` view without copying its arrays."""
    provenance_by_key = provenance_by_key or {}
    results = MetricResultSet()
    for key in frames.metric_keys:
        values = frames.values(key)
        if values is not None:
            results.add(FrameMetricResult(
                key, frames.frame, frames.time, values,
                provenance_by_key.get(key, UNSPECIFIED_PROVENANCE),
            ))
    return results


def frame_scores_from_results(results: MetricResultSet):
    """Build the shared-axis UI view when registered frame metrics align.

    Metrics on a different sampling axis and sequence metrics remain in the
    generic result set rather than being misaligned into ``FrameScores``.
    """
    from vmaf_app.core.models import FrameScores

    frame_results = [results.frame(metric.key) for metric in FRAME_METRICS]
    present = [result for result in frame_results if result is not None]
    if not present:
        return FrameScores.empty()

    # The first registered metric that is present defines the UI axis. Metrics
    # with their own sampling axis stay authoritative in MetricResultSet and
    # are simply omitted from this shared-axis projection. One independent
    # metric must never make otherwise-displayable scores disappear.
    #
    # Alignment is by frame number alone, which identifies the frame. Times
    # are derived -- frame / fps, from the reference's or the test's frame
    # rate depending on the backend -- and can differ by rounding for the
    # same frames. Requiring them to be bit-identical dropped every metric
    # but VMAF from real results: on a cached 151,919-frame film PSNR, SSIM,
    # XPSNR, SSIMULACRA2 and Butteraugli all had VMAF's frame numbers but
    # times up to 3.3e-7 s apart, so the CSV export, the "scored frames"
    # text and Video Compare lost them. The table uses the first metric's
    # times.
    reference = present[0]
    aligned = [result for result in present if np.array_equal(reference.frame, result.frame)]
    return FrameScores(reference.frame, reference.time, metrics={
        result.key: result.values for result in aligned
    })


def merge_metric_results(existing: MetricResultSet, incoming: MetricResultSet) -> MetricResultSet:
    """Return a replacement-by-key merge; unrelated metric results survive."""
    merged = existing.copy()
    for key in incoming:
        result = incoming.get(key)
        assert result is not None
        merged.add(result)
    return merged
