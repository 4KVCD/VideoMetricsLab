"""Generic metric results, independent of any particular execution backend."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from vmaf_app.core.metrics import METRIC_BY_KEY, MetricAggregation

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


LEGACY_PROVENANCE = MetricProvenance(
    implementation="legacy", implementation_version="", compute_backend="unknown",
    implementation_compatibility_id="legacy-v1",
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
    key: str
    score: float
    provenance: MetricProvenance

    def __post_init__(self) -> None:
        self.score = float(self.score)


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
    """Adapt legacy packed columns without copying their arrays."""
    provenance_by_key = provenance_by_key or {}
    results = MetricResultSet()
    for key in frames.metric_keys:
        values = frames.values(key)
        if values is not None:
            results.add(FrameMetricResult(
                key, frames.frame, frames.time, values,
                provenance_by_key.get(key, LEGACY_PROVENANCE),
            ))
    return results


def frame_scores_from_results(results: MetricResultSet):
    """Build a legacy view only when current metrics share one exact axis.

    Arbitrary future metric keys and metrics on a different sampling axis are
    intentionally excluded rather than being misaligned into FrameScores.
    """
    from vmaf_app.core.models import FrameScores

    legacy = [results.frame(key) for key in ("vmaf", "psnr", "ssim", "xpsnr", "vmaf_neg")]
    present = [result for result in legacy if result is not None]
    if not present:
        return FrameScores.empty()
    reference = present[0]
    if any(
        not (np.array_equal(reference.frame, result.frame) and np.array_equal(reference.time, result.time))
        for result in present[1:]
    ):
        return FrameScores.empty()
    return FrameScores(reference.frame, reference.time, metrics={
        result.key: result.values for result in present
    })


def merge_metric_results(existing: MetricResultSet, incoming: MetricResultSet) -> MetricResultSet:
    """Return a replacement-by-key merge; unrelated metric results survive."""
    merged = existing.copy()
    for key in incoming:
        result = incoming.get(key)
        assert result is not None
        merged.add(result)
    return merged
