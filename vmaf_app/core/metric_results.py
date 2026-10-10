"""Metric results as every backend gives them: a score per frame
(FrameMetricResult) or one for the whole video (SequenceMetricResult), each
with its provenance, collected in a MetricResultSet."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from vmaf_app.core.metrics import FRAME_METRICS, METRIC_BY_KEY, MetricAggregation

#: The identity cached and saved XPSNR scores are reused by. v2 since
#: October 2026: weighted by the source's activity (vmaf_runner._xpsnr_filter),
#: not the encode's as before, which gave other scores.
XPSNR_COMPATIBILITY_ID = "ffmpeg-xpsnr-v2"

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
            XPSNR_COMPATIBILITY_ID if key == "xpsnr" else "ffmpeg-libvmaf-v1"
        ),
        parameters=parameters or {},
    )


@dataclass(slots=True)
class FrameMetricResult:
    """A metric with a score per frame: the frame numbers, their times, the scores and the provenance."""

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
        """The one number for the whole video, by the metric's own way of averaging."""
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

    `frame`, `time` and `values` may hold a timeline: for CVVDP, the JOD of
    each one-second window, at the window's first frame. The timeline is for
    finding bad stretches; `score` is the metric (it is not the timeline's
    mean).
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
        """Whether the result has a timeline."""
        return self.values is not None and len(self.values) > 0

    @property
    def aggregate(self) -> float:
        """The one number for the whole video: the score."""
        return self.score


MetricResult = FrameMetricResult | SequenceMetricResult


class MetricResultSet:
    """The results of a comparison's metrics, by key. Each is independent,
    and frame metrics need not share the same frames."""

    def __init__(self, results: Iterable[MetricResult] = ()) -> None:
        self._results: dict[str, MetricResult] = {}
        for result in results:
            self.add(result)

    def add(self, result: MetricResult) -> None:
        """Adds or replaces a metric's result."""
        self._results[result.key] = result

    def get(self, key: str) -> MetricResult | None:
        """A metric's result, or None."""
        return self._results.get(key)

    def has(self, key: str) -> bool:
        """Whether the set has the metric."""
        return key in self._results

    def frame(self, key: str) -> FrameMetricResult | None:
        """A per-frame metric's result, or None."""
        result = self.get(key)
        return result if isinstance(result, FrameMetricResult) else None

    def sequence(self, key: str) -> SequenceMetricResult | None:
        """A whole-video metric's result, or None."""
        result = self.get(key)
        return result if isinstance(result, SequenceMetricResult) else None

    def keys(self) -> tuple[str, ...]:
        """The metrics' keys, in the order they were added."""
        return tuple(self._results)

    def __iter__(self):
        return iter(self._results)

    def __bool__(self) -> bool:
        return bool(self._results)

    def copy(self) -> MetricResultSet:
        """A new set with the same results."""
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

    # The first metric present, in the registry's order, sets the frames of
    # this view. Metrics on other frames stay in the MetricResultSet and are
    # left out of the view, so that one metric on its own frames never makes
    # the others' scores disappear.
    #
    # Frames are matched by frame number alone, which identifies the frame.
    # The times are worked out (frame / fps, from the source's or the test
    # video's frame rate, by backend) and can differ by rounding for the
    # same frames. Don't require them to be identical: on a saved 151,919-frame
    # film, every metric but VMAF had VMAF's frame numbers but times up to
    # 3.3e-7 s apart, and would be lost from the CSV export, the "scored
    # frames" text and Video Compare. The view uses the first metric's
    # times.
    reference = present[0]
    aligned = [result for result in present if np.array_equal(reference.frame, result.frame)]
    return FrameScores(reference.frame, reference.time, metrics={
        result.key: result.values for result in aligned
    })


def sampled_every(result: FrameMetricResult, step: int) -> FrameMetricResult:
    """`result` with only every `step`-th frame (0, step, 2*step...): the
    frames a subsampled request covers, as libvmaf's n_subsample scores
    them. The result itself when it holds no others."""
    if step <= 1 or not len(result.frame) or not np.any(result.frame % step):
        return result
    keep = result.frame % step == 0
    return FrameMetricResult(result.key, result.frame[keep], result.time[keep], result.values[keep],
                             result.provenance)


def as_requested(results: MetricResultSet, specs) -> MetricResultSet:
    """Returns `results` with each frame metric on the frames its request
    covers (MetricRequestSpec.coverage), or the set itself when nothing
    changes. XPSNR asked for beside subsampled libvmaf metrics covers their
    frames: in one FFmpeg run, libvmaf's log decides which frames are
    reported. But scored in a run of its own, beside VMAF on the GPU, it
    comes back for every frame, and kept that way it would fall off the
    shared frames, out of the table and the CSV."""
    steps = {spec.key: spec.coverage.step for spec in specs if spec.coverage is not None}
    changed = False
    conformed = MetricResultSet()
    for key in results:
        result = results.get(key)
        if isinstance(result, FrameMetricResult) and steps.get(key, 1) > 1:
            sampled = sampled_every(result, steps[key])
            changed |= sampled is not result
            result = sampled
        conformed.add(result)
    return conformed if changed else results


def merge_metric_results(existing: MetricResultSet, incoming: MetricResultSet) -> MetricResultSet:
    """Return a replacement-by-key merge; unrelated metric results survive."""
    merged = existing.copy()
    for key in incoming:
        result = incoming.get(key)
        assert result is not None
        merged.add(result)
    return merged
