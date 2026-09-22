"""Generic scientific requests shared by planning, caching, and metric backends."""
from __future__ import annotations

from dataclasses import dataclass

from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.models import GpuVendor


@dataclass(frozen=True, slots=True)
class FrameCoverage:
    """Which source frames a frame metric represents."""

    mode: str
    step: int = 1


@dataclass(frozen=True, slots=True)
class MetricRequestSpec:
    """Scientific identity plus execution routing for one metric."""

    key: str
    backend_id: str
    parameters: tuple[tuple[str, object], ...]
    coverage: FrameCoverage | None
    implementation_compatibility_id: str

    def identity_dict(self) -> dict:
        """Cache identity, deliberately independent of execution routing."""
        return {
            "key": self.key,
            "parameters": dict(self.parameters),
            "coverage": None if self.coverage is None else {
                "mode": self.coverage.mode,
                "step": self.coverage.step,
            },
            "implementation_compatibility_id": self.implementation_compatibility_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPreferences:
    """Performance choices that must never enter scientific/cache identity."""

    gpu_decode: bool
    gpu_vendor: GpuVendor
    n_threads: int


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Backend-neutral request for one source/distorted comparison.

    The recipe defines the pictures being compared. Metric specs define the
    requested measurements and their implementation-compatible identities.
    Execution preferences only affect how quickly the request is fulfilled.
    """

    recipe: ComparisonRecipe
    metrics: tuple[MetricRequestSpec, ...]
    execution: ExecutionPreferences

    def __post_init__(self) -> None:
        keys = [spec.key for spec in self.metrics]
        if len(keys) != len(set(keys)):
            raise ValueError("an analysis request cannot contain duplicate metric keys")

    @property
    def requested_metrics(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.metrics)

    def metric(self, key: str) -> MetricRequestSpec | None:
        return next((spec for spec in self.metrics if spec.key == key), None)
