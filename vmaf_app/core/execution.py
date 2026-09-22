"""Backend-neutral grouping of metric requests into executable tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec

if TYPE_CHECKING:
    from vmaf_app.core.metric_results import MetricResultSet


@dataclass(frozen=True, slots=True)
class MetricTask:
    backend_id: str
    metric_keys: tuple[str, ...]
    requested_specs: tuple[MetricRequestSpec, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    requested_metrics: tuple[str, ...]
    tasks: tuple[MetricTask, ...]


@dataclass(slots=True)
class TaskExecutionResult:
    metric_results: MetricResultSet | None
    error: str | None = None


def build_execution_plan(
    request: AnalysisRequest,
    cached_results: MetricResultSet | None = None,
) -> ExecutionPlan:
    """Group requested metrics by backend while preserving backend efficiency.

    A backend group is skipped only when every metric in that group is already
    cached. If one is missing, the whole group remains together so coupled
    implementations such as the current FFmpeg/libvmaf filtergraph still get
    one efficient pass. Future backends can coexist without planner changes.
    """
    if not request.metrics:
        raise ValueError("Select at least one metric to calculate.")

    groups: dict[str, list[MetricRequestSpec]] = {}
    for spec in request.metrics:
        groups.setdefault(spec.backend_id, []).append(spec)

    tasks: list[MetricTask] = []
    for backend_id, specs in groups.items():
        if cached_results is not None and all(cached_results.has(spec.key) for spec in specs):
            continue
        task_specs = tuple(specs)
        tasks.append(MetricTask(
            backend_id=backend_id,
            metric_keys=tuple(spec.key for spec in task_specs),
            requested_specs=task_specs,
        ))

    return ExecutionPlan(request.requested_metrics, tuple(tasks))
