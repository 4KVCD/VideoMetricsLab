"""Groups the metrics a request asks for into tasks, one per backend (the
program that calculates them), leaving out what is already saved
(build_execution_plan)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec

if TYPE_CHECKING:
    from vmaf_app.core.metric_results import MetricResultSet


@dataclass(frozen=True, slots=True)
class MetricTask:
    """One backend's metrics for one video: the backend, the metric keys and their request specs."""

    backend_id: str
    metric_keys: tuple[str, ...]
    requested_specs: tuple[MetricRequestSpec, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The metrics asked for, and the tasks that calculate those not already saved."""

    requested_metrics: tuple[str, ...]
    tasks: tuple[MetricTask, ...]


@dataclass(slots=True)
class TaskExecutionResult:
    """A task's results, or its error."""

    metric_results: MetricResultSet | None
    error: str | None = None


#: Backends whose metrics share only the decoding: SSIMULACRA2, Butteraugli
#: and CVVDP each have their own handler and score. Don't keep such a group
#: together: ticking CVVDP on a video with a saved SSIMULACRA2 would
#: calculate SSIMULACRA2 again too, which takes days on the CPU, without the
#: long-CPU-run warning (which counts only metrics without scores).
_INDEPENDENT_BACKENDS = frozenset({"perceptual"})


def build_execution_plan(
    request: AnalysisRequest,
    cached_results: MetricResultSet | None = None,
) -> ExecutionPlan:
    """Returns the plan: the requested metrics grouped into one task per
    backend.

    A backend's task is skipped only when every metric in it is already
    saved. If one is missing, the whole group stays together, so that
    metrics calculated together, such as FFmpeg's in one filter graph, still
    get one efficient pass. The exception is backends whose metrics are
    independent (_INDEPENDENT_BACKENDS): there a saved metric is left out,
    and only the missing ones run.
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
        if cached_results is not None and backend_id in _INDEPENDENT_BACKENDS:
            specs = [spec for spec in specs if not cached_results.has(spec.key)]
        task_specs = tuple(specs)
        tasks.append(MetricTask(
            backend_id=backend_id,
            metric_keys=tuple(spec.key for spec in task_specs),
            requested_specs=task_specs,
        ))

    return ExecutionPlan(request.requested_metrics, tuple(tasks))
