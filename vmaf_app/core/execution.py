"""Metric request identity and the small, explicit execution planner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vmaf_app.core.models import GpuVendor, VmafOptions

if TYPE_CHECKING:
    from pathlib import Path

    from vmaf_app.core.metric_results import MetricResultSet
    from vmaf_app.core.models import VideoInfo, VmafRunResult
    from vmaf_app.core.process_control import ProcessHandle


@dataclass(frozen=True, slots=True)
class FrameCoverage:
    mode: str
    step: int = 1


@dataclass(frozen=True, slots=True)
class MetricRequestSpec:
    key: str
    parameters: tuple[tuple[str, object], ...]
    coverage: FrameCoverage | None
    implementation_compatibility_id: str

    def identity_dict(self) -> dict:
        return {
            "key": self.key,
            "parameters": dict(self.parameters),
            "coverage": None if self.coverage is None else {
                "mode": self.coverage.mode, "step": self.coverage.step,
            },
            "implementation_compatibility_id": self.implementation_compatibility_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPreferences:
    gpu_decode: bool
    gpu_vendor: GpuVendor
    n_threads: int

    @classmethod
    def from_vmaf_options(cls, options: VmafOptions) -> ExecutionPreferences:
        return cls(options.gpu_decode, options.gpu_vendor, options.n_threads)


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


def metric_request_specs(options: VmafOptions) -> tuple[MetricRequestSpec, ...]:
    """Scientific request identity for each established metric.

    XPSNR alone intentionally keeps full coverage even when the libvmaf
    subsample setting is non-one. In a mixed FFmpeg run it follows the
    sampled timeline emitted by libvmaf; this long-standing behavior is part
    of its cache identity.
    """
    requested = options.requested_metrics()
    libvmaf_requested = any(key != "xpsnr" for key in requested)
    specs = []
    for key in requested:
        coverage = FrameCoverage(
            "sampled" if libvmaf_requested and options.n_subsample > 1 else "full",
            options.n_subsample if libvmaf_requested and options.n_subsample > 1 else 1,
        )
        parameters: tuple[tuple[str, object], ...] = ()
        if key in {"vmaf", "vmaf_neg"}:
            # ``model`` is resolved by the runner for Auto selections, while
            # model_choice distinguishes the bundled variants before that
            # point. A custom path gets a cheap size/mtime identity so editing
            # the file in place cannot reuse scores from its prior contents.
            model = options.model or options.model_choice
            custom_identity = ""
            if options.custom_model_path:
                path = Path(options.custom_model_path).resolve()
                try:
                    stat = path.stat()
                    custom_identity = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
                except OSError:
                    custom_identity = str(path)
            parameters = (("model", model), ("model_choice", options.model_choice),
                          ("custom_model", custom_identity))
        # The current filter implementations are scientifically stable under
        # these compatibility IDs; execution preferences intentionally stay
        # outside this identity.
        compatibility = "ffmpeg-xpsnr-v1" if key == "xpsnr" else "ffmpeg-libvmaf-v1"
        specs.append(MetricRequestSpec(key, parameters, coverage, compatibility))
    return tuple(specs)


def build_execution_plan(
    options: VmafOptions, cached_results: MetricResultSet | None = None,
) -> ExecutionPlan:
    requested = options.requested_metrics()
    if not requested:
        raise ValueError("Select at least one metric to calculate.")
    specs = metric_request_specs(options)
    # A cache adapter only returns scientifically compatible entries. Current
    # metrics remain one coupled FFmpeg pass, so either all are satisfied or
    # the original grouped calculation is retained.
    if cached_results is not None and all(cached_results.has(key) for key in requested):
        return ExecutionPlan(requested, ())
    return ExecutionPlan(requested, (
        MetricTask("legacy_ffmpeg", requested, specs),
    ))


def execute_plan(
    plan: ExecutionPlan,
    source_info: VideoInfo,
    distorted_info: VideoInfo | None,
    options: VmafOptions,
    *,
    on_progress=None,
    on_status=None,
    cancel_event=None,
    process_handle: ProcessHandle | None = None,
    result_distorted_path: Path | None = None,
) -> VmafRunResult:
    """Execute current tasks sequentially; future tasks can be added safely.

    The only production backend today is the existing grouped FFmpeg runner.
    Imports are local so planning stays headless and pure for unit tests.
    """
    if not plan.tasks:
        raise ValueError("Execution plan contains no task to execute.")
    from vmaf_app.core.vmaf_runner import run_resample_test, run_vmaf

    result = None
    for task in plan.tasks:
        if task.backend_id != "legacy_ffmpeg":
            raise ValueError(f"Unsupported metric backend {task.backend_id!r}")
        if options.resample_test is not None:
            result = run_resample_test(
                source_info, options, on_progress=on_progress, on_status=on_status,
                cancel_event=cancel_event, process_handle=process_handle,
            )
        else:
            assert distorted_info is not None
            result = run_vmaf(
                source_info, distorted_info, options, on_progress=on_progress, on_status=on_status,
                cancel_event=cancel_event, process_handle=process_handle,
                result_distorted_path=result_distorted_path,
            )
    assert result is not None
    return result
