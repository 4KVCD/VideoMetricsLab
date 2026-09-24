"""Background worker that runs one or more VMAF jobs without blocking the UI."""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_results import MetricResultSet
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import ComparisonResult, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualRunError
from vmaf_app.core.perceptual_vship import apply_vship_cpu_fallback
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError, auto_threads, run_resample_test, run_vmaf


class _TaskCancelToken:
    """Cancel one job's sibling tasks without cancelling the entire queue."""

    def __init__(self, run_cancel: threading.Event) -> None:
        self._run_cancel = run_cancel
        self._job_cancel = threading.Event()

    def is_set(self) -> bool:
        return self._run_cancel.is_set() or self._job_cancel.is_set()

    def cancel_job(self) -> None:
        self._job_cancel.set()


@dataclass
class VmafJob:
    source_info: VideoInfo
    distorted_info: VideoInfo
    options: VmafOptions
    label: str
    # Overrides the result's `distorted` identity -- see run_vmaf's
    # result_distorted_path param. None means "just use distorted_info.path"
    # (the normal case).
    result_distorted_path: Path | None = None
    # Selection lives with the job/request, not VmafOptions: standalone
    # metrics are not FFmpeg adapter configuration.
    metric_keys: tuple[str, ...] | None = None
    metric_backends: dict[str, str] = field(default_factory=dict)
    # What the row already shows for this exact recipe, and which of its
    # metrics can stand as they are. A backend group whose metrics are all
    # in cached_metrics is not run; the rest are merged back into the
    # result, so adding SSIMULACRA2 to a row with VMAF runs Vship alone.
    cached_result: ComparisonResult | None = None
    cached_metrics: MetricResultSet | None = None


#: Two, because a third buys nothing. Measured on a 24-core machine over four
#: 1080p comparisons with the app's defaults: 22.6s at one at a time, 14.4s at
#: two, 14.5s at three. Each job already asks libvmaf for every core, and every
#: extra one adds another decode reading from the same disk, so past two they
#: contend rather than overlap.
MAX_PARALLEL_JOBS = 2


class VmafWorker(QThread):
    job_started = Signal(int, str)          # job_index, label
    progress = Signal(int, int, int, float) # job_index, current_frame, total_frames, fps
    status = Signal(int, str)               # job_index, status text
    job_finished = Signal(int, object)      # job_index, ComparisonResult
    job_failed = Signal(int, str, str)      # job_index, message, stderr_tail
    # One metric group failed, the other finished: the finished metrics are
    # the result (job_index, ComparisonResult, message, stderr_tail).
    job_partially_failed = Signal(int, object, str, str)
    cancelled = Signal()
    all_finished = Signal()

    def __init__(self, jobs: list[VmafJob], parallel_jobs: int = 1, parent=None):
        super().__init__(parent)
        self._jobs = jobs
        self._parallel_jobs = max(1, min(int(parallel_jobs), MAX_PARALLEL_JOBS))
        self._cancel_event = threading.Event()
        # One handle per running job rather than one for the worker: pausing
        # or cancelling has to reach every ffmpeg that is currently up, and a
        # single handle can only ever address one pid.
        self._handles: dict[int, ProcessHandle] = {}
        self._handles_lock = threading.Lock()
        # How many lanes may hold a job at once. Changeable mid-run: a queue
        # of feature-length videos is exactly when someone notices their CPU
        # is idle, and being told to restart the batch to act on that would
        # be useless.
        self._slots = threading.Condition()
        self._active = 0
        self._paused = False
        self._cancellation_reported = False
        # How many lanes this run actually has -- set by run(), and never
        # more than there are jobs. It caps the share of the machine each
        # job is planned for: a single video runs alone whatever the
        # setting says, and should not be planned for half a CPU.
        self._lane_count = 1

    # ------------------------------------------------------------- controls
    def cancel(self) -> None:
        # The flag is set and the handles collected under one lock, so a lane
        # claiming a handle either appears in this snapshot (and is
        # terminated) or sees the flag and never starts. Setting the flag
        # outside the lock left a window in which a lane had passed its
        # cancellation check but had not yet registered its handle, so cancel
        # found nothing to kill and ffmpeg started anyway.
        with self._handles_lock:
            self._cancel_event.set()
            handles = list(self._handles.values())
        # Wakes up a paused process too, rather than only relying on the
        # (blocked, since a paused process has no more output) reader loop
        # to notice the cancel flag on its own -- see ProcessHandle.terminate.
        for handle in handles:
            handle.terminate()
        with self._slots:
            self._slots.notify_all()  # release any lane waiting for a slot

    def pause(self) -> None:
        # Applied while holding the lock, so it cannot interleave with a lane
        # claiming a handle or with resume(). Doing the fan-out afterwards
        # let a resume land between "record paused" and "pause the handle",
        # leaving that one lane suspended for good while the UI said the run
        # had resumed.
        with self._handles_lock:
            self._paused = True
            for handle in self._handles.values():
                handle.pause()

    def resume(self) -> None:
        with self._handles_lock:
            self._paused = False
            for handle in self._handles.values():
                handle.resume()

    @property
    def parallel_jobs(self) -> int:
        with self._slots:
            return self._parallel_jobs

    def set_parallel_jobs(self, count: int) -> None:
        """Changes how many videos may be scored at once, while running.

        Raising it lets a waiting lane pick up the next video immediately.
        Lowering it never interrupts a video that has already started -- it
        just stops another from beginning until enough have finished.
        """
        with self._slots:
            self._parallel_jobs = max(1, min(int(count), MAX_PARALLEL_JOBS))
            self._slots.notify_all()

    def _acquire_slot(self) -> bool:
        """Waits for room to run a job. False if the run was cancelled."""
        with self._slots:
            while self._active >= self._parallel_jobs:
                if self._cancel_event.is_set():
                    return False
                # A timeout rather than a pure wait: cancellation is
                # signalled through an Event that cannot notify this
                # condition, so the wait has to come up for air.
                self._slots.wait(0.1)
            if self._cancel_event.is_set():
                return False
            self._active += 1
            return True

    def _release_slot(self) -> None:
        with self._slots:
            self._active -= 1
            self._slots.notify_all()

    @property
    def is_paused(self) -> bool:
        with self._handles_lock:
            return self._paused

    def _live_handles(self) -> list[ProcessHandle]:
        with self._handles_lock:
            return list(self._handles.values())

    def _claim_handle(self, index: int) -> ProcessHandle | None:
        """A handle for one job, or None if the run has been cancelled.

        Everything happens under the one lock cancel/pause/resume also take,
        so this is atomic with respect to all three. A job that starts while
        the run is paused comes up paused; a job that tries to start after
        cancel never starts at all.
        """
        handle = ProcessHandle()
        with self._handles_lock:
            if self._cancel_event.is_set():
                return None
            self._handles[index] = handle
            if self._paused:
                handle.pause()
        return handle

    def _release_handle(self, index: int) -> None:
        with self._handles_lock:
            self._handles.pop(index, None)

    def _share_cores(self, options: VmafOptions) -> VmafOptions:
        """Fills in "Auto" libvmaf threads with this job's share of the CPU.

        Decided as the job starts, from the lane count in force at that
        moment: a run that was widened to two lanes part-way through gives
        every video started after that half the cores, while the one already
        running keeps what it was launched with (a thread count cannot be
        changed under a live ffmpeg). An explicit thread count is the
        user's, and passes through untouched. n_threads never reaches the
        cache key, so this changes how fast the result arrives, not what it
        is.
        """
        if options.n_threads > 0:
            return options
        concurrent = min(self.parallel_jobs, self._lane_count)
        if concurrent <= 1:
            return options  # the runner resolves Auto to every core itself
        return replace(options, n_threads=auto_threads(concurrent))

    def _report_cancelled_once(self) -> None:
        """`cancelled` means the run stopped, not that a job did -- with
        several jobs in flight they all raise Cancelled together."""
        with self._handles_lock:
            if self._cancellation_reported:
                return
            self._cancellation_reported = True
        self.cancelled.emit()

    def _execute_plan(self, index: int, job: VmafJob, options: VmafOptions,
                      handle: ProcessHandle):
        """Run independent backends together and publish one result.

        Returns (result, failure): failure is None when every task finished,
        or (message, stderr tail) when one metric group failed and the other
        finished -- the finished metrics are still the result. Raises when
        nothing was produced.
        """
        request = analysis_request_from_vmaf_options(
            options, job.metric_keys, job.metric_backends,
        )
        cached = job.cached_metrics if job.cached_result is not None and job.cached_metrics else None
        plan = build_execution_plan(request, cached)
        token = _TaskCancelToken(self._cancel_event)
        task_results: dict[str, object] = {}
        task_errors: list[tuple[object, Exception]] = []
        task_lock = threading.Lock()
        task_progress: dict[str, tuple[int, int, float]] = {}

        def report_progress(backend: str, cur: int, total: int, fps: float) -> None:
            if len(plan.tasks) == 1:
                self.progress.emit(index, cur, total, fps)
                return
            # Each pass may cover a different number of frames. Until both
            # are done, the slower completion fraction owns job progress.
            with task_lock:
                task_progress[backend] = (cur, total, fps)
                known_total = max((value[1] for value in task_progress.values()), default=0)
                fractions = [
                    1.0 if task.backend_id in task_results else
                    min(0.999, value[0] / value[1]) if value[1] > 0 else 0.0
                    for task in plan.tasks
                    for value in [task_progress.get(task.backend_id, (0, 0, 0.0))]
                ]
                fraction = min(fractions)
                remaining = []
                for task in plan.tasks:
                    if task.backend_id in task_results:
                        remaining.append(0.0)
                        continue
                    value = task_progress.get(task.backend_id)
                    if value is None or value[2] <= 0 or value[0] >= value[1]:
                        remaining = []
                        break
                    remaining.append(max(0, value[1] - value[0]) / value[2])
                overall_cur = round(known_total * fraction)
                if len(task_results) < len(plan.tasks) and known_total > 0:
                    overall_cur = min(overall_cur, known_total - 1)
                overall_fps = (
                    (known_total - overall_cur) / max(remaining)
                    if remaining and max(remaining) > 0 else 0.0
                )
            self.progress.emit(index, overall_cur, known_total, overall_fps)

        def execute_task(task) -> object:
            if task.backend_id == "ffmpeg":
                task_options = replace(options)
                for key in options.requested_metrics():
                    task_options.set_metric_enabled(key, key in task.metric_keys)
                if options.resample_test is not None:
                    return run_resample_test(
                        job.source_info, task_options,
                        on_progress=lambda cur, tot, fps: report_progress(task.backend_id, cur, tot, fps),
                        on_status=lambda msg: self.status.emit(index, msg),
                        cancel_event=token, process_handle=handle,
                    )
                return run_vmaf(
                    job.source_info, job.distorted_info, task_options,
                    on_progress=lambda cur, tot, fps: report_progress(task.backend_id, cur, tot, fps),
                    on_status=lambda msg: self.status.emit(index, msg),
                    cancel_event=token, process_handle=handle,
                    result_distorted_path=job.result_distorted_path,
                )
            if task.backend_id == "perceptual":
                return apply_vship_cpu_fallback(
                    job.source_info, job.distorted_info, request, task.requested_specs,
                    on_progress=lambda cur, tot, fps: report_progress(task.backend_id, cur, tot, fps),
                    on_status=lambda msg: self.status.emit(index, msg),
                    cancel_event=token, process_handle=handle,
                )
            raise VmafRunError(f"Unknown metric backend: {task.backend_id}")

        def run_task(task) -> None:
            try:
                output = execute_task(task)
                with task_lock:
                    task_results[task.backend_id] = output
                    last_progress = task_progress.get(task.backend_id, (1, 1, 0.0))
                if len(plan.tasks) > 1:
                    report_progress(task.backend_id, *last_progress)
            except Exception as error:
                # The sibling is left to finish: a SSIMULACRA2/Butteraugli
                # failure (an unsupported input, a tool error) used to cancel
                # the libvmaf pass and discard VMAF/PSNR/SSIM/XPSNR with it.
                with task_lock:
                    task_errors.append((task, error))

        if len(plan.tasks) > 1:
            threads = [
                threading.Thread(target=run_task, args=(task,),
                                 name=f"metric-{task.backend_id}-{index}", daemon=True)
                for task in plan.tasks
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for task in plan.tasks:
                run_task(task)

        if self._cancel_event.is_set():
            raise Cancelled("Cancelled by user")
        if task_errors and not task_results and cached is None:
            raise task_errors[0][1]

        result = task_results.get("ffmpeg")
        if result is None and cached is not None:
            # FFmpeg's metrics are all saved: the saved run is the base, so
            # its crops, frame table and file info carry over. A shallow
            # copy is enough -- merge_metric_results replaces the metric set
            # and frame table rather than editing them, and the row's own
            # result object must not change under the UI thread.
            result = copy.copy(job.cached_result)
        perceptual = task_results.get("perceptual")
        if perceptual is not None:
            if result is None:
                from vmaf_app.core.models import ComparisonResult, FrameScores
                result = ComparisonResult(
                    source=job.source_info.path,
                    distorted=job.result_distorted_path or job.distorted_info.path,
                    frames=FrameScores.empty(), fps=job.source_info.fps, model="",
                    source_crop=perceptual.source_crop, distorted_crop=perceptual.distorted_crop,
                    source_info=job.source_info, distorted_info=job.distorted_info,
                    scale_direction=options.scale_direction, scale_algorithm=options.scale_algorithm,
                    compared_frame_count=perceptual.compared_frame_count,
                )
            combined = result.metric_results.copy()
            for key in perceptual.metrics:
                value = perceptual.metrics.get(key)
                assert value is not None
                combined.add(value)
            result.merge_metric_results(combined)
        if result is None:
            if task_errors:
                raise task_errors[0][1]
            raise VmafRunError("No executable metric task was planned.")
        if cached is not None:
            # Saved metrics fill only what this run did not calculate: a
            # fresh score always wins over the saved one.
            carried = MetricResultSet(
                value for key in cached
                if not result.has_metric(key) and (value := cached.get(key)) is not None
            )
            if carried:
                result.merge_metric_results(carried)
        if not task_errors:
            return result, None
        task, error = task_errors[0]
        labels = ", ".join(metric_definition(key).label for key in task.metric_keys)
        return result, (f"{labels} failed: {error}", getattr(error, "stderr_tail", "") or "")

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        # Jobs are pulled from one shared iterator rather than dealt out in
        # advance: they finish at wildly different speeds (a 90-minute
        # feature next to a 20-second clip), and a fixed split would leave a
        # lane idle while another still had work queued.
        pending = iter(range(len(self._jobs)))
        # As many lanes as the ceiling allows, not as many as the current
        # setting: a lane blocks on a slot until there is room for it, which
        # is what lets the count be raised while the run is going. A blocked
        # lane costs a parked thread and nothing else.
        lane_count = min(MAX_PARALLEL_JOBS, len(self._jobs))
        self._lane_count = lane_count
        if lane_count <= 1:
            self._run_jobs(pending)
        else:
            lanes = [
                threading.Thread(
                    target=self._run_jobs, args=(pending,),
                    name=f"vmaf-lane-{n}", daemon=True,
                )
                for n in range(lane_count)
            ]
            for lane in lanes:
                lane.start()
            for lane in lanes:
                lane.join()

        if self._cancel_event.is_set():
            self._report_cancelled_once()
        self.all_finished.emit()

    def _run_jobs(self, pending) -> None:
        """Runs jobs from `pending` until it is empty or the run is cancelled.

        `pending` is shared between lanes; next() on an iterator is atomic
        under the GIL, so each index goes to exactly one lane.
        """
        for i in pending:
            if self._cancel_event.is_set():
                break
            if not self._acquire_slot():
                break
            handle = self._claim_handle(i)
            if handle is None:
                self._release_slot()
                break
            job = self._jobs[i]
            options = self._share_cores(job.options)
            self.job_started.emit(i, job.label)
            try:
                result, failure = self._execute_plan(i, job, options, handle)
            except (Cancelled, PerceptualCancelled):
                self._report_cancelled_once()
                break
            except (VmafRunError, PerceptualRunError) as e:
                self.job_failed.emit(i, str(e), e.stderr_tail)
                continue
            except Exception as e:
                self.job_failed.emit(i, str(e), "")
                continue
            finally:
                self._release_handle(i)
                self._release_slot()
            if failure is None:
                self.job_finished.emit(i, result)
            else:
                self.job_partially_failed.emit(i, result, *failure)
