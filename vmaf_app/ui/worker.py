"""Background worker that runs one or more VMAF jobs without blocking the UI."""
from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.cvvdp import CvvdpSettings
from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_results import MetricResultSet
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import ComparisonResult, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualRunError
from vmaf_app.core.perceptual_vship import GPU_ONLY_METRICS, GPU_WAIT_MESSAGE, apply_vship_cpu_fallback
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
    # The display CVVDP models for this video; None is the built-in default.
    cvvdp: CvvdpSettings | None = None
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

#: Videos in progress at once: one per CPU lane, plus the one on the GPU. A
#: video's CPU and GPU halves run in separate queues, so one half can run
#: ahead of the other -- but not without limit: each video in progress has a
#: line of its own in the window, and the list is still worked through in
#: order.
MAX_VIDEOS_IN_FLIGHT = MAX_PARALLEL_JOBS + 1

_CPU, _GPU = "cpu", "gpu"


class VmafWorker(QThread):
    job_started = Signal(int, str)          # job_index, label
    progress = Signal(int, int, int, float) # job_index, current_frame, total_frames, fps
    # For a video scored in two halves (FFmpeg metrics, SSIMULACRA2/
    # Butteraugli/CVVDP) side by side: job_index, then per half
    # (metric labels, current, total, fps, "waiting"|"starting"|"running"|"done").
    # Sent before each `progress` of such a video.
    halves = Signal(int, object)
    # Backend-aware progress for the status UI.  Unlike `progress`, this
    # keeps CPU and perceptual work on their own timelines, which matters
    # when the perceptual backend contains several serialized GPU passes.
    # Each item is a dict containing backend, metric_keys, current, total,
    # fps, state, and an optional phase (GPU metric number/name).
    task_progress = Signal(int, object)
    # Once, as the run starts: {job_index: [("cpu" or "gpu", passes), ...]},
    # each video's halves in task order and how many passes over its frames
    # each makes (a GPU half, one per metric). The queue ETA needs it for
    # videos that have not started yet.
    planned = Signal(object)
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
        #
        # Work is scheduled per half, not per video: the FFmpeg metrics (and
        # SSIMULACRA2/Butteraugli set to CPU) queue for the CPU lanes, the
        # Vship metrics for the one GPU, each queue in list order. A lane
        # used to take a whole video: with two lanes, the second went to
        # the next video even when all it needed was the GPU -- which it
        # then took ahead of the first video, while that video's GPU half
        # waited and the third video's CPU half never started.
        self._sched = threading.Condition()
        self._busy = {_CPU: 0, _GPU: 0}
        self._queues: dict[str, list] = {_CPU: [], _GPU: []}
        self._in_flight: set[int] = set()
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
        with self._sched:
            self._sched.notify_all()  # release any lane waiting for room

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
        with self._sched:
            return self._parallel_jobs

    def set_parallel_jobs(self, count: int) -> None:
        """Changes how many videos' CPU metrics may run at once, while running.

        Raising it lets a waiting lane pick up the next video immediately.
        Lowering it never interrupts work that has already started -- it
        just stops more from beginning until enough have finished.
        """
        with self._sched:
            self._parallel_jobs = max(1, min(int(count), MAX_PARALLEL_JOBS))
            self._sched.notify_all()

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

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        runs: list[_JobRun] = []
        for index, job in enumerate(self._jobs):
            try:
                runs.append(_JobRun(self, index, job))
            except Exception as error:  # a request that cannot be built fails its own video
                self.job_started.emit(index, job.label)
                self.job_failed.emit(index, str(error), getattr(error, "stderr_tail", "") or "")
        self.planned.emit({
            run.index: [(run.pool_of(task), len(task.requested_specs) if run.pool_of(task) == _GPU else 1)
                        for task in run.plan.tasks]
            for run in runs
        })
        for run in runs:
            if not run.plan.tasks:
                # Everything is already saved: the saved result is the result.
                if run.begin():
                    self._finish(run)
                continue
            for task in run.plan.tasks:
                self._queues[run.pool_of(task)].append((run, task))

        # A lane per CPU slot the setting may ever allow, so the count can be
        # raised mid-run: a lane with no room waits, which costs a parked
        # thread and nothing else. The GPU has one lane.
        cpu_lanes = min(MAX_PARALLEL_JOBS, len(self._queues[_CPU]))
        self._lane_count = max(1, cpu_lanes)
        lanes = [
            threading.Thread(target=self._drain_queue, args=(_CPU,), name=f"vmaf-cpu-{n}", daemon=True)
            for n in range(cpu_lanes)
        ]
        if self._queues[_GPU]:
            lanes.append(threading.Thread(target=self._drain_queue, args=(_GPU,), name="vmaf-gpu", daemon=True))
        # The last lane runs on this thread: one thread fewer, and a run with
        # a single lane stays on the worker thread as it always did.
        for lane in lanes[:-1]:
            lane.start()
        if lanes:
            lanes[-1].run()
        for lane in lanes[:-1]:
            lane.join()
        for run in runs:  # a video cancelled part-way still holds its handle
            self._release_handle(run.index)

        if self._cancel_event.is_set():
            self._report_cancelled_once()
        self.all_finished.emit()

    def _next_task(self, pool: str):
        """The next half this pool may run, in list order, or None once the
        queue is empty or the run was cancelled. Waits for room."""
        queue = self._queues[pool]
        with self._sched:
            while True:
                if self._cancel_event.is_set() or not queue:
                    return None
                capacity = self._parallel_jobs if pool == _CPU else 1
                if self._busy[pool] < capacity:
                    # The first half whose video is already in progress, or
                    # that may start a new one. Later ones may pass an earlier
                    # video only when that one cannot start yet, so the
                    # videos in progress always have a way to finish.
                    for position, (run, task) in enumerate(queue):
                        if run.started or len(self._in_flight) < MAX_VIDEOS_IN_FLIGHT:
                            del queue[position]
                            self._busy[pool] += 1
                            run.admitted.add(task.backend_id)
                            if not run.started:
                                run.started = True
                                self._in_flight.add(run.index)
                            return run, task
                # A timeout rather than a pure wait: cancellation is
                # signalled through an Event that cannot notify this
                # condition, so the wait has to come up for air.
                self._sched.wait(0.1)

    def _drain_queue(self, pool: str) -> None:
        while (picked := self._next_task(pool)) is not None:
            run, task = picked
            complete = False
            try:
                if run.begin():
                    complete = run.run_task(task)
            finally:
                with self._sched:
                    self._busy[pool] -= 1
                    self._sched.notify_all()
            if complete:
                self._finish(run)
                with self._sched:
                    self._in_flight.discard(run.index)
                    self._sched.notify_all()

    def _finish(self, run: _JobRun) -> None:
        index = run.index
        try:
            result, failure = run.finish()
        except (Cancelled, PerceptualCancelled):
            self._report_cancelled_once()
            return
        except (VmafRunError, PerceptualRunError) as e:
            self.job_failed.emit(index, str(e), e.stderr_tail)
            return
        except Exception as e:
            self.job_failed.emit(index, str(e), "")
            return
        finally:
            self._release_handle(index)
        if failure is None:
            self.job_finished.emit(index, result)
        else:
            self.job_partially_failed.emit(index, result, *failure)


class _JobRun:
    """One video: its halves, which the worker's CPU and GPU lanes run, and
    the result they make together."""

    def __init__(self, worker: VmafWorker, index: int, job: VmafJob) -> None:
        self.worker, self.index, self.job = worker, index, job
        self.request = analysis_request_from_vmaf_options(
            job.options, job.metric_keys, job.metric_backends, job.cvvdp,
        )
        self.cached = job.cached_metrics if job.cached_result is not None and job.cached_metrics else None
        self.plan = build_execution_plan(self.request, self.cached)
        self.token = _TaskCancelToken(worker._cancel_event)
        self.options = job.options
        self.handle: ProcessHandle | None = None
        # Under the worker's scheduler lock: whether the video is in progress,
        # and which of its halves a lane has taken.
        self.started = False
        self.admitted: set[str] = set()
        self.lock = threading.Lock()
        self._begun = False
        self.task_results: dict[str, object] = {}
        self.task_errors: list[tuple[object, Exception]] = []
        self.task_progress: dict[str, tuple[int, int, float]] = {}
        self.task_phases: dict[str, tuple[int, int, str]] = {}
        # Halves not running yet, and what they wait for: "GPU" or "CPU".
        self.task_waiting: dict[str, str] = {}
        # Each half's latest status message: what a half not yet reporting
        # figures is doing (black-bar detection, say).
        self.task_steps: dict[str, str] = {}
        self._finished_tasks = 0

    def pool_of(self, task) -> str:
        """The GPU queue for a half with any metric on the GPU (a GPU pass
        that fails is retried on the CPU inside it); the CPU queue otherwise."""
        if task.backend_id != "perceptual":
            return _CPU
        on_gpu = any(
            spec.key in GPU_ONLY_METRICS or self.request.execution.perceptual_backend(spec.key) == "gpu"
            for spec in task.requested_specs
        )
        return _GPU if on_gpu else _CPU

    def begin(self) -> bool:
        """Starts the video when its first half is taken: its process
        handle, its share of the cores, and job_started. False if the run
        was cancelled first. Only the first call does anything."""
        with self.lock:
            if self._begun:
                return self.handle is not None
            self._begun = True
            self.handle = self.worker._claim_handle(self.index)
            if self.handle is None:
                return False
            self.options = self.worker._share_cores(self.job.options)
            self.worker.job_started.emit(self.index, self.job.label)
            if len(self.plan.tasks) > 1:
                admitted = set(self.admitted)
                for task in self.plan.tasks:
                    if task.backend_id not in admitted:
                        self.task_waiting[task.backend_id] = "GPU" if self.pool_of(task) == _GPU else "CPU"
                snapshot = self.task_snapshots()
        if len(self.plan.tasks) > 1:
            self.worker.task_progress.emit(self.index, snapshot)
        return True

    def halves(self) -> list[tuple[str, int, int, float, str]]:
        """Each half's own progress; called with self.lock held."""
        found = []
        for task in self.plan.tasks:
            cur, total, fps = self.task_progress.get(task.backend_id, (0, 0, 0.0))
            labels = "/".join(metric_definition(key).label for key in task.metric_keys)
            found.append((labels, cur, total, fps, self._state(task)))
        return found

    def _state(self, task) -> str:
        return ("done" if task.backend_id in self.task_results else
                "waiting" if task.backend_id in self.task_waiting else
                "running" if task.backend_id in self.task_progress else "starting")

    def task_snapshots(self) -> list[dict[str, object]]:
        """Progress without collapsing unlike backends together; called with self.lock held."""
        found = []
        for task in self.plan.tasks:
            cur, total, fps = self.task_progress.get(task.backend_id, (0, 0, 0.0))
            found.append({
                "backend": task.backend_id,
                "metric_keys": task.metric_keys,
                "current": cur,
                "total": total,
                "fps": fps,
                "state": self._state(task),
                "waiting_for": self.task_waiting.get(task.backend_id),
                "step": self.task_steps.get(task.backend_id, ""),
                "phase": self.task_phases.get(task.backend_id),
            })
        return found

    def report_status(self, backend: str, message: str) -> None:
        with self.lock:
            self.task_steps[backend] = message
            if len(self.plan.tasks) > 1 and message == GPU_WAIT_MESSAGE:
                self.task_waiting[backend] = "GPU"
            phase = re.match(r"^GPU metric (\d+)/(\d+): (.+)$", message)
            if phase:
                self.task_phases[backend] = (int(phase.group(1)), int(phase.group(2)), phase.group(3))
            elif "on CPU" in message or "using CPU" in message:
                # A GPU pass has handed work to the CPU fallback (or a
                # planned CPU perceptual pass has begun); don't leave a
                # stale GPU metric number on the status line.
                self.task_phases.pop(backend, None)
            snapshot = self.task_snapshots()
        self.worker.task_progress.emit(self.index, snapshot)
        self.worker.status.emit(self.index, message)

    def report_progress(self, backend: str, cur: int, total: int, fps: float) -> None:
        worker, index, tasks = self.worker, self.index, self.plan.tasks
        if len(tasks) == 1:
            with self.lock:
                self.task_progress[backend] = (cur, total, fps)
                snapshot = self.task_snapshots()
            worker.task_progress.emit(index, snapshot)
            worker.progress.emit(index, cur, total, fps)
            return
        # Each pass may cover a different number of frames. Until both
        # are done, the slower completion fraction owns job progress.
        with self.lock:
            self.task_waiting.pop(backend, None)
            self.task_progress[backend] = (cur, total, fps)
            known_total = max((value[1] for value in self.task_progress.values()), default=0)
            fractions = [
                1.0 if task.backend_id in self.task_results else
                min(0.999, value[0] / value[1]) if value[1] > 0 else 0.0
                for task in tasks
                for value in [self.task_progress.get(task.backend_id, (0, 0, 0.0))]
            ]
            fraction = min(fractions)
            remaining = []
            for task in tasks:
                if task.backend_id in self.task_results:
                    remaining.append(0.0)
                    continue
                value = self.task_progress.get(task.backend_id)
                if value is None or value[2] <= 0 or value[0] >= value[1]:
                    remaining = []
                    break
                remaining.append(max(0, value[1] - value[0]) / value[2])
            overall_cur = round(known_total * fraction)
            if len(self.task_results) < len(tasks) and known_total > 0:
                overall_cur = min(overall_cur, known_total - 1)
            overall_fps = (
                (known_total - overall_cur) / max(remaining)
                if remaining and max(remaining) > 0 else 0.0
            )
            each = self.halves()
            snapshot = self.task_snapshots()
        worker.task_progress.emit(index, snapshot)
        worker.halves.emit(index, each)
        worker.progress.emit(index, overall_cur, known_total, overall_fps)

    def execute_task(self, task) -> object:
        job, options = self.job, self.options

        def progress(cur, tot, fps):
            self.report_progress(task.backend_id, cur, tot, fps)

        if task.backend_id == "ffmpeg":
            task_options = replace(options)
            for key in options.requested_metrics():
                task_options.set_metric_enabled(key, key in task.metric_keys)
            if options.resample_test is not None:
                return run_resample_test(
                    job.source_info, task_options,
                    on_progress=progress,
                    on_status=lambda msg: self.report_status(task.backend_id, msg),
                    cancel_event=self.token, process_handle=self.handle,
                )
            return run_vmaf(
                job.source_info, job.distorted_info, task_options,
                on_progress=progress,
                on_status=lambda msg: self.report_status(task.backend_id, msg),
                cancel_event=self.token, process_handle=self.handle,
                result_distorted_path=job.result_distorted_path,
            )
        if task.backend_id == "perceptual":
            return apply_vship_cpu_fallback(
                job.source_info, job.distorted_info, self.request, task.requested_specs,
                on_progress=progress,
                on_status=lambda msg: self.report_status(task.backend_id, msg),
                cancel_event=self.token, process_handle=self.handle,
            )
        raise VmafRunError(f"Unknown metric backend: {task.backend_id}")

    def run_task(self, task) -> bool:
        """Runs one half; True when it was the video's last."""
        with self.lock:
            self.task_waiting.pop(task.backend_id, None)
        try:
            output = self.execute_task(task)
        except Exception as error:
            # The sibling is left to finish: a SSIMULACRA2/Butteraugli
            # failure (an unsupported input, a tool error) used to cancel
            # the libvmaf pass and discard VMAF/PSNR/SSIM/XPSNR with it.
            with self.lock:
                self.task_errors.append((task, error))
        else:
            with self.lock:
                self.task_results[task.backend_id] = output
                last_progress = self.task_progress.get(task.backend_id, (1, 1, 0.0))
            if len(self.plan.tasks) > 1:
                self.report_progress(task.backend_id, *last_progress)
        with self.lock:
            self._finished_tasks += 1
            return self._finished_tasks == len(self.plan.tasks)

    def finish(self):
        """The video's result once every half has run.

        Returns (result, failure): failure is None when every task finished,
        or (message, stderr tail) when one metric group failed and the other
        finished -- the finished metrics are still the result. Raises when
        nothing was produced.
        """
        job, options, cached = self.job, self.options, self.cached
        task_results, task_errors = self.task_results, self.task_errors
        if self.worker._cancel_event.is_set():
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
        # A metric that failed while the rest of its group finished (CVVDP,
        # GPU only, beside SSIMULACRA2/Butteraugli) is reported like a
        # failed group.
        metric_failures = dict(perceptual.failures) if perceptual is not None else {}
        if not task_errors and not metric_failures:
            return result, None
        messages, stderr_tail = [], ""
        for task, error in task_errors:
            labels = ", ".join(metric_definition(key).label for key in task.metric_keys)
            messages.append(f"{labels} failed: {error}")
            stderr_tail = stderr_tail or getattr(error, "stderr_tail", "") or ""
        for key, message in metric_failures.items():
            messages.append(f"{metric_definition(key).label} failed: {message}")
        return result, ("\n".join(messages), stderr_tail)
