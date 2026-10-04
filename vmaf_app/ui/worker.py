"""Background worker that runs one or more VMAF jobs without blocking the UI."""
from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core import vmaf_cuda
from vmaf_app.core.app_log import RUN_START
from vmaf_app.core.cvvdp import CvvdpSettings
from vmaf_app.core.execution import ExecutionPlan, MetricTask, build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.metric_results import MetricResultSet, as_requested, frame_scores_from_results
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import ComparisonResult, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualRunError, PerceptualTaskOutput
from vmaf_app.core.perceptual_vship import (
    GPU_ONLY_METRICS,
    apply_vship_cpu_fallback,
    vship_passes,
)
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.status import GPU_VMAF_FAILED, GPU_WAIT, kind_of, plan_of
from vmaf_app.core.time_format import format_hms
from vmaf_app.core.vmaf_runner import (
    Cancelled,
    VmafRunError,
    analysis_bit_depth,
    auto_threads,
    run_resample_test,
    run_vmaf,
)

_log = logging.getLogger(__name__)

#: The backend of VMAF and NEG calculated on the GPU, a half of their own
#: (_JobRun._by_place): run_vmaf with those two alone.
GPU_VMAF = "vmaf_gpu"
_GPU_VMAF_KEYS = ("vmaf", "vmaf_neg")
#: The backend of SSIMULACRA2 and Butteraugli chosen for the CPU, apart from
#: Vship's metrics on the GPU ("perceptual"): see _JobRun._by_place.
PERCEPTUAL_CPU = "perceptual_cpu"


def _media(info: VideoInfo) -> str:
    """A file's shape, for the log: "3840x2160 23.976 fps vvc yuv420p10le HDR PQ, 1:45:36"."""
    hdr = {"smpte2084": " HDR PQ", "arib-std-b67": " HDR HLG"}.get(info.color_transfer or "", "")
    return (f"{info.width}x{info.height} {info.fps:.3f} fps {info.codec_name} {info.pix_fmt}{hdr}, "
            f"{format_hms(info.duration)}")


def _stderr_note(error: BaseException) -> str:
    tail = getattr(error, "stderr_tail", "") or ""
    return f"\nLast output:\n{tail}" if tail else ""


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
    # Each of a video's halves as it stands (_JobRun.task_snapshots): the
    # window's run line is built from them, metric by metric.
    task_progress = Signal(int, object)
    status = Signal(int, object)            # job_index, status text: a str, often a core.status.Status
    job_finished = Signal(int, object)      # job_index, ComparisonResult
    job_failed = Signal(int, str, str)      # job_index, message, stderr_tail
    # One metric group failed, the other finished: the finished metrics are
    # the result (job_index, ComparisonResult, message, stderr_tail).
    # job_index, result, message, stderr_tail, {metric key: why it failed}
    job_partially_failed = Signal(int, object, str, str, object)
    # job_index, ComparisonResult: the video's result so far, each time a
    # half or a GPU metric's pass finishes while the rest of it runs on --
    # shown and saved at once, not only when the whole video is done.
    result_updated = Signal(int, object)
    cancelled = Signal()
    all_finished = Signal()

    def __init__(self, jobs: list[VmafJob], parallel_jobs: int = 1, parent=None, *,
                 gpu_metrics_together: bool = False):
        super().__init__(parent)
        self._jobs = jobs
        # Settings > GPU metrics: each video's GPU metrics in one Vship pass,
        # decoding it once, rather than one pass (and one decode) per metric.
        # How the scores are calculated, never what they are.
        self.gpu_metrics_together = gpu_metrics_together
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
        _log.info("Run: cancel requested")
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
        _log.info("Run: paused")
        with self._handles_lock:
            self._paused = True
            for handle in self._handles.values():
                handle.pause()

    def resume(self) -> None:
        _log.info("Run: resumed")
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
        # A GPU probe that failed a while ago is made again (on this thread,
        # as each video is planned): the failure may have passed.
        vmaf_cuda.forget_failed_probe()
        runs: list[_JobRun] = []
        for index, job in enumerate(self._jobs):
            try:
                runs.append(_JobRun(self, index, job))
            except Exception as error:  # a request that cannot be built fails its own video
                _log.error("Video %d '%s' could not be set up: %s", index + 1, job.label, error, exc_info=error)
                self.job_started.emit(index, job.label)
                self.job_failed.emit(index, str(error), getattr(error, "stderr_tail", "") or "")
        _log.info("%s %d video(s), CPU metrics for up to %d at once, GPU metrics one video at a time%s", RUN_START,
                  len(self._jobs), self.parallel_jobs,
                  ", each video's in one pass" if self.gpu_metrics_together else "")
        for run in runs:
            _log.info("%s", run.describe())
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
        if self._cancel_event.is_set():
            # A video whose other half had not started when the run was
            # cancelled was never finished by its lanes: what it did finish
            # is still its result.
            for run in runs:
                if run.started and not run.finalized and run.task_results:
                    self._finish(run)
        for run in runs:  # a video cancelled part-way still holds its handle
            self._release_handle(run.index)

        if self._cancel_event.is_set():
            self._report_cancelled_once()
        _log.info("Run ended%s", " (cancelled)" if self._cancel_event.is_set() else "")
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
        run.finalized = True
        index = run.index
        try:
            result, failure = run.finish()
        except (Cancelled, PerceptualCancelled):
            _log.info("%s: cancelled", run.name)
            self._report_cancelled_once()
            return
        except (VmafRunError, PerceptualRunError) as e:
            _log.error("%s failed: %s%s", run.name, e, _stderr_note(e))
            self.job_failed.emit(index, str(e), e.stderr_tail)
            return
        except Exception as e:
            _log.error("%s failed: %s", run.name, e, exc_info=e)
            self.job_failed.emit(index, str(e), "")
            return
        finally:
            self._release_handle(index)
        metrics = ", ".join(metric_definition(key).label for key in result.metric_results)
        if failure is None:
            _log.info("%s finished: %s", run.name, metrics)
            self.job_finished.emit(index, result)
        else:
            message, tail, _reasons = failure
            _log.warning("%s finished with failed metrics (kept: %s):\n%s%s", run.name, metrics or "none",
                         message, f"\nLast output:\n{tail}" if tail else "")
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
        self.plan = self._by_place(build_execution_plan(self.request, self.cached))
        self.token = _TaskCancelToken(worker._cancel_event)
        self.options = job.options
        self.handle: ProcessHandle | None = None
        # Under the worker's scheduler lock: whether the video is in progress,
        # and which of its halves a lane has taken.
        self.started = False
        self.admitted: set[str] = set()
        self.finalized = False  # its result (or failure) has been sent
        self.lock = threading.Lock()
        # Held from building a snapshot under `lock` until it has been emitted:
        # the halves report from their own threads, and a snapshot built first
        # but emitted last left the window showing the older state (a half's
        # decode plan missing until the next report). Re-entrant, so a slot
        # connected directly may report again.
        self.emit_lock = threading.RLock()
        self._begun = False
        self.task_results: dict[str, object] = {}
        self.task_errors: list[tuple[object, Exception]] = []
        self.task_progress: dict[str, tuple[int, int, float]] = {}
        # Each half's passes, the metric keys each scores, in order: Vship's
        # as planned (vship_passes) and as each starts (report_pass_start),
        # a retry after the others; one pass of all its metrics otherwise.
        self.task_passes: dict[str, list[tuple[str, ...]]] = {
            task.backend_id: ([tuple(spec.key for spec in group) for group in
                               vship_passes(task.requested_specs, worker.gpu_metrics_together)]
                              if task.backend_id == "perceptual" else [task.metric_keys])
            for task in self.plan.tasks
        }
        # The pass under way: (number, count, the half's progress figure when
        # it started -- its own figures are measured from there).
        self.task_phases: dict[str, tuple[int, int, int]] = {}
        # Metrics of a half in the GPU's queue being calculated on the CPU,
        # after a GPU failure or as planned beside the GPU's: the half's
        # figures are theirs from then on (report_cpu).
        self.task_cpu_keys: dict[str, tuple[str, ...]] = {}
        # Halves not running yet, and what they wait for: "GPU" or "CPU".
        self.task_waiting: dict[str, str] = {}
        # Each half's latest status message: what a half not yet reporting
        # figures is doing (black-bar detection, say).
        self.task_steps: dict[str, str] = {}
        # Each half's latest decode plan (core.status.Status.plan): the
        # halves decode separately, and one can fall back to software while
        # the other does not.
        self.task_decode: dict[str, HwAccelPlan] = {}
        # The GPU half's finished passes, one metric each, while the half
        # runs on (see perceptual_so_far).
        self.pass_outputs: list[PerceptualTaskOutput] = []
        self._finished_tasks = 0

    def _by_place(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Each half's metrics calculated in one place, the CPU or the GPU.

        The planner groups metrics by the program that calculates them:
        FFmpeg's, and Vship's SSIMULACRA2/Butteraugli/CVVDP. With VMAF on the
        GPU, VMAF and NEG are a half of their own (GPU_VMAF) in the GPU's
        queue, ahead of Vship's metrics, and FFmpeg's half keeps the metrics
        calculated on the CPU. In one FFmpeg run with VMAF v1, PSNR, SSIM and
        XPSNR it went at their pace -- 14.6 fps for all six at 4K -- and the
        Vship metrics waited for the CPU's metrics too.

        SSIMULACRA2 or Butteraugli chosen for the CPU are a half of their own
        too (PERCEPTUAL_CPU), in the CPU's queue. In Vship's half they were
        calculated after its GPU passes, holding the GPU's turn meanwhile,
        and were shown as GPU metrics."""
        options = self.job.options
        tasks = []
        for task in plan.tasks:
            if task.backend_id == "perceptual":
                on_gpu = tuple(spec for spec in task.requested_specs if spec.key in GPU_ONLY_METRICS
                               or self.request.execution.perceptual_backend(spec.key) == "gpu")
                on_cpu = tuple(spec for spec in task.requested_specs if spec not in on_gpu)
                for backend_id, specs in (("perceptual", on_gpu), (PERCEPTUAL_CPU, on_cpu)):
                    if specs:
                        tasks.append(MetricTask(backend_id, tuple(spec.key for spec in specs), specs))
                continue
            vmaf = tuple(spec for spec in task.requested_specs if spec.key in _GPU_VMAF_KEYS)
            keys = {spec.key for spec in vmaf}
            if (task.backend_id != "ffmpeg" or not vmaf or options.resample_test is not None
                    or vmaf_cuda.scores_on_gpu("vmaf" in keys, "vmaf_neg" in keys, options.model,
                                               options.vmaf_on_gpu, analysis_bit_depth(
                                                   self.job.source_info, self.job.distorted_info)) is None):
                tasks.append(task)
                continue
            # The planner keeps FFmpeg's metrics together, so a saved one is
            # calculated again with the rest; split, a part whose metrics are
            # all saved is not.
            for backend_id, specs in (("ffmpeg", tuple(spec for spec in task.requested_specs
                                                       if spec.key not in _GPU_VMAF_KEYS)),
                                      (GPU_VMAF, vmaf)):
                if specs and not (self.cached is not None and all(self.cached.has(spec.key) for spec in specs)):
                    tasks.append(MetricTask(backend_id, tuple(spec.key for spec in specs), specs))
        return ExecutionPlan(plan.requested_metrics, tuple(tasks))

    @property
    def name(self) -> str:
        return f"Video {self.index + 1} '{self.job.label}'"

    def half_name(self, task) -> str:
        labels = ", ".join(metric_definition(key).label for key in task.metric_keys)
        return f"{'GPU' if self.pool_of(task) == _GPU else 'CPU'} metrics ({labels})"

    def describe(self) -> str:
        """The video's files, what it will calculate and how, for the log."""
        job, options = self.job, self.job.options
        halves = "; ".join(self.half_name(task) for task in self.plan.tasks) or "nothing (all saved)"
        kept = ", ".join(metric_definition(key).label for key in self.cached) if self.cached else ""
        settings = (
            f"black bars {options.crop_mode.value}, duration limit {options.duration_limit:g} s, "
            f"resolution mismatch {options.scale_direction.value} ({options.scale_algorithm}), "
            f"VMAF v0.6.1 model {options.model_choice}, VMAF v1 model {options.model_choice_v1}, "
            f"frame subsample {options.n_subsample}, GPU decode {'on' if options.gpu_decode else 'off'}"
        )
        if job.metric_backends:
            settings += ", " + ", ".join(f"{metric_definition(key).label} on {backend.upper()}"
                                         for key, backend in sorted(job.metric_backends.items()))
        if job.cvvdp is not None:
            settings += f", CVVDP display {job.cvvdp.display.describe()}"
        lines = [
            f"{self.name}:",
            f"  test   {job.distorted_info.path} ({_media(job.distorted_info)})",
            f"  source {job.source_info.path} ({_media(job.source_info)})",
            f"  calculating {halves}" + (f"; saved scores kept: {kept}" if kept else ""),
            f"  {settings}",
        ]
        if options.resample_test is not None:
            lines.append(f"  resolution test: {options.resample_test.label}")
        return "\n".join(lines)

    @staticmethod
    def pool_of(task) -> str:
        """The GPU's queue for the halves calculated on the GPU (_by_place):
        one at a time, so that none runs beside another video's GPU
        metrics. A GPU pass that fails is retried on the CPU in the same
        turn. The CPU's queue otherwise."""
        return _GPU if task.backend_id in (GPU_VMAF, "perceptual") else _CPU

    def begin(self) -> bool:
        """Starts the video when its first half is taken: its process
        handle, its share of the cores, and job_started. False if the run
        was cancelled first. Only the first call does anything."""
        with self.emit_lock:
            with self.lock:
                if self._begun:
                    return self.handle is not None
                self._begun = True
                self.handle = self.worker._claim_handle(self.index)
                if self.handle is None:
                    return False
                self.options = self.worker._share_cores(self.job.options)
                _log.info("%s: started", self.name)
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

    def _state(self, task) -> str:
        return ("done" if task.backend_id in self.task_results else
                "failed" if any(failed is task for failed, _error in self.task_errors) else
                "waiting" if task.backend_id in self.task_waiting else
                "running" if task.backend_id in self.task_progress else "starting")

    def _done_keys(self, task) -> tuple[str, ...]:
        """The half's metrics that have scores: all it scored once it is
        done, its finished passes while it runs."""
        output = self.task_results.get(task.backend_id)
        outputs = [output] if output is not None else self.pass_outputs if task.backend_id == "perceptual" else []
        found: set[str] = set()
        for output in outputs:
            found.update(output.metrics if isinstance(output, PerceptualTaskOutput) else output.metric_results)
        return tuple(key for key in task.metric_keys if key in found)

    def task_snapshots(self) -> list[dict[str, object]]:
        """Each half as it stands, for the run line; called with self.lock held."""
        found = []
        for task in self.plan.tasks:
            cur, total, fps = self.task_progress.get(task.backend_id, (0, 0, 0.0))
            found.append({
                "backend": task.backend_id,
                "metric_keys": task.metric_keys,
                "lane": self.pool_of(task),
                "passes": tuple(self.task_passes.get(task.backend_id, (task.metric_keys,))),
                "cpu_keys": self.task_cpu_keys.get(task.backend_id, ()),
                "done_keys": self._done_keys(task),
                "current": cur,
                "total": total,
                "fps": fps,
                "state": self._state(task),
                "waiting_for": self.task_waiting.get(task.backend_id),
                "step": self.task_steps.get(task.backend_id, ""),
                "decode": self.task_decode.get(task.backend_id, ""),
                "phase": self.task_phases.get(task.backend_id),
            })
        return found

    def report_status(self, backend: str, message: str) -> None:
        _log.info("%s: %s", self.name, message)
        with self.emit_lock:
            with self.lock:
                self.task_steps[backend] = message
                if (plan := plan_of(message)) is not None:
                    self.task_decode[backend] = plan
                if len(self.plan.tasks) > 1 and kind_of(message) == GPU_WAIT:
                    self.task_waiting[backend] = "GPU"
                if backend == GPU_VMAF and kind_of(message) == GPU_VMAF_FAILED:
                    # FFmpeg's libvmaf takes over, on the CPU, from frame 0.
                    self._to_cpu(backend, next(task.metric_keys for task in self.plan.tasks
                                               if task.backend_id == backend))
                snapshot = self.task_snapshots()
            self.worker.task_progress.emit(self.index, snapshot)
            self.worker.status.emit(self.index, message)

    def report_pass_start(self, backend: str, number: int, count: int, keys: tuple[str, ...]) -> None:
        """A GPU pass starting (run_vship_task's on_pass): the metrics it
        scores, and where the half's figures stand -- the pass's own are
        measured from there."""
        with self.emit_lock:
            with self.lock:
                passes = self.task_passes.setdefault(backend, [])
                passes.extend([()] * (number - len(passes)))
                passes[number - 1] = keys
                del passes[max(count, number):]
                current, total, _fps = self.task_progress.get(backend, (0, 0, 0.0))
                self.task_phases[backend] = (number, count, current)
                if backend in self.task_progress:
                    # The new pass has no rate yet: the last pass's was shown
                    # (and timed) as the new metric's.
                    self.task_progress[backend] = (current, total, 0.0)
                snapshot = self.task_snapshots()
            self.worker.task_progress.emit(self.index, snapshot)

    def report_cpu(self, backend: str, keys: tuple[str, ...]) -> None:
        """Metrics of a half the CPU is about to calculate (on_cpu)."""
        with self.emit_lock:
            with self.lock:
                self._to_cpu(backend, keys)
                snapshot = self.task_snapshots()
            self.worker.task_progress.emit(self.index, snapshot)

    def _to_cpu(self, backend: str, keys: tuple[str, ...]) -> None:
        """The half's figures are the CPU's for `keys` from now on, from 0;
        called with self.lock held."""
        self.task_cpu_keys[backend] = keys
        self.task_phases.pop(backend, None)
        self.task_progress.pop(backend, None)

    def report_progress(self, backend: str, cur: int, total: int, fps: float) -> None:
        with self.emit_lock:
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
                snapshot = self.task_snapshots()
            worker.task_progress.emit(index, snapshot)
            worker.progress.emit(index, overall_cur, known_total, overall_fps)

    def execute_task(self, task) -> object:
        job, options = self.job, self.options

        def progress(cur, tot, fps):
            self.report_progress(task.backend_id, cur, tot, fps)

        if task.backend_id in ("ffmpeg", GPU_VMAF):  # GPU_VMAF: run_vmaf with VMAF and NEG alone
            task_options = replace(options)
            for key in options.requested_metrics():
                task_options.set_metric_enabled(key, key in task.metric_keys)
            if options.resample_test is not None:
                result = run_resample_test(
                    job.source_info, task_options,
                    on_progress=progress,
                    on_status=lambda msg: self.report_status(task.backend_id, msg),
                    cancel_event=self.token, process_handle=self.handle,
                )
            else:
                result = run_vmaf(
                    job.source_info, job.distorted_info, task_options,
                    on_progress=progress,
                    on_status=lambda msg: self.report_status(task.backend_id, msg),
                    cancel_event=self.token, process_handle=self.handle,
                    result_distorted_path=job.result_distorted_path,
                )
            conformed = as_requested(result.metric_results, task.requested_specs)
            if conformed is not result.metric_results:
                result.metric_results = conformed
                result.frames = frame_scores_from_results(conformed)
            return result
        if task.backend_id in ("perceptual", PERCEPTUAL_CPU):
            return apply_vship_cpu_fallback(
                job.source_info, job.distorted_info, self.request, task.requested_specs,
                on_progress=progress,
                on_status=lambda msg: self.report_status(task.backend_id, msg),
                cancel_event=self.token, process_handle=self.handle,
                on_pass_done=self.report_pass, together=self.worker.gpu_metrics_together,
                on_pass=lambda number, count, keys: self.report_pass_start(task.backend_id, number, count, keys),
                on_cpu=lambda keys: self.report_cpu(task.backend_id, keys),
            )
        raise VmafRunError(f"Unknown metric backend: {task.backend_id}")

    def report_pass(self, output: PerceptualTaskOutput) -> None:
        """A GPU metric's pass is done while its half runs on."""
        with self.lock:
            self.pass_outputs.append(output)
        self._send_result_so_far()

    def _send_result_so_far(self) -> None:
        with self.lock:
            task_results = dict(self.task_results)
        result = self._merged(task_results, self.perceptual_so_far(task_results))
        if result is not None:
            self.worker.result_updated.emit(self.index, result)

    def run_task(self, task) -> bool:
        """Runs one half; True when it was the video's last."""
        with self.lock:
            self.task_waiting.pop(task.backend_id, None)
        half = self.half_name(task)
        _log.info("%s: %s started", self.name, half)
        started = time.monotonic()
        try:
            output = self.execute_task(task)
        except Exception as error:
            took = format_hms(time.monotonic() - started)
            if isinstance(error, (Cancelled, PerceptualCancelled)):
                _log.info("%s: %s cancelled after %s", self.name, half, took)
            else:
                _log.error("%s: %s failed after %s: %s%s", self.name, half, took, error, _stderr_note(error),
                           exc_info=error)
            # The sibling is left to finish: a SSIMULACRA2/Butteraugli
            # failure (an unsupported input, a tool error) used to cancel
            # the libvmaf pass and discard VMAF/PSNR/SSIM/XPSNR with it.
            # The window hears of it now: the half's part of the line went on
            # showing its last step -- "Detecting black bars" -- while the
            # other half ran.
            with self.emit_lock:
                with self.lock:
                    self.task_errors.append((task, error))
                    snapshot = self.task_snapshots()
                if len(self.plan.tasks) > 1:
                    self.worker.task_progress.emit(self.index, snapshot)
        else:
            _log.info("%s: %s done in %s", self.name, half, format_hms(time.monotonic() - started))
            for key, message in (getattr(output, "failures", None) or {}).items():
                _log.error("%s: %s failed: %s", self.name, metric_definition(key).label, message)
            with self.lock:
                self.task_results[task.backend_id] = output
                last_progress = self.task_progress.get(task.backend_id, (1, 1, 0.0))
            if len(self.plan.tasks) > 1:
                self.report_progress(task.backend_id, *last_progress)
        with self.lock:
            self._finished_tasks += 1
            last = self._finished_tasks == len(self.plan.tasks)
            finished_here = task.backend_id in self.task_results
        if finished_here and not last:
            # The other half goes on: this one's scores are shown and saved
            # now, not when the whole video is done.
            self._send_result_so_far()
        return last

    def finish(self):
        """The video's result once every half has run.

        Returns (result, failure): failure is None when every task finished,
        or (message, stderr tail) when one metric group failed and the other
        finished -- the finished metrics are still the result. Raises when
        nothing was produced.
        """
        cached = self.cached
        task_results, task_errors = self.task_results, self.task_errors
        # The GPU half's result -- or, if it was cancelled or failed after
        # some of its passes, those passes: their scores were already shown
        # and saved (see _send_result_so_far).
        perceptual = self.perceptual_so_far(task_results)
        if self.worker._cancel_event.is_set():
            # Cancelling keeps the halves that had finished. The halves run
            # independently, so a video's GPU metrics are often done hours
            # before its CPU metrics; the whole video used to be dropped,
            # finished scores included, when the run was cancelled.
            if not task_results and perceptual is None:
                raise Cancelled("Cancelled by user")
            task_errors = [(task, error) for task, error in task_errors
                           if not isinstance(error, (Cancelled, PerceptualCancelled))]
        if task_errors and not task_results and perceptual is None and cached is None:
            raise task_errors[0][1]
        result = self._merged(task_results, perceptual)
        if result is None:
            if task_errors:
                raise task_errors[0][1]
            raise VmafRunError("No executable metric task was planned.")
        # A metric that failed while the rest of its group finished (CVVDP,
        # GPU only, beside SSIMULACRA2/Butteraugli) is reported like a
        # failed group.
        metric_failures = dict(perceptual.failures) if perceptual is not None else {}
        if not task_errors and not metric_failures:
            return result, None
        messages, stderr_tail, reasons = [], "", {}
        for task, error in task_errors:
            labels = ", ".join(metric_definition(key).label for key in task.metric_keys)
            messages.append(f"{labels} failed: {error}")
            stderr_tail = stderr_tail or getattr(error, "stderr_tail", "") or ""
            reasons.update(dict.fromkeys(task.metric_keys, str(error)))
        for key, message in metric_failures.items():
            messages.append(f"{metric_definition(key).label} failed: {message}")
            reasons[key] = message
        return result, ("\n".join(messages), stderr_tail, reasons)

    def perceptual_so_far(self, task_results: dict) -> PerceptualTaskOutput | None:
        """SSIMULACRA2, Butteraugli and CVVDP so far: the GPU half's result,
        or until it has one its finished passes, and the CPU half's."""
        if "perceptual" in task_results:
            parts = [task_results["perceptual"]]
        else:
            with self.lock:
                parts = list(self.pass_outputs)
        if PERCEPTUAL_CPU in task_results:
            parts.append(task_results[PERCEPTUAL_CPU])
        if not parts:
            return None
        if len(parts) == 1 and ("perceptual" in task_results or PERCEPTUAL_CPU in task_results):
            return parts[0]
        metrics = MetricResultSet()
        failures: dict[str, str] = {}
        for output in parts:
            for key in output.metrics:
                metrics.add(output.metrics.get(key))
            failures.update(output.failures)
        return PerceptualTaskOutput(metrics, parts[0].source_crop, parts[0].distorted_crop,
                                    max(output.compared_frame_count for output in parts), failures)

    def _merged(self, task_results: dict, perceptual: PerceptualTaskOutput | None) -> ComparisonResult | None:
        """The video's result from the halves given: FFmpeg's (or the saved
        run), the GPU metrics, and saved scores this run does not replace.
        Always a new object: a half's own result is merged again each time
        another piece lands, and the window may be showing the last one."""
        job, options, cached = self.job, self.options, self.cached
        result = task_results.get("ffmpeg")
        gpu_vmaf = task_results.get(GPU_VMAF)
        if result is None and gpu_vmaf is not None:
            result, gpu_vmaf = gpu_vmaf, None
        if result is not None:
            result = copy.copy(result)
            if gpu_vmaf is not None:  # VMAF and NEG from their own run, beside FFmpeg's other metrics
                combined = result.metric_results.copy()
                for key in gpu_vmaf.metric_results:
                    combined.add(gpu_vmaf.metric_results.get(key))
                result.merge_metric_results(combined)
                result.model = gpu_vmaf.model  # FFmpeg's run calculated no VMAF: its model was ""
        elif cached is not None:
            # FFmpeg's metrics are all saved: the saved run is the base, so
            # its crops, frame table and file info carry over. A shallow
            # copy is enough -- merge_metric_results replaces the metric set
            # and frame table rather than editing them, and the row's own
            # result object must not change under the UI thread.
            result = copy.copy(job.cached_result)
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
            return None
        if cached is not None:
            # Saved metrics fill only what this run did not calculate: a
            # fresh score always wins over the saved one.
            carried = MetricResultSet(
                value for key in cached
                if not result.has_metric(key) and (value := cached.get(key)) is not None
            )
            if carried:
                result.merge_metric_results(carried)
                # A saved VMAF's model, when this run's halves calculated
                # none: with VMAF on the GPU a saved part of FFmpeg's metrics
                # is not calculated again, and the saved run is not the base.
                saved = job.cached_result
                if carried.has("vmaf") and not result.model:
                    result.model = saved.model
                if carried.has("vmaf_v1") and not result.model_choice_v1:
                    result.model_v1, result.model_choice_v1 = saved.model_v1, saved.model_choice_v1
        return result
