"""Runs the metrics of a list of videos. This is the job queue the window's
VmafWorker runs on its thread, kept free of Qt so that it can be used (by
the command line) and tested on its own.

Terms used here:
- a job is one video to compare with its source (VmafJob);
- a task is the part of a video's metrics that one program calculates in
  one place: FFmpeg's metrics on the CPU, VMAF on the GPU, Vship's metrics
  on the GPU, or SSIMULACRA2 and Butteraugli on the CPU. A task is also
  called a "half" (half_name), from when a video had only two of them;
- a lane is a thread that runs one task at a time. The CPU's queue has two
  lanes (one if it holds a single task), and the "parallel jobs" setting
  says how many of them may work at once. The GPU's queue has one lane.

JobScheduler takes the list of videos and runs each video's tasks in the
CPU's and the GPU's queues, in list order. JobRun is one video: its tasks'
progress, and the result they make together. What happens is reported
through RunEvents."""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from vmaf_app.core import frame_sync, vmaf_gpu
from vmaf_app.core.app_log import RUN_START
from vmaf_app.core.cvvdp import CvvdpSettings
from vmaf_app.core.execution import ExecutionPlan, MetricTask, build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.geometry import analysis_dimensions
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.metric_results import MetricResultSet, as_requested, frame_scores_from_results
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import ComparisonResult, CropMode, FrameScores, VideoInfo, VmafOptions
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

#: The task of VMAF, NEG and VMAF v1 calculated on the GPU (see
#: JobRun._by_place): a run_vmaf with those metrics alone.
GPU_VMAF = "vmaf_gpu"
_GPU_VMAF_KEYS = ("vmaf", "vmaf_neg", "vmaf_v1")
#: The task of SSIMULACRA2 and Butteraugli chosen for the CPU, kept apart
#: from Vship's metrics on the GPU ("perceptual"); see JobRun._by_place.
PERCEPTUAL_CPU = "perceptual_cpu"


def _media(info: VideoInfo) -> str:
    """A video's format, for the log: "3840x2160 23.976 fps vvc yuv420p10le
    HDR PQ, 1:45:36"."""
    hdr = {"smpte2084": " HDR PQ", "arib-std-b67": " HDR HLG"}.get(info.color_transfer or "", "")
    return (f"{info.width}x{info.height} {info.fps:.3f} fps {info.codec_name} {info.pix_fmt}{hdr}, "
            f"{format_hms(info.duration)}")


def _stderr_note(error: BaseException) -> str:
    """The end of the tool's error output carried by `error`, for the log."""
    tail = getattr(error, "stderr_tail", "") or ""
    return f"\nLast output:\n{tail}" if tail else ""


class _TaskCancelToken:
    """Cancels one job's other tasks without cancelling the whole run. It is
    passed where a threading.Event is expected (cancel_event), so it answers
    as one does: is_set and wait."""

    def __init__(self, run_cancel: threading.Event) -> None:
        self._run_cancel = run_cancel
        self._job_cancel = threading.Event()

    def is_set(self) -> bool:
        """Whether the run or the job has been cancelled."""
        return self._run_cancel.is_set() or self._job_cancel.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """As threading.Event.wait: returns True once either the run or the
        job is cancelled, or False if `timeout` seconds pass first. It waits
        on the job's event in short turns, checking the run's in between."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return False
            self._job_cancel.wait(0.05 if left is None else min(left, 0.05))
        return True


@dataclass
class VmafJob:
    """One video to compare with its source, with what to calculate."""

    source_info: VideoInfo
    distorted_info: VideoInfo
    options: VmafOptions
    label: str
    # Replaces the result's `distorted` path (see run_vmaf's
    # result_distorted_path). None, the usual case, keeps distorted_info.path.
    result_distorted_path: Path | None = None
    # The metrics to calculate, and where (CPU or GPU). They belong to the
    # job, not to VmafOptions, because the metrics outside FFmpeg are not
    # FFmpeg settings.
    metric_keys: tuple[str, ...] | None = None
    metric_backends: dict[str, str] = field(default_factory=dict)
    # The CVVDP display models for this video; None means the built-in default.
    cvvdp: CvvdpSettings | None = None
    # What the row already shows for this exact recipe, and which of its
    # metrics can be kept as they are. A task whose metrics are all in
    # cached_metrics is not run, and the kept metrics are merged back into
    # the result, so adding SSIMULACRA2 to a row with VMAF runs Vship alone.
    cached_result: ComparisonResult | None = None
    cached_metrics: MetricResultSet | None = None


#: At most two videos' CPU metrics at once, because a third gains nothing.
#: On a 24-core PC, four 1080p comparisons with the app's defaults took 22.6 s
#: one at a time, 14.4 s two at a time and 14.5 s three at a time. Each job
#: already asks libvmaf for every core, and each extra one adds another decode
#: reading from the same disk, so past two they compete rather than overlap.
MAX_PARALLEL_JOBS = 2

#: Videos in progress at once: one per CPU lane, plus the one on the GPU. A
#: video's CPU and GPU tasks run in separate queues, so one queue can run
#: ahead of the other, but not without limit: each video in progress has a
#: line of its own in the window, and the list is still worked through in
#: order.
MAX_VIDEOS_IN_FLIGHT = MAX_PARALLEL_JOBS + 1

_CPU, _GPU = "cpu", "gpu"

#: How often a task's progress figures are passed on at most
#: (JobRun.report_progress), unless they change what its line says. Each
#: scorer reports every few frames: for three tasks of a 120 fps 1080p video,
#: about 290 reports a second, two signals each, queued to the window's thread,
#: which cost 3% of a core for a line redrawn once a second.
_PROGRESS_INTERVAL_S = 0.1


def _ignore(*_args) -> None:
    """The default for an event nobody listens to."""


@dataclass
class RunEvents:
    """The events a run reports as they happen. Each is called on the thread
    of the lane it happens on (or the thread running JobScheduler.run). The
    window's VmafWorker passes them on as its Qt signals."""

    # Arguments: job index, label.
    job_started: Callable[[int, str], None] = _ignore
    # Arguments: job index, current frame, total frames, frames per second.
    progress: Callable[[int, int, int, float], None] = _ignore
    # Arguments: job index, and the state of each of the video's tasks
    # (JobRun.task_snapshots). The window builds its run line from them,
    # metric by metric.
    task_progress: Callable[[int, list], None] = _ignore
    # Arguments: job index, status text (a str, often a core.status.Status).
    status: Callable[[int, str], None] = _ignore
    # Arguments: job index, result.
    job_finished: Callable[[int, ComparisonResult], None] = _ignore
    # Arguments: job index, error message, the end of the tool's error output.
    job_failed: Callable[[int, str, str], None] = _ignore
    # Some of the video's tasks failed and the others finished, and the
    # finished metrics are the result. Arguments: job index, result, error
    # message, the end of the tool's error output, {metric key: why it failed}.
    job_partially_failed: Callable[[int, ComparisonResult, str, str, dict], None] = _ignore
    # Arguments: job index, the video's result so far. Sent each time one of
    # its tasks, or one pass of its GPU metrics, finishes while the rest
    # still run, so that the result is shown and saved at once rather than
    # only when the whole video is done.
    result_updated: Callable[[int, ComparisonResult], None] = _ignore
    # The run stopped. Sent once, however many videos it stopped.
    cancelled: Callable[[], None] = _ignore
    all_finished: Callable[[], None] = _ignore


class JobScheduler:
    """Runs the metrics of a list of videos: each video's tasks (JobRun) in
    the CPU's and the GPU's queues, on lanes of their own, reporting to
    `events`. run() blocks until everything is done or cancelled, and the
    controls (cancel, pause, resume, set_parallel_jobs) may be used from any
    thread meanwhile."""

    def __init__(self, jobs: list[VmafJob], parallel_jobs: int = 1, events: RunEvents | None = None, *,
                 gpu_metrics_together: bool = False) -> None:
        self.jobs = jobs
        self.events = events or RunEvents()
        # Settings > GPU metrics: calculate each video's GPU metrics in one
        # Vship pass, decoding the video once, rather than in one pass (and
        # one decode) per metric. This changes how the scores are calculated,
        # never what they are.
        self.gpu_metrics_together = gpu_metrics_together
        self._parallel_jobs = max(1, min(int(parallel_jobs), MAX_PARALLEL_JOBS))
        self._cancel_event = threading.Event()
        # One process handle per running job, because pausing or cancelling
        # has to reach every FFmpeg that is running, and a handle controls
        # only one process.
        self._handles: dict[int, ProcessHandle] = {}
        self._handles_lock = threading.Lock()
        # _parallel_jobs is how many CPU lanes may work at once. It can be
        # changed during a run: a list of feature-length videos is exactly
        # when someone notices that their CPU is idle, and restarting the
        # list to use it would waste the work done.
        #
        # Work is scheduled per task, not per video: FFmpeg's metrics (and
        # SSIMULACRA2 and Butteraugli set to the CPU) wait for the CPU's
        # lanes, and the GPU metrics for the GPU's, each queue in list order.
        # If a lane took a whole video instead, a second lane would take the
        # next video even when all it needed was the GPU. It would then use
        # the GPU ahead of the first video, and the third video's CPU
        # metrics would wait although the CPU had room for them.
        self._sched = threading.Condition()
        self._busy = {_CPU: 0, _GPU: 0}
        self._queues: dict[str, list] = {_CPU: [], _GPU: []}
        self._in_flight: set[int] = set()
        self._paused = False
        self._cancellation_reported = False
        # How many CPU lanes this run has: set by run(), at least one, and
        # never more than there are CPU tasks. It limits how many jobs the
        # CPU is shared between: a single video runs alone whatever the
        # setting says, so it should not be given only half of the CPU.
        self._lane_count = 1

    # ------------------------------------------------------------- controls
    def cancel(self) -> None:
        """Stops the run: kills the running processes and starts no more."""
        _log.info("Run: cancel requested")
        # The flag is set and the handles are collected under one lock, so a
        # lane claiming a handle either is in this list (and is stopped) or
        # sees the flag and never starts. With the flag set outside the lock,
        # a lane could pass its check of the flag, then register its handle
        # after this list was taken, and its FFmpeg started anyway.
        with self._handles_lock:
            self._cancel_event.set()
            handles = list(self._handles.values())
        # terminate() also wakes a paused process. Its reader loop cannot
        # notice the flag on its own, because a paused process writes nothing
        # (see ProcessHandle.terminate).
        for handle in handles:
            handle.terminate()
        with self._sched:
            self._sched.notify_all()  # wakes any lane waiting for room

    def pause(self) -> None:
        """Pauses every running process; jobs started later start paused."""
        # The handles are paused while the lock is held, so a pause cannot
        # interleave with a lane claiming a handle or with resume(). Pausing
        # them after releasing the lock would let a resume come between
        # setting the flag and pausing a handle, which leaves that lane paused
        # for good while the window says the run has resumed.
        _log.info("Run: paused")
        with self._handles_lock:
            self._paused = True
            for handle in self._handles.values():
                handle.pause()

    def resume(self) -> None:
        """Resumes every paused process."""
        _log.info("Run: resumed")
        with self._handles_lock:
            self._paused = False
            for handle in self._handles.values():
                handle.resume()

    @property
    def parallel_jobs(self) -> int:
        """How many videos' CPU metrics may run at once."""
        with self._sched:
            return self._parallel_jobs

    def set_parallel_jobs(self, count: int) -> None:
        """Changes how many videos' CPU metrics may run at once, during a run.

        Raising it lets a waiting lane start the next video at once. Lowering
        it never stops work that has already started: it only keeps more
        from starting until enough have finished.
        """
        with self._sched:
            self._parallel_jobs = max(1, min(int(count), MAX_PARALLEL_JOBS))
            self._sched.notify_all()

    @property
    def is_paused(self) -> bool:
        """Whether the run is paused."""
        with self._handles_lock:
            return self._paused

    def _claim_handle(self, index: int) -> ProcessHandle | None:
        """Returns a process handle for one job, or None if the run has been
        cancelled.

        This happens under the same lock that cancel, pause and resume take,
        so it cannot interleave with any of them. A job that starts while
        the run is paused starts paused, and a job that tries to start after
        a cancel never starts.
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
        """Forgets a job's process handle once the job is over."""
        with self._handles_lock:
            self._handles.pop(index, None)

    def _share_cores(self, options: VmafOptions) -> VmafOptions:
        """Replaces "Auto" libvmaf threads with this job's share of the CPU.

        The share is decided as the job starts, from the number of lanes at
        that moment. If a run is widened to two lanes part-way through,
        every video started after that gets half of the cores, while a video
        already running keeps the threads it started with (a running FFmpeg's
        thread count cannot be changed). A thread count the user chose is
        kept as it is. n_threads is not part of the cache key, so this
        changes how fast the result arrives, not what it is.
        """
        if options.n_threads > 0:
            return options
        concurrent = min(self.parallel_jobs, self._lane_count)
        if concurrent <= 1:
            return options  # the runner turns Auto into every core itself
        return replace(options, n_threads=auto_threads(concurrent))

    def concurrent_cpu_tasks(self) -> int:
        """How many tasks the CPU's queue may run at once now. The CPU
        metrics that score several frame pairs at a time divide the cores
        and memory by this (perceptual_cpu.scoring_workers), as _share_cores
        divides libvmaf's threads."""
        return max(1, min(self.parallel_jobs, self._lane_count))

    def _report_cancelled_once(self) -> None:
        """Sends `cancelled` once. It means the run stopped, not that a job
        did, and when several jobs are running they all raise Cancelled
        together."""
        with self._handles_lock:
            if self._cancellation_reported:
                return
            self._cancellation_reported = True
        self.events.cancelled()

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        """Runs every job, then returns once all are done or the run was
        cancelled."""
        # A check of the GPU that failed a while ago is made again (on this
        # thread, as each video is planned), because the failure may have
        # passed.
        vmaf_gpu.forget_failed_probe()
        runs: list[JobRun] = []
        for index, job in enumerate(self.jobs):
            try:
                runs.append(JobRun(self, index, job))
            except Exception as error:  # only this video fails if its request cannot be built
                _log.error("Video %d '%s' could not be set up: %s", index + 1, job.label, error, exc_info=error)
                self.events.job_started(index, job.label)
                self.events.job_failed(index, str(error), getattr(error, "stderr_tail", "") or "")
        _log.info("%s %d video(s), CPU metrics for up to %d at once, GPU metrics one video at a time%s%s", RUN_START,
                  len(self.jobs), self.parallel_jobs,
                  ", each video's in one pass" if self.gpu_metrics_together else "",
                  ", frames paired by timestamp (VML_FRAME_PAIRING)" if frame_sync.by_timestamp() else "")
        for run in runs:
            _log.info("%s", run.describe())
        for run in runs:
            if not run.plan.tasks:
                # Every metric is already saved, so the saved result is the result.
                if run.begin():
                    self._finish(run)
                continue
            for task in run.plan.tasks:
                self._queues[run.pool_of(task)].append((run, task))

        # As many CPU lanes as the setting can ever allow, so that it can be
        # raised during the run. A lane with no room waits, which costs an
        # idle thread and nothing else. The GPU has one lane.
        cpu_lanes = min(MAX_PARALLEL_JOBS, len(self._queues[_CPU]))
        self._lane_count = max(1, cpu_lanes)
        lanes = [
            threading.Thread(target=self._drain_queue, args=(_CPU,), name=f"vmaf-cpu-{n}", daemon=True)
            for n in range(cpu_lanes)
        ]
        if self._queues[_GPU]:
            lanes.append(threading.Thread(target=self._drain_queue, args=(_GPU,), name="vmaf-gpu", daemon=True))
        # The last lane runs on this thread. That saves a thread, and a run
        # with a single lane stays on the thread that called run().
        for lane in lanes[:-1]:
            lane.start()
        if lanes:
            lanes[-1].run()
        for lane in lanes[:-1]:
            lane.join()
        if self._cancel_event.is_set():
            # A video cancelled before all of its tasks started was not
            # finished by any lane, but the tasks it did finish are still
            # its result.
            for run in runs:
                if run.started and not run.finalized and run.task_results:
                    self._finish(run)
        for run in runs:  # a video cancelled part-way through still holds its process handle
            self._release_handle(run.index)

        if self._cancel_event.is_set():
            self._report_cancelled_once()
        _log.info("Run ended%s", " (cancelled)" if self._cancel_event.is_set() else "")
        self.events.all_finished()

    def _next_task(self, pool: str):
        """Returns the next task this queue may run, in list order, waiting
        until there is room for it. Returns None once the queue is empty or
        the run is cancelled."""
        with self._sched:
            while True:
                if self._cancel_event.is_set() or not self._queues[pool]:
                    return None
                if (picked := self._take_next(pool)) is not None:
                    return picked
                # cancel() and everything that makes room wake this
                # condition; the timeout is only a safety net.
                self._sched.wait(0.1)

    def _take_next(self, pool: str):
        """Takes the task this queue may start now off the queue and returns
        it, or returns None when there is no room for one. Called with
        self._sched held."""
        if self._busy[pool] >= (self._parallel_jobs if pool == _CPU else 1):
            return None
        # Take the first task whose video is already in progress, or that may
        # start a new video. A later video's task passes an earlier one's
        # only when the earlier video cannot start yet, so the videos in
        # progress can always finish.
        queue = self._queues[pool]
        for position, (run, task) in enumerate(queue):
            if run.started or len(self._in_flight) < MAX_VIDEOS_IN_FLIGHT:
                del queue[position]
                self._busy[pool] += 1
                run.admitted.add(task.backend_id)
                if not run.started:
                    run.started = True
                    self._in_flight.add(run.index)
                return run, task
        return None

    def _drain_queue(self, pool: str) -> None:
        """One lane: runs this queue's tasks one at a time until the queue is
        empty or the run is cancelled, and finishes each video whose last
        task it ran."""
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

    def _finish(self, run: JobRun) -> None:
        """Builds the video's result from its tasks and reports it, or
        reports why it failed or that it was cancelled. Then releases the
        video's process handle. All of it under the video's emit lock, as its
        result so far is sent (JobRun._send_result_so_far): the two go out in
        the order they were made, and no result so far after this."""
        with run.emit_lock:
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
                self.events.job_failed(index, str(e), e.stderr_tail)
                return
            except Exception as e:
                _log.error("%s failed: %s", run.name, e, exc_info=e)
                self.events.job_failed(index, str(e), "")
                return
            finally:
                self._release_handle(index)
            metrics = ", ".join(metric_definition(key).label for key in result.metric_results)
            if failure is None:
                _log.info("%s finished: %s", run.name, metrics)
                self.events.job_finished(index, result)
            else:
                message, tail, _reasons = failure
                _log.warning("%s finished with failed metrics (kept: %s):\n%s%s", run.name, metrics or "none",
                             message, f"\nLast output:\n{tail}" if tail else "")
                self.events.job_partially_failed(index, result, *failure)


class JobRun:
    """One video: its tasks, which the scheduler's CPU and GPU lanes run,
    their progress, and the result they make together."""

    def __init__(self, scheduler: JobScheduler, index: int, job: VmafJob) -> None:
        self.scheduler, self.index, self.job = scheduler, index, job
        self.request = analysis_request_from_vmaf_options(
            job.options, job.metric_keys, job.metric_backends, job.cvvdp,
        )
        self.cached = job.cached_metrics if job.cached_result is not None and job.cached_metrics else None
        self.plan = self._by_place(build_execution_plan(self.request, self.cached))
        self.token = _TaskCancelToken(scheduler._cancel_event)
        self.options = job.options
        self.handle: ProcessHandle | None = None
        # Changed under the scheduler's lock: whether the video is in
        # progress, and which of its tasks a lane has taken.
        self.started = False
        self.admitted: set[str] = set()
        self.finalized = False  # whether its result (or failure) has been sent
        self.lock = threading.Lock()
        # Held from building a snapshot under `lock` until the snapshot has
        # been sent. The tasks report from their own threads, and a snapshot
        # built first but sent last would leave the window showing the older
        # state (a task's decode plan missing until its next report). It is
        # re-entrant so that a slot connected directly, which runs on the
        # sending thread, may report again.
        self.emit_lock = threading.RLock()
        self._begun = False
        self.task_results: dict[str, object] = {}
        self.task_errors: list[tuple[object, Exception]] = []
        self.task_progress: dict[str, tuple[int, int, float]] = {}
        # Each task's passes in order, as the metric keys each pass scores.
        # Vship's come from the plan (vship_passes) and are updated as each
        # pass starts (report_pass_start), with any retry added after the
        # others. Every other task has one pass of all its metrics.
        self.task_passes: dict[str, list[tuple[str, ...]]] = {
            task.backend_id: ([tuple(spec.key for spec in group) for group in
                               vship_passes(task.requested_specs, scheduler.gpu_metrics_together)]
                              if task.backend_id == "perceptual" else [task.metric_keys])
            for task in self.plan.tasks
        }
        # The pass under way, per task: (pass number, pass count, the task's
        # progress count when the pass started). The pass's own progress is
        # counted from there.
        self.task_phases: dict[str, tuple[int, int, int]] = {}
        # A task's metrics that the CPU is calculating: those chosen for the
        # CPU, or a GPU task's metrics handed over after the GPU failed. From
        # then on, the task's progress figures are theirs (report_cpu).
        self.task_cpu_keys: dict[str, tuple[str, ...]] = {}
        # Tasks not running yet, and what each waits for: "GPU" or "CPU".
        self.task_waiting: dict[str, str] = {}
        # Each task's latest status message: what a task with no progress
        # figures yet is doing (detecting black bars, for example).
        self.task_steps: dict[str, str] = {}
        # Each task's latest decode plan (core.status.Status.plan). The tasks
        # decode separately, and one can fall back to software decoding while
        # another does not.
        self.task_decode: dict[str, HwAccelPlan] = {}
        # The scores of each finished pass of Vship's task, while the task
        # still runs (see perceptual_so_far).
        self.pass_outputs: list[PerceptualTaskOutput] = []
        self._finished_tasks = 0
        self._progress_sent = float("-inf")  # time.monotonic() when progress was last sent

    def _by_place(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Splits the plan's tasks so that each task's metrics are calculated
        in one place, the CPU or the GPU.

        The planner groups metrics by the program that calculates them:
        FFmpeg, or Vship (SSIMULACRA2, Butteraugli and CVVDP). With VMAF on
        the GPU, VMAF, NEG and VMAF v1 become a task of their own (GPU_VMAF)
        in the GPU's queue, ahead of Vship's metrics, and FFmpeg's task keeps
        the metrics calculated on the CPU. In the same FFmpeg as PSNR, SSIM
        and XPSNR, they would go at those metrics' pace (14.6 fps for all six
        at 4K), and Vship's metrics would wait for the CPU's too.

        SSIMULACRA2 or Butteraugli chosen for the CPU become a task of their
        own too (PERCEPTUAL_CPU), in the CPU's queue. In Vship's task, they
        would be calculated after its GPU passes, holding the GPU meanwhile,
        and be shown as GPU metrics."""
        options = self.job.options
        # The size the videos are compared at, if it is known before the run,
        # for the GPU's check of odd sizes. It is known when black bars are
        # kept and there is no resolution test. Pictures with their black
        # bars cut are always even-sized (crop_detect).
        size = (analysis_dimensions(self.job.source_info, self.job.distorted_info, options)
                if options.crop_mode is CropMode.NONE and options.resample_test is None else None)
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
            on_gpu = None
            if task.backend_id == "ffmpeg" and vmaf and options.resample_test is None:
                on_gpu = vmaf_gpu.scores_on_gpu(
                    "vmaf" in keys, "vmaf_neg" in keys, options.model, options.vmaf_on_gpu,
                    analysis_bit_depth(self.job.source_info, self.job.distorted_info), size=size,
                    compute_vmaf_v1="vmaf_v1" in keys, model_v1=options.model_v1)
            if on_gpu is None:
                tasks.append(task)
                continue
            # The ones the GPU scores (VMAF with a custom model stays on the CPU).
            vmaf = tuple(spec for spec in vmaf if spec.key in on_gpu)
            # The planner keeps FFmpeg's metrics together, so a saved one is
            # calculated again with the rest. Once they are split, a part
            # whose metrics are all saved is not run.
            for backend_id, specs in (("ffmpeg", tuple(spec for spec in task.requested_specs
                                                       if spec not in vmaf)),
                                      (GPU_VMAF, vmaf)):
                if specs and not (self.cached is not None and all(self.cached.has(spec.key) for spec in specs)):
                    tasks.append(MetricTask(backend_id, tuple(spec.key for spec in specs), specs))
        return ExecutionPlan(plan.requested_metrics, tuple(tasks))

    @property
    def name(self) -> str:
        """The video's name in the log: Video 2 'label'."""
        return f"Video {self.index + 1} '{self.job.label}'"

    def half_name(self, task) -> str:
        """The task's name in the log: "GPU metrics (SSIMULACRA2, CVVDP)"."""
        labels = ", ".join(metric_definition(key).label for key in task.metric_keys)
        return f"{'GPU' if self.pool_of(task) == _GPU else 'CPU'} metrics ({labels})"

    def describe(self) -> str:
        """The video's files, what will be calculated and how, for the log."""
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
        """Returns the queue a task waits in: the GPU's for the tasks
        calculated on the GPU (see _by_place), and the CPU's for the others.
        The GPU's queue runs one task at a time, so that no video's GPU
        metrics run beside another's. A GPU pass that fails is retried on
        the CPU in the same turn."""
        return _GPU if task.backend_id in (GPU_VMAF, "perceptual") else _CPU

    def begin(self) -> bool:
        """Starts the video when a lane takes its first task: claims its
        process handle, decides its share of the cores and sends
        job_started. Returns False if the run was cancelled first. Only the
        first call does anything, and later calls return its answer."""
        with self.emit_lock:
            with self.lock:
                if self._begun:
                    return self.handle is not None
                self._begun = True
                self.handle = self.scheduler._claim_handle(self.index)
                if self.handle is None:
                    return False
                self.options = self.scheduler._share_cores(self.job.options)
                _log.info("%s: started", self.name)
                self.scheduler.events.job_started(self.index, self.job.label)
                if len(self.plan.tasks) > 1:
                    admitted = set(self.admitted)
                    for task in self.plan.tasks:
                        if task.backend_id not in admitted:
                            self.task_waiting[task.backend_id] = "GPU" if self.pool_of(task) == _GPU else "CPU"
                    snapshot = self.task_snapshots()
            if len(self.plan.tasks) > 1:
                self.scheduler.events.task_progress(self.index, snapshot)
            return True

    def _state(self, task) -> str:
        """The task's state on the run line: done, failed, waiting, running
        or starting."""
        return ("done" if task.backend_id in self.task_results else
                "failed" if any(failed is task for failed, _error in self.task_errors) else
                "waiting" if task.backend_id in self.task_waiting else
                "running" if task.backend_id in self.task_progress else "starting")

    def _done_keys(self, task) -> tuple[str, ...]:
        """The task's metrics that have scores: all that it scored once it
        is done, or those of its finished passes while it runs."""
        output = self.task_results.get(task.backend_id)
        outputs = [output] if output is not None else self.pass_outputs if task.backend_id == "perceptual" else []
        found: set[str] = set()
        for output in outputs:
            found.update(output.metrics if isinstance(output, PerceptualTaskOutput) else output.metric_results)
        return tuple(key for key in task.metric_keys if key in found)

    def task_snapshots(self) -> list[dict[str, object]]:
        """The state of each task, for the run line. Called with self.lock held."""
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
        """Logs a task's status message, keeps it for the run line and passes
        it on."""
        _log.info("%s: %s", self.name, message)
        with self.emit_lock:
            with self.lock:
                self.task_steps[backend] = message
                if (plan := plan_of(message)) is not None:
                    self.task_decode[backend] = plan
                if len(self.plan.tasks) > 1 and kind_of(message) == GPU_WAIT:
                    self.task_waiting[backend] = "GPU"
                if backend == GPU_VMAF and kind_of(message) == GPU_VMAF_FAILED:
                    # FFmpeg's libvmaf takes over on the CPU, from frame 0.
                    self._to_cpu(backend, next(task.metric_keys for task in self.plan.tasks
                                               if task.backend_id == backend))
                snapshot = self.task_snapshots()
            self.scheduler.events.task_progress(self.index, snapshot)
            self.scheduler.events.status(self.index, message)

    def report_pass_start(self, backend: str, number: int, count: int, keys: tuple[str, ...]) -> None:
        """Records a GPU pass as it starts (run_vship_task's on_pass): the
        metrics it scores, and the task's progress count at that moment, from
        which the pass's own progress is counted."""
        with self.emit_lock:
            with self.lock:
                passes = self.task_passes.setdefault(backend, [])
                passes.extend([()] * (number - len(passes)))
                passes[number - 1] = keys
                del passes[max(count, number):]
                current, total, _fps = self.task_progress.get(backend, (0, 0, 0.0))
                self.task_phases[backend] = (number, count, current)
                if backend in self.task_progress:
                    # The new pass has no speed yet. Without this, the last
                    # pass's speed would be shown as the new metric's, and
                    # used for its time left.
                    self.task_progress[backend] = (current, total, 0.0)
                snapshot = self.task_snapshots()
            self.scheduler.events.task_progress(self.index, snapshot)

    def report_cpu(self, backend: str, keys: tuple[str, ...]) -> None:
        """Records which of a task's metrics the CPU is about to calculate
        (on_cpu)."""
        with self.emit_lock:
            with self.lock:
                self._to_cpu(backend, keys)
                snapshot = self.task_snapshots()
            self.scheduler.events.task_progress(self.index, snapshot)

    def _to_cpu(self, backend: str, keys: tuple[str, ...]) -> None:
        """From now on, the task's progress figures are the CPU's for `keys`,
        counted from 0. Called with self.lock held."""
        self.task_cpu_keys[backend] = keys
        self.task_phases.pop(backend, None)
        self.task_progress.pop(backend, None)

    def report_progress(self, backend: str, cur: int, total: int, fps: float, *, force: bool = False) -> None:
        """Passes on a task's progress figures at most every
        _PROGRESS_INTERVAL_S, keeping the latest in between, unless they
        change what its line says: the task's first figures, a task that is
        no longer waiting, its last frame, or `force` (the task has
        finished)."""
        with self.emit_lock:
            events, index, tasks = self.scheduler.events, self.index, self.plan.tasks
            now = time.monotonic()
            with self.lock:
                if not (force or backend not in self.task_progress or backend in self.task_waiting
                        or (total > 0 and cur >= total) or now - self._progress_sent >= _PROGRESS_INTERVAL_S):
                    self.task_progress[backend] = (cur, total, fps)
                    return
                self._progress_sent = now
            if len(tasks) == 1:
                with self.lock:
                    self.task_progress[backend] = (cur, total, fps)
                    snapshot = self.task_snapshots()
                events.task_progress(index, snapshot)
                events.progress(index, cur, total, fps)
                return
            # The tasks may cover different numbers of frames. Until all of
            # them are done, the video's progress is that of the task with
            # the smallest fraction done.
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
            events.task_progress(index, snapshot)
            events.progress(index, overall_cur, known_total, overall_fps)

    def execute_task(self, task) -> object:
        """Calculates one task's metrics and returns its output: a
        ComparisonResult for FFmpeg's and the GPU's VMAF, or a
        PerceptualTaskOutput for SSIMULACRA2, Butteraugli and CVVDP."""
        job, options = self.job, self.options

        def progress(cur, tot, fps):
            self.report_progress(task.backend_id, cur, tot, fps)

        if task.backend_id in ("ffmpeg", GPU_VMAF):  # GPU_VMAF is run_vmaf with VMAF, NEG and VMAF v1 alone
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
                on_pass_done=self.report_pass, together=self.scheduler.gpu_metrics_together,
                on_pass=lambda number, count, keys: self.report_pass_start(task.backend_id, number, count, keys),
                on_cpu=lambda keys: self.report_cpu(task.backend_id, keys),
                concurrent_cpu_tasks=self.scheduler.concurrent_cpu_tasks(),
            )
        raise VmafRunError(f"Unknown metric backend: {task.backend_id}")

    def report_pass(self, output: PerceptualTaskOutput) -> None:
        """Keeps the scores of a GPU pass that finished while its task still
        runs, and sends the video's result so far."""
        with self.lock:
            self.pass_outputs.append(output)
        self._send_result_so_far()

    def _send_result_so_far(self) -> None:
        """Sends the video's result so far, so that finished scores are shown
        and saved before the whole video is done.

        The emit lock is held from taking the tasks' results until the result
        is sent, and the video's whole result is sent under it too
        (JobScheduler._finish). Otherwise a result so far taken just before
        the last task stored its result could be sent after the whole result
        and replace it: the command line's table and JSON then lacked that
        task's metrics, and the saved comparison's context.json (its VMAF
        model, its frames compared) was the result so far's. Once the whole
        result is out, nothing is sent."""
        with self.emit_lock:
            if self.finalized:
                return
            with self.lock:
                task_results = dict(self.task_results)
            result = self._merged(task_results, self.perceptual_so_far(task_results))
            if result is not None:
                self.scheduler.events.result_updated(self.index, result)

    def run_task(self, task) -> bool:
        """Runs one task. Returns True when it was the video's last task to
        end."""
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
            # The video's other tasks are left to finish, so that a failure
            # of SSIMULACRA2 or Butteraugli (an unsupported input, a tool
            # error) does not lose VMAF, PSNR, SSIM and XPSNR too. The window
            # is told at once, or the failed task's part of the line would
            # keep showing its last step ("Detecting black bars") while the
            # other tasks run.
            with self.emit_lock:
                with self.lock:
                    self.task_errors.append((task, error))
                    snapshot = self.task_snapshots()
                if len(self.plan.tasks) > 1:
                    self.scheduler.events.task_progress(self.index, snapshot)
        else:
            _log.info("%s: %s done in %s", self.name, half, format_hms(time.monotonic() - started))
            for key, message in (getattr(output, "failures", None) or {}).items():
                _log.error("%s: %s failed: %s", self.name, metric_definition(key).label, message)
            with self.lock:
                self.task_results[task.backend_id] = output
                last_progress = self.task_progress.get(task.backend_id, (1, 1, 0.0))
            if len(self.plan.tasks) > 1:
                self.report_progress(task.backend_id, *last_progress, force=True)
        with self.lock:
            self._finished_tasks += 1
            last = self._finished_tasks == len(self.plan.tasks)
            finished_here = task.backend_id in self.task_results
        if finished_here and not last:
            # Other tasks are still running: this task's scores are shown and
            # saved now, not when the whole video is done.
            self._send_result_so_far()
        return last

    def finish(self):
        """Returns the video's result once all of its tasks have ended.

        Returns (result, failure). failure is None when every task finished,
        or (message, end of the tool's error output, {metric key: why it
        failed}) when some of the metrics failed and the others finished; the
        finished metrics are still the result. Raises an error when nothing
        was produced.
        """
        cached = self.cached
        task_results, task_errors = self.task_results, self.task_errors
        # The result of Vship's task or, if the task was cancelled or failed
        # after some of its passes, those passes' scores, which were already
        # shown and saved (see _send_result_so_far).
        perceptual = self.perceptual_so_far(task_results)
        if self.scheduler._cancel_event.is_set():
            # A cancelled run keeps the tasks that had finished. The tasks
            # run independently, so a video's GPU metrics are often done
            # hours before its CPU metrics, and cancelling must not throw
            # those scores away.
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
        # A metric that failed while the rest of its task finished (CVVDP,
        # which runs on the GPU only, beside SSIMULACRA2 and Butteraugli) is
        # reported like a failed task.
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
        """The scores of SSIMULACRA2, Butteraugli and CVVDP so far: the
        result of Vship's task (or, until it has one, its finished passes),
        together with the result of the CPU's PERCEPTUAL_CPU task."""
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
        """Builds the video's result from the given tasks' results: FFmpeg's
        (or the saved result), the GPU metrics, and the saved scores that
        this run does not replace. It is always a new object, because a
        task's own result is merged again each time another part arrives,
        and the window may be showing the previous one."""
        job, options, cached = self.job, self.options, self.cached
        result = task_results.get("ffmpeg")
        gpu_vmaf = task_results.get(GPU_VMAF)
        if result is None and gpu_vmaf is not None:
            result, gpu_vmaf = gpu_vmaf, None
        if result is not None:
            result = copy.copy(result)
            if gpu_vmaf is not None:  # VMAF, NEG and VMAF v1 from their own run, beside FFmpeg's other metrics
                combined = result.metric_results.copy()
                for key in gpu_vmaf.metric_results:
                    combined.add(gpu_vmaf.metric_results.get(key))
                result.merge_metric_results(combined)
                # Each VMAF's model comes from the task that calculated it.
                # FFmpeg's task calculates VMAF with a custom model, and the
                # GPU's task then calculates VMAF v1 alone, with a model of "".
                if gpu_vmaf.has_metric("vmaf"):
                    result.model = gpu_vmaf.model
                if gpu_vmaf.has_metric("vmaf_v1"):
                    result.model_v1, result.model_choice_v1 = gpu_vmaf.model_v1, gpu_vmaf.model_choice_v1
        elif cached is not None:
            # FFmpeg's metrics are all saved, so the saved result is the
            # base, and its crops, frame table and file details carry over.
            # A shallow copy is enough, because merge_metric_results replaces
            # the metric set and frame table rather than editing them. The
            # row's own result object must not change under the UI thread.
            result = copy.copy(job.cached_result)
        if perceptual is not None:
            if result is None:
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
            # Saved metrics fill in only what this run did not calculate: a
            # new score always replaces the saved one.
            carried = MetricResultSet(
                value for key in cached
                if not result.has_metric(key) and (value := cached.get(key)) is not None
            )
            if carried:
                result.merge_metric_results(carried)
                # A saved VMAF keeps its model when this run's tasks
                # calculated no VMAF. With VMAF on the GPU, a part of FFmpeg's
                # metrics that is all saved is not calculated again, and the
                # saved result is not the base, so its model is copied here.
                saved = job.cached_result
                if carried.has("vmaf") and not result.model:
                    result.model = saved.model
                if carried.has("vmaf_v1") and not result.model_choice_v1:
                    result.model_v1, result.model_choice_v1 = saved.model_v1, saved.model_choice_v1
        return result
