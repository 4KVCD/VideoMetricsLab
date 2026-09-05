"""Background worker that runs one or more VMAF jobs without blocking the UI."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.models import VideoInfo, VmafOptions
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError, run_resample_test, run_vmaf


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
    job_finished = Signal(int, object)      # job_index, VmafRunResult
    job_failed = Signal(int, str, str)      # job_index, message, stderr_tail
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
            self.job_started.emit(i, job.label)
            try:
                if job.options.resample_test is not None:
                    result = run_resample_test(
                        job.source_info,
                        job.options,
                        on_progress=lambda cur, tot, fps, idx=i: self.progress.emit(idx, cur, tot, fps),
                        on_status=lambda msg, idx=i: self.status.emit(idx, msg),
                        cancel_event=self._cancel_event,
                        process_handle=handle,
                    )
                else:
                    result = run_vmaf(
                        job.source_info,
                        job.distorted_info,
                        job.options,
                        on_progress=lambda cur, tot, fps, idx=i: self.progress.emit(idx, cur, tot, fps),
                        on_status=lambda msg, idx=i: self.status.emit(idx, msg),
                        cancel_event=self._cancel_event,
                        process_handle=handle,
                        result_distorted_path=job.result_distorted_path,
                    )
            except Cancelled:
                self._report_cancelled_once()
                break
            except VmafRunError as e:
                self.job_failed.emit(i, str(e), e.stderr_tail)
                continue
            except Exception as e:
                self.job_failed.emit(i, str(e), "")
                continue
            finally:
                self._release_handle(i)
                self._release_slot()
            self.job_finished.emit(i, result)
