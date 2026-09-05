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


#: Above this, jobs mostly contend rather than overlap: each one already asks
#: libvmaf for every core, and every extra job adds another decode reading
#: from the same disk.
MAX_PARALLEL_JOBS = 4


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
        self._paused = False
        self._cancellation_reported = False

    # ------------------------------------------------------------- controls
    def cancel(self) -> None:
        self._cancel_event.set()
        # Wakes up a paused process too, rather than only relying on the
        # (blocked, since a paused process has no more output) reader loop
        # to notice the cancel flag on its own -- see ProcessHandle.terminate.
        for handle in self._live_handles():
            handle.terminate()

    def pause(self) -> None:
        with self._handles_lock:
            self._paused = True
            handles = list(self._handles.values())
        for handle in handles:
            handle.pause()

    def resume(self) -> None:
        with self._handles_lock:
            self._paused = False
            handles = list(self._handles.values())
        for handle in handles:
            handle.resume()

    @property
    def is_paused(self) -> bool:
        with self._handles_lock:
            return self._paused

    def _live_handles(self) -> list[ProcessHandle]:
        with self._handles_lock:
            return list(self._handles.values())

    def _claim_handle(self, index: int) -> ProcessHandle:
        """A handle for one job, already paused if the run is paused.

        A job that starts while the user has the run paused must come up
        paused as well; otherwise pressing Pause and waiting would quietly
        let the next video start running at full speed.
        """
        handle = ProcessHandle()
        with self._handles_lock:
            self._handles[index] = handle
            paused = self._paused
        if paused:
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
        if self._parallel_jobs <= 1 or len(self._jobs) <= 1:
            self._run_jobs(iter(range(len(self._jobs))))
        else:
            # Jobs are pulled from one shared iterator rather than dealt out
            # in advance: they finish at wildly different speeds (a 90-minute
            # feature next to a 20-second clip), and a fixed split would leave
            # a lane idle while another still had work queued.
            pending = iter(range(len(self._jobs)))
            lanes = [
                threading.Thread(
                    target=self._run_jobs, args=(pending,),
                    name=f"vmaf-lane-{n}", daemon=True,
                )
                for n in range(min(self._parallel_jobs, len(self._jobs)))
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
            job = self._jobs[i]
            handle = self._claim_handle(i)
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
            self.job_finished.emit(i, result)
