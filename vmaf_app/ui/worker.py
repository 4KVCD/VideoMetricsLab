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


class VmafWorker(QThread):
    job_started = Signal(int, str)          # job_index, label
    progress = Signal(int, int, int, float) # job_index, current_frame, total_frames, fps
    status = Signal(int, str)               # job_index, status text
    job_finished = Signal(int, object)      # job_index, VmafRunResult
    job_failed = Signal(int, str, str)      # job_index, message, stderr_tail
    all_finished = Signal()

    def __init__(self, jobs: list[VmafJob], parent=None):
        super().__init__(parent)
        self._jobs = jobs
        self._cancel_event = threading.Event()
        self._process_handle = ProcessHandle()

    def cancel(self) -> None:
        self._cancel_event.set()
        # Wakes up a paused process too, rather than only relying on the
        # (blocked, since a paused process has no more output) reader loop
        # to notice the cancel flag on its own -- see ProcessHandle.terminate.
        self._process_handle.terminate()

    def pause(self) -> None:
        self._process_handle.pause()

    def resume(self) -> None:
        self._process_handle.resume()

    @property
    def is_paused(self) -> bool:
        return self._process_handle.is_pause_requested

    def run(self) -> None:
        for i, job in enumerate(self._jobs):
            if self._cancel_event.is_set():
                break
            self.job_started.emit(i, job.label)
            try:
                if job.options.resample_test is not None:
                    result = run_resample_test(
                        job.source_info,
                        job.options,
                        on_progress=lambda cur, tot, fps, idx=i: self.progress.emit(idx, cur, tot, fps),
                        on_status=lambda msg, idx=i: self.status.emit(idx, msg),
                        cancel_event=self._cancel_event,
                        process_handle=self._process_handle,
                    )
                else:
                    result = run_vmaf(
                        job.source_info,
                        job.distorted_info,
                        job.options,
                        on_progress=lambda cur, tot, fps, idx=i: self.progress.emit(idx, cur, tot, fps),
                        on_status=lambda msg, idx=i: self.status.emit(idx, msg),
                        cancel_event=self._cancel_event,
                        process_handle=self._process_handle,
                        result_distorted_path=job.result_distorted_path,
                    )
            except Cancelled:
                break
            except VmafRunError as e:
                self.job_failed.emit(i, str(e), e.stderr_tail)
                continue
            except Exception as e:
                self.job_failed.emit(i, str(e), "")
                continue
            self.job_finished.emit(i, result)
        self.all_finished.emit()
