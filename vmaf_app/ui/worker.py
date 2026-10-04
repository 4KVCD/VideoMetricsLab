"""The window's run of VMAF jobs: core.job_runner's JobScheduler on a
QThread of its own, its events sent on as Qt signals."""
from __future__ import annotations

from dataclasses import fields

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.job_runner import JobScheduler, RunEvents, VmafJob


class VmafWorker(QThread):
    """The signals are RunEvents' (see there), emitted from the run's lanes:
    a connection to a QObject in the window's thread is queued."""

    job_started = Signal(int, str)
    progress = Signal(int, int, int, float)
    task_progress = Signal(int, object)
    # object, not str: a str signal would send a core.status.Status on as a
    # plain str, without its decode plan and kind.
    status = Signal(int, object)
    job_finished = Signal(int, object)
    job_failed = Signal(int, str, str)
    job_partially_failed = Signal(int, object, str, str, object)
    result_updated = Signal(int, object)
    cancelled = Signal()
    all_finished = Signal()

    def __init__(self, jobs: list[VmafJob], parallel_jobs: int = 1, parent=None, *,
                 gpu_metrics_together: bool = False):
        super().__init__(parent)
        events = RunEvents(**{event.name: getattr(self, event.name).emit for event in fields(RunEvents)})
        self.scheduler = JobScheduler(jobs, parallel_jobs, events, gpu_metrics_together=gpu_metrics_together)

    def run(self) -> None:
        self.scheduler.run()

    def cancel(self) -> None:
        self.scheduler.cancel()

    def pause(self) -> None:
        self.scheduler.pause()

    def resume(self) -> None:
        self.scheduler.resume()

    @property
    def is_paused(self) -> bool:
        return self.scheduler.is_paused

    @property
    def parallel_jobs(self) -> int:
        return self.scheduler.parallel_jobs

    def set_parallel_jobs(self, count: int) -> None:
        self.scheduler.set_parallel_jobs(count)
