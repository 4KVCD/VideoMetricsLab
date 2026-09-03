"""A thread-safe handle for pausing/resuming the currently running ffmpeg
subprocess.

This is an OS-level suspend (like Task Manager's "Suspend"), not a
checkpoint: the process is frozen in place and CPU/GPU usage drops to zero,
but it must stay alive in memory to be resumed -- closing the app or
rebooting loses the run, same as a hard cancel would.
"""
from __future__ import annotations

import contextlib
import threading

import psutil


class ProcessHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._want_paused = False
        # Set when terminate() is called, so a non-zero exit can be told
        # apart from a genuine failure of the tool.
        self._terminated = False

    @property
    def was_terminated(self) -> bool:
        with self._lock:
            return self._terminated

    def attach(self, pid: int) -> None:
        """Called when a new ffmpeg process starts; re-applies a pause
        request made before this process existed (e.g. right at a job
        boundary, or during the GPU-decode-failure CPU retry)."""
        with self._lock:
            self._pid = pid
            want_paused = self._want_paused
        if want_paused:
            self._try(pid, "suspend")

    def detach(self) -> None:
        with self._lock:
            self._pid = None

    def pause(self) -> None:
        with self._lock:
            self._want_paused = True
            pid = self._pid
        if pid is not None:
            self._try(pid, "suspend")

    def resume(self) -> None:
        with self._lock:
            self._want_paused = False
            pid = self._pid
        if pid is not None:
            self._try(pid, "resume")

    def terminate(self) -> None:
        """Kills whatever process is currently attached, regardless of pause
        state. A suspended process blocks its own stdout reader forever
        (no more output is ever coming), so cancelling a paused run has to
        reach in and kill it directly rather than waiting for the reader
        loop to notice a cancellation flag that will never get checked."""
        with self._lock:
            pid = self._pid
            self._terminated = True
        if pid is not None:
            self._try(pid, "terminate")

    @property
    def is_pause_requested(self) -> bool:
        with self._lock:
            return self._want_paused

    @staticmethod
    def _try(pid: int, action: str) -> None:
        # The process finishing on its own between the check and the call
        # is normal, not an error -- pausing/resuming a dead process is a no-op.
        with contextlib.suppress(psutil.NoSuchProcess):
            getattr(psutil.Process(pid), action)()
