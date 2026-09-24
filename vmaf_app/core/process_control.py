"""A thread-safe handle for pausing/resuming the ffmpeg subprocesses of one
job.

This is an OS-level suspend (like Task Manager's "Suspend"), not a
checkpoint: the process is frozen in place and CPU/GPU usage drops to zero,
but it must stay alive in memory to be resumed -- closing the app or
rebooting loses the run, same as a hard cancel would.

One handle can address several processes at once. A job is usually one
ffmpeg, but black-bar detection runs its sample windows as a handful of
short-lived processes concurrently, and Pause or Cancel pressed during that
moment has to reach every one of them -- a handle that remembered only the
most recent pid left the others running.
"""
from __future__ import annotations

import contextlib
import threading

import psutil

from vmaf_app.core.proc import signal_tree


class ProcessHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: set[int] = set()
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
        # Suspended under the lock, not after releasing it: a resume landing
        # in that gap would run first and this stale suspend afterwards,
        # leaving the process paused for good with nothing left to resume it.
        with self._lock:
            self._pids.add(pid)
            if self._want_paused:
                self._try(pid, "suspend")

    def detach(self, pid: int | None = None) -> None:
        """Forgets one process, or every process when no pid is given.

        Concurrent callers must name their own pid: a bare detach() from one
        of several windows would drop the rest from reach of Pause and
        Cancel while they were still running.
        """
        with self._lock:
            if pid is None:
                self._pids.clear()
            else:
                self._pids.discard(pid)

    def pause(self) -> None:
        with self._lock:
            self._want_paused = True
            pids = list(self._pids)
        for pid in pids:
            self._try(pid, "suspend")

    def resume(self) -> None:
        with self._lock:
            self._want_paused = False
            pids = list(self._pids)
        for pid in pids:
            self._try(pid, "resume")

    def terminate(self) -> None:
        """Kills every attached process, regardless of pause state. A
        suspended process blocks its own stdout reader forever (no more
        output is ever coming), so cancelling a paused run has to reach in
        and kill it directly rather than waiting for the reader loop to
        notice a cancellation flag that will never get checked."""
        with self._lock:
            pids = list(self._pids)
            self._terminated = True
        for pid in pids:
            self._try(pid, "terminate")

    @property
    def is_pause_requested(self) -> bool:
        with self._lock:
            return self._want_paused

    @staticmethod
    def _try(pid: int, action: str) -> None:
        # The process finishing on its own between the check and the call
        # is normal, not an error -- pausing/resuming a dead process is a no-op.
        # The whole tree: FFmpeg can be a child of the process started, when
        # ffmpeg.exe is a launcher such as a Chocolatey shim (see
        # proc.process_tree), and pausing only the launcher paused nothing.
        with contextlib.suppress(psutil.NoSuchProcess):
            signal_tree(pid, action)
