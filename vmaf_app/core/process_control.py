"""A thread-safe handle for pausing, resuming and ending the FFmpeg processes
of one job.

Pausing is an operating-system suspend (like Task Manager's "Suspend"), not
a checkpoint: the process is frozen in place and its CPU and GPU use drop to
zero, but it must stay in memory to be resumed. Closing the app or
restarting the PC loses the run, as a Cancel would.

One handle can reach several processes at once. A job is usually one
FFmpeg, but black-bar detection runs a handful of short-lived processes at
the same time, and Pause or Cancel pressed then must reach every one of
them.
"""
from __future__ import annotations

import contextlib
import threading

import psutil

from vmaf_app.core.proc import signal_tree


class ProcessHandle:
    """The processes of one job, and whether the user has asked for them to
    be paused or ended."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: set[int] = set()
        self._want_paused = False
        # Set when terminate() is called, so that a non-zero exit can be told
        # apart from a real failure of the tool.
        self._terminated = False

    @property
    def was_terminated(self) -> bool:
        """Whether terminate() has been called (the user cancelled)."""
        with self._lock:
            return self._terminated

    def attach(self, pid: int) -> None:
        """Adds a newly started process to the handle.

        A pause requested before the process existed (at the start of a job,
        say, or during the CPU retry after a GPU decode failure) is applied
        to it. A process that starts after terminate() is ended at once:
        Cancel has been given, and otherwise the next attempt of a fallback
        would start paused, with nothing left to end it, and the run would
        wait for it forever."""
        # The suspend happens under the lock, not after releasing it: a
        # resume landing in between would run first and this stale suspend
        # afterwards, leaving the process paused with nothing to resume it.
        # terminate() sets its flag under the lock too, so a process is
        # either among those terminate() ends or is ended here.
        with self._lock:
            self._pids.add(pid)
            if self._terminated:
                self._try(pid, "terminate")
            elif self._want_paused:
                self._try(pid, "suspend")

    def detach(self, pid: int | None = None) -> None:
        """Forgets one process, or every process when no pid is given.

        Callers that run at the same time must name their own pid: a bare
        detach() from one of them would put the others out of reach of
        Pause and Cancel while they are still running.
        """
        with self._lock:
            if pid is None:
                self._pids.clear()
            else:
                self._pids.discard(pid)

    def pause(self) -> None:
        """Suspends every attached process, and any attached later."""
        with self._lock:
            self._want_paused = True
            pids = list(self._pids)
        for pid in pids:
            self._try(pid, "suspend")

    def resume(self) -> None:
        """Resumes every attached process."""
        with self._lock:
            self._want_paused = False
            pids = list(self._pids)
        for pid in pids:
            self._try(pid, "resume")

    def terminate(self) -> None:
        """Kills every attached process, paused or not. A suspended process
        blocks the thread reading its output forever (no more output is
        coming), so a paused run is cancelled by killing its processes
        directly, not by waiting for the reader to check a flag."""
        with self._lock:
            pids = list(self._pids)
            self._terminated = True
        for pid in pids:
            self._try(pid, "terminate")

    @property
    def is_pause_requested(self) -> bool:
        """Whether the user has paused the job."""
        with self._lock:
            return self._want_paused

    @staticmethod
    def _try(pid: int, action: str) -> None:
        """Applies `action` ("suspend", "resume" or "terminate") to the
        process and everything under it."""
        # A process can finish on its own between the check and the call;
        # that is normal, so acting on a process that is gone does nothing.
        # The whole tree is acted on: when ffmpeg.exe is a launcher (such as
        # a Chocolatey shim, see proc.process_tree), the real FFmpeg is its
        # child, and pausing only the launcher would pause nothing.
        with contextlib.suppress(psutil.NoSuchProcess):
            signal_tree(pid, action)
