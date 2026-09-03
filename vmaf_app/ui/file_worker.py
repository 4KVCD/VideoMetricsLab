"""Runs file writes off the UI thread.

A feature-length run is hundreds of thousands of frames, and serialising
one is seconds of work: a cached result is ~9MB of JSON, and a CSV export
of the same run is larger still. Doing that inside the handler that
finishes a run froze the window solid -- no repaint, no drag, no cancel --
at exactly the moment the user was watching for a result.

Writes go through a pool of ONE thread rather than the global pool, for
two reasons: they stay in submission order, and they cannot starve
anything else Qt is running in the background.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _WriteTask(QRunnable):
    def __init__(self, queue: FileWriteQueue, description: str, write: Callable[[], None]):
        super().__init__()
        self._queue = queue
        self._description = description
        self._write = write

    def run(self) -> None:  # runs on the pool's thread
        try:
            self._write()
        except Exception as e:
            # A QRunnable that raises takes the exception nowhere useful, so
            # failures are reported through a signal instead.
            self._queue._write_failed(self._description, str(e))
        finally:
            self._queue._task_done()


class FileWriteQueue(QObject):
    """Serialises file writes onto one background thread.

    The callables submitted here MUST NOT touch widgets: they run on
    another thread. Give them plain data (a result, a path, a label) and
    let the signals below carry the outcome back.
    """

    #: (description, error message) -- one per failed write.
    write_failed = Signal(str, str)
    #: Emitted when the last outstanding write finishes.
    became_idle = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # One thread, so writes land in the order they were asked for and a
        # long export cannot occupy every core Qt might want.
        self._pool.setMaxThreadCount(1)
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, description: str, write: Callable[[], None]) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()
        self._pool.start(_WriteTask(self, description, write))

    def wait_until_idle(self, timeout_seconds: float = 30.0) -> bool:
        """Blocks until every submitted write has finished.

        For shutdown -- closing the window while a cache write is in flight
        would lose the result -- and for tests, which need the file to exist
        before they can assert anything about it.
        """
        return self._idle.wait(timeout_seconds)

    # -------------------------------------------------- called from the pool
    def _write_failed(self, description: str, error: str) -> None:
        self.write_failed.emit(description, error)

    def _task_done(self) -> None:
        with self._lock:
            self._pending -= 1
            finished = self._pending == 0
            if finished:
                self._idle.set()
        if finished:
            self.became_idle.emit()


__all__ = ["FileWriteQueue"]
