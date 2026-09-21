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

import gc
import threading
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, QRunnable, Qt, QThreadPool, Signal, Slot

_gc_lock = threading.Lock()
_gc_suspensions = 0
_gc_was_enabled = False


def _suspend_automatic_gc() -> None:
    """Keep cyclic GC out of background threads that coexist with Qt.

    Reference counting remains active. The last completed write restores
    automatic collection on the GUI thread and performs the deferred young-
    generation collection there.
    """
    global _gc_suspensions, _gc_was_enabled
    with _gc_lock:
        if _gc_suspensions == 0:
            _gc_was_enabled = gc.isenabled()
            if _gc_was_enabled:
                gc.disable()
        _gc_suspensions += 1


def _resume_automatic_gc_on_gui_thread() -> None:
    global _gc_suspensions, _gc_was_enabled
    should_restore = False
    with _gc_lock:
        _gc_suspensions -= 1
        if _gc_suspensions == 0:
            should_restore = _gc_was_enabled
            _gc_was_enabled = False
    if should_restore:
        gc.enable()
        # Automatic collection would have selected generation zero. Do that
        # work now, on the thread that owns the application's Qt wrappers.
        gc.collect(0)


class _WriteTask(QRunnable):
    def __init__(self, queue: FileWriteQueue, description: str, write: Callable[[], None]):
        super().__init__()
        self._queue = queue
        self._description = description
        self._write = write
        # PySide must not destroy this Python-owned wrapper on the pool
        # thread. FileWriteQueue retains it until the GUI-thread completion
        # slot runs, then releasing the final reference is safe.
        self.setAutoDelete(False)

    def run(self) -> None:  # runs on the pool's thread
        _suspend_automatic_gc()
        error = ""
        try:
            self._write()
        except Exception as e:
            error = str(e)
        finally:
            self._queue._worker_done(self, self._description, error)


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
    _task_finished = Signal(object, str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # One thread, so writes land in the order they were asked for and a
        # long export cannot occupy every core Qt might want.
        self._pool.setMaxThreadCount(1)
        self._lock = threading.Lock()
        self._pending = 0
        self._tasks: set[_WriteTask] = set()
        self._idle = threading.Event()
        self._idle.set()
        self._task_finished.connect(self._finish_on_gui_thread, Qt.QueuedConnection)

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, description: str, write: Callable[[], None]) -> None:
        task = _WriteTask(self, description, write)
        with self._lock:
            self._pending += 1
            self._tasks.add(task)
            self._idle.clear()
        self._pool.start(task)

    def wait_until_idle(self, timeout_seconds: float = 30.0) -> bool:
        """Blocks until every submitted write has finished.

        For shutdown -- closing the window while a cache write is in flight
        would lose the result -- and for tests, which need the file to exist
        before they can assert anything about it.
        """
        finished = self._idle.wait(timeout_seconds)
        if finished and QCoreApplication.instance() is not None:
            # Completion is deliberately queued to the GUI thread. Tests and
            # shutdown call this method while that event loop is not turning,
            # so deliver the queued cleanup before reporting that the queue is
            # fully idle.
            QCoreApplication.processEvents()
        return finished

    # -------------------------------------------------- called from the pool
    def _worker_done(self, task: _WriteTask, description: str, error: str) -> None:
        # Queue GUI-thread cleanup before exposing the idle event. A waiter
        # that wakes can then process the already-posted completion safely.
        self._task_finished.emit(task, description, error)
        with self._lock:
            self._pending -= 1
            if self._pending == 0:
                self._idle.set()

    @Slot(object, str, str)
    def _finish_on_gui_thread(self, task: _WriteTask, description: str, error: str) -> None:
        if error:
            self.write_failed.emit(description, error)
        self._tasks.discard(task)
        _resume_automatic_gc_on_gui_thread()
        with self._lock:
            finished = self._pending == 0 and not self._tasks
        if finished:
            self.became_idle.emit()


__all__ = ["FileWriteQueue"]
