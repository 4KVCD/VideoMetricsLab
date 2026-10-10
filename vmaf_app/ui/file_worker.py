"""Runs file writes (saved results, exports) off the window's thread.

A feature-length run has hundreds of thousands of frames, and writing one
out takes seconds: a saved result is about 9 MB of JSON, and a CSV export of
the same run is larger still. Done in the handler that finishes a run, it
would freeze the window (no repainting, dragging or cancelling) at the very
moment the user is watching for a result.

The writes run on one thread rather than Qt's shared pool, for two
reasons: they stay in the order they were submitted, and they cannot crowd
out anything else Qt runs in the background.
"""
from __future__ import annotations

import collections
import gc
import threading
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal, Slot

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


def _resume_automatic_gc() -> bool:
    """Whether this ended the last suspension and turned collection back on."""
    global _gc_suspensions, _gc_was_enabled
    should_restore = False
    with _gc_lock:
        _gc_suspensions -= 1
        if _gc_suspensions == 0:
            should_restore = _gc_was_enabled
            _gc_was_enabled = False
    if should_restore:
        gc.enable()
    return should_restore


def _resume_automatic_gc_on_gui_thread() -> None:
    """Turns automatic garbage collection back on, and does the collection it would have done, on the window's thread."""
    if _resume_automatic_gc():
        # Automatic collection would have selected generation zero. Do that
        # work now, on the thread that owns the application's Qt wrappers.
        gc.collect(0)


class FileWriteQueue(QObject):
    """Serialises file writes onto one background thread.

    The callables submitted here MUST NOT touch widgets: they run on
    another thread. Give them plain data (a result, a path, a label) and
    let the signals below carry the outcome back.
    """

    #: (description, error message), sent once per failed write.
    write_failed = Signal(str, str)
    #: Emitted when the last outstanding write finishes.
    became_idle = Signal()
    _write_finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._writes: collections.deque[tuple[str, Callable[[], None]]] = collections.deque()
        # Written, for the GUI thread to finish: (write, description, error).
        self._written: collections.deque[tuple[Callable[[], None], str, str]] = collections.deque()
        # A thread is taking writes from self._writes; it ends when they run out.
        self._writing = False
        self._pending = 0  # submitted, not yet written
        self._unfinished = 0  # submitted, not yet finished on the GUI thread
        self._idle = threading.Event()
        self._idle.set()
        self._write_finished.connect(self._finish_on_gui_thread, Qt.QueuedConnection)

    @property
    def pending(self) -> int:
        """How many writes have not been written yet."""
        with self._lock:
            return self._pending

    def submit(self, description: str, write: Callable[[], None]) -> None:
        """Queues a write, to run after those already queued."""
        with self._lock:
            self._writes.append((description, write))
            self._pending += 1
            self._unfinished += 1
            self._idle.clear()
            if self._writing:
                return
            self._writing = True
        # A thread of its own, not a QThreadPool's. PySide makes the pool the
        # owner of each QRunnable started on it, and one that does not delete
        # itself after running (these must not, on the pool's thread) is never
        # freed, nor is the write it holds with its result: every saved result
        # and export would stay in memory for as long as the app ran.
        threading.Thread(target=self._write_all, name="File writes", daemon=True).start()

    def wait_until_idle(self, timeout_seconds: float = 30.0) -> bool:
        """Blocks until every submitted write has finished.

        It is for shutdown, because closing the window during a write would
        lose the result, and for tests, which need the file to exist before
        they can check anything about it.
        """
        finished = self._idle.wait(timeout_seconds)
        if finished and QCoreApplication.instance() is not None:
            # Completion is deliberately queued to the GUI thread. Tests and
            # shutdown call this method while that event loop is not turning,
            # so deliver the queued cleanup before reporting that the queue is
            # fully idle.
            QCoreApplication.processEvents()
        return finished

    # ---------------------------------------------- on the writing thread
    def _write_all(self) -> None:
        """The writing thread: runs the queued writes in turn until there are none."""
        while True:
            with self._lock:
                if not self._writes:
                    self._writing = False
                    return
                description, write = self._writes.popleft()
            _suspend_automatic_gc()
            error = ""
            try:
                write()
            except Exception as e:
                error = str(e)
            # Handed to the GUI thread, which drops the last reference to the
            # write and the result it holds: this thread lets go of it first.
            with self._lock:
                self._written.append((write, description, error))
            del write
            # Queue GUI-thread cleanup before exposing the idle event. A waiter
            # that wakes can then process the already-posted completion safely.
            try:
                self._write_finished.emit()
            except RuntimeError:
                # The window that owned this queue is gone: nothing to tell it.
                _resume_automatic_gc()
            with self._lock:
                self._pending -= 1
                if self._pending == 0:
                    self._idle.set()

    @Slot()
    def _finish_on_gui_thread(self) -> None:
        """Finishes one write on the window's thread: drops it, and reports a failure."""
        with self._lock:
            write, description, error = self._written.popleft()
            self._unfinished -= 1
            finished = self._unfinished == 0
        del write  # the last reference to it
        if error:
            self.write_failed.emit(description, error)
        _resume_automatic_gc_on_gui_thread()
        if finished:
            self.became_idle.emit()


__all__ = ["FileWriteQueue"]
