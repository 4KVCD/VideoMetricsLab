"""The background file-write queue, and the thing it exists to prevent:
the UI thread stopping while a large result is serialised.
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.ui.file_worker import FileWriteQueue


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_ui_thread_keeps_processing_events_during_a_slow_write(qapp):
    """THE point of the queue. A feature-length result is seconds of JSON
    serialisation, and doing it in the handler that finishes a run froze the
    window -- no repaint, no drag, no cancel -- exactly when the user was
    watching for the result.
    """
    queue = FileWriteQueue()
    release = threading.Event()
    queue.submit("slow write", release.wait)

    # While that write is blocked, the UI thread must still be able to run.
    ticks = 0
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and ticks < 50:
        QApplication.processEvents()
        ticks += 1

    assert ticks >= 50, "the UI thread was blocked by the write"
    assert queue.pending == 1, "the write finished early; it was not really blocking"

    release.set()
    assert queue.wait_until_idle(10.0)


def test_writes_run_in_the_order_they_were_submitted(qapp):
    queue = FileWriteQueue()
    order: list[int] = []
    for i in range(10):
        queue.submit(f"write {i}", lambda i=i: order.append(i))

    assert queue.wait_until_idle(10.0)
    assert order == list(range(10))


def test_cyclic_gc_and_runnable_cleanup_return_to_the_gui_thread(qapp):
    queue = FileWriteQueue()
    gc_state_in_worker: list[bool] = []

    queue.submit("allocation-heavy write", lambda: gc_state_in_worker.append(gc.isenabled()))

    assert queue.wait_until_idle(10.0)
    assert gc_state_in_worker == [False]
    assert gc.isenabled(), "automatic collection was not restored after the write"
    assert not queue._tasks, "the GUI-thread completion did not release the runnable"


def test_a_failing_write_is_reported_and_does_not_stop_the_queue(qapp):
    # A QRunnable that raises takes its exception nowhere useful, so a
    # failure has to come back as a signal -- and must not take the rest of
    # the batch with it.
    queue = FileWriteQueue()
    failures: list[tuple[str, str]] = []
    queue.write_failed.connect(lambda d, e: failures.append((d, e)))
    done: list[str] = []

    def boom():
        raise OSError("disk full")

    queue.submit("bad write", boom)
    queue.submit("good write", lambda: done.append("ok"))

    assert queue.wait_until_idle(10.0)
    QApplication.processEvents()  # deliver the queued signal

    assert done == ["ok"], "one failure aborted the rest of the batch"
    assert failures and failures[0][0] == "bad write"
    assert "disk full" in failures[0][1]


def test_idle_is_reported_only_once_everything_has_finished(qapp):
    queue = FileWriteQueue()
    release = threading.Event()
    idle_signals: list[bool] = []
    queue.became_idle.connect(lambda: idle_signals.append(True))
    queue.submit("first", release.wait)
    queue.submit("second", lambda: None)

    assert not queue.wait_until_idle(0.2), "reported idle with work outstanding"
    release.set()
    assert queue.wait_until_idle(10.0)
    assert queue.pending == 0
    assert idle_signals == [True]


def test_a_queue_with_nothing_submitted_is_already_idle(qapp):
    assert FileWriteQueue().wait_until_idle(0.1)


def test_finishing_a_run_does_not_block_the_window(qapp, tmp_path, monkeypatch):
    """The end-to-end version: the handler that a finished run lands in must
    return promptly even when the cache write is slow."""
    from tests.test_main_window import _fake_completed_run, _fake_video_info
    from vmaf_app.core import result_cache
    from vmaf_app.ui.main_window import MainWindow

    source = tmp_path / "source.mp4"
    distorted = tmp_path / "distorted.mp4"
    for path in (source, distorted):
        path.write_bytes(b"x")

    started = threading.Event()
    release = threading.Event()

    def slow_store(*args, **kwargs):
        started.set()
        release.wait(10.0)

    monkeypatch.setattr(result_cache, "store", slow_store)

    win = MainWindow()
    win._source_info = _fake_video_info(str(source))
    row = win._add_table_row(distorted)
    win._job_rows = [win._rows[row]]
    result = _fake_completed_run(str(distorted)).result
    result.source = source
    result.distorted = distorted

    began = time.monotonic()
    win._on_job_finished(0, result)
    elapsed = time.monotonic() - began

    assert started.wait(5.0), "the cache write never started"
    assert elapsed < 1.0, (
        f"_on_job_finished blocked for {elapsed:.2f}s waiting on the cache write"
    )
    # And the row was still updated, rather than the result being deferred
    # along with the write.
    assert win._rows[row].completed_run is not None

    release.set()
    assert win._file_writes.wait_until_idle(10.0)


def test_recompute_is_ordered_after_a_pending_cache_store(qapp, monkeypatch):
    """A clear issued while store is running must be last, or the supposedly
    ignored result is recreated as soon as the background write finishes."""
    from tests.test_main_window import _fake_completed_run, _fake_video_info
    from vmaf_app.core import result_cache
    from vmaf_app.ui.main_window import MainWindow

    started = threading.Event()
    release = threading.Event()
    order = []

    def slow_store(*args, **kwargs):
        started.set()
        release.wait(10.0)
        order.append("store")

    monkeypatch.setattr(result_cache, "store", slow_store)
    monkeypatch.setattr(result_cache, "clear", lambda *a, **k: order.append("clear"))

    win = MainWindow()
    win._source_info = _fake_video_info("source.mp4")
    row = win._add_table_row(Path("distorted.mp4"))
    win._job_rows = [win._rows[row]]
    win._on_job_finished(0, _fake_completed_run("distorted.mp4").result)
    assert started.wait(5.0)

    win._recompute_rows([row])
    release.set()
    assert win._file_writes.wait_until_idle(10.0)

    assert order == ["store", "clear"]
