"""The background file-write queue, and the thing it exists to prevent:
the UI thread stopping while a large result is serialised.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tests.factories import (
    fake_completed_run as _fake_completed_run,
)
from tests.factories import (
    fake_video_info as _fake_video_info,
)
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


def test_recompute_is_ordered_after_a_pending_cache_store(qapp, monkeypatch):
    """A clear issued while store is running must be last, or the supposedly
    ignored result is recreated as soon as the background write finishes."""
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
