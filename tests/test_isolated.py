"""vmaf_app.core.isolated: native GPU code runs in a process of its own.

The targets are module-level: the child imports them from here."""
import faulthandler
import logging
import subprocess
import threading
import time

import psutil
import pytest

from tests.factories import STDLIB_PYTHON
from vmaf_app.core.isolated import IsolatedCrashError, run_isolated
from vmaf_app.core.perceptual_cpu import PerceptualCancelled
from vmaf_app.core.vmaf_runner import VmafRunError


def _report(value, on_status=None):
    logging.getLogger("vmaf_app.test_child").warning("working on %s", value)
    on_status("first")
    on_status("second")
    return value * 2


def _log_info():
    logging.getLogger("vmaf_app.core.test_child").info("FFmpeg: ffmpeg -i a.mkv")


def test_the_childs_info_lines_reach_the_apps_log(monkeypatch):
    """The app's log is at INFO on the vmaf_app logger, the root logger at
    WARNING: the child was given the root's level and dropped every INFO
    line -- its FFmpeg commands and a pass's timing never reached the log."""
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    app = logging.getLogger("vmaf_app")
    handler = Keep()
    app.addHandler(handler)
    monkeypatch.setattr(app, "level", logging.INFO)
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    try:
        run_isolated(_log_info, what="test")
    finally:
        app.removeHandler(handler)
    assert "FFmpeg: ffmpeg -i a.mkv" in records


def _fail():
    raise VmafRunError("libvmaf failed", "the last lines FFmpeg wrote")


def _start_and_wait(process_handle=None, cancel_event=None):
    child = subprocess.Popen([STDLIB_PYTHON, "-S", "-c", "import time; time.sleep(60)"])
    process_handle.attach(child.pid)
    time.sleep(60)


def test_a_crash_in_the_child_is_an_error_here(caplog, tmp_path, monkeypatch):
    """An access violation in Vship or the GPU driver ended the app. Ending
    the child instead, it left no trace of where: its threads at the crash
    are now logged with the error, and the file they were written to is
    removed."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with caplog.at_level(logging.ERROR, logger="vmaf_app.core.isolated"),             pytest.raises(IsolatedCrashError, match="Vship crashed"):
        run_isolated(faulthandler._sigsegv, what="Vship")
    assert "its threads at the crash" in caplog.text
    assert "_child_main" in caplog.text  # the crashed thread's Python stack
    assert list(tmp_path.glob("vml-isolated-*")) == []


def test_a_child_that_ends_normally_leaves_no_crash_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    assert run_isolated(_report, 21, what="test", callbacks=("on_status",), on_status=lambda _text: None) == 42
    assert list(tmp_path.glob("vml-isolated-*")) == []


def test_the_result_callbacks_logs_and_errors_come_back(caplog):
    statuses = []
    with caplog.at_level(logging.WARNING):
        assert run_isolated(_report, 21, what="test", callbacks=("on_status",), on_status=statuses.append) == 42
    assert statuses == ["first", "second"]
    assert "working on 21" in caplog.text
    # What __init__ set beyond the message survives: the stderr tail a
    # failed video shows.
    with pytest.raises(VmafRunError, match="libvmaf failed") as raised:
        run_isolated(_fail, what="test")
    assert raised.value.stderr_tail == "the last lines FFmpeg wrote"


def test_cancel_ends_the_child_and_what_it_started():
    """The child's FFmpeg is attached to the caller's handle, so Pause and
    Cancel reach it; Cancel also ends the child."""
    attached = []

    class Handle:
        def attach(self, pid):
            attached.append(pid)

        def detach(self, pid=None):
            pass

    cancel = threading.Event()
    threading.Thread(target=lambda: (_wait_for(lambda: attached), cancel.set()), daemon=True).start()
    with pytest.raises(PerceptualCancelled):
        run_isolated(_start_and_wait, what="test", process_handle=Handle(), cancel_event=cancel,
                     cancelled=PerceptualCancelled)
    assert _wait_for(lambda: not psutil.pid_exists(attached[0])), "the child's process kept running"


def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False
