"""vmaf_app.core.isolated: native GPU code runs in a process of its own.

The targets are module-level: the child imports them from here."""
import faulthandler
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

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


def _start_and_wait(process_handle=None, cancel_event=None, on_status=None):
    # Like a Vship pass's FFmpeg: a program the child started, writing to a
    # temporary file the child made.
    descriptor, name = tempfile.mkstemp(prefix="vml-vship-pts-", suffix=".txt")
    os.close(descriptor)
    child = subprocess.Popen(
        [STDLIB_PYTHON, "-S", "-c", "import sys, time; f = open(sys.argv[1], 'w'); print(flush=True); time.sleep(60)",
         name], stdout=subprocess.PIPE)
    child.stdout.readline()  # the file is open
    on_status(name)
    process_handle.attach(child.pid)
    time.sleep(60)


def _crash_after_a_temporary_file(on_status=None):
    _descriptor, name = tempfile.mkstemp(prefix="vml-vship-pts-", suffix=".txt")
    on_status(name)
    faulthandler._sigsegv()


def test_a_crash_in_the_child_is_an_error_here(caplog, tmp_path, monkeypatch):
    """An access violation in Vship or the GPU driver ended the app. Ending
    the child instead, it left no trace of where: its threads at the crash
    are now logged with the error. The file they were written to is
    removed, and so is every temporary file the child made: a crashed Vship
    pass left its files of FFmpeg's timestamps in the temporary folder."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    made = []
    with (caplog.at_level(logging.ERROR, logger="vmaf_app.core.isolated"),
          pytest.raises(IsolatedCrashError, match="Vship crashed")):
        run_isolated(_crash_after_a_temporary_file, what="Vship", callbacks=("on_status",), on_status=made.append)
    assert "its threads at the crash" in caplog.text
    assert "_child_main" in caplog.text  # the crashed thread's Python stack
    assert made and not Path(made[0]).exists()
    assert list(tmp_path.glob("vml-*")) == []


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


def test_cancel_ends_the_child_and_what_it_started(tmp_path, monkeypatch):
    """The child's FFmpeg is attached to the caller's handle, so Pause and
    Cancel reach it; Cancel also ends the child. The child's temporary
    files are removed, even one its FFmpeg still has open: a cancelled GPU
    pass left the file for its crash report in the temporary folder, and a
    Vship pass its files of FFmpeg's timestamps."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    attached, made = [], []

    class Handle:
        def attach(self, pid):
            attached.append(pid)

        def detach(self, pid=None):
            pass

    cancel = threading.Event()
    threading.Thread(target=lambda: (_wait_for(lambda: attached), cancel.set()), daemon=True).start()
    with pytest.raises(PerceptualCancelled):
        run_isolated(_start_and_wait, what="test", callbacks=("on_status",), on_status=made.append,
                     process_handle=Handle(), cancel_event=cancel, cancelled=PerceptualCancelled)
    assert _wait_for(lambda: not psutil.pid_exists(attached[0])), "the child's process kept running"
    assert made and not Path(made[0]).exists()
    assert list(tmp_path.glob("vml-*")) == []


def _wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False
