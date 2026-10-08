import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from vmaf_app.core import app_log


@pytest.fixture
def session_log(tmp_path, monkeypatch):
    # The hooks start_logging chains to: no-ops, so pytest's own thread
    # exception hook does not report the deliberate test exceptions.
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    path = app_log.start_logging(tmp_path)
    yield path
    app_log.stop_logging()


def _text(path) -> str:
    for handler in logging.getLogger(app_log.LOGGER_NAME).handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_the_log_goes_to_the_folder_given_and_keeps_failures(session_log, tmp_path):
    assert session_log == tmp_path / "VideoMetricsLab.log"
    logging.getLogger("vmaf_app.core.example").error("Butteraugli failed: %s", "out of memory")
    text = _text(session_log)
    assert "ERROR" in text and "vmaf_app.core.example: Butteraugli failed: out of memory" in text


def test_uncaught_exceptions_are_logged_with_their_traceback(session_log):
    """A crash in the UI thread or in a run's lane left no trace once the
    window was gone."""
    try:
        raise ValueError("a bad frame")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    try:
        raise RuntimeError("lane crashed")
    except RuntimeError as error:
        threading.excepthook(SimpleNamespace(exc_type=RuntimeError, exc_value=error,
                                             exc_traceback=error.__traceback__, thread=SimpleNamespace(name="vmaf-gpu")))
    text = _text(session_log)
    assert "Uncaught exception\nTraceback" in text and "ValueError: a bad frame" in text
    assert "Uncaught exception in thread vmaf-gpu" in text and "RuntimeError: lane crashed" in text


def test_the_logs_are_exported_as_one_zip(tmp_path):
    import zipfile

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "VideoMetricsLab.log").write_bytes(b"today's session\n")
    (logs / "VideoMetricsLab.log.1").write_bytes(b"an older session\n")
    (logs / "native-crashes.log").write_bytes(b"")  # nothing in it: left out
    (logs / "notes.txt").write_bytes(b"not a log\n")
    target = tmp_path / "export.zip"
    assert app_log.export_logs(target, logs) == ["VideoMetricsLab.log", "VideoMetricsLab.log.1"]
    with zipfile.ZipFile(target) as archive:
        assert archive.read("VideoMetricsLab.log") == b"today's session\n"
        assert sorted(archive.namelist()) == ["VideoMetricsLab.log", "VideoMetricsLab.log.1"]


def test_the_session_and_run_lines_are_the_ones_the_copy_looks_for(tmp_path, monkeypatch):
    """The copy finds sessions and runs by the lines the app writes: if their
    wording changed, the copy would start in the wrong place."""
    import logging

    from vmaf_app import main as entry
    from vmaf_app.core import job_runner

    monkeypatch.setattr(app_log, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    entry.start_session_log()
    try:
        job_runner.JobScheduler([], parallel_jobs=1).run()
        logging.getLogger("vmaf_app.core.job_runner").error("a failure in the run")
        text, _ = app_log.log_text_to_share(tmp_path)
    finally:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(None)
        app_log.stop_logging()
    assert text.splitlines()[0].endswith("==== VideoMetricsLab starting ====")
    assert "Run started: 0 video(s)" in text and "a failure in the run" in text
