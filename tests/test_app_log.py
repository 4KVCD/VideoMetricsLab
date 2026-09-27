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


def test_the_package_prints_nothing_until_the_app_starts_its_log(capsys):
    logging.getLogger("vmaf_app.core.example").error("not for stderr")
    assert capsys.readouterr().err == ""


def test_stopping_restores_the_exception_hooks(tmp_path):
    hooks = (sys.excepthook, threading.excepthook)
    app_log.start_logging(tmp_path)
    assert (sys.excepthook, threading.excepthook) != hooks
    app_log.stop_logging()
    assert (sys.excepthook, threading.excepthook) == hooks


def test_each_session_is_headed_with_what_it_runs_on(tmp_path, monkeypatch):
    from PySide6.QtCore import qInstallMessageHandler

    from vmaf_app import __version__
    from vmaf_app import main as entry

    monkeypatch.setattr(app_log, "log_dir", lambda: tmp_path)
    entry.start_session_log()
    try:
        logging.getLogger("vmaf_app.qt").warning("a Qt warning")
        text = _text(tmp_path / app_log.LOG_FILE_NAME)
    finally:
        qInstallMessageHandler(None)
        app_log.stop_logging()
    assert "==== VideoMetricsLab starting ====" in text
    assert f"VideoMetricsLab {__version__}" in text and "Python " in text and "logical processors" in text
    assert "FFmpeg" in text and "GPUs: " in text and "Qt " in text
    assert "vmaf_app.qt: a Qt warning" in text


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


def test_no_log_exports_nothing(tmp_path):
    assert app_log.export_logs(tmp_path / "export.zip", tmp_path / "no-logs-here") == []
    assert not (tmp_path / "export.zip").exists()


def test_the_session_log_in_use_is_exported_up_to_its_last_line(session_log, tmp_path):
    import logging
    import zipfile

    logging.getLogger("vmaf_app.core.example").error("the failure just before exporting")
    target = tmp_path / "export.zip"
    app_log.export_logs(target, tmp_path)
    with zipfile.ZipFile(target) as archive:
        assert b"the failure just before exporting" in archive.read("VideoMetricsLab.log")

