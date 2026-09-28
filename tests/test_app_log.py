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


def _write_log(directory, name, *lines):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _session(tag: str, *, run: bool, extra=()):
    lines = [f"{tag} INFO vmaf_app.main: ==== VideoMetricsLab starting ====",
             f"{tag} INFO vmaf_app.main: VideoMetricsLab 1.3 (from source)"]
    if run:
        lines += [f"{tag} INFO vmaf_app.ui.worker: Run started: 1 video(s)",
                  f"{tag} ERROR vmaf_app.ui.worker: Video 1 'film': GPU metrics failed: CUDA error"]
    return [*lines, *extra]


def test_the_copy_starts_at_the_latest_session_that_ran_metrics(tmp_path):
    """A failed overnight run, then the app restarted: the copy still holds
    the run, and not the sessions before it."""
    _write_log(tmp_path, "VideoMetricsLab.log",
               *_session("day1", run=True), *_session("day2", run=True), *_session("day3", run=False))
    text, shortened = app_log.log_text_to_share(tmp_path)
    assert not shortened
    assert text.startswith("day2 INFO vmaf_app.main: ==== VideoMetricsLab starting ====")
    assert "day2 ERROR vmaf_app.ui.worker: Video 1 'film': GPU metrics failed: CUDA error" in text
    assert "day3" in text and "day1" not in text


def test_a_session_begun_in_the_previous_log_file_is_copied_whole(tmp_path):
    lines = _session("day1", run=True)
    _write_log(tmp_path, "VideoMetricsLab.log.1", *lines[:2])
    _write_log(tmp_path, "VideoMetricsLab.log", *lines[2:])
    text, _ = app_log.log_text_to_share(tmp_path)
    assert text.startswith("day1 INFO vmaf_app.main: ==== VideoMetricsLab starting ====") and "Run started" in text


def test_a_long_session_keeps_its_start_and_its_end(tmp_path):
    steps = [f"step {n:05d} " + "x" * 90 for n in range(2000)]
    _write_log(tmp_path, "VideoMetricsLab.log", *_session("day1", run=True, extra=[*steps, "the last line"]))
    text, shortened = app_log.log_text_to_share(tmp_path, limit=20_000)
    assert shortened and len(text) <= 20_000
    assert text.startswith("day1 INFO vmaf_app.main: ==== VideoMetricsLab starting ====")
    assert text.endswith("the last line")
    assert "lines left out here; Export log in Settings saves all of it" in text


def test_there_is_nothing_to_copy_without_a_log(tmp_path):
    assert app_log.log_text_to_share(tmp_path / "none") is None


def test_the_session_and_run_lines_are_the_ones_the_copy_looks_for(tmp_path, monkeypatch):
    """The copy finds sessions and runs by the lines the app writes: if their
    wording changed, the copy would start in the wrong place."""
    import logging

    from vmaf_app import main as entry
    from vmaf_app.ui import worker as worker_module

    monkeypatch.setattr(app_log, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    monkeypatch.setattr(sys, "excepthook", lambda *args: None)
    entry.start_session_log()
    try:
        worker_module.VmafWorker([], parallel_jobs=1).run()
        logging.getLogger("vmaf_app.ui.worker").error("a failure in the run")
        text, _ = app_log.log_text_to_share(tmp_path)
    finally:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(None)
        app_log.stop_logging()
    assert text.splitlines()[0].endswith("==== VideoMetricsLab starting ====")
    assert "Run started: 0 video(s)" in text and "a failure in the run" in text

