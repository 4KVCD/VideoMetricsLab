"""The app's log file: what ran, how it went, and every failure in full.

A metric that failed during an overnight run said why only in a tooltip,
and the tooltip went with the window: after a restart nothing was left to go
on. The log keeps each session's setup, every run and its steps, and every
failure with its full text, in <user data>/logs/VideoMetricsLab.log --
rotated at 5 MB, with four older files kept.

Only the app turns it on (start_logging, from vmaf_app.main). Imported by
tests or scripts, the package logs nowhere.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import zipfile
from pathlib import Path

from vmaf_app.core.app_paths import user_data_dir

LOGGER_NAME = "vmaf_app"
LOG_FILE_NAME = "VideoMetricsLab.log"
#: Python-level tracebacks of a hard crash (faulthandler): the process is
#: gone before anything could reach the main log.
CRASH_FILE_NAME = "native-crashes.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 4
#: The line that opens each session (vmaf_app.main.start_session_log), and
#: the one that opens each run (vmaf_app.core.job_runner).
SESSION_START = " starting ===="
RUN_START = "Run started:"
#: The most text "Copy log" puts on the clipboard: a post or chat takes this
#: much, and a session's log is a few kilobytes -- a run of four films is
#: some 30 KB -- so only an unusually long one is cut.
SHARE_LIMIT = 60_000
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

_handler: logging.Handler | None = None
_crash_file = None
_previous_hooks: tuple | None = None


def log_dir() -> Path:
    return user_data_dir() / "logs"


def start_logging(directory: Path | None = None) -> Path | None:
    """Logs to a file from now on, and records uncaught exceptions (any
    thread) and native crashes. The log file's path, or None when it could
    not be opened -- the app runs the same without it."""
    global _handler, _crash_file, _previous_hooks
    if _handler is not None:
        return Path(_handler.baseFilename)
    directory = directory or log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / LOG_FILE_NAME, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8",
        )
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    _handler = handler

    _previous_hooks = (sys.excepthook, threading.excepthook)
    previous_hook, previous_thread_hook = _previous_hooks

    def excepthook(exc_type, exc, traceback) -> None:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, traceback))
        previous_hook(exc_type, exc, traceback)

    def thread_excepthook(args) -> None:
        if args.exc_type is not SystemExit:
            logger.critical("Uncaught exception in thread %s", getattr(args.thread, "name", "?"),
                            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        previous_thread_hook(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
    try:
        _crash_file = open(directory / CRASH_FILE_NAME, "a", encoding="utf-8")  # noqa: SIM115 -- open for the session
        faulthandler.enable(_crash_file, all_threads=True)
    except (OSError, RuntimeError):
        _crash_file = None
    return Path(handler.baseFilename)


def stop_logging() -> None:
    """Undo start_logging (tests)."""
    global _handler, _crash_file, _previous_hooks
    if _handler is None:
        return
    logging.getLogger(LOGGER_NAME).removeHandler(_handler)
    _handler.close()
    _handler = None
    if _previous_hooks is not None:
        sys.excepthook, threading.excepthook = _previous_hooks
        _previous_hooks = None
    if _crash_file is not None:
        faulthandler.disable()
        _crash_file.close()
        _crash_file = None


def export_logs(destination: Path, directory: Path | None = None) -> list[str]:
    """Zips the log files -- the current log, its older rotations and the
    crash log, those with anything in them -- into destination, to attach
    to a report. The names written; empty, and nothing written, when there
    is no log yet."""
    directory = directory or log_dir()
    if _handler is not None:
        _handler.flush()
    files = sorted(path for path in directory.glob("*.log*") if path.is_file() and path.stat().st_size) \
        if directory.is_dir() else []
    if not files:
        return []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return [path.name for path in files]


def log_text_to_share(directory: Path | None = None, limit: int = SHARE_LIMIT) -> tuple[str, bool] | None:
    """The part of the log to paste into a post or chat, and whether it was
    shortened; None when there is no log yet.

    From the start of the latest session that calculated metrics -- so a
    run that failed before the app was restarted is still in it -- to the
    end. Longer than `limit`, it keeps the session's first lines (what the
    app ran on) and as much of the end as fits, saying how many lines are
    left out; Export log saves all of it."""
    directory = directory or log_dir()
    if _handler is not None:
        _handler.flush()
    current = directory / LOG_FILE_NAME
    rotated = [directory / f"{LOG_FILE_NAME}.{n}" for n in range(_BACKUPS, 0, -1)]  # oldest first
    lines: list[str] = []
    for path in [*rotated, current]:
        if path.is_file():
            lines += path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return None
    last_run = max((i for i, line in enumerate(lines) if RUN_START in line), default=len(lines) - 1)
    start = max((i for i, line in enumerate(lines[:last_run + 1]) if line.endswith(SESSION_START)), default=0)
    lines = lines[start:]
    text = "\n".join(lines)
    if len(text) <= limit:
        return text, False
    head: list[str] = []
    for line in lines[:12]:  # the session's header: version, Windows, CPU, FFmpeg, GPUs...
        if len("\n".join([*head, line])) > limit // 4:
            break
        head.append(line)
    tail: list[str] = []
    budget = limit - len("\n".join(head)) - 120
    for line in reversed(lines[len(head):]):
        if len(line) + 1 > budget:
            break
        tail.insert(0, line)
        budget -= len(line) + 1
    left_out = len(lines) - len(head) - len(tail)
    note = f"... {left_out} lines left out here; Export log in Settings saves all of it ..."
    return "\n".join([*head, note, *tail]), True


def environment_lines() -> list[str]:
    """What this session runs on, for the top of each session's log."""
    from vmaf_app import APP_NAME, __version__

    lines = [
        f"{APP_NAME} {__version__} ({'packaged build' if getattr(sys, 'frozen', False) else 'from source'})",
        f"Python {sys.version.split()[0]} on {_system()}",
        f"CPU: {_processor() or 'unknown'}, {os.cpu_count()} logical processors",
    ]
    memory = _physical_memory_gb()
    if memory:
        lines.append(f"Memory: {memory:.0f} GB")
    return lines


def _system() -> str:
    """platform.platform()'s text, without the WMI query it makes on
    Windows: 50 ms of each start."""
    if sys.platform != "win32":
        return platform.platform()
    version = sys.getwindowsversion()
    release = "11" if version.major == 10 and version.build >= 22000 else str(version.major)
    return f"Windows-{release}-{version.major}.{version.minor}.{version.build}-SP{version.service_pack_major}"


def _processor() -> str:
    """platform.processor()'s text: on Windows, what it reads from the
    environment once its WMI query (_system) has run."""
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER", "")
    return platform.processor()


def _physical_memory_gb() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong)]

    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.total_physical / 1024 ** 3
