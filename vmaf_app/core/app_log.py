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
from pathlib import Path

from vmaf_app.core.app_paths import user_data_dir

LOGGER_NAME = "vmaf_app"
LOG_FILE_NAME = "VideoMetricsLab.log"
#: Python-level tracebacks of a hard crash (faulthandler): the process is
#: gone before anything could reach the main log.
CRASH_FILE_NAME = "native-crashes.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 4
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


def environment_lines() -> list[str]:
    """What this session runs on, for the top of each session's log."""
    from vmaf_app import APP_NAME, __version__

    lines = [
        f"{APP_NAME} {__version__} ({'packaged build' if getattr(sys, 'frozen', False) else 'from source'})",
        f"Python {sys.version.split()[0]} on {platform.platform()}",
        f"CPU: {platform.processor() or 'unknown'}, {os.cpu_count()} logical processors",
    ]
    memory = _physical_memory_gb()
    if memory:
        lines.append(f"Memory: {memory:.0f} GB")
    return lines


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
