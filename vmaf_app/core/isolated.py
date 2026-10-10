"""Runs native GPU code (Vship, and libvmaf through ctypes) in a process of
its own, so that a crash in it ends that process and not the app.

A library loaded with ctypes runs in the caller's process. An access
violation in it, or in the GPU driver under it, would kill the whole app,
losing every video's progress without a message. run_isolated instead calls
a module-level function in a child process and returns what it returned, or
raises what it raised. If the child dies without either, run_isolated raises
IsolatedCrashError, which callers treat like any other failure of that step
(by falling back to the CPU, or by failing the metric).

The child is a spawned Python process (multiprocessing). This works in the
frozen app too, because main.py calls multiprocessing.freeze_support() first
thing. Everything that crosses between the processes is pickled: the
arguments, the result, every callback's arguments, and the exception with
its attributes. While the child runs:

- the callbacks named in `callbacks` run in this process, in the order the
  child called them, on the thread that called run_isolated;
- with `process_handle`, the processes the child attaches to its handle
  (FFmpeg) are attached to this one too, so Pause and Cancel reach them;
- with `cancel_event`, setting the event ends the child and everything it
  started, and raises `cancelled`;
- the child's log records are logged here, under their own logger names;
- if the child crashes, where each of its threads was (from faulthandler) is
  logged here with the crash.
"""
from __future__ import annotations

import contextlib
import ctypes
import faulthandler
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import threading
import traceback
import uuid
from collections.abc import Callable, Iterable
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import psutil

from vmaf_app.core import proc

_log = logging.getLogger(__name__)

#: How often the parent checks the cancel event while the child sends nothing.
_POLL_SECONDS = 0.05


class IsolatedCrashError(RuntimeError):
    """The child process ended without a result, most likely because native
    code crashed (exit code 0xC0000005 is an access violation)."""

    def __init__(self, what: str, exitcode: int | None):
        self.what = what
        self.exitcode = exitcode
        if self.stopped:
            # Process.terminate()'s exit code means this app ended the child:
            # multiprocessing does that to children still running when the
            # app exits. That is not a crash, and is not reported as one.
            super().__init__(f"{what} was stopped before it finished")
        else:
            super().__init__(f"{what} crashed ({describe_exit_code(exitcode)})")

    @property
    def stopped(self) -> bool:
        """Whether this app ended the child, rather than the child crashing."""
        return self.exitcode == -signal.SIGTERM

    def __reduce__(self):
        return type(self), (self.what, self.exitcode)


def describe_exit_code(exitcode: int | None) -> str:
    """An exit code as it reads best: Windows status codes in hexadecimal."""
    if exitcode is None:
        return "no exit code"
    if exitcode < 0:  # how multiprocessing reports some Windows status codes
        exitcode &= 0xFFFFFFFF
    return f"exit code 0x{exitcode:08X}" if exitcode > 0xFFFF else f"exit code {exitcode}"


def run_isolated(
    target: Callable, *args,
    what: str,
    callbacks: Iterable[str] = (),
    process_handle=None,
    cancel_event: threading.Event | None = None,
    cancelled: Callable[[str], BaseException] | None = None,
    **kwargs,
):
    """Calls target(*args, **kwargs) in a child process and returns its result
    (see the module docstring).

    `target` must be a module-level function. `callbacks` names the keyword
    arguments of `target` that are callables; those given as None are left
    out. `what` names the work in error messages."""
    calls = {name: kwargs.pop(name) for name in callbacks if kwargs.get(name) is not None}
    for name in callbacks:
        kwargs.pop(name, None)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    # Where the child writes its threads' stacks if it crashes: a native
    # crash ends the child before it can send anything.
    crash_path = Path(tempfile.gettempdir()) / f"vml-isolated-{os.getpid()}-{uuid.uuid4().hex[:12]}.txt"
    # The lowest level any of the app's loggers takes. The app's log is at
    # INFO on the app's package logger while the root logger stays at
    # WARNING, and the root's level alone would drop every INFO line the
    # child writes (its FFmpeg commands, a pass's timing). The records come
    # back here under their own logger names, whose levels decide again.
    level = min(logging.getLogger().getEffectiveLevel(),
                logging.getLogger(__name__.partition(".")[0]).getEffectiveLevel())
    child = context.Process(
        target=_child_main,
        args=(sender, level, target, args, kwargs,
              tuple(calls), process_handle is not None, cancel_event is not None, str(crash_path)),
        name=f"isolated-{what}", daemon=True,
    )
    child.start()
    proc.end_with_app(child.pid)
    sender.close()
    attached: set[int] = set()
    outcome: tuple[str, object] | None = None
    crash_threads = ""
    try:
        while outcome is None:
            if cancel_event is not None and cancel_event.is_set():
                raise (cancelled or RuntimeError)("Cancelled by user")
            ready = wait([receiver, child.sentinel], _POLL_SECONDS)
            if not ready:
                continue
            if receiver not in ready and not receiver.poll():
                break  # the child has ended, and everything it sent has been read
            try:
                message = receiver.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "call":
                calls[message[1]](*message[2])
            elif kind == "log":
                _, name, level, text = message
                logging.getLogger(name).log(level, "%s", text)
            elif kind == "attach":
                attached.add(message[1])
                if process_handle is not None:
                    process_handle.attach(message[1])
            elif kind == "detach":
                attached.discard(message[1])
                if process_handle is not None:
                    process_handle.detach(message[1])
            else:  # "result" or "error"
                outcome = message
    except BaseException:
        # A Cancel, or a callback that failed: the child's work is no longer
        # wanted, so the child and everything it started are ended now.
        _end_tree(child.pid)
        raise
    finally:
        receiver.close()
        if outcome is None or child.is_alive():
            child.join(timeout=5)
            if child.is_alive():
                _end_tree(child.pid)
                child.join(timeout=5)
        if process_handle is not None:
            for pid in attached:
                process_handle.detach(pid)
        # Read (and removed) here, however the loop was left: a Cancel or a
        # failed callback raises out of it, and the file must not be left in
        # the temp folder.
        crash_threads = _take_text(crash_path)
    if cancel_event is not None and cancel_event.is_set():
        # Cancel also ends the child's FFmpeg, so whatever the child made of
        # that (a failed read, a broken pipe) is not the outcome.
        raise (cancelled or RuntimeError)("Cancelled by user")
    if outcome is None:
        error = IsolatedCrashError(what, child.exitcode)
        if crash_threads:
            _log.error("%s; its threads at the crash:\n%s", error, crash_threads)
        raise error
    if outcome[0] == "error":
        raise _rebuild_exception(*outcome[1:])
    return outcome[1]


def _take_text(path: Path) -> str:
    """Returns the text of `path` and removes the file; "" if there is none."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    with contextlib.suppress(OSError):
        path.unlink()
    return text


def _end_tree(pid: int | None) -> None:
    """Ends the child and everything it started (FFmpeg, its launcher)."""
    if pid is None:
        return
    try:
        root = psutil.Process(pid)
        processes = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for process in processes:
        with contextlib.suppress(psutil.Error):
            process.kill()


# ------------------------------------------------------------------- child

class _Sender:
    """The child's end of the pipe. Connection.send is not thread-safe, and
    the code the child runs reports from several threads (frame readers,
    FFmpeg monitors), so sends are serialized here."""

    def __init__(self, connection) -> None:
        self._connection = connection
        self._lock = threading.Lock()

    def send(self, message) -> None:
        with self._lock:
            self._connection.send(message)


class _ForwardedHandle:
    """Stands in for the parent's ProcessHandle in the child: attach and
    detach are sent to the parent. Pausing and ending processes are left to
    the parent."""

    def __init__(self, sender: _Sender) -> None:
        self._sender = sender

    def attach(self, pid: int) -> None:
        self._sender.send(("attach", pid))

    def detach(self, pid: int | None = None) -> None:
        if pid is not None:
            self._sender.send(("detach", pid))

    @property
    def is_pause_requested(self) -> bool:
        return False

    @property
    def was_terminated(self) -> bool:
        """Always False here. Cancel reaches the child through its cancel
        event, and the parent's handle ends any process attached after a
        Cancel (ProcessHandle.attach). vmaf_runner._cancelled reads this."""
        return False


class _LogForwarder(logging.Handler):
    """Sends the child's log records to the parent, which logs them."""

    def __init__(self, sender: _Sender) -> None:
        super().__init__()
        self._sender = sender

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.exc_info:
                text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            self._sender.send(("log", record.name, record.levelno, text))
        except Exception:  # a log record must never take the work down
            pass


def _child_main(connection, log_level: int, target: Callable, args: tuple, kwargs: dict,
                callback_names: tuple[str, ...], has_handle: bool, has_cancel: bool, crash_path: str) -> None:
    """The child process: sets up crash reports, logging and the stand-ins
    for the parent's callbacks, handle and cancel event, then runs `target`
    and sends back its result or its exception."""
    try:
        crash_file = open(crash_path, "w", encoding="utf-8")  # noqa: SIM115 -- open while the process lives
        faulthandler.enable(crash_file, all_threads=True)
    except (OSError, RuntimeError):
        pass  # the work runs the same without it
    sender = _Sender(connection)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_LogForwarder(sender))
    root.setLevel(log_level)
    for name in callback_names:
        kwargs[name] = _forwarder(sender, name)
    if has_handle:
        kwargs["process_handle"] = _ForwardedHandle(sender)
    if has_cancel:
        kwargs["cancel_event"] = threading.Event()  # the parent ends this process instead
    try:
        result = target(*args, **kwargs)
    except BaseException as error:  # every failure goes back to the caller
        sender.send(("error", *_describe_exception(error)))
    else:
        sender.send(("result", result))
    finally:
        connection.close()
    _end_if_a_decoder_is_stuck()


def _end_if_a_decoder_is_stuck() -> None:
    """Ends this process at once if a GPU decoder never finished closing
    (gpu_frames.stuck_decoders). The decoder's thread still holds the
    driver's locks, and a normal exit, which unloads every library in turn,
    could wait on them forever. The result has been sent by now, so nothing
    is lost."""
    gpu_frames = sys.modules.get("vmaf_app.core.gpu_frames")
    if gpu_frames is not None and gpu_frames.stuck_decoders():
        _end_now()


def _end_now() -> None:
    """Ends this process without unloading anything."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0)
    os._exit(0)


def _forwarder(sender: _Sender, name: str) -> Callable:
    """A stand-in for the parent's callback `name`: each call is sent to the
    parent, which makes it."""
    def call(*args) -> None:
        sender.send(("call", name, args))
    return call


def _describe_exception(error: BaseException) -> tuple:
    """What the parent needs to rebuild `error`: its class by name, its args,
    its attributes and the child's traceback. The attributes are sent
    separately because an exception's own pickling keeps only its args and
    drops what __init__ set beyond them (VmafRunError's stderr_tail)."""
    cls = type(error)
    state = {}
    for key, value in getattr(error, "__dict__", {}).items():
        try:
            ForkingPickler.dumps(value)
        except Exception:
            continue
        state[key] = value
    try:
        ForkingPickler.dumps(error.args)
        args = error.args
    except Exception:
        args = tuple(str(arg) for arg in error.args)
    text = "".join(traceback.format_exception(error)).rstrip()
    return cls.__module__, cls.__qualname__, args, state, text


def _rebuild_exception(module: str, qualname: str, args: tuple, state: dict, text: str) -> BaseException:
    """Rebuilds the child's exception in the parent, with the child's
    traceback added as a note; a RuntimeError if its class cannot be found."""
    import importlib

    try:
        cls = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        error = cls.__new__(cls)
        error.args = args
        error.__dict__.update(state)
    except Exception:
        error = RuntimeError(f"{qualname}: {', '.join(map(str, args))}")
    error.add_note("In the isolated process:\n" + text)
    return error
