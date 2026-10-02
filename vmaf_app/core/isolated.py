"""Runs native GPU code -- Vship, libvmaf through ctypes -- in a process of its
own, so that a crash in it ends that process and not the app.

A library loaded with ctypes runs in the caller's process: an access
violation in it, or in the GPU driver under it, killed the whole app with
every video's progress, with no message. run_isolated calls a module-level
function in a child process instead and gives back what it returned, or
raises what it raised; if the child dies without either, it raises
IsolatedCrashError, which callers treat like any other failure of that step
(the CPU fallback, a failed metric).

The child is a spawned Python (multiprocessing), the frozen app included:
main.py calls multiprocessing.freeze_support() first thing. What crosses the
boundary is pickled: the arguments, the result, every callback's arguments
and the exception, with its attributes. While the child runs:

- callbacks named in `callbacks` run in this process, in the order the child
  made them, on the thread that called run_isolated;
- `process_handle`: the processes the child attaches to its handle (FFmpeg)
  are attached to this one, so Pause and Cancel reach them as before;
- `cancel_event`: when it is set the child and everything it started are
  ended, and `cancelled` is raised;
- the child's log records are logged here, under their own logger names.
"""
from __future__ import annotations

import contextlib
import logging
import multiprocessing
import threading
import traceback
from collections.abc import Callable, Iterable
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler

import psutil

_log = logging.getLogger(__name__)

#: How often the parent looks at the cancel event while the child is quiet.
_POLL_SECONDS = 0.05


class IsolatedCrashError(RuntimeError):
    """The child process ended without a result: a crash in native code,
    most likely (exit code 0xC0000005 is an access violation)."""

    def __init__(self, what: str, exitcode: int | None):
        self.what = what
        self.exitcode = exitcode
        super().__init__(f"{what} crashed ({describe_exit_code(exitcode)})")

    def __reduce__(self):
        return type(self), (self.what, self.exitcode)


def describe_exit_code(exitcode: int | None) -> str:
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
    """target(*args, **kwargs) in a child process; see the module docstring.

    `target` must be a module-level function. `callbacks` names keyword
    arguments of it that are callables; None ones are left out."""
    calls = {name: kwargs.pop(name) for name in callbacks if kwargs.get(name) is not None}
    for name in callbacks:
        kwargs.pop(name, None)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    # The lowest level any of the app's loggers takes: the app's own log is
    # at INFO on its package logger while the root logger stays at WARNING,
    # and the root's level alone dropped every INFO line the child wrote
    # (its FFmpeg commands, a pass's timing). Records come back to their own
    # logger names here, whose levels decide again.
    level = min(logging.getLogger().getEffectiveLevel(),
                logging.getLogger(__name__.partition(".")[0]).getEffectiveLevel())
    child = context.Process(
        target=_child_main,
        args=(sender, level, target, args, kwargs,
              tuple(calls), process_handle is not None, cancel_event is not None),
        name=f"isolated-{what}", daemon=True,
    )
    child.start()
    sender.close()
    attached: set[int] = set()
    outcome: tuple[str, object] | None = None
    try:
        while outcome is None:
            if cancel_event is not None and cancel_event.is_set():
                _end_tree(child.pid)
                raise (cancelled or RuntimeError)("Cancelled by user")
            ready = wait([receiver, child.sentinel], _POLL_SECONDS)
            if not ready:
                continue
            if receiver not in ready and not receiver.poll():
                break  # the child ended, and everything it sent has been read
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
    if cancel_event is not None and cancel_event.is_set():
        # Cancel ends the child's FFmpeg too: what the child made of that
        # (a failed read, a dead pipe) is not the outcome.
        raise (cancelled or RuntimeError)("Cancelled by user")
    if outcome is None:
        raise IsolatedCrashError(what, child.exitcode)
    if outcome[0] == "error":
        raise _rebuild_exception(*outcome[1:])
    return outcome[1]


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
    """The child's end: Connection.send is not thread-safe, and the code it
    runs reports from several threads (frame readers, FFmpeg monitors)."""

    def __init__(self, connection) -> None:
        self._connection = connection
        self._lock = threading.Lock()

    def send(self, message) -> None:
        with self._lock:
            self._connection.send(message)


class _ForwardedHandle:
    """Stands in for the parent's ProcessHandle: attach/detach go to it.
    Pausing and ending are the parent's to do."""

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


class _LogForwarder(logging.Handler):
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
                callback_names: tuple[str, ...], has_handle: bool, has_cancel: bool) -> None:
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


def _forwarder(sender: _Sender, name: str) -> Callable:
    def call(*args) -> None:
        sender.send(("call", name, args))
    return call


def _describe_exception(error: BaseException) -> tuple:
    """What rebuilds `error` in the parent: its class by name, its args and
    attributes (an exception's own pickling drops what __init__ set beyond
    args -- VmafRunError's stderr_tail), and the child's traceback."""
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
