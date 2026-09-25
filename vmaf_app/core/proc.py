"""Launching ffmpeg/ffprobe without a console window flashing up.

The app runs under pythonw.exe, which has no console of its own, so every
child process Windows starts gets a brand new console window -- ffprobe on
each added video, the version checks at startup, crop detection during a
run. They appear and vanish, and with several videos it looks like the app
is misbehaving.
"""
from __future__ import annotations

import contextlib
import os
import subprocess

# CREATE_NO_WINDOW. Defined here rather than imported from subprocess so the
# module still imports cleanly off Windows, where the flag does not exist.
_CREATE_NO_WINDOW = 0x0800_0000


def hidden_kwargs() -> dict:
    """Extra Popen/run keyword arguments that suppress the console window."""
    if os.name != "nt":
        return {}
    return {"creationflags": _CREATE_NO_WINDOW}


def run(cmd, **kwargs):
    """subprocess.run with the console window suppressed on Windows."""
    return subprocess.run(cmd, **{**hidden_kwargs(), **kwargs})


def popen(cmd, **kwargs):
    """subprocess.Popen with the console window suppressed on Windows."""
    return subprocess.Popen(cmd, **{**hidden_kwargs(), **kwargs})


def process_tree(pid: int) -> list:
    """The process and every process it started, oldest first.

    FFmpeg is not always the process the app starts. Chocolatey installs
    ffmpeg.exe as a "shim", a launcher that starts the real ffmpeg.exe as a
    child and waits for it. Suspending or ending the launcher alone left
    FFmpeg running: Pause did not pause, Cancel did not stop it, and the
    CPU perceptual-metric extraction ran ahead of scoring unchecked (364
    images waiting on disk on the GitHub runner, where the limit is 52).
    """
    import psutil

    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
    except psutil.Error:
        return []
    # Not the console host Windows gives each console program: it does no
    # work, and suspending it serves nothing.
    tree = [root]
    for child in children:
        with contextlib.suppress(psutil.Error):
            if child.name().lower() != "conhost.exe":
                tree.append(child)
    return tree


def process_root(pid: int) -> list:
    """Just the process, as a one-item tree: quick, for when its children
    have not been listed yet."""
    import psutil

    try:
        return [psutil.Process(pid)]
    except psutil.Error:
        return []


def signal_tree(pid: int, action: str) -> None:
    """Applies "suspend", "resume", "terminate" or "kill" to a process and
    its children (see process_tree). Suspending starts at the top, so the
    launcher cannot start anything meanwhile; resuming and ending start at
    the bottom, so FFmpeg is never left running under a stopped launcher.
    A process that has already exited is skipped."""
    signal_processes(process_tree(pid), action)


def signal_processes(tree: list, action: str) -> None:
    """signal_tree for a tree already listed by process_tree, for callers
    that switch it often: listing it takes ~13 ms on Windows."""
    import psutil

    for process in (tree if action == "suspend" else reversed(tree)):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            getattr(process, action)()


def terminate(process) -> None:
    """Popen.terminate(), after ending anything the process started (the
    real FFmpeg under a launcher). The process itself is ended through its
    own Popen, which knows whether it has already exited."""
    signal_processes(process_tree(process.pid)[1:], "terminate")
    process.terminate()


def kill(process) -> None:
    """Popen.kill(), after killing anything the process started."""
    signal_processes(process_tree(process.pid)[1:], "kill")
    process.kill()


def raise_current_thread_priority() -> None:
    """Makes the calling thread preempt ordinary work (Windows
    THREAD_PRIORITY_HIGHEST). For light, timing-critical threads only.
    Does nothing elsewhere or if Windows refuses."""
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST

