"""Launching ffmpeg/ffprobe without a console window flashing up.

The app runs under pythonw.exe, which has no console of its own, so every
child process Windows starts gets a brand new console window -- ffprobe on
each added video, the version checks at startup, crop detection during a
run. They appear and vanish, and with several videos it looks like the app
is misbehaving.
"""
from __future__ import annotations

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
