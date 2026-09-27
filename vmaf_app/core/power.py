"""Keeping Windows out of sleep while a run is going.

A run can take hours -- a full 4K film is several -- and a PC set to sleep
after some idle time would sleep under it: on a Modern Standby PC, once the
sleep timeout passes, Windows suspends desktop apps, stopping the app and its
FFmpeg and Vship processes mid-pass. A "system required" request, which
HandBrake holds while it encodes, keeps the PC working until it is released;
the screen may still turn off. Windows honours it for as long as it is held
on mains power, and for five minutes on battery.
"""
from __future__ import annotations

import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_system_awake(awake: bool) -> bool:
    """Hold (True) or release (False) the request. It belongs to the calling
    thread -- call it from the GUI thread, which lives as long as the app --
    and ends with that thread at the latest. False where nothing was held:
    not Windows, or Windows refused."""
    if sys.platform != "win32":
        return False
    import ctypes

    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if awake else 0)
    return ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0
