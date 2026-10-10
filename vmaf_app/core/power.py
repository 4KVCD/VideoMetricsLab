"""Keeps Windows from going to sleep while a run is going.

A run can take hours (a full 4K film takes several), and a PC set to sleep
after some idle time would go to sleep during it. On a Modern Standby PC,
once the sleep timeout passes, Windows suspends desktop apps, which stops
the app and its FFmpeg and Vship processes in the middle of a pass.

A "system required" request (HandBrake holds the same one while it encodes)
keeps the PC working until it is released; the screen may still turn off.
Windows honours it for as long as it is held on mains power, and for five
minutes on battery.
"""
from __future__ import annotations

import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_system_awake(awake: bool) -> bool:
    """Holds (True) or releases (False) the request. The request belongs to
    the calling thread and ends with it at the latest, so call this from the
    GUI thread, which lives as long as the app. Returns False when nothing
    was held: not on Windows, or Windows refused."""
    if sys.platform != "win32":
        return False
    import ctypes

    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if awake else 0)
    return ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0
