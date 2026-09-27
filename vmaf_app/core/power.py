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


# ---------------------------------------------------------- power notices
# Windows tells every top-level window when the PC is about to sleep or has
# woken, and -- once registered for -- when the screen turns off, dims or
# comes back on, and when it switches between mains and battery. The app
# only listens and logs them (MainWindow.nativeEvent): a failure during a run
# can then be read against what the PC was doing at that moment.

WM_POWERBROADCAST = 0x0218
_PBT_APMSUSPEND = 0x0004
_PBT_APMRESUMESUSPEND = 0x0007
_PBT_APMRESUMEAUTOMATIC = 0x0012
_PBT_POWERSETTINGCHANGE = 0x8013
#: GUID_CONSOLE_DISPLAY_STATE and GUID_ACDC_POWER_SOURCE.
_DISPLAY_STATE = "6fe69556-704a-47a0-8f24-c28d936fda47"
_POWER_SOURCE = "5d3e9a59-e9d5-4b00-a6bd-ff34ff516548"
_DISPLAY_STATES = {0: "Screen turned off", 1: "Screen turned on", 2: "Screen dimmed"}
_POWER_SOURCES = {0: "On mains power", 1: "On battery", 2: "On short-term power (UPS)"}


def _guid_bytes(text: str) -> bytes:
    import uuid

    return uuid.UUID(text).bytes_le


def register_power_notices(window_handle: int) -> list[int]:
    """Ask Windows to tell this window about the screen and the power
    source; the registrations, for unregister_power_notices."""
    if sys.platform != "win32" or not window_handle:
        return []
    import ctypes

    register = ctypes.windll.user32.RegisterPowerSettingNotification
    register.restype = ctypes.c_void_p
    register.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong]
    handles = []
    for guid in (_DISPLAY_STATE, _POWER_SOURCE):
        handle = register(window_handle, _guid_bytes(guid), 0)  # DEVICE_NOTIFY_WINDOW_HANDLE
        if handle:
            handles.append(handle)
    return handles


def unregister_power_notices(handles: list[int]) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    unregister = ctypes.windll.user32.UnregisterPowerSettingNotification
    unregister.argtypes = [ctypes.c_void_p]
    for handle in handles:
        unregister(handle)


def describe_power_notice(wparam: int, lparam: int) -> str | None:
    """What a WM_POWERBROADCAST says, in words; None for notices not logged.
    lparam points at a POWERBROADCAST_SETTING for PBT_POWERSETTINGCHANGE."""
    if wparam == _PBT_APMSUSPEND:
        return "Windows is going to sleep"
    if wparam in (_PBT_APMRESUMESUSPEND, _PBT_APMRESUMEAUTOMATIC):
        return "Windows woke from sleep"
    if wparam != _PBT_POWERSETTINGCHANGE or not lparam:
        return None
    import ctypes

    guid = ctypes.string_at(lparam, 16)
    length = ctypes.c_ulong.from_address(lparam + 16).value
    if length < 4:
        return None
    value = ctypes.c_ulong.from_address(lparam + 20).value
    if guid == _guid_bytes(_DISPLAY_STATE):
        return _DISPLAY_STATES.get(value, f"Screen state {value}")
    if guid == _guid_bytes(_POWER_SOURCE):
        return _POWER_SOURCES.get(value, f"Power source {value}")
    return None

