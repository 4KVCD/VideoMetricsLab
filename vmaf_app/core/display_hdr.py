"""Read the Windows HDR state for the monitor showing the application.

The frame viewer is a QWidget/QPixmap surface, so its final image is SDR.
These values let HDR previews be tone-mapped for the monitor's configured
SDR white level instead of pretending that an 8-bit PNG is native HDR.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DisplayHdrInfo:
    device_name: str = ""
    hdr_supported: bool | None = None
    hdr_enabled: bool | None = None
    bits_per_color_channel: int | None = None
    sdr_white_nits: float | None = None


def query_display_hdr(window_handle: int | None = None) -> DisplayHdrInfo:
    """Return HDR state for the monitor nearest ``window_handle``.

    Windows exposes this through the DisplayConfig APIs.  On another OS, or
    when a driver does not implement the query, unknown values are returned;
    preview extraction then uses the conservative 100-nit SDR reference.
    """
    if os.name != "nt":
        return DisplayHdrInfo()
    try:
        return _query_windows_display_hdr(window_handle)
    except (OSError, ValueError, AttributeError):
        # Monitor/driver queries must never prevent the comparison tab from
        # opening.  Unknown is materially different from reporting HDR off.
        return DisplayHdrInfo()


def _query_windows_display_hdr(window_handle: int | None) -> DisplayHdrInfo:
    from ctypes import wintypes

    class Luid(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class Rational(ctypes.Structure):
        _fields_ = [("Numerator", wintypes.UINT), ("Denominator", wintypes.UINT)]

    class SourceInfo(ctypes.Structure):
        _fields_ = [
            ("adapterId", Luid), ("id", wintypes.UINT),
            ("modeInfoIdx", wintypes.UINT), ("statusFlags", wintypes.UINT),
        ]

    class TargetInfo(ctypes.Structure):
        _fields_ = [
            ("adapterId", Luid), ("id", wintypes.UINT),
            ("modeInfoIdx", wintypes.UINT), ("outputTechnology", wintypes.UINT),
            ("rotation", wintypes.UINT), ("scaling", wintypes.UINT),
            ("refreshRate", Rational), ("scanLineOrdering", wintypes.UINT),
            ("targetAvailable", wintypes.BOOL), ("statusFlags", wintypes.UINT),
        ]

    class PathInfo(ctypes.Structure):
        _fields_ = [
            ("sourceInfo", SourceInfo), ("targetInfo", TargetInfo),
            ("flags", wintypes.UINT),
        ]

    class ModeUnion(ctypes.Union):
        _fields_ = [("opaque", ctypes.c_byte * 56)]

    class ModeInfo(ctypes.Structure):
        _fields_ = [
            ("infoType", wintypes.UINT), ("id", wintypes.UINT),
            ("adapterId", Luid), ("mode", ModeUnion),
        ]

    class DeviceInfoHeader(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.UINT), ("size", wintypes.UINT),
            ("adapterId", Luid), ("id", wintypes.UINT),
        ]

    class SourceDeviceName(ctypes.Structure):
        _fields_ = [
            ("header", DeviceInfoHeader),
            ("viewGdiDeviceName", wintypes.WCHAR * 32),
        ]

    class AdvancedColorInfo(ctypes.Structure):
        _fields_ = [
            ("header", DeviceInfoHeader), ("value", wintypes.UINT),
            ("colorEncoding", wintypes.UINT),
            ("bitsPerColorChannel", wintypes.UINT),
        ]

    class SdrWhiteLevel(ctypes.Structure):
        _fields_ = [
            ("header", DeviceInfoHeader), ("SDRWhiteLevel", wintypes.ULONG),
        ]

    class MonitorInfoEx(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    monitor_from_window = user32.MonitorFromWindow
    monitor_from_window.argtypes = [wintypes.HWND, wintypes.DWORD]
    monitor_from_window.restype = wintypes.HMONITOR
    get_monitor_info = user32.GetMonitorInfoW
    get_monitor_info.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MonitorInfoEx)]
    get_monitor_info.restype = wintypes.BOOL

    monitor = monitor_from_window(window_handle or 0, 2)  # nearest monitor
    monitor_info = MonitorInfoEx()
    monitor_info.cbSize = ctypes.sizeof(monitor_info)
    if not monitor or not get_monitor_info(monitor, ctypes.byref(monitor_info)):
        raise ctypes.WinError(ctypes.get_last_error())
    wanted_name = monitor_info.szDevice

    get_sizes = user32.GetDisplayConfigBufferSizes
    query_config = user32.QueryDisplayConfig
    get_device_info = user32.DisplayConfigGetDeviceInfo
    active_paths_only = 0x2
    error_insufficient_buffer = 122

    for _attempt in range(3):
        path_count = wintypes.UINT()
        mode_count = wintypes.UINT()
        result = get_sizes(
            active_paths_only, ctypes.byref(path_count), ctypes.byref(mode_count)
        )
        if result:
            raise ctypes.WinError(result)
        paths = (PathInfo * path_count.value)()
        modes = (ModeInfo * mode_count.value)()
        result = query_config(
            active_paths_only, ctypes.byref(path_count), paths,
            ctypes.byref(mode_count), modes, None,
        )
        if result == error_insufficient_buffer:
            continue
        if result:
            raise ctypes.WinError(result)
        break
    else:
        raise OSError("The active display list kept changing")

    for path in paths[:path_count.value]:
        source_name = SourceDeviceName()
        source_name.header.type = 1  # DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME
        source_name.header.size = ctypes.sizeof(source_name)
        source_name.header.adapterId = path.sourceInfo.adapterId
        source_name.header.id = path.sourceInfo.id
        if get_device_info(ctypes.byref(source_name)):
            continue
        if source_name.viewGdiDeviceName.casefold() != wanted_name.casefold():
            continue

        advanced = AdvancedColorInfo()
        advanced.header.type = 9  # DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO
        advanced.header.size = ctypes.sizeof(advanced)
        advanced.header.adapterId = path.targetInfo.adapterId
        advanced.header.id = path.targetInfo.id
        if get_device_info(ctypes.byref(advanced)):
            return DisplayHdrInfo(device_name=wanted_name)

        white = SdrWhiteLevel()
        white.header.type = 11  # DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL
        white.header.size = ctypes.sizeof(white)
        white.header.adapterId = path.targetInfo.adapterId
        white.header.id = path.targetInfo.id
        white_result = get_device_info(ctypes.byref(white))
        # Windows expresses this as a multiplier of 80 nits, scaled by 1000.
        white_nits = None if white_result else white.SDRWhiteLevel / 1000 * 80
        flags = advanced.value
        return DisplayHdrInfo(
            device_name=wanted_name,
            hdr_supported=bool(flags & 0x1),
            hdr_enabled=bool(flags & 0x2),
            bits_per_color_channel=int(advanced.bitsPerColorChannel),
            sdr_white_nits=white_nits,
        )

    return DisplayHdrInfo(device_name=wanted_name)
