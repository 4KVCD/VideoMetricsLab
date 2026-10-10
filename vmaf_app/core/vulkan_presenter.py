"""Video Compare's Vulkan presenter (vulkan_presenter.dll, built from
native/vulkan_presenter.cpp). It draws and presents native playback's frames
with Vulkan as d3d11_tonemap.dll's presenter does with Direct3D 11: the same
calls, and the same pictures.

On Windows, native playback draws with Direct3D 11. This presenter is kept as
the start of the one for macOS and Linux, which have no Direct3D. On Windows
it is used only when the environment variable VIDEOMETRICSLAB_PRESENTER is
set to "vulkan", to test it, and Direct3D 11's presenter takes over where it
cannot start (locked_presentation).
"""
from __future__ import annotations

import ctypes
import functools
import os
import sys
from pathlib import Path

#: The environment variable that asks for this presenter ("vulkan").
CHOICE_VARIABLE = "VIDEOMETRICSLAB_PRESENTER"


def library_path() -> Path:
    """Where the Vulkan presenter's DLL is."""
    return Path(__file__).resolve().parents[1] / "native" / "vulkan_presenter.dll"


def available() -> bool:
    """Whether the Vulkan presenter's DLL is here (Windows only)."""
    return sys.platform == "win32" and library_path().is_file()


def chosen() -> bool:
    """Whether native playback tries this presenter first: built, and asked
    for by VIDEOMETRICSLAB_PRESENTER=vulkan."""
    return available() and os.environ.get(CHOICE_VARIABLE, "").strip().casefold() == "vulkan"


@functools.cache
def _library():
    """The Vulkan presenter's DLL, loaded once."""
    library = ctypes.CDLL(str(library_path()))
    library.vmaf_present_prepare.restype = ctypes.c_int
    library.vmaf_present_problem.restype = ctypes.c_char_p
    return library


def problem() -> str:
    """Why the presenter's last call that failed did."""
    return (_library().vmaf_present_problem() or b"").decode("utf-8", "replace")


def prepare() -> str | None:
    """Makes the Vulkan device, its pipelines and the frame hand-over's
    shader ahead of the first presenter (vmaf_present_prepare); the device
    alone takes some 40 ms. Returns None, or why Vulkan cannot present here."""
    if not chosen():
        return None
    try:
        failed = _library().vmaf_present_prepare() < 0
    except OSError as error:
        return str(error)
    return (problem() or "Vulkan failed") if failed else None


def check() -> str | None:
    """None if Vulkan presents Video Compare's frames here, or why it does
    not. For the self-test, when the presenter is asked for."""
    if not available():
        return f"{library_path().name} is not built"
    return prepare()
