"""Finds FFmpeg and ffprobe, and checks that they run and are new enough.

They are looked for in the folder chosen in Settings (Settings.ffmpeg_dir),
then on the PATH, then in common install folders. The last matters because
a newly installed winget package is not on the PATH of processes that were
already running when it was installed.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from vmaf_app.core import proc as proc_util

# FFmpeg's XPSNR filter and the libvmaf option syntax the app's filter
# graphs use need a recent FFmpeg; the app is developed and tested with 9.
MINIMUM_FFMPEG_VERSION = (9,)

#: The stream every command reads from a file: the first video stream that
#: is not a picture. FFmpeg lists cover art and thumbnails as video streams
#: too, and an MP4's cover (covr) can come before its video track, so "v:0"
#: would compare, scan or play the picture. "V" leaves pictures out.
#: ffprobe.probe_video describes the same stream.
VIDEO_STREAM = "V:0"

_WINGET_PACKAGE_GLOBS = [
    r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*-full_build\bin",
    r"AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg*\bin",
]


def _candidate_dirs() -> list[Path]:
    """The common install folders to look in."""
    home = Path.home()
    candidates: list[Path] = []
    for pattern in _WINGET_PACKAGE_GLOBS:
        candidates.extend(home.glob(pattern))
    candidates.append(Path(r"C:\ffmpeg\bin"))
    candidates.append(Path(r"C:\Program Files\ffmpeg\bin"))
    return candidates


def _configured_dir() -> str | None:
    """The folder chosen in Settings, read from settings.json, where every
    process finds it, the isolated ones included. Keep it in that one place:
    a second copy (once in the registry) can disagree with it."""
    try:
        from vmaf_app.core.settings import Settings

        return Settings.load().ffmpeg_dir.strip() or None
    except Exception:
        return None


def ffmpeg_dir_changed() -> None:
    """Forgets where the tools were found and how they checked, after the
    folder in Settings has changed and been saved."""
    global _tools
    find_binary.cache_clear()
    with _tools_lock:
        _tools = None


def exe_name(name: str) -> str:
    """A tool's file name on this system ("ffmpeg.exe" on Windows)."""
    return f"{name}.exe" if os.name == "nt" else name


@cache
def find_binary(name: str) -> str:
    """Returns the path (or else the bare name) to run for `name` ("ffmpeg"
    or "ffprobe")."""
    exe = f"{name}.exe" if os.name == "nt" else name

    override = _configured_dir()
    if override:
        candidate = Path(override) / exe
        if candidate.exists():
            return str(candidate)

    on_path = shutil.which(name)
    if on_path:
        return on_path

    for d in _candidate_dirs():
        candidate = d / exe
        if candidate.exists():
            return str(candidate)

    # Last resort: the bare name, in case it is on the PATH by the time it runs.
    return name


def ffmpeg_path() -> str:
    """The FFmpeg to run."""
    return find_binary("ffmpeg")


def ffprobe_path() -> str:
    """The ffprobe to run."""
    return find_binary("ffprobe")


# ---------------------------------------------------------------- tool checks

# Matches the version line both tools print, such as
#   "ffmpeg version 9.0.1-full_build-www.gyan.dev ..."
#   "ffprobe version n7.1 Copyright ..."
# Builds from FFmpeg's development branch say "N-113411-g1234567" instead,
# which has no usable version number, so parse_version returns None for
# them rather than guessing.
_VERSION_RE = re.compile(r"\b(?:ffmpeg|ffprobe) version n?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def parse_version(version_output: str) -> tuple[int, ...] | None:
    """The version in a tool's -version output, such as (9, 0, 1), or None."""
    match = _VERSION_RE.search(version_output)
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def format_version(version: tuple[int, ...] | None) -> str:
    """A version as text, such as "9.0.1", or "unknown"."""
    return ".".join(str(p) for p in version) if version else "unknown"


@dataclass
class ToolStatus:
    """The result of checking one tool."""

    name: str
    path: str
    runnable: bool
    version: tuple[int, ...] | None = None
    error: str = ""


def check_tool(name: str) -> ToolStatus:
    """Runs `<name> -version`, which is the only way to know that the tool
    is there and runs, rather than just a path that exists."""
    path = find_binary(name)
    try:
        proc = proc_util.run([path, "-version"], capture_output=True, text=True, timeout=15)
    except Exception as e:
        return ToolStatus(name=name, path=path, runnable=False, error=str(e))
    if proc.returncode != 0:
        return ToolStatus(
            name=name, path=path, runnable=False,
            error=(proc.stderr or proc.stdout or "").strip()[:300] or f"exited with code {proc.returncode}",
        )
    return ToolStatus(name=name, path=path, runnable=True, version=parse_version(proc.stdout))


@dataclass
class ToolsStatus:
    """The result of checking FFmpeg and ffprobe."""

    ffmpeg: ToolStatus
    ffprobe: ToolStatus

    @property
    def problems(self) -> list[str]:
        """The reasons the app cannot run, for the user; empty if there are
        none."""
        issues: list[str] = []
        for tool in (self.ffmpeg, self.ffprobe):
            if not tool.runnable:
                issues.append(f"{exe_name(tool.name)} could not be run ({tool.error or 'not found'}).")
        version = self.ffmpeg.version
        if self.ffmpeg.runnable and version is not None and version < MINIMUM_FFMPEG_VERSION:
            issues.append(
                f"ffmpeg {format_version(version)} is too old -- "
                f"version {format_version(MINIMUM_FFMPEG_VERSION)} or newer is required."
            )
        return issues

    @property
    def ok(self) -> bool:
        """Whether both tools are fine."""
        return not self.problems


_tools_lock = threading.Lock()
_tools: ToolsStatus | None = None


def check_tools() -> ToolsStatus:
    """Checks both tools, once: the answer is kept, because each check
    starts two processes, and every new window asks. A check already
    running (start_tool_check) is waited for, not run again.
    ffmpeg_dir_changed() forgets the answer."""
    global _tools
    with _tools_lock:
        if _tools is None:
            _tools = ToolsStatus(ffmpeg=check_tool("ffmpeg"), ffprobe=check_tool("ffprobe"))
        return _tools


def start_tool_check() -> None:
    """Runs check_tools() in the background at the start, once the folder in
    Settings is known, so the window finds it done once it is built. Run
    while the window is built, its two processes would hold the window's
    thread for 80 ms."""
    threading.Thread(target=check_tools, name="ffmpeg-check", daemon=True).start()
