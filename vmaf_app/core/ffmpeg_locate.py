"""Finds the ffmpeg/ffprobe executables, since a freshly-installed winget
package is not on PATH for processes that were already running when it
was installed. The folder chosen in Settings (Settings.ffmpeg_dir) comes
first, then PATH, then common install locations.
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

# libvmaf's XPSNR filter and the current libvmaf option syntax this app
# builds its filtergraphs around need a recent ffmpeg; 9 is what the app is
# developed and tested against.
MINIMUM_FFMPEG_VERSION = (9,)

#: The stream every command reads from an input: the first video stream
#: that is not a picture. FFmpeg lists cover art and thumbnails as video
#: streams too, and an MP4's cover (covr) can come before its video track:
#: "v:0" then compared, scanned or played the picture. "V" leaves them out.
#: ffprobe.probe_video describes the same stream.
VIDEO_STREAM = "V:0"

_WINGET_PACKAGE_GLOBS = [
    r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*-full_build\bin",
    r"AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg*\bin",
]


def _candidate_dirs() -> list[Path]:
    home = Path.home()
    candidates: list[Path] = []
    for pattern in _WINGET_PACKAGE_GLOBS:
        candidates.extend(home.glob(pattern))
    candidates.append(Path(r"C:\ffmpeg\bin"))
    candidates.append(Path(r"C:\Program Files\ffmpeg\bin"))
    return candidates


def _configured_dir() -> str | None:
    """The folder chosen in Settings, read from settings.json -- where every
    process finds it, the isolated ones included. It used to be kept in the
    registry as well, written from two places that could disagree: "Locate
    ffmpeg.exe" wrote only the registry, so the Settings tab showed no
    folder while one was in use."""
    try:
        from vmaf_app.core.settings import Settings

        return Settings.load().ffmpeg_dir.strip() or None
    except Exception:
        return None


def ffmpeg_dir_changed() -> None:
    """After the folder in Settings changed (and was saved): finds the tools
    again, and checks them again."""
    global _tools
    find_binary.cache_clear()
    with _tools_lock:
        _tools = None


def exe_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


@cache
def find_binary(name: str) -> str:
    """Returns a path (or bare name) to invoke for `name` (ffmpeg/ffprobe)."""
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

    # Last resort: hope it's on PATH by the time we actually run it.
    return name


def ffmpeg_path() -> str:
    return find_binary("ffmpeg")


def ffprobe_path() -> str:
    return find_binary("ffprobe")


# ---------------------------------------------------------------- tool checks

# Matches the version line both tools print, e.g.
#   "ffmpeg version 9.0.1-full_build-www.gyan.dev ..."
#   "ffprobe version n7.1 Copyright ..."
# Git master builds report "N-113411-g1234567" instead, which carries no
# usable version number -- parse_version returns None for those rather than
# guessing.
_VERSION_RE = re.compile(r"\b(?:ffmpeg|ffprobe) version n?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def parse_version(version_output: str) -> tuple[int, ...] | None:
    match = _VERSION_RE.search(version_output)
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def format_version(version: tuple[int, ...] | None) -> str:
    return ".".join(str(p) for p in version) if version else "unknown"


@dataclass
class ToolStatus:
    name: str
    path: str
    runnable: bool
    version: tuple[int, ...] | None = None
    error: str = ""


def check_tool(name: str) -> ToolStatus:
    """Actually runs `<name> -version` -- the only way to know the binary is
    both present and executable, rather than just a path that exists."""
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
    ffmpeg: ToolStatus
    ffprobe: ToolStatus

    @property
    def problems(self) -> list[str]:
        """Human-readable reasons the app can't run, empty if all good."""
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
        return not self.problems


_tools_lock = threading.Lock()
_tools: ToolsStatus | None = None


def check_tools() -> ToolsStatus:
    """Cached: each call shells out twice, and this is consulted on every
    window construction. A check already running (start_tool_check) is
    waited for, not run again. ffmpeg_dir_changed() clears it."""
    global _tools
    with _tools_lock:
        if _tools is None:
            _tools = ToolsStatus(ffmpeg=check_tool("ffmpeg"), ffprobe=check_tool("ffprobe"))
        return _tools


def start_tool_check() -> None:
    """check_tools() in the background from the start, the folder in
    Settings known: the window, built meanwhile, finds it done. Run while
    the window was built, its two processes held the window's thread 80 ms."""
    threading.Thread(target=check_tools, name="ffmpeg-check", daemon=True).start()
