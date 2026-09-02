"""Finds the ffmpeg/ffprobe executables, since a freshly-installed winget
package is not on PATH for processes that were already running when it
was installed. Falls back to common install locations and lets the user
override via QSettings.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SETTINGS_ORG = "VmafApp"
_SETTINGS_APP = "VmafCalculator"

# libvmaf's XPSNR filter and the current libvmaf option syntax this app
# builds its filtergraphs around need a recent ffmpeg; 9 is what the app is
# developed and tested against.
MINIMUM_FFMPEG_VERSION = (9,)

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


def _get_setting(key: str) -> str | None:
    try:
        from PySide6.QtCore import QSettings
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        val = settings.value(key)
        return str(val) if val else None
    except Exception:
        return None


def set_ffmpeg_dir_override(dir_path: str) -> None:
    from PySide6.QtCore import QSettings
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    settings.setValue("ffmpeg_dir", dir_path)
    find_binary.cache_clear()
    check_tools.cache_clear()


def exe_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


@lru_cache(maxsize=None)
def find_binary(name: str) -> str:
    """Returns a path (or bare name) to invoke for `name` (ffmpeg/ffprobe)."""
    exe = f"{name}.exe" if os.name == "nt" else name

    override = _get_setting("ffmpeg_dir")
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
        proc = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001 - any failure here means "not usable"
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


@lru_cache(maxsize=1)
def check_tools() -> ToolsStatus:
    """Cached: each call shells out twice, and this is consulted on every
    window construction. set_ffmpeg_dir_override() clears it."""
    return ToolsStatus(ffmpeg=check_tool("ffmpeg"), ffprobe=check_tool("ffprobe"))
