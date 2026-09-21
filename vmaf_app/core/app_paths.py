"""Stable per-user storage paths shared by every app launcher."""
from __future__ import annotations

from pathlib import Path

DATA_DIR_NAME = ".videometricslab"
LEGACY_DATA_DIR_NAME = ".vmaf-calculator"


def user_data_dir() -> Path:
    """A host-independent home for settings and cached calculations.

    Qt's application-data paths can be virtualized according to the process
    that launched the app.  A directory directly below the user's home is
    stable whether the app starts from an editor, terminal, or packaged build.
    """
    target = Path.home() / DATA_DIR_NAME
    legacy = Path.home() / LEGACY_DATA_DIR_NAME
    if not target.exists() and legacy.exists():
        try:
            legacy.rename(target)
        except OSError:
            # A running older build may still have a cache file open. Keep
            # using the existing data rather than making settings/results
            # appear to vanish, and retry the rename on a later launch.
            return legacy
    return target


def settings_file() -> Path:
    return user_data_dir() / "settings.json"
