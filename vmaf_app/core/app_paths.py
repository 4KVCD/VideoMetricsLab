"""Stable per-user storage paths shared by every app launcher."""
from __future__ import annotations

from pathlib import Path

DATA_DIR_NAME = ".videometricslab"


def user_data_dir() -> Path:
    """A host-independent home for settings and cached calculations.

    Qt's application-data paths can be virtualized according to the process
    that launched the app. A directory directly below the user's home is
    stable whether the app starts from an editor, terminal, or packaged build.
    """
    return Path.home() / DATA_DIR_NAME


def settings_file() -> Path:
    return user_data_dir() / "settings.json"
