"""Session-only video colors shared by metric and bitrate plots.

This is presentation state, not result/cache metadata. Keep reservations
after a row is removed so re-adding its video restores the same color.
"""
from __future__ import annotations

import os
from pathlib import Path

PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)


class VideoColors:
    """Owned by one app window; accessed only when its UI adds a video."""

    def __init__(self) -> None:
        self._colors: dict[str, str] = {}

    @staticmethod
    def path_key(path: Path) -> str:
        try:
            full_path = str(Path(path).resolve())
        except OSError:
            full_path = os.path.abspath(path)
        # Windows paths are case-insensitive; Linux paths need not be.
        return os.path.normcase(full_path)

    def for_path(self, path: Path) -> str:
        key = self.path_key(path)
        if key not in self._colors:
            self._colors[key] = PALETTE[len(self._colors) % len(PALETTE)]
        return self._colors[key]
