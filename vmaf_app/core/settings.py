"""App-wide preferences, persisted between runs.

Deliberately separate from VmafOptions: those describe how one video is
scored and belong to a row, while these describe how the app behaves and
belong to the user.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PySide6.QtCore import QStandardPaths


@dataclass
class Settings:
    # Folder holding ffmpeg.exe/ffprobe.exe. Empty means "search PATH".
    ffmpeg_dir: str = ""
    # Where completed runs are cached. Empty means the platform default.
    cache_dir: str = ""
    # Pre-selected folder for CSV/PNG exports. Empty means "ask each time".
    export_dir: str = ""

    # What a newly added video starts with. Per-row settings are edited in
    # the Options panel; these are only the starting point.
    default_gpu_decode: bool = True
    default_compute_psnr: bool = False
    default_compute_ssim: bool = False
    default_compute_xpsnr: bool = False

    # Reuse a cached result when a video is added, instead of recomputing.
    use_cache: bool = True

    # Restore the window to the size it was closed at.
    remember_window_size: bool = True
    window_width: int = 1280
    window_height: int = 800

    @staticmethod
    def path() -> Path:
        base = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation))
        base.mkdir(parents=True, exist_ok=True)
        return base / "settings.json"

    @classmethod
    def load(cls) -> Settings:
        """Reads the saved settings, falling back to defaults for anything
        missing or unreadable -- a corrupt or half-written file must not stop
        the app starting, and a file from an older build won't mention
        fields added since."""
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except Exception:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> str | None:
        """Returns None on success, or a message to show the user."""
        try:
            self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError as e:
            return f"Could not save settings: {e}"
        return None

    # ---------------------------------------------------------------- paths
    def _dir(self, value: str) -> Path | None:
        value = value.strip()
        return Path(value) if value else None

    def ffmpeg_dir_path(self) -> Path | None:
        return self._dir(self.ffmpeg_dir)

    def cache_dir_path(self) -> Path | None:
        return self._dir(self.cache_dir)

    def export_dir_path(self) -> Path | None:
        return self._dir(self.export_dir)

    def default_extra_features(self) -> list[str]:
        """The libvmaf feature list these defaults imply. XPSNR is absent on
        purpose: it's a separate ffmpeg filter, not a libvmaf feature."""
        features = []
        if self.default_compute_psnr:
            features.append("name=psnr")
        if self.default_compute_ssim:
            features.append("name=float_ssim")
        return features
