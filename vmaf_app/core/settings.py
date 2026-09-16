"""App-wide preferences, persisted between runs.

Deliberately separate from VmafOptions: those describe how one video is
scored and belong to a row, while these describe how the app behaves and
belong to the user.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from vmaf_app.core.app_paths import settings_file

#: Bumped whenever a saved settings file needs upgrading in place. 1 turned
#: the metric defaults on for files written before all four were the default;
#: 2 turned parallel scoring on for many-core machines that had never chosen.
SETTINGS_VERSION = 2

#: Above this many logical cores, a fresh install scores two videos at once.
#: One libvmaf job leaves a big CPU largely idle (see MAX_PARALLEL_JOBS for
#: the measurement); below it, a second job mostly competes with the first.
PARALLEL_BY_DEFAULT_ABOVE_CORES = 12


def default_parallel_jobs(cores: int | None = None) -> int:
    """The parallel-jobs setting a machine with `cores` logical cores starts
    with: two above PARALLEL_BY_DEFAULT_ABOVE_CORES, otherwise one."""
    if cores is None:
        cores = os.cpu_count() or 1
    return 2 if cores > PARALLEL_BY_DEFAULT_ABOVE_CORES else 1


@dataclass
class Settings:
    # Folder holding ffmpeg.exe/ffprobe.exe. Empty means "search PATH".
    ffmpeg_dir: str = ""
    # Where completed runs are cached. Empty means the shared per-user default.
    cache_dir: str = ""
    # Pre-selected folder for CSV/PNG exports. Empty means "ask each time".
    export_dir: str = ""

    # What a newly added video starts with. Per-row settings are edited in
    # the Options panel; these are only the starting point.
    #
    # All four metrics, because they share one decode pass: PSNR and SSIM are
    # libvmaf features computed from the frame pair VMAF already holds, and
    # XPSNR is chained into the same graph. Measured on a 10s 1080p pair,
    # adding PSNR and SSIM to VMAF cost 0.01s of 2.81s, and XPSNR another
    # 0.8s -- against 2.2s for a second pass to fetch it separately. Leaving
    # them off saved almost nothing and meant re-running the whole video to
    # answer a question the first run could have answered.
    default_gpu_decode: bool = True
    default_compute_psnr: bool = True
    default_compute_ssim: bool = True
    default_compute_xpsnr: bool = True
    default_compute_vmaf: bool = True
    default_compute_vmaf_neg: bool = False
    graph_metric: str = "vmaf"

    # Which upgrades have already been applied to the saved file. See
    # SETTINGS_VERSION and _migrate.
    settings_version: int = SETTINGS_VERSION

    # How many videos to score at once (1 or 2 -- see MAX_PARALLEL_JOBS).
    # libvmaf does not saturate a modern many-core CPU on its own, so a
    # second job largely fills the gap rather than competing for it -- which
    # is why the default depends on the machine (default_parallel_jobs).
    # Kept out of VmafOptions on purpose: it changes how fast results
    # arrive, never what they are, so it must not take part in cache
    # identity.
    parallel_jobs: int = field(default_factory=default_parallel_jobs)

    # Reuse a cached result when a video is added, instead of recomputing.
    use_cache: bool = True

    # Restore the window to the size it was closed at.
    remember_window_size: bool = True
    window_width: int = 1280
    window_height: int = 800

    # Frame Compare is an SDR QWidget surface. This controls how HDR frames
    # are converted for preview and is independent of the VMAF recipe.
    frame_preview_color_mode: str = "display_aware"

    @staticmethod
    def path() -> Path:
        path = settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls) -> Settings:
        """Reads the saved settings, falling back to defaults for anything
        missing or unreadable -- a corrupt or half-written file must not stop
        the app starting, and a file from an older build won't mention
        fields added since."""
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except Exception:
            return cls()  # no file: the dataclass defaults are already current
        if not isinstance(data, dict):
            return cls()
        try:
            version = int(data.get("settings_version", 0))
        except (TypeError, ValueError, OverflowError):
            version = 0
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in known})
        if settings._migrate(version):
            # Persisted straight away, so an upgrade happens exactly once. If
            # it only lived in memory, a user who turned a metric back off
            # would find it on again at every launch.
            settings.save()
        return settings

    def _migrate(self, from_version: int) -> bool:
        """Brings a settings file written by an older build up to date.

        Returns whether anything changed. Each step is written against the
        version it upgrades from, so a file several versions behind is
        carried forward through all of them in order.
        """
        if from_version >= SETTINGS_VERSION:
            return False
        if from_version < 1:
            # All four metrics come from one decode pass, so computing only
            # VMAF saved almost no time while making PSNR/SSIM/XPSNR cost a
            # whole second run of the video to obtain later.
            self.default_compute_psnr = True
            self.default_compute_ssim = True
            self.default_compute_xpsnr = True
            self.default_compute_vmaf = True
        if from_version < 2 and self.parallel_jobs == 1:
            # One was the only default before, so a saved 1 says nothing
            # about whether it was chosen. Files that already say 2 are left
            # alone, and once this has run, so is any later choice of 1.
            self.parallel_jobs = default_parallel_jobs()
        self.settings_version = SETTINGS_VERSION
        return True

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
