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
    default_compute_vmaf_neg: bool = True
    # Every metric starts ticked (Brian, 2026-09-25): the table is for
    # comparing encodes on all of them. The perceptual metrics are passes
    # of their own (Vship on the GPU; SSIMULACRA2/Butteraugli far slower on
    # the CPU, where long videos ask first), and CVVDP shows n/a without a
    # supported GPU. Saved settings keep what the user chose.
    default_compute_ssimulacra2: bool = True
    default_compute_butteraugli: bool = True
    default_compute_cvvdp: bool = True
    # "gpu" (Vship, falling back to the CPU tools without a supported GPU)
    # or "cpu" (libjxl's tools). Set from the Options panel's GPU/CPU
    # choice, which is also what a newly added video starts with.
    default_ssimulacra2_backend: str = "gpu"
    default_butteraugli_backend: str = "gpu"
    graph_metric: str = "vmaf"

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

    # How many test videos Video Compare decodes at once: the selected one
    # plus neighbours, so switching with the arrow keys is instant. Each one
    # is a running GPU decoder (about a quarter of a gigabyte of RAM for 4K),
    # which is why it is a choice. See video_playback.neighbour_indices for
    # which neighbours.
    compare_decoded_videos: int = 3

    # Metrics left out of the test-video table (the Metrics... picker). A
    # hidden metric is not calculated either; its cached scores come back
    # when it is shown again. Stored as the hidden set rather than the shown
    # one, so a metric added in a later version appears instead of being
    # silently off for everyone who already has a settings file.
    hidden_metrics: list[str] = field(default_factory=list)

    # CVVDP presets the user saved: [{"name": ..., "settings":
    # CvvdpSettings.to_dict()}], and the preset new videos start with ("" is
    # the built-in default). Saving a preset makes it the default, since a
    # user who tunes CVVDP will keep using their own settings.
    cvvdp_presets: list[dict] = field(default_factory=list)
    cvvdp_default_preset: str = ""

    @staticmethod
    def path() -> Path:
        path = settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls) -> Settings:
        """Read saved settings, using defaults for missing or unreadable data.

        Unknown keys are ignored so a stray or future field cannot stop the
        application from starting.
        """
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(data, dict):
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
