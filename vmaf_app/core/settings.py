"""The app's preferences (Settings), saved in settings.json between sessions.

They are kept apart from VmafOptions on purpose: those say how one video is
scored and belong to a row, while these say how the app behaves and belong
to the user.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vmaf_app.core.app_paths import settings_file

_log = logging.getLogger(__name__)

#: Above this many logical cores, a new install scores two videos at once.
#: One libvmaf job leaves a big CPU largely idle (see MAX_PARALLEL_JOBS for
#: the measurement), while on fewer cores a second job mostly competes with
#: the first.
PARALLEL_BY_DEFAULT_ABOVE_CORES = 12


def default_parallel_jobs(cores: int | None = None) -> int:
    """The parallel-jobs setting a machine with `cores` logical cores starts
    with: two above PARALLEL_BY_DEFAULT_ABOVE_CORES, otherwise one."""
    if cores is None:
        cores = os.cpu_count() or 1
    return 2 if cores > PARALLEL_BY_DEFAULT_ABOVE_CORES else 1


@dataclass
class Settings:
    """The app's preferences, each with its default."""

    # The folder holding ffmpeg.exe and ffprobe.exe. Empty means "search the
    # PATH".
    ffmpeg_dir: str = ""
    # Where finished runs are saved. Empty means the per-user default.
    cache_dir: str = ""
    # The folder CSV and PNG exports go to. Empty means "ask each time".
    export_dir: str = ""

    # What a newly added video starts with. Each row's settings are edited
    # in the Options panel; these are only the starting point.
    #
    # The FFmpeg metrics all start on, because they share one decoding pass:
    # PSNR and SSIM are libvmaf features calculated from the frame pair VMAF
    # already holds, and XPSNR is in the same filter graph. On a 10 s 1080p
    # pair, adding PSNR and SSIM to VMAF cost 0.01 s of 2.81 s, and XPSNR
    # another 0.8 s, against 2.2 s for a second pass of its own. Leaving them
    # off saves almost nothing, and means running the whole video again to
    # answer a question the first run could have answered.
    default_gpu_decode: bool = True
    default_compute_psnr: bool = True
    default_compute_ssim: bool = True
    default_compute_xpsnr: bool = True
    default_compute_vmaf: bool = True
    default_compute_vmaf_neg: bool = True
    default_compute_vmaf_v1: bool = True
    # Every metric starts ticked (Brian's choice): the table is for comparing
    # encodes on all of them. The perceptual metrics are passes of their own
    # (Vship on the GPU; SSIMULACRA2 and Butteraugli are far slower on the
    # CPU, where long videos ask first), and CVVDP shows n/a without a
    # supported GPU. Saved settings keep what the user chose.
    default_compute_ssimulacra2: bool = True
    default_compute_butteraugli: bool = True
    default_compute_cvvdp: bool = True
    # "gpu" (Vship, falling back to the CPU tools without a GPU Vship can
    # use, or where its build scores the metric wrongly; see
    # perceptual_vship.SCORED_WRONGLY) or "cpu" (libjxl's tools). Set from the
    # Options panel's GPU/CPU choice, which is also what a newly added video
    # starts with.
    default_ssimulacra2_backend: str = "gpu"
    default_butteraugli_backend: str = "gpu"
    # Whether VMAF v0.6.1, NEG and v1 run on the GPU (VmafOptions.vmaf_on_gpu).
    # Set from the Options panel's GPU/CPU choice, which is also what a newly
    # added video starts with.
    default_vmaf_on_gpu: bool = True
    graph_metric: str = "vmaf"

    # How many videos to score at once (1 or 2; see MAX_PARALLEL_JOBS).
    # libvmaf alone does not fill a modern many-core CPU, so a second job
    # mostly uses idle cores rather than competing, which is why the default
    # depends on the PC (default_parallel_jobs). It is kept out of
    # VmafOptions on purpose: it changes how fast results arrive, never what
    # they are, so it must not be part of the cache key.
    parallel_jobs: int = field(default_factory=default_parallel_jobs)

    # Calculate a video's GPU metrics (SSIMULACRA2, Butteraugli, CVVDP) in
    # one Vship pass, decoding the video once, instead of one pass each (see
    # perceptual_vship.run_vship_task). Off by default, because at 4K it
    # needs about 7.3 GB of GPU memory, against 4.8 GB one at a time. Like
    # parallel_jobs, it changes how the scores are calculated, never what
    # they are, so it is not part of the cache key.
    gpu_metrics_together: bool = False

    # Which of Vship's builds calculates the GPU metrics: "auto" (CUDA on
    # NVIDIA, HIP on AMD, Vulkan on other GPUs), or "vulkan", "cuda" or "hip"
    # tried first (see perceptual_vship.VSHIP_BACKENDS). Like
    # gpu_metrics_together, it is not part of the cache key: a metric a build
    # scores wrongly is not scored on it (perceptual_vship.SCORED_WRONGLY).
    gpu_backend: str = "auto"

    # Use a saved result when a video is added, instead of calculating again.
    use_cache: bool = True

    # Ask GitHub for a newer release when the app starts
    # (vmaf_app.core.update_check); and a release the user chose to skip.
    check_for_updates: bool = True
    skipped_update_version: str = ""

    # The window's language (a code from vmaf_app.i18n.LANGUAGES). Empty
    # means Windows' language, or English when there is no translation.
    language: str = ""

    # The window's colours: "" follows Windows (light or dark, changing when
    # it changes), and "light" or "dark" stay so whatever Windows uses
    # (vmaf_app.ui.theme; issue #5).
    theme: str = ""

    # Restore the window to the size it was closed at.
    remember_window_size: bool = True
    window_width: int = 1280
    window_height: int = 800

    # The folders the source's and the test videos' file dialogs open in:
    # where a source, and a test video, was last chosen or dropped from. They
    # are kept apart, because the two are often in different places (issue
    # #4). Empty means the other one's, or else the dialogs' own default.
    last_reference_dir: str = ""
    last_test_dir: str = ""
    # Reopen at startup the source and test videos that were in the window
    # when it was closed, those that are still there. Off by default (Brian,
    # issue #4): a new session starts empty unless asked otherwise. The paths
    # are kept only while it is on.
    remember_videos: bool = False
    remembered_reference: str = ""
    remembered_tests: list[str] = field(default_factory=list)
    # Videos tab, "Highlight best/worst results": each metric's best score in
    # green and its worst in red, among the test videos. Off by default
    # (Brian, from a user's request, as FFMetrics colours its table).
    highlight_best_worst: bool = False

    # Frame Compare shows SDR pictures, so this sets how HDR frames are
    # converted for it. It does not change the metrics.
    frame_preview_color_mode: str = "display_aware"

    # How many test videos Video Compare decodes at once: the selected one
    # and its neighbours, so that switching with the arrow keys is instant.
    # Each is a running GPU decoder (about a quarter of a gigabyte of memory
    # for 4K), which is why it is a choice. See
    # video_playback.neighbour_indices for which neighbours.
    compare_decoded_videos: int = 3

    # The metrics left out of the test-video table (the Metrics... picker). A
    # hidden metric is not calculated either, and its saved scores come back
    # when it is shown again. The hidden set is stored rather than the shown
    # one, so that a metric added in a later version appears, instead of
    # being quietly off for everyone who already has a settings file.
    hidden_metrics: list[str] = field(default_factory=list)

    # The CVVDP presets the user saved, as [{"name": ..., "settings":
    # CvvdpSettings.to_dict()}], and the preset new videos start with (""
    # is the built-in default). Saving a preset makes it the default, since
    # a user who tunes CVVDP will keep using their own settings.
    cvvdp_presets: list[dict] = field(default_factory=list)
    cvvdp_default_preset: str = ""

    @staticmethod
    def path() -> Path:
        """settings.json's path, its folder made if need be."""
        path = settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls) -> Settings:
        """Reads the saved settings, using defaults for missing or unreadable
        values.

        Unknown keys are ignored, so that a stray or future field cannot stop
        the app from starting. So is a value of the wrong type, as in a
        hand-edited file ("parallel_jobs": "2" or "hidden_metrics": "psnr"):
        that setting keeps its default, and the others load. Kept as it is,
        such a value would fail later, wherever the setting is used.
        """
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(data, dict):
            return cls()
        kinds = typing.get_type_hints(cls)
        kept = {}
        for name, value in data.items():
            if name not in kinds:
                continue
            if _fits(value, kinds[name]):
                kept[name] = value
            else:
                _log.warning("Settings: %s is %r, not a %s; using its default", name, value, kinds[name])
        return cls(**kept)

    def save(self) -> str | None:
        """Saves the settings. Returns None on success, or a message to show
        the user.

        The file is written whole beside the old one, then put in its place.
        Written in place, a crash or a full disk part-way through would leave
        a cut-off file, which the next start would read as no settings at
        all."""
        path = self.path()
        try:
            descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        except OSError as e:
            return f"Could not save settings: {e}"
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(self), indent=2))
            os.replace(temporary, path)
        except OSError as e:
            with contextlib.suppress(OSError):
                temporary.unlink()
            return f"Could not save settings: {e}"
        return None

    # ---------------------------------------------------------------- paths
    def _dir(self, value: str) -> Path | None:
        """A folder setting as a path, or None when it is empty."""
        value = value.strip()
        return Path(value) if value else None

    def ffmpeg_dir_path(self) -> Path | None:
        """The FFmpeg folder, or None."""
        return self._dir(self.ffmpeg_dir)

    def cache_dir_path(self) -> Path | None:
        """The results folder, or None for the default."""
        return self._dir(self.cache_dir)

    def export_dir_path(self) -> Path | None:
        """The exports folder, or None to ask each time."""
        return self._dir(self.export_dir)

    def default_extra_features(self) -> list[str]:
        """The libvmaf features these defaults ask for. XPSNR is left out on
        purpose: it is a separate FFmpeg filter, not a libvmaf feature."""
        features = []
        if self.default_compute_psnr:
            features.append("name=psnr")
        if self.default_compute_ssim:
            features.append("name=float_ssim")
        return features


def _fits(value, kind) -> bool:
    """Whether a value read from the settings file is of a field's type:
    bool, int, str, or a list of one of them (or of dicts)."""
    if kind is bool:
        return isinstance(value, bool)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in (str, float, dict):
        return isinstance(value, kind)
    if typing.get_origin(kind) is list:
        (item,) = typing.get_args(kind) or (object,)
        return isinstance(value, list) and all(_fits(element, item) for element in value)
    return kind is object
