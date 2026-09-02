"""Turning a row's model *choice* into the concrete ffmpeg `model=` value.

This is pure decision logic with no Qt and no I/O, so it lives in core and is
testable on its own -- it used to sit in main_window.py, which meant the test
for 4K auto-selection had to import a private name out of a UI module.
"""
from __future__ import annotations

from vmaf_app.core.models import VideoInfo, VmafOptions

# A distorted video at or above this resolution is considered UHD/4K for the
# purpose of auto-selecting the 4K VMAF model (matches the "4K or higher" ask).
UHD_WIDTH_THRESHOLD = 3840
UHD_HEIGHT_THRESHOLD = 2160

DEFAULT_MODEL = "version=vmaf_v0.6.1"
UHD_MODEL = "version=vmaf_4k_v0.6.1"

# The two sentinel values a row's `model_choice` can hold instead of a real
# ffmpeg model string.
AUTO_MODEL_CHOICE = "__auto__"
CUSTOM_MODEL_CHOICE = "__custom__"


def model_for_resolution(width: int, height: int) -> str:
    """The model auto-selection would pick for a distorted video of this size."""
    if width >= UHD_WIDTH_THRESHOLD or height >= UHD_HEIGHT_THRESHOLD:
        return UHD_MODEL
    return DEFAULT_MODEL


def resolve_model(options: VmafOptions, distorted_info: VideoInfo) -> str:
    """Resolves a row's model_choice (+ custom_model_path) into the concrete
    ffmpeg model= value, applying 4K auto-selection if chosen.

    Raises ValueError if the row asks for a custom model but names no file.
    """
    if options.model_choice == CUSTOM_MODEL_CHOICE:
        if not options.custom_model_path:
            raise ValueError("No custom model file selected.")
        # run_vmaf() copies this into its per-run temp dir and references
        # it by bare filename, so the raw absolute path is fine here.
        return f"path={options.custom_model_path}"
    if options.model_choice == AUTO_MODEL_CHOICE:
        return model_for_resolution(distorted_info.width, distorted_info.height)
    return options.model_choice
