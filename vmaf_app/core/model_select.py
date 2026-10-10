"""Turns a row's VMAF model choice into the FFmpeg `model=` value it means,
choosing the 4K model for "Auto" at 4K. It has no Qt and no I/O, so it can
be tested on its own.
"""
from __future__ import annotations

from vmaf_app.core.builtin_models import builtin_choice, builtin_model_path, is_builtin_model_choice
from vmaf_app.core.models import VmafOptions

# A comparison at or above this resolution counts as 4K, for "Auto" to pick
# the 4K VMAF model.
UHD_WIDTH_THRESHOLD = 3840
UHD_HEIGHT_THRESHOLD = 2160

DEFAULT_MODEL = "version=vmaf_v0.6.1"
UHD_MODEL = "version=vmaf_4k_v0.6.1"

# The two special values a row's `model_choice` can hold instead of an
# FFmpeg model string.
AUTO_MODEL_CHOICE = "__auto__"
CUSTOM_MODEL_CHOICE = "__custom__"


def model_for_resolution(width: int, height: int) -> str:
    """The model "Auto" picks for frames compared at this size."""
    if width >= UHD_WIDTH_THRESHOLD or height >= UHD_HEIGHT_THRESHOLD:
        return UHD_MODEL
    return DEFAULT_MODEL


def resolve_model(options: VmafOptions, width: int, height: int) -> str:
    """Returns the FFmpeg model= value for a row's model_choice (and
    custom_model_path), choosing the 4K model for "Auto" at 4K.

    `width` and `height` are the size the two videos are compared at, not
    either video's own size: one is scaled to the other before libvmaf sees
    them, so a 1080p encode scaled up to a 4K master is a 4K comparison (see
    geometry.analysis_dimensions).

    Raises ValueError if the row asks for a custom model but names no file.
    """
    if options.model_choice == CUSTOM_MODEL_CHOICE:
        if not options.custom_model_path:
            raise ValueError("No custom model file selected.")
        # run_vmaf() copies the file into its run's temporary folder and
        # names it by its bare file name, so the full path is fine here.
        return f"path={options.custom_model_path}"
    if is_builtin_model_choice(options.model_choice):
        model_id = options.model_choice.removeprefix("__builtin:")
        return f"path={builtin_model_path(model_id)}"
    if options.model_choice == AUTO_MODEL_CHOICE:
        return model_for_resolution(width, height)
    return options.model_choice


# VMAF v1's Auto: below 4K, the 1080p model at three screen heights; at 4K,
# the 4K model at one and a half. Those are the viewing conditions v0.6.1's
# standard and 4K models were made for, so Auto means the same in both
# columns.
V1_DEFAULT_MODEL = builtin_choice("vmaf_v1_3d0h")
V1_UHD_MODEL = builtin_choice("vmaf_v1_1d5h_2160")


def v1_model_for_resolution(width: int, height: int) -> str:
    """The bundled VMAF v1 model Auto picks for frames compared at this size."""
    if width >= UHD_WIDTH_THRESHOLD or height >= UHD_HEIGHT_THRESHOLD:
        return V1_UHD_MODEL
    return V1_DEFAULT_MODEL


def resolve_v1_model(options: VmafOptions, width: int, height: int) -> str:
    """Returns the FFmpeg model= value for the row's VMAF v1 choice
    ("path=<file>")."""
    choice = options.model_choice_v1
    if choice == AUTO_MODEL_CHOICE:
        choice = v1_model_for_resolution(width, height)
    if not is_builtin_model_choice(choice):
        raise ValueError(f"Not a VMAF v1 model: {choice}")
    return f"path={builtin_model_path(choice.removeprefix('__builtin:'))}"


def is_v1_choice(choice: str | None) -> bool:
    """Whether a model choice is a bundled VMAF v1 model. Old rows, results
    and saved scores can carry one as their VMAF model, from before VMAF v1
    had its own column."""
    return bool(choice) and is_builtin_model_choice(choice)

