"""VMAF model files shipped with VideoMetricsLab.

The VMAF v1 models are distributed by Netflix as BSD-2-Clause-Patent
licensed data files.  They are kept as ordinary files (rather than relying on
libvmaf's installation directory) so the packaged app can pass them to any
compatible external FFmpeg build.
"""
from __future__ import annotations

from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"

# Stable IDs are serialized in VmafOptions.model_choice.  Do not serialize an
# absolute path: it differs between source and frozen installs and would make
# saved settings/results non-portable.
BUILTIN_MODEL_FILES: dict[str, str] = {
    "vmaf_v1_3d0h": "vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json",
    "vmaf_v1_5d0h": "vmaf_v1.0.16/vmaf_v1.0.16_5d0h.json",
    "vmaf_v1_1d5h_2160": "vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json",
    "vmaf_v1_3d0h_2160": "vmaf_v1.0.16/vmaf_v1.0.16_3d0h_2160.json",
    "vmaf_v1_hfr_3d0h": "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json",
    "vmaf_v1_hfr_5d0h": "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_5d0h.json",
    "vmaf_v1_hfr_1d5h_2160": "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json",
    "vmaf_v1_hfr_3d0h_2160": "vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h_2160.json",
}


def builtin_model_path(model_id: str) -> Path:
    """Return the installed path for a built-in model ID.

    A missing asset is an installation/build error, not a reason to silently
    fall back to VMAF v0 and report misleading scores.
    """
    try:
        relative = BUILTIN_MODEL_FILES[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown built-in VMAF model: {model_id}") from exc
    path = _MODEL_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Built-in VMAF model is missing: {path}")
    return path


def is_builtin_model_choice(choice: str) -> bool:
    return choice.startswith("__builtin:") and choice.removeprefix("__builtin:") in BUILTIN_MODEL_FILES


def builtin_choice(model_id: str) -> str:
    if model_id not in BUILTIN_MODEL_FILES:
        raise ValueError(f"Unknown built-in VMAF model: {model_id}")
    return f"__builtin:{model_id}"
