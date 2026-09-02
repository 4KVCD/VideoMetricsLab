"""GPU / hardware-decode capability detection.

We keep this deliberately conservative: pick a decode hwaccel that's likely
to work for the source video's codec, and let vmaf_runner fall back to
software decode if the hardware path fails to launch.
"""
from __future__ import annotations

import platform
import subprocess
from functools import lru_cache

from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import GpuVendor

# hwaccel name -> codecs it reliably decodes via ffmpeg's generic hwaccel path
_HWACCEL_CODEC_SUPPORT = {
    "cuda": {"h264", "hevc", "vp8", "vp9", "mpeg2video", "mpeg4", "av1", "vc1"},
    "qsv": {"h264", "hevc", "vp8", "vp9", "mpeg2video", "av1"},
    "d3d11va": {"h264", "hevc", "vp9", "mpeg2video", "vc1", "av1"},
}

_VENDOR_PREFERRED_HWACCEL = {
    GpuVendor.NVIDIA: "cuda",
    GpuVendor.INTEL: "qsv",
    GpuVendor.AMD: "d3d11va",
}


@lru_cache(maxsize=1)
def available_hwaccels() -> set[str]:
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return set()
    lines = [l.strip() for l in proc.stdout.splitlines()]
    names = set()
    started = False
    for line in lines:
        if line.lower().startswith("hardware acceleration methods"):
            started = True
            continue
        if started and line:
            names.add(line)
    return names


@lru_cache(maxsize=1)
def detected_gpu_vendors() -> list[GpuVendor]:
    """Best-effort detection of installed GPU vendors (Windows only)."""
    if platform.system() != "Windows":
        return []
    try:
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return []
    names = proc.stdout.lower()
    vendors = []
    if "nvidia" in names:
        vendors.append(GpuVendor.NVIDIA)
    if "intel" in names:
        vendors.append(GpuVendor.INTEL)
    if "amd" in names or "radeon" in names:
        vendors.append(GpuVendor.AMD)
    return vendors


def pick_hwaccel(vendor: GpuVendor, codec_name: str) -> str | None:
    """Returns an ffmpeg -hwaccel value to try, or None for software decode."""
    hwaccels = available_hwaccels()
    codec_name = (codec_name or "").lower()

    if vendor == GpuVendor.NONE:
        return None

    if vendor == GpuVendor.AUTO:
        for v in detected_gpu_vendors():
            candidate = _VENDOR_PREFERRED_HWACCEL.get(v)
            if candidate and candidate in hwaccels and codec_name in _HWACCEL_CODEC_SUPPORT.get(candidate, set()):
                return candidate
        return None

    candidate = _VENDOR_PREFERRED_HWACCEL.get(vendor)
    if candidate and candidate in hwaccels and codec_name in _HWACCEL_CODEC_SUPPORT.get(candidate, set()):
        return candidate
    return None
