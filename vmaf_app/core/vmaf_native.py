"""Command contract for the out-of-process, GPU-native VMAF helper.

Only native-size/crop-only, same-depth 4:2:0 comparisons are qualified.
Resizing, mixed depths and unavailable native libraries retain the raw-pipe
path. Actual decoder/layout failures are retried there too, not accepted as
partially calculated results. Timestamp matching happens inside the helper
using FFmpeg framesync on tiny tags, never on full-size pixel canvases.
"""
from __future__ import annotations

import sys
from pathlib import Path

from vmaf_app.core.gpu import bit_depth
from vmaf_app.core.models import CropBox, VideoInfo

RUNTIME = Path(__file__).resolve().parents[1] / "tools" / "vmaf_native"
EXECUTABLE = RUNTIME / "vmaf_native.exe"
KERNEL = RUNTIME / "vmaf_prepare.ptx"
_FILES = ("vmaf_native.exe", "vmaf_prepare.ptx", "libvmaf.dll", "avcodec-63.dll", "avformat-63.dll",
          "avfilter-12.dll", "avutil-61.dll", "swscale-10.dll", "swresample-7.dll")


def crop(info: VideoInfo, box: CropBox | None) -> CropBox:
    return box if box is not None else CropBox(info.width, info.height, 0, 0)


def eligible(source: VideoInfo, test: VideoInfo, source_crop: CropBox | None, test_crop: CropBox | None,
             width: int, height: int, depth: int, cpu_output: bool, hw) -> bool:
    if sys.platform != "win32" or cpu_output or depth not in (8, 10):
        return False
    if hw.source not in (None, "cuda") or hw.distorted not in (None, "cuda"):
        return False
    for info, box in ((source, source_crop), (test, test_crop)):
        if not info.path.is_file() or info.pix_fmt not in ("yuv420p", "yuv420p10le") or bit_depth(info.pix_fmt) != depth:
            return False
        rect = crop(info, box)
        if (rect.w, rect.h) != (width, height) or min(rect.w, rect.h) < 32:
            return False
        if any(value < 0 or value % 2 for value in (rect.w, rect.h, rect.x, rect.y)):
            return False
        if rect.x + rect.w > info.width or rect.y + rect.h > info.height:
            return False
    return all((RUNTIME / name).is_file() for name in _FILES)


def command(source: VideoInfo, test: VideoInfo, source_crop: CropBox | None, test_crop: CropBox | None,
            depth: int, models: dict[str, str], subsample: int, duration: float, hw, log: Path,
            *, wait: str = "blocking") -> list[str]:
    # An owned CUDA context lets GPU waits sleep rather than consume CPU.
    # Keep the automatic mode available for reproducible A/B benchmarks.
    def rect(info, box):
        c = crop(info, box)
        return f"{c.w}:{c.h}:{c.x}:{c.y}"

    args = [str(EXECUTABLE), "--reference", str(source.path.resolve()), "--test", str(test.path.resolve()),
            "--reference-crop", rect(source, source_crop), "--test-crop", rect(test, test_crop),
            "--depth", str(depth), "--subsample", str(subsample), "--duration", f"{duration:.3f}",
            "--reference-decode", hw.source or "cpu", "--test-decode", hw.distorted or "cpu",
            # libvmaf's MSVC fopen takes a narrow name. The subprocess's
            # Unicode cwd is this log's folder; an ASCII basename works even
            # when the user's Windows profile or TEMP folder is non-ASCII.
            "--output", log.name, "--ptx", str(KERNEL), "--wait", wait]
    for name, version in models.items():
        args += ["--model", f"{name}={version}"]
    return args
