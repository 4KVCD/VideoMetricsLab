"""Times VMAF's features on the GPU, on frames held in memory (no decoding):
Vulkan on every GPU, and libvmaf's CUDA code.

    python scripts/bench_vmaf_vulkan.py [REFERENCE DISTORTED] [--size 1920x1080]
        [--bits 8] [--frames 120] [--parts]

Without videos, synthetic frames. --parts also times motion, VIF and ADM
alone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_vmaf_vulkan as compare

from vmaf_app.core import vmaf_cuda, vmaf_vulkan


def timed(scorer, reference, distorted, frames: int, copy: bool = False) -> float:
    count = len(reference)
    try:
        for index in range(min(4, frames)):  # pipelines compiled, buffers touched
            scorer.add(reference[index % count], distorted[index % count])
        started = time.perf_counter()
        for index in range(frames):
            ref, dis = reference[index % count], distorted[index % count]
            scorer.add(bytearray(ref), bytearray(dis)) if copy else scorer.add(ref, dis)
        scorer.features() if hasattr(scorer, "features") else scorer.finish()
        return frames / (time.perf_counter() - started)
    finally:
        scorer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", nargs="?")
    parser.add_argument("distorted", nargs="?")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--start", type=float, default=300.0)
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    arguments = parser.parse_args()
    width, height = (int(part) for part in arguments.size.split("x"))
    if arguments.reference:
        reference, _, _ = compare.decode(arguments.reference, arguments.start, 12, (width, height), arguments.bits)
        distorted, _, _ = compare.decode(arguments.distorted, arguments.start, 12, (width, height), arguments.bits)
    else:
        reference, distorted = compare.synthetic(width, height, arguments.bits, 8, 1)
    print(f"{width}x{height} {arguments.bits}-bit, {arguments.frames} frames, VMAF and VMAF NEG")
    for device in vmaf_vulkan.devices():
        if not device.usable:
            continue

        def scorer(skip=0, device=device):
            return vmaf_vulkan.VulkanScorer(width, height, arguments.bits, compare.MODELS, device=device.index,
                                            native_double=arguments.native and device.has_double, skip=skip)

        line = f"  Vulkan, {device.name}: {timed(scorer(), reference, distorted, arguments.frames):.1f} fps"
        if arguments.parts:
            for label, skip in (("motion", 6), ("VIF", 5), ("ADM", 3), ("upload only", 7)):
                rate = timed(scorer(skip), reference, distorted, arguments.frames)
                line += f"; {label} {1000 / rate:.2f} ms"
        print(line)
    if not arguments.no_cuda and vmaf_cuda.LIBRARY_PATH.is_file():
        try:
            cuda = vmaf_cuda.GpuScorer(width, height, arguments.bits, compare.MODELS)
            print(f"  CUDA (libvmaf): {timed(cuda, reference, distorted, arguments.frames, copy=True):.1f} fps")
        except (OSError, vmaf_cuda.VmafGpuError) as error:
            print(f"  CUDA (libvmaf): unavailable ({error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
