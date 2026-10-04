"""Compares VMAF v1 with the GPU (vmaf_app.core.vmaf_v1_gpu) with libvmaf's
CPU VMAF v1 on the same frames: the four features as bit patterns and the
score per frame, and times both.

    python scripts/compare_vmaf_v1.py REFERENCE DISTORTED [--model 3d0h]
        [--frames 48] [--start 60] [--size 1920x1080] [--bits 8] [--device 0]
    python scripts/compare_vmaf_v1.py --matrix [--device 1]

--matrix: every bundled VMAF v1 model over sizes, bit depths, frame skipping
and the frames that take rare paths. Exit code 0 when all are identical.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_vmaf_vulkan as frames_of

from vmaf_app.core import vmaf_v1_gpu, vmaf_vulkan

MODELS = Path(__file__).resolve().parents[1] / "vmaf_app" / "models"


def model_path(name: str) -> Path:
    matches = sorted(MODELS.glob(f"*/vmaf_v1.0.16_{name}.json"))
    if not matches:
        raise SystemExit(f"no bundled VMAF v1 model named {name}")
    return matches[0]


def cpu_run(model: Path, width: int, height: int, bits: int, reference, distorted, step: int, threads: int):
    """libvmaf's CPU VMAF v1 (vmaf_v1_gpu.cpu_reference) and the seconds it took."""
    started = time.perf_counter()
    scored, values, scores = vmaf_v1_gpu.cpu_reference(model, width, height, bits, reference, distorted, step, threads)
    return scored, values, scores, time.perf_counter() - started


def gpu_run(model: Path, width: int, height: int, bits: int, reference, distorted, step: int, device):
    scorer = vmaf_v1_gpu.V1Scorer(width, height, bits, model, step, device=device)
    try:
        started = time.perf_counter()
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        scored, values = scorer.features()
        elapsed = time.perf_counter() - started
        scores = vmaf_v1_gpu.predict(model, scored, values)
        return scored, values, scores, elapsed
    finally:
        scorer.close()


def compare(label: str, model: Path, width: int, height: int, bits: int, reference, distorted, step: int = 1,
            device: int | None = None, threads: int = 12) -> bool:
    count = min(len(reference), len(distorted))
    reference, distorted = reference[:count], distorted[:count]
    c_frames, c_values, c_scores, c_time = cpu_run(model, width, height, bits, reference, distorted, step, threads)
    g_frames, g_values, g_scores, g_time = gpu_run(model, width, height, bits, reference, distorted, step, device)
    identical = np.array_equal(c_frames, g_frames)
    lines = []
    for feature in c_values:
        cpu, gpu = c_values[feature], g_values[feature]
        same = np.array_equal(cpu.view(np.uint64), gpu.view(np.uint64))
        identical &= same
        if not same:
            lines.append(f"    {feature}: {int(np.sum(cpu != gpu))}/{len(cpu)} frames differ, "
                         f"at most by {np.nanmax(np.abs(cpu - gpu)):.3e}")
    same = np.array_equal(c_scores, g_scores, equal_nan=True)
    identical &= same
    if not same:
        lines.append(f"    score: at most {np.nanmax(np.abs(c_scores - g_scores)):.6f} apart")
    print(f"{'IDENTICAL' if identical else 'DIFFERENT'}  {label}: {width}x{height} {bits}-bit, {count} frames"
          f"{f', every {step}' if step > 1 else ''}, {model.stem.replace('vmaf_v1.0.16_', '')}; "
          f"VMAF v1 {np.nanmean(g_scores):.6f}; CPU {count / c_time:.1f} fps, GPU {count / g_time:.1f} fps", flush=True)
    for line in lines:
        print(line)
    return bool(identical)


def matrix(device: int | None) -> bool:
    same = True
    models = sorted(MODELS.glob("*/vmaf_v1.0.16_*.json"))
    sizes = [(640, 480, 8), (642, 362, 10), (1280, 720, 8), (1365, 768, 10), (1920, 1080, 8), (3840, 2160, 10)]
    for index, model in enumerate(models):
        for width, height, bits in sizes:
            count = 4 if width * height > 2_000_000 else 7
            same &= compare("synthetic", model, width, height, bits,
                            *frames_of.synthetic(width, height, bits, count, width + index), 1, device)
    model = model_path("3d0h")
    for kind in ("black", "white", "black-white", "identical", "noise", "extremes", "still", "sharpened", "inverted"):
        for bits in (8, 10):
            same &= compare(kind, model, 640, 360, bits, *frames_of.special(kind, 640, 360, bits, 6), 1, device)
    hfr = model_path("hfr_3d0h")
    same &= compare("every 2nd frame", hfr, 640, 360, 8, *frames_of.synthetic(640, 360, 8, 9, 3), 2, device)
    same &= compare("every 5th frame", model, 640, 360, 10, *frames_of.synthetic(640, 360, 10, 11, 4), 5, device)
    for count in (1, 2, 3):
        same &= compare(f"{count} frame(s)", hfr, 640, 360, 8, *frames_of.synthetic(640, 360, 8, count, 5), 1, device)
    print("ALL IDENTICAL" if same else "SOME DIFFER")
    return same


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", nargs="?")
    parser.add_argument("distorted", nargs="?")
    parser.add_argument("--model", default="3d0h")
    parser.add_argument("--synthetic")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--size")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", type=int)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--matrix", action="store_true")
    arguments = parser.parse_args()
    for device in vmaf_vulkan.devices():
        print(f"Vulkan GPU {device.index}: {device.name}")
    if arguments.matrix:
        return 0 if matrix(arguments.device) else 1
    model = model_path(arguments.model)
    if arguments.synthetic:
        width, height = (int(part) for part in arguments.synthetic.split("x"))
        reference, distorted = frames_of.synthetic(width, height, arguments.bits, arguments.frames, 1)
        label = "synthetic"
    else:
        size = tuple(int(part) for part in arguments.size.split("x")) if arguments.size else None
        reference, width, height = frames_of.decode(arguments.reference, arguments.start, arguments.frames, size,
                                                    arguments.bits)
        distorted, _, _ = frames_of.decode(arguments.distorted, arguments.start, arguments.frames, (width, height),
                                           arguments.bits)
        label = Path(arguments.distorted).name
    same = compare(label, model, width, height, arguments.bits, reference, distorted, arguments.step,
                   arguments.device, arguments.threads)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
