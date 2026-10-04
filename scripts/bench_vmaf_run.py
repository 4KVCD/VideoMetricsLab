"""Times a whole VMAF run of the app (FFmpeg decoding, cropping and scaling,
then VMAF and VMAF NEG) on each backend, and compares their per-frame scores.

    python scripts/bench_vmaf_run.py REFERENCE DISTORTED [--seconds 30]
        [--backends cpu cuda vulkan:0 vulkan:1] [--metrics vmaf vmaf_neg vmaf_v1]
        [--repeat 2] [--no-gpu-decode]

A backend is "cpu" (FFmpeg's libvmaf), "cuda" (the bundled libvmaf) or
"vulkan:N" (vmaf_vulkan on Vulkan's GPU N; see compare_vmaf_vulkan.py for
the numbers). The first backend is what the others' scores are compared
with. Each run's best time of --repeat is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmaf_app.core import vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import VmafOptions
from vmaf_app.core.vmaf_runner import run_vmaf


def run(backend: str, source, distorted, options: VmafOptions, repeat: int):
    if backend == "cpu":
        options = VmafOptions(**{**options.__dict__, "vmaf_on_gpu": False})
    else:
        name, _, device = backend.partition(":")
        vmaf_cuda._probed = (True, backend)
        vmaf_cuda._backend = (name, int(device) if device else None)
        vmaf_v1_gpu._probed = (True, backend)  # VMAF v1's GPU half is Vulkan's with either
    best, result = float("inf"), None
    for _ in range(repeat):
        started = time.perf_counter()
        result = run_vmaf(source, distorted, options)
        best = min(best, time.perf_counter() - started)
    return best, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("distorted")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--backends", nargs="+", default=["cpu", "cuda", "vulkan:0"])
    parser.add_argument("--metrics", nargs="+", default=["vmaf", "vmaf_neg"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--no-gpu-decode", action="store_true")
    arguments = parser.parse_args()
    names = {device.index: device.name for device in vmaf_vulkan.devices()}
    source, distorted = probe_video(Path(arguments.reference)), probe_video(Path(arguments.distorted))
    options = VmafOptions(compute_vmaf="vmaf" in arguments.metrics, compute_vmaf_neg="vmaf_neg" in arguments.metrics,
                          compute_vmaf_v1="vmaf_v1" in arguments.metrics,
                          duration_limit=arguments.seconds, gpu_decode=not arguments.no_gpu_decode)
    print(f"{distorted.path.name} against {source.path.name}, first {arguments.seconds:g} s, "
          f"{' and '.join(arguments.metrics)}")
    reference = None
    for backend in arguments.backends:
        seconds, result = run(backend, source, distorted, options, arguments.repeat)
        frames = result.frames
        label = backend if not backend.startswith("vulkan:") else f"vulkan ({names[int(backend[7:])]})"
        line = f"  {label:42} {seconds:7.2f} s  {len(frames) / seconds:7.1f} fps"
        for key in arguments.metrics:
            values = np.asarray(frames.values(key), dtype=np.float64)
            line += f"  {key} {np.mean(values):.6f}"
            if reference is not None:
                other = np.asarray(reference.values(key), dtype=np.float64)
                same = len(other) == len(values) and np.array_equal(other, values)
                line += " (identical)" if same else f" (max difference {np.max(np.abs(other - values)):.6f})"
        print(line, flush=True)
        if reference is None:
            reference = frames
    return 0


if __name__ == "__main__":
    sys.exit(main())
