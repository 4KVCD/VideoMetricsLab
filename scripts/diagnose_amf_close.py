"""Runs a Vulkan VMAF comparison with AMD's decoder and traces the decoders'
shutdown, for an AMD PC: a shutdown that never returned was seen on a Radeon
780M and cannot be reproduced where the app is developed.

    .venv\\Scripts\\python.exe scripts\\diagnose_amf_close.py REFERENCE DISTORTED [--seconds 20]

REFERENCE should be longer than DISTORTED (or --seconds shorter than it), so
the reference's decoder is closed part-way through its video. VMAF is scored
with Vulkan forced on (whatever the app's probe says of this GPU), both
videos decoded by AMD's decoder in the scoring process, in a process of its
own with a time limit. It prints whether the run finished, how long it took,
whether a decoder "did not close" (the app gives a close 10 seconds and
carries on), the VMAF it got, and each shutdown step AMD's library was asked
for (VML_AMF_TRACE; the last line of a decoder that hung is the call that
never returned).

What it found (2026-10, Radeon 780M): the decoder that never returned from
Terminate had failed part-way on 10-bit H.264, which AMD's GPUs do not decode
and its library started on all the same; amf_frames.cpp now asks the
decoder's capabilities, and FFmpeg decodes such a video. Decoders of videos
the GPU does decode close in about 60 ms, stopped part-way or not. Draining
or flushing first, releasing without Terminate, and closing Vulkan before the
decoders made no difference to either.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def score(reference: Path, distorted: Path, seconds: float) -> int:
    """One run, in this process (and the scoring process it starts)."""
    import numpy as np

    from vmaf_app.core import vmaf_cuda, vmaf_vulkan
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.models import GpuVendor, VmafOptions
    from vmaf_app.core.vmaf_runner import run_vmaf

    logging.basicConfig(level=logging.INFO, format="LOG %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    device = vmaf_vulkan.best_device()
    if device is None:
        print("RESULT no GPU that Vulkan can calculate VMAF on")
        return 2
    vmaf_cuda._probed = (True, f"vulkan:{device.index}")
    vmaf_cuda._backend = ("vulkan", device.index)
    options = VmafOptions(compute_vmaf=True, compute_vmaf_neg=True, duration_limit=seconds, gpu_decode=True,
                          gpu_vendor=GpuVendor.AMD)
    result = run_vmaf(probe_video(reference), probe_video(distorted), options)
    values = np.asarray(result.frames.values("vmaf"), dtype=np.float64)
    print(f"RESULT {len(values)} frames, VMAF {np.mean(values):.6f} on {device.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("distorted", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--limit", type=float, default=300.0, help="seconds the run may take before it is ended")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        return score(args.reference, args.distorted, args.seconds)
    print(f"AMD decoder shutdown in a Vulkan VMAF run: {args.distorted.name} against {args.reference.name}, "
          f"first {args.seconds:g} s")
    with tempfile.TemporaryDirectory(prefix="vml-amf-close-") as folder:
        trace = Path(folder) / "trace.txt"
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "VML_AMF_TRACE": str(trace)}
        command = [sys.executable, str(Path(__file__).resolve()), str(args.reference), str(args.distorted),
                   "--seconds", str(args.seconds), "--run"]
        started = time.perf_counter()
        try:
            done = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=args.limit)
            output, code = done.stdout + done.stderr, str(done.returncode)
        except subprocess.TimeoutExpired as expired:
            output = (expired.stdout or b"").decode("utf-8", "replace") if isinstance(expired.stdout, bytes) \
                else (expired.stdout or "")
            code = f"ENDED after {args.limit:g} s without finishing (hung)"
        elapsed = time.perf_counter() - started
        lines = output.splitlines()
        result = next((line for line in lines if line.startswith("RESULT")), "RESULT none")
        stuck = [line for line in lines if "did not close" in line or "left open" in line]
        decoded = [line for line in lines if "decoded" in line.lower() and "LOG" in line][:4]
        print(f"exit {code}, {elapsed:.1f} s; {result[7:]}")
        print(f"decoders that did not close: {len(stuck)}")
        for line in stuck + decoded:
            print(f"   {line[:200]}")
        failures = [line for line in lines if "Traceback" in line or "Error" in line][:6]
        for line in failures:
            print(f"   ! {line[:200]}")
        steps = trace.read_text(encoding="utf-8", errors="replace").splitlines() if trace.is_file() else []
        print(f"AMD shutdown trace ({len(steps)} steps; time in ms, thread, step):")
        for line in steps:
            print(f"   {line}")
        if not steps:
            print("   (none: AMD's decoder was not used, or amf_frames.dll was not rebuilt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
