"""Serial Windows GPU VMAF comparison, no settings/cache writes.

Run with the project's Python. Requires psutil (development dependency).
Use compressed real videos, a warm pass, and equal frame counts. Reports
CPU seconds/frame as well as cores: faster throughput can use more cores
while doing less CPU work. The centered ROI is a transfer diagnostic, not
a statement about the quality of the full movie. Output scores are genuine.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmaf_app.core.vmaf_cuda import GpuAttempt
from vmaf_app.core.vmaf_runner import _gpu_pairs_stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--test-decode", choices=("cuda", "cpu"), required=True)
    p.add_argument("--test-height", type=int, required=True)
    p.add_argument("--start", type=float, default=1680)
    p.add_argument("--frames", type=int, default=720)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--ffmpeg", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root / "vmaf_app/tools/vmaf_native"
    get_times = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
    get_times.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    results, expected = [], None
    with tempfile.TemporaryDirectory(prefix="vmaf_native_bench_") as temp:
        temp = Path(temp)
        # Warm all three paths, then rotate their order to avoid always
        # giving the last one the warmest file-system/GPU caches.
        modes = ("legacy", "native_auto", "native_blocking")
        for repetition in range(a.repeat + 1):
            order = modes[repetition % 3:] + modes[:repetition % 3]
            for mode in order:
                attempt = None
                log = temp / "scores.json"
                if mode == "legacy":
                    attempt = GpuAttempt(3840, 1608, 10, {"vmaf": "vmaf_4k_v0.6.1"}, 1)
                    cmd = [str(a.ffmpeg), "-hide_banner", "-v", "error", "-nostdin", "-y"]
                    for path, hw in ((a.test, a.test_decode == "cuda"), (a.reference, True)):
                        cmd += ["-ss", str(a.start)]
                        if hw:
                            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                        cmd += ["-i", str(path)]
                    filters = []
                    for i, hw, y, name in ((0, a.test_decode == "cuda", (a.test_height-1608)//2, "test"),
                                           (1, True, 276, "ref")):
                        ops = ["hwdownload", "format=p010le"] if hw else []
                        ops += ["format=yuv420p10le", f"crop=3840:1608:0:{y}", "setpts=PTS-STARTPTS"]
                        filters.append(f"[{i}:v]" + ",".join(ops) + f"[{name}]")
                    filters.append(_gpu_pairs_stage("yuv420p10le", 3840, 1608, "test", "ref"))
                    outputs = []
                    # Output options must precede each output URL.
                    for token in attempt.output_args(0):
                        if token.startswith("\\\\.\\pipe\\"):
                            outputs += ["-frames:v", str(a.frames)]
                        outputs.append(token)
                    cmd += ["-filter_complex", ";".join(filters), *outputs]
                else:
                    cmd = [str(runtime / "vmaf_native.exe"), "--reference", str(a.reference), "--test", str(a.test),
                           "--reference-decode", "cuda", "--test-decode", a.test_decode, "--reference-crop",
                           "3840:1608:0:276", "--test-crop", f"3840:1608:0:{(a.test_height-1608)//2}",
                           "--depth", "10", "--frames", str(a.frames), "--start", str(a.start),
                           "--model", "vmaf=vmaf_4k_v0.6.1", "--output", str(log),
                           "--ptx", str(runtime / "vmaf_prepare.ptx"), "--wait",
                           "blocking" if mode == "native_blocking" else "auto"]
                print(json.dumps({"event": "start", "mode": mode, "repetition": repetition}), flush=True)
                start, own = time.perf_counter(), time.process_time()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stop = threading.Event()
                peak = [0]

                def sample(proc=proc, stop=stop, mode=mode, peak=peak):
                    target = psutil.Process(proc.pid)
                    while not stop.wait(0.05):
                        try:
                            # Include the Python scoring process for legacy.
                            rss = target.memory_info().rss
                            if mode == "legacy":
                                rss += psutil.Process().memory_info().rss
                            peak[0] = max(peak[0], rss)
                        except psutil.Error:
                            return

                monitor = threading.Thread(target=sample, daemon=True)
                monitor.start()
                try:
                    _, err = proc.communicate(timeout=600)
                    ct, et, kt, ut = (ctypes.c_uint64() for _ in range(4))
                    if not get_times(int(proc._handle), ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if proc.returncode:
                        raise RuntimeError(err.decode(errors="replace")[-4000:])
                    if attempt:
                        _, scores = attempt.finish(True)
                        attempt = None
                        scores = scores["vmaf"]
                    else:
                        scores = np.array([f["metrics"]["vmaf"] for f in json.loads(log.read_text())["frames"]])
                    wall = time.perf_counter() - start
                    own = time.process_time() - own
                    cpu = (kt.value + ut.value) / 1e7 + own
                    if len(scores) != a.frames:
                        raise RuntimeError(f"Expected {a.frames} scores, got {len(scores)}")
                    if expected is None:
                        expected = scores.copy()
                    difference = float(np.max(np.abs(scores - expected)))
                    if difference > 0.00005:
                        print(json.dumps({"expected_first": expected[:12].tolist(), "actual_first": scores[:12].tolist(),
                                          "stderr": err.decode(errors="replace")[-4000:]}), flush=True)
                        raise RuntimeError(f"Score parity failed: maximum difference {difference}")
                    item = dict(mode=mode, repetition=repetition, frames=len(scores), wall_s=wall, cpu_s=cpu,
                                cpu_ms_per_pair=cpu / len(scores) * 1000, cores=cpu / wall,
                                fps=len(scores) / wall, peak_rss_mb=peak[0] / 1e6,
                                mean=float(scores.mean()), maximum_score_difference=difference)
                    results.append(item)
                    print(json.dumps(item), flush=True)
                    a.output.write_text(json.dumps({"start_s": a.start, "roi": "3840x1608", "results": results}, indent=2))
                finally:
                    stop.set(); monitor.join(timeout=2)
                    if proc.poll() is None:
                        proc.kill(); proc.communicate()
                    if attempt:
                        attempt.finish(False)


if __name__ == "__main__":
    main()
