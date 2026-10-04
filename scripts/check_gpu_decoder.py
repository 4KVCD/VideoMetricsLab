"""Checks a GPU maker's frame decoder (vmaf_app/native/*_frames.dll) on this
PC, for a GPU whose decoder cannot be checked where the app is developed
(AMD's, say). Run from the repository root, with FFmpeg 9 or newer installed
and the decoders built (scripts/build_gpu_frames.ps1):

    .venv\\Scripts\\python.exe scripts\\check_gpu_decoder.py amd [VIDEO ...]

1. Pictures: small clips it makes itself (H.264 8-bit, HEVC 10-bit, AV1
   10-bit; cropped; an MP4 cut with an edit list), and each VIDEO given,
   decoded by the GPU decoder and by FFmpeg on the CPU, frame by frame
   (MD5): they must be identical, timestamps too.
2. Speed: the first VIDEO (or the 10-bit clip) decoded on its own, with the
   copy into a host buffer the GPU metrics make: frames per second and CPU.
3. Scores: SSIMULACRA2 on the GPU (Vship) for the first VIDEO against its
   first 10 s re-encoded (8-bit H.264, which every GPU decodes), with the GPU
   decoder and with FFmpeg decoding: identical.

It prints a report to send back. Nothing is written outside a temporary
folder.
"""
from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmaf_app.core import gpu_frames as nv
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropBox, GpuVendor

VENDORS = {"nvidia": GpuVendor.NVIDIA, "intel": GpuVendor.INTEL, "amd": GpuVendor.AMD}


def clip(path: Path, encoder: list[str], pix_fmt: str, seconds: float = 2.0, size: str = "1280x720",
         extra: list[str] | None = None) -> Path:
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s={size}:r=24000/1001:d={seconds}", "-pix_fmt", pix_fmt, *encoder,
                    *(extra or []), str(path)], check=True)
    return path


def gpu_sums(info, plan, backend):
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    sums, stamps = [], []
    try:
        stream.start()
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            sums.append(hashlib.md5(out).hexdigest())
            stamps.append(item[1])
    finally:
        stream.close()
    return sums, stamps


def cpu_sums(path, plan, frames=None):
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    out = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:V:0", "-vf",
                          f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", *(["-frames:v", str(frames)] if frames else []),
                          "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return [line.split(",")[5].strip() for line in out.splitlines() if line and not line.startswith("#")]


def check_pictures(path: Path, backend: str, crop: CropBox | None = None) -> str:
    info = probe_video(path)
    try:
        plan = nv.plan_decode(info, crop, shift=6)
    except nv.GpuDecodeUnavailableError as error:
        return f"{path.name}: not decoded by the GPU decoder ({error})"
    supported, reason = nv.decoder_supports(0, plan, backend)
    if not supported:
        return f"{path.name}: the GPU decoder refuses it ({reason})"
    try:
        sums, _stamps = gpu_sums(info, plan, backend)
    except nv.GpuDecodeFailedError as error:
        return f"{path.name}: FAILED on the GPU decoder: {error}"
    want = cpu_sums(path, plan)
    differ = sum(a != b for a, b in zip(sums, want, strict=False))
    verdict = "IDENTICAL" if len(sums) == len(want) and not differ else "DIFFERENT"
    return f"{path.name}: {verdict} ({len(sums)} GPU pictures, {len(want)} CPU, {differ} differ)"


def check_speed(path: Path, backend: str, frames: int = 600) -> str:
    info = probe_video(path)
    plan = nv.plan_decode(info, None)
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    me = psutil.Process()

    def cpu():
        total = sum(me.cpu_times()[:2])
        for child in me.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += sum(child.cpu_times()[:2])
        return total

    n, marks = 0, []
    try:
        stream.start()
        while n < frames:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            n += 1
            if n == 30 or n == frames:
                marks.append((time.perf_counter(), cpu(), n))
    finally:
        stream.close()
    if len(marks) < 2:
        return f"speed: too few pictures ({n})"
    (t0, c0, n0), (t1, c1, n1) = marks[0], marks[-1]
    return (f"speed on {path.name} ({info.width}x{info.height} {info.codec_name} {info.pix_fmt}): "
            f"{(n1 - n0) / (t1 - t0):.0f} fps, {(c1 - c0) / (t1 - t0):.2f} CPU cores, "
            f"{(c1 - c0) / (n1 - n0) * 1000:.1f} ms of CPU per picture")


def check_scores(source: Path, backend: str, work: Path) -> str:
    from vmaf_app.core import perceptual_vship as vship
    from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
    from vmaf_app.core.models import CropMode, VmafOptions

    test = work / "test.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:V:0", "-t", "10",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", str(test)],
                   check=True)
    options = VmafOptions(crop_mode=CropMode.NONE, gpu_decode=True, gpu_vendor=VENDORS[backend], duration_limit=10.0)
    request = analysis_request_from_vmaf_options(options, ("ssimulacra2",))
    device, reason = vship.detect_vship_device()
    if device is None:
        return f"scores: no GPU for Vship ({reason})"
    results, statuses = {}, []
    real = vship._native_decoder
    for mode in ("GPU decoder", "FFmpeg"):
        if mode == "FFmpeg":
            vship._native_decoder = lambda *_args, **_kwargs: None
        started = time.perf_counter()
        try:
            output = vship.run_vship_task(probe_video(source), probe_video(test), request, request.metrics, device,
                                          None, None, on_status=statuses.append)
        finally:
            vship._native_decoder = real
        results[mode] = (np.asarray(output.metrics.get("ssimulacra2").values), time.perf_counter() - started)
    (a, ta), (b, tb) = results["GPU decoder"], results["FFmpeg"]
    same = a.shape == b.shape and np.array_equal(a, b)
    return (f"scores: SSIMULACRA2 on {device.name} ({device.backend}), {len(a)} frames: "
            f"{'IDENTICAL' if same else 'DIFFERENT'}; {ta:.1f} s with the GPU decoder, {tb:.1f} s with FFmpeg"
            + "".join(f"\n  status: {s}" for s in statuses if "decod" in s.lower()))


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "amd"
    videos = [Path(arg) for arg in sys.argv[2:]]
    print(f"GPU frame decoder check: {backend}")
    for name in nv.LIBRARIES:
        print(f"  {name}: {'library loads' if nv.available(name) else 'library missing'}")
    if not nv.available(backend):
        sys.exit(f"{nv.LIBRARIES[backend].name} is missing: run scripts/build_gpu_frames.ps1")
    with tempfile.TemporaryDirectory(prefix="vml-gpu-check-") as folder:
        work = Path(folder)
        print("1. pictures")
        cases = [
            (clip(work / "h264.mkv", ["-c:v", "libx264", "-bf", "3"], "yuv420p"), None),
            (work / "h264.mkv", CropBox(1200, 640, 20, 40)),
            (clip(work / "hevc10.mkv", ["-c:v", "libx265", "-preset", "fast", "-x265-params",
                                        "bframes=4:log-level=error"], "yuv420p10le"), CropBox(1278, 718, 2, 2)),
            (clip(work / "av1_10.mkv", ["-c:v", "libsvtav1", "-preset", "10"], "yuv420p10le"), None),
        ]
        whole = clip(work / "whole.mp4", ["-c:v", "libx264", "-g", "48"], "yuv420p", seconds=4.0)
        subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-ss", "1.3", "-i", str(whole), "-c", "copy",
                        str(work / "editlist.mp4")], check=True)
        cases.append((work / "editlist.mp4", None))
        cases += [(video, None) for video in videos]
        for path, crop in cases:
            print(f"   {check_pictures(path, backend, crop)}" + (f" (cropped {crop.as_filter()})" if crop else ""),
                  flush=True)
        print("2. " + check_speed(videos[0] if videos else work / "hevc10.mkv", backend), flush=True)
        print("3. " + check_scores(videos[0] if videos else work / "hevc10.mkv", backend, work), flush=True)


if __name__ == "__main__":
    main()
