"""Checks a GPU maker's frame decoder (vmaf_app/native/*_frames.dll) on this
PC, for a GPU whose decoder cannot be checked where the app is developed
(AMD's, say). Run from the repository root, with FFmpeg 9 or newer installed
and the decoders built (scripts/build_gpu_frames.ps1):

    .venv\\Scripts\\python.exe scripts\\check_gpu_decoder.py amd [VIDEO ...] [--seconds 60]

Of each VIDEO given, the first --seconds are checked (60 unless said; 0 for
all of it): a whole film decoded twice, frame by frame, takes hours.

1. Pictures: small clips it makes itself (H.264 8-bit, HEVC 10-bit, AV1
   10-bit; cropped; an MP4 cut with an edit list), and each VIDEO given,
   decoded by the GPU decoder and by FFmpeg on the CPU, frame by frame
   (MD5): they must be identical, and so must their timestamps, each from
   its first picture.
2. Speed: the first VIDEO (or the 10-bit clip) decoded on its own, with the
   copy into a host buffer the GPU metrics make: frames per second and CPU.
3. Scores: SSIMULACRA2 on the GPU (Vship) for the first VIDEO against its
   first 10 s re-encoded (8-bit H.264, which every GPU decodes), with the GPU
   decoder and with FFmpeg decoding (its hardware decode, as the app runs
   it): identical. The FFmpeg pass's status lines say where FFmpeg decoded:
   "GPU decode failed ... decoding it in software" means its hardware decode
   did not work on this PC.
4. Scaling (Intel, AMD): a comparison at two sizes is scaled by the decoder,
   on the GPU (native/d3d11_scale.h's shader). The shader's picture must be
   the CPU scaler's, sample for sample, on this GPU; then the first VIDEO
   (or the 10-bit clip) is decoded and scaled down and up, on the GPU and
   on the CPU: identical pictures, with each one's speed and CPU. "scaled
   on the CPU" with a reason means the decoder could not scale on the GPU
   here: the pictures are still right, only slower.

It prints a report to send back. Nothing is written outside a temporary
folder.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
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
#: PCI vendor ids, for the scaling shader's own check.
PCI_VENDORS = {"intel": 0x8086, "amd": 0x1002}


def clip(path: Path, encoder: list[str], pix_fmt: str, seconds: float = 2.0, size: str = "1280x720",
         extra: list[str] | None = None) -> Path:
    # Captured: SVT-AV1 writes its settings whatever FFmpeg's log level is,
    # thirty lines into the report this prints.
    made = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                           "-i", f"testsrc2=s={size}:r=24000/1001:d={seconds}", "-pix_fmt", pix_fmt, *encoder,
                           *(extra or []), str(path)], capture_output=True, text=True)
    if made.returncode != 0:
        sys.exit(f"FFmpeg could not make {path.name}:\n{made.stderr[-2000:]}")
    return path


def gpu_sums(info, plan, backend, frames=None):
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    sums, stamps = [], []
    try:
        stream.start()
        while frames is None or len(sums) < frames:
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
    """Each picture of FFmpeg's CPU decode: its MD5 and its timestamp, in the
    stream's time base (-enc_time_base filter; the encoder's own is 1/rate)."""
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    out = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:V:0", "-vf",
                          f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", "-enc_time_base", "filter",
                          *(["-frames:v", str(frames)] if frames else []),
                          "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    rows = [line.split(",") for line in out.splitlines() if line and not line.startswith("#")]
    return [row[5].strip() for row in rows], [int(row[2]) for row in rows]


def check_pictures(path: Path, backend: str, crop: CropBox | None = None, seconds: float = 0.0) -> str:
    info = probe_video(path)
    # The first `seconds` of it, as a count of pictures for both decodes.
    frames = max(1, round(seconds * info.fps)) if seconds > 0 and info.fps > 0 else None
    try:
        plan = nv.plan_decode(info, crop, shift=6)
    except nv.GpuDecodeUnavailableError as error:
        return f"{path.name}: not decoded by the GPU decoder ({error})"
    supported, reason = nv.decoder_supports(0, plan, backend)
    if not supported:
        return f"{path.name}: the GPU decoder refuses it ({reason})"
    try:
        sums, stamps = gpu_sums(info, plan, backend, frames)
    except nv.GpuDecodeFailedError as error:
        return f"{path.name}: FAILED on the GPU decoder: {error}"
    want, want_stamps = cpu_sums(path, plan, frames)
    differ = sum(a != b for a, b in zip(sums, want, strict=False))
    # From each one's first picture: FFmpeg's start at 0, the decoder's are
    # the file's own.
    late = sum(a - stamps[0] != b - want_stamps[0] for a, b in zip(stamps, want_stamps, strict=False))
    verdict = "IDENTICAL" if len(sums) == len(want) and not differ and not late else "DIFFERENT"
    return (f"{path.name}: {verdict} ({len(sums)} GPU pictures, {len(want)} CPU, {differ} differ, "
            f"{late} timestamps differ)")


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
        # The 2-second clip made here when no VIDEO is given: too short to
        # time. It read as a failure of the decoder.
        return f"speed: not measured ({path.name} has {n} pictures; give a longer VIDEO to time the decoder)"
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
    # The limit is the test clip's own length: at 10 s, with the 2-second clip
    # made here as the source, the comparison was refused ("The duration
    # limit extends beyond the end of one of the videos").
    limit = min(10.0, probe_video(test).duration)
    options = VmafOptions(crop_mode=CropMode.NONE, gpu_decode=True, gpu_vendor=VENDORS[backend],
                          duration_limit=limit)
    request = analysis_request_from_vmaf_options(options, ("ssimulacra2",))
    # Probed in this process, so that the passes are scored in it too. The
    # app's device (detect_vship_device) is scored in a process of its own,
    # which loads perceptual_vship afresh: the "FFmpeg" pass below, made by
    # replacing _native_decoder here, was decoded by the GPU decoder there,
    # and the step compared the decoder with itself.
    device, reason = vship._probe_vship_device()
    if device is None:
        return f"scores: no GPU for Vship ({reason})"
    native_lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "in the scoring process" in record.getMessage():
                native_lines.append(record.getMessage())

    capture = Capture(level=logging.INFO)
    vship_log = logging.getLogger(vship.__name__)
    level = vship_log.level
    vship_log.addHandler(capture)
    vship_log.setLevel(logging.INFO)
    results, statuses, used = {}, {}, {}
    real = vship._native_decoder
    try:
        for mode in ("GPU decoder", "FFmpeg"):
            if mode == "FFmpeg":
                vship._native_decoder = lambda *_args, **_kwargs: None
            native_lines.clear()
            statuses[mode] = []
            started = time.perf_counter()
            try:
                output = vship.run_vship_task(probe_video(source), probe_video(test), request, request.metrics,
                                              device, None, None, on_status=statuses[mode].append)
            finally:
                vship._native_decoder = real
            results[mode] = (np.asarray(output.metrics.get("ssimulacra2").values), time.perf_counter() - started)
            used[mode] = len(native_lines)
    finally:
        vship_log.removeHandler(capture)
        vship_log.setLevel(level)
    (a, ta), (b, tb) = results["GPU decoder"], results["FFmpeg"]
    same = a.shape == b.shape and np.array_equal(a, b)
    lines = [f"scores: SSIMULACRA2 on {device.name} ({device.backend}), {len(a)} frames: "
             f"{'IDENTICAL' if same else 'DIFFERENT'}; {ta:.1f} s with the GPU decoder, {tb:.1f} s with FFmpeg"]
    if used["GPU decoder"] != 2:
        lines.append(f"  NOT A COMPARISON: the GPU decoder decoded {used['GPU decoder']} of the 2 videos in its pass")
    if used["FFmpeg"]:
        lines.append(f"  NOT A COMPARISON: the GPU decoder decoded {used['FFmpeg']} of the 2 videos in FFmpeg's pass")
    for mode in ("GPU decoder", "FFmpeg"):
        lines += [f"  {mode} pass: {s}" for s in statuses[mode] if "decod" in s.lower()]
    return "\n".join(lines)


def test_picture(width: int, height: int, depth: int, seed: int = 7) -> bytes:
    """An NV12/P010 picture of noise with hard edges, where a filter rings
    and a wrong weight, order or rounding shows."""
    rng = np.random.default_rng(seed)
    top = (1 << depth) - 1
    luma = rng.integers(0, top + 1, (height, width))
    luma[: height // 3, : width // 2] = top
    luma[height // 3: height // 2, width // 3:] = 0
    chroma = rng.integers(0, top + 1, (height // 2, width))
    chroma[: height // 5] = top
    planes = np.concatenate([luma, chroma])
    return (planes << 6).astype("<u2").tobytes() if depth > 8 else planes.astype(np.uint8).tobytes()


def check_shader(backend: str) -> str:
    """The scaling shader on this maker's GPU against the CPU scaler."""
    same = total = 0
    for depth in (8, 10):
        for (w, h), size in (((1920, 1080), (1280, 720)), ((1280, 720), (1920, 1080)), ((1920, 800), (854, 356))):
            for algorithm in ("bicubic", "bilinear", "lanczos", "spline"):
                plan = nv.DecodePlan("hevc" if depth > 8 else "h264", depth, w, h, 0, 0, w, h,
                                     6 if depth > 8 else 0, False, size[0], size[1], algorithm)
                picture = test_picture(w, h, depth)
                try:
                    on_gpu = nv.scale_picture(plan, picture, PCI_VENDORS[backend], backend)
                except nv.GpuDecodeFailedError as error:
                    return f"shader: FAILED on this GPU: {error}"
                if on_gpu is None:
                    return "shader: no Direct3D 11 GPU of this maker here"
                total += 1
                same += on_gpu == nv.scale_picture(plan, picture, None, backend)
    return f"shader: {'IDENTICAL' if same == total else 'DIFFERENT'} to the CPU scaler in {same} of {total} cases"


#: Pictures compared by MD5 at the start of a scaled decode; the rest are
#: only timed (hashing a 4K picture costs more CPU than handing it over).
_COMPARED = 48


def scaled(info, plan, backend, frames):
    """(the first pictures' MD5s, fps and ms of CPU a picture after them,
    scaled on the GPU, note)."""
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    me = psutil.Process()
    sums, count = [], 0
    try:
        stream.start()
        started, cpu = time.perf_counter(), sum(me.cpu_times()[:2])
        while count < frames:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            count += 1
            if count <= _COMPARED:
                sums.append(hashlib.md5(out).hexdigest())
                if count == _COMPARED:
                    started, cpu = time.perf_counter(), sum(me.cpu_times()[:2])
        elapsed, used = time.perf_counter() - started, sum(me.cpu_times()[:2]) - cpu
        on_gpu, note = bool(stream.stats().scaled_on_gpu), stream.scale_note()
    finally:
        stream.close()
    timed = count - _COMPARED
    if timed <= 0:  # a short clip: nothing left to time
        return sums, 0.0, 0.0, on_gpu, note
    return sums, timed / max(elapsed, 1e-6), used / timed * 1000, on_gpu, note


def check_scaling(path: Path, backend: str, frames: int = 240) -> list[str]:
    from dataclasses import replace

    info = probe_video(path)
    lines = []
    half = (max(2, info.width // 4 * 2), max(2, info.height // 4 * 2))
    # Up to twice the size, or for a video already 4K wide, down to a third.
    other = (("up", (info.width * 2, info.height * 2)) if info.width <= 1920
             else ("down", (max(2, info.width // 6 * 2), max(2, info.height // 6 * 2))))
    for name, size in (("down", half), other):
        try:
            plan = nv.plan_decode(info, None, shift=6, size=size)
            gpu = scaled(info, plan, backend, frames)
            cpu = scaled(info, replace(plan, cpu_scaling=True), backend, frames)
        except (nv.GpuDecodeUnavailableError, nv.GpuDecodeFailedError) as error:
            lines.append(f"{name} to {size[0]}x{size[1]}: FAILED: {error}")
            continue
        verdict = "IDENTICAL" if gpu[0] == cpu[0] else "DIFFERENT"
        where = "scaled on the GPU" if gpu[3] else f"scaled on the CPU ({gpu[4] or 'no reason given'})"
        speed = (f"{gpu[1]:.0f} fps, {gpu[2]:.1f} ms of CPU a picture; with CPU scaling: {cpu[1]:.0f} fps, "
                 f"{cpu[2]:.1f} ms" if gpu[1] and cpu[1] else "too short to time: give a longer VIDEO")
        lines.append(f"{name} to {size[0]}x{size[1]}: {verdict} to CPU scaling ({len(gpu[0])} pictures); {where}: "
                     f"{speed}")
    return lines


def main() -> None:
    arguments = sys.argv[1:]
    seconds = 60.0
    if "--seconds" in arguments:
        at = arguments.index("--seconds")
        seconds = float(arguments[at + 1])
        del arguments[at:at + 2]
    backend = arguments[0] if arguments else "amd"
    videos = [Path(arg) for arg in arguments[1:]]
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
        made = len(cases)  # the clips made here are short: checked whole
        cases += [(video, None) for video in videos]
        for number, (path, crop) in enumerate(cases):
            limit = seconds if number >= made else 0.0
            print(f"   {check_pictures(path, backend, crop, limit)}"
                  + (f" (cropped {crop.as_filter()})" if crop else "")
                  + (f" (its first {limit:g} s)" if limit else ""), flush=True)
        print("2. " + check_speed(videos[0] if videos else work / "hevc10.mkv", backend), flush=True)
        print("3. " + check_scores(videos[0] if videos else work / "hevc10.mkv", backend, work), flush=True)
        if backend in PCI_VENDORS:
            print("4. scaling")
            print(f"   {check_shader(backend)}", flush=True)
            for line in check_scaling(videos[0] if videos else work / "hevc10.mkv", backend):
                print(f"   {line} [{(videos[0] if videos else work / 'hevc10.mkv').name}]", flush=True)


if __name__ == "__main__":
    main()
