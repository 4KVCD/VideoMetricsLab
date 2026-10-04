"""Compares VMAF's features and scores from Vulkan (vmaf_app.core.vmaf_vulkan)
with libvmaf's CUDA ones (vmaf_app.core.vmaf_cuda) on the same frames, frame
by frame and bit by bit, and times both.

    python scripts/compare_vmaf_vulkan.py REFERENCE DISTORTED [--frames 48]
        [--start 60] [--size 1920x1080] [--bits 8] [--device 0] [--native]
    python scripts/compare_vmaf_vulkan.py --synthetic 640x360 [--bits 10]
    python scripts/compare_vmaf_vulkan.py --matrix [--device 1]

--matrix runs the comparison over sizes (odd ones, powers of two, the
smallest, 8K), both bit depths and kinds of frames that take the shaders'
rarer paths: black, white, identical, full-range noise, extremes side by
side, a still video, a sharpened one. Exit code 0 when every feature and
score is identical.
"""
from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmaf_app.core import vmaf_cuda, vmaf_vulkan

MODELS = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
# libvmaf's feature names -> the Vulkan library's columns.
FEATURES = {**vmaf_vulkan._MODEL_FEATURES[False], **vmaf_vulkan._MODEL_FEATURES[True]}


def decode(path: str, start: float, frames: int, size: tuple[int, int] | None, bits: int) -> tuple[list, int, int]:
    """`frames` frames of a video as packed 4:2:0 frames."""
    probe = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    width, height = (int(part) for part in probe.stdout.strip().split(",")[:2])
    command = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-ss", str(start), "-i", path, "-map", "0:v:0",
               "-frames:v", str(frames)]
    if size:
        width, height = size
        command += ["-vf", f"scale={width}:{height}:flags=bicubic"]
    command += ["-pix_fmt", "yuv420p" if bits == 8 else f"yuv420p{bits}le", "-f", "rawvideo", "-"]
    sample = 1 if bits == 8 else 2
    frame_bytes = (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * sample
    data = subprocess.run(command, capture_output=True, check=True).stdout
    return [bytearray(data[i:i + frame_bytes]) for i in range(0, len(data) - frame_bytes + 1, frame_bytes)], width, height


def synthetic(width: int, height: int, bits: int, frames: int, seed: int) -> tuple[list, list]:
    """Moving textured frames and a noisy, blurred copy of them."""
    rng = np.random.default_rng(seed)
    peak = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    yy, xx = np.mgrid[0:height, 0:width]
    chroma = np.full(2 * ((width + 1) // 2) * ((height + 1) // 2), peak // 2, dtype=dtype).tobytes()
    reference, distorted = [], []
    texture = rng.integers(0, peak + 1, (height + 64, width + 64)).astype(np.float64)
    for index in range(frames):
        base = 0.5 + 0.25 * np.sin((xx + 3 * index) / 17.0) + 0.2 * np.cos((yy - 2 * index) / 11.0)
        ref = np.clip(base * peak * 0.7 + texture[index:index + height, 2 * index:2 * index + width] * 0.3, 0, peak)
        dis = ref.copy()
        dis[:, 1:] = (dis[:, 1:] + dis[:, :-1]) / 2
        dis = np.clip(dis * (1.05 if index % 3 == 0 else 0.97) + rng.normal(0, peak * 0.02, dis.shape), 0, peak)
        if index % 4 == 1:  # a sharpened frame: gains above 1, which VMAF NEG limits
            dis = np.clip(ref + 0.6 * (ref - dis), 0, peak)
        reference.append(bytearray(ref.astype(dtype).tobytes() + chroma))
        distorted.append(bytearray(dis.astype(dtype).tobytes() + chroma))
    return reference, distorted


def special(kind: str, width: int, height: int, bits: int, frames: int, seed: int = 7) -> tuple[list, list]:
    """Frames that take the paths ordinary video rarely does."""
    rng = np.random.default_rng(seed)
    peak = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    chroma = np.full(2 * ((width + 1) // 2) * ((height + 1) // 2), peak // 2, dtype=dtype).tobytes()
    yy, xx = np.mgrid[0:height, 0:width]

    def frame(picture):
        return bytearray(np.clip(picture, 0, peak).astype(dtype).tobytes() + chroma)

    reference, distorted = [], []
    for index in range(frames):
        noise = rng.integers(0, peak + 1, (height, width))
        if kind == "black":
            ref, dis = np.zeros((height, width)), np.zeros((height, width))
        elif kind == "white":
            ref, dis = np.full((height, width), peak), np.full((height, width), peak)
        elif kind == "black-white":
            ref, dis = np.zeros((height, width)), np.full((height, width), peak)
        elif kind == "identical":
            ref = dis = noise
        elif kind == "noise":  # two unrelated full-range noise frames
            ref, dis = noise, rng.integers(0, peak + 1, (height, width))
        elif kind == "extremes":  # 0 and the peak side by side, shifted by a pixel in the distorted frame
            ref = ((xx // 3 + yy // 2 + index) % 2) * peak
            dis = (((xx + 1) // 3 + yy // 2 + index) % 2) * peak
        elif kind == "still":  # no motion at all
            ref = (xx * 7 + yy * 13) % (peak + 1)
            dis = ref // 2 + peak // 4
        elif kind == "sharpened":  # gains far above 1 everywhere
            base = (np.sin(xx / 5.0) * np.cos(yy / 7.0 + index) * 0.2 + 0.5) * peak
            ref, dis = base, peak / 2 + (base - peak / 2) * 3.0
        elif kind == "inverted":  # negative correlation
            ref, dis = noise, peak - noise
        else:
            raise ValueError(kind)
        reference.append(frame(ref))
        distorted.append(frame(dis))
    return reference, distorted


class _LumaScorer(vmaf_cuda.GpuScorer):
    """GpuScorer for any size: it copies the chroma as rows of (width + 1) / 2
    samples, which libvmaf's pictures do not hold for an odd width. VMAF
    uses the luma alone, so only that is copied here."""

    def _fill(self, picture, frame) -> None:
        offset, rows, row_bytes = self._planes[0]
        source = np.frombuffer(frame, dtype=np.uint8)
        target = np.ctypeslib.as_array(
            ctypes.cast(picture.data[0], ctypes.POINTER(ctypes.c_uint8)), shape=(rows, picture.stride[0]))
        target[:, :row_bytes] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)


def cuda_run(width, height, bits, reference, distorted, step):
    scorer = _LumaScorer(width, height, bits, MODELS, step)
    try:
        started = time.perf_counter()
        for ref, dis in zip(reference, distorted, strict=True):
            # libvmaf's CUDA pictures take a copy: the caller's are reused.
            scorer.add(bytearray(ref), bytearray(dis))
        frames, scores = scorer.finish()
        elapsed = time.perf_counter() - started
        lib = vmaf_cuda._load()
        lib.vmaf_feature_score_at_index.restype = ctypes.c_int
        lib.vmaf_feature_score_at_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                    ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
        rows = np.zeros((len(frames), vmaf_vulkan.FEATURE_COUNT))
        value = ctypes.c_double()
        for name, column in FEATURES.items():
            for slot, frame in enumerate(frames):
                error = lib.vmaf_feature_score_at_index(scorer._context, name.encode(), ctypes.byref(value), int(frame))
                if error:
                    raise RuntimeError(f"libvmaf has no {name} for frame {frame} ({error})")
                rows[slot, column] = value.value
        return frames, rows, scores, elapsed
    finally:
        scorer.close()


def vulkan_run(width, height, bits, reference, distorted, step, device, native):
    scorer = vmaf_vulkan.VulkanScorer(width, height, bits, MODELS, step, device=device, native_double=native)
    try:
        started = time.perf_counter()
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        frames, rows = scorer.features()
        elapsed = time.perf_counter() - started
        scores = {name: vmaf_vulkan.predict(version, frames, rows) for name, version in MODELS.items()}
        return frames, rows, scores, elapsed
    finally:
        scorer.close()


def compare(label, width, height, bits, reference, distorted, step=1, device=None, native=False) -> bool:
    count = min(len(reference), len(distorted))
    reference, distorted = reference[:count], distorted[:count]
    c_frames, c_rows, c_scores, c_time = cuda_run(width, height, bits, reference, distorted, step)
    v_frames, v_rows, v_scores, v_time = vulkan_run(width, height, bits, reference, distorted, step, device, native)
    identical = np.array_equal(c_frames, v_frames)
    lines = []
    for name, column in sorted(FEATURES.items(), key=lambda item: item[1]):
        cuda, vulkan = c_rows[:, column], v_rows[:, column]
        # NaN (0 / 0 on a flat frame) compares as its bit pattern too.
        same = np.array_equal(cuda.view(np.uint64), vulkan.view(np.uint64))
        identical &= same
        if not same:
            worst = int(np.argmax(np.abs(cuda - vulkan)))
            lines.append(f"    {name}: {int(np.sum(cuda != vulkan))}/{len(cuda)} frames differ, most at frame "
                         f"{int(c_frames[worst])}: CUDA {cuda[worst]!r} Vulkan {vulkan[worst]!r}")
    for name in MODELS:
        same = np.array_equal(c_scores[name], v_scores[name], equal_nan=True)
        identical &= same
        if not same:
            lines.append(f"    {name}: max difference {np.max(np.abs(c_scores[name] - v_scores[name])):.6f}")
    print(f"{'IDENTICAL' if identical else 'DIFFERENT'}  {label}: {width}x{height} {bits}-bit, {count} frames"
          f"{f', every {step}' if step > 1 else ''}; VMAF {np.nanmean(v_scores['vmaf']):.6f} NEG "
          f"{np.nanmean(v_scores['vmaf_neg']):.6f}; CUDA {count / c_time:.1f} fps, Vulkan {count / v_time:.1f} fps")
    for line in lines:
        print(line)
    return bool(identical)


def matrix(device: int | None) -> bool:
    """The comparisons --matrix makes; True when all are identical."""
    same = True
    sizes = [(64, 64), (66, 34), (33, 35), (127, 63), (255, 257), (641, 361), (1365, 768), (1024, 512),
             (2048, 1024), (1920, 1080), (4096, 2160)]
    for width, height in sizes:
        for bits in (8, 10):
            frames = 3 if width * height > 2_000_000 else 5
            same &= compare("synthetic", width, height, bits, *synthetic(width, height, bits, frames, width), 1,
                            device)
    for kind in ("black", "white", "black-white", "identical", "noise", "extremes", "still", "sharpened",
                 "inverted"):
        for bits in (8, 10):
            same &= compare(kind, 322, 242, bits, *special(kind, 322, 242, bits, 4), 1, device)
    same &= compare("every 2nd frame", 640, 360, 8, *synthetic(640, 360, 8, 7, 3), 2, device)
    same &= compare("every 5th frame", 640, 360, 10, *synthetic(640, 360, 10, 11, 4), 5, device)
    same &= compare("one frame", 640, 360, 8, *synthetic(640, 360, 8, 1, 5), 1, device)
    same &= compare("two frames", 640, 360, 8, *synthetic(640, 360, 8, 2, 6), 1, device)
    same &= compare("8K", 7680, 4320, 10, *synthetic(7680, 4320, 10, 2, 9), 1, device)
    print("ALL IDENTICAL" if same else "SOME DIFFER")
    return same


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", nargs="?")
    parser.add_argument("distorted", nargs="?")
    parser.add_argument("--synthetic", help="WIDTHxHEIGHT of made-up frames instead of videos")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--size")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", type=int)
    parser.add_argument("--native", action="store_true", help="use the GPU's double")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--matrix", action="store_true")
    arguments = parser.parse_args()
    for device in vmaf_vulkan.devices():
        print(f"Vulkan GPU {device.index}: {device.name}{'' if device.usable else ' (unusable)'}")
    if arguments.matrix:
        return 0 if matrix(arguments.device) else 1
    if arguments.synthetic:
        width, height = (int(part) for part in arguments.synthetic.split("x"))
        reference, distorted = synthetic(width, height, arguments.bits, arguments.frames, arguments.seed)
        label = "synthetic"
    else:
        size = tuple(int(part) for part in arguments.size.split("x")) if arguments.size else None
        reference, width, height = decode(arguments.reference, arguments.start, arguments.frames, size, arguments.bits)
        distorted, _, _ = decode(arguments.distorted, arguments.start, arguments.frames, (width, height),
                                   arguments.bits)
        label = Path(arguments.distorted).name
    same = compare(label, width, height, arguments.bits, reference, distorted, arguments.step, arguments.device,
                   arguments.native)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
