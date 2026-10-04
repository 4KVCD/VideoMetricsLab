"""Scores a whole video pair with libvmaf's CUDA code and with the Vulkan
port from one decode, and compares every frame's features and scores.

    python scripts/verify_vmaf_vulkan_movie.py REFERENCE DISTORTED OUT.npz
        [--device 0] [--seconds 0]

Both videos are decoded once, by NVIDIA's decoder in this process
(gpu_frames), cropped and scaled as the app's run does them (black bars
detected as the app detects them); every frame pair goes to both scorers,
from the same decoded pictures: CUDA copies the luma on the GPU, Vulkan
downloads it. The per-frame features (VIF scales 0-3, ADM2 and motion2, for
VMAF and for VMAF NEG) are compared as bit patterns, and the scores as
libvmaf's log gives them; all of it is saved to OUT.npz.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmaf_app.core import gpu_frames, vmaf_cuda, vmaf_vulkan
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.frame_sync import frame_pairs
from vmaf_app.core.gpu import plan_hwaccel
from vmaf_app.core.models import VmafOptions
from vmaf_app.core.vmaf_runner import (
    _auto_model_or,
    _resolve_crops,
    analysis_bit_depth,
    analysis_dimensions,
    estimate_total_frames,
)

FEATURES = {**vmaf_vulkan._MODEL_FEATURES[False], **vmaf_vulkan._MODEL_FEATURES[True]}
COLUMNS = sorted(set(FEATURES.values()))
NAMES = {column: name for name, column in FEATURES.items()}


def cuda_features(scorer: vmaf_cuda.GpuScorer, count: int) -> np.ndarray:
    lib = vmaf_cuda._load()
    lib.vmaf_feature_score_at_index.restype = ctypes.c_int
    lib.vmaf_feature_score_at_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
    rows = np.full((count, vmaf_vulkan.FEATURE_COUNT), np.nan)
    value = ctypes.c_double()
    for column in COLUMNS:
        name = NAMES[column].encode()
        for frame in range(count):
            if lib.vmaf_feature_score_at_index(scorer._context, name, ctypes.byref(value), frame) == 0:
                rows[frame, column] = value.value
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("distorted")
    parser.add_argument("out")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    arguments = parser.parse_args()

    source, distorted = probe_video(Path(arguments.reference)), probe_video(Path(arguments.distorted))
    options = VmafOptions(compute_vmaf=True, compute_vmaf_neg=True, duration_limit=arguments.seconds)
    hwaccel = plan_hwaccel(options.gpu_vendor, source.codec_name, distorted.codec_name,
                           source_pix_fmt=source.pix_fmt, distorted_pix_fmt=distorted.pix_fmt)
    source_crop, distorted_crop = _resolve_crops(source, distorted, options, None, hwaccel=hwaccel)
    width, height = analysis_dimensions(source, distorted, options, source_crop, distorted_crop)
    bit_depth = analysis_bit_depth(source, distorted)
    models = vmaf_cuda.gpu_models(True, True, _auto_model_or(options, (width, height)))
    total = estimate_total_frames(distorted, options, source)
    print(f"{distorted.path.name} against {source.path.name}: {width}x{height} {bit_depth}-bit, crops "
          f"{source_crop} / {distorted_crop}, models {models}, about {total} frames", flush=True)

    plans = []
    for info, crop in ((distorted, distorted_crop), (source, source_crop)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height))
        supported, refusal = gpu_frames.decoder_supports(0, plan)
        if not supported:
            raise SystemExit(f"NVIDIA's decoder does not take {info.path.name}: {refusal}")
        plans.append(plan)
    test = gpu_frames.GpuFrameStream(distorted, plans[0], 0, pool=6)
    ref = gpu_frames.GpuFrameStream(source, plans[1], 0, pool=6)
    cuda = vmaf_cuda.GpuScorer(width, height, bit_depth, models, 1, on_device=True)
    vulkan = vmaf_vulkan.VulkanScorer(width, height, bit_depth, models, 1, device=arguments.device)
    try:
        test.start()
        ref.start()
        test_base, ref_base = test.wait_time_base(), ref.wait_time_base()
        stop = gpu_frames.duration_in(f"{arguments.seconds + 1 / distorted.fps:.3f}", test_base) \
            if arguments.seconds > 0 else None

        def puller(stream):
            def pull():
                while True:
                    try:
                        return stream.next(100)
                    except TimeoutError:
                        continue
            return pull

        pairs = frame_pairs(puller(test), puller(ref), test_base, ref_base, test.release, ref.release)
        count = 0
        started = time.perf_counter()
        try:
            for test_slot, ref_slot, when in pairs:
                if stop is not None and when >= stop:
                    break
                if ref_slot is None:
                    raise SystemExit("the source has no frame for the test video's first")
                cuda.add_on_device(lambda address, pitch, slot=ref_slot: ref.copy_luma(slot, address, pitch),
                                   lambda address, pitch, slot=test_slot: test.copy_luma(slot, address, pitch))
                vulkan.add_decoded(lambda address, slot=ref_slot: ref.download(slot, address),
                                   lambda address, slot=test_slot: test.download(slot, address))
                count += 1
                if count % 10000 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"  {count} frames, {elapsed / 60:.1f} min, {count / elapsed:.0f} fps", flush=True)
        finally:
            pairs.close()
        test.verify()
        ref.verify()
        elapsed = time.perf_counter() - started
        print(f"decoded and scored {count} frame pairs in {elapsed / 60:.1f} min", flush=True)

        c_frames, c_scores = cuda.finish()
        c_rows = cuda_features(cuda, count)
        v_frames, v_rows = vulkan.features()
        _, v_scores = vulkan.finish()
    finally:
        test.close()
        ref.close()
        cuda.close()
        vulkan.close()

    np.savez_compressed(arguments.out, frames=c_frames, cuda_features=c_rows, vulkan_features=v_rows,
                        **{f"cuda_{k}": v for k, v in c_scores.items()}, **{f"vulkan_{k}": v for k, v in v_scores.items()})
    identical = np.array_equal(c_frames, v_frames)
    print(f"frames: CUDA {len(c_frames)}, Vulkan {len(v_frames)}")
    for column in COLUMNS:
        c, v = c_rows[:, column], v_rows[:, column]
        same = c.view(np.uint64) == v.view(np.uint64)
        identical &= bool(same.all())
        line = f"  {NAMES[column]:40} {int(same.sum())}/{len(same)} frames bit-identical"
        if not same.all():
            worst = int(np.nanargmax(np.abs(c - v)))
            line += f"; largest difference {abs(c[worst] - v[worst]):.3e} at frame {worst}"
        print(line)
    for name in models:
        c, v = c_scores[name], v_scores[name]
        same = c == v
        identical &= bool(same.all())
        line = f"  score {name:34} {int(same.sum())}/{len(same)} frames identical; mean CUDA {c.mean():.6f}, " \
               f"Vulkan {v.mean():.6f}"
        if not same.all():
            line += f"; largest difference {np.max(np.abs(c - v)):.6f}"
        print(line)
    print("ALL IDENTICAL" if identical else "DIFFERENCES FOUND", flush=True)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
