"""Manual end-to-end smoke test: not part of the pytest suite (needs real
ffmpeg + fixture videos). Run with: .venv\\Scripts\\python.exe tests\\smoke_run.py
"""
from pathlib import Path

from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropMode, GpuVendor, VmafOptions
from vmaf_app.core.stats import stats_for_run
from vmaf_app.core.vmaf_runner import run_vmaf

FIXTURES = Path(__file__).parent / "fixtures"


def main():
    source_info = probe_video(FIXTURES / "source.mp4")
    distorted_info = probe_video(FIXTURES / "distorted.mp4")
    print("source:", source_info)
    print("distorted:", distorted_info)

    options = VmafOptions(
        model="version=vmaf_v0.6.1",
        gpu_decode_source=True,
        gpu_vendor=GpuVendor.AUTO,
        crop_mode=CropMode.AUTO,
    )

    def on_progress(cur, total):
        print(f"\rprogress: {cur}/{total}", end="", flush=True)

    def on_status(msg):
        print(f"\n[status] {msg}")

    from vmaf_app.core.vmaf_runner import VmafRunError
    try:
        result = run_vmaf(source_info, distorted_info, options, on_progress=on_progress, on_status=on_status)
    except VmafRunError as e:
        print("STDERR TAIL:\n", e.stderr_tail)
        raise
    print()
    print("source_crop:", result.source_crop)
    print("distorted_crop:", result.distorted_crop)
    print("frame count:", len(result.frames))
    print("first 5 frames:", result.frames[:5])

    stats = stats_for_run(result)
    print("mean:", stats.mean, "min:", stats.minimum, "max:", stats.maximum)
    for t in stats.thresholds:
        print(f"  {t.label}: {t.percentage:.1f}% ({t.count} frames)")


if __name__ == "__main__":
    main()
