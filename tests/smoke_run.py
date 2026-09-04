r"""Manual end-to-end smoke test: not part of the pytest suite (it needs a
real ffmpeg and the fixture videos).

Run it from the repository root:

    .venv\Scripts\python.exe tests\smoke_run.py
    .venv\Scripts\python.exe tests\smoke_run.py --10bit

Running a script puts *its own* directory first on sys.path, not the
current one, so `import vmaf_app` would fail from here without the
sys.path line below -- which is why the documented command used to end in
ModuleNotFoundError.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.frame_extract import extract_frame_png
from vmaf_app.core.models import CropMode, GpuVendor, VmafOptions
from vmaf_app.core.stats import stats_for_run
from vmaf_app.core.vmaf_runner import VmafRunError, analysis_pix_fmt, run_vmaf

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--10bit", dest="ten_bit", action="store_true",
        help="compare the 10-bit fixture, to check the analysis format is not "
             "silently truncated to 8-bit",
    )
    parser.add_argument("--no-gpu", action="store_true", help="force software decode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_name = "source_10bit.mp4" if args.ten_bit else "source.mp4"
    distorted_name = source_name if args.ten_bit else "distorted.mp4"

    source_info = probe_video(FIXTURES / source_name)
    # The ordinary distorted fixture is cropped 2.35:1 content, while the
    # dedicated 10-bit fixture is full-frame 16:9. Pairing those two is a
    # geometry error, not a useful bit-depth smoke test, so the 10-bit path
    # round-trips its own fixture instead.
    distorted_info = probe_video(FIXTURES / distorted_name)
    print("source:", source_info)
    print("distorted:", distorted_info)
    print("analysis format:", analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt))

    options = VmafOptions(
        model="version=vmaf_v0.6.1",
        gpu_decode=not args.no_gpu,
        gpu_vendor=GpuVendor.AUTO,
        crop_mode=CropMode.AUTO,
    )

    # Three arguments, matching ProgressCallback: the runner reports fps
    # alongside the frame counts so callers can show an ETA.
    def on_progress(current, total, fps):
        print(f"\rprogress: {current}/{total}  ({fps:.1f} fps)", end="", flush=True)

    def on_status(message):
        print(f"\n[status] {message}")

    try:
        result = run_vmaf(
            source_info, distorted_info, options,
            on_progress=on_progress, on_status=on_status,
        )
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

    # Exercise the same real decode path used by the Frame Compare tab. PNG's
    # IHDR stores width/height at bytes 16..24 in network byte order.
    preview_frame = len(result.frames) // 2
    source_png = extract_frame_png(result, "source", preview_frame)
    distorted_png = extract_frame_png(result, "distorted", preview_frame)
    source_size = struct.unpack(">II", source_png[16:24])
    distorted_size = struct.unpack(">II", distorted_png[16:24])
    assert source_size == distorted_size
    print(
        f"frame preview: frame {preview_frame}, {source_size[0]}x{source_size[1]}, "
        f"source {len(source_png):,} bytes, distorted {len(distorted_png):,} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
