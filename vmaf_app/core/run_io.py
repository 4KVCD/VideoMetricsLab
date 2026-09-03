"""Save/load a VmafRunResult to a portable JSON file, and CSV export, so
past runs can be reloaded later and overlaid in the comparison graph."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from vmaf_app.core.models import CropBox, FrameScores, ScaleDirection, VideoInfo, VmafRunResult

FORMAT_VERSION = 1


def safe_filename_stem(label: str) -> str:
    """A series label reduced to something usable as a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


def unique_output_path(
    directory: Path, label: str, suffix: str, reserved: set[Path] | None = None
) -> Path:
    """A path in `directory` for `label` that collides with nothing.

    Batch exports name their files after the series label, and two labels
    collide easily -- the same basename from two directories, or one file
    compared twice under different options, both reduce to "movie". Writing
    them in a loop meant the second silently replaced the first, and the
    user was told N files had been written when fewer existed.

    Checks both what is already on disk and what this batch has already
    claimed (via `reserved`, which is updated in place), because within a
    single loop the earlier file may not have been written yet.
    """
    reserved = reserved if reserved is not None else set()
    stem = safe_filename_stem(label) or "run"
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate in reserved or candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    reserved.add(candidate)
    return candidate


def _crop_to_dict(c: CropBox | None) -> dict | None:
    return None if c is None else {"w": c.w, "h": c.h, "x": c.x, "y": c.y}


def _crop_from_dict(d: dict | None) -> CropBox | None:
    return None if d is None else CropBox(**d)


def _info_to_dict(v: VideoInfo) -> dict:
    return {
        "path": str(v.path), "width": v.width, "height": v.height, "fps": v.fps,
        "duration": v.duration, "nb_frames": v.nb_frames, "codec_name": v.codec_name,
        "sar": v.sar, "pix_fmt": v.pix_fmt, "bit_rate": v.bit_rate,
        "nominal_fps": v.nominal_fps,
    }


def _info_from_dict(d: dict) -> VideoInfo:
    return VideoInfo(
        path=Path(d["path"]), width=d["width"], height=d["height"], fps=d["fps"],
        duration=d["duration"], nb_frames=d["nb_frames"], codec_name=d["codec_name"],
        sar=d.get("sar", "1:1"), pix_fmt=d.get("pix_fmt", ""), bit_rate=d.get("bit_rate", 0),
        nominal_fps=d.get("nominal_fps", 0.0),
    )


def _frames_to_rows(frames: FrameScores) -> list[list]:
    """The on-disk shape is unchanged (one row per frame) so files written by
    older versions still load -- the arrays are just unpacked to write."""
    # Scores are written unrounded: they are float32, so float(v) is already
    # the shortest decimal that reads back to the same bits, and rounding to
    # 6dp would land between two float32s and break an exact reload. Only
    # `time` is rounded -- it is float64 and derived from frame/fps, where
    # microsecond precision is far beyond what anything displays.
    def column(metric: str) -> list:
        arr = frames.values(metric)
        if arr is None:
            return [None] * len(frames)
        return [None if math.isnan(v) else float(v) for v in arr]

    psnr, ssim, xpsnr = column("psnr"), column("ssim"), column("xpsnr")
    return [
        [int(frames.frame[i]), round(float(frames.time[i]), 6), float(frames.vmaf[i]),
         psnr[i], ssim[i], xpsnr[i]]
        for i in range(len(frames))
    ]


def _rows_to_frames(rows: list[list]) -> FrameScores:
    if not rows:
        return FrameScores.empty()

    def column(index: int) -> np.ndarray | None:
        # fr[5] (xpsnr) is missing in files saved before XPSNR support existed.
        values = [r[index] if len(r) > index else None for r in rows]
        if all(v is None for v in values):
            return None
        return np.array([np.nan if v is None else v for v in values], dtype=np.float32)

    return FrameScores(
        frame=np.array([r[0] for r in rows], dtype=np.int32),
        time=np.array([r[1] for r in rows], dtype=np.float64),
        vmaf=np.array([r[2] for r in rows], dtype=np.float32),
        psnr=column(3), ssim=column(4), xpsnr=column(5),
    )


def save_run(result: VmafRunResult, path: Path, label: str | None = None) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "label": label or result.distorted.stem,
        "source": str(result.source),
        "distorted": str(result.distorted),
        "fps": result.fps,
        "model": result.model,
        "source_crop": _crop_to_dict(result.source_crop),
        "distorted_crop": _crop_to_dict(result.distorted_crop),
        "source_info": _info_to_dict(result.source_info),
        "distorted_info": _info_to_dict(result.distorted_info),
        "scale_direction": result.scale_direction.value,
        "frames": _frames_to_rows(result.frames),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_run(path: Path) -> tuple[VmafRunResult, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = _rows_to_frames(data["frames"])
    result = VmafRunResult(
        source=Path(data["source"]),
        distorted=Path(data["distorted"]),
        frames=frames,
        fps=data["fps"],
        model=data["model"],
        source_crop=_crop_from_dict(data.get("source_crop")),
        distorted_crop=_crop_from_dict(data.get("distorted_crop")),
        source_info=_info_from_dict(data["source_info"]),
        distorted_info=_info_from_dict(data["distorted_info"]),
        # Missing in files saved before "test both directions" existed --
        # SOURCE_TO_DISTORTED was the only behavior then, so it's the correct
        # default for those older files, not just an arbitrary fallback.
        scale_direction=ScaleDirection(data.get("scale_direction", ScaleDirection.SOURCE_TO_DISTORTED.value)),
    )
    label = data.get("label") or result.distorted.stem
    return result, label


def export_csv(result: VmafRunResult, path: Path) -> None:
    # `x if x is not None else ""`, not `x or ""`: a metric that's genuinely
    # 0.0 (VMAF and SSIM both really do bottom out at 0 for badly degraded
    # frames) is falsy, and `or` exported it as an empty cell -- making a
    # real score indistinguishable from "this metric wasn't computed".
    def cell(value: float | None) -> float | str:
        return "" if value is None else value

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "vmaf", "psnr", "ssim", "xpsnr"])
        for fr in result.frames:
            writer.writerow([fr.frame, f"{fr.time:.6f}", fr.vmaf, cell(fr.psnr), cell(fr.ssim), cell(fr.xpsnr)])
