"""Save/load a VmafRunResult to a portable JSON file, and CSV export, so
past runs can be reloaded later and overlaid in the comparison graph."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from vmaf_app.core.models import CropBox, FrameScore, ScaleDirection, VideoInfo, VmafRunResult

FORMAT_VERSION = 1


def _crop_to_dict(c: CropBox | None) -> dict | None:
    return None if c is None else {"w": c.w, "h": c.h, "x": c.x, "y": c.y}


def _crop_from_dict(d: dict | None) -> CropBox | None:
    return None if d is None else CropBox(**d)


def _info_to_dict(v: VideoInfo) -> dict:
    return {
        "path": str(v.path), "width": v.width, "height": v.height, "fps": v.fps,
        "duration": v.duration, "nb_frames": v.nb_frames, "codec_name": v.codec_name,
        "sar": v.sar, "pix_fmt": v.pix_fmt, "bit_rate": v.bit_rate,
    }


def _info_from_dict(d: dict) -> VideoInfo:
    return VideoInfo(
        path=Path(d["path"]), width=d["width"], height=d["height"], fps=d["fps"],
        duration=d["duration"], nb_frames=d["nb_frames"], codec_name=d["codec_name"],
        sar=d.get("sar", "1:1"), pix_fmt=d.get("pix_fmt", ""), bit_rate=d.get("bit_rate", 0),
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
        "frames": [[f.frame, round(f.time, 6), f.vmaf, f.psnr, f.ssim, f.xpsnr] for f in result.frames],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_run(path: Path) -> tuple[VmafRunResult, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        # fr[5] (xpsnr) is missing in files saved before XPSNR support existed.
        FrameScore(frame=fr[0], time=fr[1], vmaf=fr[2], psnr=fr[3], ssim=fr[4], xpsnr=fr[5] if len(fr) > 5 else None)
        for fr in data["frames"]
    ]
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
        raw_log_path=None,
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
