"""Portable result persistence and CSV export."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from vmaf_app.core.metric_results import (
    UNSPECIFIED_PROVENANCE,
    FrameMetricResult,
    MetricResultSet,
    SequenceMetricResult,
    provenance_from_dict,
    provenance_to_dict,
)
from vmaf_app.core.metrics import FRAME_METRICS
from vmaf_app.core.models import (
    ComparisonResult,
    CropBox,
    FrameScores,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
)

FORMAT_VERSION = 2

#: The file extension of a saved run. A double extension rather than a bare
#: .json keeps result-file operations scoped away from unrelated JSON files.
RESULT_SUFFIX = ".metrics.json"
RESULT_FILE_FILTER = f"Analysis results (*{RESULT_SUFFIX})"


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
        "color_range": v.color_range, "color_space": v.color_space,
        "color_transfer": v.color_transfer,
        "color_primaries": v.color_primaries,
        "chroma_location": v.chroma_location,
    }


def _info_from_dict(d: dict) -> VideoInfo:
    return VideoInfo(
        path=Path(d["path"]), width=d["width"], height=d["height"], fps=d["fps"],
        duration=d["duration"], nb_frames=d["nb_frames"], codec_name=d["codec_name"],
        sar=d.get("sar", "1:1"), pix_fmt=d.get("pix_fmt", ""), bit_rate=d.get("bit_rate", 0),
        nominal_fps=d.get("nominal_fps", 0.0),
        color_range=d.get("color_range", ""),
        color_space=d.get("color_space", ""),
        color_transfer=d.get("color_transfer", ""),
        color_primaries=d.get("color_primaries", ""),
        chroma_location=d.get("chroma_location", ""),
    )


def _json_float(value: float) -> float | str | None:
    value = float(value)
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _float_from_json(value: float | int | str | None) -> float:
    if value is None:
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return float(value)


def _portable_metric_results(result: ComparisonResult) -> MetricResultSet:
    """Return authoritative results with the current shared frame view applied.

    `metric_results` is the architectural source of truth, but established
    callers can still replace `result.frames` directly. Overlaying the frame
    view here keeps those current APIs coherent without discarding generic
    sequence results or independently sampled metrics that have no frame-view
    representation.
    """
    results = result.metric_results.copy()
    for key in result.frames.metric_keys:
        values = result.frames.values(key)
        if values is None:
            continue
        current = results.frame(key)
        provenance = current.provenance if current is not None else UNSPECIFIED_PROVENANCE
        results.add(FrameMetricResult(
            key, result.frames.frame, result.frames.time, values, provenance,
        ))
    return results


def _metric_to_dict(metric) -> dict:
    common = {
        "key": metric.key,
        "provenance": provenance_to_dict(metric.provenance),
    }
    if isinstance(metric, FrameMetricResult):
        return {
            **common,
            "kind": "frame",
            "frame": [int(value) for value in metric.frame],
            "time": [_json_float(value) for value in metric.time],
            "values": [_json_float(value) for value in metric.values],
        }
    if isinstance(metric, SequenceMetricResult):
        return {**common, "kind": "sequence", "score": _json_float(metric.score)}
    raise TypeError(f"Unsupported metric result type: {type(metric).__name__}")


def _metric_from_dict(data: dict):
    if not isinstance(data, dict):
        raise TypeError("metric result must be an object")
    key = data["key"]
    if not isinstance(key, str) or not key:
        raise ValueError("metric result key must be a non-empty string")
    provenance_data = data["provenance"]
    if not isinstance(provenance_data, dict):
        raise TypeError("metric provenance must be an object")
    provenance = provenance_from_dict(provenance_data)
    kind = data.get("kind")
    if kind == "frame":
        frames, times, values = data["frame"], data["time"], data["values"]
        if not all(isinstance(column, list) for column in (frames, times, values)):
            raise TypeError("frame metric arrays must be lists")
        return FrameMetricResult(
            key,
            np.asarray(frames, dtype=np.int32),
            np.asarray([_float_from_json(value) for value in times], dtype=np.float64),
            np.asarray([_float_from_json(value) for value in values], dtype=np.float32),
            provenance,
        )
    if kind == "sequence":
        return SequenceMetricResult(key, _float_from_json(data["score"]), provenance)
    raise ValueError(f"Unsupported metric result kind: {kind!r}")


def save_run(result: ComparisonResult, path: Path, label: str | None = None) -> None:
    metrics = _portable_metric_results(result)
    payload = {
        "format_version": FORMAT_VERSION,
        "label": label or result.distorted.stem,
        "source": str(result.source),
        "distorted": str(result.distorted),
        "fps": result.fps,
        "model": result.model,
        "model_choice": result.model_choice,
        "source_crop": _crop_to_dict(result.source_crop),
        "distorted_crop": _crop_to_dict(result.distorted_crop),
        "source_info": _info_to_dict(result.source_info),
        "distorted_info": _info_to_dict(result.distorted_info),
        "scale_direction": result.scale_direction.value,
        "scale_algorithm": result.scale_algorithm,
        "resample_target": (
            None if result.resample_target is None else {
                "width": result.resample_target.width,
                "label": result.resample_target.label,
            }
        ),
        "compared_frame_count": result.compared_frame_count,
        "metric_results": [
            _metric_to_dict(metric)
            for key in metrics
            if (metric := metrics.get(key)) is not None
        ],
    }
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def load_run(path: Path) -> tuple[ComparisonResult, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported analysis result format version: {data.get('format_version')!r}"
        )
    metric_data = data.get("metric_results")
    if not isinstance(metric_data, list):
        raise TypeError("metric_results must be a list")
    metrics = MetricResultSet(_metric_from_dict(item) for item in metric_data)
    result = ComparisonResult(
        source=Path(data["source"]),
        distorted=Path(data["distorted"]),
        frames=FrameScores.empty(),
        fps=data["fps"],
        model=data["model"],
        source_crop=_crop_from_dict(data.get("source_crop")),
        distorted_crop=_crop_from_dict(data.get("distorted_crop")),
        source_info=_info_from_dict(data["source_info"]),
        distorted_info=_info_from_dict(data["distorted_info"]),
        scale_direction=ScaleDirection(data["scale_direction"]),
        scale_algorithm=data["scale_algorithm"],
        resample_target=(
            ResampleTarget(**data["resample_target"])
            if data["resample_target"] is not None
            else None
        ),
        compared_frame_count=data["compared_frame_count"],
        model_choice=data["model_choice"],
        metric_results=metrics,
    )
    return result, data["label"]


def export_csv(result: ComparisonResult, path: Path) -> None:
    """One row per frame, one column per frame metric.

    The first seven columns are what the export always had (frame, time_s,
    vmaf, vmaf_neg, psnr, ssim, xpsnr), so existing scripts keep working;
    every other frame metric follows in registry order (ssimulacra2,
    butteraugli). The export used to read only the shared frame table with
    those five columns hard-coded, so SSIMULACRA2 and Butteraugli were never
    written -- a SSIMULACRA2-only result exported rows of empty cells.

    Metrics need not share a frame axis (a subsampled perceptual metric
    scores every n-th frame), so rows are every frame any metric scored,
    and a metric with no score for a frame leaves its cell blank. A genuine
    0.0 is written as 0.0, never blank.
    """
    metrics = _portable_metric_results(result)
    columns = [(metric.key, metrics.frame(metric.key)) for metric in FRAME_METRICS]
    present = [frame for _key, frame in columns if frame is not None and len(frame.frame)]
    frames = (np.unique(np.concatenate([frame.frame for frame in present]))
              if present else np.empty(0, dtype=np.int64))
    times = np.full(len(frames), np.nan)
    values: dict[str, np.ndarray] = {}
    for key, frame in columns:
        column = np.full(len(frames), np.nan)
        if frame is not None and len(frame.frame):
            at = np.searchsorted(frames, frame.frame)
            column[at] = frame.values
            times[at] = np.where(np.isnan(times[at]), frame.time, times[at])
        values[key] = column

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", *(key for key, _frame in columns)])
        for row, frame_number in enumerate(frames):
            writer.writerow([
                int(frame_number), f"{times[row]:.6f}",
                *("" if np.isnan(values[key][row]) else float(values[key][row]) for key, _frame in columns),
            ])
