"""Version-2 internal cache: direct, independently-addressable metric files."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from vmaf_app.core.analysis_request import MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.metric_results import (
    FrameMetricResult,
    MetricResultSet,
    SequenceMetricResult,
    provenance_from_dict,
    provenance_to_dict,
)
from vmaf_app.core.models import ComparisonResult, CropBox, ResampleTarget, ScaleDirection, VideoInfo

METRIC_CACHE_FORMAT_VERSION = 2
_V2_DIR = "v2"


def file_identity(path: Path) -> str:
    path = Path(path).resolve()
    try:
        stat = path.stat()
        size, modified = stat.st_size, stat.st_mtime_ns
    except OSError:
        size, modified = -1, -1
    return f"{path}:{size}:{modified}"


def _canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _json_default(value: object):
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def recipe_hash(source: Path, distorted: Path, recipe: ComparisonRecipe) -> str:
    raw = _canonical({
        "source": file_identity(source), "distorted": file_identity(distorted),
        "recipe": recipe.identity_dict(),
    })
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def metric_identity_hash(spec: MetricRequestSpec) -> str:
    return hashlib.sha256(_canonical(spec.identity_dict()).encode("utf-8")).hexdigest()


def recipe_directory(base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe) -> Path:
    return Path(base) / _V2_DIR / recipe_hash(source, distorted, recipe)


def metric_path(directory: Path, spec: MetricRequestSpec) -> Path:
    return directory / f"{spec.key}_{metric_identity_hash(spec)}.npz"


def _crop_to_dict(crop: CropBox | None) -> dict | None:
    return None if crop is None else {"w": crop.w, "h": crop.h, "x": crop.x, "y": crop.y}


def _crop_from_dict(data: dict | None) -> CropBox | None:
    return None if data is None else CropBox(**data)


def _info_to_dict(info: VideoInfo) -> dict:
    return {
        "path": str(info.path), "width": info.width, "height": info.height,
        "fps": info.fps, "duration": info.duration, "nb_frames": info.nb_frames,
        "codec_name": info.codec_name, "sar": info.sar, "pix_fmt": info.pix_fmt,
        "bit_rate": info.bit_rate, "nominal_fps": info.nominal_fps,
        "color_range": info.color_range, "color_space": info.color_space,
        "color_transfer": info.color_transfer, "color_primaries": info.color_primaries,
    }


def _info_from_dict(data: dict) -> VideoInfo:
    return VideoInfo(path=Path(data["path"]), width=data["width"], height=data["height"],
                     fps=data["fps"], duration=data["duration"], nb_frames=data["nb_frames"],
                     codec_name=data["codec_name"], sar=data.get("sar", "1:1"),
                     pix_fmt=data.get("pix_fmt", ""), bit_rate=data.get("bit_rate", 0),
                     nominal_fps=data.get("nominal_fps", 0.0), color_range=data.get("color_range", ""),
                     color_space=data.get("color_space", ""), color_transfer=data.get("color_transfer", ""),
                     color_primaries=data.get("color_primaries", ""))


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_canonical(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _metadata(result, spec: MetricRequestSpec) -> np.ndarray:
    provenance = provenance_to_dict(result.provenance)
    if result.key != "vmaf":
        # This cache supports score reuse. Non-VMAF implementation-library
        # versions must not become part of its stored results; VMAF retains
        # version provenance alongside its model-specific request identity.
        provenance["implementation_version"] = ""
    data = {
        "format_version": METRIC_CACHE_FORMAT_VERSION,
        "kind": "frame" if isinstance(result, FrameMetricResult) else "sequence",
        "key": result.key, "request": spec.identity_dict(),
        "provenance": provenance,
    }
    return np.array(_canonical(data))


def store_metric(directory: Path, result, spec: MetricRequestSpec) -> Path:
    if result.key != spec.key:
        raise ValueError("metric result and request key differ")
    arrays = {"metadata": _metadata(result, spec)}
    if isinstance(result, FrameMetricResult):
        arrays.update(frame=result.frame, time=result.time, values=result.values)
    elif isinstance(result, SequenceMetricResult):
        arrays["score"] = np.array(result.score, dtype=np.float64)
    else:
        raise TypeError("unsupported metric result")
    path = metric_path(directory, spec)
    _atomic_npz(path, arrays)
    return path


def load_metric(directory: Path, spec: MetricRequestSpec):
    path = metric_path(directory, spec)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION
                    or metadata.get("key") != spec.key
                    or metadata.get("request") != spec.identity_dict()):
                return None
            provenance = provenance_from_dict(metadata["provenance"])
            if metadata["kind"] == "frame":
                return FrameMetricResult(spec.key, data["frame"], data["time"], data["values"], provenance)
            if metadata["kind"] == "sequence":
                return SequenceMetricResult(spec.key, float(data["score"].item()), provenance)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return None


def load_metrics(directory: Path, specs: tuple[MetricRequestSpec, ...]) -> MetricResultSet:
    results = MetricResultSet()
    for spec in specs:
        result = load_metric(directory, spec)
        if result is not None:
            results.add(result)
    return results


def _context_from_result(result: ComparisonResult, label: str, recipe: ComparisonRecipe) -> dict:
    return {
        "format_version": METRIC_CACHE_FORMAT_VERSION, "label": label,
        "source": str(result.source), "distorted": str(result.distorted), "fps": result.fps,
        "source_crop": _crop_to_dict(result.source_crop), "distorted_crop": _crop_to_dict(result.distorted_crop),
        "source_info": _info_to_dict(result.source_info), "distorted_info": _info_to_dict(result.distorted_info),
        "scale_direction": result.scale_direction.value, "scale_algorithm": result.scale_algorithm,
        "resample_target": None if result.resample_target is None else {
            "width": result.resample_target.width, "label": result.resample_target.label,
        },
        "compared_frame_count": result.compared_frame_count,
        "model": result.model, "model_choice": result.model_choice,
        "recipe": recipe.identity_dict(),
    }


def store_result(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    result: ComparisonResult, label: str, specs: tuple[MetricRequestSpec, ...],
) -> Path:
    directory = recipe_directory(base, source, distorted, recipe)
    _atomic_json(directory / "context.json", _context_from_result(result, label, recipe))
    for spec in specs:
        metric = result.metric(spec.key)
        if metric is not None:
            store_metric(directory, metric, spec)
    return directory


def load_result(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    specs: tuple[MetricRequestSpec, ...], supplemental_specs: tuple[MetricRequestSpec, ...] = (),
):
    directory = recipe_directory(base, source, distorted, recipe)
    context_path = directory / "context.json"
    if not context_path.exists():
        return None
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
        if context.get("format_version") != METRIC_CACHE_FORMAT_VERSION:
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    results = load_metrics(directory, specs)
    # Extra current metrics are direct lookups too. They preserve the UI
    # behavior of showing every compatible score already cached for a row
    # without an exponential search through metric combinations.
    for spec in supplemental_specs:
        if not results.has(spec.key):
            extra = load_metric(directory, spec)
            if extra is not None:
                results.add(extra)
    if not results:
        return None

    from vmaf_app.core.metric_results import frame_scores_from_results

    expected_frames = context.get("compared_frame_count")
    if isinstance(expected_frames, int) and expected_frames > 0:
        compatible = MetricResultSet()
        for key in results:
            metric = results.get(key)
            if isinstance(metric, FrameMetricResult) and len(metric.frame) != expected_frames:
                continue
            if metric is not None:
                compatible.add(metric)
        results = compatible
        if not results:
            return None

    frame_view = frame_scores_from_results(results)

    try:
        result = ComparisonResult(
            source=Path(context["source"]), distorted=Path(context["distorted"]),
            frames=frame_view, fps=context["fps"], model=context.get("model", ""),
            source_crop=_crop_from_dict(context.get("source_crop")), distorted_crop=_crop_from_dict(context.get("distorted_crop")),
            source_info=_info_from_dict(context["source_info"]), distorted_info=_info_from_dict(context["distorted_info"]),
            scale_direction=ScaleDirection(context.get("scale_direction", ScaleDirection.SOURCE_TO_DISTORTED.value)),
            scale_algorithm=context.get("scale_algorithm", "bicubic"),
            resample_target=ResampleTarget(**context["resample_target"]) if context.get("resample_target") else None,
            compared_frame_count=context.get("compared_frame_count", 0), model_choice=context.get("model_choice"),
            metric_results=results,
        )
        return result, context.get("label") or Path(context["distorted"]).stem
    except (KeyError, TypeError, ValueError):
        return None


def cache_summary(base: Path) -> tuple[int, int]:
    """Return ``(comparison_count, byte_count)`` for the metric cache."""
    root = Path(base) / _V2_DIR
    if not root.exists():
        return 0, 0
    contexts = list(root.glob("*/context.json"))
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with contextlib.suppress(OSError):
            total += path.stat().st_size
    return len(contexts), total


def clear_metrics(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    specs: tuple[MetricRequestSpec, ...],
) -> int:
    """Delete only the requested metric identities for one comparison recipe."""
    directory = recipe_directory(base, source, distorted, recipe)
    removed = 0
    for spec in specs:
        path = metric_path(directory, spec)
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            continue
    if directory.exists() and not any(directory.glob("*.npz")):
        with contextlib.suppress(OSError):
            shutil.rmtree(directory)
    return removed


def clear_recipe(base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe) -> int:
    directory = recipe_directory(base, source, distorted, recipe)
    if not directory.exists():
        return 0
    count = sum(1 for path in directory.rglob("*") if path.is_file())
    try:
        shutil.rmtree(directory)
    except OSError:
        return 0
    return count


def clear_all(base: Path) -> int:
    root = Path(base) / _V2_DIR
    if not root.exists():
        return 0
    count = sum(1 for path in root.glob("*/context.json") if path.is_file())
    try:
        shutil.rmtree(root)
    except OSError:
        return 0
    return count
