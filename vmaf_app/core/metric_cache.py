"""Version-2 internal cache: direct, independently-addressable metric files."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
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
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE, model_for_resolution
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
        "chroma_location": info.chroma_location,
    }


def _info_from_dict(data: dict) -> VideoInfo:
    return VideoInfo(path=Path(data["path"]), width=data["width"], height=data["height"],
                     fps=data["fps"], duration=data["duration"], nb_frames=data["nb_frames"],
                     codec_name=data["codec_name"], sar=data.get("sar", "1:1"),
                     pix_fmt=data.get("pix_fmt", ""), bit_rate=data.get("bit_rate", 0),
                     nominal_fps=data.get("nominal_fps", 0.0), color_range=data.get("color_range", ""),
                     color_space=data.get("color_space", ""), color_transfer=data.get("color_transfer", ""),
                     color_primaries=data.get("color_primaries", ""),
                     chroma_location=data.get("chroma_location", ""))


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
    # The UI request is backend-neutral, but the implementation used for a
    # perceptual score affects its numeric result. Persist under the concrete
    # implementation ID reported by provenance so a CPU score can never
    # masquerade as a Vship GPU score (or vice versa).
    if _is_auto_perceptual_spec(spec):
        spec = replace(
            spec,
            implementation_compatibility_id=result.provenance.implementation_compatibility_id,
        )
    arrays = {"metadata": _metadata(result, spec)}
    if isinstance(result, FrameMetricResult):
        arrays.update(frame=result.frame, time=result.time, values=result.values)
    elif isinstance(result, SequenceMetricResult):
        arrays["score"] = np.array(result.score, dtype=np.float64)
        if result.has_timeline:
            arrays.update(frame=result.frame, time=result.time, values=result.values)
    else:
        raise TypeError("unsupported metric result")
    path = metric_path(directory, spec)
    _atomic_npz(path, arrays)
    return path


def _is_auto_perceptual_spec(spec: MetricRequestSpec) -> bool:
    return (
        spec.backend_id == "perceptual"
        and spec.implementation_compatibility_id.endswith("-auto-or-libjxl-cpu-v1")
    )


def _auto_perceptual_compatibility(spec: MetricRequestSpec, compatibility: object) -> bool:
    value = str(compatibility or "")
    return (
        value == f"{spec.key}-vship-gpu-v1"
        or value == f"{spec.key}-libjxl-cpu-v1"
        # Accept existing cache entries written before implementation-library
        # versions were removed from compatibility IDs. The implementation
        # family/backend remains part of the ID; only its package version does not.
        or (value.startswith(f"{spec.key}-vship-") and value.endswith("-gpu-v1"))
        or (value.startswith(f"{spec.key}-libjxl-") and value.endswith("-cpu-v1"))
    )


def _load_metric_file(path: Path, spec: MetricRequestSpec):
    if not path.is_file():
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
                timeline = (data["frame"], data["time"], data["values"]) if "values" in data else (None, None, None)
                return SequenceMetricResult(spec.key, float(data["score"].item()), provenance, *timeline)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return None


def _auto_perceptual_candidates(directory: Path, spec: MetricRequestSpec):
    """Return concrete saved implementations matching a backend-neutral request."""
    candidates: list[tuple[int, Path, MetricRequestSpec]] = []
    try:
        paths = directory.glob(f"{spec.key}_*.npz")
        for path in paths:
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                request = metadata.get("request", {})
                compatibility = request.get("implementation_compatibility_id")
                provenance = metadata.get("provenance", {})
                if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION
                        or metadata.get("key") != spec.key
                        or not _auto_perceptual_compatibility(spec, compatibility)
                        or not isinstance(provenance, dict)
                        or provenance.get("implementation_compatibility_id") != compatibility):
                    continue
                concrete = replace(spec, implementation_compatibility_id=compatibility)
                if request != concrete.identity_dict():
                    continue
                # Prefer a prior GPU run, while still allowing CPU-only
                # machines to reuse the bundled libjxl result.
                rank = 0 if "-vship-" in compatibility else 1
                candidates.append((rank, path, concrete))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
    except OSError:
        return []
    return sorted(candidates, key=lambda item: item[0])


def load_other_parameters(directory: Path, spec: MetricRequestSpec) -> list[tuple[dict, float]]:
    """(parameters, score) of each whole-video score saved for `spec`'s
    metric with other parameters -- CVVDP's other displays -- and otherwise
    the same request. Never an answer for `spec` itself: a score is only
    valid for the parameters it was made with."""
    wanted = spec.identity_dict()
    rest = {name: value for name, value in wanted.items() if name != "parameters"}
    found: list[tuple[dict, float]] = []
    try:
        paths = sorted(directory.glob(f"{spec.key}_*.npz"))
    except OSError:
        return []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                request = metadata.get("request", {})
                if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION
                        or metadata.get("key") != spec.key or metadata.get("kind") != "sequence"
                        or {name: value for name, value in request.items() if name != "parameters"} != rest
                        or _canonical(request.get("parameters")) == _canonical(wanted["parameters"])):
                    continue
                score = float(data["score"].item())
                parameters = dict(request["parameters"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if math.isfinite(score):
            found.append((parameters, score))
    return found


def _vmaf_model_ran(request: dict, provenance: dict) -> str | None:
    """The model a saved VMAF score was calculated with, when that is certain.

    An explicit or bundled choice is the model. For Auto, a score calculated
    by this app records the model in its provenance; one migrated from the
    first cache format was stored under the model that ran. A score re-saved
    from an older result can carry neither -- its key's model was the row's
    leftover default, "vmaf_v0.6.1" on 4K-model scores -- so it is unknown.
    """
    parameters = request.get("parameters") or {}
    choice = parameters.get("model_choice")
    if isinstance(choice, str) and choice not in ("", AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE):
        return choice
    if choice != AUTO_MODEL_CHOICE:
        return None
    ran = (provenance.get("parameters") or {}).get("model")
    if isinstance(ran, str) and ran.startswith("version="):
        return ran
    model = parameters.get("model")
    if provenance.get("implementation") == "legacy-v1-cache" and isinstance(model, str) and model.startswith("version="):
        return model
    return None


def _auto_model_for(directory: Path) -> str | None:
    """The model Auto runs for this comparison, from the sizes and black bars
    its saved context records: the size compared at, as the run decides it."""
    from vmaf_app.core.vmaf_runner import compared_dimensions, resample_analysis_dimensions

    try:
        context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
        source = _info_from_dict(context["source_info"])
        distorted = _info_from_dict(context["distorted_info"])
        source_crop = _crop_from_dict(context.get("source_crop"))
        if context.get("resample_target"):
            return model_for_resolution(*resample_analysis_dimensions(source, source_crop))
        return model_for_resolution(*compared_dimensions(
            source, distorted, ScaleDirection(context["scale_direction"]),
            source_crop, _crop_from_dict(context.get("distorted_crop"))))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return None


def _vmaf_equivalents(directory: Path, spec: MetricRequestSpec) -> list[tuple[int, Path, MetricRequestSpec]]:
    """Saved VMAF scores for the same comparison under another key that are
    scores of the same model: an Auto row's for any Auto score, or one for
    the model Auto picks here; an explicit row's for an Auto score that
    certainly ran that model. The key used to have to match exactly, so a
    row on Auto found nothing saved with "VMAF 4K v0.6.1" chosen, though
    Auto picks that model for the same comparison, and a row's own score
    could be missed when its key's leftover model field differed."""
    wanted = spec.identity_dict()
    parameters = wanted["parameters"]
    choice = parameters.get("model_choice")
    rest = {name: value for name, value in wanted.items() if name != "parameters"}
    exact = metric_path(directory, spec)
    auto_model: list[str | None] = []  # worked out once, only if needed
    found: list[tuple[int, Path, MetricRequestSpec]] = []
    try:
        paths = sorted(directory.glob("vmaf_*.npz"))
    except OSError:
        return []
    for path in paths:
        if path == exact:
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
            request = metadata.get("request") or {}
            stored = request.get("parameters") or {}
            if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION or metadata.get("key") != "vmaf"
                    or {name: value for name, value in request.items() if name != "parameters"} != rest
                    or stored.get("custom_model", "") != parameters.get("custom_model", "")):
                continue
            if stored.get("model_choice") == choice:
                rank = 1
            else:
                ran = _vmaf_model_ran(request, metadata.get("provenance") or {})
                if ran is None:
                    continue
                if choice == AUTO_MODEL_CHOICE:
                    if not auto_model:
                        auto_model.append(_auto_model_for(directory))
                    if ran != auto_model[0]:
                        continue
                elif not (isinstance(choice, str) and choice.startswith("version=")
                          and stored.get("model_choice") == AUTO_MODEL_CHOICE and ran == choice):
                    continue
                rank = 2
            found.append((rank, path, replace(spec, parameters=tuple(stored.items()))))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda item: item[0])


def _is_vmaf_spec(spec: MetricRequestSpec) -> bool:
    return spec.key == "vmaf" and spec.backend_id == "ffmpeg"


def load_metric(directory: Path, spec: MetricRequestSpec, compute_backend: str = "gpu"):
    """Load one cached metric. `compute_backend` is the user's GPU/CPU choice
    for a perceptual metric: "cpu" accepts only a libjxl CPU score, never a
    Vship GPU one (the two can differ by a few points on the same frames);
    "gpu" prefers a Vship score and accepts a CPU one, which is what a GPU
    selection produces on a machine without a supported GPU."""
    if _is_auto_perceptual_spec(spec):
        for _rank, path, concrete in _auto_perceptual_candidates(directory, spec):
            if compute_backend == "cpu" and "-vship-" in concrete.implementation_compatibility_id:
                continue
            result = _load_metric_file(path, concrete)
            if result is not None:
                return result
        return None
    result = _load_metric_file(metric_path(directory, spec), spec)
    if result is None and _is_vmaf_spec(spec):
        for _rank, path, concrete in _vmaf_equivalents(directory, spec):
            result = _load_metric_file(path, concrete)
            if result is not None:
                break
    return result


def load_metrics(
    directory: Path, specs: tuple[MetricRequestSpec, ...],
    compute_backends: Mapping[str, str] | None = None,
) -> MetricResultSet:
    backends = compute_backends or {}
    results = MetricResultSet()
    for spec in specs:
        result = load_metric(directory, spec, backends.get(spec.key, "gpu"))
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
    compute_backends: Mapping[str, str] | None = None,
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
    results = load_metrics(directory, specs, compute_backends)
    # Extra current metrics are direct lookups too. They preserve the UI
    # behavior of showing every compatible score already cached for a row
    # without an exponential search through metric combinations.
    for spec in supplemental_specs:
        if not results.has(spec.key):
            extra = load_metric(directory, spec, (compute_backends or {}).get(spec.key, "gpu"))
            if extra is not None:
                results.add(extra)
    if not results:
        return None

    from vmaf_app.core.metric_results import frame_scores_from_results

    # Every stored metric is kept, whatever its number of scores. A metric's
    # frame axis is its own: a subsampled one holds every n-th frame, and
    # FFmpeg and Vship can legitimately end a frame apart. Comparing each
    # metric's length with context.json's compared_frame_count threw such
    # metrics away on reload -- and context.json is rewritten by every run,
    # so which metric survived depended on which run was saved last. The
    # directory is already keyed by file identity and the comparison recipe,
    # which is what makes a stored score valid for this comparison.
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
        if _is_auto_perceptual_spec(spec):
            for _rank, path, _concrete in _auto_perceptual_candidates(directory, spec):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
            continue
        if _is_vmaf_spec(spec):
            # Recalculating must not leave an equivalent score to come back.
            for _rank, path, _concrete in _vmaf_equivalents(directory, spec):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
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
