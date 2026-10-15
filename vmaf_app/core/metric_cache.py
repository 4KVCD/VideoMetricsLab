"""The metric cache: scores saved on disk, so that a run can reuse scores
calculated before instead of calculating them again.

Each comparison has a directory of its own, <base>/v2/<hash>, named by a hash
of the two files' identities (path, size and modification time) and the
comparison recipe (crop, scaling direction and the like). The directory
holds context.json, which describes the comparison, and one .npz file per
metric score, named by the metric and a hash of its request (the metric's
parameters and the implementation that answers it). Each score file can be
read, written and removed on its own.

store_result saves a run's scores and load_result reads them back;
reusable decides which scores a new run keeps.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from vmaf_app.core.analysis_request import MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.metric_results import (
    UNSPECIFIED_PROVENANCE,
    FrameMetricResult,
    MetricResultSet,
    SequenceMetricResult,
    provenance_from_dict,
    provenance_to_dict,
    sampled_every,
)
from vmaf_app.core.model_select import (
    AUTO_MODEL_CHOICE,
    CUSTOM_MODEL_CHOICE,
    is_v1_choice,
    model_for_resolution,
    v1_model_for_resolution,
)
from vmaf_app.core.models import (
    ComparisonResult,
    ResampleTarget,
    ScaleDirection,
    crop_from_dict,
    crop_to_dict,
    video_info_from_dict,
    video_info_to_dict,
)

_log = logging.getLogger(__name__)

METRIC_CACHE_FORMAT_VERSION = 2
_V2_DIR = "v2"


def file_identity(path: Path) -> str:
    """A file's identity for the cache: its full path, size and modification
    time. A file that is replaced or edited gets a new identity, so its old
    scores are not reused."""
    path = Path(path).resolve()
    try:
        stat = path.stat()
        size, modified = stat.st_size, stat.st_mtime_ns
    except OSError:
        size, modified = -1, -1
    return f"{path}:{size}:{modified}"


def _canonical(data: object) -> str:
    """`data` as compact JSON with sorted keys: the same data always gives
    the same text, and so the same hash."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _json_default(value: object):
    """How json writes the recipe's enums (their value) and objects (their fields)."""
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def recipe_hash(source: Path, distorted: Path, recipe: ComparisonRecipe) -> str:
    """The hash that names a comparison's directory: of both files'
    identities and the comparison recipe."""
    return _hash(file_identity(source), file_identity(distorted), recipe.identity_dict())


def _hash(source_identity: str, distorted_identity: str, recipe_identity: dict) -> str:
    """The SHA-256 of both files' identities and the recipe's."""
    raw = _canonical({"source": source_identity, "distorted": distorted_identity, "recipe": recipe_identity})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def metric_identity_hash(spec: MetricRequestSpec) -> str:
    """The hash of a metric request, which names its score file."""
    return hashlib.sha256(_canonical(spec.identity_dict()).encode("utf-8")).hexdigest()


#: The scaling algorithms the app offers. Older caches included the scaling
#: algorithm in a comparison's identity, so a comparison could have one
#: directory per algorithm; the identity no longer includes it
#: (ComparisonRecipe.identity_dict).
_LEGACY_SCALE_ALGORITHMS = ("bicubic", "lanczos", "bilinear", "spline")


def recipe_directory(base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe) -> Path:
    """Returns the comparison's directory in the cache.

    Directories saved under the older identity, one per scaling algorithm,
    are merged into the new one the first time it is asked for: the one for
    the recipe's own algorithm is renamed to the new name, and the scores of
    the others are moved into it (_merge_legacy). If the first cannot be
    renamed (because it is open elsewhere), it is used where it is."""
    root = Path(base) / _V2_DIR
    identities = file_identity(source), file_identity(distorted)
    identity = recipe.identity_dict()
    directory = root / _hash(*identities, identity)
    if directory.exists() or not root.is_dir():
        return directory
    algorithms = [recipe.scale_algorithm, *(a for a in _LEGACY_SCALE_ALGORITHMS if a != recipe.scale_algorithm)]
    legacy = [path for path in (root / _hash(*identities, {**identity, "scale_algorithm": algorithm})
                                for algorithm in algorithms) if path.is_dir()]
    if not legacy:
        return directory
    first, *others = legacy
    try:
        first.rename(directory)
    except OSError:
        return first
    for other in others:
        _merge_legacy(other, directory)
    return directory


def _merge_legacy(legacy: Path, directory: Path) -> None:
    """Moves the files of `legacy`, a directory of the same comparison saved
    under another scaling algorithm, into `directory`.

    When `directory` already has the same score, its own is kept and the
    other copy is removed: a comparison is the same whatever it was scaled
    with. Files that cannot be moved stay where they are."""
    for item in legacy.iterdir():
        target = directory / item.name
        with contextlib.suppress(OSError):
            if target.exists():
                if item.is_file():
                    item.unlink()
            else:
                item.rename(target)
    with contextlib.suppress(OSError):
        legacy.rmdir()


def metric_path(directory: Path, spec: MetricRequestSpec) -> Path:
    """The file a request's score is saved in: <metric>_<request hash>.npz."""
    return directory / f"{spec.key}_{metric_identity_hash(spec)}.npz"


#: Every Vship GPU score records the colour-tag mapping it was made with, as
#: the provenance parameter "color_tags". Scores from app versions 1.2 and
#: 1.2.1 have none. Those versions read an RGB video with no transfer tag as
#: BT.709, while Vship 5.1.1's FFVship (and the app since) read it as sRGB.
#: Every other input is scored as those versions scored it, so only the
#: scores of such a pair are calculated again (_stale_vship_score).
VSHIP_COLOR_TAGS = "ffvship-5.1.1"
#: The RGB pixel formats those versions scored on the GPU.
_OLD_VSHIP_RGB = re.compile(
    r"gbrp(?:(?:9|10|12|14|16)(?:le|be)?)?|rgb24|bgr24|rgba|bgra|argb|abgr|rgb0|bgr0|rgb48(?:le|be)|rgba64(?:le|be)")
_UNTAGGED = {"", "unknown", "unspecified", "reserved"}


#: Every CPU SSIMULACRA2 and Butteraugli score records how its pictures'
#: colours were read, as the provenance parameter "color_tags". The CPU tools
#: are given the pictures' colours as Vship reads them (colour.describe_png),
#: and Butteraugli's 3-norm is Vship's. A score without this value was made
#: with FFmpeg's own colour conversion and the tool's own norm (a tagged
#: BT.709 film scored 15 SSIMULACRA2 points below the GPU), so it is
#: calculated again. The "/2" version marks scores of scaled videos converted
#: to RGB after their colour tags are set: before that, FFmpeg converted an
#: untagged HD source with BT.601's matrix while scaling it.
CPU_COLOR_TAGS = "vship-5.1.1/2"


def _stale_cpu_score(provenance) -> bool:
    """Whether a CPU SSIMULACRA2 or Butteraugli score predates the current
    colour handling (CPU_COLOR_TAGS) and must be calculated again."""
    return (provenance.compute_backend == "cpu" and provenance.implementation in {"ssimulacra2", "butteraugli"}
            and provenance.parameters.get("color_tags") != CPU_COLOR_TAGS)


def _stale_vship_score(provenance, infos: tuple[object, ...]) -> bool:
    """Whether a saved score is known to be wrong and must be calculated
    again: a GPU score from app version 1.2 or 1.2.1 for a pair with an RGB
    video that has no transfer tag (VSHIP_COLOR_TAGS), or a CPU score from
    before the CPU tools read colours as Vship does (CPU_COLOR_TAGS).
    `infos` are the two videos' entries in context.json."""
    if _stale_cpu_score(provenance):
        return True
    if (provenance.compute_backend != "gpu" or not provenance.implementation.startswith("Vship/")
            or provenance.parameters.get("color_tags") == VSHIP_COLOR_TAGS):
        return False
    return any(
        isinstance(info, dict)
        and _OLD_VSHIP_RGB.fullmatch(str(info.get("pix_fmt") or "").strip().casefold()) is not None
        and str(info.get("color_transfer") or "").strip().casefold() in _UNTAGGED
        for info in infos
    )


def _compatible(spec: MetricRequestSpec, provenance) -> bool:
    """Whether a score with `provenance` was made by an implementation that
    answers `spec`, judged by the implementation's compatibility id.

    A score saved before scores recorded their provenance (shown as
    UNSPECIFIED_PROVENANCE's "unversioned") answers only libvmaf's metrics,
    whose id has never changed. XPSNR's id has changed (its weighting, see
    XPSNR_COMPATIBILITY_ID), and so have the perceptual tools' ids."""
    compatibility = provenance.implementation_compatibility_id
    if _is_auto_perceptual_spec(spec):
        return _auto_perceptual_compatibility(spec, compatibility)
    if compatibility == UNSPECIFIED_PROVENANCE.implementation_compatibility_id:
        return spec.implementation_compatibility_id == "ffmpeg-libvmaf-v1"
    return compatibility == spec.implementation_compatibility_id


def answers(spec: MetricRequestSpec, result, infos: tuple[object, ...] = ()) -> bool:
    """Whether a score made earlier (on a row, or in an opened run file)
    answers `spec` now, by the same rule a saved score must meet to be
    loaded: it was made by a compatible implementation (_compatible) and is
    not known to be wrong (_stale_vship_score). `infos` holds the two videos
    as context.json has them, or dicts with their pix_fmt and color_transfer.

    Without this check, an old score on a row (an XPSNR from v1.4, weighted
    by the encode) was kept beside new metrics and then saved as current."""
    return _compatible(spec, result.provenance) and not _stale_vship_score(result.provenance, infos)


def video_infos(*infos) -> tuple[dict, ...]:
    """What _stale_vship_score reads of each VideoInfo, as context.json has it."""
    return tuple({"pix_fmt": info.pix_fmt, "color_transfer": info.color_transfer}
                 for info in infos if info is not None)


def made_as_chosen(key: str, choice: str | None, produced: str, gpu_can_score: Callable[[str], bool]) -> bool:
    """Whether a score of `key` made on `produced` ("gpu" or "cpu") counts
    for the metric when it is set to `choice` ("gpu" or "cpu"; None for a
    metric with no such choice, where any score counts).

    SSIMULACRA2 and Butteraugli on the GPU (Vship) and on the CPU (libjxl)
    give different scores for the same frames: a little different on SDR
    (55.13 against 55.24 SSIMULACRA2 on a 1080p film), far more on HDR (35.1
    against 47.3 on a PQ film, because libjxl's tool scores PQ its own way).
    A comparison that mixed them would rank encodes on different scales, so:
    - set to CPU, only a CPU score counts;
    - set to GPU, a CPU score (left by a fallback, or by an earlier CPU
      choice) counts only where the GPU cannot calculate the metric
      (`gpu_can_score`, which is perceptual_vship.gpu_can_score). On a PC
      with a suitable GPU, the next run calculates it there."""
    if choice == "cpu":
        return produced == "cpu"
    if choice == "gpu" and produced == "cpu":
        return not gpu_can_score(key)
    return True


def reusable(result: ComparisonResult, specs: Iterable[MetricRequestSpec], backends: Mapping[str, str],
             gpu_can_score: Callable[[str], bool], infos: tuple[object, ...] = ()) -> MetricResultSet:
    """Returns the scores of `result` that a run asking for `specs` keeps
    instead of calculating them again. `result` can be a row's last run, an
    opened run file or the cache's. Both the window's Run button and the
    command line follow this rule.

    A metric's score is kept when it exists (a finite value for a
    whole-video score, at least one frame for a per-frame metric), was made
    where the metric is set to be calculated (`backends`, see
    made_as_chosen), and answers the request as a saved score must to be
    loaded (see answers). `infos` is video_infos of the two videos."""
    kept = MetricResultSet()
    for spec in specs:
        sequence = result.sequence_metric(spec.key)
        if sequence is not None:
            # One score for the video (CVVDP). A row's is always its own:
            # changing the CVVDP display drops it (MainWindow._drop_cvvdp_result).
            if np.isfinite(sequence.score) and answers(spec, sequence, infos):
                kept.add(sequence)
            continue
        metric = result.frame_metric(spec.key)
        if (metric is not None and np.any(~np.isnan(metric.values))
                and made_as_chosen(spec.key, backends.get(spec.key), metric.provenance.compute_backend, gpu_can_score)
                and answers(spec, metric, infos)):
            kept.add(metric)
    return kept


def _context_infos(directory: Path) -> tuple[object, ...]:
    """The two videos' entries in a comparison's context.json, if readable."""
    try:
        context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
        return context.get("source_info"), context.get("distorted_info")
    except (OSError, ValueError, AttributeError):
        return ()


def _atomic_json(path: Path, data: dict) -> None:
    """Writes `data` to `path` as JSON, through a temporary file that is then
    renamed, so a crash never leaves half a file."""
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
    """Writes `arrays` to `path` as a compressed .npz, through a temporary
    file that is then renamed, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _metadata(result, spec: MetricRequestSpec) -> np.ndarray:
    """The metadata saved with a score: the cache format, the kind of score,
    the request it answers and the score's provenance."""
    provenance = provenance_to_dict(result.provenance)
    if result.key != "vmaf":
        # The library version is not saved for metrics other than VMAF, so a
        # library update does not make their saved scores look different.
        # VMAF keeps its version beside its model-specific request.
        provenance["implementation_version"] = ""
    data = {
        "format_version": METRIC_CACHE_FORMAT_VERSION,
        "kind": "frame" if isinstance(result, FrameMetricResult) else "sequence",
        "key": result.key, "request": spec.identity_dict(),
        "provenance": provenance,
    }
    return np.array(_canonical(data))


def store_metric(directory: Path, result, spec: MetricRequestSpec) -> Path:
    """Saves one metric's score in the comparison's `directory`, under its
    request `spec`, and returns the file's path."""
    if result.key != spec.key:
        raise ValueError("metric result and request key differ")
    # The window asks for a perceptual metric without saying where to
    # calculate it, but the CPU and the GPU give different scores. The score
    # is saved under the implementation that made it (from its provenance),
    # so a CPU score can never pass for a Vship GPU score, or the reverse.
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
    """Whether `spec` asks for a perceptual metric without fixing where it is
    calculated (Vship on the GPU, or libjxl on the CPU)."""
    return (
        spec.backend_id == "perceptual"
        and spec.implementation_compatibility_id.endswith("-auto-or-libjxl-cpu-v1")
    )


def _auto_perceptual_compatibility(spec: MetricRequestSpec, compatibility: object) -> bool:
    """Whether a saved perceptual score's compatibility id answers `spec`,
    a request that does not fix where the metric is calculated."""
    value = str(compatibility or "")
    return (
        value == f"{spec.key}-vship-gpu-v1"
        or value == f"{spec.key}-libjxl-cpu-v1"
        # Older cache entries have the library's version in the id as well
        # ("ssimulacra2-vship-5.1.1-gpu-v1"). They are accepted: the library
        # and the backend are what matter, not the version.
        or (value.startswith(f"{spec.key}-vship-") and value.endswith("-gpu-v1"))
        or (value.startswith(f"{spec.key}-libjxl-") and value.endswith("-cpu-v1"))
    )


def _load_metric_file(path: Path, spec: MetricRequestSpec):
    """Reads the score saved in `path` if it was saved for exactly `spec`;
    None otherwise, or if the file cannot be read."""
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
                # Only the frames the request covers: an XPSNR scored beside
                # VMAF on the GPU could be saved for every frame under a
                # subsampled request (see metric_results.as_requested).
                return sampled_every(
                    FrameMetricResult(spec.key, data["frame"], data["time"], data["values"], provenance),
                    spec.coverage.step if spec.coverage is not None else 1)
            if metadata["kind"] == "sequence":
                timeline = (data["frame"], data["time"], data["values"]) if "values" in data else (None, None, None)
                return SequenceMetricResult(spec.key, float(data["score"].item()), provenance, *timeline)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return None


def _auto_perceptual_candidates(directory: Path, spec: MetricRequestSpec):
    """Returns the saved scores that answer a perceptual request which does
    not fix where the metric is calculated, as (rank, path, the request with
    the score's implementation), GPU scores first."""
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
                # A GPU score is preferred; a CPU score (from libjxl) can
                # still be reused, as it is on a PC without a suitable GPU.
                rank = 0 if "-vship-" in compatibility else 1
                candidates.append((rank, path, concrete))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
    except OSError:
        return []
    return sorted(candidates, key=lambda item: item[0])


def load_other_parameters(directory: Path, spec: MetricRequestSpec) -> list[tuple[dict, float]]:
    """Returns (parameters, score) for each whole-video score saved for
    `spec`'s metric with other parameters but otherwise the same request:
    CVVDP's scores for other displays. These are never an answer for `spec`
    itself, because a score is valid only for the parameters it was made
    with."""
    wanted = spec.identity_dict()
    rest = {name: value for name, value in wanted.items() if name != "parameters"}
    found: list[tuple[dict, float]] = []
    try:
        paths = sorted(directory.glob(f"{spec.key}_*.npz"))
    except OSError:
        return []
    infos = _context_infos(directory) if paths else ()
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                request = metadata.get("request", {})
                if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION
                        or metadata.get("key") != spec.key or metadata.get("kind") != "sequence"
                        or {name: value for name, value in request.items() if name != "parameters"} != rest
                        or _canonical(request.get("parameters")) == _canonical(wanted["parameters"])
                        or (isinstance(metadata.get("provenance"), dict)
                            and _stale_vship_score(provenance_from_dict(metadata["provenance"]), infos))):
                    continue
                score = float(data["score"].item())
                parameters = dict(request["parameters"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
        if math.isfinite(score):
            found.append((parameters, score))
    return found


def _vmaf_model_ran(request: dict, provenance: dict) -> str | None:
    """Returns the model a saved VMAF score was calculated with, or None when
    that is not certain.

    An explicitly chosen (or bundled) model is the model. For Auto, a score
    calculated by this app records the model in its provenance, and a score
    migrated from the first cache format was saved under the model that ran.
    A score re-saved from an older result can carry neither (its key could
    hold the row's leftover default, "vmaf_v0.6.1", on a 4K-model score), so
    its model is unknown.
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


def _compared_size(directory: Path) -> tuple[int, int] | None:
    """The size this comparison's frames are compared at, worked out as the
    run works it out, from the sizes and black bars in its saved context.
    Auto picks a VMAF model by this size."""
    from vmaf_app.core.geometry import compared_dimensions, resample_analysis_dimensions

    try:
        context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
        source = video_info_from_dict(context["source_info"])
        distorted = video_info_from_dict(context["distorted_info"])
        source_crop = crop_from_dict(context.get("source_crop"))
        if context.get("resample_target"):
            return resample_analysis_dimensions(source, source_crop)
        return compared_dimensions(
            source, distorted, ScaleDirection(context["scale_direction"]),
            source_crop, crop_from_dict(context.get("distorted_crop")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return None


def _auto_model_for(directory: Path) -> str | None:
    """The VMAF v0.6.1 model Auto runs for this comparison."""
    size = _compared_size(directory)
    return None if size is None else model_for_resolution(*size)


def _vmaf_equivalents(directory: Path, spec: MetricRequestSpec) -> list[tuple[int, Path, MetricRequestSpec]]:
    """Returns saved VMAF scores of the same comparison, under another key,
    that were calculated with the same model, best match first.

    For a request on Auto: any other Auto score, or a score for the model
    Auto picks for this comparison. For a request with an explicit model: an
    Auto score that certainly ran that model. This way a row on Auto finds a
    score saved with "VMAF 4K v0.6.1" chosen, when Auto picks that model for
    the same comparison."""
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


def _old_v1_scores(directory: Path, spec: MetricRequestSpec) -> list[tuple[Path, MetricRequestSpec]]:
    """Returns VMAF v1 scores saved before VMAF v1 had a column of its own:
    they were saved under the key "vmaf", with a bundled v1 model as the
    VMAF model choice. They answer a VMAF v1 request for that model, or a
    request on Auto when it is the model Auto picks for this comparison."""
    wanted = spec.identity_dict()
    choice = wanted["parameters"].get("model_choice")
    rest = {name: value for name, value in wanted.items() if name not in ("key", "parameters")}
    auto: list[str | None] = []
    found = []
    try:
        paths = sorted(directory.glob("vmaf_*.npz"))
    except OSError:
        return []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
            request = metadata.get("request") or {}
            stored = request.get("parameters") or {}
            stored_choice = stored.get("model_choice")
            if (metadata.get("format_version") != METRIC_CACHE_FORMAT_VERSION or metadata.get("key") != "vmaf"
                    or not is_v1_choice(stored_choice)
                    or {name: value for name, value in request.items() if name not in ("key", "parameters")} != rest):
                continue
            if choice == AUTO_MODEL_CHOICE:
                if not auto:
                    size = _compared_size(directory)
                    auto.append(None if size is None else v1_model_for_resolution(*size))
                if stored_choice != auto[0]:
                    continue
            elif stored_choice != choice:
                continue
            found.append((path, replace(spec, key="vmaf", parameters=tuple(stored.items()))))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    return found


def _is_v1_spec(spec: MetricRequestSpec) -> bool:
    """Whether a request is for FFmpeg's VMAF v1."""
    return spec.key == "vmaf_v1" and spec.backend_id == "ffmpeg"


def _is_vmaf_spec(spec: MetricRequestSpec) -> bool:
    """Whether a request is for FFmpeg's VMAF (v0.6.1)."""
    return spec.key == "vmaf" and spec.backend_id == "ffmpeg"


def load_metric(directory: Path, spec: MetricRequestSpec, compute_backend: str = "gpu"):
    """Loads one saved metric score, or returns None.

    `compute_backend` is the user's GPU or CPU choice for a perceptual
    metric. "cpu" accepts only a libjxl CPU score, never a Vship GPU score,
    because the two can differ by a few points on the same frames. "gpu"
    prefers a Vship score and also accepts a CPU score, which is what the
    GPU choice produces on a PC without a supported GPU.

    A score known to be wrong is passed over (see _stale_vship_score): the
    next saved score answers instead, or the metric is calculated again."""
    infos: tuple[object, ...] | None = None

    def stale(result) -> bool:
        nonlocal infos
        if result.provenance.compute_backend != "gpu":
            return _stale_cpu_score(result.provenance)
        if infos is None:
            infos = _context_infos(directory)
        return _stale_vship_score(result.provenance, infos)

    if _is_auto_perceptual_spec(spec):
        for _rank, path, concrete in _auto_perceptual_candidates(directory, spec):
            if compute_backend == "cpu" and "-vship-" in concrete.implementation_compatibility_id:
                continue
            result = _load_metric_file(path, concrete)
            if result is not None and not stale(result):
                return result
        return None
    result = _load_metric_file(metric_path(directory, spec), spec)
    if result is not None and stale(result):
        return None
    if result is None and _is_vmaf_spec(spec):
        for _rank, path, concrete in _vmaf_equivalents(directory, spec):
            result = _load_metric_file(path, concrete)
            if result is not None:
                break
    if result is None and _is_v1_spec(spec):
        for path, concrete in _old_v1_scores(directory, spec):
            old = _load_metric_file(path, concrete)
            if isinstance(old, FrameMetricResult):
                return FrameMetricResult("vmaf_v1", old.frame, old.time, old.values, old.provenance)
    return result


def load_metrics(
    directory: Path, specs: tuple[MetricRequestSpec, ...],
    compute_backends: Mapping[str, str] | None = None,
) -> MetricResultSet:
    """Loads the saved scores of every request in `specs` that has one (see
    load_metric)."""
    backends = compute_backends or {}
    results = MetricResultSet()
    for spec in specs:
        result = load_metric(directory, spec, backends.get(spec.key, "gpu"))
        if result is not None:
            results.add(result)
    return results


def _context_from_result(result: ComparisonResult, label: str, recipe: ComparisonRecipe) -> dict:
    """The comparison's context.json: its files, crops, frame rate and recipe."""
    return {
        "format_version": METRIC_CACHE_FORMAT_VERSION, "label": label,
        "source": str(result.source), "distorted": str(result.distorted), "fps": result.fps,
        "source_crop": crop_to_dict(result.source_crop), "distorted_crop": crop_to_dict(result.distorted_crop),
        "source_info": video_info_to_dict(result.source_info), "distorted_info": video_info_to_dict(result.distorted_info),
        "scale_direction": result.scale_direction.value, "scale_algorithm": result.scale_algorithm,
        "resample_target": None if result.resample_target is None else {
            "width": result.resample_target.width, "label": result.resample_target.label,
        },
        "compared_frame_count": result.compared_frame_count,
        "model": result.model, "model_choice": result.model_choice,
        "model_v1": result.model_v1, "model_choice_v1": result.model_choice_v1,
        "recipe": recipe.identity_dict(),
    }


def store_result(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    result: ComparisonResult, label: str, specs: tuple[MetricRequestSpec, ...],
) -> Path:
    """Saves a run's result: its context.json and the score of each request
    in `specs`. Returns the comparison's directory."""
    directory = recipe_directory(base, source, distorted, recipe)
    _atomic_json(directory / "context.json", _context_from_result(result, label, recipe))
    for spec in specs:
        metric = result.metric(spec.key)
        if metric is None:
            continue
        if not _compatible(spec, metric.provenance):
            # A score saved under this request would answer it from then on,
            # because loading checks the saved request, not the score's
            # provenance (_load_metric_file).
            _log.warning("%s not saved: its score is %s's, the request asks for %s's", spec.key,
                         metric.provenance.implementation_compatibility_id, spec.implementation_compatibility_id)
            continue
        store_metric(directory, metric, spec)
    return directory


def _loaded_model(results: MetricResultSet, specs: tuple[MetricRequestSpec, ...], key: str,
                  saved: tuple[str, str | None]) -> tuple[str, str | None]:
    """The model and model choice of VMAF or VMAF v1 (`key`) in a result of
    saved scores. When it was loaded: the model its score was made with, as
    its file records it, and the choice of the request it answers, as
    calculating that request would give them. Otherwise `saved`, which are
    context.json's: those of the result saved last, which can be another
    display's CVVDP alone (model "") or VMAF with another model chosen."""
    model, choice = saved
    metric = results.get(key)
    spec = next((spec for spec in specs if spec.key == key), None)
    if metric is None or spec is None:
        return model, choice
    ran = metric.provenance.parameters.get("model")
    if isinstance(ran, str) and ran:
        model = ran
    return model, dict(spec.parameters).get("model_choice", choice)


def load_result(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    specs: tuple[MetricRequestSpec, ...], supplemental_specs: tuple[MetricRequestSpec, ...] = (),
    compute_backends: Mapping[str, str] | None = None,
):
    """Loads a saved comparison: (the result, its label), or None when
    nothing is saved for it. The result holds the saved scores of `specs`
    and of `supplemental_specs` (other metrics the comparison has saved)."""
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
    # Other saved metrics are looked up directly too, so a row shows every
    # compatible score saved for it. This is done only for metrics the
    # request does not ask for: a metric it asks for has its own saved score
    # or none. Otherwise a score made another way could stand in for the
    # requested one (an XPSNR of every frame for one of every n-th frame),
    # and the row's next run would never calculate the one asked for.
    requested = {spec.key for spec in specs}
    for spec in supplemental_specs:
        if spec.key not in requested and not results.has(spec.key):
            extra = load_metric(directory, spec, (compute_backends or {}).get(spec.key, "gpu"))
            if extra is not None:
                results.add(extra)
    if not results:
        return None

    from vmaf_app.core.metric_results import frame_scores_from_results

    # Every saved metric is kept, whatever its number of scores. Each metric
    # has its own frames: a subsampled one has every n-th frame, and FFmpeg
    # and Vship can end a frame apart. Its length is not checked against
    # context.json's compared_frame_count: every run rewrites context.json,
    # so such a check would keep whichever metric was saved last. The
    # directory's name (the files' identities and the comparison recipe) is
    # what makes a saved score valid for this comparison.
    frame_view = frame_scores_from_results(results)

    try:
        # A context saved before some of these fields were recorded lacks
        # them; the recipe's values stand in, because the comparison found
        # under the recipe is the same one. The scaling algorithm is the
        # result's own (it is not part of the comparison's identity), so the
        # saved one is kept.
        target = context.get("resample_target")
        asked = (*specs, *supplemental_specs)
        model, model_choice = _loaded_model(results, asked, "vmaf",
                                            (context.get("model") or "", context.get("model_choice")))
        model_v1, model_choice_v1 = _loaded_model(results, asked, "vmaf_v1",
                                                  (context.get("model_v1") or "", context.get("model_choice_v1")))
        result = ComparisonResult(
            source=Path(context["source"]), distorted=Path(context["distorted"]),
            frames=frame_view, fps=context["fps"], model=model,
            source_crop=crop_from_dict(context.get("source_crop")), distorted_crop=crop_from_dict(context.get("distorted_crop")),
            source_info=video_info_from_dict(context["source_info"]), distorted_info=video_info_from_dict(context["distorted_info"]),
            scale_direction=ScaleDirection(context.get("scale_direction", recipe.scale_direction.value)),
            scale_algorithm=context.get("scale_algorithm") or recipe.scale_algorithm,
            resample_target=ResampleTarget(**target) if target else recipe.resample_test,
            compared_frame_count=context.get("compared_frame_count", 0), model_choice=model_choice,
            model_v1=model_v1, model_choice_v1=model_choice_v1,
            metric_results=results,
        )
        return result, context.get("label") or Path(context["distorted"]).stem
    except (KeyError, TypeError, ValueError):
        return None


def cache_summary(base: Path) -> tuple[int, int]:
    """Returns (number of comparisons, bytes used) for the metric cache.

    The cache is walked with os.scandir, whose entries come with their sizes
    on Windows. Asking for each file's size separately takes about ten
    times as long (0.3 s for a thousand saved comparisons)."""
    comparisons = total = 0

    def walk(directory: str, depth: int) -> None:
        nonlocal comparisons, total
        try:
            entries = os.scandir(directory)
        except OSError:
            return
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        walk(entry.path, depth + 1)
                    elif entry.is_file():
                        total += entry.stat().st_size
                        if depth == 1 and entry.name == "context.json":
                            comparisons += 1
                except OSError:
                    continue

    walk(str(Path(base) / _V2_DIR), 0)
    return comparisons, total


def clear_metrics(
    base: Path, source: Path, distorted: Path, recipe: ComparisonRecipe,
    specs: tuple[MetricRequestSpec, ...],
) -> int:
    """Deletes the saved scores of the requests in `specs` for one
    comparison, and returns how many files were removed. Scores that would
    answer the same requests (see _vmaf_equivalents and _old_v1_scores) are
    deleted too, so they cannot come back after a Recalculate."""
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
        equivalents = (
            [path for _rank, path, _concrete in _vmaf_equivalents(directory, spec)] if _is_vmaf_spec(spec)
            else [path for path, _concrete in _old_v1_scores(directory, spec)] if _is_v1_spec(spec)
            else []
        )
        for path in equivalents:
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


def clear_all(base: Path) -> int:
    """Deletes the whole metric cache and returns how many comparisons it
    held (0 if it could not be deleted)."""
    root = Path(base) / _V2_DIR
    if not root.exists():
        return 0
    count = sum(1 for path in root.glob("*/context.json") if path.is_file())
    try:
        shutil.rmtree(root)
    except OSError:
        return 0
    return count
