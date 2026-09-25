"""Cache facade for backend-neutral per-metric analysis results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vmaf_app.core import metric_cache
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.app_paths import user_data_dir
from vmaf_app.core.cvvdp import CvvdpSettings
from vmaf_app.core.models import ComparisonResult

_dir_override: Path | None = None


def default_cache_dir() -> Path:
    """Return the cache shared by every launcher for this OS user."""
    return user_data_dir() / "results_cache"


def set_cache_dir_override(directory: Path | None) -> None:
    """Points the cache somewhere else, per the Settings tab. None restores
    the platform's app-data folder."""
    global _dir_override
    _dir_override = directory


def cache_dir() -> Path:
    directory = _dir_override if _dir_override is not None else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cache_dir() -> Path:
    return cache_dir()


def cache_key(source: Path, distorted: Path, request: AnalysisRequest) -> str:
    """Stable token for one row's scientific request.

    The UI uses this only to reject a cache answer that finishes after the
    source, file contents, or request changed. Execution preferences therefore
    stay out of the token, exactly as they do in the metric cache itself.
    """
    raw = {
        "recipe": metric_cache.recipe_hash(source, distorted, request.recipe),
        "metrics": [spec.identity_dict() for spec in request.metrics],
    }
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_summary(directory: Path | None = None) -> tuple[int, int]:
    """Return ``(saved comparisons, bytes)`` for the active metric cache."""
    base = directory if directory is not None else _cache_dir()
    return metric_cache.cache_summary(base)


def load_cached(
    source: Path,
    distorted: Path,
    request: AnalysisRequest,
    directory: Path | None = None,
    supplemental_specs: tuple[MetricRequestSpec, ...] = (),
) -> tuple[ComparisonResult, str] | None:
    """Return every compatible cached metric available for this request.

    A partial result is useful: the UI can display finished measurements
    immediately and leave missing metrics to be calculated. Supplemental specs
    are a presentation policy supplied by the caller, not part of scientific
    request identity.
    """
    base = directory if directory is not None else _cache_dir()
    # The GPU/CPU choice is not part of cache identity, but it does decide
    # which saved implementation of a perceptual metric may answer: choosing
    # CPU must never show a GPU score (see metric_cache.load_metric).
    return metric_cache.load_result(
        base, source, distorted, request.recipe, request.metrics, supplemental_specs,
        compute_backends=dict(request.execution.perceptual_backends),
    )


def other_cvvdp_scores(
    source: Path,
    distorted: Path,
    request: AnalysisRequest,
    directory: Path | None = None,
    supplemental_specs: tuple[MetricRequestSpec, ...] = (),
) -> tuple[tuple[tuple, ...], list[tuple[CvvdpSettings, float]]]:
    """CVVDP scores saved for this comparison with other display settings,
    as (settings, JOD), and the parameters of the CVVDP request they are
    "other" to; nothing if the request has no CVVDP."""
    spec = next((s for s in (*request.metrics, *supplemental_specs) if s.key == "cvvdp"), None)
    if spec is None:
        return (), []
    base = directory if directory is not None else _cache_dir()
    found = []
    for parameters, score in metric_cache.load_other_parameters(
            metric_cache.recipe_directory(base, source, distorted, request.recipe), spec):
        try:
            found.append((CvvdpSettings.from_spec_parameters(parameters), score))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(spec.parameters), found


def store(
    source: Path,
    distorted: Path,
    result: ComparisonResult,
    label: str,
    request: AnalysisRequest,
    directory: Path | None = None,
) -> None:
    """Store a completed result in the per-metric cache."""
    base = directory if directory is not None else _cache_dir()
    metric_cache.store_result(
        base, source, distorted, request.recipe, result, label, request.metrics
    )


def clear(
    source: Path,
    distorted: Path,
    request: AnalysisRequest,
    directory: Path | None = None,
    supplemental_specs: tuple[MetricRequestSpec, ...] = (),
) -> None:
    """Forget cached metrics this request could load without touching other recipes."""
    base = directory if directory is not None else _cache_dir()
    specs_by_identity = {
        metric_cache.metric_identity_hash(spec): spec
        for spec in (*request.metrics, *supplemental_specs)
    }
    metric_cache.clear_metrics(
        base, source, distorted, request.recipe, tuple(specs_by_identity.values())
    )


def clear_all(directory: Path | None = None) -> int:
    """Remove the active metric cache and return the number of comparisons."""
    base = directory if directory is not None else _cache_dir()
    return metric_cache.clear_all(base)
