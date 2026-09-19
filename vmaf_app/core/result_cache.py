"""Persists computed VMAF results so identical inputs can be reused safely."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vmaf_app.core.app_paths import user_data_dir
from vmaf_app.core.models import VmafOptions, VmafRunResult, clone_options
from vmaf_app.core.run_io import LEGACY_RESULT_SUFFIXES, RESULT_SUFFIX, load_run, save_run

_dir_override: Path | None = None


def default_cache_dir() -> Path:
    """Return the cache shared by every launcher for this OS user.

    QStandardPaths.AppDataLocation depends on the process' application/package
    identity.  During development that made the same app use one cache when
    launched from a packaged editor and another when launched from a terminal.
    The user's home directory is stable across those host processes.
    """
    return user_data_dir() / "results_cache"


def set_cache_dir_override(directory: Path | None) -> None:
    """Points the cache somewhere else, per the Settings tab. None restores
    the platform's app-data folder."""
    global _dir_override
    _dir_override = directory


def cache_dir() -> Path:
    d = _dir_override
    if d is None:
        d = default_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    adopt_legacy_names(d)
    return d


_adopted: set[Path] = set()


def adopt_legacy_names(directory: Path) -> int:
    """Renames cached runs saved under an older suffix to the current one.

    Once per folder per process: the scan is one directory listing, but
    cache_dir() is called on every lookup. Returns how many were renamed. A
    file whose new name already exists is left alone rather than replaced;
    clear_all removes both spellings, so nothing is ever stranded.
    """
    directory = Path(directory)
    if directory in _adopted:
        return 0
    _adopted.add(directory)
    renamed = 0
    for legacy in LEGACY_RESULT_SUFFIXES:
        for path in directory.glob(f"*{legacy}"):
            target = path.with_name(path.name[: -len(legacy)] + RESULT_SUFFIX)
            if target.exists():
                continue
            try:
                path.rename(target)
                renamed += 1
            except OSError:
                pass
    return renamed


def result_files(directory: Path) -> list[Path]:
    """Every saved run in `directory`, under the current suffix or an older one."""
    files: list[Path] = []
    for suffix in (RESULT_SUFFIX, *LEGACY_RESULT_SUFFIXES):
        files.extend(Path(directory).glob(f"*{suffix}"))
    return files


def _cache_dir() -> Path:
    return cache_dir()


def _file_identity(path: Path) -> str:
    path = Path(path).resolve()
    try:
        stat = path.stat()
        size = stat.st_size
        modified = stat.st_mtime_ns
    except OSError:
        size = -1
        modified = -1
    # The absolute path prevents two different files with the same basename
    # and size from sharing a result. mtime catches an in-place replacement
    # that happens to retain its exact byte length without hashing a movie.
    return f"{path}:{size}:{modified}"


def _options_identity(options: VmafOptions) -> str:
    values = asdict(options)
    if not options.compute_vmaf_neg:
        values.pop("compute_vmaf_neg")  # preserve existing cache identities
    # Preserve the exact legacy key for every existing VMAF-enabled run.
    if options.compute_vmaf:
        values.pop("compute_vmaf")
    else:
        # A disabled model cannot affect feature-only results.
        for name in ("model", "model_choice", "custom_model_path"):
            values.pop(name, None)
    # These only control how the same decoded frames are produced or how
    # libvmaf schedules its work. They do not change which frames/metrics
    # belong in the result, so changing performance hardware or thread count
    # must not force a feature-length video to be recalculated.
    for execution_only in ("gpu_decode", "gpu_vendor", "n_threads"):
        values.pop(execution_only, None)
    custom_model = options.custom_model_path
    if custom_model and options.compute_vmaf:
        values["custom_model_path"] = _file_identity(Path(custom_model))
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def cache_key(source: Path, distorted: Path, options: VmafOptions) -> str:
    raw = (
        f"{_file_identity(source)}|{_file_identity(distorted)}|"
        f"{_options_identity(options)}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(
    source: Path, distorted: Path, options: VmafOptions, directory: Path | None = None
) -> Path:
    base = directory if directory is not None else _cache_dir()
    return base / f"{cache_key(source, distorted, options)}{RESULT_SUFFIX}"


#: Every metric, and the options that request it. The identity of a cached
#: run includes which metrics it holds, so a run is only found by asking for
#: exactly the set it was computed with -- see _candidate_options.
_ALL_METRICS = ("vmaf", "psnr", "ssim", "xpsnr", "vmaf_neg")
_FEATURE_BY_METRIC = {"psnr": "name=psnr", "ssim": "name=float_ssim"}


def _options_for_metrics(options: VmafOptions, metrics: frozenset[str]) -> VmafOptions:
    """`options` as they would have been for exactly this set of metrics.

    Everything that decides which pictures get compared -- crop, scaling,
    duration, subsampling, model -- is carried over untouched. Only the
    choice of metrics differs, which is the whole point: a run measured the
    same frames whether or not it also recorded SSIM.
    """
    candidate = clone_options(options)
    candidate.compute_vmaf = "vmaf" in metrics
    candidate.compute_vmaf_neg = "vmaf_neg" in metrics
    candidate.compute_xpsnr = "xpsnr" in metrics
    # Rebuilt in a fixed order rather than filtered in place: extra_features
    # is a list, so its ORDER is part of the identity, and it is appended to
    # in whatever order the metrics were ticked.
    features = [_FEATURE_BY_METRIC[m] for m in ("psnr", "ssim") if m in metrics]
    features += [f for f in options.extra_features
                 if f not in _FEATURE_BY_METRIC.values() and f not in features]
    candidate.extra_features = features
    return candidate


def _candidate_options(options: VmafOptions) -> list[VmafOptions]:
    """Every cached run that could answer for `options`, best first.

    A run holding a SUPERSET of the requested metrics is a complete answer
    -- it measured everything wanted and more -- and the more it measured,
    the better: every score it holds is shown, whether or not it was asked
    for. So the fullest complete answer comes first, and the run recorded
    with exactly the requested set takes its place in that order rather
    than jumping the queue. It used to come first as a cheap exact hit,
    which hid a fuller run: VMAF NEG computed on top of an existing
    four-metric run was stored as a second, fuller file, and re-adding the
    video asked for the four, found the older file first, and never showed
    the NEG scores sitting in the other one.

    A run holding a SUBSET is a partial answer worth having, because the
    alternative is discarding a finished measurement of a feature-length
    video and recomputing it from nothing. The UI already distinguishes the
    two: a partial load leaves the missing metrics as tick boxes and is
    re-run to fill the gaps, rather than being reported as done.
    """
    wanted = frozenset(options.requested_metrics())
    if not wanted:
        return [options]
    ranked = []
    for mask in range(1, 1 << len(_ALL_METRICS)):
        metrics = frozenset(
            m for i, m in enumerate(_ALL_METRICS) if mask & (1 << i)
        )
        covered = len(metrics & wanted)
        if not covered:
            continue
        ranked.append((
            wanted <= metrics,  # complete answers before partial ones
            covered,            # then whichever supplies the most of them
            len(metrics),       # then whichever recorded the most in all
            sorted(metrics),    # stable, so the choice does not vary by run
            metrics,
        ))
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]), reverse=True)
    candidates = []
    for _complete, _covered, _count, _names, metrics in ranked:
        if metrics == wanted:
            # The caller's own options carry extra_features in whatever
            # order the metrics were ticked, and that order is part of the
            # identity. Tried alongside the canonical rebuild of the same
            # set, which finds a run stored under the other order.
            candidates.append(options)
        candidates.append(_options_for_metrics(options, metrics))
    # Both historical checkbox orders were persisted in cache identities.
    # Keep those keys valid rather than changing the identity format.
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if all(f in candidate.extra_features for f in _FEATURE_BY_METRIC.values()):
            alternate = clone_options(candidate)
            alternate.extra_features = [
                {"name=psnr": "name=float_ssim", "name=float_ssim": "name=psnr"}.get(f, f)
                for f in candidate.extra_features
            ]
            expanded.append(alternate)
    if options.compute_vmaf_neg:
        # Before NEG had its own metric, it used the VMAF model selector.
        # Try those exact old identities without rewriting or deleting files.
        for candidate in list(expanded):
            if not candidate.compute_vmaf_neg:
                continue
            legacy = clone_options(candidate)
            legacy.compute_vmaf = True
            legacy.compute_vmaf_neg = False
            legacy.model = legacy.model_choice = "version=vmaf_v0.6.1neg"
            expanded.append(legacy)
    return expanded


def load_cached(
    source: Path, distorted: Path, options: VmafOptions, directory: Path | None = None
) -> tuple[VmafRunResult, str] | None:
    """Returns (result, label) for this source+distorted pair, else None.

    Matches on file identity and on every option that changes what is
    measured. The set of metrics is the one exception: a run that recorded
    a different set of them looked at the same frames in the same way, so it
    is reused rather than thrown away -- see _candidate_options.
    """
    for candidate in _candidate_options(options):
        # XPSNR alone retains all frames; a libvmaf-backed run retains only
        # its sampled frames. Never substitute between those coverage modes.
        if options.n_subsample > 1 and (
            (options.requested_metrics() == ("xpsnr",))
            != (candidate.requested_metrics() == ("xpsnr",))
        ):
            continue
        path = _cache_path(source, distorted, candidate, directory)
        if not path.exists():
            continue
        try:
            return load_run(path)
        except Exception:
            continue
    return None


def store(
    source: Path, distorted: Path, result: VmafRunResult, label: str,
    options: VmafOptions, directory: Path | None = None,
) -> None:
    """`directory` pins where this write lands.

    Cache writes are queued and run later, on another thread. Resolving the
    folder inside the queued task reads whatever the setting says by then,
    so a store submitted while folder A was configured would land in folder
    B if the user changed the setting before it ran. Callers that queue must
    capture the directory at submit time and pass it here.
    """
    save_run(result, _cache_path(source, distorted, options, directory), label=label)


def clear(
    source: Path, distorted: Path, options: VmafOptions, directory: Path | None = None
) -> None:
    """Forgets every cached run for this pair that a lookup could return.

    Not just the entry matching `options` exactly: load_cached deliberately
    also reuses runs recorded with a different set of metrics (see
    _candidate_options), so clearing only the exact match left an older run
    of the same pair alive to be found again -- and "ignore cached results"
    has to mean it. Everything else about the run still has to match, so a
    result under a different crop, scale or duration is untouched.
    """
    for candidate in _candidate_options(options):
        _cache_path(source, distorted, candidate, directory).unlink(missing_ok=True)


def clear_all(directory: Path | None = None) -> int:
    """Removes only this app's cached result files and returns the count.

    `directory` matters more here than anywhere else: the confirmation
    dialog names a folder, and deleting a different one than the user was
    shown is not a thing to leave to timing.
    """
    base = directory if directory is not None else _cache_dir()
    removed = 0
    for path in result_files(base):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
