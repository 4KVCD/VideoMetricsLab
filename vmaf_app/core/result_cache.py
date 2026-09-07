"""Persists computed VMAF results so identical inputs can be reused safely."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vmaf_app.core.app_paths import user_data_dir
from vmaf_app.core.models import VmafOptions, VmafRunResult
from vmaf_app.core.run_io import load_run, save_run

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
    return d


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
    return base / f"{cache_key(source, distorted, options)}.vmafrun.json"


def load_cached(
    source: Path, distorted: Path, options: VmafOptions, directory: Path | None = None
) -> tuple[VmafRunResult, str] | None:
    """Returns (result, label) if a cached run exists for this exact
    source+distorted file identity, else None."""
    path = _cache_path(source, distorted, options, directory)
    if not path.exists():
        return None
    try:
        return load_run(path)
    except Exception:
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
    _cache_path(source, distorted, options, directory).unlink(missing_ok=True)


def clear_all(directory: Path | None = None) -> int:
    """Removes only this app's cached result files and returns the count.

    `directory` matters more here than anywhere else: the confirmation
    dialog names a folder, and deleting a different one than the user was
    shown is not a thing to leave to timing.
    """
    base = directory if directory is not None else _cache_dir()
    removed = 0
    for path in base.glob("*.vmafrun.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
