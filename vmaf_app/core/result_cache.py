"""Persists computed VMAF results so identical inputs can be reused safely."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from vmaf_app.core.models import VmafRunResult
from vmaf_app.core.run_io import load_run, save_run

_dir_override: Path | None = None


def set_cache_dir_override(directory: Path | None) -> None:
    """Points the cache somewhere else, per the Settings tab. None restores
    the platform's app-data folder."""
    global _dir_override
    _dir_override = directory


def cache_dir() -> Path:
    d = _dir_override
    if d is None:
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        d = Path(base) / "results_cache"
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


def cache_key(source: Path, distorted: Path) -> str:
    raw = f"{_file_identity(source)}|{_file_identity(distorted)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_path(source: Path, distorted: Path) -> Path:
    return _cache_dir() / f"{cache_key(source, distorted)}.vmafrun.json"


def load_cached(source: Path, distorted: Path) -> tuple[VmafRunResult, str] | None:
    """Returns (result, label) if a cached run exists for this exact
    source+distorted file identity, else None."""
    path = _cache_path(source, distorted)
    if not path.exists():
        return None
    try:
        return load_run(path)
    except Exception:
        return None


def store(source: Path, distorted: Path, result: VmafRunResult, label: str) -> None:
    save_run(result, _cache_path(source, distorted), label=label)


def clear(source: Path, distorted: Path) -> None:
    _cache_path(source, distorted).unlink(missing_ok=True)
