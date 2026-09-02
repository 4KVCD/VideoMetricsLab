"""Persists computed VMAF results to disk, keyed by the source and distorted
files' name+size, so relaunching the app with the same files doesn't require
recomputing.

Keying is deliberately just filename+size (not a full content hash, and not
the options that were used) -- fast to check, and matches what was asked
for. "Recompute" is always available as an escape hatch if the files were
re-encoded to the same name/size, or the options used should change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from vmaf_app.core.models import VmafRunResult
from vmaf_app.core.run_io import load_run, save_run


def _cache_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    d = Path(base) / "results_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_identity(path: Path) -> str:
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = -1
    return f"{Path(path).name}:{size}"


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
    except Exception:  # noqa: BLE001 - a corrupt/unreadable cache entry just means a cache miss
        return None


def store(source: Path, distorted: Path, result: VmafRunResult, label: str) -> None:
    save_run(result, _cache_path(source, distorted), label=label)


def clear(source: Path, distorted: Path) -> None:
    _cache_path(source, distorted).unlink(missing_ok=True)
