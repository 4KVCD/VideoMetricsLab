"""Probes newly added videos, and loads their cached results, off the UI
thread.

Both steps are slow enough to be felt: ffprobe is a subprocess launch
(~70ms for a small file, more for a large one on a slow disk), and a cached
result for a feature-length run is a ~9MB JSON parse (~140ms). Doing eight
videos inline froze the window for a couple of seconds with no feedback.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core import result_cache
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import VideoInfo, VmafOptions, VmafRunResult


class ProbeWorker(QThread):
    """Emits one signal per video, in the order they were given."""

    # (path, VideoInfo or None, error message or "")
    probed = Signal(object, object, str)
    # (path, VmafRunResult, label, cache key) -- only for videos with a
    # cached result. The key lets the UI reject a result if the row's options
    # changed while this background read was in flight.
    cached_found = Signal(object, object, str, str)
    finished_all = Signal()

    def __init__(self, paths: list[Path], source: Path | None, use_cache: bool,
                 cache_options: dict[Path, VmafOptions], probe_media: bool = True,
                 parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._source = source
        self._use_cache = use_cache
        self._cache_options = cache_options
        # False when only the source changed: the distorted files are the
        # same, so only their cached results need re-checking.
        self._probe_media = probe_media
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            for path in self._paths:
                if self._cancelled:
                    break
                if self._probe_media:
                    try:
                        info: VideoInfo | None = probe_video(path)
                        error = ""
                    except Exception as e:
                        # This is a background boundary: even an unexpected
                        # probe failure must become a row error rather than
                        # silently killing the QThread before cleanup.
                        info, error = None, str(e)
                    self.probed.emit(path, info, error)

                if self._cancelled or not self._use_cache or self._source is None:
                    continue
                # A miss is the normal case and must not be reported as a
                # failure; the row simply stays unscored until it is run.
                options = self._cache_options[path]
                # Capture identity before parsing the file. If either video
                # is replaced during a long read, the UI will reject this
                # token rather than accepting old scores under the new file.
                key = result_cache.cache_key(self._source, path, options)
                cached = result_cache.load_cached(self._source, path, options)
                if cached is not None:
                    result, label = cached
                    self.cached_found.emit(path, result, label, key)
        finally:
            self.finished_all.emit()


__all__ = ["ProbeWorker", "VmafRunResult"]
