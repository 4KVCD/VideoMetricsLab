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
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.ffprobe import ProbeCancelled, probe_video
from vmaf_app.core.models import ComparisonResult, VideoInfo
from vmaf_app.core.process_control import ProcessHandle


class ProbeWorker(QThread):
    """Emits one signal per video, in the order they were given."""

    # (path, VideoInfo or None, error message or "")
    probed = Signal(object, object, str)
    # (path, ComparisonResult, label, cache key) -- only for videos with a
    # cached result. The key lets the UI reject a result if the row's options
    # changed while this background read was in flight.
    cached_found = Signal(object, object, str, str)
    # (path, CVVDP scores saved for other displays as [(CvvdpSettings, JOD)],
    # the CVVDP request parameters they were looked up for) -- for every
    # video whose lookup includes CVVDP, so an empty list clears old ones.
    other_cvvdp_found = Signal(object, object, object)
    finished_all = Signal()

    def __init__(self, paths: list[Path], source: Path | None, use_cache: bool,
                 cache_requests: dict[Path, AnalysisRequest],
                 cache_paths: dict[Path, Path] | None = None,
                 cache_supplemental: dict[Path, tuple[MetricRequestSpec, ...]] | None = None,
                 probe_media: bool = True, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._source = source
        self._use_cache = use_cache
        self._cache_requests = cache_requests
        self._cache_supplemental = cache_supplemental or {}
        # Row path -> the real file its cache identity comes from. They
        # differ only for synthetic rows (a "test both" companion, a
        # resolution test), whose own path does not exist on disk.
        self._cache_paths = cache_paths or {}
        # False when only the source changed: the distorted files are the
        # same, so only their cached results need re-checking.
        self._probe_media = probe_media
        self._cancelled = False
        # ffprobe on a large file over a slow or network disk takes many
        # seconds. Setting a flag cannot interrupt a probe already blocked
        # inside that call, so cancellation goes through the handle and
        # terminates the subprocess itself.
        self._process = ProcessHandle()

    def cancel(self) -> None:
        self._cancelled = True
        self._process.terminate()

    def run(self) -> None:
        try:
            for path in self._paths:
                if self._cancelled:
                    break
                if self._probe_media:
                    try:
                        info: VideoInfo | None = probe_video(
                            path, process_handle=self._process
                        )
                        error = ""
                    except ProbeCancelled:
                        break
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
                request = self._cache_requests[path]
                identity = self._cache_paths.get(path, path)
                # Capture identity before parsing the file. If either video
                # is replaced during a long read, the UI will reject this
                # token rather than accepting old scores under the new file.
                key = result_cache.cache_key(self._source, identity, request)
                cached = result_cache.load_cached(
                    self._source, identity, request,
                    supplemental_specs=self._cache_supplemental.get(path, ()),
                )
                if cached is not None:
                    result, label = cached
                    self.cached_found.emit(path, result, label, key)
                try:
                    parameters, others = result_cache.other_cvvdp_scores(
                        self._source, identity, request,
                        supplemental_specs=self._cache_supplemental.get(path, ()),
                    )
                except Exception:
                    # Only a hint for a tooltip: a damaged cache file must
                    # not end this loop and leave the videos after this one
                    # without their saved scores.
                    continue
                if parameters:
                    self.other_cvvdp_found.emit(path, others, parameters)
        finally:
            self.finished_all.emit()


__all__ = ["ComparisonResult", "ProbeWorker"]
