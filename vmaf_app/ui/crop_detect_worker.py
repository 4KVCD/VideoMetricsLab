"""Background black-bar detection for the Video Compare tab."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.crop_detect import CropDetectCancelled, detect_crop, detect_pair
from vmaf_app.core.models import CropBox, VideoInfo


class CropDetectWorker(QThread):
    """Resolve a pair's auto-crops without blocking the Qt event loop."""

    ready = Signal(object, object)

    def __init__(self, source: VideoInfo, distorted: VideoInfo, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._distorted = distorted
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _detect(self, info: VideoInfo) -> CropBox | None:
        try:
            return detect_crop(info, cancel_event=self._cancel)
        except CropDetectCancelled:
            raise
        except Exception:
            # Preview remains usable when a file has no stable bars or a
            # decoder cannot be sampled. The scored run remains authoritative.
            return None

    def run(self) -> None:
        try:
            if self._distorted.path == self._source.path:
                source = self._detect(self._source)
                distorted = source
            else:
                source, distorted = detect_pair(
                    lambda: self._detect(self._source),
                    lambda: self._detect(self._distorted),
                )
            if not self._cancel.is_set():
                self.ready.emit(source, distorted)
        except CropDetectCancelled:
            return
