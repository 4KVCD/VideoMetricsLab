"""Background ffprobe packet scans for the Bitrate Viewer tab."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.bitrate import BitrateCancelled, analyze_video_bitrate
from vmaf_app.core.ffprobe import ProbeCancelled, probe_video
from vmaf_app.core.models import VideoInfo
from vmaf_app.core.process_control import ProcessHandle


class BitrateWorker(QThread):
    file_started = Signal(object)                 # path
    progress = Signal(object, int, int)           # path, packets, estimate
    analyzed = Signal(object, object, object)     # path, VideoInfo, BitrateData
    failed = Signal(object, str)                  # path, readable error

    def __init__(
        self,
        jobs: list[tuple[Path, VideoInfo | None]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._jobs = list(jobs)
        self._cancelled = False
        self._process = ProcessHandle()

    def cancel(self) -> None:
        self._cancelled = True
        self._process.terminate()

    def run(self) -> None:
        for path, known_info in self._jobs:
            if self._cancelled:
                return
            self.file_started.emit(path)
            try:
                info = known_info or probe_video(path, process_handle=self._process)
                if self._cancelled:
                    return
                data = analyze_video_bitrate(
                    info,
                    process_handle=self._process,
                    on_progress=lambda done, total, p=path: self.progress.emit(
                        p, done, total
                    ),
                )
            except (ProbeCancelled, BitrateCancelled):
                return
            except Exception as exc:
                if not self._cancelled:
                    self.failed.emit(path, str(exc))
                continue
            if not self._cancelled:
                self.analyzed.emit(path, info, data)
