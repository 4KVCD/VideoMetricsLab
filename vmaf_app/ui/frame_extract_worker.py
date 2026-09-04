"""Background decoding for the Frame Compare tab."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from vmaf_app.core.frame_extract import (
    FrameExtractCancelledError,
    extract_frame_png,
)
from vmaf_app.core.models import VmafRunResult
from vmaf_app.core.process_control import ProcessHandle


class FrameExtractWorker(QThread):
    # generation, side, encoded PNG
    frame_ready = Signal(int, str, bytes)
    # generation, side, readable error
    frame_failed = Signal(int, str, str)

    def __init__(
        self,
        generation: int,
        result: VmafRunResult,
        frame: int,
        sides: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.generation = generation
        self._result = result
        self._frame = frame
        self._sides = list(sides)
        self._cancelled = False
        self._process = ProcessHandle()

    def cancel(self) -> None:
        self._cancelled = True
        self._process.terminate()

    def run(self) -> None:
        for side in self._sides:
            if self._cancelled:
                return
            try:
                png = extract_frame_png(
                    self._result, side, self._frame,
                    process_handle=self._process,
                )
            except FrameExtractCancelledError:
                return
            except Exception as exc:
                if not self._cancelled:
                    self.frame_failed.emit(self.generation, side, str(exc))
                continue
            if not self._cancelled:
                self.frame_ready.emit(self.generation, side, png)
