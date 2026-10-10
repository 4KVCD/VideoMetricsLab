"""Background decoding for the Frame Compare tab."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from vmaf_app.core.frame_extract import (
    FrameComparison,
    FrameExtractCancelledError,
    PreviewColorSettings,
    extract_frame_png,
)
from vmaf_app.core.process_control import ProcessHandle


class FrameExtractWorker(QThread):
    """Extracts one frame of each side for Frame Compare, on a thread of its own."""

    # generation, side, the frame decoded here (a QImage, null if
    # unreadable): decoded by the window's thread, a 4K frame held it 64-104 ms
    frame_ready = Signal(int, str, object)
    # generation, side, readable error
    frame_failed = Signal(int, str, str)

    def __init__(
        self,
        generation: int,
        comparison: FrameComparison,
        frame: int,
        sides: list[str],
        color_settings: PreviewColorSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.generation = generation
        self._comparison = comparison
        self._frame = frame
        self._sides = list(sides)
        self._color_settings = color_settings or PreviewColorSettings()
        self._cancelled = False
        self._process = ProcessHandle()

    def cancel(self) -> None:
        """Stops the extraction, ending FFmpeg."""
        self._cancelled = True
        self._process.terminate()

    def run(self) -> None:
        """Extracts each side's frame in turn (QThread.run)."""
        for side in self._sides:
            if self._cancelled:
                return
            try:
                png = extract_frame_png(
                    self._comparison, side, self._frame,
                    process_handle=self._process,
                    color_settings=self._color_settings,
                )
            except FrameExtractCancelledError:
                return
            except Exception as exc:
                if not self._cancelled:
                    self.frame_failed.emit(self.generation, side, str(exc))
                continue
            if not self._cancelled:
                self.frame_ready.emit(self.generation, side, QImage.fromData(png, "PNG"))
