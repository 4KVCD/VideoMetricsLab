"""Frame-locked source/distorted playback decoded entirely by ffmpeg."""
from __future__ import annotations

import subprocess
import threading
from collections import deque

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from vmaf_app.core import proc as proc_util
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings, frame_input_path
from vmaf_app.core.gpu import GpuVendor, HwAccelPlan, plan_hwaccel
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.video_playback import (
    build_audio_command,
    build_video_pair_command,
    playback_dimensions,
)


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _PairDecodeWorker(QThread):
    frame_ready = Signal(int, int, bytes, int, int)
    decode_started = Signal(int, str)
    decode_failed = Signal(int, str)
    playback_ended = Signal(int)

    def __init__(
        self,
        generation: int,
        comparison: FrameComparison,
        start_frame: int,
        color_settings: PreviewColorSettings,
        output_size: tuple[int, int],
        hwaccel: HwAccelPlan,
        *,
        realtime: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.generation = generation
        self._comparison = comparison
        self._start_frame = start_frame
        self._color_settings = color_settings
        self._output_size = output_size
        self._preferred_hwaccel = hwaccel
        self._realtime = realtime
        self._cancelled = threading.Event()
        self._process = ProcessHandle()

    def cancel(self) -> None:
        self._cancelled.set()
        self._process.terminate()

    def pause(self) -> None:
        self._process.pause()

    def resume(self) -> None:
        self._process.resume()

    @staticmethod
    def _plans(preferred: HwAccelPlan) -> list[HwAccelPlan]:
        return [preferred, HwAccelPlan()] if preferred.uses_gpu else [preferred]

    def run(self) -> None:
        width, height = self._output_size
        frame_bytes = width * 2 * height * 3
        last_error = "ffmpeg returned no video frame."
        for plan in self._plans(self._preferred_hwaccel):
            if self._cancelled.is_set():
                return
            command = build_video_pair_command(
                self._comparison,
                self._start_frame,
                self._color_settings,
                plan,
                self._output_size,
                realtime=self._realtime,
            )
            process = proc_util.popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            self._process.attach(process.pid)
            stderr_tail: deque[bytes] = deque(maxlen=80)

            def drain_stderr(pipe=process.stderr, tail=stderr_tail) -> None:
                for line in iter(pipe.readline, b""):
                    tail.append(line)

            stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
            stderr_reader.start()
            frame_number = self._start_frame
            first = True
            try:
                while not self._cancelled.is_set():
                    payload = _read_exact(process.stdout, frame_bytes)
                    if len(payload) != frame_bytes:
                        break
                    if first:
                        first = False
                        self.decode_started.emit(
                            self.generation, f"GPU decode: {plan.describe()}"
                        )
                    self.frame_ready.emit(
                        self.generation, frame_number, payload, width, height
                    )
                    frame_number += 1
                    if not self._realtime:
                        break
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                self._process.detach()
                process.stdout.close()
                stderr_reader.join(timeout=2)
                process.stderr.close()

            if self._cancelled.is_set():
                return
            if not first:
                if self._realtime:
                    self.playback_ended.emit(self.generation)
                return
            detail = b"".join(stderr_tail).decode("utf-8", errors="replace").strip()
            if detail:
                last_error = detail[-2000:]
        self.decode_failed.emit(self.generation, last_error)


class _PairedFrameWidget(QWidget):
    """Paint one half of the latest indivisible source/distorted frame pair."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self._payload: bytes | None = None
        self._image = QImage()
        self._side_width = 0
        self._height = 0
        self._show_source = False

    def set_pair(self, payload: bytes, side_width: int, height: int) -> None:
        self._payload = payload
        self._side_width = side_width
        self._height = height
        self._image = QImage(
            payload, side_width * 2, height, side_width * 2 * 3,
            QImage.Format_RGB888,
        )
        self.update()

    def show_source(self, showing: bool) -> None:
        self._show_source = bool(showing)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171717"))
        if self._image.isNull() or self._side_width <= 0 or self._height <= 0:
            return
        scale = min(self.width() / self._side_width, self.height() / self._height)
        target_w = self._side_width * scale
        target_h = self._height * scale
        target = QRectF(
            (self.width() - target_w) / 2,
            (self.height() - target_h) / 2,
            target_w,
            target_h,
        )
        source_x = 0 if self._show_source else self._side_width
        source = QRectF(source_x, 0, self._side_width, self._height)
        painter.drawImage(target, self._image, source)


class VideoCompareView(QWidget):
    """Play a paired ffmpeg stream with instant, frame-exact A/B switching."""

    position_changed = Signal(int)
    playing_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #171717;")
        self.video = _PairedFrameWidget(self)
        # There is now one paired ffmpeg surface; retain these focus aliases
        # for code that treated the old Qt surfaces as keyboard targets.
        self.source_video = self.video
        self.distorted_video = self.video
        self._comparison: FrameComparison | None = None
        self._color_settings = PreviewColorSettings()
        self._worker: _PairDecodeWorker | None = None
        self._retired_workers: set[_PairDecodeWorker] = set()
        self._generation = 0
        self._frame = 0
        self._wanted_playing = False
        self._is_playing = False
        self._showing_source = False
        self._audio_enabled = True
        self._audio_process: subprocess.Popen | None = None
        self._audio_handle: ProcessHandle | None = None

    def resizeEvent(self, event) -> None:
        self.video.setGeometry(self.rect())
        super().resizeEvent(event)

    @staticmethod
    def can_play(comparison: FrameComparison) -> tuple[bool, str]:
        if comparison.resample_target is not None:
            return (
                False,
                "Video playback is unavailable for synthetic resolution tests; "
                "use Still frame to inspect their processed output.",
            )
        source = frame_input_path(comparison, "source")
        distorted = frame_input_path(comparison, "distorted")
        missing = next((path for path in (source, distorted) if not path.is_file()), None)
        if missing is not None:
            return False, f"Video file is missing: {missing}"
        return True, ""

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def playback_requested(self) -> bool:
        return self._wanted_playing

    @property
    def position(self) -> int:
        if self._comparison is None or self._comparison.fps <= 0:
            return 0
        return round(self._frame / self._comparison.fps * 1000)

    def live_workers(self) -> list[QThread]:
        workers = list(self._retired_workers)
        if self._worker is not None:
            workers.append(self._worker)
        return [worker for worker in workers if worker.isRunning()]

    def load(
        self,
        comparison: FrameComparison,
        position_ms: int,
        *,
        playing: bool = False,
        color_settings: PreviewColorSettings | None = None,
    ) -> bool:
        available, reason = self.can_play(comparison)
        if not available:
            self.set_playing(False)
            self.status_changed.emit(reason)
            return False
        self._comparison = comparison
        self._color_settings = color_settings or PreviewColorSettings()
        self._frame = max(0, round(position_ms / 1000 * comparison.fps))
        self._wanted_playing = bool(playing)
        self._restart_decoder(realtime=bool(playing))
        return True

    def clear(self) -> None:
        self._wanted_playing = False
        self._is_playing = False
        self._comparison = None
        self._stop_decoder()
        self._stop_audio()
        self.playing_changed.emit(False)

    def set_position(self, position_ms: int) -> None:
        comparison = self._comparison
        if comparison is None or comparison.fps <= 0:
            return
        self._frame = max(0, round(position_ms / 1000 * comparison.fps))
        self._restart_decoder(realtime=self._wanted_playing)

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        self._wanted_playing = playing
        worker = self._worker
        if playing:
            if worker is not None and worker.isRunning() and not self._is_playing:
                worker.resume()
                if self._audio_handle is not None:
                    self._audio_handle.resume()
                self._is_playing = True
                self.playing_changed.emit(True)
            elif not self._is_playing and self._comparison is not None:
                self._restart_decoder(realtime=True)
        else:
            if worker is not None and worker.isRunning():
                worker.pause()
            if self._audio_handle is not None:
                self._audio_handle.pause()
            self._is_playing = False
            self.playing_changed.emit(False)

    def show_source(self, showing: bool) -> None:
        self._showing_source = bool(showing)
        self.video.show_source(showing)

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = bool(enabled)
        if not enabled:
            self._stop_audio()
        elif self._is_playing:
            self._start_audio(self._frame)

    def set_color_settings(self, settings: PreviewColorSettings) -> None:
        if settings == self._color_settings:
            return
        self._color_settings = settings
        if self._comparison is not None:
            self._restart_decoder(realtime=self._wanted_playing)

    def _restart_decoder(self, *, realtime: bool) -> None:
        comparison = self._comparison
        if comparison is None:
            return
        self._stop_decoder()
        self._stop_audio()
        self._generation += 1
        generation = self._generation
        output_size = playback_dimensions(comparison)
        plan = plan_hwaccel(
            GpuVendor.AUTO,
            comparison.source_info.codec_name,
            comparison.distorted_info.codec_name,
        )
        worker = _PairDecodeWorker(
            generation,
            comparison,
            self._frame,
            self._color_settings,
            output_size,
            plan,
            realtime=realtime,
            parent=self,
        )
        worker.frame_ready.connect(self._on_frame_ready)
        worker.decode_started.connect(self._on_decode_started)
        worker.decode_failed.connect(self._on_decode_failed)
        worker.playback_ended.connect(self._on_playback_ended)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._worker = worker
        self._is_playing = realtime
        self.status_changed.emit("Opening with ffmpeg…")
        self.playing_changed.emit(realtime)
        worker.start()

    def _stop_decoder(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            self._retired_workers.add(worker)

    def _on_worker_finished(self, worker: _PairDecodeWorker) -> None:
        self._retired_workers.discard(worker)
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def _on_frame_ready(
        self, generation: int, frame: int, payload: bytes, width: int, height: int
    ) -> None:
        if generation != self._generation:
            return
        self._frame = frame
        self.video.set_pair(payload, width, height)
        comparison = self._comparison
        if comparison is not None and comparison.fps > 0:
            self.position_changed.emit(round(frame / comparison.fps * 1000))

    def _on_decode_started(self, generation: int, detail: str) -> None:
        if generation != self._generation:
            return
        state = "Playing" if self._wanted_playing else "Paused"
        self.status_changed.emit(f"{state} · ffmpeg · {detail}")
        if self._wanted_playing:
            self._start_audio(self._frame)

    def _on_decode_failed(self, generation: int, error: str) -> None:
        if generation != self._generation:
            return
        self._wanted_playing = False
        self._is_playing = False
        self._stop_audio()
        self.playing_changed.emit(False)
        self.status_changed.emit(f"Could not decode comparison with ffmpeg: {error}")

    def _on_playback_ended(self, generation: int) -> None:
        if generation != self._generation:
            return
        self._wanted_playing = False
        self._is_playing = False
        self._stop_audio()
        self.playing_changed.emit(False)
        self.status_changed.emit("Playback ended")

    def _start_audio(self, frame: int) -> None:
        if not self._audio_enabled or self._comparison is None:
            return
        self._stop_audio()
        command = build_audio_command(self._comparison, frame)
        if command is None:
            return
        process = proc_util.popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        handle = ProcessHandle()
        handle.attach(process.pid)
        self._audio_process = process
        self._audio_handle = handle

    def _stop_audio(self) -> None:
        process = self._audio_process
        handle = self._audio_handle
        self._audio_process = None
        self._audio_handle = None
        if handle is not None:
            handle.terminate()
            handle.detach()
        if process is not None and process.poll() is None:
            process.terminate()
