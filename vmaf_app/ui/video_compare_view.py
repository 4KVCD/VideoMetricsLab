"""Frame-locked source/distorted playback with native GPU presentation."""
from __future__ import annotations

import subprocess
import threading
from collections import deque

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QWidget

from vmaf_app.core import proc as proc_util
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings, frame_input_path
from vmaf_app.core.gpu import GpuVendor, HwAccelPlan, plan_hwaccel
from vmaf_app.core.gstreamer_playback import (
    GstComparePipeline,
    GStreamerPlaybackError,
    uses_native_gstreamer,
)
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
    frame_available = Signal(int)
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
        self._frame_lock = threading.Lock()
        self._latest_frame: tuple[int, bytes, int, int] | None = None
        self._notification_pending = False

    def take_latest_frame(self) -> tuple[int, bytes, int, int] | None:
        """Return the newest decoded frame without copying it through a Qt signal.

        PySide converts a ``bytes`` signal argument to and from ``QByteArray``.
        For a UHD comparison preview that made the GUI thread copy roughly 9 MB
        per frame, while also allowing an unbounded queue of stale frames.  The
        worker now owns one replaceable slot and signals only that data is ready.
        """
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
            self._notification_pending = False
            return frame

    def _publish_frame(
        self, frame_number: int, payload: bytes, width: int, height: int
    ) -> None:
        notify = False
        with self._frame_lock:
            self._latest_frame = (frame_number, payload, width, height)
            if not self._notification_pending:
                self._notification_pending = True
                notify = True
        if notify:
            self.frame_available.emit(self.generation)

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
                # Let BufferedReader assemble each raw frame in C.  An
                # unbuffered Windows pipe often returns only a few KiB per
                # read, which made Python loop thousands of times per UHD
                # preview frame and starved the GUI thread of the GIL.
                bufsize=frame_bytes,
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
                    self._publish_frame(frame_number, payload, width, height)
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
    """Native D3D11 target, with a QImage surface for FFmpeg fallback."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors)
        self._payload: bytes | None = None
        self._image = QImage()
        self._side_width = 0
        self._height = 0
        self._show_source = False
        self._native_playback = False

    def set_native_playback(self, enabled: bool) -> None:
        self._native_playback = bool(enabled)
        if enabled:
            self._payload = None
            self._image = QImage()
        self.update()

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
        # d3d11videosink owns this child HWND while native playback is active.
        # Painting it from Qt would erase or flash over the swapchain.
        if self._native_playback:
            return
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
    """Play one synchronized source/distorted stream with instant A/B switching."""

    position_changed = Signal(int)
    playing_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #171717;")
        self.source_video = _PairedFrameWidget(self)
        self.source_video.show_source(True)
        self.distorted_video = _PairedFrameWidget(self)
        self.distorted_video.show_source(False)
        # FFmpeg fallback still paints its packed pair on this surface.  Keep
        # the public alias used by the panel and older tests.
        self.video = self.distorted_video
        self.source_video.hide()
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
        self._gst: GstComparePipeline | None = None
        self._gst_timer = QTimer(self)
        self._gst_timer.setInterval(40)
        self._gst_timer.timeout.connect(self._poll_gstreamer)

    def resizeEvent(self, event) -> None:
        self.source_video.setGeometry(self.rect())
        self.distorted_video.setGeometry(self.rect())
        super().resizeEvent(event)

    @staticmethod
    def can_play(comparison: FrameComparison) -> tuple[bool, str]:
        if comparison.resample_target is not None:
            return (
                False,
                "Video playback is unavailable for synthetic resolution tests; "
                "their processed frames are displayed automatically.",
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
        if self._gst is not None:
            try:
                self._gst.seek(position_ms)
            except Exception as exc:
                self._fall_back_from_gstreamer(str(exc))
            return
        self._restart_decoder(realtime=self._wanted_playing)

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        self._wanted_playing = playing
        if self._gst is not None:
            self._gst.set_playing(playing)
            self._is_playing = playing
            self.playing_changed.emit(playing)
            self.status_changed.emit(
                ("Playing" if playing else "Paused") + " · GStreamer · D3D11"
            )
            return
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
        if self._gst is not None:
            self._gst.set_show_source(showing)
            if showing:
                self.source_video.raise_()
            else:
                self.distorted_video.raise_()

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = bool(enabled)
        if self._gst is not None:
            self._gst.set_audio_enabled(enabled)
            return
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
        native, reason = uses_native_gstreamer(comparison, self._color_settings)
        if native:
            try:
                self._start_gstreamer(realtime)
                return
            except Exception as exc:
                reason = str(exc)
                self._stop_decoder()
        self._start_ffmpeg(realtime, reason)

    def _start_gstreamer(self, realtime: bool) -> None:
        comparison = self._comparison
        if comparison is None:
            return
        self.source_video.show()
        self.distorted_video.show()
        # winId() forces two real child HWNDs. Both GPU swapchains remain live
        # on one GStreamer clock; S only changes their Z-order.
        pipeline = GstComparePipeline(
            comparison,
            int(self.source_video.winId()),
            int(self.distorted_video.winId()),
            self._color_settings,
            show_source=self._showing_source,
            audio_enabled=self._audio_enabled,
        )
        self._gst = pipeline
        self.source_video.set_native_playback(True)
        self.distorted_video.set_native_playback(True)
        if self._showing_source:
            self.source_video.raise_()
        else:
            self.distorted_video.raise_()
        try:
            pipeline.start(self.position, realtime)
        except Exception as exc:
            # Construction can succeed while the asynchronous D3D11 state
            # change or initial seek cannot.  Never leave a half-started
            # swapchain owning the child HWND before FFmpeg takes it back.
            self._gst = None
            pipeline.stop()
            self.source_video.set_native_playback(False)
            self.distorted_video.set_native_playback(False)
            self.source_video.hide()
            if isinstance(exc, GStreamerPlaybackError):
                raise
            raise GStreamerPlaybackError(
                f"GStreamer could not start native playback: {exc}"
            ) from exc
        self._is_playing = realtime
        self.status_changed.emit("Opening with GStreamer/D3D11…")
        self.playing_changed.emit(realtime)
        self._gst_timer.start()

    def _display_pixel_size(self) -> tuple[int, int] | None:
        handle = self.window().windowHandle()
        screen = handle.screen() if handle is not None else QGuiApplication.primaryScreen()
        if screen is None:
            return None
        geometry = screen.geometry()
        ratio = screen.devicePixelRatio()
        return round(geometry.width() * ratio), round(geometry.height() * ratio)

    def _start_ffmpeg(self, realtime: bool, reason: str = "") -> None:
        comparison = self._comparison
        if comparison is None:
            return
        generation = self._generation
        output_size = playback_dimensions(comparison, self._display_pixel_size())
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
        worker.frame_available.connect(self._on_frame_available)
        worker.decode_started.connect(self._on_decode_started)
        worker.decode_failed.connect(self._on_decode_failed)
        worker.playback_ended.connect(self._on_playback_ended)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._worker = worker
        self._is_playing = realtime
        suffix = f" ({reason})" if reason else ""
        self.status_changed.emit(f"Opening with FFmpeg tone-map fallback…{suffix}")
        self.playing_changed.emit(realtime)
        worker.start()

    def _stop_decoder(self) -> None:
        self._gst_timer.stop()
        pipeline = self._gst
        self._gst = None
        if pipeline is not None:
            pipeline.stop()
        self.source_video.set_native_playback(False)
        self.distorted_video.set_native_playback(False)
        self.source_video.hide()
        self.distorted_video.show()
        self.distorted_video.raise_()
        self.distorted_video.show_source(self._showing_source)
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            self._retired_workers.add(worker)

    def _poll_gstreamer(self) -> None:
        pipeline = self._gst
        comparison = self._comparison
        if pipeline is None or comparison is None:
            return
        try:
            update = pipeline.poll()
        except Exception as exc:
            self._fall_back_from_gstreamer(str(exc))
            return
        if update.position_ms is not None and comparison.fps > 0:
            self._frame = max(0, round(update.position_ms / 1000 * comparison.fps))
            self.position_changed.emit(update.position_ms)
        if update.status:
            state = "Playing" if self._wanted_playing else "Paused"
            self.status_changed.emit(f"{state} · GStreamer · {update.status}")
        if update.error:
            self._fall_back_from_gstreamer(update.error)
            return
        if update.ended:
            self._wanted_playing = False
            self._is_playing = False
            self.playing_changed.emit(False)
            self.status_changed.emit("Playback ended")

    def _fall_back_from_gstreamer(self, error: str) -> None:
        pipeline = self._gst
        self._gst = None
        self._gst_timer.stop()
        if pipeline is not None:
            pipeline.stop()
        self.source_video.set_native_playback(False)
        self.distorted_video.set_native_playback(False)
        self.source_video.hide()
        self.distorted_video.show()
        self.distorted_video.raise_()
        self.distorted_video.show_source(self._showing_source)
        self.status_changed.emit(
            f"GStreamer playback failed; trying FFmpeg fallback: {error}"
        )
        self._start_ffmpeg(self._wanted_playing, error)

    def _on_worker_finished(self, worker: _PairDecodeWorker) -> None:
        self._retired_workers.discard(worker)
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def _on_frame_available(self, generation: int) -> None:
        if generation != self._generation:
            return
        worker = self._worker
        if worker is None:
            return
        latest = worker.take_latest_frame()
        if latest is None:
            return
        frame, payload, width, height = latest
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
