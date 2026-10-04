"""Frame-locked playback of the source against the test videos.

A shared source and the selected encode, with the encodes beside it decoding
ahead so that switching between test videos is instant. Where GStreamer can
play them on the GPU, they play natively (LockedNativePool: D3D11, frame
locked, one soundtrack); otherwise FFmpeg workers decode them
(StreamDecodeWorker) and their frames are presented here, on one clock.

This is the one playback view. It used to be two stacked by inheritance: a
base that played one pair through a single GStreamer pipeline or one FFmpeg
process, and a rolling view built on it that replaced how it decoded -- so
the base's own decoding could no longer be reached.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import replace

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QWidget

from vmaf_app.core import proc as proc_util
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings, frame_input_path
from vmaf_app.core.gpu import GpuVendor, plan_hwaccel
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.video_playback import (
    DEFAULT_COMPARE_DECODED_VIDEOS,
    build_audio_command,
    neighbour_indices,
    playback_dimensions,
    source_playback_comparison,
)
from vmaf_app.ui.playback_worker import StreamDecodeWorker


class _PairedFrameWidget(QWidget):
    """A native window a GPU swapchain presents into (LockedNativePool),
    painted dark by Qt while no native playback owns it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors)
        self._native_playback = False

    def set_native_playback(self, enabled: bool) -> None:
        self._native_playback = bool(enabled)
        self.update()

    def paintEvent(self, _event) -> None:
        # The swapchain owns this child HWND while native playback is active.
        # Painting it from Qt would erase or flash over it.
        if self._native_playback:
            return
        QPainter(self).fillRect(self.rect(), QColor("#171717"))


class _StreamSurface(_PairedFrameWidget):
    """Where FFmpeg's decoded frames are drawn, fitted to the view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._payload: bytes | None = None
        self._image = QImage()
        self._side_width = 0
        self._height = 0

    def set_frame(self, payload, size):
        self._payload = payload
        self._side_width, self._height = size
        self._image = QImage(payload, *size, size[0] * 4, QImage.Format_RGBA8888)
        self.update()

    def clear_frame(self):
        self._payload = None
        self._image = QImage()
        self.update()

    def paintEvent(self, event):
        if self._native_playback:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171717"))
        if self._image.isNull():
            return
        scale = min(self.width() / self._side_width, self.height() / self._height)
        w, h = self._side_width * scale, self._height * scale
        painter.drawImage(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), self._image)


class VideoCompareView(QWidget):
    """The source and the test videos, with bounded decode-ahead and instant
    selection.

    FFmpeg workers are producers, not clocks. A frame is presented only when
    both source and selected encode have that exact comparison frame number.
    Neighbours advance on that same clock but never hold up the selected pair.
    """

    position_changed = Signal(int)
    playing_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #171717;")
        self._comparison: FrameComparison | None = None
        self._color_settings = PreviewColorSettings()
        #: Decoders being stopped, which still count against the limit
        #: while they exit (LockedNativePool's too).
        self._retired_workers: set[QThread] = set()
        #: How many times the decoders have been started: a change that
        #: should keep them running (another test video) leaves it as it is.
        self._generation = 0
        self._frame = 0
        self._wanted_playing = False
        self._is_playing = False
        self._showing_source = False
        self._audio_enabled = True
        self._audio_process: subprocess.Popen | None = None
        self._audio_handle: ProcessHandle | None = None
        self._series: list[FrameComparison] = []
        self._selected = 0
        self._source_native = True
        self._pool = {}
        self._history = {}
        self._details = {}
        self._failures = {}
        self._desired = {}
        self._clock_frame = 0
        self._clock_started = None
        self._presented = -1
        self._buffering = True
        self._pool_active = False
        self._pool_reason = ""
        self._pool_maximum = None
        self._native_pool = None
        self._last_status = ""
        self._decoded_videos = DEFAULT_COMPARE_DECODED_VIDEOS
        self._pool_timer = QTimer(self)
        self._pool_timer.setInterval(10)
        self._pool_timer.timeout.connect(self._tick)
        self._source_surface = _StreamSurface(self)
        self._distorted_surface = _StreamSurface(self)
        self._source_surface.hide()
        self._distorted_surface.hide()

    def resizeEvent(self, event):
        for surface in (self._source_surface, self._distorted_surface):
            surface.setGeometry(self.rect())
        if self._native_pool is not None:
            self._native_pool.resize()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.clear()
        if self.live_workers():
            event.ignore()
            QTimer.singleShot(20, self.close)
            return
        super().closeEvent(event)

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
        running = [worker for worker in self._retired_workers if worker.isRunning()]
        return list(dict.fromkeys([*running, *self._pool.values()]))

    def load(self, comparison, position_ms, *, playing=False, color_settings=None, series=None, source_native=True):
        available, reason = self.can_play(comparison)
        if not available:
            self.clear()
            self.status_changed.emit(reason)
            return False
        series = list(series or [comparison])
        settings = color_settings or PreviewColorSettings()
        selected = series.index(comparison)
        if self._series == series and self._color_settings == settings and self._comparison is not None and self._source_native == source_native:
            timestamp = self.position
            changed = self._selected != selected
            self._selected = selected
            self._comparison = comparison
            self._frame = round(timestamp / 1000 * comparison.fps)
            if self._native_pool is not None:
                if changed:
                    try:
                        self._native_pool.sync(selected)
                    except Exception as exc:
                        self._native_pool.stop()
                        self._native_pool = None
                        self._start_ffmpeg(playing, str(exc))
                self.set_playing(playing)
                return True
            if self._pool_active:
                if changed:
                    self._distorted_surface.clear_frame()
                    self._sync_pool()
                    self._presented = -1
                    self._tick()
                if bool(playing) != self._wanted_playing:
                    self.set_playing(playing)
                return True
            if changed:
                self._restart_decoder(realtime=playing)
            return True
        self._series = series
        self._source_native = source_native
        self._selected = selected
        self._comparison = comparison
        self._color_settings = settings
        self._frame = round(position_ms / 1000 * comparison.fps)
        self._wanted_playing = bool(playing)
        self._restart_decoder(realtime=playing)
        return True

    def clear(self) -> None:
        self._wanted_playing = False
        self._is_playing = False
        self._comparison = None
        self._stop_decoder()
        self._stop_audio()
        self.playing_changed.emit(False)
        self._series = []

    def set_position(self, position_ms: int) -> None:
        if self._native_pool is not None:
            self._native_pool.seek(position_ms)
            self._frame = self._native_pool.frame
            return
        comparison = self._comparison
        if comparison is None or comparison.fps <= 0:
            return
        self._frame = max(0, round(position_ms / 1000 * comparison.fps))
        self._restart_decoder(realtime=self._wanted_playing)

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        if self._native_pool is not None:
            self._native_pool.set_playing(playing)
            self._wanted_playing = self._is_playing = playing
            self.playing_changed.emit(playing)
            return
        if not self._pool_active:
            # Nothing decoding: playing starts the decoders.
            self._wanted_playing = playing
            if playing:
                if not self._is_playing and self._comparison is not None:
                    self._restart_decoder(realtime=True)
            else:
                if self._audio_handle is not None:
                    self._audio_handle.pause()
                self._is_playing = False
                self.playing_changed.emit(False)
            return
        self._clock_frame = self._presented if not playing and self._presented >= 0 else self._target_frame()
        self._wanted_playing = self._is_playing = playing
        self._clock_started = time.monotonic() if playing and not self._buffering else None
        if not playing:
            self._stop_audio()
        elif not self._buffering:
            self._start_audio(self._frame)
        self.playing_changed.emit(playing)

    def show_source(self, showing: bool) -> None:
        self._showing_source = bool(showing)
        if self._native_pool is not None:
            self._native_pool.show_source(showing)
        if self._pool_active:
            (self._source_surface if showing else self._distorted_surface).raise_()

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = bool(enabled)
        if self._native_pool is not None:
            self._native_pool.set_audio_enabled(enabled)
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

    def _restart_decoder(self, *, realtime):
        if self._comparison is None:
            return
        self._generation += 1
        from vmaf_app.core.gstreamer_playback import uses_native_gstreamer

        native, reason = uses_native_gstreamer(self._comparison, self._color_settings)
        if native:
            from vmaf_app.ui.locked_native_pool import LockedNativePool

            self._stop_decoder()
            self._stop_audio()
            try:
                self._native_pool = LockedNativePool(self, self._series, self._selected,
                                                     self.position, self._color_settings, realtime,
                                                     source_native=self._source_native)
                self._is_playing = self._wanted_playing = bool(realtime)
                self._native_pool.show_source(self._showing_source)
                self._pool_timer.start()
                self.playing_changed.emit(realtime)
                return
            except Exception as exc:
                reason = str(exc)
        self._stop_decoder()
        self._stop_audio()
        self._start_ffmpeg(realtime, reason)

    def _display_pixel_size(self) -> tuple[int, int] | None:
        handle = self.window().windowHandle()
        screen = handle.screen() if handle is not None else QGuiApplication.primaryScreen()
        if screen is None:
            return None
        geometry = screen.geometry()
        ratio = screen.devicePixelRatio()
        return round(geometry.width() * ratio), round(geometry.height() * ratio)

    def _start_ffmpeg(self, realtime, reason=""):
        self._pool_reason = reason
        self._pool_maximum = self._display_pixel_size()
        self._pool_active = True
        self._wanted_playing = bool(realtime)
        self._is_playing = bool(realtime)
        self._clock_frame = round(self.position / 1000 * self._series[0].fps)
        self._clock_started = None
        self._buffering = True
        self._presented = -1
        for surface in (self._source_surface, self._distorted_surface):
            surface.clear_frame()
            surface.setGeometry(self.rect())
            surface.show()
        self.show_source(self._showing_source)
        self._status("Preparing source/current/adjacent videos · GPU tone mapping and RGB"
                     + (f" · native fallback: {reason}" if reason else ""))
        self._sync_pool()
        self._pool_timer.start()
        self.playing_changed.emit(realtime)

    def _status(self, text):
        if text != self._last_status:
            self._last_status = text
            self.status_changed.emit(text)

    @property
    def decoded_videos(self) -> int:
        """How many test videos are kept decoding: the selected one and its neighbours."""
        return self._decoded_videos

    @property
    def decoder_limit(self) -> int:
        """Decoders that may run at once: the source plus the test videos."""
        return 1 + self._decoded_videos

    def set_decoded_videos(self, count: int) -> None:
        """Applies immediately to whatever is playing: extra neighbours are
        retired, missing ones started, the selected pair is never touched."""
        count = max(1, int(count))
        if count == self._decoded_videos:
            return
        self._decoded_videos = count
        if self._native_pool is not None:
            self._native_pool.sync(self._selected)
        elif self._pool_active:
            self._sync_pool()

    def _source_recipe(self):
        return replace(source_playback_comparison(self._comparison, self._source_native),
                       fps=self._series[0].fps)

    def _sync_pool(self):
        if not self._pool_active:
            return
        source = self._source_recipe()
        crop = source.source_crop
        source_key = ("source", str(source.source_info.path), None if crop is None else (crop.w, crop.h, crop.x, crop.y), playback_dimensions(source, self._pool_maximum))
        desired = {source_key: (source, "source")}
        for index in neighbour_indices(len(self._series), self._selected, self._decoded_videos):
            desired[("distorted", index)] = (replace(self._series[index], fps=self._series[0].fps), "distorted")
        self._desired = desired
        for key in list(self._pool):
            if key not in desired:
                worker = self._pool.pop(key)
                worker.cancel()
                self._retired_workers.add(worker)
                self._history.pop(key, None)
                self._details.pop(key, None)
                self._failures.pop(key, None)
        self._launch_missing()

    def _launch_missing(self):
        if not self._pool_active:
            return
        # Retiring workers count too: rapid navigation may not transiently
        # spawn one video decoder more than allowed while the old process is exiting.
        occupied = sum(w.isRunning() for w in self._retired_workers) + len(self._pool)
        for key, (comparison, side) in self._desired.items():
            if key in self._pool or occupied >= self.decoder_limit:
                continue
            plan = plan_hwaccel(GpuVendor.AUTO, comparison.source_info.codec_name, comparison.distorted_info.codec_name,
                                source_pix_fmt=comparison.source_info.pix_fmt,
                                distorted_pix_fmt=comparison.distorted_info.pix_fmt)
            worker = StreamDecodeWorker(comparison, side, self._target_frame(), self._color_settings,
                                        plan, self._pool_maximum, self)
            worker.ready.connect(lambda detail, k=key, w=worker: self._stream_ready(k, w, detail))
            worker.failed.connect(lambda error, k=key, w=worker: self._stream_failed(k, w, error))
            worker.finished.connect(lambda k=key, w=worker: self._stream_finished(k, w))
            self._pool[key] = worker
            self._history[key] = {}
            worker.start()
            occupied += 1

    def _stream_ready(self, key, worker, detail):
        if self._pool.get(key) is worker:
            self._details[key] = detail

    def _stream_failed(self, key, worker, error):
        if self._pool.get(key) is worker:
            self._failures[key] = error
            self.status_changed.emit(f"Video {key} could not play: {error}")

    def _stream_finished(self, key, worker):
        self._retired_workers.discard(worker)
        # Keep queued frames from a naturally completed short clip until
        # consumed. Obsolete workers may be destroyed immediately.
        if self._pool.get(key) is not worker:
            worker.deleteLater()
        self._launch_missing()

    def _target_frame(self):
        if self._clock_started is None or not self._wanted_playing:
            return self._clock_frame
        return self._clock_frame + int((time.monotonic() - self._clock_started) * self._series[0].fps)

    def _tick(self):
        if self._native_pool is not None:
            try:
                position = self._native_pool.poll()
                frame = round(position / 1000 * self._comparison.fps)
                if frame != self._frame:
                    self._frame = frame
                    self.position_changed.emit(position)
                if self._native_pool.ended and self._wanted_playing:
                    self.set_playing(False)
                self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · GStreamer D3D11 · {len(self._native_pool.entries)}/{self.decoder_limit} streams · {self._native_pool.description}")
                self._check_end()
            except Exception as exc:
                self._native_pool.stop()
                self._native_pool = None
                self._start_ffmpeg(self._wanted_playing, str(exc))
            return
        if not self._pool_active:
            return
        if self._display_pixel_size() != self._pool_maximum:
            self._restart_decoder(realtime=self._wanted_playing)
            return
        self._launch_missing()
        target = self._target_frame()
        source_key = next((key for key in self._desired if key[0] == "source"), None)
        distorted_key = ("distorted", self._selected)
        # Do not discard fast-stream frames while its partner is still
        # decoding them. Backpressure holds that producer at three frames.
        limits = []
        for key in (source_key, distorted_key):
            worker = self._pool.get(key)
            available = worker.latest_frame_number() if worker is not None else None
            if available is None:
                available = max(self._history.get(key, {}), default=self._clock_frame)
            limits.append(available)
        paired_target = min(target, *limits)
        for key, worker in self._pool.items():
            history = self._history[key]
            due = paired_target if key in (source_key, distorted_key) else target
            for number, payload in worker.drain_through(due):
                history[number] = payload
            # Three past frames plus the worker's three future frames bound
            # RAM regardless of movie duration or number of files in the list.
            for number in sorted(history)[:-3]:
                del history[number]
        source = self._history.get(source_key, {})
        distorted = self._history.get(distorted_key, {})
        common = source.keys() & distorted.keys()
        if not common:
            self._buffering = True
            if self._wanted_playing and target - max(0, self._presented) > 2:
                self._stop_audio()
            error = self._failures.get(source_key) or self._failures.get(distorted_key)
            self._status(f"Selected video failed: {error}" if error else
                         "Buffering selected comparison; adjacent videos are preparing in the background")
            return
        frame = max(common)
        if frame == self._presented:
            if self._wanted_playing and target - frame > 2:
                # Don't let audio run arbitrarily ahead during sustained
                # starvation. Resume it from the next displayed frame.
                self._stop_audio()
                self._buffering = True
            return
        self._presented = frame
        fps = self._series[0].fps
        if self._wanted_playing and target - frame > 2:
            # A decoder that cannot sustain real time must not create an
            # ever-growing queue or let the comparison drift away from audio.
            self._clock_frame = frame
            self._clock_started = time.monotonic()
            self._stop_audio()
            self._buffering = True
        self._frame = round(frame / fps * self._comparison.fps)
        self._source_surface.set_frame(source[frame], playback_dimensions(self._desired[source_key][0], self._pool_maximum))
        self._distorted_surface.set_frame(distorted[frame], playback_dimensions(self._comparison, self._pool_maximum))
        if self._clock_started is None and self._wanted_playing:
            self._clock_frame = frame
            self._clock_started = time.monotonic()
        if self._buffering and self._wanted_playing:
            self._start_audio(self._frame)
        self._buffering = False
        self.position_changed.emit(round(frame / fps * 1000))
        detail = self._details.get(distorted_key, "GPU processing starting")
        if self._pool_reason:
            detail += " · SDR preview (native playback unavailable)"
        self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · {len(self._pool)}/{self.decoder_limit} streams · {detail}")
        self._check_end()

    def _check_end(self):
        counts = [item.frame_count / item.fps for item in self._series if item.frame_count > 0 and item.fps > 0]
        if counts and self._wanted_playing and self.position / 1000 >= min(counts) - 1 / self._series[0].fps:
            self.set_playing(False)
            self._status("Playback ended")

    def _start_audio(self, frame: int) -> None:
        comparison = self._comparison
        if self._pool_active and self._series:
            # Keep one soundtrack alive across visual switches: this is a
            # visual comparison, and several soundtracks at once would mix.
            # It is the source's, from the frame on screen.
            comparison = self._series[0]
            frame = max(0, round(self._presented / self._series[0].fps * comparison.fps))
        if not self._audio_enabled or comparison is None:
            return
        self._stop_audio()
        command = build_audio_command(comparison, frame)
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
            proc_util.terminate(process)

    def _stop_decoder(self):
        if self._native_pool is not None:
            self._native_pool.stop()
            self._native_pool = None
        self._pool_active = False
        self._pool_timer.stop()
        for worker in self._pool.values():
            worker.cancel()
            if worker.isRunning():
                self._retired_workers.add(worker)
            else:
                worker.deleteLater()
        self._pool.clear()
        self._history.clear()
        self._desired.clear()
        self._details.clear()
        self._failures.clear()
        self._source_surface.clear_frame()
        self._distorted_surface.clear_frame()
        self._source_surface.hide()
        self._distorted_surface.hide()
