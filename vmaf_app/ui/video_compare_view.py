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
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QRegion
from PySide6.QtWidgets import QWidget

from vmaf_app.core import proc as proc_util
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorSettings,
    comparison_dimensions,
    frame_input_path,
)
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
from vmaf_app.ui.zoom import DragsZoomedFrame, Zoom

#: How often playback is looked after while frames are due (playing, a seek,
#: another test video, decoders starting), and once paused on the frame
#: wanted. At 10 ms throughout, a paused view cost 2-4% of a core.
_BUSY_TICK_MS = 10
_SETTLED_TICK_MS = 100


class _PairedFrameWidget(DragsZoomedFrame, QWidget):
    """A native window a GPU swapchain presents into (LockedNativePool),
    painted dark by Qt while no native playback owns it. A zoomed frame is
    dragged on it (the view's zoom)."""

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
        # Painting it from Qt would erase or flash over it -- but for the
        # margins beside a zoomed frame, which the sink, drawing in its
        # rectangle alone, leaves to the window: the default grey there.
        if self._native_playback:
            area = self.parentWidget().native_area()
            if area is not None:
                painter = QPainter(self)
                painter.setClipRegion(QRegion(self.rect()).subtracted(QRegion(area)))
                painter.fillRect(self.rect(), QColor("#171717"))
            return
        QPainter(self).fillRect(self.rect(), QColor("#171717"))

    def _zoom_context(self):
        return self.parentWidget().zoom_context()

    def _zoom_dragged(self) -> None:
        self.parentWidget().zoom_changed()


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
        place = self.parentWidget().zoom_placement((self._side_width, self._height))
        if place is None:
            scale = min(self.width() / self._side_width, self.height() / self._height)
            w, h = self._side_width * scale, self._height * scale
            painter.drawImage(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), self._image)
            return
        painter.drawImage(QRectF(place.x, place.y, place.width, place.height), self._image)


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
        self._pool_screen = None  # the screen size FFmpeg's decoders started for
        self._native_pool = None
        self._last_status = ""
        self._decoded_videos = DEFAULT_COMPARE_DECODED_VIDEOS
        self._pool_timer = QTimer(self)
        self._pool_timer.setInterval(_BUSY_TICK_MS)
        self._pool_timer.timeout.connect(self._tick)
        self._on_screen = False  # between showEvent and hideEvent
        self._zoom = Zoom()
        #: Where the GPU frame last presented zoomed is drawn (device pixels).
        self._native_drawn = None
        self._source_surface = _StreamSurface(self)
        self._distorted_surface = _StreamSurface(self)
        self._source_surface.hide()
        self._distorted_surface.hide()

    def resizeEvent(self, event):
        for surface in (self._source_surface, self._distorted_surface):
            surface.setGeometry(self.rect())
            surface.refresh_cursor()
        if self._native_pool is not None:
            self._native_pool.resize()
        super().resizeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._on_screen = True
        if self._native_pool is not None or self._pool_active:
            self._wake()

    def hideEvent(self, event):
        # Off screen -- another tab, the window minimized -- nothing is
        # drawn, and the decoders wait with their queues full. Ticking on,
        # the view cost 2-4% of a core for the rest of the session once
        # Video Compare had been opened.
        self._on_screen = False
        self._pool_timer.stop()
        super().hideEvent(event)

    def _wake(self) -> None:
        """Playback looked after every 10 ms again, from now: something is
        due. Only on screen (showEvent wakes it there). Not isVisible(): a
        minimized window's widgets stay visible to Qt, and minimizing hides
        the panel after the view -- the panel then pauses it, which woke it
        again. (Leaving the tab hides the panel first, then the view.)"""
        if self._on_screen and not (self._pool_timer.isActive()
                                    and self._pool_timer.interval() == _BUSY_TICK_MS):
            # Not restarted when already ticking that fast: a slider dragged
            # seeks more often than every 10 ms, and each restart would put
            # the next tick off again.
            self._pool_timer.start(_BUSY_TICK_MS)

    def _pace(self, settled: bool) -> None:
        """Ticks slowed once paused on the frame wanted, every 10 ms otherwise."""
        interval = _SETTLED_TICK_MS if settled else _BUSY_TICK_MS
        if self._pool_timer.interval() != interval:
            self._pool_timer.setInterval(interval)

    # ------------------------------------------------------------------ zoom
    def set_zoom(self, zoom: Zoom) -> None:
        """The zoom to show frames at: the panel's, which its still frames
        use too."""
        self._zoom = zoom

    def zoom_context(self, frame=None):
        """(zoom, view size, frame size, ratio) for a `frame`-pixel picture
        of the pair -- the comparison's own size, as its metrics see it,
        when not given -- at the comparison's scale: a source decoded at its
        native size beside a smaller encode is drawn at the encode's size,
        and one of another shape (its black bars kept) keeps its shape.
        None when nothing is loaded."""
        if self._comparison is None:
            return None
        reference = comparison_dimensions(self._comparison)
        if reference[0] <= 0 or reference[1] <= 0:
            return None
        frame = frame or reference
        return (self._zoom, (self.width(), self.height()), frame,
                self.devicePixelRatioF() * frame[0] / reference[0])

    def zoom_placement(self, frame):
        """Where a `frame`-pixel picture of the pair is drawn in the view."""
        context = self.zoom_context(frame)
        return None if context is None else self._zoom.placement(*context[1:])

    def native_area(self):
        """Where the GPU frame last presented zoomed is drawn, in the view's
        own pixels (a QRect), for the surface to paint around; None when
        fitted."""
        if self._zoom.factor is None or self._native_drawn is None:
            return None
        ratio = self.devicePixelRatioF()
        x, y, width, height = (value / ratio for value in self._native_drawn)
        return QRectF(x, y, width, height).toAlignedRect()

    def native_view(self, sample):
        """For LockedPresentation, a GPU frame zoomed: the part of it in
        view (x, y, width, height in its pixels) and where that goes in the
        window (device pixels). None when it fits the window."""
        if self._zoom.factor is None:
            self._native_drawn = None
            return None
        structure = sample.get_caps().get_structure(0)
        size = (structure.get_value("width"), structure.get_value("height"))
        context = self.zoom_context(size)
        if context is None:
            return None
        (left, top, right, bottom), drawn = self._zoom.visible(*context[1:])
        ratio = self.devicePixelRatioF()
        rectangle = (round(drawn.x * ratio), round(drawn.y * ratio),
                     max(1, round(drawn.width * ratio)), max(1, round(drawn.height * ratio)))
        # Sized from the part in view, then placed: rounding each edge apart
        # (round() takes halves to even) gave 4 rows where 5 were in view.
        width = max(1, min(size[0], round((right - left) * size[0])))
        height = max(1, min(size[1], round((bottom - top) * size[1])))
        crop = (min(round(left * size[0]), size[0] - width), min(round(top * size[1]), size[1] - height),
                width, height)
        self._native_drawn = rectangle
        return crop, rectangle

    def zoom_changed(self) -> None:
        """The zoom, or the part of the frame in view, changed: shown again.
        Zoomed, FFmpeg's frames are decoded at the comparison's full size,
        not the screen's (_wanted_maximum)."""
        if self._native_pool is not None:
            try:
                self._native_pool.place()
            except Exception as exc:
                self._native_pool.stop()
                self._native_pool = None
                self._start_ffmpeg(self._wanted_playing, str(exc))
        elif self._pool_active and (wanted := self._wanted_maximum()) != self._pool_maximum:
            if self._decode_size_changes(wanted):
                self._restart_decoder(realtime=self._wanted_playing)
            else:
                self._pool_maximum = wanted  # the same frames: they go on decoding
        for surface in (self._source_surface, self._distorted_surface):
            surface.refresh_cursor()
            surface.update()

    def _wanted_maximum(self):
        """The size FFmpeg's frames are decoded at most: the screen's when
        fitted; zoomed, the comparison's own (None)."""
        return None if self._zoom.factor is not None else self._display_pixel_size()

    def _decode_size_changes(self, maximum) -> bool:
        """Whether decoding with `maximum` would give any of the pool's
        videos other frame sizes: the source, the selected encode, and the
        encodes decoded beside it, whose workers are kept for a switch."""
        if self._comparison is None or maximum == self._pool_maximum:
            return False
        decoded = [comparison for comparison, _side in self._desired.values()] or [
            self._comparison, self._source_recipe()]
        return any(playback_dimensions(comparison, maximum) != playback_dimensions(comparison, self._pool_maximum)
                   for comparison in decoded)

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
                    self._wake()
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
            self._wake()
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
            self._wake()
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
        self._wake()
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
                self._wake()
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
        self._pool_screen = self._display_pixel_size()
        self._pool_maximum = self._wanted_maximum()
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
        self._wake()
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
        # The pair on screen first: the encodes decoded beside it start once
        # it has shown a frame (_tick calls this again). Started with it, two
        # of them slowed its first frame at the tab's first opening from
        # 1.1-1.3 s to 1.5-1.6.
        on_screen = None if self._presented >= 0 else (
            {key for key in self._desired if key[0] == "source"} | {("distorted", self._selected)})
        launched = False
        for key, (comparison, side) in self._desired.items():
            if key in self._pool or occupied >= self.decoder_limit:
                continue
            if on_screen is not None and key not in on_screen:
                continue
            source_info, distorted_info = comparison.source_info, comparison.distorted_info
            plan = plan_hwaccel(GpuVendor.AUTO, source_info.codec_name, distorted_info.codec_name,
                                source_pix_fmt=source_info.pix_fmt, distorted_pix_fmt=distorted_info.pix_fmt,
                                source_size=(source_info.width, source_info.height),
                                distorted_size=(distorted_info.width, distorted_info.height))
            worker = StreamDecodeWorker(comparison, side, self._target_frame(), self._color_settings,
                                        plan, self._pool_maximum, self)
            worker.ready.connect(lambda detail, k=key, w=worker: self._stream_ready(k, w, detail))
            worker.failed.connect(lambda error, k=key, w=worker: self._stream_failed(k, w, error))
            worker.finished.connect(lambda k=key, w=worker: self._stream_finished(k, w))
            self._pool[key] = worker
            self._history[key] = {}
            worker.start()
            occupied += 1
            launched = True
        if launched and self._presented >= 0:
            self._pool_status()  # its count of streams

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
        if (pool := self._native_pool) is not None:
            try:
                position = pool.poll()
                frame = round(position / 1000 * self._comparison.fps)
                if frame != self._frame:
                    self._frame = frame
                    self.position_changed.emit(position)
                if pool.ended and self._wanted_playing:
                    self.set_playing(False)
                self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · GStreamer D3D11 · {len(pool.entries)}/{self.decoder_limit} streams · {pool.description}")
                self._check_end()
                self._pace(not self._wanted_playing and pool.pair is not None and pool.pair_index == pool.selected)
            except Exception as exc:
                self._native_pool.stop()
                self._native_pool = None
                self._start_ffmpeg(self._wanted_playing, str(exc))
            return
        if not self._pool_active:
            return
        if self._display_pixel_size() != self._pool_screen:
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
            self._pace(bool(error))
            return
        frame = max(common)
        if frame == self._presented:
            if self._wanted_playing and target - frame > 2:
                # Don't let audio run arbitrarily ahead during sustained
                # starvation. Resume it from the next displayed frame.
                self._stop_audio()
                self._buffering = True
            self._pace(not self._wanted_playing and frame >= target)
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
        self._pool_status()
        self._check_end()
        self._pace(False)  # the next tick starts the encodes decoded beside it

    def _pool_status(self) -> None:
        """The status while FFmpeg's frames are shown."""
        detail = self._details.get(("distorted", self._selected), "GPU processing starting")
        if self._pool_reason:
            detail += " · SDR preview (native playback unavailable)"
        self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · {len(self._pool)}/{self.decoder_limit} streams · {detail}")

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
