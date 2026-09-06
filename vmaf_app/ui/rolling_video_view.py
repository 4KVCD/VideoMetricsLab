"""Rolling four-video playback: a shared source and three adjacent encodes."""
from __future__ import annotations

import time
from dataclasses import replace

from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QColor, QImage, QPainter

from vmaf_app.core.frame_extract import PreviewColorSettings
from vmaf_app.core.gpu import GpuVendor, plan_hwaccel
from vmaf_app.core.video_playback import neighbour_indices, playback_dimensions
from vmaf_app.ui.playback_worker import StreamDecodeWorker
from vmaf_app.ui.video_compare_view import VideoCompareView, _PairedFrameWidget


class _StreamSurface(_PairedFrameWidget):
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


class RollingVideoCompareView(VideoCompareView):
    """The public panel API, with bounded decode-ahead and instant selection.

    FFmpeg workers are producers, not clocks. A frame is presented only when
    both source and selected encode have that exact comparison frame number.
    Neighbours advance on that same clock but never hold up the selected pair.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series = []
        self._selected = 0
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
        self._native_pool = None
        self._last_status = ""
        self._pool_timer = QTimer(self)
        self._pool_timer.setInterval(10)
        self._pool_timer.timeout.connect(self._tick)
        self._source_surface = _StreamSurface(self)
        self._distorted_surface = _StreamSurface(self)
        self._source_surface.hide()
        self._distorted_surface.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for name in ("_source_surface", "_distorted_surface"):
            surface = getattr(self, name, None)
            if surface is not None:
                surface.setGeometry(self.rect())
        if getattr(self, "_native_pool", None) is not None:
            self._native_pool.resize()

    def closeEvent(self, event):
        self.clear()
        if self.live_workers():
            event.ignore()
            QTimer.singleShot(20, self.close)
            return
        super().closeEvent(event)

    def live_workers(self):
        return list(dict.fromkeys([*super().live_workers(), *self._pool.values()]))

    def load(self, comparison, position_ms, *, playing=False, color_settings=None, series=None):
        available, reason = self.can_play(comparison)
        if not available:
            self.clear()
            self.status_changed.emit(reason)
            return False
        series = list(series or [comparison])
        settings = color_settings or PreviewColorSettings()
        selected = series.index(comparison)
        if self._series == series and self._color_settings == settings and self._comparison is not None:
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
        self._selected = selected
        self._comparison = comparison
        self._color_settings = settings
        self._frame = round(position_ms / 1000 * comparison.fps)
        self._wanted_playing = bool(playing)
        self._restart_decoder(realtime=playing)
        return True

    def clear(self):
        super().clear()
        self._series = []

    def _restart_decoder(self, *, realtime):
        if self._comparison is None:
            return
        from vmaf_app.core.gstreamer_playback import uses_native_gstreamer

        native, reason = uses_native_gstreamer(self._comparison, self._color_settings)
        if native:
            from vmaf_app.ui.native_playback_pool import NativePlaybackPool

            self._stop_decoder()
            self._stop_audio()
            try:
                self._native_pool = NativePlaybackPool(self, self._series, self._selected,
                                                       self.position, self._color_settings, realtime)
                self.source_video.hide()
                self.distorted_video.hide()
                self._is_playing = self._wanted_playing = bool(realtime)
                self._native_pool.show_source(self._showing_source)
                self._pool_timer.start()
                self.playing_changed.emit(realtime)
                return
            except Exception as exc:
                reason = str(exc)
        self._stop_decoder()
        self._stop_audio()
        self._generation += 1
        self._start_ffmpeg(realtime, reason)

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
        self.source_video.hide()
        self.distorted_video.hide()
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

    def _source_recipe(self):
        # Source processing must not depend on which encode is currently
        # selected. Decode it at its own crop resolution; the surface fits it
        # into the window. Never launch a second source decoder.
        item = self._comparison
        return replace(item, distorted_info=item.source_info,
                       distorted_crop=item.source_crop, fps=self._series[0].fps)

    def _sync_pool(self):
        if not self._pool_active:
            return
        source = self._source_recipe()
        crop = source.source_crop
        source_key = ("source", str(source.source_info.path), None if crop is None else (crop.w, crop.h, crop.x, crop.y))
        desired = {source_key: (source, "source")}
        for index in neighbour_indices(len(self._series), self._selected):
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
        # spawn a fifth video decoder while the old process is exiting.
        occupied = sum(w.isRunning() for w in self._retired_workers) + len(self._pool)
        for key, (comparison, side) in self._desired.items():
            if key in self._pool or occupied >= 4:
                continue
            plan = plan_hwaccel(GpuVendor.AUTO, comparison.source_info.codec_name, comparison.distorted_info.codec_name)
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
                self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · GStreamer D3D11 · {len(self._native_pool.entries)}/4 streams · {self._native_pool.description}")
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
        self._status(f"{'Playing' if self._wanted_playing else 'Paused'} · {len(self._pool)}/4 streams · {detail}")
        self._check_end()

    def _check_end(self):
        counts = [item.frame_count / item.fps for item in self._series if item.frame_count > 0 and item.fps > 0]
        if counts and self._wanted_playing and self.position / 1000 >= min(counts) - 1 / self._series[0].fps:
            self.set_playing(False)
            self._status("Playback ended")

    def show_source(self, showing):
        super().show_source(showing)
        if self._native_pool is not None:
            self._native_pool.show_source(showing)
        if self._pool_active:
            (self._source_surface if showing else self._distorted_surface).raise_()

    def set_position(self, position_ms):
        if not self._pool_active and self._native_pool is None:
            return super().set_position(position_ms)
        self._frame = round(position_ms / 1000 * self._comparison.fps)
        self._restart_decoder(realtime=self._wanted_playing)

    def set_playing(self, playing):
        if self._native_pool is not None:
            self._native_pool.set_playing(playing)
            self._wanted_playing = self._is_playing = bool(playing)
            self.playing_changed.emit(playing)
            return
        if not self._pool_active:
            return super().set_playing(playing)
        playing = bool(playing)
        self._clock_frame = self._presented if not playing and self._presented >= 0 else self._target_frame()
        self._wanted_playing = self._is_playing = playing
        self._clock_started = time.monotonic() if playing and not self._buffering else None
        if not playing:
            self._stop_audio()
        elif not self._buffering:
            self._start_audio(self._frame)
        self.playing_changed.emit(playing)

    def set_audio_enabled(self, enabled):
        if self._native_pool is not None:
            self._audio_enabled = bool(enabled)
            self._native_pool.set_audio_enabled(enabled)
            return
        super().set_audio_enabled(enabled)

    def _start_audio(self, frame):
        if not self._pool_active:
            return super()._start_audio(frame)
        # Keep one soundtrack alive across visual switches. This is expressly
        # a visual comparison; multiple simultaneous soundtracks would mix.
        selected = self._comparison
        try:
            self._comparison = self._series[0]
            frame = round(self._presented / self._series[0].fps * self._comparison.fps)
            super()._start_audio(max(0, frame))
        finally:
            self._comparison = selected

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
        super()._stop_decoder()
