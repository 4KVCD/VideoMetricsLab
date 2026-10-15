"""Video Compare's playback view (VideoCompareView): the source against the
test videos, frame by frame in step.

It plays the source and the selected test video, with the test videos beside
it decoding ahead so that switching between them is instant. Where GStreamer
can play them on the GPU, they play natively (LockedNativePool: frame-locked
on the GPU, with one soundtrack). Otherwise FFmpeg workers decode them
(StreamDecodeWorker), and their frames are shown here, on one clock.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import replace

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from vmaf_app.core import proc as proc_util
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorSettings,
    _would_distort,
    comparison_dimensions,
    frame_input_path,
)
from vmaf_app.core.geometry import untimed_pair_problem
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
#: wanted. At 10 ms throughout, a paused view would cost 2-4% of a core.
_BUSY_TICK_MS = 10
_SETTLED_TICK_MS = 100
#: How long FFmpeg's playback must keep up before its soundtrack, stopped when
#: the frames fell behind, starts again. Restarted at the next frame, decoders
#: slower than real time would restart it at every frame (11 times a second),
#: filling the window's thread and breaking the sound into pieces.
_AUDIO_RESUME_SECONDS = 1.0
#: How long a video file found on disk counts as there (can_play). It is
#: asked at every frame played, 120 times a second, which would ask the disk
#: as often.
_FOUND_FOR_S = 2.0
#: When each video file was last found on disk (time.monotonic), by path.
_FOUND: dict = {}


def _found_on_disk(path) -> bool:
    """Whether `path` is a file: found less than _FOUND_FOR_S ago, or on
    disk now. A file found missing is looked for again at once."""
    now = time.monotonic()
    found = _FOUND.get(path)
    if found is not None and now - found < _FOUND_FOR_S:
        return True
    if not path.is_file():
        _FOUND.pop(path, None)
        return False
    if len(_FOUND) > 256:
        _FOUND.clear()
    _FOUND[path] = now
    return True


def crops_change_nothing(pending: FrameComparison, cropped: FrameComparison) -> bool:
    """Whether the black bars found for `pending` leave its pictures as they
    are: none on either side (each crop the whole frame), the size compared
    at unchanged, and neither side boxed while they were looked for (each
    already the shape it is shown at). Then the frames playing are the ones
    the crops give."""
    size = comparison_dimensions(pending)
    if size != comparison_dimensions(cropped):
        return False
    for info, crop in ((cropped.source_info, cropped.source_crop),
                       (cropped.distorted_info, cropped.distorted_crop)):
        if crop is not None and not crop.is_noop(info.width, info.height):
            return False
        if _would_distort(pending, info, None, *size):
            return False
    return True


def _end_audio(process: subprocess.Popen, handle: ProcessHandle | None) -> None:
    """Ends a soundtrack's FFmpeg and anything it started."""
    if handle is not None:
        handle.terminate()
        handle.detach()
    if process.poll() is None:
        proc_util.terminate(process)

class _PairedFrameWidget(DragsZoomedFrame, QWidget):
    """A native window the native presenter's window sits in
    (LockedNativePool), painted dark by Qt, which shows until the presenter
    draws its first frame. A zoomed frame is dragged on it (the view's zoom)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors)

    def paintEvent(self, _event) -> None:
        """Paints it dark."""
        # Even while native playback owns it: its presenter's window covers
        # all of it once it shows a frame, but until then (a pool just
        # started) this is what shows, and unpainted it showed whatever the
        # window last drew there -- the Videos tab.
        QPainter(self).fillRect(self.rect(), QColor("#171717"))

    def _zoom_context(self):
        """The view's zoom context (see VideoCompareView.zoom_context)."""
        return self.parentWidget().zoom_context()

    def _zoom_dragged(self) -> None:
        """Tells the view the frame was dragged."""
        self.parentWidget().zoom_changed()


class _StreamSurface(_PairedFrameWidget):
    """Where FFmpeg's decoded frames are drawn, fitted to the view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # It paints all of itself: nothing behind it is painted first.
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self._payload: bytes | None = None
        self._image: QImage | None = QImage()
        self._side_width = 0
        self._height = 0
        #: Where the last paint drew a fitted frame, for what view and frame
        #: size: (key, rectangle), or None (update_frame).
        self._drawn = None

    def set_frame(self, payload, size):
        """Sets the frame it shows. The view repaints only the surface on top
        (the other is covered), and the QImage is made when it is painted,
        so the covered surface makes none."""
        self._payload = payload
        self._side_width, self._height = size
        self._image = None

    def clear_frame(self):
        """Shows no frame."""
        self._payload = None
        self._image = QImage()
        self.update()

    def shows(self, payload) -> bool:
        """Whether `payload` is the frame it shows."""
        return payload is self._payload

    def update_frame(self):
        """Paints its new frame: only where the last one was drawn, when it
        is drawn the same way. The bars beside it stay as painted; filling
        them again at each frame would take 0.2 ms of the window's thread,
        and send them to the screen again too."""
        drawn = self._drawn
        if drawn is not None and drawn[0] == self._drawn_key():
            self.update(drawn[1])
        else:
            self.update()

    def _drawn_key(self):
        """What decides where the frame is drawn: the surface's size and pixel
        ratio, and the frame's size."""
        return self.width(), self.height(), self.devicePixelRatioF(), self._side_width, self._height

    def _frame_image(self) -> QImage:
        """The frame as a QImage, made once per frame."""
        if self._image is None:
            # RGBX, not RGBA: FFmpeg's alpha is always opaque, and a frame
            # that may be translucent is blended into the window, not copied.
            self._image = QImage(self._payload, self._side_width, self._height, self._side_width * 4,
                                 QImage.Format_RGBX8888)
        return self._image

    def paintEvent(self, event):
        """Draws the frame: fitted to the view, or where the zoom puts it, with
        dark bars around it."""
        painter = QPainter(self)
        background = QColor("#171717")
        image = self._frame_image()
        view = self.parentWidget()
        self._drawn = None
        if image.isNull() or not view.fitted:
            painter.fillRect(self.rect(), background)
            place = None if image.isNull() else view.zoom_placement((self._side_width, self._height))
            if place is not None:
                image.setDevicePixelRatio(1.0)
                painter.drawImage(QRectF(place.x, place.y, place.width, place.height), image)
            return
        # Fitted, on whole device pixels. A frame decoded at the view's size
        # (VideoCompareView._wanted_maximum) is copied as it is, not scaled by
        # the pixel or two its even size leaves, and only the bars beside it
        # are filled. Each bar reaches a pixel into the frame, which the
        # frame then covers, so that rounding leaves no gap.
        ratio = self.devicePixelRatioF()
        view_w, view_h = round(self.width() * ratio), round(self.height() * ratio)
        scale = min(view_w / self._side_width, view_h / self._height)
        w, h = round(self._side_width * scale), round(self._height * scale)
        if 0 <= w - self._side_width <= 2 and 0 <= h - self._height <= 2:
            w, h = self._side_width, self._height
        x, y = (view_w - w) // 2, (view_h - h) // 2
        self._drawn = (self._drawn_key(), QRectF(x / ratio, y / ratio, w / ratio, h / ratio).toAlignedRect())
        if not self._drawn[1].contains(event.rect()):  # not just the frame (update_frame)
            bars = []
            if x > 0:
                bars.append((0, 0, x + 1, view_h))
            if x + w < view_w:
                bars.append((x + w - 1, 0, view_w - x - w + 1, view_h))
            if y > 0:
                bars.append((0, 0, view_w, y + 1))
            if y + h < view_h:
                bars.append((0, y + h - 1, view_w, view_h - y - h + 1))
            for left, top, width, height in bars:
                painter.fillRect(QRectF(left / ratio, top / ratio, width / ratio, height / ratio), background)
        if (w, h) == (self._side_width, self._height):
            image.setDevicePixelRatio(ratio)
            painter.drawImage(QPointF(x / ratio, y / ratio), image)
        else:  # decoded for another size, until the view's settles (_follow_decode_size)
            image.setDevicePixelRatio(1.0)
            painter.drawImage(QRectF(x / ratio, y / ratio, w / ratio, h / ratio), image)


class VideoCompareView(QWidget):
    """Plays the source and the test videos, decoding a few frames ahead,
    with instant switching between test videos.

    The FFmpeg workers only decode; they are not clocks. A frame is shown
    only when both the source and the selected test video have that exact
    frame number. The neighbours follow the same clock, but never hold up
    the selected pair.
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
        #: The soundtrack is to start (again) with the frames, not before
        #: _audio_resume_at: playback fell behind until then. A start (Play,
        #: a seek, another test video) is not falling behind: it clears that.
        self._audio_due = False
        self._audio_resume_at = 0.0
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
        #: The view's size settling after a resize: FFmpeg's frames are then
        #: decoded at it (_follow_decode_size).
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(300)
        self._resize_timer.timeout.connect(self._follow_decode_size)
        self._native_pool = None
        self._last_status = ""
        self._decoded_videos = DEFAULT_COMPARE_DECODED_VIDEOS
        self._pool_timer = QTimer(self)
        # Precise: while playing it ticks when each frame is due (native
        # playback: LockedNativePool.next_tick_ms) or every half frame
        # (_busy_tick_ms).
        self._pool_timer.setTimerType(Qt.PreciseTimer)
        self._pool_timer.setInterval(_BUSY_TICK_MS)
        self._pool_timer.timeout.connect(self._tick)
        self._on_screen = False  # between showEvent and hideEvent
        self._zoom = Zoom()
        self._source_surface = _StreamSurface(self)
        self._distorted_surface = _StreamSurface(self)
        self._source_surface.hide()
        self._distorted_surface.hide()

    def resizeEvent(self, event):
        """Fits the surfaces to the view, and decodes at the new size once the resize settles."""
        for surface in (self._source_surface, self._distorted_surface):
            surface.setGeometry(self.rect())
            surface.refresh_cursor()
        if self._native_pool is not None:
            self._native_pool.resize()
        if self._pool_active:
            self._resize_timer.start()
        super().resizeEvent(event)

    def event(self, event):
        """Decodes at the new size after a move to a screen of another scaling."""
        # Moved to a screen of another scaling: other device pixels, as a resize.
        if event.type() == QEvent.DevicePixelRatioChange and self._pool_active:
            self._resize_timer.start()
        return super().event(event)

    def showEvent(self, event):
        """Looks after playback again once on screen."""
        super().showEvent(event)
        self._on_screen = True
        if self._native_pool is not None or self._pool_active:
            self._wake()

    def hideEvent(self, event):
        """Stops looking after playback while off screen."""
        # Off screen (another tab, the window minimized), nothing is drawn,
        # and the decoders wait with full queues. Ticking on, the view would
        # cost 2-4% of a core for the rest of the session once Video Compare
        # had been opened.
        self._on_screen = False
        self._pool_timer.stop()
        super().hideEvent(event)

    def _wake(self) -> None:
        """Looks after playback often again from now, because something is
        due. Only on screen (showEvent wakes it there), judged by
        _on_screen rather than isVisible(): a minimized window's widgets
        stay visible to Qt, and minimizing hides the panel after the view,
        so the panel's pause would wake it again. (Leaving the tab hides the
        panel first, then the view.)"""
        busy = self._busy_tick_ms()
        if self._on_screen and not (self._pool_timer.isActive() and self._pool_timer.interval() == busy):
            # Not restarted when already ticking that fast: a slider dragged
            # seeks more often than it ticks, and each restart would put the
            # next tick off again.
            self._pool_timer.start(busy)

    def _pace(self, settled: bool) -> None:
        """Slows the ticks once paused on the frame wanted (`settled`), and
        speeds them up otherwise."""
        interval = _SETTLED_TICK_MS if settled else self._busy_tick_ms()
        if self._pool_timer.interval() != interval:
            self._pool_timer.setInterval(interval)

    def _busy_tick_ms(self) -> int:
        """The busy tick: half a frame's time while playing (4 ms at
        120 fps), and 10 ms otherwise. Ticking every 10 ms, a 120 fps video
        played natively showed only 83 frames a second."""
        fps = self._series[0].fps if self._wanted_playing and self._series else 0
        return max(2, min(_BUSY_TICK_MS, int(500 / fps))) if fps > 0 else _BUSY_TICK_MS

    # ------------------------------------------------------------------ zoom
    def set_zoom(self, zoom: Zoom) -> None:
        """The zoom to show frames at: the panel's, which its still frames
        use too."""
        self._zoom = zoom

    @property
    def fitted(self) -> bool:
        """Whether frames are fitted to the view, not zoomed."""
        return self._zoom.factor is None

    def zoom_context(self, frame=None):
        """Returns (zoom, view size, frame size, ratio) for a picture of the
        pair `frame` pixels in size (by default the comparison's own size, as
        its metrics see it), at the comparison's scale: a source decoded at
        its own size beside a smaller encode is drawn at the encode's size,
        and one of another shape (its black bars kept) keeps its shape.
        Returns None when nothing is loaded."""
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

    def native_view(self, sample):
        """For LockedPresentation, a zoomed GPU frame: returns the part of it
        in view (x, y, width, height in its pixels) and where that goes in
        the window (device pixels), or None when it fits the window."""
        if self._zoom.factor is None:
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
        # Sized from the part in view, then placed. Rounding each edge on its
        # own (round() takes halves to even) can give 4 rows where 5 are in
        # view.
        width = max(1, min(size[0], round((right - left) * size[0])))
        height = max(1, min(size[1], round((bottom - top) * size[1])))
        crop = (min(round(left * size[0]), size[0] - width), min(round(top * size[1]), size[1] - height),
                width, height)
        return crop, rectangle

    def zoom_changed(self) -> None:
        """Shows the frame again after the zoom, or the part of the frame in
        view, has changed. When zoomed, FFmpeg's frames are decoded at the
        comparison's full size, not the view's (_wanted_maximum)."""
        if self._native_pool is not None:
            try:
                self._native_pool.place()
            except Exception as exc:
                self._native_pool.stop()
                self._native_pool = None
                self._start_ffmpeg(self._wanted_playing, str(exc))
        else:
            self._follow_decode_size()
        for surface in (self._source_surface, self._distorted_surface):
            surface.refresh_cursor()
            surface.update()

    def _wanted_maximum(self):
        """The largest size FFmpeg's frames are decoded at, in device pixels:
        when fitted, the view's, because no more of a frame is ever shown and
        FFmpeg scales better than painting does; when zoomed, the
        comparison's own (None). Don't decode at the screen's size: 4K HDR
        then plays at 22-26 frames a second, each 3840x2160 (33 MB) for a
        view of 2166x959 device pixels."""
        if self._zoom.factor is not None:
            return None
        ratio = self.devicePixelRatioF()
        return max(2, round(self.width() * ratio)), max(2, round(self.height() * ratio))

    def _follow_decode_size(self) -> None:
        """Decodes FFmpeg's frames again when the size wanted gives any of
        them a new size: when fitted, the view's, once a resize has settled
        (so that they are copied as they are, not scaled by the painting);
        when zoomed, their own."""
        if not self._pool_active:
            return
        wanted = self._wanted_maximum()
        if wanted == self._pool_maximum:
            return
        if self._decode_size_changes(wanted):
            self._decode_at_new_size()
        else:
            self._pool_maximum = wanted  # the same frames: they go on decoding

    def _decode_at_new_size(self) -> None:
        """Starts FFmpeg's decoders again for frames of another size. The
        frame on screen stays until theirs come (1.4-2 s on an RTX 5090, for
        a Vulkan device and the seek); cleared, the view would be dark for
        that long."""
        self._stop_decoder(keep_frame=True)
        self._stop_audio()
        self._start_ffmpeg(self._wanted_playing, self._pool_reason, keep_frame=True)

    def _decode_size_changes(self, maximum) -> bool:
        """Whether decoding with `maximum` would give any of the decoded
        videos other frame sizes: the source, the selected test video, and
        the test videos decoded beside it, whose workers are kept for a
        switch."""
        if self._comparison is None or maximum == self._pool_maximum:
            return False
        decoded = [comparison for comparison, _side in self._desired.values()] or [
            self._comparison, self._source_recipe()]
        return any(playback_dimensions(comparison, maximum) != playback_dimensions(comparison, self._pool_maximum)
                   for comparison in decoded)

    def closeEvent(self, event):
        """Stops playback, and closes once every worker has stopped."""
        self.clear()
        if self.live_workers():
            event.ignore()
            QTimer.singleShot(20, self.close)
            return
        super().closeEvent(event)

    @staticmethod
    def can_play(comparison: FrameComparison) -> tuple[bool, str]:
        """Whether the pair can be played, and if not, why."""
        if comparison.resample_target is not None:
            return (
                False,
                "Video playback is unavailable for synthetic resolution tests; "
                "their processed frames are displayed automatically.",
            )
        source = frame_input_path(comparison, "source")
        distorted = frame_input_path(comparison, "distorted")
        missing = next((path for path in (source, distorted) if not _found_on_disk(path)), None)
        if missing is not None:
            return False, f"Video file is missing: {missing}"
        problem = untimed_pair_problem(comparison.source_info, comparison.distorted_info)
        if problem is not None:
            return False, problem
        return True, ""

    @property
    def is_playing(self) -> bool:
        """Whether it is playing now."""
        return self._is_playing

    @property
    def playback_requested(self) -> bool:
        """Whether playing was asked for (it may still be buffering)."""
        return self._wanted_playing

    @property
    def position(self) -> int:
        """The position shown, in ms from the first frame."""
        if self._comparison is None or self._comparison.fps <= 0:
            return 0
        return round(self._frame / self._comparison.fps * 1000)

    def live_workers(self) -> list[QThread]:
        """The decoders still running, those stopping included."""
        running = [worker for worker in self._retired_workers if worker.isRunning()]
        return list(dict.fromkeys([*running, *self._pool.values()]))

    def _plays_as_is(self, series) -> bool:
        """Whether `series` is the pairs playing, each as it is or with black
        bars found since that change nothing: then they play as they are.
        Loaded afresh instead, every decoder would start again (seconds at
        4K), and the first switch to another test video after a pair's bars
        were found took that long."""
        if len(self._series) != len(series):
            return False
        return all(playing == now or (playing.auto_crop_pending and not now.auto_crop_pending
                                      and crops_change_nothing(playing, now))
                   for playing, now in zip(self._series, series, strict=True))

    def load(self, comparison, position_ms, *, playing=False, color_settings=None, series=None, source_native=True):
        """Shows a comparison (one of `series`) at `position_ms`. A different
        test video of the same series keeps the decoders running; so does a
        series whose black bars, found since, change nothing
        (crops_change_nothing). Returns False if it cannot be played."""
        available, reason = self.can_play(comparison)
        if not available:
            self.clear()
            self.status_changed.emit(reason)
            return False
        series = list(series or [comparison])
        settings = color_settings or PreviewColorSettings()
        selected = series.index(comparison)
        if (self._color_settings == settings and self._comparison is not None
                and self._source_native == source_native and self._plays_as_is(series)):
            timestamp = self.position
            changed = self._selected != selected
            self._selected = selected
            self._comparison = self._series[selected]  # what plays, as it plays it
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
                    self._audio_resume_at = 0.0  # a start: the soundtrack with the first frame
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
        """Stops playback and shows nothing."""
        self._wanted_playing = False
        self._is_playing = False
        self._comparison = None
        self._stop_decoder()
        self._stop_audio()
        self.playing_changed.emit(False)
        self._series = []

    def set_position(self, position_ms: int) -> None:
        """Seeks to `position_ms`."""
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
        """Plays or pauses."""
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
        # The status is updated at once. Updated only as a frame is shown, it
        # would go on saying "Playing" when paused (by minimizing the window,
        # say).
        self._pool_status()
        self._audio_due = False
        self._audio_resume_at = 0.0
        self._clock_started = time.monotonic() if playing and not self._buffering else None
        if not playing:
            self._stop_audio()
        elif not self._buffering:
            self._start_audio(self._frame)
        self._wake()
        self.playing_changed.emit(playing)

    def show_source(self, showing: bool) -> None:
        """Shows the source's frame (True) or the test video's."""
        self._showing_source = bool(showing)
        if self._native_pool is not None:
            self._native_pool.show_source(showing)
        if self._pool_active:
            surface = self._source_surface if showing else self._distorted_surface
            surface.raise_()
            surface.update()  # its frame is current, but it was not repainted while covered

    def set_audio_enabled(self, enabled: bool) -> None:
        """Turns the soundtrack on or off."""
        self._audio_enabled = bool(enabled)
        if self._native_pool is not None:
            self._native_pool.set_audio_enabled(enabled)
            return
        if not enabled:
            self._stop_audio()
        elif self._is_playing:
            self._start_audio(self._frame)

    def set_color_settings(self, settings: PreviewColorSettings) -> None:
        """Changes how HDR is shown, decoding again if it changed."""
        if settings == self._color_settings:
            return
        self._color_settings = settings
        if self._comparison is not None:
            self._restart_decoder(realtime=self._wanted_playing)

    def _restart_decoder(self, *, realtime):
        """Starts playback again from the current frame: natively where it can, else with FFmpeg."""
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

    def _start_ffmpeg(self, realtime, reason="", *, keep_frame=False):
        """Starts playback with FFmpeg workers, saying why native playback is
        not used (`reason`)."""
        self._pool_reason = reason
        self._pool_maximum = self._wanted_maximum()
        self._pool_active = True
        self._wanted_playing = bool(realtime)
        self._is_playing = bool(realtime)
        self._clock_frame = round(self.position / 1000 * self._series[0].fps)
        self._clock_started = None
        self._buffering = True
        self._audio_due = False
        self._audio_resume_at = 0.0
        self._presented = -1
        for surface in (self._source_surface, self._distorted_surface):
            if not keep_frame:
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
        """Sends the status line, when it changed."""
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
        """Sets how many test videos are kept decoding, at once for whatever
        is playing: extra neighbours are stopped and missing ones started,
        and the selected pair is never touched."""
        count = max(1, int(count))
        if count == self._decoded_videos:
            return
        self._decoded_videos = count
        if self._native_pool is not None:
            self._native_pool.sync(self._selected)
        elif self._pool_active:
            self._sync_pool()

    def _source_recipe(self):
        """The source's playback comparison, at the series' frame rate."""
        return replace(source_playback_comparison(self._comparison, self._source_native),
                       fps=self._series[0].fps)

    def _sync_pool(self):
        """Keeps the FFmpeg workers still wanted, stops the others and starts the missing ones."""
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
                # One that has finished (at the end of a short clip, or
                # failed) goes now: kept, it held its queued frames until the
                # window closed.
                if worker.isRunning():
                    self._retired_workers.add(worker)
                else:
                    worker.deleteLater()
                self._history.pop(key, None)
                self._details.pop(key, None)
                self._failures.pop(key, None)
        self._launch_missing()

    def _launch_missing(self):
        """Starts the FFmpeg workers wanted but not running, within the decoder limit."""
        if not self._pool_active:
            return
        # Workers still stopping count too, so that quick switching never
        # runs one decoder more than allowed while an old process exits.
        occupied = sum(w.isRunning() for w in self._retired_workers) + len(self._pool)
        # The pair on screen comes first: the test videos decoded beside it
        # start once it has shown a frame (_tick calls this again). Started
        # with it, two of them slowed its first frame at the tab's first
        # opening from 1.1-1.3 s to 1.5-1.6 s.
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
        """Keeps a worker's description for the status line."""
        if self._pool.get(key) is worker:
            self._details[key] = detail

    def _stream_failed(self, key, worker, error):
        """Reports that a worker's video could not play."""
        if self._pool.get(key) is worker:
            self._failures[key] = error
            # The source is always part of the pair shown; a test video is
            # named by its number in the list, not by the view's key.
            self.status_changed.emit(f"Selected video failed: {error}" if key[0] == "source"
                                     else f"Video {key[1] + 1} could not play: {error}")

    def _stream_finished(self, key, worker):
        """Cleans up after a worker that ended, and starts any missing one."""
        self._retired_workers.discard(worker)
        # A worker that reached the end of a short clip keeps its queued
        # frames until they are shown. A worker no longer wanted can go now.
        if self._pool.get(key) is not worker:
            worker.deleteLater()
        self._launch_missing()

    def _target_frame(self):
        """The frame due now by the playback clock."""
        if self._clock_started is None or not self._wanted_playing:
            return self._clock_frame
        return self._clock_frame + int((time.monotonic() - self._clock_started) * self._series[0].fps)

    def _tick(self):
        """Advances playback: natively, or by taking each worker's frames and
        showing the newest pair both videos have."""
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
                if self._wanted_playing and not pool.buffering and not pool.ended:
                    # The next tick when the next frame is due, if ticking at
                    # all: it is never started here, while off screen
                    # (hideEvent).
                    if self._pool_timer.isActive():
                        due = pool.next_tick_ms()
                        self._pool_timer.start(due if due is not None else self._busy_tick_ms())
                else:
                    self._pace(not self._wanted_playing and pool.pair is not None and pool.pair_index == pool.selected)
            except Exception as exc:
                self._native_pool.stop()
                self._native_pool = None
                self._start_ffmpeg(self._wanted_playing, str(exc))
            return
        if not self._pool_active:
            return
        self._launch_missing()
        target = self._target_frame()
        source_key = next((key for key in self._desired if key[0] == "source"), None)
        distorted_key = ("distorted", self._selected)
        # Don't drop a faster video's frames while the other video is still
        # decoding them. The worker's queue holds the faster one at three
        # frames ahead.
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
            # Three past frames, plus the worker's three frames ahead, limit
            # the memory used, whatever the video's length or the number of
            # files in the list.
            for number in sorted(history)[:-3]:
                payload = history.pop(number)
                if not (self._source_surface.shows(payload) or self._distorted_surface.shows(payload)):
                    worker.recycle(payload)
        source = self._history.get(source_key, {})
        distorted = self._history.get(distorted_key, {})
        common = source.keys() & distorted.keys()
        if not common:
            self._buffering = True
            if self._wanted_playing and target - max(0, self._presented) > 2:
                # Starting (a seek, another test video) is not falling behind:
                # the soundtrack starts with the first frame.
                if self._presented >= 0 and self._clock_started is not None:
                    self._fell_behind()
                else:
                    self._stop_audio()
            error = self._failures.get(source_key) or self._failures.get(distorted_key)
            self._status(f"Selected video failed: {error}" if error else
                         "Buffering selected comparison; adjacent videos are preparing in the background")
            self._pace(bool(error))
            return
        frame = max(common)
        if frame == self._presented:
            if self._wanted_playing and target - frame > 2:
                # Don't let the sound run far ahead while frames keep failing
                # to come. It starts again with the frames, once they keep up.
                self._fell_behind()
                self._buffering = True
            self._pace(not self._wanted_playing and frame >= target)
            return
        self._presented = frame
        fps = self._series[0].fps
        if self._wanted_playing and target - frame > 2:
            # A decoder slower than real time must not build an ever-growing
            # queue, or let the pictures drift away from the sound.
            self._clock_frame = frame
            self._clock_started = time.monotonic()
            self._fell_behind()
            self._buffering = True
        self._frame = round(frame / fps * self._comparison.fps)
        self._source_surface.set_frame(source[frame], playback_dimensions(self._desired[source_key][0], self._pool_maximum))
        self._distorted_surface.set_frame(distorted[frame], playback_dimensions(self._comparison, self._pool_maximum))
        (self._source_surface if self._showing_source else self._distorted_surface).update_frame()
        if self._clock_started is None and self._wanted_playing:
            self._clock_frame = frame
            self._clock_started = time.monotonic()
        if self._buffering and self._wanted_playing:
            self._audio_due = True
        self._buffering = False
        if self._audio_due and self._wanted_playing and time.monotonic() >= self._audio_resume_at:
            self._audio_due = False
            self._start_audio(self._frame)
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
        """Stops playing at the end of the shortest video."""
        counts = [item.frame_count / item.fps for item in self._series if item.frame_count > 0 and item.fps > 0]
        if counts and self._wanted_playing and self.position / 1000 >= min(counts) - 1 / self._series[0].fps:
            self.set_playing(False)
            self._status("Playback ended")

    def _start_audio(self, frame: int) -> None:
        """Plays the source's soundtrack from a frame, with FFmpeg."""
        comparison = self._comparison
        if self._pool_active and self._series:
            # One soundtrack plays across switches between videos: this is a
            # comparison of pictures, and several soundtracks at once would
            # mix. It is the source's, from the frame on screen.
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

    def _fell_behind(self) -> None:
        """Stops the soundtrack because the frames fell behind the clock. It
        starts again only once they have kept up for a while."""
        self._stop_audio()
        self._audio_resume_at = time.monotonic() + _AUDIO_RESUME_SECONDS

    def _stop_audio(self) -> None:
        """Stops the soundtrack."""
        process = self._audio_process
        handle = self._audio_handle
        self._audio_process = None
        self._audio_handle = None
        if process is not None:
            # Ended from a thread of its own, because listing its process tree
            # twice would hold the window for 28 ms at each pause.
            threading.Thread(target=_end_audio, args=(process, handle), name="audio-stop", daemon=True).start()

    def _stop_decoder(self, *, keep_frame=False):
        """Stops all decoding, and clears the frame shown unless keep_frame."""
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
        if keep_frame:
            return
        self._source_surface.clear_frame()
        self._distorted_surface.clear_frame()
        self._source_surface.hide()
        self._distorted_surface.hide()
