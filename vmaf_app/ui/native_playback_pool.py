"""Up to four native GPU streams, sharing a GStreamer clock and media origin."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread

from vmaf_app.core.frame_extract import comparison_dimensions
from vmaf_app.core.gstreamer_playback import GstComparePipeline, _load_gstreamer, gstreamer_available
from vmaf_app.core.video_playback import neighbour_indices, source_playback_comparison
from vmaf_app.ui.video_compare_view import _PairedFrameWidget


class _StopNative(QThread):
    def __init__(self, pipeline, parent):
        super().__init__(parent)
        self.pipeline = pipeline

    def run(self):
        self.pipeline.stop()

    def cancel(self):
        """Already stopping: allow teardown to finish, never terminate it.

        The main window cancels all live workers during shutdown, including
        this one. Interrupting D3D11 teardown could destroy live resources.
        """


class NativePlaybackPool:
    def __init__(self, view, series, selected, position_ms, settings, playing, source_native=True):
        available, reason = gstreamer_available()
        if not available:
            raise RuntimeError(reason)
        self.view, self.series, self.settings = view, series, settings
        self.selected = selected
        self.source_native = source_native
        self.gst, _ = _load_gstreamer()
        self.clock = self.gst.SystemClock.obtain()
        self.origin = position_ms
        self.position = position_ms
        self.base = None
        self.paused_at = None
        self.playing = playing
        self.entries = {}
        self.desired = {}
        self.showing_source = False
        self.audio_enabled = view._audio_enabled
        self.error = None
        self.ended = False
        self.status_details = {}
        self.closed = False
        try:
            self.sync(selected)
        except Exception:
            self.stop()
            raise

    def sync(self, selected):
        if selected != self.selected:
            self.ended = False
        self.selected = selected
        comparison = source_playback_comparison(self.series[selected], self.source_native)
        crop = comparison.source_crop
        source_key = ("source", str(comparison.source_info.path), None if crop is None else (crop.w, crop.h, crop.x, crop.y), comparison_dimensions(comparison))
        self.desired = {source_key: (comparison, "source")}
        self.desired.update({("distorted", i): (self.series[i], "distorted")
                             for i in neighbour_indices(len(self.series), selected, self.view.decoded_videos)})
        # Raise the already-running target before any background cleanup.
        self.show_source(self.showing_source)
        for key in list(self.entries):
            if key not in self.desired:
                self._retire(self.entries.pop(key))
                self.status_details.pop(key, None)
        self.launch_missing()
        self.set_audio_enabled(self.audio_enabled)

    def _retire(self, entry):
        pipeline, surface, *_ = entry
        surface.hide()
        worker = _StopNative(pipeline, self.view)
        self.view._retired_workers.add(worker)

        def finished():
            self.view._retired_workers.discard(worker)
            surface.set_native_playback(False)
            surface.deleteLater()
            worker.deleteLater()

        worker.finished.connect(finished, Qt.QueuedConnection)
        worker.start()

    def launch_missing(self):
        if self.closed:
            return
        occupied = len(self.entries) + sum(w.isRunning() for w in self.view._retired_workers)
        for key, (comparison, side) in self.desired.items():
            if key in self.entries or occupied >= self.view.decoder_limit:
                continue
            surface = _PairedFrameWidget(self.view)
            surface.setFocusProxy(self.view)
            surface.setGeometry(self.view.rect())
            surface.set_native_playback(True)
            surface.show()
            handle = int(surface.winId())
            pipeline = None
            try:
                pipeline = GstComparePipeline(comparison, handle, handle, self.settings,
                                              single_side=side, audio_enabled=False)
                pipeline.start(self.position, False)
            except Exception:
                if pipeline is not None:
                    self._retire([pipeline, surface])
                else:
                    surface.deleteLater()
                raise
            self.entries[key] = [pipeline, surface, self.position, False]
            occupied += 1
        self.show_source(self.showing_source)

    def show_source(self, showing):
        self.showing_source = bool(showing)
        key = next((key for key in self.desired if key[0] == "source"), None) if showing else ("distorted", self.selected)
        if key in self.entries:
            self.entries[key][1].raise_()

    def set_audio_enabled(self, enabled):
        self.audio_enabled = bool(enabled)
        for key, entry in self.entries.items():
            entry[0].set_audio_enabled(enabled and key == ("distorted", self.selected))

    def set_playing(self, playing):
        playing = bool(playing)
        if self.playing == playing:
            return
        now = self.clock.get_time()
        if playing and self.paused_at is not None and self.base is not None:
            self.base += now - self.paused_at
            self.paused_at = None
        elif not playing:
            self.paused_at = now
        self.playing = playing
        for pipeline, _, start, active in self.entries.values():
            if active and self.base is not None:
                pipeline.align_clock(self.clock, self.base, start - self.origin)
                pipeline.set_playing(playing)

    def poll(self):
        self.launch_missing()
        for key, entry in self.entries.items():
            pipeline = entry[0]
            update = pipeline.poll()
            if update.error:
                raise RuntimeError(f"{key}: {update.error}")
            if getattr(update, "status", None):
                self.status_details[key] = update.status
            if getattr(update, "ended", False) and (
                key[0] == "source" or key == ("distorted", self.selected)
            ):
                self.ended = True
        needed = [entry for key, entry in self.entries.items()
                  if key[0] == "source" or key == ("distorted", self.selected)]
        if self.base is None and len(needed) == 2 and all(e[0]._ready for e in needed):
            self.base = self.clock.get_time()
            if not self.playing:
                self.paused_at = self.base
        for entry in self.entries.values():
            pipeline, _, start, active = entry
            if not active and pipeline._ready and self.base is not None:
                pipeline.align_clock(self.clock, self.base, start - self.origin)
                pipeline.set_playing(self.playing)
                entry[3] = True
        if self.base is not None:
            now = self.clock.get_time() if self.playing else (self.paused_at or self.base)
            self.position = self.origin + max(0, round((now - self.base) / self.gst.MSECOND))
        self.set_audio_enabled(self.audio_enabled)
        return self.position

    @property
    def description(self):
        return self.status_details.get(("distorted", self.selected), "Preparing GPU video")

    def resize(self):
        for _, surface, *_ in self.entries.values():
            surface.setGeometry(self.view.rect())

    def stop(self):
        self.closed = True
        for entry in self.entries.values():
            self._retire(entry)
        self.entries.clear()
