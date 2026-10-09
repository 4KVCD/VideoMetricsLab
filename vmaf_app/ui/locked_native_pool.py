"""Pair GPU samples before a single renderer; never compare independent sinks."""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QThread

from vmaf_app.core.frame_extract import comparison_dimensions
from vmaf_app.core.gstreamer_playback import GstComparePipeline, _load_gstreamer, d3d11_device
from vmaf_app.core.locked_presentation import LockedPresentation, SingleSoundtrack
from vmaf_app.core.video_playback import neighbour_indices, source_playback_comparison
from vmaf_app.ui.video_compare_view import _PairedFrameWidget

#: How long playback may run behind its soundtrack, catching up with the
#: sound playing on, before it stops the sound to wait for the decoders
#: (LockedNativePool.poll). The window's thread, held up 55-60 ms every 15 s
#: on the test PC, left the decoders waiting with their queues full; stopped
#: and sought again each time, the sound broke off.
_CATCH_UP_S = 0.2


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


class LockedNativePool:
    def __init__(self, view, series, selected, position_ms, settings, playing, source_native=True):
        self.gst, _ = _load_gstreamer()
        self.device = d3d11_device()
        self.view, self.series, self.settings = view, series, settings
        self.source_native = source_native
        self.selected = selected
        self.fps = series[0].fps
        if self.fps <= 0 or any(abs(c.distorted_info.fps - self.fps) > .01 for c in series):
            raise RuntimeError("Native frame lock currently requires matching frame rates")
        self.position = position_ms
        self.frame = round(position_ms * self.fps / 1000)
        self.playing, self.ended, self.closed = playing, False, False
        self.buffering = True
        #: Since when the frames shown have been behind the soundtrack (time.monotonic), or None.
        self.late_since = None
        self.showing_source = False
        self.entries, self.desired, self.frames, self.eos = {}, {}, {}, set()
        self.pair = None
        self.pair_index = None
        self.anchor = time.monotonic()
        self.anchor_frame = self.frame
        self.audio_running = False
        self.audio_deadline = time.monotonic() + 15
        self.status_details = {}
        self.surface = _PairedFrameWidget(view)
        self.surface.setFocusProxy(view)
        self.surface.setGeometry(view.rect())
        self.surface.set_native_playback(True)
        self.surface.show()
        self.output = self.audio = None
        try:
            self.output = LockedPresentation(int(self.surface.winId()), self.device, settings)
            self.audio = SingleSoundtrack(series[0].source_info.path, position_ms, view._audio_enabled)
            self.sync(selected)
        except Exception:
            self.stop()
            raise

    def _retire(self, pipeline, finished=None, *, decoder=True):
        worker = _StopNative(pipeline, self.view)
        worker.counts_as_decoder = decoder
        self.view._retired_workers.add(worker)

        def done():
            self.view._retired_workers.discard(worker)
            if finished:
                finished()
            worker.deleteLater()

        worker.finished.connect(done, Qt.QueuedConnection)
        worker.start()

    def sync(self, selected):
        self.selected = selected
        source = source_playback_comparison(self.series[selected], self.source_native)
        crop = source.source_crop
        self.source_key = ("source", str(source.source_info.path),
                           None if crop is None else (crop.w, crop.h, crop.x, crop.y),
                           comparison_dimensions(source))
        self.desired = {self.source_key: (source, "source")}
        self.desired.update({("distorted", i): (self.series[i], "distorted")
                             for i in neighbour_indices(len(self.series), selected, self.view.decoded_videos)})
        for key in list(self.entries):
            if key not in self.desired:
                self._retire(self.entries.pop(key)[0])
                self.frames.pop(key, None)
                self.eos.discard(key)
        self.launch_missing()
        self._choose_pair(self.frame)

    def launch_missing(self):
        occupied = len(self.entries) + sum(w.isRunning() and getattr(w, "counts_as_decoder", True)
                                          for w in self.view._retired_workers)
        for key, (comparison, side) in self.desired.items():
            if key in self.entries or occupied >= self.view.decoder_limit or self.closed:
                continue
            player = GstComparePipeline(comparison, self.settings, side, device=self.device)
            self.entries[key] = [player, None, self.position, False]
            self.frames[key] = {}
            player.start(max(0, self.position - round(1000 / self.fps)), True)
            occupied += 1

    def _choose_pair(self, target):
        source = self.frames.get(self.source_key, {})
        distorted = self.frames.get(("distorted", self.selected), {})
        common = source.keys() & distorted.keys()
        due = [f for f in common if f <= target]
        if not due:
            return False
        frame = max(due)
        if frame < self.frame:
            return False
        changed = frame != self.frame or self.pair_index != self.selected or self.pair is None
        self.frame, self.position = frame, round(frame * 1000 / self.fps)
        self.pair, self.pair_index = (source[frame], distorted[frame]), self.selected
        if changed:
            self._present(self.pair[0 if self.showing_source else 1])
        return True

    def _present(self, sample):
        view = self.view.native_view(sample)
        if view is None:
            self.output.present(sample)  # fitted to the window, as always
        else:
            self.output.present(sample, view)

    def place(self):
        """The zoom, or the part of the frame in view, changed."""
        self.surface.refresh_cursor()
        if self.pair is not None and self.pair_index == self.selected:
            self._present(self.pair[0 if self.showing_source else 1])

    def show_source(self, showing):
        self.showing_source = bool(showing)
        if self.pair is not None and self.pair_index == self.selected:
            self._present(self.pair[0 if showing else 1])

    def set_audio_enabled(self, enabled):
        self.audio.set_enabled(enabled)

    def seek(self, position_ms):
        """Reuse decoders and renderer; nearby paired samples need no video seek."""
        target = max(0, round(position_ms * self.fps / 1000))
        if target == self.frame:
            return
        self.audio.set_playing(False)
        self.audio_running = False
        self.audio.seek(round(target * 1000 / self.fps))
        self.audio_deadline = time.monotonic() + 15
        self.ended = False
        self.frame = target
        self.position = round(target * 1000 / self.fps)
        self.anchor, self.anchor_frame = time.monotonic(), target
        self.late_since = None
        self.pair = None
        self.pair_index = None
        for key, entry in self.entries.items():
            queue = self.frames[key]
            if target not in queue:
                queue.clear()
                self.eos.discard(key)
                # Start one frame early: rounding a fractional frame time to
                # milliseconds must not seek just beyond the requested frame.
                entry[0].seek(max(0, self.position - round(1000 / self.fps)))
        self.buffering = not self._choose_pair(target)

    def set_playing(self, playing):
        self.playing = bool(playing)
        self.anchor, self.anchor_frame = time.monotonic(), self.frame
        self.late_since = None
        if not playing:
            self.audio.set_playing(False)
            self.audio_running = False

    def poll(self):
        self.output.poll()
        self.launch_missing()
        audio_ms = self.audio.poll()
        if not self.audio.ready and not self.audio.failed and time.monotonic() > self.audio_deadline:
            self.audio.failed = "Audio preroll timed out"
            self.audio.set_playing(False)
        target = self.frame
        if self.playing and not self.buffering:
            target = round(audio_ms * self.fps / 1000) if self.audio_running and audio_ms is not None else (
                self.anchor_frame + int((time.monotonic() - self.anchor) * self.fps))
        # The newest frame both videos shown have: the queues' floor below.
        reach = min(max(self.frames.get(key, ()), default=self.frame)
                    for key in (self.source_key, ("distorted", self.selected)))
        for key, entry in self.entries.items():
            player = entry[0]
            update = player.poll()
            if update.error:
                raise RuntimeError(update.error)
            first_ms = player.first_frame_ms if key == self.source_key else None
            if first_ms is not None and first_ms != self.audio.offset_ms and not self.audio_running:
                # The soundtrack's clock counts from the source's first
                # frame too, known from its first preroll, before the sound
                # starts.
                self.audio.set_offset(first_ms, self.position)
            if update.status:
                self.status_details[key] = update.status
            if update.ended:
                self.eos.add(key)
            queue = self.frames[key]
            # Two frames before the one shown -- or, playing, before the one
            # about to be, as far as this queue and both videos shown have
            # come -- kept for immediate S/arrow swaps (the pair shown holds
            # its own). Counted from the frame shown while playing, the queue
            # was still full of the past when the clock moved two frames on,
            # and the frame due waited in the sink: one in five skipped at 120
            # fps. Counted as far as this queue alone had come, a video
            # decoded ahead of the other -- an encode decoded by the GPU, its
            # 4K 10-bit H.264 source by the CPU -- dropped the frames the other
            # had yet to reach: the pair waited for could never come, both
            # queues full, and playback stalled for good.
            floor = min(max(self.frame, target), max(queue, default=self.frame), reach) - 2
            for old in list(queue):
                if old < floor:
                    del queue[old]
            sink = next(iter(player._sinks.values()))
            # Two previous frames plus the current and two upcoming frames.
            # Bound retained GPU surfaces while making +/- frame reversible.
            while len(queue) < 5:
                sample = self._pull_sample(key, sink)
                if sample is None:
                    break
                time_ns = player.frame_time(sample)
                if time_ns is None:
                    raise RuntimeError("Video frame has no usable presentation timestamp")
                frame = round(time_ns / self.gst.SECOND * self.fps)
                if frame >= self.frame:
                    queue[frame] = sample
        # Negotiation/preroll work above can take time on a newly selected
        # decoder. Sample the audio clock again immediately before choosing
        # the visible pair, not using its pre-negotiation value.
        if self.playing and not self.buffering and self.audio_running:
            audio_ms = self.audio.poll()
            if audio_ms is not None:
                target = round(audio_ms * self.fps / 1000)
        matched = self._choose_pair(target)
        # Falling behind is judged by the soundtrack clock's own reading, not
        # the value carried on between its steps, which picks the frame.
        reading = self.audio.reading_ms if self.audio_running else None
        due = round(reading * self.fps / 1000) if reading is not None else target
        behind = due - self.frame > 1 or (reading is not None and reading - self.position > 1000 / self.fps)
        if not behind:
            self.late_since = None
        elif matched and not self.buffering:  # a moment to catch up in, the sound playing on
            now = time.monotonic()
            if self.late_since is None:
                self.late_since = now
            behind = now - self.late_since > _CATCH_UP_S
        if not matched or behind:
            self.late_since = None
            if not self.buffering:
                self.audio.set_playing(False)
                self.audio_running = False
                if not self.audio.failed:
                    self.audio.seek(self.position)
                    self.audio_deadline = time.monotonic() + 15
            self.buffering = True
        elif self.playing and (self.audio.ready or self.audio.failed):
            if self.buffering:
                self.anchor, self.anchor_frame = time.monotonic(), self.frame
            self.buffering = False
            if not self.audio_running and not self.audio.failed:
                self.audio.set_playing(True)
                self.audio_running = True
        required = (self.source_key, ("distorted", self.selected))
        if any(k in self.eos and not any(f > self.frame for f in self.frames[k]) for k in required):
            self.ended = True
            self.audio.set_playing(False)
        return self.position

    def _pull_sample(self, key, sink):
        return sink.emit("try-pull-sample", 0)

    @property
    def description(self):
        state = "Buffering locked pair" if self.buffering else "Frame-locked GPU pair"
        audio = "source soundtrack" if not self.audio.failed else "audio unavailable"
        details = [self.status_details.get(key, "") for key in
                   (self.source_key, ("distorted", self.selected))]
        return f"{state} · {audio} · " + " · ".join(filter(None, details))

    def resize(self):
        self.surface.setGeometry(self.view.rect())
        self.place()

    def stop(self):
        if self.closed:
            return
        self.closed = True
        for entry in self.entries.values():
            self._retire(entry[0])
        self.entries.clear()
        self.frames.clear()
        self.pair = None
        if self.audio:
            self.audio.set_enabled(False)
            self._retire(self.audio, decoder=False)
        self.surface.hide()
        if self.output:
            self._retire(self.output, self.surface.deleteLater, decoder=False)
        else:
            self.surface.deleteLater()
