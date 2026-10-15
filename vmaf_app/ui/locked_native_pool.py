"""Plays Video Compare natively (LockedNativePool): GStreamer decodes the
source and the test videos on the GPU, and each test frame is paired with the
source frame of the same number before the one presenter draws either, in
time with the source's soundtrack. The frames are always paired here, never
shown by separate sinks, so the two videos cannot drift apart."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QThread

from vmaf_app.core.frame_extract import comparison_dimensions
from vmaf_app.core.gstreamer_playback import GstComparePipeline, _load_gstreamer, d3d11_device
from vmaf_app.core.locked_presentation import LockedPresentation, SingleSoundtrack
from vmaf_app.core.video_playback import neighbour_indices, source_playback_comparison
from vmaf_app.ui.video_compare_view import _PairedFrameWidget

#: How long playback may run behind its soundtrack, catching up while the
#: sound plays on, before it stops the sound to wait for the decoders
#: (LockedNativePool.poll). On the test PC the window's thread is held up
#: 55-60 ms every 15 s, which leaves the decoders waiting with full queues;
#: stopping and seeking the sound each time would make it break off.
_CATCH_UP_S = 0.2


class _StopNative(QThread):
    """Stops a pipeline (or the presenter, or the sound) on a thread of its
    own, so that the window does not wait for its teardown."""

    def __init__(self, pipeline, parent):
        super().__init__(parent)
        self.pipeline = pipeline

    def run(self):
        """Stops it (QThread.run)."""
        self.pipeline.stop()

    def cancel(self):
        """Does nothing: it is already stopping, and its teardown must finish.

        The window cancels every running worker when it closes, this one
        included, and interrupting D3D11 teardown could destroy resources
        still in use.
        """


class LockedNativePool:
    """Plays Video Compare natively: one GstComparePipeline per video (the
    source, and the selected test video with its neighbours), with each test
    frame paired with the source frame of the same number, timed by the
    source's soundtrack."""

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
        #: Whether the pair wanted is the one on screen (poll, seek). When
        #: paused on it, nothing is buffering, although `buffering` stays set
        #: after a seek made while paused, so that playing again waits for
        #: the sound.
        self.shown = False
        #: Since when the frames shown have been behind the soundtrack (time.monotonic), or None.
        self.late_since = None
        #: The soundtrack clock's reading (ms) that the last poll chose the
        #: frame by, or None without one, and when it was read
        #: (time.monotonic); see next_tick_ms.
        self._clock_ms, self._clock_at = None, 0.0
        self.showing_source = False
        self.entries, self.desired, self.frames, self.eos = {}, {}, {}, set()
        #: Each video's shading (GstComparePipeline.shading), kept when its
        #: decoder goes, so that an encode chosen again before its decoder is
        #: back is not shown unshaded (HDR as SDR).
        self.shadings = {}
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
        """Stops a pipeline, the presenter or the sound on a thread of its own,
        then calls finished. decoder says whether it counts against the
        decoder limit while stopping."""
        worker = _StopNative(pipeline, self.view)
        worker.counts_as_decoder = decoder
        self.view._retired_workers.add(worker)

        def done():
            """Forgets the stopped worker."""
            self.view._retired_workers.discard(worker)
            if finished:
                finished()
            worker.deleteLater()

        worker.finished.connect(done, Qt.QueuedConnection)
        worker.start()

    def sync(self, selected):
        """Selects a test video: keeps the decoders still needed, stops the
        others and starts the missing ones."""
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
        """Starts the decoders wanted but not running, within the decoder limit
        (counting those still stopping)."""
        occupied = len(self.entries) + sum(w.isRunning() and getattr(w, "counts_as_decoder", True)
                                          for w in self.view._retired_workers)
        for key, (comparison, side) in self.desired.items():
            if key in self.entries or occupied >= self.view.decoder_limit or self.closed:
                continue
            player = GstComparePipeline(comparison, self.settings, side, device=self.device)
            self.entries[key] = [player, None, self.position, False]
            self.shadings[key] = player.shading
            self.frames[key] = {}
            player.start(max(0, self.position - round(1000 / self.fps)), True)
            occupied += 1

    def _choose_pair(self, target):
        """Shows the newest frame pair of the source and the selected video due
        at frame target. Returns whether there is one."""
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
            self._present(self.showing_source)
        return True

    def _present(self, source):
        """Draws the pair's source frame (with `source`) or its test frame:
        fitted to the window, or where the zoom puts it, and shaded as its
        video is (GstComparePipeline.shading)."""
        sample = self.pair[0 if source else 1]
        shading = self.shadings.get(self.source_key if source else ("distorted", self.pair_index))
        view = self.view.native_view(sample)
        if shading is not None:
            self.output.present(sample, view, shading)
        elif view is None:
            self.output.present(sample)  # fitted to the window, as always
        else:
            self.output.present(sample, view)

    def place(self):
        """Draws the frame again after the zoom, or the part of the frame in
        view, has changed."""
        self.surface.refresh_cursor()
        if self.pair is not None and self.pair_index == self.selected:
            self._present(self.showing_source)

    def show_source(self, showing):
        """Shows the source's frame (True) or the test video's."""
        self.showing_source = bool(showing)
        if self.pair is not None and self.pair_index == self.selected:
            self._present(self.showing_source)

    def set_audio_enabled(self, enabled):
        """Turns the soundtrack on or off."""
        self.audio.set_enabled(enabled)

    def seek(self, position_ms):
        """Seeks to `position_ms`, keeping the decoders and the presenter. A
        decoder that already holds the frame wanted is not sought."""
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
        self.shown = self._choose_pair(target)
        self.buffering = not self.shown

    def set_playing(self, playing):
        """Plays or pauses."""
        self.playing = bool(playing)
        self.anchor, self.anchor_frame = time.monotonic(), self.frame
        self.late_since = None
        if not playing:
            self.audio.set_playing(False)
            self.audio_running = False

    def poll(self):
        """Advances playback: reads each decoder's frames into its queue, shows
        the pair due by the soundtrack's clock, and stops the sound to wait
        whenever the frames fall behind. Returns the position shown (ms)."""
        self.output.poll()
        self.launch_missing()
        audio_ms = self.audio.poll()
        polled_at = time.monotonic()
        if not self.audio.ready and not self.audio.failed and time.monotonic() > self.audio_deadline:
            self.audio.failed = "Audio preroll timed out"
            self.audio.set_playing(False)
        target, clock_ms = self.frame, None
        if self.playing and not self.buffering:
            if self.audio_running and audio_ms is not None:
                target, clock_ms = round(audio_ms * self.fps / 1000), audio_ms
            else:
                target = self.anchor_frame + int((time.monotonic() - self.anchor) * self.fps)
        # The newest frame both videos shown have, which limits the queues'
        # floor below.
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
            # Keep two frames before the one shown (while playing, before the
            # one about to be shown), for instant S and arrow-key swaps; the
            # pair shown holds its own. While playing, the floor counts from
            # the target: counted from the frame shown, the queue would still
            # be full of past frames when the clock moved two frames on, and
            # one frame in five would be skipped at 120 fps. The floor also
            # stops at how far both videos shown have come: counted from this
            # queue alone, a video decoded ahead of the other (an encode
            # decoded by the GPU, its 4K 10-bit H.264 source by the CPU) would
            # drop frames the other has yet to reach, so the pair waited for
            # would never come, and playback would stall for good.
            floor = min(max(self.frame, target), max(queue, default=self.frame), reach) - 2
            for old in list(queue):
                if old < floor:
                    del queue[old]
            sink = next(iter(player._sinks.values()))
            # Two previous frames, the current one and two upcoming ones: a
            # limit on the GPU surfaces held, while a step back or forward
            # stays instant. Nothing is pulled while the decoder's seek is
            # still to be made (on the Seeker's thread), because its sink can
            # still give frames from before the seek, which a seek back would
            # take for frames to come.
            while len(queue) < 5 and not player.seeking:
                sample = self._pull_sample(key, sink)
                if sample is None:
                    break
                time_ns = player.frame_time(sample)
                if time_ns is None:
                    raise RuntimeError("Video frame has no usable presentation timestamp")
                frame = round(time_ns / self.gst.SECOND * self.fps)
                if frame >= self.frame:
                    queue[frame] = sample
        # The work above can take time with a newly selected decoder
        # (negotiation and preroll), so the soundtrack's clock is read again
        # before the pair is chosen, rather than using the earlier reading.
        if self.playing and not self.buffering and self.audio_running and time.monotonic() - polled_at > 0.002:
            audio_ms = self.audio.poll()
            if audio_ms is not None:
                target, clock_ms, polled_at = round(audio_ms * self.fps / 1000), audio_ms, time.monotonic()
        self._clock_ms, self._clock_at = clock_ms, polled_at
        matched = self.shown = self._choose_pair(target)
        # Falling behind is judged by the soundtrack clock's own reading, not
        # the value carried on between its steps, which picks the frame.
        reading = self.audio.reading_ms if self.audio_running else None
        due = round(reading * self.fps / 1000) if reading is not None else target
        behind = due - self.frame > 1 or (reading is not None and reading - self.position > 1000 / self.fps)
        if not behind:
            self.late_since = None
        elif matched and not self.buffering:  # a moment to catch up, with the sound playing on
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

    def next_tick_ms(self) -> int | None:
        """Returns how long to wait (ms) for the next tick while playing: 1 ms
        after the next frame becomes due, which is half a frame before its
        time, because poll rounds the clock. Ticking every half frame
        instead, most ticks would find nothing to do (250 a second at
        120 fps, 100 at 24). Returns None while that frame is due but not
        decoded yet."""
        frame_ms = 1000 / self.fps
        if self._clock_ms is not None:
            wait = (self.frame + 0.5) * frame_ms - self._clock_ms - (time.monotonic() - self._clock_at) * 1000
        else:
            wait = (self.anchor + (self.frame + 1 - self.anchor_frame) / self.fps - time.monotonic()) * 1000
        if wait <= 0:
            return None
        return min(math.ceil(wait) + 1, math.ceil(frame_ms))

    def _pull_sample(self, key, sink):
        """Takes the next decoded frame from a sink without waiting, or None."""
        return sink.emit("try-pull-sample", 0)

    @property
    def description(self):
        """The status line for the view."""
        # Paused on the pair on screen is not buffering, even after a seek.
        state = ("Buffering locked pair" if not self.shown or (self.buffering and self.playing)
                 else "Frame-locked GPU pair")
        audio = "source soundtrack" if not self.audio.failed else "audio unavailable"
        details = [self.status_details.get(key, "") for key in
                   (self.source_key, ("distorted", self.selected))]
        # The presenter drawing the frames: Vulkan, or Direct3D 11.
        presenter = getattr(self.output, "name", None)
        return f"{state} · {audio} · " + " · ".join(filter(None, [presenter, *details]))

    def resize(self):
        """Fits the drawing surface to the view again."""
        self.surface.setGeometry(self.view.rect())
        self.place()

    def stop(self):
        """Stops every decoder, the sound and the presenter, each on a thread of its own."""
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
