"""One native presentation surface and one audio-only playback pipeline."""
from __future__ import annotations

from vmaf_app.core.gstreamer_playback import _load_gstreamer


class LockedPresentation:
    def __init__(self, window, device, settings):
        from gi.repository import GstD3D11

        self.gst, video = _load_gstreamer()
        self.pipeline = self.gst.parse_launch(
            "appsrc name=input is-live=true format=time block=false max-buffers=1 "
            "leaky-type=downstream ! d3d11videosink name=output sync=false "
            "enable-last-sample=false force-aspect-ratio=true"
        )
        self.pipeline.set_context(GstD3D11.d3d11_context_new(device))
        self.source = self.pipeline.get_by_name("input")
        self.sink = self.pipeline.get_by_name("output")
        self.sink.set_property("display-format", 24 if settings.display_hdr_enabled else 28)
        video.VideoOverlay.set_window_handle(self.sink, window)
        self.pipeline.set_state(self.gst.State.PLAYING)

    def present(self, sample):
        # push-sample refs the existing GPU buffer and updates caps as needed.
        # No map(), extraction of pixels, or QImage conversion occurs here.
        result = self.source.emit("push-sample", sample)
        if result != self.gst.FlowReturn.OK:
            raise RuntimeError(f"Native presentation rejected a frame: {result}")

    def poll(self):
        message = self.pipeline.get_bus().pop_filtered(self.gst.MessageType.ERROR)
        if message:
            raise RuntimeError(message.parse_error()[0].message)

    def stop(self):
        self.pipeline.set_state(self.gst.State.NULL)


class SingleSoundtrack:
    """Audio-only source soundtrack. Never changes when selecting an encode."""
    def __init__(self, path, start_ms, enabled):
        self.gst, _ = _load_gstreamer()
        self.pipeline = self.gst.ElementFactory.make("playbin3", "comparison-audio")
        self.pipeline.set_property("uri", path.resolve().as_uri())
        # GstPlayFlags: AUDIO | SOFT_VOLUME; no video/text/visualizations.
        self.pipeline.set_property("flags", 2 | 16)
        self.pipeline.set_property("mute", not enabled)
        self.pending = start_ms
        self.ready = False
        self.failed = None
        self.seeking = False
        self.pipeline.set_state(self.gst.State.PAUSED)

    def poll(self):
        bus = self.pipeline.get_bus()
        while (message := bus.pop_filtered(self.gst.MessageType.ERROR | self.gst.MessageType.ASYNC_DONE)):
            if message.type == self.gst.MessageType.ERROR:
                self.failed = message.parse_error()[0].message
                self.pipeline.set_state(self.gst.State.NULL)
            elif self.pending is not None:
                target, self.pending = self.pending, None
                self.seek(target)
            else:
                self.ready, self.seeking = True, False
        ok, position = self.pipeline.query_position(self.gst.Format.TIME)
        return round(position / self.gst.MSECOND) if ok else None

    def seek(self, position):
        self.ready, self.seeking = False, True
        if not self.pipeline.seek_simple(self.gst.Format.TIME,
                self.gst.SeekFlags.FLUSH | self.gst.SeekFlags.ACCURATE,
                max(0, int(position)) * self.gst.MSECOND):
            self.failed = "Could not seek the soundtrack"

    def set_playing(self, playing):
        if not self.failed:
            self.pipeline.set_state(self.gst.State.PLAYING if playing else self.gst.State.PAUSED)

    def set_enabled(self, enabled):
        self.pipeline.set_property("mute", not enabled)

    def stop(self):
        self.pipeline.set_state(self.gst.State.NULL)
