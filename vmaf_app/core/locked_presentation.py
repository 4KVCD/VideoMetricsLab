"""One native presentation surface and one audio-only playback pipeline."""
from __future__ import annotations

import ctypes
import functools
from ctypes import wintypes

from vmaf_app.core.d3d11_tonemap import boxed_pointer
from vmaf_app.core.gstreamer_playback import _load_gstreamer


class _CropMeta(ctypes.Structure):
    """GstVideoCropMeta: its GstMeta (flags, info), then the rectangle."""

    _fields_ = [("flags", ctypes.c_uint), ("info", ctypes.c_void_p), ("x", ctypes.c_uint),
                ("y", ctypes.c_uint), ("width", ctypes.c_uint), ("height", ctypes.c_uint)]


@functools.cache
def _crop_api():
    """GstVideoCropMeta's info and gst_buffer_add_meta, from GStreamer's C
    API: gst_buffer_add_video_crop_meta is a C macro, which Python cannot
    call."""
    info = ctypes.CDLL("gstvideo-1.0-0.dll").gst_video_crop_meta_get_info
    info.restype, info.argtypes = ctypes.c_void_p, []
    add = ctypes.CDLL("gstreamer-1.0-0.dll").gst_buffer_add_meta
    add.restype, add.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return info(), add


def cropped(sample, crop: tuple[int, int, int, int]):
    """`sample` showing only `crop` (x, y, width, height, in its pixels): a
    new buffer on the same GPU memory, with a GstVideoCropMeta, which
    d3d11videosink draws from. The sample itself is left as it was: the
    pair keeps it to show again. Not Buffer.copy(): in PyGObject that is
    the same buffer again (a second reference), and the crop would have
    gone onto the kept one -- GStreamer refused it, as not writable."""
    gst, _video = _load_gstreamer()
    info, add = _crop_api()
    original, copy = sample.get_buffer(), gst.BufferCopyFlags
    buffer = original.copy_region(copy.FLAGS | copy.TIMESTAMPS | copy.META | copy.MEMORY, 0, original.get_size())
    address = add(boxed_pointer(buffer), info, None)
    if not address:
        raise RuntimeError("the frame could not be cropped")
    meta = _CropMeta.from_address(address)
    meta.x, meta.y, meta.width, meta.height = crop
    return gst.Sample.new(buffer, sample.get_caps(), sample.get_segment(), sample.get_info())


@functools.cache
def _user32():
    """user32 with these functions' types: an instance of its own, not
    ctypes.windll.user32, which the whole process shares."""
    user32 = ctypes.WinDLL("user32")
    user32.GetWindow.restype = ctypes.c_void_p
    user32.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.IsWindowEnabled.argtypes = [ctypes.c_void_p]
    user32.EnableWindow.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    return user32


def pass_mouse_through(window: int) -> None:
    """Mouse input over the video goes to `window`, not to the window
    d3d11videosink makes inside it, so that dragging a zoomed frame is the
    app's: Windows' hit testing (WindowFromPoint) skips a disabled child."""
    user32 = _user32()
    child = user32.GetWindow(window, 5)  # GW_CHILD
    while child:
        if user32.IsWindowEnabled(child):
            user32.EnableWindow(child, False)
        child = user32.GetWindow(child, 2)  # GW_HWNDNEXT


class LockedPresentation:
    def __init__(self, window, device, settings):
        from gi.repository import GstD3D11

        self.gst, video = _load_gstreamer()
        self.video, self.window = video, window
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
        # Where in the window the sink draws, once a zoom has placed it.
        self._rectangle = None
        self.pipeline.set_state(self.gst.State.PLAYING)

    def present(self, sample, view=None):
        """`view`: None to fit the frame to the window; else the part of it
        to show and where in the window (VideoCompareView.native_view). The
        sink smooths a zoomed frame's pixels: its sampling-method, set while
        it runs, is not taken up."""
        crop, rectangle = view if view is not None else (None, None)
        if rectangle is None and self._rectangle is not None:
            # Fitted again after a zoom: the whole window. The sink's own
            # reset, a rectangle of -1s, left its window off the middle.
            client = wintypes.RECT()
            _user32().GetClientRect(self.window, ctypes.byref(client))
            rectangle = (0, 0, max(1, client.right), max(1, client.bottom))
        if rectangle != self._rectangle:
            self.video.VideoOverlay.set_render_rectangle(self.sink, *rectangle)
            self._rectangle = rectangle
        if crop is not None:
            sample = cropped(sample, crop)
        # push-sample refs the existing GPU buffer and updates caps as needed.
        # No map(), extraction of pixels, or QImage conversion occurs here.
        result = self.source.emit("push-sample", sample)
        if result != self.gst.FlowReturn.OK:
            raise RuntimeError(f"Native presentation rejected a frame: {result}")

    def poll(self):
        message = self.pipeline.get_bus().pop_filtered(self.gst.MessageType.ERROR)
        if message:
            raise RuntimeError(message.parse_error()[0].message)
        # The sink makes its window when it first draws, after a frame.
        pass_mouse_through(self.window)

    def stop(self):
        self.pipeline.set_state(self.gst.State.NULL)


class SingleSoundtrack:
    """Audio-only source soundtrack. Never changes when selecting an encode.

    Its positions count from the source video's first frame, `offset_ms`
    into the file's timeline (GstComparePipeline.first_frame_ms), as the
    video's frames do: its clock times them."""

    offset_ms = 0

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
        return round(position / self.gst.MSECOND) - self.offset_ms if ok else None

    def set_offset(self, offset_ms, position):
        """Counts from the source video's first frame, `offset_ms` into the
        file's timeline, from now on: at `position` again, unless its first
        seek, which waits for the preroll, has still to come and takes it."""
        self.offset_ms = offset_ms
        if self.pending is None and not self.failed:
            self.seek(position)

    def seek(self, position):
        self.ready, self.seeking = False, True
        if not self.pipeline.seek_simple(self.gst.Format.TIME,
                self.gst.SeekFlags.FLUSH | self.gst.SeekFlags.ACCURATE,
                max(0, int(position) + self.offset_ms) * self.gst.MSECOND):
            self.failed = "Could not seek the soundtrack"

    def set_playing(self, playing):
        if not self.failed:
            self.pipeline.set_state(self.gst.State.PLAYING if playing else self.gst.State.PAUSED)

    def set_enabled(self, enabled):
        self.pipeline.set_property("mute", not enabled)

    def stop(self):
        self.pipeline.set_state(self.gst.State.NULL)
