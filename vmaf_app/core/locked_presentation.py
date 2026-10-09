"""Native playback's presentation -- the app's own, on the decoders' D3D11
device -- and one audio-only playback pipeline."""
from __future__ import annotations

import ctypes
import functools
import threading
import time
from collections import deque

from vmaf_app.core.d3d11_tonemap import boxed_pointer, library_path
from vmaf_app.core.gstreamer_playback import _load_gstreamer

#: The colour beside a frame: the views' background (VideoCompareView).
_BACKGROUND = (0x17 / 255,) * 3 + (1.0,)
#: DXGI colour spaces: SDR (sRGB-coded BT.709) and HDR10 (PQ in BT.2020).
_SDR_SPACE, _HDR10_SPACE = 0, 12


@functools.cache
def _presenter_api():
    """The native presenter's entry points (d3d11_tonemap.dll), and the ones
    of GStreamer's D3D11 library it needs, typed."""
    lib = ctypes.CDLL(str(library_path()))
    ints, floats = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    for name, restype, argtypes in (
        ("vmaf_present_create", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ("vmaf_present_follow", None, [ctypes.c_void_p, ctypes.c_void_p]),
        ("vmaf_present_wait", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint]),
        ("vmaf_present_frame", ctypes.c_int,
         [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ints, ints, floats, ctypes.c_int, floats]),
        ("vmaf_present_destroy", None, [ctypes.c_void_p]),
    ):
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    gst = ctypes.CDLL("gstd3d11-1.0-0.dll")
    for name, restype in (("gst_is_d3d11_memory", ctypes.c_int),
                          ("gst_d3d11_memory_get_resource_handle", ctypes.c_void_p),
                          ("gst_d3d11_memory_get_subresource_index", ctypes.c_uint),
                          ("gst_d3d11_device_get_device_handle", ctypes.c_void_p)):
        function = getattr(gst, name)
        function.restype, function.argtypes = restype, [ctypes.c_void_p]
    return lib, gst


def _object_pointer(instance) -> int:
    """The GObject a PyGObject wrapper stands for."""
    capsule = ctypes.pythonapi.PyCapsule_GetPointer
    capsule.restype, capsule.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return capsule(instance.__gpointer__, None)


def yuv_to_rgb(matrix: str, full_range: bool, ten_bit: bool) -> tuple[float, ...]:
    """The presenter's rows of Y, U, V coefficients and an offset, for a
    frame's planes as its shader samples them: NV12's 8-bit codes over 255,
    P010's 10-bit codes in the top of 16 bits over 65535. `matrix`: GStreamer's
    name for the YUV matrix (bt709, bt601, bt2020...)."""
    kr, kb = {"bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593), "smpte240m": (0.212, 0.087),
              "fcc": (0.30, 0.11)}.get(matrix, (0.2126, 0.0722))
    kg = 1 - kr - kb
    bits = 10 if ten_bit else 8
    code = 65535 / 64 if ten_bit else 255.0  # codes in a sampled value of 1
    step = 1 << (bits - 8)
    if full_range:
        y0, y_range, c0, c_range = 0, (1 << bits) - 1, 1 << (bits - 1), (1 << bits) - 1
    else:
        y0, y_range, c0, c_range = 16 * step, 219 * step, 128 * step, 224 * step
    ay, by = code / y_range, -y0 / y_range
    ac, bc = code / c_range, -c0 / c_range
    gu, gv = -2 * kb * (1 - kb) / kg, -2 * kr * (1 - kr) / kg
    return (ay, 0.0, 2 * (1 - kr) * ac, by + 2 * (1 - kr) * bc,
            ay, gu * ac, gv * ac, by + (gu + gv) * bc,
            ay, 2 * (1 - kb) * ac, 0.0, by + 2 * (1 - kb) * bc)


class LockedPresentation:
    """Native playback's frames, drawn and presented by the app's own
    presenter (d3d11_tonemap.dll's vmaf_present_*): a waitable flip-model
    swapchain in a window of its own over `window`, on the decoders' D3D11
    device. A thread waits until the swapchain takes a frame, holding no
    lock, then draws the one due and presents it holding the device's: the
    present has nothing to wait for. GStreamer's d3d11videosink, which this
    replaced, waited for the display's refresh holding that lock whenever it
    presented at the display's own rate, and the decoders, which need it for
    each frame, stalled up to 40 ms: 120 fps video showed some 100 frames a
    second, and its sound stopped now and then to wait for them."""

    def __init__(self, window, device, settings):
        self._video = _load_gstreamer()[1]
        self.window, self.device = window, device
        self._lib, self._gst = _presenter_api()
        handle = self._gst.gst_d3d11_device_get_device_handle(_object_pointer(device))
        device.lock()
        try:
            self._presenter = self._lib.vmaf_present_create(handle, window, int(settings.display_hdr_enabled is True))
        finally:
            device.unlock()
        if not self._presenter:
            raise RuntimeError("Could not set up the native video presentation")
        # The frames to draw, one each time the swapchain takes one, in order:
        # two at most, the oldest dropped past that (_present_paced).
        self._due = deque()
        self._due_lock = threading.Lock()
        self._due_ready = threading.Event()
        self._stopping = False
        self._error = None
        self._colours = {}
        self._clear = (ctypes.c_float * 4)(*_BACKGROUND)
        self._presenter_thread = threading.Thread(target=self._present_paced, name="native-present", daemon=True)
        self._presenter_thread.start()

    def present(self, sample, view=None):
        """`sample` shown at the next refresh. `view`: None to fit it to the
        window, letterboxed; else the part of it to show and where in the
        window, in device pixels (VideoCompareView.native_view)."""
        self._lib.vmaf_present_follow(self._presenter, self.window)
        with self._due_lock:
            self._due.append((sample, view))
            while len(self._due) > 2:
                self._due.popleft()
            self._due_ready.set()

    def _present_paced(self):
        """The frames due, one each time the swapchain takes one -- once per
        refresh -- in order: chosen 4 ms apart at times, two within one
        refresh would otherwise lose one. A frame it cannot draw ends it,
        the error left for poll(): the picture would stop with no word."""
        while True:
            self._due_ready.wait()
            if self._stopping:
                return
            if self._lib.vmaf_present_wait(self._presenter, 100) != 0:
                continue  # not taken yet (a hidden window): asked again
            with self._due_lock:
                sample, view = self._due.popleft()
                if not self._due:
                    self._due_ready.clear()
            if self._stopping:
                return
            try:
                self._draw(sample, view)
            except Exception as exc:
                self._error = f"Native presentation failed: {exc}"
            if self._error:
                return

    def _draw(self, sample, view):
        buffer = sample.get_buffer()
        if buffer.n_memory() != 1:
            self._error = "Native presentation needs one GPU texture a frame"
            return
        memory = buffer.peek_memory(0)
        # Kept alive (``memory``) until the native calls return.
        pointer = boxed_pointer(memory)
        if not self._gst.gst_is_d3d11_memory(pointer):
            self._error = "Native presentation received CPU memory instead of a D3D11 texture"
            return
        to_rgb, space, size = self._colours_of(sample.get_caps())
        crop, rectangle = view if view is not None else (None, None)
        # The frame's own size, not its texture's: a decoder's is padded
        # (1608 rows to 1616), and its padding was drawn as picture.
        source = (ctypes.c_int * 4)(*(crop if crop is not None else (0, 0, *size)))
        target = (ctypes.c_int * 4)(*rectangle) if rectangle is not None else None
        self.device.lock()
        try:
            resource = self._gst.gst_d3d11_memory_get_resource_handle(pointer)
            # The frame's slice of its texture: H.264's decoder gives arrays.
            subresource = self._gst.gst_d3d11_memory_get_subresource_index(pointer)
            result = self._lib.vmaf_present_frame(self._presenter, resource, subresource, source, target, to_rgb,
                                                  space, self._clear)
        finally:
            self.device.unlock()
        if result < 0:
            self._error = f"Native presentation failed: 0x{result & 0xffffffff:08x}"

    def _colours_of(self, caps):
        """(YUV-to-RGB rows or None, DXGI colour space, (width, height)) for
        frames of `caps`, worked out once."""
        key = caps.to_string()
        colours = self._colours.get(key)
        if colours is None:
            info = self._video.VideoInfo.new_from_caps(caps)
            name, colorimetry = info.finfo.format.value_nick, info.colorimetry
            rows = None
            if name in ("nv12", "p010-10le"):
                rows = (ctypes.c_float * 12)(*yuv_to_rgb(colorimetry.matrix.value_nick,
                                                         colorimetry.range.value_nick == "0-255",
                                                         name == "p010-10le"))
            space = _HDR10_SPACE if colorimetry.transfer.value_nick == "smpte2084" else _SDR_SPACE
            colours = self._colours[key] = (rows, space, (info.width, info.height))
        return colours

    def poll(self):
        if self._error:
            raise RuntimeError(self._error)

    def stop(self):
        """On any thread: the window goes with the view's (its owner's).
        Left, not freed, if its thread were still drawing a second on."""
        self._stopping = True
        self._due_ready.set()
        self._presenter_thread.join(timeout=1)
        if self._presenter and not self._presenter_thread.is_alive():
            self.device.lock()
            try:
                self._lib.vmaf_present_destroy(self._presenter)
            finally:
                self.device.unlock()
            self._presenter = None


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
        self.playing = False
        #: The clock's last reading (ms) and when it changed (time.monotonic),
        #: the smallest step it has moved in, and the last value given (poll).
        self._clock = (None, 0.0)
        self._step_ms = 10.0
        self._given_ms = None
        #: The clock's own last reading (ms, from the video's first frame),
        #: not carried on: what falling behind is judged by (LockedNativePool).
        self.reading_ms = None
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
        if not ok:
            self.reading_ms = None
            return None
        # Its sink's clock moves in steps of its buffer, 10 ms here, where a
        # frame of 120 fps video lasts 8.3 ms: one frame in five was skipped.
        # Between steps it is carried on by the time passed, never further
        # than a step, and never back while playing.
        position_ms, now = position / self.gst.MSECOND, time.monotonic()
        self.reading_ms = position_ms - self.offset_ms
        last_ms, changed_at = self._clock
        if position_ms != last_ms:
            if self.playing and last_ms is not None and 5 <= position_ms - last_ms < self._step_ms:
                self._step_ms = position_ms - last_ms
            self._clock = (position_ms, now)
        elif self.playing:
            position_ms += min((now - changed_at) * 1000, self._step_ms)
        if self.playing and self._given_ms is not None:
            position_ms = max(position_ms, self._given_ms)
        self._given_ms = position_ms if self.playing else None
        return position_ms - self.offset_ms

    def set_offset(self, offset_ms, position):
        """Counts from the source video's first frame, `offset_ms` into the
        file's timeline, from now on: at `position` again, unless its first
        seek, which waits for the preroll, has still to come and takes it."""
        self.offset_ms = offset_ms
        if self.pending is None and not self.failed:
            self.seek(position)

    def seek(self, position):
        self.ready, self.seeking = False, True
        self._clock, self._given_ms = (None, 0.0), None
        if not self.pipeline.seek_simple(self.gst.Format.TIME,
                self.gst.SeekFlags.FLUSH | self.gst.SeekFlags.ACCURATE,
                max(0, int(position) + self.offset_ms) * self.gst.MSECOND):
            self.failed = "Could not seek the soundtrack"

    def set_playing(self, playing):
        if not self.failed:
            if bool(playing) != self.playing:
                # Carried on from now: from its last change, before a pause,
                # it went a whole step on at once.
                self._clock, self._given_ms = (None, 0.0), None
            self.playing = bool(playing)
            self.pipeline.set_state(self.gst.State.PLAYING if playing else self.gst.State.PAUSED)

    def set_enabled(self, enabled):
        self.pipeline.set_property("mute", not enabled)

    def stop(self):
        self.pipeline.set_state(self.gst.State.NULL)
