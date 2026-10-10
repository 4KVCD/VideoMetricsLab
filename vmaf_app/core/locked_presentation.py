"""How native playback shows its frames, and plays the source's soundtrack.

LockedPresentation draws and presents the decoded frames with the app's own
presenter: Direct3D 11, or Vulkan when it is asked for. It runs on the GPU
of the decoders' D3D11 device. SingleSoundtrack plays the source video's
audio on its own pipeline.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import threading
import time
from collections import deque

from vmaf_app.core import d3d11_tonemap, vulkan_presenter
from vmaf_app.core.d3d11_tonemap import boxed_pointer, object_pointer
from vmaf_app.core.gstreamer_playback import Seeker, _load_gstreamer

_LOG = logging.getLogger(__name__)

#: The colour around a frame: the views' background (VideoCompareView).
_BACKGROUND = (0x17 / 255,) * 3 + (1.0,)
#: DXGI colour spaces: SDR (sRGB-coded BT.709) and HDR10 (PQ in BT.2020).
_SDR_SPACE, _HDR10_SPACE = 0, 12
#: The presenters that could not start in this process, so that each failure
#: is logged only once.
_PRESENTERS_FAILED: set[str] = set()


@functools.cache
def _presenter_api(path: str):
    """Loads a native presenter and declares its functions' types. The two
    presenters, vulkan_presenter.dll and d3d11_tonemap.dll, have the same
    functions."""
    lib = ctypes.CDLL(path)
    ints, floats = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    for name, restype, argtypes in (
        ("vmaf_present_create", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
        ("vmaf_present_follow", None, [ctypes.c_void_p, ctypes.c_void_p]),
        ("vmaf_present_wait", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint]),
        ("vmaf_present_texture", ctypes.c_int,
         [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ints, ints, floats, floats, ctypes.c_int, floats]),
        ("vmaf_present_destroy", None, [ctypes.c_void_p]),
    ):
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    # Only Vulkan's presenter has these: the reason a call failed, making the
    # swapchain before the first frame, and drawing and presenting a frame
    # after the device lock is released.
    problem = getattr(lib, "vmaf_present_problem", None)
    if problem is not None:
        problem.restype, problem.argtypes = ctypes.c_char_p, []
    for name in ("vmaf_present_start", "vmaf_present_flip"):
        function = getattr(lib, name, None)
        if function is not None:
            function.restype, function.argtypes = ctypes.c_int, [ctypes.c_void_p]
    return lib


@functools.cache
def _gstreamer_d3d11_api():
    """Loads GStreamer's D3D11 library and declares the types of the
    functions the presenters need."""
    gst = ctypes.CDLL("gstd3d11-1.0-0.dll")
    for name, restype in (("gst_is_d3d11_memory", ctypes.c_int),
                          ("gst_d3d11_memory_get_resource_handle", ctypes.c_void_p),
                          ("gst_d3d11_memory_get_subresource_index", ctypes.c_uint),
                          ("gst_d3d11_device_get_device_handle", ctypes.c_void_p)):
        function = getattr(gst, name)
        function.restype, function.argtypes = restype, [ctypes.c_void_p]
    return gst


def _presenters():
    """The native presenters to try, in order, as (name, library path):
    Vulkan's when it is asked for (vulkan_presenter.chosen), then Direct3D
    11's."""
    vulkan = [("Vulkan", vulkan_presenter.library_path())] if vulkan_presenter.chosen() else []
    return [*vulkan, ("Direct3D 11", d3d11_tonemap.library_path())]


def _problem(lib) -> str:
    """Why a presenter's last call failed, if the presenter can say (Vulkan's
    can); "" otherwise."""
    problem = getattr(lib, "vmaf_present_problem", None)
    return (problem() or b"").decode("utf-8", "replace") if problem is not None else ""


@functools.cache
def _crop_meta_api():
    """Returns gst_buffer_get_meta (typed) and the GType of GstVideoCropMeta's
    API, which together say where a frame's picture is in its texture (see
    LockedPresentation._draw)."""
    get_meta = ctypes.CDLL("gstreamer-1.0-0.dll").gst_buffer_get_meta
    get_meta.restype, get_meta.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t]
    api = ctypes.CDLL("gstvideo-1.0-0.dll").gst_video_crop_meta_api_get_type
    api.restype = ctypes.c_size_t
    return get_meta, api()


class _CropMeta(ctypes.Structure):
    """GstVideoCropMeta's layout: its GstMeta header, then the picture's
    rectangle."""
    _fields_ = [("flags", ctypes.c_int), ("info", ctypes.c_void_p),
                ("x", ctypes.c_uint), ("y", ctypes.c_uint), ("width", ctypes.c_uint), ("height", ctypes.c_uint)]


def yuv_to_rgb(matrix: str, full_range: bool, ten_bit: bool) -> tuple[float, ...]:
    """The presenter's YUV-to-RGB conversion: three rows of Y, U and V
    coefficients and an offset, for a frame's planes as the shader samples
    them. The shader sees NV12's 8-bit codes divided by 255, and P010's
    10-bit codes (in the top of 16 bits) divided by 65535. `matrix` is
    GStreamer's name for the YUV matrix (bt709, bt601, bt2020...)."""
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
    """Draws and presents native playback's frames with the app's own
    presenter (its vmaf_present_* functions): Direct3D 11's
    (d3d11_tonemap.dll), or Vulkan's (vulkan_presenter.dll) when it is asked
    for and can start.

    The presenter draws into a swapchain in a window of its own over
    `window`, on the GPU of the decoders' D3D11 device, and takes one frame
    per display refresh. A thread waits, holding no lock, until the
    swapchain can take a frame, and only then takes the device's lock to
    draw the frame that is due. Vulkan's presenter just hands the frame over
    while the lock is held, and draws and presents it after the lock is
    released. The decoders need the same lock for each frame, so it must
    never be held while waiting for the display: GStreamer's d3d11videosink
    did that, stalling the decoders by up to 40 ms, so 120 fps video showed
    about 100 frames a second and its sound stopped now and then."""

    def __init__(self, window, device, settings):
        self._video = _load_gstreamer()[1]
        self.window, self.device = window, device
        self._gst = _gstreamer_d3d11_api()
        self._crop_meta = _crop_meta_api()
        handle = self._gst.gst_d3d11_device_get_device_handle(object_pointer(device))
        hdr_display = int(settings.display_hdr_enabled is True)
        #: The presenter's library, its native handle, and its name ("Vulkan"
        #: or "Direct3D 11", for the status line).
        self._lib = self._presenter = self.name = None
        for name, path in _presenters():
            lib = _presenter_api(str(path))
            device.lock()
            try:
                presenter = lib.vmaf_present_create(handle, window, hdr_display)
            finally:
                device.unlock()
            if presenter:
                self._lib, self._presenter, self.name = lib, presenter, name
                break
            if name not in _PRESENTERS_FAILED:
                _PRESENTERS_FAILED.add(name)
                _LOG.warning("The %s video presenter could not start (%s); the next one is used",
                             name, _problem(lib) or "no reason given")
        if not self._presenter:
            raise RuntimeError("Could not set up the native video presentation")
        # The frames waiting to be drawn, in order, one each time the
        # swapchain takes a frame. At most two wait; past that the oldest is
        # dropped (see _present_paced).
        self._due = deque()
        self._due_lock = threading.Lock()
        self._due_ready = threading.Event()
        self._stopping = False
        self._error = None
        self._colours = {}
        self._shadings = {}
        self._clear = (ctypes.c_float * 4)(*_BACKGROUND)
        self._presenter_thread = threading.Thread(target=self._present_paced, name="native-present", daemon=True)
        self._presenter_thread.start()

    def present(self, sample, view=None, shading=None):
        """Shows `sample` at the next display refresh.

        `view` is None to fit the frame to the window, letterboxed, or else
        the part of the frame to show and where in the window, in device
        pixels (VideoCompareView.native_view). `shading` is what the shader
        does to the video's colours (d3d11_tonemap.shading), or None to show
        them as they are."""
        self._lib.vmaf_present_follow(self._presenter, self.window)
        with self._due_lock:
            self._due.append((sample, view, shading))
            while len(self._due) > 2:
                self._due.popleft()
            self._due_ready.set()

    def _present_paced(self):
        """The presenting thread: draws the waiting frames in order, one each
        time the swapchain takes a frame (once per refresh). Frames can be
        chosen as little as 4 ms apart, and two that fall within one refresh
        would otherwise lose one.

        A frame that cannot be drawn ends the thread, and the error is left
        for poll() to raise, so the picture never stops without a word."""
        # Vulkan's swapchain is made while the decoders start: making it
        # takes about 85 ms, which the first frame would otherwise wait for.
        start = getattr(self._lib, "vmaf_present_start", None)
        if start is not None and start(self._presenter) < 0:
            self._error = f"Native presentation failed: {_problem(self._lib) or 'no swapchain'}"
            return
        while True:
            self._due_ready.wait()
            if self._stopping:
                return
            waited = self._lib.vmaf_present_wait(self._presenter, 100)
            if waited < 0:
                self._error = f"Native presentation failed: {_problem(self._lib) or 'the swapchain failed'}"
                return
            if waited:
                continue  # the swapchain took no frame yet (a hidden window, say): wait again
            with self._due_lock:
                sample, view, shading = self._due.popleft()
                if not self._due:
                    self._due_ready.clear()
            if self._stopping:
                return
            try:
                self._draw(sample, view, shading)
            except Exception as exc:
                self._error = f"Native presentation failed: {exc}"
            if self._error:
                return

    def _draw(self, sample, view, shading):
        """Hands one frame to the presenter, and with Vulkan's, presents it.
        On a failure, sets the error poll() raises."""
        buffer = sample.get_buffer()
        if buffer.n_memory() != 1:
            self._error = "Native presentation needs one GPU texture a frame"
            return
        memory = buffer.peek_memory(0)
        # `memory` keeps the pointer's object alive until the native calls return.
        pointer = boxed_pointer(memory)
        if not self._gst.gst_is_d3d11_memory(pointer):
            self._error = "Native presentation received CPU memory instead of a D3D11 texture"
            return
        to_rgb, space, size = self._colours_of(sample.get_caps())
        shade = None
        if shading is not None:
            shade = self._shadings.get(shading)
            if shade is None:
                shade = self._shadings[shading] = (ctypes.c_float * 16)(*shading)
            # Tell the swapchain whether the shader maps the frame to SDR or
            # keeps it HDR, as PQ in BT.2020.
            space = _HDR10_SPACE if shading[0] == 2 else _SDR_SPACE
        crop, rectangle = view if view is not None else (None, None)
        # The frame's own size, not its texture's: a decoder's texture is
        # padded (1608 rows to 1616, say), and the padding must not be drawn.
        x, y, width, height = crop if crop is not None else (0, 0, *size)
        # When the converter only has to crop, it passes the decoder's frame
        # on whole, letterbox included, and a crop meta says where the
        # picture is (see GstComparePipeline).
        get_meta, crop_api = self._crop_meta
        if meta := get_meta(boxed_pointer(buffer), crop_api):
            picture = _CropMeta.from_address(meta)
            x, y = x + picture.x, y + picture.y
        source = (ctypes.c_int * 4)(x, y, width, height)
        target = (ctypes.c_int * 4)(*rectangle) if rectangle is not None else None
        self.device.lock()
        try:
            resource = self._gst.gst_d3d11_memory_get_resource_handle(pointer)
            # The frame's slice of its texture: H.264's decoder gives texture arrays.
            subresource = self._gst.gst_d3d11_memory_get_subresource_index(pointer)
            result = self._lib.vmaf_present_texture(self._presenter, resource, subresource, source, target, to_rgb,
                                                    shade, space, self._clear)
        finally:
            self.device.unlock()
        # Vulkan's presenter draws and presents the frame after the lock is
        # released, because the decoders need that lock for each frame.
        flip = getattr(self._lib, "vmaf_present_flip", None)
        if result >= 0 and flip is not None:
            result = flip(self._presenter)
        if result < 0:
            reason = _problem(self._lib)
            self._error = f"Native presentation failed: 0x{result & 0xffffffff:08x}" + (f" ({reason})" if reason else "")

    def _colours_of(self, caps):
        """Returns (YUV-to-RGB rows or None, DXGI colour space, (width,
        height)) for frames of `caps`, worked out once per kind of frame."""
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
        """Raises the presenting thread's error, if it had one."""
        if self._error:
            raise RuntimeError(self._error)

    def stop(self):
        """Stops presenting and frees the presenter; callable on any thread.
        The presenter's window goes with the view that owns it. If the
        presenting thread is still drawing a second later, the presenter is
        left, not freed."""
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
    """The source video's soundtrack, played on an audio-only pipeline. It
    does not change when another encode is selected.

    Its positions count from the source video's first frame, which is
    `offset_ms` into the file's timeline (GstComparePipeline.first_frame_ms),
    as the video frames' positions do, because its clock times the frames."""

    offset_ms = 0

    def __init__(self, path, start_ms, enabled):
        self.gst, _ = _load_gstreamer()
        self.pipeline = self.gst.ElementFactory.make("playbin3", "comparison-audio")
        self._seeker = Seeker("soundtrack-seek")
        self.pipeline.set_property("uri", path.resolve().as_uri())
        # GstPlayFlags: AUDIO | SOFT_VOLUME; no video/text/visualizations.
        self.pipeline.set_property("flags", 2 | 16)
        self.pipeline.set_property("mute", not enabled)
        self.pending = start_ms
        self.ready = False
        self.failed = None
        self.seeking = False
        self.playing = False
        #: For poll: the clock's last reading (ms), the smallest step it has
        #: moved in, and the last position returned while playing, with the
        #: time it was returned (ms, time.monotonic()).
        self._last_reading = None
        self._step_ms = 10.0
        self._given = None
        #: The clock's own last reading (ms from the video's first frame),
        #: without poll's smoothing. LockedNativePool judges by it whether
        #: playback has fallen behind.
        self.reading_ms = None
        self.pipeline.set_state(self.gst.State.PAUSED)

    def poll(self):
        """Handles the pipeline's messages and returns the soundtrack's
        position (ms from the source video's first frame), or None when it
        has none yet."""
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
        # The audio sink's clock moves in steps of its buffer, 10 ms here,
        # while a frame of 120 fps video lasts 8.3 ms, so read as it is, one
        # frame in five would be skipped. Between steps, the position is
        # carried on from the last one returned by the time that has passed.
        # It is kept between the reading and one step past it, and never
        # goes back while playing. (Carrying on from the moment a poll saw
        # the clock step instead let it fall up to a frame behind, and then
        # catch up by showing two frames at once.)
        position_ms, now = position / self.gst.MSECOND, time.monotonic()
        self.reading_ms = position_ms - self.offset_ms
        last = self._last_reading
        if self.playing and last is not None and 5 <= position_ms - last < self._step_ms:
            self._step_ms = position_ms - last
        self._last_reading = position_ms
        if not self.playing:
            self._given = None
            return position_ms - self.offset_ms
        given = position_ms
        if self._given is not None:
            given_ms, given_at = self._given
            carried = given_ms + (now - given_at) * 1000
            given = max(given_ms, min(max(carried, position_ms), position_ms + self._step_ms))
        self._given = (given, now)
        return given - self.offset_ms

    def set_offset(self, offset_ms, position):
        """From now on, counts positions from the source video's first frame,
        `offset_ms` into the file's timeline, and seeks to `position` again
        (unless the first seek, which waits for the preroll, has yet to
        happen and will use the new offset)."""
        self.offset_ms = offset_ms
        if self.pending is None and not self.failed:
            self.seek(position)

    def seek(self, position):
        """Seeks to `position`. The seek is made by the Seeker on a thread of
        its own; the soundtrack is ready again at the preroll that follows
        (see poll)."""
        self.ready, self.seeking = False, True
        self._last_reading, self._given = None, None

        def refused():
            self.failed = "Could not seek the soundtrack"
        self._seeker.seek(self.pipeline, self.gst, max(0, int(position) + self.offset_ms) * self.gst.MSECOND, refused)

    def set_playing(self, playing):
        """Plays or pauses the soundtrack."""
        if not self.failed:
            if bool(playing) != self.playing:
                # Start the smoothing again from now. Carrying on from the
                # position given before a pause would jump a whole step.
                self._last_reading, self._given = None, None
            self.playing = bool(playing)
            self.pipeline.set_state(self.gst.State.PLAYING if playing else self.gst.State.PAUSED)

    def set_enabled(self, enabled):
        """Unmutes (True) or mutes (False) the soundtrack."""
        self.pipeline.set_property("mute", not enabled)

    def stop(self):
        """Stops the soundtrack and frees its pipeline."""
        self._seeker.close()
        self.pipeline.set_state(self.gst.State.NULL)
