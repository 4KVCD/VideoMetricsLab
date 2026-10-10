"""Plays the source and the test video in step for Video Compare, with
GStreamer and Direct3D 11.

The window never receives decoded pixels. GStreamer reads and decodes the
files and runs the soundtrack's clock, and the app's own presenter
(locked_presentation) shows the frames. The frames stay on the GPU as D3D11
textures, from a hardware decoder through cropping and scaling into the
presenter's swap chain.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import os
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorMode,
    PreviewColorSettings,
    comparison_dimensions,
    frame_input_path,
    frame_video_info,
    hdr_kind,
)
from vmaf_app.core.gpu import analysis_pix_fmt
from vmaf_app.core.models import CropBox, VideoInfo


class GStreamerPlaybackError(RuntimeError):
    """The native GStreamer playback pipeline could not be used."""


class Seeker:
    """Makes a pipeline's flushing seeks on a thread of its own, only the
    latest one asked for. A flushing seek waits for the pipeline's streaming
    threads, so on the window's thread it would hold the window as long as
    one of them is held up (a seek into a file on a USB hard disk once froze
    it for 16 s). The `refused` callback given with each seek is called on
    that thread if the pipeline refuses the seek, and `done`, if given, once
    the seek has been made."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._wanted = None
        self._condition = threading.Condition()
        self._closed = False
        self._thread: threading.Thread | None = None

    def seek(self, pipeline, gst, time_ns: int, refused, done=None) -> None:
        """Asks for a seek to `time_ns`, replacing any seek not yet made."""
        with self._condition:
            if self._closed:
                return
            self._wanted = (pipeline, gst, time_ns, refused, done)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
            self._condition.notify()

    def close(self) -> None:
        """Stops taking seeks. A seek under way finishes (stopping the
        pipeline ends it)."""
        with self._condition:
            self._closed = True
            self._wanted = None
            self._condition.notify()

    def _run(self) -> None:
        """The seek thread: makes the latest seek asked for, until closed."""
        while True:
            with self._condition:
                while self._wanted is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                pipeline, gst, time_ns, refused, done = self._wanted
                self._wanted = None
            if not pipeline.seek_simple(gst.Format.TIME, gst.SeekFlags.FLUSH | gst.SeekFlags.ACCURATE, time_ns):
                refused()
            if done is not None:
                done()
            del pipeline, refused, done  # not kept alive while waiting for the next


@dataclass(frozen=True, slots=True)
class PlaybackUpdate:
    """A change in playback for the window: a status message, an error, or
    the end of the videos."""

    status: str | None = None
    error: str | None = None
    ended: bool = False


_GST: tuple[Any, Any] | None = None
_GST_ERROR: str | None = None
_GST_LOCK = threading.Lock()
_D3D11_DEVICE: Any = None
_D3D11_LOCK = threading.Lock()

#: Elements the pipelines create by name. Without any of these the GStreamer
#: path cannot be built at all, so _load_gstreamer refuses and playback uses
#: FFmpeg instead.
_PIPELINE_ELEMENTS = (
    "filesrc", "decodebin3", "d3d11upload", "d3d11convert", "videocrop",
)

#: What a complete GStreamer installation provides for this app: the pipeline
#: elements, the parsers and demuxers decodebin3 needs for the files people
#: compare, software decoders for what no GPU decodes (VVC, 10-bit H.264,
#: ProRes), and the soundtrack path. The self-test and the packaging check
#: (scripts/verify_gstreamer_bundle.py) both test against this list, so a
#: bundle that dropped a plugin fails loudly instead of quietly falling back
#: to FFmpeg for one kind of file. GPU decoders are left out on purpose,
#: because which of d3d11h264dec, d3d11h265dec, d3d11av1dec and d3d11vp9dec
#: exist depends on the GPU the registry was scanned on, not on the
#: installation.
REQUIRED_ELEMENTS = (
    *_PIPELINE_ELEMENTS,
    "queue", "capsfilter", "appsink",
    "playbin3", "uridecodebin3",  # for the soundtrack (playbin3 is built on uridecodebin3)
    "h264parse", "h265parse", "h266parse", "av1parse", "vp9parse", "mpegvideoparse",
    "matroskademux", "qtdemux", "tsdemux", "avidemux",
    "avdec_h264", "avdec_h265", "avdec_h266", "avdec_prores", "dav1ddec",
    "aacparse", "ac3parse", "dcaparse", "opusparse",
    "avdec_aac", "avdec_ac3", "avdec_eac3", "avdec_dca", "avdec_truehd",
    "opusdec", "vorbisdec", "flacdec",
    "audioconvert", "audioresample", "volume", "autoaudiosink", "wasapi2sink",
)

#: Hardware decoders, reported rather than required (see REQUIRED_ELEMENTS).
GPU_DECODERS = ("d3d11h264dec", "d3d11h265dec", "d3d11av1dec", "d3d11vp9dec", "d3d11mpeg2dec")


#: The variables that the GStreamer wheels' setup sets by putting its value
#: in front of what is there already. The setup is
#: gstreamer_libs.setup_python_environment, run by gstreamer_bundle.pth and
#: by the packaged app's runtime hook. Each of these two variables names one
#: file; the variables in _PATH_LIST_VARIABLES are lists of paths.
_SINGLE_PATH_VARIABLES = ("GST_REGISTRY_1_0", "GST_PLUGIN_SCANNER_1_0")
_PATH_LIST_VARIABLES = ("PATH", "PYGI_DLL_DIRS", "GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0",
                        "GST_PYTHONPATH_1_0", "GI_TYPELIB_PATH", "GIO_EXTRA_MODULES", "XDG_DATA_DIRS",
                        "XDG_CONFIG_DIRS")


def repair_gstreamer_environment(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Undoes the effect of GStreamer's setup running twice, and returns the
    names of the variables it changed.

    A process started by a Python process that ran the setup inherits its
    variables and runs it again. Every path is then there twice, and the two
    variables that name a file (the plugin registry and the plugin scanner)
    name no file at all ("a;a"). GStreamer then finds no registry and no
    scanner, loads every plugin into the process to scan it, and crashed
    doing so in about one start in five (heap corruption, 0xc0000374), but
    in none of 25 fresh processes once repaired. This affects any process
    started from Python, such as the test suite's workers, or the app run
    from an editor.

    It keeps a file variable's first path, and the first occurrence of each
    path in a list.
    """
    environ = os.environ if environ is None else environ
    changed = []
    for name in (*_SINGLE_PATH_VARIABLES, *_PATH_LIST_VARIABLES):
        value = environ.get(name)
        if not value:
            continue
        parts = value.split(os.pathsep)
        if name in _SINGLE_PATH_VARIABLES:
            kept = parts[:1]
        else:
            seen: set[str] = set()
            kept = []
            for part in parts:
                key = os.path.normcase(os.path.normpath(part)) if part else part
                if key not in seen:
                    seen.add(key)
                    kept.append(part)
        repaired = os.pathsep.join(kept)
        if repaired != value:
            environ[name] = repaired
            changed.append(name)
    return changed


def start_loading() -> None:
    """Loads in the background, at the window's start, what Video Compare
    needs the first time it opens videos natively: GStreamer, its D3D11
    plugin, the D3D11 device, the film grain synthesis, and the Direct3D 11
    presenter's shaders. Loading them at that first opening would hold the
    window for 0.4 s. (The Vulkan presenter, when it is chosen, is made
    ready later, once there are videos: see start_loading_presenter.) Only
    the window calls this, so uses without Video Compare (the command line,
    the scoring processes) do not pay for the time or the memory."""
    if os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen":
        threading.Thread(target=_warm_up, name="gstreamer-load", daemon=True).start()


def _warm_up() -> None:
    """start_loading's thread."""
    if not gstreamer_available()[0]:
        return
    try:
        gst, _video = _load_gstreamer()
        plugin = gst.ElementFactory.find("d3d11convert")
        if plugin is not None:
            plugin.load()
        d3d11_device()
        from vmaf_app.core import d3d11_film_grain, d3d11_tonemap, vulkan_presenter

        if not vulkan_presenter.chosen():
            d3d11_tonemap.prepare_presenter()
        d3d11_film_grain.prepare()
    except Exception:
        pass  # only a head start: Video Compare meets the same error itself, and then plays with FFmpeg


_PRESENTER_LOADING_STARTED = False


def start_loading_presenter() -> None:
    """Makes the Vulkan presenter ready in the background (its device and
    pipelines), when it is chosen (vulkan_presenter.chosen). This happens
    once per process, when Video Compare may first have something to play:
    the window calls this as its Videos table gains a row. Made at the
    window's start, it would cost 130 ms and 25 MB of memory whether Video
    Compare is used or not, and left to the first native opening, it would
    hold the window for those 130 ms."""
    global _PRESENTER_LOADING_STARTED
    from vmaf_app.core import vulkan_presenter

    if (_PRESENTER_LOADING_STARTED or os.environ.get("QT_QPA_PLATFORM", "").casefold() == "offscreen"
            or not vulkan_presenter.chosen()):
        return
    _PRESENTER_LOADING_STARTED = True
    threading.Thread(target=_prepare_presenter, name="presenter-load", daemon=True).start()


def _prepare_presenter() -> None:
    """start_loading_presenter's thread."""
    if not gstreamer_available()[0]:
        return  # no native playback, so no presenter
    try:
        from vmaf_app.core import d3d11_tonemap, vulkan_presenter

        if vulkan_presenter.prepare() is not None:
            # Vulkan cannot present here, so the Direct3D 11 presenter is used.
            d3d11_tonemap.prepare_presenter()
    except Exception:
        pass  # only a head start, as in _warm_up


def d3d11_device() -> Any:
    """Returns the D3D11 device that native playback draws on. There is one
    for the session, shared by every comparison's pipelines, and made again
    only once lost (after a GPU reset). Don't make one per comparison:
    making a device takes about 0.14 s while the driver loads, for a
    process's first device and for one made while no other is left."""
    global _D3D11_DEVICE
    with _D3D11_LOCK:
        device = _D3D11_DEVICE
        if device is None or device.get_property("device-removed-reason"):
            _load_gstreamer()  # first, because it repairs the environment that gi reads
            import gi

            gi.require_version("GstD3D11", "1.0")
            from gi.repository import GstD3D11

            device = GstD3D11.D3D11Device.new(0, 0)
            if device is None:
                raise GStreamerPlaybackError("D3D11 device unavailable")
            _D3D11_DEVICE = device
        return device


def _load_gstreamer() -> tuple[Any, Any]:
    """Imports GStreamer when it is first needed, so that uses without Video
    Compare do not pay for its start. The lock makes this happen once,
    because start_loading's thread and the window's first use may meet."""
    with _GST_LOCK:
        return _load_gstreamer_once()


def _load_gstreamer_once() -> tuple[Any, Any]:
    """_load_gstreamer's work, done under its lock."""
    global _GST, _GST_ERROR
    if _GST is not None:
        return _GST
    if _GST_ERROR is not None:
        raise GStreamerPlaybackError(_GST_ERROR)
    if changed := repair_gstreamer_environment():
        logging.getLogger(__name__).info("GStreamer's environment was set up twice; repaired %s", ", ".join(changed))
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        Gst.init(None)
        # Keep hardware decoding in the same graphics API as the processing
        # and the output. D3D12's decoders outrank D3D11's by default, and
        # would put a bridge (or a download and upload) before the app's
        # D3D11 presenter.
        for name in ("d3d11h264dec", "d3d11h265dec", "d3d11av1dec", "d3d11vp9dec"):
            factory = Gst.ElementFactory.find(name)
            if factory is not None:
                factory.set_rank(max(factory.get_rank(), int(Gst.Rank.PRIMARY) + 16))
        missing = [name for name in _PIPELINE_ELEMENTS if Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError("missing elements: " + ", ".join(missing))
        _GST = Gst, GstVideo
        return _GST
    except Exception as exc:
        _GST_ERROR = f"GStreamer is unavailable: {exc}"
        raise GStreamerPlaybackError(_GST_ERROR) from exc


def gstreamer_available() -> tuple[bool, str]:
    """Returns whether the D3D11 pipeline can be built in this process, and
    why not."""
    # Offscreen Qt tests have no real window for the presenter. Saying so
    # here also keeps headless tests from starting the graphics drivers.
    if os.environ.get("QT_QPA_PLATFORM", "").casefold() == "offscreen":
        return False, "native video output is disabled by the offscreen Qt platform"
    try:
        _load_gstreamer()
    except GStreamerPlaybackError as exc:
        return False, str(exc)
    return True, ""


def uses_native_gstreamer(
    comparison: FrameComparison, settings: PreviewColorSettings
) -> tuple[bool, str]:
    """Returns whether this comparison plays through GStreamer (native
    playback), and if not, why: FFmpeg then plays it. HDR video needs the
    presenter's shader, so it plays natively only where the shader supports
    its colours."""
    available, reason = gstreamer_available()
    if not available:
        return False, reason
    from vmaf_app.core import d3d11_tonemap

    if not d3d11_tonemap.presents():
        return False, "the native video presenter is not built (d3d11_tonemap.dll)"
    if settings.mode == PreviewColorMode.UNMANAGED:
        return False, "unmanaged preview uses FFmpeg to bypass native playback's automatic color handling"
    if settings.mode == PreviewColorMode.HDR_TO_SDR and any(
        not info.color_transfer or info.color_transfer in {"unknown", "unspecified"}
        for info in (comparison.source_info, comparison.distorted_info)
    ):
        return False, "forced HDR interpretation of untagged video uses FFmpeg"
    inputs_are_hdr = any(
        hdr_kind(info) is not None
        for info in (comparison.source_info, comparison.distorted_info)
    )
    if inputs_are_hdr:
        for info in (comparison.source_info, comparison.distorted_info):
            # The shader maps HDR to SDR, or converts it to PQ in BT.2020 for
            # an HDR display (_shading); BT.2020 PQ is presented as it comes.
            shading = _shading(info, settings)
            if shading and not d3d11_tonemap.supports(info.color_primaries, hdr=shading == "hdr"):
                return False, f"{info.color_primaries} HDR primaries use the FFmpeg color converter"
    return True, ""


def _shading(info: VideoInfo, settings: PreviewColorSettings) -> str | None:
    """Returns what the D3D11 shader does to one video's frames:

    - "sdr": HDR is mapped to SDR BT.709.
    - "hdr": HDR is kept for an HDR display, and converted to the PQ in
      BT.2020 that the display takes. Passed through as they come, Display
      P3 colours would be shown as BT.2020, oversaturated, and HLG has no
      swap chain colour space of its own.
    - None: nothing.
    """
    if hdr_kind(info) is None:
        return None
    if needs_sdr_tonemap(settings):
        return "sdr"
    from vmaf_app.core.d3d11_tonemap import hdr_primaries

    if hdr_kind(info) == "HDR10 / PQ" and hdr_primaries(info.color_primaries) == "bt2020":
        return None
    return "hdr"


#: GstVideoColorPrimaries of the primaries the HDR shader takes (d3d11_tonemap.PRIMARIES).
_GST_PRIMARIES = {"bt709": 1, "bt2020": 7, "smpte432": 11}
#: GstVideoColorMatrix by FFmpeg's names for the YUV matrix.
_GST_MATRICES = {"bt709": 3, "bt470bg": 4, "smpte170m": 4, "smpte240m": 5, "bt2020nc": 6, "bt2020c": 6}


def _hdr_colorimetry(info: VideoInfo) -> str:
    """Returns an HDR video's own colorimetry as GStreamer writes it (range,
    YUV matrix, transfer, primaries). Stated in the caps a converter is to
    give, it makes the converter leave the colours alone, for the
    presenter's shader to take as they come. It is in GStreamer's own words
    ("bt2100-pq", not "2:6:14:7"), because caps compare it as text, and an
    element that writes the name (as d3d11compositor did) would have the
    caps it gives refused."""
    from vmaf_app.core.d3d11_tonemap import hdr_primaries

    full = info.color_range.casefold() in {"pc", "full", "jpeg"}
    matrix = _GST_MATRICES.get(info.color_space.casefold(), 6)
    transfer = 14 if hdr_kind(info) == "HDR10 / PQ" else 15
    colorimetry = _load_gstreamer()[1].VideoColorimetry()
    numbers = f"{1 if full else 2}:{matrix}:{transfer}:{_GST_PRIMARIES[hdr_primaries(info.color_primaries)]}"
    return (colorimetry.to_string() if colorimetry.from_string(numbers) else None) or numbers


@functools.cache
def _crop_meta_answer():
    """Returns GStreamer's functions for answering an allocation query
    through ctypes, and the GTypes of GstVideoCropMeta's and GstVideoMeta's
    APIs, both of which videocrop asks for. Returns None if a query's type
    is not where this expects it in GstQuery. The query cannot be answered
    through PyGObject, because its wrapper holds a reference, and a query
    held twice is not writable."""
    core = ctypes.CDLL("gstreamer-1.0-0.dll")
    for name, restype, argtypes in (
        ("gst_pad_probe_info_get_query", ctypes.c_void_p, [ctypes.c_void_p]),
        ("gst_query_new_allocation", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
        ("gst_query_find_allocation_meta", ctypes.c_int, [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]),
        ("gst_query_add_allocation_meta", None, [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]),
        ("gst_mini_object_unref", None, [ctypes.c_void_p]),
    ):
        function = getattr(core, name)
        function.restype, function.argtypes = restype, argtypes
    video = ctypes.CDLL("gstvideo-1.0-0.dll")
    apis = []
    for name in ("gst_video_crop_meta_api_get_type", "gst_video_meta_api_get_type"):
        function = getattr(video, name)
        function.restype = ctypes.c_size_t
        apis.append(function())
    allocation = int(_load_gstreamer()[0].QueryType.ALLOCATION)
    offset = 64  # GST_QUERY_TYPE: after the query's GstMiniObject, 64 bytes in a 64-bit process
    query = core.gst_query_new_allocation(None, 0)
    try:
        if ctypes.c_int.from_address(query + offset).value != allocation:
            return None
    finally:
        core.gst_mini_object_unref(query)
    return core, apis, allocation, offset


def _accept_crop_meta(_pad, info):
    """A pad probe on an appsink that answers its allocation queries as
    taking a GstVideoCropMeta (see GstComparePipeline). It holds no
    reference to the pipeline."""
    from vmaf_app.core.d3d11_tonemap import boxed_pointer

    answer = _crop_meta_answer()
    if answer is not None:
        core, apis, allocation, offset = answer
        query = core.gst_pad_probe_info_get_query(boxed_pointer(info))
        if query and ctypes.c_int.from_address(query + offset).value == allocation:
            for api in apis:
                if not core.gst_query_find_allocation_meta(query, api, None):
                    core.gst_query_add_allocation_meta(query, api, None)
    return _GST[0].PadProbeReturn.OK


def needs_sdr_tonemap(settings: PreviewColorSettings) -> bool:
    """Whether HDR video is mapped to SDR: when the user chose HDR to SDR, or
    when the display is not known to be in HDR mode."""
    return settings.mode == PreviewColorMode.HDR_TO_SDR or (
        settings.mode == PreviewColorMode.DISPLAY_AWARE
        and settings.display_hdr_enabled is not True
    )


def _crop_edges(info: VideoInfo, crop: CropBox | None) -> tuple[int, int, int, int]:
    """The black-bar crop as the pixels to cut from each edge: (left, top,
    right, bottom)."""
    if crop is None:
        return 0, 0, 0, 0
    return (
        max(0, crop.x),
        max(0, crop.y),
        max(0, info.width - crop.x - crop.w),
        max(0, info.height - crop.y - crop.h),
    )


def _output_format(info: VideoInfo) -> str:
    """Returns the pixel format the presenter reads (locked_presentation):
    NV12, or P010 for video deeper than 8 bits, 12-bit video included. Don't
    give 12-bit video as P012: its frames would come in two textures, which
    the presenter cannot draw, so it would play through FFmpeg."""
    fmt = analysis_pix_fmt(info.pix_fmt, info.pix_fmt)
    return "P010_10LE" if "10" in fmt or "12" in fmt else "NV12"


def _native_colorimetry(info: VideoInfo) -> str | None:
    """The colorimetry that keeps an HDR or BT.2020 video's colour signal in
    the output caps for an HDR display, or None for other video."""
    kind = hdr_kind(info)
    if kind == "HDR10 / PQ":
        return "bt2100-pq"
    if kind == "HLG":
        return "bt2100-hlg"
    if info.color_primaries.casefold() in {"bt2020", "bt.2020"}:
        return "bt2020-10"
    return None


def output_caps_string(
    comparison: FrameComparison, settings: PreviewColorSettings, side: str
) -> str:
    """Returns the output caps for one side's frames on the GPU, keeping that
    video's colour signal."""
    width, height = comparison_dimensions(comparison)
    info = frame_video_info(comparison, side)
    fields = [
        "video/x-raw(memory:D3D11Memory)",
        f"format={_output_format(info)}",
        f"width={width}",
        f"height={height}",
        "pixel-aspect-ratio=1/1",
    ]
    colorimetry = _native_colorimetry(info)
    if colorimetry is not None and settings.display_hdr_enabled is True:
        fields.append(f"colorimetry={colorimetry}")
    return ",".join(fields)


class _FilmGrain:
    """Draws a VVC video's film grain on its frames, for GstComparePipeline.

    GStreamer's VVC decoder never applies film grain, so this does it
    instead. It reads the film grain messages from the access units that
    GStreamer's VVC parser hands to the decoder, and draws each frame's
    grain on the GPU as the decoded frame leaves d3d11upload. That is before
    the frame is cropped, scaled or shown, so the grain lands where a
    decoder would have put it.
    """

    def __init__(self, gst, gst_video, device) -> None:
        from vmaf_app.core.d3d11_film_grain import GrainDrawer
        from vmaf_app.core.film_grain import FilmGrainTracker

        self._gst, self._gst_video = gst, gst_video
        self.tracker = FilmGrainTracker()
        self._drawer = GrainDrawer(device)
        self._read_decoder_config = False
        # The tag each access unit leaves the parser with: its number
        # (FilmGrainTracker.add_access_unit). The decoder copies it onto the
        # frame it decodes from that access unit, as GstVideoDecoder copies
        # every meta without tags; a GstReferenceTimestampMeta has none.
        self._access_unit_tag = gst.Caps.from_string("timestamp/x-videometricslab-access-unit")
        # The probe on the parser's output: (pad, probe id), once there is one.
        self._parser_probe: tuple | None = None
        # The frames' layout, worked out once for their caps:
        # (caps, bit depth, plane sizes, chroma subsampling).
        self._layout: tuple | None = None
        # Whether grain has been drawn on any frame yet: for the status line.
        self.drawn = False
        # Why drawing the grain failed, if it did. It is not tried again.
        self.failure: str | None = None

    def element_added(self, _bin, _sub_bin, element) -> None:
        """Handles decodebin3's deep-element-added signal: watches the access
        units that its VVC parser hands on, once the parser is created."""
        factory = element.get_factory()
        if factory is not None and factory.get_name() == "h266parse":
            kinds = self._gst.PadProbeType.BUFFER | self._gst.PadProbeType.EVENT_FLUSH
            pad = element.get_static_pad("src")
            self._parser_probe = (pad, pad.add_probe(kinds, self._on_parser_output))

    def status(self) -> str | None:
        """What the status line says about the grain: that it is drawn, or
        was turned off by an error; None while no frame has had grain drawn."""
        if self.failure is not None:
            return "film grain off (error)"
        return "film grain synthesized" if self.drawn else None

    def _on_parser_output(self, pad, info):
        """Reads an access unit from the parser (they come in decoding
        order), or a seek's flush, after which decoding starts again."""
        if self.failure is None:
            try:
                self._read_parser_output(pad, info)
            except Exception as error:  # on GStreamer's thread: playback goes on without the grain
                self._turn_off(f"could not read the grain messages: {error!r}")
        return self._gst.PadProbeReturn.OK

    def _read_parser_output(self, pad, info) -> None:
        """Reads one access unit's grain messages and tags its buffer with
        the access unit's number, or resets the tracker at a seek's flush."""
        gst = self._gst
        if info.type & gst.PadProbeType.EVENT_FLUSH:
            if info.get_event().type == gst.EventType.FLUSH_STOP:
                self.tracker.flush()
            return
        if not self._read_decoder_config:
            # Matroska and MP4 can keep the stream's parameter sets in the caps.
            self._read_decoder_config = True
            caps = pad.get_current_caps()
            codec_data = caps.get_structure(0).get_value("codec_data") if caps is not None else None
            if codec_data is not None:
                ok, mapped = codec_data.map(gst.MapFlags.READ)
                if ok:
                    self.tracker.read_decoder_config(bytes(mapped.data))
                    codec_data.unmap(mapped)
        # Writable, to be tagged; given back to the probe as the context ends.
        with info.writable_object() as buffer:
            ok, mapped = buffer.map(gst.MapFlags.READ)
            if not ok:
                return
            pts = None if buffer.pts == gst.CLOCK_TIME_NONE else buffer.pts
            try:
                number = self.tracker.add_access_unit(mapped.data, pts)
            finally:
                buffer.unmap(mapped)
            buffer.add_reference_timestamp_meta(self._access_unit_tag, number, gst.CLOCK_TIME_NONE)

    def on_frame_uploaded(self, pad, info):
        """Draws a decoded frame's grain on it, just after it was uploaded to the GPU."""
        if self.failure is None:
            try:
                self._draw_grain(pad, info)
            except Exception as error:  # on GStreamer's thread: playback goes on without the grain
                self._turn_off(repr(error))
        return self._gst.PadProbeReturn.OK

    def _draw_grain(self, pad, info) -> None:
        """Draws the grain of the frame in this buffer, if it has any."""
        from vmaf_app.core.d3d11_film_grain import native_params
        from vmaf_app.core.d3d11_tonemap import boxed_pointer

        buffer = info.get_buffer()
        tag = buffer.get_reference_timestamp_meta(self._access_unit_tag)
        frame = self.tracker.take_frame_grain(None if tag is None else tag.timestamp,
                                              None if buffer.pts == self._gst.CLOCK_TIME_NONE else buffer.pts)
        if frame is None:
            return
        _caps, depth, sizes, chroma_shift = self._frame_layout(pad.get_current_caps())
        params = native_params(frame, depth, sizes, chroma_shift)
        if params is not None:
            error = self._drawer.draw(boxed_pointer(buffer), params)
            if error:
                self._turn_off(error)
            else:
                self.drawn = True

    def _turn_off(self, reason: str) -> None:
        """Stops drawing the grain for this video, and says why in the log."""
        self.failure = reason
        logging.getLogger(__name__).warning("Film grain could not be drawn, and is left off: %s", reason)

    def _frame_layout(self, caps) -> tuple:
        """The frames' layout for these caps, worked out once per caps:
        (caps, bit depth, plane sizes, chroma subsampling)."""
        if self._layout is None or self._layout[0] != caps.to_string():
            from vmaf_app.core.d3d11_film_grain import plane_sizes

            info = self._gst_video.VideoInfo.new_from_caps(caps)
            planes = info.finfo.n_planes
            chroma_shift = (info.finfo.w_sub[1], info.finfo.h_sub[1]) if planes > 1 else (0, 0)
            sizes = plane_sizes(info.width, info.height, chroma_shift, planes)
            self._layout = (caps.to_string(), info.finfo.depth[0], sizes, chroma_shift)
        return self._layout

    def remove_probe(self) -> None:
        """Removes the probe on the parser's output. Called once the pipeline has stopped."""
        if self._parser_probe is not None:
            pad, probe = self._parser_probe
            pad.remove_probe(probe)
            self._parser_probe = None

    def close(self) -> None:
        """Frees the grain's GPU resources. Called once the pipeline has stopped."""
        self._drawer.close()


def _start_film_grain(gst, gst_video, device, info: VideoInfo) -> _FilmGrain | None:
    """Sets up film grain drawing for a VVC video. Returns None for other
    codecs, or if the synthesis is not built or cannot start on the device."""
    from vmaf_app.core import d3d11_film_grain

    if info.codec_name != "vvc" or device is None or not d3d11_film_grain.available():
        return None
    try:
        return _FilmGrain(gst, gst_video, device)
    except RuntimeError as error:
        logging.getLogger(__name__).warning("Film grain is off for %s: %s", info.path.name, error)
        return None


class GstComparePipeline:
    """Plays one video of a comparison, its `side` ("source" or
    "distorted"): decoded, cropped, scaled and converted on the GPU into an
    appsink, as D3D11 textures that LockedNativePool presents."""

    def __init__(
        self,
        comparison: FrameComparison,
        settings: PreviewColorSettings,
        side: str,
        *,
        device=None,
    ) -> None:
        if side not in ("source", "distorted"):
            raise ValueError("invalid video side")
        gst, gst_video = _load_gstreamer()
        self.Gst = gst
        self._gst_video = gst_video
        self._comparison = comparison
        self._pipeline = gst.Pipeline.new("comparison")
        if self._pipeline is None:
            raise GStreamerPlaybackError("Could not create the GStreamer pipeline.")
        self._video_linked = {"source": False, "distorted": False}
        self._decoder_status_reported = False
        #: What the status line last said about the film grain
        #: (_FilmGrain.status). The line is sent again when that changes.
        self._reported_grain_status: str | None = None
        self._wanted_playing = False
        self._ready = False
        self._pending_initial_seek_ms: int | None = None
        self._initial_seek_sent = False
        #: Where the video's first frame is on the pipeline's timeline (ns),
        #: found at its first preroll (_prerolled_time). Frames and positions
        #: are counted from it, as the metrics and FFmpeg's playback count
        #: them.
        self._first_frame: int | None = None
        #: How the presenter's shader shows this video's frames
        #: (d3d11_tonemap.shading), or None to show them as they come; and
        #: which way, "sdr" or "hdr" (_shading).
        self.shading: tuple[float, ...] | None = None
        self._shaded: str | None = None
        self._seeker = Seeker(f"{side}-seek")
        #: The error of a seek the pipeline refused (set on the Seeker's
        #: thread), for poll() to report.
        self._seek_error: str | None = None
        #: How many seeks were asked for (on the window's thread) and made (on
        #: the Seeker's thread); see `seeking`.
        self._seeks_asked = self._seeks_made = 0
        #: Signal handlers that call back into this object, as (object, id).
        #: stop() removes them (_release).
        self._handlers: list[tuple[Any, int]] = []
        #: Pad probes that call back into this object's parts, as (pad, id).
        #: stop() removes them (_release).
        self._probes: list[tuple[Any, int]] = []
        #: The VVC video's film grain, drawn on its frames (_FilmGrain); None
        #: for other videos.
        self._film_grain: _FilmGrain | None = None
        self._device = device
        if device is not None or any(
            _shading(i, settings) for i in (comparison.source_info, comparison.distorted_info)
        ):
            import gi

            gi.require_version("GstD3D11", "1.0")
            from gi.repository import GstD3D11

            self._device = device or GstD3D11.D3D11Device.new(0, 0)
            if self._device is None:
                raise GStreamerPlaybackError("Could not create the D3D11 processing device")
            self._pipeline.set_context(GstD3D11.d3d11_context_new(self._device))

        self._decoders: dict[str, Any] = {}
        self._sinks: dict[str, Any] = {}
        try:
            self._build_video_branch(side, settings)
        except Exception:
            self.stop()
            raise
        self._bus = self._pipeline.get_bus()

    def _make(self, factory: str, name: str):
        """Makes a GStreamer element, or raises GStreamerPlaybackError."""
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise GStreamerPlaybackError(f"Could not create GStreamer element {factory}.")
        return element

    def _add(self, *elements) -> None:
        """Adds elements to the pipeline."""
        for element in elements:
            self._pipeline.add(element)

    def _build_video_branch(self, side: str, settings) -> None:
        """Builds the chain of elements that reads, decodes, crops, scales and
        converts the video into its appsink."""
        info = (
            self._comparison.source_info
            if side == "source" else self._comparison.distorted_info
        )
        crop_box = (
            self._comparison.source_crop
            if side == "source" else self._comparison.distorted_crop
        )
        # filesrc into decodebin3, not uridecodebin3. The difference is
        # urisourcebin, which uridecodebin3 puts in front of decodebin3. It
        # holds the demuxed streams in a multiqueue with no byte limit, only
        # a time limit that it raises whenever one stream looks empty while
        # another is full. For AV1 on a hardware decoder, the parser gives
        # frame-aligned buffers whose time level reads as zero, so that queue
        # would keep growing until it held the whole file: 2.7 GB of memory
        # per 4K AV1 stream against 0.35 GB for HEVC, and 5.7 GB for a
        # four-video comparison. decodebin3 on its own plugs in the same
        # elements, has the same select-stream and pad-added signals, seeks
        # the same way, and its multiqueue is bounded (10 MB or 250 ms): the
        # same AV1 stream then takes 0.39 GB.
        source = self._make("filesrc", f"{side}-file")
        source.set_property("location", str(frame_input_path(self._comparison, side).resolve()))
        decoder = self._make("decodebin3", f"{side}-decoder")
        self._handlers.append((decoder, decoder.connect("select-stream", self._select_stream, side)))
        self._handlers.append((decoder, decoder.connect("pad-added", self._pad_added, side)))
        self._film_grain = _start_film_grain(self.Gst, self._gst_video, self._device, info)
        if self._film_grain is not None:
            self._handlers.append((decoder, decoder.connect("deep-element-added", self._film_grain.element_added)))
        queue = self._make("queue", f"{side}-video-queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)
        crop = self._make("videocrop", f"{side}-crop")
        left, top, right, bottom = _crop_edges(info, crop_box)
        crop.set_property("left", left)
        crop.set_property("top", top)
        crop.set_property("right", right)
        crop.set_property("bottom", bottom)
        upload = self._make("d3d11upload", f"{side}-upload")
        gpu_memory = self._make("capsfilter", f"{side}-gpu-memory")
        gpu_memory.set_property("caps", self.Gst.Caps.from_string("video/x-raw(memory:D3D11Memory)"))
        if self._film_grain is not None:
            # Here the whole decoded frame is on the GPU, so its grain is drawn here.
            pad = gpu_memory.get_static_pad("src")
            self._probes.append((pad, pad.add_probe(self.Gst.PadProbeType.BUFFER, self._film_grain.on_frame_uploaded)))
        # On GPU memory, videocrop cannot touch the pixels. It attaches a
        # crop rectangle (GstVideoCropMeta) for the next element to apply,
        # and refuses to run unless that element says it will. d3d11convert
        # does so in GStreamer 1.28, and a letterboxed 10-bit frame comes out
        # exactly as FFmpeg crops it. Don't crop with d3d11compositor instead:
        # it converts through RGB and back (up to 82 codes apart), and asked
        # to output YUV it does not crop at all.
        #
        # The frame is stretched to the comparison's size, as the metrics
        # compare the frames. But while black bars are still being detected,
        # it is boxed in black, as the FFmpeg paths fit it, because otherwise
        # a letterboxed 16:9 source beside a 2.4:1 encode would be squashed,
        # a pair that no run compares as it is.
        width, height = comparison_dimensions(self._comparison)
        convert = self._make("d3d11convert", f"{side}-convert")
        convert.set_property("add-borders", bool(self._comparison.auto_crop_pending))
        capsfilter = self._make("capsfilter", f"{side}-output-caps")
        caps = self.Gst.Caps.from_string(
            output_caps_string(self._comparison, settings, side)
        )
        shading = _shading(info, settings)
        if shading:
            from vmaf_app.core import d3d11_tonemap

            # The presenter shades the frame as it draws it, at the size
            # shown, so the converter only crops and scales, keeping the
            # video's own colours (12-bit video as 10, which the presenter
            # reads). Shading here would convert each frame of each video to
            # 16-bit RGBA, copy it and shade it at full size.
            caps = self.Gst.Caps.from_string(
                f"video/x-raw(memory:D3D11Memory),format={_output_format(info)},"
                f"width={width},height={height},pixel-aspect-ratio=1/1,colorimetry={_hdr_colorimetry(info)}"
            )
            self.shading = d3d11_tonemap.shading(hdr_kind(info), info.color_primaries, hdr=shading == "hdr")
            self._shaded = shading
        capsfilter.set_property("caps", caps)
        sink = self._make("appsink", f"{side}-video-sink")
        # Retain references to GPU textures, not CPU-mapped pixel arrays.
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 3)
        sink.set_property("drop", False)
        sink.set_property("wait-on-eos", False)
        sink.set_property("enable-last-sample", False)
        if any((left, top, right, bottom)):
            # When cropping is all there is to do (the same sizes and formats
            # on either side), d3d11convert passes frames on as they come,
            # and asks this sink whether it takes a crop meta. The presenter
            # does (LockedPresentation._draw), so the probe says yes; the
            # appsink alone would say no, and videocrop would refuse to run.
            sink.get_static_pad("sink").add_probe(
                self.Gst.PadProbeType.QUERY_DOWNSTREAM | self.Gst.PadProbeType.PUSH, _accept_crop_meta)
        # Software decoders (including H.266's) upload each frame once,
        # before the GPU crops it.
        chain = [queue, upload, gpu_memory, crop, convert, capsfilter, sink]
        self._add(source, decoder, *chain)
        if not source.link(decoder):
            raise GStreamerPlaybackError(f"Could not open the {side} video file for decoding.")
        for first, second in pairwise(chain):
            if not first.link(second):
                raise GStreamerPlaybackError(
                    f"Could not connect the {side} GPU video branch."
                )
        self._decoders[side] = decoder
        self._sinks[side] = sink

    @staticmethod
    def _stream_caps_name(stream) -> str:
        """The media type of a stream's caps, such as "video/x-h265"."""
        caps = stream.get_caps()
        if caps is None or caps.get_size() == 0:
            return ""
        return caps.get_structure(0).get_name()

    def _select_stream(self, _decoder, _collection, stream, side: str) -> int:
        """decodebin3's select-stream signal: selects the video only, because
        the soundtrack plays on its own (locked_presentation.SingleSoundtrack)."""
        return 1 if self._stream_caps_name(stream).startswith("video/") else 0

    def _pad_added(self, _decoder, pad, side: str) -> None:
        """decodebin3's pad-added signal: links the decoded video's pad to
        the branch's queue."""
        name = pad.get_name()
        if name.startswith("video_") and not self._video_linked[side]:
            queue = self._pipeline.get_by_name(f"{side}-video-queue")
            sink_pad = queue.get_static_pad("sink")
            if pad.link(sink_pad) == self.Gst.PadLinkReturn.OK:
                self._video_linked[side] = True

    def start(self, position_ms: int, playing: bool) -> None:
        """Opens the video paused, then seeks to `position_ms` and plays if
        `playing` (poll() finishes this once the sink has prerolled)."""
        # The sink is prerolled before seeking or playing: decodebin3 cannot
        # take a reliable seek while still in READY, and starting in PLAYING
        # would briefly show frame 0 when opening at a later time.
        self._wanted_playing = bool(playing)
        self._pending_initial_seek_ms = max(0, int(position_ms))
        result = self._pipeline.set_state(self.Gst.State.PAUSED)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise GStreamerPlaybackError("GStreamer could not open the comparison.")

    def stop(self) -> None:
        """Stops the pipeline and frees what it holds."""
        self._seeker.close()
        self._pipeline.set_state(self.Gst.State.NULL)
        self._release()
        if self._film_grain is not None:
            self._film_grain.close()

    def _release(self) -> None:
        """Removes the callbacks into this object that the pipeline's
        elements hold. Each is a reference cycle (the decoder holds a bound
        method, which holds this object, which holds the pipeline) that
        passes through GStreamer's C side, where Python's garbage collector
        cannot see it. Left in place, they keep a stopped pipeline and its
        decoder's D3D11 surfaces from being freed. Moving the window between
        an HDR and an SDR display rebuilds the players each time, and 4K
        video kept 0.8-1.8 GB of the GPU's memory per move (issue #3). Once
        the pipeline is in the NULL state, nothing calls the callbacks."""
        for element, handler in self._handlers:
            if element.handler_is_connected(handler):
                element.disconnect(handler)
        self._handlers.clear()
        for pad, probe in self._probes:
            pad.remove_probe(probe)
        self._probes.clear()
        if self._film_grain is not None:
            self._film_grain.remove_probe()

    def set_playing(self, playing: bool) -> None:
        """Plays or pauses, or keeps the wish until the pipeline is ready."""
        self._wanted_playing = bool(playing)
        if not self._ready:
            return
        state = self.Gst.State.PLAYING if playing else self.Gst.State.PAUSED
        self._pipeline.set_state(state)

    def seek(self, position_ms: int) -> None:
        """Seeks to `position_ms`, counted from the video's first frame, or
        keeps it until the pipeline is ready."""
        if not self._ready:
            self._pending_initial_seek_ms = max(0, int(position_ms))
            return
        self._seek_to(position_ms, "GStreamer could not seek to that frame.")

    def _seek_to(self, position_ms: int, refused: str) -> None:
        """Has the Seeker make a seek. If the pipeline refuses it, poll()
        reports `refused`."""
        def report() -> None:
            self._seek_error = refused

        self._seeks_asked += 1
        asked = self._seeks_asked

        def made() -> None:
            self._seeks_made = max(self._seeks_made, asked)
        self._seeker.seek(self._pipeline, self.Gst, self._stream_time(position_ms), report, made)

    @property
    def seeking(self) -> bool:
        """Whether the last seek asked for is still to be made. Until it is,
        the sink can still give frames from before it (LockedNativePool)."""
        return self._seeks_made < self._seeks_asked

    def _stream_time(self, position_ms: int) -> int:
        """Where `position_ms`, counted from the video's first frame, is on
        the pipeline's timeline (ns)."""
        return max(0, int(position_ms)) * self.Gst.MSECOND + (self._first_frame or 0)

    def _prerolled_time(self) -> int:
        """Returns the timeline time of the prerolled frame (ns). At the
        first preroll, from the file's start, that is the video's first
        frame. GStreamer does not count from that frame for every file:
        VideoQ's MP4 source starts its timeline 32 ms before it, as FFmpeg
        does, while its Matroska encodes start theirs at it. Counting from
        the timeline's start instead would pair the two 4 frames apart, and
        playback would never start from the beginning, because no source
        frame would have the number 0."""
        sample = next(iter(self._sinks.values())).emit("try-pull-preroll", 0)
        if sample is None:
            return 0
        time = sample.get_segment().to_stream_time(self.Gst.Format.TIME, sample.get_buffer().pts)
        return 0 if time == self.Gst.CLOCK_TIME_NONE else time

    def frame_time(self, sample) -> int | None:
        """Returns a sample's time from the video's first frame (ns), or None
        if it has no usable timestamp."""
        time = sample.get_segment().to_stream_time(self.Gst.Format.TIME, sample.get_buffer().pts)
        if time == self.Gst.CLOCK_TIME_NONE:
            return None
        return time - (self._first_frame or 0)

    @property
    def first_frame_ms(self) -> int | None:
        """Where the video's first frame is on the file's timeline (ms), once
        known (the first preroll)."""
        return None if self._first_frame is None else round(self._first_frame / self.Gst.MSECOND)

    def _decoder_factories(self, decoder) -> list[str]:
        """The names of the video decoders that decodebin3 has plugged in."""
        factories: list[str] = []
        iterator = decoder.iterate_recurse()
        while True:
            result, element = iterator.next()
            if result == self.Gst.IteratorResult.DONE:
                break
            if result == self.Gst.IteratorResult.RESYNC:
                iterator.resync()
                continue
            if result != self.Gst.IteratorResult.OK:
                break
            if element is None:
                continue
            factory = element.get_factory()
            if factory is None:
                continue
            klass = factory.get_metadata(self.Gst.ELEMENT_METADATA_KLASS) or ""
            if "Decoder/Video" in klass:
                factories.append(factory.get_name())
        return sorted(set(factories))

    def _decoder_description(self) -> str:
        """Says which decoder the video uses, GPU or software, for the status
        line."""
        descriptions: list[str] = []
        for side, decoder in self._decoders.items():
            factories = self._decoder_factories(decoder)
            if not factories:
                descriptions.append(f"{side} decoder starting")
                continue
            # GPU decoders: Direct3D's, NVIDIA's (nvcodec) and Intel's (Quick
            # Sync). nvh265dec, which decodes 12-bit HEVC where the full
            # GStreamer is installed, must count as a GPU decoder too.
            hardware = [
                name for name in factories if name.startswith(("d3d11", "d3d12", "nv", "qsv"))
            ]
            mode = "GPU" if hardware else "software"
            selected = hardware or factories
            descriptions.append(f"{side} {mode}: {', '.join(selected)}")
        return " · ".join(descriptions)

    def _negotiated_description(self) -> str:
        """The status line once the frames' format is settled: size, format,
        colours, memory, decoder, film grain and shading."""
        caps_by_side = {
            side: sink.get_static_pad("sink").get_current_caps()
            for side, sink in self._sinks.items()
        }
        if any(caps is None or caps.get_size() == 0 for caps in caps_by_side.values()):
            return self._decoder_description()
        caps = caps_by_side.get("distorted", next(iter(caps_by_side.values())))
        assert caps is not None
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        fmt = structure.get_value("format")
        color = structure.get_value("colorimetry")
        memory = caps.get_features(0).to_string()
        details = [f"{width}×{height} {fmt}", memory, self._decoder_description()]
        if color:
            details.insert(1, str(color))
        grain = self._film_grain.status() if self._film_grain is not None else None
        if grain is not None:
            details.append(grain)
        if self._shaded == "sdr":
            details.append("GPU HDR→SDR · fixed Reinhard 1000→100 nit")
        elif self._shaded == "hdr":
            details.append("GPU colours → PQ BT.2020, HDR kept")
        return " · ".join(details)

    def poll(self) -> PlaybackUpdate:
        """Returns what changed since the last call: a status line, an error,
        or the end of the video. It also finishes opening the video: the
        first seek, then playing or staying paused."""
        error = self._seek_error
        ended = False
        while True:
            message = self._bus.pop_filtered(
                self.Gst.MessageType.ERROR
                | self.Gst.MessageType.EOS
                | self.Gst.MessageType.ASYNC_DONE
            )
            if message is None:
                break
            if message.type == self.Gst.MessageType.ERROR:
                gst_error, debug = message.parse_error()
                detail = gst_error.message
                if debug:
                    detail += f" ({debug[-1000:]})"
                error = detail
            elif message.type == self.Gst.MessageType.EOS:
                ended = True
            elif message.type == self.Gst.MessageType.ASYNC_DONE and not self._ready:
                if self._first_frame is None:
                    self._first_frame = self._prerolled_time()
                target = self._pending_initial_seek_ms or 0
                self._pending_initial_seek_ms = None
                if target > 0 and not self._initial_seek_sent:
                    self._seek_to(target, "GStreamer could not seek to the requested start frame.")
                    self._initial_seek_sent = True
                    continue
                self._ready = True
                target_state = (
                    self.Gst.State.PLAYING
                    if self._wanted_playing else self.Gst.State.PAUSED
                )
                self._pipeline.set_state(target_state)
        status = None
        grain = self._film_grain.status() if self._film_grain is not None else None
        if not self._decoder_status_reported or grain != self._reported_grain_status:
            caps = [
                sink.get_static_pad("sink").get_current_caps()
                for sink in self._sinks.values()
            ]
            if all(item is not None for item in caps):
                self._decoder_status_reported = True
                self._reported_grain_status = grain
                status = self._negotiated_description()
        return PlaybackUpdate(status, error, ended)

    def position_ms(self) -> int | None:
        """Returns where the pipeline is (ms), counted from the video's first
        frame, for reports (scripts/smoke_playback_pool.py). poll() leaves it
        out: queried at each of the view's ticks, it would make 750 pipeline
        queries a second that nothing reads."""
        ok, position = self._pipeline.query_position(self.Gst.Format.TIME)
        return round((position - (self._first_frame or 0)) / self.Gst.MSECOND) if ok else None
