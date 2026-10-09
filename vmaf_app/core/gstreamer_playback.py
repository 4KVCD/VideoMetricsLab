"""GStreamer/D3D11 playback for synchronized source/distorted comparison.

The UI deliberately never receives decoded pixels.  GStreamer owns demuxing,
decoding and the soundtrack's clock, and the app's own presenter
(locked_presentation) shows the frames: D3D11 textures remain on the GPU from a
hardware decoder through crop/scale and into its swapchain.
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
    """A pipeline's flushing seeks, made on a thread of its own, the latest
    only. A flushing seek waits for the pipeline's streaming threads: made
    on the window's thread, it held the window as long as one of them was
    held up (a seek into a file on a USB hard disk froze it 16 s once).
    `refused`, given with each seek, is called on that thread if the
    pipeline refuses it; `done`, if given, once it has been made."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._wanted = None
        self._condition = threading.Condition()
        self._closed = False
        self._thread: threading.Thread | None = None

    def seek(self, pipeline, gst, time_ns: int, refused, done=None) -> None:
        with self._condition:
            if self._closed:
                return
            self._wanted = (pipeline, gst, time_ns, refused, done)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
            self._condition.notify()

    def close(self) -> None:
        """No more seeks; one under way finishes (stopping the pipeline
        ends it)."""
        with self._condition:
            self._closed = True
            self._wanted = None
            self._condition.notify()

    def _run(self) -> None:
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
#: to FFmpeg for one kind of file. GPU decoders are deliberately absent:
#: which of d3d11h264dec, d3d11h265dec, d3d11av1dec and d3d11vp9dec exist
#: depends on the GPU the registry was scanned on, not on the installation.
REQUIRED_ELEMENTS = (
    *_PIPELINE_ELEMENTS,
    "queue", "capsfilter", "appsink",
    "playbin3", "uridecodebin3",  # the soundtrack: playbin3 is built on uridecodebin3
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


#: Variables the GStreamer wheels' setup (gstreamer_libs.setup_python_
#: environment, run by gstreamer_bundle.pth and the packaged app's runtime
#: hook) sets by putting its value in front of what is there already: each
#: names one file, and the others are path lists.
_SINGLE_PATH_VARIABLES = ("GST_REGISTRY_1_0", "GST_PLUGIN_SCANNER_1_0")
_PATH_LIST_VARIABLES = ("PATH", "PYGI_DLL_DIRS", "GST_PLUGIN_PATH_1_0", "GST_PLUGIN_SYSTEM_PATH_1_0",
                        "GST_PYTHONPATH_1_0", "GI_TYPELIB_PATH", "GIO_EXTRA_MODULES", "XDG_DATA_DIRS",
                        "XDG_CONFIG_DIRS")


def repair_gstreamer_environment(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Undoes GStreamer's setup having run twice. A process started by a
    Python process that ran it inherits its variables and runs it again:
    every path is then there twice, and the two variables that name a file
    -- the plugin registry and the plugin scanner -- name none ("a;a").
    GStreamer then found no registry and no scanner, loaded every plugin
    into the process to scan it, and crashed doing so in about one start in
    five (heap corruption, 0xc0000374): measured in fresh processes, 0 of 25
    once repaired. That is any process started from Python -- the test
    suite's workers, the app run from an editor's launcher.

    Keeps a file variable's first path and a list's first occurrence of
    each path. Returns the names of the variables it changed."""
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
    """What Video Compare's first native opening loads, loaded in the
    background at the window's startup: GStreamer, its D3D11 plugin and the
    D3D11 device. Loaded by that opening instead, they held the
    window for 0.4 s then. The window's only: metric-only use (the command
    line, the scoring processes) does not pay their start or their memory."""
    if os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen":
        threading.Thread(target=_warm_up, name="gstreamer-load", daemon=True).start()


def _warm_up() -> None:
    if not gstreamer_available()[0]:
        return
    try:
        gst, _video = _load_gstreamer()
        plugin = gst.ElementFactory.find("d3d11convert")
        if plugin is not None:
            plugin.load()
        d3d11_device()
    except Exception:
        pass  # only a head start: Video Compare meets the same error itself, and plays with FFmpeg


def d3d11_device() -> Any:
    """The D3D11 device native playback draws on: one for the session,
    shared by each comparison's pipelines, and made again only once lost (a
    GPU reset). Each comparison made its own before: a process's first took
    about 0.14 s, the driver loading, and so did one made while no other was
    left."""
    global _D3D11_DEVICE
    with _D3D11_LOCK:
        device = _D3D11_DEVICE
        if device is None or device.get_property("device-removed-reason"):
            _load_gstreamer()  # first: it repairs the environment gi reads
            import gi

            gi.require_version("GstD3D11", "1.0")
            from gi.repository import GstD3D11

            device = GstD3D11.D3D11Device.new(0, 0)
            if device is None:
                raise GStreamerPlaybackError("D3D11 device unavailable")
            _D3D11_DEVICE = device
        return device


def _load_gstreamer() -> tuple[Any, Any]:
    """Import lazily so metric-only use does not pay GStreamer's start cost.
    Once: start_loading's thread and the window's first use may meet."""
    with _GST_LOCK:
        return _load_gstreamer_once()


def _load_gstreamer_once() -> tuple[Any, Any]:
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
        # Keep hardware decode in the same graphics API as processing/output.
        # D3D12 currently outranks D3D11 by default and can introduce a bridge
        # (or download/upload) before this application's D3D11 renderer.
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
    """Whether the D3D11 pipeline can be built in this process."""
    # Offscreen Qt tests have no real HWND for the presenter.  Keeping this
    # explicit also prevents native graphics drivers from being initialized by
    # otherwise headless unit tests.
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
    """Require the explicit GPU shader for HDR content on an SDR output."""
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
    """What the D3D11 shader does to one video's frames: "sdr", HDR mapped
    to SDR BT.709; "hdr", HDR kept for an HDR display and converted to the
    PQ in BT.2020 it takes -- passed through as they came, Display P3 colours
    were shown as BT.2020, oversaturated, and HLG has no swapchain colour
    space of its own; None, nothing."""
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
    """An HDR video's own colorimetry, as GStreamer writes it (range, YUV
    matrix, transfer, primaries): stated in the caps a converter is to give,
    it converts none of the colours, which the presenter's shader takes as
    they come. In GStreamer's own words -- "bt2100-pq", not "2:6:14:7":
    caps compare them as text, and an element that writes the name, as
    d3d11compositor did, has the caps it gives refused."""
    from vmaf_app.core.d3d11_tonemap import hdr_primaries

    full = info.color_range.casefold() in {"pc", "full", "jpeg"}
    matrix = _GST_MATRICES.get(info.color_space.casefold(), 6)
    transfer = 14 if hdr_kind(info) == "HDR10 / PQ" else 15
    colorimetry = _load_gstreamer()[1].VideoColorimetry()
    numbers = f"{1 if full else 2}:{matrix}:{transfer}:{_GST_PRIMARIES[hdr_primaries(info.color_primaries)]}"
    return (colorimetry.to_string() if colorimetry.from_string(numbers) else None) or numbers


@functools.cache
def _crop_meta_answer():
    """GStreamer's functions to answer an allocation query through ctypes,
    and the GTypes of the APIs of GstVideoCropMeta and GstVideoMeta, which
    videocrop asks for both; None if a query's type is not where GstQuery
    has it. Through PyGObject the query cannot be answered: its wrapper
    holds a reference, and a query held twice is not writable."""
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
    """Pad probe on an appsink: its allocation queries answered as taking a
    GstVideoCropMeta (GstComparePipeline). Holds nothing of the pipeline."""
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
    return settings.mode == PreviewColorMode.HDR_TO_SDR or (
        settings.mode == PreviewColorMode.DISPLAY_AWARE
        and settings.display_hdr_enabled is not True
    )


def _crop_edges(info: VideoInfo, crop: CropBox | None) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, 0, 0, 0
    return (
        max(0, crop.x),
        max(0, crop.y),
        max(0, info.width - crop.x - crop.w),
        max(0, info.height - crop.y - crop.h),
    )


def _output_format(info: VideoInfo) -> str:
    """NV12, or P010 for deeper video -- 12-bit as 10: what the presenter
    reads (locked_presentation). Given as P012, 12-bit frames came in two
    textures, which it cannot draw, and played through FFmpeg."""
    fmt = analysis_pix_fmt(info.pix_fmt, info.pix_fmt)
    return "P010_10LE" if "10" in fmt or "12" in fmt else "NV12"


def _native_colorimetry(info: VideoInfo) -> str | None:
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
    """GPU surface caps for one side, preserving that input's colour signal."""
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


class GstComparePipeline:
    """One video of a comparison -- its `side`, "source" or "distorted" --
    decoded, cropped, scaled and converted on the GPU into an appsink, as
    D3D11 textures LockedNativePool presents.

    It could also draw both videos itself, into two window handles through
    d3d11videosink, with the test video's soundtrack: the way of the view
    playback was before LockedNativePool, which nothing used any more."""

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
        gst, _gst_video = _load_gstreamer()
        self.Gst = gst
        self._comparison = comparison
        self._pipeline = gst.Pipeline.new("comparison")
        if self._pipeline is None:
            raise GStreamerPlaybackError("Could not create the GStreamer pipeline.")
        self._video_linked = {"source": False, "distorted": False}
        self._decoder_status_reported = False
        self._wanted_playing = False
        self._ready = False
        self._pending_initial_seek_ms: int | None = None
        self._initial_seek_sent = False
        #: Where the video's first frame is on the pipeline's timeline (ns),
        #: from its first preroll (_prerolled_time): frames and positions
        #: count from it, as the metrics and FFmpeg's playback count them.
        self._first_frame: int | None = None
        #: How the presenter's shader shows this video's frames
        #: (d3d11_tonemap.shading), None as they come; and which way, "sdr"
        #: or "hdr" (_shading).
        self.shading: tuple[float, ...] | None = None
        self._shaded: str | None = None
        self._seeker = Seeker(f"{side}-seek")
        #: A seek the pipeline refused (Seeker, on its thread), for poll().
        self._seek_error: str | None = None
        #: Seeks asked for (the window's thread) and made (the Seeker's): `seeking`.
        self._seeks_asked = self._seeks_made = 0
        #: Signal handlers that call back into this object, (object, id):
        #: stop() removes them (_release).
        self._handlers: list[tuple[Any, int]] = []
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
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise GStreamerPlaybackError(f"Could not create GStreamer element {factory}.")
        return element

    def _add(self, *elements) -> None:
        for element in elements:
            self._pipeline.add(element)

    def _build_video_branch(self, side: str, settings) -> None:
        info = (
            self._comparison.source_info
            if side == "source" else self._comparison.distorted_info
        )
        crop_box = (
            self._comparison.source_crop
            if side == "source" else self._comparison.distorted_crop
        )
        # filesrc into decodebin3, not uridecodebin3. The difference is
        # urisourcebin, which uridecodebin3 puts in front of decodebin3: it
        # holds the demuxed streams in a multiqueue with no byte limit, only
        # a time limit that it grows whenever one stream looks empty while
        # another is full. For AV1 on a hardware decoder the parser emits
        # frame-aligned buffers whose time level reads as zero, so that
        # queue kept growing until it held the whole file: 2.7 GB of RAM per
        # 4K AV1 stream against 0.35 GB for HEVC, and 5.7 GB for a
        # four-video comparison. decodebin3 on its own has the same
        # autoplugging, the same select-stream and pad-added signals, seeks
        # the same way, and its multiqueue is bounded (10 MB / 250 ms): the
        # same AV1 stream then costs 0.39 GB.
        source = self._make("filesrc", f"{side}-file")
        source.set_property("location", str(frame_input_path(self._comparison, side).resolve()))
        decoder = self._make("decodebin3", f"{side}-decoder")
        self._handlers.append((decoder, decoder.connect("select-stream", self._select_stream, side)))
        self._handlers.append((decoder, decoder.connect("pad-added", self._pad_added, side)))
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
        # On GPU memory videocrop cannot touch pixels; it attaches a crop
        # rectangle (GstVideoCropMeta) for the next element to honour, and
        # refuses to run unless that element says it will. d3d11convert does
        # in GStreamer 1.28: a letterboxed 10-bit frame comes out as FFmpeg
        # crops it, code for code. It did not when cropping came in, and
        # d3d11compositor cropped instead -- through RGB and back, up to 82
        # codes apart, and, given YUV to give, never: every letterboxed HDR
        # comparison played through FFmpeg. Stretched to the comparison's
        # size, as the metrics compare the frames -- but while black bars
        # are still being detected, boxed in black, as the FFmpeg paths fit
        # it: a letterboxed 16:9 source beside a 2.4:1 encode is squashed
        # otherwise, a pair no run compares as it is.
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

            # Shaded by the presenter as it draws the frame shown, at the
            # size shown: the converter only crops and scales, the video's
            # own colours kept (12-bit video as 10, which the presenter
            # reads). Shaded here, each frame of each video was converted
            # to 16-bit RGBA, copied and shaded at full size.
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
            # Cropped with nothing else to do -- the sizes and formats the
            # same either side -- d3d11convert passes frames on as they come,
            # and asks this sink whether it takes a crop meta: the presenter
            # does (LockedPresentation._draw). Asked of the sink as it is,
            # videocrop refused to run.
            sink.get_static_pad("sink").add_probe(
                self.Gst.PadProbeType.QUERY_DOWNSTREAM | self.Gst.PadProbeType.PUSH, _accept_crop_meta)
        # CPU-only decoders (including H.266) upload once, before GPU cropping.
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
        caps = stream.get_caps()
        if caps is None or caps.get_size() == 0:
            return ""
        return caps.get_structure(0).get_name()

    def _select_stream(self, _decoder, _collection, stream, side: str) -> int:
        # The video only: the soundtrack plays on its own
        # (locked_presentation.SingleSoundtrack).
        return 1 if self._stream_caps_name(stream).startswith("video/") else 0

    def _pad_added(self, _decoder, pad, side: str) -> None:
        name = pad.get_name()
        if name.startswith("video_") and not self._video_linked[side]:
            queue = self._pipeline.get_by_name(f"{side}-video-queue")
            sink_pad = queue.get_static_pad("sink")
            if pad.link(sink_pad) == self.Gst.PadLinkReturn.OK:
                self._video_linked[side] = True

    def start(self, position_ms: int, playing: bool) -> None:
        # Preroll both sinks before seeking or playing.  decodebin3 cannot
        # accept a reliable seek while still in READY, and starting PLAYING
        # would briefly display frame zero when opening at a later timestamp.
        self._wanted_playing = bool(playing)
        self._pending_initial_seek_ms = max(0, int(position_ms))
        result = self._pipeline.set_state(self.Gst.State.PAUSED)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise GStreamerPlaybackError("GStreamer could not open the comparison.")

    def stop(self) -> None:
        self._seeker.close()
        self._pipeline.set_state(self.Gst.State.NULL)
        self._release()

    def _release(self) -> None:
        """Removes the callbacks into this object that the pipeline's
        elements hold. Each is a cycle -- the decoder holds a bound method,
        which holds this object, which holds the pipeline -- that
        passes through GStreamer's C side, where Python's collector cannot
        see it: a stopped pipeline, with its decoder's D3D11 surfaces, was
        never freed. Moving the window between an HDR and an SDR display
        rebuilds the players each time; 4K video kept 0.8-1.8 GB of the
        GPU's memory a move (issue #3). After NULL nothing calls them."""
        for element, handler in self._handlers:
            if element.handler_is_connected(handler):
                element.disconnect(handler)
        self._handlers.clear()

    def set_playing(self, playing: bool) -> None:
        self._wanted_playing = bool(playing)
        if not self._ready:
            return
        state = self.Gst.State.PLAYING if playing else self.Gst.State.PAUSED
        self._pipeline.set_state(state)

    def seek(self, position_ms: int) -> None:
        if not self._ready:
            self._pending_initial_seek_ms = max(0, int(position_ms))
            return
        self._seek_to(position_ms, "GStreamer could not seek to that frame.")

    def _seek_to(self, position_ms: int, refused: str) -> None:
        """A seek made by the Seeker; refused, poll() reports `refused`."""
        def report() -> None:
            self._seek_error = refused

        self._seeks_asked += 1
        asked = self._seeks_asked

        def made() -> None:
            self._seeks_made = max(self._seeks_made, asked)
        self._seeker.seek(self._pipeline, self.Gst, self._stream_time(position_ms), report, made)

    @property
    def seeking(self) -> bool:
        """Whether the seek asked for last is still to be made: until it is,
        the sink can still give frames from before it (LockedNativePool)."""
        return self._seeks_made < self._seeks_asked

    def _stream_time(self, position_ms: int) -> int:
        """Where `position_ms`, counted from the video's first frame, is on
        the pipeline's timeline (ns)."""
        return max(0, int(position_ms)) * self.Gst.MSECOND + (self._first_frame or 0)

    def _prerolled_time(self) -> int:
        """The timeline's time of the frame prerolled (ns): at the first
        preroll, from the file's start, the video's first frame. GStreamer
        does not count from that frame for every file: VideoQ's MP4 source
        starts its timeline 32 ms before it, as FFmpeg does, where its
        Matroska encodes start theirs at it. Counted from the timeline's
        start, the two were paired 4 frames apart and never started
        playing from the beginning: no source frame had the number 0."""
        sample = next(iter(self._sinks.values())).emit("try-pull-preroll", 0)
        if sample is None:
            return 0
        time = sample.get_segment().to_stream_time(self.Gst.Format.TIME, sample.get_buffer().pts)
        return 0 if time == self.Gst.CLOCK_TIME_NONE else time

    def frame_time(self, sample) -> int | None:
        """A sample's time from the video's first frame (ns); None where it
        has no usable timestamp."""
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
        descriptions: list[str] = []
        for side, decoder in self._decoders.items():
            factories = self._decoder_factories(decoder)
            if not factories:
                descriptions.append(f"{side} decoder starting")
                continue
            # GPU decoders: Direct3D's, NVIDIA's (nvcodec) and Intel's (Quick
            # Sync). nvh265dec, which decodes 12-bit HEVC where the full
            # GStreamer is installed, was called software: "CPU decode".
            hardware = [
                name for name in factories if name.startswith(("d3d11", "d3d12", "nv", "qsv"))
            ]
            mode = "GPU" if hardware else "software"
            selected = hardware or factories
            descriptions.append(f"{side} {mode}: {', '.join(selected)}")
        return " · ".join(descriptions)

    def _negotiated_description(self) -> str:
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
        if self._shaded == "sdr":
            details.append("GPU HDR→SDR · fixed Reinhard 1000→100 nit")
        elif self._shaded == "hdr":
            details.append("GPU colours → PQ BT.2020, HDR kept")
        return " · ".join(details)

    def poll(self) -> PlaybackUpdate:
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
        if not self._decoder_status_reported:
            caps = [
                sink.get_static_pad("sink").get_current_caps()
                for sink in self._sinks.values()
            ]
            if all(item is not None for item in caps):
                self._decoder_status_reported = True
                status = self._negotiated_description()
        return PlaybackUpdate(status, error, ended)

    def position_ms(self) -> int | None:
        """Where the pipeline is, counted from the video's first frame (a
        report's; poll() leaves it out: queried at each of the view's ticks,
        it was 750 pipeline queries a second, read by nothing)."""
        ok, position = self._pipeline.query_position(self.Gst.Format.TIME)
        return round((position - (self._first_frame or 0)) / self.Gst.MSECOND) if ok else None
