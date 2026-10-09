"""GStreamer/D3D11 playback for synchronized source/distorted comparison.

The UI deliberately never receives decoded pixels.  GStreamer owns demuxing,
decoding, clocks and presentation, while D3D11 textures remain on the GPU from
a hardware decoder through crop/scale and into the swapchain.
"""
from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class PlaybackUpdate:
    position_ms: int | None = None
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
    "d3d11compositor", "queue", "capsfilter", "capssetter", "appsink", "appsrc",
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
    background at the window's startup: GStreamer, the plugin of the video
    sink and the D3D11 device. Loaded by that opening instead, they held the
    window for 0.4 s then. The window's only: metric-only use (the command
    line, the scoring processes) does not pay their start or their memory."""
    if os.environ.get("QT_QPA_PLATFORM", "").casefold() != "offscreen":
        threading.Thread(target=_warm_up, name="gstreamer-load", daemon=True).start()


def _warm_up() -> None:
    if not gstreamer_available()[0]:
        return
    try:
        gst, _video = _load_gstreamer()
        sink = gst.ElementFactory.find("d3d11videosink")
        if sink is not None:
            sink.load()
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
    # Offscreen Qt tests have no real HWND for GstVideoOverlay.  Keeping this
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
    if settings.mode == PreviewColorMode.UNMANAGED:
        return False, "unmanaged preview uses FFmpeg to bypass automatic sink color handling"
    if settings.mode == PreviewColorMode.HDR_TO_SDR and any(
        not info.color_transfer or info.color_transfer in {"unknown", "unspecified"}
        for info in (comparison.source_info, comparison.distorted_info)
    ):
        return False, "forced HDR interpretation of untagged video uses FFmpeg"
    inputs_are_hdr = any(
        hdr_kind(info) is not None
        for info in (comparison.source_info, comparison.distorted_info)
    )
    if inputs_are_hdr and needs_sdr_tonemap(settings):
        from vmaf_app.core.d3d11_tonemap import available, supports

        if not available():
            return False, "native HDR-to-SDR helper is not built; using FFmpeg tone mapping"
        for info in (comparison.source_info, comparison.distorted_info):
            if hdr_kind(info) and not supports(info.color_primaries):
                return False, f"{info.color_primaries} HDR primaries use the FFmpeg color converter"
    return True, ""


#: GstVideoColorPrimaries of the primaries the HDR shader takes (d3d11_tonemap.PRIMARIES).
_GST_PRIMARIES = {"bt709": 1, "bt2020": 7, "smpte432": 11}


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
    fmt = analysis_pix_fmt(info.pix_fmt, info.pix_fmt)
    if "12" in fmt:
        return "P012_LE"
    if "10" in fmt:
        return "P010_10LE"
    return "NV12"


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
        self._tone_mappers = []
        self._tone_error = None
        #: Signal handlers and pad probes that call back into this object,
        #: (object, id): stop() removes them (_release).
        self._handlers: list[tuple[Any, int]] = []
        self._probes: list[tuple[Any, int]] = []
        self._device = device
        if device is not None or (needs_sdr_tonemap(settings) and any(
            hdr_kind(i) for i in (comparison.source_info, comparison.distorted_info)
        )):
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
        # not, so every letterboxed comparison used to fail here and fall
        # back to FFmpeg. d3d11compositor does: with one input stretched to
        # the output size it is the same crop-scale-convert pass on the GPU,
        # checked frame for frame against d3d11convert on the same source.
        # It re-times its output to the frame grid, which d3d11convert does
        # not, so the uncropped case keeps the element it always had.
        width, height = comparison_dimensions(self._comparison)
        if any((left, top, right, bottom)):
            convert = self._make("d3d11compositor", f"{side}-convert")
            convert.set_property("background", 1)  # black, not the checkerboard
            picture = convert.request_pad_simple("sink_%u")
            if picture is None:
                raise GStreamerPlaybackError(f"Could not request the {side} compositor input.")
            picture.set_property("width", width)
            picture.set_property("height", height)
        else:
            convert = self._make("d3d11convert", f"{side}-convert")
        capsfilter = self._make("capsfilter", f"{side}-output-caps")
        caps = self.Gst.Caps.from_string(
            output_caps_string(self._comparison, settings, side)
        )
        tone_map = hdr_kind(info) is not None and needs_sdr_tonemap(settings)
        retag = None
        if tone_map:
            from vmaf_app.core.d3d11_tonemap import D3D11ToneMapper, hdr_primaries

            # Force a private, high-precision converter output, not an 8-bit
            # intermediate or the decoder's reference surface. Keep PQ/HLG
            # encoded values until our explicit highlight mapping stage.
            caps = self.Gst.Caps.from_string(
                "video/x-raw(memory:D3D11Memory),format=RGBA64_LE,"
                f"width={width},height={height},pixel-aspect-ratio=1/1,"
                # Gst colour enum tuple: full range, RGB matrix, PQ/HLG, and
                # the video's own primaries, which the shader converts (the
                # converter leaves them as they are). A YUV bt2100-pq
                # shorthand would leave limited-range RGB values for the
                # shader to misinterpret.
                f"colorimetry=1:1:{14 if hdr_kind(info) == 'HDR10 / PQ' else 15}:"
                f"{_GST_PRIMARIES[hdr_primaries(info.color_primaries)]}"
            )
            mapper = D3D11ToneMapper(self._device, hdr_kind(info), info.color_primaries)
            self._tone_mappers.append(mapper)
            retag = self._make("capssetter", f"{side}-sdr-caps")
            retag.set_property("replace", True)
            retag.set_property("caps", self.Gst.Caps.from_string(
                "video/x-raw(memory:D3D11Memory),format=RGBA64_LE,"
                f"width={width},height={height},pixel-aspect-ratio=1/1,"
                "colorimetry=sRGB"
            ))
            pad = capsfilter.get_static_pad("src")
            self._probes.append((pad, pad.add_probe(
                self.Gst.PadProbeType.BUFFER | self.Gst.PadProbeType.EVENT_DOWNSTREAM,
                self._tone_probe, (mapper, retag),
            )))
        capsfilter.set_property("caps", caps)
        sink = self._make("appsink", f"{side}-video-sink")
        # Retain references to GPU textures, not CPU-mapped pixel arrays.
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 3)
        sink.set_property("drop", False)
        sink.set_property("wait-on-eos", False)
        sink.set_property("enable-last-sample", False)
        # CPU-only decoders (including H.266) upload once, before GPU cropping.
        chain = [queue, upload, gpu_memory, crop, convert, capsfilter]
        if retag is not None:
            chain.append(retag)
        chain.append(sink)
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

    def _tone_probe(self, _pad, probe, processing):
        mapper, retag = processing
        if probe.type & self.Gst.PadProbeType.EVENT_DOWNSTREAM:
            event = probe.get_event()
            if event.type == self.Gst.EventType.CAPS:
                caps = event.parse_caps().copy()
                caps.set_value("colorimetry", "sRGB")
                # remove_field on a GI structure wrapper edits a copy, so use
                # writable caps via their serialized structure here (once per
                # negotiation, never per frame).
                structure = caps.get_structure(0).copy()
                for field in ("mastering-display-info", "content-light-level"):
                    structure.remove_field(field)
                output = self.Gst.Caps.new_empty()
                output.append_structure_full(structure, caps.get_features(0).copy())
                retag.set_property("caps", output)
            return self.Gst.PadProbeReturn.OK
        try:
            mapper.render(probe.get_buffer())
        except Exception as exc:
            self._tone_error = str(exc)
            return self.Gst.PadProbeReturn.DROP
        return self.Gst.PadProbeReturn.OK

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
        self._pipeline.set_state(self.Gst.State.NULL)
        for mapper in self._tone_mappers:
            mapper.close()
        self._release()

    def _release(self) -> None:
        """Removes the callbacks into this object that the pipeline's
        elements hold. Each is a cycle -- the decoder or a pad holds a bound
        method, which holds this object, which holds the pipeline -- that
        passes through GStreamer's C side, where Python's collector cannot
        see it: a stopped pipeline, with its decoder's D3D11 surfaces, was
        never freed. Moving the window between an HDR and an SDR display
        rebuilds the players each time; 4K video kept 0.8-1.8 GB of the
        GPU's memory a move (issue #3). After NULL nothing calls them."""
        for element, handler in self._handlers:
            if element.handler_is_connected(handler):
                element.disconnect(handler)
        for pad, probe in self._probes:
            pad.remove_probe(probe)
        self._handlers.clear()
        self._probes.clear()

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
        flags = self.Gst.SeekFlags.FLUSH | self.Gst.SeekFlags.ACCURATE
        if not self._pipeline.seek_simple(self.Gst.Format.TIME, flags, self._stream_time(position_ms)):
            raise GStreamerPlaybackError("GStreamer could not seek to that frame.")

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
            hardware = [
                name for name in factories if name.startswith(("d3d11", "d3d12"))
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
        if self._tone_mappers:
            details.append("GPU HDR→SDR · fixed Reinhard 1000→100 nit")
        return " · ".join(details)

    def poll(self) -> PlaybackUpdate:
        error = self._tone_error
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
                    flags = self.Gst.SeekFlags.FLUSH | self.Gst.SeekFlags.ACCURATE
                    if not self._pipeline.seek_simple(
                        self.Gst.Format.TIME, flags, self._stream_time(target)
                    ):
                        error = "GStreamer could not seek to the requested start frame."
                    else:
                        self._initial_seek_sent = True
                    continue
                self._ready = True
                target_state = (
                    self.Gst.State.PLAYING
                    if self._wanted_playing else self.Gst.State.PAUSED
                )
                self._pipeline.set_state(target_state)
        ok, position = self._pipeline.query_position(self.Gst.Format.TIME)
        position_ms = round((position - (self._first_frame or 0)) / self.Gst.MSECOND) if ok else None
        status = None
        if not self._decoder_status_reported:
            caps = [
                sink.get_static_pad("sink").get_current_caps()
                for sink in self._sinks.values()
            ]
            if all(item is not None for item in caps):
                self._decoder_status_reported = True
                status = self._negotiated_description()
        return PlaybackUpdate(position_ms, status, error, ended)
