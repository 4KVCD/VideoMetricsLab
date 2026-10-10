"""Decodes videos in the scoring process for the GPU metrics, without FFmpeg
decoding them.

On NVIDIA GPUs the pictures are decoded straight into GPU memory
(vmaf_app/native/nvdec_frames.dll). On Intel's and AMD's GPUs they are
decoded into system memory through their makers' decoder libraries
(vpl_frames.dll with oneVPL, amf_frames.dll with AMF); when AMD's decoder
hands its pictures over to the GPU, Windows' own decoders are tried first
(mf_frames.dll). scripts/build_gpu_frames.ps1 builds all of them from
native/, and they share one C API (native/gpu_frames.h).

Why: when FFmpeg decodes for the GPU metrics, it decodes on the GPU, copies
each picture back to system memory, converts it and writes it to a pipe,
which the scoring process reads into memory that the GPU then copies it
from again. At 4K that is several CPU copies of 25 MB per frame and per
video, most of a GPU metric's CPU use, and at 200 frames per second more
than a pipe carries. Here FFmpeg only copies the compressed stream out of
its container (-c:v copy, a few MB per second), and the GPU's decoder
decodes it in this process. NVIDIA's pictures stay on the GPU: libvmaf reads
them there, and the GPU's copy engine fills Vship's page-locked buffers.
Intel's and AMD's decoders hand their pictures over in system memory, and
one CPU pass puts each into Vship's buffer (on Intel's integrated GPUs,
system memory is the GPU's own memory).

The pictures are the same, sample for sample, as FFmpeg's decoding gives:
the same decoder hardware, the stream's display area, a crop rounded as
FFmpeg's crop filter rounds it (its edges to even numbers, for 4:2:0), and
P016's 10-bit samples either kept in the top bits (Vship reads them as
16-bit) or shifted down, exactly as FFmpeg converts them to yuv420p10le.
8-bit samples are widened to 10 bits as FFmpeg widens them. A picture the
comparison scales is scaled here with FFmpeg's filter for the chosen
algorithm (native/scale_filter.h), on the GPU for every maker (with a
compute shader on Intel's and AMD's, native/d3d11_scale.h, that gives the
CPU scaler's result). That is not FFmpeg's result to the sample (it is
within 1 of it on film), which the user decided is the same comparison (see
ComparisonRecipe.identity_dict). The pictures that come out, and their
timestamps, are checked against the packets that went in
(GpuFrameStream.verify): a picture the decoder dropped or added fails the
run, and the caller makes the run again through FFmpeg.

A video FFmpeg would decode in software (one with no GPU decoder, such as
VVC, AV1 on an older GPU or 10-bit H.264, or with GPU decoding turned off)
is decoded here too, on the CPU, by the "software" decoder
(software_frames.dll). It uses FFmpeg 9's own decoders, built for the app
(vmaf_app/tools/ffmpeg), and the bundled GStreamer's dav1d, so its pictures
need no pipe either: at 4K a Windows pipe carries 60-80 frames a second,
while the decoders give 200 or more. The pictures are FFmpeg's decoding's,
sample for sample, film grain included.

Whatever is not decoded here (another GPU, a codec or format the decoder
does not take, an interlaced or damaged stream) is decoded by FFmpeg
instead: GpuDecodeUnavailableError is raised before the first picture, and
GpuDecodeFailedError after it.
"""
from __future__ import annotations

import _winapi
import contextlib
import ctypes
import heapq
import importlib.util
import logging
import msvcrt
import os
import queue
import re
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, ffmpeg_path, ffprobe_path
from vmaf_app.core.models import CropBox, VideoInfo

_log = logging.getLogger(__name__)
#: ConnectNamedPipe's answers when the writer has already connected (535),
#: or has connected, written and closed (232).
_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA = 535, 232

_NATIVE = Path(__file__).resolve().parents[1] / "native"
#: Each decoder library, all with one C API (native/gpu_frames.h): NVIDIA's
#: decoder through nvcuvid, Intel's through oneVPL, AMD's through AMF (and
#: Windows' Media Foundation), and the CPU's through FFmpeg's libraries and
#: dav1d (SOFTWARE).
LIBRARIES = {"nvidia": _NATIVE / "nvdec_frames.dll", "intel": _NATIVE / "vpl_frames.dll",
             "amd": _NATIVE / "amf_frames.dll", "amd-mf": _NATIVE / "mf_frames.dll",
             "software": _NATIVE / "software_frames.dll"}
LIBRARY_PATH = LIBRARIES["nvidia"]
#: Windows' own decoders (Media Foundation's) on AMD's GPUs, which can only
#: hand their pictures over to the GPU. They are opened first where AMD's
#: decoder would hand over, and AMD's own (AMF) where they cannot be
#: (GpuFrameStream).
MEDIA_FOUNDATION = "amd-mf"
#: The decoder for the videos FFmpeg would decode in software.
SOFTWARE = "software"
#: The environment variable that, set to "ffmpeg", sends the videos FFmpeg
#: decodes in software to the scoring process through FFmpeg's pipes
#: instead of the software decoder, to compare the two.
SOFTWARE_VARIABLE = "VML_SOFTWARE_DECODE"
#: The software decoder's libraries: FFmpeg 9.0.2's libavutil and libavcodec
#: with only the decoders it needs (LGPL, built by
#: scripts/build_ffmpeg_decoders.ps1), and the bundled GStreamer's dav1d,
#: which is in its wheel's package (also a folder of the packaged app, see
#: scripts/gstreamer_bundle.py). The libraries they import are beside them
#: or are Windows'.
FFMPEG_FOLDER = Path(__file__).resolve().parents[1] / "tools" / "ffmpeg"
_FFMPEG_LIBRARIES = ("avutil-61.dll", "avcodec-63.dll")
_DAV1D_PACKAGE, _DAV1D_LIBRARY = "gstreamer_plugins_libs", "dav1d.dll"
#: The decoders whose pictures can be handed straight to Vulkan VMAF's and
#: libvmaf's memory on the GPU (GpuFrameStream.pin, import_memory): NVIDIA's
#: through CUDA, and AMD's through Vulkan (native/amf_handover.h) when opened
#: with handover.
HANDOVER_BACKENDS = ("nvidia", "amd", MEDIA_FOUNDATION)

#: For each FFmpeg codec name: NVDEC's codec number (cudaVideoCodec; for VVC,
#: which NVDEC does not have, the app's own 100), and the bitstream filters
#: that turn the container's packets into what the decoder's parser reads.
#:
#: H.264, HEVC and VVC get Annex B start codes, with the parameter sets from
#: the stream's configuration before every keyframe (mp4toannexb adds them
#: only before an IDR, and an open-GOP H.264 stream cut at a recovery point
#: has none, so NVDEC decoded nothing of it). AV1's packets are already in its
#: low-overhead OBU format, and its sequence header goes to the parser
#: separately. MPEG-2's sequence header, which a container may keep only in
#: its configuration, is put before every keyframe too. VP9's packets need
#: nothing, and FFV1's configuration goes to its decoder separately
#: (_configuration).
#:
#: VVC's packets are given decode times that count up from far below any
#: presentation time (setts). Matroska stores presentation times only, and
#: FFmpeg's demuxer gives most of a VVC stream's packets decode times equal to
#: their presentation times, out of order, and the rest none. FFmpeg's stream
#: copy then "repairs" them by overwriting the presentation times (292 became
#: 627 in one film), but the packet listing needs those as they are. Nothing
#: reads these made-up decode times.
_CODECS = {"h264": (4, "h264_mp4toannexb,dump_extra=freq=keyframe"),
           "hevc": (8, "hevc_mp4toannexb,dump_extra=freq=keyframe"), "av1": (11, None),
           "vvc": (100, "vvc_mp4toannexb,dump_extra=freq=keyframe,setts=pts=PTS:dts=N-1000000000"),
           "vp9": (10, None), "mpeg2video": (1, "dump_extra=freq=keyframe"), "ffv1": (101, None)}
#: The codecs the GPU decoders are used for. The others are left to the
#: software decoder (NVIDIA's would take VP9 and MPEG-2, but they are untried).
_GPU_CODECS = frozenset({"h264", "hevc", "av1"})
#: Packet flags (AV_PKT_FLAG_*).
_KEY, _DISCARD = 0x1, 0x4
#: FFmpeg warnings that mean the timestamps it copied may not be the ones its
#: decoding would give a picture (it rewrites non-monotonic ones as it muxes).
_TIMESTAMP_WARNING = re.compile(r"timestamp|\bdts\b|\bpts\b|monoton", re.IGNORECASE)
#: The pixel formats FFmpeg decodes these codecs to, with their bit depths.
#: 4:2:0 only: 4:2:2 and 4:4:4 are not decoded here.
_DEPTHS = {"yuv420p": 8, "yuvj420p": 8, "yuv420p10le": 10, "yuv420p10be": 10}

_NVF_FRAME, _NVF_END, _NVF_ERROR, _NVF_ABORTED, _NVF_TIMEOUT = 1, 0, -1, -2, 2
_ES_PIPE_BYTES = 8 * 1024 * 1024
_NOPTS = -(1 << 63)


#: How long a decoder library may take to close a decoder. It normally takes
#: milliseconds, but AMD's has been seen never to return (on a Radeon 780M,
#: closing a decoder stopped part-way while Vulkan was in use in the same
#: process).
_CLOSE_SECONDS = 10.0
_stuck_closes = 0


def stuck_decoders() -> int:
    """How many decoders this process gave up closing because their library
    never returned. Such a process cannot be trusted to shut down by itself
    (the stuck call holds the driver's locks), so isolated's child process
    ends itself."""
    return _stuck_closes


class GpuDecodeUnavailableError(RuntimeError):
    """This video cannot be decoded here, so FFmpeg decodes it."""


class GpuDecodeFailedError(RuntimeError):
    """Decoding failed after it had started; the caller decodes the video
    again through FFmpeg. `backend` names the decoder, where it is known."""

    def __init__(self, message: str = "", backend: str | None = None) -> None:
        super().__init__(message)
        self.backend = backend


def decoding_failed_status(error: BaseException) -> str:
    """The status shown when decoding in the scoring process failed after it
    had started, naming the GPU's decoder or the app's own."""
    what = "Decoding in the app" if getattr(error, "backend", None) == SOFTWARE else "GPU decoding"
    return f"{what} failed ({error}); decoding through FFmpeg instead…"


# ------------------------------------------------------------------ binding

class _Params(ctypes.Structure):
    """The decoder libraries' nvf_params (native/gpu_frames.h): how to decode
    one video."""
    _fields_ = [("device", ctypes.c_int), ("codec", ctypes.c_int), ("bit_depth", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("crop_x", ctypes.c_int), ("crop_y", ctypes.c_int), ("crop_w", ctypes.c_int), ("crop_h", ctypes.c_int),
                ("shift", ctypes.c_int), ("luma_only", ctypes.c_int), ("pool", ctypes.c_int),
                ("extradata", ctypes.c_void_p), ("extradata_size", ctypes.c_int),
                ("out_w", ctypes.c_int), ("out_h", ctypes.c_int), ("scaler", ctypes.c_int), ("widen", ctypes.c_int),
                ("cpu_scaling", ctypes.c_int), ("handover", ctypes.c_int)]


class _Info(ctypes.Structure):
    """The decoder libraries' nvf_info (native/gpu_frames.h): what a decoder
    found in the stream, and how many pictures it decoded and handed out."""
    _fields_ = [("coded_width", ctypes.c_int), ("coded_height", ctypes.c_int),
                ("display_left", ctypes.c_int), ("display_top", ctypes.c_int),
                ("display_right", ctypes.c_int), ("display_bottom", ctypes.c_int),
                ("bit_depth", ctypes.c_int), ("chroma_format", ctypes.c_int), ("progressive", ctypes.c_int),
                ("decode_surfaces", ctypes.c_int),
                ("decoded", ctypes.c_longlong), ("displayed", ctypes.c_longlong), ("frame_bytes", ctypes.c_longlong),
                ("scaled_on_gpu", ctypes.c_int)]


_libraries: dict[str, ctypes.CDLL] = {}
_library_lock = threading.Lock()


def _software_library_paths() -> list[Path | None]:
    """The paths of libavutil, libavcodec and dav1d; None for any that is
    not there."""
    spec = importlib.util.find_spec(_DAV1D_PACKAGE)
    folders = [Path(folder) / "bin" for folder in (spec.submodule_search_locations or [])] if spec else []
    dav1d = next((folder / _DAV1D_LIBRARY for folder in folders if (folder / _DAV1D_LIBRARY).is_file()), None)
    ffmpeg = [FFMPEG_FOLDER / name if (FFMPEG_FOLDER / name).is_file() else None for name in _FFMPEG_LIBRARIES]
    return [*ffmpeg, dav1d]


def _software_libraries() -> tuple[int | None, ...]:
    """Loads libavutil, libavcodec and dav1d by their paths and returns their
    module handles (None for any that is not there or did not load).

    The libraries they import are found beside them, because ctypes loads a
    library by its path with its own folder searched. So no folder is added
    to this process's DLL search path: GStreamer's folder has MinGW runtimes
    that would then be found before other libraries' own."""
    handles: list[int | None] = []
    for path in _software_library_paths():
        handle = None
        if path is not None:
            try:
                handle = ctypes.CDLL(str(path))._handle
            except OSError as error:
                _log.info("The software decoder cannot use %s: %s", path.name, error)
        handles.append(handle)
    return tuple(handles)


def _load(backend: str = "nvidia") -> ctypes.CDLL:
    """Loads a decoder library once and declares its functions' types.
    Raises GpuDecodeUnavailableError when it is not bundled or will not load."""
    with _library_lock:
        if backend not in _libraries:
            if backend == SOFTWARE and os.environ.get(SOFTWARE_VARIABLE, "").casefold() == "ffmpeg":
                raise GpuDecodeUnavailableError(f"{SOFTWARE_VARIABLE} is set to ffmpeg")
            path = LIBRARY_PATH if backend == "nvidia" else LIBRARIES[backend]
            if not path.is_file():
                raise GpuDecodeUnavailableError(f"the frame decoder ({path.name}) is not bundled")
            try:
                lib = ctypes.CDLL(str(path))
            except OSError as error:
                raise GpuDecodeUnavailableError(f"the GPU frame decoder could not be loaded: {error}") from error
            handle, text = ctypes.c_void_p, ctypes.c_char_p
            for name, restype, argtypes in (
                ("nvf_open", handle, [ctypes.POINTER(_Params), text, ctypes.c_int]),
                ("nvf_push", ctypes.c_int, [handle, ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]),
                ("nvf_finish", ctypes.c_int, [handle]),
                ("nvf_pop", ctypes.c_int,
                 [handle, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong)]),
                ("nvf_download", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_void_p]),
                ("nvf_copy_luma", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_longlong]),
                ("nvf_release", None, [handle, ctypes.c_int]),
                ("nvf_abort", None, [handle]),
                ("nvf_close", None, [handle]),
                ("nvf_error", ctypes.c_int, [handle, text, ctypes.c_int]),
                ("nvf_info", None, [handle, ctypes.POINTER(_Info)]),
                ("nvf_supports", ctypes.c_int,
                 [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, text, ctypes.c_int]),
            ):
                function = getattr(lib, name)
                function.restype, function.argtypes = restype, argtypes
            if backend in HANDOVER_BACKENDS:  # pictures handed over with no CPU copy (GpuFrameStream.pin, import_memory)
                planes, pitches = ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong)
                imported = ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_void_p)
                for name, restype, argtypes in (
                    ("nvf_pin", ctypes.c_int, [handle, ctypes.c_void_p, ctypes.c_ulonglong]),
                    ("nvf_unpin", ctypes.c_int, [handle, ctypes.c_void_p]),
                    ("nvf_download_planes", ctypes.c_int, [handle, ctypes.c_int, planes, pitches]),
                    ("nvf_copy_planes", ctypes.c_int,
                     [handle, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong), pitches]),
                    # NVIDIA's CUDA imports any export; AMD's Vulkan only one from its own GPU and driver.
                    ("nvf_import", ctypes.c_int, [handle, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int, *imported])
                    if backend == "nvidia" else
                    ("nvf_import_vulkan", ctypes.c_int, [handle, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint,
                                                         ctypes.c_char_p, ctypes.c_char_p, *imported]),
                    ("nvf_unimport", None, [handle, ctypes.c_void_p]),
                ):
                    function = getattr(lib, name)
                    function.restype, function.argtypes = restype, argtypes
            if hasattr(lib, "nvf_timeline"):  # AMD's: copies the GPU waits for, rather than the CPU
                value = ctypes.POINTER(ctypes.c_ulonglong)
                for name, restype, argtypes in (
                    ("nvf_timeline", ctypes.c_int, [handle, ctypes.POINTER(ctypes.c_void_p)]),
                    ("nvf_copy_luma_async", ctypes.c_int,
                     [handle, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_longlong, value]),
                    ("nvf_copy_planes_async", ctypes.c_int,
                     [handle, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_longlong),
                      value]),
                ):
                    function = getattr(lib, name)
                    function.restype, function.argtypes = restype, argtypes
            if hasattr(lib, "nvf_slot_texture"):  # a slot's picture in place (Windows' decoder)
                lib.nvf_slot_texture.restype = ctypes.c_int
                lib.nvf_slot_texture.argtypes = [handle, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                                 ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            if backend in _GPU_SCALING:  # the decoders that scale with native/d3d11_scale.h
                lib.nvf_scale_note.restype, lib.nvf_scale_note.argtypes = ctypes.c_int, [handle, text, ctypes.c_int]
                lib.nvf_scale_test.restype = ctypes.c_int
                lib.nvf_scale_test.argtypes = [ctypes.c_int, ctypes.POINTER(_Params), ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_void_p, text, ctypes.c_int]
            if backend == SOFTWARE:
                lib.nvf_set_libraries.restype = ctypes.c_int
                lib.nvf_set_libraries.argtypes = [ctypes.c_void_p] * 3
                lib.nvf_libraries_error.restype, lib.nvf_libraries_error.argtypes = ctypes.c_int, [text, ctypes.c_int]
                if lib.nvf_set_libraries(*_software_libraries()) != 0:
                    # The decoder then refuses what it cannot decode (nvf_supports).
                    error = ctypes.create_string_buffer(512)
                    lib.nvf_libraries_error(error, len(error))
                    _log.info("The software decoder: %s", error.value.decode(errors="replace"))
            _libraries[backend] = lib
        return _libraries[backend]


# ------------------------------------------------------------------- plans

#: The app's scaling algorithms (VmafOptions.scale_algorithm), mapped to the
#: decoders' filters (native/scale_filter.h).
_SCALERS = {"bilinear": 0, "bicubic": 1, "lanczos": 2, "spline": 3}
#: How an 8-bit video is widened to the 10 bits it is compared at, as FFmpeg
#: converts it: shifted left by 2 (WIDEN_SHIFT), or, for full range luma,
#: also with its top bits repeated in the bottom ones (WIDEN_REPEAT).
WIDEN_SHIFT, WIDEN_REPEAT = 1, 2


@dataclass(frozen=True)
class DecodePlan:
    """How one video is decoded here: its codec and bit depth, the rectangle
    handed on (FFmpeg's crop, rounded as FFmpeg rounds it), the size it is
    scaled to and with which filter, and the sample layout."""

    codec: str
    bit_depth: int
    width: int
    height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    #: The right shift of 16-bit samples: 6 for yuv420p10le, 0 to keep P016's
    #: layout (Vship reads it as 16-bit).
    shift: int = 0
    #: Decode only the luma plane (VMAF reads nothing else).
    luma_only: bool = False
    #: The size the crop is scaled to (0: not scaled), and the filter.
    out_w: int = 0
    out_h: int = 0
    scaler: str = "bicubic"
    #: Hand 8-bit pictures back as 10-bit (WIDEN_SHIFT or WIDEN_REPEAT; 0: no).
    widen: int = 0
    #: Intel and AMD only: 1 scales on the CPU even where the GPU could (the
    #: same picture, slower), to check one against the other. 2 is for the
    #: tests: the GPU's first picture is spoiled, as a faulty driver's would
    #: be, so the decoder finds that it differs from the CPU's.
    cpu_scaling: int = 0

    @property
    def bytes_per_sample(self) -> int:
        """The bytes per sample of the pictures handed back."""
        return 2 if self.bit_depth > 8 or self.widen else 1

    @property
    def output_size(self) -> tuple[int, int]:
        """The size of the pictures handed back."""
        return (self.out_w or self.crop_w, self.out_h or self.crop_h)

    @property
    def scaled(self) -> bool:
        """Whether the pictures are scaled."""
        return self.output_size != (self.crop_w, self.crop_h)

    @property
    def frame_bytes(self) -> int:
        """The bytes of one picture handed back, its planes packed."""
        width, height = self.output_size
        luma = width * height * self.bytes_per_sample
        if self.luma_only:
            return luma
        return luma + 2 * ((width + 1) // 2) * ((height + 1) // 2) * self.bytes_per_sample


def plan_decode(info: VideoInfo, crop: CropBox | None, *, shift: int = 0, luma_only: bool = False,
                size: tuple[int, int] | None = None, algorithm: str = "bicubic", widen: int = 0) -> DecodePlan:
    """The plan for decoding the video here, cropped to `crop` and scaled to
    `size` with `algorithm`. Raises GpuDecodeUnavailableError, with the
    reason, when FFmpeg must decode it instead.

    A picture scaled here is not exactly what FFmpeg's scale filter gives,
    but the user decided that a comparison is the same however it is scaled
    (see ComparisonRecipe.identity_dict)."""
    codec = (info.codec_name or "").casefold()
    if codec not in _CODECS:
        raise GpuDecodeUnavailableError(f"{info.codec_name or 'this codec'} is decoded by FFmpeg")
    depth = _DEPTHS.get((info.pix_fmt or "").casefold())
    if depth is None:
        raise GpuDecodeUnavailableError(f"{info.pix_fmt or 'this pixel format'} is decoded by FFmpeg")
    if info.width <= 0 or info.height <= 0:
        raise GpuDecodeUnavailableError("the video's size is unknown")
    if info.width & 1 or info.height & 1:
        # The decoders refuse it only once started (NVIDIA's decodes to an
        # even size), so the pass would start, fail and be made again.
        raise GpuDecodeUnavailableError("an odd-sized video is decoded by FFmpeg")
    if crop is None or crop.is_noop(info.width, info.height):
        x, y, w, h = 0, 0, info.width, info.height
    else:
        # FFmpeg's crop filter keeps a 4:2:0 crop on whole chroma samples: its
        # left and top edges move to the even sample at or before them, and
        # an odd width or height loses its last column or row.
        x, y, w, h = crop.x & ~1, crop.y & ~1, crop.w & ~1, crop.h & ~1
    if w <= 0 or h <= 0 or x + w > info.width or y + h > info.height:
        raise GpuDecodeUnavailableError("the crop is outside the picture")
    out_w, out_h = size if size is not None else (w, h)
    if out_w <= 0 or out_h <= 0:
        raise GpuDecodeUnavailableError("the size it is scaled to is empty")
    if widen and depth != 8:
        raise GpuDecodeUnavailableError("only 8-bit video is widened")
    return DecodePlan(codec, depth, info.width, info.height, x, y, w, h,
                      shift if depth > 8 else 0, luma_only, out_w, out_h,
                      algorithm if algorithm in _SCALERS else "bicubic", widen)


#: The decoders that scale on their GPU (native/d3d11_scale.h); they leave a
#: note when they scale on the CPU instead.
_GPU_SCALING = ("intel", "amd", MEDIA_FOUNDATION)


def decodes_codec(codec: str, backend: str) -> bool:
    """Whether `backend`'s decoder is used for `codec` (FFmpeg's name)."""
    return codec in _CODECS and (backend == SOFTWARE or codec in _GPU_CODECS)


def _params(plan: DecodePlan, device: int = 0, pool: int = 1, extradata=None, extradata_size: int = 0,
            handover: bool = False) -> _Params:
    """The plan as the decoder libraries take it."""
    return _Params(device, _CODECS[plan.codec][0], plan.bit_depth, plan.width, plan.height,
                   plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h, plan.shift, int(plan.luma_only),
                   pool, extradata, extradata_size, plan.output_size[0], plan.output_size[1], _SCALERS[plan.scaler],
                   plan.widen, int(plan.cpu_scaling), int(handover))


def hands_over_textures(backend: str) -> bool:
    """Whether `backend`'s decoder may give its pictures in place, as
    textures (GpuFrameStream.slot_texture): AMD's, through Windows' own
    decoder (native/mf_frames.cpp), where that is bundled. Whether a given
    stream's pictures come that way depends on its codec."""
    if backend != "amd":
        return False
    try:
        return hasattr(_load(MEDIA_FOUNDATION), "nvf_slot_texture")
    except GpuDecodeUnavailableError:
        return False


def can_hand_over(plan: DecodePlan, backend: str) -> bool:
    """Whether `backend`'s decoder hands `plan`'s pictures over on the GPU
    (GpuFrameStream's handover): NVIDIA's always, and AMD's when they are
    neither scaled nor widened (native/amf_handover.h)."""
    if backend == "nvidia":
        return True
    return backend == "amd" and not plan.scaled and not plan.widen


def scale_picture(plan: DecodePlan, picture: bytes, vendor: int | None, backend: str = "intel") -> bytes | None:
    """For the tests: scales one NV12 or P010 picture (luma rows, then
    interleaved chroma rows, 10-bit samples in the top bits of 16) as `plan`
    says, with the decoders' scaler: on the CPU (`vendor` None), or with its
    shader on the first GPU of PCI vendor `vendor` (0x1414 is Windows'
    software device). Returns None when this PC has no such GPU."""
    lib = _load(backend)
    params = _params(plan)
    out = ctypes.create_string_buffer(plan.frame_bytes)
    error = ctypes.create_string_buffer(2048)
    result = lib.nvf_scale_test(vendor or 0, ctypes.byref(params), picture, int(vendor is not None), out, error,
                                len(error))
    if result == 1:
        return None
    if result != 0:
        raise GpuDecodeFailedError(error.value.decode(errors="replace") or "scaling failed")
    return out.raw


def software_bundled() -> bool:
    """Whether the software decoder can be used, judged from its files alone
    (nothing is loaded): its library and its FFmpeg libraries are there, and
    SOFTWARE_VARIABLE is not set to ffmpeg. This is for planning a run in
    the app's process; the scoring process finds out for sure
    (decoder_supports)."""
    if os.environ.get(SOFTWARE_VARIABLE, "").casefold() == "ffmpeg" or not LIBRARIES[SOFTWARE].is_file():
        return False
    return all(path is not None for path in _software_library_paths()[:len(_FFMPEG_LIBRARIES)])


def software_codecs() -> tuple[list[str], str]:
    """The codecs (FFmpeg's names) the software decoder can decode on this
    PC, after loading its libraries, and why when it can decode none. For
    the self-test."""
    try:
        lib = _load(SOFTWARE)
    except GpuDecodeUnavailableError as error:
        return [], str(error)
    codecs, why = [], ""
    for codec, (number, _filters) in _CODECS.items():
        error = ctypes.create_string_buffer(512)
        if lib.nvf_supports(0, number, 8, 64, 64, error, len(error)):
            codecs.append(codec)
        elif not why:
            why = error.value.decode(errors="replace")
    return codecs, "" if codecs else why


def decoder_pair(decoder: str | tuple[str, str]) -> tuple[str, str]:
    """(the source's decoder, the test video's), from `decoder`, which is
    either one decoder for both or the two."""
    return (decoder, decoder) if isinstance(decoder, str) else (decoder[0], decoder[1])


def device_of(backend: str, gpu: int) -> int:
    """The device a decoder is opened on: `gpu` for NVIDIA's (a CUDA device
    number), and 0 for the others, which find their GPU themselves (or need
    none)."""
    return gpu if backend == "nvidia" else 0


def available(backend: str = "nvidia") -> bool:
    """Whether the decoder library is bundled and loads (the GPU is not asked)."""
    try:
        _load(backend)
    except GpuDecodeUnavailableError:
        return False
    return True


def decoder_supports(device: int, plan: DecodePlan, backend: str = "nvidia") -> tuple[bool, str]:
    """Whether GPU `device`'s decoder takes the plan's codec, bit depth and
    size, and why not; for the software decoder, whether its libraries
    decode the codec.

    The software decoder is never asked to scale: it would scale on one
    thread as each picture is handed on, so a 4K source compared at 1080p
    ran at 52 frames a second, against 128 with FFmpeg scaling it on several
    threads before its pipe."""
    if not decodes_codec(plan.codec, backend):
        return False, f"{plan.codec} is not decoded by the {backend} decoder"
    if backend == SOFTWARE and plan.scaled:
        return False, "the software decoder does not scale; FFmpeg scales it"
    try:
        lib = _load(backend)
    except GpuDecodeUnavailableError as error:
        return False, str(error)
    error = ctypes.create_string_buffer(512)
    ok = lib.nvf_supports(device, _CODECS[plan.codec][0], plan.bit_depth, plan.width, plan.height, error, len(error))
    return bool(ok), error.value.decode(errors="replace")


# ------------------------------------------------------------- the packets

_TB_RE = re.compile(r"#tb 0: (\d+)/(\d+)")


class _PacketReader:
    """An FFmpeg that copies the video's first video stream out of its
    container. It writes the packets to its standard output (-f data), and
    each packet's timestamp and size to a framecrc listing on a named pipe
    served here. A thread reads the listing as it arrives, so neither of
    FFmpeg's outputs waits for the other."""

    def __init__(self, path: Path, codec: str, process_handle=None) -> None:
        bsf = _CODECS[codec][1]
        filters = ["-bsf:v", bsf] if bsf else []
        self.pipe_path = rf"\\.\pipe\vml-gpu-frames-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._pipe = _winapi.CreateNamedPipe(
            self.pipe_path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,
            1, 1024 * 1024, 1024 * 1024, 0, _winapi.NULL)
        # -copyinkf copies every packet, as a decoder would be given them
        # (packets() checks that the first is a keyframe).
        self.command = [
            # -y: the listing's pipe already exists (this process serves it).
            ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
            "-i", str(Path(path).resolve()),
            "-map", f"0:{VIDEO_STREAM}", "-c:v", "copy", "-copyinkf", *filters, "-f", "data", "pipe:1",
            "-map", f"0:{VIDEO_STREAM}", "-c:v", "copy", "-copyinkf", *filters, "-flush_packets", "1", "-f", "framecrc",
            self.pipe_path,
        ]
        self._process_handle = process_handle
        self._listing: queue.Queue[tuple[int, int, int] | BaseException | None] = queue.Queue()
        self.time_base: Fraction | None = None
        self._time_base_known = threading.Event()
        self.process: subprocess.Popen | None = None
        self._stdout = None
        self._stderr: list[str] = []
        #: A warning from FFmpeg about timestamps (see _TIMESTAMP_WARNING).
        self.timestamp_warning: str | None = None
        self._threads: list[threading.Thread] = []
        self._closed = False

    def start(self) -> None:
        """Starts FFmpeg, and the threads that read its listing and errors."""
        read_handle, write_handle = _winapi.CreatePipe(None, _ES_PIPE_BYTES)
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
        try:
            self.process = proc_util.popen(self.command, stdin=subprocess.DEVNULL, stdout=write_fd,
                                           stderr=subprocess.PIPE)
        except BaseException:
            os.close(write_fd)
            _winapi.CloseHandle(read_handle)
            raise
        finally:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        self._stdout = open(msvcrt.open_osfhandle(read_handle, os.O_RDONLY), "rb", buffering=0)  # noqa: SIM115
        if self._process_handle is not None:
            self._process_handle.attach(self.process.pid)
        for target, name in ((self._read_listing, "listing"), (self._drain_stderr, "stderr")):
            thread = threading.Thread(target=target, name=f"gpu-frames-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _read_listing(self) -> None:
        """The listing thread: reads the time base and each packet's
        timestamp, size and flags, and checks the decode times go up."""
        try:
            try:
                _winapi.ConnectNamedPipe(self._pipe, _winapi.NULL)
            except OSError as error:
                # ERROR_PIPE_CONNECTED: FFmpeg connected first. ERROR_NO_DATA:
                # FFmpeg connected, wrote and closed before this connected;
                # what it wrote is still there to read, so neither is an
                # error (treating them as one lost all of a short run on a
                # busy PC).
                if error.winerror not in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
                    raise
            fd = msvcrt.open_osfhandle(self._pipe, os.O_RDONLY)
            self._pipe = None
            last_dts = None
            with open(fd, "rb") as listing:
                for raw in listing:
                    line = raw.decode("ascii", errors="replace").strip()
                    if line.startswith("#"):
                        match = _TB_RE.match(line)
                        if match:
                            self.time_base = Fraction(int(match.group(1)), int(match.group(2)))
                            self._time_base_known.set()
                        continue
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) < 6 or not fields[0].isdigit():
                        continue
                    printed = next((f for f in fields[6:] if f.startswith("F=")), None)
                    # framecrc prints a packet's flags only when they are not
                    # just "keyframe".
                    flags = _KEY if printed is None else int(printed[2:], 16)
                    # A decode time that does not go up is one FFmpeg's muxer
                    # rewrote (to the previous one, and the presentation time
                    # with it). It is caught here, with the packet, because
                    # FFmpeg's warning about it can come only after a run
                    # that stopped early has already used the packet.
                    dts = int(fields[1])
                    if dts != _NOPTS:
                        if last_dts is not None and dts <= last_dts:
                            raise GpuDecodeFailedError("the video's decode timestamps do not go up")
                        last_dts = dts
                    self._listing.put((int(fields[2]), int(fields[4]), flags))
        except BaseException as error:  # passed on to packets(), which raises it
            self._listing.put(error)
        finally:
            if self._pipe is not None:
                with contextlib.suppress(OSError):
                    _winapi.CloseHandle(self._pipe)
                self._pipe = None
            self._time_base_known.set()
            self._listing.put(None)

    def _drain_stderr(self) -> None:
        """The error thread: keeps the start of FFmpeg's error output, and
        notes its first warning about timestamps."""
        assert self.process is not None and self.process.stderr is not None
        with contextlib.suppress(OSError, ValueError):
            for raw in self.process.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if len(self._stderr) < 50:
                    self._stderr.append(line)
                if self.timestamp_warning is None and _TIMESTAMP_WARNING.search(line):
                    self.timestamp_warning = line

    def _next_listing(self):
        """Returns the next (pts, size, flags), or None at the end of the
        listing. If FFmpeg ended without opening the listing's pipe, the pipe
        is connected to here, so that the wait ends."""
        while True:
            try:
                return self._listing.get(timeout=0.2)
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    with contextlib.suppress(OSError), open(self.pipe_path, "wb"):
                        pass

    def packets(self):
        """Yields (packet, pts, flags) in decode order, then checks that FFmpeg
        succeeded and did not rewrite any timestamps."""
        first = True
        while True:
            item = self._next_listing()
            if item is None:
                break
            if isinstance(item, GpuDecodeFailedError):
                raise item
            if isinstance(item, BaseException):
                raise GpuDecodeFailedError(f"reading FFmpeg's packet list failed: {item}") from item
            pts, size, flags = item
            if pts == _NOPTS:
                raise GpuDecodeFailedError("a packet has no timestamp")
            if first and not flags & _KEY:
                raise GpuDecodeFailedError("the video does not start with a keyframe")
            if self.timestamp_warning is not None:
                raise GpuDecodeFailedError(f"FFmpeg rewrote the video's timestamps: {self.timestamp_warning}")
            first = False
            data = bytearray(size)
            view = memoryview(data)
            filled = 0
            while filled < size:
                count = self._stdout.readinto(view[filled:])
                if not count:
                    raise GpuDecodeFailedError("FFmpeg's packets ended before its packet list")
                filled += count
            yield data, pts, flags
        code = self.process.wait() if self.process is not None else 0
        for thread in self._threads:
            thread.join(timeout=10)  # so that every warning has been read
        if self._closed:
            return
        if self.timestamp_warning is not None:
            raise GpuDecodeFailedError(f"FFmpeg rewrote the video's timestamps: {self.timestamp_warning}")
        if code != 0 or first:
            # No packets at all is a failure too: FFmpeg can exit with 0 when
            # it refused to open an output.
            raise GpuDecodeFailedError(f"FFmpeg failed copying the video's packets (exit code {code})"
                                   + (f": {self.stderr_text()}" if self.stderr_text() else ""))

    def wait_time_base(self, timeout: float) -> Fraction | None:
        """The stream's time base, once FFmpeg has listed it (or None)."""
        self._time_base_known.wait(timeout)
        return self.time_base

    def stderr_text(self) -> str:
        """The end of FFmpeg's error output, for messages."""
        return "\n".join(self._stderr).strip()[-500:]

    def close(self) -> None:
        """Ends FFmpeg and the reading threads, and closes the pipes."""
        self._closed = True
        if self.process is not None:
            if self.process.poll() is None:
                with contextlib.suppress(OSError):
                    proc_util.terminate(self.process)
            with contextlib.suppress(Exception):
                self.process.wait(timeout=10)
            if self._process_handle is not None:
                self._process_handle.detach(self.process.pid)
        with contextlib.suppress(OSError), open(self.pipe_path, "wb"):
            pass  # releases a listing thread still waiting for FFmpeg to connect
        for thread in self._threads:
            thread.join(timeout=10)
        if self._stdout is not None:
            with contextlib.suppress(OSError):
                self._stdout.close()
        if self._pipe is not None:
            with contextlib.suppress(OSError):
                _winapi.CloseHandle(self._pipe)
            self._pipe = None


def _extradata(path: Path) -> bytes:
    """The video stream's configuration (FFmpeg's extradata), as ffprobe
    shows it; empty when there is none or it cannot be read."""
    command = [ffprobe_path(), "-v", "error", "-select_streams", VIDEO_STREAM, "-show_entries", "stream=extradata",
               "-show_data", "-of", "default=noprint_wrappers=1", str(Path(path).resolve())]
    try:
        output = proc_util.run(command, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return b""
    data = bytearray()
    for line in output.splitlines():
        match = re.match(r"^[0-9a-f]{8}: ((?:[0-9a-f]{2,4} ?)+)", line.strip())
        if match:
            data += bytes.fromhex(match.group(1).replace(" ", ""))
    return bytes(data)


def _av1_sequence_header(path: Path) -> bytes:
    """The AV1 sequence header OBUs from the stream's configuration (its av1C
    box without the box's own 4-byte header), as FFmpeg's NVDEC decoder
    gives them to NVIDIA's parser. Empty when there are none."""
    data = _extradata(path)
    if len(data) > 4 and data[0] & 0x80:
        return data[4:]
    return b""


def _configuration(path: Path, codec: str) -> bytes:
    """The part of the stream's configuration a decoder is given besides the
    packets: AV1's sequence header, or FFV1's configuration record (which a
    version 3 stream needs to read its frames)."""
    if codec == "av1":
        return _av1_sequence_header(path)
    return _extradata(path) if codec == "ffv1" else b""


# ----------------------------------------------------------------- streams

class GpuFrameStream:
    """One video decoded on GPU `device`, its pictures held in a pool of
    `pool` slots in GPU memory, in display order. A thread feeds FFmpeg's
    packets to the decoder; next() hands out the pictures, and each slot is
    given back with release() once its picture has been copied.

    With `handover`, AMD's decoder keeps its pictures on the GPU, for pin,
    download_planes, import_memory and copy_luma (NVIDIA's decoder always
    does). It does so where it can (can_hand_over): through Windows' own
    decoder where that opens (_open_media_foundation), and otherwise through
    AMF. Where it cannot, or when its Vulkan cannot start, it decodes without
    handover, and self.handover says so."""

    def __init__(self, info: VideoInfo, plan: DecodePlan, device: int = 0, *, pool: int = 4,
                 process_handle=None, backend: str = "nvidia", handover: bool = False) -> None:
        self.info = info
        self.plan = plan
        self.backend = backend
        if not decodes_codec(plan.codec, backend):
            raise GpuDecodeUnavailableError(f"{plan.codec} is not decoded by the {backend} decoder")
        self._lib = _load(backend)
        self._handle = None
        self._reader = _PacketReader(info.path, plan.codec, process_handle)
        self._extradata = _configuration(info.path, plan.codec)
        self._extradata_buffer = ctypes.create_string_buffer(self._extradata) if self._extradata else None
        extradata = ctypes.cast(self._extradata_buffer, ctypes.c_void_p) if self._extradata_buffer else None
        #: Whether the pictures are handed over on the GPU (so pin and
        #: import_memory work).
        self.handover = backend == "nvidia" or (handover and can_hand_over(plan, backend))
        error = ctypes.create_string_buffer(1024)
        handle = None
        if self.handover and backend == "amd":
            handle = self._open_media_foundation(plan, device, pool, extradata)
        if self.handover and backend != "nvidia" and not handle:
            params = _params(plan, device, pool, extradata, len(self._extradata), handover=True)
            handle = self._lib.nvf_open(ctypes.byref(params), error, len(error))
            if not handle:
                _log.info("%s: the GPU's pictures go through system memory (%s)", info.path.name,
                          error.value.decode(errors="replace"))
                self.handover = False
        if not handle:
            params = _params(plan, device, pool, extradata, len(self._extradata))
            handle = self._lib.nvf_open(ctypes.byref(params), error, len(error))
        if not handle:
            self._reader.close()  # closes its pipe
            raise GpuDecodeUnavailableError(error.value.decode(errors="replace") or f"{self.decoder} could not start")
        self._handle = handle
        self.frame_bytes = plan.frame_bytes
        self._feed_error: BaseException | None = None
        #: The timestamps of the packets fed whose pictures have not come out
        #: yet, smallest first (a heap), and how many packets were fed.
        self._waiting: list[int] = []
        self._fed_count = 0
        self._fed_lock = threading.Lock()
        #: The timestamps of packets the demuxer marked "discard": the frames
        #: an MP4 edit list cuts off. They are decoded (later frames may refer
        #: to them) but never handed out, as FFmpeg's decoding drops them.
        self._discard: set[int] = set()
        self._shown_count = 0
        self._last_shown: int | None = None
        self._finished = False
        self._closing = False
        self._feeder = threading.Thread(target=self._feed, name="gpu-frames-feed", daemon=True)

    def _open_media_foundation(self, plan: DecodePlan, device: int, pool: int, extradata) -> int | None:
        """Opens AMD's GPU decoder through Windows' own decoder (Media
        Foundation's, native/mf_frames.cpp) rather than AMF: the same decoder
        hardware, faster and with less CPU. Returns its handle, and the
        stream then uses that library. Returns None where it cannot be used
        (not bundled, or no decoder for the codec on the GPU; HEVC's and
        AV1's come with Windows' video extensions), and AMF is used instead."""
        try:
            lib = _load(MEDIA_FOUNDATION)
        except GpuDecodeUnavailableError as error:
            _log.info("%s: decoded by AMF (%s)", self.info.path.name, error)
            return None
        error = ctypes.create_string_buffer(1024)
        params = _params(plan, device, pool, extradata, len(self._extradata), handover=True)
        handle = lib.nvf_open(ctypes.byref(params), error, len(error))
        if not handle:
            _log.info("%s: decoded by AMF, not Windows' decoder (%s)", self.info.path.name,
                      error.value.decode(errors="replace"))
            return None
        self._lib = lib
        return handle

    @property
    def decoder(self) -> str:
        """The decoder, for messages: "the GPU's decoder" or "the software decoder"."""
        return "the software decoder" if self.backend == SOFTWARE else "the GPU's decoder"

    def wait_time_base(self, timeout: float = 60.0) -> Fraction:
        """The stream's time base, once FFmpeg has reported it (it does so
        before the first packet). Used when pairing by timestamp."""
        self._reader.wait_time_base(timeout)
        return self.time_base

    @property
    def time_base(self) -> Fraction:
        """The stream's time base; GpuDecodeFailedError if FFmpeg gave none."""
        time_base = self._reader.time_base
        if time_base is None:
            raise GpuDecodeFailedError("FFmpeg did not report the stream's time base")
        return time_base

    def start(self) -> None:
        """Starts FFmpeg and the thread that feeds its packets to the decoder."""
        self._reader.start()
        self._feeder.start()

    def _feed(self) -> None:
        """The feeding thread: gives FFmpeg's packets to the decoder, noting
        each packet's timestamp for the checks in _take and verify."""
        lib, handle = self._lib, self._handle
        try:
            for data, pts, flags in self._reader.packets():
                with self._fed_lock:
                    heapq.heappush(self._waiting, pts)
                    self._fed_count += 1
                    if flags & _DISCARD:
                        self._discard.add(pts)
                buffer = (ctypes.c_ubyte * len(data)).from_buffer(data)
                code = lib.nvf_push(handle, buffer, len(data), pts)
                if code != 0:
                    return  # a decoder error, or a close: next() reports it
            lib.nvf_finish(handle)
        except BaseException as error:
            if not self._closing:
                self._feed_error = error
            lib.nvf_abort(handle)

    def _error(self) -> str:
        """The decoder's last error message."""
        text = ctypes.create_string_buffer(1024)
        self._lib.nvf_error(self._handle, text, len(text))
        return text.value.decode(errors="replace")

    def next(self, timeout_ms: int = 100) -> tuple[int, int] | None:
        """Returns the next picture's (slot, pts), or None at the end of the
        video. Raises TimeoutError after `timeout_ms`, so the caller can check
        for Cancel, and GpuDecodeFailedError when decoding failed or the
        decoder's pictures do not match the packets (see verify)."""
        slot, pts = ctypes.c_int(), ctypes.c_longlong()
        while True:
            code = self._lib.nvf_pop(self._handle, timeout_ms, ctypes.byref(slot), ctypes.byref(pts))
            if code != _NVF_FRAME:
                break
            try:
                discarded = self._take(pts.value)
            except GpuDecodeFailedError as error:
                self.release(slot.value)
                error.backend = self.backend
                raise
            if not discarded:
                return slot.value, pts.value
            self.release(slot.value)
        if code == _NVF_TIMEOUT:
            raise TimeoutError
        if code == _NVF_END:
            self._finished = True
            self.verify()
            return None
        if self._feed_error is not None:
            raise GpuDecodeFailedError(str(self._feed_error), self.backend) from self._feed_error
        if code == _NVF_ABORTED:
            raise GpuDecodeFailedError("decoding was stopped", self.backend)
        raise GpuDecodeFailedError(self._error() or f"{self.decoder} failed", self.backend)

    def pictures(self, check_cancel: Callable[[], None]) -> Callable[[], int | None]:
        """This video's pictures for frame_sync.frame_pairs: a function that
        returns the next picture's slot, or None at the video's end, and
        calls `check_cancel` every 100 ms while it waits for a picture."""
        def next_picture() -> int | None:
            while True:
                check_cancel()
                try:
                    picture = self.next(100)
                except TimeoutError:
                    continue
                return None if picture is None else picture[0]
        return next_picture

    def _take(self, pts: int) -> bool:
        """Checks the picture stamped `pts`, as it comes out, against the
        packets fed: it must come in timestamp order, and be the next
        packet's (every packet stamped earlier has had its picture). Checked
        here, a picture the decoder dropped or made up fails at once, not a
        whole pass later at the end. Returns whether the picture is one the
        demuxer discards."""
        if self._last_shown is not None and pts <= self._last_shown:
            raise GpuDecodeFailedError(f"{self.decoder} gave pictures out of order")
        with self._fed_lock:
            # An earlier-stamped packet is in the heap only if it was fed, so
            # its picture was due before this one. A packet fed after this
            # picture came out would come out of order, which fails above.
            if self._waiting and self._waiting[0] < pts:
                raise GpuDecodeFailedError(
                    f"{self.decoder} gave {self._shown_count} pictures for {self._shown_count + 1} packets")
            if not self._waiting or self._waiting[0] != pts:
                raise GpuDecodeFailedError(f"{self.decoder} gave pictures the packets do not have")
            heapq.heappop(self._waiting)
            discarded = pts in self._discard
        self._shown_count += 1
        self._last_shown = pts
        return discarded

    def verify(self) -> None:
        """Raises GpuDecodeFailedError unless the pictures handed out match the
        packets, one each, in timestamp order, as FFmpeg's decoding of the
        stream gives them. Each picture is checked as it comes out (_take);
        this checks, after the end, that no packet is left without its
        picture."""
        if not self._finished:
            return
        with self._fed_lock:
            fed = self._fed_count
        if self._shown_count != fed:
            raise GpuDecodeFailedError(f"{self.decoder} gave {self._shown_count} pictures for {fed} packets",
                                       self.backend)

    def release(self, slot: int) -> None:
        """Gives a slot back to the decoder once its picture has been copied."""
        self._lib.nvf_release(self._handle, slot)

    def download(self, slot: int, address: int) -> None:
        """Copies the slot's picture, planes packed, to page-locked memory at
        `address` (frame_bytes long)."""
        if self._lib.nvf_download(self._handle, slot, address) != 0:
            raise GpuDecodeFailedError(self._error() or f"{self.decoder} failed", self.backend)

    def copy_luma(self, slot: int, address: int, pitch: int) -> None:
        """Copies the slot's luma plane to GPU memory at `address`, with rows
        `pitch` bytes apart."""
        if self._lib.nvf_copy_luma(self._handle, slot, address, pitch) != 0:
            raise GpuDecodeFailedError(self._error() or "the GPU's decoder failed")

    def timeline(self) -> int | None:
        """The timeline semaphore that copy_luma_async's copies signal (AMD's
        handover), as a Win32 handle for another Vulkan device on the GPU to
        import (the caller closes it). None where there is none; the copies
        are then waited for instead."""
        if not self.handover or not hasattr(self._lib, "nvf_timeline"):
            return None
        handle = ctypes.c_void_p()
        if self._lib.nvf_timeline(self._handle, ctypes.byref(handle)) != 0 or not handle.value:
            return None
        return handle.value

    def copy_luma_async(self, slot: int, address: int, pitch: int) -> int:
        """Like copy_luma, but without waiting for the copy. Returns the
        timeline semaphore's value once the copy is done, for the GPU to wait
        for (0 if the copy was waited for after all)."""
        value = ctypes.c_ulonglong()
        if self._lib.nvf_copy_luma_async(self._handle, slot, address, pitch, ctypes.byref(value)) != 0:
            raise GpuDecodeFailedError(self._error() or "the GPU's decoder failed")
        return value.value

    @property
    def gives_textures(self) -> bool:
        """Whether slot_texture gives this stream's pictures."""
        return self.handover and hasattr(self._lib, "nvf_slot_texture")

    def slot_texture(self, slot: int) -> tuple[int, int, int, int, int]:
        """The slot's picture where the decoder left it: its texture's KMT
        handle (a Direct3D 11 texture shared without a mutex), the texture's
        width and height, and the position (x, y) of the picture's first
        sample in it. Valid until the slot is released."""
        texture, width, height = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_uint()
        x, y = ctypes.c_int(), ctypes.c_int()
        if self._lib.nvf_slot_texture(self._handle, slot, ctypes.byref(texture), ctypes.byref(width),
                                      ctypes.byref(height), ctypes.byref(x), ctypes.byref(y)) != 0:
            raise GpuDecodeFailedError(self._error() or "the decoder's picture is not a texture")
        return texture.value, width.value, height.value, x.value, y.value

    def copy_planes_async(self, slot: int, addresses: tuple[int, ...], pitches: tuple[int, ...]) -> int:
        """Like copy_planes, but without waiting, as copy_luma_async."""
        destinations = (ctypes.c_ulonglong * 3)(*addresses)
        rows = (ctypes.c_longlong * 3)(*pitches)
        value = ctypes.c_ulonglong()
        if self._lib.nvf_copy_planes_async(self._handle, slot, destinations, rows, ctypes.byref(value)) != 0:
            raise GpuDecodeFailedError(self._error() or "the GPU's decoder failed")
        return value.value

    # Pictures handed over without a CPU copy (NVIDIA's decoder, and AMD's
    # with handover).

    def pin(self, address: int, size: int) -> bool:
        """Page-locks `size` bytes of this process's memory at `address`, so
        that the GPU writes downloads into it directly (otherwise the driver
        copies them into ordinary memory on the CPU). Returns False where it
        cannot; downloads then work as before. With AMD's decoder, memory
        pinned by one stream is written into by every stream of the process
        that hands over."""
        return self.handover and self._lib.nvf_pin(self._handle, address, size) == 0

    def unpin(self, address: int) -> None:
        """Undoes pin."""
        if self._handle is not None:
            self._lib.nvf_unpin(self._handle, address)

    def download_planes(self, slot: int, addresses: tuple[int | None, ...], pitches: tuple[int, ...]) -> None:
        """Copies the slot's Y, U and V planes each to its own address, with
        rows `pitches` bytes apart (a libvmaf picture's planes). An address
        of None skips that plane."""
        planes = (ctypes.c_void_p * 3)(*addresses)
        rows = (ctypes.c_longlong * 3)(*pitches)
        if self._lib.nvf_download_planes(self._handle, slot, planes, rows) != 0:
            raise GpuDecodeFailedError(self._error() or "the GPU's decoder failed")

    def copy_planes(self, slot: int, addresses: tuple[int, ...], pitches: tuple[int, ...]) -> None:
        """Copies the slot's Y, U and V planes on the GPU into memory that
        import_memory imported, each to its own address (0 skips that plane),
        with rows `pitches` bytes apart."""
        destinations = (ctypes.c_ulonglong * 3)(*addresses)
        rows = (ctypes.c_longlong * 3)(*pitches)
        if self._lib.nvf_copy_planes(self._handle, slot, destinations, rows) != 0:
            raise GpuDecodeFailedError(self._error() or "the GPU's decoder failed")

    def import_memory(self, handle: int, size: int, exporter=None) -> tuple[int, int] | None:
        """Imports GPU memory that Vulkan allocated on this GPU and exported
        as a Win32 handle (vmaf_vulkan's vv_export: one buffer's allocation
        of `size` bytes), so copy_luma can write into it. Returns (its
        address, the value unimport takes), or None where the driver cannot
        share it.

        With AMD's decoder, memory one stream imports is copied into by
        every stream of the process that hands over. `exporter`
        (libvmaf-fast's SharedDevice, from vv_shared_device) names the GPU,
        the driver and the memory type; without it, or for another GPU or
        driver, nothing is imported."""
        if not self.handover:
            return None
        address, memory = ctypes.c_ulonglong(), ctypes.c_void_p()
        if self.backend == "nvidia":
            result = self._lib.nvf_import(self._handle, handle, size, 1, ctypes.byref(address), ctypes.byref(memory))
        elif exporter is None:
            return None
        else:
            result = self._lib.nvf_import_vulkan(self._handle, handle, size, exporter.memory_type,
                                                 bytes(exporter.device_uuid), bytes(exporter.driver_uuid),
                                                 ctypes.byref(address), ctypes.byref(memory))
        if result != 0:
            return None
        return address.value, memory.value

    def unimport(self, memory: int) -> None:
        """Undoes import_memory."""
        if self._handle is not None:
            self._lib.nvf_unimport(self._handle, memory)

    def abort(self) -> None:
        """Ends every wait in next() and in the feeding thread; callable from
        any thread."""
        if self._handle is not None:
            self._closing = True
            self._lib.nvf_abort(self._handle)

    def stats(self) -> _Info:
        """What the decoder found in the stream, and its picture counts."""
        info = _Info()
        self._lib.nvf_info(self._handle, ctypes.byref(info))
        return info

    def scale_note(self) -> str:
        """Why the pictures are scaled on the CPU when the GPU was meant to
        scale them (Intel, AMD); empty when the GPU scales them, or nothing
        is scaled."""
        if self.backend not in _GPU_SCALING or self._handle is None:
            return ""
        text = ctypes.create_string_buffer(1024)
        self._lib.nvf_scale_note(self._handle, text, len(text))
        return text.value.decode(errors="replace")

    def close(self) -> None:
        """Stops decoding and frees everything; safe to call twice."""
        if self._handle is None:
            return
        self._closing = True
        if note := self.scale_note():
            _log.warning("%s: scaled on the CPU, not the GPU: %s", self.info.path.name, note)
        self._lib.nvf_abort(self._handle)
        self._reader.close()
        if self._feeder.ident is not None:
            self._feeder.join(timeout=30)
        if not self._feeder.is_alive():
            # The decoder is closed on a thread of its own, waited for only so
            # long: a close that never returned would hang the whole run at
            # its end, scores and all (AMD's decoder did this after failing
            # on 10-bit H.264, which it is no longer given).
            closer = threading.Thread(target=self._lib.nvf_close, args=(self._handle,), name="gpu-frames-close",
                                      daemon=True)
            closer.start()
            closer.join(_CLOSE_SECONDS)
            if closer.is_alive():
                global _stuck_closes
                _stuck_closes += 1
                _log.error("%s: the %s decoder did not close in %.0f s; it is left open", self.info.path.name,
                           self.backend, _CLOSE_SECONDS)
        else:  # never seen; leaking the decoder is safer than freeing it under the running thread
            _log.error("The %s decoder's feeding thread did not stop; its decoder is left open", self.backend)
        self._handle = None
