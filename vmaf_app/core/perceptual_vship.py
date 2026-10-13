"""Calculates SSIMULACRA2, Butteraugli and CVVDP on the GPU, through Vship's
C API. The main entry points are run_vship_task and apply_vship_cpu_fallback.

The API is Vship 5.1's (Vship_InitHandler and Vship_ComputeHandler; see
https://codeberg.org/Line-fr/Vship/src/tag/v5.1.1/src/VshipAPI.h). The
older per-metric functions are a deprecated compatibility layer in 5.1.

SSIMULACRA2 and Butteraugli score each frame pair on its own. CVVDP is
temporal, and scores the video as a whole (see _CvvdpLane).

FFmpeg decodes, crops, samples and scales the frames. It is the same FFmpeg
as the rest of the app, so every codec it reads works, VVC included, with
hardware decoding for each video whose codec the GPU decodes. It streams
tightly packed frames into rings of pinned host buffers (see _FrameStream).
A video that the GPU's own decoder can decode (NVIDIA's with Vship's CUDA
build, Intel's or AMD's with any build) is decoded in the scoring process
instead (gpu_frames, _NativeFrameStream). FFmpeg then only copies its
packets out of the container, and each picture goes into the ring without
FFmpeg's CPU copies and the pipe; the pictures are the same.

Vship converts each frame from the colour space it is described in
(_vship_colorspace), and calculates the metric on the GPU: through Vship's
CUDA build on NVIDIA, its HIP build on AMD, or its Vulkan build, which runs
on any GPU with a Vulkan driver (see VSHIP_BUILDS). Vship takes a frame as
three planes, so a hardware-decoded frame that crosses the pipe in the
decoder's NV12 or P010 layout has only its chroma split into planes here
(see _passthrough_format). No FFVship or FFMS2 program is bundled.
"""
from __future__ import annotations

import collections
import contextlib
import ctypes
import logging
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from vmaf_app.core import frame_sync_by_time, gpu_frames, temp_files
from vmaf_app.core import proc as proc_util
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.colour import UnsupportedColourError, video_colour
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.cvvdp import VSHIP_MODEL_KEY, CvvdpSettings, vship_display_json
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, ffmpeg_path
from vmaf_app.core.frame_coverage import short_comparison
from vmaf_app.core.frame_sync import (
    by_timestamp,
    duration_args,
    frame_pairs,
    frame_rate,
    frames_within,
    numbering_filters,
)
from vmaf_app.core.geometry import content_size
from vmaf_app.core.gpu import (
    GPU_PASS,
    GPU_WAIT_MESSAGE,
    PCI_VENDORS,
    HwAccelPlan,
    hw_native_format,
    hwaccel_args,
    pick_decode,
)
from vmaf_app.core.isolated import IsolatedCrashError, run_isolated
from vmaf_app.core.metric_cache import VSHIP_COLOR_TAGS
from vmaf_app.core.metric_results import (
    FrameMetricResult,
    MetricProvenance,
    MetricResultSet,
    SequenceMetricResult,
)
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import CropBox, GpuVendor, ScaleDirection, VideoInfo
from vmaf_app.core.perceptual_cpu import (
    LONG_CPU_RUN_SECONDS,
    ComparisonCutShortError,
    PerceptualCancelled,
    PerceptualRunError,
    PerceptualTaskOutput,
    _validate_pair,
    compared_seconds,
)
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.status import GPU_PASS as GPU_PASS_STATUS
from vmaf_app.core.status import GPU_WAIT, Status

BACKEND_ID = "perceptual"
_METRICS = {"ssimulacra2", "butteraugli", "cvvdp"}
#: Scored on the GPU only: the official CPU implementation needs PyTorch
#: (almost 1 GB) and takes seconds per 4K frame.
GPU_ONLY_METRICS = frozenset({"cvvdp"})


_log = logging.getLogger(__name__)


class VshipUnavailableError(PerceptualRunError):
    """Vship could not be loaded or run, so the metrics can safely be
    calculated on the CPU instead."""


class VshipPassesFailedError(VshipUnavailableError):
    """Every metric's GPU pass failed. The message is the first error's text,
    and `failures` holds each metric's own reason, so that no metric is
    reported with another's reason (such as CVVDP given SSIMULACRA2's
    out-of-memory error)."""

    def __init__(self, message: str, failures: dict[str, str]) -> None:
        super().__init__(message)
        self.failures = dict(failures)


class _Subsampling(ctypes.Structure):
    """The chroma subsampling in Vship_ColorSpace_t: two shifts."""

    _fields_ = [("subw", ctypes.c_int), ("subh", ctypes.c_int)]


class _Crop(ctypes.Structure):
    """The crop in Vship_ColorSpace_t."""

    _fields_ = [(name, ctypes.c_int) for name in ("top", "bottom", "left", "right")]


class _Colorspace(ctypes.Structure):
    """Vship_ColorSpace_t: how Vship reads a frame's samples."""

    _fields_ = [
        ("width", ctypes.c_int64), ("height", ctypes.c_int64),
        ("target_width", ctypes.c_int64), ("target_height", ctypes.c_int64),
        ("sample", ctypes.c_int), ("range", ctypes.c_int),
        ("subsampling", _Subsampling), ("chromaLocation", ctypes.c_int),
        ("colorFamily", ctypes.c_int), ("YUVMatrix", ctypes.c_int),
        ("transferFunction", ctypes.c_int), ("primaries", ctypes.c_int),
        ("crop", _Crop),
    ]


class _Version(ctypes.Structure):
    """Vship's version, and the backend it was built for."""

    _fields_ = [
        ("major", ctypes.c_int), ("minor", ctypes.c_int),
        ("minorMinor", ctypes.c_int), ("backend", ctypes.c_int),
    ]


class _DeviceInfo(ctypes.Structure):
    """What Vship reports about a GPU."""

    _fields_ = [
        ("name", ctypes.c_char * 256), ("VRAMSize", ctypes.c_uint64),
        ("integrated", ctypes.c_int), ("MultiProcessorCount", ctypes.c_int),
        ("WarpSize", ctypes.c_int),
        # Added in Vship 5.0. GetDeviceInfo writes this trailing feature matrix
        # even for backends where it is unused; reserve the full native struct.
        ("vulkanFeatureMatrix", ctypes.c_bool * 26),
    ]


#: Vship_Handle: one metric's handler, opaque to the caller.
_Handle = ctypes.c_void_p

#: Vship_StructType: which init or score struct a void* argument holds.
_INIT_SSIMULACRA2, _INIT_BUTTERAUGLI, _INIT_CVVDP = 1, 2, 3
_SCORE_SSIMULACRA2, _SCORE_BUTTERAUGLI, _SCORE_CVVDP = 4, 5, 6


class _InitSsimulacra2(ctypes.Structure):
    """Vship_InitSSIMULACRA2_1."""
    _fields_ = [("structType", ctypes.c_int), ("src_colorspace", _Colorspace),
                ("dis_colorspace", _Colorspace), ("gpu_id", ctypes.c_int)]


class _InitButteraugli(ctypes.Structure):
    """Vship_InitButteraugli_1."""
    _fields_ = [("structType", ctypes.c_int), ("src_colorspace", _Colorspace),
                ("dis_colorspace", _Colorspace), ("Qnorm", ctypes.c_int),
                ("intensity_multiplier", ctypes.c_float), ("gpu_id", ctypes.c_int)]


class _InitCvvdp(ctypes.Structure):
    """Vship_InitCVVDP_1."""
    _fields_ = [("structType", ctypes.c_int), ("src_colorspace", _Colorspace),
                ("dis_colorspace", _Colorspace), ("fps", ctypes.c_float),
                ("resizeToDisplay", ctypes.c_bool), ("model_key_cstr", ctypes.c_char_p),
                ("model_config_json_cstr", ctypes.c_char_p), ("gpu_id", ctypes.c_int)]


class _ScoreSsimulacra2(ctypes.Structure):
    """Vship_ScoreSSIMULACRA2."""
    _fields_ = [("structType", ctypes.c_int), ("score", ctypes.c_double)]


class _ScoreButteraugli(ctypes.Structure):
    """Vship_ScoreButteraugli. dstp NULL: no distortion map."""
    _fields_ = [("structType", ctypes.c_int), ("normQ", ctypes.c_double), ("norm3", ctypes.c_double),
                ("norminf", ctypes.c_double), ("dstp", ctypes.c_void_p), ("dststride", ctypes.c_int64)]


class _ScoreCvvdp(ctypes.Structure):
    """Vship_ScoreCVVDP. dstp NULL: no distortion map."""
    _fields_ = [("structType", ctypes.c_int), ("score", ctypes.c_double),
                ("dstp", ctypes.c_void_p), ("dststride", ctypes.c_int64)]


_U8P = ctypes.POINTER(ctypes.c_uint8)
_PLANES = ctypes.POINTER(_U8P)
_I64_3 = ctypes.c_int64 * 3
#: Vship_Sample_t for each integer bit depth. Vship takes any of these for
#: any chroma subsampling (Vship_ColorSpace_t.subsampling is two shifts),
#: and half and float samples too; the layouts are FFmpeg's.
_VSHIP_ENUMS = {8: 2, 9: 3, 10: 5, 12: 7, 14: 9, 16: 11}
#: Subsampling shifts (width, height) of FFmpeg's planar YUV layouts. 4:1:0
#: (yuv410p) has one chroma sample per 4x4 block, so (2, 2). Don't use (2, 1)
#: as FFVship 5.1.1 does (Vship issue 21): Vship then reads chroma planes
#: twice the height FFmpeg wrote, and the run stops partway through the
#: first frame.
_SUBSAMPLING = {"410": (2, 2), "411": (2, 0), "420": (1, 1), "422": (1, 0), "440": (0, 1), "444": (0, 0)}
#: The first Vship version that reads 4:1:0 correctly (Vship issue 21). Older
#: versions give nonsense with subsampling (2, 2) (SSIMULACRA2 -52477,
#: Butteraugli not finite), so for them FFmpeg upsamples 4:1:0 to 4:4:4.
_READS_410_SINCE = (5, 1, 2)


@dataclass(frozen=True, slots=True)
class _ImageFormat:
    """The pixel format FFmpeg sends (pixel_format), and how Vship reads it.
    `family` is this module's own: 1 for RGB, whose planes are all full
    size. Vship 5.1 itself tells RGB from YUV by the matrix (0 is RGB), and
    does not use its colorFamily field."""

    pixel_format: str
    family: int
    sample: int
    subw: int
    subh: int
    full_range: bool = False  # yuvj: full range when the stream has no range tag
    # FFmpeg's planar RGB formats are ordered G, B, R; Vship expects R, G, B.
    plane_order: tuple[int, int, int] = (0, 1, 2)

    def frame_layout(self, width: int, height: int) -> tuple[int, tuple[int, int, int], _I64_3, tuple[int, int, int]]:
        """Returns a frame's layout at this size: (total bytes, each plane's
        bytes, each plane's row stride, the order Vship reads the planes in)."""
        bytes_per_sample = 1 if self.sample == _VSHIP_ENUMS[8] else 2
        if self.family == 1:
            sizes = (width * height * bytes_per_sample,) * 3
            strides = _I64_3(width * bytes_per_sample, width * bytes_per_sample, width * bytes_per_sample)
        else:
            chroma_width = (width + (1 << self.subw) - 1) >> self.subw
            chroma_height = (height + (1 << self.subh) - 1) >> self.subh
            sizes = (
                width * height * bytes_per_sample,
                chroma_width * chroma_height * bytes_per_sample,
                chroma_width * chroma_height * bytes_per_sample,
            )
            strides = _I64_3(
                width * bytes_per_sample, chroma_width * bytes_per_sample,
                chroma_width * bytes_per_sample,
            )
        return sum(sizes), sizes, strides, self.plane_order


@dataclass(slots=True)
class _LoadedVship:
    """A Vship build loaded in this process: its backend, its DLL's path, the
    library, and the DLL folder added for its dependencies."""

    backend: str
    path: Path
    library: ctypes.CDLL
    dll_directory: object


@dataclass(frozen=True, slots=True)
class VshipDevice:
    """A GPU that Vship can use, and the Vship build that runs on it."""

    backend: str  # "vulkan", "cuda" or "hip": the Vship build it runs on (VSHIP_BUILDS)
    name: str
    gpu_id: int
    version: str
    #: The library, if it was loaded in this process. It is None in the app
    #: itself, because Vship is probed and run in a process of its own (see
    #: _run_vship_pass), which loads the build again (_library).
    loaded: _LoadedVship | None
    #: Who made the GPU: NVIDIA for CUDA, AMD for HIP, and for Vulkan what
    #: the Vulkan loader says (GpuVendor.NONE for another maker). None when
    #: it could not be told.
    vendor: GpuVendor | None = None


#: Vship 5.1's API, the one this module uses. A library without it (4.x)
#: is refused at the probe rather than used through old entry points.
_API_FUNCTIONS = (
    "Vship_GetVersion", "Vship_GetDeviceCount", "Vship_GetDeviceInfo", "Vship_GPUFullCheck",
    "Vship_GetErrorMessage", "Vship_GetDetailedLastError", "Vship_PinnedMalloc2", "Vship_PinnedFree2",
    "Vship_InitHandler", "Vship_FreeHandler", "Vship_ComputeHandler", "Vship_GetDetailedLastErrorHandler",
    "Vship_ResetScore",
)


def _has_api(lib) -> bool:
    """Whether the library has Vship 5.1's API."""
    return all(hasattr(lib, name) for name in _API_FUNCTIONS)


def _configure_api(lib: ctypes.CDLL) -> None:
    """Declares the API's functions."""
    lib.Vship_GetVersion.argtypes = []
    lib.Vship_GetVersion.restype = _Version
    lib.Vship_GetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.Vship_GetDeviceCount.restype = ctypes.c_int
    lib.Vship_GetDeviceInfo.argtypes = [ctypes.POINTER(_DeviceInfo), ctypes.c_int]
    lib.Vship_GetDeviceInfo.restype = ctypes.c_int
    lib.Vship_GPUFullCheck.argtypes = [ctypes.c_int]
    lib.Vship_GPUFullCheck.restype = ctypes.c_int
    lib.Vship_GetErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.Vship_GetErrorMessage.restype = ctypes.c_int
    lib.Vship_GetDetailedLastError.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.Vship_GetDetailedLastError.restype = ctypes.c_int
    # The GPU is named in each call: Vulkan allocates per device.
    lib.Vship_PinnedMalloc2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_int]
    lib.Vship_PinnedMalloc2.restype = ctypes.c_int
    lib.Vship_PinnedFree2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.Vship_PinnedFree2.restype = ctypes.c_int
    # One handler per metric, made from an init struct and scored into a
    # score struct: both start with their Vship_StructType.
    lib.Vship_InitHandler.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
    lib.Vship_InitHandler.restype = ctypes.c_int
    lib.Vship_FreeHandler.argtypes = [_Handle]
    lib.Vship_FreeHandler.restype = ctypes.c_int
    lib.Vship_ComputeHandler.argtypes = [
        _Handle, ctypes.c_void_p, _PLANES, _PLANES, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]
    lib.Vship_ComputeHandler.restype = ctypes.c_int
    lib.Vship_GetDetailedLastErrorHandler.argtypes = [_Handle, ctypes.c_char_p, ctypes.c_int]
    lib.Vship_GetDetailedLastErrorHandler.restype = ctypes.c_int
    # Clears CVVDP's score accumulation only, not its temporal history.
    lib.Vship_ResetScore.argtypes = [_Handle]
    lib.Vship_ResetScore.restype = ctypes.c_int


def _handler_error(lib: ctypes.CDLL, handle: _Handle, code: int) -> str:
    """Returns why a call on `handle` failed: the handler's own last error,
    or else the text for `code`. The handler's own error is used because
    other handlers, on other threads, may have failed since."""
    detail = ctypes.create_string_buffer(1024)
    with contextlib.suppress(Exception):
        lib.Vship_GetDetailedLastErrorHandler(handle, detail, len(detail))
        if detail.value:
            return detail.value.decode("utf-8", errors="replace").strip()
    return _message(lib, code)


def _message(lib: ctypes.CDLL, code: int | None = None) -> str:
    """Returns Vship's text for the error `code`, or its last detailed error
    when `code` is None."""
    buffer = ctypes.create_string_buffer(1024)
    try:
        if code is None:
            lib.Vship_GetDetailedLastError(buffer, len(buffer))
        else:
            lib.Vship_GetErrorMessage(code, buffer, len(buffer))
        text = buffer.value.decode("utf-8", errors="replace").strip()
        return text or f"Vship error {code}"
    except Exception:
        return f"Vship error {code}"


#: Vship's builds, bundled in tools/vship/<folder>: CUDA runs on NVIDIA,
#: HIP on AMD, and Vulkan on any GPU with a Vulkan driver (NVIDIA, AMD and
#: Intel).
VSHIP_BUILDS: dict[str, str] = {"cuda": "nvidia", "hip": "amd", "vulkan": "vulkan"}
#: Settings > GPU metrics > backend: "auto", or the build to try first. Auto
#: tries CUDA, then HIP, then Vulkan: the fastest build whose scores agree
#: with the reference on each GPU (CUDA is faster than Vulkan on NVIDIA for
#: SSIMULACRA2 and CVVDP), and Vulkan for a GPU neither of the others can use.
#: If the chosen build cannot run, the others are tried after it.
VSHIP_BACKENDS = ("auto", "vulkan", "cuda", "hip")
DEFAULT_VSHIP_BACKEND = "auto"
_AUTO_ORDER = ("cuda", "hip", "vulkan")
_BACKEND_LABELS = {"vulkan": "Vulkan", "cuda": "CUDA", "hip": "HIP"}
#: The (build, GPU maker, metric) combinations that a Vship build scores
#: wrongly. Such a metric is calculated on the CPU instead, as on a PC
#: without a usable GPU. A GPU whose maker could not be told counts as every
#: maker. Add an entry only for a bundled build measured to disagree with
#: CUDA and libjxl on the same frames.
#:
#: It is empty since the bundled Vulkan build is Vship 5.1.2. Older Vulkan
#: builds scored SSIMULACRA2 far too high on NVIDIA GPUs (Vship issue 18,
#: fixed in commit 97d0dc5), because NVIDIA's driver miscompiles a small
#: two-dimensional array in the blur. With 5.1.2, an RTX 5090 scores 45.5019
#: where CUDA scores 45.5020. AMD's Vulkan scores have not been measured.
SCORED_WRONGLY: frozenset[tuple[str, GpuVendor, str]] = frozenset()
#: The largest frame, in pixels, that a (build, GPU maker, metric)
#: combination is given, with the GPUs and sizes that limit covers, for the
#: reason shown with the video. Above it, the metric is not calculated.
#: Vship 5.1.2's Vulkan CVVDP hangs Intel's GPU above 1080p: on a Core
#: Ultra 9 285K (driver 32.0.101.8860), 4K HEVC stopped after 3 to 6
#: frames, with one handler or two. Windows reset the GPU (LiveKernelEvent
#: 141, and the screen goes black for a moment), and the pass failed with
#: "Failed to Submit commandBuffer to queue". At 1920x1080, all 240 frames
#: were scored, 13 a second. Sizes in between are untested, so they are
#: refused too. This has not been reported to Vship yet.
GPU_PIXEL_LIMITS: dict[tuple[str, GpuVendor, str], tuple[int, str]] = {
    ("vulkan", GpuVendor.INTEL, "cvvdp"): (1920 * 1080, "Intel GPUs above 1920x1080"),
}
_backend = DEFAULT_VSHIP_BACKEND

# The probe's result and when it was made. One probe serves the whole
# session: callers that arrive while it runs wait for it rather than
# starting their own.
_PROBE_LOCK = threading.Lock()
_probed: tuple[tuple[VshipDevice | None, str], float] | None = None
#: A failed probe older than this is made again before the next GPU run.
FAILED_PROBE_RETRY_SECONDS = 60.0


def set_vship_backend(backend: str) -> None:
    """Sets the Vship build to try first, or "auto" (Settings > GPU
    metrics). An unknown name means "auto". A probe made for another choice
    is forgotten, so that the next probe (start_vship_probe, or the next GPU
    run) uses this one."""
    global _backend, _probed
    backend = backend if backend in VSHIP_BACKENDS else DEFAULT_VSHIP_BACKEND
    with _PROBE_LOCK:
        if backend != _backend:
            _backend = backend
            _probed = None


def vship_backend() -> str:
    """The Vship build to try first, or "auto"."""
    return _backend


def backend_label(backend: str) -> str:
    """The name shown for a Vship build: "Vulkan", "CUDA" or "HIP"."""
    return _BACKEND_LABELS.get(backend, backend)


def scores_correctly(device: VshipDevice, key: str) -> bool:
    """Whether `device`'s Vship build scores `key` correctly on its GPU (see
    SCORED_WRONGLY). A GPU of unknown make is taken to be affected."""
    return not any(device.backend == backend and key == metric and device.vendor in (vendor, None)
                   for backend, vendor, metric in SCORED_WRONGLY)


def size_refusal(device: VshipDevice, key: str, size: tuple[int, int],
                 computed: tuple[int, int] | None = None) -> str | None:
    """Returns why `device` does not calculate `key` on frames of `size` (see
    GPU_PIXEL_LIMITS), or None if it does. `computed` is the size the metric
    is calculated at, where that differs (CVVDP scaled to the display)."""
    limit = GPU_PIXEL_LIMITS.get((device.backend, device.vendor, key))
    at = computed or size
    if limit is None or at[0] * at[1] <= limit[0]:
        return None
    detail = (f"this comparison is {size[0]}x{size[1]}" if at == size else
              f"this comparison is {size[0]}x{size[1]}, scaled to the display: {at[0]}x{at[1]}")
    return f"on {limit[1]} it is not calculated, as Vship's Vulkan build hangs the GPU ({detail})"


def cvvdp_computed_size(size: tuple[int, int], settings: CvvdpSettings) -> tuple[int, int]:
    """Returns the size Vship calculates CVVDP at for frames of `size`: their
    own size, or with resize_to_display, the larger of their own size and
    their size fitted to the display (keeping the aspect ratio). The larger
    is taken because it is not known whether Vship shrinks a frame larger
    than the display, and the size decides whether Intel's GPU hangs (see
    GPU_PIXEL_LIMITS)."""
    if not settings.resize_to_display:
        return size
    scale = min(settings.display.width / size[0], settings.display.height / size[1])
    fitted = (round(size[0] * scale), round(size[1] * scale))
    return fitted if fitted[0] * fitted[1] > size[0] * size[1] else size


def gpu_can_score(key: str) -> bool:
    """Whether SSIMULACRA2 or Butteraugli set to the GPU is calculated on the
    GPU here: when Vship can use a GPU, and its build scores the metric
    correctly. The GPU is probed once."""
    device, _reason = detect_vship_device()
    return device is not None and scores_correctly(device, key)


def detect_vship_device() -> tuple[VshipDevice | None, str]:
    """Returns a fully checked GPU for Vship, or a reason the user can read
    for why there is none.

    The probe is made once and its result kept; see start_vship_probe and
    forget_failed_vship_probe."""
    global _probed
    with _PROBE_LOCK:
        if _probed is None:
            stopped = []
            _probed = (_probe_isolated(stopped), time.monotonic())
            device, reason = _probed[0]
            if device is not None:
                _log.info("Vship %s (%s) GPU: %s", device.version, backend_label(device.backend), device.name)
            else:
                # A probe stopped because the app is closing is not worth a
                # warning; the other probes log that at INFO too.
                _log.log(logging.INFO if stopped else logging.WARNING, "Vship GPU unavailable: %s", reason)
        return _probed[0]


def start_vship_probe() -> None:
    """Probes the GPU in the background at startup. The first probe loads
    the Vship library and starts the GPU driver, which would freeze the
    window for that long if it ran on the window's thread when the first
    video is added."""
    threading.Thread(target=detect_vship_device, name="vship-probe", daemon=True).start()


def forget_failed_vship_probe() -> None:
    """Forgets a probe that failed a while ago, so that the next run probes
    again. Call it before a GPU run, off the window's thread. The failure
    may have been temporary (a driver being updated or restarted, a GPU
    briefly unavailable); without this, every GPU metric would run on the
    CPU until the app restarted."""
    global _probed
    with _PROBE_LOCK:
        if (_probed is not None and _probed[0][0] is None
                and time.monotonic() - _probed[1] >= FAILED_PROBE_RETRY_SECONDS):
            _probed = None


def _probe_order(choice: str | None = None) -> tuple[str, ...]:
    """The builds the probe tries, in turn, for "auto" or a chosen build."""
    choice = _backend if choice is None else choice
    return _AUTO_ORDER if choice == "auto" else (choice, *(build for build in _AUTO_ORDER if build != choice))


class _VkApplicationInfo(ctypes.Structure):
    """Vulkan's VkApplicationInfo, to start the Vulkan instance of the checks
    below."""

    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class _VkInstanceCreateInfo(ctypes.Structure):
    """Vulkan's VkInstanceCreateInfo."""

    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


def _vulkan_unavailable() -> str | None:
    """Returns why no GPU is usable through Vulkan here, or None if one is.

    The Vulkan loader itself is asked, before Vship's Vulkan build is
    loaded. Where the loader is installed but no driver answers (a virtual
    machine, GitHub's test runner), Vship's Vulkan builds before 5.1.2
    failed their DLL initialisation (WinError 1114), and the process then
    crashed with an access violation as it exited. The bundled build starts
    Vulkan on first use, and loads there too (tests/test_perceptual_vship.py),
    but the loader is still asked first, because it says why without
    loading Vship."""
    try:
        vulkan = ctypes.WinDLL("vulkan-1.dll")
    except OSError:
        return "no Vulkan driver is installed (vulkan-1.dll was not found)"
    app = _VkApplicationInfo(0, None, b"VideoMetricsLab", 1, None, 0, 1 << 22)  # Vulkan 1.0
    info = _VkInstanceCreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    vulkan.vkCreateInstance.restype = ctypes.c_int32
    result = vulkan.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if result != 0:
        return f"no Vulkan driver answered (VkResult {result})"
    try:
        count = ctypes.c_uint32(0)
        vulkan.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
        result = vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        if result != 0 or count.value == 0:
            return "no GPU with a Vulkan driver was found"
        return None
    finally:
        vulkan.vkDestroyInstance(instance, None)


def _vulkan_vendor(name: str) -> GpuVendor | None:
    """Returns who made the Vulkan GPU called `name` (Vship names a GPU by its
    VkPhysicalDeviceProperties.deviceName). The Vulkan loader is asked,
    because names do not always say (NVIDIA's "Quadro" and "Tesla" cards).
    Returns GpuVendor.NONE for another maker, or None if no GPU has the
    name."""
    try:
        vulkan = ctypes.WinDLL("vulkan-1.dll")
    except OSError:
        return None
    app = _VkApplicationInfo(0, None, b"VideoMetricsLab", 1, None, 0, 1 << 22)  # Vulkan 1.0
    info = _VkInstanceCreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    vulkan.vkCreateInstance.restype = ctypes.c_int32
    if vulkan.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance)) != 0:
        return None
    try:
        count = ctypes.c_uint32(0)
        vulkan.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
        if vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0 or not count.value:
            return None
        devices = (ctypes.c_void_p * count.value)()
        vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)
        for device in devices[:count.value]:
            # VkPhysicalDeviceProperties: apiVersion, driverVersion, vendorID,
            # deviceID, deviceType (4 bytes each), then deviceName[256].
            properties = (ctypes.c_ubyte * 1024)()
            vulkan.vkGetPhysicalDeviceProperties(ctypes.c_void_p(device), properties)
            raw = bytes(properties)
            if raw[20:276].split(b"\0", 1)[0].decode("utf-8", errors="replace") == name:
                return PCI_VENDORS.get(int.from_bytes(raw[8:12], "little"), GpuVendor.NONE)
        return None
    finally:
        vulkan.vkDestroyInstance(instance, None)


#: The GPU maker each build runs on. For Vulkan, the loader is asked
#: (_vulkan_vendor).
_BUILD_VENDORS = {"cuda": GpuVendor.NVIDIA, "hip": GpuVendor.AMD}


def _probe_vship_device() -> tuple[VshipDevice | None, str]:
    """Tries the Vship builds in turn (_probe_order), and returns the first
    usable GPU found, a discrete one first, or the reasons none was found."""
    if os.name != "nt":
        return None, "Vship GPU acceleration is only bundled for Windows."
    tools = Path(__file__).resolve().parents[1] / "tools" / "vship"
    failures: list[str] = []
    for backend in _probe_order():
        label = backend_label(backend)
        path = tools / VSHIP_BUILDS[backend] / "libvship.dll"
        if not path.is_file():
            failures.append(f"Vship's {label} library is missing")
            continue
        if backend == "vulkan":
            unavailable = _vulkan_unavailable()
            if unavailable is not None:
                failures.append(f"{label}: {unavailable}")
                continue
        try:
            dll_directory = os.add_dll_directory(str(path.parent))
            lib = ctypes.CDLL(str(path))
            if not _has_api(lib):
                failures.append(f"Vship's {label} library is older than 5.1")
                continue
            _configure_api(lib)
            loaded = _LoadedVship(backend, path, lib, dll_directory)
            count = ctypes.c_int()
            error = lib.Vship_GetDeviceCount(ctypes.byref(count))
            if error != 0:
                failures.append(f"{label}: {_message(lib, error)}")
                continue
            usable: list[tuple[bool, int, str]] = []
            for gpu_id in range(count.value):
                error = lib.Vship_GPUFullCheck(gpu_id)
                if error != 0:
                    failures.append(f"{label} GPU {gpu_id}: {_message(lib, error)}")
                    continue
                info = _DeviceInfo()
                error = lib.Vship_GetDeviceInfo(ctypes.byref(info), gpu_id)
                if error != 0:
                    failures.append(f"{label} GPU {gpu_id}: {_message(lib, error)}")
                    continue
                name = bytes(info.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
                usable.append((bool(info.integrated), gpu_id, name or f"{label} GPU {gpu_id}"))
            if usable:
                # Vulkan lists every GPU, integrated ones included, in the
                # driver's order; a discrete GPU is preferred here.
                _integrated, gpu_id, name = min(usable)
                version = lib.Vship_GetVersion()
                version_text = f"{version.major}.{version.minor}.{version.minorMinor}"
                vendor = _BUILD_VENDORS.get(backend) or _vulkan_vendor(name)
                return VshipDevice(backend, name, gpu_id, version_text, loaded, vendor), ""
            if count.value == 0:
                failures.append(f"{label}: no GPU was found")
        except (OSError, AttributeError, TypeError) as error:
            failures.append(f"Vship's {label} library could not be loaded: {error}")
    return None, "; ".join(failures) or "No GPU that Vship can use was found."


def _probe_isolated(stopped: list | None = None) -> tuple[VshipDevice | None, str]:
    """Runs _probe_vship_device in a process of its own. Loading Vship
    starts the GPU driver, and a crash there would take the app down (Vship
    5.1.1's Vulkan build crashed as it closed, on a PC whose only GPU is
    Intel's). A crash now ends only that process, and the GPU metrics are
    calculated on the CPU. If given, `stopped` gets True appended when the
    app ended the probe (because it is closing) rather than the probe
    failing."""
    try:
        return run_isolated(_probe_in_own_process, _backend, what="Vship's GPU probe")
    except IsolatedCrashError as error:
        if error.stopped and stopped is not None:
            stopped.append(True)
        return None, str(error)


def _probe_in_own_process(choice: str) -> tuple[VshipDevice | None, str]:
    """The probe in its own process (see _probe_isolated). The device is
    returned without the library, which belongs to the process that loaded
    it."""
    global _backend
    _backend = choice  # this process starts with the default
    device, reason = _probe_vship_device()
    return (replace(device, loaded=None) if device is not None else None), reason


#: The builds a pass's process has loaded (see _library).
_loaded_builds: dict[str, _LoadedVship] = {}


def _library(device: VshipDevice) -> ctypes.CDLL:
    """Returns the Vship library `device` runs on: the probe's own, or, in
    the process a pass runs in (which gets the device without its library),
    the build loaded there once."""
    if device.loaded is not None:
        return device.loaded.library
    loaded = _loaded_builds.get(device.backend)
    if loaded is None:
        path = Path(__file__).resolve().parents[1] / "tools" / "vship" / VSHIP_BUILDS[device.backend] / "libvship.dll"
        dll_directory = os.add_dll_directory(str(path.parent))
        lib = ctypes.CDLL(str(path))
        _configure_api(lib)
        loaded = _loaded_builds[device.backend] = _LoadedVship(device.backend, path, lib, dll_directory)
    return loaded.library


def _reads_410(version: str | None) -> bool:
    """Whether Vship `version` ("5.1.2") reads 4:1:0 correctly (see
    _READS_410_SINCE). An unknown version is taken not to."""
    try:
        return tuple(int(part) for part in (version or "").split(".")[:3]) >= _READS_410_SINCE
    except ValueError:
        return False


def _image_format(info: VideoInfo, version: str | None = None) -> _ImageFormat:
    """Returns the planar layout FFmpeg sends a decoded video in, and how
    Vship `version` reads it. FFmpeg converts to it exactly: a semi-planar
    or big-endian layout to the planar little-endian one, with alpha dropped
    and neutral chroma added to a monochrome video. 4:1:0 is sent as it is
    to a Vship that reads it, and upsampled to 4:4:4 for an older one (see
    _READS_410_SINCE)."""
    name = (info.pix_fmt or "").strip().casefold()
    if name in {"nv12", "nv21"}:
        return _format_yuv("420", 8)
    semi_planar = {"p010le": ("420", 10), "p010be": ("420", 10), "p012le": ("420", 12),
                   "p016le": ("420", 16), "p016be": ("420", 16),
                   "nv16": ("422", 8), "p210le": ("422", 10), "p212le": ("422", 12), "p216le": ("422", 16),
                   "nv24": ("444", 8), "p410le": ("444", 10), "p412le": ("444", 12), "p416le": ("444", 16)}
    if name in semi_planar:
        return _format_yuv(*semi_planar[name])

    # yuv, yuvj (full range when untagged) and yuva (alpha dropped).
    yuv = re.fullmatch(r"yuv(j|a)?(410|411|420|422|440|444)p(?:(9|10|12|14|16)(?:le|be)?)?", name)
    if yuv:
        sampling = yuv.group(2)
        if sampling == "410" and not _reads_410(version):
            sampling = "444"
        image = _format_yuv(sampling, int(yuv.group(3) or 8))
        return replace(image, full_range=yuv.group(1) == "j")
    # Monochrome (AV1 can be): the luma as it is, with neutral chroma.
    gray = re.fullmatch(r"gray(?:(9|10|12|14|16)(?:le|be)?)?", name)
    if gray:
        return _format_yuv("420", int(gray.group(1) or 8))

    rgb = re.fullmatch(r"gbra?p(?:(9|10|12|14|16)(?:le|be)?)?", name)
    if rgb:
        depth = int(rgb.group(1) or 8)
    elif name in {"rgb24", "bgr24", "rgba", "bgra", "argb", "abgr", "rgb0", "bgr0", "0rgb", "0bgr"}:
        depth = 8
    elif name in {"rgb48le", "rgb48be", "bgr48le", "bgr48be", "rgba64le", "rgba64be", "bgra64le", "bgra64be"}:
        depth = 16
    else:
        raise VshipUnavailableError(
            f"The GPU metrics cannot read the decoded pixel format {info.pix_fmt or '(unknown)'}.")
    fmt = "gbrp" + (f"{depth}le" if depth > 8 else "")
    # FFmpeg's planar RGB is ordered G, B, R; Vship reads R, G, B.
    return _ImageFormat(fmt, 1, _VSHIP_ENUMS[depth], 0, 0, True, (2, 0, 1))


def _passthrough_format(info: VideoInfo, hwaccel: str | None) -> tuple[_ImageFormat, str] | None:
    """Returns the planar layout Vship is given and the layout FFmpeg sends,
    when a hardware-decoded 4:2:0 frame can cross the pipe exactly as it is
    downloaded; otherwise None.

    NVDEC (and D3D11VA and QSV) give back NV12 or P010: a luma plane, then U
    and V interleaved in one plane. Vship only takes separate planes, and
    having FFmpeg rearrange the whole frame costs 10-12 ms of CPU per 4K
    frame, about half of what feeding Vship costs. Sent as it is, the luma
    plane is read straight into pinned memory and only the chroma is split,
    which numpy does in under 1 ms (_FrameStream._fill).

    P010 keeps each 10-bit sample in the top bits of 16. Declared to Vship
    as 16-bit and limited range, that is exactly the same picture: Vship
    brings limited-range samples to 8-bit scale by dividing by 2^(depth-8),
    so (v << 6) / 256 and v / 4 are the same float (FullRange in Vship
    5.1.1's gpuColorToLinear/rangeToFull.hpp; the 16-bit read masks nothing
    off). Full range divides by 2^depth - 1 instead, where the two differ,
    so full-range 10-bit video keeps FFmpeg's conversion. 8-bit NV12 is
    exact in either range.
    """
    if not hwaccel:
        return None
    image = _image_format(info)
    native = hw_native_format(info.pix_fmt)
    if image.pixel_format == "yuv420p" and native == "nv12":
        return image, "nv12"
    range_name = (info.color_range or "").casefold()
    if (image.pixel_format == "yuv420p10le" and native == "p010le" and not image.full_range
            and range_name not in {"pc", "jpeg", "full"}):
        return _ImageFormat("yuv420p16le", 0, _VSHIP_ENUMS[16], 1, 1), "p010le"
    return None


def _format_yuv(sampling: str, depth: int) -> _ImageFormat:
    """The planar YUV format for a subsampling ("420") and bit depth."""
    subw, subh = _SUBSAMPLING[sampling]
    suffix = "" if depth == 8 else f"{depth}le"
    return _ImageFormat(f"yuv{sampling}p{suffix}", 0, _VSHIP_ENUMS[depth], subw, subh)


def _vship_colorspace(info: VideoInfo, image: _ImageFormat, width: int, height: int) -> _Colorspace:
    """Returns the frame's colour space for Vship, from the stream's tags,
    read as colour.video_colour reads them for both the GPU and the CPU."""
    try:
        colour = video_colour(info, rgb=image.family == 1, full_range_untagged=image.full_range)
    except UnsupportedColourError as error:
        raise VshipUnavailableError({
            "matrix": f"Vship does not support the {error.value} color matrix.",
            "transfer": f"Vship does not support the {error.value} transfer function.",
            "primaries": f"Vship does not support the {error.value} color primaries.",
            "range": f"Vship does not support the {error.value} range tag.",
        }[error.kind]) from error
    location = (info.chroma_location or "left").casefold().replace("-", "")
    # Vship_ChromaLocation_t has no bottom or bottom-left siting.
    locations = {"left": 0, "center": 1, "topleft": 2, "top": 3,
                 "unspecified": 0, "unknown": 0}
    if location not in locations:
        raise VshipUnavailableError(f"Vship does not support {info.chroma_location} chroma siting.")

    return _Colorspace(
        width, height, -1, -1, image.sample, int(colour.full_range),
        _Subsampling(image.subw, image.subh), locations[location], image.family,
        colour.matrix, colour.transfer, colour.primaries, _Crop(0, 0, 0, 0),
    )


def _scaled_sizes(
    source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The sizes the source's and the test video's frames are scaled to:
    both the same, by the comparison's scaling direction."""
    source_size = content_size(source, source_crop)
    distorted_size = content_size(distorted, distorted_crop)
    if source_size == distorted_size:
        return source_size, distorted_size
    if recipe.scale_direction is ScaleDirection.DISTORTED_TO_SOURCE:
        return source_size, source_size
    return distorted_size, distorted_size


def _filter_chain(
    info: VideoInfo, crop: CropBox | None, target_size: tuple[int, int],
    pixel_format: str, step: int, algorithm: str, hwaccel: str | None = None,
    numbered_at: Fraction | None = None,
) -> str:
    """Returns the filters a video's frames go through on their way to
    Vship: every step-th frame is kept, then brought from the GPU, cropped,
    scaled and converted. `numbered_at` is the test video's frame rate, at
    which the frames are first numbered, as every metric's are (frame_sync).
    It is None when pairing by timestamp, where each video's times start at
    0 instead."""
    operations: list[str] = [] if numbered_at is None else numbering_filters(numbered_at)
    if step > 1:
        # select comes before the rest, so that the frames it drops are not
        # downloaded, cropped or scaled. Nothing after it drops a frame, so
        # n counts the same frames wherever it stands.
        operations.append(f"select=not(mod(n\\,{step}))")
    if hwaccel:
        # A hardware-decoded surface is brought to system memory in its
        # native layout first; crop, scale and the final format conversion
        # then run exactly as they do for a software-decoded input, so the
        # pictures Vship scores do not depend on which decoder produced them.
        operations.append(f"hwdownload,format={hw_native_format(info.pix_fmt)}")
    if crop is not None and not crop.is_noop(info.width, info.height):
        operations.append(crop.as_filter())
    current_size = content_size(info, crop)
    if current_size != target_size:
        operations.append(f"scale={target_size[0]}:{target_size[1]}:flags={algorithm}")
    if numbered_at is None:
        operations.append("setpts=PTS-STARTPTS")
    operations.append(f"format={pixel_format}")
    return ",".join(operations)


#: Vship handlers working at once per metric. One handler leaves the GPU
#: idle between its own kernels, and two keep it busy: at 4K 10-bit on an
#: RTX 5090 (Vship's CUDA build), SSIMULACRA2 went from 217 to 277 pairs a
#: second, Butteraugli from 82 to 95, and both together from 61 to 72.
#: (FFVship runs 8 GPU streams by default, and its documentation calls 3
#: usually enough; each holds its own GPU memory.)
_LANES_PER_METRIC = 2
#: Frames in flight per video: one per lane being scored, one waiting and
#: one being filled, so that decoding, transfer and GPU work overlap instead
#: of taking turns. A 4K 10-bit frame is 25 MB of pinned memory, so this is
#: 250 MB for a 4K pair.
_RING_SLOTS = _LANES_PER_METRIC + 3
#: The size of the pipe between FFmpeg and this process. Windows' default
#: is a few kilobytes, which holds a raw 4K stream near 55 fps however fast
#: the decoder is. With 64 MB, one 4K 10-bit HEVC stream went from 55 to
#: 107 fps alone, and from 46 to 78 fps with the two videos in parallel.
_PIPE_BYTES = proc_util.FRAME_PIPE_BYTES
#: How many of a decoding FFmpeg's last error lines are kept for a failure's
#: message, which shows their last 2,000 bytes.
_STDERR_LINES_KEPT = 100
_EOF = -1
#: What FFmpeg writes about each frame it sends to Vship (-stats_enc_pre)
#: when pairing by timestamp (frame_sync.by_timestamp): the picture's
#: timestamp as it leaves the filter chain, in the chain's time base
#: (-enc_time_base filter, the stream's). That is the timestamp libvmaf's
#: frame sync is given at the end of the same chain, so the frames are
#: paired by it as libvmaf pairs them (frame_sync_by_time).
#:
#: Don't use the decoder's timestamp ({ptsi} {tbi}) instead: a file that
#: stores no presentation times (H.264 with B-frames in AVI) has none,
#: because FFmpeg works its frames' times out after decoding.
_TIMESTAMP_FORMAT = "{pts} {tb}"
#: How long a frame's timestamp may be missing once the frame has arrived.
#: FFmpeg writes and flushes it before the frame, so it is there at once.
_TIMESTAMP_WAIT_SECONDS = 10.0
_NO_PTS = -(1 << 63)  # AV_NOPTS_VALUE


def _with_timestamps(command: list[str], path: Path) -> list[str]:
    """Returns `command`, whose last argument is its output, made to also
    write each piped frame's timestamp to `path` (_TIMESTAMP_FORMAT), one
    line per frame."""
    return [*command[:-1], "-enc_time_base", "filter", "-stats_enc_pre", str(path),
            "-stats_enc_pre_fmt", _TIMESTAMP_FORMAT, command[-1]]


class _Timestamps:
    """Reads the timestamp lines one FFmpeg writes as it sends frames
    (_with_timestamps), one line per frame read. A frame's line is written
    before the frame, so it is in the file by the time the frame has been
    read; the only wait is for a line caught part-way through."""

    def __init__(self, path: Path, process: subprocess.Popen, label: str,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._path, self._process, self._label, self._clock = path, process, label, clock
        self._file = None
        self._pending = b""
        self.time_base: Fraction | None = None

    def next(self) -> int:
        """The next frame's timestamp, in self.time_base."""
        deadline = None
        while True:
            line, newline, rest = self._pending.partition(b"\n")
            if newline:
                self._pending = rest
                return self._parse(line)
            if self._file is None:
                with contextlib.suppress(OSError):
                    self._file = open(self._path, "rb")  # noqa: SIM115 -- closed in close()
            data = self._file.read() if self._file is not None else b""
            if data:
                self._pending += data
                continue
            ended = self._process.poll() is not None
            now = self._clock()
            deadline = now + _TIMESTAMP_WAIT_SECONDS if deadline is None else deadline
            if ended or now >= deadline:
                raise self._missing("FFmpeg wrote none")
            time.sleep(0.001)

    def _missing(self, what: str) -> VshipUnavailableError:
        """The pass's failure for a frame without a usable timestamp. The
        window gets one fixed message (it has a translation), and the log
        gets `what`."""
        _log.error("Vship: no timestamp for a %s frame: %s", self._label, what)
        return VshipUnavailableError(f"FFmpeg did not give the timestamp of a {self._label} frame.")

    def _parse(self, line: bytes) -> int:
        """Returns the timestamp on one line, checking its time base."""
        try:
            pts_text, base_text = line.decode("ascii").split()
            pts, time_base = int(pts_text), Fraction(base_text)
        except (UnicodeDecodeError, ValueError, ZeroDivisionError) as error:
            raise self._missing(f"FFmpeg wrote {line[:80]!r}") from error
        if pts == _NO_PTS or time_base <= 0:
            raise self._missing(f"FFmpeg wrote {line[:80]!r}: no timestamp")
        if self.time_base is None:
            self.time_base = time_base
        elif time_base != self.time_base:
            raise self._missing(f"its time base changed from {self.time_base} to {time_base}")
        return pts

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class _PassRate:
    """A pass's frame rate for its progress reports, timed from its first
    frame pair rather than from the pass's start.

    The start (FFmpeg opening both videos, the first 4K frames decoded,
    Vship's handlers set up) takes a second or more that says nothing about
    the pass's speed. Counted in, it would show "1.7 fps, 0:05:39 left" at
    the start of a pass over an 8-second clip. So there is no rate until
    there is some time to measure over.

    The rate is in the video's frames, as the progress is: with frame
    subsampling, each compared pair covers `step` frames. Pairs per second
    against frames left would make a sampled pass's time left `step` times
    too long.
    """

    WARMUP_SECONDS = 0.5

    def __init__(self, step: int = 1, clock: Callable[[], float] = time.perf_counter) -> None:
        self._step = step
        self._clock = clock
        self._first_at: float | None = None
        self._first_pairs = 0

    def frames_per_second(self, pairs: int) -> float:
        """The rate after `pairs` frame pairs have been handed to Vship."""
        now = self._clock()
        if self._first_at is None:
            self._first_at, self._first_pairs = now, pairs
            return 0.0
        span = now - self._first_at
        return (pairs - self._first_pairs) * self._step / span if span >= self.WARMUP_SECONDS else 0.0


#: The planar layouts FFmpeg's YUV4MPEG muxer writes (with -strict -1 above
#: 8 bits). 4:1:0, 4:4:0, planar RGB, and the semi-planar layouts a GPU
#: decoder gives back (NV12, P010) are sent as raw video.
_Y4M_FORMAT = re.compile(r"yuv(411|420|422|444)p(?:(9|10|12|14|16)le)?")
_Y4M_FRAME = b"FRAME\n"


def _piped_as(pixel_format: str) -> list[str]:
    """Returns FFmpeg's output arguments for sending frames of
    `pixel_format` down the pipe.

    YUV4MPEG is used where the muxer takes the layout. It holds the same
    planes as raw video, each frame after a "FRAME" line, and FFmpeg writes
    it straight from its frames, while the rawvideo encoder first copies
    every frame into a packet. On a 4K 10-bit HEVC video, that is 23 ms of
    FFmpeg's CPU per frame against 35."""
    if _Y4M_FORMAT.fullmatch(pixel_format):
        return ["-pix_fmt", pixel_format, "-f", "yuv4mpegpipe", "-strict", "-1", "pipe:1"]
    return ["-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"]


def _is_y4m(command: list[str]) -> bool:
    """Whether an FFmpeg command sends YUV4MPEG."""
    return "yuv4mpegpipe" in command


def _gather(reader, count: int, process: subprocess.Popen, stopping: threading.Event) -> None:
    """Waits until `count` bytes of a frame are in the pipe (or as much as it
    can hold), FFmpeg has ended, or the stream is stopping. Read as it
    comes, a 4K frame arrives in some 300 pieces, each waking the reading
    thread: 6 ms of CPU per frame, against 4 ms when taken in one read."""
    if os.name != "nt":
        return
    import msvcrt
    from ctypes import wintypes

    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(reader.fileno()))
    except (OSError, ValueError):
        return
    # Never wait for more than half of what the pipe holds: with its pipe
    # full, FFmpeg would wait for this side to read while this side waited
    # for the rest of the frame. (A pipe of Windows' default few kilobytes,
    # as the tests' stand-ins have, is read as it comes.)
    held = wintypes.DWORD()
    if not ctypes.windll.kernel32.GetNamedPipeInfo(handle, None, None, ctypes.byref(held), None):
        return
    count = min(count, held.value // 2)
    available = wintypes.DWORD()
    peek = ctypes.windll.kernel32.PeekNamedPipe
    while not stopping.is_set():
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            return  # closed and empty: the read says so
        if available.value >= count or process.poll() is not None:
            return
        time.sleep(0.001)


def _spawn_raw_ffmpeg(command: list[str]) -> tuple[subprocess.Popen, object]:
    """Starts FFmpeg writing frames to a large pipe; returns (process, reader)."""
    return proc_util.popen_piped(command, _PIPE_BYTES)


def _read_exact(reader, view: memoryview) -> int:
    """Fills `view` from the pipe with whole-buffer reads, and returns the
    number of bytes read.

    It reads straight into pinned memory, with no intermediate bytes
    objects and no second copy. Reading 1 MB chunks into new bytes objects
    and copying each one would hold a 4K stream to 6 fps.
    """
    offset, total = 0, len(view)
    while offset < total:
        count = reader.readinto(view[offset:])
        if not count:
            break
        offset += count
    return offset


class _LaneFailedError(Exception):
    """A scoring lane stopped; its error is in the pass's failure list."""


class _FramesApartError(Exception):
    """Raised only when pairing by timestamp. With frame subsampling, the
    source gives only every step-th frame, as the test video does, and while
    the two videos' frames line up, those are the frames the test video's
    are paired with. A pair further apart than _APART_FRAMES shows that they
    do not, and the pass is then made again with every source frame
    (_score_vship_pass)."""


#: How far apart, in frames, a subsampled pair's two frames may be while the
#: source frame is still the nearest of all its frames: with evenly spaced
#: frames, any other is at least three quarters of a frame away.
_APART_FRAMES = 0.25


class _FrameStream:
    """One video decoded by FFmpeg, feeding a ring of pinned frame buffers.

    A background thread reads frames into free slots and hands them over in
    order, and the scoring thread gives each slot back once Vship is done
    with it. Both the pipe read and the Vship call release the GIL, so the
    two videos and the GPU all make progress at once. If hardware decoding
    fails before giving a frame, the same pictures are decoded in software
    instead, and `on_software` is called first. With `timestamps`, each
    frame's timestamp is read too (pts, time_base), for pairing by
    timestamp.
    """

    def __init__(
        self, lib: ctypes.CDLL, frame_bytes: int, commands: list[list[str]],
        process_handle: ProcessHandle | None, label: str,
        interleaved_chroma: tuple[int, int, type[np.integer]] | None = None,
        on_software: Callable[[], None] | None = None, gpu_id: int = 0, timestamps: bool = False,
    ) -> None:
        self.buffers = _pinned_ring(lib, frame_bytes, gpu_id)
        self.views = [memoryview(buffer.array).cast("B") for buffer in self.buffers]
        # Set when FFmpeg sends NV12 or P010: the luma goes straight into
        # the slot, and the U/V pairs into one staging buffer, from which
        # they are split into the slot's U and V planes. None when the frame
        # arrives already planar.
        self._split = None
        if interleaved_chroma is not None:
            luma, plane, dtype = interleaved_chroma
            staging = np.empty(2 * plane, dtype=np.uint8)
            count = plane // np.dtype(dtype).itemsize
            self._split = (
                luma, memoryview(staging),
                staging.view(dtype).reshape(-1, 2),
                [(np.frombuffer(view, dtype, count, luma), np.frombuffer(view, dtype, count, luma + plane))
                 for view in self.views],
            )
        self._commands = commands
        self._process_handle = process_handle
        self._label = label
        self._on_software = on_software
        self._timestamps = timestamps
        #: When pairing by timestamp, each slot's frame's timestamp in
        #: time_base (known from the first frame), set before the slot is
        #: handed over.
        self.pts = [0] * _RING_SLOTS
        self.time_base: Fraction | None = None
        self._free: queue.Queue[int] = queue.Queue()
        self._filled: queue.Queue[int | BaseException] = queue.Queue()
        for slot in range(_RING_SLOTS):
            self._free.put(slot)
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._reader = None
        self._thread = threading.Thread(target=self._run, name=f"vship-{label}", daemon=True)

    def start(self) -> None:
        """Starts decoding."""
        self._thread.start()

    def _run(self) -> None:
        """The reading thread: runs each FFmpeg command in turn (hardware
        decoding first) until one succeeds."""
        try:
            for attempt, command in enumerate(self._commands):
                frames, code, stderr = self._decode(command)
                if self._stopping.is_set():
                    return
                if code == 0:
                    self._filled.put(_EOF)
                    return
                if frames == 0 and attempt + 1 < len(self._commands):
                    # Hardware decode refused this stream: decode it in software.
                    if self._on_software is not None:
                        self._on_software()
                    continue
                raise VshipUnavailableError(
                    f"FFmpeg failed while decoding the {self._label} for Vship"
                    + (f": {stderr}" if stderr else ".")
                )
        except BaseException as error:  # handed to the scoring thread, never lost
            self._filled.put(error)

    def _decode(self, command: list[str]) -> tuple[int, int, str]:
        """Runs one FFmpeg command into the ring, with a timestamp file when
        pairing by timestamp. Returns (frames, exit code, error output)."""
        if not self._timestamps:
            return self._decode_with(command, None)
        descriptor, name = tempfile.mkstemp(prefix=temp_files.prefix("vml-vship-pts-"), suffix=".txt")
        os.close(descriptor)
        stamps_path = Path(name)
        try:
            return self._decode_with(_with_timestamps(command, stamps_path), stamps_path)
        finally:
            with contextlib.suppress(OSError):
                stamps_path.unlink()

    def _decode_with(self, command: list[str], stamps_path: Path | None) -> tuple[int, int, str]:
        """Pipes the frames of one FFmpeg command into the ring; with
        `stamps_path`, each frame's timestamp is read from it too."""
        try:
            process, reader = _spawn_raw_ffmpeg(command)
        except OSError as error:
            raise VshipUnavailableError(f"Could not start FFmpeg for Vship: {error}") from error
        stamps = _Timestamps(stamps_path, process, self._label) if stamps_path is not None else None
        with self._lock:
            self._process, self._reader = process, reader
        if self._process_handle is not None:
            self._process_handle.attach(process.pid)
        # FFmpeg's last error lines, for a failure's message. Its whole error
        # output was held until it ended: without end over a long film if it
        # warned on every frame.
        stderr_tail: collections.deque[bytes] = collections.deque(maxlen=_STDERR_LINES_KEPT)
        drain = threading.Thread(
            target=lambda: stderr_tail.extend(process.stderr) if process.stderr else None,
            daemon=True,
        )
        drain.start()
        frames = 0
        y4m = _is_y4m(command)
        try:
            if y4m and not self._y4m_header(reader):
                y4m = None  # FFmpeg wrote nothing: its exit code says why
            while y4m is not None and not self._stopping.is_set():
                slot = self._free.get()
                if slot == _EOF:
                    break
                _gather(reader, len(self.views[slot]) + (len(_Y4M_FRAME) if y4m else 0), process, self._stopping)
                if y4m and not self._y4m_frame(reader):
                    self._free.put(slot)
                    break
                received = self._fill(reader, slot)
                if received == len(self.views[slot]):
                    if stamps is not None:
                        try:
                            self.pts[slot] = stamps.next()
                        except BaseException:
                            self._free.put(slot)
                            raise
                        self.time_base = stamps.time_base
                    self._filled.put(slot)
                    frames += 1
                    continue
                self._free.put(slot)
                if received or y4m:  # a "FRAME" line with no frame after it
                    raise VshipUnavailableError(f"FFmpeg ended partway through a {self._label} frame.")
                break
        finally:
            with contextlib.suppress(OSError):
                reader.close()
            code = process.wait()
            if stamps is not None:
                stamps.close()
            drain.join(timeout=5)
            if self._process_handle is not None:
                self._process_handle.detach(process.pid)
        message = b"".join(stderr_tail)[-2000:].decode("utf-8", errors="replace").strip()
        return frames, code, message

    def _y4m_header(self, reader) -> bool:
        """Reads a YUV4MPEG stream's first line; False if there is no stream."""
        line, byte = bytearray(), bytearray(1)
        while reader.readinto(byte):
            if byte == b"\n":
                if not line.startswith(b"YUV4MPEG2 "):
                    break
                return True
            line += byte
            if len(line) > 1024:
                break
        if line:
            raise VshipUnavailableError(f"FFmpeg's {self._label} frames are not a YUV4MPEG stream.")
        return False

    def _y4m_frame(self, reader) -> bool:
        """Reads the "FRAME" line before a frame; False at the stream's end."""
        marker = bytearray(len(_Y4M_FRAME))
        received = _read_exact(reader, memoryview(marker))
        if received == 0:
            return False
        if marker != _Y4M_FRAME:
            raise VshipUnavailableError(f"FFmpeg's {self._label} frames are not where they were expected.")
        return True

    def _fill(self, reader, slot: int) -> int:
        """Reads one frame into `slot`, and returns the number of bytes read."""
        if self._split is None:
            return _read_exact(reader, self.views[slot])
        luma, staging, pairs, planes = self._split
        received = _read_exact(reader, self.views[slot][:luma])
        if received < luma:
            return received
        chroma = _read_exact(reader, staging)
        if chroma == len(staging):
            # Strided copies. numpy releases the GIL for them, so the other
            # video's reader and the scoring lanes keep going meanwhile.
            u, v = planes[slot]
            u[:] = pairs[:, 0]
            v[:] = pairs[:, 1]
        return received + chroma

    def next(self, cancel_event: threading.Event | None, abort: threading.Event | None = None) -> int:
        """Returns the next filled slot, or _EOF. Raises the reader's error,
        PerceptualCancelled on cancel, or _LaneFailedError once `abort` is
        set.

        A lane that fails keeps the slots of the frames it was handed, so
        after a failure the ring can fill with slots that nobody will give
        back: the reader then waits for a free slot, and no frame ever comes.
        Waiting on `abort` as well lets the pass end with the lane's error.
        """
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user")
            if abort is not None and abort.is_set():
                raise _LaneFailedError
            try:
                item = self._filled.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    def release(self, slot: int) -> None:
        """Gives a slot back once Vship is done with it."""
        self._free.put(slot)

    def close(self) -> None:
        """Stops decoding and the reading thread, and frees the ring."""
        self._stopping.set()
        self._free.put(_EOF)  # wake a reader waiting for a slot
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError):
                proc_util.terminate(process)
        if self._thread.ident is not None:  # never started: nothing to wait for
            self._thread.join(timeout=10)
        # Pinned memory is freed only once the reader cannot be writing to it.
        if not self._thread.is_alive():
            for buffer in self.buffers:
                buffer.close()


def _pinned_ring(lib: ctypes.CDLL, frame_bytes: int, gpu_id: int) -> list[_PinnedBuffer]:
    """Returns _RING_SLOTS pinned frame buffers. They are allocated one by
    one, so that a failure part-way through frees those already allocated;
    otherwise they (pinned memory, 25 MB each for 4K 10-bit) would stay
    allocated for the rest of the session, once per failed attempt."""
    buffers: list[_PinnedBuffer] = []
    try:
        for _ in range(_RING_SLOTS):
            buffers.append(_PinnedBuffer(lib, frame_bytes, gpu_id))
    except BaseException:
        for buffer in buffers:
            buffer.close()
        raise
    return buffers


class _FrameSelection:
    """Says which decoded pictures a video's FFmpeg chain would send to
    Vship (_filter_chain and the command's -t): its frames are numbered,
    select keeps every step-th, and the output's -t keeps the first `limit`
    numbered frames (frame_sync.frames_within; None keeps every frame)."""

    def __init__(self, step: int, limit: int | None) -> None:
        self._step = step
        self._limit = limit
        self._number = 0

    def take(self) -> bool | None:
        """For the next decoded picture: True if it is piped, False if it is
        skipped, None if the output has ended at it."""
        number = self._number
        self._number += 1
        if self._limit is not None and number >= self._limit:
            return None
        return number % self._step == 0


class _TimedFrameSelection:
    """_FrameSelection when pairing by timestamp. The chain's select filter
    keeps every step-th picture, setpts starts their times at 0, and the
    output's -t is FFmpeg's trim filter: it keeps a picture while its time
    is below the limit in the stream's time base
    (frame_sync_by_time.duration_in), and ends the output at the first
    picture that is not."""

    def __init__(self, step: int, duration_limit: str | None) -> None:
        self._step = step
        self._duration_limit = duration_limit
        self._index = 0
        self._first: int | None = None
        self._limit: int | None = None

    def take(self, pts: int, time_base: Fraction) -> bool | None:
        """For the next decoded picture: True if it is piped, False if it is
        skipped, None if the output has ended at it."""
        index = self._index
        self._index += 1
        if index % self._step:
            return False
        if self._first is None:
            self._first = pts
            if self._duration_limit is not None:
                self._limit = frame_sync_by_time.duration_in(self._duration_limit, time_base)
        if self._limit is not None and pts - self._first >= self._limit:
            return None
        return True


class _NativeFrameStream:
    """One video decoded in this process by the GPU's decoder
    (gpu_frames.GpuFrameStream), feeding a ring of pinned frame buffers as
    _FrameStream does. Each picture that FFmpeg's chain would send
    (_FrameSelection) goes straight into a free slot: copied there by the
    GPU from NVIDIA's decoder, or by the CPU in one pass from Intel's or
    AMD's. A decoding failure after the start is raised from next() as
    GpuDecodeFailedError, and the pass is then made again through FFmpeg.

    `limit` is the number of frames the duration limit keeps (None keeps
    every frame), or, when pairing by timestamp (`by_time`), the limit as
    -t takes it (_TimedFrameSelection)."""

    def __init__(self, lib: ctypes.CDLL, frame_bytes: int, decoder: gpu_frames.GpuFrameStream, step: int,
                 limit: int | str | None, label: str, gpu_id: int = 0, by_time: bool = False) -> None:
        self.buffers = _pinned_ring(lib, frame_bytes, gpu_id)
        self._decoder = decoder
        self._by_time = by_time
        self._selection = _TimedFrameSelection(step, limit) if by_time else _FrameSelection(step, limit)
        self.pts = [0] * _RING_SLOTS  # as _FrameStream's, when pairing by timestamp
        self.time_base: Fraction | None = None
        self._label = label
        self._free: queue.Queue[int] = queue.Queue()
        self._filled: queue.Queue[int | BaseException] = queue.Queue()
        for slot in range(_RING_SLOTS):
            self._free.put(slot)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"vship-gpu-decode-{label}", daemon=True)

    def start(self) -> None:
        """Starts decoding."""
        self._decoder.start()
        self._thread.start()

    def _run(self) -> None:
        """The reading thread: copies each selected picture into a free slot."""
        decoder = self._decoder
        try:
            while not self._stopping.is_set():
                try:
                    item = decoder.next(100)
                except TimeoutError:
                    continue
                if item is None:
                    break  # the end, checked (GpuFrameStream.verify)
                picture, pts = item
                try:
                    taken = self._selection.take(pts, decoder.time_base) if self._by_time else self._selection.take()
                    if taken is None:
                        decoder.verify()  # the pictures up to here
                        break
                    if not taken:
                        continue
                    slot = self._free.get()
                    if slot == _EOF:
                        return
                    decoder.download(picture, self.buffers[slot].address.value)
                    if self._by_time:
                        self.pts[slot], self.time_base = pts, decoder.time_base
                    self._filled.put(slot)
                finally:
                    decoder.release(picture)
            self._filled.put(_EOF)
        except BaseException as error:  # handed to the scoring thread, never lost
            self._filled.put(error)

    next = _FrameStream.next

    def release(self, slot: int) -> None:
        """Gives a slot back once Vship is done with it."""
        self._free.put(slot)

    def close(self) -> None:
        """Stops decoding and the reading thread, and frees the ring."""
        self._stopping.set()
        self._free.put(_EOF)  # wake a reader waiting for a slot
        self._decoder.abort()
        if self._thread.ident is not None:
            self._thread.join(timeout=30)
        if not self._thread.is_alive():
            self._decoder.close()
            # Pinned memory is freed only once nothing can be copying to it.
            for buffer in self.buffers:
                buffer.close()
        else:
            _log.error("The %s's GPU decoding thread did not stop; its memory is left allocated", self._label)


#: For each -hwaccel the app picks for FFmpeg (gpu.pick_hwaccel), the GPU
#: decoder that decodes the same video in the scoring process: NVIDIA's
#: through nvcuvid, Intel's through oneVPL, AMD's through AMF.
_GPU_DECODERS = {"cuda": "nvidia", "qsv": "intel", "d3d11va": "amd"}


def _decoded_here(hwaccel: str | None, device: VshipDevice) -> str | None:
    """Returns the decoder that decodes, in the scoring process, a video
    FFmpeg would decode with `hwaccel`: the GPU's, or the software decoder
    for a video FFmpeg would decode in software (`hwaccel` None). Returns
    None when FFmpeg must decode it. NVIDIA's decoder copies each picture
    into the ring with the GPU, so the ring must be CUDA's page-locked
    memory (Vship's CUDA build). Intel's, AMD's and the software decoder
    give pictures in system memory, which the CPU copies into any build's
    ring in one pass."""
    if hwaccel is None:
        return gpu_frames.SOFTWARE if gpu_frames.software_bundled() else None
    backend = _GPU_DECODERS.get(hwaccel)
    if backend == "nvidia" and device.backend != "cuda":
        return None
    return backend


def _native_decoder(info: VideoInfo, crop: CropBox | None, size: tuple[int, int], image_format: _ImageFormat,
                    frame_bytes: int, gpu_id: int, process_handle: ProcessHandle | None,
                    label: str, backend: str = "nvidia", algorithm: str = "bicubic") -> gpu_frames.GpuFrameStream | None:
    """Returns the decoder for one video of a pass, when GPU decoder
    `backend` can give it in the layout Vship is told: 8-bit planes, or
    P016's 16-bit samples as they are or shifted to 10-bit (as FFmpeg
    converts full-range 10-bit). Where it is scaled, it is scaled to `size`
    with `algorithm`, by the GPU, or by the CPU on Intel and AMD if their
    GPU scaling fails its check (gpu_frames.scale_note). Returns None, with
    the reason logged, when FFmpeg decodes the video."""
    shifts = {"yuv420p": 0, "yuv420p16le": 0, "yuv420p10le": 6}
    reason = None
    if image_format.pixel_format not in shifts:
        reason = f"Vship reads it as {image_format.pixel_format}"
    if reason is None:
        try:
            plan = gpu_frames.plan_decode(info, crop, shift=shifts[image_format.pixel_format], size=size,
                                            algorithm=algorithm)
            if plan.frame_bytes != frame_bytes:
                raise gpu_frames.GpuDecodeUnavailableError(
                    f"its frames would be {plan.frame_bytes} bytes, not {frame_bytes}")
            # Asked first, because a stream the decoder refuses once the pass
            # has started (10-bit H.264, say) means making the whole pass again.
            supported, refusal = gpu_frames.decoder_supports(gpu_id, plan, backend)
            if not supported:
                raise gpu_frames.GpuDecodeUnavailableError(refusal)
            decoder = gpu_frames.GpuFrameStream(info, plan, gpu_id, pool=4, process_handle=process_handle,
                                               backend=backend)
        except gpu_frames.GpuDecodeUnavailableError as error:
            reason = str(error)
        else:
            _log.info("Vship: the %s is decoded by the %s decoder in the scoring process", label, backend)
            return decoder
    _log.info("Vship: the %s is decoded by FFmpeg (%s)", label, reason)
    return None


class _PinnedBuffer:
    """One frame's buffer in Vship's pinned (page-locked) memory."""

    def __init__(self, lib: ctypes.CDLL, size: int, gpu_id: int) -> None:
        self.lib = lib
        self.gpu_id = gpu_id
        self.address = ctypes.c_void_p()
        error = lib.Vship_PinnedMalloc2(ctypes.byref(self.address), size, gpu_id)
        if error != 0 or not self.address.value:
            raise VshipUnavailableError(f"Could not allocate Vship pinned frame memory: {_message(lib, error)}")
        self.array = (ctypes.c_uint8 * size).from_address(self.address.value)

    def planes(self, layout: _ImageFormat, sizes: tuple[int, int, int]) -> _PLANES:
        """The addresses of the frame's three planes, in the order Vship reads them."""
        ptrs: list[_U8P] = []
        offsets = (0, sizes[0], sizes[0] + sizes[1])
        base = self.address.value
        for plane in layout.plane_order:
            ptrs.append(ctypes.cast(base + offsets[plane], _U8P))
        return (_U8P * 3)(*ptrs)

    def close(self) -> None:
        """Frees the buffer."""
        if self.address.value:
            self.lib.Vship_PinnedFree2(self.address, self.gpu_id)
            self.address = ctypes.c_void_p()


class _ScoreArray:
    """Each frame's score, in packed float64 chunks, written by frame index.

    Lanes finish frames out of order, so scores are stored by position
    rather than appended. They are kept in chunks rather than one array
    because a feature-length video's frame count is only estimated at the
    start, and growing one array would move it under a lane that is writing
    to it. The dispatching thread adds a new chunk before any lane can be
    handed an index in it, and existing chunks never move. 172,800 frames
    (two hours at 24 fps) take 1.4 MB, where a dict of Python floats takes
    about 17 MB.
    """

    _CHUNK = 8192

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def reserve(self, index: int) -> None:
        """Makes room for `index`. Called before the index is dispatched."""
        while index >= len(self._chunks) * self._CHUNK:
            self._chunks.append(np.full(self._CHUNK, np.nan))

    def __setitem__(self, index: int, value: float) -> None:
        self._chunks[index // self._CHUNK][index % self._CHUNK] = value

    def values(self, count: int) -> np.ndarray:
        """The first `count` scores, as float32."""
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)[:count].astype(np.float32)


class _MetricLane:
    """One Vship handler for one metric, on its own thread.

    Frame pairs are dealt to the lanes in turn. Each lane writes its score
    by frame index, so the order in which they finish does not matter. A
    pair's pinned slots go back to their readers once every metric has
    scored it.
    """

    def __init__(self, device: VshipDevice, key: str, src: _Colorspace, dist: _Colorspace,
                 source_planes, distorted_planes, src_strides: _I64_3, dist_strides: _I64_3,
                 scores: _ScoreArray, finished: Callable[[int], None], abort: threading.Event) -> None:
        self.key = key
        self._args = (device, src, dist)
        self._planes = (source_planes, distorted_planes)
        self._strides = (src_strides, dist_strides)
        self._scores, self._finished, self._abort = scores, finished, abort
        self.jobs: queue.Queue[tuple[int, int, int] | None] = queue.Queue()
        #: Set when the handler failed. The lane then hands frames back
        #: unscored, as _CvvdpLane does, so that the other metrics can finish.
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"vship-{key}", daemon=True)

    def start(self) -> None:
        """Starts the lane's thread."""
        self._thread.start()

    def _run(self) -> None:
        """The lane's thread: scores each pair it is given until stopped."""
        device, src, dist = self._args
        lib = _library(device)
        handler = None
        try:
            handler = _init_handler(device, self.key, src, dist)
        except BaseException as error:
            self.error = error
        try:
            while True:
                job = self.jobs.get()
                if job is None or self._abort.is_set():
                    return
                index, source_slot, distorted_slot = job
                if self.error is None:
                    try:
                        self._scores[index] = _compute_metric(
                            device, self.key, handler, self._planes[0][source_slot],
                            self._planes[1][distorted_slot], *self._strides,
                        )
                    except BaseException as error:
                        self.error = error
                self._finished(index)
        finally:
            if handler is not None:
                with contextlib.suppress(Exception):
                    lib.Vship_FreeHandler(handler)

    def stop(self) -> None:
        """Tells the lane to stop after the pairs already given."""
        self.jobs.put(None)

    def join(self) -> None:
        """Waits for the lane's thread to end."""
        self._thread.join()


def _init_handler(device: VshipDevice, key: str, src: _Colorspace, dist: _Colorspace) -> _Handle:
    """Makes a Vship handler for SSIMULACRA2 or Butteraugli."""
    lib = _library(device)
    if key == "ssimulacra2":
        init = _InitSsimulacra2(_INIT_SSIMULACRA2, src, dist, device.gpu_id)
    else:
        # Vship's defaults (doc/BUTTERAUGLI.md): the 2-norm, reported beside
        # the 3-norm the app shows, for a 203-nit display.
        init = _InitButteraugli(_INIT_BUTTERAUGLI, src, dist, 2, 203.0, device.gpu_id)
    handle = _Handle()
    error = lib.Vship_InitHandler(ctypes.byref(handle), ctypes.byref(init))
    if error != 0:
        raise VshipUnavailableError(f"Could not initialize Vship {key}: {_message(lib, None)}")
    return handle


def _compute_metric(
    device: VshipDevice, key: str, handler: _Handle, source_planes: _PLANES,
    distorted_planes: _PLANES, source_strides: _I64_3, distorted_strides: _I64_3,
) -> float:
    """Scores one frame pair with a handler, and returns the score."""
    lib = _library(device)
    label = "SSIMULACRA2" if key == "ssimulacra2" else "Butteraugli"
    # Butteraugli's distortion map is not wanted: its dstp stays NULL.
    score = (_ScoreSsimulacra2(_SCORE_SSIMULACRA2) if key == "ssimulacra2"
             else _ScoreButteraugli(_SCORE_BUTTERAUGLI))
    error = lib.Vship_ComputeHandler(handler, ctypes.byref(score), source_planes, distorted_planes,
                                     source_strides, distorted_strides)
    if error != 0:
        raise VshipUnavailableError(f"Vship {label} failed: {_handler_error(lib, handler, error)}")
    value = float(score.score if key == "ssimulacra2" else score.norm3)
    if not math.isfinite(value):
        raise VshipUnavailableError(f"Vship {label} returned a non-finite value.")
    return value


# ------------------------------------------------------------------ CVVDP
#
# Vship pools CVVDP's per-frame quality q as ColorVideoVDP does: a running
# sum of q^2 over the n frames scored since the last reset, reported as
# JOD(sqrt(sum / n)), except that a single frame is reported as
# JOD(q * IMAGE_INT). Vship_ResetScore clears only that sum (the temporal
# filters keep their history), so resetting it every second gives a JOD per
# second, and turning each second's JOD back into its sum rebuilds the
# whole video's score. On 2 s of the Beekeeper 4K AV1 encode, the rebuilt
# score matched an unreset handler's to 6e-7 JOD; on 49 and 73 frames at
# 23.976 fps, which end in a one-frame second, to 2e-7 and 3e-7 (pooling
# that frame like any other would be off by 0.01). The constants are
# Vship's (and ColorVideoVDP's) jod_a, jod_exp and image_int, from
# src/HIP/cvvdp/parameters.hpp and CVVDPComputingImplementation::run in
# Vship 5.1.1.
_JOD_A = 0.0439569391310215
_JOD_EXP = 0.9302042722702026
_IMAGE_INT = 0.577918291091919
_JOD_LINEAR_A = _JOD_A * 0.1 ** (_JOD_EXP - 1.0)


def _jod_from_quality(q: float) -> float:
    """CVVDP's JOD for a pooled quality q."""
    return 10.0 - (_JOD_A * q ** _JOD_EXP if q > 0.1 else _JOD_LINEAR_A * q)


def _quality_from_jod(jod: float) -> float:
    """The pooled quality q that gives a JOD (the inverse of _jod_from_quality)."""
    gap = max(0.0, 10.0 - jod)
    linear = gap / _JOD_LINEAR_A
    return linear if linear <= 0.1 else (gap / _JOD_A) ** (1.0 / _JOD_EXP)


def pool_cvvdp_windows(windows: list[tuple[int, int, float]]) -> float:
    """The whole video's JOD from (first frame, frames, JOD) windows."""
    total = sum(frames for _first, frames, _jod in windows)
    if total == 1:
        return windows[0][2]
    squares = 0.0
    for _first, frames, jod in windows:
        q = _quality_from_jod(jod)
        squares += (q / _IMAGE_INT) ** 2 if frames == 1 else frames * q * q
    return _jod_from_quality(math.sqrt(squares / total))


def _init_cvvdp(device: VshipDevice, src: _Colorspace, dist: _Colorspace,
                settings: CvvdpSettings, fps: float) -> _Handle:
    """Makes a Vship handler for CVVDP, with the display model and frame rate."""
    lib = _library(device)
    if not (math.isfinite(fps) and fps > 0):
        # CVVDP models how the eye integrates over time, so the frame rate is
        # part of the score, and must not be quietly replaced by a guess.
        raise VshipUnavailableError(
            "CVVDP needs the video's frame rate, and none was found (it was read as "
            f"{fps:g} fps)."
        )
    # The display is given as JSON text, which Vship parses itself when it
    # starts with "{" (DisplayModel::parseJson). Don't pass a file instead:
    # Vship opens it with a narrow-character path, which fails for a
    # Windows user name outside the ANSI code page.
    config = vship_display_json(settings.display).encode("utf-8")
    init = _InitCvvdp(_INIT_CVVDP, src, dist, fps, bool(settings.resize_to_display),
                      VSHIP_MODEL_KEY.encode(), config, device.gpu_id)
    handle = _Handle()
    error = lib.Vship_InitHandler(ctypes.byref(handle), ctypes.byref(init))
    if error != 0:
        raise VshipUnavailableError(f"Could not initialize Vship CVVDP: {_message(lib, None)}")
    return handle


def _compute_cvvdp(device: VshipDevice, handler: _Handle, source_planes: _PLANES,
                   distorted_planes: _PLANES, source_strides: _I64_3, distorted_strides: _I64_3) -> float:
    """Scores the next frame pair; returns the JOD of the frames since the last reset."""
    lib = _library(device)
    score = _ScoreCvvdp(_SCORE_CVVDP)  # no distortion map: dstp stays NULL
    error = lib.Vship_ComputeHandler(handler, ctypes.byref(score), source_planes, distorted_planes,
                                     source_strides, distorted_strides)
    if error != 0:
        raise VshipUnavailableError(f"Vship CVVDP failed: {_handler_error(lib, handler, error)}")
    value = float(score.score)
    if not math.isfinite(value):
        raise VshipUnavailableError("Vship CVVDP returned a non-finite value.")
    return value


def _reset_cvvdp_score(device: VshipDevice, handler: _Handle) -> None:
    """Starts a new CVVDP window: clears the score sum, not the history."""
    lib = _library(device)
    error = lib.Vship_ResetScore(handler)
    if error != 0:
        raise VshipUnavailableError(f"Vship CVVDP failed: {_handler_error(lib, handler, error)}")


def _free_cvvdp(device: VshipDevice, handler: _Handle) -> None:
    """Frees a CVVDP handler."""
    _library(device).Vship_FreeHandler(handler)


class _CvvdpLane:
    """CVVDP's one Vship handler, fed every frame pair in order.

    CVVDP judges each frame together with the ones before it, so its frames
    cannot be dealt in turn to two handlers, as SSIMULACRA2's are. The score
    is reset at each new second of video, which gives the JOD of every
    second (the per-second graph), and the overall JOD is pooled from those
    seconds (pool_cvvdp_windows).

    A CVVDP failure does not end the pass for SSIMULACRA2 and Butteraugli:
    the lane keeps the error, keeps handing each frame back so that its ring
    slots are freed, and the pass reports CVVDP as failed.
    """

    def __init__(self, device: VshipDevice, src: _Colorspace, dist: _Colorspace,
                 settings: CvvdpSettings, fps: float, source_planes, distorted_planes,
                 src_strides: _I64_3, dist_strides: _I64_3,
                 finished: Callable[[int], None], abort: threading.Event) -> None:
        self._args = (device, src, dist, settings, fps)
        self._planes = (source_planes, distorted_planes)
        self._strides = (src_strides, dist_strides)
        self._finished, self._abort = finished, abort
        self.jobs: queue.Queue[tuple[int, int, int] | None] = queue.Queue()
        self.error: BaseException | None = None
        #: (first frame index, frames, JOD) for each second scored.
        self.windows: list[tuple[int, int, float]] = []
        self._thread = threading.Thread(target=self._run, name="vship-cvvdp", daemon=True)

    def start(self) -> None:
        """Starts the lane's thread."""
        self._thread.start()

    def _run(self) -> None:
        """The lane's thread: scores every pair in order, one window per second."""
        device, src, dist, settings, fps = self._args
        handler = None
        try:
            handler = _init_cvvdp(device, src, dist, settings, fps)
        except BaseException as error:
            self.error = error
        start = count = second = 0
        jod = math.nan
        try:
            while True:
                job = self.jobs.get()
                if job is None or self._abort.is_set():
                    return
                index, source_slot, distorted_slot = job
                if self.error is None:
                    try:
                        this_second = int(index / fps)
                        if count and this_second != second:
                            self.windows.append((start, count, jod))
                            _reset_cvvdp_score(device, handler)
                            start, count = index, 0
                        second = this_second
                        jod = _compute_cvvdp(device, handler, self._planes[0][source_slot],
                                             self._planes[1][distorted_slot], *self._strides)
                        count += 1
                    except BaseException as error:
                        self.error = error
                self._finished(index)
        finally:
            if self.error is None and count:
                self.windows.append((start, count, jod))
            if handler is not None:
                with contextlib.suppress(Exception):
                    _free_cvvdp(device, handler)

    def stop(self) -> None:
        """Tells the lane to stop after the pairs already given."""
        self.jobs.put(None)

    def join(self) -> None:
        """Waits for the lane's thread to end."""
        self._thread.join()


#: One Vship pass at a time in the whole app, because a pass already fills
#: the GPU. At 4K, two passes at once scored no faster than one after the
#: other (84.5 against 83.6 pairs a second, Vship's CUDA build), while
#: holding twice the GPU memory (9.1 against 4.5 GB, more than most cards
#: have). Running two jobs in parallel still pays off (23% at 1080p, 27% at
#: 4K), because one job's VMAF, PSNR and SSIM, which are CPU work, overlap
#: the other's Vship pass. Only this GPU pass waits its turn, never black-bar
#: detection or the FFmpeg metrics. The lock is shared with VMAF on the GPU
#: (gpu.GPU_PASS).
_gpu_pass = GPU_PASS


def vship_passes(
    specs: tuple[MetricRequestSpec, ...], together: bool = False,
) -> list[tuple[MetricRequestSpec, ...]]:
    """Returns the Vship passes `specs` are scored in, in order: one per
    metric, or with `together` one per set of frames scored, so that every
    metric scoring the same frames shares one pass and the video is decoded
    once for them all. CVVDP scores every frame, so with frame subsampling
    it keeps a pass of its own."""
    if not together:
        return [(spec,) for spec in specs]
    groups: dict[int, list[MetricRequestSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.coverage.step if spec.coverage is not None else 1, []).append(spec)
    return [tuple(group) for group in groups.values()]


def run_vship_task(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    on_pass_done: Callable[[PerceptualTaskOutput], None] | None = None,
    together: bool = False,
    on_pass: Callable[[int, int, tuple[str, ...]], None] | None = None,
) -> PerceptualTaskOutput:
    """Scores `specs` on the GPU once no other Vship pass is running.

    `on_pass_done` gets each pass's scores as it finishes, before the next
    pass starts, so the window shows a score the moment it exists. `on_pass`
    is called as each pass starts, with its number, how many passes there
    are, and the keys of the metrics it scores, because the run line's
    figures are per metric.

    By default each metric gets a pass of its own, and the video is decoded
    again for each. Scoring them in one pass holds every metric's GPU memory
    at once: 7.3 GB for SSIMULACRA2, Butteraugli and CVVDP together on a
    full 3840x2160 frame (Vship's CUDA build; its Vulkan build reports no
    memory size to check against), more than an 8 GB card has free. That
    gains little speed where the GPU decodes: 34 fps together against about
    31.5 fps one after the other at 4K, and 91 against 90.5 fps at 1080p
    (RTX 5090, Beekeeper AV1 and a synthetic SDR clip). One at a time, the
    peak is the largest single metric's: SSIMULACRA2 2.7 GB, Butteraugli
    4.4 GB, CVVDP 4.8 GB.

    `together` (Settings > GPU metrics) scores them in one pass anyway (see
    vship_passes). Where the CPU decodes (4K VVC, which no GPU decodes),
    decoding each video once instead of once per metric saves far more than
    the memory costs. A metric that fails in a shared pass, most likely out
    of GPU memory, is calculated again in a pass of its own.

    A metric whose pass fails is reported in the output's `failures` while
    the others still run. Only when every pass fails is the first error
    raised, and the caller may then retry on the CPU.

    A metric this GPU does not calculate at this size (GPU_PIXEL_LIMITS) is
    not started: it is reported as failed with the reason, before any GPU
    work.
    """
    task = _VshipTask(
        source, distorted, request, specs, device, source_crop, distorted_crop,
        on_progress=on_progress, on_status=on_status, cancel_event=cancel_event, process_handle=process_handle,
        on_pass_done=on_pass_done, together=together, on_pass=on_pass,
    )
    return task.run()


class _VshipTask:
    """Scores one task's Vship metrics, in one pass or several (see
    run_vship_task, which makes one)."""

    def __init__(
        self, source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
        specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
        source_crop: CropBox | None, distorted_crop: CropBox | None, *,
        on_progress: Callable[[int, int, float], None] | None,
        on_status: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
        process_handle: ProcessHandle | None,
        on_pass_done: Callable[[PerceptualTaskOutput], None] | None,
        together: bool,
        on_pass: Callable[[int, int, tuple[str, ...]], None] | None,
    ) -> None:
        """Keeps the task's arguments; run() scores it."""
        self.source, self.distorted, self.request, self.specs, self.device = source, distorted, request, specs, device
        self.source_crop, self.distorted_crop = source_crop, distorted_crop
        self.on_progress, self.on_status, self.cancel_event = on_progress, on_status, cancel_event
        self.process_handle, self.on_pass_done, self.together, self.on_pass = (
            process_handle, on_pass_done, together, on_pass)
        # What the passes made so far.
        self.metrics = MetricResultSet()
        self.failures: dict[str, str] = {}
        self.errors: list[BaseException] = []
        self.frames = 0
        # How far each pass run so far got, in its own progress units. The
        # task's progress is one count across every pass, with a retry's
        # after the passes before it. Otherwise the task would go from 100%
        # back to 0% when a metric that failed in the shared pass is
        # calculated again, and its time left would be that retry's alone.
        self.reached: list[int] = []

    def run(self) -> PerceptualTaskOutput:
        """Scores the task (see run_vship_task)."""
        refused = self._refused_for_size()
        if refused:
            return self._run_without(refused)
        self._wait_for_gpu()
        try:
            if len(self.specs) <= 1:
                if self.on_pass is not None:
                    self.on_pass(1, 1, tuple(spec.key for spec in self.specs))
                return self._run_pass(self.specs, self.on_progress)
            return self._run_passes()
        finally:
            _gpu_pass.release()

    def _refused_for_size(self) -> dict[str, str]:
        """The metrics this GPU does not calculate at the size compared at
        (GPU_PIXEL_LIMITS), each with the reason."""
        device, specs = self.device, self.specs
        if not any((device.backend, device.vendor, spec.key) in GPU_PIXEL_LIMITS for spec in specs):
            return {}
        size = _scaled_sizes(self.source, self.distorted, self.request.recipe, self.source_crop,
                             self.distorted_crop)[0]

        def computed(spec: MetricRequestSpec) -> tuple[int, int]:
            """The size `spec` is calculated at."""
            # CVVDP set to fill the display is calculated at the display's
            # size, so a 1080p comparison on the default 4K display would
            # otherwise get through at the size that hangs Intel's GPU.
            if spec.key != "cvvdp":
                return size
            return cvvdp_computed_size(size, CvvdpSettings.from_spec_parameters(spec.parameters))

        return {spec.key: reason for spec in specs if (reason := size_refusal(device, spec.key, size, computed(spec)))}

    def _run_without(self, refused: dict[str, str]) -> PerceptualTaskOutput:
        """Scores the metrics not `refused`, which are reported as failed
        with their reasons."""
        for key, reason in refused.items():
            _log.warning("%s not started: %s", metric_definition(key).label, reason)
        specs = tuple(spec for spec in self.specs if spec.key not in refused)
        if not specs:
            raise VshipPassesFailedError(next(iter(refused.values())), refused)
        try:
            output = run_vship_task(
                self.source, self.distorted, self.request, specs, self.device, self.source_crop,
                self.distorted_crop, on_progress=self.on_progress, on_status=self.on_status,
                cancel_event=self.cancel_event, process_handle=self.process_handle,
                on_pass_done=self.on_pass_done, together=self.together, on_pass=self.on_pass)
        except (PerceptualCancelled, ComparisonCutShortError):
            raise
        except Exception as error:
            # The others failed too. Each keeps its own reason (the caller
            # reads `failures`), so that CVVDP is not given theirs.
            reasons = getattr(error, "failures", None) or {spec.key: str(error) for spec in specs}
            raise VshipPassesFailedError(str(error), {**reasons, **refused}) from error
        return replace(output, failures={**output.failures, **refused})

    def _wait_for_gpu(self) -> None:
        """Waits until no other Vship pass is running, and takes the GPU."""
        if _gpu_pass.acquire(blocking=False):
            return
        if self.on_status:
            self.on_status(Status(GPU_WAIT_MESSAGE, kind=GPU_WAIT))
        while not _gpu_pass.acquire(timeout=0.1):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user")

    def _run_pass(self, specs: tuple[MetricRequestSpec, ...],
                  progress: Callable[[int, int, float], None] | None) -> PerceptualTaskOutput:
        """One Vship pass scoring `specs`."""
        return _run_vship_pass(
            self.source, self.distorted, self.request, specs, self.device, self.source_crop, self.distorted_crop,
            on_progress=progress, on_status=self.on_status,
            cancel_event=self.cancel_event, process_handle=self.process_handle,
        )

    def _run_passes(self) -> PerceptualTaskOutput:
        """Scores the metrics in the passes vship_passes groups them into,
        then each metric that failed in a shared pass in a pass of its own."""
        passes = vship_passes(self.specs, self.together)
        self._run_groups(passes, 0, len(passes))
        shared = [spec for group in passes if len(group) > 1 for spec in group if spec.key in self.failures]
        if shared:
            self._retry_alone(shared, len(passes))
        if not self.metrics:
            if not self.errors:
                raise VshipUnavailableError("Vship produced no scores.")
            raise VshipPassesFailedError(str(self.errors[0]), self.failures) from self.errors[0]
        return PerceptualTaskOutput(self.metrics, self.source_crop, self.distorted_crop, self.frames, self.failures)

    def _retry_alone(self, shared: list[MetricRequestSpec], passes_before: int) -> None:
        """Scores each metric that failed in a shared pass in a pass of its
        own, numbered after the `passes_before` passes."""
        labels = " and ".join(metric_definition(spec.key).label for spec in shared)
        _log.warning("%s failed in the GPU pass shared with the other metrics; calculating %s in a "
                     "pass of %s own", labels, *(("it", "its") if len(shared) == 1 else ("them", "their")))
        if self.on_status:
            self.on_status(f"{labels} failed in the shared GPU pass; calculating "
                           f"{'it' if len(shared) == 1 else 'each'} in a pass of its own…")
        for spec in shared:
            del self.failures[spec.key]
        failed_before = list(self.errors)
        self.errors.clear()
        # Numbered after the passes before them ("GPU metric 2/2" after a
        # shared pass), so that the window counts the shared pass as done.
        self._run_groups([(spec,) for spec in shared], passes_before, passes_before + len(shared))
        self.errors.extend(failed_before)  # the retries' own errors first

    def _run_groups(self, passes: list[tuple[MetricRequestSpec, ...]], first: int, count: int) -> None:
        """Runs `passes` as passes first + 1 onwards of `count`, keeping
        each one's scores, or its error as each of its metrics' failure."""
        for offset, group in enumerate(passes):
            number = first + offset
            labels = " + ".join(metric_definition(spec.key).label for spec in group)
            if self.on_status and count > 1:
                self.on_status(Status(f"GPU metric {number + 1}/{count}: {labels}", kind=GPU_PASS_STATUS))
            if self.on_pass is not None:
                self.on_pass(number + 1, count, tuple(spec.key for spec in group))
            self.reached.append(0)
            progress = (self._pass_progress(len(self.reached) - 1, sum(self.reached), count - number)
                        if self.on_progress is not None else None)
            try:
                output = self._run_pass(group, progress)
            except (PerceptualCancelled, ComparisonCutShortError):
                # A video that ends early is cut short for every metric.
                # It is the file's fault, the same in each pass and on the CPU.
                raise
            except Exception as error:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise PerceptualCancelled("Cancelled by user") from error
                _log.error("GPU metric %d/%d (%s) failed: %s", number + 1, count, labels, error)
                self.errors.append(error)
                self.failures.update(dict.fromkeys((spec.key for spec in group), str(error)))
                continue
            for key in output.metrics:
                self.metrics.add(output.metrics.get(key))
            self.failures.update(output.failures)
            self.frames = max(self.frames, output.compared_frame_count)
            if self.on_pass_done is not None and output.metrics:
                self.on_pass_done(output)

    def _pass_progress(self, slot: int, done: int, left: int) -> Callable[[int, int, float], None]:
        """Reports a pass's progress as the task's: after `done` units of the
        passes before it, with `left` passes of its length to go, this one
        included (they score the same frames)."""
        on_progress = self.on_progress
        assert on_progress is not None  # only asked for when there is one

        def progress(current: int, total: int, fps: float) -> None:
            self.reached[slot] = current
            on_progress(done + current, done + left * total, fps)
        return progress


def _run_vship_pass(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    """Runs one Vship pass (_score_vship_pass) in a process of its own.

    Vship and the GPU driver under it run in the process that loads them,
    so a crash in either would take the app down, with every video's
    progress and no message. Instead it ends the pass's process: the pass
    fails like any other GPU failure (its metrics are calculated on the CPU,
    or fail with the reason), and the run goes on. The pass's FFmpeg
    processes are attached to `process_handle`, so Pause and Cancel reach
    them.

    A device whose library this process has already loaded is scored here,
    because a crash it causes would be this process's anyway. The app's
    devices come from the isolated probe without a library."""
    if device.loaded is not None:
        return _score_vship_pass(
            source, distorted, request, specs, device, source_crop, distorted_crop,
            on_progress=on_progress, on_status=on_status, cancel_event=cancel_event, process_handle=process_handle,
        )
    try:
        return run_isolated(
            _score_vship_pass, source, distorted, request, specs, device, source_crop, distorted_crop,
            what="Vship", callbacks=("on_progress", "on_status"), on_progress=on_progress, on_status=on_status,
            process_handle=process_handle, cancel_event=cancel_event, cancelled=PerceptualCancelled,
        )
    except IsolatedCrashError as error:
        labels = ", ".join(metric_definition(spec.key).label for spec in specs)
        _log.error("Vship pass (%s) failed: %s", labels, error)
        raise VshipUnavailableError(f"Vship GPU calculation failed: {error}") from error


def _score_vship_pass(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    """Runs one Vship pass, with the videos that the GPU's decoder takes
    decoded in this process (_NativeFrameStream). If that decoding fails
    after it has started (a damaged stream, pictures that do not match the
    packets), the pass is made again with FFmpeg decoding. When pairing by
    timestamp, a subsampled pass whose frames do not line up with the
    source's (_FramesApartError) is made again with every source frame."""
    kwargs = {"on_progress": on_progress, "on_status": on_status, "cancel_event": cancel_event,
              "process_handle": process_handle}
    native, every_source_frame = True, False
    while True:
        try:
            return _score_vship_pass_with(source, distorted, request, specs, device, source_crop, distorted_crop,
                                          native=native, every_source_frame=every_source_frame, **kwargs)
        except gpu_frames.GpuDecodeFailedError as error:
            if not native:
                raise
            _log.warning("Decoding in the Vship pass failed; decoding through FFmpeg instead: %s", error)
            if on_status:
                on_status(gpu_frames.decoding_failed_status(error))
            native = False
        except _FramesApartError as error:
            if every_source_frame:
                raise VshipUnavailableError(str(error)) from error
            _log.info("Vship pass: %s; it is made again with every source frame", error)
            every_source_frame = True


def _score_vship_pass_with(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    native: bool,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    every_source_frame: bool = False,
) -> PerceptualTaskOutput:
    """_score_vship_pass's single attempt: `native` decodes in this process
    where it can, and `every_source_frame` takes every source frame when
    pairing by timestamp."""
    vship_pass = _VshipPass(
        source, distorted, request, specs, device, source_crop, distorted_crop, native=native,
        on_progress=on_progress, on_status=on_status, cancel_event=cancel_event, process_handle=process_handle,
        every_source_frame=every_source_frame,
    )
    return vship_pass.run()


@dataclass(frozen=True)
class _VideoPlan:
    """How one video of a Vship pass is decoded and handed to Vship."""

    info: VideoInfo
    crop: CropBox | None
    side: str  # "source" or "distorted", as HwAccelPlan names them
    label: str  # "reference" or "test video", for messages
    hwaccel: str | None  # how FFmpeg would decode it; None for software
    passthrough: tuple[_ImageFormat, str] | None  # see _passthrough_format
    image_format: _ImageFormat
    size: tuple[int, int]  # the size it is compared at
    color: _Colorspace
    frame_bytes: int
    plane_sizes: tuple[int, int, int]
    strides: _I64_3


class _VshipPass:
    """One pass of Vship's metrics over a pair of videos: both videos are
    decoded once, their frames are paired, and each pair is handed to every
    metric's lanes (threads of their own) and then given back to its
    decoder.

    A pair's two decoder slots are counted: a slot goes back to its stream
    once the pairing and every metric are done with it. When pairing by
    timestamp, one source frame can be in more than one pair.
    """

    def __init__(
        self, source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
        specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
        source_crop: CropBox | None, distorted_crop: CropBox | None, *,
        native: bool,
        on_progress: Callable[[int, int, float], None] | None,
        on_status: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
        process_handle: ProcessHandle | None,
        every_source_frame: bool,
    ) -> None:
        """Checks the request and plans the pass; run() makes it. Raises
        VshipUnavailableError for a request Vship cannot score."""
        if not specs or any(spec.backend_id != BACKEND_ID or spec.key not in _METRICS for spec in specs):
            raise ValueError("Vship task requires supported perceptual metric specs")
        if request.recipe.resample_test is not None:
            raise VshipUnavailableError("The GPU metrics do not support resolution round-trip tests yet.")
        _validate_pair(source, distorted, request.recipe)
        if cancel_event is not None and cancel_event.is_set():
            raise PerceptualCancelled("Cancelled by user")
        self.source, self.distorted, self.request, self.specs, self.device = source, distorted, request, specs, device
        self.native, self.process_handle = native, process_handle
        self.on_progress, self.on_status, self.cancel_event = on_progress, on_status, cancel_event
        self.lib = _library(device)
        self.videos = self._plan_videos(source_crop, distorted_crop)

        self.step = specs[0].coverage.step if specs[0].coverage is not None else 1
        if any((spec.coverage.step if spec.coverage is not None else 1) != self.step for spec in specs):
            raise VshipUnavailableError("Perceptual metrics in one task must use the same frame coverage.")
        self.cvvdp_spec = next((spec for spec in specs if spec.key == "cvvdp"), None)
        self.frame_specs = tuple(spec for spec in specs if spec.key != "cvvdp")
        if self.cvvdp_spec is not None and self.step != 1:
            raise VshipUnavailableError("CVVDP scores every frame; it cannot be subsampled.")

        # Where each video is decoded, reported as FFmpeg's runs report it, so
        # that the window's run line can show "Decoder: Source: GPU, test video:
        # CPU" for a video with only GPU metrics too.
        self.decode = {video.side: video.hwaccel for video in self.videos}
        self.decode_lock = threading.Lock()

        # Pairing by frame number, both videos' frames are numbered at the test
        # video's frame rate (frame_sync) and both are cut at the duration limit,
        # so they keep the same frames. Pairing by timestamp, only the test video
        # is cut.
        self.by_time = by_timestamp()
        self.numbering_rate = frame_rate(distorted)
        limit = request.recipe.duration_limit
        self.duration_limit = f"{limit:.6f}" if limit > 0 else None
        expected = min(source.estimated_frame_count, distorted.estimated_frame_count)
        if self.by_time and limit > 0:
            expected = min(expected, max(1, math.ceil(limit * source.fps)))
        elif self.duration_limit is not None:
            expected = min(expected, max(1, frames_within(self.duration_limit, self.numbering_rate)))
        self.expected_frames = expected
        self.total_units = max(1, math.ceil(expected / self.step)) * self.step

        # Each test frame gives one pair, as in libvmaf: every step-th frame, up
        # to the duration limit. When pairing by timestamp, the source gives the
        # frames among which frame_pairs finds each test frame's pair, by time.
        # So it is never cut at the limit, and it is subsampled as the test video
        # is while the two line up (_FramesApartError). A source without a frame
        # rate gives every frame.
        self.every_source_frame = self.by_time and (every_source_frame or source.fps <= 0)
        self.check_apart = self.by_time and self.step > 1 and not self.every_source_frame

        self.streams: list[_FrameStream | _NativeFrameStream] = []
        self.lanes: list[_MetricLane | _CvvdpLane] = []
        self.lanes_by_metric: dict[str, list[_MetricLane]] = {}
        self.cvvdp_lane: _CvvdpLane | None = None
        self.scores: dict[str, _ScoreArray] = {spec.key: _ScoreArray() for spec in self.frame_specs}
        self.abort = threading.Event()
        self.slots_lock = threading.Lock()
        #: Each pair being scored: frame index -> [metrics left, source slot, test slot].
        self.pending: dict[int, list[int]] = {}
        #: For each stream (source, test): slot -> how many hold it.
        self.held: tuple[dict[int, int], dict[int, int]] = ({}, {})

    def _plan_videos(self, source_crop: CropBox | None,
                     distorted_crop: CropBox | None) -> tuple[_VideoPlan, _VideoPlan]:
        """How each video is decoded, sized and laid out for Vship.

        Decoding follows the row's GPU decode setting, per video and per codec:
        NVDEC for HEVC, AV1 and H.264 on NVIDIA, and software where the GPU
        has no decoder (VVC) or where FFmpeg's hardware decoding does not give
        the video's own pictures (gpu.pick_decode: 4:2:2, 4:4:4, 12-bit, an
        odd size, 10-bit H.264 on Intel and AMD). Otherwise hardware decoding
        is bit-exact, so it changes only the speed."""
        execution, recipe = self.request.execution, self.request.recipe
        vendor = execution.gpu_vendor if execution.gpu_decode else GpuVendor.NONE
        sizes = _scaled_sizes(self.source, self.distorted, recipe, source_crop, distorted_crop)
        plans = []
        for info, crop, size, side, label in ((self.source, source_crop, sizes[0], "source", "reference"),
                                             (self.distorted, distorted_crop, sizes[1], "distorted", "test video")):
            hwaccel = pick_decode(vendor, info.codec_name, info.pix_fmt, info.width, info.height)
            passthrough = _passthrough_format(info, hwaccel)
            image_format = passthrough[0] if passthrough else _image_format(info, self.device.version)
            frame_bytes, plane_sizes, strides, _ = image_format.frame_layout(*size)
            plans.append(_VideoPlan(info, crop, side, label, hwaccel, passthrough, image_format, size,
                                    _vship_colorspace(info, image_format, *size), frame_bytes, plane_sizes, strides))
        return plans[0], plans[1]

    def _labels(self) -> str:
        """The pass's metrics, as the log names them."""
        return ", ".join(metric_definition(spec.key).label for spec in self.specs)

    # ------------------------------------------------------------------ the pass

    def run(self) -> PerceptualTaskOutput:
        """Makes the pass and returns its scores. A metric whose lanes fail
        is reported in the output's failures while the others keep their
        scores; the pass fails only when every metric does."""
        started = time.perf_counter()
        try:
            # One at a time, so that if the test video's buffers cannot be
            # allocated, the source's stream is already in `streams` and the
            # finally block below frees it.
            for video in self.videos:
                self.streams.append(self._open_stream(video))
            if self.on_status:
                self.on_status(Status.decoding(f"Vship GPU ({self.device.name}): calculating {self._labels()}",
                                               HwAccelPlan(**self.decode)))
            for stream in self.streams:
                stream.start()
            self._start_lanes()
            frames, rate = self._feed_pairs()
            self._stop_lanes()
            return self._finish(frames, rate, started)
        except (PerceptualCancelled, ComparisonCutShortError, gpu_frames.GpuDecodeFailedError, _FramesApartError):
            # A video cut short is the file's fault, not the GPU's: the CPU would
            # stop as short. A failed GPU decode is made again through FFmpeg, and
            # a subsampled pass whose frames do not line up is made again with
            # every source frame (_score_vship_pass).
            raise
        except VshipUnavailableError as error:
            _log.error("Vship pass (%s) failed: %s", self._labels(), error, exc_info=error)
            raise
        except Exception as error:
            _log.error("Vship pass (%s) failed", self._labels(), exc_info=error)
            raise VshipUnavailableError(f"Vship GPU calculation failed: {error}") from error
        finally:
            # Lanes first: they read the pinned frames the streams own, so the
            # streams may only free that memory once no lane can touch it.
            self.abort.set()
            self._stop_lanes()
            for stream in self.streams:
                stream.close()

    def _open_stream(self, video: _VideoPlan) -> _FrameStream | _NativeFrameStream:
        """One video's frame stream: decoded in this process where it can
        be, by the same GPU decoder FFmpeg would use, else by FFmpeg."""
        cut = video.side == "distorted" or not self.by_time
        every = 1 if video.side == "source" and self.every_source_frame else self.step
        backend = _decoded_here(video.hwaccel, self.device) if self.native else None
        if backend is not None:
            decoder = _native_decoder(video.info, video.crop, video.size, video.image_format, video.frame_bytes,
                                      self.device.gpu_id, self.process_handle, video.label, backend,
                                      self.request.recipe.scale_algorithm)
            if decoder is not None:
                if self.by_time:
                    limit = self.duration_limit if cut else None
                else:
                    limit = frames_within(self.duration_limit, self.numbering_rate)
                try:
                    return _NativeFrameStream(self.lib, video.frame_bytes, decoder, every, limit, video.label,
                                              self.device.gpu_id, self.by_time)
                except BaseException:
                    decoder.close()
                    raise
        split = None
        if video.passthrough is not None:
            dtype = np.uint16 if video.image_format.sample != _VSHIP_ENUMS[8] else np.uint8
            split = (video.plane_sizes[0], video.plane_sizes[1], dtype)
        pixel_format = video.passthrough[1] if video.passthrough else video.image_format.pixel_format
        return _FrameStream(self.lib, video.frame_bytes, self._ffmpeg_commands(video, pixel_format, every, cut),
                            self.process_handle, video.label, split, self._decoded_in_software(video.side),
                            self.device.gpu_id, self.by_time)

    def _ffmpeg_commands(self, video: _VideoPlan, pixel_format: str, every: int, cut: bool) -> list[list[str]]:
        """The FFmpeg commands to try for one video: with hardware decoding
        first if it has any, then in software."""
        # The software retry sends the same layout as the hardware attempt,
        # because Vship's handlers are set up for one layout per video.
        attempts = []
        for accel in ([video.hwaccel, None] if video.hwaccel else [None]):
            filter_chain = _filter_chain(
                video.info, video.crop, video.size, pixel_format, every,
                self.request.recipe.scale_algorithm, accel, None if self.by_time else self.numbering_rate,
            )
            command = [
                ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
                *hwaccel_args(accel),
                "-i", str(video.info.path.resolve()), "-map", f"0:{VIDEO_STREAM}", "-an", "-sn", "-dn",
                "-vf", filter_chain,
            ]
            if cut:
                command += duration_args(self.duration_limit)
            command += ["-fps_mode", "passthrough", *_piped_as(pixel_format)]
            attempts.append(command)
        return attempts

    def _decoded_in_software(self, side: str) -> Callable[[], None]:
        """The callback that reports `side` falling back to software decoding."""
        def report() -> None:
            with self.decode_lock:  # the two inputs' threads can both fall back
                self.decode[side] = None
                if self.on_status:
                    self.on_status(Status.decoding(
                        f"GPU decode failed for the {'test video' if side == 'distorted' else side}, "
                        "decoding it in software", HwAccelPlan(**self.decode)))
        return report

    def _start_lanes(self) -> None:
        """Starts CVVDP's lane, if it is asked for, and each frame metric's
        lanes, on the planes of the streams' buffers."""
        source, distorted = self.videos
        source_stream, distorted_stream = self.streams
        source_planes = [buffer.planes(source.image_format, source.plane_sizes) for buffer in source_stream.buffers]
        distorted_planes = [buffer.planes(distorted.image_format, distorted.plane_sizes)
                            for buffer in distorted_stream.buffers]
        if self.cvvdp_spec is not None:
            self.cvvdp_lane = _CvvdpLane(
                self.device, source.color, distorted.color,
                CvvdpSettings.from_spec_parameters(self.cvvdp_spec.parameters), self.source.fps,
                source_planes, distorted_planes, source.strides, distorted.strides, self._pair_done, self.abort,
            )
            self.lanes.append(self.cvvdp_lane)
        for spec in self.frame_specs:
            self.lanes_by_metric[spec.key] = [
                _MetricLane(self.device, spec.key, source.color, distorted.color, source_planes, distorted_planes,
                            source.strides, distorted.strides, self.scores[spec.key], self._pair_done, self.abort)
                for _ in range(_LANES_PER_METRIC)
            ]
            self.lanes.extend(self.lanes_by_metric[spec.key])
        for lane in self.lanes:
            lane.start()

    def _stop_lanes(self) -> None:
        """Stops every lane and waits for it."""
        for lane in self.lanes:
            lane.stop()
        for lane in self.lanes:
            lane.join()

    def _lane_errors(self) -> dict[str, BaseException]:
        """Each failed metric's first error. One failed lane fails its metric,
        because the other lane's frames alone would leave holes."""
        errors = {key: next((lane.error for lane in lanes if lane.error is not None), None)
                  for key, lanes in self.lanes_by_metric.items()}
        if self.cvvdp_lane is not None:
            errors["cvvdp"] = self.cvvdp_lane.error
        return {key: error for key, error in errors.items() if error is not None}

    # ------------------------------------------------------------------ decoder slots

    def _hold(self, side: int, slot: int) -> None:
        """One more holder of stream `side`'s `slot`."""
        with self.slots_lock:
            self.held[side][slot] = self.held[side].get(slot, 0) + 1

    def _drop(self, side: int, slot: int) -> None:
        """One holder of stream `side`'s `slot` is done with it; the last one
        gives it back to the stream."""
        with self.slots_lock:
            holders = self.held[side][slot] - 1
            if holders:
                self.held[side][slot] = holders
                return
            del self.held[side][slot]
        self.streams[side].release(slot)

    def _pair_done(self, index: int) -> None:
        """One metric is done with pair `index`; the last one drops its slots."""
        with self.slots_lock:
            entry = self.pending[index]
            entry[0] -= 1
            if entry[0]:
                return
            del self.pending[index]
        self._drop(0, entry[1])
        self._drop(1, entry[2])

    def _frames_of(self, side: int) -> Callable[[], int | None]:
        """frame_pairs' source of stream `side`'s frames: each call gives the
        next frame's slot, held for the pairing, or None at the stream's end."""
        stream = self.streams[side]

        def next_frame() -> int | None:
            slot = stream.next(self.cancel_event)
            if slot == _EOF:
                return None
            self._hold(side, slot)
            return slot
        return next_frame

    def _timed_frames_of(self, side: int, first: int) -> Callable[[], tuple[int, int] | None]:
        """frame_sync_by_time.frame_pairs' source of stream `side`'s frames,
        starting with `first` (already taken, for its time base): each call
        gives the next frame's slot and timestamp, or None at the end."""
        stream, waiting = self.streams[side], [first]

        def pull() -> tuple[int, int] | None:
            slot = waiting.pop() if waiting else stream.next(self.cancel_event)
            if slot == _EOF:
                return None
            self._hold(side, slot)
            return slot, stream.pts[slot]
        return pull

    # ------------------------------------------------------------------ pairing

    def _pairs(self) -> Generator[tuple[int, int]]:
        """The (test slot, source slot) pairs: each test frame with the
        source frame of the same number, as every metric pairs them
        (frame_sync), until either video ends; or by timestamp. The longer
        video's reader is stopped when the streams close."""
        if self.by_time:
            return self._timed_pairs()
        return frame_pairs(self._frames_of(1), self._frames_of(0), lambda slot: self._drop(1, slot),
                           lambda slot: self._drop(0, slot))

    def _timed_pairs(self) -> Generator[tuple[int, int]]:
        """Yields the (test slot, source slot) pairs when pairing by
        timestamp: each test frame with the source frame that libvmaf pairs
        it with by timestamp (frame_sync_by_time). The comparison is the
        frames both videos have (shortest=1), so it ends with the test video,
        or at its first frame after the source's end."""
        source_stream, distorted_stream = self.streams
        first_source = source_stream.next(self.cancel_event)
        first_test = distorted_stream.next(self.cancel_event)
        if _EOF in (first_source, first_test):
            for stream, slot in ((source_stream, first_source), (distorted_stream, first_test)):
                if slot != _EOF:
                    stream.release(slot)
            return
        test_base, source_base = distorted_stream.time_base, source_stream.time_base
        source_start = source_stream.pts[first_source]
        pairs = frame_sync_by_time.frame_pairs(
            self._timed_frames_of(1, first_test), self._timed_frames_of(0, first_source), test_base, source_base,
            lambda slot: self._drop(1, slot), lambda slot: self._drop(0, slot))
        try:
            for distorted_slot, source_slot, when in pairs:
                if source_slot is None:
                    raise VshipUnavailableError("The source has no frame at the test video's first.")
                if self.check_apart:
                    test_time = when * test_base
                    apart = test_time - (source_stream.pts[source_slot] - source_start) * source_base
                    if abs(apart) * Fraction(self.source.fps) > _APART_FRAMES:
                        raise _FramesApartError(
                            f"the test frame at {float(test_time):.3f} s is {float(apart) * 1000:.1f} ms from "
                            f"the nearest of the source frames sampled with it (one in {self.step})")
                yield distorted_slot, source_slot
        finally:
            pairs.close()

    def _feed_pairs(self) -> tuple[int, _PassRate]:
        """Hands every pair to each metric's lanes; returns how many pairs,
        and the pass's rate."""
        frame = 0
        rate = _PassRate(self.step)
        pairs = self._pairs()
        try:
            for distorted_slot, source_slot in pairs:
                errors = self._lane_errors()
                if len(errors) == len(self.specs):
                    # Nothing left to score: stop now rather than decode
                    # the rest of the video (a film took ~30 minutes to
                    # report a CVVDP handler that had failed to start).
                    raise next(iter(errors.values()))
                # No abort here: a failed lane keeps handing its frames back,
                # so the ring never fills with slots that nobody will give
                # back, and the check above ends the pass once no metric is
                # left.
                with self.slots_lock:
                    self.pending[frame] = [len(self.specs), source_slot, distorted_slot]
                    self.held[0][source_slot] += 1
                    self.held[1][distorted_slot] += 1
                for spec in self.frame_specs:
                    self.scores[spec.key].reserve(frame)
                for spec in self.frame_specs:
                    self.lanes_by_metric[spec.key][frame % _LANES_PER_METRIC].jobs.put(
                        (frame, source_slot, distorted_slot))
                if self.cvvdp_lane is not None:
                    self.cvvdp_lane.jobs.put((frame, source_slot, distorted_slot))
                frame += 1
                if self.on_progress:
                    self.on_progress(min(frame * self.step, self.total_units), self.total_units,
                                     rate.frames_per_second(frame))
        finally:
            pairs.close()  # hands back the frames the pairing holds
        return frame, rate

    # ------------------------------------------------------------------ results

    def _finish(self, frames: int, rate: _PassRate, started: float) -> PerceptualTaskOutput:
        """Checks the pass and returns its scores."""
        step = self.step
        if frames == 0:
            raise VshipUnavailableError("FFmpeg produced no frame pairs for Vship.")
        if (short := short_comparison(self.expected_frames, frames * step, self.source.fps, step)) is not None:
            raise ComparisonCutShortError(short)
        # A metric whose handler failed (out of GPU memory on a smaller card,
        # say) is reported on its own, and the others keep their scores.
        errors = self._lane_errors()
        if len(errors) == len(self.specs):
            raise next(iter(errors.values()))
        metric_failures = {key: str(error) for key, error in errors.items()}
        results = self._results(frames, metric_failures)

        elapsed = max(time.perf_counter() - started, 1e-6)
        _log.info("Vship pass (%s) on %s: %d frame pairs in %.1f s, %.1f pairs/s",
                  self._labels(), self.device.name, frames, elapsed, frames / elapsed)
        for key, error in errors.items():
            _log.error("Vship %s failed in its pass: %s", metric_definition(key).label, error, exc_info=error)
        if self.on_progress:
            # A pass too short to time from its first frame is timed over its
            # whole length instead.
            self.on_progress(frames * step, frames * step,
                             rate.frames_per_second(frames) or frames * step / elapsed)
        return PerceptualTaskOutput(results, self.videos[0].crop, self.videos[1].crop, frames * step, metric_failures)

    def _results(self, frames: int, metric_failures: dict[str, str]) -> MetricResultSet:
        """Each metric's scores that did not fail: per frame, and CVVDP's per
        second of video."""
        device, source = self.device, self.source
        frame_numbers = np.arange(frames, dtype=np.int32) * self.step
        times = frame_numbers.astype(np.float64) / max(source.fps, 1.0)
        results = MetricResultSet()
        parameters = {
            "gpu_backend": device.backend,
            "gpu_name": device.name,
            "input": "ffmpeg frames (hardware-decoded NV12/P010 split into planes); "
                     "native range/transfer and primaries",
            "coverage_step": self.step,
            "butteraugli_norm": "3-norm",
            # How the videos' colour tags were read (_vship_colorspace). The
            # cache recalculates GPU scores from v1.2, which have none.
            "color_tags": VSHIP_COLOR_TAGS,
        }
        for spec in self.frame_specs:
            if spec.key in metric_failures:
                continue
            results.add(FrameMetricResult(
                spec.key, frame_numbers, times,
                self.scores[spec.key].values(frames),
                MetricProvenance(
                    implementation=f"Vship/{spec.key}",
                    implementation_version=f"Vship {device.version}",
                    compute_backend="gpu",
                    implementation_compatibility_id=f"{spec.key}-vship-gpu-v1",
                    parameters=parameters,
                ),
            ))
        if self.cvvdp_lane is not None and self.cvvdp_lane.error is None and self.cvvdp_spec is not None:
            windows = self.cvvdp_lane.windows
            first = np.array([start for start, _count, _jod in windows], dtype=np.int32)
            cvvdp_parameters = dict(self.cvvdp_spec.parameters)
            results.add(SequenceMetricResult(
                "cvvdp", pool_cvvdp_windows(windows),
                MetricProvenance(
                    implementation="Vship/cvvdp",
                    implementation_version=f"Vship {device.version}",
                    compute_backend="gpu",
                    implementation_compatibility_id=self.cvvdp_spec.implementation_compatibility_id,
                    parameters={
                        "gpu_backend": device.backend, "gpu_name": device.name,
                        "display": cvvdp_parameters["display"],
                        "resize_to_display": cvvdp_parameters["resize_to_display"],
                        "timeline": "JOD of each second of video",
                        "color_tags": VSHIP_COLOR_TAGS,
                    },
                ),
                frame=first, time=first.astype(np.float64) / max(source.fps, 1.0),
                values=[jod for _start, _count, jod in windows],
            ))
        return results


def apply_vship_cpu_fallback(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    on_pass_done: Callable[[PerceptualTaskOutput], None] | None = None,
    together: bool = False,
    on_pass: Callable[[int, int, tuple[str, ...]], None] | None = None,
    on_cpu: Callable[[tuple[str, ...]], None] | None = None,
    concurrent_cpu_tasks: int = 1,
) -> PerceptualTaskOutput:
    """Calculates SSIMULACRA2, Butteraugli and CVVDP each where it is set to
    run (GPU or CPU), and moves a metric from the GPU to the CPU when the
    GPU cannot calculate it. `together` scores the GPU metrics in one pass
    (see run_vship_task), and `on_pass` is called as each GPU pass starts.
    `concurrent_cpu_tasks` is how many tasks share the CPU while this one
    runs, which sets the CPU metrics' share (perceptual_cpu.scoring_workers).

    `on_cpu` gets the keys of the metrics the CPU is about to calculate,
    just before it starts on them: those chosen for the CPU, and those meant
    for the GPU and handed over (no usable GPU, a failed GPU pass, or a GPU
    this build of Vship scores wrongly on). The progress reported after it
    is the CPU's own, from 0, and the run line shows those metrics as CPU
    metrics, with the CPU's figures.

    CVVDP runs on the GPU only. Without a usable GPU it fails on its own
    (reported in the output's `failures`), and the other metrics are scored
    as they would have been without it.
    """
    routing = _MetricRouting(
        source, distorted, request, specs, on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
        process_handle=process_handle, on_pass_done=on_pass_done, together=together, on_pass=on_pass, on_cpu=on_cpu,
        concurrent_cpu_tasks=concurrent_cpu_tasks,
    )
    return routing.run()


class _MetricRouting:
    """Calculates a task's SSIMULACRA2, Butteraugli and CVVDP each where it
    is set to run, and moves a metric from the GPU to the CPU when the GPU
    cannot calculate it (see apply_vship_cpu_fallback, which makes one)."""

    def __init__(
        self, source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
        specs: tuple[MetricRequestSpec, ...], *,
        on_progress: Callable[[int, int, float], None] | None,
        on_status: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
        process_handle: ProcessHandle | None,
        on_pass_done: Callable[[PerceptualTaskOutput], None] | None,
        together: bool,
        on_pass: Callable[[int, int, tuple[str, ...]], None] | None,
        on_cpu: Callable[[tuple[str, ...]], None] | None,
        concurrent_cpu_tasks: int,
    ) -> None:
        """Sorts the metrics by where they are set to run."""
        self.source, self.distorted, self.request, self.specs = source, distorted, request, specs
        self.on_progress, self.on_status, self.cancel_event = on_progress, on_status, cancel_event
        self.process_handle, self.on_pass_done, self.together = process_handle, on_pass_done, together
        self.on_pass, self.on_cpu, self.concurrent_cpu_tasks = on_pass, on_cpu, concurrent_cpu_tasks
        self.cpu_specs = tuple(spec for spec in specs if self._backend(spec) == "cpu")
        self.gpu_specs = tuple(spec for spec in specs if self._backend(spec) == "gpu")
        if len(self.cpu_specs) + len(self.gpu_specs) != len(specs):
            raise ValueError("perceptual metric backend must be either GPU or CPU")
        self.gpu_only = tuple(spec for spec in self.gpu_specs if spec.key in GPU_ONLY_METRICS)
        # What may be retried on the CPU if the GPU cannot be used.
        self.retryable = tuple(spec for spec in specs if spec.key not in GPU_ONLY_METRICS)

    def _backend(self, spec: MetricRequestSpec) -> str:
        """Where a metric is set to run: "gpu" or "cpu"."""
        return "gpu" if spec.key in GPU_ONLY_METRICS else self.request.execution.perceptual_backend(spec.key)

    def run(self) -> PerceptualTaskOutput:
        """Calculates the metrics (see apply_vship_cpu_fallback)."""
        # An explicit CPU selection must not probe Vship or touch a compute GPU.
        if not self.gpu_specs:
            return self._on_cpu(self.cpu_specs)

        forget_failed_vship_probe()
        device, reason = detect_vship_device()
        if device is None:
            return self._without_gpu(reason)

        self._move_wrongly_scored(device)
        if not self.gpu_specs:
            return self._on_cpu(self.cpu_specs)

        from vmaf_app.core.perceptual_cpu import _resolve_crops

        crops = _resolve_crops(
            self.source, self.distorted, self.request.recipe, self.cancel_event, self.process_handle,
            self.on_status,
        )
        try:
            gpu_output = run_vship_task(
                self.source, self.distorted, self.request, self.gpu_specs, device, *crops,
                on_progress=self.on_progress, on_status=self.on_status, cancel_event=self.cancel_event,
                process_handle=self.process_handle, on_pass_done=self.on_pass_done, together=self.together,
                on_pass=self.on_pass,
            )
        except (PerceptualCancelled, ComparisonCutShortError):
            # Don't retry a video cut short (one that ends early) on the CPU: it
            # would stop as short, after hours instead of the GPU's minutes.
            raise
        except Exception as error:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user") from error
            return self._after_gpu_failure(error, crops)
        return self._after_gpu(gpu_output, crops)

    # ------------------------------------------------------------------ the CPU

    def _run_on_cpu(self, specs: tuple[MetricRequestSpec, ...],
                    crops: tuple[CropBox | None, CropBox | None] | None = None) -> PerceptualTaskOutput:
        """perceptual_cpu.run_perceptual_task for `specs`, with this task's
        CPU share, and the black bars already found if there are `crops`."""
        from vmaf_app.core import perceptual_cpu

        extra = {} if crops is None else {"resolved_crops": crops}
        return perceptual_cpu.run_perceptual_task(
            self.source, self.distorted, self.request, specs,
            on_progress=self.on_progress, on_status=self.on_status,
            cancel_event=self.cancel_event, process_handle=self.process_handle,
            concurrent_tasks=self.concurrent_cpu_tasks, **extra,
        )

    def _announce_cpu(self, handed: tuple[MetricRequestSpec, ...]) -> None:
        """Reports the metrics the CPU is about to calculate (on_cpu)."""
        if self.on_cpu is not None and handed:
            self.on_cpu(tuple(spec.key for spec in handed))

    def _on_cpu(self, specs: tuple[MetricRequestSpec, ...], failures: dict[str, str] | None = None,
                crops: tuple[CropBox | None, CropBox | None] | None = None) -> PerceptualTaskOutput:
        """Calculates `specs` on the CPU, with `failures` added to the output."""
        self._announce_cpu(specs)
        output = self._run_on_cpu(specs, crops)
        return replace(output, failures={**output.failures, **failures}) if failures else output

    # ------------------------------------------------------------------ when the GPU cannot

    def _gpu_only_failed(self, reason: str, error: BaseException | None = None, *,
                         no_gpu: bool = True, reasons: dict[str, str] | None = None) -> dict[str, str]:
        """Returns the failures of the GPU-only metrics, or raises them if
        nothing else was asked for. `no_gpu` means no usable GPU was found.
        Otherwise the GPU pass itself failed, possibly for a reason that has
        nothing to do with the GPU (the two videos' frame rates differing,
        say), so the message does not blame the GPU, which would send people
        looking for the wrong problem."""
        def message(spec: MetricRequestSpec) -> str:
            label = metric_definition(spec.key).label
            own = (reasons or {}).get(spec.key, reason)
            return (f"{label} needs a GPU that Vship can use, and could not use one: {own}"
                    if no_gpu else f"{label} could not be calculated: {own}")

        messages = {spec.key: message(spec) for spec in self.gpu_only}
        if not self.retryable:
            raise PerceptualRunError("\n".join(messages.values())) from error
        return messages

    def _without_gpu(self, reason: str) -> PerceptualTaskOutput:
        """Calculates on the CPU what can be, when no usable GPU was found."""
        failures = self._gpu_only_failed(reason) if self.gpu_only else {}
        if self.on_status:
            self.on_status(f"Vship GPU unavailable ({reason}); using CPU reference metrics…")
        return self._on_cpu(self.retryable, failures)

    def _move_wrongly_scored(self, device: VshipDevice) -> None:
        """Moves to the CPU each metric this build of Vship scores wrongly on
        `device`, planned like a CPU choice. As on a PC with no GPU for it,
        the Videos tab asks before a long CPU run
        (MainWindow._confirm_long_cpu_perceptual)."""
        wrong = tuple(spec for spec in self.gpu_specs if not scores_correctly(device, spec.key))
        if not wrong:
            return
        _log.info("%s on the CPU: Vship's %s build scores it wrongly on %s",
                  " and ".join(metric_definition(spec.key).label for spec in wrong), backend_label(device.backend),
                  device.name)
        self.gpu_specs = tuple(spec for spec in self.gpu_specs if spec not in wrong)
        self.cpu_specs = self.cpu_specs + wrong

    def _too_long_for_cpu(self) -> bool:
        """Whether the comparison is too long to move to the CPU unasked."""
        return compared_seconds(self.source, self.distorted, self.request.recipe.duration_limit) > LONG_CPU_RUN_SECONDS

    def _after_gpu_failure(self, error: Exception,
                           crops: tuple[CropBox | None, CropBox | None]) -> PerceptualTaskOutput:
        """Calculates every metric that can be on the CPU after the GPU task
        failed, unless the video is too long to do that unasked."""
        failures = (self._gpu_only_failed(str(error), error, no_gpu=False,
                                          reasons=getattr(error, "failures", None)) if self.gpu_only else {})
        gpu_retry = tuple(spec for spec in self.gpu_specs if spec.key not in GPU_ONLY_METRICS)
        if gpu_retry and self._too_long_for_cpu():
            # Nobody agreed to a CPU run of this length: the Videos tab asks
            # before one, but a GPU failure during a run cannot. Falling back
            # without asking would mean days of CPU work and terabytes of
            # temporary pictures for a film. So the metric fails, with the
            # reason, and the video keeps its other metrics.
            labels = " and ".join(metric_definition(spec.key).label for spec in gpu_retry)
            # CVVDP's own failure goes into the same message, because the
            # advice about the CPU does not apply to a GPU-only metric.
            raise PerceptualRunError(
                f"GPU scoring failed ({error}). It was not retried on the CPU, which would take "
                "hours to days for a video over 10 minutes. "
                f"Choose CPU for {labels} to calculate it on the CPU anyway."
                + "".join(f"\n{message}" for message in dict.fromkeys(failures.values()))
            ) from error
        if self.on_status:
            self.on_status(f"Vship GPU compute failed ({error}); using CPU reference metrics…")
        # After a GPU failure, all metrics run together on the CPU, so that
        # the frames are extracted only once and the result stays whole.
        return self._on_cpu(self.retryable, failures, crops)

    # ------------------------------------------------------------------ after the GPU task

    def _after_gpu(self, gpu_output: PerceptualTaskOutput,
                   crops: tuple[CropBox | None, CropBox | None]) -> PerceptualTaskOutput:
        """Calculates the CPU's metrics once the GPU task is done, with any
        metric that failed on the GPU, and combines the two."""
        # A metric that failed on the GPU while the others finished is retried
        # on the CPU on the same terms as a failed pass: only for videos up to
        # ten minutes, which nobody has to agree to.
        failures = dict(gpu_output.failures)
        gpu_failed = tuple(spec for spec in self.gpu_specs
                           if spec.key in failures and spec.key not in GPU_ONLY_METRICS)
        if gpu_failed and self._too_long_for_cpu():
            for spec in gpu_failed:
                failures[spec.key] = (
                    f"GPU scoring failed ({failures[spec.key]}). It was not retried on the CPU, which would "
                    "take hours to days for a video over 10 minutes. Choose CPU for "
                    f"{metric_definition(spec.key).label} to calculate it on the CPU anyway."
                )
            gpu_failed = ()
        # Kept for the message if the CPU retry fails too: the GPU failure is
        # the first cause, and without it a GPU user would be sent to fix a CPU
        # tool.
        gpu_reasons = {spec.key: failures.pop(spec.key) for spec in gpu_failed}
        cpu_specs = self.cpu_specs + gpu_failed
        if not cpu_specs:
            return replace(gpu_output, failures=failures) if failures != gpu_output.failures else gpu_output
        self._announce_cpu(cpu_specs)

        if self.on_status:
            if gpu_failed:
                labels = " and ".join(metric_definition(spec.key).label for spec in gpu_failed)
                self.on_status(f"{labels} failed on the GPU; calculating it on the CPU…")
            else:
                self.on_status("Calculating selected perceptual metric(s) on CPU…")

        def failed_on_cpu(key: str, reason: str) -> str:
            """A CPU failure's message, with the GPU failure first if there was one."""
            if key in gpu_reasons:
                return f"GPU scoring failed ({gpu_reasons[key]}); the CPU retry failed too: {reason}"
            return reason

        # The GPU pass has finished, so a CPU failure from here on fails only the
        # CPU metrics, and does not throw away what the GPU scored (a finished
        # CVVDP pass lost to a missing libjxl tool, say). The progress from here
        # is a stage of its own, from 0 (on_cpu).
        try:
            cpu_output = self._run_on_cpu(cpu_specs, crops)
        except PerceptualCancelled:
            raise
        except Exception as error:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user") from error
            if not gpu_output.metrics:
                if gpu_reasons:
                    raise PerceptualRunError(failed_on_cpu(cpu_specs[-1].key, str(error))) from error
                raise
            failures.update({spec.key: failed_on_cpu(spec.key, str(error)) for spec in cpu_specs})
            return replace(gpu_output, failures=failures)
        return self._combined(gpu_output, cpu_output, failures)

    def _combined(self, gpu_output: PerceptualTaskOutput, cpu_output: PerceptualTaskOutput,
                  failures: dict[str, str]) -> PerceptualTaskOutput:
        """The GPU's and the CPU's scores as one output.

        Each metric keeps its own frames. Vship and the CPU tools can end a
        frame apart (each stops where the shorter video's pictures do), and
        the cache and the window take each metric's own frames, so a
        difference must not throw the CPU's finished scores away."""
        combined = MetricResultSet()
        for spec in self.specs:
            if spec.key in failures:
                continue
            value = gpu_output.metrics.get(spec.key) or cpu_output.metrics.get(spec.key)
            if value is None:
                raise PerceptualRunError(f"Selected backend did not produce {spec.key}.")
            combined.add(value)
        return PerceptualTaskOutput(
            combined, gpu_output.source_crop, gpu_output.distorted_crop,
            max(gpu_output.compared_frame_count, cpu_output.compared_frame_count), failures,
        )


