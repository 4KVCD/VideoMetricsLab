"""GPU implementation of the perceptual metrics via Vship's C API.

The API is Vship 5.1's (Vship_InitHandler / Vship_ComputeHandler, see
https://codeberg.org/Line-fr/Vship/src/tag/v5.1.1/src/VshipAPI.h); the older
per-metric functions are a deprecated compatibility layer in 5.1.

SSIMULACRA2 and Butteraugli score each frame pair on its own; CVVDP is
temporal and scores the video (see _CvvdpLane).

FFmpeg decodes, crops, samples and scales -- the same FFmpeg as the rest of
the app, so every codec it reads works, VVC included, with hardware decode
per input where the GPU has one. It streams tightly packed frames into rings
of pinned host buffers (see _FrameStream). A video the GPU's decoder
decodes -- NVIDIA's with Vship's CUDA build, Intel's or AMD's with any -- is
decoded in the scoring process instead (gpu_frames, _NativeFrameStream):
FFmpeg only copies its packets out of the container, and each picture goes
into the ring without FFmpeg's CPU copies and the pipe -- the same pictures.
Vship converts each
frame from the colorspace it is described in (_vship_colorspace) and
computes the metric on
the GPU -- through Vship's CUDA build on NVIDIA, its HIP build on AMD, or
its Vulkan build, which runs on any GPU with a Vulkan driver (see
VSHIP_BUILDS). Vship takes a frame as three planes, so a hardware-decoded
frame
crossing the pipe in the decoder's NV12/P010 layout has only its chroma
split into planes here (see _passthrough_format). No FFVship or FFMS2
executable is bundled.
"""
from __future__ import annotations

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
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from vmaf_app.core import gpu_frames
from vmaf_app.core import proc as proc_util
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.colour import UnsupportedColourError, video_colour
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.cvvdp import VSHIP_MODEL_KEY, CvvdpSettings, vship_display_json
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, ffmpeg_path
from vmaf_app.core.frame_coverage import short_comparison
from vmaf_app.core.frame_sync import frame_pairs
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
    """The GPU implementation could not be loaded or run; CPU fallback is safe."""


class VshipPassesFailedError(VshipUnavailableError):
    """Every metric's GPU pass failed. Raised with the first error's text,
    and `failures` holds each metric's own reason, so a metric is never
    reported with another's (CVVDP given SSIMULACRA2's out-of-memory)."""

    def __init__(self, message: str, failures: dict[str, str]) -> None:
        super().__init__(message)
        self.failures = dict(failures)


class _Subsampling(ctypes.Structure):
    _fields_ = [("subw", ctypes.c_int), ("subh", ctypes.c_int)]


class _Crop(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ("top", "bottom", "left", "right")]


class _Colorspace(ctypes.Structure):
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
    _fields_ = [
        ("major", ctypes.c_int), ("minor", ctypes.c_int),
        ("minorMinor", ctypes.c_int), ("backend", ctypes.c_int),
    ]


class _DeviceInfo(ctypes.Structure):
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
#: (yuv410p) has one chroma sample per 4x4 block, (2, 2). It was (2, 1), as
#: in FFVship 5.1.1 (Vship issue 21): Vship then read chroma planes twice the
#: height FFmpeg wrote, and the run stopped partway through the first frame.
_SUBSAMPLING = {"410": (2, 2), "411": (2, 0), "420": (1, 1), "422": (1, 0), "440": (0, 1), "444": (0, 0)}
#: The first Vship that reads 4:1:0 (Vship issue 21). Before it, subsampling
#: (2, 2) gave nonsense on the same frames -- SSIMULACRA2 -52477, Butteraugli
#: not finite -- so for an older build FFmpeg upsamples 4:1:0 to 4:4:4.
_READS_410_SINCE = (5, 1, 2)


@dataclass(frozen=True, slots=True)
class _ImageFormat:
    """What FFmpeg pipes (pixel_format) and how Vship reads it. `family` is
    this module's: 1 for RGB, planes all full size. Vship 5.1 itself tells
    RGB from YUV by the matrix (0 is RGB); its colorFamily field is unused."""

    pixel_format: str
    family: int
    sample: int
    subw: int
    subh: int
    full_range: bool = False  # yuvj: full range when the stream has no range tag
    # FFmpeg's planar RGB formats are ordered G, B, R; Vship expects R, G, B.
    plane_order: tuple[int, int, int] = (0, 1, 2)

    def frame_layout(self, width: int, height: int) -> tuple[int, tuple[int, int, int], _I64_3, tuple[int, int, int]]:
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
    backend: str
    path: Path
    library: ctypes.CDLL
    dll_directory: object


@dataclass(frozen=True, slots=True)
class VshipDevice:
    backend: str  # "vulkan", "cuda" or "hip": the Vship build it runs on (VSHIP_BUILDS)
    name: str
    gpu_id: int
    version: str
    #: The library, where it was loaded in this process. None in the app
    #: itself: Vship is probed and run in a process of its own (see
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
    return all(hasattr(lib, name) for name in _API_FUNCTIONS)


def _configure_api(lib: ctypes.CDLL) -> None:
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
    """Why a call on `handle` failed: the handler's own last error -- other
    handlers on other threads may have failed since -- or the code's text."""
    detail = ctypes.create_string_buffer(1024)
    with contextlib.suppress(Exception):
        lib.Vship_GetDetailedLastErrorHandler(handle, detail, len(detail))
        if detail.value:
            return detail.value.decode("utf-8", errors="replace").strip()
    return _message(lib, code)


def _message(lib: ctypes.CDLL, code: int | None = None) -> str:
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
#: HIP on AMD, and Vulkan on any GPU with a Vulkan driver -- NVIDIA, AMD and
#: Intel.
VSHIP_BUILDS: dict[str, str] = {"cuda": "nvidia", "hip": "amd", "vulkan": "vulkan"}
#: Settings > GPU metrics > backend: "auto", or the build to try first. Auto
#: tries CUDA, then HIP, then Vulkan: the fastest build whose scores agree
#: with the reference on each GPU (CUDA is faster than Vulkan on NVIDIA for
#: SSIMULACRA2 and CVVDP), and Vulkan for a GPU neither of the others can use.
#: Whatever is chosen, the other builds follow if it cannot run.
VSHIP_BACKENDS = ("auto", "vulkan", "cuda", "hip")
DEFAULT_VSHIP_BACKEND = "auto"
_AUTO_ORDER = ("cuda", "hip", "vulkan")
_BACKEND_LABELS = {"vulkan": "Vulkan", "cuda": "CUDA", "hip": "HIP"}
#: (build, GPU maker, metric) a Vship build scores wrongly: the metric is
#: calculated on the CPU instead, as on a PC without a usable GPU. A GPU whose
#: maker could not be told counts as every maker. Add an entry only for a
#: bundled build measured to disagree with CUDA and libjxl on the same frames.
#:
#: Empty since the bundled Vulkan build is Vship 5.1.2 (its fix is commit 97d0dc5).
#: Before it, Vship's Vulkan build (5.1.1, and commit 0732ed3) scored
#: SSIMULACRA2 far too high on NVIDIA GPUs -- HoneyBee 4K against a CRF 22
#: HEVC encode: libjxl 47.4, CUDA 45.5, Vulkan 62.9; +7 at 1080p, +1 at 360p
#: (Vship issue 18) -- because NVIDIA's driver miscompiles a small
#: two-dimensional array in its blur, which 5.1.2 flattens. An RTX 5090 now
#: scores 45.5019 there against CUDA's 45.5020, and Intel's scores are
#: unchanged. Butteraugli and CVVDP always agreed. AMD's Vulkan is
#: unmeasured.
SCORED_WRONGLY: frozenset[tuple[str, GpuVendor, str]] = frozenset()
#: (build, GPU maker, metric) -> the largest frame it is given, in pixels,
#: and the GPUs and sizes that is, for the reason the video shows: above it
#: the metric is not calculated. Vship 5.1.2's Vulkan CVVDP hangs Intel's
#: GPU above 1080p. On a Core Ultra 9 285K's (driver 32.0.101.8860), 4K
#: HEVC stopped after 3 to 6
#: frames, with one handler or two: Windows reset the GPU (LiveKernelEvent
#: 141; the screen goes black for a moment) and the pass failed with "Failed
#: to Submit commandBuffer to queue". At 1920x1080 all 240 frames were
#: scored, 13 a second. Sizes in between are untested, so they are refused
#: too. Not reported to Vship yet.
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
    """ "auto" or the Vship build to try first (Settings > GPU metrics); an
    unknown name is "auto". A probe made for another choice is forgotten, so
    the next one -- start_vship_probe, or the next GPU run -- uses this."""
    global _backend, _probed
    backend = backend if backend in VSHIP_BACKENDS else DEFAULT_VSHIP_BACKEND
    with _PROBE_LOCK:
        if backend != _backend:
            _backend = backend
            _probed = None


def vship_backend() -> str:
    return _backend


def backend_label(backend: str) -> str:
    """ "Vulkan", "CUDA" or "HIP"."""
    return _BACKEND_LABELS.get(backend, backend)


def scores_correctly(device: VshipDevice, key: str) -> bool:
    """Whether `device`'s Vship build scores `key` right on its GPU
    (SCORED_WRONGLY); a GPU of unknown make is assumed to be affected."""
    return not any(device.backend == backend and key == metric and device.vendor in (vendor, None)
                   for backend, vendor, metric in SCORED_WRONGLY)


def size_refusal(device: VshipDevice, key: str, size: tuple[int, int]) -> str | None:
    """Why `device` does not calculate `key` on frames of `size`
    (GPU_PIXEL_LIMITS), or None where it does."""
    limit = GPU_PIXEL_LIMITS.get((device.backend, device.vendor, key))
    if limit is None or size[0] * size[1] <= limit[0]:
        return None
    return (f"on {limit[1]} it is not calculated, as Vship's Vulkan build hangs the GPU "
            f"(this comparison is {size[0]}x{size[1]})")


def gpu_can_score(key: str) -> bool:
    """Whether SSIMULACRA2 or Butteraugli set to GPU is calculated on the GPU
    here: a GPU Vship can use, whose build scores it right. Probes once."""
    device, _reason = detect_vship_device()
    return device is not None and scores_correctly(device, key)


def detect_vship_device() -> tuple[VshipDevice | None, str]:
    """Return a fully verified GPU for Vship, or a user-readable reason.

    Probed once and cached; see start_vship_probe and forget_failed_vship_probe."""
    global _probed
    with _PROBE_LOCK:
        if _probed is None:
            _probed = (_probe_isolated(), time.monotonic())
            device, reason = _probed[0]
            if device is not None:
                _log.info("Vship %s (%s) GPU: %s", device.version, backend_label(device.backend), device.name)
            else:
                _log.warning("Vship GPU unavailable: %s", reason)
        return _probed[0]


def start_vship_probe() -> None:
    """Probe in the background, at startup. The first probe loads the Vship
    library and initializes the GPU driver, and ran on the UI thread when the
    first video was added, freezing the window for that long."""
    threading.Thread(target=detect_vship_device, name="vship-probe", daemon=True).start()


def forget_failed_vship_probe() -> None:
    """Before a GPU run, off the UI thread: redo a probe that failed a while
    ago. The failure may have been passing -- a driver being updated or
    restarted, a GPU briefly unavailable -- and was kept for the whole
    session, running every GPU metric on the CPU until the app restarted."""
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
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


def _vulkan_unavailable() -> str | None:
    """Why no GPU is usable through Vulkan here, or None if one is.

    Asked of the Vulkan loader itself before Vship's Vulkan build is
    loaded. Where the loader is installed but no driver answers -- a virtual
    machine, GitHub's test runner -- Vship's Vulkan builds before 5.1.2
    failed their DLL initialisation (WinError 1114), and the process then
    crashed with an access violation as it exited. The bundled build starts
    Vulkan on first use and loads there (tests/test_perceptual_vship.py); the
    loader is still asked first, as it says why without loading Vship."""
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
    """Who made the Vulkan GPU called `name`, from the Vulkan loader: Vship
    names a GPU by its VkPhysicalDeviceProperties.deviceName. Asked of the
    loader because names do not always say (NVIDIA's "Quadro" and "Tesla"
    cards). GpuVendor.NONE for another maker, None if no GPU has the name."""
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


#: The GPU maker each build runs on; Vulkan's is asked of the loader.
_BUILD_VENDORS = {"cuda": GpuVendor.NVIDIA, "hip": GpuVendor.AMD}


def _probe_vship_device() -> tuple[VshipDevice | None, str]:
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
                # Vulkan lists every GPU, an integrated one too, in the
                # driver's order: a discrete GPU comes first here.
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


def _probe_isolated() -> tuple[VshipDevice | None, str]:
    """_probe_vship_device in a process of its own. Loading Vship starts the
    GPU driver, and a crash there took the app down: Vship 5.1.1's Vulkan
    build on a PC whose only GPU is Intel's crashed it as it closed. Now it
    ends that process, and the GPU metrics are calculated on the CPU."""
    try:
        return run_isolated(_probe_in_own_process, _backend, what="Vship's GPU probe")
    except IsolatedCrashError as error:
        return None, str(error)


def _probe_in_own_process(choice: str) -> tuple[VshipDevice | None, str]:
    """Run by _probe_isolated: the device goes back without the library,
    which belongs to the process that loaded it."""
    global _backend
    _backend = choice  # this process starts with the default
    device, reason = _probe_vship_device()
    return (replace(device, loaded=None) if device is not None else None), reason


#: The builds a pass's process has loaded (see _library).
_loaded_builds: dict[str, _LoadedVship] = {}


def _library(device: VshipDevice) -> ctypes.CDLL:
    """The Vship library `device` runs on: the probe's own, or, in the
    process a pass runs in (which gets the device without it), loaded there
    once."""
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
    """Whether Vship `version` ("5.1.2") reads 4:1:0 (_READS_410_SINCE); an
    unknown version is taken not to."""
    try:
        return tuple(int(part) for part in (version or "").split(".")[:3]) >= _READS_410_SINCE
    except ValueError:
        return False


def _image_format(info: VideoInfo, version: str | None = None) -> _ImageFormat:
    """The planar layout FFmpeg pipes a decoded video in, and how Vship
    `version` reads it. FFmpeg converts to it exactly: a semi-planar or
    big-endian layout to the planar little-endian one, alpha dropped, a
    monochrome video given neutral chroma. 4:1:0 goes as it is to a Vship
    that reads it, upsampled to 4:4:4 for an older one (_READS_410_SINCE)."""
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
    # Monochrome (AV1 can be): the luma as it is, chroma neutral.
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
    """The planar layout Vship is given, and the layout FFmpeg pipes, when a
    hardware-decoded 4:2:0 frame can cross the pipe exactly as it downloads.

    NVDEC (and D3D11VA/QSV) hand back NV12 or P010: a luma plane, then U and
    V interleaved in one plane. Vship only takes separate planes, and having
    FFmpeg rearrange the whole frame cost 10-12 ms of CPU per 4K frame --
    about half of what feeding Vship cost. Piped as-is, the luma plane is
    read straight into pinned memory and only the chroma is split, which
    numpy does in under 1 ms (_FrameStream._fill).

    P010 keeps each 10-bit sample in the top bits of 16. Declared to Vship as
    16-bit, limited range, that is the same picture exactly: Vship brings
    limited-range samples to 8-bit scale by dividing by 2^(depth-8), so
    (v << 6) / 256 and v / 4 are the same float (FullRange in Vship 5.1.1's
    gpuColorToLinear/rangeToFull.hpp; the 16-bit read masks nothing off).
    Full range divides by 2^depth - 1 instead, where the two differ, so
    full-range 10-bit keeps FFmpeg's conversion. 8-bit NV12 is exact in
    either range.
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
    subw, subh = _SUBSAMPLING[sampling]
    suffix = "" if depth == 8 else f"{depth}le"
    return _ImageFormat(f"yuv{sampling}p{suffix}", 0, _VSHIP_ENUMS[depth], subw, subh)


def _vship_colorspace(info: VideoInfo, image: _ImageFormat, width: int, height: int) -> _Colorspace:
    """The frame's colorspace for Vship, from the stream's tags, read as
    colour.video_colour reads them for both implementations."""
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
) -> str:
    operations: list[str] = []
    if step > 1:
        # First: the frames it drops are not downloaded, cropped or scaled
        # (they were: at step 24, 23 of every 24). Nothing after it drops a
        # frame, so n counts the same frames wherever it stands.
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
    operations.extend(("setpts=PTS-STARTPTS", f"format={pixel_format}"))
    return ",".join(operations)


#: Concurrent Vship handlers per metric. One handler leaves the GPU idle
#: between its own kernels; two keep it busy. Measured at 4K 10-bit on an
#: RTX 5090 (Vship's CUDA build): SSIMULACRA2 217 -> 277 pairs/s,
#: Butteraugli 82 -> 95, both metrics together 61 -> 72. (FFVship runs 8
#: GPU streams by default, -g, and its documentation calls 3 usually
#: enough; each holds its own VRAM.)
_LANES_PER_METRIC = 2
#: Frames in flight per video: one per lane being scored, one waiting and
#: one being filled, so decode, transfer and GPU compute overlap instead of
#: taking turns. A 4K 10-bit frame is 25 MB of pinned memory, so this is
#: 250 MB for a 4K pair.
_RING_SLOTS = _LANES_PER_METRIC + 3
#: The pipe between FFmpeg and this process. Windows' default is a few
#: kilobytes, which caps a raw 4K stream near 55 fps no matter how fast the
#: decoder is; 64 MB doubles that. Measured on one 4K 10-bit HEVC stream:
#: 55 -> 107 fps alone, 46 -> 78 fps with the two inputs in parallel.
_PIPE_BYTES = proc_util.FRAME_PIPE_BYTES
_EOF = -1
#: What FFmpeg writes about each frame it pipes to Vship (-stats_enc_pre):
#: the picture's timestamp as it leaves the filter chain, in the chain's
#: time base (-enc_time_base filter: the stream's) -- the timestamp
#: libvmaf's frame sync is given at the end of the same chain. The frames
#: are paired by it, as libvmaf pairs them (frame_sync).
#:
#: It was the decoder's ({ptsi} {tbi}), which is the same number less the
#: first frame's where the file stores presentation times. A file that
#: stores none (H.264 with B-frames in AVI) has no decoder timestamp --
#: FFmpeg works its frames' times out after decoding -- and every GPU pass
#: on one failed with "FFmpeg gave a test video frame no timestamp".
_TIMESTAMP_FORMAT = "{pts} {tb}"
#: How long a frame's timestamp may be missing once the frame has arrived.
#: FFmpeg writes it, and flushes it, before the frame: it is there at once.
_TIMESTAMP_WAIT_SECONDS = 10.0
_NO_PTS = -(1 << 63)  # AV_NOPTS_VALUE


def _with_timestamps(command: list[str], path: Path) -> list[str]:
    """`command`, whose last argument is its output, also writing each piped
    frame's timestamp to `path` (_TIMESTAMP_FORMAT), a line per frame."""
    return [*command[:-1], "-enc_time_base", "filter", "-stats_enc_pre", str(path),
            "-stats_enc_pre_fmt", _TIMESTAMP_FORMAT, command[-1]]


class _Timestamps:
    """The timestamp lines one FFmpeg writes as it pipes frames
    (_with_timestamps), read a line per frame read. A frame's line is
    written before the frame, so it is in the file by the time the frame
    has been read; waiting is only for a line caught part-way."""

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
        """The pass's failure for a frame without a usable timestamp: one
        message for the window (it has a translation), `what` in the log."""
        _log.error("Vship: no timestamp for a %s frame: %s", self._label, what)
        return VshipUnavailableError(f"FFmpeg did not give the timestamp of a {self._label} frame.")

    def _parse(self, line: bytes) -> int:
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

    The start -- FFmpeg opening both inputs, the first 4K frames decoded,
    Vship's handlers set up -- takes a second or more that is not the rate
    the pass runs at. Counted in, it put "1.7 fps, 0:05:39 left" on the run
    line at the start of a pass over an 8-second clip. No rate until there
    is some time to measure over.

    The rate is in the video's frames, as the progress is: with frame
    subsampling each compared pair covers `step` frames. Pairs per second
    against frames left made a sampled pass's time left `step` times too
    long (the CPU tools already counted frames).
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


#: The planar layouts FFmpeg's YUV4MPEG muxer writes (-strict -1 for more
#: than 8 bits); 4:1:0, 4:4:0, planar RGB and the semi-planar layouts a GPU
#: decoder hands back (NV12, P010) go as raw video.
_Y4M_FORMAT = re.compile(r"yuv(411|420|422|444)p(?:(9|10|12|14|16)le)?")
_Y4M_FRAME = b"FRAME\n"


def _piped_as(pixel_format: str) -> list[str]:
    """FFmpeg's output arguments for piping frames of `pixel_format`.

    YUV4MPEG where the muxer takes the layout: the same planes as raw
    video, each after a "FRAME" line, which FFmpeg writes straight from its
    frames -- the rawvideo encoder first copies every frame into a packet
    it allocates. Measured on a 4K 10-bit HEVC video FFmpeg decodes: 23 ms
    of FFmpeg's CPU a frame against 35."""
    if _Y4M_FORMAT.fullmatch(pixel_format):
        return ["-pix_fmt", pixel_format, "-f", "yuv4mpegpipe", "-strict", "-1", "pipe:1"]
    return ["-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"]


def _is_y4m(command: list[str]) -> bool:
    return "yuv4mpegpipe" in command


def _gather(reader, count: int, process: subprocess.Popen, stopping: threading.Event) -> None:
    """Waits until `count` bytes of a frame are in the pipe (or all it can
    hold), FFmpeg has ended, or the stream is stopping. Read as it comes, a
    4K frame arrives in some 300 pieces, each a wake-up of the reading
    thread: 6 ms of CPU a frame against 4 taken in one read."""
    if os.name != "nt":
        return
    import msvcrt
    from ctypes import wintypes

    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(reader.fileno()))
    except (OSError, ValueError):
        return
    # Never for more than half of what the pipe holds: FFmpeg, its pipe
    # full, would wait for this side to read while this side waited for the
    # rest of the frame. (A pipe of Windows' default few kilobytes, as the
    # tests' stand-ins have, is read as it comes.)
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
    """Start FFmpeg writing raw frames to a large pipe; returns (process, reader)."""
    return proc_util.popen_piped(command, _PIPE_BYTES)


def _read_exact(reader, view: memoryview) -> int:
    """Fill `view` from the pipe with whole-buffer reads; returns bytes read.

    readinto straight into pinned memory: no intermediate bytes objects and
    no second copy. The previous transport read 1 MB chunks into new bytes
    objects and memmoved each one, which held a 4K stream to 6 fps.
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
    """With frame subsampling the source gives only its step-th frames, as
    the test video does: while the two videos' frames line up, those are the
    frames the test video's are paired with. A pair further apart than
    _APART_FRAMES shows they do not; the pass is then made again with every
    source frame (_score_vship_pass)."""


#: How far apart, in frames, a subsampled pair's two frames may be while the
#: source frame is still the nearest of all its frames: evenly spaced, any
#: other is at least three quarters of a frame away.
_APART_FRAMES = 0.25


class _FrameStream:
    """One input's FFmpeg decode, feeding a ring of pinned frame buffers.

    A background thread reads frames into free slots and hands them over in
    order; the scoring thread returns each slot once Vship is done with it.
    Both the pipe read and the Vship call release the GIL, so the two inputs
    and the GPU all make progress at once. If hardware decode fails before
    delivering a frame, the same pictures are decoded in software instead,
    and `on_software` is called first.
    """

    def __init__(
        self, lib: ctypes.CDLL, frame_bytes: int, commands: list[list[str]],
        process_handle: ProcessHandle | None, label: str,
        interleaved_chroma: tuple[int, int, type[np.integer]] | None = None,
        on_software: Callable[[], None] | None = None, gpu_id: int = 0,
    ) -> None:
        self.buffers = _pinned_ring(lib, frame_bytes, gpu_id)
        self.views = [memoryview(buffer.array).cast("B") for buffer in self.buffers]
        # (luma bytes, bytes of one chroma plane, sample type) when FFmpeg
        # pipes NV12/P010: the luma goes straight into the slot, the U/V pairs
        # into one staging buffer, and are then split into the slot's U and
        # V planes. None when the frame arrives already planar.
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
        #: Each slot's frame's timestamp, in time_base (known from the
        #: first frame): set before the slot is handed over.
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
        self._thread.start()

    def _run(self) -> None:
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
        descriptor, name = tempfile.mkstemp(prefix="vml-vship-pts-", suffix=".txt")
        os.close(descriptor)
        stamps_path = Path(name)
        try:
            return self._decode_with(_with_timestamps(command, stamps_path), stamps_path)
        finally:
            with contextlib.suppress(OSError):
                stamps_path.unlink()

    def _decode_with(self, command: list[str], stamps_path: Path) -> tuple[int, int, str]:
        try:
            process, reader = _spawn_raw_ffmpeg(command)
        except OSError as error:
            raise VshipUnavailableError(f"Could not start FFmpeg for Vship: {error}") from error
        stamps = _Timestamps(stamps_path, process, self._label)
        with self._lock:
            self._process, self._reader = process, reader
        if self._process_handle is not None:
            self._process_handle.attach(process.pid)
        stderr_tail: list[bytes] = []
        drain = threading.Thread(
            target=lambda: stderr_tail.append(process.stderr.read()[-2000:]) if process.stderr else None,
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
            stamps.close()
            drain.join(timeout=5)
            if self._process_handle is not None:
                self._process_handle.detach(process.pid)
        message = b"".join(stderr_tail).decode("utf-8", errors="replace").strip()
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
        """Read one frame into `slot`; returns the bytes read."""
        if self._split is None:
            return _read_exact(reader, self.views[slot])
        luma, staging, pairs, planes = self._split
        received = _read_exact(reader, self.views[slot][:luma])
        if received < luma:
            return received
        chroma = _read_exact(reader, staging)
        if chroma == len(staging):
            # Strided copies; numpy releases the GIL for them, so the other
            # input's reader and the scoring lanes keep going meanwhile.
            u, v = planes[slot]
            u[:] = pairs[:, 0]
            v[:] = pairs[:, 1]
        return received + chroma

    def next(self, cancel_event: threading.Event | None, abort: threading.Event | None = None) -> int:
        """The next filled slot, or _EOF. Raises the reader's error, on
        cancel, or with _LaneFailedError once `abort` is set.

        A lane that fails keeps the slots of the frames it was handed, so
        after a failure the ring can fill with slots nobody will release: the
        reader then waits for a free slot and no frame ever comes. Waiting on
        `abort` as well is what lets the pass end with the lane's error.
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
        self._free.put(slot)

    def close(self) -> None:
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
    """_RING_SLOTS pinned frame buffers, allocated one by one so a failure
    part-way frees what was already allocated: a list comprehension left
    those buffers -- pinned RAM, 25 MB each for 4K 10-bit -- allocated for
    the rest of the session, once per failed attempt."""
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
    """Which decoded pictures an input's FFmpeg chain pipes to Vship
    (_filter_chain and the command's -t): its select filter keeps every
    step-th picture, setpts starts them at 0, and the output's -t is FFmpeg's
    trim filter, which keeps a picture while its time is below the limit in
    the stream's time base (gpu_frames.duration_in) and ends the output at
    the first that is not."""

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
                self._limit = gpu_frames.duration_in(self._duration_limit, time_base)
        if self._limit is not None and pts - self._first >= self._limit:
            return None
        return True


class _NativeFrameStream:
    """One input decoded in this process by the GPU's decoder
    (gpu_frames.GpuFrameStream), feeding a ring of pinned frame buffers as
    _FrameStream does: each picture FFmpeg's chain would pipe
    (_FrameSelection) goes straight into a free slot -- copied there by the
    GPU from NVIDIA's decoder, by the CPU in one pass from Intel's or AMD's.
    A decoding failure after the start is raised from next() as
    GpuDecodeFailedError: the pass is then made again through FFmpeg."""

    def __init__(self, lib: ctypes.CDLL, frame_bytes: int, decoder: gpu_frames.GpuFrameStream, step: int,
                 duration_limit: str | None, label: str, gpu_id: int = 0) -> None:
        self.buffers = _pinned_ring(lib, frame_bytes, gpu_id)
        self._decoder = decoder
        self._selection = _FrameSelection(step, duration_limit)
        self.pts = [0] * _RING_SLOTS  # as _FrameStream's
        self.time_base: Fraction | None = None
        self._label = label
        self._free: queue.Queue[int] = queue.Queue()
        self._filled: queue.Queue[int | BaseException] = queue.Queue()
        for slot in range(_RING_SLOTS):
            self._free.put(slot)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"vship-gpu-decode-{label}", daemon=True)

    def start(self) -> None:
        self._decoder.start()
        self._thread.start()

    def _run(self) -> None:
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
                    taken = self._selection.take(pts, decoder.time_base)
                    if taken is None:
                        decoder.verify()  # the pictures up to here
                        break
                    if not taken:
                        continue
                    slot = self._free.get()
                    if slot == _EOF:
                        return
                    decoder.download(picture, self.buffers[slot].address.value)
                    self.pts[slot], self.time_base = pts, decoder.time_base
                    self._filled.put(slot)
                finally:
                    decoder.release(picture)
            self._filled.put(_EOF)
        except BaseException as error:  # handed to the scoring thread, never lost
            self._filled.put(error)

    next = _FrameStream.next

    def release(self, slot: int) -> None:
        self._free.put(slot)

    def close(self) -> None:
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


#: The GPU decoder that decodes in the scoring process what FFmpeg would
#: decode with each -hwaccel the app picks (gpu.pick_hwaccel): NVIDIA's
#: through nvcuvid, Intel's through oneVPL, AMD's through AMF.
_GPU_DECODERS = {"cuda": "nvidia", "qsv": "intel", "d3d11va": "amd"}


def _decoded_here(hwaccel: str | None, device: VshipDevice) -> str | None:
    """The decoder an input FFmpeg would decode with `hwaccel` is decoded by
    in the scoring process, or None: the GPU's, or, for an input FFmpeg would
    decode in software (`hwaccel` None), the software decoder. NVIDIA's copies
    each picture with the GPU into the ring, which must then be CUDA's
    page-locked memory (Vship's CUDA build); Intel's, AMD's and the software
    decoder hand pictures over in system memory, and the CPU copies them into
    any build's ring in one pass."""
    if hwaccel is None:
        return gpu_frames.SOFTWARE if gpu_frames.software_bundled() else None
    backend = _GPU_DECODERS.get(hwaccel)
    if backend == "nvidia" and device.backend != "cuda":
        return None
    return backend


def _native_decoder(info: VideoInfo, crop: CropBox | None, size: tuple[int, int], image_format: _ImageFormat,
                    frame_bytes: int, gpu_id: int, process_handle: ProcessHandle | None,
                    label: str, backend: str = "nvidia", algorithm: str = "bicubic") -> gpu_frames.GpuFrameStream | None:
    """The decoder for one input of a pass, when GPU decoder `backend` can
    give it in the layout Vship is told (8-bit planes, P016's 16-bit samples
    as they are, or shifted to 10-bit, as FFmpeg converts full-range 10-bit),
    scaled to `size` with `algorithm` where it is (by the GPU; by the CPU on
    Intel and AMD if their GPU scaling fails its check, gpu_frames.scale_note).
    None, with the reason logged, when FFmpeg decodes it."""
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
            # Asked first: a stream the decoder refuses once the pass has
            # started (10-bit H.264, say) makes the whole pass again.
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
    def __init__(self, lib: ctypes.CDLL, size: int, gpu_id: int) -> None:
        self.lib = lib
        self.gpu_id = gpu_id
        self.address = ctypes.c_void_p()
        error = lib.Vship_PinnedMalloc2(ctypes.byref(self.address), size, gpu_id)
        if error != 0 or not self.address.value:
            raise VshipUnavailableError(f"Could not allocate Vship pinned frame memory: {_message(lib, error)}")
        self.array = (ctypes.c_uint8 * size).from_address(self.address.value)

    def planes(self, layout: _ImageFormat, sizes: tuple[int, int, int]) -> _PLANES:
        ptrs: list[_U8P] = []
        offsets = (0, sizes[0], sizes[0] + sizes[1])
        base = self.address.value
        for plane in layout.plane_order:
            ptrs.append(ctypes.cast(base + offsets[plane], _U8P))
        return (_U8P * 3)(*ptrs)

    def close(self) -> None:
        if self.address.value:
            self.lib.Vship_PinnedFree2(self.address, self.gpu_id)
            self.address = ctypes.c_void_p()


class _ScoreArray:
    """Per-frame scores in packed float64 chunks, written by frame index.

    Lanes finish frames out of order, so scores are stored by position rather
    than appended. Chunks rather than one array because a feature-length
    video's frame count is only estimated up front, and growing one array
    would reallocate it under a lane that is writing to it; a new chunk is
    added by the dispatching thread before any lane can be handed an index
    in it, and existing chunks never move. 172,800 frames (two hours at 24
    fps) take 1.4 MB, where a dict of Python floats took about 17 MB.
    """

    _CHUNK = 8192

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []

    def reserve(self, index: int) -> None:
        """Make room for `index`; called before the index is dispatched."""
        while index >= len(self._chunks) * self._CHUNK:
            self._chunks.append(np.full(self._CHUNK, np.nan))

    def __setitem__(self, index: int, value: float) -> None:
        self._chunks[index // self._CHUNK][index % self._CHUNK] = value

    def values(self, count: int) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)[:count].astype(np.float32)


class _MetricLane:
    """One Vship handler for one metric, on its own thread.

    Frame pairs are dealt to lanes round-robin; each lane writes its score
    by frame index, so completion order does not matter. The pinned slots
    of a pair go back to their readers once every metric has scored it.
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
        #: Set when the handler failed; the lane then hands frames back
        #: unscored, like _CvvdpLane, so the other metrics can finish.
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"vship-{key}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
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
        self.jobs.put(None)

    def join(self) -> None:
        self._thread.join()


def _init_handler(device: VshipDevice, key: str, src: _Colorspace, dist: _Colorspace) -> _Handle:
    lib = _library(device)
    if key == "ssimulacra2":
        init = _InitSsimulacra2(_INIT_SSIMULACRA2, src, dist, device.gpu_id)
    else:
        # Vship's defaults (doc/BUTTERAUGLI.md): the 2-norm, beside the
        # 3-norm the app graphs, for a 203-nit display.
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
# Vship pools CVVDP's per-frame quality q the way ColorVideoVDP does: a
# running sum of q^2 over the n frames scored since the last reset, reported
# as JOD(sqrt(sum / n)) -- except for a single frame, reported as
# JOD(q * IMAGE_INT). Vship_ResetScore clears only that sum (the temporal
# filters keep their history), so resetting it at each second gives a JOD
# per second, and inverting each second's JOD back to its sum rebuilds the
# score of the whole video. On 2 s of the Beekeeper 4K AV1 encode the
# rebuilt score matched an unreset handler's to 6e-7 JOD; on 49 and 73
# frames at 23.976 fps, which end in a one-frame second, to 2e-7 and 3e-7
# (pooling that frame like any other would be off by 0.01). The constants are
# Vship's (and ColorVideoVDP's) jod_a, jod_exp and image_int: src/HIP/cvvdp/
# parameters.hpp and CVVDPComputingImplementation::run in Vship 5.1.1.
_JOD_A = 0.0439569391310215
_JOD_EXP = 0.9302042722702026
_IMAGE_INT = 0.577918291091919
_JOD_LINEAR_A = _JOD_A * 0.1 ** (_JOD_EXP - 1.0)


def _jod_from_quality(q: float) -> float:
    return 10.0 - (_JOD_A * q ** _JOD_EXP if q > 0.1 else _JOD_LINEAR_A * q)


def _quality_from_jod(jod: float) -> float:
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
    lib = _library(device)
    if not (math.isfinite(fps) and fps > 0):
        # CVVDP models how the eye integrates over time, so the frame rate
        # is part of the score; it used to be replaced by 1 fps silently.
        raise VshipUnavailableError(
            "CVVDP needs the video's frame rate, and none was found (it was read as "
            f"{fps:g} fps)."
        )
    # The display as JSON text: Vship parses a config that starts with "{"
    # itself (DisplayModel::parseJson), with no file to write and no path
    # for its narrow-character open to fail on -- a Windows user name
    # outside the ANSI code page broke the file this used to pass.
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
    lib = _library(device)
    error = lib.Vship_ResetScore(handler)
    if error != 0:
        raise VshipUnavailableError(f"Vship CVVDP failed: {_handler_error(lib, handler, error)}")


def _free_cvvdp(device: VshipDevice, handler: _Handle) -> None:
    _library(device).Vship_FreeHandler(handler)


class _CvvdpLane:
    """CVVDP's one Vship handler, fed every frame pair in order.

    CVVDP judges each frame together with the ones before it, so its frames
    cannot be dealt round-robin to two handlers like SSIMULACRA2's. The
    score is reset at each new second of video, giving the JOD of every
    second (the per-second curve); the overall JOD is pooled from those
    seconds (pool_cvvdp_windows).

    A CVVDP failure does not end the pass for SSIMULACRA2/Butteraugli: the
    lane keeps the error, keeps handing each frame back so its ring slots
    are freed, and the pass reports CVVDP as failed.
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
        self._thread.start()

    def _run(self) -> None:
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
        self.jobs.put(None)

    def join(self) -> None:
        self._thread.join()


#: One Vship pass at a time in the whole app. A pass already fills the GPU:
#: at 4K two passes at once scored no faster than one after the other (84.5
#: vs 83.6 pairs/s, Vship's CUDA build) while holding twice the VRAM (9.1 vs 4.5 GB, more than
#: most cards have). Running two jobs in parallel still pays off -- 23% at
#: 1080p, 27% at 4K -- because one job's VMAF/PSNR/SSIM pass, which is CPU
#: work, overlaps the other's Vship pass; only this GPU pass is serialized,
#: never crop detection or the FFmpeg metrics. Shared with VMAF on the GPU
#: (gpu.GPU_PASS).
_gpu_pass = GPU_PASS


def vship_passes(
    specs: tuple[MetricRequestSpec, ...], together: bool = False,
) -> list[tuple[MetricRequestSpec, ...]]:
    """The Vship passes `specs` are scored in, in order: one per metric, or
    with `together` one per set of frames scored -- every metric scoring
    the same frames in one pass, so the video is decoded once for them all.
    CVVDP scores every frame, so with frame subsampling it keeps a pass of
    its own."""
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
    starts: the window shows a score the moment it exists. `on_pass` gets
    each pass as it starts: its number, how many there are, and the keys of
    the metrics it scores -- the run line's figures are per metric.

    One metric at a time by default: each gets a pass of its own, and the
    video is decoded again for each. Scoring them in one pass holds every
    metric's GPU memory at once -- 7.3 GB for SSIMULACRA2, Butteraugli and
    CVVDP together on a full 3840x2160 frame (Vship's CUDA build; its Vulkan
    build reports no memory size to check against), more than an 8 GB card has
    free -- for little speed where the GPU decodes: 34 fps together against
    about 31.5 fps one after the other at 4K, and 91 against 90.5 fps at
    1080p (RTX 5090, Beekeeper AV1 and a synthetic SDR clip). One at a
    time, the peak is the largest single metric: SSIMULACRA2 2.7 GB,
    Butteraugli 4.4 GB, CVVDP 4.8 GB.

    `together` (Settings > GPU metrics) scores them in one pass anyway (see
    vship_passes): where the CPU decodes -- 4K VVC, which no GPU decodes --
    decoding each video once instead of once per metric saves far more
    than the memory costs. A metric that fails in a shared pass, out of GPU
    memory most likely, is calculated again in a pass of its own.

    A metric whose pass fails is reported in the output's `failures` while
    the others still run; only when every pass fails is the first error
    raised (and the caller may retry on the CPU).

    A metric this GPU does not calculate at this size (GPU_PIXEL_LIMITS) is
    not started: it is a failure with the reason, before any GPU work.
    """
    refused = {}
    if any((device.backend, device.vendor, spec.key) in GPU_PIXEL_LIMITS for spec in specs):
        size = _scaled_sizes(source, distorted, request.recipe, source_crop, distorted_crop)[0]
        refused = {spec.key: reason for spec in specs if (reason := size_refusal(device, spec.key, size))}
    if refused:
        for key, reason in refused.items():
            _log.warning("%s not started: %s", metric_definition(key).label, reason)
        specs = tuple(spec for spec in specs if spec.key not in refused)
        if not specs:
            raise VshipPassesFailedError(next(iter(refused.values())), refused)
        output = run_vship_task(
            source, distorted, request, specs, device, source_crop, distorted_crop,
            on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle, on_pass_done=on_pass_done, together=together, on_pass=on_pass)
        return replace(output, failures={**output.failures, **refused})
    if not _gpu_pass.acquire(blocking=False):
        if on_status:
            on_status(Status(GPU_WAIT_MESSAGE, kind=GPU_WAIT))
        while not _gpu_pass.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user")
    try:
        if len(specs) <= 1:
            if on_pass is not None:
                on_pass(1, 1, tuple(spec.key for spec in specs))
            return _run_vship_pass(
                source, distorted, request, specs, device, source_crop, distorted_crop,
                on_progress=on_progress, on_status=on_status,
                cancel_event=cancel_event, process_handle=process_handle,
            )
        metrics = MetricResultSet()
        failures: dict[str, str] = {}
        errors: list[BaseException] = []
        frames = 0
        # How far each pass run so far got, in its own progress units: the
        # figure is one count across every pass, a retry's after the passes
        # before it. A retry reported only itself, so the half went from
        # 100% back to 0% when a metric that failed in the shared pass was
        # calculated again, and its time left was that retry's alone.
        reached: list[int] = []

        def pass_progress(slot: int, done: int, left: int) -> Callable[[int, int, float], None]:
            """A pass's progress as the task's: after `done` units of the
            passes before it, with `left` passes of its length to go (this
            one included -- they score the same frames)."""
            def progress(current: int, total: int, fps: float) -> None:
                reached[slot] = current
                on_progress(done + current, done + left * total, fps)
            return progress

        def run_passes(passes: list[tuple[MetricRequestSpec, ...]], first: int, count: int) -> None:
            """Runs `passes` as passes first + 1 onwards of `count`."""
            nonlocal frames
            for offset, group in enumerate(passes):
                number = first + offset
                labels = " + ".join(metric_definition(spec.key).label for spec in group)
                if on_status and count > 1:
                    on_status(Status(f"GPU metric {number + 1}/{count}: {labels}", kind=GPU_PASS_STATUS))
                if on_pass is not None:
                    on_pass(number + 1, count, tuple(spec.key for spec in group))
                reached.append(0)
                progress = (pass_progress(len(reached) - 1, sum(reached), count - number)
                            if on_progress is not None else None)
                try:
                    output = _run_vship_pass(
                        source, distorted, request, group, device, source_crop, distorted_crop,
                        on_progress=progress, on_status=on_status,
                        cancel_event=cancel_event, process_handle=process_handle,
                    )
                except PerceptualCancelled:
                    raise
                except Exception as error:
                    if cancel_event is not None and cancel_event.is_set():
                        raise PerceptualCancelled("Cancelled by user") from error
                    _log.error("GPU metric %d/%d (%s) failed: %s", number + 1, count, labels, error)
                    errors.append(error)
                    failures.update(dict.fromkeys((spec.key for spec in group), str(error)))
                    continue
                for key in output.metrics:
                    metrics.add(output.metrics.get(key))
                failures.update(output.failures)
                frames = max(frames, output.compared_frame_count)
                if on_pass_done is not None and output.metrics:
                    on_pass_done(output)

        passes = vship_passes(specs, together)
        run_passes(passes, 0, len(passes))
        shared = [spec for group in passes if len(group) > 1 for spec in group if spec.key in failures]
        if shared:
            labels = " and ".join(metric_definition(spec.key).label for spec in shared)
            _log.warning("%s failed in the GPU pass shared with the other metrics; calculating %s in a "
                         "pass of %s own", labels, *(("it", "its") if len(shared) == 1 else ("them", "their")))
            if on_status:
                on_status(f"{labels} failed in the shared GPU pass; calculating "
                          f"{'it' if len(shared) == 1 else 'each'} in a pass of its own…")
            for spec in shared:
                del failures[spec.key]
            failed_before = list(errors)
            errors.clear()
            # Numbered after the passes before them -- "GPU metric 2/2" after
            # a shared pass -- so the window counts the shared pass as done.
            run_passes([(spec,) for spec in shared], len(passes), len(passes) + len(shared))
            errors.extend(failed_before)  # the retries' own errors first
        if not metrics:
            if not errors:
                raise VshipUnavailableError("Vship produced no scores.")
            raise VshipPassesFailedError(str(errors[0]), failures) from errors[0]
        return PerceptualTaskOutput(metrics, source_crop, distorted_crop, frames, failures)
    finally:
        _gpu_pass.release()


def _run_vship_pass(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    """One Vship pass (_score_vship_pass), in a process of its own.

    Vship and the GPU driver under it run in the process that loads them: a
    crash in either took the app down, every video's progress with it, with
    no message. Now it ends the pass's process: the pass fails like any
    other GPU failure -- its metrics are calculated on the CPU, or fail with
    the reason -- and the run goes on. The pass's FFmpeg processes are
    attached to `process_handle` as before, so Pause and Cancel reach them.

    A device whose library this process has already loaded is scored here:
    the crash it could cause is this process's anyway. The app's devices
    come from the isolated probe without one."""
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
    """One Vship pass, with the videos NVIDIA's decoder takes decoded in
    this process (_NativeFrameStream). If that decoding fails after it has
    started -- a damaged stream, pictures that are not the packets' -- the
    pass is made again with FFmpeg decoding, as before."""
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
    if not specs or any(spec.backend_id != BACKEND_ID or spec.key not in _METRICS for spec in specs):
        raise ValueError("Vship task requires supported perceptual metric specs")
    if request.recipe.resample_test is not None:
        raise VshipUnavailableError("The GPU metrics do not support resolution round-trip tests yet.")
    _validate_pair(source, distorted, request.recipe)
    if cancel_event is not None and cancel_event.is_set():
        raise PerceptualCancelled("Cancelled by user")

    lib = _library(device)

    # Decode follows the row's GPU-decode setting, per input and per codec:
    # NVDEC for HEVC/AV1/H.264 on NVIDIA, software where the GPU has no
    # decoder (VVC) or where FFmpeg's hardware decode does not give the
    # video's own pictures (gpu.pick_decode: 4:2:2, 4:4:4, 12-bit, an
    # odd size, 10-bit H.264 on Intel and AMD). Hardware decode is otherwise bit-exact: it
    # changes speed only.
    vendor = request.execution.gpu_vendor if request.execution.gpu_decode else GpuVendor.NONE

    def decoder_of(info: VideoInfo) -> str | None:
        return pick_decode(vendor, info.codec_name, info.pix_fmt, info.width, info.height)

    src_hwaccel, dist_hwaccel = decoder_of(source), decoder_of(distorted)
    src_passthrough = _passthrough_format(source, src_hwaccel)
    dist_passthrough = _passthrough_format(distorted, dist_hwaccel)
    src_format = src_passthrough[0] if src_passthrough else _image_format(source, device.version)
    dist_format = dist_passthrough[0] if dist_passthrough else _image_format(distorted, device.version)
    src_size, dist_size = _scaled_sizes(source, distorted, request.recipe, source_crop, distorted_crop)
    src_color = _vship_colorspace(source, src_format, *src_size)
    dist_color = _vship_colorspace(distorted, dist_format, *dist_size)
    src_layout = src_format.frame_layout(*src_size)
    dist_layout = dist_format.frame_layout(*dist_size)
    src_frame_bytes, src_plane_sizes, src_strides, _ = src_layout
    dist_frame_bytes, dist_plane_sizes, dist_strides, _ = dist_layout

    step = specs[0].coverage.step if specs[0].coverage is not None else 1
    if any((spec.coverage.step if spec.coverage is not None else 1) != step for spec in specs):
        raise VshipUnavailableError("Perceptual metrics in one task must use the same frame coverage.")
    cvvdp_spec = next((spec for spec in specs if spec.key == "cvvdp"), None)
    frame_specs = tuple(spec for spec in specs if spec.key != "cvvdp")
    if cvvdp_spec is not None and step != 1:
        raise VshipUnavailableError("CVVDP scores every frame; it cannot be subsampled.")
    # Where each video is decoded, in the "(GPU decode: ...)" form FFmpeg's
    # runs report it in: the window's run line shows it as "Decoder: Source:
    # GPU, test video: CPU". Only FFmpeg's metrics reported it, so a video
    # with only GPU metrics never said where it was decoded.
    decode = {"source": src_hwaccel, "distorted": dist_hwaccel}
    decode_lock = threading.Lock()

    def decoded_in_software(side: str) -> Callable[[], None]:
        def report() -> None:
            with decode_lock:  # the two inputs' threads can both fall back
                decode[side] = None
                if on_status:
                    on_status(Status.decoding(
                        f"GPU decode failed for the {'test video' if side == 'distorted' else side}, "
                        "decoding it in software", HwAccelPlan(**decode)))
        return report

    streams: list[_FrameStream | _NativeFrameStream] = []
    lanes: list[_MetricLane | _CvvdpLane] = []
    cvvdp_lane: _CvvdpLane | None = None
    started = time.perf_counter()
    scores: dict[str, _ScoreArray] = {spec.key: _ScoreArray() for spec in frame_specs}
    abort = threading.Event()
    pending: dict[int, list[int]] = {}  # frame index -> [metrics left, source slot, test slot]
    #: Per stream (source, test): slot -> holders. A slot goes back to its
    #: stream once the pairing and every pair it is in are done with it: a
    #: source frame can be in more than one pair.
    held: tuple[dict[int, int], dict[int, int]] = ({}, {})
    pending_lock = threading.Lock()
    expected_frames = min(source.estimated_frame_count, distorted.estimated_frame_count)
    if request.recipe.duration_limit > 0:
        expected_frames = min(expected_frames, max(1, math.ceil(request.recipe.duration_limit * source.fps)))
    expected_samples = max(1, math.ceil(expected_frames / step))
    total_units = expected_samples * step

    def commands(info: VideoInfo, crop: CropBox | None, target: tuple[int, int],
                 hwaccel: str | None, pixel_format: str, every: int, cut: bool) -> list[list[str]]:
        # The software retry pipes the same layout as the hardware attempt:
        # Vship's handlers are set up for one layout per input.
        attempts = []
        for accel in ([hwaccel, None] if hwaccel else [None]):
            filter_chain = _filter_chain(
                info, crop, target, pixel_format, every,
                request.recipe.scale_algorithm, accel,
            )
            command = [
                ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
                *hwaccel_args(accel),
                "-i", str(info.path.resolve()), "-map", f"0:{VIDEO_STREAM}", "-an", "-sn", "-dn",
                "-vf", filter_chain,
            ]
            if cut and request.recipe.duration_limit > 0:
                command += ["-t", f"{request.recipe.duration_limit:.6f}"]
            command += ["-fps_mode", "passthrough", *_piped_as(pixel_format)]
            attempts.append(command)
        return attempts

    # Each test frame is one pair, as each of libvmaf's is: every step-th,
    # up to the duration limit. The source gives the frames frame_pairs
    # finds each one's pair among -- by its time, so it is never cut at the
    # limit, and it is subsampled as the test video is while the two line
    # up (_FramesApartError); a source without a frame rate gives every frame.
    every_source_frame = every_source_frame or source.fps <= 0
    check_apart = step > 1 and not every_source_frame
    try:
        def stream(info, crop, size, hwaccel, image_format, passthrough, frame_bytes, plane_sizes, label, side):
            cut = side == "distorted"
            every = 1 if side == "source" and every_source_frame else step
            # Decoded here, by the GPU decoder FFmpeg would decode it with.
            backend = _decoded_here(hwaccel, device) if native else None
            if backend is not None:
                decoder = _native_decoder(info, crop, size, image_format, frame_bytes, device.gpu_id,
                                          process_handle, label, backend, request.recipe.scale_algorithm)
                if decoder is not None:
                    limit = (f"{request.recipe.duration_limit:.6f}"
                             if cut and request.recipe.duration_limit > 0 else None)
                    try:
                        return _NativeFrameStream(lib, frame_bytes, decoder, every, limit, label, device.gpu_id)
                    except BaseException:
                        decoder.close()
                        raise
            split = None
            if passthrough is not None:
                dtype = np.uint16 if image_format.sample != _VSHIP_ENUMS[8] else np.uint8
                split = (plane_sizes[0], plane_sizes[1], dtype)
            pixel_format = passthrough[1] if passthrough else image_format.pixel_format
            return _FrameStream(lib, frame_bytes, commands(info, crop, size, hwaccel, pixel_format, every, cut),
                                process_handle, label, split, decoded_in_software(side), device.gpu_id)

        # Appended one at a time: if the test video's buffers cannot be
        # allocated, the reference's stream is already in `streams`, so the
        # finally block below frees it. Built as one list, it leaked.
        streams.append(stream(source, source_crop, src_size, src_hwaccel, src_format, src_passthrough,
                              src_frame_bytes, src_plane_sizes, "reference", "source"))
        streams.append(stream(distorted, distorted_crop, dist_size, dist_hwaccel, dist_format,
                              dist_passthrough, dist_frame_bytes, dist_plane_sizes, "test video", "distorted"))
        if on_status:
            labels = ", ".join(metric_definition(spec.key).label for spec in specs)
            on_status(Status.decoding(f"Vship GPU ({device.name}): calculating {labels}", HwAccelPlan(**decode)))
        for stream in streams:
            stream.start()
        source_stream, distorted_stream = streams
        source_planes = [buffer.planes(src_format, src_plane_sizes) for buffer in source_stream.buffers]
        distorted_planes = [buffer.planes(dist_format, dist_plane_sizes) for buffer in distorted_stream.buffers]

        def drop(side: int, slot: int) -> None:
            """One holder of stream `side`'s `slot` is done with it."""
            with pending_lock:
                holders = held[side][slot] - 1
                if holders:
                    held[side][slot] = holders
                    return
                del held[side][slot]
            streams[side].release(slot)

        def finished(index: int) -> None:
            with pending_lock:
                entry = pending[index]
                entry[0] -= 1
                if entry[0]:
                    return
                del pending[index]
            drop(0, entry[1])
            drop(1, entry[2])

        def puller(side: int, first: int) -> Callable[[], tuple[int, int] | None]:
            """frame_pairs' source of stream `side`'s frames, starting with
            `first` (taken already, for its time base)."""
            stream, waiting = streams[side], [first]

            def pull() -> tuple[int, int] | None:
                slot = waiting.pop() if waiting else stream.next(cancel_event)
                if slot == _EOF:
                    return None
                with pending_lock:
                    held[side][slot] = held[side].get(slot, 0) + 1
                return slot, stream.pts[slot]
            return pull

        by_metric: dict[str, list[_MetricLane]] = {}
        if cvvdp_spec is not None:
            cvvdp_lane = _CvvdpLane(
                device, src_color, dist_color, CvvdpSettings.from_spec_parameters(cvvdp_spec.parameters),
                source.fps, source_planes, distorted_planes, src_strides, dist_strides, finished, abort,
            )
            lanes.append(cvvdp_lane)
        for spec in frame_specs:
            by_metric[spec.key] = [
                _MetricLane(device, spec.key, src_color, dist_color, source_planes, distorted_planes,
                            src_strides, dist_strides, scores[spec.key], finished, abort)
                for _ in range(_LANES_PER_METRIC)
            ]
            lanes.extend(by_metric[spec.key])
        for lane in lanes:
            lane.start()

        def lane_errors() -> dict[str, BaseException]:
            """Each failed metric's first error. One failed lane fails its
            metric: the other lane's frames alone would leave holes."""
            errors = {key: next((lane.error for lane in metric_lanes if lane.error is not None), None)
                      for key, metric_lanes in by_metric.items()}
            if cvvdp_lane is not None:
                errors["cvvdp"] = cvvdp_lane.error
            return {key: error for key, error in errors.items() if error is not None}

        frame = 0
        rate = _PassRate(step)
        # Each test frame with the source frame libvmaf pairs it with, by
        # timestamp (frame_sync): the same pairs VMAF is calculated on. They
        # used to be paired by position, which a frame dropped from the test
        # video put one apart for the rest of the video. The comparison is
        # the frames both have (shortest=1): it ends with the test video, or
        # at its first frame past the source's end; the longer input's
        # reader is stopped when the streams close.
        first_source = source_stream.next(cancel_event)
        first_test = distorted_stream.next(cancel_event)
        if _EOF in (first_source, first_test):
            for stream, slot in ((source_stream, first_source), (distorted_stream, first_test)):
                if slot != _EOF:
                    stream.release(slot)
        else:
            test_base, source_base = distorted_stream.time_base, source_stream.time_base
            source_start = source_stream.pts[first_source]
            pairs = frame_pairs(puller(1, first_test), puller(0, first_source), test_base, source_base,
                                lambda slot: drop(1, slot), lambda slot: drop(0, slot))
            try:
                for distorted_slot, source_slot, when in pairs:
                    errors = lane_errors()
                    if len(errors) == len(specs):
                        # Nothing left to score: stop now rather than decode
                        # the rest of the video (a film took ~30 minutes to
                        # report a CVVDP handler that had failed to start).
                        raise next(iter(errors.values()))
                    # No abort here: a failed lane keeps handing its frames
                    # back, so the ring never fills with slots nobody will
                    # release, and the check above ends the pass once no
                    # metric is left.
                    if source_slot is None:
                        raise VshipUnavailableError("The source has no frame at the test video's first.")
                    if check_apart:
                        test_time = when * test_base
                        apart = test_time - (source_stream.pts[source_slot] - source_start) * source_base
                        if abs(apart) * Fraction(source.fps) > _APART_FRAMES:
                            raise _FramesApartError(
                                f"the test frame at {float(test_time):.3f} s is {float(apart) * 1000:.1f} ms from "
                                f"the nearest of the source frames sampled with it (one in {step})")
                    with pending_lock:
                        pending[frame] = [len(specs), source_slot, distorted_slot]
                        held[0][source_slot] += 1
                        held[1][distorted_slot] += 1
                    for spec in frame_specs:
                        scores[spec.key].reserve(frame)
                    for spec in frame_specs:
                        by_metric[spec.key][frame % _LANES_PER_METRIC].jobs.put(
                            (frame, source_slot, distorted_slot))
                    if cvvdp_lane is not None:
                        cvvdp_lane.jobs.put((frame, source_slot, distorted_slot))
                    frame += 1
                    if on_progress:
                        on_progress(min(frame * step, total_units), total_units, rate.frames_per_second(frame))
            finally:
                pairs.close()  # hands back the frames the pairing holds
        for lane in lanes:
            lane.stop()
        for lane in lanes:
            lane.join()
        if frame == 0:
            raise VshipUnavailableError("FFmpeg produced no frame pairs for Vship.")
        if (short := short_comparison(expected_frames, frame * step, source.fps, step)) is not None:
            raise ComparisonCutShortError(short)
        # A metric whose handler failed (out of VRAM on a smaller card, say)
        # is reported on its own; the others keep their scores. A SSIMULACRA2
        # or Butteraugli failure used to end the pass and take CVVDP with it.
        errors = lane_errors()
        if len(errors) == len(specs):
            raise next(iter(errors.values()))
        metric_failures = {key: str(error) for key, error in errors.items()}

        frame_numbers = np.arange(frame, dtype=np.int32) * step
        times = frame_numbers.astype(np.float64) / max(source.fps, 1.0)
        results = MetricResultSet()
        parameters = {
            "gpu_backend": device.backend,
            "gpu_name": device.name,
            "input": "ffmpeg frames (hardware-decoded NV12/P010 split into planes); "
                     "native range/transfer and primaries",
            "coverage_step": step,
            "butteraugli_norm": "3-norm",
            # How the videos' color tags were read (_vship_colorspace); the
            # cache recalculates GPU scores of v1.2, which has none.
            "color_tags": VSHIP_COLOR_TAGS,
        }
        for spec in frame_specs:
            if spec.key in metric_failures:
                continue
            results.add(FrameMetricResult(
                spec.key, frame_numbers, times,
                scores[spec.key].values(frame),
                MetricProvenance(
                    implementation=f"Vship/{spec.key}",
                    implementation_version=f"Vship {device.version}",
                    compute_backend="gpu",
                    implementation_compatibility_id=f"{spec.key}-vship-gpu-v1",
                    parameters=parameters,
                ),
            ))
        if cvvdp_lane is not None and cvvdp_lane.error is None:
            windows = cvvdp_lane.windows
            first = np.array([start for start, _count, _jod in windows], dtype=np.int32)
            results.add(SequenceMetricResult(
                "cvvdp", pool_cvvdp_windows(windows),
                MetricProvenance(
                    implementation="Vship/cvvdp",
                    implementation_version=f"Vship {device.version}",
                    compute_backend="gpu",
                    implementation_compatibility_id=cvvdp_spec.implementation_compatibility_id,
                    parameters={
                        "gpu_backend": device.backend, "gpu_name": device.name,
                        "display": dict(cvvdp_spec.parameters)["display"],
                        "resize_to_display": dict(cvvdp_spec.parameters)["resize_to_display"],
                        "timeline": "JOD of each second of video",
                        "color_tags": VSHIP_COLOR_TAGS,
                    },
                ),
                frame=first, time=first.astype(np.float64) / max(source.fps, 1.0),
                values=[jod for _start, _count, jod in windows],
            ))
        elapsed = max(time.perf_counter() - started, 1e-6)
        _log.info("Vship pass (%s) on %s: %d frame pairs in %.1f s, %.1f pairs/s",
                  ", ".join(metric_definition(spec.key).label for spec in specs), device.name, frame, elapsed,
                  frame / elapsed)
        for key, error in errors.items():
            _log.error("Vship %s failed in its pass: %s", metric_definition(key).label, error, exc_info=error)
        if on_progress:
            # A pass too short to time from its first frame keeps the old
            # figure: its whole length.
            on_progress(frame * step, frame * step, rate.frames_per_second(frame) or frame * step / elapsed)
        return PerceptualTaskOutput(results, source_crop, distorted_crop, frame * step, metric_failures)
    except (PerceptualCancelled, ComparisonCutShortError, gpu_frames.GpuDecodeFailedError, _FramesApartError):
        # The file's fault, not the GPU's: the CPU would stop as short. A
        # failed GPU decode is made again through FFmpeg, and a subsampled
        # pass whose frames do not line up with every source frame
        # (_score_vship_pass).
        raise
    except VshipUnavailableError as error:
        _log.error("Vship pass (%s) failed: %s", ", ".join(metric_definition(spec.key).label for spec in specs),
                   error, exc_info=error)
        raise
    except Exception as error:
        _log.error("Vship pass (%s) failed", ", ".join(metric_definition(spec.key).label for spec in specs),
                   exc_info=error)
        raise VshipUnavailableError(f"Vship GPU calculation failed: {error}") from error
    finally:
        # Lanes first: they read the pinned frames the streams own, so the
        # streams may only free that memory once no lane can touch it.
        abort.set()
        for lane in lanes:
            lane.stop()
        for lane in lanes:
            lane.join()
        for stream in streams:
            stream.close()


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
    """Run selected backends, with a per-metric GPU-to-CPU fallback.
    `together` scores the GPU metrics in one pass (see run_vship_task), and
    `on_pass` hears of each GPU pass as it starts. `concurrent_cpu_tasks` is
    how many tasks share the CPU while this one runs: the CPU metrics take
    their share of it (perceptual_cpu.scoring_workers).

    `on_cpu` gets the keys of the metrics the CPU is about to calculate,
    just before it starts on them: ones chosen for the CPU, and ones meant
    for the GPU and handed over -- no usable GPU, a failed GPU pass, or a
    GPU this build of Vship scores wrongly on. Progress reported after it
    is the CPU's own, from 0: the run line shows those metrics as CPU
    metrics, with the CPU's figures.

    CVVDP runs on the GPU only. Without a usable GPU it fails on its own --
    reported in the output's `failures` -- and the other metrics are scored
    as they would have been without it.
    """
    from vmaf_app.core import perceptual_cpu
    from vmaf_app.core.perceptual_cpu import _resolve_crops

    def run_perceptual_task(*args, **kwargs) -> PerceptualTaskOutput:
        return perceptual_cpu.run_perceptual_task(*args, concurrent_tasks=concurrent_cpu_tasks, **kwargs)

    def backend(spec: MetricRequestSpec) -> str:
        return "gpu" if spec.key in GPU_ONLY_METRICS else request.execution.perceptual_backend(spec.key)

    cpu_specs = tuple(spec for spec in specs if backend(spec) == "cpu")
    gpu_specs = tuple(spec for spec in specs if backend(spec) == "gpu")
    if len(cpu_specs) + len(gpu_specs) != len(specs):
        raise ValueError("perceptual metric backend must be either GPU or CPU")
    gpu_only = tuple(spec for spec in gpu_specs if spec.key in GPU_ONLY_METRICS)
    # What may be retried on the CPU if the GPU cannot be used.
    retryable = tuple(spec for spec in specs if spec.key not in GPU_ONLY_METRICS)

    def gpu_only_failed(reason: str, error: BaseException | None = None, *,
                        no_gpu: bool = True, reasons: dict[str, str] | None = None) -> dict[str, str]:
        """The failures for the GPU-only metrics; raises if nothing else was
        asked for. `no_gpu` is for when no usable GPU was found; otherwise
        the GPU pass itself failed, possibly for a reason that has nothing to
        do with the GPU (the two videos' frame rates differing, say), and
        blaming the GPU sent people looking for the wrong problem."""
        def message(spec: MetricRequestSpec) -> str:
            label = metric_definition(spec.key).label
            own = (reasons or {}).get(spec.key, reason)
            return (f"{label} needs a GPU that Vship can use, and could not use one: {own}"
                    if no_gpu else f"{label} could not be calculated: {own}")

        messages = {spec.key: message(spec) for spec in gpu_only}
        if not retryable:
            raise PerceptualRunError("\n".join(messages.values())) from error
        return messages

    def with_failures(output: PerceptualTaskOutput, failures: dict[str, str]) -> PerceptualTaskOutput:
        return replace(output, failures={**output.failures, **failures}) if failures else output

    def to_cpu(handed: tuple[MetricRequestSpec, ...]) -> None:
        """The metrics the CPU is about to calculate."""
        if on_cpu is not None and handed:
            on_cpu(tuple(spec.key for spec in handed))

    # An explicit CPU selection must not probe Vship or touch a compute GPU.
    if not gpu_specs:
        to_cpu(cpu_specs)
        return run_perceptual_task(
            source, distorted, request, cpu_specs,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
        )

    forget_failed_vship_probe()
    device, reason = detect_vship_device()
    if device is None:
        failures = gpu_only_failed(reason) if gpu_only else {}
        if on_status:
            on_status(f"Vship GPU unavailable ({reason}); using CPU reference metrics…")
        to_cpu(retryable)
        return with_failures(run_perceptual_task(
            source, distorted, request, retryable,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
        ), failures)

    # A metric this build scores wrongly goes to the CPU, planned like a CPU
    # choice: as on a PC with no GPU for it, the Videos tab asks before a
    # long CPU run (MainWindow._confirm_long_cpu_perceptual).
    wrong = tuple(spec for spec in gpu_specs if not scores_correctly(device, spec.key))
    if wrong:
        _log.info("%s on the CPU: Vship's %s build scores it wrongly on %s",
                  " and ".join(metric_definition(spec.key).label for spec in wrong), backend_label(device.backend),
                  device.name)
        gpu_specs = tuple(spec for spec in gpu_specs if spec not in wrong)
        cpu_specs = cpu_specs + wrong
        if not gpu_specs:
            to_cpu(cpu_specs)
            return run_perceptual_task(
                source, distorted, request, cpu_specs,
                on_progress=on_progress, on_status=on_status,
                cancel_event=cancel_event, process_handle=process_handle,
            )

    crops = _resolve_crops(
        source, distorted, request.recipe, cancel_event, process_handle, on_status,
    )
    try:
        gpu_output = run_vship_task(
            source, distorted, request, gpu_specs, device, *crops,
            on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle, on_pass_done=on_pass_done, together=together, on_pass=on_pass,
        )
    except PerceptualCancelled:
        raise
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            raise PerceptualCancelled("Cancelled by user") from error
        failures = (gpu_only_failed(str(error), error, no_gpu=False,
                                    reasons=getattr(error, "failures", None)) if gpu_only else {})
        gpu_retry = tuple(spec for spec in gpu_specs if spec.key not in GPU_ONLY_METRICS)
        if gpu_retry and compared_seconds(source, distorted, request.recipe.duration_limit) > LONG_CPU_RUN_SECONDS:
            # Nobody agreed to a CPU run of this length: the Videos tab asks
            # before one, but a GPU failure mid-run cannot. Falling back
            # silently meant days of CPU work and terabytes of temporary
            # images for a film. The metric fails instead, with the reason;
            # the video keeps its other metrics.
            labels = " and ".join(metric_definition(spec.key).label for spec in gpu_retry)
            # CVVDP's own failure goes into the same message: raising here
            # used to drop it, leaving only advice about the CPU that does
            # not apply to a GPU-only metric.
            raise PerceptualRunError(
                f"GPU scoring failed ({error}). It was not retried on the CPU, which would take "
                "hours to days for a video over 10 minutes. "
                f"Choose CPU for {labels} to calculate it on the CPU anyway."
                + "".join(f"\n{message}" for message in dict.fromkeys(failures.values()))
            ) from error
        if on_status:
            on_status(f"Vship GPU compute failed ({error}); using CPU reference metrics…")
        to_cpu(retryable)
        # Run all metrics together after a GPU failure so CPU frame extraction
        # happens only once, and the returned result remains atomic.
        return with_failures(run_perceptual_task(
            source, distorted, request, retryable,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
            resolved_crops=crops,
        ), failures)

    # A metric that failed on the GPU while the others finished is retried
    # on the CPU on the same terms as a failed pass: only for videos up to
    # ten minutes, which nobody has to agree to.
    failures = dict(gpu_output.failures)
    gpu_failed = tuple(spec for spec in gpu_specs if spec.key in failures and spec.key not in GPU_ONLY_METRICS)
    if gpu_failed and compared_seconds(source, distorted, request.recipe.duration_limit) > LONG_CPU_RUN_SECONDS:
        for spec in gpu_failed:
            failures[spec.key] = (
                f"GPU scoring failed ({failures[spec.key]}). It was not retried on the CPU, which would "
                "take hours to days for a video over 10 minutes. Choose CPU for "
                f"{metric_definition(spec.key).label} to calculate it on the CPU anyway."
            )
        gpu_failed = ()
    # Kept for the message if the CPU retry fails too: the GPU failure is
    # the first cause, and dropping it sent a GPU user to fix a CPU tool.
    gpu_reasons = {spec.key: failures.pop(spec.key) for spec in gpu_failed}
    cpu_specs = cpu_specs + gpu_failed
    if not cpu_specs:
        return replace(gpu_output, failures=failures) if failures != gpu_output.failures else gpu_output
    to_cpu(cpu_specs)

    if on_status:
        if gpu_failed:
            labels = " and ".join(metric_definition(spec.key).label for spec in gpu_failed)
            on_status(f"{labels} failed on the GPU; calculating it on the CPU…")
        else:
            on_status("Calculating selected perceptual metric(s) on CPU…")

    def failed_on_cpu(key: str, reason: str) -> str:
        if key in gpu_reasons:
            return f"GPU scoring failed ({gpu_reasons[key]}); the CPU retry failed too: {reason}"
        return reason

    # The GPU pass has finished: a CPU failure from here on fails only the
    # CPU metrics. It used to raise out of here and throw away what the GPU
    # had scored -- a finished CVVDP pass lost to a missing libjxl tool.
    try:
        cpu_output = run_perceptual_task(
            source, distorted, request, cpu_specs,
            on_progress=on_progress,  # a stage of its own, from 0 (on_cpu)
            on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle, resolved_crops=crops,
        )
    except PerceptualCancelled:
        raise
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            raise PerceptualCancelled("Cancelled by user") from error
        if not gpu_output.metrics:
            if gpu_reasons:
                raise PerceptualRunError(failed_on_cpu(cpu_specs[-1].key, str(error))) from error
            raise
        failures.update({spec.key: failed_on_cpu(spec.key, str(error)) for spec in cpu_specs})
        return replace(gpu_output, failures=failures)
    # Each metric is kept on its own frames. Vship and the CPU tools can end
    # a frame apart -- each stops where the shorter input's pictures do --
    # and that used to throw the CPU's finished scores away, where the cache
    # and the window have always taken each metric's own frames.
    combined = MetricResultSet()
    for spec in specs:
        if spec.key in failures:
            continue
        value = gpu_output.metrics.get(spec.key) or cpu_output.metrics.get(spec.key)
        if value is None:
            raise PerceptualRunError(f"Selected backend did not produce {spec.key}.")
        assert value is not None
        combined.add(value)
    return PerceptualTaskOutput(
        combined, gpu_output.source_crop, gpu_output.distorted_crop,
        max(gpu_output.compared_frame_count, cpu_output.compared_frame_count), failures,
    )
