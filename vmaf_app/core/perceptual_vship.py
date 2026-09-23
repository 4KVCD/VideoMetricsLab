"""GPU implementation of the frame-based perceptual metrics via Vship's C API.

FFmpeg remains responsible for decoding, cropping, sampling, and scaling. It
streams tightly packed planar frames into two pinned host buffers; Vship only
does the metric computation on a supported NVIDIA CUDA or AMD HIP device.
This avoids bundling FFVship/FFMS2 executables and keeps video decode behavior
under the same FFmpeg installation used by the rest of the app.
"""
from __future__ import annotations

import contextlib
import ctypes
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from vmaf_app.core import proc as proc_util
from vmaf_app.core.analysis_request import AnalysisRequest, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo
from vmaf_app.core.perceptual_cpu import (
    PerceptualCancelled,
    PerceptualRunError,
    PerceptualTaskOutput,
    _content_size,
    _validate_pair,
)
from vmaf_app.core.process_control import ProcessHandle

BACKEND_ID = "perceptual"
_METRICS = {"ssimulacra2", "butteraugli"}


class VshipUnavailableError(PerceptualRunError):
    """The GPU implementation could not be loaded or run; CPU fallback is safe."""


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


class _Handler(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint)]


class _ButteraugliScore(ctypes.Structure):
    _fields_ = [
        ("normQ", ctypes.c_double), ("norm3", ctypes.c_double),
        ("norminf", ctypes.c_double),
    ]


_U8P = ctypes.POINTER(ctypes.c_uint8)
_PLANES = ctypes.POINTER(_U8P)
_I64_3 = ctypes.c_int64 * 3
_VSHIP_ENUMS = {8: 2, 9: 3, 10: 5, 12: 7, 14: 9, 16: 11}
_YUV_LAYOUTS = {
    # Vship's format adapter supports these planar layouts / sample depths.
    "410": {8}, "411": {8}, "420": {8, 9, 10, 12, 14, 16},
    "422": {8, 9, 10, 12, 14, 16}, "440": {8, 10, 12},
    "444": {8, 9, 10, 12, 14, 16},
}


@dataclass(frozen=True, slots=True)
class _ImageFormat:
    pixel_format: str
    family: int
    sample: int
    subw: int
    subh: int
    full_range: bool = False
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
    vendor: str
    path: Path
    library: ctypes.CDLL
    dll_directory: object


@dataclass(frozen=True, slots=True)
class VshipDevice:
    vendor: str
    name: str
    gpu_id: int
    version: str
    loaded: _LoadedVship


def _configure_api(lib: ctypes.CDLL) -> None:
    lib.Vship_GetVersion.argtypes = []
    lib.Vship_GetVersion.restype = _Version
    lib.Vship_GetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.Vship_GetDeviceCount.restype = ctypes.c_int
    lib.Vship_GetDeviceInfo.argtypes = [ctypes.POINTER(_DeviceInfo), ctypes.c_int]
    lib.Vship_GetDeviceInfo.restype = ctypes.c_int
    lib.Vship_GPUFullCheck.argtypes = [ctypes.c_int]
    lib.Vship_GPUFullCheck.restype = ctypes.c_int
    lib.Vship_SetDevice.argtypes = [ctypes.c_int]
    lib.Vship_SetDevice.restype = ctypes.c_int
    lib.Vship_GetErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.Vship_GetErrorMessage.restype = ctypes.c_int
    lib.Vship_GetDetailedLastError.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.Vship_GetDetailedLastError.restype = ctypes.c_int
    lib.Vship_PinnedMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
    lib.Vship_PinnedMalloc.restype = ctypes.c_int
    lib.Vship_PinnedFree.argtypes = [ctypes.c_void_p]
    lib.Vship_PinnedFree.restype = ctypes.c_int
    lib.Vship_SSIMU2Init.argtypes = [ctypes.POINTER(_Handler), _Colorspace, _Colorspace]
    lib.Vship_SSIMU2Init.restype = ctypes.c_int
    lib.Vship_SSIMU2Free.argtypes = [_Handler]
    lib.Vship_SSIMU2Free.restype = ctypes.c_int
    lib.Vship_ComputeSSIMU2.argtypes = [
        _Handler, ctypes.POINTER(ctypes.c_double), _PLANES, _PLANES,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]
    lib.Vship_ComputeSSIMU2.restype = ctypes.c_int
    lib.Vship_SSIMU2GetDetailedLastError.argtypes = [_Handler, ctypes.c_char_p, ctypes.c_int]
    lib.Vship_SSIMU2GetDetailedLastError.restype = ctypes.c_int
    lib.Vship_ButteraugliInit.argtypes = [
        ctypes.POINTER(_Handler), _Colorspace, _Colorspace, ctypes.c_int, ctypes.c_float,
    ]
    lib.Vship_ButteraugliInit.restype = ctypes.c_int
    lib.Vship_ButteraugliFree.argtypes = [_Handler]
    lib.Vship_ButteraugliFree.restype = ctypes.c_int
    lib.Vship_ComputeButteraugli.argtypes = [
        _Handler, ctypes.POINTER(_ButteraugliScore), ctypes.c_void_p, ctypes.c_int64,
        _PLANES, _PLANES, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]
    lib.Vship_ComputeButteraugli.restype = ctypes.c_int
    lib.Vship_ButteraugliGetDetailedLastError.argtypes = [_Handler, ctypes.c_char_p, ctypes.c_int]
    lib.Vship_ButteraugliGetDetailedLastError.restype = ctypes.c_int


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


@lru_cache(maxsize=1)
def detect_vship_device() -> tuple[VshipDevice | None, str]:
    """Return a fully verified NVIDIA/AMD device, or a user-readable reason."""
    if os.name != "nt":
        return None, "Vship GPU acceleration is only bundled for Windows."
    tools = Path(__file__).resolve().parents[1] / "tools" / "vship"
    failures: list[str] = []
    for vendor in ("nvidia", "amd"):
        path = tools / vendor / "libvship.dll"
        if not path.is_file():
            failures.append(f"{vendor.upper()} Vship library is missing")
            continue
        try:
            dll_directory = os.add_dll_directory(str(path.parent))
            lib = ctypes.CDLL(str(path))
            _configure_api(lib)
            loaded = _LoadedVship(vendor, path, lib, dll_directory)
            count = ctypes.c_int()
            error = lib.Vship_GetDeviceCount(ctypes.byref(count))
            if error != 0:
                failures.append(f"{vendor.upper()}: {_message(lib, error)}")
                continue
            for gpu_id in range(count.value):
                error = lib.Vship_GPUFullCheck(gpu_id)
                if error != 0:
                    failures.append(f"{vendor.upper()} GPU {gpu_id}: {_message(lib, error)}")
                    continue
                info = _DeviceInfo()
                error = lib.Vship_GetDeviceInfo(ctypes.byref(info), gpu_id)
                if error != 0:
                    failures.append(f"{vendor.upper()} GPU {gpu_id}: {_message(lib, error)}")
                    continue
                version = lib.Vship_GetVersion()
                name = bytes(info.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
                version_text = f"{version.major}.{version.minor}.{version.minorMinor}"
                return VshipDevice(vendor, name or f"{vendor.upper()} GPU {gpu_id}", gpu_id,
                                   version_text, loaded), ""
            if count.value == 0:
                failures.append(f"No {vendor.upper()} Vship GPU was detected")
        except (OSError, AttributeError, TypeError) as error:
            failures.append(f"{vendor.upper()} Vship library unavailable: {error}")
    return None, "; ".join(failures) or "No supported NVIDIA CUDA or AMD HIP GPU was detected."


def _image_format(info: VideoInfo) -> _ImageFormat:
    name = (info.pix_fmt or "").strip().casefold()
    if name in {"nv12", "nv21"}:
        return _ImageFormat("yuv420p", 0, _VSHIP_ENUMS[8], 1, 1)
    packed_yuv = {"p010le": ("420", 10), "p016le": ("420", 16),
                  "p210le": ("422", 10), "p216le": ("422", 16),
                  "p410le": ("444", 10), "p416le": ("444", 16)}
    if name in packed_yuv:
        sampling, depth = packed_yuv[name]
        return _format_yuv(sampling, depth)

    yuv = re.fullmatch(r"yuvj?(410|411|420|422|440|444)p(?:(9|10|12|14|16)(?:le|be)?)?", name)
    if yuv:
        sampling = yuv.group(1)
        depth = int(yuv.group(2) or 8)
        image = _format_yuv(sampling, depth)
        return replace(image, full_range=name.startswith("yuvj"))

    rgb_depth = None
    rgb = re.fullmatch(r"gbrp(?:(9|10|12|14|16)(?:le|be)?)?", name)
    if name in {"rgb24", "bgr24", "rgba", "bgra", "argb", "abgr", "rgb0", "bgr0"}:
        rgb = re.fullmatch(r".+", name)
    elif name in {"rgb48le", "rgb48be", "rgba64le", "rgba64be"}:
        rgb = re.fullmatch(r".+", name)
        rgb_depth = 16
    else:
        rgb_depth = None
    if rgb:
        depth = rgb_depth or (int(rgb.group(1) or 8) if rgb.lastindex else 8)
        if depth not in _VSHIP_ENUMS:
            raise VshipUnavailableError(f"Vship does not support the {depth}-bit RGB format {info.pix_fmt}.")
        fmt = "gbrp" + (f"{depth}le" if depth > 8 else "")
        return _ImageFormat(fmt, 1, _VSHIP_ENUMS[depth], 0, 0, True, (2, 0, 1))
    raise VshipUnavailableError(f"Vship does not support the decoded pixel format {info.pix_fmt or '(unknown)'}.")


def _format_yuv(sampling: str, depth: int) -> _ImageFormat:
    if depth not in _YUV_LAYOUTS.get(sampling, set()):
        raise VshipUnavailableError(f"Vship does not support {sampling} {depth}-bit YUV video.")
    subw, subh = {
        "410": (2, 1), "411": (2, 0), "420": (1, 1),
        "422": (1, 0), "440": (0, 1), "444": (0, 0),
    }[sampling]
    suffix = "" if depth == 8 else f"{depth}le"
    return _ImageFormat(f"yuv{sampling}p{suffix}", 0, _VSHIP_ENUMS[depth], subw, subh)


def _vship_colorspace(info: VideoInfo, image: _ImageFormat, width: int, height: int) -> _Colorspace:
    matrix_name = (info.color_space or "").casefold().replace(".", "")
    matrix_values = {
        "rgb": 0, "bt709": 1, "bt470bg": 5, "smpte170m": 6,
        "bt2020nc": 9, "bt2020ncl": 9, "bt2020c": 10, "bt2020cl": 10,
        "ictcp": 14,
    }
    if image.family == 1:
        matrix = 0
    elif matrix_name in {"", "unknown", "unspecified", "reserved"}:
        matrix = 1 if height > 650 else 5
    elif matrix_name in matrix_values:
        matrix = matrix_values[matrix_name]
    else:
        raise VshipUnavailableError(f"Vship does not support the {info.color_space} color matrix.")
    if matrix == 14:
        raise VshipUnavailableError("Vship's current GPU metric API does not support ICtCp input.")

    transfer_name = (info.color_transfer or "").casefold().replace(".", "").replace("-", "")
    transfer_values = {
        "bt709": 1, "gamma22": 4, "gamma28": 5, "smpte170m": 6,
        "linear": 8, "iec6196621": 13, "smpte2084": 16,
        "smpte428": 17, "aribstdb67": 18,
    }
    if transfer_name in {"", "unknown", "unspecified", "reserved"}:
        transfer = 5 if matrix == 5 else 16 if matrix in {9, 10} else 1
    elif transfer_name in transfer_values:
        transfer = transfer_values[transfer_name]
    else:
        raise VshipUnavailableError(f"Vship does not support the {info.color_transfer} transfer function.")

    primaries_name = (info.color_primaries or "").casefold().replace(".", "")
    primaries_values = {"bt709": 1, "bt470m": 4, "bt470bg": 5, "bt2020": 9}
    if primaries_name in {"", "unknown", "unspecified", "reserved"}:
        primaries = 5 if matrix == 5 else 9 if matrix in {9, 10} else 1
    elif primaries_name in primaries_values:
        primaries = primaries_values[primaries_name]
    else:
        raise VshipUnavailableError(f"Vship does not support the {info.color_primaries} color primaries.")

    range_name = (info.color_range or "").casefold()
    if range_name in {"pc", "jpeg", "full"} or (not range_name and image.full_range):
        value_range = 1
    elif range_name in {"", "unknown", "unspecified", "tv", "mpeg", "limited"}:
        value_range = 1 if image.family == 1 else 0
    else:
        raise VshipUnavailableError(f"Vship does not support the {info.color_range} range tag.")
    location = (info.chroma_location or "left").casefold().replace("-", "")
    locations = {"left": 0, "center": 1, "topleft": 2, "top": 3,
                 "unspecified": 0, "unknown": 0}
    if location not in locations:
        raise VshipUnavailableError(f"Vship does not support {info.chroma_location} chroma siting.")

    return _Colorspace(
        width, height, -1, -1, image.sample, value_range,
        _Subsampling(image.subw, image.subh), locations[location], image.family,
        matrix, transfer, primaries, _Crop(0, 0, 0, 0),
    )


def _scaled_sizes(
    source: VideoInfo, distorted: VideoInfo, recipe: ComparisonRecipe,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    source_size = _content_size(source, source_crop)
    distorted_size = _content_size(distorted, distorted_crop)
    if source_size == distorted_size:
        return source_size, distorted_size
    if recipe.scale_direction is ScaleDirection.DISTORTED_TO_SOURCE:
        return source_size, source_size
    return distorted_size, distorted_size


def _filter_chain(
    info: VideoInfo, crop: CropBox | None, target_size: tuple[int, int],
    pixel_format: str, step: int, algorithm: str,
) -> str:
    operations: list[str] = []
    if crop is not None and not crop.is_noop(info.width, info.height):
        operations.append(crop.as_filter())
    current_size = _content_size(info, crop)
    if current_size != target_size:
        operations.append(f"scale={target_size[0]}:{target_size[1]}:flags={algorithm}")
    if step > 1:
        operations.append(f"select=not(mod(n\\,{step}))")
    operations.extend(("setpts=PTS-STARTPTS", f"format={pixel_format}"))
    return ",".join(operations)


def _read_pinned_frame(
    process: subprocess.Popen, destination: int, frame_bytes: int,
    processes: tuple[subprocess.Popen, ...], cancel_event: threading.Event | None,
) -> bool:
    """Read one exact raw frame into pinned memory, polling for cancellation.

    Keep pipe reads on the calculation thread rather than handing CUDA-pinned
    memory to Python's buffered pipe reader. Reads are incremental, so only two
    frames are resident; ProcessHandle can terminate a blocked FFmpeg read.
    """
    if process.stdout is None:
        raise VshipUnavailableError("FFmpeg did not provide a video frame pipe.")
    offset = 0
    while offset < frame_bytes:
        if cancel_event is not None and cancel_event.is_set():
            for child in processes:
                with contextlib.suppress(OSError):
                    child.terminate()
            raise PerceptualCancelled("Cancelled by user")

        chunk = process.stdout.read(min(1024 * 1024, frame_bytes - offset))
        if not chunk:
            break
        ctypes.memmove(destination + offset, chunk, len(chunk))
        offset += len(chunk)

    if offset == frame_bytes:
        return True
    if offset:
        raise VshipUnavailableError("FFmpeg ended partway through a raw video frame.")
    return False


class _PinnedBuffer:
    def __init__(self, lib: ctypes.CDLL, size: int) -> None:
        self.lib = lib
        self.address = ctypes.c_void_p()
        error = lib.Vship_PinnedMalloc(ctypes.byref(self.address), size)
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
            self.lib.Vship_PinnedFree(self.address)
            self.address = ctypes.c_void_p()


def _init_handler(device: VshipDevice, key: str, src: _Colorspace, dist: _Colorspace) -> _Handler:
    lib = device.loaded.library
    handler = _Handler()
    if key == "ssimulacra2":
        error = lib.Vship_SSIMU2Init(ctypes.byref(handler), src, dist)
    else:
        # Match FFVship's default 2-norm setting and 203-nit target, but graph
        # the 3-norm value to preserve the app's existing Butteraugli meaning.
        error = lib.Vship_ButteraugliInit(ctypes.byref(handler), src, dist, 2, ctypes.c_float(203.0))
    if error != 0:
        raise VshipUnavailableError(f"Could not initialize Vship {key}: {_message(lib, error)}")
    return handler


def _compute_metric(
    device: VshipDevice, key: str, handler: _Handler, source_planes: _PLANES,
    distorted_planes: _PLANES, source_strides: _I64_3, distorted_strides: _I64_3,
) -> float:
    lib = device.loaded.library
    if key == "ssimulacra2":
        score = ctypes.c_double()
        error = lib.Vship_ComputeSSIMU2(handler, ctypes.byref(score), source_planes,
                                         distorted_planes, source_strides, distorted_strides)
        if error != 0:
            detail = ctypes.create_string_buffer(1024)
            lib.Vship_SSIMU2GetDetailedLastError(handler, detail, len(detail))
            raise VshipUnavailableError(f"Vship SSIMULACRA2 failed: {detail.value.decode(errors='replace') or _message(lib, error)}")
        value = float(score.value)
        if not math.isfinite(value):
            raise VshipUnavailableError("Vship SSIMULACRA2 returned a non-finite value.")
        return value
    score = _ButteraugliScore()
    error = lib.Vship_ComputeButteraugli(handler, ctypes.byref(score), None, 0,
                                          source_planes, distorted_planes,
                                          source_strides, distorted_strides)
    if error != 0:
        detail = ctypes.create_string_buffer(1024)
        lib.Vship_ButteraugliGetDetailedLastError(handler, detail, len(detail))
        raise VshipUnavailableError(f"Vship Butteraugli failed: {detail.value.decode(errors='replace') or _message(lib, error)}")
    value = float(score.norm3)
    if not math.isfinite(value):
        raise VshipUnavailableError("Vship Butteraugli returned a non-finite value.")
    return value


def run_vship_task(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    if not specs or any(spec.backend_id != BACKEND_ID or spec.key not in _METRICS for spec in specs):
        raise ValueError("Vship task requires supported perceptual metric specs")
    if request.recipe.resample_test is not None:
        raise VshipUnavailableError("Vship does not support resolution round-trip tests yet.")
    _validate_pair(source, distorted, request.recipe)
    if cancel_event is not None and cancel_event.is_set():
        raise PerceptualCancelled("Cancelled by user")

    lib = device.loaded.library
    error = lib.Vship_SetDevice(device.gpu_id)
    if error != 0:
        raise VshipUnavailableError(f"Could not select the Vship GPU: {_message(lib, error)}")

    src_format, dist_format = _image_format(source), _image_format(distorted)
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
    if on_status:
        on_status(f"Vship GPU ({device.name}): calculating {', '.join(spec.key.upper() for spec in specs)}…")

    handlers: dict[str, _Handler] = {}
    initialized: list[tuple[str, _Handler]] = []
    src_buffer: _PinnedBuffer | None = None
    dist_buffer: _PinnedBuffer | None = None
    process_list: list[subprocess.Popen] = []
    started = time.perf_counter()
    values: dict[str, list[float]] = {spec.key: [] for spec in specs}
    expected_frames = min(source.estimated_frame_count, distorted.estimated_frame_count)
    if request.recipe.duration_limit > 0:
        expected_frames = min(expected_frames, max(1, math.ceil(request.recipe.duration_limit * source.fps)))
    expected_samples = max(1, math.ceil(expected_frames / step))
    total_units = expected_samples * step

    try:
        src_buffer = _PinnedBuffer(lib, src_frame_bytes)
        dist_buffer = _PinnedBuffer(lib, dist_frame_bytes)
        for spec in specs:
            handler = _init_handler(device, spec.key, src_color, dist_color)
            handlers[spec.key] = handler
            initialized.append((spec.key, handler))

        for info, crop, target, image_format in (
            (source, source_crop, src_size, src_format),
            (distorted, distorted_crop, dist_size, dist_format),
        ):
            filter_chain = _filter_chain(
                info, crop, target, image_format.pixel_format, step,
                request.recipe.scale_algorithm,
            )
            command = [
                ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(info.path.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", filter_chain,
            ]
            if request.recipe.duration_limit > 0:
                command += ["-t", f"{request.recipe.duration_limit:.6f}"]
            command += ["-fps_mode", "passthrough", "-pix_fmt", image_format.pixel_format,
                        "-f", "rawvideo", "pipe:1"]
            try:
                process = proc_util.popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0,
                )
            except OSError as error:
                raise VshipUnavailableError(f"Could not start FFmpeg for Vship: {error}") from error
            process_list.append(process)
            if process_handle is not None:
                process_handle.attach(process.pid)

        frame = 0
        while True:
            source_has_frame = _read_pinned_frame(
                process_list[0], ctypes.addressof(src_buffer.array), src_frame_bytes,
                tuple(process_list), cancel_event,
            )
            distorted_has_frame = _read_pinned_frame(
                process_list[1], ctypes.addressof(dist_buffer.array), dist_frame_bytes,
                tuple(process_list), cancel_event,
            )
            if not source_has_frame and not distorted_has_frame:
                break
            if source_has_frame != distorted_has_frame:
                raise VshipUnavailableError("The source and test produced different frame counts for Vship.")
            source_planes = src_buffer.planes(src_format, src_plane_sizes)
            distorted_planes = dist_buffer.planes(dist_format, dist_plane_sizes)
            for spec in specs:
                values[spec.key].append(_compute_metric(
                    device, spec.key, handlers[spec.key], source_planes,
                    distorted_planes, src_strides, dist_strides,
                ))
            frame += 1
            if on_progress:
                on_progress(min(frame * step, total_units), total_units, 0.0)
            if cancel_event is not None and cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user")

        for process in process_list:
            code = process.wait(timeout=30)
            if process_handle is not None:
                process_handle.detach(process.pid)
            if code != 0:
                raise VshipUnavailableError("FFmpeg failed while streaming frames to Vship.")
        if frame == 0:
            raise VshipUnavailableError("FFmpeg produced no frame pairs for Vship.")

        frame_numbers = np.arange(frame, dtype=np.int32) * step
        times = frame_numbers.astype(np.float64) / max(source.fps, 1.0)
        results = MetricResultSet()
        parameters = {
            "gpu_vendor": device.vendor,
            "gpu_name": device.name,
            "input": "ffmpeg planar frames; native range/transfer and primaries",
            "coverage_step": step,
            "butteraugli_norm": "3-norm",
        }
        for spec in specs:
            results.add(FrameMetricResult(
                spec.key, frame_numbers, times, np.asarray(values[spec.key], dtype=np.float32),
                MetricProvenance(
                    implementation=f"Vship/{spec.key}",
                    implementation_version=f"Vship {device.version}",
                    compute_backend="gpu",
                    implementation_compatibility_id=f"{spec.key}-vship-{device.version}-gpu-v1",
                    parameters=parameters,
                ),
            ))
        elapsed = max(time.perf_counter() - started, 1e-6)
        if on_progress:
            on_progress(frame * step, frame * step, frame / elapsed)
        return PerceptualTaskOutput(results, source_crop, distorted_crop, frame * step)
    except PerceptualCancelled:
        raise
    except VshipUnavailableError:
        raise
    except Exception as error:
        raise VshipUnavailableError(f"Vship GPU calculation failed: {error}") from error
    finally:
        for process in process_list:
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process_handle is not None:
                process_handle.detach(process.pid)
            if process.stdout is not None:
                with contextlib.suppress(OSError):
                    process.stdout.close()
        for key, handler in reversed(initialized):
            with contextlib.suppress(Exception):
                if key == "ssimulacra2":
                    lib.Vship_SSIMU2Free(handler)
                else:
                    lib.Vship_ButteraugliFree(handler)
        if dist_buffer is not None:
            dist_buffer.close()
        if src_buffer is not None:
            src_buffer.close()


def apply_vship_cpu_fallback(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    """Shared fallback entry point; import the CPU code lazily to avoid cycles."""
    from vmaf_app.core.perceptual_cpu import _resolve_crops, run_perceptual_task

    device, reason = detect_vship_device()
    crops = _resolve_crops(
        source, distorted, request.recipe, cancel_event, process_handle, on_status,
    )
    if device is not None:
        try:
            return run_vship_task(
                source, distorted, request, specs, device, *crops,
                on_progress=on_progress, on_status=on_status,
                cancel_event=cancel_event, process_handle=process_handle,
            )
        except PerceptualCancelled:
            raise
        except Exception as error:
            reason = str(error)
    if on_status:
        on_status(f"Vship GPU unavailable ({reason}); using CPU reference metrics…")
    return run_perceptual_task(
        source, distorted, request, specs,
        on_progress=on_progress, on_status=on_status,
        cancel_event=cancel_event, process_handle=process_handle,
        resolved_crops=crops,
    )
