"""GPU implementation of the frame-based perceptual metrics via Vship's C API.

FFmpeg remains responsible for decoding, cropping, sampling, and scaling. It
streams tightly packed frames into rings of pinned host buffers (see
_FrameStream); Vship only does the metric computation on a supported NVIDIA
CUDA or AMD HIP device. A hardware-decoded frame crosses the pipe in the
decoder's own NV12/P010 layout and only its chroma is split into planes here
(see _passthrough_format). Because FFmpeg decodes, every codec it supports works,
VVC included, and hardware decode is used per input where the GPU has one.
This avoids bundling FFVship/FFMS2 executables and keeps video decode behavior
under the same FFmpeg installation used by the rest of the app.
"""
from __future__ import annotations

import contextlib
import ctypes
import math
import os
import queue
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
from vmaf_app.core.gpu import hw_native_format, hwaccel_args, pick_hwaccel
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.models import CropBox, GpuVendor, ScaleDirection, VideoInfo
from vmaf_app.core.perceptual_cpu import (
    LONG_CPU_RUN_SECONDS,
    PerceptualCancelled,
    PerceptualRunError,
    PerceptualTaskOutput,
    _content_size,
    _validate_pair,
    compared_seconds,
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
    (v << 6) / 256 and v / 4 are the same float. Full range divides by
    2^depth - 1 instead, where the two differ, so full-range 10-bit keeps
    FFmpeg's conversion. 8-bit NV12 is exact in either range.
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
    pixel_format: str, step: int, algorithm: str, hwaccel: str | None = None,
) -> str:
    operations: list[str] = []
    if hwaccel:
        # A hardware-decoded surface is brought to system memory in its
        # native layout first; crop, scale and the final format conversion
        # then run exactly as they do for a software-decoded input, so the
        # pictures Vship scores do not depend on which decoder produced them.
        operations.append(f"hwdownload,format={hw_native_format(info.pix_fmt)}")
    if crop is not None and not crop.is_noop(info.width, info.height):
        operations.append(crop.as_filter())
    current_size = _content_size(info, crop)
    if current_size != target_size:
        operations.append(f"scale={target_size[0]}:{target_size[1]}:flags={algorithm}")
    if step > 1:
        operations.append(f"select=not(mod(n\\,{step}))")
    operations.extend(("setpts=PTS-STARTPTS", f"format={pixel_format}"))
    return ",".join(operations)


#: Concurrent Vship handlers per metric. One handler leaves the GPU idle
#: between its own kernels; two keep it busy. Measured at 4K 10-bit on an
#: RTX 5090: SSIMULACRA2 217 -> 277 pairs/s, Butteraugli 82 -> 95, both
#: metrics together 61 -> 72. This is also how FFVship runs Vship.
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
_PIPE_BYTES = 64 * 1024 * 1024
_EOF = -1


def _spawn_raw_ffmpeg(command: list[str]) -> tuple[subprocess.Popen, object]:
    """Start FFmpeg writing raw frames to a large pipe; returns (process, reader)."""
    if os.name == "nt":
        import _winapi
        import msvcrt

        read_handle, write_handle = _winapi.CreatePipe(None, _PIPE_BYTES)
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
        try:
            process = proc_util.popen(
                command, stdin=subprocess.DEVNULL, stdout=write_fd, stderr=subprocess.PIPE,
            )
        except BaseException:
            os.close(write_fd)
            _winapi.CloseHandle(read_handle)
            raise
        os.close(write_fd)  # the child holds its own copy; EOF arrives when it exits
        reader = open(msvcrt.open_osfhandle(read_handle, os.O_RDONLY), "rb", buffering=0)  # noqa: SIM115
        return process, reader
    process = proc_util.popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    return process, process.stdout


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


class _FrameStream:
    """One input's FFmpeg decode, feeding a ring of pinned frame buffers.

    A background thread reads frames into free slots and hands them over in
    order; the scoring thread returns each slot once Vship is done with it.
    Both the pipe read and the Vship call release the GIL, so the two inputs
    and the GPU all make progress at once. If hardware decode fails before
    delivering a frame, the same pictures are decoded in software instead.
    """

    def __init__(
        self, lib: ctypes.CDLL, frame_bytes: int, commands: list[list[str]],
        process_handle: ProcessHandle | None, label: str,
        interleaved_chroma: tuple[int, int, type[np.integer]] | None = None,
    ) -> None:
        self.buffers = [_PinnedBuffer(lib, frame_bytes) for _ in range(_RING_SLOTS)]
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
        self._free: queue.Queue[int] = queue.Queue()
        self._filled: queue.Queue[int | BaseException] = queue.Queue()
        for slot in range(_RING_SLOTS):
            self._free.put(slot)
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._reader = None
        self.used_hardware = False
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
                    self.used_hardware = attempt == 0 and len(self._commands) > 1
                    self._filled.put(_EOF)
                    return
                if frames == 0 and attempt + 1 < len(self._commands):
                    continue  # hardware decode refused this stream: decode it in software
                raise VshipUnavailableError(
                    f"FFmpeg failed while decoding the {self._label} for Vship"
                    + (f": {stderr}" if stderr else ".")
                )
        except BaseException as error:  # handed to the scoring thread, never lost
            self._filled.put(error)

    def _decode(self, command: list[str]) -> tuple[int, int, str]:
        try:
            process, reader = _spawn_raw_ffmpeg(command)
        except OSError as error:
            raise VshipUnavailableError(f"Could not start FFmpeg for Vship: {error}") from error
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
        try:
            while not self._stopping.is_set():
                slot = self._free.get()
                if slot == _EOF:
                    break
                received = self._fill(reader, slot)
                if received == len(self.views[slot]):
                    self._filled.put(slot)
                    frames += 1
                    continue
                self._free.put(slot)
                if received:
                    raise VshipUnavailableError(f"FFmpeg ended partway through a {self._label} frame.")
                break
        finally:
            with contextlib.suppress(OSError):
                reader.close()
            code = process.wait()
            drain.join(timeout=5)
            if self._process_handle is not None:
                self._process_handle.detach(process.pid)
        message = b"".join(stderr_tail).decode("utf-8", errors="replace").strip()
        return frames, code, message

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
                process.terminate()
        self._thread.join(timeout=10)
        # Pinned memory is freed only once the reader cannot be writing to it.
        if not self._thread.is_alive():
            for buffer in self.buffers:
                buffer.close()


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
                 scores: _ScoreArray, finished: Callable[[int], None],
                 failed: Callable[[BaseException], None], abort: threading.Event) -> None:
        self.key = key
        self._args = (device, src, dist)
        self._planes = (source_planes, distorted_planes)
        self._strides = (src_strides, dist_strides)
        self._scores, self._finished, self._failed, self._abort = scores, finished, failed, abort
        self.jobs: queue.Queue[tuple[int, int, int] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"vship-{key}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        device, src, dist = self._args
        lib = device.loaded.library
        handler = None
        try:
            # The CUDA/HIP device is per thread, and a handler allocates on
            # the device current when it is created.
            error = lib.Vship_SetDevice(device.gpu_id)
            if error != 0:
                raise VshipUnavailableError(f"Could not select the Vship GPU: {_message(lib, error)}")
            handler = _init_handler(device, self.key, src, dist)
            while True:
                job = self.jobs.get()
                if job is None or self._abort.is_set():
                    return
                index, source_slot, distorted_slot = job
                self._scores[index] = _compute_metric(
                    device, self.key, handler, self._planes[0][source_slot],
                    self._planes[1][distorted_slot], *self._strides,
                )
                self._finished(index)
        except BaseException as error:
            self._failed(error)
        finally:
            if handler is not None:
                with contextlib.suppress(Exception):
                    (lib.Vship_SSIMU2Free if self.key == "ssimulacra2" else lib.Vship_ButteraugliFree)(handler)

    def stop(self) -> None:
        self.jobs.put(None)

    def join(self) -> None:
        self._thread.join()


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


#: One Vship pass at a time in the whole app. A pass already fills the GPU:
#: at 4K two passes at once scored no faster than one after the other (84.5
#: vs 83.6 pairs/s) while holding twice the VRAM (9.1 vs 4.5 GB, more than
#: most cards have). Running two jobs in parallel still pays off -- 23% at
#: 1080p, 27% at 4K -- because one job's VMAF/PSNR/SSIM pass, which is CPU
#: work, overlaps the other's Vship pass; only this GPU pass is serialized,
#: never crop detection or the FFmpeg metrics.
_gpu_pass = threading.Lock()


def run_vship_task(
    source: VideoInfo, distorted: VideoInfo, request: AnalysisRequest,
    specs: tuple[MetricRequestSpec, ...], device: VshipDevice,
    source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    on_progress: Callable[[int, int, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> PerceptualTaskOutput:
    """Scores `specs` on the GPU once no other Vship pass is running."""
    if not _gpu_pass.acquire(blocking=False):
        if on_status:
            on_status("Waiting for the GPU: another video's SSIMULACRA2/Butteraugli pass is running…")
        while not _gpu_pass.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise PerceptualCancelled("Cancelled by user")
    try:
        return _run_vship_pass(
            source, distorted, request, specs, device, source_crop, distorted_crop,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
        )
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

    # Decode follows the row's GPU-decode setting, per input and per codec:
    # NVDEC for HEVC/AV1/H.264 on NVIDIA, software where the GPU has no
    # decoder (VVC). Hardware decode is bit-exact, so it changes speed only.
    vendor = request.execution.gpu_vendor if request.execution.gpu_decode else GpuVendor.NONE
    src_hwaccel = pick_hwaccel(vendor, source.codec_name)
    dist_hwaccel = pick_hwaccel(vendor, distorted.codec_name)
    src_passthrough = _passthrough_format(source, src_hwaccel)
    dist_passthrough = _passthrough_format(distorted, dist_hwaccel)
    src_format = src_passthrough[0] if src_passthrough else _image_format(source)
    dist_format = dist_passthrough[0] if dist_passthrough else _image_format(distorted)
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

    streams: list[_FrameStream] = []
    lanes: list[_MetricLane] = []
    started = time.perf_counter()
    scores: dict[str, _ScoreArray] = {spec.key: _ScoreArray() for spec in specs}
    abort = threading.Event()
    failures: list[BaseException] = []
    pending: dict[int, list[int]] = {}  # frame index -> [metrics left, source slot, test slot]
    pending_lock = threading.Lock()
    expected_frames = min(source.estimated_frame_count, distorted.estimated_frame_count)
    if request.recipe.duration_limit > 0:
        expected_frames = min(expected_frames, max(1, math.ceil(request.recipe.duration_limit * source.fps)))
    expected_samples = max(1, math.ceil(expected_frames / step))
    total_units = expected_samples * step

    def commands(info: VideoInfo, crop: CropBox | None, target: tuple[int, int],
                 hwaccel: str | None, pixel_format: str) -> list[list[str]]:
        # The software retry pipes the same layout as the hardware attempt:
        # Vship's handlers are set up for one layout per input.
        attempts = []
        for accel in ([hwaccel, None] if hwaccel else [None]):
            filter_chain = _filter_chain(
                info, crop, target, pixel_format, step,
                request.recipe.scale_algorithm, accel,
            )
            command = [
                ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin",
                *hwaccel_args(accel),
                "-i", str(info.path.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", filter_chain,
            ]
            if request.recipe.duration_limit > 0:
                command += ["-t", f"{request.recipe.duration_limit:.6f}"]
            command += ["-fps_mode", "passthrough", "-pix_fmt", pixel_format,
                        "-f", "rawvideo", "pipe:1"]
            attempts.append(command)
        return attempts

    def failed(error: BaseException) -> None:
        failures.append(error)
        abort.set()

    try:
        def stream(info, crop, size, hwaccel, image_format, passthrough, frame_bytes, plane_sizes, label):
            split = None
            if passthrough is not None:
                dtype = np.uint16 if image_format.sample != _VSHIP_ENUMS[8] else np.uint8
                split = (plane_sizes[0], plane_sizes[1], dtype)
            pixel_format = passthrough[1] if passthrough else image_format.pixel_format
            return _FrameStream(lib, frame_bytes, commands(info, crop, size, hwaccel, pixel_format),
                                process_handle, label, split)

        streams = [
            stream(source, source_crop, src_size, src_hwaccel, src_format, src_passthrough,
                   src_frame_bytes, src_plane_sizes, "reference"),
            stream(distorted, distorted_crop, dist_size, dist_hwaccel, dist_format, dist_passthrough,
                   dist_frame_bytes, dist_plane_sizes, "test video"),
        ]
        for stream in streams:
            stream.start()
        source_stream, distorted_stream = streams
        source_planes = [buffer.planes(src_format, src_plane_sizes) for buffer in source_stream.buffers]
        distorted_planes = [buffer.planes(dist_format, dist_plane_sizes) for buffer in distorted_stream.buffers]

        def finished(index: int) -> None:
            with pending_lock:
                entry = pending[index]
                entry[0] -= 1
                if entry[0]:
                    return
                del pending[index]
            source_stream.release(entry[1])
            distorted_stream.release(entry[2])

        by_metric: dict[str, list[_MetricLane]] = {}
        for spec in specs:
            by_metric[spec.key] = [
                _MetricLane(device, spec.key, src_color, dist_color, source_planes, distorted_planes,
                            src_strides, dist_strides, scores[spec.key], finished, failed, abort)
                for _ in range(_LANES_PER_METRIC)
            ]
            lanes.extend(by_metric[spec.key])
        for lane in lanes:
            lane.start()

        frame = 0
        while True:
            if failures:
                raise failures[0]
            try:
                source_slot = source_stream.next(cancel_event, abort)
                distorted_slot = distorted_stream.next(cancel_event, abort)
            except _LaneFailedError:
                raise failures[0] from None
            if source_slot == _EOF or distorted_slot == _EOF:
                # The shorter input has ended: the comparison is the frames
                # both have, exactly as libvmaf scores it (framesync with
                # shortest=1). Refusing here failed any pair a frame or two
                # apart -- and with it the whole job, VMAF included. The
                # longer input's reader is stopped when the streams close.
                for stream, slot in ((source_stream, source_slot), (distorted_stream, distorted_slot)):
                    if slot != _EOF:
                        stream.release(slot)
                break
            with pending_lock:
                pending[frame] = [len(specs), source_slot, distorted_slot]
            for spec in specs:
                scores[spec.key].reserve(frame)
            for spec in specs:
                by_metric[spec.key][frame % _LANES_PER_METRIC].jobs.put((frame, source_slot, distorted_slot))
            frame += 1
            if on_progress:
                elapsed = max(time.perf_counter() - started, 1e-6)
                on_progress(min(frame * step, total_units), total_units, frame / elapsed)
        for lane in lanes:
            lane.stop()
        for lane in lanes:
            lane.join()
        if failures:
            raise failures[0]
        if frame == 0:
            raise VshipUnavailableError("FFmpeg produced no frame pairs for Vship.")

        frame_numbers = np.arange(frame, dtype=np.int32) * step
        times = frame_numbers.astype(np.float64) / max(source.fps, 1.0)
        results = MetricResultSet()
        parameters = {
            "gpu_vendor": device.vendor,
            "gpu_name": device.name,
            "input": "ffmpeg frames (hardware-decoded NV12/P010 split into planes); "
                     "native range/transfer and primaries",
            "coverage_step": step,
            "butteraugli_norm": "3-norm",
        }
        for spec in specs:
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
) -> PerceptualTaskOutput:
    """Run selected backends, with a per-metric GPU-to-CPU fallback."""
    from vmaf_app.core.perceptual_cpu import _resolve_crops, run_perceptual_task

    cpu_specs = tuple(spec for spec in specs if request.execution.perceptual_backend(spec.key) == "cpu")
    gpu_specs = tuple(spec for spec in specs if request.execution.perceptual_backend(spec.key) == "gpu")
    if len(cpu_specs) + len(gpu_specs) != len(specs):
        raise ValueError("perceptual metric backend must be either GPU or CPU")

    # An explicit CPU selection must not probe Vship or touch a compute GPU.
    if not gpu_specs:
        return run_perceptual_task(
            source, distorted, request, cpu_specs,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
        )

    device, reason = detect_vship_device()
    if device is None:
        if on_status:
            on_status(f"Vship GPU unavailable ({reason}); using CPU reference metrics…")
        return run_perceptual_task(
            source, distorted, request, specs,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
        )

    crops = _resolve_crops(
        source, distorted, request.recipe, cancel_event, process_handle, on_status,
    )
    try:
        gpu_output = run_vship_task(
            source, distorted, request, gpu_specs, device, *crops,
            on_progress=(
                (lambda cur, total, fps: on_progress(cur, total * 2, fps))
                if cpu_specs and on_progress else on_progress
            ),
            on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle,
        )
    except PerceptualCancelled:
        raise
    except Exception as error:
        if cancel_event is not None and cancel_event.is_set():
            raise PerceptualCancelled("Cancelled by user") from error
        if compared_seconds(source, distorted, request.recipe.duration_limit) > LONG_CPU_RUN_SECONDS:
            # Nobody agreed to a CPU run of this length: the Videos tab asks
            # before one, but a GPU failure mid-run cannot. Falling back
            # silently meant days of CPU work and terabytes of temporary
            # images for a film. The metric fails instead, with the reason;
            # the video keeps its other metrics.
            labels = " and ".join(metric_definition(spec.key).label for spec in gpu_specs)
            raise PerceptualRunError(
                f"GPU scoring failed ({error}). It was not retried on the CPU, which would take "
                "hours to days and a lot of temporary disk space for a video over 10 minutes. "
                f"Choose CPU for {labels} to calculate it on the CPU anyway."
            ) from error
        if on_status:
            on_status(f"Vship GPU compute failed ({error}); using CPU reference metrics…")
        # Run all metrics together after a GPU failure so CPU frame extraction
        # happens only once, and the returned result remains atomic.
        return run_perceptual_task(
            source, distorted, request, specs,
            on_progress=on_progress, on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle,
            resolved_crops=crops,
        )

    if not cpu_specs:
        return gpu_output

    if on_status:
        on_status("Calculating selected perceptual metric(s) on CPU…")

    def report_cpu_progress(cur: int, total: int, fps: float) -> None:
        if on_progress:
            on_progress(total + cur, total * 2, fps)

    cpu_output = run_perceptual_task(
        source, distorted, request, cpu_specs,
        on_progress=report_cpu_progress,
        on_status=on_status, cancel_event=cancel_event,
        process_handle=process_handle, resolved_crops=crops,
    )
    if gpu_output.compared_frame_count != cpu_output.compared_frame_count:
        raise PerceptualRunError("GPU and CPU perceptual metrics produced different frame counts.")
    combined = MetricResultSet()
    for spec in specs:
        value = gpu_output.metrics.get(spec.key) or cpu_output.metrics.get(spec.key)
        if value is None:
            raise PerceptualRunError(f"Selected backend did not produce {spec.key}.")
        assert value is not None
        combined.add(value)
    return PerceptualTaskOutput(
        combined, gpu_output.source_crop, gpu_output.distorted_crop,
        gpu_output.compared_frame_count,
    )
