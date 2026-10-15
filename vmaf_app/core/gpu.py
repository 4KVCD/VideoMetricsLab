"""Finds out which GPUs this PC has and how FFmpeg can decode each video on
them (its -hwaccel), and works out pixel formats and bit depths.

The choice is cautious: a -hwaccel likely to work for the video's codec,
with vmaf_runner falling back to software decoding if the hardware path
fails to start. The two videos of a comparison are decided separately.
They are unrelated streams (a 10-bit HEVC master against an AV1 encode of
it, say), so one can be decodable on this PC's GPU while the other is not,
and a single shared answer would have to be "no" whenever either was not.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.models import GpuVendor

#: One GPU pass at a time in the whole app: a Vship pass (see
#: perceptual_vship._gpu_pass) or VMAF on the GPU (vmaf_runner). Each
#: already fills the GPU, and two at once can run a card out of memory.
GPU_PASS = threading.Lock()
#: Sent through on_status while a pass waits for another video's pass to
#: end. Sent with the status kind GPU_WAIT, it lets the run line show which
#: of the video's tasks is waiting.
GPU_WAIT_MESSAGE = "Waiting for the GPU: another video's GPU pass is running…"

# For each -hwaccel, the codecs it reliably decodes through FFmpeg's generic
# hardware decoding.
_HWACCEL_CODEC_SUPPORT = {
    "cuda": {"h264", "hevc", "vp8", "vp9", "mpeg2video", "mpeg4", "av1", "vc1"},
    "qsv": {"h264", "hevc", "vp8", "vp9", "mpeg2video", "av1"},
    "d3d11va": {"h264", "hevc", "vp9", "mpeg2video", "vc1", "av1"},
}

_VENDOR_PREFERRED_HWACCEL = {
    GpuVendor.NVIDIA: "cuda",
    GpuVendor.INTEL: "qsv",
    GpuVendor.AMD: "d3d11va",
}


def start_hwaccel_probe() -> None:
    """Asks FFmpeg for its -hwaccels in the background at the window's
    start. Asked at Video Compare's first opening instead, FFmpeg's start
    would hold the window for 40 ms."""
    threading.Thread(target=available_hwaccels, name="hwaccel-probe", daemon=True).start()


def available_hwaccels() -> set[str]:
    """The -hwaccel methods the FFmpeg in use was built with. The answer is
    kept per FFmpeg, so pointing the app at another one (Settings, or
    "Locate ffmpeg.exe") asks again."""
    return set(_hwaccels_of(ffmpeg_path()))


@lru_cache(maxsize=4)
def _hwaccels_of(executable: str) -> frozenset[str]:
    """The -hwaccel methods one FFmpeg lists."""
    try:
        proc = proc_util.run(
            [executable, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return frozenset()
    lines = [l.strip() for l in proc.stdout.splitlines()]
    names = set()
    started = False
    for line in lines:
        if line.lower().startswith("hardware acceleration methods"):
            started = True
            continue
        if started and line:
            names.add(line)
    return frozenset(names)


#: The GPU makers' PCI vendor IDs, as DXGI and Vulkan report them.
PCI_VENDORS = {0x10DE: GpuVendor.NVIDIA, 0x1002: GpuVendor.AMD, 0x1022: GpuVendor.AMD, 0x8086: GpuVendor.INTEL}


@lru_cache(maxsize=1)
def detected_gpu_vendors() -> list[GpuVendor]:
    """The makers of the GPUs DirectX lists (Windows only): NVIDIA first,
    then Intel, then AMD, the order in which "auto" tries their decoders.
    With VML_GPU_VENDOR set to nvidia, intel or amd, only that maker, as on
    a PC with no other: for testing one maker's route on a PC that has
    several (CUDA_VISIBLE_DEVICES and VK_LOADER_DRIVERS_DISABLE hide the
    others from CUDA and Vulkan).

    DXGI is asked in this process, which takes milliseconds. Don't use a
    PowerShell query (Get-CimInstance Win32_VideoController) instead: it
    takes 0.3 to 0.4 s, or seconds when cold, on the window's thread while
    the window is being built."""
    if sys.platform != "win32":  # not platform.system(): a 50 ms WMI query on Windows
        return []
    try:
        found = {PCI_VENDORS.get(vendor_id) for vendor_id in _dxgi_vendor_ids()}
    except OSError:
        return []
    named = os.environ.get("VML_GPU_VENDOR", "").strip().lower()
    return [vendor for vendor in (GpuVendor.NVIDIA, GpuVendor.INTEL, GpuVendor.AMD)
            if vendor in found and (not named or vendor.name.lower() == named)]


class _Guid(ctypes.Structure):
    """A Windows GUID."""

    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]


class _AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    """DXGI_ADAPTER_DESC1."""

    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32),
                ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuidLow", ctypes.c_uint32),
                ("AdapterLuidHigh", ctypes.c_int32), ("Flags", ctypes.c_uint32)]


_IID_IDXGIFACTORY1 = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1,
                                                                            0xB3, 0x87))
_DXGI_ERROR_NOT_FOUND = -0x7785FFFE  # 0x887A0002 as a signed HRESULT
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
# Vtable slots: IUnknown's three, IDXGIObject's four, then the interface's.
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10


def _com_call(interface: ctypes.c_void_p, slot: int, *args, argtypes=()) -> int:
    """Calls a COM method by its slot in the interface's vtable."""
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return method(interface, *args)


def _dxgi_vendor_ids() -> list[int]:
    """The PCI vendor ID of each hardware adapter DXGI lists. Raises OSError
    when DXGI cannot be asked."""
    factory = ctypes.c_void_p()
    result = ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFACTORY1), ctypes.byref(factory))
    if result < 0 or not factory:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{result & 0xFFFFFFFF:08x}")
    vendors = []
    try:
        for index in range(64):
            adapter = ctypes.c_void_p()
            result = _com_call(factory, _ENUM_ADAPTERS1, index, ctypes.byref(adapter),
                               argtypes=(ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)))
            if result == _DXGI_ERROR_NOT_FOUND:
                break
            if result < 0 or not adapter:
                raise OSError(f"IDXGIFactory1::EnumAdapters1 failed: 0x{result & 0xFFFFFFFF:08x}")
            try:
                description = _AdapterDesc1()
                if (_com_call(adapter, _GET_DESC1, ctypes.byref(description),
                              argtypes=(ctypes.POINTER(_AdapterDesc1),)) >= 0
                        and not description.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE):
                    vendors.append(description.VendorId)
            finally:
                _com_call(adapter, _RELEASE)
    finally:
        _com_call(factory, _RELEASE)
    return vendors


def pick_hwaccel(vendor: GpuVendor, codec_name: str) -> str | None:
    """Returns the FFmpeg -hwaccel to try for a codec on this GPU maker (or
    the first one found, for "auto"), or None for software decoding."""
    hwaccels = available_hwaccels()
    codec_name = (codec_name or "").lower()

    if vendor == GpuVendor.NONE:
        return None

    if vendor == GpuVendor.AUTO:
        for v in detected_gpu_vendors():
            candidate = _VENDOR_PREFERRED_HWACCEL.get(v)
            if candidate and candidate in hwaccels and codec_name in _HWACCEL_CODEC_SUPPORT.get(candidate, set()):
                return candidate
        return None

    candidate = _VENDOR_PREFERRED_HWACCEL.get(vendor)
    if candidate and candidate in hwaccels and codec_name in _HWACCEL_CODEC_SUPPORT.get(candidate, set()):
        return candidate
    return None


@dataclass(frozen=True)
class HwAccelPlan:
    """The FFmpeg -hwaccel to use for each video of one run.

    None for a video means it is decoded in software. Both None is the
    all-CPU plan, which every fallback eventually reaches.
    """

    source: str | None = None
    distorted: str | None = None

    @property
    def uses_gpu(self) -> bool:
        """Whether either video is decoded on the GPU."""
        return self.source is not None or self.distorted is not None

    def describe(self) -> str:
        """The plan for the status line, which shows which video was really
        decoded on the GPU; otherwise a quiet fallback for one video would
        look the same as a run that never tried."""
        if not self.uses_gpu:
            return "off"
        return f"source {self.source or 'cpu'}, distorted {self.distorted or 'cpu'}"


def downloads_from_gpu(pix_fmt: str, width: int = 0, height: int = 0) -> bool:
    """Whether FFmpeg's hardware decoding, as the app runs it (-hwaccel X
    -hwaccel_output_format X, then hwdownload,format=hw_native_format), gives
    a video of this pixel format and size the same pictures as its software
    decoding does.

    The format must be 4:2:0 at 8 or 10 bits (the NV12 and P010 surfaces). A
    4:2:2, 4:4:4 or 12-bit video decodes to another surface, the download
    fails, and every run would start again in software (checked on an RTX
    5090 with HEVC, H.264 and AV1). An unknown format is left to try.

    The width and height must be even. For an odd-sized video (AV1 and VP9
    can be), FFmpeg's NVIDIA decoding gives the decoder's own picture, a
    sample wider or a row taller (854x480 for 853x479), with whatever the
    decoder left in the added column and row, and for an odd height the
    chroma a row out (FFmpeg 9.0.1, RTX 5090: the luma identical to the
    software decode's, the chroma 26 dB PSNR from it). With default
    settings, an 854x479 AV1 pair scored SSIMULACRA2 20.4 where its pictures
    score 45.1. Intel's decoding gave the video's own picture; AMD's is
    unchecked, and a size this rare is not worth a rule per maker. An
    unknown size (0) is left to try."""
    if width & 1 or height & 1:
        return False
    name = (pix_fmt or "").casefold()
    if not name:
        return True
    return name in {"nv12", "p010le", "p010be"} or (
        name.startswith(("yuv420p", "yuvj420p")) and bit_depth(name) <= 10)


def pick_decode(vendor: GpuVendor, codec_name: str, pix_fmt: str, width: int = 0, height: int = 0) -> str | None:
    """Returns the FFmpeg -hwaccel to start this video with, or None for
    software decoding: pick_hwaccel's choice, for a format and size that
    downloads_from_gpu accepts, and for H.264 above 8 bits only NVIDIA's.

    Intel's and AMD's GPUs do not decode 10-bit H.264. Asked to with
    -hwaccel qsv or d3d11va, FFmpeg decodes it in software, the frames reach
    hwdownload in system memory, and every run would fail after 0.3 s
    ("Error reinitializing filters!" on a Radeon 780M, "Invalid argument"
    with FFmpeg 9 here) and start again in software. NVIDIA's RTX 50 series
    does decode it (on an RTX 5090, the software decode's pictures, by MD5);
    an older card fails and starts again in software."""
    if not downloads_from_gpu(pix_fmt, width, height):
        return None
    hwaccel = pick_hwaccel(vendor, codec_name)
    if hwaccel != "cuda" and (codec_name or "").casefold() == "h264" and pix_fmt and bit_depth(pix_fmt) > 8:
        return None
    return hwaccel


def plan_hwaccel(
    vendor: GpuVendor, source_codec: str, distorted_codec: str | None = None, *,
    source_pix_fmt: str = "", distorted_pix_fmt: str = "",
    source_size: tuple[int, int] = (0, 0), distorted_size: tuple[int, int] = (0, 0),
) -> HwAccelPlan:
    """Chooses FFmpeg's hardware decoding for each video separately: by
    codec, and only for a pixel format and size (width, height) that it
    decodes as the software decoder does (pick_decode).

    `distorted_codec` is None for a resolution test, which has only one
    file, so there is nothing to decide for the test video.
    """
    return HwAccelPlan(
        source=pick_decode(vendor, source_codec, source_pix_fmt, *source_size),
        distorted=(pick_decode(vendor, distorted_codec, distorted_pix_fmt, *distorted_size)
                   if distorted_codec is not None else None),
    )


# ---------------------------------------------------------------- pixel formats
# These are here rather than in vmaf_runner because black-bar detection
# needs them too, and vmaf_runner imports crop_detect.

#: For each comparison bit depth, the planar 4:2:0 format both videos are
#: converted to before they meet. libvmaf compares two streams that must
#: have the same format, so one is picked for the pair.
_ANALYSIS_FORMAT_BY_DEPTH = {8: "yuv420p", 10: "yuv420p10le", 12: "yuv420p12le"}


def analysis_pix_fmt(*pix_fmts: str) -> str:
    """Returns the common format the videos are converted to before they are
    compared.

    It takes the deepest of the videos, so a 10-bit master compared with an
    8-bit encode raises the encode to 10 bits rather than cutting the master
    to 8. Don't force 8-bit yuv420p: on an HDR or 10-bit comparison it
    throws away two bits of both videos, and puts a floor under PSNR and
    XPSNR that has nothing to do with the encode being measured.
    """
    depth = max((bit_depth(f) for f in pix_fmts), default=8)
    if depth <= 8:
        return _ANALYSIS_FORMAT_BY_DEPTH[8]
    if depth <= 10:
        return _ANALYSIS_FORMAT_BY_DEPTH[10]
    # libvmaf accepts up to 12 bits, so deeper sources (16-bit intermediates)
    # are compared at 12 rather than dropped back to 8.
    return _ANALYSIS_FORMAT_BY_DEPTH[12]


def hw_native_format(pix_fmt: str) -> str:
    """Returns the system-memory pixel format that a cuda, qsv or d3d11va
    surface downloads to, by the video's bit depth: video deeper than 8 bits
    decodes to a P010-family surface, and everything else to NV12."""
    if bit_depth(pix_fmt) > 8:
        return "p010le"
    return "nv12"


#: The depth digits come right after the planar "p", at the end of the
#: name: yuv420p10le, gbrp12be, yuv444p16le. Matching on that "p" keeps nv12
#: (8-bit semi-planar, whose 12 is part of the name) and rgb24 (8-bit, whose
#: 24 is bits per pixel) from being read as deep.
_PLANAR_DEPTH_RE = re.compile(r"p(\d{1,2})(?:le|be)?$")
#: Single-plane formats put the digits straight after the plane name.
_GRAY_DEPTH_RE = re.compile(r"^(?:gray|ya)(\d{1,2})(?:le|be)?$")
#: The semi-planar hardware-surface formats: digits in the middle, and
#: "p010" means 10 significant bits stored in 16.
_HW_SURFACE_DEPTHS = (("p016", 16), ("p012", 12), ("p010", 10))


def bit_depth(pix_fmt: str) -> int:
    """Returns the bit depth written in an FFmpeg pixel format's name.

    FFmpeg puts the depth in the name rather than reporting it separately,
    and the 8-bit names have no depth digits at all. Anything not recognised
    is taken as 8-bit, which is the safe direction: it costs precision only
    for a format the app would not handle specially anyway.
    """
    name = (pix_fmt or "").lower()
    if not name:
        return 8
    for token, depth in _HW_SURFACE_DEPTHS:
        if name.startswith(token):
            return depth
    for pattern in (_PLANAR_DEPTH_RE, _GRAY_DEPTH_RE):
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return 8


#: FFmpeg's pixel format for each -hwaccel's frames, where it is not the
#: hwaccel's own name (cuda and qsv are both).
_HWACCEL_OUTPUT_FORMATS = {"d3d11va": "d3d11"}


def hwaccel_output_format(hwaccel: str) -> str:
    """Returns the -hwaccel_output_format that keeps `hwaccel`'s frames on
    the GPU for hwdownload: the name of its pixel format.

    Don't use the hwaccel's own name: for d3d11va (AMD's) it is not a pixel
    format. FFmpeg then says "Unrecognised hwaccel output format: d3d11va"
    and gives the frames in system memory, the graph's hwdownload refuses
    them, and every run on an AMD GPU starts again in software (found on a
    Radeon 780M)."""
    return _HWACCEL_OUTPUT_FORMATS.get(hwaccel, hwaccel)


#: -hwaccels whose decoding FFmpeg runs on one thread (hwaccel_args).
_ONE_THREAD_HWACCELS = frozenset({"cuda"})


def hwaccel_args(hwaccel: str | None) -> list[str]:
    """Returns the -hwaccel options for one video. FFmpeg reads them as
    options of the next -i on the command line, which lets the two videos
    be decoded differently.

    NVIDIA's decoding runs on one decoder thread. FFmpeg's frame threads
    each hold pictures of their own, and a GPU decoding the video gains
    nothing from them: a 4K HEVC 10-bit film decoded and downloaded as fast
    in 260 MB less memory, and 1080p in 50 MB less; XPSNR beside PSNR and
    SSIM took 0.8 GB instead of 1.3 GB. Intel's took the same memory either
    way. A video that NVIDIA's decoder does not take is not left to decode
    slowly on one CPU thread: the frames are kept on the GPU here
    (hwaccel_output_format) for a graph that downloads them, so the run
    fails, and is made again without -hwaccel."""
    if not hwaccel:
        return []
    threads = ["-threads", "1"] if hwaccel in _ONE_THREAD_HWACCELS else []
    return [*threads, "-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel_output_format(hwaccel)]
