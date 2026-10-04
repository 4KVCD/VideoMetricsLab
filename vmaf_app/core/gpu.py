"""GPU / hardware-decode capability detection.

We keep this deliberately conservative: pick a decode hwaccel that's likely
to work for a given video's codec, and let vmaf_runner fall back to
software decode if the hardware path fails to launch.

The two inputs of a comparison are decided independently. They are
unrelated bitstreams -- a 10-bit HEVC master against an AV1 encode of it --
so one can be GPU-decodable on this machine while the other is not, and a
single shared answer would have to be "no" whenever either side was
unsupported.
"""
from __future__ import annotations

import ctypes
import platform
import re
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
#: Sent through on_status while a pass waits for another video's to end; the
#: worker recognises it to say which half of a video is waiting.
GPU_WAIT_MESSAGE = "Waiting for the GPU: another video's GPU pass is running…"

# hwaccel name -> codecs it reliably decodes via ffmpeg's generic hwaccel path
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


def available_hwaccels() -> set[str]:
    """The -hwaccel methods the FFmpeg in use was built with. Asked once
    per FFmpeg: the answer was kept for the session whatever FFmpeg it came
    from, so pointing the app at another one (Settings, or "Locate
    ffmpeg.exe") kept the first one's list."""
    return set(_hwaccels_of(ffmpeg_path()))


@lru_cache(maxsize=4)
def _hwaccels_of(executable: str) -> frozenset[str]:
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
    """The makers of the GPUs DirectX lists (Windows only), NVIDIA first,
    then Intel, then AMD -- the order "auto" tries their decoders in.

    Asked of DXGI in-process, in milliseconds. It was a PowerShell query
    (Get-CimInstance Win32_VideoController): 0.3 to 0.4 s warm, seconds
    cold, on the UI thread while the window was being built."""
    if platform.system() != "Windows":
        return []
    try:
        found = {PCI_VENDORS.get(vendor_id) for vendor_id in _dxgi_vendor_ids()}
    except OSError:
        return []
    return [vendor for vendor in (GpuVendor.NVIDIA, GpuVendor.INTEL, GpuVendor.AMD) if vendor in found]


class _Guid(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]


class _AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
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
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return method(interface, *args)


def _dxgi_vendor_ids() -> list[int]:
    """The PCI vendor ID of each hardware adapter DXGI lists. OSError when
    DXGI cannot be asked."""
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
    """Returns an ffmpeg -hwaccel value to try, or None for software decode."""
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
    """The ffmpeg -hwaccel to use for each input of one run.

    None on either side means "decode this one in software". Both being
    None is the all-CPU plan, which is also what every fallback eventually
    reaches.
    """

    source: str | None = None
    distorted: str | None = None

    @property
    def uses_gpu(self) -> bool:
        return self.source is not None or self.distorted is not None

    def describe(self) -> str:
        """For the status line, so it is visible which input actually got
        hardware decode -- otherwise a silent per-input fallback looks
        identical to a run that never tried."""
        if not self.uses_gpu:
            return "off"
        return f"source {self.source or 'cpu'}, distorted {self.distorted or 'cpu'}"


def downloads_from_gpu(pix_fmt: str, width: int = 0, height: int = 0) -> bool:
    """Whether FFmpeg's hardware decode as the app runs it (-hwaccel X
    -hwaccel_output_format X, then hwdownload,format=hw_native_format) gives
    a video of this pixel format and size as its software decode does.

    The format: 4:2:0 at 8 or 10 bits, the NV12 and P010 surfaces. A 4:2:2,
    4:4:4 or 12-bit video decodes to another surface, the download fails,
    and the run started again in software -- every run (checked on an RTX
    5090 with HEVC, H.264 and AV1). An unknown format is left to try.

    The size: an even width and height. Of an odd-sized video (AV1 and VP9
    can be) FFmpeg's NVIDIA decode gives the decoder's own picture, a sample
    wider or a row taller -- 854x480 for 853x479 -- with whatever the decoder
    left in the added column and row, and for an odd height its chroma a row
    out (FFmpeg 9.0.1, RTX 5090: luma identical to the software decode,
    chroma 26 dB PSNR from it). With default settings an 854x479 AV1 pair
    scored SSIMULACRA2 20.4 where its pictures score 45.1, and 853x480, its
    black bars measured on the padded picture, 16.1 for 44.0. Intel's decode
    gave the video's own picture; AMD's is unchecked, and a size this rare
    is not worth a rule per maker. An unknown size (0) is left to try."""
    if width & 1 or height & 1:
        return False
    name = (pix_fmt or "").casefold()
    if not name:
        return True
    return name in {"nv12", "p010le", "p010be"} or (
        name.startswith(("yuv420p", "yuvj420p")) and bit_depth(name) <= 10)


def plan_hwaccel(
    vendor: GpuVendor, source_codec: str, distorted_codec: str | None = None, *,
    source_pix_fmt: str = "", distorted_pix_fmt: str = "",
    source_size: tuple[int, int] = (0, 0), distorted_size: tuple[int, int] = (0, 0),
) -> HwAccelPlan:
    """Chooses FFmpeg's hardware decode for each input separately: by codec,
    and only for a pixel format and a size (width, height) it gives as the
    software decode does (downloads_from_gpu).

    `distorted_codec` of None is the round-trip-test case: there is only one
    input file, so there is nothing to decide for the distorted side.
    """
    def pick(codec: str, pix_fmt: str, size: tuple[int, int]) -> str | None:
        return pick_hwaccel(vendor, codec) if downloads_from_gpu(pix_fmt, *size) else None

    return HwAccelPlan(
        source=pick(source_codec, source_pix_fmt, source_size),
        distorted=(pick(distorted_codec, distorted_pix_fmt, distorted_size)
                   if distorted_codec is not None else None),
    )


# ---------------------------------------------------------------- pixel formats
# Here rather than in vmaf_runner because crop detection needs them too, and
# vmaf_runner imports crop_detect.

#: Analysis bit depth -> the planar 4:2:0 format both branches are converted
#: to before they meet. libvmaf compares two streams that must agree on
#: format, so one has to be picked for the pair.
_ANALYSIS_FORMAT_BY_DEPTH = {8: "yuv420p", 10: "yuv420p10le", 12: "yuv420p12le"}


def analysis_pix_fmt(*pix_fmts: str) -> str:
    """The common format the inputs are converted to before comparison.

    Takes the *deepest* of the inputs, so a 10-bit master compared against
    an 8-bit encode promotes the encode rather than truncating the master.
    Everything used to be forced to 8-bit yuv420p, which quietly discarded
    two bits of both sides on any HDR/10-bit comparison and put a floor
    under PSNR/XPSNR that had nothing to do with the encode being measured.
    """
    depth = max((bit_depth(f) for f in pix_fmts), default=8)
    if depth <= 8:
        return _ANALYSIS_FORMAT_BY_DEPTH[8]
    if depth <= 10:
        return _ANALYSIS_FORMAT_BY_DEPTH[10]
    # libvmaf accepts up to 12-bit; deeper sources (16-bit intermediates)
    # are analysed at 12 rather than being dropped back to 8.
    return _ANALYSIS_FORMAT_BY_DEPTH[12]


def hw_native_format(pix_fmt: str) -> str:
    """The system-memory pixel format a cuda/qsv/d3d11va hw surface downloads
    to, based on that input's bit depth. 10/12-bit 4:2:0 video decodes to a
    p010-family surface; everything else (including 8-bit) decodes to nv12."""
    if bit_depth(pix_fmt) > 8:
        return "p010le"
    return "nv12"


#: Depth digits sit immediately after the planar marker and at the end of
#: the name: yuv420p10le, gbrp12be, yuv444p16le. Anchoring on that "p" is
#: what keeps nv12 (8-bit semi-planar, whose 12 is part of the *name*) and
#: rgb24 (8-bit, whose 24 is bits per pixel) from being read as deep.
_PLANAR_DEPTH_RE = re.compile(r"p(\d{1,2})(?:le|be)?$")
#: Single-plane formats put the digits straight after the plane name.
_GRAY_DEPTH_RE = re.compile(r"^(?:gray|ya)(\d{1,2})(?:le|be)?$")
#: The semi-planar hardware-surface formats: digits in the middle, and
#: "p010" means 10 significant bits stored in 16.
_HW_SURFACE_DEPTHS = (("p016", 16), ("p012", 12), ("p010", 10))


def bit_depth(pix_fmt: str) -> int:
    """The bit depth encoded in an ffmpeg pixel-format name.

    ffmpeg spells depth into the name rather than reporting it separately,
    and the 8-bit names carry no depth digits at all. Anything unrecognised
    is treated as 8-bit, which is the safe direction: it costs precision
    only for a format the pipeline was never going to handle specially.
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
    """The -hwaccel_output_format that keeps `hwaccel`'s frames on the GPU
    for hwdownload: the name of its pixel format.

    It was the hwaccel's name, which for d3d11va -- AMD's -- is no pixel
    format: FFmpeg said "Unrecognised hwaccel output format: d3d11va", gave
    the frames in system memory, the graph's hwdownload refused them, and
    the run started again in software. Every run on an AMD GPU: right
    scores, never a hardware decode. Found on a Radeon 780M."""
    return _HWACCEL_OUTPUT_FORMATS.get(hwaccel, hwaccel)


def hwaccel_args(hwaccel: str | None) -> list[str]:
    """The -hwaccel options for ONE input. ffmpeg reads these as per-input
    options, applying to the next -i on the command line, which is what
    allows the two inputs to be decoded differently."""
    if not hwaccel:
        return []
    return ["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel_output_format(hwaccel)]
