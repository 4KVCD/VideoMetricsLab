"""Pointer-only bridge to the private D3D11 HDR preview shader.

The shader receives a private converter output texture, never decoder-owned
reference frames. This module does not map, copy or allocate CPU pixel buffers.
"""
from __future__ import annotations

import ctypes
import functools
import sys
from pathlib import Path

#: CIE xy of the red, green and blue of the primaries the shader converts
#: to BT.709 from, by FFmpeg's names; all with D65 white. Others (DCI-P3's
#: own white, SMPTE 431) are left to FFmpeg's converter.
PRIMARIES = {
    "bt709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "bt2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "smpte432": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),  # Display P3
}
_D65 = (0.3127, 0.3290)


def library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "d3d11_tonemap.dll"


def available() -> bool:
    return sys.platform == "win32" and library_path().is_file()


def hdr_primaries(tag: str) -> str:
    """An HDR video's primaries by FFmpeg's name: untagged, BT.2020."""
    tag = tag.casefold()
    return "bt2020" if tag in {"", "unknown", "unspecified", "bt.2020"} else tag


@functools.cache
def _entry_points() -> frozenset[str]:
    """Which of the shader's later entry points the built DLL has: one built
    before them takes every video for BT.2020, and maps it to SDR only."""
    if not available():
        return frozenset()
    try:
        library = ctypes.CDLL(str(library_path()))
    except OSError:
        return frozenset()
    return frozenset(name for name in ("vmaf_tonemap_create_primaries", "vmaf_hdr_convert_create")
                     if hasattr(library, name))


def converts_primaries() -> bool:
    """Whether the built shader maps HDR video of any PRIMARIES to SDR."""
    return "vmaf_tonemap_create_primaries" in _entry_points()


def converts_for_hdr() -> bool:
    """Whether the built shader converts HDR video to BT.2020, HDR kept."""
    return "vmaf_hdr_convert_create" in _entry_points()


def supports(primaries: str, *, hdr: bool = False) -> bool:
    """Whether HDR video of these primaries is shown right: mapped to SDR
    by the shader, or (`hdr`) on an HDR display, which takes BT.2020 --
    other primaries converted to it by the shader."""
    primaries = hdr_primaries(primaries)
    if primaries == "bt2020":
        return True
    return primaries in PRIMARIES and (converts_for_hdr() if hdr else converts_primaries())


def _to_xyz(primaries: str):
    import numpy as np

    columns = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in PRIMARIES[primaries]]).T
    x, y = _D65
    return columns * np.linalg.solve(columns, [x / y, 1.0, (1 - x - y) / y])


def conversion(primaries: str, target: str = "bt709") -> tuple[tuple[float, ...], tuple[float, ...]]:
    """For linear RGB in `primaries`: its luminance weights, and the matrix
    to linear RGB in `target`'s, by rows."""
    import numpy as np

    to_xyz = _to_xyz(hdr_primaries(primaries))
    return tuple(to_xyz[1].tolist()), tuple((np.linalg.inv(_to_xyz(target)) @ to_xyz).ravel().tolist())


def boxed_pointer(boxed) -> int:
    """The native pointer a PyGObject boxed wrapper (a Gst.Memory) holds.

    PyGObject has no public way to give it. Its hash is that pointer, and
    the wrapper stores it right after the Python object header; neither is
    promised, so both are read, and a pointer they do not agree on is
    refused rather than handed to native code, where a wrong one crashes
    the app. Checked against gst_buffer_peek_memory on PyGObject 3.52."""
    try:
        stored = ctypes.c_void_p.from_address(id(boxed) + object.__basicsize__).value
    except (ValueError, OSError) as error:
        raise RuntimeError("Could not read the GPU frame's native pointer") from error
    hashed = hash(boxed) & 0xFFFFFFFFFFFFFFFF
    if not stored or stored != hashed:
        raise RuntimeError("Could not find the GPU frame's native pointer (PyGObject changed)")
    return stored


class D3D11ToneMapper:
    def __init__(self, device, kind: str, primaries: str = "bt2020", *, hdr: bool = False):
        """HDR video of `kind` and `primaries` mapped to SDR BT.709 or, with
        `hdr`, kept HDR and converted to BT.2020 for an HDR display."""
        self.device = device
        self.kind = 1 if kind == "HDR10 / PQ" else 2
        self.hdr = hdr
        self.handle = None
        self.lib = ctypes.CDLL(str(library_path()))
        self.lib.vmaf_tonemap_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.vmaf_tonemap_create.restype = ctypes.c_void_p
        # The entry point given the video's primaries, and its arguments past
        # the frame and kind; a shader built before it is given BT.2020
        # video alone, mapped to SDR (supports()).
        self.colours = None
        floats = ctypes.POINTER(ctypes.c_float)
        if hdr:
            if not hasattr(self.lib, "vmaf_hdr_convert_create"):
                raise RuntimeError("The HDR shader was built before it converted HDR for HDR displays")
            create = self.lib.vmaf_hdr_convert_create
            create.argtypes = [ctypes.c_void_p, ctypes.c_int, floats]
            create.restype = ctypes.c_void_p
            self.colours = create, ((ctypes.c_float * 9)(*conversion(primaries, "bt2020")[1]),)
        elif hasattr(self.lib, "vmaf_tonemap_create_primaries"):
            create = self.lib.vmaf_tonemap_create_primaries
            create.argtypes = [ctypes.c_void_p, ctypes.c_int, floats, floats]
            create.restype = ctypes.c_void_p
            luma, to_bt709 = conversion(primaries)
            self.colours = create, ((ctypes.c_float * 3)(*luma), (ctypes.c_float * 9)(*to_bt709))
        elif hdr_primaries(primaries) != "bt2020":
            raise RuntimeError("The HDR shader was built before it took a video's primaries")
        self.lib.vmaf_tonemap_render.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.vmaf_tonemap_render.restype = ctypes.c_int32
        self.lib.vmaf_tonemap_destroy.argtypes = [ctypes.c_void_p]
        self.lib.vmaf_tonemap_destroy.restype = None
        # GStreamer's Python bundle registers its DLL directory at startup.
        self.gst = ctypes.CDLL("gstd3d11-1.0-0.dll")
        self.gst.gst_is_d3d11_memory.argtypes = [ctypes.c_void_p]
        self.gst.gst_is_d3d11_memory.restype = ctypes.c_int
        self.gst.gst_d3d11_memory_get_resource_handle.argtypes = [ctypes.c_void_p]
        self.gst.gst_d3d11_memory_get_resource_handle.restype = ctypes.c_void_p

    def render(self, buffer) -> None:
        if buffer.n_memory() != 1:
            raise RuntimeError("Tone mapper requires one private RGBA16 GPU texture")
        memory = buffer.peek_memory(0)
        # Kept alive (``memory``) until the native calls return.
        pointer = boxed_pointer(memory)
        if not self.gst.gst_is_d3d11_memory(pointer):
            raise RuntimeError("Tone mapper received CPU memory instead of a D3D11 texture")
        self.device.lock()
        try:
            resource = self.gst.gst_d3d11_memory_get_resource_handle(pointer)
            if not self.handle:
                if self.colours:
                    create, colours = self.colours
                    self.handle = create(resource, self.kind, *colours)
                else:
                    self.handle = self.lib.vmaf_tonemap_create(resource, self.kind)
                if not self.handle:
                    raise RuntimeError("Could not initialize the D3D11 HDR shader")
            result = self.lib.vmaf_tonemap_render(self.handle, resource)
            if result < 0:
                raise RuntimeError(f"D3D11 HDR shader failed: 0x{result & 0xffffffff:08x}")
        finally:
            self.device.unlock()

    def close(self) -> None:
        if self.handle:
            self.lib.vmaf_tonemap_destroy(self.handle)
            self.handle = None
