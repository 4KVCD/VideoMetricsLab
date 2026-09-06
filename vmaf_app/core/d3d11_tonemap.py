"""Pointer-only bridge to the private D3D11 HDR preview shader.

The shader receives a private converter output texture, never decoder-owned
reference frames. This module does not map, copy or allocate CPU pixel buffers.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "d3d11_tonemap.dll"


def available() -> bool:
    return sys.platform == "win32" and library_path().is_file()


class D3D11ToneMapper:
    def __init__(self, device, kind: str):
        self.device = device
        self.kind = 1 if kind == "HDR10 / PQ" else 2
        self.handle = None
        self.lib = ctypes.CDLL(str(library_path()))
        self.lib.vmaf_tonemap_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.vmaf_tonemap_create.restype = ctypes.c_void_p
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
        # PyGObject miniobject hashes expose the wrapped native pointer. Keep
        # the wrapper alive until the native call returns.
        pointer = hash(memory)
        if not self.gst.gst_is_d3d11_memory(pointer):
            raise RuntimeError("Tone mapper received CPU memory instead of a D3D11 texture")
        self.device.lock()
        try:
            resource = self.gst.gst_d3d11_memory_get_resource_handle(pointer)
            if not self.handle:
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
