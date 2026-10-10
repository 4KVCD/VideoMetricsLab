"""Draws film grain on Video Compare's VVC frames, on the GPU.

This wraps d3d11_film_grain.dll (native/d3d11_film_grain.cpp), which
synthesizes H.274 film grain on a decoded frame's planes in place, on
GStreamer's D3D11 device. film_grain works out which grain each frame has;
this module turns that into the parameters the DLL takes.

The synthesis is FFmpeg's, bit for bit, for 8-bit video. FFmpeg does not
synthesize grain above 8 bits; there it follows JVET's reference software,
VTM. Where a message leaves model values out, the standard's defaults are
used (a horizontal cutoff of 8, and a vertical cutoff equal to the
horizontal one); FFmpeg uses zeros there.

Optional: without the DLL, VVC videos play as before, without their grain.
"""
from __future__ import annotations

import ctypes
import functools
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from vmaf_app.core.d3d11_tonemap import object_pointer
from vmaf_app.core.film_grain import FilmGrainH274, FrameGrain

# The DLL's cutoff frequencies are indices into its pattern database:
# H.274's cutoffs 2 to 14 become 0 to 12.
_MIN_CUTOFF, _MAX_CUTOFF = 2, 14


class GrainPlane(ctypes.Structure):
    """One plane's grain, as the DLL takes it (GrainPlane in the C++ code)."""

    _fields_ = [
        ("width", ctypes.c_int),  # the plane's picture size, in samples
        ("height", ctypes.c_int),
        ("intervals", ctypes.c_int),  # how many intensity intervals; 0 for no grain
        ("lower", ctypes.c_uint8 * 256),  # each interval's bounds, in 8-bit codes
        ("upper", ctypes.c_uint8 * 256),
        ("scale", ctypes.c_int16 * 256),  # each interval's grain scale
        ("h", ctypes.c_uint8 * 256),  # its horizontal and vertical cutoff indices (0-12)
        ("v", ctypes.c_uint8 * 256),
    ]


class GrainParams(ctypes.Structure):
    """One frame's grain, as the DLL takes it (GrainParams in the C++ code)."""

    _fields_ = [
        ("depth", ctypes.c_int),  # the video's bit depth: 8 to 12
        ("log2_scale_factor", ctypes.c_int),
        ("poc", ctypes.c_int),  # the frame's picture order count: the pattern's seed
        ("plane", GrainPlane * 3),
    ]


def library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "d3d11_film_grain.dll"


def available() -> bool:
    return sys.platform == "win32" and library_path().is_file()


@functools.cache
def _load_library():
    """The DLL with its functions typed, or None if it is missing or will not load."""
    if not available():
        return None
    try:
        library = ctypes.CDLL(str(library_path()))
    except OSError:
        return None
    params = ctypes.POINTER(GrainParams)
    pointers = ctypes.POINTER(ctypes.c_void_p)
    for name, result_type, argument_types in (
        ("vmaf_grain_prepare", ctypes.c_int, []),
        ("vmaf_grain_create", ctypes.c_void_p, [ctypes.c_void_p]),
        ("vmaf_grain_apply", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, params]),
        ("vmaf_grain_error", ctypes.c_char_p, [ctypes.c_void_p]),
        ("vmaf_grain_destroy", None, [ctypes.c_void_p]),
        ("vmaf_grain_planes", ctypes.c_int, [pointers, pointers, ctypes.c_int, params, ctypes.c_int]),
    ):
        function = getattr(library, name)
        function.restype, function.argtypes = result_type, argument_types
    return library


def plane_sizes(width: int, height: int, chroma_shift: tuple[int, int], planes: int) -> list[tuple[int, int]]:
    """The picture's size in each of its planes.

    `chroma_shift` is the chroma planes' subsampling as powers of two,
    horizontally and vertically: (1, 1) for 4:2:0. Odd sizes round up.
    """
    chroma = (-(-width >> chroma_shift[0]), -(-height >> chroma_shift[1]))
    return [(width, height), chroma, chroma][:planes]


def native_params(frame: FrameGrain, depth: int, sizes: Sequence[tuple[int, int]],
                  chroma_shift: tuple[int, int]) -> GrainParams | None:
    """Turns a frame's grain into the DLL's parameters.

    `sizes` are the frame's plane sizes, `chroma_shift` its chroma
    subsampling (see plane_sizes) and `depth` its bit depth. Returns None if
    the grain cannot be drawn: H.274's autoregressive model (model 1), which
    FFmpeg does not synthesize either.
    """
    grain = frame.grain
    if grain.model_id != 0:
        return None
    params = GrainParams()
    params.depth = depth
    params.log2_scale_factor = grain.log2_scale_factor
    params.poc = frame.seed
    for component, (width, height) in enumerate(sizes[:3]):
        plane = params.plane[component]
        plane.width, plane.height = width, height
        intervals = grain.intervals[component] if component < len(grain.intervals) else ()
        plane.intervals = min(len(intervals), 256)
        for index, (lower, upper, values) in enumerate(intervals[:256]):
            scale, horizontal, vertical = _model_values(values, component > 0, chroma_shift)
            plane.lower[index], plane.upper[index] = lower, upper
            plane.scale[index] = max(-32768, min(32767, scale))
            plane.h[index] = min(max(horizontal, _MIN_CUTOFF), _MAX_CUTOFF) - _MIN_CUTOFF
            plane.v[index] = min(max(vertical, _MIN_CUTOFF), _MAX_CUTOFF) - _MIN_CUTOFF
    return params


def _model_values(values: tuple[int, ...], chroma: bool, chroma_shift: tuple[int, int]) -> tuple[int, int, int]:
    """An interval's (scale, horizontal cutoff, vertical cutoff), adapted to
    a subsampled chroma plane.

    The grain is specified at full resolution, so a subsampled chroma plane
    gets cutoffs doubled in the directions it is subsampled in, and a
    smaller scale: half for 4:2:0 (as FFmpeg and VTM do), 1/sqrt(2) for
    4:2:2 (as VTM does).
    """
    scale = values[0]
    horizontal = values[1] if len(values) > 1 else 8
    vertical = values[2] if len(values) > 2 else horizontal
    if chroma and chroma_shift == (1, 1):
        scale >>= 1
        horizontal *= 2
        vertical *= 2
    elif chroma and chroma_shift == (1, 0):
        scale = (scale * 181) >> 8  # 181 / 256 is 1/sqrt(2)
        horizontal *= 2
    return scale, horizontal, vertical


def prepare() -> None:
    """Builds the grain pattern database and compiles the shaders ahead of
    the first VVC comparison (about 25 ms, once per process), so that
    opening one does not wait for them. Called from the app's start-up
    warm-up thread."""
    library = _load_library()
    if library is not None:
        library.vmaf_grain_prepare()


def check() -> str | None:
    """Runs the synthesis once on a small frame, on Windows' software
    renderer. Returns None if it works, or why it does not. For the self-test."""
    import numpy as np

    if not available():
        return f"{library_path().name} is not built"
    grain = FilmGrainH274(model_id=0, blending_mode_id=0, log2_scale_factor=3,
                          intervals=(((0, 255, (40, 8, 8)),), (), ()), persistence=True)
    params = native_params(FrameGrain(grain, seed=0), 8, [(32, 32)], (1, 1))
    plane = np.full((32, 32), 128, np.uint8)
    try:
        grained = synthesize([plane], params, software=True)[0]
    except (OSError, RuntimeError) as error:
        return str(error)
    return None if not np.array_equal(grained, plane) else "the synthesis drew no grain"


def synthesize(planes: Sequence, params: GrainParams, software: bool = False) -> list:
    """Draws grain on planes held in memory and returns the grained planes.

    `planes` are numpy arrays of codes: uint8 for 8-bit video, uint16 above.
    The synthesis runs on the GPU, or on Windows' software renderer if
    `software` is set (no GPU is needed then). For the tests and the checks
    against FFmpeg; playback draws on GStreamer's frames instead (GrainDrawer).
    """
    import numpy as np

    library = _load_library()
    if library is None:
        raise RuntimeError(f"{library_path().name} is not built")
    dtype = np.uint16 if params.depth > 8 else np.uint8
    inputs = [np.ascontiguousarray(plane, dtype=dtype) for plane in planes]
    outputs = [np.empty_like(plane) for plane in inputs]
    input_pointers = (ctypes.c_void_p * 3)(*[plane.ctypes.data for plane in inputs])
    output_pointers = (ctypes.c_void_p * 3)(*[plane.ctypes.data for plane in outputs])
    result = library.vmaf_grain_planes(input_pointers, output_pointers, len(inputs), ctypes.byref(params),
                                       int(software))
    if result:
        raise RuntimeError(f"film grain synthesis failed ({result})")
    return outputs


def _device_handle(device) -> int:
    """The ID3D11Device behind a GstD3D11Device."""
    get_handle = ctypes.CDLL("gstd3d11-1.0-0.dll").gst_d3d11_device_get_device_handle
    get_handle.restype, get_handle.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
    return get_handle(object_pointer(device))


class GrainDrawer:
    """Draws film grain on one stream's frames on GStreamer's D3D11 device
    (a GstD3D11Device).

    Each call into the DLL holds the device's lock, as GStreamer's own
    elements do when they use the device.
    """

    def __init__(self, device) -> None:
        library = _load_library()
        if library is None:
            raise RuntimeError("film grain is unavailable")
        self._library, self._device = library, device
        self._lock = threading.Lock()
        device.lock()
        try:
            self._native = library.vmaf_grain_create(_device_handle(device))
        finally:
            device.unlock()
        if not self._native:
            raise RuntimeError("could not set up the film grain synthesis on the D3D11 device")

    def draw(self, buffer_pointer: int, params: GrainParams) -> str | None:
        """Draws the grain of `params` on the frame in `buffer_pointer` (a
        GstBuffer of D3D11 textures), in place. Returns None, or why it failed."""
        with self._lock:
            if not self._native:
                return "the film grain drawer is closed"
            self._device.lock()
            try:
                result = self._library.vmaf_grain_apply(self._native, buffer_pointer, ctypes.byref(params))
                error = self._library.vmaf_grain_error(self._native) if result else None
            finally:
                self._device.unlock()
        if result:
            return (error or b"").decode("utf-8", "replace") or f"error {result}"
        return None

    def close(self) -> None:
        """Frees the drawer's GPU resources."""
        with self._lock:
            if self._native:
                self._device.lock()
                try:
                    self._library.vmaf_grain_destroy(self._native)
                finally:
                    self._device.unlock()
                self._native = None
