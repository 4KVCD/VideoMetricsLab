"""Native playback's presenter (d3d11_tonemap.dll): the facts its shader
works from. It shows each frame in the display's colours as it draws it --
HDR video mapped to SDR, or converted to PQ in BT.2020 for an HDR display
(shading()). This module does not map, copy or allocate CPU pixel buffers.
"""
from __future__ import annotations

import ctypes
import functools
import sys
from pathlib import Path

#: CIE xy of the red, green and blue of the primaries the shader converts
#: from, by FFmpeg's names; all with D65 white. Others (DCI-P3's own white,
#: SMPTE 431) are left to FFmpeg's converter.
PRIMARIES = {
    "bt709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "bt2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "smpte432": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),  # Display P3
}
_D65 = (0.3127, 0.3290)
#: The light mapped to SDR white and HLG's display peak; SDR white (nits).
_PEAK_NITS, _WHITE_NITS = 1000.0, 100.0


def library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "d3d11_tonemap.dll"


def available() -> bool:
    return sys.platform == "win32" and library_path().is_file()


def hdr_primaries(tag: str) -> str:
    """An HDR video's primaries by FFmpeg's name: untagged, BT.2020."""
    tag = tag.casefold()
    return "bt2020" if tag in {"", "unknown", "unspecified", "bt.2020"} else tag


@functools.cache
def presents() -> bool:
    """Whether the built DLL has native playback's presenter, its shading
    included (locked_presentation.LockedPresentation): one built before it
    shaded HDR video in each decoder's pipeline."""
    if not available():
        return False
    try:
        return hasattr(ctypes.CDLL(str(library_path())), "vmaf_present_texture")
    except OSError:
        return False


def prepare_presenter() -> None:
    """The presenter's shaders compiled ahead of its first use, once a
    process (vmaf_present_prepare): each presenter compiled them, 10 ms of
    the window's time whenever Video Compare opened a pair."""
    if presents():
        prepare = getattr(ctypes.CDLL(str(library_path())), "vmaf_present_prepare", None)
        if prepare is not None:  # a DLL built before it compiles them for its first presenter
            prepare()


def supports(primaries: str, *, hdr: bool = False) -> bool:
    """Whether HDR video of these primaries is shown right by the shader:
    mapped to SDR, or (`hdr`) for an HDR display, which takes PQ in BT.2020,
    converted to that."""
    return hdr_primaries(primaries) in PRIMARIES and presents()


def shading(kind: str, primaries: str, *, hdr: bool) -> tuple[float, ...]:
    """What the presenter's shader does to HDR video of `kind` ("HDR10 /
    PQ" or "HLG") in `primaries`, as vmaf_present_texture takes it: mapped to
    SDR BT.709, or (`hdr`) kept HDR and converted to PQ in BT.2020. The
    curve, fixed for every video compared: extended Reinhard, 1000 nits to
    SDR white at 100."""
    luma, matrix = conversion(primaries, "bt2020" if hdr else "bt709")
    return (2.0 if hdr else 1.0, 1.0 if kind == "HDR10 / PQ" else 2.0, *luma, *matrix, _PEAK_NITS, _WHITE_NITS)


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
