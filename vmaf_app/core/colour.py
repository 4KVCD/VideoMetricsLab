"""How a video's colours are read for the perceptual metrics: its matrix,
transfer, primaries and range, from its tags -- the H.273 numbers -- with
untagged ones guessed as FFVship (Vship's own tool) guesses them.

One reading for both implementations: Vship is given these numbers on the
GPU (perceptual_vship._vship_colorspace), and the CPU tools get pictures
converted and described the same way (perceptual_cpu). Read differently,
the two scored the same frames far apart: a tagged BT.709 film, 40.4
SSIMULACRA2 on the CPU against 55.1 on the GPU.
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from vmaf_app.core.models import VideoInfo

#: FFmpeg's tag names (as ffprobe prints them, and their aliases) for the
#: values of Vship's enums in VshipColor.h -- the H.273 numbers. The
#: mapping is FFVship's (src/ffvship_utility/ffmpegToVshipColorFormat.hpp,
#: Vship 5.1.1); a tag missing here has no Vship value.
MATRICES = {"gbr": 0, "rgb": 0, "bt709": 1, "bt470bg": 5, "smpte170m": 6, "ycgco": 8, "ycocg": 8,
            "bt2020nc": 9, "bt2020ncl": 9, "bt2020c": 10, "bt2020cl": 10, "ictcp": 14,
            "ycgco-re": 16, "ycgco-ro": 17}
#: BT.2020's own 10- and 12-bit curves are BT.709's (H.273), as FFVship maps them.
TRANSFERS = {"bt709": 1, "bt470m": 4, "gamma22": 4, "bt470bg": 5, "gamma28": 5, "smpte170m": 6,
             "smpte240m": 7, "linear": 8, "iec61966-2-1": 13, "srgb": 13, "iec61966_2_1": 13,
             "bt2020-10": 1, "bt2020_10bit": 1, "bt2020-12": 1, "bt2020_12bit": 1,
             "smpte2084": 16, "smpte428": 17, "smpte428_1": 17, "arib-std-b67": 18}
PRIMARIES = {"bt709": 1, "bt470m": 4, "bt470bg": 5, "smpte170m": 6, "smpte240m": 7, "bt2020": 9,
             "smpte432": 12}
UNTAGGED = {"", "unknown", "unspecified", "reserved"}

#: The matrices' names in FFmpeg's filters (setparams' colorspace).
FFMPEG_MATRICES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 8: "ycgco", 9: "bt2020nc", 10: "bt2020c",
                   14: "ictcp", 16: "ycgco-re", 17: "ycgco-ro"}
#: The transfers whose light is relative to the display's white (SDR); the
#: others -- PQ, ST 428, HLG -- are HDR.
SDR_TRANSFERS = frozenset({1, 4, 5, 6, 7, 8, 13})


class UnsupportedColourError(ValueError):
    """A tag with no value here: `kind` is "matrix", "transfer",
    "primaries", "range" or "siting", `value` the tag."""

    def __init__(self, kind: str, value: str) -> None:
        super().__init__(f"unsupported {kind}: {value}")
        self.kind = kind
        self.value = value


@dataclass(frozen=True, slots=True)
class VideoColour:
    matrix: int        # 0 for RGB
    transfer: int
    primaries: int
    full_range: bool

    @property
    def hdr(self) -> bool:
        return self.transfer not in SDR_TRANSFERS


def video_colour(info: VideoInfo, *, rgb: bool, full_range_untagged: bool = False) -> VideoColour:
    """`info`'s colours. Untagged values are guessed as FFVship guesses
    them: the matrix by the video's height (BT.709 above 650 lines, else
    BT.470BG), the transfer and primaries from the matrix (BT.470BG's, or
    PQ and BT.2020 for BT.2020 and ICtCp); untagged RGB is sRGB, full range.
    `full_range_untagged`: a format that is full range when untagged (yuvj).

    The height is the video's own. It was the size compared at, so an
    untagged 1080p source scaled to a 640x360 encode was read as BT.470BG."""
    matrix_name = (info.color_space or "").casefold()
    if rgb:
        matrix = 0
    elif matrix_name in UNTAGGED:
        matrix = 1 if info.height > 650 else 5
    elif matrix_name in MATRICES:
        matrix = MATRICES[matrix_name]
    else:
        raise UnsupportedColourError("matrix", info.color_space)

    transfer_name = (info.color_transfer or "").casefold()
    if transfer_name in UNTAGGED:
        transfer = 13 if matrix == 0 else 5 if matrix == 5 else 16 if matrix in {9, 10, 14} else 1
    elif transfer_name in TRANSFERS:
        transfer = TRANSFERS[transfer_name]
    else:
        raise UnsupportedColourError("transfer", info.color_transfer)

    primaries_name = (info.color_primaries or "").casefold()
    if primaries_name in UNTAGGED:
        primaries = 5 if matrix == 5 else 9 if matrix in {9, 10, 14} else 1
    elif primaries_name in PRIMARIES:
        primaries = PRIMARIES[primaries_name]
    else:
        raise UnsupportedColourError("primaries", info.color_primaries)

    range_name = (info.color_range or "").casefold()
    if range_name in {"pc", "jpeg", "full"} or (not range_name and full_range_untagged):
        full_range = True
    elif range_name in {"", "unknown", "unspecified", "tv", "mpeg", "limited"}:
        full_range = rgb
    else:
        raise UnsupportedColourError("range", info.color_range)
    return VideoColour(matrix, transfer, primaries, full_range)


def is_rgb_format(pix_fmt: str) -> bool:
    """Whether FFmpeg's pixel format holds RGB rather than YUV."""
    name = (pix_fmt or "").casefold()
    return name.startswith(("gbr", "rgb", "bgr", "argb", "abgr", "0rgb", "0bgr"))


def colour_of(info: VideoInfo) -> VideoColour | None:
    """`info`'s colours as both implementations read them, or None for a tag
    neither can (left to FFmpeg's own conversion)."""
    name = (info.pix_fmt or "").casefold()
    try:
        return video_colour(info, rgb=is_rgb_format(name), full_range_untagged=name.startswith("yuvj"))
    except UnsupportedColourError:
        return None


# ------------------------------------------------------- PNG colour chunks

#: Each transfer that is a plain power curve in Vship, and its exponent:
#: BT.709 (and BT.2020's SDR curves) is the 2.4 gamma of a display
#: (BT.1886), as Vship takes it, not the camera curve H.273 defines.
_GAMMAS = {1: 2.4, 4: 2.2, 5: 2.8, 6: 2.8, 7: 2.4}
#: (white, red, green, blue) chromaticities of each primaries code, as x, y.
_CHROMATICITIES = {
    1: ((0.3127, 0.3290), (0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    4: ((0.310, 0.316), (0.67, 0.33), (0.21, 0.71), (0.14, 0.08)),
    5: ((0.3127, 0.3290), (0.64, 0.33), (0.29, 0.60), (0.15, 0.06)),
    6: ((0.3127, 0.3290), (0.630, 0.340), (0.310, 0.595), (0.155, 0.070)),
    7: ((0.3127, 0.3290), (0.630, 0.340), (0.310, 0.595), (0.155, 0.070)),
    9: ((0.3127, 0.3290), (0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    12: ((0.3127, 0.3290), (0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
}
#: The chunks that say how a PNG's values are to be read; what FFmpeg wrote
#: of them is replaced.
_COLOUR_CHUNKS = {b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"cICP", b"mDCV", b"cLLI"}


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png_colour_chunks(colour: VideoColour) -> bytes:
    """The chunks that make libjxl's tools read an RGB picture's values as
    Vship reads the video's: a plain power curve as gAMA with cHRM (no
    H.273 code is a plain 2.4 gamma), every other transfer -- sRGB, linear,
    PQ, HLG -- as cICP."""
    gamma = _GAMMAS.get(colour.transfer)
    if gamma is not None and colour.primaries in _CHROMATICITIES:
        points = [round(value * 100000) for point in _CHROMATICITIES[colour.primaries] for value in point]
        return (_png_chunk(b"gAMA", struct.pack(">I", round(100000 / gamma)))
                + _png_chunk(b"cHRM", struct.pack(">8I", *points)))
    return _png_chunk(b"cICP", bytes((colour.primaries, colour.transfer, 0, 1)))


def describe_png(path: Path, colour: VideoColour) -> None:
    """Rewrites the colour chunks of the PNG at `path` (one FFmpeg wrote)
    as png_colour_chunks gives them for `colour`."""
    path = Path(path)
    data = path.read_bytes()
    parts, position = [data[:8]], 8
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        whole = data[position:position + 12 + length]
        if kind == b"IHDR":
            parts += [whole, png_colour_chunks(colour)]
        elif kind not in _COLOUR_CHUNKS:
            parts.append(whole)
        position += 12 + length
    temporary = path.with_name(path.name + ".colour")
    temporary.write_bytes(b"".join(parts))
    os.replace(temporary, path)
