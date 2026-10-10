"""How a video's colours are read for the perceptual metrics: its matrix,
transfer, primaries and range, as H.273 numbers taken from its tags. Values
the video does not tag are guessed as FFVship (Vship's own tool) guesses
them.

Both implementations read colours the same way: Vship is given these
numbers on the GPU (perceptual_vship._vship_colorspace), and the CPU tools
are given pictures converted and described the same way (perceptual_cpu).
When the two read colours differently, they scored the same frames far
apart: a tagged BT.709 film scored 40.4 SSIMULACRA2 on the CPU and 55.1 on
the GPU.
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from vmaf_app.core.models import VideoInfo

#: FFmpeg's tag names (as ffprobe prints them, with their aliases), mapped to
#: the values of Vship's enums in VshipColor.h, which are the H.273 numbers.
#: The mapping is FFVship's (src/ffvship_utility/ffmpegToVshipColorFormat.hpp
#: in Vship 5.1.1). A tag missing here has no Vship value.
MATRICES = {"gbr": 0, "rgb": 0, "bt709": 1, "bt470bg": 5, "smpte170m": 6, "ycgco": 8, "ycocg": 8,
            "bt2020nc": 9, "bt2020ncl": 9, "bt2020c": 10, "bt2020cl": 10, "ictcp": 14,
            "ycgco-re": 16, "ycgco-ro": 17}
#: BT.2020's 10- and 12-bit curves are BT.709's curve (H.273), and FFVship
#: maps them that way.
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
#: The transfers whose light is relative to the display's white (SDR). The
#: others (PQ, ST 428 and HLG) are HDR.
SDR_TRANSFERS = frozenset({1, 4, 5, 6, 7, 8, 13})


class UnsupportedColourError(ValueError):
    """A colour tag that has no value here. `kind` is "matrix", "transfer",
    "primaries", "range" or "siting"; `value` is the tag."""

    def __init__(self, kind: str, value: str) -> None:
        super().__init__(f"unsupported {kind}: {value}")
        self.kind = kind
        self.value = value


@dataclass(frozen=True, slots=True)
class VideoColour:
    """A video's colours as H.273 numbers, and whether it is full range."""

    matrix: int        # 0 for RGB
    transfer: int
    primaries: int
    full_range: bool

    @property
    def hdr(self) -> bool:
        """Whether the transfer is an HDR one."""
        return self.transfer not in SDR_TRANSFERS


def video_colour(info: VideoInfo, *, rgb: bool, full_range_untagged: bool = False) -> VideoColour:
    """Returns the video's colours.

    Untagged values are guessed as FFVship guesses them: the matrix from the
    video's height (BT.709 above 650 lines, otherwise BT.470BG), and the
    transfer and primaries from the matrix (BT.470BG's for BT.470BG; PQ and
    BT.2020 for BT.2020 and ICtCp; BT.709's otherwise). Untagged RGB is
    sRGB, full range. `full_range_untagged` says the pixel format is full
    range when untagged (yuvj formats).

    The height used is the video's own, not the size it is compared at:
    otherwise an untagged 1080p source scaled to a 640x360 encode would be
    read as BT.470BG."""
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
    """The video's colours as both implementations read them, or None for a
    tag neither can read (such a video is left to FFmpeg's own conversion)."""
    name = (info.pix_fmt or "").casefold()
    try:
        return video_colour(info, rgb=is_rgb_format(name), full_range_untagged=name.startswith("yuvj"))
    except UnsupportedColourError:
        return None


# ------------------------------------------------------- PNG colour chunks

#: Each transfer that Vship treats as a plain power curve, with its exponent.
#: Vship takes BT.709 (and BT.2020's SDR curves) as a display's 2.4 gamma
#: (BT.1886), not as the camera curve H.273 defines.
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
#: The PNG chunks that say how its values are to be read. Any of these that
#: FFmpeg wrote are replaced.
_COLOUR_CHUNKS = {b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"cICP", b"mDCV", b"cLLI"}


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    """One PNG chunk: length, type, body and CRC."""
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png_colour_chunks(colour: VideoColour) -> bytes:
    """The PNG chunks that make libjxl's tools read an RGB picture's values
    as Vship reads the video's. A plain power curve is described with gAMA
    and cHRM, because no H.273 code means a plain 2.4 gamma. Every other
    transfer (sRGB, linear, PQ, HLG) is described with cICP."""
    gamma = _GAMMAS.get(colour.transfer)
    if gamma is not None and colour.primaries in _CHROMATICITIES:
        points = [round(value * 100000) for point in _CHROMATICITIES[colour.primaries] for value in point]
        return (_png_chunk(b"gAMA", struct.pack(">I", round(100000 / gamma)))
                + _png_chunk(b"cHRM", struct.pack(">8I", *points)))
    return _png_chunk(b"cICP", bytes((colour.primaries, colour.transfer, 0, 1)))


def describe_png(path: Path, colour: VideoColour) -> None:
    """Replaces the colour chunks of the PNG at `path` (one FFmpeg wrote)
    with those png_colour_chunks gives for `colour`."""
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
