"""Draws the app's icon at every size Windows and macOS ask for, and writes
cpp/app/icons: VideoMetricsLab.ico (Windows), VideoMetricsLab.icns (macOS),
VideoMetricsLab-256.png, and VideoMetricsLab.svg and VideoMetricsLab-small.svg,
the same drawing for editing tools.

    python scripts/make_icons.py

The icon: an encoded picture's blocks, a magnifying glass over them, and in
its lens four blocks enlarged, the faulty one marked in amber. At 32 px and
below, where that outline would be thinner than a pixel, the faulty block is
filled amber instead and the glass is drawn bolder. The drawing is the shapes
below, in units of a 128-unit square; this script is its only source.
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "cpp" / "app" / "icons"

TILE = "#1E293B"
GLASS = "#F8FAFC"
AMBER = "#F59E0B"
LENS = (56, 56, 30)  # the lens's centre and radius

#: The encoded picture: 4 by 4 blocks of sky and grass.
MOSAIC = [["#2B7FB0", "#3489BA", "#2B7FB0", "#3489BA"],
          ["#3489BA", "#2B7FB0", "#3489BA", "#2B7FB0"],
          ["#2E8B57", "#3A9B5E", "#2E8B57", "#3489BA"],
          ["#23704A", "#2E8B57", "#23704A", "#2E8B57"]]
#: The same picture in 2 by 2 blocks, for the small sizes.
SMALL_MOSAIC = [["#3489BA", "#2B7FB0"],
                ["#2E8B57", "#23704A"]]


def mosaic(colours: list[list[str]]) -> list[tuple]:
    """The picture's blocks, filling 16..110 with 2-unit gaps."""
    count = len(colours)
    pitch = 96 / count
    return [("rect", 16 + column * pitch, 16 + row * pitch, 16 + column * pitch + pitch - 2,
             16 + row * pitch + pitch - 2, colour)
            for row, line in enumerate(colours) for column, colour in enumerate(line)]


def drawing(small: bool) -> list[tuple]:
    """The icon's shapes, painted in order."""
    cx, cy, r = LENS
    lens = [("rect", 26, 26, 56, 56, "#3489BA"), ("rect", 56, 26, 86, 56, "#2B7FB0"),
            ("rect", 26, 56, 56, 86, "#2E8B57")]
    if small:
        lens.append(("rect", 56, 56, 86, 86, AMBER))
    else:
        lens += [("rect", 56, 56, 86, 86, "#C9A43A"), ("frame", 58.5, 58.5, 83.5, 83.5, 3, AMBER)]
    ring, handle = (10, 7) if small else (8, 6)
    return [("roundrect", 0, 0, 128, 128, 28, TILE),
            *mosaic(SMALL_MOSAIC if small else MOSAIC),
            ("clip", cx, cy, r, lens),
            ("ring", cx, cy, r - ring / 2, r + ring / 2, GLASS),
            ("capsule", 79, 79, 106, 106, handle, GLASS)]


# ---------------------------------------------------------------- drawing

def rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255


def coverage(shape: tuple, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which samples (x, y: their positions in the drawing's units) a shape covers."""
    kind = shape[0]
    if kind == "rect":
        x0, y0, x1, y1 = shape[1:5]
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    if kind == "roundrect":
        x0, y0, x1, y1, radius = shape[1:6]
        nearest_x = np.clip(x, x0 + radius, x1 - radius)
        nearest_y = np.clip(y, y0 + radius, y1 - radius)
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & \
            ((x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius ** 2)
    if kind == "frame":
        x0, y0, x1, y1, width = shape[1:6]
        half = width / 2
        outer = (x >= x0 - half) & (x < x1 + half) & (y >= y0 - half) & (y < y1 + half)
        inner = (x >= x0 + half) & (x < x1 - half) & (y >= y0 + half) & (y < y1 - half)
        return outer & ~inner
    if kind == "ring":
        cx, cy, inner, outer = shape[1:5]
        distance = (x - cx) ** 2 + (y - cy) ** 2
        return (distance >= inner ** 2) & (distance <= outer ** 2)
    if kind == "capsule":
        x0, y0, x1, y1, radius = shape[1:6]
        dx, dy = x1 - x0, y1 - y0
        along = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        return (x - x0 - along * dx) ** 2 + (y - y0 - along * dy) ** 2 <= radius ** 2
    raise ValueError(kind)


def paint(shapes: list[tuple], x: np.ndarray, y: np.ndarray, colour: np.ndarray, alpha: np.ndarray,
          within: np.ndarray | None = None) -> None:
    for shape in shapes:
        if shape[0] == "clip":
            cx, cy, radius, inside = shape[1:5]
            clip = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            paint(inside, x, y, colour, alpha, clip if within is None else clip & within)
            continue
        covered = coverage(shape, x, y)
        if within is not None:
            covered &= within
        colour[covered] = rgb(shape[-1])
        alpha[covered] = 1


def render(size: int, small: bool, tile: float = 1.0) -> np.ndarray:
    """The icon as size x size RGBA bytes, the tile filling `tile` of the
    square and centred (macOS's icons leave a margin), with 8 x 8 samples a
    pixel (4 x 4 above 256 px)."""
    samples = 8 if size <= 256 else 4
    span = size * tile
    offset = (size - span) / 2
    shapes = drawing(small)
    image = np.zeros((size, size, 4), dtype=np.float32)
    rows_at_once = max(1, 4096 // (size * samples) * 8)
    for top in range(0, size, rows_at_once):
        rows = min(rows_at_once, size - top)
        sy = (top + (np.arange(rows * samples, dtype=np.float32) + 0.5) / samples - offset) * 128 / span
        sx = ((np.arange(size * samples, dtype=np.float32) + 0.5) / samples - offset) * 128 / span
        x, y = np.meshgrid(sx, sy)
        colour = np.zeros((*x.shape, 3), dtype=np.float32)
        alpha = np.zeros(x.shape, dtype=np.float32)
        paint(shapes, x, y, colour, alpha)
        # Averaged as premultiplied colour, then made straight again.
        premultiplied = np.concatenate([colour * alpha[..., None], alpha[..., None]], axis=-1)
        block = premultiplied.reshape(rows, samples, size, samples, 4).mean(axis=(1, 3))
        straight = np.where(block[..., 3:] > 0, block[..., :3] / np.maximum(block[..., 3:], 1e-9), 0)
        image[top:top + rows] = np.concatenate([straight, block[..., 3:]], axis=-1)
    return np.round(image * 255).astype(np.uint8)


# ----------------------------------------------------------------- files

def png(pixels: np.ndarray) -> bytes:
    """An RGBA image as PNG."""
    height, width = pixels.shape[:2]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico(images: list[tuple[int, bytes]]) -> bytes:
    """A Windows icon of PNG images (size, data), as Windows Vista on reads them."""
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, data = b"", b""
    for size, image in images:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(image), offset + len(data))
        data += image
    return header + entries + data


def icns(images: list[tuple[str, bytes]]) -> bytes:
    """A macOS icon of PNG images (type, data)."""
    body = b"".join(kind.encode() + struct.pack(">I", 8 + len(image)) + image for kind, image in images)
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def svg(small: bool) -> str:
    """The drawing as SVG."""
    def element(shape: tuple) -> str:
        kind = shape[0]
        if kind == "rect":
            x0, y0, x1, y1, colour = shape[1:]
            return f'<rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}" fill="{colour}"/>'
        if kind == "roundrect":
            x0, y0, x1, y1, radius, colour = shape[1:]
            return f'<rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}" rx="{radius:g}" ' \
                   f'fill="{colour}"/>'
        if kind == "frame":
            x0, y0, x1, y1, width, colour = shape[1:]
            return f'<rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" height="{y1 - y0:g}" fill="none" ' \
                   f'stroke="{colour}" stroke-width="{width:g}"/>'
        if kind == "ring":
            cx, cy, inner, outer, colour = shape[1:]
            return f'<circle cx="{cx:g}" cy="{cy:g}" r="{(inner + outer) / 2:g}" fill="none" stroke="{colour}" ' \
                   f'stroke-width="{outer - inner:g}"/>'
        if kind == "capsule":
            x0, y0, x1, y1, radius, colour = shape[1:]
            return f'<line x1="{x0:g}" y1="{y0:g}" x2="{x1:g}" y2="{y1:g}" stroke="{colour}" ' \
                   f'stroke-width="{2 * radius:g}" stroke-linecap="round"/>'
        if kind == "clip":
            inside = "".join(element(part) for part in shape[4])
            return f'<g clip-path="url(#lens)">{inside}</g>'
        raise ValueError(kind)

    cx, cy, radius = LENS
    shapes = "\n".join(element(shape) for shape in drawing(small))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">\n'
            f'<defs><clipPath id="lens"><circle cx="{cx}" cy="{cy}" r="{radius}"/></clipPath></defs>\n'
            f'{shapes}\n</svg>\n')


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Windows: the small (16 px at 100%) and large (32 px) icons at every
    # scaling from 100% to 300%, so that none is scaled: at 175%, say, the
    # title bar's 28 px and the taskbar's 56 px; and Explorer's 48 to 256.
    sizes = (16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 128, 256)
    windows = [(size, png(render(size, small=size <= 32))) for size in sizes]
    (OUTPUT / "VideoMetricsLab.ico").write_bytes(ico(windows))
    # macOS: Apple's grid, the tile 824 of 1024 units wide; each size and
    # its double (@2x) under its own type.
    tile = 824 / 1024
    mac = {size: png(render(size, small=size <= 32, tile=tile)) for size in (16, 32, 64, 128, 256, 512, 1024)}
    (OUTPUT / "VideoMetricsLab.icns").write_bytes(icns([
        ("icp4", mac[16]), ("ic11", mac[32]), ("icp5", mac[32]), ("ic12", mac[64]), ("ic07", mac[128]),
        ("ic13", mac[256]), ("ic08", mac[256]), ("ic14", mac[512]), ("ic09", mac[512]), ("ic10", mac[1024])]))
    (OUTPUT / "VideoMetricsLab-256.png").write_bytes(dict(windows)[256])
    (OUTPUT / "VideoMetricsLab.svg").write_text(svg(small=False), encoding="utf-8", newline="\n")
    (OUTPUT / "VideoMetricsLab-small.svg").write_text(svg(small=True), encoding="utf-8", newline="\n")
    for path in sorted(OUTPUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
