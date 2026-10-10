"""Makes the film grain test's fixtures: tests/fixtures/vvc_grain.266 (a small
VVC stream with film grain messages) and tests/fixtures/vvc_grain.npz
(FFmpeg's decoded frames of it, with and without the grain).

How the stream is made:
  1. FFmpeg's libvvenc encodes a 64x48, 8-bit, 17-frame test pattern.
  2. A film grain characteristics SEI message is inserted into some access
     units, following SCHEDULE below. The schedule covers what the grain
     synthesis has to get right: two luma intensity intervals with a gap
     between them, horizontal and vertical cutoffs that differ, a plane with
     no grain, a message that persists to the following frames, one that
     applies to its own frame only, and a cancel.
  3. FFmpeg decodes the stream twice, once applying the grain (FFmpeg
     synthesizes H.274 grain for 8-bit 4:2:0 video) and once without it.
     ffprobe reports each frame's grain seed, which is its picture order
     count. FFmpeg's VVC decoder reports a message that does not persist
     for its own frame, but does not draw it: the stream's frames record
     which frames FFmpeg actually changed.

The test checks the app's parser and GPU synthesis against these results.

Usage (needs FFmpeg 8 or newer, built with libvvenc, on PATH):
    python tests/fixtures/make_vvc_grain.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vmaf_app.core import film_grain

WIDTH, HEIGHT, FRAMES = 64, 48, 17

# A grain message is (log2_scale_factor, persistence_flag, components). Each
# of the three components (Y, Cb, Cr) is None (no grain on that plane) or a
# list of intensity intervals: (lower bound, upper bound, model values),
# where the model values are (scale, horizontal cutoff, vertical cutoff).
GRAIN_A = (3, True, [
    [(16, 100, (60, 10, 4)), (120, 235, (36, 14, 14))],  # luma: blocks averaging 101-119 get no grain
    [(0, 255, (40, 4, 12))],
    None,  # no grain on Cr
])
GRAIN_B = (2, True, [
    [(0, 255, (30, 6, 6))],
    None,
    [(60, 200, (50, 12, 8))],
])
GRAIN_ONCE = (4, False, [
    [(0, 255, (120, 8, 14))],
    [(0, 255, (60, 14, 2))],
    [(0, 255, (60, 2, 14))],
])
CANCEL = "cancel"

# Which access unit (in decoding order) carries which message. Access units
# not listed carry none: the previous message still applies if it persists.
SCHEDULE = {0: GRAIN_A, 8: GRAIN_B, 12: CANCEL, 14: GRAIN_ONCE}


class BitWriter:
    """Writes bits most significant first, as H.274's syntax is written."""

    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, bit_count: int, value: int) -> None:
        for i in reversed(range(bit_count)):
            self.bits.append((value >> i) & 1)

    def write_unsigned_exp_golomb(self, value: int) -> None:
        code = value + 1
        self.write(code.bit_length() - 1, 0)
        self.write(code.bit_length(), code)

    def write_signed_exp_golomb(self, value: int) -> None:
        self.write_unsigned_exp_golomb(2 * value - 1 if value > 0 else -2 * value)

    def to_bytes(self) -> bytes:
        """The bits as an SEI payload: one that does not end on a byte boundary
        ends with a 1 bit and then zeros (sei_payload_bit_equal_to_one)."""
        padded = list(self.bits)
        if len(padded) % 8:
            padded.append(1)
            padded += [0] * (-len(padded) % 8)
        return bytes(int("".join(map(str, padded[i:i + 8])), 2) for i in range(0, len(padded), 8))


def film_grain_payload(message) -> bytes:
    """The payload of a film grain characteristics SEI message (H.274 8.5)."""
    writer = BitWriter()
    if message == CANCEL:
        writer.write(1, 1)  # fg_characteristics_cancel_flag
        return writer.to_bytes()
    log2_scale_factor, persistence, components = message
    writer.write(1, 0)  # fg_characteristics_cancel_flag
    writer.write(2, 0)  # fg_model_id: frequency filtering
    writer.write(1, 0)  # fg_separate_colour_description_present_flag
    writer.write(2, 0)  # fg_blending_mode_id: additive
    writer.write(4, log2_scale_factor)
    for component in components:
        writer.write(1, component is not None)  # fg_comp_model_present_flag
    for component in components:
        if component is None:
            continue
        writer.write(8, len(component) - 1)  # fg_num_intensity_intervals_minus1
        writer.write(3, 2)  # fg_num_model_values_minus1: three values
        for lower, upper, values in component:
            writer.write(8, lower)
            writer.write(8, upper)
            for value in values:
                writer.write_signed_exp_golomb(value)
    writer.write(1, int(persistence))  # fg_characteristics_persistence_flag
    return writer.to_bytes()


def add_emulation_prevention(rbsp: bytes) -> bytes:
    """Inserts the 0x03 bytes that keep a NAL unit's payload from containing a start code."""
    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def film_grain_sei_nal(temporal_id: int, message) -> bytes:
    """A prefix SEI NAL unit holding one film grain characteristics message."""
    payload = film_grain_payload(message)
    header = bytes([0, (film_grain.PREFIX_SEI_NUT << 3) | (temporal_id + 1)])
    body = bytes([film_grain.FILM_GRAIN_CHARACTERISTICS, len(payload)]) + payload + b"\x80"
    return header + add_emulation_prevention(body)


def run(*args: str) -> bytes:
    return subprocess.run(args, capture_output=True, check=True).stdout


def split_access_units(stream: bytes) -> list[list[bytes]]:
    """The stream's NAL units, grouped into access units (each starts with an AUD)."""
    access_units: list[list[bytes]] = []
    for unit in film_grain.nal_units(stream):
        if film_grain.nal_type(unit) == 20 or not access_units:
            access_units.append([])
        access_units[-1].append(unit)
    return access_units


def main() -> None:
    encoded = Path(tempfile.mkdtemp()) / "encoded.266"
    run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=24",
        "-frames:v", str(FRAMES), "-vf", "noise=alls=12:allf=t,format=yuv420p10le", "-c:v", "libvvenc",
        "-preset", "faster", "-qp", "28", "-vvenc-params", "internal-bitdepth=8:IntraPeriod=16",
        "-f", "vvc", str(encoded))

    stream = bytearray()
    for index, units in enumerate(split_access_units(encoded.read_bytes())):
        message = SCHEDULE.get(index)
        inserted = message is None
        for unit in units:
            is_slice = 0 <= film_grain.nal_type(unit) <= 11
            if is_slice and not inserted:
                stream += b"\x00\x00\x00\x01" + film_grain_sei_nal(film_grain.temporal_id(unit), message)
                inserted = True
            stream += b"\x00\x00\x00\x01" + unit
    fixture = ROOT / "tests" / "fixtures" / "vvc_grain.266"
    fixture.write_bytes(bytes(stream))

    def decode(*options: str) -> np.ndarray:
        raw = run("ffmpeg", "-v", "error", *options, "-i", str(fixture), "-fps_mode", "passthrough",
                  "-f", "rawvideo", "-pix_fmt", "yuv420p", "-")
        return np.frombuffer(raw, np.uint8).reshape(-1, WIDTH * HEIGHT * 3 // 2)

    grained = decode()
    plain = decode("-export_side_data", "film_grain")
    report = json.loads(run("ffprobe", "-v", "error", "-export_side_data", "film_grain", "-show_frames",
                            "-show_entries", "frame=side_data", "-of", "json", str(fixture)))
    entries = report.get("packets_and_frames") or report.get("frames", [])
    frames = [entry for entry in entries if entry.get("type", "frame") == "frame"]
    # Each frame's grain seed in display order, -1 for a frame without grain.
    seeds = []
    for frame in frames:
        grain = [side for side in frame.get("side_data_list", []) if side["side_data_type"] == "Film grain parameters"]
        seeds.append(int(grain[0]["seed"]) if grain else -1)
    applied = (grained != plain).any(axis=1)
    np.savez_compressed(ROOT / "tests" / "fixtures" / "vvc_grain.npz",
                        plain=plain, grained=grained, seeds=np.array(seeds), applied=applied)
    print(f"{fixture.name}: {len(stream)} bytes, {len(plain)} frames, seeds {seeds}, "
          f"frames FFmpeg drew grain on: {int(applied.sum())}")


if __name__ == "__main__":
    main()
