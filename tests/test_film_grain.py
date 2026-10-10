"""VVC film grain in Video Compare: reading the grain messages, working out
each frame's picture order count, and synthesizing the grain on the GPU,
checked against FFmpeg.

tests/fixtures/make_vvc_grain.py made the fixtures: a small 8-bit VVC stream
with a schedule of film grain messages, and FFmpeg's frames of it with and
without the grain (FFmpeg synthesizes H.274 grain for 8-bit 4:2:0 video).
"""
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import d3d11_film_grain, film_grain

FIXTURES = Path(__file__).parent / "fixtures"
WIDTH, HEIGHT = 64, 48


def _access_units(stream: bytes) -> list[bytes]:
    """The stream's access units (each starts with an access unit delimiter)."""
    groups: list[list[bytes]] = []
    for unit in film_grain.nal_units(stream):
        if film_grain.nal_type(unit) == 20 or not groups:
            groups.append([])
        groups[-1].append(unit)
    return [b"".join(b"\x00\x00\x00\x01" + unit for unit in group) for group in groups]


def _planes(frame: np.ndarray) -> list[np.ndarray]:
    """An 8-bit 4:2:0 frame's Y, Cb and Cr planes."""
    chroma = (WIDTH // 2) * (HEIGHT // 2)
    luma = frame[:WIDTH * HEIGHT].reshape(HEIGHT, WIDTH)
    cb = frame[WIDTH * HEIGHT:WIDTH * HEIGHT + chroma].reshape(HEIGHT // 2, WIDTH // 2)
    cr = frame[WIDTH * HEIGHT + chroma:].reshape(HEIGHT // 2, WIDTH // 2)
    return [luma, cb, cr]


def _fixture_grain() -> dict[int, film_grain.FrameGrain]:
    """The grain of each frame of the fixture, by picture order count."""
    tracker = film_grain.FilmGrainTracker()
    frames = {}
    for access_unit in _access_units((FIXTURES / "vvc_grain.266").read_bytes()):
        frame = tracker.take_frame_grain(tracker.add_access_unit(access_unit, pts=None))
        if frame is not None:
            frames[frame.seed] = frame
    return frames


def _bits(*fields) -> bytes:
    """A bit string's bytes, zero-padded. Each field is (value, width), or
    ("ue", value) for an unsigned Exp-Golomb code."""
    text = ""
    for first, second in fields:
        if first == "ue":
            code = bin(second + 1)[2:]
            text += "0" * (len(code) - 1) + code
        else:
            text += format(first, f"0{second}b")
    text += "0" * (-len(text) % 8)
    return bytes(int(text[i:i + 8], 2) for i in range(0, len(text), 8))


def test_vvc_film_grain_matches_ffmpeg(subtests):
    expected = np.load(FIXTURES / "vvc_grain.npz")
    # In display order, which is picture order count order: the frame's seed
    # (its picture order count), or -1 where FFmpeg found no grain.
    expected_seeds = [int(seed) for seed in expected["seeds"]]

    with subtests.test("a grain message is read as FFmpeg reads it"):
        # VVenc's default message, from a real encode.
        payload = bytes.fromhex("00b8020afa0680d01a0020afa0d068340082be8341a0d6")
        assert film_grain.parse_film_grain(payload) == film_grain.FilmGrainH274(
            model_id=0, blending_mode_id=0, log2_scale_factor=2, persistence=True,
            intervals=(((10, 250, (26, 26, 26)),), ((10, 250, (13, 13, 13)),), ((10, 250, (13, 13, 13)),)))
        assert film_grain.parse_film_grain(b"\x80") is None  # a cancel
        with pytest.raises(ValueError):
            film_grain.parse_film_grain(payload[:3])  # cut short
        assert film_grain.remove_emulation_prevention(b"\x00\x00\x03\x01\x00\x00\x03") == b"\x00\x00\x01\x00\x00"

    with subtests.test("each frame gets the grain FFmpeg gives it, seeded by its picture order count"):
        frames = _fixture_grain()
        assert sorted(frames) == [seed for seed in expected_seeds if seed >= 0]

    with subtests.test("frames are found by their access unit's number in any order, or else by timestamp"):
        access_units = _access_units((FIXTURES / "vvc_grain.266").read_bytes())
        # A raw stream: no timestamps, and the frames come out of the
        # decoder in display order, not in the order of their access units.
        tracker = film_grain.FilmGrainTracker()
        numbers = [tracker.add_access_unit(access_unit, pts=None) for access_unit in access_units]
        grains = [tracker.take_frame_grain(number) for number in reversed(numbers)][::-1]
        assert sorted(frame.seed for frame in grains if frame is not None) == sorted(_fixture_grain())
        assert all(tracker.take_frame_grain(number) is None for number in numbers)  # each is taken once
        # By timestamp: the frames before the one asked for are forgotten,
        # and a frame the decoder moved to a seek's target is still found.
        tracker = film_grain.FilmGrainTracker()
        for index, access_unit in enumerate(access_units):
            tracker.add_access_unit(access_unit, pts=1000 * index)
        assert None not in grains[1:6]
        assert tracker.take_frame_grain(None, 3000) == grains[3]
        assert tracker.take_frame_grain(None, 1000) is None
        assert tracker.take_frame_grain(None, 5500) == grains[5]

    with subtests.test("a seek forgets the grain that persisted before it"):
        tracker = film_grain.FilmGrainTracker()
        first, second = _access_units((FIXTURES / "vvc_grain.266").read_bytes())[:2]
        before = tracker.add_access_unit(first, pts=0)
        tracker.flush()
        after = tracker.add_access_unit(second, pts=1)
        assert tracker.take_frame_grain(before) is None and tracker.take_frame_grain(after) is None

    with subtests.test("picture order counts are worked out from the last picture others may refer to"):
        tracker = film_grain.FilmGrainTracker()
        sps = film_grain.SequenceParameterSet(log2_max_poc_lsb=4, poc_msb_cycle_length=0, extra_picture_header_bits=0)

        def count(poc_lsb, slice_type=0, non_reference=False):  # slice type 0: an ordinary picture
            header = film_grain.PictureHeader(poc_lsb, None, sps, non_reference)
            return tracker._picture_order_count(header, slice_type, 0, None)

        assert count(0, film_grain.IDR_N_LP) == 0
        assert count(8) == 8
        assert count(15, non_reference=True) == 15
        # Worked out from 8, as H.266 says and FFmpeg does: from 15, it was 18.
        assert count(2) == 2

    with subtests.test("a damaged parameter set cannot hold the parser up"):
        sps = _bits((0, 4), (0, 4), (0, 3), (1, 2), (2, 2),  # ids, one sublayer, 4:2:0, 128x128 CTUs
                    (0, 1), (0, 1), (0, 1),  # no profile, no gradual refresh, no resampling
                    ("ue", 64), ("ue", 64), (0, 1),  # 64x64, no conformance window
                    (1, 1), ("ue", 1 << 30), (1, 1), (1, 1))  # a billion subpictures, all the same size
        with pytest.raises(ValueError):
            film_grain.parse_sps(sps)

    with subtests.test("4:2:0 chroma is adapted as FFmpeg's synthesis does, 4:2:2 as VTM's"):
        grain = film_grain.FilmGrainH274(0, 0, 3, ((), ((0, 255, (40, 4, 12)),), ((0, 255, (40,)),)), True)
        frame = film_grain.FrameGrain(grain, seed=0)
        sizes = [(64, 48), (32, 24), (32, 24)]
        params = d3d11_film_grain.native_params(frame, 8, sizes, (1, 1))
        cb, cr = params.plane[1], params.plane[2]
        assert (cb.scale[0], cb.h[0], cb.v[0]) == (20, 8 - 2, 14 - 2)  # half the scale, cutoffs doubled
        assert (cr.scale[0], cr.h[0], cr.v[0]) == (20, 14 - 2, 14 - 2)  # inferred cutoffs 8, then doubled
        assert params.plane[0].intervals == 0
        params = d3d11_film_grain.native_params(frame, 8, [(64, 48), (32, 48), (32, 48)], (1, 0))
        assert (params.plane[1].scale[0], params.plane[1].h[0], params.plane[1].v[0]) == (28, 6, 10)

    if not d3d11_film_grain.available():
        pytest.skip("d3d11_film_grain.dll is not built (scripts/build_d3d11_film_grain.ps1)")

    with subtests.test("the synthesis gives FFmpeg's frames exactly"):
        frames = _fixture_grain()
        sizes = d3d11_film_grain.plane_sizes(WIDTH, HEIGHT, (1, 1), 3)
        for display_index, seed in enumerate(expected_seeds):
            plain, grained = _planes(expected["plain"][display_index]), _planes(expected["grained"][display_index])
            if seed < 0:
                continue  # no grain on this frame
            params = d3d11_film_grain.native_params(frames[seed], 8, sizes, (1, 1))
            ours = d3d11_film_grain.synthesize(plain, params, software=True)
            if expected["applied"][display_index]:
                for plane, (got, want) in enumerate(zip(ours, grained, strict=True)):
                    assert np.array_equal(got, want), f"frame {display_index}, plane {plane}"
            else:
                # FFmpeg's VVC decoder reports a message that does not persist
                # for its own frame but never draws it (it marks the message
                # used before deciding whether to draw). H.274 applies it to
                # that frame, and so do we.
                assert any(not np.array_equal(got, base) for got, base in zip(ours, plain, strict=True))

    with subtests.test("above 8 bits, blocks are averaged in 8-bit codes and the grain is shifted up, as VTM's"):
        frames = _fixture_grain()
        sizes = d3d11_film_grain.plane_sizes(WIDTH, HEIGHT, (1, 1), 3)
        # Codes kept away from the ends, so that neither depth clips.
        plain = [np.clip(plane, 40, 215) for plane in _planes(expected["plain"][1])]
        eight = d3d11_film_grain.synthesize(plain, d3d11_film_grain.native_params(frames[1], 8, sizes, (1, 1)),
                                            software=True)
        ten = d3d11_film_grain.synthesize([plane.astype(np.uint16) * 4 for plane in plain],
                                          d3d11_film_grain.native_params(frames[1], 10, sizes, (1, 1)),
                                          software=True)
        assert any(not np.array_equal(a, b) for a, b in zip(eight, plain, strict=True))
        for got, base in zip(ten, eight, strict=True):
            assert np.array_equal(got, base.astype(np.uint16) * 4)
