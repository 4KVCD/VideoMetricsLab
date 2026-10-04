"""How the perceptual metrics read a video's colours (vmaf_app.core.colour)."""
import struct
from pathlib import Path

import pytest

from vmaf_app.core.colour import (
    UnsupportedColourError,
    VideoColour,
    colour_of,
    describe_png,
    png_colour_chunks,
    video_colour,
)
from vmaf_app.core.models import VideoInfo


def _info(height=1080, **tags):
    return VideoInfo(Path("v.mkv"), height * 16 // 9, height, 24.0, 1.0, 24, "h264", pix_fmt="yuv420p", **tags)


def test_untagged_video_is_guessed_as_ffvship_guesses_it():
    assert video_colour(_info(1080), rgb=False) == VideoColour(1, 1, 1, False)
    assert video_colour(_info(480), rgb=False) == VideoColour(5, 5, 5, False)
    assert video_colour(_info(1080, color_space="bt2020nc"), rgb=False) == VideoColour(9, 16, 9, False)
    assert video_colour(_info(1080), rgb=True) == VideoColour(0, 13, 1, True)


def test_the_guess_goes_by_the_videos_own_height():
    """It went by the size compared at: an untagged 1080p source scaled to a
    640x360 encode was read as BT.470BG."""
    assert video_colour(_info(1080), rgb=False).matrix == 1


def test_tags_are_read_and_an_unknown_one_says_which():
    hdr = video_colour(_info(2160, color_space="bt2020nc", color_transfer="smpte2084",
                             color_primaries="bt2020", color_range="tv"), rgb=False)
    assert hdr == VideoColour(9, 16, 9, False) and hdr.hdr
    with pytest.raises(UnsupportedColourError) as raised:
        video_colour(_info(1080, color_transfer="log100"), rgb=False)
    assert (raised.value.kind, raised.value.value) == ("transfer", "log100")
    assert colour_of(_info(1080, color_transfer="log100")) is None


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    found, position = [], 8
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        found.append((kind, data[position + 8:position + 8 + length]))
        position += 12 + length
    return found


def _png(*extra: bytes) -> bytes:
    from vmaf_app.core.colour import _png_chunk

    header = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 16, 2, 0, 0, 0))
    return b"\x89PNG\r\n\x1a\n" + header + b"".join(extra) + _png_chunk(b"IDAT", b"pixels") + _png_chunk(b"IEND", b"")


def test_bt709_is_described_as_a_24_gamma_not_the_camera_curve():
    """FFmpeg tags a BT.709 picture cICP 1/1, which libjxl reads with the
    camera curve; Vship and a display use a 2.4 gamma. A tagged BT.709
    film scored 40.4 SSIMULACRA2 on the CPU against 55.1 on the GPU."""
    kinds = dict(_chunks(b"\x89PNG\r\n\x1a\n" + png_colour_chunks(VideoColour(1, 1, 1, False))))
    assert struct.unpack(">I", kinds[b"gAMA"])[0] == 41667
    assert struct.unpack(">8I", kinds[b"cHRM"])[:2] == (31270, 32900)
    assert b"cICP" not in kinds


def test_hdr_is_described_by_its_h273_codes():
    kinds = dict(_chunks(b"\x89PNG\r\n\x1a\n" + png_colour_chunks(VideoColour(9, 16, 9, False))))
    assert kinds == {b"cICP": bytes((9, 16, 0, 1))}


def test_ffmpegs_colour_chunks_are_replaced(tmp_path):
    from vmaf_app.core.colour import _png_chunk

    path = tmp_path / "frame.png"
    path.write_bytes(_png(_png_chunk(b"cICP", bytes((1, 1, 0, 1))), _png_chunk(b"gAMA", struct.pack(">I", 45455))))
    describe_png(path, VideoColour(1, 1, 1, False))
    kinds = [kind for kind, _body in _chunks(path.read_bytes())]
    assert kinds == [b"IHDR", b"gAMA", b"cHRM", b"IDAT", b"IEND"]
