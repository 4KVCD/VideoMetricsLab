"""How every metric pairs the two videos' frames: the test video's frame N
with the source's frame N, counted from each video's first frame, until
either video ends. A test video that drops, repeats or adds a frame is
compared off by that frame from there on. (Videos whose frame rates differ
are refused before they are compared: geometry.pair_problem.)

FFmpeg's two-input filters -- libvmaf, xpsnr, and the overlay that pairs the
frames for VMAF on the GPU -- pair frames by their timestamps. So both
videos' frames are first stamped with their numbers (numbering_filters):
frame N at N frame times of the test video, in both videos. Those filters
then pair frame N with frame N, and a duration limit (duration_args, an
output's -t) keeps the same frames of either video: those whose numbered
time is below the limit.

Where the frames reach this process instead -- from the GPU decoders, and
from FFmpeg's pipes to Vship and to VMAF on the GPU -- frame_pairs pairs the
n-th frames of the two videos, and frames_within gives the number of frames
a duration limit keeps, as FFmpeg's -t keeps them.

The old way, pairing by timestamp, is kept for now (frame_sync_by_time):
with the environment variable VML_FRAME_PAIRING=timestamp, every metric
pairs the frames as it did before (by_timestamp). The helpers here that
build FFmpeg's options (framesync_opts, pairing_filters, duration_args) and
decoded_frame_pairs follow that switch.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, TypeVar

from vmaf_app.core.models import VideoInfo

if TYPE_CHECKING:
    from vmaf_app.core.gpu_frames import GpuFrameStream

T = TypeVar("T")
U = TypeVar("U")

#: The environment variable that brings back the old pairing by timestamp
#: ("timestamp"), kept for now to compare the two ways.
PAIRING_VARIABLE = "VML_FRAME_PAIRING"


def by_timestamp() -> bool:
    """Whether frames are paired the old way, by timestamp
    (VML_FRAME_PAIRING=timestamp), rather than by frame number."""
    return os.environ.get(PAIRING_VARIABLE, "").strip().casefold() == "timestamp"


#: FFmpeg's frame sync for its two-input filters: the comparison ends with
#: the shorter video (shortest=1), and no video's last frame is repeated
#: past its end (repeatlast=0). With FFmpeg's defaults, the frames of a test
#: video longer than its source -- two frames of encoder padding are routine
#: -- were compared with a frozen copy of the source's last frame, and their
#: poor scores lowered the video's.
FRAMESYNC_OPTS = ["shortest=1", "repeatlast=0"]


def framesync_opts() -> list[str]:
    """The frame sync options for FFmpeg's two-input filters (libvmaf, xpsnr,
    the GPU's overlay): FRAMESYNC_OPTS, or when pairing by timestamp the old
    ones, which also take the nearest timestamp (frame_sync_by_time)."""
    if by_timestamp():
        from vmaf_app.core import frame_sync_by_time

        return list(frame_sync_by_time.FRAMESYNC_OPTS)
    return list(FRAMESYNC_OPTS)


#: The frame rate taken for a video that gives none.
_FALLBACK_FRAME_RATE = Fraction(25)


def frame_rate(info: VideoInfo) -> Fraction:
    """`info`'s frame rate as an exact fraction -- 24000/1001, not 23.976... --
    which numbered frames are timed by (25 for a video that gives none)."""
    if info.fps <= 0:
        return _FALLBACK_FRAME_RATE
    return Fraction(info.fps).limit_denominator(1_000_000)


def numbering_filters(rate: Fraction) -> list[str]:
    """FFmpeg filters that stamp each frame with its number: frame N at N /
    `rate` seconds, in microseconds rounded down (FFmpeg's setpts truncates).
    Given the test video's rate on both inputs, FFmpeg's two-input filters
    pair frame N with frame N. Microseconds, not 1/rate: a duration limit
    (duration_args) between two frames is held to the microsecond, not
    rounded to a whole frame."""
    return ["settb=AVTB", f"setpts=N*{1_000_000 * rate.denominator}/{rate.numerator}"]


def pairing_filters(rate: Fraction) -> list[str]:
    """The filters at the end of each video's chain, before an FFmpeg filter
    pairs the two videos: numbering_filters at `rate` (the test video's), or
    when pairing by timestamp, each video's times started at 0, as before."""
    if by_timestamp():
        return ["setpts=PTS-STARTPTS"]
    return numbering_filters(rate)


def duration_args(duration_limit: str | None) -> list[str]:
    """An FFmpeg output's options for a duration limit (`duration_limit`, as
    -t takes it; none if None): its -t, held against the numbered frames'
    own times (-enc_time_base filter). Without it, FFmpeg holds -t against
    each frame's time in the encoder's time base, rounded to the output's
    own frame rate -- the source's, on an FFmpeg that writes the source --
    and two outputs could keep different frames. When pairing by timestamp,
    the plain -t, as before."""
    if not duration_limit:
        return []
    if by_timestamp():
        return ["-t", duration_limit]
    return ["-enc_time_base", "filter", "-t", duration_limit]


def frames_within(duration_limit: str | None, rate: Fraction) -> int | None:
    """How many numbered frames (numbering_filters, at `rate`) an output's
    duration limit keeps (duration_args) -- those whose time is below it --
    or None without a limit."""
    if not duration_limit:
        return None
    limit_us = _microseconds(duration_limit)
    # Frame N is kept while N * 1000000 / rate < limit_us, to the microsecond.
    return max(0, -(-(limit_us * rate.numerator) // (1_000_000 * rate.denominator)))


def _microseconds(duration: str) -> int:
    """`duration`, in seconds as -t takes it ("30.042"), in whole
    microseconds, as FFmpeg reads it: exactly, the digits past the sixth
    decimal dropped."""
    return int(Decimal(duration).scaleb(6).to_integral_value(rounding=ROUND_DOWN))


def frame_pairs(test: Callable[[], T | None], source: Callable[[], U | None],
                release_test: Callable[[T], None], release_source: Callable[[U], None],
                limit: int | None = None) -> Iterator[tuple[T, U]]:
    """The two videos' frames in pairs, each video's n-th frame with the
    other's, until either video ends or `limit` pairs have been given.
    `test` and `source` give their video's next frame, in display order, or
    None at its end. A pair's frames are released when the next pair is
    asked for, or when the pairing is closed; a test frame taken after the
    source has ended is released at once."""
    paired = 0
    while limit is None or paired < limit:
        test_frame = test()
        if test_frame is None:
            return
        source_frame = source()
        if source_frame is None:
            release_test(test_frame)
            return
        try:
            yield test_frame, source_frame
        finally:
            release_test(test_frame)
            release_source(source_frame)
        paired += 1


def decoded_frame_pairs(test: GpuFrameStream, ref: GpuFrameStream, distorted: VideoInfo,
                        duration_limit: str | None, check_cancel: Callable[[], None],
                        release_test: Callable[[int], None],
                        release_ref: Callable[[int], None]) -> Iterator[tuple[int, int]]:
    """The (test slot, source slot) pairs of two videos decoded in the app
    (gpu_frames.GpuFrameStream, both started), as many as the duration limit
    (the FFmpeg outputs' -t, as text) keeps: frame N with frame N, or when
    pairing by timestamp, as before (frame_sync_by_time.decoded_pairs).
    `check_cancel` is called every 100 ms while a picture is waited for."""
    if by_timestamp():
        from vmaf_app.core import frame_sync_by_time

        return frame_sync_by_time.decoded_pairs(test, ref, duration_limit, check_cancel, release_test, release_ref)
    limit = frames_within(duration_limit, frame_rate(distorted))
    return frame_pairs(test.pictures(check_cancel), ref.pictures(check_cancel), release_test, release_ref, limit)
