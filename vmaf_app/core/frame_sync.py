"""How every metric pairs the two videos' frames.

The test video's frame N is compared with the source's frame N, counting
from each video's first frame, until either video ends. A test video that
drops, repeats or adds a frame is therefore compared one frame off from
that point on. Videos whose frame rates differ are refused before they are
compared (geometry.pair_problem).

FFmpeg's two-input filters (libvmaf, xpsnr, and the overlay that pairs the
frames for VMAF on the GPU) pair frames by their timestamps. So each video's
frames are first restamped with their numbers (numbering_filters): frame N
gets the time of N frames of the test video, in both videos. The filters
then pair frame N with frame N. A duration limit (an output's -t, see
duration_args) then keeps the same frames of both videos: those whose new
time is below the limit.

Where the frames come into this process instead (from the GPU decoders, and
from FFmpeg's pipes to Vship and to VMAF on the GPU), frame_pairs pairs the
two videos' n-th frames, and frames_within gives the number of frames a
duration limit keeps, the same number FFmpeg's -t keeps.

The old way of pairing, by timestamp, is kept for now in frame_sync_by_time.
With the environment variable VML_FRAME_PAIRING=timestamp, every metric
pairs frames as it did before (see by_timestamp). The functions here that
build FFmpeg's options (framesync_opts, pairing_filters, duration_args), and
decoded_frame_pairs, follow that switch.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Generator
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


#: FFmpeg's frame sync options for its two-input filters. The comparison ends
#: with the shorter video (shortest=1), and no video's last frame is repeated
#: past its end (repeatlast=0). With FFmpeg's defaults, the extra frames of a
#: test video longer than its source (two frames of encoder padding are
#: common) were compared with a frozen copy of the source's last frame, and
#: their poor scores lowered the video's score.
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
    """The video's frame rate as an exact fraction (24000/1001, not 23.976...),
    which numbered frames are timed by; 25 for a video that gives none."""
    if info.fps <= 0:
        return _FALLBACK_FRAME_RATE
    return Fraction(info.fps).limit_denominator(1_000_000)


def numbering_filters(rate: Fraction) -> list[str]:
    """FFmpeg filters that restamp each frame by its number: frame N gets the
    time N / `rate` seconds, in whole microseconds, rounded down (FFmpeg's
    setpts truncates).

    With the test video's rate on both inputs, FFmpeg's two-input filters
    pair frame N with frame N. The time base is microseconds rather than
    1 / rate so that a duration limit between two frames (duration_args) is
    held to the microsecond instead of being rounded to a whole frame."""
    return ["settb=AVTB", f"setpts=N*{1_000_000 * rate.denominator}/{rate.numerator}"]


def pairing_filters(rate: Fraction) -> list[str]:
    """The filters at the end of each video's chain, before an FFmpeg filter
    pairs the two videos: numbering_filters at `rate` (the test video's), or
    when pairing by timestamp, each video's times started at 0, as before."""
    if by_timestamp():
        return ["setpts=PTS-STARTPTS"]
    return numbering_filters(rate)


def duration_args(duration_limit: str | None) -> list[str]:
    """An FFmpeg output's options for a duration limit, given as -t takes it
    (no options for None).

    The limit is held against the frames' new, numbered times
    (-enc_time_base filter). Without that, FFmpeg also checks -t against
    each frame's time in the encoder's time base, which is rounded to the
    output's own frame rate (the source's, for an FFmpeg that writes the
    source), and two outputs could keep different frames. When pairing by
    timestamp, the plain -t is used, as before."""
    if not duration_limit:
        return []
    if by_timestamp():
        return ["-t", duration_limit]
    return ["-enc_time_base", "filter", "-t", duration_limit]


def frames_within(duration_limit: str | None, rate: Fraction) -> int | None:
    """How many frames, numbered at `rate` (numbering_filters), an output's
    duration limit keeps (duration_args): those whose time is below the
    limit. None when there is no limit."""
    if not duration_limit:
        return None
    limit_us = _microseconds(duration_limit)
    # Frame N is kept while N * 1000000 / rate < limit_us, to the microsecond.
    return max(0, -(-(limit_us * rate.numerator) // (1_000_000 * rate.denominator)))


def _microseconds(duration: str) -> int:
    """A duration in seconds, as -t takes it ("30.042"), in whole
    microseconds, read as FFmpeg reads it: exactly, with any digits past the
    sixth decimal dropped."""
    return int(Decimal(duration).scaleb(6).to_integral_value(rounding=ROUND_DOWN))


def frame_pairs(test: Callable[[], T | None], source: Callable[[], U | None],
                release_test: Callable[[T], None], release_source: Callable[[U], None],
                limit: int | None = None) -> Generator[tuple[T, U]]:
    """Yields the two videos' frames in pairs, each video's n-th frame with
    the other's, until either video ends or `limit` pairs have been given.

    `test` and `source` return their video's next frame, in display order,
    or None at its end. A pair's frames are released when the next pair is
    asked for, or when the pairing is closed. A test frame taken after the
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
                        release_ref: Callable[[int], None]) -> Generator[tuple[int, int]]:
    """Yields (test slot, source slot) pairs of two videos decoded in the app
    (both gpu_frames.GpuFrameStream, already started): frame N with frame N,
    as many as the duration limit (the FFmpeg outputs' -t, as text) keeps.
    When pairing by timestamp, the pairs are made as before
    (frame_sync_by_time.decoded_pairs). `check_cancel` is called every
    100 ms while a picture is waited for."""
    if by_timestamp():
        from vmaf_app.core import frame_sync_by_time

        return frame_sync_by_time.decoded_pairs(test, ref, duration_limit, check_cancel, release_test, release_ref)
    limit = frames_within(duration_limit, frame_rate(distorted))
    return frame_pairs(test.pictures(check_cancel), ref.pictures(check_cancel), release_test, release_ref, limit)
