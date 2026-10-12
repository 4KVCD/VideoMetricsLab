"""Pairs the two videos' frames by timestamp. This is the old way of pairing,
kept for now beside the pairing by frame number (frame_sync), and used only
when the environment variable VML_FRAME_PAIRING is "timestamp" (see
frame_sync.by_timestamp).

It works out, from the two videos' timestamps, the frame pairs that FFmpeg's
libvmaf filter would compare. It follows FFmpeg's frame sync
(libavfilter/framesync.c) with the options vmaf_runner gives libvmaf and the
GPU's overlay pairing (FRAMESYNC_OPTS: shortest=1, repeatlast=0,
ts_sync_mode=nearest). VMAF on the GPU uses it for videos decoded in the
app's own process (the score_decoded functions), where no FFmpeg filter
graph pairs the frames.

With these options, FFmpeg's frame sync pairs two inputs as follows:

- Each input's timestamps start at 0 (the filter chains'
  setpts=PTS-STARTPTS). The second input's timestamps are rescaled to the
  first's time base and rounded to nearest, because the frame sync's time
  base is that of its only synchronising input.
- Each frame of the first input (the test video, the "main" input) gives one
  pair, when it is the earliest frame that either input has next.
- The second input (the source) moves on to its next frame when that frame
  is the earliest next frame, or ("nearest") when that frame is strictly
  nearer to the time being decided than its current frame. So a source frame
  stamped a millisecond after a test frame is paired with that test frame,
  not with the one before.
- When the test video ends, so does the comparison. When the source ends,
  its end is stamped one tick after its last frame, and the comparison ends
  at the first test frame at or after that time (shortest=1).

tests/test_frame_sync_by_time.py checks this against FFmpeg's own pairing
(the overlay that vmaf_runner._gpu_pairs_stage uses) on many patterns of
timestamps.
"""
from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Generic, TypeVar

from vmaf_app.core.gpu_frames import GpuDecodeFailedError, GpuFrameStream

T = TypeVar("T")
U = TypeVar("U")

#: The frame sync options for libvmaf and xpsnr, which are both framesync
#: filters. framesync's defaults are wrong for measurement:
#:
#: - repeatlast=1 repeats the second input's last frame after it ends, and
#:   eof_action=repeat keeps the comparison going. A test video two frames
#:   longer than the source (common encoder padding) would get two extra
#:   scores, of real test frames against a frozen copy of the source's last
#:   frame. Those score very low (48 and 31 on a 30-frame test clip that
#:   otherwise scores about 100) and pull the average down.
#: - The default ts_sync_mode pairs each test frame with the last source
#:   frame at or before its timestamp. Two files with the same frames can
#:   have timestamps a millisecond apart (MKV stores whole milliseconds, and
#:   each program rounds frame times its own way), so a test frame stamped
#:   1 ms early would be compared with the source's previous frame. At scene
#:   cuts and in motion, that gave VMAF 0 and XPSNR about 16 dB on frames
#:   that SSIMULACRA2, paired frame by frame, scored 93. "nearest" takes the
#:   source frame nearest in time, which is the same frame whichever way the
#:   two timestamps differ, as long as they differ by less than half a frame.
FRAMESYNC_OPTS = ["shortest=1", "repeatlast=0", "ts_sync_mode=nearest"]

_END = (1 << 63) - 1  # INT64_MAX: the end time of an input that ended before giving any frame


def _ffmpeg_time(text: str) -> int:
    """Converts an FFmpeg duration in seconds ("30.042") to microseconds as
    av_parse_time reads it: exactly, cut off after the sixth decimal."""
    return int(Decimal(text).scaleb(6).to_integral_value(rounding="ROUND_DOWN"))


def rescale(value: int, source: Fraction, target: Fraction) -> int:
    """Converts `value` from time base `source` to time base `target` as
    av_rescale_q does: rounded to nearest, with halves rounded away from
    zero."""
    b = source.numerator * target.denominator
    c = target.numerator * source.denominator
    product = value * b
    if product >= 0:
        return (product + c // 2) // c
    return -((-product + c // 2) // c)


def duration_in(text: str, time_base: Fraction) -> int:
    """Converts FFmpeg's output option -t `text` to `time_base`, as FFmpeg's
    trim filter holds it. A picture is kept while its time from the first
    picture is below the result."""
    return rescale(_ffmpeg_time(text), Fraction(1, 1_000_000), time_base)


class _Input(Generic[T]):
    """One input of the frame sync: its current frame and its next one, with
    their times in the frame sync's time base."""

    def __init__(self, pull: Callable[[], tuple[T, int] | None], release: Callable[[T], None],
                 to_sync_time: Callable[[int], int], nearest: bool) -> None:
        self.pull = pull
        self.release = release
        self.to_sync_time = to_sync_time
        self.nearest = nearest
        self.first: int | None = None    # the first timestamp (STARTPTS in setpts=PTS-STARTPTS)
        self.ended = False               # whether all of the input's frames have been pulled
        self.done = False                # whether its end has been taken (framesync's STATE_EOF)
        self.frame: T | None = None
        self.pts: int | None = None      # None until a frame is taken (AV_NOPTS_VALUE)
        self.next: T | None = None
        self.next_pts: int | None = None
        self.has_next = False

    def fill(self) -> None:
        """Fetches the input's next frame, or its end, and stamps it as
        framesync does with after=EXT_STOP (framesync's consume_from_fifos,
        for one input)."""
        if self.has_next or self.done:
            return
        item = None if self.ended else self.pull()
        if item is None:
            self.ended = True
            self.next = None
            self.next_pts = _END if self.pts is None else self.pts + 1
        else:
            frame, pts = item
            if self.first is None:
                self.first = pts
            self.next, self.next_pts = frame, self.to_sync_time(pts - self.first)
        self.has_next = True

    def take(self) -> None:
        """Makes the next frame the current one, and releases the frame it
        replaces."""
        old = self.frame
        self.frame, self.pts = self.next, self.next_pts
        self.next, self.next_pts, self.has_next = None, None, False
        if self.frame is None:
            self.done = True
        if old is not None:
            self.release(old)

    def drop(self) -> None:
        """Releases the frames the input still holds."""
        for frame in (self.frame, self.next):
            if frame is not None:
                self.release(frame)
        self.frame = self.next = None


def frame_pairs(
    main: Callable[[], tuple[T, int] | None], source: Callable[[], tuple[U, int] | None],
    main_time_base: Fraction, source_time_base: Fraction,
    release_main: Callable[[T], None], release_source: Callable[[U], None],
) -> Iterator[tuple[T, U | None, int]]:
    """Yields (test frame, source frame, time) for each pair, with the time
    in the test video's time base, counted from its first frame.

    `main` and `source` return each video's next (frame, pts) in display
    order, or None at its end. A frame is passed to its release function
    once no later pair can use it, which is only after the next pair is
    asked for, so a pair's frames stay valid until then. The source frame is
    None only where FFmpeg's filter would have none (the source has no frame
    by the time of the test video's first), and callers refuse that."""
    test = _Input(main, release_main, lambda pts: pts, nearest=False)
    ref = _Input(source, release_source, lambda pts: rescale(pts, source_time_base, main_time_base), nearest=True)
    inputs = (test, ref)
    try:
        while True:
            ready = ended = False
            while not (ready or ended):
                for put in inputs:
                    put.fill()
                if test.ended:
                    # The test video's end drops framesync's sync level to 0,
                    # which ends the comparison (framesync_sync_level_update).
                    return
                now = min(put.next_pts for put in inputs if put.has_next)
                if now == _END:
                    return
                for put in inputs:
                    if not put.has_next:
                        continue
                    if put.next_pts == now or (put.nearest and put.next_pts != _END and put.pts is not None
                                               and put.next_pts - now < now - put.pts):
                        put.take()
                        if put is test and test.frame is not None:
                            ready = True
                        if put.done:  # with after=EXT_STOP (shortest=1), an input's end ends the comparison
                            ended, ready = True, False
            if ended:
                return
            yield test.frame, ref.frame, test.pts
    finally:
        test.drop()
        ref.drop()


def decoded_pairs(test: GpuFrameStream, ref: GpuFrameStream, duration_limit: str | None,
                  check_cancel: Callable[[], None], release_test: Callable[[int], None],
                  release_ref: Callable[[int], None]) -> Generator[tuple[int, int]]:
    """Yields the (test slot, source slot) pairs of two videos decoded in the
    app, paired by timestamp as FFmpeg's overlay pairs them (see
    frame_pairs). Pairs are given while the test frame's time from the first
    frame is below `duration_limit` (the output's -t, as text) in the test
    video's time base, which is where FFmpeg's trim filter cuts. Both streams
    must already be started."""
    test_base, ref_base = test.wait_time_base(), ref.wait_time_base()
    stop = duration_in(duration_limit, test_base) if duration_limit else None

    def puller(stream: GpuFrameStream) -> Callable[[], tuple[int, int] | None]:
        def pull() -> tuple[int, int] | None:
            while True:
                check_cancel()
                try:
                    return stream.next(100)
                except TimeoutError:
                    continue
        return pull

    pairs = frame_pairs(puller(test), puller(ref), test_base, ref_base, release_test, release_ref)
    try:
        for test_slot, ref_slot, when in pairs:
            if stop is not None and when >= stop:
                return
            if ref_slot is None:
                raise GpuDecodeFailedError("the source has no frame for the test video's first")
            yield test_slot, ref_slot
    finally:
        pairs.close()
