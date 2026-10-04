"""The frame pairs FFmpeg's libvmaf filter compares, worked out from the two
videos' timestamps: FFmpeg's frame sync (libavfilter/framesync.c) with the
options vmaf_runner gives libvmaf and the GPU's overlay pairing
(FRAMESYNC_OPTS: shortest=1, repeatlast=0, ts_sync_mode=nearest). VMAF on
the GPU uses it for the videos decoded in its own process
(vmaf_cuda.score_decoded), where no FFmpeg filter graph pairs them.

How that frame sync pairs two inputs, for this configuration:

- each input's timestamps start at 0 (the chains' setpts=PTS-STARTPTS), and
  the second input's are rescaled to the first's time base, rounded to
  nearest (framesync's time base is that of its only synchronising input);
- each frame of the first input (the test video, the "main" input) gives
  one pair, when it is the earliest frame either input has next;
- the second (the source) moves on to a frame of its own when that frame is
  the earliest next, or -- "nearest" -- when that frame is strictly nearer
  the time being decided than the frame it has: a source frame stamped a
  millisecond after a test frame pairs with it, not with the one before;
- when the test video ends, so does the comparison; when the source ends,
  its end is stamped one tick after its last frame, and the comparison ends
  at the first test frame at or past it (shortest=1).

tests/test_frame_sync.py checks it against FFmpeg's own pairing (the
overlay vmaf_runner._gpu_pairs_stage uses) on many patterns of timestamps.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Generic, TypeVar

from vmaf_app.core.gpu_frames import rescale

T = TypeVar("T")
U = TypeVar("U")

#: Both libvmaf and xpsnr are framesync filters, and framesync's defaults are
#: wrong for measurement: repeatlast=true extends the last frame of the
#: secondary input past its EOF, and eof_action=repeat keeps the comparison
#: going. A distorted file two frames longer than the source -- routine
#: encoder padding, and well inside the duration tolerance -- therefore got
#: two extra "scores" comparing real distorted frames against a frozen copy
#: of the source's final frame. Those frames score terribly (48 and 31 on a
#: 30-frame fixture that is otherwise ~100) and drag the aggregate down, so
#: the run silently reports a worse encode than was delivered.
#:
#: The default ts_sync_mode pairs each distorted frame with the last source
#: frame at or before its timestamp. Two files with the same frames can have
#: timestamps a millisecond apart -- MKV stores whole milliseconds, and each
#: program rounds frame times from its own clock -- and a distorted frame
#: stamped 1 ms early was compared with the source's previous frame: VMAF 0
#: and XPSNR ~16 dB at scene cuts and in motion, on an anime episode whose
#: SSIMULACRA2 (paired frame by frame) was 93 on the same frame. "nearest"
#: takes the source frame nearest in time, the same frame whichever way the
#: two timestamps are off by less than half a frame.
FRAMESYNC_OPTS = ["shortest=1", "repeatlast=0", "ts_sync_mode=nearest"]

_END = (1 << 63) - 1  # INT64_MAX: an input that ended with nothing to extrapolate from


class _Input(Generic[T]):
    def __init__(self, pull: Callable[[], tuple[T, int] | None], release: Callable[[T], None],
                 to_sync_time: Callable[[int], int], nearest: bool) -> None:
        self.pull = pull
        self.release = release
        self.to_sync_time = to_sync_time
        self.nearest = nearest
        self.first: int | None = None    # the STARTPTS of its setpts
        self.ended = False               # the input's frames have all been pulled
        self.done = False                # its end has been taken (framesync's STATE_EOF)
        self.frame: T | None = None
        self.pts: int | None = None      # None: no frame taken yet (AV_NOPTS_VALUE)
        self.next: T | None = None
        self.next_pts: int | None = None
        self.has_next = False

    def fill(self) -> None:
        """framesync's consume_from_fifos for one input: its next frame, or
        its end, stamped as framesync stamps it with after=EXT_STOP."""
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
        old = self.frame
        self.frame, self.pts = self.next, self.next_pts
        self.next, self.next_pts, self.has_next = None, None, False
        if self.frame is None:
            self.done = True
        if old is not None:
            self.release(old)

    def drop(self) -> None:
        for frame in (self.frame, self.next):
            if frame is not None:
                self.release(frame)
        self.frame = self.next = None


def frame_pairs(
    main: Callable[[], tuple[T, int] | None], source: Callable[[], tuple[U, int] | None],
    main_time_base: Fraction, source_time_base: Fraction,
    release_main: Callable[[T], None], release_source: Callable[[U], None],
) -> Iterator[tuple[T, U | None, int]]:
    """(test frame, source frame, time) for each pair, the time in the test
    video's time base from its first frame. `main` and `source` give each
    video's next (frame, pts) in display order, or None at its end; a frame
    is handed to its release function once no later pair can use it -- after
    the next pair is asked for, so a pair's frames stay valid until then.
    The source frame is None only where FFmpeg's filter would see none (the
    source has no frame by the test video's first); callers refuse that."""
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
                    # The test video's end drops the sync level to 0: the
                    # comparison is over (framesync_sync_level_update).
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
                        if put.done:  # after=EXT_STOP (shortest=1): the end of the comparison
                            ended, ready = True, False
            if ended:
                return
            yield test.frame, ref.frame, test.pts
    finally:
        test.drop()
        ref.drop()
