"""LockedPresentation: native playback's frames, presented by the app's own
presenter; SingleSoundtrack: the clock they are chosen by."""
from types import SimpleNamespace

import pytest

from vmaf_app.core import locked_presentation
from vmaf_app.core.locked_presentation import SingleSoundtrack, yuv_to_rgb


def test_the_presenters_colours_and_the_soundtracks_clock(subtests, monkeypatch, tmp_path):
    """The presenter's shader samples NV12's 8-bit codes over 255 and P010's
    10-bit codes in the top of 16 bits over 65535; its rows turn them into
    RGB. The soundtrack's clock, which picks the frames, moves in steps of
    10 ms, where a frame of 120 fps video lasts 8.3 ms: one frame in five was
    skipped until it was carried on between its steps."""

    def rgb(rows, y, u, v):
        return tuple(rows[4 * i] * y + rows[4 * i + 1] * u + rows[4 * i + 2] * v + rows[4 * i + 3] for i in range(3))

    for matrix, red in (("bt709", (63, 102, 240)), ("bt601", (81, 90, 240)), ("bt2020", (74, 97, 240))):
        with subtests.test("8-bit limited range", matrix=matrix):
            rows = yuv_to_rgb(matrix, full_range=False, ten_bit=False)
            assert rgb(rows, 235 / 255, 128 / 255, 128 / 255) == pytest.approx((1, 1, 1), abs=1e-9)
            assert rgb(rows, 16 / 255, 128 / 255, 128 / 255) == pytest.approx((0, 0, 0), abs=1e-9)
            # Pure red's codes, as each matrix gives them (Y, Cb, Cr).
            assert rgb(rows, *(code / 255 for code in red)) == pytest.approx((1, 0, 0), abs=0.01)
    with subtests.test("P010, BT.2020 limited range: its white and black"):
        rows = yuv_to_rgb("bt2020", full_range=False, ten_bit=True)
        assert rgb(rows, (940 << 6) / 65535, (512 << 6) / 65535, (512 << 6) / 65535) == pytest.approx((1, 1, 1), abs=1e-9)
        assert rgb(rows, (64 << 6) / 65535, (512 << 6) / 65535, (512 << 6) / 65535) == pytest.approx((0, 0, 0), abs=1e-9)
    with subtests.test("8-bit full range"):
        rows = yuv_to_rgb("bt709", full_range=True, ten_bit=False)
        assert rgb(rows, 1.0, 128 / 255, 128 / 255) == pytest.approx((1, 1, 1), abs=1e-9)
        assert rgb(rows, 0.0, 128 / 255, 128 / 255) == pytest.approx((0, 0, 0), abs=1e-9)

    # The soundtrack on a fake playbin3, its sink's clock at `sink_ms`, timed
    # by a fake monotonic clock (`now`, seconds).
    now, sink_ms = [100.0], [1000]
    pipeline = SimpleNamespace(
        set_property=lambda *_: None, set_state=lambda _state: None,
        get_bus=lambda: SimpleNamespace(pop_filtered=lambda _mask: None),
        query_position=lambda _format: (True, sink_ms[0] * 1_000_000))
    gst = SimpleNamespace(
        ElementFactory=SimpleNamespace(make=lambda *_: pipeline),
        State=SimpleNamespace(PAUSED="paused", PLAYING="playing", NULL="null"),
        MessageType=SimpleNamespace(ERROR=1, ASYNC_DONE=2), Format=SimpleNamespace(TIME="time"),
        MSECOND=1_000_000)
    monkeypatch.setattr(locked_presentation, "_load_gstreamer", lambda: (gst, None))
    monkeypatch.setattr(locked_presentation, "time", SimpleNamespace(monotonic=lambda: now[0]))
    track = SingleSoundtrack(tmp_path / "source.mkv", 0, True)
    track.pending, track.ready = None, True
    track.set_playing(True)
    with subtests.test("carried on between its 10 ms steps: 120 fps video, no frame skipped"):
        frames = []
        for tick in range(250):  # a second of the view's 4 ms ticks
            now[0] = 100 + tick * 0.004
            sink_ms[0] = 1000 + tick * 4 // 10 * 10
            frames.append(round(track.poll() * 120 / 1000))
            assert track.reading_ms == sink_ms[0]  # what falling behind is judged by: not carried
        assert sorted(set(frames)) == list(range(120, 241))  # to 1996 ms: frame 240 is due at 1995.8
    with subtests.test("never further than a step, and never back while playing"):
        now[0] += 1.0  # the sink's clock stopped a second
        stopped = track.poll()
        assert stopped == sink_ms[0] + 10
        sink_ms[0] -= 5
        assert track.poll() == stopped
    with subtests.test("paused, then playing again a second later: no step at once"):
        track.set_playing(False)
        assert track.poll() == sink_ms[0]
        now[0] += 1.0
        track.set_playing(True)
        assert track.poll() == sink_ms[0]
        now[0] += 0.004
        assert track.poll() == pytest.approx(sink_ms[0] + 4)
    with subtests.test("polled once a frame, as the view's ticks come when a frame is due: each frame once"):
        # Carried on from when a poll saw it step, it fell up to a poll
        # behind, then caught up at once: a frame twice, then two at once.
        track.set_playing(False)
        track.set_playing(True)
        start_s, start_ms, frames = now[0], sink_ms[0], []
        for tick in range(120):
            now[0] = start_s + (tick + 0.1) / 120
            sink_ms[0] = start_ms + int((now[0] - start_s) * 1000) // 10 * 10
            frames.append(round(track.poll() * 120 / 1000))
        assert frames == list(range(frames[0], frames[0] + 120))
