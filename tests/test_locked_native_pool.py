from types import SimpleNamespace

from vmaf_app.ui import locked_native_pool
from vmaf_app.ui.locked_native_pool import LockedNativePool


def pool():
    instance = object.__new__(LockedNativePool)
    instance.source_key = "source"
    instance.selected = 0
    instance.frame, instance.fps, instance.position = 10, 25, 400
    instance.pair = None
    instance.pair_index = None
    instance.late_since = None
    instance._clock_ms, instance._clock_at = None, 0.0
    instance.entries = {}
    instance.shadings = {}
    instance.showing_source = False
    instance.frames = {"source": {10: "s10", 11: "s11", 12: "s12"},
                       ("distorted", 0): {10: "d10"}}
    output = []
    instance.output = SimpleNamespace(present=output.append)
    instance.view = SimpleNamespace(native_view=lambda sample: None)  # fitted to the window
    return instance, output


def test_a_zoomed_frame_is_presented_with_the_part_in_view():
    """Zoomed, the view says which part of the frame to show and where
    (VideoCompareView.native_view); fitted, frames go as they always did."""
    instance, _output = pool()
    presented = []
    instance.output = SimpleNamespace(present=lambda *args: presented.append(args))
    instance.view = SimpleNamespace(native_view=lambda sample: ((1, 2, 3, 4), (0, 0, 9, 9)))
    instance.surface = SimpleNamespace(refresh_cursor=lambda: None, update=lambda: None)
    assert instance._choose_pair(12)
    assert presented == [("d10", ((1, 2, 3, 4), (0, 0, 9, 9)))]
    instance.view = SimpleNamespace(native_view=lambda sample: None)
    instance.place()  # the zoom changed: the same frame again
    assert presented[-1] == ("d10",)
    # Shown again after its decoder went -- the encode chosen again before
    # it was back -- the frame is shaded as its video is: it was shown
    # unshaded, HDR as SDR.
    instance.shadings[("distorted", 0)] = (1.0, 1.0)
    instance.entries = {}
    instance.place()
    assert presented[-1] == ("d10", None, (1.0, 1.0))


def test_fast_source_never_advances_past_matching_distorted_frame():
    instance, output = pool()
    assert instance._choose_pair(12)
    assert instance.frame == 10
    assert instance.pair == ("s10", "d10")
    assert output == ["d10"]
    instance.frames[("distorted", 0)][11] = "d11"
    assert instance._choose_pair(12)
    assert instance.frame == 11
    assert instance.pair == ("s11", "d11")


def test_switch_uses_same_frame_from_new_encode():
    instance, output = pool()
    instance._choose_pair(10)
    instance.frames[("distorted", 1)] = {10: "new10"}
    instance.selected = 1
    instance._choose_pair(10)
    assert instance.pair == ("s10", "new10")
    instance.show_source(True)
    assert output == ["d10", "new10", "s10"]


def test_decoder_stall_pauses_audio_and_waits_for_seek_before_resume(monkeypatch):
    """Behind its soundtrack for a moment -- the window's thread held up, the
    decoders waiting with their queues full -- playback catches up with the
    sound playing on: stopped and sought again each time, it broke off every
    15 s. Behind for longer, it stops the sound, waits for the decoders and
    seeks the sound to the frame shown."""
    now = [100.0]
    monkeypatch.setattr(locked_native_pool, "time", SimpleNamespace(monotonic=lambda: now[0]))
    instance, _output = pool()
    calls = []
    audio = SimpleNamespace(ready=True, failed=None, position=520, offset_ms=0, reading_ms=None)

    def poll():
        audio.reading_ms = audio.position  # its clock's own reading: no carrying on here
        return audio.position

    audio.poll = poll
    audio.set_playing = lambda playing: calls.append(("playing", playing))

    def seek(position):
        calls.append(("seek", position))
        audio.ready = False
        audio.position = position

    audio.seek = seek
    instance.audio = audio
    instance.audio_deadline = now[0] + 15
    instance.output.poll = lambda: None
    instance.launch_missing = lambda: None
    instance._pull_sample = lambda key, sink: None
    instance.playing, instance.buffering, instance.audio_running = True, False, True
    instance.status_details, instance.eos = {}, set()
    instance.anchor, instance.anchor_frame = now[0], 10
    instance.entries = {key: [SimpleNamespace(
        poll=lambda: SimpleNamespace(error=None, status=None, ended=False),
        first_frame_ms=None, _sinks={"video": object()}, seeking=False)] for key in instance.frames}
    instance.poll()  # frame 13 due, 10 the newest pair
    assert calls == [] and not instance.buffering
    assert instance.pair == ("s10", "d10")
    now[0] += 0.1
    instance.frames["source"][13] = "s13"
    instance.frames[("distorted", 0)].update({11: "d11", 12: "d12", 13: "d13"})
    instance.poll()  # caught up, the sound never stopped
    assert calls == [] and not instance.buffering
    assert instance.pair == ("s13", "d13")
    # Frame 14 is chosen from 540 ms on the clock (13.5 frames of 40 ms):
    # the next tick 1 ms past it, not every half frame.
    assert instance.next_tick_ms() == 21
    now[0] += 0.015
    assert instance.next_tick_ms() == 6
    audio.position = 640  # frame 16 due; the decoders stalled at 13
    instance.poll()
    assert calls == []
    assert instance.next_tick_ms() is None  # due, not decoded: the view's busy pace
    now[0] += 0.25
    instance.poll()
    assert calls == [("playing", False), ("seek", 520)]
    assert instance.buffering and not instance.audio_running
    assert instance.pair == ("s13", "d13")
    instance.poll()
    assert len(calls) == 2  # asynchronous audio seek still pending
    audio.ready = True
    instance.poll()
    assert calls[-1] == ("playing", True)
    assert instance.audio_running and not instance.buffering
    instance.frames["source"][14] = "s14"
    instance.frames[("distorted", 0)][14] = "d14"
    audio.position = 560
    instance.poll()
    assert instance.pair == ("s14", "d14")
    # The encode decoded ahead of its source, the clock ahead of both: the
    # encode's queue keeps the frames the source has yet to reach. Trimmed
    # from the clock, it dropped them, and the pair waited for never came.
    instance.frames["source"] = {13: "s13", 14: "s14"}
    instance.frames[("distorted", 0)] = {frame: f"d{frame}" for frame in range(14, 19)}
    audio.position = 720  # frame 18 due
    instance.poll()
    assert 14 in instance.frames[("distorted", 0)]
    # Paused on the pair on screen it is locked, not buffering -- though
    # still set to wait for the sound's seek, as after one made while paused.
    instance.set_playing(False)
    instance.buffering = True
    instance.poll()
    assert instance.description.startswith("Frame-locked GPU pair")
    instance.frame = 15  # sought to a frame not decoded yet
    instance.poll()
    assert instance.description.startswith("Buffering locked pair")
    # A video whose seek is still to be made gives the pool no frames: its
    # sink can still hold frames from before the seek, which a seek back
    # took for frames to come.
    pulled = []
    instance._pull_sample = lambda key, sink: pulled.append(key)
    for entry in instance.entries.values():
        entry[0].seeking = True
    instance.poll()
    assert pulled == []
