import time
from types import SimpleNamespace

from vmaf_app.ui.locked_native_pool import LockedNativePool


def pool():
    instance = object.__new__(LockedNativePool)
    instance.source_key = "source"
    instance.selected = 0
    instance.frame, instance.fps, instance.position = 10, 25, 400
    instance.pair = None
    instance.pair_index = None
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


def test_decoder_stall_pauses_audio_and_waits_for_seek_before_resume():
    instance, _output = pool()
    calls = []
    audio = SimpleNamespace(ready=True, failed=None, position=520, offset_ms=0)
    audio.poll = lambda: audio.position
    audio.set_playing = lambda playing: calls.append(("playing", playing))

    def seek(position):
        calls.append(("seek", position))
        audio.ready = False
        audio.position = position

    audio.seek = seek
    instance.audio = audio
    instance.audio_deadline = time.monotonic() + 15
    instance.output.poll = lambda: None
    instance.launch_missing = lambda: None
    instance._pull_sample = lambda key, sink: None
    instance.playing, instance.buffering, instance.audio_running = True, False, True
    instance.status_details, instance.eos = {}, set()
    instance.anchor, instance.anchor_frame = time.monotonic(), 10
    instance.entries = {key: [SimpleNamespace(
        poll=lambda: SimpleNamespace(error=None, status=None, ended=False),
        first_frame_ms=None, _sinks={"video": object()})] for key in instance.frames}
    instance.poll()
    assert calls == [("playing", False), ("seek", 400)]
    assert instance.buffering and not instance.audio_running
    assert instance.pair == ("s10", "d10")
    instance.poll()
    assert len(calls) == 2  # asynchronous audio seek still pending
    audio.ready = True
    instance.poll()
    assert calls[-1] == ("playing", True)
    assert instance.audio_running and not instance.buffering
    instance.frames[("distorted", 0)][11] = "d11"
    audio.position = 440
    instance.poll()
    assert instance.pair == ("s11", "d11")
