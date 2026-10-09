from dataclasses import replace
from enum import IntFlag
from pathlib import Path
from types import SimpleNamespace

from vmaf_app.core import gstreamer_playback
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorSettings,
)
from vmaf_app.core.models import CropBox, VideoInfo


def _info(path: str, *, transfer: str = "smpte2084") -> VideoInfo:
    return VideoInfo(
        path=Path(path),
        width=3840,
        height=2160,
        fps=24000 / 1001,
        duration=120,
        nb_frames=2877,
        codec_name="hevc",
        pix_fmt="yuv420p10le",
        color_range="tv",
        color_space="bt2020nc",
        color_transfer=transfer,
        color_primaries="bt2020",
    )


def _comparison() -> FrameComparison:
    return FrameComparison(
        source_info=_info("source.mkv"),
        distorted_info=replace(_info("distorted.mkv"), height=1608),
        source_crop=CropBox(3840, 1608, 0, 276),
        distorted_crop=CropBox(3840, 1608, 0, 0),
        fps=24000 / 1001,
        frame_count=2877,
    )


def test_native_hdr_caps_keep_full_cropped_resolution_and_precision():
    caps = gstreamer_playback.output_caps_string(
        _comparison(),
        PreviewColorSettings(display_hdr_enabled=True),
        "source",
    )

    assert "memory:D3D11Memory" in caps
    assert "format=P010_10LE" in caps
    assert "width=3840" in caps
    assert "height=1608" in caps
    assert "colorimetry=bt2100-pq" in caps


def test_mixed_hdr_and_sdr_inputs_keep_independent_native_caps():
    comparison = _comparison()
    comparison = replace(
        comparison,
        distorted_info=replace(
            comparison.distorted_info,
            pix_fmt="yuv420p",
            color_transfer="bt709",
            color_primaries="bt709",
            color_space="bt709",
        ),
    )
    settings = PreviewColorSettings(display_hdr_enabled=True)

    source_caps = gstreamer_playback.output_caps_string(
        comparison, settings, "source"
    )
    distorted_caps = gstreamer_playback.output_caps_string(
        comparison, settings, "distorted"
    )

    assert "format=P010_10LE" in source_caps
    assert "colorimetry=bt2100-pq" in source_caps
    assert "format=NV12" in distorted_caps
    assert "bt2100-pq" not in distorted_caps


class _MessageType(IntFlag):
    ERROR = 1
    EOS = 2
    ASYNC_DONE = 4


class _FakePipeline:
    def __init__(self) -> None:
        self.states = []
        self.seeks = []

    def set_state(self, state):
        self.states.append(state)
        return "success"

    def seek_simple(self, fmt, flags, position):
        self.seeks.append((fmt, flags, position))
        return True

    def query_position(self, _fmt):
        return False, 0


class _FakeBus:
    def __init__(self) -> None:
        self.messages = []

    def pop_filtered(self, _types):
        return self.messages.pop(0) if self.messages else None


def test_initial_seek_waits_for_both_native_sinks_to_preroll():
    """And counts from the video's first frame, which the first preroll,
    from the file's start, gives: VideoQ's MP4 source has it 32 ms into its
    timeline, where its encodes have it at 0 -- frames counted from the
    timeline's start were paired 4 apart and never played from frame 0."""
    player = object.__new__(gstreamer_playback.GstComparePipeline)
    player.Gst = SimpleNamespace(
        State=SimpleNamespace(PAUSED="paused", PLAYING="playing"),
        StateChangeReturn=SimpleNamespace(FAILURE="failure"),
        Format=SimpleNamespace(TIME="time"),
        SeekFlags=SimpleNamespace(FLUSH=1, ACCURATE=2),
        MessageType=_MessageType,
        MSECOND=1_000_000,
        CLOCK_TIME_NONE=2**64 - 1,
    )
    player._pipeline = _FakePipeline()
    player._bus = _FakeBus()
    player._decoder_status_reported = True
    player._ready = False
    player._initial_seek_sent = False
    player._pending_initial_seek_ms = None
    player._tone_error = None
    player._first_frame = None

    def sample(pts):
        return SimpleNamespace(get_segment=lambda: SimpleNamespace(to_stream_time=lambda _format, time: time),
                               get_buffer=lambda: SimpleNamespace(pts=pts))

    player._sinks = {"source": SimpleNamespace(emit=lambda _signal, _timeout: sample(32_031_000))}

    player.start(2500, True)

    assert player._pipeline.states == ["paused"]
    assert player._pipeline.seeks == []

    player._bus.messages.append(SimpleNamespace(type=_MessageType.ASYNC_DONE))
    player.poll()

    assert player._ready is False
    assert player._pipeline.seeks[-1][-1] == 2_500_000_000 + 32_031_000
    assert player.first_frame_ms == 32
    assert player.frame_time(sample(5_032_031_000)) == 5_000_000_000

    player._bus.messages.append(SimpleNamespace(type=_MessageType.ASYNC_DONE))
    player.poll()

    assert player._ready is True
    assert player._pipeline.states[-1] == "playing"
