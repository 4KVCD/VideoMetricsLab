from dataclasses import replace
from enum import IntFlag
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmaf_app.core import d3d11_tonemap, gstreamer_playback
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorMode,
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


def test_display_aware_hdr_uses_gpu_shader_when_windows_hdr_is_off(
    monkeypatch,
):
    monkeypatch.setattr(
        gstreamer_playback, "gstreamer_available", lambda: (True, "")
    )
    monkeypatch.setattr(d3d11_tonemap, "available", lambda: True)

    native, reason = gstreamer_playback.uses_native_gstreamer(
        _comparison(), PreviewColorSettings(display_hdr_enabled=True)
    )
    assert native is True
    assert reason == ""

    native, reason = gstreamer_playback.uses_native_gstreamer(
        _comparison(), PreviewColorSettings(display_hdr_enabled=False)
    )
    assert native is True
    assert reason == ""


def test_explicit_hdr_to_sdr_keeps_the_high_quality_ffmpeg_fallback(monkeypatch):
    monkeypatch.setattr(
        gstreamer_playback, "gstreamer_available", lambda: (True, "")
    )
    monkeypatch.setattr(d3d11_tonemap, "available", lambda: False)

    native, reason = gstreamer_playback.uses_native_gstreamer(
        _comparison(),
        PreviewColorSettings(
            mode=PreviewColorMode.HDR_TO_SDR,
            display_hdr_enabled=True,
        ),
    )

    assert native is False
    assert "FFmpeg tone mapping" in reason


@pytest.mark.parametrize("hdr_display", [True, False, None])
def test_explicit_hdr_to_sdr_uses_native_shader_regardless_of_display(monkeypatch, hdr_display):
    monkeypatch.setattr(gstreamer_playback, "gstreamer_available", lambda: (True, ""))
    monkeypatch.setattr(d3d11_tonemap, "available", lambda: True)
    settings = PreviewColorSettings(PreviewColorMode.HDR_TO_SDR, hdr_display)
    assert gstreamer_playback.uses_native_gstreamer(_comparison(), settings) == (True, "")


def test_unmanaged_does_not_accidentally_use_sink_hdr_processing(monkeypatch):
    monkeypatch.setattr(gstreamer_playback, "gstreamer_available", lambda: (True, ""))
    settings = PreviewColorSettings(PreviewColorMode.UNMANAGED)
    native, reason = gstreamer_playback.uses_native_gstreamer(_comparison(), settings)
    assert not native
    assert "bypass" in reason


def test_explicit_untagged_hdr_retains_ffmpeg_interpretation(monkeypatch):
    monkeypatch.setattr(gstreamer_playback, "gstreamer_available", lambda: (True, ""))
    comparison = _comparison()
    comparison = replace(comparison, source_info=replace(comparison.source_info, color_transfer=""))
    native, reason = gstreamer_playback.uses_native_gstreamer(
        comparison, PreviewColorSettings(PreviewColorMode.HDR_TO_SDR))
    assert not native
    assert "untagged" in reason


def test_sdr_uses_native_gpu_presentation_even_if_windows_hdr_is_off(monkeypatch):
    monkeypatch.setattr(
        gstreamer_playback, "gstreamer_available", lambda: (True, "")
    )
    comparison = _comparison()
    comparison = replace(
        comparison,
        source_info=replace(
            comparison.source_info,
            pix_fmt="yuv420p",
            color_transfer="bt709",
            color_primaries="bt709",
            color_space="bt709",
        ),
        distorted_info=replace(
            comparison.distorted_info,
            pix_fmt="yuv420p",
            color_transfer="bt709",
            color_primaries="bt709",
            color_space="bt709",
        ),
    )

    native, reason = gstreamer_playback.uses_native_gstreamer(
        comparison, PreviewColorSettings(display_hdr_enabled=False)
    )

    assert native is True
    assert reason == ""
    assert "format=NV12" in gstreamer_playback.output_caps_string(
        comparison, PreviewColorSettings(display_hdr_enabled=False), "distorted"
    )


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
    player = object.__new__(gstreamer_playback.GstComparePipeline)
    player.Gst = SimpleNamespace(
        State=SimpleNamespace(PAUSED="paused", PLAYING="playing"),
        StateChangeReturn=SimpleNamespace(FAILURE="failure"),
        Format=SimpleNamespace(TIME="time"),
        SeekFlags=SimpleNamespace(FLUSH=1, ACCURATE=2),
        MessageType=_MessageType,
        MSECOND=1_000_000,
    )
    player._pipeline = _FakePipeline()
    player._bus = _FakeBus()
    player._decoder_status_reported = True
    player._ready = False
    player._initial_seek_sent = False
    player._pending_initial_seek_ms = None
    player._tone_error = None

    player.start(2500, True)

    assert player._pipeline.states == ["paused"]
    assert player._pipeline.seeks == []

    player._bus.messages.append(SimpleNamespace(type=_MessageType.ASYNC_DONE))
    player.poll()

    assert player._ready is False
    assert player._pipeline.seeks[-1][-1] == 2_500_000_000

    player._bus.messages.append(SimpleNamespace(type=_MessageType.ASYNC_DONE))
    player.poll()

    assert player._ready is True
    assert player._pipeline.states[-1] == "playing"
