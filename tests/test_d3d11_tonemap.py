import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmaf_app.core import d3d11_tonemap, gstreamer_playback, locked_presentation
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings
from vmaf_app.core.models import VideoInfo


def _presentation(monkeypatch, result=0):
    """A LockedPresentation with its native side faked: draws return
    `result`; the calls to lock the device, to unlock it and to draw, kept."""
    calls = []
    monkeypatch.setattr(locked_presentation, "boxed_pointer", lambda _memory: 789)
    presentation = object.__new__(locked_presentation.LockedPresentation)
    presentation.device = SimpleNamespace(lock=lambda: calls.append("lock"), unlock=lambda: calls.append("unlock"))
    presentation._gst = SimpleNamespace(gst_is_d3d11_memory=lambda p: True,
                                        gst_d3d11_memory_get_resource_handle=lambda p: 123,
                                        gst_d3d11_memory_get_subresource_index=lambda p: 7)

    def draw(*args):
        calls.append(("draw", args[6], args[7]))  # the shading and the colour space
        presentation.drawn_from = list(args[3])  # the part of the frame's texture drawn
        presentation.drawn_slice = args[2]  # the frame's slice of its texture
        return result
    presentation._lib = SimpleNamespace(vmaf_present_texture=draw)
    presentation._crop_meta = (lambda _buffer, _api: None, 0)  # no crop meta on the frames
    presentation._presenter, presentation._error, presentation._shadings = 1, None, {}
    presentation._clear = None
    presentation._colours_of = lambda caps: (None, locked_presentation._SDR_SPACE, (16, 16))
    return presentation, calls


def _sample(count=1):
    buffer = SimpleNamespace(n_memory=lambda: count, peek_memory=lambda i: object())
    return SimpleNamespace(get_buffer=lambda: buffer, get_caps=lambda: None)


def test_draw_failures_unlock_the_device_and_cpu_memory_is_refused(monkeypatch, subtests):
    with subtests.test("a native failure unlocks the device, and is reported"):
        presentation, calls = _presentation(monkeypatch, -1)
        presentation._draw(_sample(), None, None)
        assert calls == ["lock", ("draw", None, locked_presentation._SDR_SPACE), "unlock"]
        assert "ffffffff" in presentation._error
    with subtests.test("CPU memory is refused before the native draw"):
        presentation, calls = _presentation(monkeypatch)
        presentation._gst.gst_is_d3d11_memory = lambda p: False
        presentation._draw(_sample(), None, None)
        assert calls == [] and "CPU memory" in presentation._error
    with subtests.test("shaded HDR: the shading reaches the draw, the swapchain told SDR or HDR10"):
        for hdr, space in ((False, locked_presentation._SDR_SPACE), (True, locked_presentation._HDR10_SPACE)):
            presentation, calls = _presentation(monkeypatch)
            shading = d3d11_tonemap.shading("HDR10 / PQ", "smpte432", hdr=hdr)
            presentation._draw(_sample(), None, shading)
            drawn = calls[1]
            assert list(drawn[1]) == pytest.approx(shading) and drawn[2] == space
    with subtests.test("cropped and passed on whole: drawn from where its crop meta puts the picture"):
        # A letterboxed source cropped with nothing else to do: the
        # decoder's frame, its bars in it (GstComparePipeline).
        presentation, _calls = _presentation(monkeypatch)
        presentation._draw(_sample(), None, None)
        assert presentation.drawn_from == [0, 0, 16, 16]
        # The frame's slice of its texture goes with it: H.264's decoder
        # gives arrays, which the presenter copies from (present_source).
        assert presentation.drawn_slice == 7
        picture = locked_presentation._CropMeta(x=0, y=276, width=16, height=16)
        presentation._crop_meta = (lambda _buffer, _api: ctypes.addressof(picture), 0)
        presentation._draw(_sample(), None, None)
        assert presentation.drawn_from == [0, 276, 16, 16]
        presentation._draw(_sample(), ((2, 3, 8, 8), (0, 0, 100, 100)), None)  # zoomed: the part in view
        assert presentation.drawn_from == [2, 279, 8, 8]


def test_a_videos_primaries_reach_the_shader(monkeypatch, subtests):
    """Display P3 HDR (VideoQ's) plays natively, its colours converted from
    P3: it was refused, the shader taking every video for BT.2020. On an
    HDR display, which takes BT.2020, it is converted to that, HDR kept: it
    was passed through as it came, its colours oversaturated."""
    with subtests.test("BT.2020's conversion is the shader's as it was, P3's as published"):
        luma, to_bt709 = d3d11_tonemap.conversion("bt2020")
        assert luma == pytest.approx((.2627, .678, .0593), abs=1e-5)
        assert to_bt709 == pytest.approx((1.660491, -.587641, -.072850, -.124550, 1.132900, -.008349,
                                          -.018151, -.100579, 1.118730), abs=1e-6)
        assert d3d11_tonemap.conversion("smpte432")[1] == pytest.approx(
            (1.224940, -.224940, 0, -.042057, 1.042057, 0, -.019638, -.078636, 1.098274), abs=1e-6)
        assert d3d11_tonemap.conversion("") == d3d11_tonemap.conversion("bt2020")
        assert d3d11_tonemap.conversion("smpte432", "bt2020")[1] == pytest.approx(
            (.753833, .198597, .047570, .045744, .941777, .012479, -.001210, .017602, .983609), abs=1e-6)

    def comparison(primaries):
        info = VideoInfo(path=Path("p3.mkv"), width=3840, height=2160, fps=120.0, duration=10.0, nb_frames=1200,
                         codec_name="hevc", pix_fmt="yuv420p10le", color_range="tv", color_space="bt2020nc",
                         color_transfer="smpte2084", color_primaries=primaries)
        return FrameComparison(source_info=info, distorted_info=info, fps=120.0, frame_count=1200)

    with subtests.test("what the presenter's shader is given"):
        sdr = d3d11_tonemap.shading("HDR10 / PQ", "smpte432", hdr=False)
        hdr = d3d11_tonemap.shading("HLG", "smpte432", hdr=True)
        assert sdr[:2] == (1.0, 1.0) and hdr[:2] == (2.0, 2.0)
        assert sdr[2:5] == pytest.approx(d3d11_tonemap.conversion("smpte432")[0])
        assert sdr[5:14] == pytest.approx(d3d11_tonemap.conversion("smpte432")[1])
        assert hdr[5:14] == pytest.approx(d3d11_tonemap.conversion("smpte432", "bt2020")[1])
        assert sdr[14:] == hdr[14:] == (1000.0, 100.0)

    monkeypatch.setattr(gstreamer_playback, "gstreamer_available", lambda: (True, ""))
    for hdr_display in (False, True):
        settings = PreviewColorSettings(display_hdr_enabled=hdr_display)
        for primaries, presents, native in (("smpte432", True, True), ("smpte432", False, False),
                                            ("bt2020", True, True), ("smpte431", True, False)):
            with subtests.test("native playback", hdr_display=hdr_display, primaries=primaries,
                               presenter_built=presents):
                monkeypatch.setattr(d3d11_tonemap, "presents", lambda presents=presents: presents)
                assert gstreamer_playback.uses_native_gstreamer(comparison(primaries), settings)[0] is native
                expected = None if primaries == "bt2020" and hdr_display else ("hdr" if hdr_display else "sdr")
                assert gstreamer_playback._shading(comparison(primaries).source_info, settings) == expected
