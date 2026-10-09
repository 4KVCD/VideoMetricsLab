from pathlib import Path
from types import SimpleNamespace

import pytest

from vmaf_app.core import d3d11_tonemap, gstreamer_playback
from vmaf_app.core.d3d11_tonemap import D3D11ToneMapper
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings
from vmaf_app.core.models import VideoInfo


@pytest.fixture(autouse=True)
def _fake_frame_pointer(monkeypatch, request):
    """The fake buffers' memory is a plain object: its pointer is made up
    (boxed_pointer has tests of its own below)."""
    if "pointer" not in request.node.name and "readings" not in request.node.name:
        monkeypatch.setattr(d3d11_tonemap, "boxed_pointer", lambda _memory: 789)


def _mapper(result=0):
    calls = []
    mapper = object.__new__(D3D11ToneMapper)
    mapper.device = SimpleNamespace(lock=lambda: calls.append("lock"), unlock=lambda: calls.append("unlock"))
    mapper.kind, mapper.handle, mapper.colours = 1, None, None
    mapper.gst = SimpleNamespace(gst_is_d3d11_memory=lambda p: True,
                                 gst_d3d11_memory_get_resource_handle=lambda p: 123)
    mapper.lib = SimpleNamespace(vmaf_tonemap_create=lambda p, k: 456,
                                 vmaf_tonemap_render=lambda h, p: result,
                                 vmaf_tonemap_destroy=lambda h: calls.append("destroy"))
    return mapper, calls


def _buffer(count=1):
    return SimpleNamespace(n_memory=lambda: count, peek_memory=lambda i: object())


def test_shader_failures_unlock_the_device_and_cpu_memory_is_refused(subtests):
    with subtests.test("a native failure unlocks the device"):
        mapper, calls = _mapper(-1)
        with pytest.raises(RuntimeError, match="ffffffff"):
            mapper.render(_buffer())
        assert calls == ["lock", "unlock"]
    with subtests.test("CPU memory is refused before the native render"):
        mapper, calls = _mapper()
        mapper.gst.gst_is_d3d11_memory = lambda p: False
        with pytest.raises(RuntimeError, match="CPU memory"):
            mapper.render(_buffer())
        assert calls == []


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

    monkeypatch.setattr(gstreamer_playback, "gstreamer_available", lambda: (True, ""))
    monkeypatch.setattr(d3d11_tonemap, "available", lambda: True)
    for hdr_display in (False, True):
        settings = PreviewColorSettings(display_hdr_enabled=hdr_display)
        for primaries, converts, native in (("smpte432", True, True), ("smpte432", False, False),
                                            ("bt2020", False, True), ("smpte431", True, False)):
            with subtests.test("native playback", hdr_display=hdr_display, primaries=primaries,
                               shader_converts=converts):
                monkeypatch.setattr(d3d11_tonemap, "converts_primaries", lambda converts=converts: converts)
                monkeypatch.setattr(d3d11_tonemap, "converts_for_hdr", lambda converts=converts: converts)
                assert gstreamer_playback.uses_native_gstreamer(comparison(primaries), settings)[0] is native
                expected = None if primaries == "bt2020" and hdr_display else ("hdr" if hdr_display else "sdr")
                assert gstreamer_playback._shading(comparison(primaries).source_info, settings) == expected
