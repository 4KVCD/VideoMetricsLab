from dataclasses import replace
from pathlib import Path

from vmaf_app.core import video_playback
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorSettings,
)
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import (
    build_audio_command,
    build_video_series_command,
    playback_dimensions,
)


def _info(path: str, codec: str = "hevc") -> VideoInfo:
    return VideoInfo(
        path=Path(path),
        width=3840,
        height=2160,
        fps=24000 / 1001,
        duration=120.0,
        nb_frames=2877,
        codec_name=codec,
        pix_fmt="yuv420p10le",
        color_range="tv",
        color_space="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
    )


def _comparison() -> FrameComparison:
    return FrameComparison(
        source_info=_info("source.mkv"),
        distorted_info=replace(
            _info("distorted.mkv", "vvc"), width=3840, height=1608
        ),
        source_crop=CropBox(3840, 1608, 0, 276),
        distorted_crop=CropBox(3840, 1608, 0, 0),
        fps=24000 / 1001,
        frame_count=2877,
    )


def test_playback_size_preserves_cropped_content_shape():
    assert playback_dimensions(_comparison()) == (3840, 1608)


def test_ffmpeg_playback_commands(monkeypatch, tmp_path, subtests):
    with subtests.test("the soundtrack is the source's, from the frame shown"):
        # As native playback's: it was the test video's, which encodes often
        # do not carry.
        monkeypatch.setattr(video_playback, "ffplay_path", lambda: tmp_path / "ffplay.exe")
        command = build_audio_command(_comparison(), 48)
        assert command[command.index("-i") + 1] == str(Path("source.mkv").resolve())
        assert float(command[command.index("-ss") + 1]) == 48 / (24000 / 1001)
    # The GPU's own decoder, never FFmpeg's Vulkan one: on an RTX 5090 that
    # turned the lower part of some HEVC frames green. NVDEC's pictures are
    # handed to Vulkan on the GPU ("interop"), others through system memory.
    for decoder, mode in (("cuda", "interop"), ("cuda", "transfer"), ("d3d11va", "transfer")):
        with subtests.test("the GPU's own decoder", decoder=decoder, mode=mode):
            command = build_video_series_command(
                [_comparison()], 0, PreviewColorSettings(), [HwAccelPlan(decoder, decoder)],
                realtime=True, processing=mode)
            graph = command[command.index("-filter_complex") + 1]
            assert [command[i + 1] for i, a in enumerate(command) if a == "-hwaccel"] == [decoder] * 2
            if mode == "interop":
                assert "vulkan=preview@decode" in command
                assert [command[i + 1] for i, a in enumerate(command) if a == "-hwaccel_device"] == ["decode"] * 2
                assert "hwdownload,format=p010le" not in graph and graph.count("hwupload,libplacebo") == 2
            else:
                assert graph.count("hwdownload,format=p010le,hwupload,libplacebo") == 2


def test_frame_compare_playback_has_no_qt_multimedia_dependency():
    module = (
        Path(__file__).resolve().parent.parent
        / "vmaf_app" / "ui" / "video_compare_view.py"
    ).read_text(encoding="utf-8")

    assert "QtMultimedia" not in module
