from dataclasses import replace
from pathlib import Path

from vmaf_app.core import gpu
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorMode,
    PreviewColorSettings,
)
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import (
    build_audio_command,
    build_video_pair_command,
    playback_dimensions,
)
from vmaf_app.ui.video_compare_view import _PairDecodeWorker


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
    assert playback_dimensions(_comparison()) == (1920, 804)


def test_pair_command_crops_tone_maps_and_packs_one_frame_clock():
    settings = PreviewColorSettings(mode=PreviewColorMode.HDR_TO_SDR)
    command = build_video_pair_command(
        _comparison(),
        240,
        settings,
        HwAccelPlan(source="cuda"),
        (1280, 536),
        realtime=True,
    )
    graph = command[command.index("-filter_complex") + 1]

    assert "[0:v]hwdownload,format=p010le" in graph
    assert "crop=3840:1608:0:276" in graph
    assert "scale=1280:536" in graph
    assert graph.count("tonemap=mobius") == 2
    assert "[source][distorted]hstack=inputs=2:shortest=1[out]" in graph
    assert command.count("-readrate") == 2
    assert command.count("-hwaccel") == 1


def test_paused_pair_decode_requests_exactly_one_frame():
    command = build_video_pair_command(
        _comparison(), 24, PreviewColorSettings(), HwAccelPlan(),
        (1280, 536), realtime=False,
    )

    assert command[command.index("-frames:v") + 1] == "1"
    assert "-readrate" not in command


def test_audio_uses_ffplay_without_requesting_a_video_decoder(monkeypatch, tmp_path):
    from vmaf_app.core import video_playback

    player = tmp_path / "ffplay.exe"
    player.write_bytes(b"")
    monkeypatch.setattr(video_playback, "ffplay_path", lambda: player)

    command = build_audio_command(_comparison(), 24)

    assert command is not None
    assert command[0] == str(player)
    assert "-nodisp" in command
    assert "-vn" in command
    assert command[command.index("-i") + 1].endswith("distorted.mkv")


def test_vvc_stays_on_ffmpeg_software_while_the_source_uses_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "available_hwaccels", lambda: {"cuda"})
    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [gpu.GpuVendor.NVIDIA])

    assert gpu.plan_hwaccel(gpu.GpuVendor.AUTO, "hevc", "vvc") == HwAccelPlan(
        source="cuda", distorted=None
    )


def test_frame_compare_playback_has_no_qt_multimedia_dependency():
    module = (
        Path(__file__).resolve().parent.parent
        / "vmaf_app" / "ui" / "video_compare_view.py"
    ).read_text(encoding="utf-8")

    assert "QtMultimedia" not in module


def test_pair_worker_coalesces_frames_without_copying_payload_through_qt():
    worker = _PairDecodeWorker(
        7,
        _comparison(),
        0,
        PreviewColorSettings(),
        (2, 2),
        HwAccelPlan(),
        realtime=True,
    )
    notifications: list[int] = []
    worker.frame_available.connect(notifications.append)

    worker._publish_frame(10, b"old", 2, 2)
    worker._publish_frame(11, b"new", 2, 2)

    assert notifications == [7]
    assert worker.take_latest_frame() == (11, b"new", 2, 2)

    worker._publish_frame(12, b"next", 2, 2)
    assert notifications == [7, 7]


def test_pair_worker_uses_a_full_frame_pipe_buffer():
    module = (
        Path(__file__).resolve().parent.parent
        / "vmaf_app" / "ui" / "video_compare_view.py"
    ).read_text(encoding="utf-8")

    assert "bufsize=frame_bytes" in module
    assert "bufsize=0" not in module
