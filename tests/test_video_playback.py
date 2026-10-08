from dataclasses import replace
from pathlib import Path

from vmaf_app.core.frame_extract import (
    FrameComparison,
)
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import (
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


def test_frame_compare_playback_has_no_qt_multimedia_dependency():
    module = (
        Path(__file__).resolve().parent.parent
        / "vmaf_app" / "ui" / "video_compare_view.py"
    ).read_text(encoding="utf-8")

    assert "QtMultimedia" not in module
