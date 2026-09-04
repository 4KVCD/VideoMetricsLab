from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core.frame_extract import (
    build_frame_command,
    comparison_dimensions,
    frame_filter,
    frame_input_path,
)
from vmaf_app.core.models import (
    CropBox,
    FrameScores,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafRunResult,
)


def _info(path: str, width: int, height: int) -> VideoInfo:
    return VideoInfo(
        path=Path(path), width=width, height=height, fps=24.0,
        duration=10.0, nb_frames=240, codec_name="h264",
    )


def _result(*, direction=ScaleDirection.SOURCE_TO_DISTORTED) -> VmafRunResult:
    source = _info("source.mkv", 3840, 2160)
    distorted = _info("distorted.mkv", 1920, 1080)
    return VmafRunResult(
        source=source.path,
        distorted=distorted.path,
        frames=FrameScores(
            frame=np.arange(240, dtype=np.int32),
            time=np.arange(240, dtype=np.float64) / 24,
            vmaf=np.full(240, 90, dtype=np.float32),
        ),
        fps=24.0,
        model="m",
        source_crop=None,
        distorted_crop=None,
        source_info=source,
        distorted_info=distorted,
        scale_direction=direction,
        scale_algorithm="lanczos",
        compared_frame_count=240,
    )


def test_source_to_distorted_previews_share_distorted_dimensions():
    result = _result()

    assert comparison_dimensions(result) == (1920, 1080)
    assert "scale=1920:1080:flags=lanczos" in frame_filter(result, "source")
    assert "scale=" not in frame_filter(result, "distorted")


def test_distorted_to_source_previews_share_source_dimensions():
    result = _result(direction=ScaleDirection.DISTORTED_TO_SOURCE)

    assert comparison_dimensions(result) == (3840, 2160)
    assert "scale=" not in frame_filter(result, "source")
    assert "scale=3840:2160:flags=lanczos" in frame_filter(result, "distorted")


def test_each_side_gets_its_own_crop_before_scaling():
    result = _result()
    result.source_crop = CropBox(3840, 1608, 0, 276)
    result.distorted_crop = CropBox(1920, 804, 0, 138)

    assert "crop=3840:1608:0:276,scale=1920:804" in frame_filter(result, "source")
    assert frame_filter(result, "distorted").startswith("crop=1920:804:0:138")
    assert comparison_dimensions(result) == (1920, 804)


def test_resolution_test_recreates_the_downscale_upscale_distortion():
    result = _result()
    result.distorted_info = result.source_info
    result.resample_target = ResampleTarget(width=1920, label="1080p")

    distorted_filter = frame_filter(result, "distorted")

    assert frame_input_path(result, "distorted") == result.source_info.path
    assert "scale=1920:1080:flags=lanczos" in distorted_filter
    assert "scale=3840:2160:flags=lanczos" in distorted_filter
    assert "scale=" not in frame_filter(result, "source")


def test_command_seeks_just_before_the_requested_frame_timestamp():
    result = _result()

    command = build_frame_command(result, "distorted", 24)

    seek = command[command.index("-ss") + 1]
    assert float(seek) == pytest.approx((24 - 0.125) / 24.0)
    assert command[command.index("-i") + 1].endswith("distorted.mkv")
    assert command[-4:] == ["-f", "image2pipe", "-c:v", "png", "pipe:1"][-4:]
    assert any("setsar=1,format=rgb24" in argument for argument in command)
