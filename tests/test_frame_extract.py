from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorMode,
    PreviewColorSettings,
    build_frame_command,
    comparison_dimensions,
    frame_filter,
    frame_input_path,
)
from vmaf_app.core.models import (
    ComparisonResult,
    CropBox,
    FrameScores,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
)


def _info(path: str, width: int, height: int) -> VideoInfo:
    return VideoInfo(
        path=Path(path), width=width, height=height, fps=24.0,
        duration=10.0, nb_frames=240, codec_name="h264",
    )


def _result(*, direction=ScaleDirection.SOURCE_TO_DISTORTED) -> FrameComparison:
    """A finished run's geometry, as the extractor now consumes it.

    Extraction takes the preprocessing recipe rather than a whole run, so
    the same code path serves a pair that has never been measured.
    """
    return FrameComparison.from_result(_run_result(direction=direction))


def _run_result(*, direction=ScaleDirection.SOURCE_TO_DISTORTED) -> ComparisonResult:
    source = _info("source.mkv", 3840, 2160)
    distorted = _info("distorted.mkv", 1920, 1080)
    return ComparisonResult(
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


def test_each_side_gets_its_own_crop_before_scaling():
    result = replace(
        _result(),
        source_crop=CropBox(3840, 1608, 0, 276),
        distorted_crop=CropBox(1920, 804, 0, 138),
    )

    assert "crop=3840:1608:0:276,format=yuv420p,scale=1920:804" in frame_filter(result, "source")
    assert frame_filter(result, "distorted").startswith("crop=1920:804:0:138")
    assert comparison_dimensions(result) == (1920, 804)


def test_resolution_test_recreates_the_downscale_upscale_distortion():
    base = _result()
    result = replace(
        base,
        distorted_info=base.source_info,
        resample_target=ResampleTarget(width=1920, label="1080p"),
    )

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


def test_display_aware_mode_tone_maps_tagged_pq_to_the_monitor_white_level():
    result = _result()
    result.distorted_info.color_transfer = "smpte2084"
    result.distorted_info.color_primaries = "bt2020"
    result.distorted_info.color_space = "bt2020nc"
    result.distorted_info.color_range = "tv"
    settings = PreviewColorSettings(
        mode=PreviewColorMode.DISPLAY_AWARE,
        display_hdr_enabled=True,
        display_sdr_white_nits=203.0,
    )

    chain = frame_filter(result, "distorted", settings)

    assert "tin=smpte2084" in chain
    assert "npl=203" in chain
    assert "tonemap=mobius" in chain
    assert "p=bt709:t=bt709:m=bt709" in chain
