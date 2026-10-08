"""vmaf_app.core.geometry: a comparison's sizes, shapes and pair checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from vmaf_app.core.geometry import (
    compared_dimensions,
    content_size,
    display_aspect_ratio,
    pair_problem,
)
from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo


def _video(width=1920, height=1080, fps=24.0, duration=10.0, **kwargs) -> VideoInfo:
    return VideoInfo(Path("v.mkv"), width, height, fps, duration, round(fps * duration), "hevc", **kwargs)


def test_the_picture_compared_is_the_crops_or_the_videos():
    assert content_size(_video(), None) == (1920, 1080)
    assert content_size(_video(), CropBox(w=1920, h=800, x=0, y=140)) == (1920, 800)


@pytest.mark.parametrize(("direction", "size"), [
    (ScaleDirection.SOURCE_TO_DISTORTED, (1280, 720)), (ScaleDirection.DISTORTED_TO_SOURCE, (1920, 1080))])
def test_one_side_is_scaled_to_the_other(direction, size):
    assert compared_dimensions(_video(), _video(1280, 720), direction) == size


def test_the_shape_counts_non_square_pixels():
    assert display_aspect_ratio(_video(1440, 1080, sar="4:3")) == pytest.approx(16 / 9)
    assert display_aspect_ratio(_video(sar="N/A")) == pytest.approx(16 / 9)


@pytest.mark.parametrize(("source", "test", "limit", "problem"), [
    (_video(), _video(), 0.0, None),
    (_video(), _video(fps=25.0), 0.0, "Frame rates do not match (24.000 vs 25.000 fps)."),
    # A whole rate against its NTSC one: a frame apart every 1000 frames.
    (_video(), _video(fps=24000 / 1001), 0.0, "Frame rates do not match (24.000 vs 23.976 fps)."),
    (_video(fps=30000 / 1001), _video(fps=30.0), 0.0, "Frame rates do not match (29.970 vs 30.000 fps)."),
    (_video(fps=60.0), _video(fps=60000 / 1001), 0.0, "Frame rates do not match (60.000 vs 59.940 fps)."),
    (_video(fps=12.0), _video(fps=12000 / 1001), 0.0, "Frame rates do not match (12.000 vs 11.988 fps)."),
    # One rate as two containers write it.
    (_video(fps=24000 / 1001), _video(fps=2997 / 125), 0.0, None),
    (_video(fps=24000 / 1001), _video(fps=23.9752), 0.0, None),
    (_video(), _video(duration=12.0), 0.0, "Durations do not match (10.000 vs 12.000 seconds)."),
    (_video(), _video(duration=12.0), 8.0, None),
    (_video(), _video(), 12.0, "The duration limit extends beyond the end of one of the videos."),
    (_video(nominal_fps=30.0), _video(), 0.0, "Variable-frame-rate video is not supported safely yet."),
])
def test_two_videos_are_compared_only_on_timelines_that_agree(source, test, limit, problem):
    found = pair_problem(source, test, limit)
    assert (found is None) if problem is None else found.startswith(problem)


@pytest.mark.parametrize("nominal", [1_200_000.0, 120.0])  # no timing in the stream, or its own rate
@pytest.mark.parametrize("side", ["source", "test"])
def test_a_raw_stream_is_refused_for_what_it_is_not_as_variable_frame_rate(nominal, side):
    """A raw HEVC stream, which ffprobe gives FFmpeg's 25 fps as an average,
    was refused as variable-frame-rate video, which sent people converting
    frame rates; it needs a container that gives it its rate."""
    raw = VideoInfo(Path("HoneyBee.hevc"), 3840, 2160, 25.0, 0.0, 0, "hevc", nominal_fps=nominal,
                    average_fps=25.0, format_name="hevc")
    pair = (raw, _video()) if side == "source" else (_video(), raw)
    assert pair_problem(*pair, 0.0) == (
        "HoneyBee.hevc is a raw HEVC stream with no timestamps, so its frame rate is unknown. Put it in a "
        "container with its frame rate first, e.g. mkvmerge -o video.mkv --default-duration 0:120fps <file> "
        "(MKVToolNix), with the video's own frame rate.")
    raw.codec_name = "h264"
    assert " raw H.264 stream " in pair_problem(*pair, 0.0)


def test_a_raw_stream_that_was_compared_still_is():
    """A raw stream whose rates agree (MPEG-2's are in its headers) and match
    the other video's was compared before; only the message of a refused one
    changed."""
    raw = VideoInfo(Path("a.m2v"), 1920, 1080, 24.0, 0.0, 0, "mpeg2video", nominal_fps=24.0, average_fps=24.0,
                    format_name="mpegvideo")
    assert pair_problem(raw, _video(), 0.0) is None


def test_the_cpu_tools_refuse_durations_that_do_not_match_as_the_window_does():
    """Their own check had no rule for it."""
    from vmaf_app.core.comparison_recipe import ComparisonRecipe
    from vmaf_app.core.models import CropMode
    from vmaf_app.core.perceptual_cpu import PerceptualRunError, _validate_pair

    recipe = ComparisonRecipe(CropMode.NONE, "bicubic", ScaleDirection.SOURCE_TO_DISTORTED, 0.0, None)
    with pytest.raises(PerceptualRunError, match="Durations do not match"):
        _validate_pair(_video(), _video(duration=12.0), recipe)
