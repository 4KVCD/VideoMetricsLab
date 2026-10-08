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


def test_two_videos_are_compared_only_on_timelines_that_agree(subtests):
    def check(source, test, limit, problem):
        found = pair_problem(source, test, limit)
        assert (found is None) if problem is None else found.startswith(problem)

    for source, test, limit, problem in [
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
    ]:
        with subtests.test(source=source, test=test, limit=limit, problem=problem):
            check(source, test, limit, problem)
