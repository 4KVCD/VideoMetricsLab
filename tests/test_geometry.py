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
    (_video(), _video(duration=12.0), 0.0, "Durations do not match (10.000 vs 12.000 seconds)."),
    (_video(), _video(duration=12.0), 8.0, None),
    (_video(), _video(), 12.0, "The duration limit extends beyond the end of one of the videos."),
    (_video(nominal_fps=30.0), _video(), 0.0, "Variable-frame-rate video is not supported safely yet."),
])
def test_two_videos_are_compared_only_on_timelines_that_agree(source, test, limit, problem):
    found = pair_problem(source, test, limit)
    assert (found is None) if problem is None else found.startswith(problem)


def test_the_cpu_tools_refuse_durations_that_do_not_match_as_the_window_does():
    """Their own check had no rule for it."""
    from vmaf_app.core.comparison_recipe import ComparisonRecipe
    from vmaf_app.core.models import CropMode
    from vmaf_app.core.perceptual_cpu import PerceptualRunError, _validate_pair

    recipe = ComparisonRecipe(CropMode.NONE, "bicubic", ScaleDirection.SOURCE_TO_DISTORTED, 0.0, None)
    with pytest.raises(PerceptualRunError, match="Durations do not match"):
        _validate_pair(_video(), _video(duration=12.0), recipe)
