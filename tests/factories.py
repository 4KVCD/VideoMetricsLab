"""Shared object factories for tests.

Keep ordinary object construction here rather than importing helpers from one
test module into another. Pytest fixtures and global environment isolation stay
in ``conftest.py``.
"""
from __future__ import annotations

from pathlib import Path

from vmaf_app.core.models import ComparisonResult, FrameScore, VideoInfo


def fake_video_info(name: str | Path) -> VideoInfo:
    return VideoInfo(
        path=Path(name),
        width=1920,
        height=1080,
        fps=30.0,
        duration=5.0,
        nb_frames=150,
        codec_name="h264",
    )


def fake_run_result(
    name: str | Path,
    *,
    source: str | Path = "source.mp4",
    vmaf: float = 90.0,
    n_frames: int = 10,
) -> ComparisonResult:
    info = fake_video_info(name)
    frames = [
        FrameScore(frame=i, time=i / info.fps, vmaf=vmaf)
        for i in range(n_frames)
    ]
    return ComparisonResult(
        source=Path(source),
        distorted=Path(name),
        frames=frames,
        fps=info.fps,
        model="version=vmaf_v0.6.1",
        source_crop=None,
        distorted_crop=None,
        source_info=info,
        distorted_info=info,
    )


def fake_completed_run(name: str | Path):
    # Local import keeps core-only tests from importing the UI just by importing
    # this factory module.
    from vmaf_app.ui.main_window import CompletedRun

    return CompletedRun(fake_run_result(name), str(name))
