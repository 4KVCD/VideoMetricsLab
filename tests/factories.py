"""Shared object factories for tests.

Keep ordinary object construction here rather than importing helpers from one
test module into another. Pytest fixtures and global environment isolation stay
in ``conftest.py``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from vmaf_app.core.models import ComparisonResult, FrameScore, VideoInfo

#: The real Python interpreter, for the stand-in processes the tests start
#: (fake FFmpeg, fake metric tools, launchers). A virtual environment's
#: python.exe is itself a launcher that starts the real one as a child; a
#: pause, the CPU throttle or a cancel landing while it creates that child
#: makes Windows refuse the creation ("Access is denied"), which a parallel
#: run's load turns into an occasional failure. Stand-ins use the standard
#: library only.
STDLIB_PYTHON = getattr(sys, "_base_executable", sys.executable)


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


def decode_plan(text: str):
    """A decode plan from its description, as HwAccelPlan.describe gives it:
    "off", or "source cuda, distorted cpu"."""
    from vmaf_app.core.gpu import HwAccelPlan

    if text == "off":
        return HwAccelPlan()
    sides = dict(part.split(" ", 1) for part in text.split(", "))
    return HwAccelPlan(**{side: None if sides.get(side, "cpu") == "cpu" else sides[side]
                          for side in ("source", "distorted")})


def status(text: str):
    """The core.status.Status a runner sends with these words: the decode
    plan they name, if any, and the kind of step they say it is."""
    from vmaf_app.core.gpu import GPU_WAIT_MESSAGE
    from vmaf_app.core.status import GPU_PASS, GPU_VMAF_FAILED, GPU_WAIT, STARTING, Status

    plan, brief = None, text
    if match := re.fullmatch(r"(.*) \(GPU decode: ([^)]*)\)(?:\.\.\.|\u2026)", text):
        brief, plan = match.group(1), decode_plan(match.group(2))
    kind = (STARTING if text.startswith("Running ffmpeg") else GPU_PASS if text.startswith("GPU metric ")
            else GPU_WAIT if text == GPU_WAIT_MESSAGE
            else GPU_VMAF_FAILED if text.startswith("VMAF on the GPU failed") else "")
    return Status(text, plan=plan, kind=kind, brief=brief)

