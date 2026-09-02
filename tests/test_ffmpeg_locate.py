import pytest

from vmaf_app.core import ffmpeg_locate
from vmaf_app.core.ffmpeg_locate import (
    MINIMUM_FFMPEG_VERSION,
    ToolsStatus,
    ToolStatus,
    format_version,
    parse_version,
)


@pytest.mark.parametrize("line, expected", [
    ("ffmpeg version 9.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025", (9, 0, 1)),
    ("ffmpeg version n7.1 Copyright (c) 2000-2024 the FFmpeg developers", (7, 1)),
    ("ffprobe version 9.0.1-full_build-www.gyan.dev", (9, 0, 1)),
    ("ffmpeg version 6.1.1-3ubuntu5 Copyright", (6, 1, 1)),
    ("ffmpeg version 10 Copyright", (10,)),
])
def test_parse_version_handles_the_common_build_flavours(line, expected):
    assert parse_version(line) == expected


def test_parse_version_returns_none_for_git_master_builds():
    # Git builds report "N-113411-g1234567" -- no usable version number, so
    # this must say "don't know" rather than guess a number.
    assert parse_version("ffmpeg version N-113411-g1234567 Copyright") is None
    assert parse_version("something else entirely") is None


def test_format_version_reads_back_nicely():
    assert format_version((9, 0, 1)) == "9.0.1"
    assert format_version(None) == "unknown"


def _status(ffmpeg_ok=True, ffprobe_ok=True, version=(9, 0, 1)) -> ToolsStatus:
    return ToolsStatus(
        ffmpeg=ToolStatus("ffmpeg", "ffmpeg.exe", ffmpeg_ok, version, "" if ffmpeg_ok else "not found"),
        ffprobe=ToolStatus("ffprobe", "ffprobe.exe", ffprobe_ok, version, "" if ffprobe_ok else "not found"),
    )


def test_everything_present_and_new_enough_is_ok():
    assert _status().ok
    assert _status().problems == []


def test_missing_ffprobe_is_a_problem_even_when_ffmpeg_works():
    # ffprobe was never checked before -- every video is probed with it, so
    # a missing ffprobe breaks the app just as thoroughly as a missing ffmpeg.
    status = _status(ffprobe_ok=False)
    assert not status.ok
    assert any("ffprobe" in p for p in status.problems)


def test_missing_ffmpeg_is_a_problem():
    status = _status(ffmpeg_ok=False)
    assert not status.ok
    assert any("ffmpeg" in p for p in status.problems)


def test_ffmpeg_older_than_the_minimum_is_rejected():
    status = _status(version=(6, 1, 1))
    assert not status.ok
    assert any("too old" in p for p in status.problems)


def test_ffmpeg_at_exactly_the_minimum_is_accepted():
    assert _status(version=MINIMUM_FFMPEG_VERSION).ok


def test_unknown_version_is_not_treated_as_too_old():
    # A git master build is almost certainly newer than the minimum; refusing
    # to run on it just because the version string is unparseable would be
    # worse than letting it through.
    assert _status(version=None).ok


def test_check_tool_reports_a_binary_that_cannot_be_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_locate, "find_binary", lambda name: str(tmp_path / "nope.exe"))
    status = ffmpeg_locate.check_tool("ffmpeg")
    assert status.runnable is False
    assert status.error


def test_check_tool_parses_a_real_version_from_the_binary(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "ffmpeg version 9.0.1-full_build-www.gyan.dev\nbuilt with gcc"
        stderr = ""

    monkeypatch.setattr(ffmpeg_locate, "find_binary", lambda name: "ffmpeg")
    monkeypatch.setattr(ffmpeg_locate.subprocess, "run", lambda *a, **k: FakeProc())

    status = ffmpeg_locate.check_tool("ffmpeg")
    assert status.runnable is True
    assert status.version == (9, 0, 1)
