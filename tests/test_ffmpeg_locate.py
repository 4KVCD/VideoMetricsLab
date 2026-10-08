import pytest

from vmaf_app.core import ffmpeg_locate
from vmaf_app.core.ffmpeg_locate import (
    ToolsStatus,
    ToolStatus,
    parse_version,
)


def test_parse_version_handles_the_common_build_flavours(subtests):
    def check(line, expected):
        assert parse_version(line) == expected

    for line, expected in [
        ("ffmpeg version 9.0.1-full_build-www.gyan.dev Copyright (c) 2000-2025", (9, 0, 1)),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024 the FFmpeg developers", (7, 1)),
        ("ffprobe version 9.0.1-full_build-www.gyan.dev", (9, 0, 1)),
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright", (6, 1, 1)),
        ("ffmpeg version 10 Copyright", (10,)),
    ]:
        with subtests.test(line=line, expected=expected):
            check(line, expected)


def _status(ffmpeg_ok=True, ffprobe_ok=True, version=(9, 0, 1)) -> ToolsStatus:
    return ToolsStatus(
        ffmpeg=ToolStatus("ffmpeg", "ffmpeg.exe", ffmpeg_ok, version, "" if ffmpeg_ok else "not found"),
        ffprobe=ToolStatus("ffprobe", "ffprobe.exe", ffprobe_ok, version, "" if ffprobe_ok else "not found"),
    )


def test_missing_ffprobe_is_a_problem_even_when_ffmpeg_works():
    # ffprobe was never checked before -- every video is probed with it, so
    # a missing ffprobe breaks the app just as thoroughly as a missing ffmpeg.
    status = _status(ffprobe_ok=False)
    assert not status.ok
    assert any("ffprobe" in p for p in status.problems)


def test_ffmpeg_older_than_the_minimum_is_rejected():
    status = _status(version=(6, 1, 1))
    assert not status.ok
    assert any("too old" in p for p in status.problems)


def test_check_tool_parses_a_real_version_from_the_binary(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "ffmpeg version 9.0.1-full_build-www.gyan.dev\nbuilt with gcc"
        stderr = ""

    monkeypatch.setattr(ffmpeg_locate, "find_binary", lambda name: "ffmpeg")
    monkeypatch.setattr(ffmpeg_locate.proc_util, "run", lambda *a, **k: FakeProc())

    status = ffmpeg_locate.check_tool("ffmpeg")
    assert status.runnable is True
    assert status.version == (9, 0, 1)


@pytest.fixture
def forget_lookups():
    ffmpeg_locate.ffmpeg_dir_changed()
    yield
    ffmpeg_locate.ffmpeg_dir_changed()


def test_the_folder_chosen_in_settings_comes_first(tmp_path, forget_lookups):
    # Read from settings.json, where every process finds it. It used to be
    # kept in the registry too, and "Locate ffmpeg.exe" wrote only there.
    from vmaf_app.core.settings import Settings

    chosen = tmp_path / ffmpeg_locate.exe_name("ffmpeg")
    chosen.write_bytes(b"")
    settings = Settings.load()
    settings.ffmpeg_dir = str(tmp_path)
    assert settings.save() is None
    ffmpeg_locate.ffmpeg_dir_changed()

    assert ffmpeg_locate.find_binary("ffmpeg") == str(chosen)
