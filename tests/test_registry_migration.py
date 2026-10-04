"""The ffmpeg folder's move from the registry, where it was kept under the
app's old name, into settings.json."""
from __future__ import annotations

import pytest
from PySide6 import QtCore

from vmaf_app.core.settings import Settings
from vmaf_app.main import adopt_registry_ffmpeg_dir


class FakeRegistry:
    values: dict[str, str] = {}

    def __init__(self, organisation: str, application: str) -> None:
        assert (organisation, application) == ("VmafApp", "VmafCalculator")

    def value(self, key: str):
        return self.values.get(key)

    def remove(self, key: str) -> None:
        self.values.pop(key, None)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(FakeRegistry, "values", {})
    monkeypatch.setattr(QtCore, "QSettings", FakeRegistry)
    return FakeRegistry.values


def test_the_registry_folder_moves_into_the_settings(registry):
    registry["ffmpeg_dir"] = r"D:\tools\ffmpeg\bin"

    assert adopt_registry_ffmpeg_dir() == r"D:\tools\ffmpeg\bin"
    assert Settings.load().ffmpeg_dir == r"D:\tools\ffmpeg\bin"
    assert "ffmpeg_dir" not in registry
    assert adopt_registry_ffmpeg_dir() is None  # once


def test_a_folder_already_in_the_settings_is_kept(registry):
    settings = Settings.load()
    settings.ffmpeg_dir = r"C:\ffmpeg\bin"
    settings.save()
    registry["ffmpeg_dir"] = r"D:\old\bin"

    assert adopt_registry_ffmpeg_dir() is None
    assert Settings.load().ffmpeg_dir == r"C:\ffmpeg\bin"
    assert "ffmpeg_dir" not in registry


def test_nothing_in_the_registry_changes_nothing(registry):
    assert adopt_registry_ffmpeg_dir() is None
    assert Settings.load().ffmpeg_dir == ""
