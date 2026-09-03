"""Shared test fixtures.

The important one is `isolate_user_state`: without it the suite reads and
writes the *user's real* settings file and results cache. That is not just
untidy -- a test that ticks "compute PSNR by default" persisted it, which
then leaked into every later test in the run AND into the installed app.
"""
from __future__ import annotations

import os

import pytest

# Set before Qt is imported anywhere: the startup ffmpeg check opens a modal
# dialog unless it sees this, and a modal in a test blocks the run forever
# rather than failing it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vmaf_app.core import result_cache
from vmaf_app.core.settings import Settings


@pytest.fixture(autouse=True)
def isolate_user_state(tmp_path, monkeypatch):
    """Points settings and the results cache at a per-test temp folder."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(Settings, "path", staticmethod(lambda: settings_file))

    cache = tmp_path / "results_cache"
    cache.mkdir()
    result_cache.set_cache_dir_override(cache)
    yield
    result_cache.set_cache_dir_override(None)
