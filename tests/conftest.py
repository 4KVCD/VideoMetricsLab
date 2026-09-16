"""Shared test fixtures.

The important one is `isolate_user_state`: without it the suite reads and
writes the *user's real* settings file and results cache. That is not just
untidy -- a test that ticks "compute PSNR by default" persisted it, which
then leaked into every later test in the run AND into the installed app.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Set before Qt is imported anywhere: the startup ffmpeg check opens a modal
# dialog unless it sees this, and a modal in a test blocks the run forever
# rather than failing it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vmaf_app.core import crop_detect, result_cache
from vmaf_app.core.settings import SETTINGS_VERSION, Settings


def _real_user_cache_dir() -> Path:
    """Where the installed app keeps results. Nothing in the suite may touch
    it, so it is resolved once here to be recognised and refused."""
    return result_cache.default_cache_dir()


@pytest.fixture(autouse=True)
def isolate_user_state(tmp_path, monkeypatch):
    """Points settings and the results cache at a per-test temp folder."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(Settings, "path", staticmethod(lambda: settings_file))

    cache = tmp_path / "results_cache"
    cache.mkdir()

    # Written as a real settings file rather than only set through the
    # override, because MainWindow's constructor re-applies whatever its
    # loaded Settings say. With cache_dir blank that is None -- "use the
    # platform folder" -- so simply building a window silently un-isolated
    # the suite and pointed it back at the user's own cache.
    #
    # parallel_jobs is pinned because its default depends on the core count
    # of the machine running the suite (see default_parallel_jobs), and the
    # version is current so the migration that would re-derive it stays out
    # of the baseline. Tests of either write their own file.
    settings_file.write_text(
        json.dumps({
            "cache_dir": str(cache),
            "parallel_jobs": 1,
            "settings_version": SETTINGS_VERSION,
        }),
        encoding="utf-8",
    )

    real_override = result_cache.set_cache_dir_override
    real_cache = _real_user_cache_dir()

    def guarded_override(directory):
        # A backstop for the same leak arriving by another route (a test
        # that saves a fresh Settings, say). Failing loudly is the point:
        # the previous symptom was a test quietly reading and writing real
        # user data, which stays invisible until it corrupts something.
        if directory is None:
            real_override(cache)
            return
        if Path(directory) == real_cache:
            raise AssertionError(
                f"a test pointed the results cache at the user's real folder "
                f"({real_cache}); it must stay inside tmp_path"
            )
        real_override(Path(directory))

    monkeypatch.setattr(result_cache, "set_cache_dir_override", guarded_override)
    result_cache.set_cache_dir_override(cache)
    # Detected black bars are remembered per file for the life of the
    # process; a test's answer must not leak into the next one's.
    crop_detect.clear_cache()
    yield
    real_override(None)
    crop_detect.clear_cache()
