"""Settings read from and written to settings.json (tests/conftest.py gives
each test a file of its own)."""
from __future__ import annotations

import json

from vmaf_app.core import settings as settings_module
from vmaf_app.core.settings import Settings


def _write(data) -> None:
    Settings.path().write_text(json.dumps(data), encoding="utf-8")


def _temporary_files() -> list[str]:
    return [path.name for path in Settings.path().parent.glob(Settings.path().name + ".*")]


def test_a_value_of_the_wrong_type_keeps_its_default_and_the_rest_load():
    """Kept as they were, these failed later, wherever they were used."""
    _write({"parallel_jobs": "2", "hidden_metrics": "psnr", "use_cache": 1, "window_width": True,
            "cvvdp_presets": [{"name": "a"}, "b"], "language": "de", "compare_decoded_videos": 5})

    loaded = Settings.load()

    defaults = Settings()
    assert loaded.parallel_jobs == defaults.parallel_jobs
    assert loaded.hidden_metrics == []
    assert loaded.use_cache is True
    assert loaded.window_width == defaults.window_width
    assert loaded.cvvdp_presets == []
    assert (loaded.language, loaded.compare_decoded_videos) == ("de", 5)


def test_values_of_the_right_type_round_trip():
    saved = Settings(parallel_jobs=1, hidden_metrics=["psnr", "ssim"], use_cache=False,
                     cvvdp_presets=[{"name": "mine", "settings": {}}], language="fr")
    assert saved.save() is None
    assert Settings.load() == saved


def test_a_save_that_fails_keeps_the_saved_settings(monkeypatch):
    """Written in place, a save that failed part-way left a truncated file,
    read at the next start as no settings at all."""
    assert Settings(language="es").save() is None

    def fail(*_args):
        raise OSError("disk full")

    with monkeypatch.context() as patched:  # only this; conftest's own patches stay
        patched.setattr(settings_module.os, "replace", fail)
        assert "disk full" in Settings(language="ja").save()

    assert Settings.load().language == "es"
    assert _temporary_files() == []
