"""The packaged app's GStreamer runtime hook (scripts/pyi_rth_gstreamer.py),
run against a fake bundle."""
from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "pyi_rth_gstreamer.py"


def test_the_registry_is_kept_with_the_users_data_not_in_the_shared_temp_folder(monkeypatch, tmp_path):
    """The wheels' setup sets the registry itself (to %TEMP%/gstreamer-1.0),
    so the hook's setdefault never took effect."""
    fake = types.ModuleType("gstreamer_libs")
    fake.setup_python_environment = lambda: os.environ.update(
        {"GST_REGISTRY_1_0": str(tmp_path / "gstreamer-1.0" / "wheel-registry.bin")})
    monkeypatch.setitem(sys.modules, "gstreamer_libs", fake)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("GST_REGISTRY_1_0", "")

    runpy.run_path(str(HOOK))

    assert os.environ["GST_REGISTRY_1_0"] == str(Path.home() / ".videometricslab" / "gstreamer-registry.bin")
