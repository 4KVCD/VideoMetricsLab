"""GStreamer's environment, set up twice: in any process started from a
Python process that set it up already (repair_gstreamer_environment)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vmaf_app.core.gstreamer_playback import repair_gstreamer_environment


def test_the_repaired_registry_and_scanner_are_real_files():
    """In this process as it is -- a test worker started from pytest's own
    process inherits GStreamer's setup and runs it again."""
    environ = dict(os.environ)
    if "GST_REGISTRY_1_0" not in environ:
        pytest.skip("GStreamer's Python wheels are not installed")
    repair_gstreamer_environment(environ)
    assert os.pathsep not in environ["GST_REGISTRY_1_0"]
    scanner = Path(environ["GST_PLUGIN_SCANNER_1_0"])
    assert scanner.with_name(scanner.name + ".exe").is_file() or scanner.is_file()
