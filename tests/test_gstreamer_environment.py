"""GStreamer's environment, set up twice: in any process started from a
Python process that set it up already (repair_gstreamer_environment)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vmaf_app.core.gstreamer_playback import repair_gstreamer_environment

REGISTRY = "C:/venv/Temp/gstreamer-1.0/registry-win-amd64.bin"
SCANNER = "C:/venv/gstreamer_libs/libexec/gstreamer-1.0/gst-plugin-scanner"
PLUGINS = ["C:/venv/gstreamer_cli/lib/gstreamer-1.0", "C:/venv/gstreamer_plugins/lib/gstreamer-1.0"]


def _twice(*parts: str) -> str:
    return os.pathsep.join([*parts, *parts])


def test_a_file_named_twice_is_named_once():
    environ = {"GST_REGISTRY_1_0": _twice(REGISTRY), "GST_PLUGIN_SCANNER_1_0": _twice(SCANNER)}

    assert repair_gstreamer_environment(environ) == ["GST_REGISTRY_1_0", "GST_PLUGIN_SCANNER_1_0"]
    assert environ == {"GST_REGISTRY_1_0": REGISTRY, "GST_PLUGIN_SCANNER_1_0": SCANNER}


def test_a_path_list_keeps_each_path_once_in_its_first_place():
    user = "C:/Users/me/bin"
    environ = {"GST_PLUGIN_PATH_1_0": _twice(*PLUGINS),
               "PATH": os.pathsep.join([PLUGINS[1], user, PLUGINS[1].upper(), PLUGINS[0]])}

    assert sorted(repair_gstreamer_environment(environ)) == ["GST_PLUGIN_PATH_1_0", "PATH"]
    assert environ["GST_PLUGIN_PATH_1_0"] == os.pathsep.join(PLUGINS)
    assert environ["PATH"] == os.pathsep.join([PLUGINS[1], user, PLUGINS[0]])


def test_an_environment_set_up_once_is_left_alone():
    environ = {"GST_REGISTRY_1_0": REGISTRY, "GST_PLUGIN_PATH_1_0": os.pathsep.join(PLUGINS), "OTHER": "x;x"}
    before = dict(environ)

    assert repair_gstreamer_environment(environ) == []
    assert environ == before


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
