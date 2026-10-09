"""The packaging policy for GStreamer: what the Windows build ships of it.

Two layers. The dependency walk is tested on made-up graphs, because that is
where a subtle bug would hide. The policy is then run against the wheels
actually installed in the development environment, when they are, to prove
the allow-list names real plugins, that everything it reaches is kept, and
that the things imports cannot reveal are kept anyway.
"""
from __future__ import annotations

import re
import sysconfig
from pathlib import Path

import pytest

from scripts.gstreamer_bundle import (
    KEEP_PLUGINS,
    PACKAGES,
    collect,
    pe_imports,
)
from vmaf_app.core import gstreamer_playback
from vmaf_app.core.gstreamer_playback import _PIPELINE_ELEMENTS, REQUIRED_ELEMENTS

MIB = 1024 ** 2


# ------------------------------------------------------- against the real wheels

SITE = Path(sysconfig.get_paths()["purelib"])
wheels_installed = all((SITE / package).is_dir() for package in PACKAGES)
needs_wheels = pytest.mark.skipif(not wheels_installed, reason="GStreamer wheels are not installed")
#: The checks against the installed wheels are the slow ones: run with --packaging.
packaging = pytest.mark.packaging


@pytest.fixture(scope="module")
def bundle():
    pytest.importorskip("pefile")
    if not wheels_installed:
        pytest.skip("GStreamer wheels are not installed")
    datas, report = collect(SITE)
    return datas, report, {Path(source) for source, _ in datas}


@packaging
@needs_wheels
def test_every_allow_listed_plugin_exists_and_is_shipped(bundle):
    _datas, report, _kept = bundle
    assert set(report["kept_plugins"]) >= set(KEEP_PLUGINS)


@packaging
@needs_wheels
def test_every_import_a_shipped_binary_makes_is_shipped(bundle):
    """The closure property itself, checked on the real files: nothing kept
    imports a wheel DLL that was left out."""
    _datas, _report, kept = bundle
    shipped_names = {p.name.lower() for p in kept if p.suffix.lower() in {".dll", ".pyd", ".exe"}}
    all_wheel_names = {
        p.name.lower() for package in PACKAGES for p in (SITE / package).rglob("*")
        if p.suffix.lower() in {".dll", ".pyd", ".exe"}
    }
    unmet = {}
    for path in kept:
        if path.suffix.lower() not in {".dll", ".pyd", ".exe"}:
            continue
        missing = {name for name in pe_imports(path) if name in all_wheel_names and name not in shipped_names}
        if missing:
            unmet[path.name] = sorted(missing)
    assert not unmet


@packaging
@needs_wheels
def test_the_bundle_stays_small(bundle):
    """The whole point. 288 MiB of wheels became 50; this fails long before
    a careless addition takes it back."""
    _datas, report, _kept = bundle
    assert report["bundled_bytes"] < 70 * MIB, f"{report['bundled_bytes'] / MIB:.1f} MiB"
    assert report["original_bytes"] > 4 * report["bundled_bytes"]


# ---------------------------------------------------- the list the checks run on

def test_required_elements_cover_everything_the_pipelines_create_by_name():
    """A new _make("something") in the playback code has to reach the
    self-test and the packaging check, or a bundle can pass both and fail
    the first time someone plays a video."""
    source = Path(gstreamer_playback.__file__).read_text(encoding="utf-8")
    made = set(re.findall(r'_make\(\s*"([a-z0-9_]+)"', source))
    made |= set(re.findall(r'"appsink" if self\._sample_output else "([a-z0-9_]+)"', source))
    assert made, "the scan found nothing; has the pipeline code changed shape?"
    assert made <= set(REQUIRED_ELEMENTS), sorted(made - set(REQUIRED_ELEMENTS))
    assert set(_PIPELINE_ELEMENTS) <= set(REQUIRED_ELEMENTS)
    assert "playbin3" in REQUIRED_ELEMENTS  # locked_presentation.py's soundtrack, by make
