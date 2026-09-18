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
    KEEP_TOOLS,
    PACKAGES,
    collect,
    dependency_closure,
    pe_imports,
)
from vmaf_app.core import gstreamer_playback
from vmaf_app.core.gstreamer_playback import _PIPELINE_ELEMENTS, REQUIRED_ELEMENTS

MIB = 1024 ** 2


# ------------------------------------------------------------ the dependency walk

def test_closure_follows_imports_transitively_and_ignores_unshipped_names():
    plugin, lib, deeper, unrelated = (Path(n) for n in ("gstx.dll", "a.dll", "b.dll", "z.dll"))
    graph = {plugin: {"a.dll", "kernel32.dll"}, lib: {"b.dll"}, deeper: set(), unrelated: set()}

    kept = dependency_closure({plugin}, list(graph), graph.__getitem__)

    assert kept == {plugin, lib, deeper}  # kernel32 is the system's, z.dll is nobody's


def test_closure_keeps_every_copy_of_a_name_that_exists_in_two_packages():
    """Which copy Windows loads depends on the search order at run time.
    Keeping one and guessing is how a build works on the build machine only."""
    plugin = Path("plugins/gstdav1d.dll")
    copies = [Path("gstreamer_libs/bin/dav1d.dll"), Path("gstreamer_plugins_libs/bin/dav1d.dll")]
    graph = {plugin: {"dav1d.dll"}, copies[0]: set(), copies[1]: set()}

    assert dependency_closure({plugin}, list(graph), graph.__getitem__) == {plugin, *copies}


def test_closure_survives_import_cycles():
    a, b = Path("a.dll"), Path("b.dll")
    graph = {a: {"b.dll"}, b: {"a.dll"}}

    assert dependency_closure({a}, list(graph), graph.__getitem__) == {a, b}


# ------------------------------------------------------- against the real wheels

SITE = Path(sysconfig.get_paths()["purelib"])
wheels_installed = all((SITE / package).is_dir() for package in PACKAGES)
needs_wheels = pytest.mark.skipif(not wheels_installed, reason="GStreamer wheels are not installed")


@pytest.fixture(scope="module")
def bundle():
    pytest.importorskip("pefile")
    if not wheels_installed:
        pytest.skip("GStreamer wheels are not installed")
    datas, report = collect(SITE)
    return datas, report, {Path(source) for source, _ in datas}


@needs_wheels
def test_every_allow_listed_plugin_exists_and_is_shipped(bundle):
    _datas, report, _kept = bundle
    assert set(report["kept_plugins"]) >= set(KEEP_PLUGINS)


@needs_wheels
def test_no_plugin_outside_the_allow_list_is_shipped(bundle):
    _datas, report, _kept = bundle
    assert set(report["kept_plugins"]) == set(KEEP_PLUGINS)


@needs_wheels
def test_what_imports_cannot_reveal_is_shipped_anyway(bundle):
    """The scanner is spawned, not imported; GLib spawns it through helpers;
    the tone mapper loads gstd3d11 by name through ctypes; gi needs its
    extension modules and the typelibs they describe."""
    _datas, _report, kept = bundle
    names = {p.name.lower() for p in kept}
    for rel in KEEP_TOOLS:
        assert SITE / rel in kept, rel
    assert "gstd3d11-1.0-0.dll" in names
    assert any(p.name.startswith("_gi.") and p.suffix == ".pyd" for p in kept)
    assert any(p.name.startswith("_gi_gst.") and p.suffix == ".pyd" for p in kept)
    assert any(p.name == "Gst-1.0.typelib" for p in kept)
    assert any(p.name == "GstD3D11-1.0.typelib" for p in kept)
    for package in PACKAGES:
        assert SITE / package / "__init__.py" in kept, f"{package} must remain importable"


@needs_wheels
def test_the_dead_weight_is_gone(bundle):
    _datas, _report, kept = bundle
    names = {p.name.lower() for p in kept}
    for unwanted in (
        "libstdc++-6.dll",        # 25 MB that nothing in the wheels imports
        "x265.dll", "x264-164.dll", "svtav1enc.dll",  # encoders
        "gstrswebrtc.dll", "gstaws.dll", "gstburn.dll",  # streaming and cloud
        "rsvg-2-2.dll", "cairo-2.dll", "harfbuzz.dll", "pango-1.0-0.dll",  # text and vector rendering
        "libcrypto-3-x64.dll", "libssl-3-x64.dll",  # TLS
        "vvdec.dll",              # VVC comes from libav's own decoder
        "gstpython.dll",          # Python-implemented elements; the app has none
    ):
        assert unwanted not in names, unwanted
    assert not any(p.name.startswith("_gi_cairo") for p in kept)
    assert not any("__pycache__" in p.parts for p in kept)
    assert not any(part in {"share", "etc"} for p in kept for part in p.relative_to(SITE).parts)


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


@needs_wheels
def test_the_bundle_stays_small(bundle):
    """The whole point. 288 MiB of wheels became 50; this fails long before
    a careless addition takes it back."""
    _datas, report, _kept = bundle
    assert report["bundled_bytes"] < 70 * MIB, f"{report['bundled_bytes'] / MIB:.1f} MiB"
    assert report["original_bytes"] > 4 * report["bundled_bytes"]


@needs_wheels
def test_datas_keep_each_file_at_its_wheel_relative_path(bundle):
    datas, _report, _kept = bundle
    for source, destination in datas:
        relative = Path(source).relative_to(SITE)
        assert destination == relative.parent.as_posix()
        assert not destination.startswith(("/", "\\")) and ".." not in destination
    assert len({(d, Path(s).name) for s, d in datas}) == len(datas), "two files would land on one path"


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
    for name in ("appsrc", "playbin3"):  # locked_presentation.py, by parse_launch and make
        assert name in REQUIRED_ELEMENTS
