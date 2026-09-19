"""What the Windows build ships of PySide6.

The policy is checked on a made-up TOC first (the rules), then on the
PySide6 actually installed (the closure holds, the dead weight is gone), and
its module list is checked against the app's own imports so a new
`from PySide6 import QtSomething` cannot ship without the module.
"""
from __future__ import annotations

import re
import sysconfig
from pathlib import Path

import pytest

from scripts.qt_bundle import KEEP_PLUGINS, QT_MODULES, prune

MIB = 1024 ** 2
APP = Path(__file__).resolve().parent.parent / "vmaf_app"
SITE = Path(sysconfig.get_paths()["purelib"])


def test_qt_modules_are_exactly_what_the_app_imports():
    imported = set()
    for path in APP.rglob("*.py"):
        imported |= set(re.findall(r"PySide6\.(Qt[A-Za-z]+)", path.read_text(encoding="utf-8")))
    assert imported == set(QT_MODULES), sorted(imported ^ set(QT_MODULES))


# -------------------------------------------------------------------- the rules

def _toc(*dests):
    return [(d, str(Path("/site") / d), "BINARY") for d in dests]


def test_prune_keeps_the_modules_the_plugins_and_what_they_import_and_nothing_else():
    binaries = _toc(
        "PySide6/QtCore.pyd", "PySide6/QtGui.pyd", "PySide6/QtWidgets.pyd", "PySide6/QtNetwork.pyd",
        "PySide6/Qt6Core.dll", "PySide6/Qt6Gui.dll", "PySide6/Qt6Widgets.dll", "PySide6/Qt6Network.dll",
        "PySide6/Qt6Quick.dll", "PySide6/opengl32sw.dll", "PySide6/avcodec-61.dll", "PySide6/pyside6.abi3.dll",
        "PySide6/plugins/platforms/qwindows.dll", "PySide6/plugins/platforms/qdirect2d.dll",
        "PySide6/plugins/styles/qmodernwindowsstyle.dll", "PySide6/plugins/imageformats/qjpeg.dll",
        "PySide6/plugins/imageformats/qico.dll", "PySide6/plugins/imageformats/qgif.dll",
        "PySide6/plugins/imageformats/qsvg.dll", "PySide6/plugins/iconengines/qsvgicon.dll",
        "PySide6/plugins/imageformats/qpdf.dll", "PySide6/plugins/tls/qschannelbackend.dll",
        "shiboken6/shiboken6.abi3.dll", "numpy/_core/_multiarray_umath.pyd",
    )
    datas = _toc("PySide6/translations/qt_de.qm", "PySide6/translations/qtbase_fr.qm", "vmaf_app/native/d3d11_tonemap.dll")
    graph = {
        "qtcore.pyd": {"qt6core.dll", "pyside6.abi3.dll"}, "qtgui.pyd": {"qt6gui.dll", "pyside6.abi3.dll"},
        "qtwidgets.pyd": {"qt6widgets.dll"}, "qtnetwork.pyd": {"qt6network.dll"},
        "qt6gui.dll": {"qt6core.dll"}, "qt6widgets.dll": {"qt6gui.dll", "qt6core.dll"},
        "qt6quick.dll": {"qt6gui.dll"}, "qwindows.dll": {"qt6gui.dll"}, "qpdf.dll": {"qt6pdf.dll"},
    }

    kept_b, kept_d, report = prune(binaries, datas, imports=lambda p: graph.get(p.name.lower(), set()))

    kept = {d for d, _s, _t in kept_b}
    assert {"PySide6/QtCore.pyd", "PySide6/QtGui.pyd", "PySide6/QtWidgets.pyd", "PySide6/Qt6Core.dll",
            "PySide6/Qt6Gui.dll", "PySide6/Qt6Widgets.dll", "PySide6/pyside6.abi3.dll",
            "PySide6/plugins/platforms/qwindows.dll", "PySide6/plugins/styles/qmodernwindowsstyle.dll",
            "PySide6/plugins/imageformats/qjpeg.dll", "PySide6/plugins/iconengines/qsvgicon.dll"} <= kept
    for gone in ("PySide6/QtNetwork.pyd", "PySide6/Qt6Network.dll", "PySide6/Qt6Quick.dll", "PySide6/opengl32sw.dll",
                 "PySide6/avcodec-61.dll", "PySide6/plugins/platforms/qdirect2d.dll",
                 "PySide6/plugins/imageformats/qpdf.dll", "PySide6/plugins/tls/qschannelbackend.dll"):
        assert gone not in kept, gone
    # Untouched: everything outside PySide6/, and PySide6 data that is not a translation.
    assert "shiboken6/shiboken6.abi3.dll" in kept and "numpy/_core/_multiarray_umath.pyd" in kept
    assert [d for d, _s, _t in kept_d] == ["vmaf_app/native/d3d11_tonemap.dll"]
    assert "PySide6/translations/qt_de.qm" in report["dropped"]


def test_prune_refuses_an_analysis_that_lost_a_module_the_app_imports():
    binaries = _toc("PySide6/QtCore.pyd", "PySide6/QtGui.pyd", "PySide6/Qt6Core.dll",
                    *(f"PySide6/plugins/{p}" for p in KEEP_PLUGINS))
    with pytest.raises(RuntimeError, match="QtWidgets"):
        prune(binaries, [], imports=lambda p: set())


def test_prune_refuses_when_a_named_plugin_does_not_exist():
    binaries = _toc(*(f"PySide6/{m}.pyd" for m in QT_MODULES), "PySide6/plugins/platforms/qwindows.dll")
    with pytest.raises(RuntimeError, match="KEEP_PLUGINS"):
        prune(binaries, [], imports=lambda p: set())


# ------------------------------------------------------ the installed PySide6

def _installed_toc():
    binaries, datas = [], []
    for path in (SITE / "PySide6").rglob("*"):
        if not path.is_file():
            continue
        dest = "PySide6/" + path.relative_to(SITE / "PySide6").as_posix()
        if path.suffix.lower() in {".dll", ".pyd"}:
            binaries.append((dest, str(path), "BINARY"))
        elif path.suffix.lower() == ".qm":
            datas.append((dest, str(path), "DATA"))
    return binaries, datas


@pytest.mark.skipif(not (SITE / "PySide6" / "Qt6Core.dll").is_file(), reason="PySide6 is not installed here")
def test_against_the_installed_pyside6():
    pytest.importorskip("pefile")
    from scripts.gstreamer_bundle import pe_imports

    binaries, datas = _installed_toc()
    kept_b, _kept_d, report = prune(binaries, datas)
    names = {Path(d).name.lower() for d, _s, _t in kept_b}

    for needed in ("qt6core.dll", "qt6gui.dll", "qt6widgets.dll", "qtcore.pyd", "qtgui.pyd", "qtwidgets.pyd",
                   "pyside6.abi3.dll", "qwindows.dll", "qmodernwindowsstyle.dll"):
        assert needed in names, needed
    for gone in ("opengl32sw.dll", "avcodec-61.dll", "qt6quick.dll", "qt6qml.dll", "qt6pdf.dll", "qt6network.dll",
                 "qtnetwork.pyd", "qt6opengl.dll", "qdirect2d.dll", "qoffscreen.dll", "qtvirtualkeyboardplugin.dll"):
        assert gone not in names, gone
    assert not any("translations/" in d for d in report["kept"])

    # The closure property on the real files: nothing kept imports a PySide6 DLL that was dropped.
    all_names = {Path(d).name.lower() for d, _s, _t in binaries}
    unmet = {Path(s).name: sorted(n for n in pe_imports(Path(s)) if n in all_names and n not in names)
             for d, s, _t in kept_b if d.startswith("PySide6/")}
    assert not {k: v for k, v in unmet.items() if v}

    assert report["qt_bytes_after"] < 60 * MIB, f"{report['qt_bytes_after'] / MIB:.1f} MiB"
