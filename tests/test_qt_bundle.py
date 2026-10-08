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

from scripts.qt_bundle import KEEP_TRANSLATIONS, QT_MODULES, prune

MIB = 1024 ** 2
APP = Path(__file__).resolve().parent.parent / "vmaf_app"
SITE = Path(sysconfig.get_paths()["purelib"])


def test_qt_modules_are_exactly_what_the_app_imports():
    imported = set()
    for path in APP.rglob("*.py"):
        imported |= set(re.findall(r"PySide6\.(Qt[A-Za-z]+)", path.read_text(encoding="utf-8")))
    assert imported == set(QT_MODULES), sorted(imported ^ set(QT_MODULES))


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
@pytest.mark.packaging
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
    assert all(Path(d).name in KEEP_TRANSLATIONS for d in report["kept"] if "translations/" in d)

    # The closure property on the real files: nothing kept imports a PySide6 DLL that was dropped.
    all_names = {Path(d).name.lower() for d, _s, _t in binaries}
    unmet = {Path(s).name: sorted(n for n in pe_imports(Path(s)) if n in all_names and n not in names)
             for d, s, _t in kept_b if d.startswith("PySide6/")}
    assert not {k: v for k, v in unmet.items() if v}

    assert report["qt_bytes_after"] < 60 * MIB, f"{report['qt_bytes_after'] / MIB:.1f} MiB"
