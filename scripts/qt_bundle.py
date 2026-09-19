"""Which of PySide6's files the Windows build ships.

PyInstaller's Qt hooks bundle by category -- everything a QtGui application
might conceivably load -- so a build that imports three Qt modules got a
software OpenGL renderer, QtMultimedia's FFmpeg, the whole QML stack (behind
one virtual-keyboard plugin), a PDF renderer, the network module with its
TLS backends, four platform plugins and 96 translations: 110 MB, of which
the app reaches 41.

This trims PyInstaller's analysis after the fact, the same way
gstreamer_bundle decides what to ship: an allow-list of the Qt modules the
app imports and the plugins it needs, their DLL dependencies read from the
import tables, and nothing else from the PySide6 folder.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

# Through the package: a bare `gstreamer_bundle` would find the GStreamer
# wheel of the same name on sys.path instead of the sibling module.
from scripts.gstreamer_bundle import dependency_closure, pe_imports

#: The Qt modules the app imports. tests/test_qt_bundle.py checks this
#: against the source, so adding `from PySide6 import QtSvg` somewhere fails
#: a test until it is listed here and the build ships it.
QT_MODULES = ("QtCore", "QtGui", "QtWidgets")

#: Qt plugins, relative to PySide6/plugins/. Qt finds plugins by scanning
#: directories, so a missing one is simply not offered; these are the ones
#: the app would notice missing.
KEEP_PLUGINS: dict[str, str] = {
    "platforms/qwindows.dll": "the Windows platform plugin; a desktop app needs exactly one",
    "styles/qmodernwindowsstyle.dll": "the native Windows look",
    "imageformats/qjpeg.dll": "JPEG",
    "imageformats/qico.dll": "ICO",
    "imageformats/qgif.dll": "GIF",
    "imageformats/qsvg.dll": "SVG images",
    "iconengines/qsvgicon.dll": "SVG icons",
}

#: Everything under these top-level bundle folders is subject to the policy.
QT_ROOTS = ("PySide6",)
#: shiboken6 is 2 MB and every part of it is needed; left alone.


def _under(dest: str, root: str) -> bool:
    return PurePosixPath(dest.replace("\\", "/")).parts[:1] == (root,)


def _relative(dest: str) -> PurePosixPath:
    return PurePosixPath(*PurePosixPath(dest.replace("\\", "/")).parts[1:])


def prune(binaries: list, datas: list, imports=pe_imports) -> tuple[list, list, dict]:
    """Filters PyInstaller TOC lists ((dest, source, type) tuples).

    Keeps, under PySide6/: the extension modules for QT_MODULES, the
    KEEP_PLUGINS, and every DLL any of those reach through import tables.
    Drops the rest of the folder's binaries and all translations. Entries
    outside PySide6/ pass through untouched.
    """
    def is_qt(entry) -> bool:
        return any(_under(entry[0], root) for root in QT_ROOTS)

    qt_binaries = [entry for entry in binaries if is_qt(entry)]
    others = [entry for entry in binaries if not is_qt(entry)]
    qt_datas = [entry for entry in datas if is_qt(entry)]

    seeds: set[Path] = set()
    for dest, src, _type in qt_binaries:
        rel = _relative(dest)
        if len(rel.parts) == 1 and rel.suffix.lower() == ".pyd" and rel.stem in QT_MODULES:
            seeds.add(Path(src))
        if rel.parts[:1] == ("plugins",) and rel.relative_to("plugins").as_posix() in KEEP_PLUGINS:
            seeds.add(Path(src))
    missing_modules = [m for m in QT_MODULES if not any(_relative(d).stem == m and _relative(d).suffix.lower() == ".pyd" for d, _s, _t in qt_binaries)]
    if missing_modules:
        raise RuntimeError(f"PyInstaller's analysis did not include {missing_modules}; is the app still importing them?")
    missing_plugins = [p for p in KEEP_PLUGINS if not any(_relative(d).as_posix() == f"plugins/{p}" for d, _s, _t in qt_binaries)]
    if missing_plugins:
        raise RuntimeError(f"Qt plugins named in KEEP_PLUGINS are not in this PySide6: {missing_plugins}")

    kept_sources = dependency_closure(seeds, [Path(src) for _d, src, _t in qt_binaries], imports)
    kept_binaries = others + [entry for entry in qt_binaries if Path(entry[1]) in kept_sources]
    dropped_binaries = [entry for entry in qt_binaries if Path(entry[1]) not in kept_sources]

    # The app installs no translator, so Qt's own dialogs are English
    # regardless; 96 .qm files would only ever be read by code that is not there.
    dropped_datas = [entry for entry in qt_datas if _relative(entry[0]).parts[:1] == ("translations",)]
    kept_datas = [entry for entry in datas if entry not in dropped_datas]

    def size(entries) -> int:
        return sum(Path(e[1]).stat().st_size for e in entries if Path(e[1]).is_file())

    kept_qt = [entry for entry in kept_binaries if is_qt(entry)] + [entry for entry in kept_datas if is_qt(entry)]
    report = {
        "qt_bytes_before": size(qt_binaries) + size(qt_datas),
        "qt_bytes_after": size(kept_qt),
        "kept": sorted(entry[0] for entry in kept_qt),
        "dropped": sorted(entry[0] for entry in dropped_binaries + dropped_datas),
    }
    return kept_binaries, kept_datas, report
