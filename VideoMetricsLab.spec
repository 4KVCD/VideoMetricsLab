# PyInstaller spec for a self-contained Windows build.
#
# Everything the app needs at runtime is inside the output folder except
# FFmpeg, which is deliberately not bundled -- see docs/BUILD.md.
#
# Build it with scripts/build_release.ps1 rather than calling pyinstaller
# directly: the tone-map DLL has to exist before this file is read.
import sys
import sysconfig
from pathlib import Path

PROJECT = Path(SPECPATH)
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])

# --- GStreamer -------------------------------------------------------------
# The wheels are ordinary directories whose own paths are computed from
# __file__ (see gstreamer_libs.environment), so every shipped file keeps its
# path relative to site-packages and each wheel still finds its own bin/,
# plugin and typelib directories.
#
# Not every file is shipped. The wheels are a complete media framework (302
# MB); the app plays two videos through D3D11 and one soundtrack, which
# needs about a sixth of that. scripts/gstreamer_bundle.py
# holds the allow-list of plugins and computes their dependencies from the
# DLL import tables; scripts/verify_gstreamer_bundle.py then proves the
# packaged runtime can still do everything the app asks, as part of the
# build. Change what is shipped there, not here.
#
# The packages are shipped as DATA and excluded from analysis on purpose.
# Frozen as modules their __file__ would point inside the archive, and every
# path they derive from it -- GST_PLUGIN_PATH, GI_TYPELIB_PATH, the plugin
# scanner executable -- would point at files that are not there.
sys.path.insert(0, str(PROJECT))
from scripts.gstreamer_bundle import PACKAGES as GSTREAMER_PACKAGES  # noqa: E402
from scripts.gstreamer_bundle import collect as collect_gstreamer  # noqa: E402

datas, gstreamer_report = collect_gstreamer(SITE_PACKAGES)
print(
    "GStreamer: shipping %d of the wheels' files, %.0f of %.0f MB"
    % (len(datas), gstreamer_report["bundled_bytes"] / 1e6, gstreamer_report["original_bytes"] / 1e6)
)

# --- the GPU HDR->SDR shader ----------------------------------------------
# vmaf_app.core.d3d11_tonemap resolves this as <package>/native/<name>, so it
# has to land in the same place relative to vmaf_app.
TONEMAP_DLL = PROJECT / "vmaf_app" / "native" / "d3d11_tonemap.dll"
if TONEMAP_DLL.is_file():
    datas.append((str(TONEMAP_DLL), "vmaf_app/native"))
else:
    print("WARNING: d3d11_tonemap.dll is missing; the build will fall back "
          "to FFmpeg tone mapping. Run scripts/build_d3d11_tonemap.ps1 first.")

# Netflix's VMAF v1 models are data files, not part of the external FFmpeg
# installation.  Keep them beside the package so the app can pass a portable
# path= model to any compatible libvmaf build.
VMAF_MODELS = PROJECT / "vmaf_app" / "models"
datas.append((str(VMAF_MODELS), "vmaf_app/models"))

a = Analysis(
    [str(PROJECT / "vmaf_app" / "main.py")],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=datas,
    # `gi` is shipped as data, so nothing traced its imports and the stdlib
    # modules it reaches for were left out of the bundle. The first symptom
    # was GStreamer failing to load with "No module named 'optparse'" --
    # which, because playback falls back to FFmpeg, would otherwise have been
    # invisible until someone tried to play a video. Collected by scanning
    # the gi tree for stdlib imports; see docs/BUILD.md.
    hiddenimports=[
        "asyncio",
        "collections",
        "contextlib",
        "ctypes",
        "functools",
        "importlib",
        "inspect",
        "optparse",
        "pkgutil",
        "platform",
        "random",
        "re",
        "selectors",
        "signal",
        "socket",
        "threading",
        "types",
        "typing",
        "warnings",
        "weakref",
    ],
    hookspath=[],
    runtime_hooks=[str(PROJECT / "scripts" / "pyi_rth_gstreamer.py")],
    excludes=[
        # Shipped as data above; freezing them too would shadow those copies
        # with ones whose __file__ points into the archive.
        *GSTREAMER_PACKAGES,
        "gstreamer_bundle",
        "gi",
        # Never imported by this app, and each drags in a lot.
        "tkinter",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtPositioning",
        "PySide6.QtWebSockets",
        "PySide6.QtWebChannel",
        "PySide6.QtDesigner",
        "PySide6.QtHelp",
        "PySide6.QtTest",
        "PySide6.QtSql",
    ],
    noarchive=False,
    optimize=0,
)

# --- Qt --------------------------------------------------------------------
# PyInstaller's Qt hooks bundle by category -- a software OpenGL renderer,
# QtMultimedia's FFmpeg, the QML stack, PDF, networking and 96 translations
# for an app that imports QtCore, QtGui and QtWidgets. scripts/qt_bundle.py
# keeps those three modules, the plugins the app needs and their import
# closure, and drops the rest of the PySide6 folder after the analysis.
from scripts.qt_bundle import prune as prune_qt  # noqa: E402

a.binaries, a.datas, qt_report = prune_qt(a.binaries, a.datas)
print(
    "Qt: keeping %d of PySide6's files, %.0f of %.0f MB"
    % (len(qt_report["kept"]), qt_report["qt_bytes_after"] / 1e6, qt_report["qt_bytes_before"] / 1e6)
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoMetricsLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A GUI app: no console window. Startup failures are still reachable by
    # running it from a terminal, which is what docs/BUILD.md says to do.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VideoMetricsLab",
)
