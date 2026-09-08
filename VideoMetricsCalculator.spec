# PyInstaller spec for a self-contained Windows build.
#
# Everything the app needs at runtime is inside the output folder except
# FFmpeg, which is deliberately not bundled -- see docs/BUILD.md.
#
# Build it with scripts/build_release.ps1 rather than calling pyinstaller
# directly: the tone-map DLL has to exist before this file is read.
import sysconfig
from pathlib import Path

PROJECT = Path(SPECPATH)
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])

# --- GStreamer -------------------------------------------------------------
# The wheels are ordinary directories whose own paths are computed from
# __file__ (see gstreamer_libs.environment), so copying each tree intact and
# putting it back under the same package name is enough for them to find
# their own plugins, typelibs and scanner.
#
# They are shipped as DATA and excluded from analysis on purpose. Frozen as
# modules their __file__ would point inside the archive, and every path they
# derive from it -- GST_PLUGIN_PATH, GI_TYPELIB_PATH, the plugin scanner
# executable -- would point at files that are not there.
GSTREAMER_PACKAGES = [
    "gstreamer_libs",              # core libraries, typelibs, plugin scanner
    "gstreamer_plugins",           # base/good plugin set
    "gstreamer_plugins_libs",      # their shared dependencies
    "gstreamer_plugins_restricted",
    "gstreamer_plugins_gpl",
    "gstreamer_plugins_gpl_restricted",
    "gstreamer_python",            # the `gi` bindings live in here
    "gstreamer_ext_runtime",       # Windows runtime shims
    # gstreamer_cli (gst-launch et al) and gstreamer_gtk (GTK video sinks)
    # are omitted: the app drives the pipeline through `gi` and presents with
    # d3d11. Both are optional imports in gstreamer_libs.gstreamer_env, so
    # their absence is handled rather than fatal. Together they are ~35 MB.
]

datas = []
for package in GSTREAMER_PACKAGES:
    source = SITE_PACKAGES / package
    if not source.is_dir():
        raise SystemExit(
            f"{package} is not installed. Run:\n"
            f"    pip install -r requirements.txt"
        )
    datas.append((str(source), package))

# --- the GPU HDR->SDR shader ----------------------------------------------
# vmaf_app.core.d3d11_tonemap resolves this as <package>/native/<name>, so it
# has to land in the same place relative to vmaf_app.
TONEMAP_DLL = PROJECT / "vmaf_app" / "native" / "d3d11_tonemap.dll"
if TONEMAP_DLL.is_file():
    datas.append((str(TONEMAP_DLL), "vmaf_app/native"))
else:
    print("WARNING: d3d11_tonemap.dll is missing; the build will fall back "
          "to FFmpeg tone mapping. Run scripts/build_d3d11_tonemap.ps1 first.")

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

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoMetricsCalculator",
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
    name="VideoMetricsCalculator",
)
