"""PyInstaller runtime hook: make the bundled GStreamer findable.

Outside a bundle this happens through `gstreamer_bundle.pth`, which the
interpreter runs at startup and which calls setup_python_environment() to
put GStreamer's bin/plugin/typelib directories into the environment and the
DLL search path. A frozen app does not process .pth files, so without this
`import gi` fails and playback silently falls back to FFmpeg.

It runs before the entry script, which is what it has to do: the environment
variables are read by GStreamer when it initialises, not when it is imported.
"""
import os
import sys


def _setup() -> None:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is None:
        return  # running from source; the .pth has already done this
    # The GStreamer packages are shipped as data directories at the top of
    # the bundle. _MEIPASS is already on sys.path, so they import as normal
    # packages and compute their own paths from __file__ -- pointing at the
    # copies inside this bundle rather than at a machine that has GStreamer
    # installed some other way.
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    try:
        import gstreamer_libs
    except ImportError:
        return  # built without GStreamer; the app falls back to FFmpeg
    try:
        gstreamer_libs.setup_python_environment()
    except Exception as error:  # pragma: no cover - defensive
        # Never fatal. Metric calculation does not need GStreamer at all, and
        # comparison playback has an FFmpeg path; taking the whole app down
        # because a video sink is unavailable would be far worse.
        print(f"GStreamer setup failed, falling back to FFmpeg: {error}",
              file=sys.stderr)
        return
    # A registry cached next to a previous install (or from the build
    # machine) makes GStreamer look for plugins at paths that do not exist
    # here. Keep it beside the user's own data instead.
    os.environ.setdefault(
        "GST_REGISTRY_1_0",
        os.path.join(
            os.path.expanduser("~"), ".vmaf-calculator", "gstreamer-registry.bin"
        ),
    )


_setup()
