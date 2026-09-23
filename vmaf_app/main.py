from __future__ import annotations

import contextlib
import sys

from PySide6.QtWidgets import QApplication

from vmaf_app import APP_NAME, __version__
from vmaf_app.core.app_paths import user_data_dir
from vmaf_app.ui.main_window import MainWindow


def self_test() -> str:
    """A report on external tools and bundled runtime components.

    Exists for the packaged build: it has no console, so when it fails to
    start or silently falls back there is otherwise nothing to look at.
    Checks the pieces that are found at runtime rather than at build time.
    """
    lines = [f"{APP_NAME} {__version__} self-test (Python {sys.version.split()[0]})"]
    frozen = getattr(sys, "frozen", False)
    lines.append(f"  packaged build: {'yes' if frozen else 'no, running from source'}")

    from vmaf_app.core.ffmpeg_locate import check_tools, format_version

    status = check_tools()
    if status.ok:
        lines.append(f"  OK    ffmpeg {format_version(status.ffmpeg.version)} and ffprobe")
    else:
        for problem in status.problems:
            lines.append(f"  FAIL  {problem}")

    try:
        from vmaf_app.core.gstreamer_playback import GPU_DECODERS, REQUIRED_ELEMENTS, _load_gstreamer

        gst, _ = _load_gstreamer()
        version = ".".join(str(part) for part in gst.version()[:3])
        lines.append(f"  OK    GStreamer {version}")
        # Element by element rather than plugin by plugin: the packaged
        # build ships a pruned plugin set (scripts/gstreamer_bundle.py), and
        # a plugin can be present while the decoder someone needs is not.
        missing = [name for name in REQUIRED_ELEMENTS if gst.ElementFactory.find(name) is None]
        if missing:
            lines.append(f"  FAIL  GStreamer elements missing: {', '.join(missing)}")
        else:
            lines.append(f"  OK    all {len(REQUIRED_ELEMENTS)} GStreamer elements the app uses")
        gpu = [name for name in GPU_DECODERS if gst.ElementFactory.find(name) is not None]
        lines.append(
            f"  OK    GPU decoders on this machine: {', '.join(gpu)}" if gpu else
            "  WARN  no D3D11 GPU decoders registered; video decodes in software"
        )
    except Exception as error:
        lines.append(f"  WARN  GStreamer unavailable, playback falls back to FFmpeg: {error}")

    # Which Qt platform plugin, style and image formats loaded. The packaged
    # build ships a pruned PySide6 (scripts/qt_bundle.py); Qt would silently
    # fall back to a plain style or refuse an image format if one were missing.
    app = QApplication.instance()
    if app is not None:
        from PySide6.QtCore import qVersion
        from PySide6.QtGui import QImageReader

        formats = sorted(bytes(f).decode() for f in QImageReader.supportedImageFormats())
        lines.append(
            f"  OK    Qt {qVersion()} on '{app.platformName()}', style '{app.style().objectName()}', "
            f"images: {', '.join(formats)}"
        )

    from vmaf_app.core import d3d11_tonemap

    if d3d11_tonemap.available():
        lines.append(f"  OK    GPU HDR tone-map shader ({d3d11_tonemap.library_path().name})")
    else:
        lines.append("  WARN  GPU HDR tone-map shader absent; FFmpeg tone mapping is used")

    from vmaf_app.core.perceptual_cpu import find_metric_executable

    for metric in ("ssimulacra2", "butteraugli"):
        tool = find_metric_executable(metric)
        lines.append(f"  OK    {metric} ({tool})" if tool else f"  WARN  {metric} tool absent")

    from vmaf_app.core.perceptual_vship import detect_vship_device

    vship_device, vship_reason = detect_vship_device()
    if vship_device is not None:
        lines.append(
            f"  OK    Vship {vship_device.version} GPU metrics "
            f"({vship_device.vendor.upper()}: {vship_device.name})"
        )
    else:
        lines.append(f"  WARN  Vship GPU metrics unavailable; CPU fallback is enabled ({vship_reason})")

    return "\n".join(lines)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    if "--self-test" in sys.argv:
        report = self_test()
        # Both, because neither alone reaches every caller: a packaged build
        # has no console to print to, and an automated check has no one to
        # dismiss a dialog.
        with contextlib.suppress(OSError, ValueError):
            print(report)  # a windowed build has no usable stdout
        destination = user_data_dir() / "self-test.txt"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report, encoding="utf-8")
        except OSError:
            destination = None
        if "--quiet" not in sys.argv:
            from PySide6.QtWidgets import QMessageBox

            box = QMessageBox()
            box.setWindowTitle("Self-test")
            box.setText(report + (f"\n\nSaved to {destination}" if destination else ""))
            box.exec()
        return 0 if "FAIL" not in report else 1

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
