from __future__ import annotations

import contextlib
import logging
import multiprocessing
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from vmaf_app import APP_NAME, __version__, i18n
from vmaf_app.core import app_log
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

    # Vship runs in processes of its own (vmaf_app.core.isolated): in the
    # packaged build that is the executable started again, which only works
    # while main() calls multiprocessing.freeze_support() first.
    import os

    from vmaf_app.core.isolated import run_isolated

    try:
        child = run_isolated(os.getpid, what="a child process")
        lines.append(f"  OK    GPU libraries run in a process of their own (started process {child})")
    except Exception as error:
        lines.append(f"  FAIL  no process for the GPU libraries could be started: {error}")

    from vmaf_app.core.perceptual_vship import backend_label, detect_vship_device, set_vship_backend
    from vmaf_app.core.settings import Settings

    set_vship_backend(Settings.load().gpu_backend)

    vship_device, vship_reason = detect_vship_device()
    if vship_device is not None:
        lines.append(
            f"  OK    Vship {vship_device.version} GPU metrics "
            f"({backend_label(vship_device.backend)}: {vship_device.name})"
        )
    else:
        lines.append(f"  WARN  Vship GPU metrics unavailable; CPU fallback is enabled ({vship_reason})")

    from vmaf_app.core import vmaf_cuda

    available, text = vmaf_cuda.gpu_vmaf_available()
    lines.append(f"  OK    VMAF on the GPU ({text})" if available
                 else f"  WARN  VMAF on the GPU unavailable; FFmpeg's libvmaf is used ({text})")

    return "\n".join(lines)


_QT_LOG_LEVELS = {"QtDebugMsg": logging.DEBUG, "QtInfoMsg": logging.INFO, "QtWarningMsg": logging.WARNING,
                  "QtCriticalMsg": logging.ERROR, "QtFatalMsg": logging.CRITICAL}


def _log_qt_message(mode, _context, message: str) -> None:
    logging.getLogger("vmaf_app.qt").log(_QT_LOG_LEVELS.get(getattr(mode, "name", ""), logging.WARNING), message)


def start_session_log() -> None:
    """The log file for this session, headed with what it runs on."""
    if app_log.start_logging() is None:
        return
    log = logging.getLogger("vmaf_app.main")
    log.info("==== %s%s", APP_NAME, app_log.SESSION_START)
    for line in app_log.environment_lines():
        log.info("%s", line)
    from PySide6.QtCore import qInstallMessageHandler, qVersion

    log.info("Qt %s", qVersion())
    from vmaf_app.core.ffmpeg_locate import check_tools, format_version
    from vmaf_app.core.gpu import detected_gpu_vendors

    tools = check_tools()
    if tools.ok:
        log.info("FFmpeg %s: %s", format_version(tools.ffmpeg.version), tools.ffmpeg.path)
    else:
        for problem in tools.problems:
            log.warning("FFmpeg: %s", problem)
    log.info("GPUs: %s", ", ".join(vendor.name for vendor in detected_gpu_vendors()) or "none detected")
    qInstallMessageHandler(_log_qt_message)


def apply_language(app: QApplication, chosen: str) -> str:
    """Shows the app in `chosen` -- or, when that is empty, Windows' display
    language -- and English where there is no translation. Qt's own dialogs
    and buttons follow (its translations for the language, where it has
    them), and a right-to-left language mirrors the window. The language
    applied."""
    from PySide6.QtCore import QLibraryInfo, Qt, QTranslator

    windows = i18n.windows_language()
    code = i18n.set_language(chosen or windows)
    logging.getLogger("vmaf_app.main").info(
        "Language: %s (%s; Windows: %s)", code, "chosen in Settings" if chosen else "as Windows", windows)
    if code == "en":
        return code
    translator = QTranslator(app)
    for directory in (QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath),
                      str(Path(sys.modules["PySide6"].__file__).parent / "translations")):
        if translator.load(f"qtbase_{code}", directory):
            app.installTranslator(translator)
            break
    if code in i18n.RIGHT_TO_LEFT:
        app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    return code


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

    start_session_log()
    from vmaf_app.core.settings import Settings

    apply_language(app, Settings.load().language)
    from vmaf_app.core.perceptual_vship import set_vship_backend, start_vship_probe

    set_vship_backend(Settings.load().gpu_backend)
    start_vship_probe()  # done by the time the first video is added
    from vmaf_app.core import vmaf_cuda

    vmaf_cuda.start_gpu_vmaf_probe()
    window = MainWindow()
    window.show()
    window.check_for_updates()  # once, now, and at no other time
    return app.exec()


if __name__ == "__main__":
    # The packaged app starts itself again for the processes Vship and
    # libvmaf run in (vmaf_app.core.isolated): there this runs that process's
    # work and exits, before any window.
    multiprocessing.freeze_support()
    sys.exit(main())
