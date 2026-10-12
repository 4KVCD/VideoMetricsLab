"""The app's entry point: starts the window, or with --self-test, checks the
app's tools and libraries and reports what it found.

The packaged build has a second program, VideoMetricsLab-cli.exe, that runs
the command line (vmaf_app.cli) instead of the window.
"""
from __future__ import annotations

import contextlib
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vmaf_app import APP_NAME, __version__, i18n
from vmaf_app.core import app_log
from vmaf_app.core.app_paths import user_data_dir

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

# Qt and the window are imported in main(), not here. Every process the app
# starts for Vship and libvmaf (vmaf_app.core.isolated) runs this module again
# before multiprocessing hands it its work, and importing them here would add
# about 0.45 s to the start of each such process.


class SelfTestReport:
    """The self-test's lines, and whether any check failed. Each check
    records its failure as it is made; reading "FAIL" back from the text
    would wrongly fail a run whose paths or messages contain that word."""

    def __init__(self, title: str) -> None:
        self.lines = [title]
        self.failed = False

    def ok(self, text: str) -> None:
        self.lines.append(f"  OK    {text}")

    def warn(self, text: str) -> None:
        self.lines.append(f"  WARN  {text}")

    def fail(self, text: str) -> None:
        self.lines.append(f"  FAIL  {text}")
        self.failed = True

    def note(self, text: str) -> None:
        self.lines.append(f"  {text}")

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def self_test() -> SelfTestReport:
    """Checks the external tools and the components found at run time (not
    at build time), and returns a report on them.

    It exists for the packaged build, which has no console: when the build
    fails to start, or quietly falls back to something slower, the report
    is the only thing to look at.
    """
    report = SelfTestReport(f"{APP_NAME} {__version__} self-test (Python {sys.version.split()[0]})")
    frozen = getattr(sys, "frozen", False)
    report.note(f"packaged build: {'yes' if frozen else 'no, running from source'}")

    from vmaf_app.core.ffmpeg_locate import check_tools, format_version

    status = check_tools()
    if status.ok:
        report.ok(f"ffmpeg {format_version(status.ffmpeg.version)} and ffprobe")
    else:
        for problem in status.problems:
            report.fail(problem)

    try:
        from vmaf_app.core.gstreamer_playback import GPU_DECODERS, REQUIRED_ELEMENTS, _load_gstreamer

        gst, _ = _load_gstreamer()
        version = ".".join(str(part) for part in gst.version()[:3])
        report.ok(f"GStreamer {version}")
        # Checked element by element rather than plugin by plugin: the
        # packaged build ships a pruned set of plugins
        # (scripts/gstreamer_bundle.py), and a plugin can be present while
        # the decoder someone needs is not.
        missing = [name for name in REQUIRED_ELEMENTS if gst.ElementFactory.find(name) is None]
        if missing:
            report.fail(f"GStreamer elements missing: {', '.join(missing)}")
        else:
            report.ok(f"all {len(REQUIRED_ELEMENTS)} GStreamer elements the app uses")
        gpu = [name for name in GPU_DECODERS if gst.ElementFactory.find(name) is not None]
        if gpu:
            report.ok(f"GPU decoders on this machine: {', '.join(gpu)}")
        else:
            report.warn("no D3D11 GPU decoders registered; video decodes in software")
    except Exception as error:
        report.warn(f"GStreamer unavailable, playback falls back to FFmpeg: {error}")

    # Which Qt platform plugin, style and image formats loaded. The packaged
    # build ships a pruned PySide6 (scripts/qt_bundle.py), and if a piece
    # were missing Qt would quietly fall back to a plain style or refuse an
    # image format.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        from PySide6.QtCore import qVersion
        from PySide6.QtGui import QImageReader

        formats = sorted(bytes(f).decode() for f in QImageReader.supportedImageFormats())
        report.ok(f"Qt {qVersion()} on '{app.platformName()}', style '{app.style().objectName()}', "
                  f"images: {', '.join(formats)}")

    from vmaf_app.core import d3d11_tonemap

    if d3d11_tonemap.presents():
        report.ok(f"Native playback's presenter and HDR shader ({d3d11_tonemap.library_path().name})")
    else:
        report.warn("Native playback's presenter absent or out of date; videos play through FFmpeg")

    from vmaf_app.core import vulkan_presenter

    if vulkan_presenter.chosen():  # asked for by VIDEOMETRICSLAB_PRESENTER=vulkan
        problem = vulkan_presenter.check()
        if problem is None:
            report.ok(f"Video Compare draws with Vulkan ({vulkan_presenter.library_path().name})")
        else:
            report.warn(f"Video Compare draws with Direct3D 11: the Vulkan presenter is unavailable ({problem})")

    from vmaf_app.core import d3d11_film_grain

    problem = d3d11_film_grain.check()
    if problem is None:
        report.ok(f"Film grain for VVC videos in Video Compare ({d3d11_film_grain.library_path().name})")
    else:
        report.warn(f"Film grain for VVC videos unavailable ({problem}); they play without it")

    from vmaf_app.core import gpu_frames

    for backend, maker in (("nvidia", "NVIDIA"), ("intel", "Intel"), ("amd", "AMD")):
        library = gpu_frames.LIBRARIES[backend].name
        if gpu_frames.available(backend):
            report.ok(f"{maker} frame decoder for the GPU metrics ({library})")
        else:
            report.warn(f"{maker} frame decoder absent ({library}); FFmpeg decodes the GPU metrics' videos")
    # The software decoder uses the app's FFmpeg libraries
    # (vmaf_app/tools/ffmpeg) and GStreamer's dav1d, so the codecs it reports
    # show that the packaged build found them.
    library = gpu_frames.LIBRARIES[gpu_frames.SOFTWARE].name
    codecs, why = gpu_frames.software_codecs()
    if codecs:
        report.ok(f"software frame decoder ({library}): {', '.join(codecs)}")
    else:
        report.warn(f"software frame decoder absent ({library}: {why}); "
                    "FFmpeg's pipes carry the videos it decodes in software")

    from vmaf_app.core.perceptual_cpu import find_metric_executable

    for metric in ("ssimulacra2", "butteraugli"):
        tool = find_metric_executable(metric)
        if tool:
            report.ok(f"{metric} ({tool})")
        else:
            report.warn(f"{metric} tool absent")

    # Vship runs in processes of its own (vmaf_app.core.isolated). In the
    # packaged build each is the executable started again, which works only
    # because the script calls multiprocessing.freeze_support() first.
    import os

    from vmaf_app.core.isolated import run_isolated

    try:
        child = run_isolated(os.getpid, what="a child process")
        report.ok(f"GPU libraries run in a process of their own (started process {child})")
    except Exception as error:
        report.fail(f"no process for the GPU libraries could be started: {error}")

    from vmaf_app.core.perceptual_vship import backend_label, detect_vship_device, set_vship_backend
    from vmaf_app.core.settings import Settings

    set_vship_backend(Settings.load().gpu_backend)

    vship_device, vship_reason = detect_vship_device()
    if vship_device is not None:
        report.ok(f"Vship {vship_device.version} GPU metrics "
                  f"({backend_label(vship_device.backend)}: {vship_device.name})")
    else:
        report.warn(f"Vship GPU metrics unavailable; CPU fallback is enabled ({vship_reason})")

    from vmaf_app.core import vmaf_gpu

    vmaf_gpu.set_gpu_backend(Settings.load().gpu_backend)
    available, text = vmaf_gpu.gpu_vmaf_available()
    if available:
        report.ok(f"VMAF on the GPU ({text})")
    else:
        report.warn(f"VMAF on the GPU unavailable; FFmpeg's libvmaf is used ({text})")
    from vmaf_app.core import vmaf_v1_gpu

    available, text = vmaf_v1_gpu.available()
    if available:
        report.ok(f"VMAF v1 with the GPU ({text})")
    else:
        report.warn(f"VMAF v1 with the GPU unavailable; FFmpeg's libvmaf is used ({text})")

    return report


_QT_LOG_LEVELS = {"QtDebugMsg": logging.DEBUG, "QtInfoMsg": logging.INFO, "QtWarningMsg": logging.WARNING,
                  "QtCriticalMsg": logging.ERROR, "QtFatalMsg": logging.CRITICAL}


def _log_qt_message(mode, _context, message: str) -> None:
    """Qt's own messages, written to the app's log at the matching level."""
    logging.getLogger("vmaf_app.qt").log(_QT_LOG_LEVELS.get(getattr(mode, "name", ""), logging.WARNING), message)


#: Where older versions kept the ffmpeg folder, before settings.json held it:
#: the registry, under the app's old name.
_REGISTRY_ORG = "VmafApp"
_REGISTRY_APP = "VmafCalculator"


def adopt_registry_ffmpeg_dir() -> str | None:
    """Moves an ffmpeg folder still kept in the registry into settings.json,
    unless settings.json names one already. The registry's copy is removed
    either way, so this happens only once. Returns the folder moved, or
    None."""
    from PySide6.QtCore import QSettings

    from vmaf_app.core.settings import Settings

    registry = QSettings(_REGISTRY_ORG, _REGISTRY_APP)
    value = str(registry.value("ffmpeg_dir") or "").strip()
    if not value:
        return None
    settings = Settings.load()
    adopted = None
    if not settings.ffmpeg_dir.strip():
        settings.ffmpeg_dir = value
        if settings.save():
            return None  # not saved: the registry keeps it for next time
        adopted = value
    registry.remove("ffmpeg_dir")
    return adopted


def start_session_log() -> None:
    """Starts this session's log, headed with what the app runs on."""
    if app_log.start_logging() is None:
        return
    log = logging.getLogger("vmaf_app.main")
    log.info("==== %s%s", APP_NAME, app_log.SESSION_START)
    for line in app_log.environment_lines():
        log.info("%s", line)
    from PySide6.QtCore import qInstallMessageHandler, qVersion

    log.info("Qt %s", qVersion())
    qInstallMessageHandler(_log_qt_message)


def log_tools() -> None:
    """Writes FFmpeg's version and the GPUs found to this session's log.
    It is called once the window is up, when the tools' check started at
    launch (start_tool_check) has finished; waiting for that check while
    starting the log would delay the start by 80 ms."""
    from vmaf_app.core.ffmpeg_locate import check_tools, format_version
    from vmaf_app.core.gpu import detected_gpu_vendors

    log = logging.getLogger("vmaf_app.main")
    tools = check_tools()
    if tools.ok:
        log.info("FFmpeg %s: %s", format_version(tools.ffmpeg.version), tools.ffmpeg.path)
    else:
        for problem in tools.problems:
            log.warning("FFmpeg: %s", problem)
    log.info("GPUs: %s", ", ".join(vendor.name for vendor in detected_gpu_vendors()) or "none detected")


def apply_language(app: QApplication, chosen: str) -> str:
    """Shows the app in the `chosen` language, or Windows' display language
    when none is chosen, with English where there is no translation.
    Returns the language applied.

    Qt's own dialogs and buttons follow, using Qt's translations where it
    has them, and a right-to-left language mirrors the window."""
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
    """Starts the window (or runs the self-test) and returns the exit code."""
    # This comes before anything looks for ffmpeg, the self-test included.
    # The tools' check then runs while the window's modules load.
    adopted = adopt_registry_ffmpeg_dir()
    from vmaf_app.core.ffmpeg_locate import start_tool_check

    start_tool_check()
    from PySide6.QtWidgets import QApplication

    from vmaf_app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    if "--self-test" in sys.argv:
        report = self_test()
        # The report is both printed and saved to a file, because neither
        # alone reaches every caller: a packaged build has no console to
        # print to, and an automated check has no one to close a dialog.
        with contextlib.suppress(OSError, ValueError):
            print(report.text)  # a windowed build has no usable stdout
        destination = user_data_dir() / "self-test.txt"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report.text, encoding="utf-8")
        except OSError:
            destination = None
        if "--quiet" not in sys.argv:
            from PySide6.QtWidgets import QMessageBox

            box = QMessageBox()
            box.setWindowTitle("Self-test")
            box.setText(report.text + (f"\n\nSaved to {destination}" if destination else ""))
            box.exec()
        return 1 if report.failed else 0

    start_session_log()
    if adopted:
        logging.getLogger("vmaf_app.main").info("ffmpeg folder moved from the registry to the settings: %s", adopted)
    from vmaf_app.core.settings import Settings

    settings = Settings.load()
    apply_language(app, settings.language)
    # The theme is applied before the window is built, because the window
    # reads it to colour itself.
    from vmaf_app.ui import theme

    theme.apply_theme(settings.theme)
    logging.getLogger("vmaf_app.main").info("Theme: %s", settings.theme or "as Windows")
    from vmaf_app.core.perceptual_vship import set_vship_backend, start_vship_probe

    set_vship_backend(Settings.load().gpu_backend)
    start_vship_probe()  # done by the time the first video is added
    from vmaf_app.core import vmaf_gpu

    vmaf_gpu.set_gpu_backend(Settings.load().gpu_backend)
    vmaf_gpu.start_gpu_vmaf_probe()
    from vmaf_app.core import vmaf_v1_gpu

    vmaf_v1_gpu.start_probe()
    window = MainWindow()
    window.show()
    log_tools()
    # Prepare Video Compare's first opening in the background now: load
    # GStreamer and its D3D11 output, and list FFmpeg's hardware decoders.
    from vmaf_app.core import gpu, gstreamer_playback

    gstreamer_playback.start_loading()
    gpu.start_hwaccel_probe()
    window.check_for_updates()  # once, now, and at no other time
    return app.exec()


if __name__ == "__main__":
    # The packaged app starts itself again for the processes Vship and
    # libvmaf run in (vmaf_app.core.isolated). In such a process, this call
    # runs the process's work and exits, before any window is made.
    multiprocessing.freeze_support()
    # The packaged build's second program, VideoMetricsLab-cli.exe, is this
    # program with a console (see VideoMetricsLab.spec): it runs the command
    # line and makes no window.
    if Path(sys.executable).stem.casefold().endswith("-cli"):
        from vmaf_app import cli

        sys.exit(cli.main())
    sys.exit(main())
