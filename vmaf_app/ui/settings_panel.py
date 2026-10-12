"""The Settings tab: the app-wide settings, saved as soon as they are edited.

The panel applies what is global itself: the theme, the GPU backends, the
saved-results folder and the FFmpeg folder. It tells the window, with
signals, about the changes that concern the window: the defaults for newly
added videos, how many test videos Video Compare decodes, and the FFmpeg
folder, which the window then checks again.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vmaf_app import APP_NAME, i18n
from vmaf_app.core import app_log, perceptual_vship, result_cache, vmaf_gpu
from vmaf_app.core.cvvdp import DEFAULT_PRESET
from vmaf_app.core.cvvdp import presets as cvvdp_presets
from vmaf_app.core.ffmpeg_locate import check_tools, ffmpeg_dir_changed, format_version
from vmaf_app.core.models import GpuVendor
from vmaf_app.core.settings import Settings
from vmaf_app.i18n import ntr, tr, tr_message
from vmaf_app.ui import theme
from vmaf_app.ui.file_worker import FileWriteQueue
from vmaf_app.ui.formatting import preset_label

_log = logging.getLogger(__name__)

#: The "Default metrics" check boxes, in the order they are shown: the
#: metric's key and its label.
_DEFAULT_METRICS = (
    ("vmaf", "VMAF v0.6.1"), ("vmaf_v1", "VMAF v1"), ("vmaf_neg", "VMAF NEG"), ("psnr", "PSNR"),
    ("ssim", "SSIM"), ("xpsnr", "XPSNR"), ("ssimulacra2", "SSIMULACRA2"), ("butteraugli", "Butteraugli"),
    ("cvvdp", "CVVDP"),
)


class SettingsPanel(QWidget):
    """The Settings tab. `settings` is the window's own Settings, which this
    panel edits and saves; `file_writes` is the window's queue of
    background writes, which clearing the saved results goes through."""

    #: The defaults for newly added videos may have changed (any edit).
    defaults_changed = Signal()
    #: How many test videos Video Compare keeps decoding (any edit).
    decoded_videos_changed = Signal(int)
    #: The FFmpeg folder changed and was saved: the window checks the tools.
    ffmpeg_folder_changed = Signal()
    # The saved results (generation, folder, comparisons, bytes), counted off
    # the UI thread (refresh_status).
    _cache_counted = Signal(object)

    def __init__(self, settings: Settings, file_writes: FileWriteQueue, gpu_vendors: list[GpuVendor],
                 parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._file_writes = file_writes
        self._count_generation = 0
        # What clearing the saved results removed, while it is under way.
        self._clear_result: list[int] | None = None
        self._cache_counted.connect(self._show_cache_summary)
        file_writes.became_idle.connect(self._on_file_writes_idle)
        self._build(gpu_vendors)

    def _build(self, gpu_vendors: list[GpuVendor]) -> None:
        """Builds the tab's boxes, one under the other, and the status line."""
        layout = QVBoxLayout(self)
        for box in (self._build_ffmpeg_box(), self._build_storage_box(), self._build_defaults_box(),
                    self._build_gpu_box(gpu_vendors), self._build_compare_box(), self._build_window_box()):
            layout.addWidget(box)
        layout.addStretch(1)
        self.status = QLabel()
        theme.style(self.status, "color: {muted};")
        layout.addWidget(self.status)
        # The tools and saved results are filled in when the tab is shown
        # (refresh_status).

    def _build_ffmpeg_box(self) -> QGroupBox:
        """The folder ffmpeg.exe is in, and whether it was found."""
        settings = self._settings
        tools_box = QGroupBox("ffmpeg")
        tools_form = QFormLayout(tools_box)
        self.ffmpeg_edit = QLineEdit(settings.ffmpeg_dir)
        self.ffmpeg_edit.setPlaceholderText(tr("blank = search PATH"))
        self.ffmpeg_edit.editingFinished.connect(self._on_edited)
        browse_ffmpeg = QPushButton(tr("Browse..."))
        browse_ffmpeg.clicked.connect(self._on_pick_ffmpeg_dir)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self.ffmpeg_edit, stretch=1)
        ffmpeg_row.addWidget(browse_ffmpeg)
        tools_form.addRow(tr("ffmpeg folder:"), ffmpeg_row)
        self.ffmpeg_status = QLabel()
        tools_form.addRow("", self.ffmpeg_status)
        return tools_box

    def _build_storage_box(self) -> QGroupBox:
        """Where saved results and exports go, and the log files."""
        settings = self._settings
        storage_box = QGroupBox(tr("Storage"))
        storage_form = QFormLayout(storage_box)
        self.cache_edit = QLineEdit(settings.cache_dir)
        self.cache_edit.setPlaceholderText(tr("blank = shared user-profile folder"))
        self.cache_edit.editingFinished.connect(self._on_edited)
        browse_cache = QPushButton(tr("Browse..."))
        browse_cache.clicked.connect(self._on_pick_cache_dir)
        open_cache = QPushButton(tr("Open"))
        open_cache.clicked.connect(self._on_open_cache_dir)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.cache_edit, stretch=1)
        cache_row.addWidget(browse_cache)
        cache_row.addWidget(open_cache)
        storage_form.addRow(tr("Saved results:"), cache_row)

        self.cache_summary = QLabel()
        clear_cache = QPushButton(tr("Clear saved results"))
        clear_cache.clicked.connect(self.clear_saved_results)
        summary_row = QHBoxLayout()
        summary_row.addWidget(self.cache_summary, stretch=1)
        summary_row.addWidget(clear_cache)
        storage_form.addRow("", summary_row)

        self.export_edit = QLineEdit(settings.export_dir)
        self.export_edit.setPlaceholderText(tr("blank = ask each time"))
        self.export_edit.editingFinished.connect(self._on_edited)
        browse_export = QPushButton(tr("Browse..."))
        browse_export.clicked.connect(self._on_pick_export_dir)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_edit, stretch=1)
        export_row.addWidget(browse_export)
        storage_form.addRow(tr("Export folder:"), export_row)

        self.use_cache = QCheckBox(tr("Reuse a saved result when a video is added again"))
        self.use_cache.setChecked(settings.use_cache)
        self.use_cache.toggled.connect(self._on_edited)
        storage_form.addRow("", self.use_cache)

        # Where each session's log goes: runs, steps and every failure in
        # full, kept after the window that showed them is closed.
        self.log_label = QLabel(str(app_log.log_dir()))
        self.log_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log_label.setToolTip(
            tr("Each session's setup, every run and its steps, and every failure with its full text. "
            "Send these files along when reporting a problem."))
        open_logs = QPushButton(tr("Open"))
        open_logs.clicked.connect(self._on_open_log_dir)
        self.export_log_btn = QPushButton(tr("Export log..."))
        self.export_log_btn.setToolTip(
            tr("Save the log files as one .zip file, to attach when reporting a problem."))
        self.export_log_btn.clicked.connect(self._on_export_log)
        self.copy_log_btn = QPushButton(tr("Copy log"))
        self.copy_log_btn.setToolTip(
            tr("Copy the latest session that calculated metrics to the clipboard, to paste into a post or chat."))
        self.copy_log_btn.clicked.connect(self._on_copy_log)
        log_row = QHBoxLayout()
        log_row.addWidget(self.log_label, stretch=1)
        log_row.addWidget(open_logs)
        log_row.addWidget(self.copy_log_btn)
        log_row.addWidget(self.export_log_btn)
        storage_form.addRow(tr("Log files:"), log_row)
        return storage_box

    def _build_defaults_box(self) -> QGroupBox:
        """What newly added videos start with."""
        settings = self._settings
        defaults_box = QGroupBox(tr("Defaults for newly added videos"))
        defaults_layout = QVBoxLayout(defaults_box)
        hint = QLabel(
            tr("These are only a starting point -- each video's own settings are "
            "edited in the Videos tab.")
        )
        theme.style(hint, "color: {muted}; font-style: italic;")
        defaults_layout.addWidget(hint)
        self.default_gpu = QCheckBox(tr("Use GPU decoding"))
        self.default_gpu.setChecked(settings.default_gpu_decode)
        self.default_gpu.toggled.connect(self._on_edited)
        defaults_layout.addWidget(self.default_gpu)

        metrics_row = QHBoxLayout()
        metrics_row.addWidget(QLabel(tr("Default metrics:")))
        #: The "Default metrics" check boxes, by metric key.
        self.default_metrics: dict[str, QCheckBox] = {}
        for key, label in _DEFAULT_METRICS:
            box = QCheckBox(label)
            box.setChecked(getattr(settings, f"default_compute_{key}"))
            box.toggled.connect(self._on_edited)
            self.default_metrics[key] = box
            metrics_row.addWidget(box)
        metrics_row.addStretch(1)
        defaults_layout.addLayout(metrics_row)

        cvvdp_row = QHBoxLayout()
        cvvdp_row.addWidget(QLabel(tr("CVVDP display:")))
        self.cvvdp_default = QComboBox()
        self.cvvdp_default.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cvvdp_default.setToolTip(
            tr("The display CVVDP models for newly added videos, whatever their format. "
            "Saving a preset of your own in the Options panel makes it the default; "
            "choose here to change that.")
        )
        self.fill_cvvdp_default_combo()
        self.cvvdp_default.currentIndexChanged.connect(self._on_edited)
        cvvdp_row.addWidget(self.cvvdp_default)
        cvvdp_row.addStretch(1)
        defaults_layout.addLayout(cvvdp_row)
        return defaults_box

    def _build_gpu_box(self, gpu_vendors: list[GpuVendor]) -> QGroupBox:
        """Vship's build, and whether the GPU metrics share one pass."""
        settings = self._settings
        gpu_box = QGroupBox(tr("GPU metrics"))
        gpu_layout = QVBoxLayout(gpu_box)
        # Which Vship build: Auto unless the user has a reason to pick one.
        # Vulkan runs on any GPU. CUDA is offered with an NVIDIA GPU and HIP
        # with an AMD one, or when already chosen, so that a GPU detection
        # missed does not lose the choice.
        self.gpu_backend = QComboBox()
        choices = [(tr("Auto (CUDA on NVIDIA, HIP on AMD, Vulkan on other GPUs)"), "auto"),
                   (tr("Vulkan (any GPU)"), "vulkan")]
        if GpuVendor.NVIDIA in gpu_vendors or settings.gpu_backend == "cuda":
            choices.append((tr("CUDA (NVIDIA)"), "cuda"))
        if GpuVendor.AMD in gpu_vendors or settings.gpu_backend == "hip":
            choices.append((tr("HIP (AMD)"), "hip"))
        for label, backend in choices:
            self.gpu_backend.addItem(label, backend)
        self.gpu_backend.setCurrentIndex(max(0, self.gpu_backend.findData(settings.gpu_backend)))
        self.gpu_backend.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.gpu_backend.setToolTip(tr(
            "Which of Vship's builds calculates the GPU metrics. Auto uses the fastest one whose scores agree with "
            "the reference: CUDA on NVIDIA, HIP on AMD, and Vulkan on other GPUs, such as Intel's. Vulkan runs on "
            "any GPU. If the chosen build cannot run here, the others are tried."))
        self.gpu_backend.currentIndexChanged.connect(self._on_edited)
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel(tr("GPU backend:")))
        backend_row.addWidget(self.gpu_backend)
        backend_row.addStretch(1)
        gpu_layout.addLayout(backend_row)
        self.gpu_together = QCheckBox(
            tr("Calculate SSIMULACRA2, Butteraugli and CVVDP together, in one pass per video"))
        self.gpu_together.setToolTip(
            tr("Each video is decoded once for all its GPU metrics instead of once for each. Where the "
            "GPU decodes the videos this gains little; where the CPU does -- 4K VVC, which no GPU "
            "decodes -- it saves most of the decoding.\n\n"
            "Very heavy on GPU memory at 4K: about 7.3 GB for all three at once, against at most "
            "4.8 GB one at a time (CVVDP). A metric that fails in the shared pass, out of GPU memory "
            "for example, is calculated again on its own.\n\nThe scores are the same either way. "
            "Applies from the next run."))
        self.gpu_together.setChecked(settings.gpu_metrics_together)
        self.gpu_together.toggled.connect(self._on_edited)
        gpu_layout.addWidget(self.gpu_together)
        gpu_note = QLabel(
            tr("Decodes each video once instead of once per metric: much less CPU work for 4K VVC. "
            "Very heavy on GPU memory at 4K: about 7.3 GB for all three together."))
        gpu_note.setWordWrap(True)
        theme.style(gpu_note, "color: {muted};")
        gpu_layout.addWidget(gpu_note)
        return gpu_box

    def _build_compare_box(self) -> QGroupBox:
        """How many test videos Video Compare keeps decoding."""
        settings = self._settings
        compare_box = QGroupBox(tr("Video Compare"))
        compare_layout = QVBoxLayout(compare_box)
        # A dropdown, not a spin box: 1 to 9 is one click away.
        self.decoded_videos = QComboBox()
        for count in range(1, 10):
            self.decoded_videos.addItem(str(count), count)
        self.decoded_videos.setCurrentIndex(max(0, self.decoded_videos.findData(settings.compare_decoded_videos)))
        # As wide as its widest entry and no wider: a form row would
        # otherwise stretch a one-digit dropdown across the window.
        self.decoded_videos.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.decoded_videos.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.decoded_videos.setToolTip(
            tr("How many test videos keep decoding while one is shown, so the "
            "left and right arrows switch to a video that is already running: "
            "the selected one, then the one to its right, its left, the next "
            "right, the next left, and so on. Each is a running GPU decoder "
            "(about 250 MB of RAM for 4K), plus the source. Takes effect "
            "immediately, even during playback.")
        )
        self.decoded_videos.currentIndexChanged.connect(self._on_edited)
        # The dropdown first, at the left edge, with its label beside it, as
        # the check boxes above sit. A form row would put the field wherever
        # the label column ends.
        decoded_row = QHBoxLayout()
        decoded_row.addWidget(self.decoded_videos)
        decoded_row.addWidget(
            QLabel(tr("Number of test videos decoded simultaneously for fast comparison switching"))
        )
        decoded_row.addStretch(1)
        compare_layout.addLayout(decoded_row)
        return compare_box

    def _build_window_box(self) -> QGroupBox:
        """Language, theme, and what the app reopens and checks at startup."""
        settings = self._settings
        window_box = QGroupBox(tr("Window"))
        window_layout = QVBoxLayout(window_box)
        # Each language in its own name, so it can be found by someone who
        # cannot read the one the window is in.
        self.language = QComboBox()
        windows = i18n.windows_language()
        self.language.addItem(
            tr("Same as Windows ({language})", language=i18n.LANGUAGES.get(windows, "English")), "")
        for code, name in i18n.LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.setCurrentIndex(max(0, self.language.findData(settings.language)))
        self.language.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.language.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.language.setToolTip(tr(
            "The language of the window. Takes effect when the app is next started. The log, saved results "
            "and exported files stay in English."))
        self.language.currentIndexChanged.connect(self._on_edited)
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel(tr("Language:")))
        language_row.addWidget(self.language)
        language_row.addStretch(1)
        window_layout.addLayout(language_row)
        # Light or dark, at once; "Same as Windows" also follows Windows
        # changing while the app is open.
        self.theme = QComboBox()
        for label, value in ((tr("Same as Windows"), ""), (tr("Light"), "light"), (tr("Dark"), "dark")):
            self.theme.addItem(label, value)
        self.theme.setCurrentIndex(max(0, self.theme.findData(settings.theme)))
        self.theme.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.theme.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.theme.currentIndexChanged.connect(self._on_edited)
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel(tr("Theme:")))
        theme_row.addWidget(self.theme)
        theme_row.addStretch(1)
        window_layout.addLayout(theme_row)
        self.remember_size = QCheckBox(tr("Reopen at the size the window was last closed at"))
        self.remember_size.setChecked(settings.remember_window_size)
        self.remember_size.toggled.connect(self._on_edited)
        window_layout.addWidget(self.remember_size)
        self.remember_videos = QCheckBox(tr("Reopen the videos that were open when the app was last closed"))
        self.remember_videos.setToolTip(tr(
            "The reference and the test videos, those still there. Resolution tests are not reopened."))
        self.remember_videos.setChecked(settings.remember_videos)
        self.remember_videos.toggled.connect(self._on_edited)
        window_layout.addWidget(self.remember_videos)
        self.check_updates = QCheckBox(tr("Check GitHub for a newer version when the app starts"))
        self.check_updates.setToolTip(
            tr("Once, at startup, asks GitHub's public list of releases whether there is a newer one, "
            "and says so only if there is. Nothing is sent but the request; GitHub sees your "
            "address, as it would for any page."))
        self.check_updates.setChecked(settings.check_for_updates)
        self.check_updates.toggled.connect(self._on_edited)
        window_layout.addWidget(self.check_updates)
        return window_box

    # ------------------------------------------------------------------ for the window

    def refresh_status(self) -> None:
        """Reads the tools and the cache again, so that the tab shows what is
        there. Called when the tab is shown, when its folders change and
        when the saved results are cleared.

        The saved results are counted on a thread of their own. Don't count
        them on the window's thread: a thousand take 0.3 s, which held the
        window at startup and at each tick box."""
        tools = check_tools()
        problems = tools.problems
        ok = not problems
        self.ffmpeg_status.setText(
            tr("ffmpeg {version} and ffprobe found.", version=format_version(tools.ffmpeg.version))
            if ok else " ".join(tr_message(problem) for problem in problems)
        )
        theme.style(self.ffmpeg_status, "color: {good};" if ok else "color: {failed};")

        directory = result_cache.cache_dir()
        self._count_generation += 1
        generation = self._count_generation

        def count() -> None:
            """Counts the saved results, off the window's thread."""
            comparisons, size_bytes = result_cache.cache_summary(directory)
            with contextlib.suppress(RuntimeError):  # the window closed meanwhile
                self._cache_counted.emit((generation, directory, comparisons, size_bytes))

        threading.Thread(target=count, name="cache-count", daemon=True).start()

    def fill_cvvdp_default_combo(self) -> None:
        """Lists every CVVDP preset in Settings, the default selected. The
        window calls it again when a preset is saved or deleted."""
        combo = self.cvvdp_default
        combo.blockSignals(True)
        try:
            combo.clear()
            for preset in cvvdp_presets(self._settings.cvvdp_presets):
                combo.addItem(preset_label(preset), preset.name)
            chosen = self._settings.cvvdp_default_preset or DEFAULT_PRESET.name
            combo.setCurrentIndex(max(0, combo.findData(chosen)))
        finally:
            combo.blockSignals(False)

    def show_ffmpeg_folder(self, directory: str) -> None:
        """Shows a FFmpeg folder the window has already saved (after the
        user located ffmpeg.exe), without treating it as an edit."""
        self.ffmpeg_edit.setText(directory)

    def show_default_metric(self, key: str, checked: bool) -> None:
        """Ticks or unticks a "Default metrics" box to match a default the
        window has already saved (from the table's column heading), without
        treating it as an edit."""
        box = self.default_metrics.get(key)
        if box is not None:
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)

    def clear_saved_results(self) -> None:
        """Deletes the saved results, after asking."""
        directory = result_cache.cache_dir()
        count, _size_bytes = result_cache.cache_summary(directory)
        if count == 0 and self._file_writes.pending == 0:
            self.status.setText(tr("There are no saved results to clear."))
            return
        if self._clear_result is not None:
            self.status.setText(tr("Saved results are already being cleared."))
            return
        confirm = QMessageBox.question(
            self, tr("Clear saved results"),
            tr("Delete saved results from {directory}?\n\nVideos already scored will have to be recomputed.", directory=directory),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        # Cache writes and deletion share the same serial queue. If a run
        # just finished, its pending store therefore lands before clear_all
        # rather than recreating an entry after the user cleared everything.
        self._clear_result = []
        cleared = self._clear_result
        # `directory` is the folder the confirmation dialog just named. Don't
        # look the folder up inside the queued task: if the cache setting
        # changes before the task runs, it deletes a folder the user was never
        # asked about.
        self._file_writes.submit("clear saved results", lambda: cleared.append(result_cache.clear_all(directory)))
        self.status.setText(tr("Clearing saved results..."))

    # ------------------------------------------------------------------ inside the tab

    def _show_cache_summary(self, counted: tuple) -> None:
        """Shows how many results are saved and their size, unless a newer count was asked for."""
        generation, directory, count, size_bytes = counted
        if generation != self._count_generation:
            return  # counted again since, perhaps in another folder
        self.cache_summary.setText(
            ntr("{count} saved result, {size:.1f} MB in {directory}", "{count} saved results, {size:.1f} MB in {directory}",
                count, size=size_bytes / 1048576, directory=directory)
        )

    def _on_file_writes_idle(self) -> None:
        """Says how many saved results were removed, once clearing them is done."""
        if self._clear_result is not None:
            removed = sum(self._clear_result)
            self._clear_result = None
            self.status.setText(ntr("Removed {count} saved result.", "Removed {count} saved results.", removed))
            self.refresh_status()

    def _on_edited(self, *_args) -> None:
        """Saves the tab, applies what is global, and tells the window."""
        settings = self._settings
        before = copy.copy(settings)  # to see what this edit changed
        self._read_controls()
        self.decoded_videos_changed.emit(settings.compare_decoded_videos)
        self._apply_global_changes(before)
        # Only the starting point for new rows; existing rows keep whatever
        # they were given.
        self.defaults_changed.emit()
        error = settings.save()
        if settings.ffmpeg_dir != before.ffmpeg_dir:
            # After the save: the lookup reads the folder from the file.
            ffmpeg_dir_changed()
            self.ffmpeg_folder_changed.emit()
        self.status.setText(
            tr_message(error) if error else
            tr("Settings saved. The new language shows when the app is next started.")
            if settings.language != before.language else tr("Settings saved."))
        if settings.ffmpeg_dir != before.ffmpeg_dir or settings.cache_dir != before.cache_dir:
            self.refresh_status()

    def _read_controls(self) -> None:
        """Copies every control's value into the settings."""
        settings = self._settings
        settings.ffmpeg_dir = self.ffmpeg_edit.text()
        settings.cache_dir = self.cache_edit.text()
        settings.export_dir = self.export_edit.text()
        settings.use_cache = self.use_cache.isChecked()
        settings.default_gpu_decode = self.default_gpu.isChecked()
        for key, box in self.default_metrics.items():
            setattr(settings, f"default_compute_{key}", box.isChecked())
        chosen = self.cvvdp_default.currentData()
        settings.cvvdp_default_preset = "" if chosen in (None, DEFAULT_PRESET.name) else chosen
        settings.compare_decoded_videos = int(self.decoded_videos.currentData())
        settings.remember_window_size = self.remember_size.isChecked()
        settings.remember_videos = self.remember_videos.isChecked()
        if not settings.remember_videos:
            # Paths are kept only while asked for.
            settings.remembered_reference = ""
            settings.remembered_tests = []
        settings.check_for_updates = self.check_updates.isChecked()
        settings.language = self.language.currentData() or ""
        settings.theme = self.theme.currentData() or ""
        settings.gpu_metrics_together = self.gpu_together.isChecked()
        settings.gpu_backend = self.gpu_backend.currentData() or "auto"

    def _apply_global_changes(self, before: Settings) -> None:
        """Applies at once what changed since `before` and is not the
        window's alone: the theme, the GPU backends and the saved-results
        folder."""
        settings = self._settings
        if settings.theme != before.theme:
            theme.apply_theme(settings.theme)  # the window's changeEvent follows
        if settings.gpu_backend != before.gpu_backend:
            # Probed again now, off the UI thread, for the next run and for
            # what the table says about GPU metrics.
            perceptual_vship.set_vship_backend(settings.gpu_backend)
            perceptual_vship.start_vship_probe()
            vmaf_gpu.set_gpu_backend(settings.gpu_backend)  # VMAF follows it: Vulkan or CUDA
            vmaf_gpu.start_gpu_vmaf_probe()
        if settings.cache_dir != before.cache_dir:
            result_cache.set_cache_dir_override(settings.cache_dir_path())

    def _pick_directory(self, title: str, edit: QLineEdit) -> None:
        """Lets the user pick a folder for `edit`, and saves it."""
        directory = QFileDialog.getExistingDirectory(self, title, edit.text())
        if directory:
            edit.setText(directory)
            self._on_edited()

    def _on_pick_ffmpeg_dir(self) -> None:
        """Lets the user pick the folder of ffmpeg.exe."""
        self._pick_directory(tr("Select the folder containing ffmpeg.exe"), self.ffmpeg_edit)

    def _on_pick_cache_dir(self) -> None:
        """Lets the user pick the folder for saved results."""
        self._pick_directory(tr("Where should saved results be kept?"), self.cache_edit)

    def _on_pick_export_dir(self) -> None:
        """Lets the user pick the folder for exports."""
        self._pick_directory(tr("Where should exports be written?"), self.export_edit)

    def _on_copy_log(self) -> None:
        """Puts the part of the log worth sharing on the clipboard."""
        try:
            shared = app_log.log_text_to_share()
        except OSError as error:
            self.status.setText(tr("Could not read the log: {error}", error=error))
            return
        if shared is None:
            self.status.setText(tr("There is no log to copy yet."))
            return
        text, shortened = shared
        QApplication.clipboard().setText(text)
        lines = text.count("\n") + 1
        self.status.setText(
            ntr("Copied the log to the clipboard ({count} line, the middle left out -- Export log saves all of it).",
                "Copied the log to the clipboard ({count} lines, the middle left out -- Export log saves all of it).",
                lines) if shortened else
            ntr("Copied the log to the clipboard ({count} line).", "Copied the log to the clipboard ({count} lines).", lines))

    def _on_export_log(self) -> None:
        """Saves the log files as one .zip wherever the user picks."""
        start = Path(self._settings.export_dir) if self._settings.export_dir else Path.home() / "Documents"
        name = f"{APP_NAME} log {time.strftime('%Y-%m-%d %H%M')}.zip"
        path, _ = QFileDialog.getSaveFileName(self, tr("Export log"), str(start / name), tr("Zip archive (*.zip)"))
        if not path:
            return
        try:
            names = app_log.export_logs(Path(path))
        except OSError as error:
            self.status.setText(tr("Could not export the log: {error}", error=error))
            return
        if not names:
            self.status.setText(tr("There is no log to export yet."))
            return
        _log.info("Log exported to %s", path)
        self.status.setText(tr("Log exported to {path}", path=path))

    def _on_open_log_dir(self) -> None:
        """Opens the log folder in the file manager."""
        directory = app_log.log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _on_open_cache_dir(self) -> None:
        """Opens the saved-results folder in the file manager."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result_cache.cache_dir())))
