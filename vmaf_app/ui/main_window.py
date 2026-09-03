"""Main window: pick a reference video and one or more distorted videos,
configure VMAF options per video, run, and browse/compare completed results.

The distorted-file list is a table (path / media info / VMAF score) so each
file's score -- or live "Frame N" progress while it's running -- is visible
at a glance in the row itself, the way FFMetrics shows it.

Each row carries its own VmafOptions (crop handling, model, GPU decode,
etc.) instead of one shared global setting. The Options panel acts as an
inspector for whichever row(s) are selected in the table: selecting a row
loads its settings into the panel, and editing the panel writes back to
every currently-selected row (so you can batch-edit several at once, or
give a single video its own crop/model/etc. independent of the rest).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTime, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from vmaf_app.core import result_cache
from vmaf_app.core.ffmpeg_locate import check_tools, exe_name, format_version, set_ffmpeg_dir_override
from vmaf_app.core.ffprobe import ProbeError, probe_video
from vmaf_app.core.gpu import detected_gpu_vendors
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE, resolve_model
from vmaf_app.core.models import (
    RESAMPLE_TARGET_CHOICES,
    CropMode,
    GpuVendor,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
    clone_options,
    synthetic_resample_distorted_path,
    synthetic_scale_direction_variant_path,
)
from vmaf_app.core.run_io import load_run, save_run, unique_output_path
from vmaf_app.core.settings import Settings
from vmaf_app.core.stats import stats_for_run
from vmaf_app.core.time_format import format_hms
from vmaf_app.core.vmaf_runner import VmafRunError, estimate_total_frames, validate_video_pair
from vmaf_app.ui.formatting import NOT_COMPUTED, bitrate_string, media_info_string, vmaf_band_colour
from vmaf_app.ui.graph_panel import GraphPanel
from vmaf_app.ui.probe_worker import ProbeWorker
from vmaf_app.ui.widgets import CheckableHeaderView, FillColumnTable
from vmaf_app.ui.worker import VmafJob, VmafWorker

_MODEL_CHOICES = [
    ("Auto (recommended: picks 4K model for UHD distorted video)", AUTO_MODEL_CHOICE),
    ("VMAF v0.6.1 (default, standard viewing)", "version=vmaf_v0.6.1"),
    ("VMAF v0.6.1neg (no enhancement gain)", "version=vmaf_v0.6.1neg"),
    ("VMAF 4K v0.6.1 (4K / large-screen viewing)", "version=vmaf_4k_v0.6.1"),
    ("Custom model file...", CUSTOM_MODEL_CHOICE),
]

_SCALE_ALGORITHMS = ["bicubic", "lanczos", "bilinear", "spline"]

_GPU_VENDOR_BY_INDEX = {0: GpuVendor.AUTO, 1: GpuVendor.NVIDIA, 2: GpuVendor.INTEL, 3: GpuVendor.AMD}
_GPU_VENDOR_INDEX = {v: k for k, v in _GPU_VENDOR_BY_INDEX.items()}

COL_CHECK, COL_PATH, COL_INFO, COL_SCALING, COL_BITRATE, COL_PSNR, COL_SSIM, COL_VMAF, COL_XPSNR = range(9)

# Tab order. A frame-comparison tab is planned between Graph and Settings;
# adding it means inserting here and in _build_ui.
TAB_VIDEOS, TAB_GRAPH, TAB_SETTINGS = range(3)

# The metric columns, in table order: (column, label, the VmafOptions field or
# libvmaf feature it maps to). VMAF has no toggle -- it's what the app exists
# to compute, and the whole pipeline is built on libvmaf.
_METRIC_COLUMNS = [
    (COL_PSNR, "PSNR", "name=psnr"),
    (COL_SSIM, "SSIM", "name=float_ssim"),
    (COL_VMAF, "VMAF", None),
    (COL_XPSNR, "XPSNR", "xpsnr"),
]

class CompletedRun:
    """A finished run plus the label it's shown under and its summary stats,
    computed once here rather than recomputed everywhere it's displayed."""

    def __init__(self, result, label: str):
        self.result = result
        self.label = label
        self.stats = stats_for_run(result)
        self.graph_identity = object()


@dataclass
class RowData:
    path: Path
    video_info: VideoInfo | None = None
    completed_run: CompletedRun | None = None
    options: VmafOptions = field(default_factory=VmafOptions)
    # True only for a "Test both" companion row (see
    # _add_opposite_scale_direction_rows), where options.scale_direction is
    # set explicitly and unambiguously to whichever direction this row
    # exists to represent -- unlike a normal row, where it's just whatever
    # the panel's default happened to be when the row was added, and a
    # cached/loaded result's own recorded direction is more trustworthy.
    scale_direction_pinned: bool = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VMAF Calculator")
        # Settings first: the ffmpeg location and what new rows default to
        # both come from them, so they must be applied before the startup
        # tool check or any row is added.
        self._settings = Settings.load()
        # Only when one is actually configured. Calling this unconditionally
        # wrote to persistent QSettings on every launch, and each call clears
        # the tool-lookup cache -- which re-probes ffmpeg by spawning it.
        if self._settings.ffmpeg_dir_path() is not None:
            self._apply_ffmpeg_setting()
        result_cache.set_cache_dir_override(self._settings.cache_dir_path())
        if self._settings.remember_window_size:
            self.resize(self._settings.window_width, self._settings.window_height)
        else:
            self.resize(1280, 800)
        self.setMinimumSize(1050, 650)

        self._source_info: VideoInfo | None = None
        self._rows: list[RowData] = []
        self._worker: VmafWorker | None = None
        self._probe_worker: ProbeWorker | None = None
        self._probe_workers: list[ProbeWorker] = []
        self._probe_generation = 0
        # These track the active run by RowData *identity*, not by table row
        # index: removing a row mid-run shifts every later index down, which
        # used to make a finishing job write its result to the wrong row --
        # or crash with IndexError when the shifted index ran off the end.
        self._job_rows: list[RowData] = []  # job index -> the row that job belongs to
        self._job_total_frames: list[int] = []  # job index -> estimated frame count, for queue ETA
        self._job_cache_options: list[VmafOptions] = []
        self._checked_rows_for_run: list[RowData] = []  # rows checked when Run was clicked, incl. already-scored ones
        self._current_job_index: int | None = None
        self._run_failed_count = 0
        self._run_was_cancelled = False

        # Per-video settings machinery: the Options panel is an inspector for
        # whichever rows are selected, not one global setting.
        self._default_options = self._options_from_settings()  # what a newly-added row starts with
        self._panel_target_rows: list[int] = []  # rows the panel currently edits
        self._panel_custom_model_path: str | None = None  # staging for the panel's "Custom model" choice
        self._syncing_panel = False  # guards against write-back while populating the panel programmatically

        self._build_ui()
        self._check_ffmpeg(prompt=True)  # startup check: both tools present, ffmpeg new enough
        self._on_table_selection_changed()

    def closeEvent(self, event) -> None:
        # A run still in flight owns a live ffmpeg subprocess. Without
        # cancelling it here, closing the window leaves ffmpeg running in the
        # background chewing CPU/GPU with nothing to report to, and tears
        # down a QThread that's still executing.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        for worker in self._probe_workers:
            if worker.isRunning():
                worker.cancel()
                worker.wait(5000)
        # Remember the window size, if asked to. The graph is a tab now, so
        # there is no second window to tear down.
        if self._settings.remember_window_size:
            self._settings.window_width = self.width()
            self._settings.window_height = self.height()
            self._settings.save()
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Banner + its action button live in one row that is actually added
        # to the layout -- the button used to be built into a throwaway
        # layout that was never attached, so it never appeared and the
        # warning was unactionable.
        banner_row = QHBoxLayout()
        self._ffmpeg_banner = QLabel()
        self._ffmpeg_banner.setStyleSheet("background: #fff3cd; padding: 6px; border: 1px solid #ffe08a;")
        self._ffmpeg_banner.setWordWrap(True)
        self._ffmpeg_banner.setVisible(False)
        self._locate_ffmpeg_btn = QPushButton("Locate ffmpeg.exe...")
        self._locate_ffmpeg_btn.clicked.connect(self._on_locate_ffmpeg)
        self._locate_ffmpeg_btn.setVisible(False)
        banner_row.addWidget(self._ffmpeg_banner, stretch=1)
        banner_row.addWidget(self._locate_ffmpeg_btn)
        root.addLayout(banner_row)

        # Tabs, not separate windows: the graph lives beside the run that
        # produced it, its series survive switching away and back, and there
        # is no second taskbar entry to manage. A frame-comparison tab is
        # planned and slots in between Graph and Settings.
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        videos_page = QWidget()
        videos_layout = QVBoxLayout(videos_page)
        videos_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)
        videos_layout.addWidget(splitter, stretch=1)
        # Videos on top, options underneath it (not beside it), so the file
        # table gets the full window width -- it's the part that grows with
        # the number of encodes being compared.
        splitter.addWidget(self._build_files_panel())
        splitter.addWidget(self._build_options_panel())
        splitter.addWidget(self._build_run_panel())
        splitter.setSizes([420, 260, 120])
        self.tabs.addTab(videos_page, "Videos")

        self.graph_panel = GraphPanel()
        self.tabs.addTab(self.graph_panel, "Graph")
        # Opening the tab is enough; pressing a button to populate it was
        # a leftover from when it was a separate window that had to be
        # opened explicitly.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.tabs.addTab(self._build_settings_panel(), "Settings")

    # ------------------------------------------------------------------ settings tab
    def _options_from_settings(self) -> VmafOptions:
        """The options a newly added video starts from. Only a starting
        point: each row's own settings are edited in the Options panel."""
        return VmafOptions(
            gpu_decode=self._settings.default_gpu_decode,
            extra_features=self._settings.default_extra_features(),
            compute_xpsnr=self._settings.default_compute_xpsnr,
        )

    def _apply_ffmpeg_setting(self) -> None:
        """Points the finder at the configured folder, or clears the override
        so it goes back to searching PATH and the known install locations.

        set_ffmpeg_dir_override writes to persistent QSettings, so this is
        only called when the setting actually changes -- not on every launch.
        """
        configured = self._settings.ffmpeg_dir_path()
        set_ffmpeg_dir_override(str(configured) if configured else "")

    def _build_settings_panel(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        tools_box = QGroupBox("ffmpeg")
        tools_form = QFormLayout(tools_box)
        self.settings_ffmpeg_edit = QLineEdit(self._settings.ffmpeg_dir)
        self.settings_ffmpeg_edit.setPlaceholderText("blank = search PATH")
        self.settings_ffmpeg_edit.editingFinished.connect(self._on_settings_edited)
        browse_ffmpeg = QPushButton("Browse...")
        browse_ffmpeg.clicked.connect(self._on_pick_ffmpeg_dir)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_row.addWidget(self.settings_ffmpeg_edit, stretch=1)
        ffmpeg_row.addWidget(browse_ffmpeg)
        tools_form.addRow("ffmpeg folder:", ffmpeg_row)
        self.settings_ffmpeg_status = QLabel()
        tools_form.addRow("", self.settings_ffmpeg_status)
        outer.addWidget(tools_box)

        storage_box = QGroupBox("Storage")
        storage_form = QFormLayout(storage_box)
        self.settings_cache_edit = QLineEdit(self._settings.cache_dir)
        self.settings_cache_edit.setPlaceholderText("blank = default app data folder")
        self.settings_cache_edit.editingFinished.connect(self._on_settings_edited)
        browse_cache = QPushButton("Browse...")
        browse_cache.clicked.connect(self._on_pick_cache_dir)
        open_cache = QPushButton("Open")
        open_cache.clicked.connect(self._on_open_cache_dir)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.settings_cache_edit, stretch=1)
        cache_row.addWidget(browse_cache)
        cache_row.addWidget(open_cache)
        storage_form.addRow("Saved results:", cache_row)

        self.settings_cache_summary = QLabel()
        clear_cache = QPushButton("Clear saved results")
        clear_cache.clicked.connect(self._on_clear_cache)
        summary_row = QHBoxLayout()
        summary_row.addWidget(self.settings_cache_summary, stretch=1)
        summary_row.addWidget(clear_cache)
        storage_form.addRow("", summary_row)

        self.settings_export_edit = QLineEdit(self._settings.export_dir)
        self.settings_export_edit.setPlaceholderText("blank = ask each time")
        self.settings_export_edit.editingFinished.connect(self._on_settings_edited)
        browse_export = QPushButton("Browse...")
        browse_export.clicked.connect(self._on_pick_export_dir)
        export_row = QHBoxLayout()
        export_row.addWidget(self.settings_export_edit, stretch=1)
        export_row.addWidget(browse_export)
        storage_form.addRow("Export folder:", export_row)

        self.settings_use_cache = QCheckBox("Reuse a saved result when a video is added again")
        self.settings_use_cache.setChecked(self._settings.use_cache)
        self.settings_use_cache.toggled.connect(self._on_settings_edited)
        storage_form.addRow("", self.settings_use_cache)
        outer.addWidget(storage_box)

        defaults_box = QGroupBox("Defaults for newly added videos")
        defaults_layout = QVBoxLayout(defaults_box)
        hint = QLabel(
            "These are only a starting point -- each video's own settings are "
            "edited in the Videos tab."
        )
        hint.setStyleSheet("color: #666; font-style: italic;")
        defaults_layout.addWidget(hint)
        self.settings_default_gpu = QCheckBox("Use GPU decoding")
        self.settings_default_gpu.setChecked(self._settings.default_gpu_decode)
        self.settings_default_gpu.toggled.connect(self._on_settings_edited)
        defaults_layout.addWidget(self.settings_default_gpu)

        metrics_row = QHBoxLayout()
        metrics_row.addWidget(QLabel("Also compute:"))
        self.settings_default_psnr = QCheckBox("PSNR")
        self.settings_default_ssim = QCheckBox("SSIM")
        self.settings_default_xpsnr = QCheckBox("XPSNR")
        for box, value in (
            (self.settings_default_psnr, self._settings.default_compute_psnr),
            (self.settings_default_ssim, self._settings.default_compute_ssim),
            (self.settings_default_xpsnr, self._settings.default_compute_xpsnr),
        ):
            box.setChecked(value)
            box.toggled.connect(self._on_settings_edited)
            metrics_row.addWidget(box)
        metrics_row.addStretch(1)
        defaults_layout.addLayout(metrics_row)
        outer.addWidget(defaults_box)

        window_box = QGroupBox("Window")
        window_layout = QVBoxLayout(window_box)
        self.settings_remember_size = QCheckBox("Reopen at the size the window was last closed at")
        self.settings_remember_size.setChecked(self._settings.remember_window_size)
        self.settings_remember_size.toggled.connect(self._on_settings_edited)
        window_layout.addWidget(self.settings_remember_size)
        outer.addWidget(window_box)

        outer.addStretch(1)
        self.settings_status = QLabel()
        self.settings_status.setStyleSheet("color: #666;")
        outer.addWidget(self.settings_status)

        self._refresh_settings_status()
        return page

    def _refresh_settings_status(self) -> None:
        """Re-reads the tools and the cache so the Settings tab reports what
        is actually there, not what was there at startup."""
        tools = check_tools()
        problems = tools.problems
        ok = not problems
        self.settings_ffmpeg_status.setText(
            f"ffmpeg {format_version(tools.ffmpeg.version)} and ffprobe found."
            if ok else " ".join(problems)
        )
        self.settings_ffmpeg_status.setStyleSheet("color: #207020;" if ok else "color: #a03030;")

        directory = result_cache.cache_dir()
        entries = list(directory.glob("*.vmafrun.json"))
        total = sum(f.stat().st_size for f in entries) / 1_048_576
        self.settings_cache_summary.setText(
            f"{len(entries)} saved result(s), {total:.1f} MB in {directory}"
        )

    def _on_settings_edited(self, *_args) -> None:
        before_ffmpeg = self._settings.ffmpeg_dir
        before_cache = self._settings.cache_dir
        self._settings.ffmpeg_dir = self.settings_ffmpeg_edit.text()
        self._settings.cache_dir = self.settings_cache_edit.text()
        self._settings.export_dir = self.settings_export_edit.text()
        self._settings.use_cache = self.settings_use_cache.isChecked()
        self._settings.default_gpu_decode = self.settings_default_gpu.isChecked()
        self._settings.default_compute_psnr = self.settings_default_psnr.isChecked()
        self._settings.default_compute_ssim = self.settings_default_ssim.isChecked()
        self._settings.default_compute_xpsnr = self.settings_default_xpsnr.isChecked()
        self._settings.remember_window_size = self.settings_remember_size.isChecked()

        if self._settings.ffmpeg_dir != before_ffmpeg:
            self._apply_ffmpeg_setting()
            self._check_ffmpeg(prompt=False)
        if self._settings.cache_dir != before_cache:
            result_cache.set_cache_dir_override(self._settings.cache_dir_path())

        # Only the starting point for new rows; existing rows keep whatever
        # they were given.
        self._default_options = self._options_from_settings()
        error = self._settings.save()
        self.settings_status.setText(error or "Settings saved.")
        self._refresh_settings_status()

    def _pick_directory(self, title: str, edit: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, title, edit.text())
        if directory:
            edit.setText(directory)
            self._on_settings_edited()

    def _on_pick_ffmpeg_dir(self) -> None:
        self._pick_directory("Select the folder containing ffmpeg.exe", self.settings_ffmpeg_edit)

    def _on_pick_cache_dir(self) -> None:
        self._pick_directory("Where should saved results be kept?", self.settings_cache_edit)

    def _on_pick_export_dir(self) -> None:
        self._pick_directory("Where should exports be written?", self.settings_export_edit)

    def _on_open_cache_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result_cache.cache_dir())))

    def _on_clear_cache(self) -> None:
        directory = result_cache.cache_dir()
        entries = list(directory.glob("*.vmafrun.json"))
        if not entries:
            self.settings_status.setText("There are no saved results to clear.")
            return
        confirm = QMessageBox.question(
            self, "Clear saved results",
            f"Delete {len(entries)} saved result(s) from {directory}?\n\n"
            "Videos already scored will have to be recomputed.",
        )
        if confirm != QMessageBox.Yes:
            return
        # Only this app's own entries, by extension -- never the folder
        # itself, which may be somewhere the user also keeps other things.
        removed = 0
        for entry in entries:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                pass
        self.settings_status.setText(f"Removed {removed} saved result(s).")
        self._refresh_settings_status()

    def _build_files_panel(self) -> QWidget:
        files_box = QGroupBox("Videos")
        self.files_box = files_box
        files_layout = QVBoxLayout(files_box)

        files_layout.addWidget(QLabel("Reference (source) video:"))
        src_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        src_browse = QPushButton("Browse...")
        src_browse.clicked.connect(self._on_browse_source)
        src_row.addWidget(self.source_edit, stretch=1)
        src_row.addWidget(src_browse)
        files_layout.addLayout(src_row)
        self.source_info_label = QLabel("No source selected.")
        self.source_info_label.setStyleSheet("color: #666;")
        files_layout.addWidget(self.source_info_label)

        files_layout.addWidget(QLabel(
            "Distorted video(s) to compare against the source "
            "(select one or more below to edit their settings underneath; "
            "tick a metric's column header to compute it):"
        ))
        metric_cols = [c for c, _, _ in _METRIC_COLUMNS]
        self.distorted_table = FillColumnTable(
            0, 9, fill_column=COL_PATH,
            other_columns=[COL_CHECK, COL_INFO, COL_SCALING, COL_BITRATE, *metric_cols],
        )
        self.metric_header = CheckableHeaderView(
            # VMAF has no checkbox: it's what the app computes, always.
            {COL_PSNR: False, COL_SSIM: False, COL_XPSNR: False},
            self.distorted_table,
        )
        self.distorted_table.setHorizontalHeader(self.metric_header)
        self.metric_header.sectionToggled.connect(self._on_metric_column_toggled)
        self.distorted_table.setHorizontalHeaderLabels(
            ["", "File name", "Media info", "Scaling", "Bitrate", "   PSNR", "   SSIM", "VMAF", "   XPSNR"]
        )
        self.distorted_table.verticalHeader().setVisible(False)
        self.distorted_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.distorted_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.distorted_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.distorted_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.distorted_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.distorted_table.customContextMenuRequested.connect(self._on_table_context_menu)
        # All columns are Interactive (drag-resizable), including PATH --
        # FillColumnTable makes PATH additionally auto-fill whatever space is
        # left over (see its docstring) instead of sitting at a fixed width
        # with wasted space beside it. INFO/BITRATE/VMAF are also kept snug to
        # their actual content automatically (see _set_row_info/
        # _set_row_vmaf_text) the way ResizeToContents used to look. A
        # horizontal scrollbar still appears if the columns don't all fit,
        # rather than squeezing everything down to fit the window.
        header = self.distorted_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        # File name and Media info grow with the window, so their headers are
        # left-aligned to stay above their content instead of drifting into
        # the middle of an empty column.
        for col in (COL_PATH, COL_INFO):
            head_item = self.distorted_table.horizontalHeaderItem(col)
            if head_item is not None:
                head_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setStretchLastSection(False)
        self.distorted_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.distorted_table.setColumnWidth(COL_CHECK, 28)
        self.distorted_table.setColumnWidth(COL_INFO, 150)
        self.distorted_table.setColumnWidth(COL_SCALING, 90)
        self.distorted_table.setColumnWidth(COL_BITRATE, 65)
        for col, _, _ in _METRIC_COLUMNS:
            self.distorted_table.setColumnWidth(col, 78)
        files_layout.addWidget(self.distorted_table, stretch=1)

        dist_btn_row = QHBoxLayout()
        add_dist_btn = QPushButton("Add files...")
        add_dist_btn.clicked.connect(self._on_add_distorted)
        add_resample_btn = QPushButton("Add resolution test...")
        add_resample_btn.setToolTip(
            "Tests VMAF for downscaling the source to a lower resolution and "
            "scaling it back up -- no separate distorted file needed."
        )
        add_resample_btn.clicked.connect(self._on_add_resample_test)
        remove_dist_btn = QPushButton("Remove selected")
        remove_dist_btn.clicked.connect(self._on_remove_distorted)
        self.remove_all_btn = QPushButton("Remove all")
        self.remove_all_btn.clicked.connect(self._on_remove_all_distorted)
        dist_btn_row.addWidget(add_dist_btn)
        dist_btn_row.addWidget(add_resample_btn)
        dist_btn_row.addWidget(remove_dist_btn)
        dist_btn_row.addWidget(self.remove_all_btn)
        dist_btn_row.addStretch(1)
        files_layout.addLayout(dist_btn_row)

        return files_box

    def _build_options_panel(self) -> QWidget:
        # --- per-video options inspector (below the file table) ---
        options_box = QGroupBox("Options")
        options_layout = QVBoxLayout(options_box)
        self.options_box = options_box

        self.panel_target_label = QLabel("")
        self.panel_target_label.setStyleSheet("color: #666; font-style: italic;")
        self.panel_target_label.setWordWrap(True)
        options_layout.addWidget(self.panel_target_label)

        form = QFormLayout()
        self.model_combo = QComboBox()
        for name, _ in _MODEL_CHOICES:
            self.model_combo.addItem(name)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("VMAF model:", self.model_combo)

        self.gpu_checkbox = QCheckBox("Use GPU decoding")
        self.gpu_checkbox.setToolTip(
            "Hardware-decodes the source and the distorted video. Each is "
            "decided separately, and either one falls back to the CPU on its "
            "own if this GPU can't decode its format."
        )
        self.gpu_checkbox.setChecked(True)
        self.gpu_vendor_combo = QComboBox()
        self.gpu_vendor_combo.addItems(["Auto-detect", "NVIDIA", "Intel", "AMD"])
        self.gpu_checkbox.stateChanged.connect(
            lambda st: self.gpu_vendor_combo.setEnabled(bool(st))
        )
        self.gpu_checkbox.stateChanged.connect(
            lambda _value: self._on_panel_field_edited("gpu")
        )
        self.gpu_vendor_combo.currentIndexChanged.connect(
            lambda _value: self._on_panel_field_edited("gpu")
        )
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addWidget(self.gpu_vendor_combo)
        form.addRow("GPU decode:", gpu_row)

        detected = detected_gpu_vendors()
        if detected:
            names = ", ".join(v.value.upper() for v in detected)
            form.addRow("", QLabel(f"Detected GPU(s): {names}"))

        self.crop_combo = QComboBox()
        self.crop_combo.addItems([
            "Auto-detect black bars (recommended)",
            "None (use full frame)",
        ])
        self.crop_combo.currentIndexChanged.connect(
            lambda _value: self._on_panel_field_edited("crop_mode")
        )
        form.addRow("Black-bar handling:", self.crop_combo)

        # The two groups sit side by side rather than stacked: the panel is
        # full window width now, so stacking them left a lot of empty space
        # to the right and pushed the file table up.
        columns = QHBoxLayout()
        basic_column = QWidget()
        basic_column.setLayout(form)
        columns.addWidget(basic_column, stretch=1)
        options_layout.addLayout(columns)

        adv_box = QGroupBox("Advanced")
        adv_box.setCheckable(False)
        adv_form = QFormLayout(adv_box)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 128)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText("Auto")
        self.threads_spin.valueChanged.connect(
            lambda _value: self._on_panel_field_edited("n_threads")
        )
        adv_form.addRow("libvmaf threads:", self.threads_spin)

        self.subsample_spin = QSpinBox()
        self.subsample_spin.setRange(1, 60)
        self.subsample_spin.setValue(1)
        self.subsample_spin.valueChanged.connect(
            lambda _value: self._on_panel_field_edited("n_subsample")
        )
        adv_form.addRow("Frame subsample (1 = every frame):", self.subsample_spin)

        self.duration_edit = QTimeEdit()
        self.duration_edit.setDisplayFormat("HH:mm:ss.zzz")
        self.duration_edit.setTime(QTime(0, 0, 0, 0))
        self.duration_edit.timeChanged.connect(
            lambda _value: self._on_panel_field_edited("duration_limit")
        )
        adv_form.addRow("Duration limit (00:00:00.000 = full video):", self.duration_edit)

        self.scale_algo_combo = QComboBox()
        self.scale_algo_combo.addItems(_SCALE_ALGORITHMS)
        self.scale_algo_combo.currentIndexChanged.connect(
            lambda _value: self._on_panel_field_edited("scale_algorithm")
        )
        adv_form.addRow("Scaling algorithm:", self.scale_algo_combo)

        self.scale_direction_combo = QComboBox()
        self.scale_direction_combo.addItems([
            "Scale source down to match distorted (default)",
            "Scale distorted up to match source",
            "Test both (adds a comparison row)",
        ])
        self.scale_direction_combo.setToolTip(
            "When the two resolutions differ: either evaluate quality at the resolution actually\n"
            "delivered (source scaled to match distorted -- the default), or as if the distorted\n"
            "video were upscaled back to the source's native resolution for playback.\n"
            "\"Test both\" doesn't change this row -- it adds a second row for the same distorted\n"
            "file using the other direction, so you can run and compare both."
        )
        self.scale_direction_combo.currentIndexChanged.connect(self._on_scale_direction_combo_changed)
        adv_form.addRow("Resolution mismatch:", self.scale_direction_combo)

        metrics_hint = QLabel(
            "PSNR / SSIM / XPSNR are toggled from their column headers in the table above."
        )
        metrics_hint.setStyleSheet("color: #666; font-style: italic;")
        metrics_hint.setWordWrap(True)
        adv_form.addRow(metrics_hint)

        columns.addWidget(adv_box, stretch=1)
        options_layout.addStretch(1)

        return options_box

    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run VMAF")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.pause_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch(1)

        load_btn = QPushButton("Load saved run...")
        load_btn.clicked.connect(self._on_load_saved_run)
        save_btn = QPushButton("Save selected...")
        save_btn.clicked.connect(self._on_save_selected)
        compare_btn = QPushButton("Compare selected")
        compare_btn.clicked.connect(self._on_compare_selected)
        self.show_graph_btn = QPushButton("Show graph")
        self.show_graph_btn.setToolTip("Reopens the comparison graph as you last left it (e.g. after closing it).")
        self.show_graph_btn.clicked.connect(self._on_show_graph_clicked)
        run_row.addWidget(load_btn)
        run_row.addWidget(save_btn)
        run_row.addWidget(compare_btn)
        run_row.addWidget(self.show_graph_btn)
        layout.addLayout(run_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.progress_detail_label = QLabel("")
        self.progress_detail_label.setStyleSheet("color: #666;")
        layout.addWidget(self.progress_detail_label)
        layout.addStretch(1)

        return panel

    # ------------------------------------------------------------------ ffmpeg availability
    def _check_ffmpeg(self, *, prompt: bool = False) -> bool:
        """Verifies ffmpeg AND ffprobe both actually run, and that ffmpeg is
        new enough. Returns whether everything checks out. With prompt=True
        (startup), offers a file picker to point at ffmpeg.exe rather than
        just complaining in a banner."""
        status = check_tools()
        if status.ok:
            self._ffmpeg_banner.setVisible(False)
            return True

        problems = status.problems
        self._ffmpeg_banner.setText("  ".join(problems) + "  Click \"Locate ffmpeg.exe...\" to point at it.")
        self._ffmpeg_banner.setVisible(True)
        self._locate_ffmpeg_btn.setVisible(True)

        # A headless run (tests -- see tests/conftest.py) has nobody to answer
        # a modal dialog, so it would block forever rather than prompt.
        if prompt and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            answer = QMessageBox.warning(
                self, "ffmpeg not usable",
                "\n".join(problems)
                + f"\n\nffmpeg was looked for at:\n  {status.ffmpeg.path}"
                + f"\nffprobe at:\n  {status.ffprobe.path}"
                + "\n\nWould you like to point the app at ffmpeg.exe now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                return self._on_locate_ffmpeg()
        return False

    def _on_locate_ffmpeg(self) -> bool:
        """Asks for ffmpeg.exe itself (not its folder -- a file picker is far
        less ambiguous), then expects ffprobe alongside it, asking separately
        only if it isn't there."""
        ffmpeg_exe = exe_name("ffmpeg")
        ffprobe_exe = exe_name("ffprobe")
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Select {ffmpeg_exe}", "", f"{ffmpeg_exe} ({ffmpeg_exe});;All files (*)",
        )
        if not chosen:
            return False

        directory = Path(chosen).parent
        if not (directory / ffprobe_exe).exists():
            QMessageBox.information(
                self, "ffprobe needed too",
                f"{ffprobe_exe} wasn't found next to {ffmpeg_exe}.\n\n"
                f"The app needs both. Please select {ffprobe_exe} as well "
                f"(it normally ships in the same folder).",
            )
            probe_chosen, _ = QFileDialog.getOpenFileName(
                self, f"Select {ffprobe_exe}", str(directory), f"{ffprobe_exe} ({ffprobe_exe});;All files (*)",
            )
            if not probe_chosen:
                return False
            if Path(probe_chosen).parent != directory:
                QMessageBox.warning(
                    self, "Different folders",
                    f"{ffmpeg_exe} and {ffprobe_exe} need to be in the same folder for the app to find both.",
                )
                return False

        set_ffmpeg_dir_override(str(directory))
        status = check_tools()
        if not status.ok:
            QMessageBox.warning(self, "Still not usable", "\n".join(status.problems))
            self._check_ffmpeg()
            return False
        self._ffmpeg_banner.setVisible(False)
        self._locate_ffmpeg_btn.setVisible(False)
        self.status_label.setText(
            f"Using ffmpeg {format_version(status.ffmpeg.version)} from {directory}"
        )
        return True

    # ------------------------------------------------------------------ source selection
    def _on_browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select reference (source) video")
        if not path:
            return
        try:
            info = probe_video(Path(path))
        except ProbeError as e:
            QMessageBox.critical(self, "Could not read video", str(e))
            return
        self._source_info = info
        self.source_edit.setText(path)
        self.source_info_label.setText(
            f"{media_info_string(info)}, {bitrate_string(info)}  ({format_hms(info.duration, decimals=1)})"
        )
        # A different/newly-picked source might match a previously cached
        # (source, distorted) pair for rows that are already in the table.
        # Each of those is a multi-MB JSON parse, so with a few long videos
        # loaded this froze the window for seconds; it goes to the worker,
        # which already knows how to load a cached result and report it.
        if self._rows:
            # Scores belong to a (source, distorted) pair, so a new source
            # invalidates every one of them until the cache says otherwise.
            for row_data in self._rows:
                row_data.completed_run = None
            self._reload_cached_for_all_rows()

    # ------------------------------------------------------------------ distorted-file table
    def _add_table_row(self, path: Path) -> int:
        row = self.distorted_table.rowCount()
        self.distorted_table.insertRow(row)

        check_item = QTableWidgetItem()
        check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        check_item.setCheckState(Qt.Checked)
        self.distorted_table.setItem(row, COL_CHECK, check_item)

        path_item = QTableWidgetItem(path.name)
        path_item.setToolTip(str(path))  # full path still available on hover
        self.distorted_table.setItem(row, COL_PATH, path_item)
        self.distorted_table.setItem(row, COL_INFO, QTableWidgetItem("Probing..."))
        self.distorted_table.setItem(row, COL_SCALING, QTableWidgetItem(""))
        self.distorted_table.setItem(row, COL_BITRATE, QTableWidgetItem(""))
        for col, _, _ in _METRIC_COLUMNS:
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignCenter)
            self.distorted_table.setItem(row, col, item)

        # New rows start with a copy of whatever the panel last showed, so
        # adding several similar files in a row doesn't mean reconfiguring
        # each one from scratch -- but it's still an independent copy from
        # this point on, so editing one row never affects another.
        self._rows.append(RowData(path=path, options=clone_options(self._default_options)))
        self._set_row_metrics(row)
        return row

    def _set_row_metrics(self, row: int) -> None:
        """Fills the PSNR/SSIM/VMAF/XPSNR cells for a row: the mean score if
        it's been computed, "N/A" if that metric isn't switched on for this
        row, and blank while it's enabled but not computed yet."""
        row_data = self._rows[row]
        run = row_data.completed_run
        opts = row_data.options
        enabled = {
            COL_PSNR: "name=psnr" in opts.extra_features,
            COL_SSIM: "name=float_ssim" in opts.extra_features,
            COL_VMAF: True,
            COL_XPSNR: opts.compute_xpsnr,
        }
        for col, _, _ in _METRIC_COLUMNS:
            item = self.distorted_table.item(row, col)
            if item is None:
                continue
            if not enabled[col]:
                item.setText(NOT_COMPUTED)
                item.setForeground(QColor("#999"))
                item.setFont(QFont())
                item.setBackground(QColor(0, 0, 0, 0))
                continue
            value = self._metric_mean(run, col) if run is not None else None
            if value is None:
                item.setText("")
                item.setBackground(QColor(0, 0, 0, 0))
                item.setFont(QFont())
                continue
            item.setText(f"{value:.4f}" if col == COL_SSIM else f"{value:.2f}")
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            item.setForeground(QColor("#000"))
            # Only VMAF has a universally meaningful "good/bad" scale to
            # colour against (0-100); dB and SSIM don't.
            item.setBackground(vmaf_band_colour(value) if col == COL_VMAF else QColor(0, 0, 0, 0))
        self.distorted_table.resizeColumnToContents(COL_VMAF)

    @staticmethod
    def _metric_mean(run: CompletedRun, column: int) -> float | None:
        if column == COL_VMAF:
            return run.stats.mean
        metric = {COL_PSNR: "psnr", COL_SSIM: "ssim", COL_XPSNR: "xpsnr"}[column]
        values = run.result.frames.values(metric)
        if values is None or len(values) == 0:
            return None
        mean = float(np.nanmean(values))
        return None if math.isnan(mean) else mean

    def _resize_mismatch(self, row: int, distorted_info: VideoInfo) -> tuple[str, str]:
        """(short tag, full explanation) for which of the two resolutions got
        resized to match the other, when they differ -- important now that a
        row can exist for either direction (see "Test both"), so it's not
        ambiguous which one a given row/result represents.

        The short tag goes in its own narrow Scaling column and the full
        sentence is the tooltip: spelled out inline it made Media info far
        too wide to scan.

        Uses the *actual* direction a completed run used (what really
        produced its scores, and still right even if the row's settings were
        edited afterward), falling back to the row's current setting before
        it's been run -- UNLESS the row is a "Test both" companion, whose
        direction is fixed and known for certain by construction (see
        RowData.scale_direction_pinned): a result cached before that field
        existed loads as SOURCE_TO_DISTORTED regardless of what actually
        produced it, which is simply wrong for a row that exists only to
        represent DISTORTED_TO_SOURCE.
        """
        source_info = self._source_info
        if source_info is None:
            return "", ""
        if (source_info.width, source_info.height) == (distorted_info.width, distorted_info.height):
            return "", "Source and distorted are the same resolution -- no scaling needed."
        row_data = self._rows[row]
        if row_data.scale_direction_pinned:
            direction = row_data.options.scale_direction
        else:
            direction = (
                row_data.completed_run.result.scale_direction if row_data.completed_run is not None
                else row_data.options.scale_direction
            )
        if direction == ScaleDirection.DISTORTED_TO_SOURCE:
            return "↑ distorted", (
                f"Distorted upscaled {distorted_info.width}x{distorted_info.height} -> "
                f"{source_info.width}x{source_info.height} to match the source."
            )
        return "↓ source", (
            f"Source downscaled {source_info.width}x{source_info.height} -> "
            f"{distorted_info.width}x{distorted_info.height} to match the distorted video."
        )

    def _set_row_info(self, row: int, info: VideoInfo | None, error: str | None = None) -> None:
        item = self.distorted_table.item(row, COL_INFO)
        scaling_item = self.distorted_table.item(row, COL_SCALING)
        if error:
            item.setText("Probe failed")
            item.setToolTip(error)
            item.setForeground(Qt.red)
            self.distorted_table.item(row, COL_BITRATE).setText("")
            scaling_item.setText("")
            scaling_item.setToolTip("")
        else:
            item.setText(media_info_string(info))
            # Back to normal text: the placeholder shown while probing greys
            # this cell out, and leaving it grey makes a probed row look
            # disabled.
            item.setForeground(self.distorted_table.palette().text())
            item.setToolTip(format_hms(info.duration, decimals=1))
            self.distorted_table.item(row, COL_BITRATE).setText(bitrate_string(info))
            tag, explanation = self._resize_mismatch(row, info)
            scaling_item.setText(tag)
            scaling_item.setToolTip(explanation)
            self._rows[row].video_info = info
        # Keeps these snug to whatever's actually in them (never wider than
        # needed) while staying user-draggable in between updates.
        self.distorted_table.resizeColumnToContents(COL_INFO)
        self.distorted_table.resizeColumnToContents(COL_BITRATE)
        self.distorted_table.resizeColumnToContents(COL_SCALING)

    def _set_row_vmaf_text(self, row: int, text: str, *, bold: bool = False, color=None) -> None:
        item = self.distorted_table.item(row, COL_VMAF)
        item.setText(text)
        font = QFont()
        font.setBold(bold)
        item.setFont(font)
        if color is not None:
            item.setForeground(color)
        self.distorted_table.resizeColumnToContents(COL_VMAF)

    def _set_resample_row_info(self, row: int, target: ResampleTarget) -> None:
        info = self._source_info
        assert info is not None
        down_h = max(2, round(target.width * info.height / info.width / 2) * 2)
        item = self.distorted_table.item(row, COL_INFO)
        item.setText(f"Downscale to {target.width}x{down_h}, upscale back to {info.width}x{info.height}")
        item.setToolTip(format_hms(info.duration, decimals=1))
        self.distorted_table.item(row, COL_BITRATE).setText("N/A")
        self.distorted_table.resizeColumnToContents(COL_INFO)
        self.distorted_table.resizeColumnToContents(COL_BITRATE)

    def _on_add_resample_test(self) -> None:
        if self._source_info is None:
            QMessageBox.warning(self, "No source", "Please select a reference (source) video first.")
            return

        targets = [
            target for target in RESAMPLE_TARGET_CHOICES
            if target.width < self._source_info.width
        ]
        if not targets:
            QMessageBox.information(
                self, "No smaller resolution available",
                f"The smallest resolution test is {RESAMPLE_TARGET_CHOICES[-1].width} pixels wide, "
                f"which is not below this source's {self._source_info.width}-pixel width.",
            )
            return
        labels = [t.label for t in targets]
        label, ok = QInputDialog.getItem(
            self, "Add resolution test", "Downscale to (then scale back up):",
            labels, min(1, len(labels) - 1), editable=False,
        )
        if not ok:
            return
        target = next(t for t in targets if t.label == label)

        synthetic_path = synthetic_resample_distorted_path(self._source_info.path, target)
        if any(r.path == synthetic_path for r in self._rows):
            QMessageBox.information(
                self, "Already added", f"A {label} resolution test for this source is already in the list."
            )
            return

        row = self._add_table_row(synthetic_path)
        self._rows[row].options.resample_test = target
        self._rows[row].video_info = self._source_info
        self._set_resample_row_info(row, target)
        self._try_load_cached_result(row)

    def _on_add_distorted(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select distorted video(s)")
        if not paths:
            return
        existing = {r.path for r in self._rows}
        new_paths = [Path(p) for p in paths if Path(p) not in existing]
        if not new_paths:
            return

        # The rows appear immediately; probing each file and loading its
        # cached result happen on a worker, because together they are a
        # couple of seconds for a handful of long videos and used to freeze
        # the window for the whole time.
        for path in new_paths:
            row = self._add_table_row(path)
            self._set_row_status(row, "Reading...")
        self._start_probe(new_paths)

    def _reload_cached_for_all_rows(self) -> None:
        """Re-checks every row against the current source, in the background.

        Called when the source changes: which cached result applies depends
        on the (source, distorted) pair, so every row's score may now be
        different -- or gone.
        """
        for row in range(len(self._rows)):
            self._set_row_metrics(row)
        self._start_probe([r.path for r in self._rows], probe_again=False)

    def _set_row_status(self, row: int, text: str) -> None:
        item = self.distorted_table.item(row, COL_INFO)
        if item is not None:
            item.setText(text)
            item.setForeground(QColor("#999"))

    def _start_probe(self, paths: list[Path], *, probe_again: bool = True) -> None:
        """Probes `paths` in the background, filling their rows as results
        arrive. A probe already running is cancelled first -- the newer
        selection is the one the user is waiting on.

        `probe_again=False` skips re-reading the media info, for when only
        the source changed and the distorted files themselves have not.
        """
        if self._probe_worker is not None and self._probe_worker.isRunning():
            # Keep the old QThread alive until it exits. Replacing the only
            # reference after a fixed two-second wait could destroy a still-
            # running worker and crash Qt on a slow/network file.
            self._probe_worker.cancel()
        self._probe_generation += 1
        generation = self._probe_generation
        source = self._source_info.path if self._source_info else None
        worker = ProbeWorker(
            paths, source, self._settings.use_cache,
            {rd.path: clone_options(rd.options) for rd in self._rows if rd.path in paths},
            probe_media=probe_again,
        )
        self._probe_worker = worker
        self._probe_workers.append(worker)
        worker.probed.connect(
            lambda path, info, error, g=generation:
            self._on_probed_if_current(g, path, info, error)
        )
        worker.cached_found.connect(
            lambda path, result, label, g=generation:
            self._on_cached_if_current(g, path, result, label)
        )
        worker.finished_all.connect(
            lambda g=generation, w=worker: self._on_probe_finished(g, w)
        )
        self.status_label.setText(f"Reading {len(paths)} video(s)...")
        worker.start()

    def _on_probed_if_current(self, generation: int, path: Path, info, error: str) -> None:
        if generation == self._probe_generation:
            self._on_probed(path, info, error)

    def _on_cached_if_current(self, generation: int, path: Path, result, label: str) -> None:
        if generation == self._probe_generation:
            self._on_cached_found(path, result, label)

    def _on_probed(self, path: Path, info, error: str) -> None:
        row = self._row_index_of_path(path)
        if row is None:
            return  # removed while the probe was in flight
        if info is None:
            self._set_row_info(row, None, error=error)
        else:
            self._set_row_info(row, info)

    def _on_cached_found(self, path: Path, result, label: str) -> None:
        row = self._row_index_of_path(path)
        if row is None:
            return
        row_data = self._rows[row]
        if row_data.completed_run is not None:
            return
        run = CompletedRun(result, label)
        row_data.completed_run = run
        row_data.video_info = result.distorted_info
        self._set_row_info(row, result.distorted_info)
        self._set_row_metrics(row)
        # Keep the graph in step as results land, so opening the tab shows
        # everything without any further action.
        self.graph_panel.add_run(
            result, label, identity=run.graph_identity
        )

    def _on_probe_finished(
        self, generation: int | None = None, worker: ProbeWorker | None = None
    ) -> None:
        if worker is not None:
            if worker in self._probe_workers:
                self._probe_workers.remove(worker)
            worker.deleteLater()
        if generation is None or generation == self._probe_generation:
            self.status_label.setText("Ready.")
            self._on_table_selection_changed()

    def _row_index_of_path(self, path: Path) -> int | None:
        for i, row in enumerate(self._rows):
            if row.path == path:
                return i
        return None

    def _on_remove_distorted(self) -> None:
        rows = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()}, reverse=True)
        for row in rows:
            # The graph goes with it: a curve whose row is gone can no longer
            # be removed from anywhere.
            completed = self._rows[row].completed_run
            if completed is not None:
                self.graph_panel.remove_by_identity(completed.graph_identity)
            self.distorted_table.removeRow(row)
            del self._rows[row]
        self._on_table_selection_changed()

    # ------------------------------------------------------------------ persistent result cache
    def _on_remove_all_distorted(self) -> None:
        if not self._rows:
            return
        # Confirmed, because it can discard a lot of completed work from the
        # table at once. The cached results themselves are untouched, so
        # re-adding a video brings its scores straight back.
        answer = QMessageBox.question(
            self, "Remove all videos",
            f"Remove all {len(self._rows)} video(s) from the list?\n\n"
            "Saved results are kept -- re-adding a video shows its scores again.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        for row_data in self._rows:
            if row_data.completed_run is not None:
                self.graph_panel.remove_by_identity(row_data.completed_run.graph_identity)
        self.distorted_table.setRowCount(0)
        self._rows.clear()
        self._on_table_selection_changed()

    def _try_load_cached_result(self, row: int) -> bool:
        """If a previous run for this exact (source, distorted) file name+size
        pair was cached to disk, loads it into the row instead of leaving it
        to be recomputed. Returns whether a cached result was applied."""
        if self._source_info is None:
            return False
        row_data = self._rows[row]
        if row_data.completed_run is not None:
            return False
        cached = result_cache.load_cached(
            self._source_info.path, row_data.path, row_data.options
        )
        if cached is None:
            return False
        result, label = cached
        run = CompletedRun(result, label)
        row_data.completed_run = run
        row_data.video_info = result.distorted_info
        if row_data.options.resample_test is not None:
            self._set_resample_row_info(row, row_data.options.resample_test)
        else:
            self._set_row_info(row, result.distorted_info)
        self._set_row_metrics(row)
        self.distorted_table.item(row, COL_VMAF).setToolTip(
            f"mean {run.stats.mean:.2f}   min {run.stats.minimum:.2f}   max {run.stats.maximum:.2f}   "
            f"({run.stats.count} frames)\nLoaded from a previous run (same filename+size) -- "
            f"right-click to recompute."
        )
        return True

    def _on_table_context_menu(self, pos) -> None:
        rows = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)
        recompute_action = menu.addAction("Recompute VMAF (ignore cached/previous result)")
        chosen = menu.exec(self.distorted_table.viewport().mapToGlobal(pos))
        if chosen == recompute_action:
            self._recompute_rows(rows)

    def _recompute_rows(self, rows: list[int]) -> None:
        for row in rows:
            row_data = self._rows[row]
            row_data.completed_run = None
            self._set_row_metrics(row)
            self.distorted_table.item(row, COL_VMAF).setToolTip("")
            if self._source_info is not None:
                result_cache.clear(
                    self._source_info.path, row_data.path, row_data.options
                )
            # Refreshes the resize-mismatch note (Info column) back to the
            # row's *current* settings -- without this it kept showing
            # whatever the just-cleared run had actually used until the next
            # run finished, which is stale/misleading in between.
            if row_data.video_info is not None and row_data.options.resample_test is None:
                self._set_row_info(row, row_data.video_info)
        self.status_label.setText(
            f"Cleared {len(rows)} result(s) -- make sure they're checked, then click Run VMAF to recompute."
        )

    def _add_opposite_scale_direction_rows(self, rows: list[int]) -> None:
        """For each selected row with a mismatched resolution, adds a second
        row for the *same* distorted file with the opposite ScaleDirection,
        so both "scale source down" and "scale distorted up" can be run and
        compared side by side instead of having to pick one and re-run to
        see the other.
        """
        if self._source_info is None:
            QMessageBox.warning(self, "No source", "Please select a reference (source) video first.")
            return

        added = 0
        for row in rows:
            row_data = self._rows[row]
            if row_data.options.resample_test is not None:
                continue  # scale direction doesn't apply to a resolution round-trip test row
            if row_data.scale_direction_pinned:
                # Already a companion row -- the row it was created from is
                # its opposite direction, so the pair is complete. Without
                # this, "Test both" on a companion produced a redundant
                # third row named "x [upscale-...] [downscale-...]".
                continue
            info = row_data.video_info
            if info is None or (info.width, info.height) == (self._source_info.width, self._source_info.height):
                continue  # no probed info yet, or resolutions already match -- both directions are equivalent

            opposite = (
                ScaleDirection.DISTORTED_TO_SOURCE
                if row_data.options.scale_direction == ScaleDirection.SOURCE_TO_DISTORTED
                else ScaleDirection.SOURCE_TO_DISTORTED
            )
            synthetic_path = synthetic_scale_direction_variant_path(row_data.path, opposite)
            if any(r.path == synthetic_path for r in self._rows):
                continue  # already added

            new_row = self._add_table_row(synthetic_path)
            new_row_data = self._rows[new_row]
            new_row_data.options = clone_options(row_data.options)
            new_row_data.options.scale_direction = opposite
            new_row_data.scale_direction_pinned = True
            new_row_data.video_info = info
            self._set_row_info(new_row, info)
            self._try_load_cached_result(new_row)
            added += 1

        if added == 0:
            QMessageBox.information(
                self, "Nothing to add",
                "Selected row(s) either already have a matching-opposite row, don't have a resolution "
                "mismatch against the source, or aren't a normal comparison (e.g. a resolution test)."
            )
        else:
            self.status_label.setText(
                f"Added {added} row(s) testing the opposite scaling direction -- check them and click Run VMAF."
            )

    # ------------------------------------------------------------------ per-video settings panel
    def _on_table_selection_changed(self) -> None:
        rows = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()})
        self._panel_target_rows = rows

        if not rows:
            self.options_box.setEnabled(False)
            self.panel_target_label.setText(
                "Select one or more videos in the table to view or edit their settings."
            )
            return

        self.options_box.setEnabled(True)
        if len(rows) == 1:
            self.panel_target_label.setText(f"Settings for: {self._rows[rows[0]].path.name}")
        else:
            self.panel_target_label.setText(
                f"Settings for {len(rows)} selected videos -- editing anything below applies to all of them."
            )
        self._write_panel_options(self._rows[rows[0]].options)

    def _write_panel_options(self, opts: VmafOptions) -> None:
        """Populates the panel widgets from `opts` without triggering the
        write-back handlers that would otherwise fire on every setValue()."""
        self._syncing_panel = True
        try:
            model_index = next(
                (i for i, (_, key) in enumerate(_MODEL_CHOICES) if key == opts.model_choice), 0
            )
            self.model_combo.setCurrentIndex(model_index)
            self._panel_custom_model_path = opts.custom_model_path

            self.gpu_checkbox.setChecked(opts.gpu_decode)
            self.gpu_vendor_combo.setCurrentIndex(_GPU_VENDOR_INDEX.get(opts.gpu_vendor, 0))
            self.gpu_vendor_combo.setEnabled(opts.gpu_decode)

            self.crop_combo.setCurrentIndex(0 if opts.crop_mode == CropMode.AUTO else 1)

            self.threads_spin.setValue(opts.n_threads)
            self.subsample_spin.setValue(opts.n_subsample)
            algo_index = _SCALE_ALGORITHMS.index(opts.scale_algorithm) if opts.scale_algorithm in _SCALE_ALGORITHMS else 0
            self.scale_algo_combo.setCurrentIndex(algo_index)
            self.scale_direction_combo.setCurrentIndex(
                1 if opts.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE else 0
            )

            ms = round(opts.duration_limit * 1000)
            self.duration_edit.setTime(QTime(0, 0, 0, 0).addMSecs(ms))

            # Which metrics to compute lives in the table's column headers
            # now, so reflect this row's options back onto them.
            self.metric_header.set_checked(COL_PSNR, "name=psnr" in opts.extra_features)
            self.metric_header.set_checked(COL_SSIM, "name=float_ssim" in opts.extra_features)
            self.metric_header.set_checked(COL_XPSNR, opts.compute_xpsnr)
        finally:
            self._syncing_panel = False

    def _read_panel_options(self) -> VmafOptions:
        extra_features = []
        if self.metric_header.is_checked(COL_PSNR):
            extra_features.append("name=psnr")
        if self.metric_header.is_checked(COL_SSIM):
            extra_features.append("name=float_ssim")

        vendor = _GPU_VENDOR_BY_INDEX[self.gpu_vendor_combo.currentIndex()] if self.gpu_checkbox.isChecked() else GpuVendor.NONE
        crop_mode = CropMode.AUTO if self.crop_combo.currentIndex() == 0 else CropMode.NONE
        duration_limit = QTime(0, 0, 0, 0).msecsTo(self.duration_edit.time()) / 1000.0
        model_choice = _MODEL_CHOICES[self.model_combo.currentIndex()][1]
        scale_direction = (
            ScaleDirection.DISTORTED_TO_SOURCE if self.scale_direction_combo.currentIndex() == 1
            else ScaleDirection.SOURCE_TO_DISTORTED
        )

        return VmafOptions(
            model="",  # resolved per-job at run time via resolve_model(), once each row's distorted video is known
            model_choice=model_choice,
            custom_model_path=self._panel_custom_model_path,
            extra_features=extra_features,
            n_threads=self.threads_spin.value(),
            n_subsample=self.subsample_spin.value(),
            scale_algorithm=self.scale_algo_combo.currentText(),
            scale_direction=scale_direction,
            compute_xpsnr=self.metric_header.is_checked(COL_XPSNR),
            duration_limit=duration_limit,
            gpu_decode=self.gpu_checkbox.isChecked(),
            gpu_vendor=vendor,
            crop_mode=crop_mode,
        )

    def _on_metric_column_toggled(self, column: int, checked: bool) -> None:
        """A metric's column header is a global switch: it applies to every
        row (and becomes the default for rows added later), matching how
        FFMetrics treats its metric columns. Rows that already have a score
        keep it -- unticking just means "don't compute this next run"."""
        if self._syncing_panel:
            return
        feature = next(f for c, _, f in _METRIC_COLUMNS if c == column)
        # Every existing row, plus the template new rows are cloned from --
        # only the metric flag is touched, so each row keeps its own model,
        # crop, GPU and scaling settings.
        for row, rd in enumerate(self._rows):
            opts = rd.options
            before = clone_options(opts)
            if column == COL_XPSNR:
                opts.compute_xpsnr = checked
            elif checked and feature not in opts.extra_features:
                opts.extra_features.append(feature)
            elif not checked and feature in opts.extra_features:
                opts.extra_features.remove(feature)
            if opts != before:
                self._invalidate_completed_result(row)
        opts = self._default_options
        if column == COL_XPSNR:
            opts.compute_xpsnr = checked
        elif checked and feature not in opts.extra_features:
            opts.extra_features.append(feature)
        elif not checked and feature in opts.extra_features:
            opts.extra_features.remove(feature)
        for row in range(len(self._rows)):
            self._set_row_metrics(row)

    def _invalidate_completed_result(self, row: int) -> None:
        """Marks a row stale after an option that affects its run changes."""
        row_data = self._rows[row]
        if row_data.completed_run is None:
            return
        graph_identity = row_data.completed_run.graph_identity
        row_data.completed_run = None
        if not self.graph_panel.remove_by_identity(graph_identity):
            # Backward-compatible fallback for a series added directly by
            # path before row-scoped graph identities existed.
            self.graph_panel.remove_by_path(row_data.path)
        self._set_row_metrics(row)
        self.distorted_table.item(row, COL_VMAF).setToolTip("")

    def _on_panel_edited(self, *_args) -> None:
        """Replace all options (kept for programmatic callers/tests).

        Widget signals use _on_panel_field_edited so editing one control in
        a mixed multi-row selection cannot copy unrelated values from the
        first selected row over all the others.
        """
        if self._syncing_panel:
            return
        new_options = self._read_panel_options()
        self._default_options = clone_options(new_options)
        for row in self._panel_target_rows:
            if self._rows[row].options != new_options:
                self._invalidate_completed_result(row)
            self._rows[row].options = clone_options(new_options)

    def _on_panel_field_edited(self, field_name: str) -> None:
        if self._syncing_panel:
            return
        panel = self._read_panel_options()
        fields = {
            "gpu": ("gpu_decode", "gpu_vendor"),
            "model": ("model", "model_choice", "custom_model_path"),
        }.get(field_name, (field_name,))

        def apply(target: VmafOptions) -> bool:
            changed = False
            for name in fields:
                value = getattr(panel, name)
                if getattr(target, name) != value:
                    setattr(target, name, value)
                    changed = True
            return changed

        apply(self._default_options)
        for row in self._panel_target_rows:
            if apply(self._rows[row].options):
                self._invalidate_completed_result(row)

    def _on_scale_direction_combo_changed(self, index: int) -> None:
        if self._syncing_panel:
            return
        if index == 2:
            # "Test both" isn't a real per-row value (ScaleDirection only has
            # the two actual states) -- it's a one-shot action that adds a
            # sibling row for the other direction, leaving this row's own
            # direction untouched. Revert the combo to reflect that.
            self._add_opposite_scale_direction_rows(self._panel_target_rows)
            if self._panel_target_rows:
                self._write_panel_options(self._rows[self._panel_target_rows[0]].options)
            return
        self._on_panel_field_edited("scale_direction")

    def _on_model_changed(self, index: int) -> None:
        if self._syncing_panel:
            return
        if _MODEL_CHOICES[index][1] == CUSTOM_MODEL_CHOICE:
            path, _ = QFileDialog.getOpenFileName(self, "Select VMAF model file (.json)")
            if path:
                self._panel_custom_model_path = path
            else:
                self.model_combo.setCurrentIndex(0)  # reverts to Auto; re-enters this handler, then falls through below
                return
        self._on_panel_field_edited("model")

    # ------------------------------------------------------------------ run
    def _checked_rows(self) -> list[int]:
        rows = []
        for row in range(self.distorted_table.rowCount()):
            if self.distorted_table.item(row, COL_CHECK).checkState() == Qt.Checked:
                rows.append(row)
        return rows

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._source_info is None:
            QMessageBox.warning(self, "No source", "Please select a reference (source) video.")
            return
        checked_rows = self._checked_rows()
        if not checked_rows:
            QMessageBox.warning(self, "No distorted videos", "Check at least one distorted video to run.")
            return

        # Checking a row you already have a score for (e.g. it was checked
        # before you added more files) shouldn't silently recompute it --
        # skip anything already scored. To force a redo, remove and re-add
        # the row (or uncheck/recheck won't do it -- that's intentional).
        already_scored_rows = [r for r in checked_rows if self._rows[r].completed_run is not None]
        rows_to_run = [r for r in checked_rows if self._rows[r].completed_run is None]

        if not rows_to_run:
            self.status_label.setText(
                f"All {len(already_scored_rows)} checked video(s) already have a VMAF score -- nothing to run."
            )
            already_done_runs = [self._rows[r].completed_run for r in already_scored_rows]
            if already_done_runs:
                self._open_or_update_graph(already_done_runs)
            return

        jobs = []
        job_rows = []
        job_total_frames = []
        for row in rows_to_run:
            row_data = self._rows[row]
            dist_info = row_data.video_info
            if dist_info is None:
                try:
                    dist_info = probe_video(row_data.path)
                    row_data.video_info = dist_info
                    self._set_row_info(row, dist_info)
                except ProbeError as e:
                    self._set_row_info(row, None, error=str(e))
                    continue
            try:
                if row_data.options.resample_test is None:
                    validate_video_pair(self._source_info, dist_info, row_data.options)
                model = resolve_model(row_data.options, dist_info)
            except (ValueError, VmafRunError) as e:
                QMessageBox.warning(self, "Invalid options", f"{row_data.path.name}: {e}")
                return
            job_options = replace(row_data.options, model=model)
            jobs.append(VmafJob(
                self._source_info, dist_info, job_options, label=row_data.path.stem,
                result_distorted_path=row_data.path,
            ))
            job_rows.append(row_data)
            # A resample test's output timeline is driven by the source (see
            # run_resample_test), not this row's (synthetic) "distorted" info.
            reference_for_frames = self._source_info if job_options.resample_test is not None else dist_info
            job_total_frames.append(estimate_total_frames(reference_for_frames, job_options))

        if not jobs:
            return

        self._job_rows = job_rows
        self._job_total_frames = job_total_frames
        self._job_cache_options = [clone_options(rd.options) for rd in job_rows]
        # Held as RowData, not indices, so removing a row mid-run can't
        # silently repoint these at a different row.
        self._checked_rows_for_run = [self._rows[r] for r in checked_rows]
        self._current_job_index = None
        self._run_failed_count = 0
        self._run_was_cancelled = False
        if already_scored_rows:
            self.status_label.setText(
                f"Skipping {len(already_scored_rows)} already-scored video(s); running {len(jobs)}..."
            )
        self._set_run_ui_active(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self._worker = VmafWorker(jobs, self)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.progress.connect(self._on_job_progress)
        self._worker.status.connect(self._on_job_status)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.job_failed.connect(self._on_job_failed)
        self._worker.cancelled.connect(self._on_run_cancelled)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _set_run_ui_active(self, active: bool) -> None:
        """Freezes every input that can change the meaning of a live job."""
        self.files_box.setEnabled(not active)
        self.options_box.setEnabled(not active and bool(self._panel_target_rows))
        self.tabs.setTabEnabled(TAB_SETTINGS, not active)
        self.run_btn.setEnabled(not active)
        self.pause_btn.setEnabled(active)
        self.cancel_btn.setEnabled(active)

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_pause_clicked(self) -> None:
        if self._worker is None:
            return
        if self.pause_btn.isChecked():
            self._worker.pause()
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused.")
        else:
            self._worker.resume()
            self.pause_btn.setText("Pause")
            self.status_label.setText("Resumed.")

    def _on_job_started(self, index: int, label: str) -> None:
        self._current_job_index = index
        self.status_label.setText(f"Running: {label}  (file {index + 1} of {len(self._job_rows)})")
        self.progress_bar.setValue(0)
        self.progress_detail_label.setText("")

    def _on_job_progress(self, index: int, current: int, total: int, fps: float) -> None:
        # Live progress (queued/starting/frame N of M/paused) belongs in the
        # status bar above -- it already shows the running file, fps and ETA
        # -- not the VMAF column, which is for the final score.
        pct = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(min(pct, 100))

        file_count = len(self._job_rows)
        if fps > 0:
            file_eta = format_hms(max(0, total - current) / fps)
            frames_done_before_this_job = sum(self._job_total_frames[:index])
            queue_total = sum(self._job_total_frames)
            queue_remaining = max(0, queue_total - (frames_done_before_this_job + current))
            queue_eta = format_hms(queue_remaining / fps)
            self.progress_detail_label.setText(
                f"{fps:.1f} fps   |   File ETA: {file_eta}   |   Queue ETA: {queue_eta}"
                f"   (file {index + 1} of {file_count})"
            )
        else:
            self.progress_detail_label.setText(f"file {index + 1} of {file_count}")

    def _on_job_status(self, index: int, message: str) -> None:
        self.status_label.setText(message)

    def _row_index_of(self, row_data: RowData) -> int | None:
        """The table row this RowData currently sits at, or None if it was
        removed while the run was in flight. Compared by identity, not `==`:
        two different rows can be field-equal (e.g. the same file added
        twice before probing), and list.index() would find the wrong one."""
        for i, rd in enumerate(self._rows):
            if rd is row_data:
                return i
        return None

    def _on_job_finished(self, index: int, result) -> None:
        row_data = self._job_rows[index]
        row = self._row_index_of(row_data)
        if row is None:
            return  # the row was removed mid-run; nothing to write the result to
        label = row_data.path.stem
        # The job owns the source/distorted identities it was launched with.
        # Never key a result from an old in-flight job using whatever source
        # happens to be selected by the time it finishes.
        cache_options = (
            self._job_cache_options[index]
            if index < len(self._job_cache_options) else row_data.options
        )
        result_cache.store(
            result.source, result.distorted, result, label, cache_options
        )
        if self._source_info is None or self._source_info.path != result.source:
            self._set_row_status(row, "Finished for the previous source; select it again to load the result.")
            return
        run = CompletedRun(result, label)
        row_data.completed_run = run
        if row_data.options.resample_test is None:
            self._set_row_info(row, result.distorted_info)  # refresh the resize-mismatch note against the actual run
        self._set_row_metrics(row)
        self.distorted_table.item(row, COL_VMAF).setToolTip(
            f"mean {run.stats.mean:.2f}   min {run.stats.minimum:.2f}   max {run.stats.maximum:.2f}   "
            f"({run.stats.count} frames)"
        )
        # Straight onto the graph: a run that has finished is a curve, and
        # waiting for a button press to see it serves nobody.
        self.graph_panel.add_run(
            result, label, identity=run.graph_identity
        )

    def _on_job_failed(self, index: int, message: str, stderr_tail: str) -> None:
        self._run_failed_count += 1
        row = self._row_index_of(self._job_rows[index])
        if row is None:
            return  # the row was removed mid-run
        self._set_row_vmaf_text(row, "Failed", color=Qt.red)
        detail = f"{message}\n\n{stderr_tail}" if stderr_tail else message
        self.distorted_table.item(row, COL_VMAF).setToolTip(detail)

    def _on_run_cancelled(self) -> None:
        self._run_was_cancelled = True

    def _on_all_finished(self) -> None:
        self._set_run_ui_active(False)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self._current_job_index = None
        if self._run_was_cancelled:
            self.status_label.setText("Cancelled.")
        elif self._run_failed_count:
            self.status_label.setText(
                f"Finished with {self._run_failed_count} failed video(s)."
            )
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText("Done.")
            self.progress_bar.setValue(100)
        self.progress_detail_label.setText("")
        # Includes rows that were already scored and skipped, not just ones
        # run this batch, so the comparison graph reflects everything checked.
        # Rows removed mid-run are skipped rather than indexed into.
        finished_runs = [
            rd.completed_run for rd in self._checked_rows_for_run
            if rd.completed_run and self._row_index_of(rd) is not None
        ]
        if finished_runs:
            self._open_or_update_graph(finished_runs)

    # ------------------------------------------------------------------ results actions
    def _selected_runs(self) -> list[CompletedRun]:
        rows = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()})
        return [self._rows[r].completed_run for r in rows if self._rows[r].completed_run]

    def _on_load_saved_run(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load saved VMAF run", "", "VMAF run (*.vmafrun.json *.json)")
        if not path:
            return
        try:
            result, label = load_run(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Failed to load run", str(e))
            return
        run = CompletedRun(result, label)
        row = self._add_table_row(result.distorted)
        row_data = self._rows[row]
        row_data.video_info = result.distorted_info
        row_data.completed_run = run
        # The optional-metric columns are driven by row options. Seed those
        # flags from the data that is actually present in the saved result,
        # otherwise valid PSNR/SSIM/XPSNR arrays render as "N/A".
        row_data.options.extra_features = []
        if result.frames.has("psnr"):
            row_data.options.extra_features.append("name=psnr")
        if result.frames.has("ssim"):
            row_data.options.extra_features.append("name=float_ssim")
        row_data.options.compute_xpsnr = result.frames.has("xpsnr")
        row_data.options.model = result.model
        row_data.options.scale_direction = result.scale_direction
        self.distorted_table.item(row, COL_CHECK).setCheckState(Qt.Unchecked)
        self._set_row_info(row, result.distorted_info)
        self._set_row_metrics(row)
        self.distorted_table.item(row, COL_VMAF).setToolTip("Loaded from saved run")

    def _on_save_selected(self) -> None:
        runs = self._selected_runs()
        if not runs:
            QMessageBox.information(self, "Nothing selected", "Select one or more completed rows to save.")
            return
        if len(runs) == 1:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save VMAF run", f"{runs[0].label}.vmafrun.json", "VMAF run (*.vmafrun.json)"
            )
            if path:
                save_run(runs[0].result, Path(path), label=runs[0].label)
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose folder to save runs into")
        if not directory:
            return
        reserved: set[Path] = set()
        for run in runs:
            save_run(
                run.result,
                unique_output_path(Path(directory), run.label, ".vmafrun.json", reserved),
                label=run.label,
            )

    def _on_compare_selected(self) -> None:
        runs = self._selected_runs()
        if not runs:
            QMessageBox.information(self, "Nothing selected", "Select one or more completed rows to compare.")
            return
        self._open_or_update_graph(runs)

    def _on_show_graph_clicked(self) -> None:
        # Syncs in every currently-completed row every time -- not just
        # whatever was completed the first time this was clicked -- so
        # checking back mid-run (e.g. 4 of 8 done) shows all 4, not just
        # however many were done the first time it was opened.
        all_runs = [r.completed_run for r in self._rows if r.completed_run]
        if not all_runs and not self.graph_panel._entries:
            QMessageBox.information(
                self, "No results yet", "Run or load at least one VMAF result before opening the graph."
            )
            return
        self._open_or_update_graph(all_runs)

    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_GRAPH:
            self._sync_graph()

    def _sync_graph(self) -> None:
        """Makes the graph show every completed row.

        add_run replaces a series with the same distorted path rather than
        stacking a duplicate, so this is safe to call as often as it likes --
        on every tab switch, and whenever a row gains a result.
        """
        for row in self._rows:
            if row.completed_run is not None:
                self.graph_panel.add_run(
                    row.completed_run.result, row.completed_run.label,
                    identity=row.completed_run.graph_identity, restore=False,
                )

    def _open_or_update_graph(self, runs: list[CompletedRun]) -> None:
        """Adds runs to the graph tab and brings it to the front."""
        for run in runs:
            self.graph_panel.add_run(
                run.result, run.label, identity=run.graph_identity
            )
        self.tabs.setCurrentIndex(TAB_GRAPH)
