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

import contextlib
import copy
import os
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTime, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from vmaf_app import APP_NAME, __version__
from vmaf_app.core import perceptual_vship, result_cache
from vmaf_app.core.builtin_models import builtin_choice
from vmaf_app.core.cvvdp import (
    DEFAULT_PRESET,
    CvvdpDisplay,
    CvvdpSettings,
    default_settings,
    matching_preset,
    preset_named,
    with_user_preset,
    without_user_preset,
)
from vmaf_app.core.cvvdp import presets as cvvdp_presets
from vmaf_app.core.ffmpeg_locate import check_tools, exe_name, format_version, set_ffmpeg_dir_override
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    displayable_metric_specs,
)
from vmaf_app.core.frame_extract import FrameComparison
from vmaf_app.core.gpu import detected_gpu_vendors
from vmaf_app.core.metric_results import MetricResultSet, frame_scores_from_results
from vmaf_app.core.metrics import FRAME_METRICS, METRICS, MetricDefinition, MetricKind, metric_definition
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE, resolve_model
from vmaf_app.core.models import (
    RESAMPLE_TARGET_CHOICES,
    CropBox,
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
from vmaf_app.core.perceptual_cpu import LONG_CPU_RUN_SECONDS
from vmaf_app.core.run_io import RESULT_FILE_FILTER, RESULT_SUFFIX, load_run, save_run, unique_output_path
from vmaf_app.core.settings import Settings
from vmaf_app.core.stats import aggregate_scores
from vmaf_app.core.time_format import format_hms
from vmaf_app.core.vmaf_runner import (
    VmafRunError,
    analysis_dimensions,
    estimate_total_frames,
    resample_analysis_dimensions,
    validate_video_pair,
)
from vmaf_app.ui.bitrate_panel import BitratePanel
from vmaf_app.ui.file_worker import FileWriteQueue
from vmaf_app.ui.formatting import bitrate_string, media_info_string
from vmaf_app.ui.frame_compare_panel import FrameComparePanel, FrameComparisonEntry
from vmaf_app.ui.graph_panel import GraphPanel
from vmaf_app.ui.probe_worker import ProbeWorker
from vmaf_app.ui.widgets import CheckableHeaderView, FillColumnTable
from vmaf_app.ui.worker import MAX_PARALLEL_JOBS, VmafJob, VmafWorker

_MODEL_CHOICES = [
    ("Auto (analysis resolution, VMAF v0.6.1)", AUTO_MODEL_CHOICE),
    ("VMAF v0.6.1 (default, standard viewing)", "version=vmaf_v0.6.1"),
    ("VMAF 4K v0.6.1 (4K / large-screen viewing)", "version=vmaf_4k_v0.6.1"),
    ("VMAF v1 (1080p / 3H)", builtin_choice("vmaf_v1_3d0h")),
    ("VMAF v1 (1080p phone / 5H)", builtin_choice("vmaf_v1_5d0h")),
    ("VMAF v1 (4K / 1.5H)", builtin_choice("vmaf_v1_1d5h_2160")),
    ("VMAF v1 (4K / 3H, up to 110)", builtin_choice("vmaf_v1_3d0h_2160")),
    ("VMAF v1 HFR (1080p / 3H)", builtin_choice("vmaf_v1_hfr_3d0h")),
    ("VMAF v1 HFR (1080p phone / 5H)", builtin_choice("vmaf_v1_hfr_5d0h")),
    ("VMAF v1 HFR (4K / 1.5H)", builtin_choice("vmaf_v1_hfr_1d5h_2160")),
    ("VMAF v1 HFR (4K / 3H, up to 110)", builtin_choice("vmaf_v1_hfr_3d0h_2160")),
    ("Custom model file...", CUSTOM_MODEL_CHOICE),
]

_SCALE_ALGORITHMS = ["bicubic", "lanczos", "bilinear", "spline"]

_GPU_VENDOR_BY_INDEX = {0: GpuVendor.AUTO, 1: GpuVendor.NVIDIA, 2: GpuVendor.INTEL, 3: GpuVendor.AMD}
_GPU_VENDOR_INDEX = {v: k for k, v in _GPU_VENDOR_BY_INDEX.items()}

(
    COL_CHECK,
    COL_PATH,
    COL_INFO,
    COL_BLACK_BARS,
    COL_SCALING,
    COL_BITRATE,
    COL_PSNR,
    COL_SSIM,
    COL_VMAF,
    COL_XPSNR,
    COL_VMAF_NEG,
    COL_SSIMULACRA2,
    COL_BUTTERAUGLI,
    COL_CVVDP,
) = range(14)

#: Row states worth colouring the file name for. Everything else the old
#: Status column reported is now visible in the metric columns themselves --
#: an unticked box is "not selected", a ticked empty one is "not calculated
#: yet", a score is "done" -- so only these two, which nothing else shows,
#: need a mark of their own.
_STATE_COLOURS = {
    "failed": "#a03030",
    "stale": "#8a6d00",
}

# Metric Graphs displays results; Video Compare and Bitrate Viewer also work
# independently of calculation. Settings remains the final page.
TAB_VIDEOS, TAB_GRAPH, TAB_FRAME_COMPARE, TAB_BITRATE, TAB_SETTINGS = range(5)

@dataclass(frozen=True)
class MetricColumn:
    """A stable physical table column bound to one registry metric."""
    column: int
    key: str

    @property
    def metric(self) -> MetricDefinition:
        return metric_definition(self.key)


# Keep these physical indices and visual order exactly as the established UI.
_METRIC_COLUMNS = (
    MetricColumn(COL_VMAF, "vmaf"), MetricColumn(COL_VMAF_NEG, "vmaf_neg"),
    MetricColumn(COL_PSNR, "psnr"), MetricColumn(COL_SSIM, "ssim"),
    MetricColumn(COL_XPSNR, "xpsnr"),
    MetricColumn(COL_SSIMULACRA2, "ssimulacra2"), MetricColumn(COL_BUTTERAUGLI, "butteraugli"),
    MetricColumn(COL_CVVDP, "cvvdp"),
)
_METRIC_COLUMN_BY_INDEX = {item.column: item for item in _METRIC_COLUMNS}
_METRIC_COLUMN_SET = frozenset(_METRIC_COLUMN_BY_INDEX)

class _StayOpenMenu(QMenu):
    """A menu of check boxes that stays open while boxes are ticked, so
    several metrics can be switched in one visit."""

    def mouseReleaseEvent(self, event) -> None:
        action = self.actionAt(event.position().toPoint())
        if action is not None and action.isCheckable() and action.isEnabled():
            action.trigger()
            return
        super().mouseReleaseEvent(event)


#: The largest display resolution the CVVDP display editor accepts (8K).
_CVVDP_MAX_DISPLAY_PIXELS = 8192


class CvvdpDisplayDialog(QDialog):
    """Edits the display CVVDP models, and saves it as a preset. Every value
    changes the score.

    `own_preset` is the name of the user's preset being edited, if the
    display is one: its name can then be changed ("Save preset" renames and
    updates it). `new_preset` is the "Add preset..." form, which can only
    save a new preset. `taken_names` are every preset name in use, which a
    new name may not repeat.

    After exec(), `action` says which button closed it: "save" (update or
    rename `own_preset`), "save_new", or "apply" (use without saving).
    """

    def __init__(self, display: CvvdpDisplay, parent=None, *, own_preset: str | None = None,
                 new_preset: bool = False, taken_names: frozenset[str] = frozenset()):
        super().__init__(parent)
        self.setWindowTitle("New CVVDP preset" if new_preset else "CVVDP display")
        self.action = ""
        self._original = display
        self._own_preset = own_preset
        self._taken_names = taken_names
        form = QFormLayout(self)
        intro = QLabel(
            "CVVDP predicts how visible the differences are to someone watching this "
            "display, from this distance, in this light. Each value below changes the score, "
            "so compare videos scored for the same display."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        self.name_edit = QLineEdit(own_preset or "")
        self.name_edit.setPlaceholderText(
            "Name for the new preset" if new_preset or own_preset is None else ""
        )
        self.name_edit.setToolTip(
            "Change it and click Save preset to rename this preset." if own_preset
            else "Type a name and click Save as new preset to keep this display as a preset."
        )
        form.addRow("Preset name:", self.name_edit)

        def spin(low, high, value, decimals, suffix, step, tip):
            box = QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setSuffix(suffix)
            box.setValue(value)
            box.setToolTip(tip)
            return box

        self.width_spin, self.height_spin = QSpinBox(), QSpinBox()
        for box, value in ((self.width_spin, display.width), (self.height_spin, display.height)):
            # Up to 8K. With "Scale the video to fill the display", Vship
            # works at the display's resolution: an 8K display took CVVDP
            # alone to +7.3 GB of VRAM on a 4K video, a 16384x16384 one to
            # +15.9 GB -- a display that does not exist, costing more than
            # most GPUs have.
            box.setRange(16, _CVVDP_MAX_DISPLAY_PIXELS)
            box.setValue(value)
            box.setToolTip("The display's own resolution in pixels (not the video's).")
        resolution = QHBoxLayout()
        resolution.addWidget(self.width_spin)
        resolution.addWidget(QLabel("x"))
        resolution.addWidget(self.height_spin)
        resolution.addStretch(1)
        form.addRow("Resolution:", resolution)
        self.diagonal_spin = spin(1, 1000, display.diagonal_inches, 1, " in", 1,
                                  "The screen's diagonal size.")
        form.addRow("Screen size:", self.diagonal_spin)
        self.distance_spin = spin(0.05, 50, display.viewing_distance_m, 4, " m", 0.05,
                                  "How far the viewer's eyes are from the screen. Closer makes "
                                  "small artifacts easier to see.")
        self.distance_note = QLabel()
        self.distance_note.setStyleSheet("color: #666;")
        distance = QHBoxLayout()
        distance.addWidget(self.distance_spin)
        distance.addWidget(self.distance_note)
        distance.addStretch(1)
        form.addRow("Viewing distance:", distance)
        self.peak_spin = spin(1, 10000, display.peak_luminance, 0, " cd/m\u00b2", 50,
                              "The display's peak brightness (nits): about 200 for an office "
                              "monitor, 600-1500 for an HDR monitor, 1000-4000 for an HDR TV.")
        form.addRow("Peak brightness:", self.peak_spin)
        self.contrast_spin = spin(1, 10_000_000, display.contrast, 0, " : 1", 100,
                                  "Peak to black: about 1000:1 for a typical LCD, 1,000,000:1 "
                                  "for OLED or the official HDR displays.")
        form.addRow("Contrast:", self.contrast_spin)
        self.ambient_spin = spin(0, 100_000, display.ambient_lux, 1, " lux", 10,
                                 "Light falling on the screen: about 250 lux in an office, 50 in a "
                                 "lit living room, 5-10 watching a film with the lights low, 0 in the dark.")
        form.addRow("Room light:", self.ambient_spin)
        self.reflectivity_spin = spin(0, 99.9, display.reflectivity * 100, 2, " %", 0.1,
                                      "How much of the room light the screen reflects back at the "
                                      "viewer; 0.5% is the official models' value.")
        form.addRow("Screen reflectivity:", self.reflectivity_spin)
        self.exposure_spin = spin(0.01, 100, display.exposure, 2, "", 0.1,
                                  "Brightness multiplier for the pictures; 1 shows them as encoded.")
        form.addRow("Exposure:", self.exposure_spin)
        self.hdr_check = QCheckBox("HDR display")
        self.hdr_check.setChecked(display.hdr)
        self.hdr_check.setToolTip("An HDR display, as in the official HDR display models. Off: an SDR one.")
        form.addRow("", self.hdr_check)
        buttons = QDialogButtonBox()
        self.save_button = None
        if own_preset is not None and not new_preset:
            self.save_button = buttons.addButton("Save preset", QDialogButtonBox.AcceptRole)
            self.save_button.setToolTip(f'Save these values (and the name above) as your preset "{own_preset}".')
            self.save_button.clicked.connect(lambda: self._finish("save"))
        self.save_new_button = buttons.addButton("Save as new preset", QDialogButtonBox.AcceptRole)
        self.save_new_button.setToolTip(
            "Keep this display as a preset of your own under the name above. It becomes "
            "the display newly added videos start with."
        )
        self.save_new_button.clicked.connect(lambda: self._finish("save_new"))
        self.apply_button = None
        if not new_preset:
            self.apply_button = buttons.addButton("Apply without saving", QDialogButtonBox.AcceptRole)
            self.apply_button.setToolTip("Use this display for the selected videos without saving a preset.")
            self.apply_button.clicked.connect(lambda: self._finish("apply"))
        cancel = buttons.addButton(QDialogButtonBox.Cancel)
        cancel.clicked.connect(self.reject)
        form.addRow(buttons)
        for box in (self.width_spin, self.height_spin, self.diagonal_spin, self.distance_spin):
            box.valueChanged.connect(self._update_distance_note)
        self._update_distance_note()

    def display(self) -> CvvdpDisplay:
        """The edited display. A field the user did not change keeps its
        exact original value: the boxes show fewer decimals than a display
        can have, and reading 0.7472 m back as 0.747 made "Apply without
        saving" with nothing changed drop the CVVDP score and turn the
        preset into "Custom"."""
        original = self._original

        def value(box, before: float, scale: float = 1.0) -> float:
            shown = round(before * scale, box.decimals())
            return before if abs(box.value() - shown) < 10 ** -(box.decimals() + 2) else box.value() / scale

        return CvvdpDisplay(
            width=self.width_spin.value(), height=self.height_spin.value(),
            diagonal_inches=value(self.diagonal_spin, original.diagonal_inches),
            viewing_distance_m=value(self.distance_spin, original.viewing_distance_m),
            peak_luminance=value(self.peak_spin, original.peak_luminance),
            contrast=value(self.contrast_spin, original.contrast),
            ambient_lux=value(self.ambient_spin, original.ambient_lux),
            reflectivity=value(self.reflectivity_spin, original.reflectivity, 100),
            exposure=value(self.exposure_spin, original.exposure),
            hdr=self.hdr_check.isChecked(),
        )

    def _update_distance_note(self, *_args) -> None:
        self.distance_note.setText(f"= {self.display().distance_in_heights:.2f} x screen height")

    def preset_name(self) -> str:
        return self.name_edit.text().strip()

    def _name_problem(self, action: str) -> str | None:
        """Why the typed name cannot be saved with `action`, or None."""
        name = self.preset_name()
        if action == "apply":
            return None
        if not name:
            return "Type a name for the preset."
        renaming_own = action == "save" and name == self._own_preset
        if name in self._taken_names and not renaming_own:
            return f'There is already a preset called "{name}". Choose another name.'
        return None

    def _finish(self, action: str) -> None:
        problem = self._name_problem(action)
        if problem is None:
            try:
                self.display().validated()
            except ValueError as error:
                problem = str(error)
        if problem is not None:
            QMessageBox.warning(self, self.windowTitle(), problem)
            return
        self.action = action
        self.accept()


#: The metrics with their own pass and a GPU/CPU choice, not a libvmaf feature.
_PERCEPTUAL_METRIC_KEYS = ("ssimulacra2", "butteraugli")
#: Every metric ticked outside VmafOptions (see RowData.extra_metric_keys).
#: CVVDP runs in the same GPU pass but has no CPU choice.
_EXTRA_METRIC_KEYS = (*_PERCEPTUAL_METRIC_KEYS, "cvvdp")

#: A row whose job finished some metrics and failed others (see
#: MainWindow._on_job_partially_failed).
_PARTLY_FAILED = "Partly failed"

#: Longer than this, CPU SSIMULACRA2/Butteraugli asks for confirmation first.
_CPU_PERCEPTUAL_WARNING_SECONDS = LONG_CPU_RUN_SECONDS
#: CPU tool seconds per megapixel of a compared frame pair, measured with the
#: bundled libjxl 0.12.0 tools on a 3840x1608 (6.17 MP) Beekeeper pair:
#: SSIMULACRA2 1.1 s, Butteraugli 2.0 s. A rough guide -- CPUs differ.
_CPU_PERCEPTUAL_SECONDS_PER_MEGAPIXEL = {"ssimulacra2": 1.1 / 6.17, "butteraugli": 2.0 / 6.17}


def _rough_duration(seconds: float) -> str:
    """"about 3 days", "about 5 hours", "about 40 minutes"."""
    if seconds >= 2 * 86400:
        return f"about {seconds / 86400:.0f} days"
    if seconds >= 2 * 3600:
        return f"about {seconds / 3600:.0f} hours"
    return f"about {max(1, round(seconds / 60))} minutes"


class CompletedRun:
    """A finished run, its display label, and its graph identity.

    Metric-specific statistics belong to the graph; the video table reads
    metric aggregates directly from the result.
    """

    def __init__(self, result, label: str):
        self.result = result
        self.label = label
        self.graph_identity = object()


@dataclass
class RowData:
    path: Path
    # The file actually decoded, when `path` is a synthetic stand-in.
    #
    # A "Test both directions" companion row and a resolution test both carry
    # a well-formed but NON-EXISTENT path, so they read as separate rows and
    # separate graph series. That path is useless for cache identity: it has
    # no size and no modification time, so replacing the real video leaves
    # the key unchanged and a stale score loads for the new content. Cache
    # lookups therefore key on this instead -- the row still keeps its own
    # options, and scale_direction/resample_test already tell the two
    # variants of one file apart.
    media_path: Path | None = None
    video_info: VideoInfo | None = None
    completed_run: CompletedRun | None = None
    options: VmafOptions = field(default_factory=VmafOptions)
    # Non-FFmpeg metrics deliberately live outside VmafOptions. The row keeps
    # one generic selection alongside the existing FFmpeg adapter settings.
    extra_metric_keys: set[str] = field(default_factory=set)
    # Compute preference is execution-only and must not invalidate scores or
    # cache entries. GPU is the default; Vship falls back to CPU if unavailable.
    metric_backends: dict[str, str] = field(default_factory=lambda: {
        "ssimulacra2": "gpu", "butteraugli": "gpu",
    })
    # The display CVVDP models for this row. Part of what a CVVDP score
    # means, so part of its cache identity; no other metric depends on it.
    cvvdp: CvvdpSettings = field(default_factory=CvvdpSettings)
    # The CVVDP preset the row was given, if any: only to tell apart presets
    # with the same display (see matching_preset), never part of a score.
    cvvdp_preset: str = ""
    # CVVDP scores saved for this comparison with other displays, as
    # [(CvvdpSettings, JOD)]: named in the tooltip of an empty CVVDP cell,
    # never shown as the row's score.
    cvvdp_elsewhere: list = field(default_factory=list)
    analysis_status: str = ""
    # What the old Status column's tooltip carried: an ffmpeg error, or how
    # many frames a loaded result holds. Now shown on the file name, which
    # is the only cell that is always present and always about the row as a
    # whole.
    status_detail: str = ""
    # True only for a "Test both" companion row (see
    # _add_opposite_scale_direction_rows), where options.scale_direction is
    # set explicitly and unambiguously to whichever direction this row
    # exists to represent -- unlike a normal row, where it's just whatever
    # the panel's default happened to be when the row was added, and a
    # cached/loaded result's own recorded direction is more trustworthy.
    scale_direction_pinned: bool = False
    # A stable, hashable token for this row in the Frame Compare tab, used
    # before it has a result to be identified by. RowData itself cannot
    # serve: it is a mutable dataclass, so it is unhashable, and the frame
    # cache keys on this.
    frame_identity: object = field(default_factory=object)

    @property
    def identity_path(self) -> Path:
        """The path cache identity is taken from -- the real file when this
        row's own path is a synthetic stand-in."""
        return self.media_path if self.media_path is not None else self.path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        # Settings first: the ffmpeg location and what new rows default to
        # both come from them, so they must be applied before the startup
        # tool check or any row is added.
        self._settings = Settings.load()
        # Serialising a feature-length result is seconds of work; done here
        # it froze the window at the moment a run finished. See FileWriteQueue.
        self._file_writes = FileWriteQueue(self)
        self._file_writes.write_failed.connect(self._on_file_write_failed)
        self._file_writes.became_idle.connect(self._on_file_writes_idle)
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
        # Media probing and cache lookups are separate lanes on purpose.
        # They used to share one worker slot and one generation counter, so
        # starting a cache lookup cancelled whatever media probe was running
        # AND invalidated its results -- while the replacement never probed,
        # because a cache lookup does not read media info. Rows were left
        # permanently on "Reading..." with nothing outstanding to fill them.
        self._cache_worker: ProbeWorker | None = None
        self._cache_lookup_paths: list[Path] = []  # the rows the running lookup was asked about
        self._cache_generation = 0
        self._source_probe_worker: ProbeWorker | None = None
        self._probe_workers: list[ProbeWorker] = []
        self._source_probe_generation = 0
        # These track the active run by RowData *identity*, not by table row
        # index: removing a row mid-run shifts every later index down, which
        # used to make a finishing job write its result to the wrong row --
        # or crash with IndexError when the shifted index ran off the end.
        self._job_rows: list[RowData] = []  # job index -> the row that job belongs to
        self._job_total_frames: list[int] = []  # job index -> estimated frame count, for queue ETA
        self._job_cache_options: list[VmafOptions] = []
        self._job_cvvdp: list[CvvdpSettings] = []
        self._checked_rows_for_run: list[RowData] = []  # rows checked when Run was clicked, incl. already-scored ones
        # Live per-job figures, keyed by job index. With several videos in
        # flight the bar can no longer track "the" running job -- there is
        # more than one -- so it tracks the whole queue instead, which is
        # also the number a user actually wants while waiting.
        self._job_frames_done: dict[int, int] = {}
        self._job_fps: dict[int, float] = {}
        self._job_decode_status: dict[int, str] = {}
        # Each half's progress for a video scored in two halves; see
        # VmafWorker.halves.
        self._job_halves: dict[int, list] = {}
        # Backend-aware task progress.  This is deliberately separate from
        # the legacy combined progress value: CPU and perceptual work have
        # different clocks, and GPU perceptual metrics may be serialized.
        self._job_task_progress: dict[int, list[dict[str, object]]] = {}
        self._job_gpu_fallback: set[int] = set()
        self._job_gpu_metrics: set[int] = set()
        self._running_jobs: list[int] = []
        self._finished_jobs: set[int] = set()
        # job index -> which of the per-video lines it owns. Held for the
        # life of the job so a line never jumps to a different video.
        self._job_line_slot: dict[int, int] = {}
        self._run_failed_count = 0
        self._run_was_cancelled = False
        # Whether a run owns the window's settings right now. Read by
        # every background handler that would otherwise re-enable them.
        self._run_active = False
        self._run_started_at: float | None = None
        self._run_elapsed_timer = QTimer(self)
        self._run_elapsed_timer.setInterval(1000)
        self._run_elapsed_timer.timeout.connect(self._update_run_status)
        self._cache_clear_result: list[int] | None = None
        # Set once the user has asked to close: background work has been
        # told to stop and the window closes itself when it actually has.
        self._closing = False

        # Per-video settings machinery: the Options panel is an inspector for
        # whichever rows are selected, not one global setting.
        self._default_options = self._options_from_settings()  # what a newly-added row starts with
        self._default_extra_metric_keys = self._extra_metrics_from_settings()
        self._default_metric_backends = {
            key: "cpu" if getattr(self._settings, f"default_{key}_backend", "gpu") == "cpu" else "gpu"
            for key in _PERCEPTUAL_METRIC_KEYS
        }
        self._default_cvvdp = self._cvvdp_from_settings()
        known = {item.key for item in _METRIC_COLUMNS}
        self._hidden_metrics: set[str] = {key for key in self._settings.hidden_metrics if key in known}
        if len(self._hidden_metrics) >= len(known):
            self._hidden_metrics.clear()  # a settings file that hides everything shows everything
        self._panel_target_rows: list[int] = []  # rows the panel currently edits
        self._panel_custom_model_path: str | None = None  # staging for the panel's "Custom model" choice
        self._syncing_panel = False  # guards against write-back while populating the panel programmatically
        self._syncing_table = False  # ditto for the table's own metric tick boxes

        self._build_ui()
        self._check_ffmpeg(prompt=True)  # startup check: both tools present, ffmpeg new enough
        self._on_table_selection_changed()

    def _close_when_idle(self) -> None:
        """Retries the close once the work it was waiting on has finished."""
        if self._closing:
            self.close()

    def _live_workers(self) -> list:
        """Every background thread that must finish before the UI it writes
        into can be destroyed."""
        workers = [w for w in self._probe_workers if w.isRunning()]
        if self._worker is not None and self._worker.isRunning():
            workers.append(self._worker)
        workers.extend(self.frame_compare_panel.live_workers())
        workers.extend(self.bitrate_panel.live_workers())
        return workers

    def closeEvent(self, event) -> None:
        # Shutdown is asynchronous rather than a blocking wait. A run owns a
        # live ffmpeg, and a probe owns a live ffprobe that can take many
        # seconds on a large file; the old code waited a flat five seconds
        # per worker on the GUI thread and then closed anyway -- destroying
        # widgets those threads were still posting into, which is a crash and
        # an orphaned subprocess rather than a slow exit.
        if not self._closing:
            self._closing = True
            self.frame_compare_panel.cancel()
            self.bitrate_panel.cancel()
            for worker in self._live_workers():
                worker.cancel()

        pending = self._live_workers()
        # Closing mid-write would lose a cached/saved result or truncate a
        # graph CSV export. The graph owns a separate serial queue, so both
        # must drain too.
        writes_finished = self._file_writes.wait_until_idle(0.5)
        graph_writes_finished = self.graph_panel.wait_until_file_writes_idle(0.5)
        if pending or not writes_finished or not graph_writes_finished:
            self.status_label.setText(
                "Finishing up; the window will close on its own."
            )
            # Re-check shortly. Each cancelled worker also calls back here as
            # it finishes, so this timer is only a backstop for the write
            # queues, which have no completion signal of their own.
            QTimer.singleShot(200, self._close_when_idle)
            event.ignore()
            return
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
        # is no second taskbar entry to manage. Video Compare and the
        # independent Bitrate Viewer sit between Metric Graphs and Settings.
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
        splitter.setSizes([380, 300, 120])
        self.tabs.addTab(videos_page, "Videos")

        self.graph_panel = GraphPanel()
        self.graph_panel.set_preferred_metric(self._settings.graph_metric)
        self.graph_panel.metric_changed.connect(self._on_graph_metric_changed)
        self.tabs.addTab(self.graph_panel, "Metric Graphs")
        # Opening the tab is enough; pressing a button to populate it was
        # a leftover from when it was a separate window that had to be
        # opened explicitly.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.frame_compare_panel = FrameComparePanel(
            color_mode=self._settings.frame_preview_color_mode,
            decoded_videos=self._settings.compare_decoded_videos,
        )
        self.frame_compare_panel.color_mode_changed.connect(
            self._on_frame_color_mode_changed
        )
        self.tabs.addTab(self.frame_compare_panel, "Video Compare")

        self.bitrate_panel = BitratePanel()
        self.tabs.addTab(self.bitrate_panel, "Bitrate Viewer")

        self.tabs.addTab(self._build_settings_panel(), "Settings")

    # ------------------------------------------------------------------ settings tab
    def _options_from_settings(self) -> VmafOptions:
        """The options a newly added video starts from. Only a starting
        point: each row's own settings are edited in the Options panel."""
        return VmafOptions(
            gpu_decode=self._settings.default_gpu_decode,
            extra_features=self._settings.default_extra_features(),
            compute_xpsnr=self._settings.default_compute_xpsnr,
            compute_vmaf=self._settings.default_compute_vmaf,
            compute_vmaf_neg=self._settings.default_compute_vmaf_neg,
        )

    def _extra_metrics_from_settings(self) -> set[str]:
        """The perceptual metrics a newly added video starts with ticked."""
        return {
            key for key in _EXTRA_METRIC_KEYS
            if getattr(self._settings, f"default_compute_{key}", False) is True
        }

    def _cvvdp_from_settings(self) -> CvvdpSettings:
        """The CVVDP display a newly added video starts with: the preset
        chosen in Settings -- the user's own, once they have saved one --
        or the built-in default. The same for every video, whatever its
        format."""
        return default_settings(self._settings.cvvdp_presets, self._settings.cvvdp_default_preset)

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
        self.settings_cache_edit.setPlaceholderText("blank = shared user-profile folder")
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
        metrics_row.addWidget(QLabel("Default metrics:"))
        self.settings_default_psnr = QCheckBox("PSNR")
        self.settings_default_vmaf = QCheckBox("VMAF")
        self.settings_default_vmaf_neg = QCheckBox("VMAF NEG")
        self.settings_default_ssim = QCheckBox("SSIM")
        self.settings_default_xpsnr = QCheckBox("XPSNR")
        self.settings_default_ssimulacra2 = QCheckBox("SSIMULACRA2")
        self.settings_default_butteraugli = QCheckBox("Butteraugli")
        self.settings_default_cvvdp = QCheckBox("CVVDP")
        for box, value in (
            (self.settings_default_vmaf, self._settings.default_compute_vmaf),
            (self.settings_default_vmaf_neg, self._settings.default_compute_vmaf_neg),
            (self.settings_default_psnr, self._settings.default_compute_psnr),
            (self.settings_default_ssim, self._settings.default_compute_ssim),
            (self.settings_default_xpsnr, self._settings.default_compute_xpsnr),
            (self.settings_default_ssimulacra2, self._settings.default_compute_ssimulacra2),
            (self.settings_default_butteraugli, self._settings.default_compute_butteraugli),
            (self.settings_default_cvvdp, self._settings.default_compute_cvvdp),
        ):
            box.setChecked(value)
            box.toggled.connect(self._on_settings_edited)
            metrics_row.addWidget(box)
        metrics_row.addStretch(1)
        defaults_layout.addLayout(metrics_row)

        cvvdp_row = QHBoxLayout()
        cvvdp_row.addWidget(QLabel("CVVDP display:"))
        self.settings_cvvdp_default = QComboBox()
        self.settings_cvvdp_default.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.settings_cvvdp_default.setToolTip(
            "The display CVVDP models for newly added videos, whatever their format. "
            "Saving a preset of your own in the Options panel makes it the default; "
            "choose here to change that."
        )
        self._fill_cvvdp_default_combo()
        self.settings_cvvdp_default.currentIndexChanged.connect(self._on_settings_edited)
        cvvdp_row.addWidget(self.settings_cvvdp_default)
        cvvdp_row.addStretch(1)
        defaults_layout.addLayout(cvvdp_row)
        outer.addWidget(defaults_box)

        compare_box = QGroupBox("Video Compare")
        compare_layout = QVBoxLayout(compare_box)
        # A dropdown, not a spin box: 1 to 9 is one click away.
        self.settings_decoded_videos = QComboBox()
        for count in range(1, 10):
            self.settings_decoded_videos.addItem(str(count), count)
        self.settings_decoded_videos.setCurrentIndex(
            max(0, self.settings_decoded_videos.findData(self._settings.compare_decoded_videos))
        )
        # As wide as its widest entry and no wider: a form row would
        # otherwise stretch a one-digit dropdown across the window.
        self.settings_decoded_videos.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.settings_decoded_videos.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.settings_decoded_videos.setToolTip(
            "How many test videos keep decoding while one is shown, so the "
            "left and right arrows switch to a video that is already running: "
            "the selected one, then the one to its right, its left, the next "
            "right, the next left, and so on. Each is a running GPU decoder "
            "(about 250 MB of RAM for 4K), plus the source. Takes effect "
            "immediately, even during playback."
        )
        self.settings_decoded_videos.currentIndexChanged.connect(self._on_settings_edited)
        # Dropdown first, at the left edge, with its label beside it -- the
        # way the check boxes above sit -- rather than a form row that puts
        # the field wherever the label column ends.
        decoded_row = QHBoxLayout()
        decoded_row.addWidget(self.settings_decoded_videos)
        decoded_row.addWidget(
            QLabel("Number of test videos decoded simultaneously for fast comparison switching")
        )
        decoded_row.addStretch(1)
        compare_layout.addLayout(decoded_row)
        outer.addWidget(compare_box)

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
        count, size_bytes = result_cache.cache_summary(directory)
        self.settings_cache_summary.setText(
            f"{count} saved result(s), {size_bytes / 1_048_576:.1f} MB in {directory}"
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
        self._settings.default_compute_vmaf = self.settings_default_vmaf.isChecked()
        self._settings.default_compute_vmaf_neg = self.settings_default_vmaf_neg.isChecked()
        self._settings.default_compute_ssim = self.settings_default_ssim.isChecked()
        self._settings.default_compute_xpsnr = self.settings_default_xpsnr.isChecked()
        self._settings.default_compute_ssimulacra2 = self.settings_default_ssimulacra2.isChecked()
        self._settings.default_compute_butteraugli = self.settings_default_butteraugli.isChecked()
        self._settings.default_compute_cvvdp = self.settings_default_cvvdp.isChecked()
        chosen = self.settings_cvvdp_default.currentData()
        self._settings.cvvdp_default_preset = "" if chosen in (None, DEFAULT_PRESET.name) else chosen
        self._settings.compare_decoded_videos = int(self.settings_decoded_videos.currentData())
        self.frame_compare_panel.set_decoded_videos(self._settings.compare_decoded_videos)
        self._settings.remember_window_size = self.settings_remember_size.isChecked()

        if self._settings.ffmpeg_dir != before_ffmpeg:
            self._apply_ffmpeg_setting()
            self._check_ffmpeg(prompt=False)
        if self._settings.cache_dir != before_cache:
            result_cache.set_cache_dir_override(self._settings.cache_dir_path())

        # Only the starting point for new rows; existing rows keep whatever
        # they were given.
        self._default_options = self._options_from_settings()
        self._default_extra_metric_keys = self._extra_metrics_from_settings()
        self._default_cvvdp = self._cvvdp_from_settings()
        error = self._settings.save()
        self.settings_status.setText(error or "Settings saved.")
        self._refresh_settings_status()

    def _fill_cvvdp_default_combo(self) -> None:
        """Lists every CVVDP preset in Settings, the default selected."""
        combo = self.settings_cvvdp_default
        combo.blockSignals(True)
        try:
            combo.clear()
            for preset in cvvdp_presets(self._settings.cvvdp_presets):
                combo.addItem(preset.name, preset.name)
            chosen = self._settings.cvvdp_default_preset or DEFAULT_PRESET.name
            combo.setCurrentIndex(max(0, combo.findData(chosen)))
        finally:
            combo.blockSignals(False)

    def _on_frame_color_mode_changed(self, mode: str) -> None:
        self._settings.frame_preview_color_mode = mode
        error = self._settings.save()
        if error:
            self.status_label.setText(error)

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
        count, _size_bytes = result_cache.cache_summary(directory)
        if count == 0 and self._file_writes.pending == 0:
            self.settings_status.setText("There are no saved results to clear.")
            return
        if self._cache_clear_result is not None:
            self.settings_status.setText("Saved results are already being cleared.")
            return
        confirm = QMessageBox.question(
            self, "Clear saved results",
            f"Delete saved results from {directory}?\n\n"
            "Videos already scored will have to be recomputed.",
        )
        if confirm != QMessageBox.Yes:
            return
        # Cache writes and deletion share the same serial queue. If a run
        # just finished, its pending store therefore lands before clear_all
        # rather than recreating an entry after the user cleared everything.
        self._cache_clear_result = []
        # `directory` is the folder the confirmation dialog just named.
        # Resolving it inside the queued task instead would delete whatever
        # folder is configured by the time it runs -- so changing the cache
        # setting between confirming and the queue draining would wipe a
        # folder the user was never asked about.
        self._file_writes.submit(
            "clear saved results",
            lambda: self._cache_clear_result.append(
                result_cache.clear_all(directory)
            ),
        )
        self.settings_status.setText("Clearing saved results...")

    def _build_files_panel(self) -> QWidget:
        files_box = QGroupBox("Videos")
        self.files_box = files_box
        files_layout = QVBoxLayout(files_box)

        files_layout.addWidget(QLabel("Reference video:"))
        src_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_browse_btn = src_browse = QPushButton("Browse...")
        src_browse.clicked.connect(self._on_browse_source)
        src_row.addWidget(self.source_edit, stretch=1)
        src_row.addWidget(src_browse)
        files_layout.addLayout(src_row)
        self.source_info_label = QLabel("No reference selected.")
        self.source_info_label.setStyleSheet("color: #666;")
        files_layout.addWidget(self.source_info_label)

        # The metric picker belongs to the table: every metric is a column
        # in it, and more are coming. So it sits on the table's top-right
        # corner, in the row that introduces the table, with no gap below.
        self.metrics_btn = QPushButton("Add/remove metrics")
        self.metrics_btn.setToolTip(
            "Choose which metrics appear in the test-video table. A hidden metric "
            "is not calculated; scores already saved for it come back when it is "
            "shown again. Also available by right-clicking the column headers."
        )
        self.metrics_btn.clicked.connect(
            lambda: self._show_metrics_menu(
                self.metrics_btn.mapToGlobal(self.metrics_btn.rect().bottomLeft())
            )
        )
        table_heading = QHBoxLayout()
        table_heading.addWidget(QLabel(
            "Test videos to compare against the reference. Check rows to calculate; "
            "select rows to edit their settings below. Metric header shortcuts apply to all rows."
        ), stretch=1, alignment=Qt.AlignBottom)
        table_heading.addWidget(self.metrics_btn, alignment=Qt.AlignBottom)
        # One block, heading row and table, with no spacing between them, so
        # the button reads as part of the table rather than as a separate row.
        table_block = QVBoxLayout()
        table_block.setSpacing(0)
        table_block.addLayout(table_heading)
        metric_cols = [item.column for item in _METRIC_COLUMNS]
        self.distorted_table = FillColumnTable(
            0, len(_METRIC_COLUMNS) + 6, fill_column=COL_PATH,
            other_columns=[
                COL_CHECK, COL_INFO, COL_BLACK_BARS, COL_SCALING, COL_BITRATE,
                *metric_cols,
            ],
        )
        # Seeded from the same defaults a new row gets, rather than a fixed
        # set: hardcoded here, the headers contradicted the Settings tab the
        # moment either was changed.
        self.metric_header = CheckableHeaderView(
            {
                col: self._default_metric_ticked(col)
                for item in _METRIC_COLUMNS for col in (item.column,)
            },
            self.distorted_table,
        )
        self.distorted_table.setHorizontalHeader(self.metric_header)
        for visual, item in enumerate(_METRIC_COLUMNS, start=6):
            self.metric_header.moveSection(self.metric_header.visualIndex(item.column), visual)
        self.metric_header.sectionToggled.connect(self._on_metric_column_toggled)
        self.distorted_table.setHorizontalHeaderLabels(
            [
                "", "File name", "Media info", "Black bars", "Scaling", "Video bitrate",
                f"   {metric_definition('psnr').table_header}", f"   {metric_definition('ssim').table_header}",
                f"   {metric_definition('vmaf').table_header}", f"   {metric_definition('xpsnr').table_header}",
                f"   {metric_definition('vmaf_neg').table_header}",
                f"   {metric_definition('ssimulacra2').table_header}",
                f"   {metric_definition('butteraugli').table_header}",
                f"   {metric_definition('cvvdp').table_header}",
            ]
        )
        self.distorted_table.verticalHeader().setVisible(False)
        # Keep internal column indices stable, but omit crop status from the
        # test-video table. Cropping remains available in the row options.
        self.distorted_table.setColumnHidden(COL_BLACK_BARS, True)
        self.metric_header.setContextMenuPolicy(Qt.CustomContextMenu)
        self.metric_header.customContextMenuRequested.connect(
            lambda pos: self._show_metrics_menu(self.metric_header.mapToGlobal(pos))
        )
        self._apply_metric_visibility()
        self.distorted_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.distorted_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.distorted_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.distorted_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.distorted_table.itemChanged.connect(self._on_table_item_changed)
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
        # Never narrower than the heading itself needs. "Video bitrate" was
        # set to 65px and rendered as "'ideo bitrat" -- the widths here are
        # picked for the values, which are shorter than several of the
        # headings, and nothing widened the column until a row arrived with
        # content to measure. An empty table showed a clipped heading, and a
        # renamed one could clip again silently.
        starting_widths = {
            COL_CHECK: 28,
            COL_INFO: 150,
            COL_BLACK_BARS: 78,
            COL_SCALING: 90,
            COL_BITRATE: 65,
            **{col: (110 if col in (COL_PSNR, COL_XPSNR, COL_SSIMULACRA2, COL_BUTTERAUGLI) else 78)
               for item in _METRIC_COLUMNS for col in (item.column,)},
        }
        for col, width in starting_widths.items():
            self.distorted_table.setColumnWidth(
                col, max(width, header.sectionSizeHint(col))
            )
        table_block.addWidget(self.distorted_table, stretch=1)
        files_layout.addLayout(table_block, stretch=1)

        dist_btn_row = QHBoxLayout()
        add_dist_btn = QPushButton("Add files...")
        add_dist_btn.clicked.connect(self._on_add_distorted)
        add_resample_btn = QPushButton("Add resolution test...")
        add_resample_btn.setToolTip(
            "Calculates selected metrics for downscaling the reference to a lower resolution and "
            "scaling it back up -- no separate test file needed."
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

        # Frozen during a run, one control at a time rather than by disabling
        # the whole box: a disabled QGroupBox disables its children, and that
        # took the video table's scrollbar with it -- so a queue longer than
        # the window could not be scrolled precisely while it was running,
        # which is exactly when someone wants to watch it.
        self._file_action_widgets = [
            self.source_browse_btn, add_dist_btn, add_resample_btn,
            remove_dist_btn, self.remove_all_btn,
            # What a run calculates is fixed when it starts.
            self.metrics_btn,
        ]

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

        # No "Metrics to calculate" box here any more: each metric is now
        # ticked in its own column of the table, on the row it applies to.
        # A separate panel meant the choice was made in one place and read
        # back in another, and it could only ever describe the current
        # selection -- so seeing what four rows were set to took four clicks.

        form = QFormLayout()
        performance_box = QGroupBox("Performance")
        performance_form = QFormLayout(performance_box)
        metric_options_box = QGroupBox("Metric-specific settings")
        metric_options_form = QFormLayout(metric_options_box)
        self.model_combo = QComboBox()
        for name, _ in _MODEL_CHOICES:
            self.model_combo.addItem(name)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_combo.setToolTip(
            "VMAF only. Auto selects the v0 model using the resolution after "
            "calculation cropping and scaling. Bundled v1 models require a "
            "recent libvmaf-capable FFmpeg build."
        )
        metric_options_form.addRow("VMAF model:", self.model_combo)

        self.gpu_checkbox = QCheckBox("Use GPU decoding")
        self.gpu_checkbox.setToolTip(
            "Hardware-decodes the reference and the test video. Each is "
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
        performance_form.addRow("GPU decode:", gpu_row)

        self.ssimulacra2_backend_combo = QComboBox()
        self.ssimulacra2_backend_combo.addItems(["GPU", "CPU"])
        self.ssimulacra2_backend_combo.setToolTip(
            "GPU uses Vship when a supported device is available. If GPU computation "
            "cannot run, the metric automatically falls back to CPU. Choose CPU to "
            "always use the bundled libjxl reference implementation."
        )
        self.ssimulacra2_backend_combo.currentIndexChanged.connect(
            lambda _index: self._on_metric_backend_changed("ssimulacra2")
        )
        performance_form.addRow("SSIMULACRA2 compute:", self.ssimulacra2_backend_combo)

        self.butteraugli_backend_combo = QComboBox()
        self.butteraugli_backend_combo.addItems(["GPU", "CPU"])
        self.butteraugli_backend_combo.setToolTip(
            "GPU uses Vship when a supported device is available. If GPU computation "
            "cannot run, the metric automatically falls back to CPU. Choose CPU to "
            "always use the bundled libjxl reference implementation."
        )
        self.butteraugli_backend_combo.currentIndexChanged.connect(
            lambda _index: self._on_metric_backend_changed("butteraugli")
        )
        performance_form.addRow("Butteraugli compute:", self.butteraugli_backend_combo)

        detected = detected_gpu_vendors()
        if detected:
            names = ", ".join(v.value.upper() for v in detected)
            performance_form.addRow("", QLabel(f"Detected GPU(s): {names}"))

        self.crop_combo = QComboBox()
        self.crop_combo.addItems([
            "Auto-detect (recommended)",
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
        basic_column = QGroupBox("Video preparation (calculation only)")
        basic_column.setLayout(form)
        columns.addWidget(basic_column, stretch=1)
        options_layout.addLayout(columns)

        adv_form = form

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 128)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText("Auto")
        self.threads_spin.valueChanged.connect(
            lambda _value: self._on_panel_field_edited("n_threads")
        )
        performance_form.addRow("libvmaf threads:", self.threads_spin)
        self.threads_spin.setToolTip(
            "Controls VMAF, PSNR and SSIM extraction in libvmaf, not XPSNR "
            "or video decoding.\n\nAuto uses every core, or half of them "
            "for each video when two are calculated in parallel."
        )

        self.subsample_spin = QSpinBox()
        self.subsample_spin.setRange(1, 60)
        self.subsample_spin.setValue(1)
        self.subsample_spin.valueChanged.connect(
            lambda _value: self._on_panel_field_edited("n_subsample")
        )
        metric_options_form.addRow("libvmaf frame subsample:", self.subsample_spin)
        self.subsample_spin.setToolTip("1 = every frame. Applies to VMAF, PSNR and SSIM. XPSNR is computed every frame; combined runs retain values at libvmaf's sampled frames.")

        # CVVDP's display: a preset, the display itself, and whether the
        # video is scaled to fill it (see vmaf_app.core.cvvdp).
        self.cvvdp_preset_combo = QComboBox()
        self.cvvdp_preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cvvdp_preset_combo.setToolTip(
            "The display CVVDP predicts visible differences on. The score depends on the "
            "display as much as on the videos, so compare videos scored for the same one.\n\n"
            "New videos start with the preset chosen in Settings. Saving a preset of your "
            "own makes it that default."
        )
        self.cvvdp_preset_combo.currentIndexChanged.connect(self._on_cvvdp_preset_chosen)
        metric_options_form.addRow("CVVDP display:", self.cvvdp_preset_combo)
        self.cvvdp_display_label = QLabel()
        self.cvvdp_display_label.setStyleSheet("color: #666;")
        self.cvvdp_display_label.setWordWrap(True)
        metric_options_form.addRow("", self.cvvdp_display_label)
        cvvdp_buttons = QHBoxLayout()
        self.cvvdp_add_btn = QPushButton("Add preset...")
        self.cvvdp_add_btn.setToolTip(
            "Make a new preset of your own, starting from the display shown here. "
            "It becomes the display newly added videos start with; the selected "
            "videos keep theirs until you choose it for them."
        )
        self.cvvdp_add_btn.clicked.connect(self._on_cvvdp_add_preset)
        self.cvvdp_edit_btn = QPushButton("Edit display...")
        self.cvvdp_edit_btn.setToolTip(
            "Change the display's size, distance, brightness, room light and more; "
            "rename your preset or save the display as a new one."
        )
        self.cvvdp_edit_btn.clicked.connect(self._on_cvvdp_edit_display)
        self.cvvdp_delete_btn = QPushButton("Delete preset")
        self.cvvdp_delete_btn.setToolTip("Delete this preset of yours. Built-in presets cannot be deleted.")
        self.cvvdp_delete_btn.clicked.connect(self._on_cvvdp_delete_preset)
        for button in (self.cvvdp_add_btn, self.cvvdp_edit_btn, self.cvvdp_delete_btn):
            cvvdp_buttons.addWidget(button)
        cvvdp_buttons.addStretch(1)
        metric_options_form.addRow("", cvvdp_buttons)
        self.cvvdp_resize_check = QCheckBox("Scale the video to fill the display")
        self.cvvdp_resize_check.setToolTip(
            "Off (the official default): the video is shown pixel for pixel, so a 1080p "
            "video covers a quarter of a 4K display. On: it is scaled, keeping its shape, "
            "to fill the display.\n\nSet per video: choosing or saving a display preset "
            "does not change it."
        )
        # checkStateChanged, not toggled: a partly ticked box (a mixed
        # selection) already counts as checked, so ticking it fully would not
        # toggle anything.
        self.cvvdp_resize_check.checkStateChanged.connect(
            lambda state: None if state == Qt.PartiallyChecked
            else self._on_cvvdp_resize_toggled(state == Qt.Checked)
        )
        metric_options_form.addRow("", self.cvvdp_resize_check)

        self.duration_edit = QTimeEdit()
        self.duration_edit.setDisplayFormat("HH:mm:ss.zzz")
        self.duration_edit.setTime(QTime(0, 0, 0, 0))
        self.duration_edit.timeChanged.connect(
            lambda _value: self._on_panel_field_edited("duration_limit")
        )
        adv_form.addRow("Duration limit (0 = full):", self.duration_edit)

        self.scale_algo_combo = QComboBox()
        self.scale_algo_combo.addItems(_SCALE_ALGORITHMS)
        self.scale_algo_combo.currentIndexChanged.connect(
            lambda _value: self._on_panel_field_edited("scale_algorithm")
        )
        adv_form.addRow("Scaling algorithm:", self.scale_algo_combo)

        self.scale_direction_combo = QComboBox()
        self.scale_direction_combo.addItems([
            "Source downscaled to test",
            "Test upscaled to source",
            "Test both (adds a comparison row)",
        ])
        self.scale_direction_combo.setToolTip(
            "When the two resolutions differ: either evaluate quality at the resolution actually\n"
            "delivered (source scaled to match test -- the default), or as if the test\n"
            "video were upscaled back to the reference's native resolution for playback.\n"
            "\"Test both\" doesn't change this row -- it adds a second row for the same test\n"
            "file using the other direction, so you can run and compare both."
        )
        self.scale_direction_combo.currentIndexChanged.connect(self._on_scale_direction_combo_changed)
        adv_form.addRow("Resolution mismatch:", self.scale_direction_combo)

        metrics_hint = QLabel(
            "Cropping and scaling here affect scores. Video Compare's tone mapping and "
            "playback resolution are display-only settings and do not change calculated metrics."
        )
        metrics_hint.setStyleSheet("color: #666; font-style: italic;")
        metrics_hint.setWordWrap(True)
        options_layout.addWidget(metrics_hint)

        columns.addWidget(metric_options_box, stretch=1)
        columns.addWidget(performance_box, stretch=1)
        options_layout.addStretch(1)

        return options_box

    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Calculate metrics")
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
        # Here rather than in Settings, and deliberately still editable while
        # a run is going: a long queue is exactly when someone notices their
        # CPU is half idle, and the Settings tab is locked during a run.
        cores = os.cpu_count() or 1
        run_row.addSpacing(16)
        # A checkbox rather than a number: the choice is only ever one or
        # two (see MAX_PARALLEL_JOBS), and a bare "2" said nothing about
        # what it was counting.
        self.parallel_jobs_check = QCheckBox(
            "Calculate CPU metrics for 2 videos in parallel"
        )
        self.parallel_jobs_check.setChecked(
            self._settings.parallel_jobs >= MAX_PARALLEL_JOBS
        )
        self.parallel_jobs_check.setToolTip(
            f"Runs CPU metric work for two videos simultaneously on this {cores}-core "
            "machine. libvmaf does not keep a many-core CPU busy on its "
            "own, so a second video largely fills the idle capacity "
            "rather than competing for it. libvmaf threads left on Auto "
            "are shared: each video gets half the cores.\n\nCan be "
            "changed while a run is in progress: ticking it starts another "
            "video straight away, unticking it lets the running ones finish "
            "first.\n\nOnly affects how fast results arrive, never what "
            "they are."
        )
        self.parallel_jobs_check.toggled.connect(self._on_parallel_jobs_changed)
        run_row.addWidget(self.parallel_jobs_check)
        run_row.addStretch(1)

        load_btn = QPushButton("Load analysis results...")
        load_btn.clicked.connect(self._on_load_saved_run)
        self.save_btn = save_btn = QPushButton("Save selected results...")
        save_btn.clicked.connect(self._on_save_selected)
        # One button, not two. "Plot selected results" claimed to plot a
        # subset, but opening the tab syncs every completed row into it
        # (see _sync_graph), so both buttons left exactly the same graph on
        # screen -- the selection one merely refused to open when nothing
        # selected had a result.
        self.show_graph_btn = QPushButton("Open metric graphs")
        self.show_graph_btn.setToolTip("Opens Metric Graphs with the current results and graph settings.")
        self.show_graph_btn.clicked.connect(self._on_show_graph_clicked)
        run_row.addWidget(load_btn)
        run_row.addWidget(save_btn)
        run_row.addWidget(self.show_graph_btn)
        layout.addLayout(run_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        # One bar per video that is actually running, so two at once are two
        # readable lines rather than a single bar flickering between them.
        # Built once and hidden, because rows appearing and disappearing
        # mid-run would shift everything below them on every job boundary.
        # One line of text per running video. No bar: the percentage is the
        # only thing a bar was conveying, it says it exactly rather than
        # approximately, and two bars stacked above a third read as a block
        # of chrome rather than as a status.
        self.job_progress_labels: list[QLabel] = []
        for _ in range(MAX_PARALLEL_JOBS):
            line = QLabel()
            line.setStyleSheet("color: #444;")
            line.setVisible(False)
            layout.addWidget(line)
            self.job_progress_labels.append(line)

        # No overall bar: each running video already has one, and a third
        # bar summarising them was just more to read. The last line is the
        # one thing those bars cannot say -- when the whole queue ends.
        # No separate queue line: the ETA rides on the status line above,
        # which otherwise only said how many videos were running.
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
        path, _ = QFileDialog.getOpenFileName(self, "Select reference video")
        if not path:
            return
        self.source_info_label.setText(f"Reading {Path(path).name}...")
        self.status_label.setText("Reading reference video...")
        self._start_source_probe(Path(path))

    def _start_source_probe(self, path: Path) -> None:
        """Reads a selected reference without freezing the main window."""
        if self._source_probe_worker is not None and self._source_probe_worker.isRunning():
            self._source_probe_worker.cancel()
        self._source_probe_generation += 1
        generation = self._source_probe_generation
        worker = ProbeWorker([path], None, False, {}, parent=self)
        self._source_probe_worker = worker
        self._probe_workers.append(worker)
        worker.probed.connect(
            lambda selected, info, error, g=generation:
            self._on_source_probed_if_current(g, selected, info, error)
        )
        worker.finished_all.connect(
            lambda g=generation, w=worker: self._on_source_probe_finished(g, w)
        )
        worker.start()

    def _on_source_probed_if_current(
        self, generation: int, path: Path, info, error: str
    ) -> None:
        if generation != self._source_probe_generation:
            return
        if info is None:
            previous = self._source_info
            self.source_info_label.setText(
                (
                    f"{media_info_string(previous)}, {bitrate_string(previous)}  "
                    f"({format_hms(previous.duration, decimals=1)})"
                )
                if previous is not None else "No reference selected."
            )
            QMessageBox.critical(self, "Could not read video", error)
            return
        self._apply_source_info(path, info)

    def _apply_source_info(self, path: Path, info: VideoInfo) -> None:
        self._source_info = info
        self.source_edit.setText(str(path))
        self.source_info_label.setText(
            f"{media_info_string(info)}, {bitrate_string(info)}  ({format_hms(info.duration, decimals=1)})"
        )
        # A different/newly-picked source might match a previously cached
        # (source, distorted) pair for rows that are already in the table.
        # Each of those is a multi-MB JSON parse, so with a few long videos
        # loaded this froze the window for seconds; it goes to the worker,
        # which already knows how to load a cached result and report it.
        # A resolution test is derived from the reference, unlike a distorted
        # file, which merely gets compared against it -- so it cannot survive
        # the reference changing underneath it.
        dropped = self._remove_rows_owned_by_the_previous_source()

        if self._rows:
            # Scores belong to a (source, distorted) pair, so a new source
            # invalidates every one of them until the cache says otherwise.
            for row in range(len(self._rows)):
                completed = self._rows[row].completed_run
                if completed is not None and self._same_source(
                    completed.result.source, path
                ):
                    # This result was measured against exactly the reference
                    # just selected, so selecting it CONFIRMS the result
                    # rather than invalidating it. Discarding it here is what
                    # made loading a saved run and then picking its own
                    # source wipe the run.
                    continue
                # A curve is the visual form of the same (source,
                # distorted) result. Clearing only the table value left the
                # old source's curve on screen under the newly selected
                # source, which is a dangerously plausible comparison.
                self._invalidate_completed_result(row)
            self._reload_cached_for_all_rows()

        if dropped:
            QMessageBox.information(
                self, "Resolution tests removed",
                f"{len(dropped)} resolution test(s) belonged to the previous "
                "source and have been removed:\n\n"
                + "\n".join(f"  {name}" for name in dropped)
                + "\n\nAdd them again to test the new source.",
            )

    def _remove_rows_owned_by_the_previous_source(self) -> list[str]:
        """Drops resolution-test rows and returns what was removed.

        Such a row has no distorted file of its own: it downscales and
        re-upscales THE SOURCE, so its synthetic path, its media info, its
        description and its identity all come from the reference that was
        selected when it was added. Leaving it in place after the reference
        changed left a row describing one video while the job would have run
        against another -- and a target width chosen as a downscale of a
        3840-wide master is an UPSCALE of a 1280-wide one, which the test was
        never meant to measure.

        Removing them is deliberate rather than migrating them: the target
        list depends on the new source's width, two migrated rows can
        collapse onto the same test, and silently rewriting what a row means
        is worse than saying it is gone.
        """
        removed: list[str] = []
        for row in range(len(self._rows) - 1, -1, -1):
            row_data = self._rows[row]
            if row_data.options.resample_test is None:
                continue
            if row_data.completed_run is not None:
                self.graph_panel.remove_by_identity(row_data.completed_run.graph_identity)
            self.distorted_table.removeRow(row)
            del self._rows[row]
            removed.append(row_data.path.name)
        if removed:
            self._on_table_selection_changed()
            self._sync_frame_compare()
        return list(reversed(removed))

    def _show_ready(self) -> None:
        """Says "Ready." once reading videos or saved results is done -- in
        place of the "Reading..." message only. Anything else on the status
        line is a message for the user (a preset saved, results cleared),
        and a lookup finishing a moment later used to overwrite it."""
        text = self.status_label.text()
        if not text or text.startswith("Reading"):
            self.status_label.setText("Ready.")

    def _on_source_probe_finished(self, generation: int, worker: ProbeWorker) -> None:
        if worker in self._probe_workers:
            self._probe_workers.remove(worker)
        worker.deleteLater()
        if generation == self._source_probe_generation:
            self._source_probe_worker = None
            if not self._run_active:
                self._show_ready()
            self._on_table_selection_changed()

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
        bars_item = QTableWidgetItem()
        bars_item.setTextAlignment(Qt.AlignCenter)
        self.distorted_table.setItem(row, COL_BLACK_BARS, bars_item)
        self.distorted_table.setItem(row, COL_SCALING, QTableWidgetItem(""))
        self.distorted_table.setItem(row, COL_BITRATE, QTableWidgetItem(""))
        for metric_column in _METRIC_COLUMNS:
            col = metric_column.column
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignCenter)
            self.distorted_table.setItem(row, col, item)

        # New rows start with a copy of whatever the panel last showed, so
        # adding several similar files in a row doesn't mean reconfiguring
        # each one from scratch -- but it's still an independent copy from
        # this point on, so editing one row never affects another.
        self._rows.append(RowData(
            path=path, options=clone_options(self._default_options),
            extra_metric_keys=set(self._default_extra_metric_keys),
            metric_backends=dict(self._default_metric_backends),
            cvvdp=self._default_cvvdp,
            cvvdp_preset=self._settings.cvvdp_default_preset,
        ))
        self._set_row_metrics(row)
        return row

    def _set_row_metrics(self, row: int) -> None:
        """Each metric cell shows its score, or a tick box for calculating it.

        A metric that has been measured shows only the number: there is no
        decision left to make about it, and a tick box beside a finished
        score invited un-ticking it as though that would undo the
        measurement. A metric without a score shows a check box instead, in
        the very column its result will land in, so choosing what to
        calculate is one click on the row it applies to.

        Previously calculated metrics survive selection edits -- ticking
        another metric changes what is requested, not what was measured.
        """
        row_data = self._rows[row]
        run = row_data.completed_run
        # setCheckState/setData below emit itemChanged, which is also how a
        # real click reaches _on_table_item_changed. Without this the window
        # would read its own repaint back as the user asking for a change.
        self._syncing_table = True
        try:
            for metric_column in _METRIC_COLUMNS:
                col = metric_column.column
                item = self.distorted_table.item(row, col)
                if item is None:
                    continue
                enabled = self._row_metric_enabled(row_data, col)
                value = self._metric_mean(run, col) if run is not None else None
                unavailable = (
                    self._metric_unavailable_reason(row_data, metric_column.key)
                    if value is None else None
                )
                if unavailable is not None:
                    # No tick box: there is nothing to choose. The row's own
                    # selection is left alone for when it applies again.
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setData(Qt.CheckStateRole, None)
                    item.setText("n/a")
                    item.setToolTip(unavailable)
                    item.setForeground(QColor("#888"))
                    item.setBackground(QColor(0, 0, 0, 0))
                    item.setFont(QFont())
                    continue
                if value is None:
                    item.setFlags(
                        Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    )
                    item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
                    failed = enabled and row_data.analysis_status in {"Failed", _PARTLY_FAILED}
                    item.setText("Failed" if failed else "")
                    item.setToolTip((
                        "This metric failed on the last run. Untick to skip it."
                        if failed else
                        "Ticked: calculated on the next run. Untick to skip it."
                        if enabled else
                        "Not selected. Tick to calculate this metric."
                    ) + (self._cvvdp_elsewhere_note(row_data) if metric_column.key == "cvvdp" else ""))
                    item.setForeground(
                        QColor("#a03030") if failed else self.distorted_table.palette().text()
                    )
                    item.setBackground(QColor(0, 0, 0, 0))
                    item.setFont(QFont())
                    continue
                # Measured: the score replaces the tick box entirely. Passing
                # no value for CheckStateRole is what removes the indicator --
                # Qt draws one for any item that merely *has* the role set.
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setData(Qt.CheckStateRole, None)
                frame_metric = run.result.frame_metric(metric_column.key)
                text = metric_column.metric.format_value(value)
                if (frame_metric is not None and metric_column.key in row_data.metric_backends
                        and not self._backend_matches(row_data, metric_column.key, frame_metric)):
                    # Visible without hovering: a score from the other
                    # implementation is on a different scale from the rest.
                    text += f" ({frame_metric.provenance.compute_backend.upper()})"
                item.setText(text)
                if metric_column.metric.kind is MetricKind.SEQUENCE:
                    item.setToolTip(self._cvvdp_note(row_data))
                else:
                    item.setToolTip(
                        "Mean of calculated frame scores."
                        + self._identical_frames_note(run, col)
                        + self._backend_note(row_data, metric_column.key, frame_metric)
                    )
                font = QFont()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QColor("#000"))
                item.setBackground(QColor(0, 0, 0, 0))
        finally:
            self._syncing_table = False
        self._refresh_row_state(row)
        self._set_row_black_bars(row)
        self._set_row_scaling(row)

    def _row_state(self, row_data: RowData) -> str:
        """How this row's analysis stands, in one phrase.

        No longer a column of its own: the metric cells already answer it.
        An unticked box means the metric was not asked for, a ticked empty
        one means it has not been measured yet, and a number means it has --
        so "Not calculated", "Partially calculated" and "Complete" were
        restating what the same row showed three columns to the left. What
        survives here is what those cells cannot say: which stage a live job
        is at, and why a finished job is not being shown.
        """
        return row_data.analysis_status or (
            "No metrics selected" if not self._requested_metrics(row_data) else
            "Complete" if self._has_requested_results(row_data) else
            "Partially calculated" if row_data.completed_run is not None
            else "Not calculated"
        )

    def _refresh_row_state(self, row: int) -> None:
        """Puts the row's state on its file name: the full path, then the
        state and any detail, with a colour for the two cases nothing else
        on the row reveals."""
        item = self.distorted_table.item(row, COL_PATH)
        if item is None:
            return
        row_data = self._rows[row]
        state = self._row_state(row_data)
        lines = [str(row_data.path), state]
        if row_data.status_detail:
            lines.append(row_data.status_detail)
        item.setToolTip("\n\n".join(lines))
        if state == "Failed":
            colour = _STATE_COLOURS["failed"]
        elif state.startswith("Finished"):
            # Done, cached, and deliberately not displayed -- see
            # _on_job_finished. Without a mark the row would look untouched.
            colour = _STATE_COLOURS["stale"]
        else:
            item.setForeground(self.distorted_table.palette().text())
            return
        item.setForeground(QColor(colour))

    @staticmethod
    def _identical_frames_note(run: CompletedRun, column: int) -> str:
        """Says when frames were held out of the mean, and why.

        Without it the number silently describes fewer frames than the run
        measured, which is worse than the "inf" it replaced.
        """
        metric = _METRIC_COLUMN_BY_INDEX[column].metric
        values = run.result.frames.values(metric.key)
        if values is None:
            return ""
        identical = int(np.isposinf(np.asarray(values, dtype=np.float64)).sum())
        if not identical:
            return ""
        return (
            f"\n\n{identical} of {len(values)} frames were identical to the reference "
            "and scored infinity; they contribute zero distortion and are included "
            "in the frame count for the XPSNR sequence average."
        )

    def _cvvdp_elsewhere_note(self, row_data: RowData) -> str:
        """For an empty CVVDP cell: the scores saved for other displays. A
        CVVDP score is only valid for the display it was made for, so a
        video whose display differs from its earlier run showed no score at
        all, with no hint that one existed or which display to choose to
        see it again."""
        others = [(settings, score) for settings, score in row_data.cvvdp_elsewhere
                  if not settings.same_as(row_data.cvvdp)]
        if not others:
            return ""
        lines = []
        for settings, score in sorted(others, key=lambda item: -item[1]):
            preset = matching_preset(settings, self._settings.cvvdp_presets)
            name = preset.name if preset else f"Custom ({settings.display.describe()})"
            scaled = ", video scaled to fill it" if settings.resize_to_display else ""
            lines.append(f"\u2022 {name}{scaled}: {score:.3f} JOD")
        return ("\n\nNo CVVDP score for this video's display yet. Saved for other displays "
                "(choose one in CVVDP display to show it; they are not scores for this display):\n"
                + "\n".join(lines))

    def _cvvdp_note(self, row_data: RowData) -> str:
        """Tooltip for a CVVDP score: what the number means and which display it is for."""
        preset = matching_preset(row_data.cvvdp, self._settings.cvvdp_presets, row_data.cvvdp_preset)
        display = row_data.cvvdp.display.describe()
        return (
            "CVVDP of the whole video, in JOD (just-objectionable differences): 10 means no "
            "visible difference, and one JOD lower means 75% of viewers would pick the "
            "reference as better.\n\n"
            f"For the display {preset.name if preset else '(custom)'}: {display}"
            + (", video scaled to fill it" if row_data.cvvdp.resize_to_display else "")
            + ".\n\nThe JOD of each second is plotted in Metric Graphs."
        )

    @staticmethod
    def _metric_mean(run: CompletedRun, column: int) -> float | None:
        metric = _METRIC_COLUMN_BY_INDEX[column].metric
        sequence = run.result.sequence_metric(metric.key)
        if sequence is not None:
            return sequence.score  # one score for the video, not a mean
        result = run.result.frame_metric(metric.key)
        values = result.values if result is not None else run.result.frames.values(metric.key)
        if values is None or len(values) == 0:
            return None
        return aggregate_scores(values, metric.aggregation)

    @staticmethod
    def _crop_detail(label: str, info: VideoInfo, crop: CropBox | None) -> str:
        """One video's crop, spelled out for the tooltip."""
        if crop is None:
            return f"{label}: not checked for black bars; no crop was applied."
        if crop.is_noop(info.width, info.height):
            return f"{label}: no black bars. Compared in full at {info.width}x{info.height}."
        sides = (
            ("top", max(0, crop.y)),
            ("bottom", max(0, info.height - crop.y - crop.h)),
            ("left", max(0, crop.x)),
            ("right", max(0, info.width - crop.x - crop.w)),
        )
        cut = ", ".join(f"{name} {value} px" for name, value in sides if value)
        return (
            f"{label}: black bars cropped off -- {cut}.\n"
            f"Compared at {crop.w}x{crop.h} instead of {info.width}x{info.height}."
        )

    @staticmethod
    def _has_black_bars(info: VideoInfo, crop: CropBox | None) -> bool | None:
        """True/False, or None when nothing was cropped off so nothing is known."""
        return None if crop is None else not crop.is_noop(info.width, info.height)

    def _set_row_black_bars(self, row: int, *, probe_failed: bool = False) -> None:
        """Whether the test video has black bars -- and nothing more.

        The cell used to read like "S TB276 . D none": a per-side pixel
        breakdown of both videos, in a column barely wide enough for the
        heading. The question actually being asked of it has two answers,
        so the cell gives one of those and the pixel counts (and what was
        cropped off the reference) wait on hover for when the answer is
        surprising.

        Auto crop is resolved during a run rather than media probing, and can
        differ between rows because duration limits are per-row. The table
        therefore shows a pending state until a result (including a cached or
        loaded result) supplies the crop boxes that were actually used.
        """
        item = self.distorted_table.item(row, COL_BLACK_BARS)
        if item is None:
            return

        def show(text: str, tooltip: str, *, muted: bool = False) -> None:
            item.setText(text)
            item.setToolTip(tooltip)
            item.setForeground(
                QColor("#999") if muted else self.distorted_table.palette().text()
            )

        if probe_failed:
            show("Unknown", "Black bars could not be checked because the video could not be read.", muted=True)
            return

        row_data = self._rows[row]
        completed = row_data.completed_run
        if completed is None:
            if row_data.options.crop_mode == CropMode.NONE:
                show("Off", "Black-bar detection is disabled for this row; the full frames will be compared.", muted=True)
            elif row_data.options.crop_mode == CropMode.MANUAL:
                show("Manual", "A manual crop is configured; the detected sides appear after the run.", muted=True)
            else:
                show("Pending", "Black bars will be detected when this row is run.", muted=True)
            return

        result = completed.result
        # A resolution test has no separate test file: both branches come
        # from the reference, so the reference's own bars are the answer.
        resample = row_data.options.resample_test is not None
        subject_crop = result.source_crop if resample else result.distorted_crop
        subject_info = result.source_info if resample else result.distorted_info

        details = [self._crop_detail("Test video", subject_info, subject_crop)] if not resample else []
        details.append(self._crop_detail("Reference", result.source_info, result.source_crop))
        tooltip = "\n\n".join(details)

        has_bars = self._has_black_bars(subject_info, subject_crop)
        if has_bars is None:
            show("Off", "Black-bar detection was disabled for this run; no crop was applied.", muted=True)
        else:
            show("Yes" if has_bars else "No", tooltip)

    @staticmethod
    def _content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
        """The picture that actually reaches the comparison, bars removed.
        Mirrors vmaf_runner's own _content_size, which decides the same thing
        for the filtergraph."""
        return (crop.w, crop.h) if crop else (info.width, info.height)

    def _resize_mismatch(self, row: int) -> tuple[str, str]:
        """(short tag, full explanation) for which of the two videos gets
        resized to match the other, when they differ -- important now that a
        row can exist for either direction (see "Test both"), so it's not
        ambiguous which one a given row/result represents.

        Compares the two videos *after* black bars come off, because that is
        what the run itself compares. A 1920x1080 letterboxed reference and a
        1920x804 encode of the same film are not a resize at all: they are
        the same picture, one of them still carrying its bars. Reading the
        stored heights literally, this column used to announce a downscale
        that never happens.

        The short tag goes in its own narrow Scaling column and the full
        sentence is the tooltip: spelled out inline it made Media info far
        too wide to scan.

        Uses the *actual* crops and direction a completed run used (what
        really produced its scores, and still right even if the row's
        settings were edited afterward), falling back to the row's current
        setting before it's been run -- UNLESS the row is a "Test both"
        companion, whose direction is fixed and known for certain by
        construction (see RowData.scale_direction_pinned): a result cached
        before that field existed loads as SOURCE_TO_TEST regardless of what
        actually produced it, which is simply wrong for a row that exists
        only to represent DISTORTED_TO_SOURCE.
        """
        row_data = self._rows[row]
        run = row_data.completed_run
        if run is not None:
            source_info, distorted_info = run.result.source_info, run.result.distorted_info
            ref = self._content_size(source_info, run.result.source_crop)
            dist = self._content_size(distorted_info, run.result.distorted_crop)
        else:
            source_info, distorted_info = self._source_info, row_data.video_info
            if source_info is None or distorted_info is None:
                return "", ""
            ref = (source_info.width, source_info.height)
            dist = (distorted_info.width, distorted_info.height)

        if ref == dist:
            return "", (
                f"Reference and test video are compared at the same "
                f"{ref[0]}x{ref[1]} -- no scaling needed."
            )

        if run is None and row_data.options.crop_mode != CropMode.NONE and (
            ref[0] == dist[0] or ref[1] == dist[1]
        ):
            # One dimension already matches and only the other differs, which
            # is exactly what a letterbox or pillarbox looks like. Black bars
            # are detected when the row runs, and removing them may well
            # leave the two the same size, so there is nothing to claim yet.
            return "Pending", (
                f"The reference is {source_info.width}x{source_info.height} and the test "
                f"video {distorted_info.width}x{distorted_info.height}, a difference in one "
                "dimension only -- the shape of black bars on one of them.\n\n"
                "Bars are detected when this row runs, and are removed before the two are "
                "compared, so whether any scaling is needed is known then."
            )

        if row_data.scale_direction_pinned:
            direction = row_data.options.scale_direction
        else:
            direction = (
                run.result.scale_direction if run is not None
                else row_data.options.scale_direction
            )
        cropped = " (after black bars)" if (
            ref != (source_info.width, source_info.height)
            or dist != (distorted_info.width, distorted_info.height)
        ) else ""
        if direction == ScaleDirection.DISTORTED_TO_SOURCE:
            return "Test upscaled to source", (
                f"Test video upscaled {dist[0]}x{dist[1]} -> "
                f"{ref[0]}x{ref[1]}{cropped} to match the reference."
            )
        return "\u2193 source", (
            f"Reference downscaled {ref[0]}x{ref[1]} -> "
            f"{dist[0]}x{dist[1]}{cropped} to match the test video."
        )

    def _set_row_scaling(self, row: int) -> None:
        item = self.distorted_table.item(row, COL_SCALING)
        if item is None:
            return
        tag, explanation = self._resize_mismatch(row)
        item.setText(tag)
        item.setToolTip(explanation)
        self.distorted_table.resizeColumnToContents(COL_SCALING)

    def _set_row_info(self, row: int, info: VideoInfo | None, error: str | None = None) -> None:
        item = self.distorted_table.item(row, COL_INFO)
        scaling_item = self.distorted_table.item(row, COL_SCALING)
        if error:
            self._set_row_status(row, "Failed", error)
            item.setText("Probe failed")
            item.setToolTip(error)
            item.setForeground(Qt.red)
            self.distorted_table.item(row, COL_BITRATE).setText("")
            scaling_item.setText("")
            scaling_item.setToolTip("")
            self._set_row_black_bars(row, probe_failed=True)
        else:
            item.setText(media_info_string(info))
            # Back to normal text: the placeholder shown while probing greys
            # this cell out, and leaving it grey makes a probed row look
            # disabled.
            item.setForeground(self.distorted_table.palette().text())
            item.setToolTip(format_hms(info.duration, decimals=1))
            self.distorted_table.item(row, COL_BITRATE).setText(bitrate_string(info))
            self._rows[row].video_info = info
            self._set_row_scaling(row)
            if self._rows[row].analysis_status == "Reading...":
                self._rows[row].analysis_status = ""
                self._set_row_metrics(row)
            self._set_row_black_bars(row)
        # Keeps these snug to whatever's actually in them (never wider than
        # needed) while staying user-draggable in between updates.
        self.distorted_table.resizeColumnToContents(COL_INFO)
        self.distorted_table.resizeColumnToContents(COL_BITRATE)
        self.distorted_table.resizeColumnToContents(COL_SCALING)

    def _set_row_vmaf_text(self, row: int, text: str, *, bold: bool = False, color=None) -> None:
        item = self.distorted_table.item(row, COL_VMAF)
        # Text in this cell replaces the tick box, the same way a score does:
        # a check indicator beside "Frame 900/1200" reads as something to
        # click, and it is not.
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(Qt.CheckStateRole, None)
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
        self._set_row_black_bars(row)
        self.distorted_table.resizeColumnToContents(COL_INFO)
        self.distorted_table.resizeColumnToContents(COL_BITRATE)

    def _on_add_resample_test(self) -> None:
        if self._source_info is None:
            QMessageBox.warning(self, "No reference", "Please select a reference video first.")
            return

        targets = [
            target for target in RESAMPLE_TARGET_CHOICES
            if target.width < self._source_info.width
        ]
        if not targets:
            QMessageBox.information(
                self, "No smaller resolution available",
                f"The smallest resolution test is {RESAMPLE_TARGET_CHOICES[-1].width} pixels wide, "
                f"which is not below this reference's {self._source_info.width}-pixel width.",
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
                self, "Already added", f"A {label} resolution test for this reference is already in the list."
            )
            return

        row = self._add_table_row(synthetic_path)
        # A round-trip test decodes only the reference; the "distorted" side is
        # synthesised in the filtergraph. The reference is therefore the file
        # whose identity the cache must follow, and the target resolution is
        # already part of the options.
        self._rows[row].media_path = self._source_info.path
        self._rows[row].options.resample_test = target
        self._rows[row].video_info = self._source_info
        self._set_resample_row_info(row, target)
        # Redrawn now it is a resolution test: drawn when added, its
        # SSIMULACRA2/Butteraugli/CVVDP cells showed tick boxes instead of
        # n/a until something else happened to redraw the row.
        self._set_row_metrics(row)
        self._try_load_cached_result(row)

    def _on_add_distorted(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select test videos")
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
        self._start_media_probe(new_paths)
        self._start_cache_lookup(new_paths)

    def _reload_cached_for_all_rows(self) -> None:
        """Re-checks every row against the current source, in the background.

        Called when the reference changes: which cached result applies depends
        on the (source, distorted) pair, so every row's score may now be
        different -- or gone.
        """
        for row in range(len(self._rows)):
            self._set_row_metrics(row)
        self._start_cache_lookup([r.path for r in self._rows])

    def _reload_cached_for_rows(self, rows: list[int]) -> None:
        """Checks newly-selected score settings without blocking the UI.

        Cache identity includes the calculation options. When a user returns
        to a model/subsample/crop/metric combination they already ran, that
        result should reappear instead of requiring another feature-length
        computation.
        """
        if self._source_info is None or not self._settings.use_cache:
            return
        paths = list(dict.fromkeys(self._rows[row].path for row in rows))
        if paths:
            self._start_cache_lookup(paths)

    def _set_row_status(self, row: int, text: str, detail: str = "") -> None:
        self._rows[row].analysis_status = text
        self._rows[row].status_detail = detail
        self._refresh_row_state(row)

    def _start_media_probe(self, paths: list[Path]) -> None:
        """Reads media info for `paths` in the background.

        Deliberately cancels nothing. A probe answers a question about one
        file -- its resolution, frame rate, codec -- and that answer stays
        true whatever the user does next, so there is never a reason to throw
        one away. Cancelling media probes to start some other piece of
        background work is what stranded rows on "Reading..." forever.
        """
        if not paths:
            return
        worker = ProbeWorker(paths, None, False, {}, probe_media=True)
        self._probe_workers.append(worker)
        # No generation guard: the result describes the file, not the state
        # of the window when it was asked for. _on_probed drops it only if
        # the row has since been removed.
        worker.probed.connect(self._on_probed)
        worker.finished_all.connect(lambda w=worker: self._on_probe_finished(worker=w))
        self.status_label.setText(f"Reading {len(paths)} video(s)...")
        worker.start()

    def _start_cache_lookup(self, paths: list[Path]) -> None:
        """Looks for cached results for `paths`, without disturbing probing.

        This lane IS generation-guarded, and does cancel its predecessor:
        which cached result applies depends on the current source and the
        row's current options, so an answer computed against superseded
        state must not be shown.
        """
        if not paths or self._source_info is None or not self._settings.use_cache:
            return
        # The lookup being replaced may not have answered for all its rows
        # yet, and its answers are about to be ignored: those rows are asked
        # again here. Only the changed rows used to be, so editing one row
        # while saved results were loading left the others empty, to be
        # recalculated in full by the next run.
        paths = list(dict.fromkeys([*self._cancel_cache_lookup(), *paths]))
        self._cache_generation += 1
        generation = self._cache_generation
        cache_rows = [rd for rd in self._rows if rd.path in paths]
        paths = [rd.path for rd in cache_rows]  # rows removed meanwhile are not asked about
        if not paths:
            return
        self._cache_lookup_paths = paths
        worker = ProbeWorker(
            paths, self._source_info.path, True,
            {
                rd.path: self._analysis_request(rd, clone_options(rd.options))
                for rd in cache_rows
            },
            cache_paths={rd.path: rd.identity_path for rd in cache_rows},
            cache_supplemental={
                rd.path: displayable_metric_specs(clone_options(rd.options), rd.cvvdp)
                for rd in cache_rows
            },
            probe_media=False,
        )
        self._cache_worker = worker
        self._probe_workers.append(worker)
        worker.cached_found.connect(
            lambda path, result, label, key, g=generation:
            self._on_cached_if_current(g, path, result, label, key)
        )
        worker.other_cvvdp_found.connect(
            lambda path, others, parameters, g=generation:
            self._on_other_cvvdp_found(g, path, others, parameters)
        )
        worker.finished_all.connect(
            lambda g=generation, w=worker: self._on_cache_lookup_finished(g, w)
        )
        worker.start()

    def _on_other_cvvdp_found(self, generation: int, path: Path, others: list, parameters) -> None:
        if generation != self._cache_generation:
            return
        row = self._row_index_of_path(path)
        if row is None or tuple(self._rows[row].cvvdp.spec_parameters()) != tuple(parameters):
            return  # the row's display changed meanwhile; its own lookup follows
        self._rows[row].cvvdp_elsewhere = list(others)
        self._set_row_metrics(row)

    def _cancel_cache_lookup(self) -> list[Path]:
        """Stops the running cache lookup, if any; returns the rows it was
        asked about, whose answers will now be ignored.

        "Running" lasts until its finished_all signal has been handled
        (_on_cache_lookup_finished clears _cache_worker), not until its
        thread exits: answers it already sent can still be queued for this
        thread, and bumping the generation discards them. Checking only
        isRunning() lost those rows' saved scores when a row was edited in
        that window."""
        if self._cache_worker is None:
            return []
        # The thread stays referenced in _probe_workers until it exits;
        # dropping the only reference could destroy a running QThread.
        self._cache_worker.cancel()
        interrupted = list(self._cache_lookup_paths)
        self._cache_worker, self._cache_lookup_paths = None, []
        return interrupted

    def _on_cache_lookup_finished(self, generation: int, worker: ProbeWorker) -> None:
        if worker is self._cache_worker:
            self._cache_worker = None
        self._on_probe_finished(worker=worker)

    def _on_cached_if_current(
        self, generation: int, path: Path, result, label: str, key: str
    ) -> None:
        if generation != self._cache_generation or self._source_info is None:
            return
        row = self._row_index_of_path(path)
        if row is None:
            return
        current_key = result_cache.cache_key(
            self._source_info.path, self._rows[row].identity_path,
            self._analysis_request(self._rows[row]),
        )
        if key == current_key:
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
            # The cached answer replaces what the row shows when it has
            # everything the row already shows and more -- ticked or not,
            # since saved scores show whatever is ticked. It used to have to
            # hold every requested metric, so one never calculated (CVVDP
            # ticked beside cached SSIMULACRA2) threw away the rest of it:
            # the row kept showing VMAF only and SSIMULACRA2/Butteraugli
            # never appeared.
            existing = row_data.completed_run.result
            shown = set(existing.metric_results.keys())
            cached = set(result.metric_results.keys())
            if not cached - shown:
                return
            # Scores the row shows that the answer lacks are kept beside it.
            # It used to have to hold them all, and a row showing a GPU
            # SSIMULACRA2 score, then set to CPU, is never given that score
            # back by the cache: every later answer was thrown away, such as
            # the saved CVVDP score of a display switched to.
            if missing := shown - cached:
                result = copy.copy(result)
                result.metric_results = MetricResultSet([
                    *(result.metric_results.get(key) for key in result.metric_results),
                    *(existing.metric_results.get(key) for key in missing),
                ])
                result.frames = frame_scores_from_results(result.metric_results)
        previous = row_data.completed_run
        run = CompletedRun(result, label)
        if previous is not None:
            # The row's graph series is updated in place: its colour and
            # its hidden or removed state stay. Replacing it under a new
            # identity reset them whenever a display switched back to had a
            # saved CVVDP score.
            run.graph_identity = previous.graph_identity
        row_data.completed_run = run
        row_data.analysis_status = ""
        row_data.analysis_status = "Complete (cached)" if self._has_requested_results(row_data) else ""
        # Old cache files predate the persisted frame-preview recipe. The
        # cache key still identifies these exact row options, so restore the
        # missing pieces from the row that found the cache entry.
        result.scale_algorithm = row_data.options.scale_algorithm
        result.resample_target = row_data.options.resample_test
        # Scores/crops come from the cache, current media descriptors do not.
        # Older runs omitted HDR tags; replacing a fresh probe with that
        # snapshot silently disabled tone mapping in Frame Compare.
        self._set_row_info(row, row_data.video_info or result.distorted_info)
        self._set_row_metrics(row)
        # Keep the graph in step as results land, so opening the tab shows
        # everything without any further action.
        self.graph_panel.add_run(
            result, label, identity=run.graph_identity, restore=previous is None
        )
        self._sync_frame_compare()

    def _on_probe_finished(
        self, generation: int | None = None, worker: ProbeWorker | None = None
    ) -> None:
        if worker is not None:
            if worker in self._probe_workers:
                self._probe_workers.remove(worker)
            worker.deleteLater()
        if any(w.isRunning() for w in self._probe_workers):
            return  # another lane is still reading, so this is not done yet
        if not self._run_active:
            # A run owns the status line while it lasts; overwriting it with
            # "Ready." made a live job look finished.
            self._show_ready()
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
        self._sync_frame_compare()

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
        self._sync_frame_compare()

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
            self._source_info.path, row_data.identity_path,
            self._analysis_request(row_data),
            supplemental_specs=displayable_metric_specs(row_data.options, row_data.cvvdp),
        )
        if cached is None:
            return False
        result, label = cached
        result.scale_algorithm = row_data.options.scale_algorithm
        result.resample_target = row_data.options.resample_test
        run = CompletedRun(result, label)
        row_data.completed_run = run
        row_data.analysis_status = ""
        row_data.video_info = result.distorted_info
        if row_data.options.resample_test is not None:
            self._set_resample_row_info(row, row_data.options.resample_test)
        else:
            self._set_row_info(row, result.distorted_info)
        row_data.status_detail = (
            f"{len(result.frames)} scored frames; metrics: "
            + ", ".join(metric.label for metric in METRICS if result.has_metric(metric.key))
            + "\nLoaded from a previous run (matching files and calculation settings) -- "
            "right-click to recompute."
        )
        self._set_row_metrics(row)
        self._sync_frame_compare()
        return True

    def _on_table_context_menu(self, pos) -> None:
        if self._run_active:
            # Its only entry throws away cached results and re-queues rows,
            # which would fight the run currently using them.
            return
        rows = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)
        recompute_action = menu.addAction("Recalculate selected metrics (ignore cached/previous results)")
        chosen = menu.exec(self.distorted_table.viewport().mapToGlobal(pos))
        if chosen == recompute_action:
            self._recompute_rows(rows)

    def _recompute_rows(self, rows: list[int]) -> None:
        # Captured now, not when each queued deletion runs -- see
        # result_cache.store's note on why the folder cannot be resolved late.
        cache_directory = result_cache.cache_dir()
        # A cache read already in flight must not put back the exact result
        # the user just asked to ignore -- but the other rows it was loading
        # are asked about again below.
        interrupted = self._cancel_cache_lookup()
        self._cache_generation += 1
        recomputed = {self._rows[row].path for row in rows}
        for row in rows:
            row_data = self._rows[row]
            # Only the metrics being recalculated leave the row: saved scores
            # of unticked metrics are neither recalculated nor deleted, and
            # clearing the whole row hid them until the video was re-added.
            row_data.analysis_status = ""
            self._drop_metric_results(row, set(self._requested_metrics(row_data)))
            row_data.status_detail = ""
            if self._source_info is not None:
                self._file_writes.submit(
                    f"clear cached result for {row_data.path.name}",
                    partial(
                        result_cache.clear,
                        self._source_info.path, row_data.identity_path,
                        # The ticked metrics only. Unticked ones whose saved
                        # scores are on show are not being recalculated, so
                        # their scores are not deleted either; passing every
                        # FFmpeg metric here deleted a shown, unticked PSNR.
                        self._analysis_request(row_data, clone_options(row_data.options)),
                        cache_directory,
                    ),
                )
            # Refreshes the resize-mismatch note (Info column) back to the
            # row's *current* settings -- without this it kept showing
            # whatever the just-cleared run had actually used until the next
            # run finished, which is stale/misleading in between.
            if row_data.video_info is not None and row_data.options.resample_test is None:
                self._set_row_info(row, row_data.video_info)
        self.status_label.setText(
            f"Cleared {len(rows)} result(s) -- make sure they're checked, then click Calculate metrics to recompute."
        )
        self._sync_frame_compare()
        others = [path for path in interrupted if path not in recomputed]
        if others:
            self._start_cache_lookup(others)

    def _add_opposite_scale_direction_rows(self, rows: list[int]) -> None:
        """For each selected row with a mismatched resolution, adds a second
        row for the *same* distorted file with the opposite ScaleDirection,
        so both "scale source down" and "scale distorted up" can be run and
        compared side by side instead of having to pick one and re-run to
        see the other.
        """
        if self._source_info is None:
            QMessageBox.warning(self, "No reference", "Please select a reference video first.")
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
            # The same comparison in the other direction: the same metric
            # ticks, GPU/CPU choices and CVVDP display. It used to take the
            # defaults for the ticks and choices, so a video set to CPU
            # SSIMULACRA2 with CVVDP ticked got a companion without CVVDP
            # and with SSIMULACRA2 on the GPU.
            new_row_data.extra_metric_keys = set(row_data.extra_metric_keys)
            new_row_data.metric_backends = dict(row_data.metric_backends)
            new_row_data.cvvdp = row_data.cvvdp
            new_row_data.cvvdp_preset = row_data.cvvdp_preset
            new_row_data.scale_direction_pinned = True
            # The companion decodes the SAME file as the row it came from;
            # only the scale direction differs, and that is already part of
            # the cache identity through the options.
            new_row_data.media_path = row_data.identity_path
            new_row_data.video_info = info
            self._set_row_info(new_row, info)
            # Redrawn with the copied choices: the row was drawn with the
            # defaults when added, and its cells kept showing those (every
            # metric ticked) while a run followed the copied ones.
            self._set_row_metrics(new_row)
            self._try_load_cached_result(new_row)
            added += 1

        if added == 0:
            QMessageBox.information(
                self, "Nothing to add",
                "Selected row(s) either already have a matching-opposite row, don't have a resolution "
                "mismatch against the reference, or aren't a normal comparison (e.g. a resolution test)."
            )
        else:
            self.status_label.setText(
                f"Added {added} row(s) testing the opposite scaling direction -- check them and click Calculate metrics."
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
            self._syncing_panel = True
            try:
                self.ssimulacra2_backend_combo.setCurrentIndex(
                    0 if self._default_metric_backends["ssimulacra2"] == "gpu" else 1
                )
                self.butteraugli_backend_combo.setCurrentIndex(
                    0 if self._default_metric_backends["butteraugli"] == "gpu" else 1
                )
            finally:
                self._syncing_panel = False
            return

        # A run owns these settings until it finishes. This is reached from
        # background completions as well as from the user clicking a row --
        # a probe finishing mid-run used to re-enable the whole panel, and
        # anything changed there was then attached to a result computed with
        # the previous settings.
        self.options_box.setEnabled(not self._run_active)
        if len(rows) == 1:
            self.panel_target_label.setText(f"Editing settings for: {self._rows[rows[0]].path.name}")
        else:
            self.panel_target_label.setText(
                f"Editing settings for {len(rows)} selected videos -- editing anything below applies to all of them."
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

            selected_backends = (
                self._rows[self._panel_target_rows[0]].metric_backends
                if self._panel_target_rows else self._default_metric_backends
            )
            self.ssimulacra2_backend_combo.setCurrentIndex(
                0 if selected_backends.get("ssimulacra2", "gpu") == "gpu" else 1
            )
            self.butteraugli_backend_combo.setCurrentIndex(
                0 if selected_backends.get("butteraugli", "gpu") == "gpu" else 1
            )

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

            # Keep the global header shortcuts and selected-row inspector
            # consistent without firing their write-back signals.
            for metric_column in _METRIC_COLUMNS:
                enabled = (
                    metric_column.key in self._requested_metrics(self._rows[self._panel_target_rows[0]])
                    if self._panel_target_rows else self._default_metric_ticked(metric_column.column)
                )
                self.metric_header.set_checked(metric_column.column, enabled)
            selected_options = [self._rows[r].options for r in self._panel_target_rows] or [opts]
            self.model_combo.setEnabled(any(o.compute_vmaf for o in selected_options))
            uses_libvmaf = any(o.compute_vmaf or o.compute_vmaf_neg or o.extra_features for o in selected_options)
            self.threads_spin.setEnabled(uses_libvmaf)
            # Also while it is above 1: CVVDP is unavailable then, and the
            # way back to it must not be a disabled control.
            self.subsample_spin.setEnabled(uses_libvmaf or any(o.n_subsample > 1 for o in selected_options))
            self._show_cvvdp_settings(
                [self._rows[r].cvvdp for r in self._panel_target_rows] or [self._default_cvvdp],
                self._rows[self._panel_target_rows[0]].cvvdp_preset if self._panel_target_rows
                else self._settings.cvvdp_default_preset,
            )
        finally:
            self._syncing_panel = False

    def _show_cvvdp_settings(self, selected: list[CvvdpSettings], preferred: str = "") -> None:
        """Shows the selected rows' CVVDP settings in the Options panel: the
        preset they match (or "Custom"), the display in one line, and the
        resize box.

        Rows that differ show as "Mixed" in the dropdown and a partly ticked
        box. Showing only the first row's values made re-choosing the preset
        it already had (or clicking the box) do nothing for the other rows,
        or set them the wrong way round.
        """
        settings = selected[0]
        mixed_display = len({repr(s.display.identity()) for s in selected}) > 1
        mixed_resize = len({s.resize_to_display for s in selected}) > 1
        combo = self.cvvdp_preset_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            user_presets = self._settings.cvvdp_presets
            for preset in cvvdp_presets(user_presets):
                combo.addItem(preset.name, preset.name)
                combo.setItemData(
                    combo.count() - 1, preset.description or "Your saved preset.", Qt.ToolTipRole
                )
            match = None if mixed_display else matching_preset(settings, user_presets, preferred)
            if mixed_display:
                combo.addItem("Mixed (the selected videos use different displays)", None)
                combo.setCurrentIndex(combo.count() - 1)
            elif match is None:
                combo.addItem("Custom (not saved as a preset)", None)
                combo.setCurrentIndex(combo.count() - 1)
            else:
                combo.setCurrentIndex(combo.findData(match.name))
        finally:
            combo.blockSignals(False)
        self.cvvdp_delete_btn.setEnabled(match is not None and not match.builtin)
        self.cvvdp_display_label.setText(
            "The selected videos use different displays; choosing a preset sets them all."
            if mixed_display else settings.display.describe()
        )
        box = self.cvvdp_resize_check
        box.blockSignals(True)
        box.setTristate(mixed_resize)
        box.setCheckState(Qt.PartiallyChecked if mixed_resize
                          else Qt.Checked if settings.resize_to_display else Qt.Unchecked)
        box.blockSignals(False)

    def _read_panel_options(self) -> VmafOptions:
        extra_features = [
            metric.ffmpeg_binding.libvmaf_feature
            for metric in FRAME_METRICS
            if metric.ffmpeg_binding is not None
            and metric.ffmpeg_binding.libvmaf_feature is not None
            and self.metric_header.is_checked(next(item.column for item in _METRIC_COLUMNS if item.key == metric.key))
        ]

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
            compute_xpsnr=self.metric_header.is_checked(_METRIC_COLUMN_BY_INDEX[COL_XPSNR].column),
            compute_vmaf=self.metric_header.is_checked(_METRIC_COLUMN_BY_INDEX[COL_VMAF].column),
            compute_vmaf_neg=self.metric_header.is_checked(_METRIC_COLUMN_BY_INDEX[COL_VMAF_NEG].column),
            duration_limit=duration_limit,
            gpu_decode=self.gpu_checkbox.isChecked(),
            gpu_vendor=vendor,
            crop_mode=crop_mode,
        )

    def _analysis_request(self, row_data: RowData, options: VmafOptions | None = None,
                          cvvdp: CvvdpSettings | None = None):
        """The row's request, as runs and every cache lookup see it. One
        place, so the cache can never be asked with different settings from
        the ones a run is stored under."""
        return analysis_request_from_vmaf_options(
            options if options is not None else row_data.options,
            self._requested_metrics(row_data), row_data.metric_backends,
            cvvdp if cvvdp is not None else row_data.cvvdp,
        )

    def _default_metric_ticked(self, column: int) -> bool:
        """Whether new rows start with this metric ticked -- the header's
        tick when no row is selected. SSIMULACRA2, Butteraugli and CVVDP
        live outside VmafOptions, so reading only the options showed their
        header unticked even with the metric on by default, and the first
        header click did nothing visible."""
        key = _METRIC_COLUMN_BY_INDEX[column].key
        return self._metric_enabled(self._default_options, column) or key in self._default_extra_metric_keys

    @staticmethod
    def _metric_enabled(options: VmafOptions, column: int) -> bool:
        metric = _METRIC_COLUMN_BY_INDEX[column].key
        return metric_definition(metric).ffmpeg_binding is not None and options.metric_enabled(metric)

    @staticmethod
    def _set_metric_option(options: VmafOptions, column: int, checked: bool) -> None:
        metric = _METRIC_COLUMN_BY_INDEX[column].key
        if metric_definition(metric).ffmpeg_binding is None:
            raise ValueError(f"{metric} is not an FFmpeg option")
        options.set_metric_enabled(metric, checked)

    def _confirm_long_cpu_perceptual(self, job_rows: list[RowData]) -> bool:
        """Asks before CPU SSIMULACRA2/Butteraugli on videos over ten minutes.

        The CPU tools score one still-image pair at a time, 1-2 s per tool
        for a 4K pair: days for a film. Only metrics this run will actually
        calculate count -- one already cached is not rerun.

        A metric set to GPU counts too when no supported GPU was found: it
        will run on the CPU just the same, and used to do so without a word.
        """
        lines = []
        gpu_missing: bool | None = None  # probed once, and only if needed
        for row_data in job_rows:
            info = row_data.video_info
            if info is None:
                continue
            reusable = self._reusable_results(row_data)
            pending = [
                key for key in ("ssimulacra2", "butteraugli")
                if key in self._requested_metrics(row_data) and not reusable.has(key)
            ]
            if any(row_data.metric_backends.get(key) != "cpu" for key in pending) and gpu_missing is None:
                gpu_missing = not self._vship_available()
            cpu_keys = [
                key for key in pending
                if row_data.metric_backends.get(key) == "cpu" or gpu_missing
            ]
            cpu_metrics = [
                metric_definition(key).label + ("" if row_data.metric_backends.get(key) == "cpu"
                                                else " (set to GPU, but no supported GPU was found)")
                for key in cpu_keys
            ]
            if not cpu_metrics:
                continue
            seconds = min(info.duration, self._source_info.duration) if self._source_info else info.duration
            if row_data.options.duration_limit > 0:
                seconds = min(seconds, row_data.options.duration_limit)
            if seconds <= _CPU_PERCEPTUAL_WARNING_SECONDS:
                continue
            frames = seconds * (info.fps or 24.0) / max(1, row_data.options.n_subsample)
            megapixels = info.width * info.height / 1e6
            scoring = frames * megapixels * sum(_CPU_PERCEPTUAL_SECONDS_PER_MEGAPIXEL[key] for key in cpu_keys)
            lines.append(
                f"\u2022 {row_data.path.name}: {format_hms(seconds)}, {' and '.join(cpu_metrics)} "
                f"on CPU \u2014 {_rough_duration(scoring)} of scoring"
            )
        if not lines:
            return True
        answer = QMessageBox.warning(
            self, "CPU perceptual metrics on long videos",
            "Calculating SSIMULACRA2 or Butteraugli on the CPU is not recommended for "
            "videos longer than 10 minutes:\n\n" + "\n".join(lines) + "\n\n"
            "The CPU tools score one still image pair at a time: about 1 s for SSIMULACRA2 "
            "and 2 s for Butteraugli per 4K frame, so a film takes days (the estimates "
            "above assume a CPU like the one they were measured on). Choose GPU for these "
            "metrics, or set a duration limit.\n\n"
            "Calculate anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    @staticmethod
    def _vship_available() -> bool:
        """Whether GPU SSIMULACRA2/Butteraugli can run here. The probe is
        cached by perceptual_vship, so only the first call loads anything."""
        return perceptual_vship.detect_vship_device()[0] is not None

    @staticmethod
    def _selected_metrics(row_data: RowData) -> tuple[str, ...]:
        """The row's tick boxes, including metrics whose column is hidden."""
        requested = set(row_data.options.requested_metrics()) | row_data.extra_metric_keys
        return tuple(metric.key for metric in METRICS if metric.key in requested)

    def _requested_metrics(self, row_data: RowData) -> tuple[str, ...]:
        """What a run of this row calculates: its ticks, less hidden metrics
        and metrics the row's kind of comparison cannot produce.

        Hiding a metric takes it out of every run and every "already
        calculated?" check, but leaves the row's tick untouched, so showing
        the metric again restores the choice as it was.
        """
        return tuple(
            key for key in self._selected_metrics(row_data)
            if key not in self._hidden_metrics and self._metric_supported(row_data, key)
        )

    @staticmethod
    def _metric_supported(row_data: RowData, key: str) -> bool:
        """Whether this row's comparison can produce `key` at all."""
        return MainWindow._metric_unavailable_reason(row_data, key) is None

    @staticmethod
    def _metric_unavailable_reason(row_data: RowData, key: str) -> str | None:
        """Why this row cannot produce `key`, for its "n/a" cell; None if it can.

        Neither perceptual backend (Vship, libjxl CPU tools) implements the
        resolution round-trip test, which derives both sides from the source
        at run time. Requesting SSIMULACRA2/Butteraugli on such a row failed
        the whole job -- VMAF included -- with "do not support resolution
        round-trip tests yet".

        CVVDP also needs every frame (it judges each one together with the
        frames before it) and a supported GPU (it has no CPU implementation
        here).
        """
        label = metric_definition(key).label
        if row_data.options.resample_test is not None and metric_definition(key).backend_id == "perceptual":
            return (f"{label} is not available for resolution round-trip tests; the other "
                    "selected metrics are still calculated.")
        if key == "cvvdp":
            if row_data.options.n_subsample > 1:
                return ("CVVDP judges each frame together with the frames before it, so it "
                        "needs every frame: it is not available while libvmaf frame subsample "
                        "is above 1.")
            if not MainWindow._vship_available():
                return ("CVVDP is calculated on the GPU only, and no supported NVIDIA or AMD "
                        "GPU was found.")
        return None

    @classmethod
    def _row_metric_enabled(cls, row_data: RowData, column: int) -> bool:
        key = _METRIC_COLUMN_BY_INDEX[column].key
        return key in cls._selected_metrics(row_data)

    # ------------------------------------------------------ metrics picker
    def _apply_metric_visibility(self) -> None:
        for item in _METRIC_COLUMNS:
            self.distorted_table.setColumnHidden(item.column, item.key in self._hidden_metrics)

    def _show_metrics_menu(self, position) -> None:
        if self._run_active:
            return
        menu = _StayOpenMenu(self)
        menu.setToolTipsVisible(True)
        shown = [item for item in _METRIC_COLUMNS if item.key not in self._hidden_metrics]
        for item in _METRIC_COLUMNS:
            action = menu.addAction(item.metric.label)
            action.setCheckable(True)
            action.setChecked(item.key not in self._hidden_metrics)
            # The last visible metric cannot be hidden: a table with no
            # metrics could calculate nothing.
            if len(shown) == 1 and item in shown:
                action.setEnabled(False)
                action.setToolTip("At least one metric must stay shown.")
            action.toggled.connect(
                lambda checked, key=item.key, m=menu: self._on_metric_visibility_toggled(key, checked, m)
            )
        menu.exec(position)

    def _on_metric_visibility_toggled(self, key: str, shown: bool, menu: QMenu | None = None) -> None:
        if shown:
            self._hidden_metrics.discard(key)
        else:
            if len(self._hidden_metrics) + 1 >= len(_METRIC_COLUMNS):
                return
            self._hidden_metrics.add(key)
        self._settings.hidden_metrics = [item.key for item in _METRIC_COLUMNS if item.key in self._hidden_metrics]
        self._settings.save()
        self._apply_metric_visibility()
        if menu is not None:
            visible = [a for a in menu.actions() if a.isChecked()]
            for action in menu.actions():
                action.setEnabled(not (len(visible) == 1 and action.isChecked()))
        for row in range(len(self._rows)):
            self._set_row_metrics(row)
        if shown and self._rows:
            # Scores saved for the metric while it was hidden come back.
            self._reload_cached_for_rows(list(range(len(self._rows))))

    def _apply_metric_selection(
        self, rows: list[int], column: int, checked: bool, *, set_default: bool = True,
    ) -> None:
        """Turns one metric on or off for `rows`.

        `set_default` carries the choice to rows added later. True for the
        header shortcut, which is a statement about the whole table; False
        for a tick in one row's own cell, which says nothing about files
        that are not there yet.
        """
        if self._syncing_panel or self._run_active:
            return
        for row in rows:
            rd = self._rows[row]
            key = _METRIC_COLUMN_BY_INDEX[column].key
            if metric_definition(key).ffmpeg_binding is None:
                if checked:
                    rd.extra_metric_keys.add(key)
                else:
                    rd.extra_metric_keys.discard(key)
            else:
                self._set_metric_option(rd.options, column, checked)
            rd.analysis_status = ""
            # The existing scores remain valid: selecting another metric
            # changes the requested output, not the measured pictures.
            self._set_row_metrics(row)
        if set_default:
            key = _METRIC_COLUMN_BY_INDEX[column].key
            if metric_definition(key).ffmpeg_binding is None:
                if checked:
                    self._default_extra_metric_keys.add(key)
                else:
                    self._default_extra_metric_keys.discard(key)
            else:
                self._set_metric_option(self._default_options, column, checked)
        self._reload_cached_for_rows(rows)
        if self._panel_target_rows:
            self._write_panel_options(self._rows[self._panel_target_rows[0]].options)

    def _on_table_item_changed(self, item) -> None:
        """A metric tick box in the table was clicked.

        Applies to every selected row when the clicked row is one of them,
        so ticking PSNR across a selection is still a single click -- and to
        that row alone otherwise, which is what clicking a row you had not
        selected plainly means.
        """
        if self._syncing_table:
            return
        column = item.column()
        if column not in _METRIC_COLUMN_SET:
            return
        row = item.row()
        if row >= len(self._rows):
            # A row still being built: setItem fires this before the RowData
            # it describes exists. _set_row_metrics fills its boxes in after.
            return
        if not item.flags() & Qt.ItemIsUserCheckable:
            return  # a cell showing a score or live progress, not a tick box
        checked = item.checkState() == Qt.Checked
        if checked == self._row_metric_enabled(self._rows[row], column):
            # itemChanged also fires for text, colour and font edits. Only a
            # box that now disagrees with the row it stands for is a click.
            return
        if self._run_active:
            # Nothing may change mid-run; put the box back the way it was.
            self._set_row_metrics(row)
            return
        selected = sorted({idx.row() for idx in self.distorted_table.selectedIndexes()})
        rows = selected if row in selected else [row]
        self._apply_metric_selection(rows, column, checked, set_default=False)

    def _on_metric_column_toggled(self, column: int, checked: bool) -> None:
        """Header shortcuts explicitly apply to all rows; inspector to selection."""
        self._apply_metric_selection(list(range(len(self._rows))), column, checked)

    def _reusable_results(self, row_data: RowData) -> MetricResultSet:
        """The requested metrics the row already has and a run can keep.

        A metric counts when it has at least one score, produced the way
        the row asks for. SSIMULACRA2/Butteraugli on the GPU (Vship) and on
        the CPU (libjxl) differ by a few points on the same frames (44.47 vs
        46.89 in one 640x360 test), so a comparison mixing them ranks
        encodes on different scales:
        - set to CPU, only a CPU score counts;
        - set to GPU, a CPU score -- left by a fallback, or by the CPU
          choice earlier -- counts only when no supported GPU is present,
          where the CPU is the only way it can be calculated. With a GPU,
          the next run recalculates it there.
        """
        reusable = MetricResultSet()
        if row_data.completed_run is None:
            return reusable
        result = row_data.completed_run.result
        for key in self._requested_metrics(row_data):
            sequence = result.sequence_metric(key)
            if sequence is not None:
                # One score for the video. It is always the row's own:
                # changing the CVVDP display drops it (_drop_cvvdp_result).
                if np.isfinite(sequence.score):
                    reusable.add(sequence)
                continue
            metric = result.frame_metric(key)
            if metric is None or not np.any(~np.isnan(metric.values)):
                continue
            if not self._backend_matches(row_data, key, metric):
                continue
            reusable.add(metric)
        return reusable

    def _backend_matches(self, row_data: RowData, key: str, metric) -> bool:
        """Whether `metric`'s score was produced the way the row asks for it
        (see _reusable_results). Metrics without a GPU/CPU choice always do."""
        choice = row_data.metric_backends.get(key)
        produced = metric.provenance.compute_backend
        if choice == "cpu":
            return produced == "cpu"
        if choice == "gpu" and produced == "cpu":
            return not self._vship_available()
        return True

    def _backend_note(self, row_data: RowData, key: str, metric) -> str:
        """Tooltip text for a SSIMULACRA2/Butteraugli score produced the
        other way from the row's GPU/CPU choice -- the rare case marked
        "(CPU)"/"(GPU)" in the cell. A score produced the chosen way needs
        no note: nearly all of them are GPU scores on GPU rows."""
        choice = row_data.metric_backends.get(key)
        if choice is None or metric is None or self._backend_matches(row_data, key, metric):
            return ""
        produced = metric.provenance.compute_backend.upper()
        wanted = "GPU" if choice == "gpu" else "CPU"
        return (f"\n\nCalculated on the {produced}, but this row is set to {wanted}; the two give "
                f"different numbers, so the next run recalculates it on the {wanted}.")

    def _has_requested_results(self, row_data: RowData) -> bool:
        requested = self._requested_metrics(row_data)
        return bool(requested) and set(self._reusable_results(row_data).keys()) == set(requested)

    def _invalidate_completed_result(self, row: int) -> None:
        """Marks a row stale after an option that affects its run changes."""
        row_data = self._rows[row]
        row_data.analysis_status = ""
        if row_data.completed_run is None:
            self._set_row_metrics(row)
            return
        graph_identity = row_data.completed_run.graph_identity
        row_data.completed_run = None
        if not self.graph_panel.remove_by_identity(graph_identity):
            # Backward-compatible fallback for a series added directly by
            # path before row-scoped graph identities existed.
            self.graph_panel.remove_by_path(row_data.path)
        self._set_row_metrics(row)
        row_data.status_detail = ""
        self._sync_frame_compare()

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
        changed_rows = []
        for row in self._panel_target_rows:
            if self._rows[row].options != new_options:
                self._invalidate_completed_result(row)
                changed_rows.append(row)
            self._rows[row].options = clone_options(new_options)
            self._set_row_black_bars(row)
        self._reload_cached_for_rows(changed_rows)

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
        execution_only = field_name in {"gpu", "n_threads"}
        changed_rows = []
        for row in self._panel_target_rows:
            if apply(self._rows[row].options):
                if not execution_only:
                    self._invalidate_completed_result(row)
                    changed_rows.append(row)
                self._set_row_black_bars(row)
        self._reload_cached_for_rows(changed_rows)

    def _on_metric_backend_changed(self, metric_key: str) -> None:
        """Apply one standalone metric's compute preference without cache churn."""
        if self._syncing_panel:
            return
        combo = {
            "ssimulacra2": self.ssimulacra2_backend_combo,
            "butteraugli": self.butteraugli_backend_combo,
        }.get(metric_key)
        if combo is None:
            raise ValueError(f"Unknown standalone metric: {metric_key}")
        backend = "gpu" if combo.currentIndex() == 0 else "cpu"
        self._default_metric_backends[metric_key] = backend
        for row in self._panel_target_rows:
            self._rows[row].metric_backends[metric_key] = backend
        # The last choice is what the next session starts with, too.
        setattr(self._settings, f"default_{metric_key}_backend", backend)
        error = self._settings.save()
        if error:
            self.status_label.setText(error)

    # ------------------------------------------------------------------ CVVDP display
    def _apply_cvvdp(self, change, rows: list[int] | None = None, preset_name: str | None = None) -> None:
        """Gives the selected rows (or `rows`) new CVVDP settings:
        `change(old) -> new`, and `preset_name` as the preset they were given.

        Only CVVDP's score depends on them. A row keeps every other score,
        loses a CVVDP score made for the old settings, and gets back one
        already cached for the new settings, if there is one.
        """
        if self._syncing_panel or self._run_active:
            return
        changed = []
        for row in self._panel_target_rows if rows is None else rows:
            row_data = self._rows[row]
            new = change(row_data.cvvdp)
            if preset_name is not None and preset_name != row_data.cvvdp_preset:
                row_data.cvvdp_preset = preset_name
                self._set_row_metrics(row)  # the score's tooltip names the preset
            if new.same_as(row_data.cvvdp):
                continue
            row_data.cvvdp = new
            self._drop_cvvdp_result(row)
            changed.append(row)
        self._reload_cached_for_rows(changed)
        if self._panel_target_rows:
            self._write_panel_options(self._rows[self._panel_target_rows[0]].options)

    def _drop_cvvdp_result(self, row: int) -> None:
        """Removes the row's CVVDP score, keeping its other scores."""
        self._drop_metric_results(row, {"cvvdp"})

    def _drop_metric_results(self, row: int, keys: set[str]) -> None:
        """Removes some of the row's scores, keeping the rest on screen and
        on the graph. The result is copied, not edited: the graph and Video
        Compare hold the old one."""
        row_data = self._rows[row]
        run = row_data.completed_run
        if run is None or not any(run.result.has_metric(key) for key in keys):
            self._set_row_metrics(row)
            return
        kept = MetricResultSet(
            value for key in run.result.metric_results
            if key not in keys and (value := run.result.metric_results.get(key)) is not None
        )
        row_data.analysis_status = ""
        if kept:
            result = copy.copy(run.result)
            result.metric_results = kept
            result.frames = frame_scores_from_results(kept)
            row_data.completed_run = CompletedRun(result, run.label)
            # The same graph series, updated in place: re-adding it under a
            # new identity gave it a new colour and brought it back even if
            # it had been hidden or removed from the graph.
            row_data.completed_run.graph_identity = run.graph_identity
            self.graph_panel.add_run(result, run.label, identity=run.graph_identity, restore=False)
        else:
            if not self.graph_panel.remove_by_identity(run.graph_identity):
                self.graph_panel.remove_by_path(row_data.path)
            row_data.completed_run = None
            row_data.status_detail = ""
        self._set_row_metrics(row)
        self._sync_frame_compare()

    def _on_cvvdp_preset_chosen(self, index: int) -> None:
        if self._syncing_panel:
            return
        name = self.cvvdp_preset_combo.itemData(index)
        preset = preset_named(name, self._settings.cvvdp_presets) if name else None
        if preset is not None:
            # The display only: "Scale the video to fill the display" is
            # the video's own setting, and choosing a built-in preset used to
            # switch it off without a word.
            self._apply_cvvdp(lambda old: replace(old, display=preset.settings.display),
                              preset_name=preset.name)

    def _on_cvvdp_resize_toggled(self, checked: bool) -> None:
        self._apply_cvvdp(lambda old: replace(old, resize_to_display=checked))

    def _cvvdp_preset_names(self) -> frozenset[str]:
        return frozenset(preset.name for preset in cvvdp_presets(self._settings.cvvdp_presets))

    def _on_cvvdp_add_preset(self) -> None:
        """A new preset of the user's own, starting from the first selected
        row's display. Like every preset they save, it becomes the default
        for new videos: someone who tunes CVVDP will keep using it."""
        if not self._panel_target_rows or self._run_active:
            return
        settings = self._rows[self._panel_target_rows[0]].cvvdp
        dialog = CvvdpDisplayDialog(settings.display, self, new_preset=True,
                                    taken_names=self._cvvdp_preset_names())
        if dialog.exec() != QDialog.Accepted:
            return
        self._save_cvvdp_preset(dialog, settings, replacing=None, apply_to_rows=False)

    def _on_cvvdp_edit_display(self) -> None:
        """Edits the first selected row's display. Saving can rename and
        update the user's preset it came from, or keep it as a new one;
        "Apply without saving" only changes the selected rows."""
        if not self._panel_target_rows or self._run_active:
            return
        first = self._rows[self._panel_target_rows[0]]
        settings = first.cvvdp
        match = matching_preset(settings, self._settings.cvvdp_presets, first.cvvdp_preset)
        own = match.name if match is not None and not match.builtin else None
        dialog = CvvdpDisplayDialog(settings.display, self, own_preset=own,
                                    taken_names=self._cvvdp_preset_names())
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.action == "apply":
            display = dialog.display()
            self._apply_cvvdp(lambda old: replace(old, display=display))
            return
        self._save_cvvdp_preset(dialog, settings, replacing=own if dialog.action == "save" else None)

    def _save_cvvdp_preset(self, dialog: CvvdpDisplayDialog, settings: CvvdpSettings,
                           replacing: str | None, apply_to_rows: bool = True) -> None:
        """Saves the dialog's display as a user preset -- replacing (and so
        renaming) the preset `replacing` if given -- and, from the display
        editor, gives it to the selected rows. A new preset becomes the
        default for new videos; a renamed default stays the default under
        its new name.

        "Add preset..." does not touch the rows (apply_to_rows=False):
        making a preset is not choosing it, and applying it there dropped
        the selected video's CVVDP score.
        """
        name = dialog.preset_name()
        saved = CvvdpSettings(dialog.display())  # a preset is a display, not the resize choice
        presets = self._settings.cvvdp_presets
        # The other videos using the preset being replaced, found before it
        # changes. Only the selected ones used to be updated: the rest were
        # left with the old values and shown as "Custom", unasked.
        others = [] if replacing is None or not apply_to_rows else [
            row for row, row_data in enumerate(self._rows)
            if row not in self._panel_target_rows
            and (match := matching_preset(row_data.cvvdp, presets, row_data.cvvdp_preset)) is not None
            and match.name == replacing
        ]
        if replacing is not None:
            presets = without_user_preset(presets, replacing)
        try:
            self._settings.cvvdp_presets = with_user_preset(presets, name, saved)
        except ValueError as error:
            QMessageBox.warning(self, "CVVDP preset", str(error))
            return
        if replacing is None or self._settings.cvvdp_default_preset == replacing:
            self._settings.cvvdp_default_preset = name
        self._default_cvvdp = self._cvvdp_from_settings()
        error = self._settings.save()
        self._fill_cvvdp_default_combo()
        if apply_to_rows:
            self._apply_cvvdp(lambda old: replace(old, display=saved.display), preset_name=name)
        if others:
            if saved.display.identity() == settings.display.identity():
                # Renamed only: the videos keep their values, and the name.
                self._apply_cvvdp(lambda old: old, rows=others, preset_name=name)
            elif QMessageBox.question(
                self, "CVVDP preset",
                f'{len(others)} other video{"s" if len(others) != 1 else ""} use{"" if len(others) != 1 else "s"} '
                f'your preset "{replacing}". Update {"them" if len(others) != 1 else "it"} to the saved '
                "values too?\n\nA CVVDP score made for the old values is cleared; one already saved for "
                "the new values is shown instead. Choose No to keep the old values (shown as Custom).",
            ) == QMessageBox.Yes:
                self._apply_cvvdp(lambda old: replace(old, display=saved.display), rows=others, preset_name=name)
        # _apply_cvvdp redraws the panel only when a row changed; a rename
        # changes no row but does change the dropdown.
        if self._panel_target_rows:
            self._write_panel_options(self._rows[self._panel_target_rows[0]].options)
        if replacing is not None:
            done = (f'Saved your CVVDP preset "{name}"' if name == replacing
                    else f'Renamed your CVVDP preset "{replacing}" to "{name}" and saved it')
        elif apply_to_rows:
            done = f'Saved the CVVDP preset "{name}". Videos added from now on use it (change that in Settings)'
        else:
            done = (f'Saved the CVVDP preset "{name}". Videos added from now on use it (change that in '
                    "Settings); choose it in the CVVDP display list to use it for the selected videos")
        self.status_label.setText(error or done + ".")

    def _on_cvvdp_delete_preset(self) -> None:
        name = self.cvvdp_preset_combo.currentData()
        preset = preset_named(name, self._settings.cvvdp_presets) if name else None
        if preset is None or preset.builtin:
            return
        answer = QMessageBox.question(
            self, "Delete CVVDP preset",
            f'Delete your preset "{name}"?\n\nVideos using it keep their settings.',
        )
        if answer != QMessageBox.Yes:
            return
        self._settings.cvvdp_presets = without_user_preset(self._settings.cvvdp_presets, name)
        if self._settings.cvvdp_default_preset == name:
            self._settings.cvvdp_default_preset = ""
        self._default_cvvdp = self._cvvdp_from_settings()
        error = self._settings.save()
        self._fill_cvvdp_default_combo()
        if self._panel_target_rows:
            self._write_panel_options(self._rows[self._panel_target_rows[0]].options)
        self.status_label.setText(error or f'Deleted the CVVDP preset "{name}".')

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
        if self._source_probe_worker is not None and self._source_probe_worker.isRunning():
            QMessageBox.information(
                self, "Still reading reference", "Wait for the reference video to finish loading."
            )
            return
        if self._source_info is None:
            QMessageBox.warning(self, "No reference", "Please select a reference video.")
            return
        checked_rows = self._checked_rows()
        if not checked_rows:
            QMessageBox.warning(self, "No test videos", "Check at least one test video to calculate metrics.")
            return

        if any(not self._requested_metrics(self._rows[r]) for r in checked_rows):
            QMessageBox.warning(self, "No metrics selected", "Select at least one metric for every checked video, or uncheck videos you do not want to calculate.")
            return

        # Checking a row you already have a score for (e.g. it was checked
        # before you added more files) shouldn't silently recompute it --
        # skip rows with every requested metric. Right-click to explicitly
        # recalculate; merely rechecking a row never forces a redo.
        already_scored_rows = [r for r in checked_rows if self._has_requested_results(self._rows[r])]
        rows_to_run = [r for r in checked_rows if not self._has_requested_results(self._rows[r])]

        if not rows_to_run:
            self.status_label.setText(
                f"All {len(already_scored_rows)} checked video(s) already have all requested metrics -- nothing to run."
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
                still_reading = any(worker.isRunning() for worker in self._probe_workers)
                QMessageBox.information(
                    self,
                    "Still reading videos" if still_reading else "Unreadable video",
                    (
                        "Wait for every checked test video to finish loading before running."
                        if still_reading else
                        f"{row_data.path.name} could not be read. Remove it or add the file again to retry."
                    ),
                )
                return
            try:
                if row_data.options.resample_test is None:
                    validate_video_pair(self._source_info, dist_info, row_data.options)
                # Sized by what the run will actually compare, not by the
                # distorted file's own resolution -- one side is scaled to
                # the other before libvmaf sees it. The runner re-resolves
                # this after auto-crop, which it cannot know here.
                if row_data.options.resample_test is not None:
                    analysis_size = resample_analysis_dimensions(
                        self._source_info, row_data.options.manual_source_crop
                    )
                else:
                    analysis_size = analysis_dimensions(
                        self._source_info, dist_info, row_data.options,
                        row_data.options.manual_source_crop,
                        row_data.options.manual_distorted_crop,
                    )
                model = resolve_model(row_data.options, *analysis_size) if row_data.options.compute_vmaf else ""
            except (ValueError, VmafRunError) as e:
                QMessageBox.warning(self, "Invalid options", f"{row_data.path.name}: {e}")
                return
            job_options = replace(row_data.options, model=model)
            jobs.append(VmafJob(
                self._source_info, dist_info, job_options, label=row_data.path.stem,
                result_distorted_path=row_data.path, metric_keys=self._requested_metrics(row_data),
                metric_backends=dict(row_data.metric_backends),
                cvvdp=row_data.cvvdp,
                # Only a result for the row's current settings is ever
                # attached (a settings change clears completed_run), so its
                # scores belong to this exact recipe.
                cached_result=row_data.completed_run.result if row_data.completed_run else None,
                cached_metrics=self._reusable_results(row_data),
            ))
            job_rows.append(row_data)
            # A resample test's output timeline is driven by the reference (see
            # run_resample_test), not this row's (synthetic) "distorted" info.
            reference_for_frames = self._source_info if job_options.resample_test is not None else dist_info
            job_total_frames.append(estimate_total_frames(reference_for_frames, job_options))

        if not jobs:
            return
        if not self._confirm_long_cpu_perceptual(job_rows):
            return

        self._job_rows = job_rows
        for rd in job_rows:
            self._set_row_status(self._row_index_of(rd), "Queued")
        self._job_total_frames = job_total_frames
        self._job_cache_options = [clone_options(rd.options) for rd in job_rows]
        self._job_cvvdp = [rd.cvvdp for rd in job_rows]
        # Held as RowData, not indices, so removing a row mid-run can't
        # silently repoint these at a different row.
        self._checked_rows_for_run = [self._rows[r] for r in checked_rows]
        self._job_frames_done = {}
        self._job_fps = {}
        self._job_decode_status.clear()
        self._job_halves.clear()
        self._job_task_progress.clear()
        self._job_gpu_fallback.clear()
        self._job_gpu_metrics = {
            index for index, row in enumerate(job_rows)
            if any(
                metric_definition(key).backend_id == "perceptual"
                and (key == "cvvdp" or row.metric_backends.get(key, "gpu") == "gpu")
                for key in self._requested_metrics(row)
            )
        }
        self._running_jobs = []
        self._finished_jobs = set()
        self._job_line_slot = {}
        for line in self.job_progress_labels:
            line.setVisible(False)
            line.clear()
        self._run_failed_count = 0
        self._run_was_cancelled = False
        self._run_started_at = time.monotonic()
        self._run_elapsed_timer.start()
        if already_scored_rows:
            self.status_label.setText(
                f"Skipping {len(already_scored_rows)} already-scored video(s); running {len(jobs)}..."
            )
        self._set_run_ui_active(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.cancel_btn.setEnabled(True)

        self._worker = VmafWorker(jobs, self._parallel_jobs(), self)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.halves.connect(self._job_halves.__setitem__)
        self._worker.task_progress.connect(self._on_task_progress)
        self._worker.progress.connect(self._on_job_progress)
        self._worker.status.connect(self._on_job_status)
        self._worker.job_finished.connect(self._on_job_finished)
        self._worker.job_failed.connect(self._on_job_failed)
        self._worker.job_partially_failed.connect(self._on_job_partially_failed)
        self._worker.cancelled.connect(self._on_run_cancelled)
        self._worker.all_finished.connect(self._on_all_finished)
        self._worker.start()

    def _set_run_ui_active(self, active: bool) -> None:
        """Freezes every input that can change the meaning of a live job."""
        self._run_active = active
        # The table itself stays live so the queue can be scrolled and read
        # while it runs. Nothing reachable through it can change a running
        # job: the metric tick boxes refuse edits mid-run (see
        # _on_table_item_changed), the per-row Calculate checkboxes are read
        # once when the run starts, and the context menu is suppressed below.
        for widget in self._file_action_widgets:
            widget.setEnabled(not active)
        self.options_box.setEnabled(not active and bool(self._panel_target_rows))
        self.tabs.setTabEnabled(TAB_SETTINGS, not active)
        self.run_btn.setEnabled(not active)
        self.pause_btn.setEnabled(active)
        self.cancel_btn.setEnabled(active)
        # Deliberately left enabled: it changes how fast the queue drains,
        # never what any result means, and it is during a run that someone
        # wants it.
        self.parallel_jobs_check.setEnabled(True)

    def _parallel_jobs(self) -> int:
        """How many videos may be scored at once, as a count."""
        return MAX_PARALLEL_JOBS if self.parallel_jobs_check.isChecked() else 1

    def _on_graph_metric_changed(self, metric: str) -> None:
        self._settings.graph_metric = metric
        self._settings.save()

    def _on_parallel_jobs_changed(self, _checked: bool) -> None:
        """Applies the choice now, and remembers it for next time.

        A run already in progress picks it up: ticking lets a waiting lane
        start the next video immediately, and unticking stops another from
        starting without interrupting anything already going.
        """
        count = self._parallel_jobs()
        self._settings.parallel_jobs = count
        self._settings.save()
        if self._worker is not None and self._worker.isRunning():
            self._worker.set_parallel_jobs(count)

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
        row = self._row_index_of(self._job_rows[index])
        if row is not None:
            self._set_row_status(row, "Calculating")
        if index not in self._running_jobs:
            self._running_jobs.append(index)
        self._job_frames_done.setdefault(index, 0)
        self._assign_progress_line(index, label)
        self._update_run_status()

    def _assign_progress_line(self, index: int, label: str) -> None:
        """Gives this job a line of its own, reusing one a finished job left."""
        if index in self._job_line_slot:
            return
        taken = set(self._job_line_slot.values())
        for slot, line in enumerate(self.job_progress_labels):
            if slot in taken:
                continue
            self._job_line_slot[index] = slot
            line.setText(f"{label} — starting…")
            line.setVisible(True)
            return

    def _mark_job_over(self, index: int) -> None:
        """Retires a job from the live figures.

        Its frame count is pinned to the job's estimate so queue progress
        does not stall on whatever the last progress line happened to say,
        and its rate is dropped so a finished job stops inflating the
        combined fps the queue ETA is computed from.
        """
        self._finished_jobs.add(index)
        self._job_fps.pop(index, None)
        self._job_decode_status.pop(index, None)
        self._job_halves.pop(index, None)
        self._job_task_progress.pop(index, None)
        self._job_gpu_fallback.discard(index)
        slot = self._job_line_slot.pop(index, None)
        if slot is not None:
            self.job_progress_labels[slot].setVisible(False)
        if 0 <= index < len(self._job_total_frames):
            self._job_frames_done[index] = self._job_total_frames[index]
        self._update_run_status()

    def _job_label(self, index: int) -> str:
        if 0 <= index < len(self._job_rows):
            return self._job_rows[index].path.stem
        return f"video {index + 1}"

    def _update_run_status(self) -> None:
        """How much is running, and when the queue ends.

        Deliberately nameless: every running video has its own line directly
        below carrying its name, percentage, rate and time left, so listing
        the names here said the same thing twice -- and with two long file
        names it was the longest line on screen for no information.
        """
        running = [i for i in self._running_jobs if i not in self._finished_jobs]
        total = len(self._job_rows)
        elapsed_text = (
            f"Elapsed: {format_hms(time.monotonic() - self._run_started_at)}"
            if self._run_started_at is not None else ""
        )
        if not running:
            if self._run_active and total:
                self.status_label.setText(
                    f"Queued {total} video(s)"
                    + (f"   ·   {elapsed_text}" if elapsed_text else "")
                )
            return
        if len(running) == 1:
            summary = f"Running {running[0] + 1} of {total}"
        else:
            summary = f"Running {len(running)} of {total} together"
        if self._gpu_metrics_in_run():
            self.status_label.setText(
                f"{summary}"
                + (f"   ·   {elapsed_text}" if elapsed_text else "")
                + "   ·   CPU metrics may run in parallel; GPU metric passes are sequential"
            )
            return
        seconds = self._queue_eta_seconds()
        eta = "calculating..." if seconds is None else format_hms(seconds)
        self.status_label.setText(
            f"{summary}"
            + (f"   ·   {elapsed_text}" if elapsed_text else "")
            + f"   ·   Queue ETA: {eta}"
        )

    def _gpu_metrics_in_run(self) -> bool:
        """Whether this run includes work serialized through the GPU pass."""
        return bool(
            self._job_gpu_metrics
            - self._finished_jobs
            - self._job_gpu_fallback
        )

    def _on_task_progress(self, index: int, snapshots: list[dict[str, object]]) -> None:
        """Store backend-specific progress before rendering a job line."""
        self._job_task_progress[index] = snapshots
        self._render_job_progress(index)

    def _task_display_label(self, index: int, task: dict[str, object]) -> str:
        keys = tuple(task.get("metric_keys", ()))
        labels = "/".join(metric_definition(key).label for key in keys)
        backend = task.get("backend")
        if backend == "ffmpeg":
            return f"CPU metrics: {labels}"
        phase = task.get("phase")
        if phase:
            _number, _count, name = phase
            return f"GPU metric {phase[0]}/{phase[1]}: {name}"
        row = self._job_rows[index] if 0 <= index < len(self._job_rows) else None
        choices = [
            "cpu" if index in self._job_gpu_fallback else
            ("gpu" if key == "cvvdp" else (row.metric_backends.get(key, "gpu") if row else "gpu"))
            for key in keys
        ]
        if choices and all(choice == "gpu" for choice in choices):
            return f"GPU metrics: {labels}"
        if choices and all(choice == "cpu" for choice in choices):
            return f"CPU metrics: {labels}"
        return f"Perceptual metrics: {labels}"

    def _task_detail(self, index: int, task: dict[str, object], multiple: bool) -> str:
        label = self._task_display_label(index, task)
        state = task.get("state")
        if state == "waiting":
            return f"{label} waiting for the GPU"
        if state == "starting":
            return f"{label} starting"
        if state == "done":
            return f"{label} complete"
        current = int(task.get("current", 0) or 0)
        total = int(task.get("total", 0) or 0)
        fps = float(task.get("fps", 0.0) or 0.0)
        phase = task.get("phase")
        phase_total = total
        if phase and total > 0:
            # Vship reports a cumulative total across its serialized passes.
            phase_total = max(1, round(total / max(1, int(phase[1]))))
            current = max(0, current - (int(phase[0]) - 1) * phase_total)
            current = min(current, phase_total)
        pct = min(100.0, 100.0 * current / total) if total > 0 else 0.0
        if phase and phase_total > 0:
            pct = min(100.0, 100.0 * current / phase_total)
        parts = [label]
        if multiple or phase:
            parts.append(f"{pct:.1f}%")
        if fps > 0:
            parts.append(f"{fps:.1f} fps")
            if phase_total > 0:
                parts.append(f"{format_hms(max(0, phase_total - current) / fps)} left")
        return " ".join(parts)

    def _render_job_progress(self, index: int) -> None:
        slot = self._job_line_slot.get(index)
        if slot is None:
            return
        total = self._job_total_frames[index] if 0 <= index < len(self._job_total_frames) else 0
        current = self._job_frames_done.get(index, 0)
        pct = min(100, int(100 * current / total)) if total > 0 else 0
        parts = [f"{self._job_label(index)} — {pct}%"]
        if decode_status := self._job_decode_status.get(index):
            parts.append(decode_status)
        snapshots = self._job_task_progress.get(index, ())
        if snapshots:
            parts.extend(
                self._task_detail(index, task, len(snapshots) > 1)
                for task in snapshots
                if task.get("state") != "done" or len(snapshots) == 1
            )
        else:
            fps = self._job_fps.get(index, 0.0)
            if fps > 0:
                parts.extend((f"{fps:.1f} fps", f"{format_hms(max(0, total - current) / fps)} left"))
            else:
                parts.extend(self._halves_detail(index))
        self.job_progress_labels[slot].setText("   ·   ".join(parts))

    def _on_job_progress(self, index: int, current: int, total: int, fps: float) -> None:
        # Live progress (queued/starting/frame N of M/paused) belongs in the
        # status bar above -- it already shows the running file, fps and ETA
        # -- not the VMAF column, which is for the final score.
        self._job_frames_done[index] = current
        self._job_fps[index] = fps
        self._render_job_progress(index)
        self._update_run_status()

    def _halves_detail(self, index: int) -> list[str]:
        """For a video scored in two halves when one has not started: each
        half on its own. The video's percentage is the slower half's, and
        its rate and time left are unknown until both run, so the line said
        only "0%" -- for hours when the SSIMULACRA2/Butteraugli/CVVDP half
        waited for another video's GPU pass while VMAF was being calculated."""
        detail = []
        for labels, current, total, fps, state in self._job_halves.get(index, ()):
            if state == "waiting":
                detail.append(f"{labels} waiting for the GPU (another video is using it)")
            elif state == "starting":
                detail.append(f"{labels} starting")
            elif state == "running" and total > 0:
                pct = min(100.0, 100 * current / total)
                detail.append(f"{labels} {pct:.1f}%" + (f" at {fps:.1f} fps" if fps > 0 else ""))
        return detail if any(state in ("waiting", "starting") for *_rest, state in
                             self._job_halves.get(index, ())) else []

    def _queue_eta_seconds(self) -> float | None:
        """When the LAST video will finish, not when the work would be done
        if it divided evenly.

        Dividing the remaining frames by the summed frame rate assumes every
        lane stays busy until the same instant. Two jobs with 10 and 1000
        seconds left would be reported as about 505 seconds, when the queue
        plainly cannot end before the 1000-second one does. So this schedules
        the remaining work over the lanes and takes the longest lane.

        None while nothing has reported a rate yet -- there is no basis for a
        guess, and a wrong number is worse than none.
        """
        # Perceptual GPU work is a serialized sequence of metric passes, so a
        # video-level FPS cannot describe the queue. The UI reports each
        # current GPU pass on its own line instead of presenting a false ETA.
        if self._gpu_metrics_in_run():
            return None
        observed = [rate for rate in self._job_fps.values() if rate > 0]
        if not observed:
            return None
        # Queued videos have no rate of their own yet; the videos already
        # running are the only evidence available for how fast this machine
        # is getting through this material.
        reference_fps = sum(observed) / len(observed)

        running_seconds: list[float] = []
        queued_seconds: list[float] = []
        for job, job_total in enumerate(self._job_total_frames):
            if job in self._finished_jobs:
                continue
            remaining = max(0, job_total - self._job_frames_done.get(job, 0))
            rate = self._job_fps.get(job, 0.0)
            if rate > 0:
                running_seconds.append(remaining / rate)
            else:
                queued_seconds.append(remaining / reference_fps)

        # A lane per video already running (there can be more than the
        # current setting, if it was lowered mid-run), plus any idle lanes.
        lanes = list(running_seconds)
        lanes += [0.0] * max(0, self._parallel_jobs() - len(lanes))
        if not lanes:
            return 0.0
        # Longest first onto the earliest-free lane: the usual greedy
        # schedule, and it avoids the optimism of assuming a perfect split.
        for seconds in sorted(queued_seconds, reverse=True):
            lanes.sort()
            lanes[0] += seconds
        return max(lanes)

    def _on_job_status(self, index: int, message: str) -> None:
        """Phase messages belong to the video they came from.

        Putting them all in the one status line meant that with two videos
        running, whichever lane spoke last owned the label -- so "Detecting
        black bars..." or a GPU-decode fallback notice appeared with no way
        to tell which file it was about, and it erased the line naming what
        was running.
        """
        if (
            "using CPU" in message
            or "calculating it on the CPU" in message
            or "on CPU" in message
        ):
            self._job_gpu_fallback.add(index)
        slot = self._job_line_slot.get(index)
        if slot is not None:
            # The runner sends the active plan on each attempt, including
            # software fallbacks. Keep it when progress replaces this phase
            # message; each parallel job owns its own decode plan.
            marker = "(GPU decode: "
            if marker in message:
                plan = message.split(marker, 1)[1].split(")", 1)[0]
                self._job_decode_status[index] = (
                    "Decode: source CPU, test CPU" if plan == "off" else
                    "Decode: " + plan.replace("distorted", "test").replace("cpu", "CPU")
                )
            if index in self._job_task_progress:
                # Keep backend-specific rates and pass progress visible. A
                # generic phase message must not replace the useful task
                # snapshot, especially while a GPU pass is waiting.
                self._render_job_progress(index)
            else:
                self.job_progress_labels[slot].setText(
                    f"{self._job_label(index)} — {message.replace('distorted', 'test')}"
                )
        self._update_run_status()

    def _row_index_of(self, row_data: RowData) -> int | None:
        """The table row this RowData currently sits at, or None if it was
        removed while the run was in flight. Compared by identity, not `==`:
        two different rows can be field-equal (e.g. the same file added
        twice before probing), and list.index() would find the wrong one."""
        for i, rd in enumerate(self._rows):
            if rd is row_data:
                return i
        return None

    def _on_job_finished(self, index: int, result) -> int | None:
        """Shows and caches a finished job. Returns the row it was shown on,
        or None when it was not shown (row removed, source or settings
        changed since the job started)."""
        self._mark_job_over(index)
        row_data = self._job_rows[index]
        row = self._row_index_of(row_data)
        if row is None:
            return None  # the row was removed mid-run; nothing to write the result to
        label = row_data.path.stem
        # The job owns the reference/distorted identities it was launched with.
        # Never key a result from an old in-flight job using whatever source
        # happens to be selected by the time it finishes.
        cache_options = (
            self._job_cache_options[index]
            if index < len(self._job_cache_options) else row_data.options
        )
        cache_cvvdp = self._job_cvvdp[index] if index < len(self._job_cvvdp) else row_data.cvvdp
        # Off the UI thread: this is the ~9MB JSON write that used to
        # stall the window every time a run finished. Every argument is
        # plain data, so nothing the worker touches is a widget.
        self._file_writes.submit(
            f"cache {label}",
            partial(
                result_cache.store,
                # The key uses the row's REAL file, not result.distorted --
                # which for a synthetic row is a path that does not exist and
                # so carries no size or mtime to notice a replacement by.
                result.source, row_data.identity_path, result, label,
                self._analysis_request(row_data, cache_options, cache_cvvdp),
                result_cache.cache_dir(),
            ),
        )
        # Packet bitrate is useful alongside quality metrics and is much
        # cheaper than decoding VMAF. Add both physical streams to the
        # independent viewer; its queue deduplicates the reference when several
        # distorted jobs finish together. A resolution round trip has no
        # second file, so only its source is scanned.
        bitrate_infos = [result.source_info]
        if result.resample_target is None:
            bitrate_infos.append(result.distorted_info)
        self.bitrate_panel.add_and_analyze(bitrate_infos)
        if self._source_info is None or self._source_info.path != result.source:
            self._set_row_status(row, "Finished for the previous source; select it again to load the result.")
            return None
        if cache_options != row_data.options or not cache_cvvdp.same_as(row_data.cvvdp):
            # The row's settings changed after this job was launched, so the
            # result does not describe what the row now says. It is still
            # cached above under the options it really used -- returning to
            # them brings it straight back -- but showing it here would
            # label it with settings it was never computed with.
            self._set_row_status(
                row,
                "Finished with the previous settings; change them back to see the result.",
            )
            return None
        if row_data.completed_run is not None:
            previous = row_data.completed_run
            # Saved scores the row was showing for metrics this run did not
            # calculate (unticked ones) stay on show; the run's result held
            # only what it calculated, so they used to vanish. Merged into a
            # copy: the cache write queued above holds this result object.
            carried = MetricResultSet(
                value for key in previous.result.metric_results
                if not result.has_metric(key) and key not in self._requested_metrics(row_data)
                and (value := previous.result.metric(key)) is not None
            )
            if carried:
                result = copy.copy(result)
                result.merge_metric_results(carried)
            if not self.graph_panel.remove_by_identity(previous.graph_identity):
                self.graph_panel.remove_by_path(row_data.path)
        run = CompletedRun(result, label)
        row_data.completed_run = run
        row_data.analysis_status = ""
        if row_data.options.resample_test is None:
            self._set_row_info(row, result.distorted_info)  # refresh the resize-mismatch note against the actual run
        row_data.status_detail = (
            f"{len(result.frames)} scored frames; metrics: "
            + ", ".join(metric.label for metric in METRICS if result.has_metric(metric.key))
        )
        self._set_row_metrics(row)
        # Straight onto the graph: a run that has finished is a curve, and
        # waiting for a button press to see it serves nobody.
        self.graph_panel.add_run(
            result, label, identity=run.graph_identity
        )
        self._sync_frame_compare()
        return row

    def _on_job_partially_failed(self, index: int, result, message: str, stderr_tail: str) -> None:
        """One metric group failed, the other finished: the finished scores
        are shown and cached like any result, and the metrics that failed
        say so in their own cells."""
        self._run_failed_count += 1
        row = self._on_job_finished(index, result)
        if row is None:
            return
        self._set_row_status(
            row, _PARTLY_FAILED, f"{message}\n\n{stderr_tail}" if stderr_tail else message
        )
        self._set_row_metrics(row)

    def _on_job_failed(self, index: int, message: str, stderr_tail: str) -> None:
        self._mark_job_over(index)
        self._run_failed_count += 1
        row = self._row_index_of(self._job_rows[index])
        if row is None:
            return  # the row was removed mid-run
        self._set_row_status(
            row, "Failed", f"{message}\n\n{stderr_tail}" if stderr_tail else message
        )
        self._set_row_metrics(row)

    def _on_run_cancelled(self) -> None:
        self._run_was_cancelled = True

    def _on_all_finished(self) -> None:
        for rd in self._job_rows:
            row = self._row_index_of(rd)
            if row is not None and rd.analysis_status in {"Calculating", "Queued"}:
                rd.analysis_status = "Cancelled" if self._run_was_cancelled else ""
                self._set_row_metrics(row)
        self._run_elapsed_timer.stop()
        self._set_run_ui_active(False)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        for line in self.job_progress_labels:
            line.setVisible(False)
        if self._run_was_cancelled:
            self.status_label.setText("Cancelled.")
        elif self._run_failed_count:
            self.status_label.setText(
                f"Finished with {self._run_failed_count} failed video(s)."
            )
        else:
            self.status_label.setText("Done.")
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

    @staticmethod
    def _same_source(a: Path | None, b: Path | None) -> bool:
        """Whether two paths name the same reference video.

        Normalised, because a saved run records whatever path was used when
        it ran -- relative or absolute, either slash, any case on Windows --
        and a textual comparison would call the same file two different
        sources.
        """
        if a is None or b is None:
            return False
        try:
            return Path(a).resolve() == Path(b).resolve()
        except OSError:
            return Path(a) == Path(b)

    def _on_load_saved_run(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load analysis results", "", RESULT_FILE_FILTER)
        if not path:
            return
        try:
            result, label = load_run(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Failed to load run", str(e))
            return
        # A result belongs to a (source, distorted) PAIR. Dropping it into
        # the table under whatever source happens to be selected presents a
        # comparison that was never made -- the row showed one video's score
        # underneath a different reference.
        if self._source_info is None:
            # Nothing to contradict: adopt the run's own reference, so the
            # window and the result agree about what was compared.
            self._adopt_source_from_run(result)
        elif not self._same_source(self._source_info.path, result.source)                 and not self._offer_to_switch_source(result):
            return

        run = CompletedRun(result, label)
        row = self._add_table_row(result.distorted)
        row_data = self._rows[row]
        row_data.video_info = result.distorted_info
        row_data.completed_run = run
        row_data.analysis_status = ""
        # The optional-metric columns are driven by row options. Seed those
        # flags from the data that is actually present in the saved result,
        # otherwise valid PSNR/SSIM/XPSNR arrays render as "N/A".
        row_data.options.extra_features = []
        if result.frames.has("psnr"):
            row_data.options.extra_features.append("name=psnr")
        if result.frames.has("ssim"):
            row_data.options.extra_features.append("name=float_ssim")
        row_data.options.compute_xpsnr = result.frames.has("xpsnr")
        row_data.options.compute_vmaf = result.frames.has("vmaf")
        row_data.options.compute_vmaf_neg = result.frames.has("vmaf_neg")
        row_data.options.model = result.model
        # Preserve the model selection when a saved run is reopened.  Without
        # this, a bundled VMAF v1 result would appear in the table correctly
        # but a later cache lookup/recalculation would silently revert the row
        # to Auto (v0).  Older files have no model_choice, so infer the two
        # legacy forms where it is unambiguous.
        if result.model_choice:
            row_data.options.model_choice = result.model_choice
            if result.model_choice == CUSTOM_MODEL_CHOICE and result.model.startswith("path="):
                row_data.options.custom_model_path = result.model.removeprefix("path=")
        elif result.model.startswith("version="):
            row_data.options.model_choice = result.model
        elif result.model.startswith("path="):
            row_data.options.model_choice = CUSTOM_MODEL_CHOICE
            row_data.options.custom_model_path = result.model.removeprefix("path=")
        row_data.options.scale_direction = result.scale_direction
        row_data.options.scale_algorithm = result.scale_algorithm
        row_data.options.resample_test = result.resample_target
        # A CVVDP score is only meaningful with the display it was made for;
        # the row takes that display, so the score and the row agree.
        cvvdp_result = result.sequence_metric("cvvdp")
        if cvvdp_result is not None:
            row_data.extra_metric_keys.add("cvvdp")
            with contextlib.suppress(KeyError, TypeError, ValueError):
                row_data.cvvdp = CvvdpSettings.from_spec_parameters(cvvdp_result.provenance.parameters)
        if result.resample_target is not None:
            row_data.media_path = result.source_info.path
        self.distorted_table.item(row, COL_CHECK).setCheckState(Qt.Unchecked)
        self._set_row_info(row, result.distorted_info)
        self._set_row_metrics(row)
        self._rows[row].status_detail = "Loaded from saved run"
        self._refresh_row_state(row)
        self._sync_frame_compare()

    def _adopt_source_from_run(self, result) -> None:
        """Points the window at the reference a loaded run was measured
        against, so the two cannot disagree."""
        self._source_info = result.source_info
        self.source_edit.setText(str(result.source))
        info = result.source_info
        self.source_info_label.setText(
            f"{media_info_string(info)}, {bitrate_string(info)}  "
            f"({format_hms(info.duration, decimals=1)})  [from saved run]"
        )

    def _offer_to_switch_source(self, result) -> bool:
        """Asks before showing a run measured against a different reference.

        Returns whether to go ahead. Answering yes switches the window to
        the run's own source, which is the only arrangement in which the
        table and the result mean the same thing.
        """
        answer = QMessageBox.question(
            self, "Different reference video",
            f"This saved run was measured against:\n    {result.source}\n\n"
            f"but the selected reference is:\n    {self._source_info.path}\n\n"
            "Scores from two different references cannot be compared. "
            "Switch the reference to the one this run used?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        self._adopt_source_from_run(result)
        # Every other row was scored against the previous reference.
        for row in range(len(self._rows)):
            self._invalidate_completed_result(row)
        return True

    def _on_save_selected(self) -> None:
        runs = self._selected_runs()
        if not runs:
            QMessageBox.information(self, "Nothing selected", "Select one or more completed rows to save.")
            return
        if len(runs) == 1:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save analysis results", f"{runs[0].label}{RESULT_SUFFIX}",
                f"Analysis results (*{RESULT_SUFFIX})",
            )
            if path:
                self._submit_save(runs[0].result, Path(path), runs[0].label)
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose folder to save runs into")
        if not directory:
            return
        reserved: set[Path] = set()
        for run in runs:
            self._submit_save(
                run.result,
                unique_output_path(Path(directory), run.label, RESULT_SUFFIX, reserved),
                run.label,
            )

    def _submit_save(self, result, path: Path, label: str) -> None:
        # Only the action doing the writing is disabled, so the rest of the
        # window stays usable while a long save runs.
        self.save_btn.setEnabled(False)
        self._file_writes.submit(
            f"save {path.name}", partial(save_run, result, path, label=label)
        )

    def _on_file_writes_idle(self) -> None:
        self.save_btn.setEnabled(True)
        if self._cache_clear_result is not None:
            removed = sum(self._cache_clear_result)
            self._cache_clear_result = None
            self.settings_status.setText(f"Removed {removed} saved result(s).")
            self._refresh_settings_status()

    def _on_file_write_failed(self, description: str, error: str) -> None:
        # Reported in the status line rather than a modal: these finish in
        # the background, and a dialog stealing focus minutes later is worse
        # than the failure it announces.
        self.status_label.setText(f"Could not {description}: {error}")

    def _on_show_graph_clicked(self) -> None:
        # Syncs in every currently-completed row every time -- not just
        # whatever was completed the first time this was clicked -- so
        # checking back mid-run (e.g. 4 of 8 done) shows all 4, not just
        # however many were done the first time it was opened.
        all_runs = [r.completed_run for r in self._rows if r.completed_run]
        if not all_runs and not self.graph_panel._entries:
            QMessageBox.information(
                self, "No results yet", "Calculate metrics or load analysis results to view metric graphs."
            )
            return
        self._open_or_update_graph(all_runs)

    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_GRAPH:
            self._sync_graph()
        elif index == TAB_FRAME_COMPARE:
            self._sync_frame_compare()

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

    def _sync_frame_compare(self) -> None:
        """Makes Frame Compare mirror every Videos-table row it can render.

        Deliberately not limited to scored rows. Looking at a source and an
        encode side by side is useful in its own right -- to check framing,
        to find a scene worth measuring, to see whether an encode is even
        the same content -- and requiring a finished run first made the tab
        unavailable exactly when it is most useful: before committing to a
        feature-length calculation.
        """
        entries = []
        for row in self._rows:
            entry = self._frame_comparison_entry(row)
            if entry is not None:
                entries.append(entry)
        self.frame_compare_panel.set_runs(entries)

    def _frame_comparison_entry(self, row: RowData) -> FrameComparisonEntry | None:
        """One Frame Compare entry for a row, scored or not."""
        if row.completed_run is not None:
            # A finished run knows the geometry it really used, including
            # crops that were detected at run time.
            return FrameComparisonEntry(
                identity=row.completed_run.graph_identity,
                label=row.completed_run.label,
                comparison=replace(
                    FrameComparison.from_result(row.completed_run.result),
                    source_info=self._source_info or row.completed_run.result.source_info,
                    distorted_info=row.video_info or row.completed_run.result.distorted_info,
                ),
                scores=row.completed_run.result.frames,
            )

        if self._source_info is None:
            return None  # nothing to compare against yet
        options = row.options
        resample = options.resample_test
        if resample is None and row.video_info is None:
            return None  # the test file has not been read yet

        # Crops that a run would apply are only known once it has run:
        # auto-detection measures the video. Manual and "none" are known
        # now, so those are exact; auto is previewed uncropped and says so.
        auto_crop_pending = options.crop_mode == CropMode.AUTO
        source_crop = distorted_crop = None
        if options.crop_mode == CropMode.MANUAL:
            source_crop = options.manual_source_crop
            distorted_crop = options.manual_distorted_crop

        distorted_info = self._source_info if resample is not None else row.video_info
        reference = self._source_info if resample is not None else distorted_info
        other = None if resample is not None else self._source_info
        return FrameComparisonEntry(
            identity=row.frame_identity,
            label=row.path.stem,
            comparison=FrameComparison(
                source_info=self._source_info,
                distorted_info=distorted_info,
                source_crop=source_crop,
                distorted_crop=distorted_crop,
                scale_direction=options.scale_direction,
                scale_algorithm=options.scale_algorithm,
                resample_target=resample,
                fps=reference.fps,
                # The same bound a run would use, so the timeline cannot
                # offer frames past the end of the shorter input.
                frame_count=estimate_total_frames(reference, options, other),
                auto_crop_pending=auto_crop_pending,
            ),
        )

    def _open_or_update_graph(self, runs: list[CompletedRun]) -> None:
        """Adds runs to the graph tab and brings it to the front."""
        for run in runs:
            self.graph_panel.add_run(
                run.result, run.label, identity=run.graph_identity
            )
        self.tabs.setCurrentIndex(TAB_GRAPH)
