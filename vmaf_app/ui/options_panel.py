"""The Options panel of the Videos tab: the settings of the videos selected
in the table.

The panel shows a video's options (show_options, show_cvvdp) and reads
them back (read_options). It decides nothing about the rows itself: each
control the user edits is reported to the window with a signal, and the
window applies the change to the selected rows. Which metrics a video
calculates is not here: it is ticked in the table's own columns.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTime, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
)

from vmaf_app.core.builtin_models import builtin_choice
from vmaf_app.core.cvvdp import CvvdpSettings, matching_preset
from vmaf_app.core.cvvdp import presets as cvvdp_presets
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE
from vmaf_app.core.models import CropMode, GpuVendor, ScaleDirection, VmafOptions
from vmaf_app.i18n import N_, tr
from vmaf_app.ui import theme
from vmaf_app.ui.formatting import display_description, preset_label

# The VMAF v0.6.1 column's models. VMAF v1 has a column and a list of its
# own (V1_MODEL_CHOICES).
MODEL_CHOICES = [
    (N_("Auto (analysis resolution)"), AUTO_MODEL_CHOICE),
    (N_("VMAF v0.6.1 (standard viewing)"), "version=vmaf_v0.6.1"),
    (N_("VMAF 4K v0.6.1 (4K / large-screen viewing)"), "version=vmaf_4k_v0.6.1"),
    (N_("Custom model file..."), CUSTOM_MODEL_CHOICE),
]

V1_MODEL_CHOICES = [
    (N_("Auto (analysis resolution)"), AUTO_MODEL_CHOICE),
    (N_("VMAF v1 (1080p / 3H)"), builtin_choice("vmaf_v1_3d0h")),
    (N_("VMAF v1 (1080p phone / 5H)"), builtin_choice("vmaf_v1_5d0h")),
    (N_("VMAF v1 (4K / 1.5H)"), builtin_choice("vmaf_v1_1d5h_2160")),
    (N_("VMAF v1 (4K / 3H, up to 110)"), builtin_choice("vmaf_v1_3d0h_2160")),
    (N_("VMAF v1 HFR (1080p / 3H)"), builtin_choice("vmaf_v1_hfr_3d0h")),
    (N_("VMAF v1 HFR (1080p phone / 5H)"), builtin_choice("vmaf_v1_hfr_5d0h")),
    (N_("VMAF v1 HFR (4K / 1.5H)"), builtin_choice("vmaf_v1_hfr_1d5h_2160")),
    (N_("VMAF v1 HFR (4K / 3H, up to 110)"), builtin_choice("vmaf_v1_hfr_3d0h_2160")),
]

SCALE_ALGORITHMS = ["bicubic", "lanczos", "bilinear", "spline"]

#: The GPU decode dropdown's entries, by index, and the index of each vendor.
GPU_VENDOR_BY_INDEX = {0: GpuVendor.AUTO, 1: GpuVendor.NVIDIA, 2: GpuVendor.INTEL, 3: GpuVendor.AMD}
GPU_VENDOR_INDEX = {v: k for k, v in GPU_VENDOR_BY_INDEX.items()}


class OptionsPanel(QGroupBox):
    """The Options panel. `gpu_vendors` are the GPUs found on this PC:
    without one, VMAF can only be calculated on the CPU."""

    #: A control was edited: the name of the VmafOptions field it sets
    #: ("gpu" for the decode box and vendor, "model_v1", "crop_mode"...).
    field_edited = Signal(str)
    #: The VMAF v0.6.1 model dropdown changed to this index (MODEL_CHOICES).
    model_chosen = Signal(int)
    #: A Vship metric's GPU/CPU choice changed ("ssimulacra2", "butteraugli").
    metric_backend_changed = Signal(str)
    #: The CVVDP preset dropdown changed to this index.
    cvvdp_preset_chosen = Signal(int)
    #: "Scale the video to fill the display" was ticked (True) or unticked.
    cvvdp_resize_toggled = Signal(bool)
    cvvdp_add_requested = Signal()
    cvvdp_edit_requested = Signal()
    cvvdp_delete_requested = Signal()
    #: The resolution mismatch dropdown changed to this index (2: "Test both").
    scale_direction_chosen = Signal(int)

    def __init__(self, gpu_vendors: list[GpuVendor], parent=None) -> None:
        super().__init__(tr("Options"), parent)
        #: The model file chosen for "Custom model file...", until it is
        #: applied to the rows.
        self.custom_model_path: str | None = None
        self.vmaf_gpu_possible = bool(gpu_vendors)
        self._build(gpu_vendors)

    def _build(self, gpu_vendors: list[GpuVendor]) -> None:
        """Builds the panel: which videos it edits, then three boxes side by
        side (video preparation, metric-specific settings, performance),
        then a note on what affects scores."""
        layout = QVBoxLayout(self)

        self.target_label = QLabel("")
        theme.style(self.target_label, "color: {muted}; font-style: italic;")
        self.target_label.setWordWrap(True)
        layout.addWidget(self.target_label)

        # Side by side rather than stacked: the panel is the window's full
        # width, so stacking the boxes would leave empty space to the right
        # and push the video table up.
        columns = QHBoxLayout()
        forms = []
        for title, build in ((tr("Video preparation (calculation only)"), self._build_preparation),
                             (tr("Metric-specific settings"), self._build_metric_settings),
                             (tr("Performance"), lambda form: self._build_performance(form, gpu_vendors))):
            box = QGroupBox(title)
            form = QFormLayout(box)
            build(form)
            forms.append(form)
            columns.addWidget(box, stretch=1)
        layout.addLayout(columns)

        metrics_hint = QLabel(
            tr("Cropping and scaling here affect scores. Video Compare's tone mapping and "
            "playback resolution are display-only settings and do not change calculated metrics.")
        )
        theme.style(metrics_hint, "color: {muted}; font-style: italic;")
        metrics_hint.setWordWrap(True)
        layout.addWidget(metrics_hint)
        layout.addStretch(1)

        # Room for longer words than English's. A field that does not fit
        # beside its label goes under it, and a dropdown can be narrower than
        # its longest entry, which it shows in full when opened. Without
        # this, the German panel needs 1,590 px, more than the window's
        # default 1,280, and is cut off at the right. In English nothing
        # moves.
        for form in forms:
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)

    def _build_preparation(self, form: QFormLayout) -> None:
        """Black bars, duration, scaling algorithm and resolution mismatch."""
        self.crop_combo = QComboBox()
        self.crop_combo.addItems([
            tr("Auto-detect (recommended)"),
            tr("None (use full frame)"),
        ])
        self.crop_combo.currentIndexChanged.connect(lambda _value: self.field_edited.emit("crop_mode"))
        form.addRow(tr("Black-bar handling:"), self.crop_combo)

        self.duration_edit = QTimeEdit()
        # In a right-to-left window (Arabic) QTimeEdit writes the sections
        # backwards ("000.00:00:00"); times read left to right in every
        # language. The direction must be set before the display format.
        self.duration_edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.duration_edit.setDisplayFormat("HH:mm:ss.zzz")
        self.duration_edit.setTime(QTime(0, 0, 0, 0))
        self.duration_edit.timeChanged.connect(lambda _value: self.field_edited.emit("duration_limit"))
        form.addRow(tr("Duration limit (0 = full):"), self.duration_edit)

        self.scale_algo_combo = QComboBox()
        self.scale_algo_combo.addItems(SCALE_ALGORITHMS)
        self.scale_algo_combo.currentIndexChanged.connect(lambda _value: self.field_edited.emit("scale_algorithm"))
        form.addRow(tr("Scaling algorithm:"), self.scale_algo_combo)

        self.scale_direction_combo = QComboBox()
        self.scale_direction_combo.addItems([
            tr("Source downscaled to test"),
            tr("Test upscaled to source"),
            tr("Test both (adds a comparison row)"),
        ])
        self.scale_direction_combo.setToolTip(
            tr("When the two resolutions differ: either evaluate quality at the resolution actually\n"
            "delivered (source scaled to match test -- the default), or as if the test\n"
            "video were upscaled back to the reference's native resolution for playback.\n"
            "\"Test both\" doesn't change this row -- it adds a second row for the same test\n"
            "file using the other direction, so you can run and compare both.")
        )
        self.scale_direction_combo.currentIndexChanged.connect(self.scale_direction_chosen.emit)
        form.addRow(tr("Resolution mismatch:"), self.scale_direction_combo)

    def _build_metric_settings(self, form: QFormLayout) -> None:
        """The VMAF models, the frame subsample, and CVVDP's display."""
        self.model_combo = QComboBox()
        for name, _ in MODEL_CHOICES:
            self.model_combo.addItem(tr(name))
        self.model_combo.currentIndexChanged.connect(self.model_chosen.emit)
        self.model_combo.setToolTip(
            tr("The VMAF v0.6.1 column's model. Auto picks the standard or the 4K "
            "model from the size frames are compared at, after black bars are "
            "removed and one video is scaled to the other.")
        )
        form.addRow(tr("VMAF v0.6.1 model:"), self.model_combo)
        self.model_v1_combo = QComboBox()
        for name, _ in V1_MODEL_CHOICES:
            self.model_v1_combo.addItem(tr(name))
        self.model_v1_combo.currentIndexChanged.connect(lambda _index: self.field_edited.emit("model_v1"))
        self.model_v1_combo.setToolTip(
            tr("The VMAF v1 column's model (Netflix's bundled VMAF v1 models). Auto "
            "picks the 1080p model at three screen heights, or the 4K model at "
            "one and a half for 4K comparisons, as for VMAF v0.6.1.")
        )
        form.addRow(tr("VMAF v1 model:"), self.model_v1_combo)

        self.subsample_spin = QSpinBox()
        self.subsample_spin.setRange(1, 60)
        self.subsample_spin.setValue(1)
        self.subsample_spin.valueChanged.connect(lambda _value: self.field_edited.emit("n_subsample"))
        form.addRow(tr("libvmaf frame subsample:"), self.subsample_spin)
        self.subsample_spin.setToolTip(tr("1 = every frame. Applies to VMAF, PSNR and SSIM. XPSNR is computed every frame; combined runs retain values at libvmaf's sampled frames."))

        self._build_cvvdp(form)

    def _build_cvvdp(self, form: QFormLayout) -> None:
        """CVVDP's display: a preset, the display itself, the buttons that
        add, edit and delete presets, and whether the video is scaled to
        fill the display (see vmaf_app.core.cvvdp)."""
        self.cvvdp_preset_combo = QComboBox()
        self.cvvdp_preset_combo.setToolTip(
            tr("The display CVVDP predicts visible differences on. The score depends on the "
            "display as much as on the videos, so compare videos scored for the same one.\n\n"
            "New videos start with the preset chosen in Settings. Saving a preset of your "
            "own makes it that default.")
        )
        self.cvvdp_preset_combo.currentIndexChanged.connect(self.cvvdp_preset_chosen.emit)
        form.addRow(tr("CVVDP display:"), self.cvvdp_preset_combo)
        self.cvvdp_display_label = QLabel()
        theme.style(self.cvvdp_display_label, "color: {muted};")
        self.cvvdp_display_label.setWordWrap(True)
        form.addRow("", self.cvvdp_display_label)

        buttons = QHBoxLayout()
        self.cvvdp_add_btn = QPushButton(tr("Add preset..."))
        self.cvvdp_add_btn.setToolTip(
            tr("Make a new preset of your own, starting from the display shown here. "
            "It becomes the display newly added videos start with; the selected "
            "videos keep theirs until you choose it for them.")
        )
        self.cvvdp_add_btn.clicked.connect(self.cvvdp_add_requested.emit)
        self.cvvdp_edit_btn = QPushButton(tr("Edit display..."))
        self.cvvdp_edit_btn.setToolTip(
            tr("Change the display's size, distance, brightness, room light and more; "
            "rename your preset or save the display as a new one.")
        )
        self.cvvdp_edit_btn.clicked.connect(self.cvvdp_edit_requested.emit)
        self.cvvdp_delete_btn = QPushButton(tr("Delete preset"))
        self.cvvdp_delete_btn.setToolTip(tr("Delete this preset of yours. Built-in presets cannot be deleted."))
        self.cvvdp_delete_btn.clicked.connect(self.cvvdp_delete_requested.emit)
        for button in (self.cvvdp_add_btn, self.cvvdp_edit_btn, self.cvvdp_delete_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        form.addRow("", buttons)

        self.cvvdp_resize_check = QCheckBox(tr("Scale the video to fill the display"))
        self.cvvdp_resize_check.setToolTip(
            tr("Off (the official default): the video is shown pixel for pixel, so a 1080p "
            "video covers a quarter of a 4K display. On: it is scaled, keeping its shape, "
            "to fill the display.\n\nSet per video: choosing or saving a display preset "
            "does not change it.")
        )
        # checkStateChanged, not toggled: a partly ticked box (a mixed
        # selection) already counts as checked, so ticking it fully would not
        # toggle anything.
        self.cvvdp_resize_check.checkStateChanged.connect(
            lambda state: None if state == Qt.CheckState.PartiallyChecked
            else self.cvvdp_resize_toggled.emit(state == Qt.CheckState.Checked)
        )
        form.addRow("", self.cvvdp_resize_check)

    def _build_performance(self, form: QFormLayout, gpu_vendors: list[GpuVendor]) -> None:
        """GPU decoding, where each metric is calculated, the GPUs found, and
        libvmaf's threads."""
        self.gpu_checkbox = QCheckBox(tr("Use GPU decoding"))
        self.gpu_checkbox.setToolTip(
            tr("Hardware-decodes the reference and the test video. Each is "
            "decided separately, and either one falls back to the CPU on its "
            "own if this GPU can't decode its format.")
        )
        self.gpu_checkbox.setChecked(True)
        self.gpu_vendor_combo = QComboBox()
        self.gpu_vendor_combo.addItems([tr("Auto-detect"), "NVIDIA", "Intel", "AMD"])
        self.gpu_checkbox.stateChanged.connect(lambda state: self.gpu_vendor_combo.setEnabled(bool(state)))
        self.gpu_checkbox.stateChanged.connect(lambda _value: self.field_edited.emit("gpu"))
        self.gpu_vendor_combo.currentIndexChanged.connect(lambda _value: self.field_edited.emit("gpu"))
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addWidget(self.gpu_vendor_combo)
        form.addRow(tr("GPU decode:"), gpu_row)

        # VMAF v0.6.1 and NEG: libvmaf has GPU code (CUDA) for their features,
        # ported to Vulkan (vmaf_vulkan), which every GPU uses unless the GPU
        # backend is CUDA. VMAF v1: all four of its features on the GPU, with
        # Vulkan, where the engine supports the model's options (vmaf_v1_gpu).
        self.vmaf_backend_combo = QComboBox()
        self.vmaf_backend_combo.addItems(["GPU", "CPU"])
        self.vmaf_backend_combo.setToolTip(
            tr("With GPU, VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated on the GPU. VMAF v0.6.1 and NEG: with "
            "Vulkan on every GPU, or with CUDA on an NVIDIA GPU when Settings > GPU metrics > GPU backend is "
            "CUDA. Both give the same scores. VMAF v1: all four of its features on the GPU with "
            "Vulkan; for a model whose options the GPU code does not support, its banding and colour features "
            "(CAMBI, SpEED) on the CPU. "
            "PSNR, SSIM and XPSNR are calculated on the CPU either way. The GPU's scores agree with the CPU's to "
            "within a thousandth of a point on every frame, so a saved score is kept whichever is chosen. If "
            "the GPU calculation fails, they are calculated on the CPU instead.")
        )
        self.vmaf_backend_combo.currentIndexChanged.connect(lambda _index: self.field_edited.emit("vmaf_on_gpu"))
        form.addRow(tr("VMAF compute:"), self.vmaf_backend_combo)

        # SSIMULACRA2's and Butteraugli's GPU/CPU choice, the same for both.
        vship_tip = tr("GPU uses Vship when a supported device is available. If GPU computation "
                       "cannot run, the metric automatically falls back to CPU. Choose CPU to "
                       "always use the bundled libjxl reference implementation.")
        self.ssimulacra2_backend_combo = QComboBox()
        self.ssimulacra2_backend_combo.addItems(["GPU", "CPU"])
        self.ssimulacra2_backend_combo.setToolTip(vship_tip)
        self.ssimulacra2_backend_combo.currentIndexChanged.connect(
            lambda _index: self.metric_backend_changed.emit("ssimulacra2"))
        form.addRow(tr("SSIMULACRA2 compute:"), self.ssimulacra2_backend_combo)
        self.butteraugli_backend_combo = QComboBox()
        self.butteraugli_backend_combo.addItems(["GPU", "CPU"])
        self.butteraugli_backend_combo.setToolTip(vship_tip)
        self.butteraugli_backend_combo.currentIndexChanged.connect(
            lambda _index: self.metric_backend_changed.emit("butteraugli"))
        form.addRow(tr("Butteraugli compute:"), self.butteraugli_backend_combo)

        # Without a GPU there is no choice: it shows CPU, which is where VMAF
        # is calculated (see show_vmaf_backend).
        if not self.vmaf_gpu_possible:
            self.vmaf_backend_combo.setEnabled(False)
            self.vmaf_backend_combo.setToolTip(
                tr("No GPU was found: VMAF v0.6.1 and VMAF NEG are calculated on the CPU."))
        if gpu_vendors:
            names = ", ".join(v.value.upper() for v in gpu_vendors)
            form.addRow("", QLabel(tr("Detected GPU(s): {names}", names=names)))

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 128)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText(tr("Auto"))
        self.threads_spin.valueChanged.connect(lambda _value: self.field_edited.emit("n_threads"))
        form.addRow(tr("libvmaf threads:"), self.threads_spin)
        self.threads_spin.setToolTip(
            tr("Controls VMAF, PSNR and SSIM extraction in libvmaf, not XPSNR "
            "or video decoding.\n\nAuto uses every core, or half of them "
            "for each video when two are calculated in parallel.")
        )

    # ------------------------------------------------------------------ showing options

    def show_options(self, options: VmafOptions, metric_backends: dict[str, str],
                     selected: list[VmafOptions]) -> None:
        """Shows `options` (the first selected video's) in the controls.
        `metric_backends` are its Vship metrics' GPU/CPU choices, and
        `selected` every selected video's options, which decide what can be
        edited. The window blocks its own handlers while this runs."""
        self.model_combo.setCurrentIndex(
            next((i for i, (_, key) in enumerate(MODEL_CHOICES) if key == options.model_choice), 0))
        self.custom_model_path = options.custom_model_path
        self.model_v1_combo.setCurrentIndex(
            next((i for i, (_, key) in enumerate(V1_MODEL_CHOICES) if key == options.model_choice_v1), 0))

        self.gpu_checkbox.setChecked(options.gpu_decode)
        self.gpu_vendor_combo.setCurrentIndex(GPU_VENDOR_INDEX.get(options.gpu_vendor, 0))
        self.gpu_vendor_combo.setEnabled(options.gpu_decode)

        self.show_backends(options.vmaf_on_gpu, metric_backends)

        self.crop_combo.setCurrentIndex(0 if options.crop_mode == CropMode.AUTO else 1)

        self.threads_spin.setValue(options.n_threads)
        self.subsample_spin.setValue(options.n_subsample)
        algo_index = SCALE_ALGORITHMS.index(options.scale_algorithm) if options.scale_algorithm in SCALE_ALGORITHMS else 0
        self.scale_algo_combo.setCurrentIndex(algo_index)
        self.scale_direction_combo.setCurrentIndex(
            1 if options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE else 0
        )

        ms = round(options.duration_limit * 1000)
        self.duration_edit.setTime(QTime(0, 0, 0, 0).addMSecs(ms))

        self.model_combo.setEnabled(any(o.compute_vmaf for o in selected))
        self.model_v1_combo.setEnabled(any(o.compute_vmaf_v1 for o in selected))
        uses_libvmaf = any(o.compute_vmaf or o.compute_vmaf_neg or o.compute_vmaf_v1 or o.extra_features
                           for o in selected)
        self.threads_spin.setEnabled(uses_libvmaf)
        # Also while it is above 1: CVVDP is unavailable then, and the
        # way back to it must not be a disabled control.
        self.subsample_spin.setEnabled(uses_libvmaf or any(o.n_subsample > 1 for o in selected))

    def show_backends(self, vmaf_on_gpu: bool, metric_backends: dict[str, str]) -> None:
        """Shows the GPU/CPU choices: VMAF's, and each Vship metric's."""
        self.show_vmaf_backend(vmaf_on_gpu)
        self.ssimulacra2_backend_combo.setCurrentIndex(0 if metric_backends.get("ssimulacra2", "gpu") == "gpu" else 1)
        self.butteraugli_backend_combo.setCurrentIndex(0 if metric_backends.get("butteraugli", "gpu") == "gpu" else 1)

    def show_vmaf_backend(self, on_gpu: bool) -> None:
        """The VMAF GPU/CPU choice as a run would take it: CPU when there is
        no GPU, whatever the video is set to."""
        self.vmaf_backend_combo.setCurrentIndex(0 if on_gpu and self.vmaf_gpu_possible else 1)

    def show_cvvdp(self, selected: list[CvvdpSettings], preferred: str, user_presets: list) -> None:
        """Shows the selected videos' CVVDP settings: the preset they match
        (or "Custom"), the display in one line, and the resize box.
        `preferred` is the preset the first one was given, to tell apart
        presets with the same display; `user_presets` are the user's own.

        Videos that differ show as "Mixed" in the dropdown and a partly
        ticked box. Don't show only the first video's values: choosing the
        preset it already has, or clicking the box, would then do nothing
        for the other videos, or set them the wrong way round.
        """
        settings = selected[0]
        mixed_display = len({repr(s.display.identity()) for s in selected}) > 1
        mixed_resize = len({s.resize_to_display for s in selected}) > 1
        combo = self.cvvdp_preset_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            for preset in cvvdp_presets(user_presets):
                combo.addItem(preset_label(preset), preset.name)
                combo.setItemData(
                    combo.count() - 1, tr(preset.description) if preset.description else tr("Your saved preset."),
                    Qt.ItemDataRole.ToolTipRole,
                )
            match = None if mixed_display else matching_preset(settings, user_presets, preferred)
            if mixed_display:
                combo.addItem(tr("Mixed (the selected videos use different displays)"), None)
                combo.setCurrentIndex(combo.count() - 1)
            elif match is None:
                combo.addItem(tr("Custom (not saved as a preset)"), None)
                combo.setCurrentIndex(combo.count() - 1)
            else:
                combo.setCurrentIndex(combo.findData(match.name))
        finally:
            combo.blockSignals(False)
        self.cvvdp_delete_btn.setEnabled(match is not None and not match.builtin)
        self.cvvdp_display_label.setText(
            tr("The selected videos use different displays; choosing a preset sets them all.")
            if mixed_display else display_description(settings.display)
        )
        box = self.cvvdp_resize_check
        box.blockSignals(True)
        box.setTristate(mixed_resize)
        box.setCheckState(Qt.CheckState.PartiallyChecked if mixed_resize
                          else Qt.CheckState.Checked if settings.resize_to_display else Qt.CheckState.Unchecked)
        box.blockSignals(False)

    # ------------------------------------------------------------------ reading options

    def read_options(self, *, extra_features: list[str], compute_vmaf: bool, compute_vmaf_neg: bool,
                     compute_vmaf_v1: bool, compute_xpsnr: bool) -> VmafOptions:
        """The options the controls show, as VmafOptions. The metrics are
        not in the panel, so the window gives them, from the table's
        column headings."""
        vendor = (GPU_VENDOR_BY_INDEX[self.gpu_vendor_combo.currentIndex()] if self.gpu_checkbox.isChecked()
                  else GpuVendor.NONE)
        crop_mode = CropMode.AUTO if self.crop_combo.currentIndex() == 0 else CropMode.NONE
        duration_limit = QTime(0, 0, 0, 0).msecsTo(self.duration_edit.time()) / 1000.0
        scale_direction = (
            ScaleDirection.DISTORTED_TO_SOURCE if self.scale_direction_combo.currentIndex() == 1
            else ScaleDirection.SOURCE_TO_DISTORTED
        )
        return VmafOptions(
            model="",  # resolved per-job at run time via resolve_model(), once each row's distorted video is known
            model_choice=MODEL_CHOICES[self.model_combo.currentIndex()][1],
            model_choice_v1=V1_MODEL_CHOICES[self.model_v1_combo.currentIndex()][1],
            compute_vmaf_v1=compute_vmaf_v1,
            custom_model_path=self.custom_model_path,
            extra_features=extra_features,
            n_threads=self.threads_spin.value(),
            n_subsample=self.subsample_spin.value(),
            scale_algorithm=self.scale_algo_combo.currentText(),
            scale_direction=scale_direction,
            compute_xpsnr=compute_xpsnr,
            compute_vmaf=compute_vmaf,
            compute_vmaf_neg=compute_vmaf_neg,
            duration_limit=duration_limit,
            gpu_decode=self.gpu_checkbox.isChecked(),
            gpu_vendor=vendor,
            vmaf_on_gpu=self.vmaf_backend_combo.currentIndex() == 0,
            crop_mode=crop_mode,
        )

    def metric_backend(self, metric_key: str) -> str:
        """A Vship metric's GPU/CPU choice: "gpu" or "cpu"."""
        combo = {"ssimulacra2": self.ssimulacra2_backend_combo,
                 "butteraugli": self.butteraugli_backend_combo}.get(metric_key)
        if combo is None:
            raise ValueError(f"Unknown standalone metric: {metric_key}")
        return "gpu" if combo.currentIndex() == 0 else "cpu"
