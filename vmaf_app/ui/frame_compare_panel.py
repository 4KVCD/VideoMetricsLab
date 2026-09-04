"""Interactive source/distorted still-frame comparison tab."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vmaf_app.core.display_hdr import DisplayHdrInfo, query_display_hdr
from vmaf_app.core.frame_extract import (
    PreviewColorMode,
    PreviewColorSettings,
    frame_video_info,
    hdr_kind,
)
from vmaf_app.core.models import VmafRunResult
from vmaf_app.core.time_format import format_hms
from vmaf_app.ui.frame_extract_worker import FrameExtractWorker


@dataclass(frozen=True)
class FrameComparisonEntry:
    identity: object
    label: str
    result: VmafRunResult


def parse_timestamp(value: str) -> float:
    """Accept seconds, M:SS, or H:MM:SS and return seconds."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(not part.strip() for part in parts):
        raise ValueError("Use seconds, M:SS, or H:MM:SS.sss")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Use seconds, M:SS, or H:MM:SS.sss") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("Timestamp cannot be negative")
    if any(not math.isfinite(number) for number in numbers):
        raise ValueError("Timestamp must be a finite number")
    if any(not number.is_integer() for number in numbers[:-1]):
        raise ValueError("Hours and minutes must be whole numbers")
    if len(numbers) > 1 and numbers[-1] >= 60:
        raise ValueError("Seconds must be below 60")
    if len(numbers) > 2 and numbers[-2] >= 60:
        raise ValueError("Minutes must be below 60")
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


class FrameView(QScrollArea):
    """Fit-to-window or pixel-for-pixel image view with retained scroll."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("QScrollArea { background: #171717; border: 1px solid #444; }")
        self._label = QLabel("Run or load a VMAF result to compare its frames.")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: #ddd; background: #171717;")
        self._image: QImage | None = None
        self._fit = True
        self.setWidget(self._label)
        self.setWidgetResizable(True)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.MouseFocusReason)
        super().mousePressEvent(event)

    def set_fit(self, fit: bool) -> None:
        self._fit = fit
        self.setWidgetResizable(fit)
        self._refresh()

    def set_message(self, message: str) -> None:
        self._image = None
        self._label.setPixmap(QPixmap())
        self._label.setText(message)
        self._label.setAlignment(Qt.AlignCenter)
        self.setWidgetResizable(True)

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._label.setText("")
        self.setWidgetResizable(self._fit)
        self._refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        if self._fit:
            available = self.viewport().size()
            pixmap = QPixmap.fromImage(self._image).scaled(
                available,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._label.setPixmap(pixmap)
            self._label.resize(available)
        else:
            pixmap = QPixmap.fromImage(self._image)
            self._label.setPixmap(pixmap)
            self._label.resize(pixmap.size())


class FrameComparePanel(QWidget):
    """One synchronized frame, switched between source and distortions."""

    color_mode_changed = Signal(str)

    _CACHE_LIMIT = 12
    _CACHE_BYTE_LIMIT = 192 * 1024 * 1024

    def __init__(
        self,
        parent=None,
        color_mode: str | PreviewColorMode = PreviewColorMode.DISPLAY_AWARE,
    ) -> None:
        super().__init__(parent)
        self._entries: list[FrameComparisonEntry] = []
        self._current_index = 0
        self._frame = 0
        self._showing_source = False
        self._generation = 0
        self._workers: set[FrameExtractWorker] = set()
        self._window_filters_installed = False
        try:
            self._color_mode = PreviewColorMode(color_mode)
        except ValueError:
            self._color_mode = PreviewColorMode.DISPLAY_AWARE
        self._display_hdr = DisplayHdrInfo()
        self._cache: OrderedDict[tuple, QImage] = OrderedDict()
        self._errors: dict[tuple, str] = {}

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.previous_video_btn = QPushButton("←")
        self.previous_video_btn.setToolTip("Previous distorted video (Left arrow)")
        self.previous_video_btn.clicked.connect(lambda: self.cycle_distorted(-1))
        self.video_combo = QComboBox()
        self.video_combo.setMinimumContentsLength(30)
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
        self.next_video_btn = QPushButton("→")
        self.next_video_btn.setToolTip("Next distorted video (Right arrow)")
        self.next_video_btn.clicked.connect(lambda: self.cycle_distorted(1))
        self.fit_checkbox = QCheckBox("Fit to window")
        self.fit_checkbox.setChecked(True)
        top.addWidget(QLabel("Distorted video:"))
        top.addWidget(self.previous_video_btn)
        top.addWidget(self.video_combo, stretch=1)
        top.addWidget(self.next_video_btn)
        top.addWidget(self.fit_checkbox)
        root.addLayout(top)

        color_row = QHBoxLayout()
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItem(
            "Display-aware (recommended)", PreviewColorMode.DISPLAY_AWARE.value
        )
        self.color_mode_combo.addItem(
            "HDR → SDR (100 nit; assume PQ if untagged)",
            PreviewColorMode.HDR_TO_SDR.value,
        )
        self.color_mode_combo.addItem(
            "Unmanaged (diagnostic)", PreviewColorMode.UNMANAGED.value
        )
        mode_index = self.color_mode_combo.findData(self._color_mode.value)
        self.color_mode_combo.setCurrentIndex(max(0, mode_index))
        self.color_mode_combo.setToolTip(
            "Display-aware detects PQ/HLG tags and uses the Windows monitor's "
            "SDR-white setting. The fixed mode can recover an untagged PQ file."
        )
        self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        self.color_status_label = QLabel()
        self.color_status_label.setStyleSheet("color: #666;")
        color_row.addWidget(QLabel("HDR preview:"))
        color_row.addWidget(self.color_mode_combo)
        color_row.addWidget(self.color_status_label, stretch=1)
        root.addLayout(color_row)

        self.showing_label = QLabel("No completed results")
        font = self.showing_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.showing_label.setFont(font)
        self.showing_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.showing_label)

        self.viewer = FrameView()
        self.fit_checkbox.toggled.connect(self.viewer.set_fit)
        root.addWidget(self.viewer, stretch=1)

        seek = QHBoxLayout()
        self.previous_frame_btn = QPushButton("− Frame")
        self.previous_frame_btn.clicked.connect(lambda: self.set_frame(self._frame - 1))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self.set_frame)
        self.timestamp_edit = QLineEdit("0:00:00.000")
        self.timestamp_edit.setMaximumWidth(120)
        self.timestamp_edit.setToolTip("Enter seconds, M:SS, or H:MM:SS.sss")
        self.timestamp_edit.editingFinished.connect(self._on_timestamp_committed)
        self.next_frame_btn = QPushButton("+ Frame")
        self.next_frame_btn.clicked.connect(lambda: self.set_frame(self._frame + 1))
        seek.addWidget(self.previous_frame_btn)
        seek.addWidget(QLabel("Frame:"))
        seek.addWidget(self.frame_spin)
        seek.addWidget(QLabel("Timestamp:"))
        seek.addWidget(self.timestamp_edit)
        seek.addWidget(self.next_frame_btn)
        root.addLayout(seek)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self._on_slider_changed)
        root.addWidget(self.timeline)

        self.detail_label = QLabel("No frame selected.")
        self.detail_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.detail_label)
        guide = QLabel(
            "Hold S: show source   ·   ←/→: switch distorted video   ·   "
            "Click the image first if a frame/timestamp field has focus"
        )
        guide.setAlignment(Qt.AlignCenter)
        guide.setStyleSheet("color: #666;")
        guide.setWordWrap(True)
        root.addWidget(guide)

        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(120)
        self._seek_timer.timeout.connect(self._request_current_frames)
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(200)
        self._display_timer.timeout.connect(self._on_display_maybe_changed)

        # Filter this panel's own widgets rather than QApplication globally.
        # A global filter keeps every discarded MainWindow alive and makes
        # repeated window creation progressively slower (especially in the
        # test suite); all relevant key events originate inside this page.
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)
        self._update_enabled_state()

    # ------------------------------------------------------------ public API
    def set_runs(self, entries: list[FrameComparisonEntry]) -> None:
        old_identity = self.current_entry.identity if self.current_entry else None
        self._entries = list(entries)
        self._generation += 1
        self._cancel_workers()
        # A file that was missing or temporarily unreadable may have been
        # restored since the previous synchronization; errors are retryable.
        self._errors.clear()

        current = next(
            (i for i, entry in enumerate(self._entries) if entry.identity is old_identity),
            0,
        )
        self._current_index = min(current, max(0, len(self._entries) - 1))
        blocked = self.video_combo.blockSignals(True)
        self.video_combo.clear()
        self.video_combo.addItems([entry.label for entry in self._entries])
        if self._entries:
            self.video_combo.setCurrentIndex(self._current_index)
        self.video_combo.blockSignals(blocked)

        maximum = min(
            (max(1, entry.result.compared_frame_count) for entry in self._entries),
            default=1,
        ) - 1
        self._frame = min(self._frame, maximum)
        self._set_ranges(maximum)
        self._update_enabled_state()
        self._update_labels()
        if not self._entries:
            self.viewer.set_message("Run or load a VMAF result to compare its frames.")
        elif self.isVisible():
            self._seek_timer.start()

    @property
    def current_entry(self) -> FrameComparisonEntry | None:
        if not self._entries:
            return None
        return self._entries[self._current_index]

    def set_frame(self, frame: int) -> None:
        if not self._entries:
            return
        bounded = max(0, min(int(frame), self.frame_spin.maximum()))
        changed = bounded != self._frame
        self._frame = bounded
        self._sync_seek_widgets()
        self._update_labels()
        if changed or self._current_image() is None:
            self._generation += 1
            self._seek_timer.start()

    def cycle_distorted(self, direction: int) -> None:
        if len(self._entries) < 2:
            return
        self._current_index = (self._current_index + direction) % len(self._entries)
        blocked = self.video_combo.blockSignals(True)
        self.video_combo.setCurrentIndex(self._current_index)
        self.video_combo.blockSignals(blocked)
        self._generation += 1
        self._update_labels()
        self._show_or_request()

    def live_workers(self) -> list[FrameExtractWorker]:
        return [worker for worker in self._workers if worker.isRunning()]

    def cancel(self) -> None:
        """Stop pending and active extraction during application shutdown."""
        self._seek_timer.stop()
        self._generation += 1
        self._cancel_workers()

    # -------------------------------------------------------------- lifecycle
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._window_filters_installed:
            top = self.window()
            top.installEventFilter(self)
            for child in top.findChildren(QWidget):
                child.installEventFilter(self)
            self._window_filters_installed = True
        self._refresh_display_hdr()
        if self._entries:
            self._show_or_request()

    def hideEvent(self, event) -> None:
        self._seek_timer.stop()
        self._display_timer.stop()
        self._generation += 1
        self._cancel_workers()
        self._showing_source = False
        self._update_labels()
        super().hideEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if not self.isVisible():
            return super().eventFilter(watched, event)
        event_type = event.type()
        if (
            watched is self.window()
            and event_type in (QEvent.Move, QEvent.ScreenChangeInternal)
        ):
            # MonitorFromWindow must run after Windows has committed the move.
            self._display_timer.start()
        key = event.key() if event_type in (QEvent.KeyPress, QEvent.KeyRelease) else None
        if key == Qt.Key_S and event.modifiers() == Qt.NoModifier:
            if event_type == QEvent.KeyPress and not event.isAutoRepeat():
                self._showing_source = True
                self._update_labels()
                self._show_or_request()
            elif event_type == QEvent.KeyRelease and not event.isAutoRepeat():
                self._showing_source = False
                self._update_labels()
                self._show_or_request()
            return True
        if event_type == QEvent.KeyPress and key in (Qt.Key_Left, Qt.Key_Right):
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QAbstractSpinBox, QSlider, QComboBox)):
                return super().eventFilter(watched, event)
            if not event.isAutoRepeat() and event.modifiers() == Qt.NoModifier:
                self.cycle_distorted(-1 if key == Qt.Key_Left else 1)
            return True
        if event_type in (QEvent.WindowDeactivate, QEvent.Hide) and self._showing_source:
            self._showing_source = False
            self._update_labels()
            self._show_or_request()
        return super().eventFilter(watched, event)

    # --------------------------------------------------------------- controls
    def _on_video_selected(self, index: int) -> None:
        if not 0 <= index < len(self._entries) or index == self._current_index:
            return
        self._current_index = index
        self._generation += 1
        self._update_labels()
        self._show_or_request()

    def _on_slider_changed(self, value: int) -> None:
        self.set_frame(value)

    def _on_color_mode_changed(self, index: int) -> None:
        value = self.color_mode_combo.itemData(index)
        try:
            mode = PreviewColorMode(value)
        except ValueError:
            return
        if mode == self._color_mode:
            return
        self._color_mode = mode
        self._generation += 1
        self._cancel_workers()
        self._update_labels()
        self.color_mode_changed.emit(mode.value)
        self._show_or_request()

    def _on_timestamp_committed(self) -> None:
        entry = self.current_entry
        if entry is None:
            return
        try:
            seconds = parse_timestamp(self.timestamp_edit.text())
        except ValueError as exc:
            self.timestamp_edit.setStyleSheet("border: 1px solid #c33;")
            self.timestamp_edit.setToolTip(str(exc))
            return
        self.timestamp_edit.setStyleSheet("")
        self.timestamp_edit.setToolTip("Enter seconds, M:SS, or H:MM:SS.sss")
        self.set_frame(round(seconds * entry.result.fps))

    def _set_ranges(self, maximum: int) -> None:
        for widget in (self.frame_spin, self.timeline):
            blocked = widget.blockSignals(True)
            widget.setRange(0, maximum)
            widget.setValue(self._frame)
            widget.blockSignals(blocked)

    def _sync_seek_widgets(self) -> None:
        for widget in (self.frame_spin, self.timeline):
            blocked = widget.blockSignals(True)
            widget.setValue(self._frame)
            widget.blockSignals(blocked)
        entry = self.current_entry
        seconds = self._frame / entry.result.fps if entry and entry.result.fps > 0 else 0
        blocked = self.timestamp_edit.blockSignals(True)
        self.timestamp_edit.setText(format_hms(seconds, decimals=3))
        self.timestamp_edit.blockSignals(blocked)

    def _update_enabled_state(self) -> None:
        available = bool(self._entries)
        for widget in (
            self.video_combo, self.frame_spin, self.timestamp_edit, self.timeline,
            self.previous_frame_btn, self.next_frame_btn,
        ):
            widget.setEnabled(available)
        several = len(self._entries) > 1
        self.previous_video_btn.setEnabled(several)
        self.next_video_btn.setEnabled(several)

    def _update_labels(self) -> None:
        entry = self.current_entry
        if entry is None:
            self.showing_label.setText("No completed results")
            self.detail_label.setText("No frame selected.")
            self.color_status_label.setText("No video selected.")
            return
        side = "SOURCE" if self._showing_source else "DISTORTED"
        name = entry.result.source_info.path.name if self._showing_source else entry.label
        suffix = "" if self._showing_source else f" {self._current_index + 1} of {len(self._entries)}"
        self.showing_label.setText(f"{side}{suffix} — {name}")
        seconds = self._frame / entry.result.fps if entry.result.fps > 0 else 0
        idx = int(np.searchsorted(entry.result.frames.frame, self._frame))
        score = None
        if idx < len(entry.result.frames) and int(entry.result.frames.frame[idx]) == self._frame:
            candidate = float(entry.result.frames.vmaf[idx])
            score = candidate if math.isfinite(candidate) else None
        score_text = f"VMAF {score:.2f}" if score is not None else "VMAF not scored for this frame"
        self.detail_label.setText(
            f"Frame {self._frame:,}   ·   {format_hms(seconds, decimals=3)}   ·   {score_text}"
        )
        self._update_color_status()

    def _color_settings(self) -> PreviewColorSettings:
        return PreviewColorSettings(
            mode=self._color_mode,
            display_hdr_enabled=self._display_hdr.hdr_enabled,
            display_sdr_white_nits=self._display_hdr.sdr_white_nits,
        )

    def _refresh_display_hdr(self) -> bool:
        try:
            window_handle = int(self.window().winId())
        except (RuntimeError, TypeError):
            window_handle = None
        current = query_display_hdr(window_handle)
        if current != self._display_hdr:
            self._display_hdr = current
            self._generation += 1
            self._cancel_workers()
            self._update_labels()
            return True
        return False

    def _on_display_maybe_changed(self) -> None:
        if self.isVisible() and self._refresh_display_hdr():
            self._show_or_request()

    def _update_color_status(self) -> None:
        entry = self.current_entry
        if entry is None:
            return
        side = "source" if self._showing_source else "distorted"
        kind = hdr_kind(frame_video_info(entry.result, side))
        if self._color_mode == PreviewColorMode.UNMANAGED:
            self.color_status_label.setText(
                f"{kind or 'SDR / untagged'} input · tone mapping off"
            )
            return
        settings = self._color_settings()
        if self._color_mode == PreviewColorMode.HDR_TO_SDR:
            source = kind or "Untagged input (assuming HDR10 / PQ)"
            self.color_status_label.setText(
                f"{source} · fixed HDR → SDR at {settings.target_nits:g} nit"
            )
            return
        if kind is None:
            self.color_status_label.setText("SDR / untagged input · no tone mapping")
            return
        if self._display_hdr.hdr_enabled is True:
            self.color_status_label.setText(
                f"{kind} · display-aware HDR → SDR · Windows HDR on · "
                f"SDR white {settings.target_nits:g} nit"
            )
        elif self._display_hdr.hdr_enabled is False:
            self.color_status_label.setText(
                f"{kind} · HDR → SDR at 100 nit · Windows HDR off"
            )
        else:
            self.color_status_label.setText(
                f"{kind} · HDR → SDR at 100 nit · display HDR state unavailable"
            )

    # --------------------------------------------------------------- decoding
    def _cache_key(self, side: str) -> tuple | None:
        entry = self.current_entry
        return (
            None if entry is None else
            (entry.identity, self._frame, side, self._color_settings().cache_token)
        )

    def _current_image(self) -> QImage | None:
        side = "source" if self._showing_source else "distorted"
        key = self._cache_key(side)
        return None if key is None else self._cache.get(key)

    def _show_or_request(self) -> None:
        image = self._current_image()
        if image is not None:
            self.viewer.set_image(image)
            key = self._cache_key("source" if self._showing_source else "distorted")
            if key is not None:
                self._cache.move_to_end(key)
            return
        side = "source" if self._showing_source else "distorted"
        key = self._cache_key(side)
        if key in self._errors:
            self.viewer.set_message(f"Could not load frame:\n{self._errors[key]}")
        else:
            self.viewer.set_message(f"Loading {side} frame {self._frame:,}…")
        self._seek_timer.start()

    def _request_current_frames(self) -> None:
        entry = self.current_entry
        if entry is None or not self.isVisible():
            return
        self._refresh_display_hdr()
        generation = self._generation
        sides = []
        preferred = "source" if self._showing_source else "distorted"
        for side in (preferred, "distorted" if preferred == "source" else "source"):
            key = self._cache_key(side)
            if key not in self._cache and key not in self._errors:
                sides.append(side)
        if not sides:
            self._show_or_request()
            return
        self._cancel_workers()
        worker = FrameExtractWorker(
            generation, entry.result, self._frame, sides,
            color_settings=self._color_settings(), parent=self,
        )
        worker.frame_ready.connect(self._on_frame_ready)
        worker.frame_failed.connect(self._on_frame_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._workers.add(worker)
        worker.start()

    def _on_frame_ready(self, generation: int, side: str, png: bytes) -> None:
        if generation != self._generation:
            return
        image = QImage.fromData(png, "PNG")
        if image.isNull():
            self._on_frame_failed(generation, side, "ffmpeg returned an unreadable image.")
            return
        key = self._cache_key(side)
        if key is None:
            return
        self._cache[key] = image
        self._cache.move_to_end(key)
        while len(self._cache) > self._CACHE_LIMIT or (
            len(self._cache) > 2
            and sum(cached.sizeInBytes() for cached in self._cache.values())
            > self._CACHE_BYTE_LIMIT
        ):
            self._cache.popitem(last=False)
        current_side = "source" if self._showing_source else "distorted"
        if side == current_side:
            self.viewer.set_image(image)

    def _on_frame_failed(self, generation: int, side: str, error: str) -> None:
        if generation != self._generation:
            return
        key = self._cache_key(side)
        if key is not None:
            self._errors[key] = error
        current_side = "source" if self._showing_source else "distorted"
        if side == current_side:
            self.viewer.set_message(f"Could not load frame:\n{error}")

    def _on_worker_finished(self, worker: FrameExtractWorker) -> None:
        self._workers.discard(worker)
        worker.deleteLater()

    def _cancel_workers(self) -> None:
        for worker in self.live_workers():
            worker.cancel()
