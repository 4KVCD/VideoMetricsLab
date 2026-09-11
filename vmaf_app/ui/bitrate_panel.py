"""Independent packet-level bitrate viewer with frame/second/GOP plots."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vmaf_app.core.bitrate import (
    BitrateData,
    BitratePlot,
    bitrate_summary,
    frame_plot,
    gop_plot,
    second_plot,
)
from vmaf_app.core.models import VideoInfo
from vmaf_app.core.time_format import format_hms
from vmaf_app.ui.bitrate_worker import BitrateWorker
from vmaf_app.ui.chart import ChartSeries, ChartWidget
from vmaf_app.ui.formatting import media_info_string

_COL_USE, _COL_FILE, _COL_MEDIA, _COL_DURATION, _COL_FRAMES, _COL_AVG, _COL_MIN, _COL_MAX, _COL_STATUS = range(9)

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


@dataclass
class BitrateEntry:
    path: Path
    color: str
    identity: object = field(default_factory=object)
    info: VideoInfo | None = None
    data: BitrateData | None = None
    enabled: bool = True
    status: str = "Ready to analyze"
    plots: dict[tuple[str, bool], BitratePlot] = field(default_factory=dict)


def _path_key(path: Path) -> str:
    try:
        value = str(path.resolve())
    except OSError:
        value = str(path)
    return value.casefold()


def _rate_text(kbps: float) -> str:
    if kbps >= 10_000:
        return f"{kbps / 1000:.2f} Mb/s"
    if kbps >= 1000:
        return f"{kbps / 1000:.3f} Mb/s"
    return f"{kbps:.0f} kb/s"


class BitratePanel(QWidget):
    """Owns files and analysis independently of the VMAF Videos tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._entries: OrderedDict[str, BitrateEntry] = OrderedDict()
        self._pending: OrderedDict[str, None] = OrderedDict()
        self._active_keys: set[str] = set()
        self._worker: BitrateWorker | None = None
        self._stopping = False
        self._current_plots: dict[int, tuple[BitrateEntry, BitratePlot]] = {}
        self._populating = False

        root = QVBoxLayout(self)
        file_controls = QHBoxLayout()
        self.add_btn = QPushButton("Add files…")
        self.add_btn.clicked.connect(self._on_add_files)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.analyze_btn = QPushButton("Calculate bitrate")
        self.analyze_btn.setToolTip("Calculate bitrate for videos with the Use checkbox checked.")
        self.analyze_btn.clicked.connect(self._on_analyze_checked)
        file_controls.addWidget(self.add_btn)
        file_controls.addWidget(self.remove_btn)
        file_controls.addWidget(self.analyze_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.cancel)
        file_controls.addWidget(self.stop_btn)
        file_controls.addStretch(1)
        root.addLayout(file_controls)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Use", "Video file", "Media info", "Duration", "Frames",
            "Average", "Minimum", "Maximum", "Status",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_COL_FILE, QHeaderView.Stretch)
        for column in range(self.table.columnCount()):
            if column != _COL_FILE:
                header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.setMaximumHeight(230)
        self.table.itemChanged.connect(self._on_table_item_changed)
        root.addWidget(self.table)

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Plot view:"))
        self.view_group = QButtonGroup(self)
        self.frame_radio = QRadioButton("Frame based")
        self.second_radio = QRadioButton("Second based")
        self.gop_radio = QRadioButton("GOP based")
        self.second_radio.setChecked(True)
        for index, button in enumerate(
            (self.frame_radio, self.second_radio, self.gop_radio)
        ):
            self.view_group.addButton(button, index)
            plot_controls.addWidget(button)
        self.view_group.idClicked.connect(lambda _index: self._refresh_plot())
        self.adjust_start_checkbox = QCheckBox("Adjust stream start time to zero")
        self.adjust_start_checkbox.setChecked(True)
        self.adjust_start_checkbox.toggled.connect(lambda _checked: self._refresh_plot())
        plot_controls.addSpacing(18)
        plot_controls.addWidget(self.adjust_start_checkbox)
        plot_controls.addStretch(1)
        self.reset_zoom_btn = QPushButton("Reset zoom")
        self.reset_zoom_btn.clicked.connect(lambda: self.chart.reset_view())
        self.export_btn = QPushButton("Export PNG…")
        self.export_btn.clicked.connect(self._on_export_png)
        plot_controls.addWidget(self.reset_zoom_btn)
        plot_controls.addWidget(self.export_btn)
        root.addLayout(plot_controls)

        self.chart = ChartWidget(y_axis_label="Video bitrate (kb/s)", fixed_y_max=None)
        self.chart.hovered.connect(self._on_hovered)
        self.chart.left.connect(self._on_hover_left)
        root.addWidget(self.chart, stretch=1)

        self.hover_label = QLabel(
            "Hover over the plot to inspect values. Scroll to zoom, drag to pan, "
            "and double-click to reset."
        )
        self.hover_label.setTextFormat(Qt.PlainText)
        root.addWidget(self.hover_label)

        status_row = QHBoxLayout()
        self.status_label = QLabel(
            "Frame view shows encoded video-frame size; second and GOP views show video-only bitrate."
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(220)
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.progress)
        root.addLayout(status_row)
        self._update_buttons()

    # -------------------------------------------------------------- public API
    def add_files(
        self,
        paths: list[Path],
        infos: dict[Path, VideoInfo] | None = None,
        *,
        analyze: bool = False,
    ) -> None:
        infos = infos or {}
        added_keys: list[str] = []
        for path_value in paths:
            path = Path(path_value)
            key = _path_key(path)
            info = infos.get(path)
            if key in self._entries:
                if info is not None:
                    self._entries[key].info = info
                if analyze and self._entries[key].data is None:
                    added_keys.append(key)
                continue
            self._entries[key] = BitrateEntry(
                path=path,
                info=info,
                color=_COLORS[len(self._entries) % len(_COLORS)],
            )
            added_keys.append(key)
        self._rebuild_table()
        self._refresh_plot()
        if analyze:
            self._queue_keys(added_keys)

    def add_and_analyze(self, infos: list[VideoInfo]) -> None:
        by_path = {info.path: info for info in infos}
        self.add_files(list(by_path), by_path, analyze=True)

    def live_workers(self) -> list[BitrateWorker]:
        return [self._worker] if self._worker is not None and self._worker.isRunning() else []

    def cancel(self) -> None:
        for key in self._pending:
            if key in self._entries:
                self._entries[key].status = "Stopped"
        self._pending.clear()
        if self._worker is not None:
            self._stopping = True
            self.status_label.setText("Stopping bitrate analysis…")
            self._worker.cancel()
        self._refresh_table_values()
        self._update_buttons()

    # --------------------------------------------------------------- file list
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.add_files(paths)
        event.acceptProposedAction()

    def _on_add_files(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(self, "Add videos for bitrate analysis")
        if paths:
            self.add_files([Path(path) for path in paths])

    def _on_remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        keys = list(self._entries)
        for row in rows:
            if 0 <= row < len(keys):
                key = keys[row]
                self._entries.pop(key, None)
                self._pending.pop(key, None)
        self._rebuild_table()
        self._refresh_plot()

    def _on_analyze_checked(self) -> None:
        if self._worker is not None:
            return
        keys = [key for key, entry in self._entries.items() if entry.enabled]
        completed = [key for key in keys if self._entries[key].data is not None]
        if completed:
            answer = QMessageBox.question(
                self, "Recalculate bitrate?",
                f"{len(completed)} checked video(s) already have bitrate results.\n\n"
                "Recalculate those files? Choose No to keep their results and "
                "calculate only files without results.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                keys = [key for key in keys if key not in completed]
        if keys:
            self._queue_keys(keys)

    def _queue_keys(self, keys: list[str]) -> None:
        if self._stopping:
            return
        for key in keys:
            entry = self._entries.get(key)
            if entry is not None and key not in self._active_keys:
                entry.status = "Queued"
                self._pending[key] = None
        self._refresh_table_values()
        self._start_pending()

    def _start_pending(self) -> None:
        if self._worker is not None:
            return
        jobs = []
        for key in list(self._pending):
            self._pending.pop(key, None)
            entry = self._entries.get(key)
            if entry is not None:
                jobs.append((entry.path, entry.info))
        if not jobs:
            self._worker = None
            self._update_buttons()
            return
        worker = BitrateWorker(jobs, parent=self)
        self._active_keys = {_path_key(path) for path, _info in jobs}
        self._worker = worker
        worker.file_started.connect(self._on_file_started)
        worker.progress.connect(self._on_progress)
        worker.analyzed.connect(self._on_analyzed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._update_buttons()
        worker.start()

    def _on_file_started(self, path: Path) -> None:
        if self._stopping:
            return
        entry = self._entries.get(_path_key(path))
        if entry is not None:
            entry.status = "Reading video packets…"
            self.status_label.setText(f"Analyzing {path.name}…")
            self.progress.setValue(0)
            self._refresh_table_values()

    def _on_progress(self, path: Path, done: int, total: int) -> None:
        if self._stopping:
            return
        self.progress.setValue(min(99, round(done / max(1, total) * 100)))
        entry = self._entries.get(_path_key(path))
        if entry is not None:
            entry.status = f"{done:,} packets"
            row = self._row_for_key(_path_key(path))
            if row >= 0 and self.table.item(row, _COL_STATUS) is not None:
                self.table.item(row, _COL_STATUS).setText(entry.status)

    def _on_analyzed(self, path: Path, info: VideoInfo, data: BitrateData) -> None:
        entry = self._entries.get(_path_key(path))
        if entry is None:
            return
        entry.info = info
        entry.data = data
        entry.plots.clear()
        entry.status = "Complete"
        self.progress.setValue(100)
        self._refresh_table_values()
        self._refresh_plot()

    def _on_failed(self, path: Path, error: str) -> None:
        entry = self._entries.get(_path_key(path))
        if entry is None:
            return
        entry.status = "Failed"
        row = self._row_for_key(_path_key(path))
        self._refresh_table_values()
        self.table.item(row, _COL_STATUS).setToolTip(error)
        self.status_label.setText(f"Could not analyze {path.name}: {error}")

    def _on_worker_finished(self, worker: BitrateWorker) -> None:
        if worker is not self._worker:
            worker.deleteLater()
            return
        stopped = self._stopping
        if stopped:
            for key in self._active_keys:
                entry = self._entries.get(key)
                if entry is not None and entry.status not in ("Complete", "Failed"):
                    entry.status = "Stopped"
            self._refresh_table_values()
        self._stopping = False
        if worker is self._worker:
            self._worker = None
            self._active_keys.clear()
        worker.deleteLater()
        if self._pending:
            self._start_pending()
        else:
            self.status_label.setText(
                "Bitrate analysis stopped. Completed results are kept." if stopped else
                "Bitrate analysis complete. Values contain the first video stream only."
            )
            self._update_buttons()

    # --------------------------------------------------------------- table UI
    def _row_for_key(self, key: str) -> int:
        try:
            return list(self._entries).index(key)
        except ValueError:
            return -1

    def _rebuild_table(self) -> None:
        self._populating = True
        try:
            self.table.setRowCount(len(self._entries))
            for row, entry in enumerate(self._entries.values()):
                self._set_table_row(row, entry)
        finally:
            self._populating = False
        self._update_buttons()

    def _refresh_table_values(self) -> None:
        self._populating = True
        try:
            for row, entry in enumerate(self._entries.values()):
                self._set_table_row(row, entry)
        finally:
            self._populating = False

    def _set_table_row(self, row: int, entry: BitrateEntry) -> None:
        if row < 0:
            return
        use = self.table.item(row, _COL_USE) or QTableWidgetItem()
        use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        use.setCheckState(Qt.Checked if entry.enabled else Qt.Unchecked)
        use.setBackground(QColor(entry.color))
        self.table.setItem(row, _COL_USE, use)
        values = [
            entry.path.name,
            media_info_string(entry.info) if entry.info else "",
            format_hms(entry.data.duration, decimals=2) if entry.data else "",
            f"{entry.data.frame_count:,}" if entry.data else "",
            "", "", "", entry.status,
        ]
        if entry.data is not None:
            summary = bitrate_summary(entry.data)
            values[4:7] = [
                _rate_text(summary.average_kbps),
                _rate_text(summary.minimum_kbps),
                _rate_text(summary.maximum_kbps),
            ]
        for column, value in zip(range(1, self.table.columnCount()), values, strict=True):
            item = self.table.item(row, column) or QTableWidgetItem()
            item.setText(value)
            if column == _COL_FILE:
                item.setToolTip(str(entry.path))
            self.table.setItem(row, column, item)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != _COL_USE:
            return
        entries = list(self._entries.values())
        if 0 <= item.row() < len(entries):
            entries[item.row()].enabled = item.checkState() == Qt.Checked
            self._refresh_plot()

    def _update_buttons(self) -> None:
        running = self._worker is not None
        self.analyze_btn.setEnabled(bool(self._entries) and not running)
        self.stop_btn.setEnabled(running and not self._stopping)
        self.remove_btn.setEnabled(bool(self._entries))
        self.export_btn.setEnabled(self.chart.has_data() if hasattr(self, "chart") else False)

    # ------------------------------------------------------------------- plot
    def _view_name(self) -> str:
        if self.frame_radio.isChecked():
            return "frame"
        if self.gop_radio.isChecked():
            return "gop"
        return "second"

    def _entry_plot(self, entry: BitrateEntry) -> BitratePlot | None:
        if entry.data is None:
            return None
        key = (self._view_name(), self.adjust_start_checkbox.isChecked())
        if key not in entry.plots:
            function = {"frame": frame_plot, "second": second_plot, "gop": gop_plot}[key[0]]
            entry.plots[key] = function(entry.data, key[1])
        return entry.plots[key]

    def _refresh_plot(self) -> None:
        self.chart.clear()
        self._current_plots.clear()
        axis_label = (
            "Frame size (kbit)" if self._view_name() == "frame"
            else "GOP bitrate (kb/s)" if self._view_name() == "gop"
            else "Video bitrate (kb/s)"
        )
        self.chart.y_axis_label = axis_label
        for entry in self._entries.values():
            if not entry.enabled:
                continue
            plot = self._entry_plot(entry)
            if plot is None:
                continue
            series_id = id(entry.identity)
            self.chart.set_series(
                series_id,
                ChartSeries(plot.times, plot.values, entry.color),
            )
            self._current_plots[series_id] = (entry, plot)
        self.chart.reset_view()
        self._on_hover_left()
        self._update_buttons()

    def _on_hovered(self, time_value: float, _value: float) -> None:
        self.chart.set_cursor_time(time_value)
        lines = [f"Time: {format_hms(time_value, decimals=3)}"]
        for entry, plot in self._current_plots.values():
            if not len(plot.times):
                continue
            index = int(np.searchsorted(plot.times, time_value, side="left"))
            index = min(len(plot.times) - 1, max(0, index))
            if index and abs(plot.times[index - 1] - time_value) <= abs(plot.times[index] - time_value):
                index -= 1
            lines.append(
                f"{entry.path.name}: {float(plot.values[index]):,.2f} {plot.value_label}"
            )
        self.hover_label.setText("\n".join(lines))

    def _on_hover_left(self) -> None:
        self.chart.set_cursor_time(None)
        self.hover_label.setText(
            "Hover over the plot to inspect values. Scroll to zoom, drag to pan, "
            "and double-click to reset."
        )

    def _on_export_png(self) -> None:
        if not self.chart.has_data():
            QMessageBox.information(self, "No bitrate data", "Analyze at least one video first.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export bitrate plot", "bitrate_plot.png", "PNG image (*.png)"
        )
        if path and not self.chart.render_to_pixmap().save(path):
            QMessageBox.warning(self, "Export failed", f"Could not write {path}")
