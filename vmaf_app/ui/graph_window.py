"""Expanded VMAF/PSNR/SSIM/XPSNR-vs-time graph window.

Supports overlaying multiple runs (e.g. several distorted encodes compared
against the same or different sources) as separate colored curves, with a
shared time-synced hover readout and a side-by-side stats comparison table.
Each metric (VMAF, PSNR, SSIM, XPSNR) gets its own tab/plot -- they're
different scales (0-100, dB, 0-1, dB) that don't belong on one axis -- while
the series list and stats table at the top are shared across all of them,
since it's the same set of runs either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from vmaf_app.core.models import FrameScore, VmafRunResult
from vmaf_app.core.run_io import export_csv, load_run, save_run
from vmaf_app.core.stats import DEFAULT_THRESHOLDS, VmafStats, compute_stats
from vmaf_app.core.time_format import format_hms
from vmaf_app.ui.chart import ChartSeries, ChartWidget

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# How many "x steps" to search either side of the cursor for a point to lock
# onto -- see the step calculation in _MetricPage._on_mouse_moved.
_HOVER_SEARCH_STEPS = 5


@dataclass
class MetricSpec:
    key: str  # "vmaf", "psnr", "ssim", "xpsnr" -- also the FrameScore attribute name
    label: str  # tab title / series-list column label
    axis_label: str  # plot Y-axis label
    value_format: str  # format spec for hover-text values, e.g. "{:.2f}"
    fixed_y_max: float | None  # VMAF's natural ceiling (100); None = autoscale to the data
    thresholds: list[tuple[str, float]] = field(default_factory=list)  # only meaningful on VMAF's fixed 0-100 scale

    def value(self, frame: FrameScore) -> float | None:
        return getattr(frame, self.key)


METRICS: list[MetricSpec] = [
    MetricSpec("vmaf", "VMAF", "VMAF", "{:.2f}", fixed_y_max=100.0, thresholds=DEFAULT_THRESHOLDS),
    MetricSpec("psnr", "PSNR", "PSNR (dB)", "{:.2f}", fixed_y_max=None),
    MetricSpec("ssim", "SSIM", "SSIM", "{:.4f}", fixed_y_max=None),
    MetricSpec("xpsnr", "XPSNR", "XPSNR (dB)", "{:.2f}", fixed_y_max=None),
]


@dataclass
class SeriesEntry:
    result: VmafRunResult
    label: str
    color: str
    checkbox: QCheckBox
    times: np.ndarray
    step: float  # typical time delta between consecutive points in this series
    visible: bool = True


@dataclass
class _MetricCurve:
    stats: VmafStats
    # Held once per add_run rather than re-derived on every hover move --
    # that per-call work was a measured CPU bottleneck on a long run.
    values: np.ndarray
    visible: bool = True


class _MetricPage(QWidget):
    """One metric's own plot + crosshair + hover readout. Curves for a given
    series only exist here if that run actually has this metric's data (e.g.
    a run without "Also compute PSNR" checked has no curve on the PSNR page).
    """

    def __init__(self, metric: MetricSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metric = metric
        self._curves: dict[int, _MetricCurve] = {}  # series_id -> stats/values, only entries with data
        self._hover_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chart = ChartWidget(y_axis_label=metric.axis_label, fixed_y_max=metric.fixed_y_max)
        layout.addWidget(self.chart, stretch=1)

        self.no_data_label = QLabel(
            f"No {metric.label} data among the currently visible series -- "
            f"tick the {metric.label} column header before running to see it here."
        )
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet("color: #888; font-style: italic; padding: 12px;")
        self.no_data_label.setVisible(False)
        layout.addWidget(self.no_data_label)

        self.hover_label = QLabel(
            "Hover to inspect a point (locks onto the lowest nearby score at or below "
            "your cursor, so dips are easy to land on). Scroll to zoom, drag to pan, "
            "double-click to reset."
        )
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.hover_label.setStyleSheet("font-family: Consolas, monospace; padding: 6px;")
        # Fixed height + plain text + no wrap, all for the same reason: this
        # is rewritten on every mouse move, and anything that lets its size
        # hint change invalidates the layout of the whole tab (chart, stats
        # table and all) on each one. That relayout, not the painting, was
        # measured as ~76% of the total cost of a hover.
        self.hover_label.setFixedHeight(90)
        self.hover_label.setWordWrap(False)
        self.hover_label.setTextFormat(Qt.PlainText)  # skips Qt's rich-text sniffing per update
        # Kept to its natural width rather than stretched across the window:
        # repaint cost is proportional to the damaged area, and a full-width
        # strip made every mouse move repaint ~2400x90px of mostly blank
        # space. A stretch to its right takes up the slack instead.
        self.hover_label.setMaximumWidth(560)
        hover_row = QHBoxLayout()
        hover_row.setContentsMargins(0, 0, 0, 0)
        hover_row.addWidget(self.hover_label)
        hover_row.addStretch(1)
        layout.addLayout(hover_row)

        self.chart.left.connect(self._on_pointer_left)
        # chart.hovered is connected by GraphWindow -- the handler needs the
        # shared series map that only GraphWindow owns.

    # ------------------------------------------------------------------ curves
    def set_curve(self, series_id: int, entry: SeriesEntry, color: str) -> None:
        """(Re)builds this series' curve on this page from its current data,
        or removes it if the run has no data for this metric."""
        self.remove_curve(series_id)
        # PSNR/SSIM/XPSNR are computed for a whole run or not at all -- it's
        # a per-run option, never a per-frame one -- so the array is either
        # present or None, and lines up index-for-index with entry.times.
        values = entry.result.frames.values(self.metric.key)
        if values is None or len(values) == 0:
            self._update_no_data_label()
            return
        self.chart.set_series(series_id, ChartSeries(
            times=entry.times, values=values, color=color, visible=entry.visible,
        ))
        self._curves[series_id] = _MetricCurve(
            stats=compute_stats(values, self.metric.thresholds), values=values, visible=entry.visible,
        )
        self._update_no_data_label()

    def remove_curve(self, series_id: int) -> None:
        if self._curves.pop(series_id, None) is None:
            return
        self.chart.remove_series(series_id)
        self._update_no_data_label()

    def set_visible(self, series_id: int, visible: bool) -> None:
        curve = self._curves.get(series_id)
        if curve is None:
            return
        curve.visible = visible
        self.chart.set_series_visible(series_id, visible)

    def _update_no_data_label(self) -> None:
        self.no_data_label.setVisible(not self._curves)

    def _on_pointer_left(self) -> None:
        self.chart.set_cursor_time(None)

    def _set_hover_text(self, text: str) -> None:
        # Dragging across one frame's worth of pixels reports the same thing
        # every time; repainting it again is pure waste.
        if text != self._hover_text:
            self._hover_text = text
            self.hover_label.setText(text)

    # ------------------------------------------------------------------ hover
    @staticmethod
    def _nearest_index_by_time(times: np.ndarray, x: float) -> int:
        idx = int(np.searchsorted(times, x, side="left"))
        if idx <= 0:
            return 0
        if idx >= len(times):
            return len(times) - 1
        return idx if (times[idx] - x) < (x - times[idx - 1]) else idx - 1

    def _find_hover_index(
        self, entry: SeriesEntry, values: np.ndarray, x: float, y: float, half_window: float,
    ) -> int:
        """Finds the frame to report for this series at the cursor.

        Rather than the single nearest-in-time point (which makes a sharp,
        narrow dip nearly impossible to land the cursor on), this looks at
        every point within `half_window` of the cursor's time position and:
        prefers the one closest in time that's at or below the cursor's Y
        position -- so hovering anywhere near a dip "grabs" it; and falls
        back to the single lowest-value point in that neighbourhood if
        nothing there is at or below the cursor's Y.

        Vectorised: zoomed out over a long run this window spans thousands
        of frames, and it runs on every mouse move.
        """
        times = entry.times
        lo = int(np.searchsorted(times, x - half_window, side="left"))
        hi = int(np.searchsorted(times, x + half_window, side="right"))
        if lo >= hi:
            return self._nearest_index_by_time(times, x)

        window = values[lo:hi]
        at_or_below = np.flatnonzero(window <= y)
        if at_or_below.size:
            nearest = np.abs(times[lo:hi][at_or_below] - x).argmin()
            return lo + int(at_or_below[nearest])
        return lo + int(np.nanargmin(window))

    def _find_shared_hover_time(
        self, pages: list[tuple[SeriesEntry, np.ndarray]], x: float, y: float, x_per_pixel: float,
    ) -> float:
        """The multi-series equivalent of _find_hover_index: picks ONE target
        time using the same dip-snap rule (nearest-in-time among points
        at/below the cursor's Y across ALL visible series pooled together,
        falling back to the single lowest point if none qualify) so every
        series' readout refers to the exact same moment.
        """
        best_below_time: float | None = None
        best_below_dist = 0.0
        fallback_time: float | None = None
        fallback_val = 0.0

        for entry, values in pages:
            times = entry.times
            half_window = max(entry.step, x_per_pixel) * _HOVER_SEARCH_STEPS
            lo = int(np.searchsorted(times, x - half_window, side="left"))
            hi = int(np.searchsorted(times, x + half_window, side="right"))
            if lo >= hi:
                continue
            window_times, window_values = times[lo:hi], values[lo:hi]

            lowest = int(np.nanargmin(window_values))
            if fallback_time is None or window_values[lowest] < fallback_val:
                fallback_time, fallback_val = float(window_times[lowest]), float(window_values[lowest])

            at_or_below = np.flatnonzero(window_values <= y)
            if at_or_below.size:
                distances = np.abs(window_times[at_or_below] - x)
                nearest = int(distances.argmin())
                if best_below_time is None or distances[nearest] < best_below_dist:
                    best_below_time = float(window_times[at_or_below[nearest]])
                    best_below_dist = float(distances[nearest])

        if best_below_time is not None:
            return best_below_time
        if fallback_time is not None:
            return fallback_time
        return x  # nothing nearby in any series -- just use the raw cursor time

    def on_hover(self, x: float, y: float, entries_by_id: dict[int, SeriesEntry]) -> None:
        """Cursor moved to time `x`, value `y`: pick the frame(s) to report
        and put the crosshair on the chosen moment."""
        x_per_pixel = self.chart.seconds_per_pixel()
        visible = [
            (entries_by_id[sid], c.values)
            for sid, c in self._curves.items() if c.visible
        ]
        if not visible:
            self.chart.set_cursor_time(x)
            self._set_hover_text(f"Time: {format_hms(x, decimals=2)}")
            return

        lines = [f"Time: {format_hms(x, decimals=2)}"]
        found: list[tuple[str, float]] = []

        if len(visible) > 1:
            # Every series reports the SAME moment -- snapping each to its own
            # nearest dip would compare different frames against each other.
            target_time = self._find_shared_hover_time(visible, x, y, x_per_pixel)
            picks = [(entry, self._nearest_index_by_time(entry.times, target_time)) for entry, _ in visible]
        else:
            entry, values = visible[0]
            half_window = max(entry.step, x_per_pixel) * _HOVER_SEARCH_STEPS
            picks = [(entry, self._find_hover_index(entry, values, x, y, half_window))]

        for entry, idx in picks:
            fr = entry.result.frames[idx]
            val = self.metric.value(fr)
            lines.append(
                f"[{entry.label}]  frame {fr.frame:>6}   t={format_hms(fr.time, decimals=2)}   "
                f"{self.metric.label}={self.metric.value_format.format(val)}"
            )
            found.append((entry.label, val))

        if len(found) == 2:
            (label_a, val_a), (label_b, val_b) = found
            lines.append(f"Δ ({label_a} − {label_b}) = {val_a - val_b:+.2f}")

        self.chart.set_cursor_time(float(picks[0][0].times[picks[0][1]]))
        self.hover_label.setText("\n".join(lines))


class GraphWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Explicitly a real top-level Window, not a Dialog/Tool implicitly
        # inheriting owned-window semantics from having a `parent` -- an
        # owned window doesn't get its own taskbar button on Windows, and
        # minimizing it shrinks it to a small title bar near the corner of
        # the screen instead of the taskbar. Being a real top-level window
        # in the same process as the main window is what makes Windows group
        # them together in the taskbar (same app), not this flag by itself.
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("VMAF Comparison Graph")
        self.resize(1280, 800)

        self._entries: dict[int, SeriesEntry] = {}
        self._next_id = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # --- top: series list (left) + statistics (right), shared across all
        # metric tabs -- side by side instead of stacked, so this whole
        # section takes less vertical space away from the plot below.
        top = QWidget()
        top_layout = QHBoxLayout(top)

        series_col = QVBoxLayout()
        series_col.addWidget(QLabel("<b>Series</b>"))
        self.series_list_layout = QVBoxLayout()
        self.series_list_layout.setAlignment(Qt.AlignTop)
        series_list_container = QWidget()
        series_list_container.setLayout(self.series_list_layout)
        series_col.addWidget(series_list_container)
        series_col.addStretch(1)
        top_layout.addLayout(series_col)

        stats_col = QVBoxLayout()
        stats_col.addWidget(QLabel("<b>Statistics</b>"))
        self.stats_table = QTableWidget()
        self.stats_table.setMaximumHeight(160)
        stats_col.addWidget(self.stats_table)
        top_layout.addLayout(stats_col, stretch=1)

        root.addWidget(top)

        # --- below: one tab per metric, each with its own full-width plot ---
        # Each tab's plot is its own OpenGL-backed widget, and the *first*
        # one created in the whole process pays a large one-time GL driver
        # initialization cost (~170MB, measured) -- building all 4 up front
        # means paying a smaller (but non-zero, ~15-20MB each) share of that
        # for tabs the user may never look at. Only VMAF (the default,
        # always-visible tab) is built eagerly; PSNR/SSIM/XPSNR are built
        # lazily the first time their tab is actually selected.
        self.tabs = QTabWidget()
        self._pages: dict[str, _MetricPage] = {}
        vmaf_page = self._build_page(METRICS[0])
        self.tabs.addTab(vmaf_page, METRICS[0].label)
        for metric in METRICS[1:]:
            self.tabs.addTab(QWidget(), metric.label)  # placeholder, replaced on first visit
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, stretch=1)

        root.addWidget(self._build_action_bar())

        self._setup_stats_table()

    def _build_page(self, metric: MetricSpec) -> _MetricPage:
        page = _MetricPage(metric)
        page.chart.hovered.connect(lambda x, y, p=page: p.on_hover(x, y, self._entries))
        self._pages[metric.key] = page
        return page

    def _on_tab_changed(self, index: int) -> None:
        metric = METRICS[index]
        if metric.key not in self._pages:
            page = self._build_page(metric)
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, page, metric.label)
            self.tabs.setCurrentIndex(index)
            # Backfill whatever's already loaded -- add_run() only pushed
            # curves into pages that existed at the time.
            for sid, entry in self._entries.items():
                page.set_curve(sid, entry, entry.color)
        self._refresh_stats_table()

    # ------------------------------------------------------------------ UI setup
    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)

        add_btn = QPushButton("Add saved run...")
        add_btn.clicked.connect(self._on_add_saved_run)
        layout.addWidget(add_btn)

        export_png_btn = QPushButton("Export graph as PNG")
        export_png_btn.clicked.connect(self._on_export_png)
        layout.addWidget(export_png_btn)

        export_csv_btn = QPushButton("Export CSV...")
        export_csv_btn.clicked.connect(self._on_export_csv)
        layout.addWidget(export_csv_btn)

        layout.addStretch(1)
        return bar

    def _current_metric(self) -> MetricSpec:
        return METRICS[self.tabs.currentIndex()] if self.tabs.currentIndex() >= 0 else METRICS[0]

    def _setup_stats_table(self) -> None:
        metric = self._current_metric()
        headers = ["Series", "Mean", "Median", "StDev", "Min", "Max", "10% Low", "5% Low", "1% Low", "0.1% Low"]
        headers += [f"{cmp_op} {thresh:g}" for cmp_op, thresh in metric.thresholds]
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.stats_table.verticalHeader().setVisible(False)

    # ------------------------------------------------------------------ public API
    def add_run(self, result: VmafRunResult, label: str | None = None) -> None:
        # Re-adding the same distorted file (e.g. re-selecting rows that are
        # already shown, or the window being reopened and repopulated)
        # replaces its existing series instead of stacking a duplicate.
        for existing_id, entry in list(self._entries.items()):
            if Path(entry.result.distorted) == Path(result.distorted):
                self.remove_run(existing_id)

        label = label or Path(result.distorted).stem
        color = _PALETTE[self._next_id % len(_PALETTE)]
        sid = self._next_id
        self._next_id += 1

        times = result.frames.time
        step = float(times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0

        # This list IS the legend -- a color swatch beside each label plays
        # the same role a legend's line sample would, without sitting on top
        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f"background-color: {color}; border-radius: 2px;")

        checkbox = QCheckBox(label)
        checkbox.setChecked(True)
        checkbox.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
        checkbox.stateChanged.connect(lambda state, s=sid: self._on_visibility_changed(s, state))
        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(24)
        remove_btn.clicked.connect(lambda _, s=sid: self.remove_run(s))
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 8)
        row_layout.addWidget(swatch)
        row_layout.addWidget(checkbox)
        row_layout.addWidget(remove_btn)
        self.series_list_layout.addWidget(row)

        entry = SeriesEntry(
            result=result, label=label, color=color, checkbox=checkbox,
            times=times, step=step, visible=True,
        )
        entry._row_widget = row  # type: ignore[attr-defined]
        self._entries[sid] = entry

        for page in self._pages.values():
            page.set_curve(sid, entry, color)

        self._refresh_stats_table()

    def remove_run(self, series_id: int) -> None:
        entry = self._entries.pop(series_id, None)
        if entry is None:
            return
        for page in self._pages.values():
            page.remove_curve(series_id)
        entry._row_widget.setParent(None)  # type: ignore[attr-defined]
        self._refresh_stats_table()

    # ------------------------------------------------------------------ interaction
    def _on_visibility_changed(self, series_id: int, state: int) -> None:
        entry = self._entries.get(series_id)
        if entry is None:
            return
        entry.visible = bool(state)
        for page in self._pages.values():
            page.set_visible(series_id, entry.visible)
        self._refresh_stats_table()

    # ------------------------------------------------------------------ actions
    def _on_add_saved_run(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open saved VMAF run", "", "VMAF run (*.vmafrun.json *.json)")
        if not path:
            return
        try:
            result, label = load_run(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to load run", str(e))
            return
        self.add_run(result, label)

    def _on_export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export graph", "vmaf_graph.png", "PNG image (*.png)")
        if not path:
            return
        page = self._pages[self._current_metric().key]
        page.chart.render_to_pixmap().save(path)

    def _on_export_csv(self) -> None:
        if not self._entries:
            QMessageBox.information(self, "No data", "There are no series to export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not directory:
            return
        for entry in self._entries.values():
            safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in entry.label)
            out_path = Path(directory) / f"{safe_label}.csv"
            export_csv(entry.result, out_path)
        QMessageBox.information(self, "Export complete", f"Exported {len(self._entries)} CSV file(s) to {directory}")

    def save_run_for_later(self, result: VmafRunResult, label: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save VMAF run", f"{label}.vmafrun.json", "VMAF run (*.vmafrun.json)"
        )
        if not path:
            return
        save_run(result, Path(path), label=label)

    # ------------------------------------------------------------------ stats table
    def _refresh_stats_table(self) -> None:
        self._setup_stats_table()
        metric = self._current_metric()
        page = self._pages[metric.key]
        visible_ids = [sid for sid, c in page._curves.items() if c.visible]
        self.stats_table.setRowCount(len(visible_ids))
        for row, sid in enumerate(visible_ids):
            entry = self._entries[sid]
            s = page._curves[sid].stats
            values = [entry.label] + [v for _, v in s.summary]
            values += [f"{t.percentage:.1f}%" for t in s.thresholds]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setForeground(QColor(entry.color))
                self.stats_table.setItem(row, col, item)
