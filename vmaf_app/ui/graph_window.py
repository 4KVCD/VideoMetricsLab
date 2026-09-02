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

import bisect
import os
from dataclasses import dataclass, field
from pathlib import Path

import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 - registers pg.exporters.ImageExporter
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from vmaf_app.core.models import FrameScore, VmafRunResult
from vmaf_app.core.run_io import export_csv, load_run, save_run
from vmaf_app.core.stats import DEFAULT_THRESHOLDS, VmafStats, compute_stats
from vmaf_app.core.time_format import format_hms

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

# How many "x steps" to search either side of the cursor for a point to lock
# onto -- see the step calculation in _MetricPage._on_mouse_moved.
_HOVER_SEARCH_STEPS = 5

# Smallest Y span a bounded-scale plot (VMAF) will ever show, so a run whose
# scores are all identical still gets a real axis instead of a zero-height
# (and then pyqtgraph-substituted, wildly wrong) one -- see _update_y_range.
_Y_AXIS_MIN_SPAN = 5


@dataclass
class MetricSpec:
    key: str  # "vmaf", "psnr", "ssim", "xpsnr" -- also the FrameScore attribute name
    label: str  # tab title / series-list column label
    axis_label: str  # plot Y-axis label
    value_format: str  # format spec for hover-text values, e.g. "{:.2f}"
    fixed_y_max: float | None  # VMAF's natural ceiling (100); None = let pyqtgraph auto-range
    thresholds: list[tuple[str, float]] = field(default_factory=list)  # only meaningful on VMAF's fixed 0-100 scale

    def value(self, frame: FrameScore) -> float | None:
        return getattr(frame, self.key)


METRICS: list[MetricSpec] = [
    MetricSpec("vmaf", "VMAF", "VMAF", "{:.2f}", fixed_y_max=100.0, thresholds=DEFAULT_THRESHOLDS),
    MetricSpec("psnr", "PSNR", "PSNR (dB)", "{:.2f}", fixed_y_max=None),
    MetricSpec("ssim", "SSIM", "SSIM", "{:.4f}", fixed_y_max=None),
    MetricSpec("xpsnr", "XPSNR", "XPSNR (dB)", "{:.2f}", fixed_y_max=None),
]


class TimeAxisItem(pg.AxisItem):
    """An axis that renders tick values as H:M:S instead of raw seconds."""

    def tickStrings(self, values, scale, spacing):
        return [format_hms(v) for v in values]


class _CrosshairOverlay(QWidget):
    """Draws just the vertical hover line, as a plain-QPainter widget stacked
    on top of the plot's viewport instead of a pyqtgraph scene item.

    A pyqtgraph item living in the same GL-backed scene as the curves means
    every setPos() on it invalidates that scene -- and QOpenGLWidget redraws
    its whole framebuffer on any invalidation, so moving only the crosshair
    was repainting every curve and axis too on every mouse move. This widget
    is a separate, ordinary (non-GL) sibling that Qt composites on top of the
    plot's backing store, so updating it only repaints this thin transparent
    layer -- the curves/axes underneath are untouched.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.line_x: float | None = None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.line_x is None:
            return
        painter = QPainter(self)
        painter.setPen(pg.mkPen("#999", width=1))
        painter.drawLine(int(self.line_x), 0, int(self.line_x), self.height())
        painter.end()


@dataclass
class SeriesEntry:
    result: VmafRunResult
    label: str
    color: str
    checkbox: QCheckBox
    times: list[float]
    step: float  # typical time delta between consecutive points in this series
    visible: bool = True


@dataclass
class _MetricCurve:
    curve: pg.PlotDataItem
    stats: VmafStats
    # Cached once per add_run rather than rebuilt on every hover move -- see
    # _on_mouse_moved, which used to be the actual measured CPU bottleneck
    # for exactly this kind of per-call list-building on a long run.
    values: list[float]


class _MetricPage(QWidget):
    """One metric's own plot + crosshair + hover readout. Curves for a given
    series only exist here if that run actually has this metric's data (e.g.
    a run without "Also compute PSNR" checked has no curve on the PSNR page).
    """

    def __init__(self, metric: MetricSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metric = metric
        self._curves: dict[int, _MetricCurve] = {}  # series_id -> curve/stats, only entries with data
        self._autorange_frozen = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        use_opengl = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        pg.setConfigOptions(antialias=False, background="w", foreground="k", useOpenGL=use_opengl)

        self.plot_widget = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.plot_widget.setLabel("bottom", "Time (H:M:S)")
        self.plot_widget.setLabel("left", metric.axis_label)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        if metric.fixed_y_max is not None:
            self.plot_widget.setYRange(0, metric.fixed_y_max, padding=0)
        layout.addWidget(self.plot_widget)

        self.no_data_label = QLabel(
            f"No {metric.label} data among the currently visible series -- "
            f"check \"Also compute {metric.label}\" before running to see it here."
        )
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet("color: #888; font-style: italic; padding: 12px;")
        self.no_data_label.setVisible(False)
        layout.addWidget(self.no_data_label)

        self.hover_label = QLabel(
            "Hover over the graph to inspect a point (locks onto the lowest nearby "
            "score at or below your cursor, so dips are easy to land on)."
        )
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.hover_label.setStyleSheet("font-family: Consolas, monospace; padding: 6px;")
        self.hover_label.setMinimumHeight(90)
        self.hover_label.setWordWrap(True)
        layout.addWidget(self.hover_label)

        viewport = self.plot_widget.viewport()
        self._crosshair_overlay = _CrosshairOverlay(viewport)
        self._crosshair_overlay.setGeometry(viewport.rect())
        self._crosshair_overlay.show()
        self._crosshair_overlay.raise_()
        viewport.installEventFilter(self)

        # The mouse-move proxy is created by GraphWindow, not here -- it
        # needs the shared _entries map (by series id -> SeriesEntry) that
        # only GraphWindow owns, since the same series appears on every page.

    # ------------------------------------------------------------------ curves
    def set_curve(self, series_id: int, entry: SeriesEntry, color: str) -> None:
        """(Re)builds this series' curve on this page from its current data,
        or removes it if the run has no data for this metric."""
        self.remove_curve(series_id)
        # PSNR/SSIM/XPSNR are always computed for every frame of a run or
        # none at all -- it's a per-run option (a checkbox before running),
        # never a per-frame one -- so checking the first frame is enough to
        # know whether this metric applies to the whole run, and `values`
        # always lines up index-for-index with `entry.times` with no gaps.
        if not entry.result.frames or self.metric.value(entry.result.frames[0]) is None:
            self._update_no_data_label()
            return
        values = [self.metric.value(f) for f in entry.result.frames]
        curve = self.plot_widget.plot(entry.times, values, pen=pg.mkPen(color, width=2))
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        curve.setVisible(entry.visible)
        stats = compute_stats(values, self.metric.thresholds)
        self._curves[series_id] = _MetricCurve(curve=curve, stats=stats, values=values)
        self._update_no_data_label()
        self._update_y_range()
        self._unfreeze_autorange()

    def remove_curve(self, series_id: int) -> None:
        entry = self._curves.pop(series_id, None)
        if entry is None:
            return
        self.plot_widget.removeItem(entry.curve)
        self._update_no_data_label()
        self._update_y_range()
        self._unfreeze_autorange()

    def set_visible(self, series_id: int, visible: bool) -> None:
        entry = self._curves.get(series_id)
        if entry is None:
            return
        entry.curve.setVisible(visible)
        self._update_y_range()

    def _update_no_data_label(self) -> None:
        self.no_data_label.setVisible(not self._curves)

    # ------------------------------------------------------------------ Y range
    def _update_y_range(self) -> None:
        if self.metric.fixed_y_max is None:
            return  # unbounded metric (PSNR/SSIM/XPSNR) -- pyqtgraph's own autorange handles it
        visible_stats = [c.stats for c in self._curves.values() if c.curve.isVisible()]
        y_min = 0 if not visible_stats else int(min(s.minimum for s in visible_stats) // 5) * 5
        # A run where every frame scores the ceiling (a lossless or
        # near-lossless encode) floors to the ceiling too, and asking for a
        # zero-height range makes pyqtgraph substitute its own -- which came
        # out as 50..150, i.e. half the plot showing impossible >100 scores.
        # Always leave at least one 5-point band below the ceiling.
        y_min = min(y_min, self.metric.fixed_y_max - _Y_AXIS_MIN_SPAN)
        # Capped exactly at the metric's ceiling -- no headroom margin above
        # it, so the highest points sit right at the plot's own top edge
        # instead of leaving a strip of empty space above 100.
        self.plot_widget.setYRange(y_min, self.metric.fixed_y_max, padding=0)

    def _unfreeze_autorange(self) -> None:
        """Re-enables autorange after the data actually changes, so the next
        paint still fits new/removed data -- _on_mouse_moved freezes it again
        on the next hover. See _on_mouse_moved for why it's frozen at all."""
        self._autorange_frozen = False
        vb = self.plot_widget.getPlotItem().vb
        vb.enableAutoRange(x=True, y=(self.metric.fixed_y_max is None))

    # ------------------------------------------------------------------ hover
    @staticmethod
    def _nearest_index_by_time(times: list[float], x: float) -> int:
        idx = bisect.bisect_left(times, x)
        if idx <= 0:
            return 0
        if idx >= len(times):
            return len(times) - 1
        return idx if (times[idx] - x) < (x - times[idx - 1]) else idx - 1

    def _find_hover_index(
        self, entry: SeriesEntry, values: list[float], x: float, y: float, half_window: float,
    ) -> int:
        """Finds the frame to report for this series at the cursor.

        Rather than the single nearest-in-time point (which makes a sharp,
        narrow dip nearly impossible to land the cursor on), this looks at
        every point within `half_window` of the cursor's time position and:
        prefers the one closest in time that's at or below the cursor's Y
        position -- so hovering anywhere near a dip "grabs" it; and falls
        back to the single lowest-value point in that neighborhood if
        nothing there is at or below the cursor's Y.

        Only used when a single series is visible on this page -- see
        _find_shared_hover_time for why comparing multiple series needs a
        shared target time instead.
        """
        times = entry.times
        lo = bisect.bisect_left(times, x - half_window)
        hi = bisect.bisect_right(times, x + half_window)
        if lo >= hi:
            return self._nearest_index_by_time(times, x)

        best_below_idx = -1
        best_below_dist = 0.0
        fallback_idx = lo
        fallback_val = values[lo]
        for i in range(lo, hi):
            v = values[i]
            if v <= y:
                dist = abs(times[i] - x)
                if best_below_idx == -1 or dist < best_below_dist:
                    best_below_idx = i
                    best_below_dist = dist
            if v < fallback_val:
                fallback_val = v
                fallback_idx = i
        return best_below_idx if best_below_idx != -1 else fallback_idx

    def _find_shared_hover_time(
        self, pages: list[tuple[SeriesEntry, list[float]]], x: float, y: float, x_per_pixel: float,
    ) -> float:
        """The multi-series equivalent of _find_hover_index: picks ONE target
        time using the same dip-snap rule (nearest-in-time among points
        at/below the cursor's Y across ALL visible series pooled together,
        falling back to the single lowest point if none qualify) so every
        series' readout refers to the exact same moment.
        """
        found_below = False
        best_below_time = 0.0
        best_below_dist = 0.0
        found_any = False
        fallback_time = 0.0
        fallback_val = 0.0
        for entry, values in pages:
            step = max(entry.step, x_per_pixel)
            half_window = step * _HOVER_SEARCH_STEPS
            times = entry.times
            lo = bisect.bisect_left(times, x - half_window)
            hi = bisect.bisect_right(times, x + half_window)
            for i in range(lo, hi):
                t = times[i]
                v = values[i]
                if not found_any or v < fallback_val:
                    fallback_time, fallback_val = t, v
                found_any = True
                if v <= y:
                    dist = abs(t - x)
                    if not found_below or dist < best_below_dist:
                        best_below_time = t
                        best_below_dist = dist
                        found_below = True

        if not found_any:
            return x
        return best_below_time if found_below else fallback_time

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.plot_widget.viewport() and event.type() == QEvent.Resize:
            self._crosshair_overlay.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def _on_mouse_moved(self, evt, entries_by_id: dict[int, SeriesEntry]) -> None:
        pos = evt[0]
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            self._crosshair_overlay.line_x = None
            self._crosshair_overlay.update()
            return
        view_box = self.plot_widget.getPlotItem().vb
        if not self._autorange_frozen:
            # Left on since the last data change, autorange makes the
            # ViewBox re-evaluate "should I refit the view" on every single
            # repaint -- real, measured CPU cost, for zero benefit once
            # hovering (not changing data) is all that's happening. Frozen
            # here rather than immediately on a data change because doing it
            # there can race ahead of the widget having real geometry.
            view_box.enableAutoRange(x=False, y=False)
            self._autorange_frozen = True
        mouse_point = view_box.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        self._crosshair_overlay.line_x = self.plot_widget.mapFromScene(pos).x()
        self._crosshair_overlay.update()

        x_range = view_box.viewRange()[0]
        viewport_px = max(1.0, view_box.width())
        x_per_pixel = (x_range[1] - x_range[0]) / viewport_px

        # (entry, cached values) for every series actually plotted (has data
        # for this metric) and currently visible.
        visible = [
            (entries_by_id[sid], c.values)
            for sid, c in self._curves.items() if c.curve.isVisible()
        ]

        lines = [f"Time: {format_hms(x, decimals=2)}"]
        found: list[tuple[str, float]] = []

        if len(visible) > 1:
            target_time = self._find_shared_hover_time(visible, x, y, x_per_pixel)
            for entry, values in visible:
                idx = self._nearest_index_by_time(entry.times, target_time)
                fr = entry.result.frames[idx]
                val = self.metric.value(fr)
                lines.append(
                    f"[{entry.label}]  frame {fr.frame:>6}   t={format_hms(fr.time, decimals=2)}   "
                    f"{self.metric.label}={self.metric.value_format.format(val)}"
                )
                found.append((entry.label, val))
        else:
            for entry, values in visible:
                step = max(entry.step, x_per_pixel)
                half_window = step * _HOVER_SEARCH_STEPS
                idx = self._find_hover_index(entry, values, x, y, half_window)
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
        page._proxy = pg.SignalProxy(
            page.plot_widget.scene().sigMouseMoved, rateLimit=60,
            slot=lambda evt, p=page: p._on_mouse_moved(evt, self._entries),
        )
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

        times = [f.time for f in result.frames]
        step = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0

        # This list IS the legend -- a color swatch beside each label plays
        # the same role a legend's line sample would, without sitting on top
        # of the data the way pyqtgraph's own in-plot legend used to.
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
        exporter = pg.exporters.ImageExporter(page.plot_widget.plotItem)
        exporter.export(path)

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
        visible_ids = [sid for sid, c in page._curves.items() if c.curve.isVisible()]
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
