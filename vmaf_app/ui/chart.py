"""A line chart made for per-frame metric scores (ChartWidget).

It is drawn by hand rather than with pyqtgraph, which costs far more than
the app needs:

- pyqtgraph draws whole vector paths, so a feature-length run (hundreds of
  thousands of points) keeps up only with an OpenGL viewport, whose first
  context alone takes about 174 MB, and with NVIDIA's driver busy-waiting on
  each present, the CPU cost of hovering grows with the window's area.
- Here the line is reduced to one min/max pair per pixel column before it is
  drawn, so the drawing cost depends on the window's width, not the video's
  length. That is cheap enough for plain raster painting.
- The drawing is kept in a pixmap, so moving the crosshair repaints only the
  few pixel columns that changed.

On the same 3 x 144,000 points in a 2400x1300 window, hovering costs 1.5 ms
of CPU instead of pyqtgraph's 34.7 ms, and the chart takes about 49 MB of
memory instead of 259 MB.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from vmaf_app.core.time_format import format_hms
from vmaf_app.ui import theme

# Room for the axis labels around the plotting area.
_MARGIN_LEFT = 62
_MARGIN_RIGHT = 12
_MARGIN_TOP = 10
_MARGIN_BOTTOM = 38

# Tick spacings that read naturally on a time axis, in seconds.
_TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400]
_TARGET_TICKS = 8


@dataclass
class ChartSeries:
    """One line: its times (seconds, ascending), its scores, its colour, and whether it is shown."""

    times: np.ndarray   # float64 seconds, ascending
    values: np.ndarray  # float32 scores, same length as times
    color: str
    visible: bool = True


def _nice_time_step(span: float) -> float:
    """A tick spacing that gives roughly _TARGET_TICKS ticks across `span`."""
    if span <= 0:
        return 1.0
    ideal = span / _TARGET_TICKS
    for step in _TIME_STEPS:
        if step >= ideal:
            return float(step)
    # Beyond the table, keep going in whole hours.
    return float(round(ideal / 3600) * 3600 or 3600)


def _nice_value_step(span: float) -> float:
    """A 1/2/5-times-power-of-ten tick spacing for a value axis."""
    if span <= 0:
        return 1.0
    raw = span / _TARGET_TICKS
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiple in (1, 2, 5, 10):
        if magnitude * multiple >= raw:
            return float(magnitude * multiple)
    return float(magnitude * 10)


class ChartWidget(QWidget):
    """Draws one metric for any number of series.

    It knows nothing about which frame the cursor should snap to: it reports
    the time and value under the pointer, and draws the crosshair wherever
    it is told to, leaving snapping to dips to the caller.
    """

    hovered = Signal(float, float)   # (time under cursor, value under cursor)
    left = Signal()                  # pointer left the plotting area
    view_changed = Signal()          # zoom/pan happened

    def __init__(self, y_axis_label: str = "", fixed_y_max: float | None = None, parent=None,
                 invert_y: bool = False):
        super().__init__(parent)
        self.y_axis_label = y_axis_label
        self.fixed_y_max = fixed_y_max
        # Lowest value at the top, for a metric where lower is better
        # (Butteraugli: 0 is identical), so "up is better" on every graph.
        self.invert_y = invert_y
        self._series: dict[int, ChartSeries] = {}
        self._cache: QPixmap | None = None
        self._cache_dark = False  # the theme the cache was drawn in (theme.chart_colours)
        self._cursor_x: int | None = None
        self._view_x: tuple[float, float] | None = None   # None = fit all data
        self._y_range: tuple[float, float] = (0.0, 1.0)
        self._pan_origin: QPoint | None = None
        self._pan_view: tuple[float, float] | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(240, 140)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_OpaquePaintEvent)  # we paint every pixel; skip Qt's pre-clear
        # While the size keeps changing (the window dragged by an edge), the
        # last drawing is shown stretched, and the chart is drawn again once
        # the size has settled. Drawn at every size, a dozen two-hour lines
        # take 54 ms a step: 18 steps a second, or 4 on a slower PC.
        self._resize_settle = QTimer(self)
        self._resize_settle.setSingleShot(True)
        self._resize_settle.setInterval(100)
        self._resize_settle.timeout.connect(self.update)

    # ------------------------------------------------------------------ data
    def set_series(self, key: int, series: ChartSeries) -> None:
        """Adds or replaces a line."""
        self._series[key] = series
        self._invalidate(refit=True)

    def remove_series(self, *keys: int) -> None:
        """Removes lines, fitting the axes again only once however many go."""
        removed = [key for key in keys if self._series.pop(key, None) is not None]
        if removed:
            self._invalidate(refit=True)

    def set_series_visible(self, key: int, visible: bool) -> None:
        """Shows or hides a line."""
        series = self._series.get(key)
        if series is not None and series.visible != visible:
            series.visible = visible
            self._invalidate(refit=True)

    def clear(self) -> None:
        """Removes every line."""
        self._series.clear()
        self._invalidate(refit=True)

    def has_data(self) -> bool:
        """Whether any shown line has points."""
        return any(s.visible and len(s.times) for s in self._series.values())

    def _visible_series(self) -> list[ChartSeries]:
        """The shown lines that have points."""
        return [s for s in self._series.values() if s.visible and len(s.times)]

    # ------------------------------------------------------------------ ranges
    def data_x_range(self) -> tuple[float, float]:
        """The time range of the shown lines."""
        visible = self._visible_series()
        if not visible:
            return 0.0, 1.0
        lo = min(float(s.times[0]) for s in visible)
        hi = max(float(s.times[-1]) for s in visible)
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def x_range(self) -> tuple[float, float]:
        """The time range in view: zoomed, or all of it."""
        return self._view_x if self._view_x is not None else self.data_x_range()

    def y_range(self) -> tuple[float, float]:
        """The value range of the axis."""
        return self._y_range

    def reset_view(self) -> None:
        """Shows all of the time range again."""
        self._view_x = None
        self._invalidate(refit=True)
        self.view_changed.emit()

    def _recompute_y_range(self) -> None:
        """Fits the value axis to the shown lines."""
        # A line can be all NaN (a metric's column with no value for any
        # frame). nanmin and nanmax warn and return NaN for it, which would
        # give the axis a NaN range and quietly draw a blank chart, so such
        # lines are skipped.
        visible = [s for s in self._visible_series() if np.isfinite(s.values).any()]
        if not visible:
            self._y_range = (0.0, self.fixed_y_max or 1.0)
            return
        lo = min(float(np.min(s.values[np.isfinite(s.values)])) for s in visible)
        hi = max(float(np.max(s.values[np.isfinite(s.values)])) for s in visible)
        if self.fixed_y_max is not None:
            # A bounded scale (VMAF): the metric's ceiling when the scores fit
            # under it, but a higher axis when a model can go above it (VMAF
            # v1 4K/3H, say). Real scores are never cut off at the ceiling.
            top = max(self.fixed_y_max, float(np.ceil(hi / 5.0) * 5.0))
            bottom = int(np.floor(lo / 5.0)) * 5
            bottom = min(bottom, top - 5)
            self._y_range = (float(bottom), top)
        else:
            # Unbounded (decibels, SSIM): a little padding, so that the
            # extremes are not drawn right on the frame.
            pad = (hi - lo) * 0.05 or max(abs(hi) * 0.01, 0.5)
            self._y_range = (lo - pad, hi + pad)

    # ------------------------------------------------------------------ geometry
    def plot_rect(self) -> QRect:
        """Where the lines are drawn, inside the axes."""
        return QRect(
            _MARGIN_LEFT, _MARGIN_TOP,
            max(1, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM),
        )

    def time_at(self, px: int) -> float:
        """The time at a pixel column."""
        rect = self.plot_rect()
        x0, x1 = self.x_range()
        frac = (px - rect.left()) / max(1, rect.width())
        return x0 + frac * (x1 - x0)

    def _height_fraction(self, value_fraction):
        """Where a value sits, as a fraction of the plot's height from the
        bottom: its fraction of the Y range, flipped when invert_y. The same
        flip maps a height back to a value, so it serves both directions.
        Works on floats and numpy arrays alike."""
        return 1.0 - value_fraction if self.invert_y else value_fraction

    def value_at(self, py: int) -> float:
        """The value at a pixel row."""
        rect = self.plot_rect()
        y0, y1 = self._y_range
        frac = self._height_fraction((rect.bottom() - py) / max(1, rect.height()))
        return y0 + frac * (y1 - y0)

    def pixel_for_time(self, t: float) -> int:
        """The pixel column of a time."""
        rect = self.plot_rect()
        x0, x1 = self.x_range()
        return rect.left() + round((t - x0) / max(1e-9, x1 - x0) * rect.width())

    def time_tick_labels(self) -> list[str]:
        """The time axis's labels for the current view, in order: the same
        ones _draw_axes draws."""
        x0, x1 = self.x_range()
        step = _nice_time_step(x1 - x0)
        tick = np.ceil(x0 / step) * step
        labels = []
        while tick <= x1:
            labels.append(format_hms(float(tick)))
            tick += step
        return labels

    def value_ticks(self, rect: QRect | None = None) -> list[tuple[int, str]]:
        """The value axis's grid lines for the current range, as (pixel row,
        label), lowest value first: the same ones _draw_axes draws."""
        rect = rect or self.plot_rect()
        y0, y1 = self._y_range
        step = _nice_value_step(y1 - y0)
        value = np.ceil(y0 / step) * step
        ticks = []
        while value <= y1:
            frac = self._height_fraction((value - y0) / max(1e-9, y1 - y0))
            # "+ 0.0" turns the -0.0 that ceil() gives for a range starting
            # just below zero (Butteraugli's padding) into 0, not "-0".
            ticks.append((rect.bottom() - int(frac * rect.height()), f"{float(value) + 0.0:g}"))
            value += step
        return ticks

    def seconds_per_pixel(self) -> float:
        """How many seconds one pixel column covers."""
        x0, x1 = self.x_range()
        return (x1 - x0) / max(1, self.plot_rect().width())

    # ------------------------------------------------------------------ painting
    def _invalidate(self, refit: bool = False) -> None:
        """Draws the chart again, fitting the value axis first with refit."""
        if refit:
            self._recompute_y_range()
        self._cache = None
        self.update()

    def resizeEvent(self, event) -> None:
        """Draws again once a resize settles, or afresh after a resize while hidden."""
        if self._cache is not None:
            if self.isVisible():
                self._resize_settle.start()  # drawn again at the size it settles on (paintEvent)
            else:
                # Resized while hidden (the window resized on another tab),
                # it is drawn afresh when shown, not stretched for 100 ms.
                self._cache = None
        super().resizeEvent(event)

    def changeEvent(self, event) -> None:
        """Draws again in the new colours when the theme changes."""
        # The window's theme changed: drawn again in its colours.
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange, QEvent.StyleChange):
            self._cache = None
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event) -> None:
        """Copies the kept drawing to the screen, with the crosshair."""
        if not self._cache_matches_display():
            if self._cache is not None and self._resize_settle.isActive():
                painter = QPainter(self)
                painter.drawPixmap(self.rect(), self._cache)  # stretched, still being resized
                painter.end()
                return
            self._rebuild_cache()
        painter = QPainter(self)
        # Only the damaged area is copied, so moving the crosshair repaints a
        # few pixel columns, not the whole chart. Qt clips to the paint
        # event; drawing at the logical origin keeps the high-DPI pixmap's
        # source coordinates from being taken for logical ones.
        painter.drawPixmap(0, 0, self._cache)
        if self._cursor_x is not None:
            painter.setPen(QPen(theme.chart_colours(self._cache_dark).crosshair, 1))
            rect = self.plot_rect()
            painter.drawLine(self._cursor_x, rect.top(), self._cursor_x, rect.bottom())
        painter.end()

    def _rebuild_cache(self) -> None:
        """Draws the chart again into the kept pixmap."""
        self._cache_dark = theme.is_dark()
        self._cache = self._render(theme.chart_colours(self._cache_dark))

    def _render(self, colours: theme.ChartColours) -> QPixmap:
        """The chart without the crosshair, in `colours`, at the screen's pixel ratio."""
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(ceil(self.width() * ratio), ceil(self.height() * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(colours.background)
        painter = QPainter(pixmap)
        painter.setFont(self.font())
        rect = self.plot_rect()
        self._draw_axes(painter, rect, colours)
        for series in self._visible_series():
            self._draw_series(painter, rect, series)
        painter.setPen(QPen(colours.axis, 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.end()
        return pixmap

    def _cache_matches_display(self) -> bool:
        """Whether the kept drawing still fits the size, pixel ratio and theme."""
        ratio = self.devicePixelRatioF()
        # Drawn again when the window's theme changed since (theme.py).
        return (self._cache is not None
                and self._cache_dark == theme.is_dark()
                and self._cache.devicePixelRatioF() == ratio
                and self._cache.width() == ceil(self.width() * ratio)
                and self._cache.height() == ceil(self.height() * ratio))

    def _draw_axes(self, painter: QPainter, rect: QRect, colours: theme.ChartColours) -> None:
        """Draws the grid, the axes' labels and the value axis's title."""
        x0, x1 = self.x_range()
        metrics = QFontMetrics(painter.font())

        painter.setPen(QPen(colours.grid, 1))
        step = _nice_time_step(x1 - x0)
        tick = np.ceil(x0 / step) * step
        time_ticks = []
        while tick <= x1:
            px = self.pixel_for_time(float(tick))
            painter.drawLine(px, rect.top(), px, rect.bottom())
            time_ticks.append((px, float(tick)))
            tick += step

        value_ticks = self.value_ticks(rect)
        for py, _label in value_ticks:
            painter.drawLine(rect.left(), py, rect.right(), py)

        painter.setPen(QPen(colours.text, 1))
        for px, tick_time in time_ticks:
            label = format_hms(tick_time)
            painter.drawText(px - metrics.horizontalAdvance(label) // 2,
                             rect.bottom() + metrics.height() + 2, label)
        for py, label in value_ticks:
            painter.drawText(rect.left() - metrics.horizontalAdvance(label) - 6,
                             py + metrics.ascent() // 2, label)

        if self.y_axis_label:
            painter.save()
            painter.translate(14, rect.center().y())
            painter.rotate(-90)
            painter.drawText(-metrics.horizontalAdvance(self.y_axis_label) // 2, 0, self.y_axis_label)
            painter.restore()

    def _draw_series(self, painter: QPainter, rect: QRect, series: ChartSeries) -> None:
        """Draws one line: point by point when there are few, else as one min/max stroke per pixel column."""
        x0, x1 = self.x_range()
        y0, y1 = self._y_range
        times, values = series.times, series.values

        lo = int(np.searchsorted(times, x0, side="left"))
        hi = int(np.searchsorted(times, x1, side="right"))
        lo = max(0, lo - 1)
        hi = min(len(times), hi + 1)
        if hi - lo < 1:
            return

        width = rect.width()
        painter.setPen(QPen(QColor(series.color), 1))
        y_span = max(1e-9, y1 - y0)

        def to_py(v: np.ndarray) -> np.ndarray:
            """The pixel rows of values."""
            frac = self._height_fraction((v - y0) / y_span)
            return rect.bottom() - frac * rect.height()

        count = hi - lo
        if count <= width * 2:
            # Few enough points to draw honestly as a polyline.
            visible_values = values[lo:hi].astype(np.float64)
            good = np.isfinite(visible_values)
            if not good.any():
                return
            xs = rect.left() + (times[lo:hi] - x0) / max(1e-9, x1 - x0) * width
            # Draw each contiguous finite section separately: joining across
            # NaN/∞ would invent a line through a missing frame, and passing
            # infinity into QPointF is undefined.
            boundaries = np.flatnonzero(np.diff(np.r_[False, good, False]))
            for start, end in boundaries.reshape(-1, 2):
                ys = to_py(visible_values[start:end])
                points = [
                    QPointF(float(x), float(y))
                    for x, y in zip(xs[start:end], ys, strict=True)
                ]
                if len(points) == 1:
                    painter.drawPoint(points[0])
                else:
                    painter.drawPolyline(QPolygonF(points))
            return

        # More frames than pixels: reduce to one min/max pair per column so
        # the drawing cost is bounded by window width, and no dip is lost.
        edges = np.linspace(lo, hi, width + 1).astype(np.int64)
        starts = edges[:-1]
        # reduceat needs strictly ascending starts; equal neighbours would
        # silently yield the raw element instead of a reduction.
        keep = np.ones(len(starts), dtype=bool)
        keep[1:] = starts[1:] > starts[:-1]
        starts_unique = starts[keep]
        columns = np.flatnonzero(keep)

        # Restrict the final bucket to the visible slice. Treat infinities
        # like gaps, not enormous finite values that erase the curve.
        selected = values[lo:hi]
        finite = np.where(np.isfinite(selected), selected, np.inf)
        mins = np.minimum.reduceat(finite, starts_unique - lo)
        finite_max = np.where(np.isfinite(selected), selected, -np.inf)
        maxs = np.maximum.reduceat(finite_max, starts_unique - lo)
        good = np.isfinite(mins) & np.isfinite(maxs)
        if not good.any():
            return

        xs = rect.left() + columns[good]
        top = to_py(maxs[good].astype(np.float64))
        bottom = to_py(mins[good].astype(np.float64))
        painter.drawLines([
            QLineF(float(x), float(t), float(x), float(b))
            for x, t, b in zip(xs, top, bottom, strict=True)
        ])

    # ------------------------------------------------------------------ cursor
    def set_cursor_time(self, t: float | None) -> None:
        """Moves the crosshair, repainting only the columns it vacated and
        the ones it now occupies."""
        px = None if t is None else self.pixel_for_time(t)
        if px is not None:
            rect = self.plot_rect()
            if not (rect.left() <= px <= rect.right()):
                px = None
        if px == self._cursor_x:
            return
        old, self._cursor_x = self._cursor_x, px
        rect = self.plot_rect()
        for x in (old, px):
            if x is not None:
                # Pen caps and fractional display scaling can touch pixels
                # outside either endpoint. Restore that fringe at BOTH ends.
                self.update(QRect(x - 1, rect.top() - 2, 3, rect.height() + 4))

    # ------------------------------------------------------------------ input
    def mouseMoveEvent(self, event) -> None:
        """Pans while dragging; otherwise reports the time and value under the pointer."""
        pos = event.position().toPoint()
        if self._pan_origin is not None and self._pan_view is not None:
            span = self._pan_view[1] - self._pan_view[0]
            shift = (self._pan_origin.x() - pos.x()) / max(1, self.plot_rect().width()) * span
            self._view_x = (self._pan_view[0] + shift, self._pan_view[1] + shift)
            self._invalidate()
            self.view_changed.emit()
            return
        if not self.plot_rect().contains(pos):
            self.left.emit()
            return
        self.hovered.emit(self.time_at(pos.x()), self.value_at(pos.y()))

    def leaveEvent(self, event) -> None:
        """Reports that the pointer left."""
        self.left.emit()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Starts panning."""
        if event.button() == Qt.LeftButton:
            self._pan_origin = event.position().toPoint()
            self._pan_view = self.x_range()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        """Stops panning."""
        if event.button() == Qt.LeftButton:
            self._pan_origin = None
            self._pan_view = None
            self.unsetCursor()

    def mouseDoubleClickEvent(self, event) -> None:
        """Shows all of the time range again."""
        self.reset_view()

    def wheelEvent(self, event) -> None:
        """Zooms the time axis around the pointer, so whatever is under the
        cursor stays under it."""
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 0.8 ** steps
        x0, x1 = self.x_range()
        anchor = self.time_at(event.position().toPoint().x())
        new_span = (x1 - x0) * factor

        data_lo, data_hi = self.data_x_range()
        full_span = data_hi - data_lo
        if new_span >= full_span:
            self.reset_view()
            return
        new_span = max(new_span, self.seconds_per_pixel() * 8)  # don't zoom past a few pixels of data

        left_share = (anchor - x0) / max(1e-9, x1 - x0)
        new_x0 = anchor - left_share * new_span
        new_x1 = new_x0 + new_span
        # Keep the view over the data rather than drifting off the end.
        if new_x0 < data_lo:
            new_x0, new_x1 = data_lo, data_lo + new_span
        if new_x1 > data_hi:
            new_x0, new_x1 = data_hi - new_span, data_hi
        self._view_x = (new_x0, new_x1)
        self._invalidate()
        self.view_changed.emit()

    def render_to_pixmap(self) -> QPixmap:
        """The current chart for PNG export: in the light theme's colours
        whatever the window shows, as an exported file is read elsewhere."""
        exported = self._render(theme.chart_colours(dark=False))
        # Export compositors use physical pixel dimensions for layout.
        exported.setDevicePixelRatio(1.0)
        return exported
