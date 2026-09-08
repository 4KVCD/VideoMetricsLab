import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.ui.chart import ChartSeries, ChartWidget, _nice_time_step, _nice_value_step


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _series(values, fps=30.0, color="#4C72B0"):
    values = np.asarray(values, dtype=np.float32)
    times = np.arange(len(values), dtype=np.float64) / fps
    return ChartSeries(times=times, values=values, color=color)


def _chart(qapp, fixed_y_max=100.0, size=(800, 400)):
    chart = ChartWidget(y_axis_label="VMAF", fixed_y_max=fixed_y_max)
    chart.resize(*size)
    return chart


# ------------------------------------------------------------------ tick spacing

@pytest.mark.parametrize("span, expected", [
    (10, 2), (60, 10), (600, 120), (7200, 900),
])
def test_time_ticks_land_on_readable_intervals(span, expected):
    # The first step at or above span/_TARGET_TICKS, so ticks are never
    # crowded (erring toward fewer, always on round intervals).
    assert _nice_time_step(span) == expected
    assert span / _nice_time_step(span) <= 8


def test_value_ticks_use_1_2_5_steps():
    assert _nice_value_step(100) in (10, 20)
    assert _nice_value_step(1.0) in (0.1, 0.2)
    assert _nice_value_step(0) == 1.0  # degenerate span must not divide by zero


def test_chart_cache_tracks_display_pixel_density(qapp, monkeypatch):
    chart = _chart(qapp)
    monkeypatch.setattr(chart, "devicePixelRatioF", lambda: 1.5)
    exported = chart.render_to_pixmap()
    assert chart._cache.width() == 1200
    assert chart._cache.height() == 600
    assert chart._cache.devicePixelRatioF() == 1.5
    assert exported.devicePixelRatioF() == 1
    monkeypatch.setattr(chart, "devicePixelRatioF", lambda: 2.0)
    chart.render_to_pixmap()
    assert chart._cache.width() == 1600
    assert chart._cache.devicePixelRatioF() == 2


# ------------------------------------------------------------------ ranges

def test_y_range_is_capped_at_the_ceiling_with_no_headroom(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 50))
    assert chart.y_range()[1] == 100.0


def test_y_range_floors_the_bottom_below_the_lowest_score(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([95.0] * 10 + [62.0] + [95.0] * 10))
    assert chart.y_range()[0] == 60.0


def test_y_range_keeps_a_real_span_when_every_score_is_the_ceiling(qapp):
    # A lossless run scores 100 on every frame; a zero-height axis would be
    # undrawable (and previously made pyqtgraph invent a 50..150 range).
    chart = _chart(qapp)
    chart.set_series(0, _series([100.0] * 20))
    low, high = chart.y_range()
    assert high == 100.0
    assert low < high


def test_all_nan_series_does_not_poison_the_axis(qapp):
    # A metric column can exist with no value on any frame (e.g. XPSNR parsed
    # from a stats file that came up empty). nanmin/nanmax return NaN for it,
    # which made the whole y-range NaN and rendered a blank chart.
    chart = _chart(qapp)
    chart.set_series(0, _series([np.nan] * 50))
    low, high = chart.y_range()
    assert np.isfinite(low) and np.isfinite(high) and low < high

    # ...and a real series alongside it still sets the range on its own.
    chart.set_series(1, _series([70.0] * 50))
    assert chart.y_range()[0] == pytest.approx(70.0, abs=10.0)
    chart.render_to_pixmap()  # must not raise


def test_unbounded_metric_autoscales_around_its_data(qapp):
    chart = _chart(qapp, fixed_y_max=None)  # e.g. PSNR in dB
    chart.set_series(0, _series([40.0, 42.0, 38.0]))
    low, high = chart.y_range()
    assert low < 38.0 and high > 42.0


@pytest.mark.parametrize("count", [12, 12000])
def test_xpsnr_infinities_do_not_hide_finite_curve(qapp, count):
    chart = _chart(qapp, fixed_y_max=None)
    values = np.resize([35., 40., np.inf, np.nan, 38.], count)
    chart.set_series(0, _series(values, color="#ff0000"))
    assert np.isfinite(chart.y_range()).all()
    image = chart.render_to_pixmap().toImage()
    assert any(image.pixelColor(x, y).name() == "#ff0000"
               for x in range(image.width()) for y in range(image.height()))


def test_hidden_series_is_excluded_from_the_range(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 20))
    chart.set_series(1, _series([40.0] * 20))
    assert chart.y_range()[0] == 40.0

    chart.set_series_visible(1, False)
    assert chart.y_range()[0] == 90.0


# ------------------------------------------------------------------ coordinate mapping

def test_time_and_pixel_mapping_round_trip(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 300))
    for t in (0.0, 2.5, 9.9):
        assert chart.time_at(chart.pixel_for_time(t)) == pytest.approx(t, abs=chart.seconds_per_pixel())


# ------------------------------------------------------------------ rendering

def test_rendering_downsamples_instead_of_drawing_every_point(qapp):
    # 200,000 points into an 800px-wide chart must not attempt 200,000 draw
    # operations -- that was exactly why the old chart needed a GPU context.
    chart = _chart(qapp)
    chart.set_series(0, _series(np.random.default_rng(0).uniform(80, 100, 200_000)))
    pixmap = chart.render_to_pixmap()
    assert pixmap.width() == 800 and pixmap.height() == 400


def test_a_narrow_dip_survives_downsampling(qapp):
    # The whole point of min/max-per-column: a 3-frame dip in a 200k-frame
    # run must still be visible, not averaged away.
    values = np.full(200_000, 95.0, dtype=np.float32)
    values[100_000:100_003] = 10.0
    chart = _chart(qapp)
    chart.set_series(0, _series(values))
    image = chart.render_to_pixmap().toImage()

    rect = chart.plot_rect()
    low, high = chart.y_range()
    dip_y = rect.bottom() - int((10.0 - low) / (high - low) * rect.height())
    # somewhere along the row where the dip bottoms out, the curve was drawn
    row_has_ink = any(
        image.pixelColor(x, dip_y).value() < 200
        for x in range(rect.left(), rect.right())
    )
    assert row_has_ink


def test_empty_chart_renders_without_error(qapp):
    chart = _chart(qapp)
    assert chart.has_data() is False
    chart.render_to_pixmap()  # must not raise


# ------------------------------------------------------------------ crosshair

def test_cursor_only_repaints_the_columns_it_touches(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 300))
    chart.show()
    qapp.processEvents()

    updates = []
    original = chart.update

    def spy(*args):
        if args:
            updates.append(args[0])
        return original(*args)

    chart.update = spy  # type: ignore[method-assign]
    chart.set_cursor_time(5.0)

    assert updates, "moving the crosshair should request a repaint"
    assert all(r.width() <= 4 for r in updates), f"expected thin damage rects, got {updates}"


def test_setting_the_same_cursor_position_does_not_repaint(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 300))
    chart.set_cursor_time(5.0)

    updates = []
    chart.update = lambda *a: updates.append(a)  # type: ignore[method-assign]
    chart.set_cursor_time(5.0)
    assert updates == []


# ------------------------------------------------------------------ zoom / pan

def test_reset_view_returns_to_the_full_data_range(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 600))
    full = chart.x_range()

    chart._view_x = (1.0, 2.0)
    assert chart.x_range() == (1.0, 2.0)

    chart.reset_view()
    assert chart.x_range() == full


def test_zoom_keeps_the_view_inside_the_data(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 600))
    data_lo, data_hi = chart.data_x_range()

    chart._view_x = (data_lo, data_lo + (data_hi - data_lo) / 4)
    lo, hi = chart.x_range()
    assert lo >= data_lo and hi <= data_hi
