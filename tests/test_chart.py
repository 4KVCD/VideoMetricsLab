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

def test_time_ticks_land_on_readable_intervals(subtests):
    def check(span, expected):
        # The first step at or above span/_TARGET_TICKS, so ticks are never
        # crowded (erring toward fewer, always on round intervals).
        assert _nice_time_step(span) == expected
        assert span / _nice_time_step(span) <= 8

    for span, expected in [
        (10, 2), (60, 10), (600, 120), (7200, 900),
    ]:
        with subtests.test(span=span, expected=expected):
            check(span, expected)


def test_value_ticks_use_1_2_5_steps():
    assert _nice_value_step(100) in (10, 20)
    assert _nice_value_step(1.0) in (0.1, 0.2)
    assert _nice_value_step(0) == 1.0  # degenerate span must not divide by zero


# ------------------------------------------------------------------ ranges


def test_y_range_expands_to_show_vmaf_v1_scores_above_100(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([99.0, 105.0, 110.0]))

    assert chart.y_range() == (95.0, 110.0)


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


# ------------------------------------------------------------------ coordinate mapping

def test_time_and_pixel_mapping_round_trip(qapp):
    chart = _chart(qapp)
    chart.set_series(0, _series([90.0] * 300))
    for t in (0.0, 2.5, 9.9):
        assert chart.time_at(chart.pixel_for_time(t)) == pytest.approx(t, abs=chart.seconds_per_pixel())


# ------------------------------------------------------------------ rendering


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


# ------------------------------------------------------------------ zoom / pan


def test_an_inverted_axis_puts_the_lowest_value_at_the_top(qapp):
    """For Butteraugli, where 0 is best: 0 at the top, so a better encode is
    higher on screen like on every other graph. Drawing, gridlines and the
    value reported under the pointer all follow the same flip."""
    upright, inverted = _chart(qapp, fixed_y_max=None), ChartWidget(invert_y=True)
    inverted.resize(800, 400)
    for chart in (upright, inverted):
        chart.set_series(1, _series([0.0, 1.5, 3.0]))
    rect = inverted.plot_rect()
    y0, y1 = inverted.y_range()

    pixel = 1.5 * (y1 - y0) / rect.height()  # QRect.bottom() is top + height - 1
    assert inverted.value_at(rect.top()) == pytest.approx(y0, abs=pixel)
    assert inverted.value_at(rect.bottom()) == pytest.approx(y1, abs=pixel)
    assert upright.value_at(rect.top()) == pytest.approx(y1, abs=pixel)

    ticks = inverted.value_ticks()
    assert ticks[0][1] == "0"
    rows = [py for py, _label in ticks]
    assert rows == sorted(rows), "rising values must run down the inverted axis"
    assert ticks[0][0] < ticks[-1][0]
