"""Video Compare's zoom (vmaf_app.ui.zoom): no widgets, only the geometry."""
from dataclasses import astuple

import pytest

from vmaf_app.ui.zoom import Zoom, parse_percent, percent_text

VIEW = (1000.0, 500.0)
FRAME = (3840, 1600)


def _at(zoom, ratio=1.0):
    return astuple(zoom.placement(VIEW, FRAME, ratio))


def test_fit_puts_the_whole_frame_in_the_view_centred():
    zoom = Zoom()
    assert _at(zoom) == pytest.approx((0.0, 41.667, 1000.0, 416.667), abs=1e-3)
    assert not zoom.can_drag(VIEW, FRAME, 1.0)
    assert not zoom.drag(100, 0, VIEW, FRAME, 1.0)


def test_100_percent_is_one_frame_pixel_on_one_screen_pixel():
    zoom = Zoom()
    zoom.factor = 1.0
    assert _at(zoom) == pytest.approx((-1420.0, -550.0, 3840.0, 1600.0))
    # At 175% Windows scaling a logical pixel is 1.75 screen pixels.
    assert _at(zoom, 1.75)[2:] == pytest.approx((3840 / 1.75, 1600 / 1.75))


def test_dragging_moves_the_frame_and_stops_at_its_edges():
    zoom = Zoom()
    zoom.factor = 1.0
    assert zoom.drag(100, 40, VIEW, FRAME, 1.0)
    assert _at(zoom) == pytest.approx((-1320.0, -510.0, 3840.0, 1600.0))
    # Far past the left edge: the frame's left edge stops at the view's.
    assert zoom.drag(10_000, 0, VIEW, FRAME, 1.0)
    assert _at(zoom)[0] == pytest.approx(0.0, abs=1e-9)
    assert not zoom.drag(10, 0, VIEW, FRAME, 1.0)  # no further
    assert zoom.drag(-10_000, -10_000, VIEW, FRAME, 1.0)
    x, y, width, height = _at(zoom)
    assert (x + width, y + height) == pytest.approx((1000.0, 500.0))


def test_a_typed_zoom_is_read_as_a_percentage_within_bounds():
    assert parse_percent("150") == 1.5
    assert parse_percent(" 150 % ") == 1.5
    assert parse_percent("12,5%") == 0.125
    assert parse_percent("5") == 0.1 and parse_percent("100000") == 16.0
    assert parse_percent("Fit") is None and parse_percent("") is None and parse_percent("nan") is None
    assert percent_text(1.5) == "150%" and percent_text(0.125) == "12.5%" and percent_text(1.0) == "100%"
