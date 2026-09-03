"""The two custom table widgets, driven through real mouse events."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QTableWidget

from vmaf_app.ui.widgets import CheckableHeaderView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def header(qapp):
    """A 4-column table whose last three columns carry checkboxes, laid out
    the way the distorted-files table's metric columns are."""
    table = QTableWidget(1, 4)
    view = CheckableHeaderView({1: False, 2: True, 3: False})
    table.setHorizontalHeader(view)
    table.setHorizontalHeaderLabels(["Path", "PSNR", "SSIM", "XPSNR"])
    for col in range(4):
        table.setColumnWidth(col, 120)
    table.resize(600, 200)
    table.show()
    QApplication.processEvents()
    # A yield rather than a return so the local `table` stays referenced for
    # the duration of the test: the header is owned by the table on the C++
    # side, and letting the last Python reference drop deletes both.
    yield view
    table.deleteLater()


def _press(view: CheckableHeaderView, point: QPoint) -> None:
    view.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(point), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier,
    ))


def _toggles(view: CheckableHeaderView) -> list[tuple[int, bool]]:
    seen: list[tuple[int, bool]] = []
    view.sectionToggled.connect(lambda s, v: seen.append((s, v)))
    return seen


def test_clicking_the_indicator_toggles_that_metric(header):
    seen = _toggles(header)

    _press(header, header.section_indicator_rect(1).center())

    assert header.is_checked(1) is True
    assert seen == [(1, True)]


def test_clicking_the_label_away_from_the_indicator_does_not_toggle(header):
    # The reported defect: anywhere in the section counted as the checkbox,
    # so reading the header by clicking it changed what the next run would
    # compute.
    seen = _toggles(header)
    rect = header.section_indicator_rect(1)
    away = QPoint(rect.right() + 30, rect.center().y())

    _press(header, away)

    assert header.is_checked(1) is False
    assert seen == []


def test_clicking_near_the_section_boundary_does_not_toggle(header):
    # Grabbing the boundary is how a column gets resized. Toggling there
    # meant the metric flipped every time the column was dragged.
    seen = _toggles(header)
    boundary = QPoint(
        header.sectionViewportPosition(2) + header.sectionSize(2) - 1,
        header.height() // 2,
    )

    _press(header, boundary)

    assert header.is_checked(2) is True, "the metric was toggled by a resize drag"
    assert seen == []


def test_a_section_with_no_checkbox_is_untouched(header):
    seen = _toggles(header)

    _press(header, QPoint(header.sectionViewportPosition(0) + 5, header.height() // 2))

    assert seen == []


def test_the_clickable_area_matches_where_the_checkbox_is_drawn(header):
    # If these ever diverge the checkbox becomes a target you can see but
    # not hit, which is worse than the original bug.
    for section in (1, 2, 3):
        rect = header.section_indicator_rect(section)
        assert rect.width() > 0 and rect.height() > 0
        assert header.logicalIndexAt(rect.center()) == section
        assert rect.left() >= header.sectionViewportPosition(section)
        assert rect.right() <= header.sectionViewportPosition(section) + header.sectionSize(section)


def test_toggling_is_still_reported_for_every_checkable_section(header):
    seen = _toggles(header)

    for section in (1, 2, 3):
        _press(header, header.section_indicator_rect(section).center())

    assert seen == [(1, True), (2, False), (3, True)]
