"""The two custom table widgets, driven through real mouse events."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
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
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)


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
