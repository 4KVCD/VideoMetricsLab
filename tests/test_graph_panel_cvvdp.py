"""CVVDP in Metric Graphs: its per-second JOD tab and its whole-video score."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tests.factories import fake_run_result
from vmaf_app.core.metric_results import MetricProvenance, MetricResultSet, SequenceMetricResult
from vmaf_app.core.metrics import METRICS
from vmaf_app.ui.graph_panel import _MEAN_COLUMNS, GraphPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _result(name: str, score: float, seconds: list[float], fps: float = 24.0):
    result = fake_run_result(name, n_frames=len(seconds) * int(fps))
    starts = np.arange(len(seconds)) * int(fps)
    result.merge_metric_results(MetricResultSet([SequenceMetricResult(
        "cvvdp", score, MetricProvenance("Vship/cvvdp", "5.1.1", "gpu", "cvvdp-vship-gpu-v1"),
        frame=starts, time=starts / fps, values=seconds,
    )]))
    return result


def _cvvdp_page(panel):
    panel.tabs.setCurrentIndex([m.key for m in METRICS].index("cvvdp"))
    return panel._pages["cvvdp"]


def test_the_cvvdp_tab_plots_each_second_at_its_middle(qapp):
    panel = GraphPanel()
    panel.add_run(_result("a.mkv", 9.61, [10.0, 9.5, 9.2]), "a")
    page = _cvvdp_page(panel)
    (curve,) = page._curves.values()
    np.testing.assert_allclose(curve.times, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(curve.starts, [0.0, 1.0, 2.0])
    assert list(curve.frames) == [0, 24, 48]
    assert "JOD of each second" in panel.metric_hint.text()
    panel.close()


def test_the_readout_names_the_second_the_cursor_is_over(qapp):
    panel = GraphPanel()
    panel.add_run(_result("a.mkv", 9.61, [10.0, 9.5, 9.2]), "a")
    page = _cvvdp_page(panel)
    page.on_hover(1.9, 11.0, panel._entries)  # late in the second that starts at 1 s
    assert "second from frame     24" in page._hover_text
    assert "t=0:00:01.00" in page._hover_text and "CVVDP=9.500" in page._hover_text
    panel.close()


def test_go_to_frame_reports_the_second_holding_that_frame(qapp):
    panel = GraphPanel()
    panel.add_run(_result("a.mkv", 9.61, [10.0, 9.5, 9.2]), "a")
    page = _cvvdp_page(panel)
    assert page.show_frame(50, panel._entries)
    assert "second from frame     48" in page._hover_text and "CVVDP=9.200" in page._hover_text
    panel.close()


def test_the_stats_table_shows_the_whole_video_score_not_the_mean_of_seconds(qapp):
    panel = GraphPanel()
    # The seconds average 9.567; the video's own score is 9.61.
    panel.add_run(_result("a.mkv", 9.61, [10.0, 9.5, 9.2]), "a")
    _cvvdp_page(panel)
    column = _MEAN_COLUMNS[[m.key for m in METRICS].index("cvvdp")]
    assert panel.stats_table.item(0, column).text() == "9.610"
    assert "whole video" in panel.stats_table.horizontalHeaderItem(column).toolTip()
    headers, rows = panel._export_table()
    assert headers[1] == "Whole video" and rows[0][2][0] == "9.610"
    panel.close()


def test_a_run_without_cvvdp_leaves_its_tab_empty_and_its_cell_blank(qapp):
    panel = GraphPanel()
    panel.add_run(fake_run_result("plain.mkv"), "plain")
    page = _cvvdp_page(panel)
    assert page._curves == {}
    column = _MEAN_COLUMNS[[m.key for m in METRICS].index("cvvdp")]
    assert panel.stats_table.item(0, column).text() == "—"
    panel.close()
