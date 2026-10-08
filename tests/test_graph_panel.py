import re
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import ComparisonResult, FrameScore, VideoInfo
from vmaf_app.ui.graph_panel import (
    _XPSNR_INFINITY_PLOT_DB,
    METRICS,
    GraphPanel,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _row_for(win, label: str):
    """The merged table's first-column item for a series, by label."""
    for row in range(win.stats_table.rowCount()):
        item = win.stats_table.item(row, 0)
        if item is not None and item.text() == label:
            return item
    raise AssertionError(f"no row for {label!r}")


@pytest.mark.parametrize("all_infinite", [False, True])
def test_xpsnr_infinity_is_capped_for_plot_only(qapp, all_infinite):
    panel = GraphPanel()
    result = _fake_result("test.mp4", with_other_metrics=True)
    result.frames.xpsnr[:] = np.inf
    if not all_infinite:
        result.frames.xpsnr[0] = 110.0  # genuine finite scores are not capped
        result.frames.xpsnr[1] = np.nan
    try:
        panel.add_run(result, "test")
        panel.tabs.setCurrentIndex(5)
        page = panel._pages["xpsnr"]
        plotted = next(iter(page.chart._series.values())).values
        assert (plotted[2:] == _XPSNR_INFINITY_PLOT_DB).all()
        assert np.isposinf(result.frames.xpsnr[2:]).all()
        curve = next(iter(page._curves.values()))
        assert np.isposinf(curve.stats.maximum)
        if all_infinite:
            assert np.isposinf(curve.stats.mean)
        else:
            assert plotted[0] == 110
            assert np.isnan(plotted[1])
        assert "123 dB" in panel.metric_hint.text()
        # Above every finite score in the series, so a perfect frame draws as
        # the best one rather than dipping below merely-good frames.
        finite = plotted[np.isfinite(plotted)]
        assert (finite <= _XPSNR_INFINITY_PLOT_DB).all()
        assert "retain infinity" in panel.metric_hint.text()
        assert not panel.render_export_image().isNull()
    finally:
        panel.close()


def _fake_result(distorted_name: str, vmaf_value: float = 90.0, with_other_metrics: bool = False) -> ComparisonResult:
    info = VideoInfo(
        path=Path(distorted_name), width=1920, height=1080, fps=30.0, duration=5.0,
        nb_frames=10, codec_name="h264",
    )
    frames = [
        FrameScore(
            frame=i, time=i / 30.0, vmaf=vmaf_value,
            psnr=45.0 if with_other_metrics else None,
            ssim=0.98 if with_other_metrics else None,
            xpsnr=40.0 if with_other_metrics else None,
        )
        for i in range(10)
    ]
    return ComparisonResult(
        source=Path("source.mp4"), distorted=Path(distorted_name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _values_result(name: str, values: list[float]) -> ComparisonResult:
    info = VideoInfo(
        path=Path(name), width=1920, height=1080, fps=30.0, duration=len(values) / 30.0,
        nb_frames=len(values), codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=v) for i, v in enumerate(values)]
    return ComparisonResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _long_result(name: str, n_frames: int, fps: float = 30.0) -> ComparisonResult:
    info = VideoInfo(
        path=Path(name), width=1920, height=1080, fps=fps, duration=n_frames / fps,
        nb_frames=n_frames, codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / fps, vmaf=90.0) for i in range(n_frames)]
    return ComparisonResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=fps,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _hover(win: GraphPanel, metric: str, time: float, value: float) -> None:
    win._pages[metric].on_hover(time, value, win._entries)


# ------------------------------------------------------------------ window basics


def test_readding_the_same_file_replaces_its_series_instead_of_duplicating(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", vmaf_value=80.0))
    win.add_run(_fake_result("b.mp4", vmaf_value=85.0))
    assert len(win._entries) == 2

    # Re-adding "a.mp4" (e.g. re-selecting the same row and clicking Compare
    # again) must replace, not stack, its series.
    win.add_run(_fake_result("a.mp4", vmaf_value=99.0))
    assert len(win._entries) == 2

    a_entries = [e for e in win._entries.values() if Path(e.result.distorted) == Path("a.mp4")]
    assert len(a_entries) == 1
    assert a_entries[0].result.frames[0].vmaf == 99.0


# ------------------------------------------------------------------ multi-series color + visibility


def test_unchecking_a_series_hides_its_curve_on_every_page_and_drops_it_from_stats(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    entry_a = next((sid, e) for sid, e in win._entries.items() if Path(e.result.distorted) == Path("a.mp4"))
    sid_a = entry_a[0]

    win.set_series_visible(sid_a, False)

    assert win._pages["vmaf"]._curves[sid_a].visible is False
    # The stats table doubles as the series list, so a hidden series KEEPS
    # its row (just unchecked) -- dropping it would leave no way to switch
    # the series back on.
    assert win.stats_table.rowCount() == 2
    assert _row_for(win, "a").checkState() == Qt.Unchecked

    win.set_series_visible(sid_a, True)
    assert win._pages["vmaf"]._curves[sid_a].visible is True
    assert win.stats_table.rowCount() == 2
    assert _row_for(win, "a").checkState() == Qt.Checked


# ------------------------------------------------------------------ per-metric tabs


def test_run_with_extra_metrics_gets_a_curve_on_every_tab(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True))
    sid = next(iter(win._entries))
    for i in range(win.tabs.count()):  # visit every tab so its (lazily-built) page exists
        win.tabs.setCurrentIndex(i)

    for key in ("vmaf", "psnr", "ssim", "xpsnr"):
        assert sid in win._pages[key]._curves
    assert win._pages["psnr"].no_data_label.isVisible() is False
    assert win._pages["psnr"]._curves[sid].values[0] == 45.0
    assert win._pages["ssim"]._curves[sid].values[0] == 0.98
    assert win._pages["xpsnr"]._curves[sid].values[0] == 40.0


def test_removing_a_run_removes_its_curve_from_every_page(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True))
    sid = next(iter(win._entries))
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)

    win.remove_run(sid)

    for key in ("vmaf", "psnr", "ssim", "xpsnr"):
        assert sid not in win._pages[key]._curves


def test_stats_table_reflects_the_currently_active_tab(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True))

    win.tabs.setCurrentIndex(0)  # VMAF
    assert win.stats_table.rowCount() == 1
    headers_vmaf = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "> 95" in headers_vmaf  # VMAF's own bands
    # Every metric's mean is present whichever tab is showing, because the
    # comparison being made is between encodes, not between tabs.
    assert [m.label for m in METRICS] == headers_vmaf[1:1 + len(METRICS)]

    win.tabs.setCurrentIndex(3)  # PSNR
    assert win.stats_table.rowCount() == 1
    headers_psnr = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "> 95" not in headers_psnr  # each metric brings its own bands
    assert "> 41" in headers_psnr
    assert [m.label for m in METRICS] == headers_psnr[1:1 + len(METRICS)]
    # "Mean" is not a column any more: each metric's mean is its own column.
    assert "Mean" not in headers_psnr
    assert "Median" in headers_psnr


# ------------------------------------------------------------------ step-based hover radius

def test_series_step_matches_the_time_between_consecutive_frames(qapp):
    win = GraphPanel()
    win.add_run(_long_result("a.mp4", n_frames=7200, fps=30.0))  # a 4-minute 30fps run
    entry = next(iter(win._entries.values()))

    assert entry.step == pytest.approx(1 / 30, rel=1e-6)


# ------------------------------------------------------------------ shared frame across series

def test_multi_series_hover_reports_the_same_frame_for_every_series(qapp):
    # series a dips hard at frame 5; series b dips hard at a DIFFERENT frame
    # (8). Independently dip-snapping each series (the single-series
    # behavior) would report frame 5 for a and frame 8 for b -- not a real
    # comparison. Both must report the same frame.
    win = GraphPanel()
    win.show()
    values_a = [95, 95, 95, 95, 95, 30, 95, 95, 95, 95]
    values_b = [95, 95, 95, 95, 95, 95, 95, 95, 40, 95]
    win.add_run(_values_result("a.mp4", values_a), "a")
    win.add_run(_values_result("b.mp4", values_b), "b")
    entry_a = next(e for e in win._entries.values() if e.label == "a")

    page = win._pages["vmaf"]
    # Hover at frame a's dip -- below the 95 baseline, so it qualifies as
    # "at or below cursor Y" for whichever series is checked first.
    _hover(win, "vmaf", float(entry_a.times[5]), 70)

    text = page.hover_label.text()
    frame_a = int(re.search(r"\[a\]\s+frame\s+(\d+)", text).group(1))
    frame_b = int(re.search(r"\[b\]\s+frame\s+(\d+)", text).group(1))
    assert frame_a == frame_b


# ------------------------------------------------------------------ jump to frame

def test_going_to_a_frame_reports_every_visible_series_at_that_exact_frame(qapp):
    # Unlike hovering, which snaps to a nearby dip so a curve is easy to
    # land on, this must report the frame asked for -- that is the point of
    # comparing two encodes at one moment.
    win = GraphPanel()
    win.add_run(_values_result("a.mp4", [90.0, 91.0, 20.0, 93.0]), "a")
    win.add_run(_values_result("b.mp4", [80.0, 81.0, 82.0, 83.0]), "b")

    page = win._pages["vmaf"]
    assert page.show_frame(1, win._entries) is True

    text = page.hover_label.text()
    assert text.startswith("Frame 1")
    assert "VMAF v0.6.1=91.00" in text and "VMAF v0.6.1=81.00" in text
    assert "VMAF v0.6.1=20.00" not in text, "must not snap to the nearby dip"


def test_a_frame_missing_from_one_run_is_reported_not_faked(qapp):
    # A shorter or subsampled run simply may not have that frame; showing a
    # neighbour's score under the requested number would be a lie.
    win = GraphPanel()
    win.add_run(_values_result("long.mp4", [90.0] * 10), "long")
    win.add_run(_values_result("short.mp4", [80.0] * 3), "short")

    page = win._pages["vmaf"]
    assert page.show_frame(7, win._entries) is True  # the long one has it
    text = page.hover_label.text()
    assert "not in this run" in text
    assert "VMAF v0.6.1=90.00" in text


# ------------------------------------------------------------- PNG export


def test_the_exported_png_is_written_and_readable(qapp, tmp_path, monkeypatch):
    from PySide6.QtGui import QPixmap

    from vmaf_app.ui import graph_panel as graph_panel_module

    panel = GraphPanel()
    panel.add_run(_values_result("a.mkv", [95.0] * 10), "encode-a")
    out = tmp_path / "graph.png"
    monkeypatch.setattr(
        graph_panel_module.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(out), "PNG image (*.png)"),
    )

    panel._on_export_png()

    assert out.exists() and out.stat().st_size > 0
    assert not QPixmap(str(out)).isNull(), "the file is not a readable image"


# ------------------------------------------------------------------ lower-is-better (Butteraugli)

def _butteraugli_result(name: str, values: list[float]) -> ComparisonResult:
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet

    result = _values_result(name, [90.0] * len(values))
    provenance = MetricProvenance("butteraugli", "0.12", "cpu", "butteraugli-libjxl-cpu-v1")
    result.metric_results = MetricResultSet([FrameMetricResult(
        "butteraugli", list(range(len(values))), [i / 30.0 for i in range(len(values))], values, provenance,
    )])
    return result


def _butteraugli_page(win: GraphPanel):
    win.tabs.setCurrentIndex(next(i for i, m in enumerate(METRICS) if m.key == "butteraugli"))
    return win._pages["butteraugli"]


def test_butteraugli_detail_columns_are_the_high_tail(qapp):
    win = GraphPanel()
    values = [float(v) for v in range(100)]  # 0 is best, 99 worst
    win.add_run(_butteraugli_result("a.mp4", values))
    _butteraugli_page(win)
    headers = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "10% High" in headers and "0.1% High" in headers
    assert "10% Low" not in headers
    cell = win.stats_table.item(0, headers.index("10% High")).text()
    assert float(cell) == pytest.approx(89.1)  # the 90th percentile, not the 10th

    win.tabs.setCurrentIndex(0)  # VMAF keeps its low tail
    headers = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "10% Low" in headers and "10% High" not in headers


# ------------------------------------------------------------------ other metrics beside the selected one


def _rows(page) -> list[str]:
    return [line for line in page.hover_label.text().splitlines() if line.startswith("[")]
