import re
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import FrameScore, VideoInfo, VmafRunResult
from vmaf_app.ui.graph_window import GraphWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_result(distorted_name: str, vmaf_value: float = 90.0, with_other_metrics: bool = False) -> VmafRunResult:
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
    return VmafRunResult(
        source=Path("source.mp4"), distorted=Path(distorted_name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _values_result(name: str, values: list[float]) -> VmafRunResult:
    info = VideoInfo(
        path=Path(name), width=1920, height=1080, fps=30.0, duration=len(values) / 30.0,
        nb_frames=len(values), codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=v) for i, v in enumerate(values)]
    return VmafRunResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _dip_result(name: str) -> VmafRunResult:
    return _values_result(name, [95, 95, 93, 95, 95, 60, 95, 95, 95, 95])  # small low at 2, sharp dip at 5


def _long_result(name: str, n_frames: int, fps: float = 30.0) -> VmafRunResult:
    info = VideoInfo(
        path=Path(name), width=1920, height=1080, fps=fps, duration=n_frames / fps,
        nb_frames=n_frames, codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / fps, vmaf=90.0) for i in range(n_frames)]
    return VmafRunResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=fps,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


def _hover(win: GraphWindow, metric: str, time: float, value: float) -> None:
    win._pages[metric].on_hover(time, value, win._entries)


def _hover_middle(win: GraphWindow, metric: str) -> None:
    """Hover the middle of the chart's current view, at mid-height."""
    chart = win._pages[metric].chart
    x0, x1 = chart.x_range()
    y0, y1 = chart.y_range()
    _hover(win, metric, (x0 + x1) / 2, (y0 + y1) / 2)


# ------------------------------------------------------------------ window basics

def test_closing_the_window_preserves_its_series(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4"))
    win.show()

    win.close()
    assert len(win._entries) == 1

    win.show()
    assert len(win._entries) == 1
    assert win.isVisible()


def test_window_is_a_real_top_level_window_not_an_owned_dialog(qapp):
    # Windows only gives a real taskbar button (and groups it with the main
    # window) to genuine top-level windows -- a widget that inherits owned
    # (Dialog/Tool-like) semantics from having a Qt `parent` doesn't get one,
    # and minimizes into a small title bar near the screen corner instead.
    win = GraphWindow(parent=None)
    assert bool(win.windowFlags() & Qt.Window)


def test_readding_the_same_file_replaces_its_series_instead_of_duplicating(qapp):
    win = GraphWindow()
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


def test_readding_all_series_after_reopen_does_not_duplicate(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    win.close()

    # Simulates MainWindow's "Compare selected" reusing the same (now hidden)
    # window and re-adding the same runs the user re-selected.
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    win.show()

    assert len(win._entries) == 2


# ------------------------------------------------------------------ multi-series color + visibility

def test_each_series_gets_a_distinct_color(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    win.add_run(_fake_result("c.mp4"))

    colors = {e.color for e in win._entries.values()}
    assert len(colors) == 3  # all distinct


def test_unchecking_a_series_hides_its_curve_on_every_page_and_drops_it_from_stats(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    entry_a = next((sid, e) for sid, e in win._entries.items() if Path(e.result.distorted) == Path("a.mp4"))
    sid_a = entry_a[0]

    entry_a[1].checkbox.setChecked(False)

    assert win._pages["vmaf"]._curves[sid_a].visible is False
    assert win.stats_table.rowCount() == 1  # only the still-checked series shows in the stats table

    entry_a[1].checkbox.setChecked(True)
    assert win._pages["vmaf"]._curves[sid_a].visible is True
    assert win.stats_table.rowCount() == 2


# ------------------------------------------------------------------ per-metric tabs

def test_graph_has_one_tab_per_metric(qapp):
    win = GraphWindow()
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert labels == ["VMAF", "PSNR", "SSIM", "XPSNR"]


def test_non_default_tabs_are_not_built_until_first_visited(qapp):
    # Each tab's plot is its own OpenGL widget; building all 4 up front
    # meant paying real memory for tabs the user may never look at (an
    # actual measured ~15-20MB apiece on top of the ~170MB one-time GL
    # driver cost the first plot pays regardless). Only the default (VMAF)
    # tab should exist right away.
    win = GraphWindow()
    assert "vmaf" in win._pages
    assert "psnr" not in win._pages
    assert "ssim" not in win._pages
    assert "xpsnr" not in win._pages

    win.tabs.setCurrentIndex(1)  # PSNR
    assert "psnr" in win._pages
    assert "ssim" not in win._pages  # still untouched


def test_run_without_extra_metrics_only_gets_a_vmaf_curve(qapp):
    win = GraphWindow()
    win.show()  # a widget inside an inactive QTabWidget page never reports isVisible()==True
    win.add_run(_fake_result("a.mp4", with_other_metrics=False))
    sid = next(iter(win._entries))
    for i in range(1, win.tabs.count()):  # visit PSNR/SSIM/XPSNR so their (lazy) pages exist
        win.tabs.setCurrentIndex(i)
    win.tabs.setCurrentIndex(1)  # back to PSNR -- isVisible() below needs it to be the active tab

    assert sid in win._pages["vmaf"]._curves
    assert sid not in win._pages["psnr"]._curves
    assert sid not in win._pages["ssim"]._curves
    assert sid not in win._pages["xpsnr"]._curves
    assert win._pages["psnr"].no_data_label.isVisible() is True


def test_run_with_extra_metrics_gets_a_curve_on_every_tab(qapp):
    win = GraphWindow()
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
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True))
    sid = next(iter(win._entries))
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)

    win.remove_run(sid)

    for key in ("vmaf", "psnr", "ssim", "xpsnr"):
        assert sid not in win._pages[key]._curves


def test_stats_table_reflects_the_currently_active_tab(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True))

    win.tabs.setCurrentIndex(0)  # VMAF
    assert win.stats_table.rowCount() == 1
    headers_vmaf = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "> 95" in headers_vmaf  # VMAF-specific threshold columns

    win.tabs.setCurrentIndex(1)  # PSNR
    assert win.stats_table.rowCount() == 1
    headers_psnr = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "> 95" not in headers_psnr  # thresholds are VMAF-only, don't make sense on a dB scale
    assert "Mean" in headers_psnr


def test_stats_table_excludes_series_with_no_data_for_the_active_metric(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True), "a")
    win.add_run(_fake_result("b.mp4", with_other_metrics=False), "b")

    win.tabs.setCurrentIndex(0)  # VMAF -- both series have it
    assert win.stats_table.rowCount() == 2

    win.tabs.setCurrentIndex(1)  # PSNR -- only "a" has it
    assert win.stats_table.rowCount() == 1
    assert win.stats_table.item(0, 0).text() == "a"


# ------------------------------------------------------------------ stats table columns (extensible)

def test_stats_table_includes_01_percent_low_column(qapp):
    win = GraphWindow()
    headers = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "1% Low" in headers
    assert "0.1% Low" in headers


# ------------------------------------------------------------------ Y-aware hover snapping

def test_hover_locks_onto_a_dip_below_cursor_y_even_if_not_exactly_under_cursor(qapp):
    win = GraphWindow()
    win.add_run(_dip_result("a.mp4"))
    entry = next(iter(win._entries.values()))
    page = win._pages["vmaf"]

    # Cursor sits exactly at the dip's time, at Y=70 -- well above the dip
    # (60) but below the surrounding baseline (95), so only the dip qualifies.
    values = entry.result.frames.vmaf
    idx = page._find_hover_index(entry, values, x=entry.times[5], y=70, half_window=0.05)
    assert idx == 5


def test_hover_falls_back_to_local_minimum_when_nothing_is_below_cursor_y(qapp):
    win = GraphWindow()
    win.add_run(_dip_result("a.mp4"))
    entry = next(iter(win._entries.values()))
    page = win._pages["vmaf"]

    # Cursor near frame 2 (values 95,93,95 nearby), Y=70 -- nothing in this
    # neighborhood is <=70 (the big dip at frame 5 is outside the window), so
    # it should fall back to the lowest VMAF in the neighborhood (93 @ idx 2).
    values = entry.result.frames.vmaf
    idx = page._find_hover_index(entry, values, x=entry.times[2], y=70, half_window=0.05)
    assert idx == 2


def test_hover_diff_shown_for_exactly_two_visible_series(qapp):
    win = GraphWindow()
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))
    win.add_run(_fake_result("b.mp4", vmaf_value=80.0))

    page = win._pages["vmaf"]
    _hover_middle(win, "vmaf")

    assert "Δ" in page.hover_label.text()  # the delta (Δ) line appears
    assert "10.00" in page.hover_label.text()  # 90 - 80 = 10


def test_hover_diff_not_shown_for_a_single_series(qapp):
    win = GraphWindow()
    win.show()
    win.add_run(_fake_result("a.mp4"))

    page = win._pages["vmaf"]
    _hover_middle(win, "vmaf")

    assert "Δ" not in page.hover_label.text()


def test_hover_on_psnr_tab_reports_psnr_not_vmaf(qapp):
    win = GraphWindow()
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0, with_other_metrics=True))
    win.tabs.setCurrentIndex(1)  # PSNR -- lazily builds its page

    page = win._pages["psnr"]
    _hover_middle(win, "psnr")

    assert "PSNR=45.00" in page.hover_label.text()
    assert "VMAF=" not in page.hover_label.text()


# ------------------------------------------------------------------ step-based hover radius

def test_series_step_matches_the_time_between_consecutive_frames(qapp):
    win = GraphWindow()
    win.add_run(_long_result("a.mp4", n_frames=7200, fps=30.0))  # a 4-minute 30fps run
    entry = next(iter(win._entries.values()))

    assert entry.step == pytest.approx(1 / 30, rel=1e-6)


def test_hover_finds_a_narrow_dip_even_when_zoomed_out_over_a_long_run(qapp):
    # Regression check: zoomed out over a long run, each screen pixel covers
    # many frames, so the search has to span what a pixel represents or it
    # misses dips entirely -- a fixed handful of *frames* (independent of
    # zoom) was nowhere near enough at this scale and just reported whatever
    # nearly-baseline value happened to be within a few frames of the cursor.
    n, fps = 50_000, 30.0  # ~28 minutes
    values = [95.0] * n
    dip_start = 25_000
    for i in range(dip_start, dip_start + 10):
        values[i] = 30.0
    info = VideoInfo(path=Path("a.mp4"), width=1920, height=1080, fps=fps, duration=n / fps, nb_frames=n, codec_name="h264")
    frames = [FrameScore(frame=i, time=i / fps, vmaf=v) for i, v in enumerate(values)]
    result = VmafRunResult(
        source=Path("source.mp4"), distorted=Path("a.mp4"), frames=frames, fps=fps,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )

    win = GraphWindow()
    win.resize(1000, 700)
    win.add_run(result, "a")
    win.show()
    qapp.processEvents()

    page = win._pages["vmaf"]
    dip_time = (dip_start + 5) / fps
    _hover(win, "vmaf", dip_time, 90)  # over the dip's X, well above its Y

    assert "VMAF=30.00" in page.hover_label.text()


# ------------------------------------------------------------------ H:M:S time formatting

def test_hover_text_uses_hms_format(qapp):
    win = GraphWindow()
    win.show()
    win.add_run(_long_result("a.mp4", n_frames=7200, fps=30.0))  # runs to 0:04:00

    page = win._pages["vmaf"]
    # Hover near the middle of the plot (~2 minutes in).
    _hover_middle(win, "vmaf")

    text = page.hover_label.text()
    assert "Time: 0:0" in text  # H:M:S, not a bare "120.00s" style value
    assert "s" not in text.split("Time:")[1].split("\n")[0]  # no trailing "s" unit on the time itself


def test_graph_x_axis_renders_hms_ticks(qapp):
    win = GraphWindow()
    win.add_run(_long_result("a.mp4", n_frames=30 * 3725, fps=30.0))  # just over an hour
    labels = win._pages["vmaf"].chart.time_tick_labels()
    assert labels, "expected some time ticks"
    assert all(label.count(":") == 2 for label in labels), labels  # H:M:S, not bare seconds
    assert labels[0].startswith("0:")


# ------------------------------------------------------------------ Y-axis auto-scaling

def test_y_axis_bottom_rounds_down_to_nearest_5_below_lowest_score(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))
    sid = next(iter(win._entries))
    # one real low point (62 -> floors the axis to 60)
    result = win._entries[sid].result
    vmaf = result.frames.vmaf.copy()
    vmaf[3] = 62.0
    result.frames = result.frames.with_values("vmaf", vmaf)
    win.add_run(result, win._entries[sid].label)  # rebuild the curve/stats

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[0] == 60


def test_y_axis_bottom_stays_at_0_when_nothing_is_below_60(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))  # every frame is 90.0, well above 60

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[0] == 90 - (90 % 5)  # 90 is already a multiple of 5, so floor(90/5)*5 == 90


def test_y_axis_stays_sane_when_every_score_is_identical(qapp):
    # Regression test: a lossless/near-lossless run where every frame scores
    # exactly 100 floored the axis bottom to 100 too, and asking pyqtgraph
    # for a zero-height range made it substitute its own -- which came out
    # as 50..150, i.e. half the plot showing impossible >100 scores.
    win = GraphWindow()
    win.add_run(_fake_result("perfect.mp4", vmaf_value=100.0))

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[1] == 100          # still capped at the ceiling
    assert y_range[0] < y_range[1]    # and not a degenerate zero-height range
    assert y_range[0] >= 90           # tight around the data, not wildly zoomed out


def test_y_axis_top_is_capped_exactly_at_100_with_no_headroom(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[1] == 100  # no margin/wasted space above VMAF's ceiling


def test_y_axis_ignores_hidden_series_lowest_point(qapp):
    win = GraphWindow()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))
    win.add_run(_fake_result("b.mp4", vmaf_value=40.0))
    entry_b = next(e for e in win._entries.values() if e.label == "b")  # label is the stem, not the full filename

    entry_b.checkbox.setChecked(False)  # hide the low series

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[0] == 90  # bottom reflects only the visible (90.0) series, not the hidden 40.0 one


# ------------------------------------------------------------------ shared frame across series

def test_multi_series_hover_reports_the_same_frame_for_every_series(qapp):
    # series a dips hard at frame 5; series b dips hard at a DIFFERENT frame
    # (8). Independently dip-snapping each series (the single-series
    # behavior) would report frame 5 for a and frame 8 for b -- not a real
    # comparison. Both must report the same frame.
    win = GraphWindow()
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


def test_single_series_hover_still_uses_independent_dip_snap(qapp):
    # With only one series visible, the original per-series dip-snap
    # behavior (not the shared-time logic, which only matters for
    # comparing 2+ series) still applies.
    win = GraphWindow()
    win.add_run(_dip_result("a.mp4"))
    entry = next(iter(win._entries.values()))
    page = win._pages["vmaf"]

    values = entry.result.frames.vmaf
    idx = page._find_hover_index(entry, values, x=entry.times[5], y=70, half_window=0.05)
    assert idx == 5


# ------------------------------------------------------------------ view fitting

def test_adding_a_longer_run_expands_the_view_to_show_it(qapp):
    win = GraphWindow()
    win.resize(1000, 700)
    win.show()
    win.add_run(_values_result("a.mp4", [90.0] * 100))  # ~3.3s at 30fps
    qapp.processEvents()

    chart = win._pages["vmaf"].chart
    assert chart.x_range()[1] < 10

    win.add_run(_values_result("b.mp4", [90.0] * 100_000))  # ~3333s at 30fps
    qapp.processEvents()

    assert chart.x_range()[1] > 1000  # now spans the longer run, not just the first


def test_hovering_does_not_change_the_view(qapp):
    # Hovering must never re-fit or shift the axes -- only move the crosshair.
    win = GraphWindow()
    win.resize(1000, 700)
    win.show()
    win.add_run(_dip_result("a.mp4"))
    qapp.processEvents()

    chart = win._pages["vmaf"].chart
    before_x, before_y = chart.x_range(), chart.y_range()
    _hover_middle(win, "vmaf")
    assert chart.x_range() == before_x
    assert chart.y_range() == before_y
