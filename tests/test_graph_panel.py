import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import FrameScore, VideoInfo, VmafRunResult
from vmaf_app.ui.graph_panel import GraphPanel


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


def _hover(win: GraphPanel, metric: str, time: float, value: float) -> None:
    win._pages[metric].on_hover(time, value, win._entries)


def _hover_middle(win: GraphPanel, metric: str) -> None:
    """Hover the middle of the chart's current view, at mid-height."""
    chart = win._pages[metric].chart
    x0, x1 = chart.x_range()
    y0, y1 = chart.y_range()
    _hover(win, metric, (x0 + x1) / 2, (y0 + y1) / 2)


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


def test_refreshing_a_hidden_series_does_not_make_it_visible_again(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", vmaf_value=80.0))
    sid = next(iter(win._entries))
    win.set_series_visible(sid, False)

    win.add_run(_fake_result("a.mp4", vmaf_value=99.0))

    assert len(win._entries) == 1
    assert win._entries[sid].visible is False
    assert win._pages["vmaf"]._curves[sid].visible is False


# ------------------------------------------------------------------ multi-series color + visibility

def test_each_series_gets_a_distinct_color(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))
    win.add_run(_fake_result("c.mp4"))

    colors = {e.color for e in win._entries.values()}
    assert len(colors) == 3  # all distinct


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

def test_graph_has_one_tab_per_metric(qapp):
    win = GraphPanel()
    labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert labels == ["VMAF", "PSNR", "SSIM", "XPSNR"]


def test_non_default_tabs_are_not_built_until_first_visited(qapp):
    # Each tab's plot caches a QPixmap of its drawn curves (~3.4MB at a
    # 1700x900 window, measured), so building all 4 up front meant paying
    # real memory for tabs the user may never look at. Only the default
    # (VMAF) tab should exist right away.
    win = GraphPanel()
    assert "vmaf" in win._pages
    assert "psnr" not in win._pages
    assert "ssim" not in win._pages
    assert "xpsnr" not in win._pages

    win.tabs.setCurrentIndex(1)  # PSNR
    assert "psnr" in win._pages
    assert "ssim" not in win._pages  # still untouched


def test_run_without_extra_metrics_only_gets_a_vmaf_curve(qapp):
    win = GraphPanel()
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
    assert "> 95" in headers_vmaf  # VMAF-specific threshold columns

    win.tabs.setCurrentIndex(1)  # PSNR
    assert win.stats_table.rowCount() == 1
    headers_psnr = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "> 95" not in headers_psnr  # thresholds are VMAF-only, don't make sense on a dB scale
    assert "Mean" in headers_psnr


def test_stats_table_blanks_series_with_no_data_for_the_active_metric(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", with_other_metrics=True), "a")
    win.add_run(_fake_result("b.mp4", with_other_metrics=False), "b")

    win.tabs.setCurrentIndex(0)  # VMAF -- both series have it
    assert win.stats_table.rowCount() == 2

    win.tabs.setCurrentIndex(1)  # PSNR -- only "a" has it
    # Both rows stay -- the table is the series list, so dropping "b" here
    # would make it un-removable and un-toggleable from this tab. "b" just
    # has no statistics to show.
    assert win.stats_table.rowCount() == 2
    assert _row_for(win, "a").text() == "a"
    a_row = win.stats_table.row(_row_for(win, "a"))
    b_row = win.stats_table.row(_row_for(win, "b"))
    assert win.stats_table.item(a_row, 1).text() != ""
    assert win.stats_table.item(b_row, 1).text() == ""


# ------------------------------------------------------------------ stats table columns (extensible)

def test_stats_table_includes_01_percent_low_column(qapp):
    win = GraphPanel()
    headers = [win.stats_table.horizontalHeaderItem(c).text() for c in range(win.stats_table.columnCount())]
    assert "1% Low" in headers
    assert "0.1% Low" in headers


# ------------------------------------------------------------------ Y-aware hover snapping

def test_hover_locks_onto_a_dip_below_cursor_y_even_if_not_exactly_under_cursor(qapp):
    win = GraphPanel()
    win.add_run(_dip_result("a.mp4"))
    entry = next(iter(win._entries.values()))
    page = win._pages["vmaf"]

    # Cursor sits exactly at the dip's time, at Y=70 -- well above the dip
    # (60) but below the surrounding baseline (95), so only the dip qualifies.
    values = entry.result.frames.vmaf
    idx = page._find_hover_index(entry, values, x=entry.times[5], y=70, half_window=0.05)
    assert idx == 5


def test_hover_falls_back_to_local_minimum_when_nothing_is_below_cursor_y(qapp):
    win = GraphPanel()
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
    win = GraphPanel()
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))
    win.add_run(_fake_result("b.mp4", vmaf_value=80.0))

    page = win._pages["vmaf"]
    _hover_middle(win, "vmaf")

    assert "Δ" in page.hover_label.text()  # the delta (Δ) line appears
    assert "10.00" in page.hover_label.text()  # 90 - 80 = 10


def test_hover_diff_not_shown_for_a_single_series(qapp):
    win = GraphPanel()
    win.show()
    win.add_run(_fake_result("a.mp4"))

    page = win._pages["vmaf"]
    _hover_middle(win, "vmaf")

    assert "Δ" not in page.hover_label.text()


def test_hover_on_psnr_tab_reports_psnr_not_vmaf(qapp):
    win = GraphPanel()
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0, with_other_metrics=True))
    win.tabs.setCurrentIndex(1)  # PSNR -- lazily builds its page

    page = win._pages["psnr"]
    _hover_middle(win, "psnr")

    assert "PSNR=45.00" in page.hover_label.text()
    assert "VMAF=" not in page.hover_label.text()


# ------------------------------------------------------------------ step-based hover radius

def test_series_step_matches_the_time_between_consecutive_frames(qapp):
    win = GraphPanel()
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

    win = GraphPanel()
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
    win = GraphPanel()
    win.show()
    win.add_run(_long_result("a.mp4", n_frames=7200, fps=30.0))  # runs to 0:04:00

    page = win._pages["vmaf"]
    # Hover near the middle of the plot (~2 minutes in).
    _hover_middle(win, "vmaf")

    text = page.hover_label.text()
    assert "Time: 0:0" in text  # H:M:S, not a bare "120.00s" style value
    assert "s" not in text.split("Time:")[1].split("\n")[0]  # no trailing "s" unit on the time itself


def test_graph_x_axis_renders_hms_ticks(qapp):
    win = GraphPanel()
    win.add_run(_long_result("a.mp4", n_frames=30 * 3725, fps=30.0))  # just over an hour
    labels = win._pages["vmaf"].chart.time_tick_labels()
    assert labels, "expected some time ticks"
    assert all(label.count(":") == 2 for label in labels), labels  # H:M:S, not bare seconds
    assert labels[0].startswith("0:")


# ------------------------------------------------------------------ Y-axis auto-scaling


def test_hiding_a_series_reaches_the_charts_y_range(qapp):
    # The y-range rules themselves are covered directly in test_chart.py;
    # what this checks is that the window's visibility toggle actually
    # reaches the chart that computes them.
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4", vmaf_value=90.0))
    win.add_run(_fake_result("b.mp4", vmaf_value=40.0))

    sid_b = next(sid for sid, e in win._entries.items() if e.label == "b")
    win.set_series_visible(sid_b, False)  # hide the low series

    y_range = win._pages["vmaf"].chart.y_range()
    assert y_range[0] == 90  # bottom reflects only the visible (90.0) series, not the hidden 40.0 one


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


def test_single_series_hover_still_uses_independent_dip_snap(qapp):
    # With only one series visible, the original per-series dip-snap
    # behavior (not the shared-time logic, which only matters for
    # comparing 2+ series) still applies.
    win = GraphPanel()
    win.add_run(_dip_result("a.mp4"))
    entry = next(iter(win._entries.values()))
    page = win._pages["vmaf"]

    values = entry.result.frames.vmaf
    idx = page._find_hover_index(entry, values, x=entry.times[5], y=70, half_window=0.05)
    assert idx == 5


# ------------------------------------------------------------------ view fitting

def test_adding_a_longer_run_expands_the_view_to_show_it(qapp):
    win = GraphPanel()
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
    win = GraphPanel()
    win.resize(1000, 700)
    win.show()
    win.add_run(_dip_result("a.mp4"))
    qapp.processEvents()

    chart = win._pages["vmaf"].chart
    before_x, before_y = chart.x_range(), chart.y_range()
    _hover_middle(win, "vmaf")
    assert chart.x_range() == before_x
    assert chart.y_range() == before_y


# ------------------------------------------------------------------ top panel sizing

def test_the_table_is_capped_at_four_rows(qapp):
    from vmaf_app.ui.graph_panel import _VISIBLE_SERIES_ROWS

    win = GraphPanel()
    win.resize(1400, 900)
    win.show()
    for i in range(7):  # more series than the panel shows at once
        win.add_run(_fake_result(f"{i}.mp4", vmaf_value=90.0 + i), f"s{i}")
    qapp.processEvents()

    row_height = win.stats_table.verticalHeader().defaultSectionSize()
    header = win.stats_table.horizontalHeader().sizeHint().height()
    # The table stops growing at four rows and scrolls past that, rather
    # than pushing the plot off the bottom of the window.
    assert win.stats_table.maximumHeight() <= header + _VISIBLE_SERIES_ROWS * row_height + 30
    assert win.stats_table.rowCount() == 7  # all series still listed, just scrolled


# ------------------------------------------------- the merged series/stats table

def test_the_stats_table_is_also_the_series_list(qapp):
    # There used to be a separate series list beside the stats table, which
    # repeated the same list of videos twice. One row per series now carries
    # its swatch, checkbox and name alongside its statistics.
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))

    assert win.stats_table.rowCount() == 2
    item = _row_for(win, "a")
    assert item.checkState() == Qt.Checked
    assert item.data(Qt.DecorationRole) is not None, "row should carry its colour swatch"
    assert not hasattr(win, "series_scroll"), "the separate series list should be gone"


def test_ticking_the_row_checkbox_toggles_the_curve(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    sid = next(iter(win._entries))

    _row_for(win, "a").setCheckState(Qt.Unchecked)
    assert win._pages["vmaf"]._curves[sid].visible is False

    _row_for(win, "a").setCheckState(Qt.Checked)
    assert win._pages["vmaf"]._curves[sid].visible is True


def test_a_series_with_no_data_for_this_metric_keeps_its_row(qapp):
    # Only VMAF is computed here, so on the PSNR tab this series has no
    # curve. Its row must still be present (blank stats) or there would be
    # no way to see or remove it from that tab.
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    win.tabs.setCurrentIndex(1)  # PSNR

    assert win.stats_table.rowCount() == 1
    assert _row_for(win, "a").text() == "a"
    assert win.stats_table.item(0, 1).text() == ""


def test_clicking_a_rows_remove_cell_removes_the_series(qapp):
    win = GraphPanel()
    win.add_run(_fake_result("a.mp4"))
    win.add_run(_fake_result("b.mp4"))

    last_col = win.stats_table.columnCount() - 1
    row = next(r for r in range(win.stats_table.rowCount())
               if win.stats_table.item(r, 0).text() == "a")
    assert win.stats_table.item(row, last_col).text() == "✕"
    win.stats_table.cellClicked.emit(row, last_col)

    assert win.stats_table.rowCount() == 1
    assert len(win._entries) == 1
    assert _row_for(win, "b").text() == "b"


# ------------------------------------------------------- hover readout sizing

def _hover_label_fits(page) -> tuple[bool, bool]:
    """(width fits, height fits) for whatever the readout currently shows."""
    from PySide6.QtGui import QFontMetrics
    fm = QFontMetrics(page.hover_label.font())
    lines = page.hover_label.text().split("\n")
    widest = max(fm.horizontalAdvance(line) for line in lines)
    return widest <= page.hover_label.width(), \
        len(lines) * fm.lineSpacing() <= page.hover_label.height()


def test_hover_readout_is_not_clipped_by_long_series_names(qapp):
    # The readout used to be pinned to 560x90px, which cut real content: long
    # encode names ran off the right mid-number, and the delta line -- the
    # longest of the lot -- lost its value entirely.
    long_a = "Top Gun Maverick 1080p QP 24 fast 1 sub"
    long_b = "Top Gun Maverick 1080p QP 24 faster 1 sub"
    win = GraphPanel()
    win.resize(1700, 950)
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=91.0), long_a)
    win.add_run(_fake_result("b.mp4", vmaf_value=93.0), long_b)
    qapp.processEvents()

    _hover_middle(win, "vmaf")
    page = win._pages["vmaf"]

    text = page.hover_label.text()
    assert "Δ" in text and text.strip().endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
    width_ok, height_ok = _hover_label_fits(page)
    assert width_ok, f"readout clipped horizontally: {page.hover_label.width()}px"
    assert height_ok, f"readout clipped vertically: {page.hover_label.height()}px"


def test_hover_readout_fits_every_series_when_many_are_shown(qapp):
    win = GraphPanel()
    win.resize(1700, 950)
    win.show()
    for i in range(6):
        win.add_run(_fake_result(f"{i}.mp4", vmaf_value=90.0 + i), f"encode-number-{i}-with-a-long-name")
    qapp.processEvents()

    _hover_middle(win, "vmaf")
    page = win._pages["vmaf"]

    assert len(page.hover_label.text().split("\n")) == 7  # time + 6 series
    width_ok, height_ok = _hover_label_fits(page)
    assert width_ok and height_ok


def test_hover_readout_size_does_not_change_while_hovering(qapp):
    # The size is derived from the series set, never from the text under the
    # cursor: letting it change per mouse move re-laid out the whole tab
    # (chart and table included), which measured as ~76% of a hover's cost.
    win = GraphPanel()
    win.resize(1700, 950)
    win.show()
    win.add_run(_fake_result("a.mp4", vmaf_value=91.0), "an-encode-with-a-long-name")
    qapp.processEvents()

    page = win._pages["vmaf"]
    chart = page.chart
    x0, x1 = chart.x_range()
    y0, y1 = chart.y_range()

    _hover(win, "vmaf", x0 + (x1 - x0) * 0.1, (y0 + y1) / 2)
    before = page.hover_label.size()
    for frac in (0.3, 0.5, 0.7, 0.9):
        _hover(win, "vmaf", x0 + (x1 - x0) * frac, (y0 + y1) / 2)
        assert page.hover_label.size() == before, f"readout resized at {frac}"


def test_the_hover_readout_is_measured_in_the_font_it_renders_in(qapp):
    # The font was set via stylesheet, which never reaches widget.font(), so
    # the width was measured in the proportional default while rendering in
    # wider monospace -- clipping the last characters of every line.
    from PySide6.QtGui import QFontMetrics

    win = GraphPanel()
    page = win._pages["vmaf"]
    fm = QFontMetrics(page.hover_label.font())
    widest = max(fm.horizontalAdvance(line) for line in page.hover_label.text().splitlines())
    assert widest <= page.hover_label.maximumWidth(), "the placeholder is clipped"


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
    assert "VMAF=91.00" in text and "VMAF=81.00" in text
    assert "VMAF=20.00" not in text, "must not snap to the nearby dip"


def test_going_to_a_frame_shows_the_difference_for_exactly_two_series(qapp):
    win = GraphPanel()
    win.add_run(_values_result("a.mp4", [90.0, 90.0]), "a")
    win.add_run(_values_result("b.mp4", [80.0, 80.0]), "b")

    win._pages["vmaf"].show_frame(0, win._entries)
    assert "Δ" in win._pages["vmaf"].hover_label.text()


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
    assert "VMAF=90.00" in text


def test_the_frame_control_is_bounded_by_what_is_plotted(qapp):
    win = GraphPanel()
    assert win.frame_spin.maximum() == 0, "nothing plotted yet"

    win.add_run(_values_result("a.mp4", [90.0] * 25), "a")
    assert win.frame_spin.maximum() == 24
    assert win.frame_spin.minimum() == 0

    sid = next(iter(win._entries))
    win.remove_run(sid)
    assert win.frame_spin.maximum() == 0, "range follows removal"


def test_going_to_a_frame_with_nothing_visible_says_so(qapp):
    win = GraphPanel()
    win.add_run(_values_result("a.mp4", [90.0, 91.0]), "a")
    sid = next(iter(win._entries))
    win.set_series_visible(sid, False)

    page = win._pages["vmaf"]
    assert page.show_frame(0, win._entries) is False
    assert "no visible series" in page.hover_label.text()


# ------------------------------------------------------------------ removal

def test_removing_a_series_by_its_distorted_path(qapp):
    win = GraphPanel()
    win.add_run(_values_result("a.mp4", [90.0]), "a")
    win.add_run(_values_result("b.mp4", [80.0]), "b")

    assert win.remove_by_path(Path("a.mp4")) is True
    assert len(win._entries) == 1
    assert Path(next(iter(win._entries.values())).result.distorted) == Path("b.mp4")

    assert win.remove_by_path(Path("never-added.mp4")) is False


# ------------------------------------------------- missing metric values

def _page(panel: GraphPanel, key: str):
    """The page for a metric, selecting its tab first -- pages other than
    VMAF are only built when their tab is first visited."""
    from vmaf_app.ui.graph_panel import METRICS
    panel.tabs.setCurrentIndex([m.key for m in METRICS].index(key))
    return panel._pages[key]


def _nan_metric_result(name: str, metric: str, n: int = 10) -> VmafRunResult:
    """A run that HAS the metric column but no finite value in it.

    This is not hypothetical: libvmaf writes the key with a null/NaN when a
    feature fails on a frame, and an n_subsample run scores only every Nth
    frame. A whole column of them is the degenerate end of the same case.
    """
    info = VideoInfo(
        path=Path(name), width=1920, height=1080, fps=30.0, duration=n / 30.0,
        nb_frames=n, codec_name="h264",
    )
    frames = [
        FrameScore(
            frame=i, time=i / 30.0, vmaf=90.0,
            psnr=float("nan") if metric == "psnr" else None,
            ssim=float("nan") if metric == "ssim" else None,
            xpsnr=float("nan") if metric == "xpsnr" else None,
        )
        for i in range(n)
    ]
    return VmafRunResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
        source_info=info, distorted_info=info,
    )


@pytest.mark.parametrize("metric", ["psnr", "ssim", "xpsnr"])
def test_hovering_an_all_nan_metric_does_not_crash(qapp, metric):
    # np.nanargmin raises ValueError on an all-NaN slice, so this used to
    # take the whole hover handler down on every mouse move over the page.
    panel = GraphPanel()
    panel.add_run(_nan_metric_result("a.mkv", metric), "a")
    page = _page(panel, metric)

    page.on_hover(0.1, 50.0, panel._entries)

    assert f"no {page.metric.label}" in page.hover_label.text()


@pytest.mark.parametrize("metric", ["psnr", "ssim", "xpsnr"])
def test_hovering_two_all_nan_series_reports_no_delta(qapp, metric):
    # The multi-series path pools every series' window before picking one
    # shared moment, so it had its own copy of the nanargmin call.
    panel = GraphPanel()
    panel.add_run(_nan_metric_result("a.mkv", metric), "a")
    panel.add_run(_nan_metric_result("b.mkv", metric), "b")
    page = _page(panel, metric)

    page.on_hover(0.1, 50.0, panel._entries)
    text = page.hover_label.text()

    assert "Δ" not in text, "a difference was reported between two missing values"
    assert text.count(f"no {page.metric.label}") == 2


def test_a_series_with_some_missing_frames_still_reports_the_finite_ones(qapp):
    panel = GraphPanel()
    result = _nan_metric_result("a.mkv", "ssim")
    result.frames.ssim[5] = 0.9876  # one real value in a column of NaN
    panel.add_run(result, "a")
    page = _page(panel, "ssim")

    page.show_frame(5, panel._entries)
    assert "0.9876" in page.hover_label.text()

    page.show_frame(4, panel._entries)
    assert "no SSIM" in page.hover_label.text()


def test_a_delta_is_only_taken_between_two_finite_values(qapp):
    panel = GraphPanel()
    good = _nan_metric_result("a.mkv", "ssim")
    good.frames.ssim[:] = 0.99
    missing = _nan_metric_result("b.mkv", "ssim")
    panel.add_run(good, "a")
    panel.add_run(missing, "b")
    page = _page(panel, "ssim")

    page.show_frame(3, panel._entries)
    text = page.hover_label.text()

    assert "0.9900" in text
    assert "no SSIM" in text
    assert "Δ" not in text
