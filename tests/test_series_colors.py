"""Shared presentation colors, independent of metric results/cache state."""
import os
import subprocess
from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog, QDialogButtonBox, QStyle, QStyleOptionViewItem

from tests.factories import fake_run_result
from vmaf_app.core.bitrate import analyze_video_bitrate
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropMode, VmafOptions, synthetic_scale_direction_variant_path
from vmaf_app.core.vmaf_runner import run_vmaf
from vmaf_app.ui.bitrate_panel import BitratePanel
from vmaf_app.ui.graph_panel import GraphPanel
from vmaf_app.ui.main_window import MainWindow
from vmaf_app.ui.series_colors import PALETTE, VideoColors


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_full_paths_distinguish_equal_names_and_normalize_aliases(tmp_path, monkeypatch):
    colors = VideoColors()
    first = tmp_path / "a" / "encode.mkv"
    second = tmp_path / "b" / "encode.mkv"
    first.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    assert colors.for_path(first) == PALETTE[0]
    assert colors.for_path(second) == PALETTE[1]
    assert colors.for_path(Path("a") / "." / "encode.mkv") == PALETTE[0]
    if os.name == "nt":
        assert colors.for_path(Path(str(first).upper())) == PALETTE[0]
    else:
        assert colors.for_path(first.with_name("ENCODE.mkv")) != PALETTE[0]


def test_palette_cycles_without_overwriting_old_reservations(tmp_path):
    colors = VideoColors()
    paths = [tmp_path / f"encode-{i}.mkv" for i in range(len(PALETTE) + 2)]
    assigned = [colors.for_path(path) for path in paths]
    assert assigned == [PALETTE[i % len(PALETTE)] for i in range(len(paths))]
    assert [colors.for_path(path) for path in reversed(paths)] == list(reversed(assigned))
    assert VideoColors().for_path(paths[-1]) == PALETTE[0]  # New session, no persisted mapping.


@pytest.mark.parametrize("bitrate_first", [False, True])
def test_panels_share_colors_in_different_orders_and_restore_removed_videos(qapp, tmp_path, bitrate_first):
    colors = VideoColors()
    graph, bitrate = GraphPanel(colors=colors), BitratePanel(colors=colors)
    paths = [tmp_path / f"encode-{i}.mkv" for i in range(3)]
    try:
        if bitrate_first:
            bitrate.add_files(list(reversed(paths)))
        for path in paths:
            graph.add_run(fake_run_result(path))
        if not bitrate_first:
            bitrate.add_files(list(reversed(paths)))
        expected = {entry.result.distorted: entry.color for entry in graph._entries.values()}
        assert {entry.path: entry.color for entry in bitrate._entries.values()} == expected
        sid = next(sid for sid, entry in graph._entries.items() if entry.result.distorted == paths[1])
        graph.remove_run(sid)
        row = next(i for i, entry in enumerate(bitrate._entries.values()) if entry.path == paths[1])
        bitrate.table.selectRow(row)
        bitrate._on_remove_selected()
        extra = tmp_path / "new-encode.mkv"
        graph.add_run(fake_run_result(extra))
        bitrate.add_files([extra, paths[1]])
        graph.add_run(fake_run_result(paths[1]))
        actual = {entry.result.distorted: entry.color for entry in graph._entries.values()}
        assert actual[paths[1]] == expected[paths[1]]
        assert {entry.path: entry.color for entry in bitrate._entries.values()} == actual
        page = graph._pages["vmaf"]
        for key, entry in graph._entries.items():
            assert page.chart._series[key].color == entry.color
        for row, entry in enumerate(bitrate._entries.values()):
            assert bitrate.table.item(row, 0).data(Qt.DecorationRole).toImage().pixelColor(0, 0).name() == entry.color.lower()
    finally:
        graph.close()
        bitrate.close()


def test_multiple_runs_and_synthetic_variants_use_physical_video_color(qapp, tmp_path):
    colors = VideoColors()
    graph = GraphPanel(colors=colors)
    path = tmp_path / "encode.mkv"
    first, variant = fake_run_result(path), fake_run_result(path)
    variant.distorted = synthetic_scale_direction_variant_path(path, variant.scale_direction)
    try:
        graph.add_run(first, identity=object())
        graph.add_run(variant, identity=object())
        assert len(graph._entries) == 2
        assert {entry.color for entry in graph._entries.values()} == {colors.for_path(path)}
        # Updating a stable row to another video must update its color too.
        entry = next(iter(graph._entries.values()))
        other = tmp_path / "different.mkv"
        graph.add_run(fake_run_result(other), identity=entry.identity)
        assert entry.color == colors.for_path(other)
    finally:
        graph.close()


def test_main_window_injects_one_session_registry_into_both_tabs(qapp):
    window = MainWindow()
    try:
        assert window.graph_panel._video_colors is window.bitrate_panel._video_colors
    finally:
        window.close()


def test_picker_recolors_both_tabs_and_all_runs_without_recomputing(qapp, tmp_path, monkeypatch):
    colors = VideoColors()
    graph, bitrate = GraphPanel(colors=colors), BitratePanel(colors=colors)
    path = tmp_path / "test.mkv"
    try:
        graph.add_run(fake_run_result(path), identity="first")
        graph.add_run(fake_run_result(path), identity="second")
        bitrate.add_files([path])
        graph.show()
        bitrate.show()
        qapp.processEvents()
        page = graph._pages["vmaf"]
        originals = {sid: curve for sid, curve in page._curves.items()}
        page.chart._view_x = (0.1, 0.2)
        bitrate.chart._view_x = (0.1, 0.2)
        # Recoloring must not take the score/statistics or bitrate plotting path.
        monkeypatch.setattr(page, "set_curve", lambda *a: pytest.fail("recalculated statistics"))
        monkeypatch.setattr(bitrate, "_refresh_plot", lambda: pytest.fail("rebuilt bitrate plot"))
        colors.set_color(path, QColor("#123456"))
        assert all(e.color == "#123456" for e in graph._entries.values())
        assert next(iter(bitrate._entries.values())).color == "#123456"
        assert all(s.color == "#123456" for s in page.chart._series.values())
        assert originals == page._curves
        assert page.chart._view_x == bitrate.chart._view_x == (0.1, 0.2)
        assert all(e.visible for e in graph._entries.values())
        assert next(iter(bitrate._entries.values())).enabled
        assert not graph.render_export_image().isNull()
        assert VideoColors().for_path(path) == PALETTE[0]
    finally:
        graph.close()
        bitrate.close()


@pytest.mark.parametrize("tab", ["graph", "bitrate"])
def test_click_square_opens_picker_not_checkbox_or_name(qapp, tmp_path, monkeypatch, tab):
    colors = VideoColors()
    graph, bitrate = GraphPanel(colors=colors), BitratePanel(colors=colors)
    path = tmp_path / "test.mkv"
    opened = []
    monkeypatch.setattr(colors, "choose", lambda path, parent: opened.append(path))
    try:
        graph.add_run(fake_run_result(path))
        bitrate.add_files([path])
        panel, table = (graph, graph.stats_table) if tab == "graph" else (bitrate, bitrate.table)
        panel.resize(1100, 650)
        panel.show()
        qapp.processEvents()
        index = table.model().index(0, 0)
        option = QStyleOptionViewItem()
        table.initViewItemOption(option)
        option.rect = table.visualRect(index)
        option.widget = table
        rect = table.itemDelegateForColumn(0).swatch_rect(option, index)
        assert not rect.isEmpty()
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=rect.center())
        assert opened == [path]
        assert table.item(0, 0).checkState() == Qt.Checked
        # Outside the swatch still performs the usual checkbox action.
        table.itemDelegateForColumn(0).initStyleOption(option, index)
        checkbox = table.style().subElementRect(QStyle.SE_ItemViewItemCheckIndicator, option, table)
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=checkbox.center())
        assert opened == [path]
        assert table.item(0, 0).checkState() == Qt.Unchecked
        # Clicking the name is not a shortcut to the picker either.
        name_index = table.model().index(0, 0 if tab == "graph" else 1)
        name_option = QStyleOptionViewItem()
        table.initViewItemOption(name_option)
        name_option.rect = table.visualRect(name_index)
        name_option.widget = table
        table.itemDelegateForIndex(name_index).initStyleOption(name_option, name_index)
        name_rect = table.style().subElementRect(QStyle.SE_ItemViewItemText, name_option, table)
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=name_rect.center())
        assert opened == [path]
    finally:
        graph.close()
        bitrate.close()


def test_real_dialog_accept_cancel_reset_and_retained_color(qapp, tmp_path):
    colors = VideoColors()
    path = tmp_path / "test.mkv"
    changes = []
    colors.changed.connect(lambda *args: changes.append(args))
    original = colors.for_path(path)

    def finish(accept, reset=False):
        dialog = qapp.activeModalWidget()
        assert isinstance(dialog, QColorDialog)
        dialog.setCurrentColor(QColor("#abcdef"))
        if reset:
            box = dialog.findChild(QDialogButtonBox)
            next(b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.ResetRole).click()
            assert dialog.currentColor().name().lower() == original.lower()
        dialog.accept() if accept else dialog.reject()

    QTimer.singleShot(0, lambda: finish(False))
    colors.choose(path, None)
    assert colors.for_path(path) == original and not changes
    QTimer.singleShot(0, lambda: finish(True))
    colors.choose(path, None)
    assert colors.for_path(path) == "#abcdef" and len(changes) == 1
    QTimer.singleShot(0, lambda: finish(False, reset=True))
    colors.choose(path, None)
    assert colors.for_path(path) == "#abcdef"
    QTimer.singleShot(0, lambda: finish(True, reset=True))
    colors.choose(path, None)
    assert colors.for_path(path).lower() == original.lower()
    colors.set_color(path, QColor())
    assert len(changes) == 2


def test_real_encoded_files_metrics_and_bitrate_curves_keep_colors(qapp, tmp_path):
    """Real FFmpeg/ffprobe workflow, with no user media or cache writes."""
    ffmpeg = str(ffmpeg_path())
    source = tmp_path / "reference.mkv"
    paths = [tmp_path / "H264 QP20.mkv", tmp_path / "H264 QP35.mkv"]
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=12", "-t", "0.5", "-c:v", "ffv1", str(source)],
                   check=True, capture_output=True)
    for path, qp in zip(paths, (20, 35), strict=True):
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-c:v", "libx264",
                        "-preset", "ultrafast", "-qp", str(qp), str(path)], check=True, capture_output=True)
    reference = probe_video(source)
    infos = {path: probe_video(path) for path in paths}
    options = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"], crop_mode=CropMode.NONE,
                          gpu_decode=False, n_threads=2)
    results = {path: run_vmaf(reference, infos[path], options) for path in paths}
    scores = {path: result.frames.values("psnr").copy() for path, result in results.items()}
    colors = VideoColors()
    graph, bitrate = GraphPanel(colors=colors), BitratePanel(colors=colors)
    try:
        for path in paths:
            graph.add_run(results[path])
        bitrate.add_files(list(reversed(paths)), infos)
        for path in paths:
            bitrate._on_analyzed(path, infos[path], analyze_video_bitrate(infos[path]))
        graph.show()
        bitrate.show()
        qapp.processEvents()
        for button in (bitrate.frame_radio, bitrate.second_radio, bitrate.gop_radio):
            button.click()
            qapp.processEvents()
            assert len(bitrate.chart._series) == 2
            for key, plotted in bitrate.chart._series.items():
                entry, _plot = bitrate._current_plots[key]
                assert plotted.color == colors.for_path(entry.path)
        expected = {entry.result.distorted: entry.color for entry in graph._entries.values()}
        assert {entry.path: entry.color for entry in bitrate._entries.values()} == expected
        assert len(bitrate.chart._series) == 2
        # Real plotted packet data must survive recoloring in either tab.
        old_plots = [entry.plots.copy() for entry in bitrate._entries.values()]
        colors.set_color(paths[0], QColor("#876543"))
        for key, plotted in bitrate.chart._series.items():
            entry, _plot = bitrate._current_plots[key]
            assert plotted.color == colors.for_path(entry.path)
        assert [entry.plots for entry in bitrate._entries.values()] == old_plots
        expected[paths[0]] = "#876543"
        for sid in list(graph._entries):
            graph.remove_run(sid)
        for path in reversed(paths):
            graph.add_run(results[path])
        assert {entry.result.distorted: entry.color for entry in graph._entries.values()} == expected
        for path, result in results.items():
            assert (result.frames.values("psnr") == scores[path]).all()
        assert not graph.render_export_image().isNull()
        assert not bitrate.chart.grab().isNull()
    finally:
        graph.close()
        bitrate.close()
