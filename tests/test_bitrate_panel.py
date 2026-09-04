from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication

from vmaf_app.core.bitrate import BitrateData
from vmaf_app.core.models import VideoInfo
from vmaf_app.ui.bitrate_panel import BitratePanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _info(path: Path) -> VideoInfo:
    return VideoInfo(
        path=path, width=1920, height=1080, fps=2.0,
        duration=2.0, nb_frames=4, codec_name="h264", pix_fmt="yuv420p",
    )


def _data(path: Path) -> BitrateData:
    return BitrateData(
        path=path,
        times=np.array([0.0, 0.5, 1.0, 1.5]),
        durations=np.full(4, 0.5),
        sizes=np.array([1000, 500, 2000, 500]),
        keyframes=np.array([True, False, True, False]),
    )


def test_files_can_be_added_without_a_vmaf_run(qapp, tmp_path):
    panel = BitratePanel()
    path = tmp_path / "encode.mkv"

    panel.add_files([path])

    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 1).text() == "encode.mkv"
    assert panel.table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert next(iter(panel._entries.values())).data is None
    assert panel.analyze_btn.isEnabled()


def test_duplicate_paths_are_not_added_twice(qapp, tmp_path):
    panel = BitratePanel()
    path = tmp_path / "encode.mkv"

    panel.add_files([path, path, path.parent / "." / path.name])

    assert panel.table.rowCount() == 1


def test_completed_analysis_populates_stats_and_all_three_plot_views(qapp, tmp_path):
    panel = BitratePanel()
    path = tmp_path / "encode.mkv"
    panel.add_files([path])
    panel._on_analyzed(path, _info(path), _data(path))

    assert panel.table.item(0, 4).text() == "4"
    assert panel.table.item(0, 5).text()
    assert panel.chart.has_data()
    assert panel.chart.y_axis_label == "Video bitrate (kb/s)"

    panel.frame_radio.click()
    assert panel.chart.y_axis_label == "Frame size (kbit)"
    panel.gop_radio.click()
    assert panel.chart.y_axis_label == "GOP bitrate (kb/s)"
    assert len(panel._current_plots) == 1


def test_use_checkbox_hides_and_restores_a_curve(qapp, tmp_path):
    panel = BitratePanel()
    path = tmp_path / "encode.mkv"
    panel.add_files([path])
    panel._on_analyzed(path, _info(path), _data(path))

    panel.table.item(0, 0).setCheckState(Qt.Unchecked)
    assert not panel.chart.has_data()
    panel.table.item(0, 0).setCheckState(Qt.Checked)
    assert panel.chart.has_data()


def test_metric_workflow_api_deduplicates_repeated_sources(qapp, tmp_path, monkeypatch):
    panel = BitratePanel()
    source = _info(tmp_path / "source.mkv")
    distorted = _info(tmp_path / "distorted.mkv")
    queued = []
    monkeypatch.setattr(panel, "_queue_keys", lambda keys: queued.extend(keys))

    panel.add_and_analyze([source, distorted, source])

    assert panel.table.rowCount() == 2
    assert len(set(queued)) == 2
