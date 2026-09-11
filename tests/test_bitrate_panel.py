from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMessageBox

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
    assert panel.analyze_btn.text() == "Calculate bitrate"
    assert "Use checkbox" in panel.analyze_btn.toolTip()


def test_duplicate_paths_are_not_added_twice(qapp, tmp_path):
    panel = BitratePanel()
    path = tmp_path / "encode.mkv"

    panel.add_files([path, path, path.parent / "." / path.name])

    assert panel.table.rowCount() == 1


def test_stop_cancels_worker_and_queue_and_preserves_results(qapp, tmp_path):
    from unittest.mock import Mock

    panel = BitratePanel()
    paths = [tmp_path / f"{i}.mkv" for i in range(3)]
    panel.add_files(paths)
    keys = list(panel._entries)
    panel._on_analyzed(paths[0], _info(paths[0]), _data(paths[0]))
    saved = panel._entries[keys[0]].data
    panel._entries[keys[1]].status = "Reading video packets…"
    panel._entries[keys[2]].status = "Queued"
    panel._active_keys = set(keys[:2])
    panel._pending[keys[2]] = None
    worker = Mock()
    panel._worker = worker
    panel._update_buttons()
    assert panel.analyze_btn.text() == "Stop"
    assert panel.analyze_btn.isEnabled()
    panel.analyze_btn.click()
    worker.cancel.assert_called_once()
    assert not panel._pending
    assert panel.analyze_btn.text() == "Stopping…"
    assert not panel.analyze_btn.isEnabled()
    panel._on_progress(paths[1], 20, 100)
    assert "Stopping" in panel.status_label.text()
    panel._on_worker_finished(worker)
    assert panel._entries[keys[0]].data is saved
    assert [entry.status for entry in panel._entries.values()] == ["Complete", "Stopped", "Stopped"]
    assert "stopped" in panel.status_label.text()
    assert panel.analyze_btn.isEnabled()
    assert panel.analyze_btn.text() == "Calculate bitrate"


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


@pytest.mark.parametrize("answer", [QMessageBox.Yes, QMessageBox.No])
def test_recalculate_confirmation_only_applies_to_checked_results(qapp, tmp_path, monkeypatch, answer):
    panel = BitratePanel()
    paths = [tmp_path / f"{i}.mkv" for i in range(3)]
    panel.add_files(paths)
    for path in paths[:2]:
        panel._on_analyzed(path, _info(path), _data(path))
    panel.table.item(1, 0).setCheckState(Qt.Unchecked)
    messages = []
    def ask(*args):
        messages.append(args[2])
        assert args[-1] == QMessageBox.No
        return answer
    monkeypatch.setattr(QMessageBox, "question", ask)
    queued = []
    monkeypatch.setattr(panel, "_queue_keys", lambda keys: queued.extend(keys))
    panel.analyze_btn.click()
    keys = list(panel._entries)
    assert queued == ([keys[0], keys[2]] if answer == QMessageBox.Yes else [keys[2]])
    assert len(messages) == 1 and "1 checked video(s)" in messages[0]


def test_no_recalculation_leaves_completed_results_and_no_work(qapp, tmp_path, monkeypatch):
    panel = BitratePanel()
    path = tmp_path / "done.mkv"
    panel.add_files([path])
    panel._on_analyzed(path, _info(path), _data(path))
    saved = next(iter(panel._entries.values())).data
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    queued = []
    monkeypatch.setattr(panel, "_queue_keys", lambda keys: queued.append(keys))
    panel.analyze_btn.click()
    assert not queued
    assert next(iter(panel._entries.values())).data is saved


def test_new_files_calculate_without_confirmation(qapp, tmp_path, monkeypatch):
    panel = BitratePanel()
    panel.add_files([tmp_path / "new.mkv"])
    def unexpected(*args):
        pytest.fail("New files must not ask to recalculate")
    monkeypatch.setattr(QMessageBox, "question", unexpected)
    queued = []
    monkeypatch.setattr(panel, "_queue_keys", lambda keys: queued.extend(keys))
    panel.analyze_btn.click()
    assert queued == list(panel._entries)
