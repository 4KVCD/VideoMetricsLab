from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import FrameScores, VideoInfo, VmafRunResult
from vmaf_app.ui.frame_compare_panel import (
    FrameComparePanel,
    FrameComparisonEntry,
    parse_timestamp,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _entry(name: str, score: float = 90.0, count: int = 120) -> FrameComparisonEntry:
    source = VideoInfo(
        path=Path("source.mkv"), width=1920, height=1080, fps=24.0,
        duration=5.0, nb_frames=count, codec_name="h264",
    )
    distorted = VideoInfo(
        path=Path(f"{name}.mkv"), width=1920, height=1080, fps=24.0,
        duration=5.0, nb_frames=count, codec_name="h264",
    )
    result = VmafRunResult(
        source=source.path,
        distorted=distorted.path,
        frames=FrameScores(
            frame=np.arange(count, dtype=np.int32),
            time=np.arange(count, dtype=np.float64) / 24,
            vmaf=np.full(count, score, dtype=np.float32),
        ),
        fps=24.0, model="m", source_crop=None, distorted_crop=None,
        source_info=source, distorted_info=distorted,
        compared_frame_count=count,
    )
    return FrameComparisonEntry(identity=object(), label=name, result=result)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12.5", 12.5), ("1:02.5", 62.5), ("2:03:04.25", 7384.25)],
)
def test_timestamp_parser_accepts_user_friendly_forms(text, seconds):
    assert parse_timestamp(text) == seconds


@pytest.mark.parametrize(
    "text", ["", "1:60", "1:60:00", "nope", "-1", "nan", "inf", "1.5:02"]
)
def test_timestamp_parser_rejects_ambiguous_values(text):
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_runs_populate_selector_and_common_frame_range(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first", count=120), _entry("second", count=90)])

    assert panel.video_combo.count() == 2
    assert panel.video_combo.currentText() == "first"
    assert panel.frame_spin.maximum() == 89
    assert panel.timeline.maximum() == 89
    assert "DISTORTED 1 of 2" in panel.showing_label.text()


def test_frame_and_timestamp_stay_synchronized(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("encode")])

    panel.set_frame(24)

    assert panel.frame_spin.value() == 24
    assert panel.timeline.value() == 24
    assert panel.timestamp_edit.text() == "0:00:01.000"
    assert "Frame 24" in panel.detail_label.text()
    assert "VMAF 90.00" in panel.detail_label.text()


def test_switching_distortions_preserves_frame_and_wraps(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first"), _entry("second")])
    panel.set_frame(48)

    panel.cycle_distorted(-1)

    assert panel.video_combo.currentText() == "second"
    assert panel.frame_spin.value() == 48
    assert "DISTORTED 2 of 2" in panel.showing_label.text()


def test_left_and_right_switch_when_the_viewer_has_focus(qapp, monkeypatch):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first"), _entry("second")])
    monkeypatch.setattr(panel, "_show_or_request", lambda: None)
    panel.show()
    panel.viewer.setFocus()

    QTest.keyClick(panel.viewer, Qt.Key_Right)
    assert panel.video_combo.currentText() == "second"
    QTest.keyClick(panel.viewer, Qt.Key_Left)
    assert panel.video_combo.currentText() == "first"
    panel.close()


def test_holding_s_temporarily_shows_source(qapp, monkeypatch):
    panel = FrameComparePanel()
    panel.set_runs([_entry("encode")])
    monkeypatch.setattr(panel, "_show_or_request", lambda: None)
    panel.show()
    panel.viewer.setFocus()

    QTest.keyPress(panel.viewer, Qt.Key_S)
    assert panel._showing_source is True
    assert panel.showing_label.text().startswith("SOURCE")

    QTest.keyRelease(panel.viewer, Qt.Key_S)
    assert panel._showing_source is False
    assert panel.showing_label.text().startswith("DISTORTED")
    panel.close()


def test_missing_subsampled_frame_is_not_given_a_neighbouring_score(qapp):
    entry = _entry("subsampled")
    entry.result.frames = entry.result.frames[::2]
    panel = FrameComparePanel()
    panel.set_runs([entry])

    panel.set_frame(3)

    assert "not scored for this frame" in panel.detail_label.text()
