import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import ComparisonResult, FrameScores, VideoInfo
from vmaf_app.core.settings import Settings
from vmaf_app.core.stats import SQUARE_MEAN_ROOT, compute_stats
from vmaf_app.ui.graph_panel import GraphPanel
from vmaf_app.ui.main_window import COL_XPSNR, CompletedRun, MainWindow


@pytest.mark.parametrize("data", [None, [], "text", 12])
def test_non_object_settings_recovers(data):
    Settings.path().write_text(json.dumps(data))
    assert isinstance(Settings.load(), Settings)


def test_mixed_infinite_percentiles_are_not_missing():
    stats = compute_stats([40] + [float("inf")] * 99, [], SQUARE_MEAN_ROOT)
    assert stats.mean == pytest.approx(80)
    assert all(np.isposinf(v) for v in [stats.percentile_10, stats.percentile_5,
                                      stats.percentile_1, stats.percentile_0_1])
    finite = compute_stats([10, 20, 30, 40], [])
    assert finite.percentile_10 == pytest.approx(13)


def test_identical_frame_tooltips_explain_sequence_average(tmp_path):
    app = QApplication.instance() or QApplication([])
    info = VideoInfo(path=tmp_path / "clip.mkv", width=192, height=108,
                     fps=24, duration=1, nb_frames=2, codec_name="ffv1")
    result = ComparisonResult(source=info.path, distorted=info.path,
                          frames=FrameScores(np.arange(2), np.arange(2)/24, None,
                                             xpsnr=[40, float("inf")]),
                          fps=24, model="", source_crop=None, distorted_crop=None,
                          source_info=info, distorted_info=info)
    note = MainWindow._identical_frames_note(CompletedRun(result, "test"), COL_XPSNR)
    assert "excluded" not in note
    assert "zero distortion" in note
    panel = GraphPanel()
    try:
        panel.add_run(result, "test")
        panel.tabs.setCurrentIndex(4)
        app.processEvents()
        tooltip = panel.stats_table.item(0, 5).toolTip()
        assert "excluded" not in tooltip
        assert "zero distortion" in tooltip
    finally:
        panel.close()
