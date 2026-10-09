from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.frame_extract import FrameComparison
from vmaf_app.core.models import ComparisonResult, FrameScores, VideoInfo
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
    result = ComparisonResult(
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
    return FrameComparisonEntry(
        identity=object(), label=name,
        comparison=FrameComparison.from_result(result),
        scores=result.frames,
    )


def _unscored_entry(name: str, count: int = 120) -> FrameComparisonEntry:
    """A pair that has only been probed -- no run, no scores."""
    source = VideoInfo(
        path=Path("source.mkv"), width=1920, height=1080, fps=24.0,
        duration=5.0, nb_frames=count, codec_name="h264",
    )
    distorted = VideoInfo(
        path=Path(f"{name}.mkv"), width=1280, height=720, fps=24.0,
        duration=5.0, nb_frames=count, codec_name="h264",
    )
    return FrameComparisonEntry(
        identity=object(), label=name,
        comparison=FrameComparison(
            source_info=source, distorted_info=distorted,
            fps=24.0, frame_count=count,
        ),
    )


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12.5", 12.5), ("1:02.5", 62.5), ("2:03:04.25", 7384.25)],
)
def test_timestamp_parser_accepts_user_friendly_forms(text, seconds):
    assert parse_timestamp(text) == seconds


def test_timestamp_parser_rejects_ambiguous_values(subtests):
    def check(text):
        with pytest.raises(ValueError):
            parse_timestamp(text)

    for text in ["", "1:60", "1:60:00", "nope", "-1", "nan", "inf", "1.5:02"]:
        with subtests.test(text=text):
            check(text)


def test_runs_populate_selector_and_common_frame_range(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first", count=120), _entry("second", count=90)])

    assert panel.video_combo.count() == 2
    assert panel.video_combo.currentText() == "first"
    assert panel.frame_spin.maximum() == 89
    assert panel.timeline.maximum() == 89
    assert "TEST 1 of 2" in panel.showing_label.text()


def test_frame_and_timestamp_stay_synchronized(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("encode")])

    panel.set_frame(24)

    assert panel.frame_spin.value() == 24
    assert panel.timeline.value() == 24
    assert panel.timestamp_edit.text() == "0:00:01.000"
    assert "Frame 24" in panel.detail_label.full_text()
    assert "VMAF v0.6.1 90.00" in panel.detail_label.full_text()


def test_switching_distortions_preserves_frame_and_wraps(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first"), _entry("second")])
    panel.set_frame(48)

    panel.cycle_distorted(-1)

    assert panel.video_combo.currentText() == "second"
    assert panel.frame_spin.value() == 48
    assert "TEST 2 of 2" in panel.showing_label.text()


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
    assert panel.showing_label.text().startswith("TEST")
    panel.close()


# ----------------------------- comparing frames with no metrics calculated

def test_an_unscored_pair_is_shown_and_says_it_has_no_score(qapp):
    """The tab used to require a finished VMAF run before it would show
    anything, which made it unavailable exactly when it is most useful --
    before committing to a feature-length calculation."""
    panel = FrameComparePanel()
    panel.set_runs([_unscored_entry("encode")])

    assert panel.video_combo.count() == 1
    assert panel.frame_spin.isEnabled()
    assert panel.frame_spin.maximum() == 119
    assert "No metric results loaded" in panel.detail_label.full_text()
    assert "VMAF" not in panel.detail_label.full_text()
    panel.close()


def test_preview_crop_cache_reuses_source_across_test_switches(qapp, tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace

    from vmaf_app.core.models import CropBox

    first = _physical_entry(tmp_path, "first")
    second = _physical_entry(tmp_path, "second")
    entries = [
        dc_replace(item, comparison=dc_replace(item.comparison, auto_crop_pending=True))
        for item in (first, second)
    ]
    calls = []

    def fake_detect(info, **_kwargs):
        calls.append(info.path)
        return CropBox(1920, 816, 0, 132)

    monkeypatch.setattr("vmaf_app.ui.crop_detect_worker.detect_crop", fake_detect)
    panel = FrameComparePanel()
    panel.set_runs(entries)
    panel._ensure_auto_crop(panel.current_entry)
    for _ in range(100):
        qapp.processEvents()
        if not panel._crop_workers:
            break
        QTest.qWait(5)
    panel.cycle_distorted(1)
    panel._ensure_auto_crop(panel.current_entry)
    for _ in range(100):
        qapp.processEvents()
        if not panel._crop_workers:
            break
        QTest.qWait(5)

    assert calls.count(tmp_path / "source.mkv") == 1
    assert set(calls) == {
        tmp_path / "source.mkv", tmp_path / "first.mkv", tmp_path / "second.mkv"
    }
    panel.close()


def test_zoom_fits_by_default_and_takes_a_chosen_or_typed_percentage(qapp):
    """A user: "a comparison tool needs zoom". Fit to window as before, or
    a zoom chosen or typed in, which playback shares with the still frames."""
    panel = FrameComparePanel()
    assert panel.zoom.factor is None and panel.zoom_combo.currentText() == "Fit to window"
    index = panel.zoom_combo.findData(1.0)
    panel.zoom_combo.setCurrentIndex(index)
    panel.zoom_combo.activated.emit(index)
    assert panel.zoom.factor == 1.0 and panel.zoom_combo.currentText() == "100%"

    def typed(text):
        panel.zoom_combo.setEditText(text)
        panel.zoom_combo.lineEdit().editingFinished.emit()

    typed("150 %")
    assert panel.zoom.factor == 1.5 and panel.zoom_combo.currentText() == "150%"
    typed("not a zoom")
    assert panel.zoom.factor == 1.5 and panel.zoom_combo.currentText() == "150%"
    typed("Fit to window")
    assert panel.zoom.factor is None
    assert panel._ensure_video_view()._zoom is panel.zoom
    panel.close()


def test_a_zoomed_still_frame_is_dragged_and_a_fitted_one_is_not(qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage

    from vmaf_app.ui.frame_compare_panel import FrameView

    viewer = FrameView()
    viewer.resize(400, 200)
    viewer.show()
    image = QImage(800, 400, QImage.Format_RGB32)
    image.fill(0)
    viewer.set_image(image)
    ratio = viewer.devicePixelRatioF()

    def drag(start, end):
        QTest.mousePress(viewer, Qt.LeftButton, Qt.NoModifier, QPoint(*start))
        QTest.mouseMove(viewer, QPoint(*end))
        QTest.mouseRelease(viewer, Qt.LeftButton, Qt.NoModifier, QPoint(*end))

    drag((200, 100), (150, 100))
    assert viewer.zoom.centre == (0.5, 0.5)  # fitted: nothing to move
    viewer.zoom.factor = 1.0
    viewer.zoom_changed()
    assert viewer.cursor().shape() == Qt.OpenHandCursor
    drag((200, 100), (150, 100))
    assert viewer.zoom.centre[0] == pytest.approx(0.5 + 50 * ratio / 800)
    viewer.close()


# ------------------------------------------------------------ video playback

def _physical_entry(tmp_path, name="encode") -> FrameComparisonEntry:
    entry = _unscored_entry(name)
    source_path = tmp_path / "source.mkv"
    distorted_path = tmp_path / f"{name}.mkv"
    source_path.write_bytes(b"source")
    distorted_path.write_bytes(b"distorted")
    comparison = replace(
        entry.comparison,
        source_info=replace(entry.comparison.source_info, path=source_path),
        distorted_info=replace(entry.comparison.distorted_info, path=distorted_path),
    )
    return replace(entry, comparison=comparison)


def test_s_switches_between_the_same_frames_of_the_two_videos(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([_physical_entry(tmp_path)])
    panel.show()
    assert panel.video_view is not None
    view = panel.video_view
    source, test = bytes(16), bytes(range(16))
    view._source_surface.set_frame(source, (2, 2))
    view._distorted_surface.set_frame(test, (2, 2))

    view.show_source(True)

    assert view._showing_source is True
    assert view._source_surface._payload is source and view._distorted_surface._payload is test
    view.show_source(False)
    assert view._showing_source is False
    assert view._source_surface._payload is source and view._distorted_surface._payload is test
    panel.close()
