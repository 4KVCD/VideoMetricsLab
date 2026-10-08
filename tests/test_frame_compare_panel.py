from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.display_hdr import DisplayHdrInfo
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorMode
from vmaf_app.core.models import ComparisonResult, FrameScores, ResampleTarget, VideoInfo
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


def test_playback_status_hides_diagnostics_but_keeps_decode_modes(qapp):
    panel = FrameComparePanel()
    raw = ("Paused · GStreamer D3D11 · 4/4 streams · Buffering locked pair · "
           "source soundtrack · 3840x1608 RGBA64_LE sRGB · memory:D3D11Memory · "
           "source GPU: d3d11h265dec · distorted software: avdec_h266")
    panel._on_video_status_changed(raw)
    assert panel._concise_playback_status() == "Paused · Source: GPU decode · Test: CPU decode"
    assert "RGBA64_LE" in panel.advanced_info_btn.toolTip()
    assert "distorted" not in panel.advanced_info_btn.toolTip()
    panel._on_video_status_changed("Could not decode test video: missing codec")
    assert "missing codec" in panel._concise_playback_status()


def test_a_reference_without_a_soundtrack_is_not_shown_as_an_error(qapp):
    """It put the whole diagnostic line -- wider than the window, cut off at
    both ends -- under every comparison whose reference has no audio."""
    panel = FrameComparePanel()
    panel._on_video_status_changed(
        "Paused · GStreamer D3D11 · 3/4 streams · Frame-locked GPU pair · audio unavailable · "
        "3840x2160 P010_10LE · memory:D3D11Memory · source software: avdec_ffv1 · distorted GPU: d3d11h264dec")
    assert panel._concise_playback_status() == "Paused · Source: CPU decode · Test: GPU decode"
    assert "audio unavailable" in panel.advanced_info_btn.toolTip()


def test_advanced_info_opens_on_tooltip_event_and_click(qapp, monkeypatch):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip
    calls = []
    monkeypatch.setattr(QToolTip, "showText", lambda *args: calls.append(args))
    panel = FrameComparePanel()
    panel._on_video_status_changed("Paused · source GPU: d3d11 · memory:D3D11Memory")
    button = panel.advanced_info_btn
    qapp.sendEvent(button, QHelpEvent(QEvent.ToolTip, QPoint(1, 1), QPoint(10, 10)))
    assert len(calls) == 1
    assert "<br>" in calls[0][1]
    assert "memory:D3D11Memory" in calls[0][1]
    button.click()
    assert len(calls) == 2


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
    assert "TEST 1 of 2" in panel.showing_label.text()


def test_frame_and_timestamp_stay_synchronized(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("encode")])

    panel.set_frame(24)

    assert panel.frame_spin.value() == 24
    assert panel.timeline.value() == 24
    assert panel.timestamp_edit.text() == "0:00:01.000"
    assert "Frame 24" in panel.detail_label.text()
    assert "VMAF v0.6.1 90.00" in panel.detail_label.text()


def test_switching_distortions_preserves_frame_and_wraps(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("first"), _entry("second")])
    panel.set_frame(48)

    panel.cycle_distorted(-1)

    assert panel.video_combo.currentText() == "second"
    assert panel.frame_spin.value() == 48
    assert "TEST 2 of 2" in panel.showing_label.text()


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
    assert panel.showing_label.text().startswith("TEST")
    panel.close()


def test_missing_subsampled_frame_is_not_given_a_neighbouring_score(qapp):
    entry = _entry("subsampled")
    entry = replace(entry, scores=entry.scores[::2])
    panel = FrameComparePanel()
    panel.set_runs([entry])

    panel.set_frame(3)

    assert "not calculated for this frame" in panel.detail_label.text()


def test_hdr_preview_control_explains_the_effective_display_aware_conversion(qapp):
    entry = _entry("hdr")
    entry.comparison.distorted_info.color_transfer = "smpte2084"
    panel = FrameComparePanel()
    panel._display_hdr = DisplayHdrInfo(
        device_name=r"\\.\DISPLAY1",
        hdr_supported=True,
        hdr_enabled=True,
        bits_per_color_channel=10,
        sdr_white_nits=203.0,
    )

    panel.set_runs([entry])

    assert panel.color_mode_combo.currentData() == PreviewColorMode.DISPLAY_AWARE.value
    assert "HDR10 / PQ" in panel.color_status_label.text()
    assert "Windows HDR on" in panel.color_status_label.text()
    assert "203 nit" in panel.color_status_label.text()


def test_fixed_hdr_to_sdr_option_is_explicit_about_untagged_input(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("untagged")])
    automatic_key = panel._cache_key("distorted")
    fixed = panel.color_mode_combo.findData(PreviewColorMode.HDR_TO_SDR.value)

    panel.color_mode_combo.setCurrentIndex(fixed)

    assert "assuming HDR10 / PQ" in panel.color_status_label.text()
    assert "100 nit" in panel.color_status_label.text()
    assert panel._cache_key("distorted") != automatic_key



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
    assert "No metric results loaded" in panel.detail_label.text()
    assert "VMAF" not in panel.detail_label.text()
    panel.close()


def test_an_unscored_pair_seeks_like_a_scored_one(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_unscored_entry("encode")])

    panel.set_frame(48)

    assert panel.frame_spin.value() == 48
    assert panel.timeline.value() == 48
    assert panel.timestamp_edit.text() == "0:00:02.000"  # 48 / 24fps
    panel.close()


def test_scored_and_unscored_pairs_coexist(qapp):
    # Adding one encode to a table that already has a measured one must not
    # hide either of them.
    panel = FrameComparePanel()
    panel.set_runs([_entry("measured", count=120), _unscored_entry("fresh", count=90)])

    assert panel.video_combo.count() == 2
    # The timeline is bounded by the shorter of the two, as before.
    assert panel.frame_spin.maximum() == 89

    panel.set_frame(10)
    assert "VMAF v0.6.1 90.00" in panel.detail_label.text()
    panel.cycle_distorted(1)
    assert "No metric results loaded" in panel.detail_label.text()
    panel.close()


def test_pending_auto_crop_is_disclosed_rather_than_implied(qapp):
    """Auto-crop is measured during a run, so before one the frames are
    uncropped and a scored comparison would differ. Saying nothing would
    make the preview quietly misleading."""
    from dataclasses import replace as dc_replace

    entry = _unscored_entry("encode")
    entry = dc_replace(
        entry, comparison=dc_replace(entry.comparison, auto_crop_pending=True)
    )
    panel = FrameComparePanel()
    panel.set_runs([entry])

    assert "black bars not detected yet" in panel.detail_label.text()
    panel.close()


def test_unscored_auto_crop_is_detected_once_in_the_background(qapp, tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace

    from vmaf_app.core.models import CropBox

    entry = _physical_entry(tmp_path)
    entry = dc_replace(
        entry, comparison=dc_replace(entry.comparison, auto_crop_pending=True)
    )
    calls = []

    def fake_detect(info, **_kwargs):
        calls.append(info.path)
        # The same picture in each video's own pixels: 1920x1080, 1280x720.
        return CropBox(1920, 816, 0, 132) if info.width == 1920 else CropBox(1280, 544, 0, 88)

    monkeypatch.setattr("vmaf_app.ui.crop_detect_worker.detect_crop", fake_detect)
    panel = FrameComparePanel()
    panel.set_runs([entry])
    panel._ensure_auto_crop(panel.current_entry)
    for _ in range(100):
        qapp.processEvents()
        if not panel._crop_workers:
            break
        QTest.qWait(5)

    assert panel.current_entry.comparison.auto_crop_pending is False
    assert panel.current_entry.comparison.source_crop == CropBox(1920, 816, 0, 132)
    assert panel.current_entry.comparison.distorted_crop == CropBox(1280, 544, 0, 88)
    assert set(calls) == {tmp_path / "source.mkv", tmp_path / "encode.mkv"}
    panel.close()


def test_a_video_opens_when_its_files_black_bars_are_already_known(qapp, tmp_path, monkeypatch):
    """The same files removed and added again: their black bars were known,
    so the entry was swapped for a cropped copy, and the player was asked for
    the uncropped one, which was not among the videos it was given
    (ValueError): Video Compare stayed blank."""
    from vmaf_app.core.models import CropBox
    from vmaf_app.ui.video_compare_view import VideoCompareView

    entry = _physical_entry(tmp_path)
    entry = replace(entry, comparison=replace(entry.comparison, auto_crop_pending=True))
    monkeypatch.setattr(VideoCompareView, "_restart_decoder", lambda self, **_kwargs: None)
    panel = FrameComparePanel()
    panel._auto_crop_files[panel._crop_file_key(tmp_path / "source.mkv")] = CropBox(1920, 816, 0, 132)
    panel._auto_crop_files[panel._crop_file_key(tmp_path / "encode.mkv")] = CropBox(1280, 544, 0, 88)
    panel.set_runs([entry])

    panel._load_current_video()

    shown = panel.video_view._comparison
    assert shown is panel.current_entry.comparison and not shown.auto_crop_pending
    assert shown.source_crop == CropBox(1920, 816, 0, 132)
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


def test_a_scored_entry_never_claims_a_pending_crop(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("measured")])

    assert "black bars" not in panel.detail_label.text()
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


def test_the_empty_message_does_not_demand_a_vmaf_run(qapp):
    panel = FrameComparePanel()
    panel.set_runs([])

    message = panel.viewer._label.text()
    assert "VMAF" not in message
    assert "source" in message.lower()
    panel.close()


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


def test_video_mode_keeps_two_surfaces_ready_for_instant_s_switch(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([_physical_entry(tmp_path)])
    panel.show()
    panel.show()
    assert panel.video_view is not None
    QTest.keyPress(panel.video_view, Qt.Key_S)
    assert panel._showing_source is True
    assert panel.video_view._showing_source is True

    QTest.keyRelease(panel.video_view, Qt.Key_S)
    assert panel._showing_source is False
    assert panel.video_view._showing_source is False
    panel.close()


def test_switching_video_keeps_decoder_generation_and_preserves_seek(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([
        _physical_entry(tmp_path, "first"),
        _physical_entry(tmp_path, "second"),
    ])
    panel.set_frame(24)
    panel.show()
    assert panel.video_view is not None
    previous_generation = panel.video_view._generation

    panel.cycle_distorted(1)

    assert panel.video_view._generation == previous_generation
    assert panel.video_view._comparison.distorted_info.path.name == "second.mkv"
    assert panel.video_view.position == 1000
    panel.close()


def test_new_gpu_surfaces_receive_source_and_navigation_keys(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([_physical_entry(tmp_path, "a"), _physical_entry(tmp_path, "b")])
    panel.show()
    panel.show()
    view = panel.video_view
    view._distorted_surface.setFocus()
    QTest.keyPress(view._distorted_surface, Qt.Key_S)
    assert panel._showing_source
    QTest.keyRelease(view._source_surface, Qt.Key_S)
    assert not panel._showing_source
    QTest.keyClick(view._distorted_surface, Qt.Key_Right)
    assert panel._current_index == 1
    panel.close()


def test_source_resolution_option_preserves_seek_and_metric_recipe(qapp, tmp_path):
    panel = FrameComparePanel()
    entry = _physical_entry(tmp_path, "resolution")
    original = entry.comparison
    panel.set_runs([entry])
    assert panel.source_resolution_combo.isEnabled()
    panel.show()
    panel.set_frame(24)
    assert panel.video_view._source_native
    panel.source_resolution_combo.setCurrentIndex(1)
    assert not panel.video_view._source_native
    assert panel.video_view.position == 1000
    assert panel.current_entry.comparison is original
    panel.source_resolution_combo.setCurrentIndex(0)
    assert panel.video_view._source_native
    assert panel.video_view.position == 1000
    panel.close()


def test_top_arrow_button_switches_while_playback_is_requested(
    qapp, tmp_path, monkeypatch
):
    panel = FrameComparePanel()
    panel.set_runs([
        _physical_entry(tmp_path, "first"),
        _physical_entry(tmp_path, "second"),
    ])
    panel.set_frame(24)
    panel.show()
    view = panel.video_view
    assert view is not None
    view._wanted_playing = True
    view._is_playing = True
    loads = []
    monkeypatch.setattr(
        view, "load",
        lambda comparison, position, **kwargs: loads.append(
            (comparison, position, kwargs)
        ) or True,
    )

    panel.next_video_btn.click()

    assert panel.video_combo.currentText() == "second"
    assert loads[-1][1] == 1000
    assert loads[-1][2]["playing"] is True
    panel.cancel()
    panel.close()


def test_frame_controls_seek_both_video_players(qapp, tmp_path, monkeypatch):
    panel = FrameComparePanel()
    panel.set_runs([_physical_entry(tmp_path)])
    panel.show()
    assert panel.video_view is not None
    assert not hasattr(panel, "view_mode_combo")
    assert panel.content_stack.currentWidget() is panel.video_view
    assert panel.play_btn.isEnabled()
    assert panel.previous_frame_btn.isEnabled()
    assert panel.next_frame_btn.isEnabled()
    positions = []
    monkeypatch.setattr(panel.video_view, "set_position", positions.append)

    panel.set_frame(24)

    assert positions == [1000]
    panel.close()


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


def test_synthetic_resolution_test_stays_still_frame_only(qapp):
    entry = _entry("resolution")
    entry = replace(
        entry,
        comparison=replace(
            entry.comparison,
            resample_target=ResampleTarget(width=1280, label="720p"),
        ),
    )
    panel = FrameComparePanel()
    panel.set_runs([entry])

    assert not panel.play_btn.isEnabled()
    panel.show()
    assert not panel.is_video_mode
    assert panel.content_stack.currentWidget() is panel.viewer
    panel.close()


def test_video_mode_discloses_native_hdr_or_actual_sdr_fallback(qapp, tmp_path, monkeypatch):
    entry = _physical_entry(tmp_path, "hdr")
    monkeypatch.setattr(FrameComparePanel, "_refresh_display_hdr", lambda self: None)
    entry.comparison.distorted_info.color_transfer = "smpte2084"
    entry.comparison.distorted_info.color_primaries = "bt2020"
    panel = FrameComparePanel()
    panel._display_hdr = DisplayHdrInfo(
        device_name=r"\\.\DISPLAY1",
        hdr_supported=True,
        hdr_enabled=True,
        bits_per_color_channel=10,
        sdr_white_nits=203.0,
    )
    panel.set_runs([entry])

    panel.show()

    # Offscreen cannot create the native swapchain, so do not claim HDR
    # presentation simply because the monitor metadata says HDR is enabled.
    assert "SDR preview (native HDR unavailable)" in panel.color_status_label.text()
    panel.video_view._pool_active = False
    panel._update_color_status()
    status = panel.color_status_label.text()
    assert "native 10-bit D3D11 presentation" in status
    assert "tone mapping off" in status
    assert "HDR → SDR" not in status
    panel.close()


def test_the_video_being_watched_reloads_only_when_it_changed(qapp, monkeypatch):
    """Any video's result arriving during a run reloaded the video being
    watched, restarting it -- several times a video once scores land as
    each metric finishes."""
    panel = FrameComparePanel()
    monkeypatch.setattr(FrameComparePanel, "is_video_mode", property(lambda self: True))
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    loads = []
    monkeypatch.setattr(panel, "_load_current_video", lambda *a, **k: loads.append(panel.current_entry.label))
    watched, other = _entry("a"), _entry("b")
    panel.set_runs([watched, other])
    assert loads == ["a"]
    panel.set_runs([replace(watched, scores=None), _entry("b", score=50.0)])  # new scores: same video
    assert loads == ["a"]
    moved = replace(watched, comparison=replace(watched.comparison, frame_count=60))
    panel.set_runs([moved, other])  # what is on screen changed
    assert loads == ["a", "a"]
    panel.close()

