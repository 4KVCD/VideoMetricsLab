from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.display_hdr import DisplayHdrInfo
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorMode
from vmaf_app.core.models import FrameScores, ResampleTarget, VideoInfo, VmafRunResult
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
    assert "VMAF 90.00" in panel.detail_label.text()


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
    assert "VMAF 90.00" in panel.detail_label.text()
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


def test_a_scored_entry_never_claims_a_pending_crop(qapp):
    panel = FrameComparePanel()
    panel.set_runs([_entry("measured")])

    assert "black bars" not in panel.detail_label.text()
    panel.close()


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
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
    assert panel.video_view is not None
    QTest.keyPress(panel.video_view.distorted_video, Qt.Key_S)
    assert panel._showing_source is True
    assert panel.video_view.video._show_source is True

    QTest.keyRelease(panel.video_view.source_video, Qt.Key_S)
    assert panel._showing_source is False
    assert panel.video_view.video._show_source is False
    panel.close()


def test_switching_video_keeps_decoder_generation_and_preserves_seek(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([
        _physical_entry(tmp_path, "first"),
        _physical_entry(tmp_path, "second"),
    ])
    panel.set_frame(24)
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
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
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
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
    assert not panel.source_resolution_combo.isEnabled()
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
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
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
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
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
    assert panel.video_view is not None
    positions = []
    monkeypatch.setattr(panel.video_view, "set_position", positions.append)

    panel.set_frame(24)

    assert positions == [1000]
    panel.close()


def test_s_switches_halves_of_the_same_decoded_frame_pair(qapp, tmp_path):
    panel = FrameComparePanel()
    panel.set_runs([_physical_entry(tmp_path)])
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
    assert panel.video_view is not None
    view = panel.video_view
    payload = bytes(range(12))
    view.video.set_pair(payload, 2, 1)

    view.show_source(True)

    assert view.video._show_source is True
    assert view.video._payload is payload
    view.show_source(False)
    assert view.video._show_source is False
    assert view.video._payload is payload
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
    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))
    assert "synthetic resolution tests" in panel.color_status_label.text()
    panel.close()


def test_video_mode_discloses_native_hdr_or_actual_sdr_fallback(qapp, tmp_path):
    entry = _physical_entry(tmp_path, "hdr")
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

    panel.view_mode_combo.setCurrentIndex(panel.view_mode_combo.findData("video"))

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
