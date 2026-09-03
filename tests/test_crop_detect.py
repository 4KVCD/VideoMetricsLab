import threading
from pathlib import Path

import pytest

from vmaf_app.core import crop_detect
from vmaf_app.core.crop_detect import _SAMPLE_WINDOW_SECONDS, CropDetectError, _sample_offsets
from vmaf_app.core.models import VideoInfo


def test_crop_samples_never_start_beyond_the_last_full_window():
    for duration in (1.0, 3.0, 5.0, 10.0, 60.0):
        latest_valid_start = max(0.0, duration - _SAMPLE_WINDOW_SECONDS)
        offsets = _sample_offsets(duration)
        assert offsets
        assert all(0.0 <= offset <= latest_valid_start for offset in offsets)


def test_very_short_clip_is_sampled_once_from_the_start():
    assert _sample_offsets(2.0) == [0.0]


def test_auto_crop_failure_is_reported_instead_of_silently_using_full_frame(monkeypatch):
    info = VideoInfo(
        path=Path("broken.mp4"), width=1920, height=1080, fps=30.0,
        duration=10.0, nb_frames=300, codec_name="h264",
    )
    monkeypatch.setattr(crop_detect, "_run_single_window", lambda *a, **kw: None)

    with pytest.raises(CropDetectError, match=r"None \(use full frame\)"):
        crop_detect.detect_crop(info)


def test_crop_detection_stops_before_another_window_after_cancel(monkeypatch):
    info = VideoInfo(
        path=Path("movie.mp4"), width=1920, height=1080, fps=30.0,
        duration=60.0, nb_frames=1800, codec_name="h264",
    )
    cancel = threading.Event()
    calls = []

    def first_window(*args, **kwargs):
        calls.append(args[1])
        cancel.set()
        return crop_detect.CropBox(1920, 1080, 0, 0)

    monkeypatch.setattr(crop_detect, "_run_single_window", first_window)

    with pytest.raises(crop_detect.CropDetectCancelled):
        crop_detect.detect_crop(info, cancel_event=cancel)
    assert len(calls) == 1
