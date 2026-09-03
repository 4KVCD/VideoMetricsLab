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


# ---------------------------------------- sampling inside a duration limit

def _info(duration: float = 10.0) -> VideoInfo:
    return VideoInfo(
        path=Path("movie.mkv"), width=320, height=180, fps=30.0,
        duration=duration, nb_frames=int(duration * 30), codec_name="h264",
    )


def _recorded_windows(monkeypatch) -> list[tuple[float, float]]:
    """Captures every (start, window) crop detection actually reads."""
    seen: list[tuple[float, float]] = []

    def fake_window(path, start, window, limit, **kwargs):
        seen.append((start, window))
        return crop_detect.CropBox(w=320, h=180, x=0, y=0)

    monkeypatch.setattr(crop_detect, "_run_single_window", fake_window)
    return seen


@pytest.mark.parametrize("limit", [0.8, 2.0, 5.0])
def test_no_crop_sample_reaches_past_the_duration_limit(monkeypatch, limit):
    """A film that is full-frame for its opening seconds and letterboxed
    afterwards was measured on footage the comparison never looks at, so the
    detected bars were cropped away from content that really is there.

    Verified against a generated fixture (10s, full-frame for 1s then
    letterboxed): with a 0.8s limit this returned 320x100+0+40 before, and
    320x180 after.
    """
    seen = _recorded_windows(monkeypatch)

    crop_detect.detect_crop(_info(10.0), duration_limit=limit)

    assert seen, "no samples were taken"
    for start, window in seen:
        assert start >= 0.0
        assert start + window <= limit + 1e-6, (
            f"a sample read {start}..{start + window}s, past the {limit}s limit"
        )


def test_a_limit_shorter_than_one_window_still_takes_a_sample(monkeypatch):
    # The analysis window is 3s by default. A 0.8s limit has to shrink it
    # rather than read 3s of footage or give up and sample nothing.
    seen = _recorded_windows(monkeypatch)

    crop_detect.detect_crop(_info(10.0), duration_limit=0.8)

    assert len(seen) == 1
    start, window = seen[0]
    assert (start, round(window, 6)) == (0.0, 0.8)


def test_no_limit_samples_the_whole_video_as_before(monkeypatch):
    seen = _recorded_windows(monkeypatch)

    crop_detect.detect_crop(_info(10.0))

    assert [round(s, 3) for s, _w in seen] == [1.0, 2.5, 4.0, 5.5, 7.0]
    assert {w for _s, w in seen} == {_SAMPLE_WINDOW_SECONDS}


def test_a_limit_longer_than_the_video_changes_nothing(monkeypatch):
    seen = _recorded_windows(monkeypatch)

    crop_detect.detect_crop(_info(10.0), duration_limit=30.0)

    assert [round(s, 3) for s, _w in seen] == [1.0, 2.5, 4.0, 5.5, 7.0]


@pytest.mark.parametrize("resample", [False, True])
def test_both_run_paths_pass_the_duration_limit_through(monkeypatch, resample):
    """The round-trip-test path resolves crop separately, and had its own
    copy of the same omission."""
    from vmaf_app.core import vmaf_runner
    from vmaf_app.core.models import CropMode, ResampleTarget, VmafOptions

    seen: list[float] = []

    def fake_detect(info, **kwargs):
        seen.append(kwargs.get("duration_limit"))
        raise crop_detect.CropDetectCancelled("stop here")

    monkeypatch.setattr(vmaf_runner, "detect_crop", fake_detect)
    options = VmafOptions(
        crop_mode=CropMode.AUTO, duration_limit=1.5,
        resample_test=ResampleTarget(width=160, label="160") if resample else None,
    )
    source = _info(10.0)

    with pytest.raises(vmaf_runner.Cancelled):
        if resample:
            vmaf_runner.run_resample_test(source, options)
        else:
            vmaf_runner._resolve_crops(source, _info(10.0), options, None)

    assert seen and all(limit == 1.5 for limit in seen)
