from pathlib import Path

import pytest

from vmaf_app.core import crop_detect
from vmaf_app.core.crop_detect import _SAMPLE_WINDOW_SECONDS, CropDetectError, _sample_offsets
from vmaf_app.core.models import CropBox, VideoInfo


def test_crop_samples_never_start_beyond_the_last_full_window():
    for duration in (1.0, 3.0, 5.0, 10.0, 60.0):
        latest_valid_start = max(0.0, duration - _SAMPLE_WINDOW_SECONDS)
        offsets = _sample_offsets(duration)
        assert offsets
        assert all(0.0 <= offset <= latest_valid_start for offset in offsets)


def test_auto_crop_failure_is_reported_instead_of_silently_using_full_frame(monkeypatch):
    info = VideoInfo(
        path=Path("broken.mp4"), width=1920, height=1080, fps=30.0,
        duration=10.0, nb_frames=300, codec_name="h264",
    )
    def window(*a, **kw):
        raise CropDetectError("ffmpeg exited with code 1")

    monkeypatch.setattr(crop_detect, "_run_single_window", window)

    with pytest.raises(CropDetectError, match=r"None \(use full frame\)"):
        crop_detect.detect_crop(info)


# ------------------------------------------------------------- the cache

def _real_file(tmp_path, name="movie.mkv", duration=60.0) -> VideoInfo:
    path = tmp_path / name
    path.write_bytes(b"x" * 1000)
    return VideoInfo(
        path=path, width=1920, height=1080, fps=30.0,
        duration=duration, nb_frames=int(duration * 30), codec_name="h264",
    )


def test_a_files_bars_are_detected_once_per_process(monkeypatch, tmp_path):
    """Six encodes of one film detected the source's bars six times over --
    thirty ffmpeg processes for one answer."""
    info = _real_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        crop_detect, "_run_single_window",
        lambda *a, **kw: (calls.append(a[1]), crop_detect.CropBox(1920, 800, 0, 140))[1],
    )

    first = crop_detect.detect_crop(info)
    launched = len(calls)
    second = crop_detect.detect_crop(info)

    assert first == second
    assert launched == 5
    assert len(calls) == launched, "the second call ran detection again"


def test_a_replaced_file_is_detected_afresh(monkeypatch, tmp_path):
    info = _real_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        crop_detect, "_run_single_window",
        lambda *a, **kw: (calls.append(a[1]), crop_detect.CropBox(1920, 800, 0, 140))[1],
    )
    crop_detect.detect_crop(info)

    info.path.write_bytes(b"y" * 2000)  # new size: a different file
    crop_detect.detect_crop(info)

    assert len(calls) == 10


# ------------------------------------------------------- GPU decode


def test_a_failed_gpu_window_falls_back_to_software(monkeypatch):
    # No free decoder session, an unsupported profile: the metric run falls
    # back to the CPU, and so does this. Same pixels, same box.
    seen = []

    def fake_launch(cmd, cancel_event, process_handle):
        seen.append(cmd)
        if "-hwaccel" in cmd:
            return 1, "Failed to initialise hwaccel"
        return 0, "crop=1920:800:0:140"

    monkeypatch.setattr(crop_detect, "_launch_window", fake_launch)

    box = crop_detect._run_single_window("movie.mkv", 5.0, 3.0, 0.1, hwaccel="cuda")

    assert box == crop_detect.CropBox(1920, 800, 0, 140)
    assert len(seen) == 2
    assert "-hwaccel" not in seen[1]
    assert seen[1][seen[1].index("-vf") + 1].startswith("cropdetect=")


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


def _sized(width, height):
    from vmaf_app.core.models import VideoInfo

    return VideoInfo(Path(f"{width}x{height}.mkv"), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_two_boxes_a_row_apart_become_the_picture_both_show():
    """An encode's soft bar edge put its box two rows from the source's:
    1920x800 against 1920x802, compared by scaling one onto the other."""
    from vmaf_app.core.crop_detect import common_picture

    source, test = common_picture(_sized(1920, 1080), _sized(1920, 1080),
                                  CropBox(1920, 800, 0, 140), CropBox(1920, 802, 0, 138))
    assert source == test == CropBox(1920, 800, 0, 140)


def test_the_same_picture_at_two_sizes_is_matched_in_each_ones_pixels():
    from vmaf_app.core.crop_detect import common_picture

    source, test = common_picture(_sized(3840, 2160), _sized(1920, 1080),
                                  CropBox(3840, 1600, 0, 280), CropBox(1920, 804, 0, 138))
    assert source == CropBox(3840, 1600, 0, 280)
    assert test == CropBox(1920, 800, 0, 140)


def test_equally_common_boxes_go_to_the_larger(monkeypatch):
    """A dark stretch reads its dark picture as bar: its box is too tight.
    A tie went to whichever window answered first."""
    crop_detect.clear_cache()
    boxes = iter([CropBox(1920, 696, 0, 192), CropBox(1920, 800, 0, 140), CropBox(1920, 696, 0, 192),
                  CropBox(1920, 800, 0, 140), None])
    monkeypatch.setattr(crop_detect, "_run_single_window", lambda *a, **k: next(boxes))
    info = VideoInfo(path=Path("tie.mkv"), width=1920, height=1080, fps=24.0, duration=600.0, nb_frames=14400,
                     codec_name="hevc")
    assert crop_detect.detect_crop(info) == CropBox(1920, 800, 0, 140)
