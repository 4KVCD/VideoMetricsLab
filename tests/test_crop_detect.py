import threading
import time
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


def test_cancel_wins_over_windows_that_already_answered(monkeypatch):
    """The windows run concurrently, so some may have a box in hand by the
    time Cancel lands. A partial vote is not an answer the caller asked for:
    cancellation is reported, and nothing is returned."""
    info = VideoInfo(
        path=Path("movie.mp4"), width=1920, height=1080, fps=30.0,
        duration=60.0, nb_frames=1800, codec_name="h264",
    )
    cancel = threading.Event()

    def window(*args, **kwargs):
        cancel.set()
        return crop_detect.CropBox(1920, 1080, 0, 0)

    monkeypatch.setattr(crop_detect, "_run_single_window", window)

    with pytest.raises(crop_detect.CropDetectCancelled):
        crop_detect.detect_crop(info, cancel_event=cancel)


def test_a_window_does_not_launch_once_cancelled():
    # The real window checks before starting a process, so a cancel that
    # lands while others are running never starts another ffmpeg.
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(crop_detect.CropDetectCancelled):
        crop_detect._run_single_window("movie.mp4", 0.0, 3.0, 0.1, cancel_event=cancel)


def test_windows_run_at_the_same_time(monkeypatch):
    """Five sequential 4K windows measured 6.9s; five concurrent ones 2.0s.
    They are independent samples, and nothing about the vote needs them in
    order."""
    info = VideoInfo(
        path=Path("movie.mp4"), width=1920, height=1080, fps=30.0,
        duration=60.0, nb_frames=1800, codec_name="h264",
    )
    running = 0
    peak = 0
    lock = threading.Lock()

    def window(*args, **kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return crop_detect.CropBox(1920, 800, 0, 140)

    monkeypatch.setattr(crop_detect, "_run_single_window", window)

    assert crop_detect.detect_crop(info) == crop_detect.CropBox(1920, 800, 0, 140)
    assert peak > 1, "the windows ran one after another"


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


def test_a_different_scored_stretch_is_a_different_answer(monkeypatch, tmp_path):
    # The samples are taken from inside the stretch that will be scored, so
    # a duration limit changes what is measured and must not reuse the
    # full-length answer.
    info = _real_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        crop_detect, "_run_single_window",
        lambda *a, **kw: (calls.append(a[1]), crop_detect.CropBox(1920, 800, 0, 140))[1],
    )

    crop_detect.detect_crop(info)
    crop_detect.detect_crop(info, duration_limit=10.0)

    assert len(calls) == 10


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


def test_a_failed_detection_is_not_remembered(monkeypatch, tmp_path):
    info = _real_file(tmp_path)
    monkeypatch.setattr(crop_detect, "_run_single_window", lambda *a, **kw: None)
    with pytest.raises(CropDetectError):
        crop_detect.detect_crop(info)

    monkeypatch.setattr(
        crop_detect, "_run_single_window",
        lambda *a, **kw: crop_detect.CropBox(1920, 800, 0, 140),
    )
    assert crop_detect.detect_crop(info) == crop_detect.CropBox(1920, 800, 0, 140)


def test_two_callers_for_one_file_share_a_single_detection(monkeypatch, tmp_path):
    """Two parallel lanes starting on the same source at the same moment
    used to launch ten processes for one answer. The second now waits for
    the first."""
    info = _real_file(tmp_path)
    calls = []
    lock = threading.Lock()

    def window(*a, **kw):
        with lock:
            calls.append(a[1])
        time.sleep(0.1)
        return crop_detect.CropBox(1920, 800, 0, 140)

    monkeypatch.setattr(crop_detect, "_run_single_window", window)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(crop_detect.detect_crop(info)))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [crop_detect.CropBox(1920, 800, 0, 140)] * 2
    assert len(calls) == 5


# ------------------------------------------------------- GPU decode

@pytest.mark.parametrize("hwaccel", ["cuda", "qsv", "d3d11va"])
def test_a_window_decodes_on_whatever_gpu_the_run_will_use(monkeypatch, hwaccel):
    """No vendor is named in crop detection. It takes the decoder the run's
    own plan chose -- NVIDIA, Intel or AMD, whatever this machine has and
    this ffmpeg build supports -- and passes it through unchanged."""
    seen = []

    def fake_launch(cmd, cancel_event, process_handle):
        seen.append(cmd)
        return 0, "crop=1920:800:0:140"

    monkeypatch.setattr(crop_detect, "_launch_window", fake_launch)

    box = crop_detect._run_single_window(
        "movie.mkv", 5.0, 3.0, 0.1, hwaccel=hwaccel, download_format="p010le"
    )

    assert box == crop_detect.CropBox(1920, 800, 0, 140)
    (cmd,) = seen
    assert cmd[cmd.index("-hwaccel") + 1] == hwaccel
    assert cmd[cmd.index("-hwaccel_output_format") + 1] == hwaccel
    assert cmd.index("-hwaccel") < cmd.index("-i")  # a per-input option
    assert "hwdownload,format=p010le,cropdetect=" in cmd[cmd.index("-vf") + 1]


def test_crop_detection_names_no_vendor_of_its_own():
    # Belt and braces for the above: the module has no idea what a GPU is.
    source = Path(crop_detect.__file__).read_text(encoding="utf-8").lower()
    for word in ("cuda", "qsv", "d3d11", "nvidia", "intel", "amd", "vaapi", "videotoolbox"):
        assert word not in source, f"crop_detect hardcodes {word!r}"


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


def test_no_gpu_means_no_fallback_attempt(monkeypatch):
    seen = []
    monkeypatch.setattr(
        crop_detect, "_launch_window",
        lambda cmd, *_: (seen.append(cmd), (1, "boom"))[1],
    )
    with pytest.raises(CropDetectError):
        crop_detect._run_single_window("movie.mkv", 5.0, 3.0, 0.1)
    assert len(seen) == 1


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

    assert sorted(round(s, 3) for s, _w in seen) == [1.0, 2.5, 4.0, 5.5, 7.0]
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
