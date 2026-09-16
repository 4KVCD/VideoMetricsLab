from pathlib import Path

import pytest

from vmaf_app.core import result_cache
from vmaf_app.core.models import (
    CropMode,
    FrameScore,
    GpuVendor,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
    VmafRunResult,
)

OPTIONS = VmafOptions()


def test_reverse_stored_order_is_found_and_cleared(tmp_path):
    source = _make_file(tmp_path / "source.mkv", 10)
    distorted = _make_file(tmp_path / "distorted.mkv", 5)
    reverse = VmafOptions(extra_features=["name=float_ssim", "name=psnr"])
    normal = VmafOptions(extra_features=["name=psnr", "name=float_ssim"])
    result_cache.store(source, distorted, _fake_result(source, distorted), "reverse", reverse)
    result_cache.store(source, distorted, _fake_result(source, distorted), "partial", VmafOptions())
    assert result_cache.load_cached(source, distorted, normal)[1] == "reverse"
    result_cache.clear(source, distorted, normal)
    assert result_cache.load_cached(source, distorted, reverse) is None


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)


def test_default_cache_is_stable_under_the_user_profile(monkeypatch, tmp_path):
    """The default must not depend on Qt's launcher/package identity."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert result_cache.default_cache_dir() == (
        tmp_path / ".vmaf-calculator" / "results_cache"
    )


def test_settings_and_cache_share_the_same_stable_data_root(monkeypatch, tmp_path):
    from vmaf_app.core.app_paths import settings_file

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert settings_file().parent == result_cache.default_cache_dir().parent


def _make_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _fake_result(source: Path, distorted: Path) -> VmafRunResult:
    info = VideoInfo(path=distorted, width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=10, codec_name="h264")
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    return VmafRunResult(
        source=source, distorted=distorted, frames=frames, fps=30.0, model="version=vmaf_v0.6.1",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )


def test_store_then_load_cached_round_trips(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)

    assert result_cache.load_cached(source, distorted, OPTIONS) is None  # nothing cached yet

    result = _fake_result(source, distorted)
    result_cache.store(source, distorted, result, label="my-run", options=OPTIONS)

    loaded = result_cache.load_cached(source, distorted, OPTIONS)
    assert loaded is not None
    loaded_result, label = loaded
    assert label == "my-run"
    assert len(loaded_result.frames) == len(result.frames)


def test_cache_miss_when_distorted_file_size_differs(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted_v1 = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(source, distorted_v1, _fake_result(source, distorted_v1), label="v1", options=OPTIONS)

    # Same name, different size (e.g. the file was re-encoded) -- must miss.
    distorted_v2 = _make_file(tmp_path / "distorted.mp4", 999)
    assert result_cache.load_cached(source, distorted_v2, OPTIONS) is None


def test_cache_miss_when_filename_differs_even_with_same_size(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "a.mp4", 500)
    b = _make_file(tmp_path / "b.mp4", 500)
    result_cache.store(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert result_cache.load_cached(source, b, OPTIONS) is None


def test_cache_miss_for_same_filename_and_size_in_a_different_directory(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "encode-a" / "movie.mp4", 500)
    b = _make_file(tmp_path / "encode-b" / "movie.mp4", 500)
    result_cache.store(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert result_cache.load_cached(source, b, OPTIONS) is None


def test_cache_miss_when_a_same_size_file_is_replaced_in_place(tmp_path):
    import os

    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "movie.mp4", 500)
    result_cache.store(source, distorted, _fake_result(source, distorted), label="old", options=OPTIONS)
    old_mtime = distorted.stat().st_mtime_ns
    distorted.write_bytes(b"y" * 500)
    os.utime(distorted, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    assert result_cache.load_cached(source, distorted, OPTIONS) is None


def test_clear_removes_the_cached_entry(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(source, distorted, _fake_result(source, distorted), label="x", options=OPTIONS)
    assert result_cache.load_cached(source, distorted, OPTIONS) is not None

    result_cache.clear(source, distorted, OPTIONS)
    assert result_cache.load_cached(source, distorted, OPTIONS) is None


def test_clear_on_nonexistent_entry_does_not_raise(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.clear(source, distorted, OPTIONS)  # never stored -- must not raise


def test_cache_miss_when_calculation_options_change(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    original = VmafOptions(n_subsample=1, extra_features=["name=psnr"])
    changed = VmafOptions(n_subsample=5, extra_features=["name=psnr"])
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="original", options=original
    )

    assert result_cache.load_cached(source, distorted, changed) is None


@pytest.mark.parametrize(
    "changed",
    [
        VmafOptions(gpu_decode=False),
        VmafOptions(gpu_vendor=GpuVendor.NVIDIA),
        VmafOptions(n_threads=12),
        VmafOptions(gpu_decode=False, gpu_vendor=GpuVendor.AMD, n_threads=4),
    ],
)
def test_cache_hit_survives_execution_only_option_changes(tmp_path, changed):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),
    )

    assert result_cache.load_cached(source, distorted, changed) is not None


@pytest.mark.parametrize(
    "changed",
    [
        VmafOptions(model_choice="version=vmaf_v0.6.1neg"),
        VmafOptions(crop_mode=CropMode.NONE),
        VmafOptions(scale_algorithm="lanczos"),
        VmafOptions(scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        VmafOptions(duration_limit=2.0),
        VmafOptions(n_subsample=5),
    ],
)
def test_cache_miss_survives_score_or_output_option_changes(tmp_path, changed):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),
    )

    assert result_cache.load_cached(source, distorted, changed) is None


# ------------------------------------------------------- reuse across metrics


@pytest.mark.parametrize(
    "asked_for",
    [
        VmafOptions(extra_features=["name=psnr"]),
        VmafOptions(compute_xpsnr=True),
        VmafOptions(extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True),
    ],
)
def test_asking_for_more_metrics_still_finds_an_earlier_run(tmp_path, asked_for):
    """A finished measurement must not be discarded for wanting more from it.

    Which metrics a run recorded is part of its cache identity, so when the
    default changed from VMAF alone to all four, every previously scored
    video became a cache miss and offered to recompute itself from nothing.
    The frames compared were identical; only the set of numbers written down
    differed.
    """
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),  # VMAF only
    )

    found = result_cache.load_cached(source, distorted, asked_for)
    assert found is not None
    assert found[1] == "original"


def test_clearing_forgets_runs_recorded_with_other_metric_sets(tmp_path):
    """"Ignore cached results" has to mean every run a lookup could return.

    Clearing only the exact-options entry left an older run of the same pair
    alive, and load_cached reuses those across metric sets -- so a row told
    to forget its results could pick one up again later.
    """
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    all_four = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="older vmaf-only run", options=VmafOptions(),
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="today's run", options=all_four,
    )
    # Same pair, but a setting that changes what was measured.
    elsewhere = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True,
        n_subsample=5,
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="a different recipe", options=elsewhere,
    )

    result_cache.clear(source, distorted, all_four)

    assert result_cache.load_cached(source, distorted, all_four) is None
    # ...but a run that measured different pictures was never in scope.
    survivor = result_cache.load_cached(source, distorted, elsewhere)
    assert survivor is not None and survivor[1] == "a different recipe"


def test_asking_for_fewer_metrics_finds_the_fuller_run(tmp_path):
    # A run holding everything asked for and more is a complete answer.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="all four",
        options=VmafOptions(
            extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True
        ),
    )

    assert result_cache.load_cached(source, distorted, VmafOptions()) is not None


def test_a_fuller_run_is_preferred_over_a_thinner_one(tmp_path):
    # Both could answer; the one carrying more of what was asked for wins,
    # so a second run fills in fewer gaps.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="vmaf only", options=VmafOptions(),
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="vmaf and psnr",
        options=VmafOptions(extra_features=["name=psnr"]),
    )

    found = result_cache.load_cached(
        source, distorted,
        VmafOptions(extra_features=["name=psnr", "name=float_ssim"]),
    )
    assert found is not None and found[1] == "vmaf and psnr"


def test_a_fuller_run_beats_an_exact_match_that_recorded_less(tmp_path):
    """VMAF NEG was computed on top of a finished four-metric run, which
    stored a second, fuller file beside the first. Re-adding the video asked
    for the four again; the exact match was tried first and won, and the
    NEG scores sat unseen in the other file. Every score a run holds is
    shown, so the run holding the most of them is the one to load."""
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    four = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True
    )
    five = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True,
        compute_vmaf_neg=True,
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="four", options=four,
    )
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="four and NEG", options=five,
    )

    found = result_cache.load_cached(source, distorted, four)
    assert found is not None and found[1] == "four and NEG"

    # Still found under its own name, and forgotten together with the rest.
    assert result_cache.load_cached(source, distorted, five)[1] == "four and NEG"
    result_cache.clear(source, distorted, four)
    assert result_cache.load_cached(source, distorted, five) is None


def test_reuse_across_metrics_still_respects_how_frames_were_compared(tmp_path):
    # The relaxation is ONLY about which metrics were recorded. A run that
    # cropped differently, or sampled different frames, measured different
    # pictures and must never be offered for a different setting.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(n_subsample=1),
    )

    asked = VmafOptions(
        n_subsample=5, extra_features=["name=psnr"], compute_xpsnr=True
    )
    assert result_cache.load_cached(source, distorted, asked) is None


def test_metric_order_does_not_hide_a_cached_run(tmp_path):
    """extra_features is a list, so its order is part of the identity, and it
    is appended to in whatever order the metrics were ticked. Ticking SSIM
    before PSNR must not lose the run that ticking PSNR first produced."""
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    result_cache.store(
        source, distorted, _fake_result(source, distorted), label="original",
        options=VmafOptions(extra_features=["name=psnr", "name=float_ssim"]),
    )

    reversed_order = VmafOptions(extra_features=["name=float_ssim", "name=psnr"])
    assert result_cache.load_cached(source, distorted, reversed_order) is not None
