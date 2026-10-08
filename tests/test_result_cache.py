from pathlib import Path

import pytest

from vmaf_app.core import result_cache
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    displayable_metric_specs,
)
from vmaf_app.core.models import (
    ComparisonResult,
    CropMode,
    FrameScore,
    GpuVendor,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
)


def _cache_request(options):
    return analysis_request_from_vmaf_options(options)


def _cache_key(source, distorted, options):
    return result_cache.cache_key(source, distorted, _cache_request(options))


def _load_cached(source, distorted, options, directory=None):
    return result_cache.load_cached(
        source, distorted, _cache_request(options), directory,
        displayable_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(
        source, distorted, result, label, _cache_request(options), directory
    )


def _clear_cached(source, distorted, options, directory=None):
    # The ticked metrics only, as the window clears them for a recalculation.
    return result_cache.clear(source, distorted, _cache_request(options), directory)

OPTIONS = VmafOptions()


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)


def _make_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _fake_result(source: Path, distorted: Path) -> ComparisonResult:
    info = VideoInfo(path=distorted, width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=10, codec_name="h264")
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    return ComparisonResult(
        source=source, distorted=distorted, frames=frames, fps=30.0, model="version=vmaf_v0.6.1",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )


def test_store_then_load_cached_round_trips(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)

    assert _load_cached(source, distorted, OPTIONS) is None  # nothing cached yet

    result = _fake_result(source, distorted)
    _store_cached(source, distorted, result, label="my-run", options=OPTIONS)

    loaded = _load_cached(source, distorted, OPTIONS)
    assert loaded is not None
    loaded_result, label = loaded
    assert label == "my-run"
    assert len(loaded_result.frames) == len(result.frames)


def test_cache_keys_differ_between_vmaf_model_choices(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    v0 = VmafOptions(model_choice="version=vmaf_v0.6.1")
    v1 = VmafOptions(model_choice="__builtin:vmaf_v1_3d0h")

    assert _cache_key(source, distorted, v0) != _cache_key(
        source, distorted, v1
    )


def test_cache_miss_when_distorted_file_size_differs(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted_v1 = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(source, distorted_v1, _fake_result(source, distorted_v1), label="v1", options=OPTIONS)

    # Same name, different size (e.g. the file was re-encoded) -- must miss.
    distorted_v2 = _make_file(tmp_path / "distorted.mp4", 999)
    assert _load_cached(source, distorted_v2, OPTIONS) is None


def test_cache_miss_when_a_same_size_file_is_replaced_in_place(tmp_path):
    import os

    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "movie.mp4", 500)
    _store_cached(source, distorted, _fake_result(source, distorted), label="old", options=OPTIONS)
    old_mtime = distorted.stat().st_mtime_ns
    distorted.write_bytes(b"y" * 500)
    os.utime(distorted, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    assert _load_cached(source, distorted, OPTIONS) is None


def test_clear_removes_the_cached_entry(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(source, distorted, _fake_result(source, distorted), label="x", options=OPTIONS)
    assert _load_cached(source, distorted, OPTIONS) is not None

    _clear_cached(source, distorted, OPTIONS)
    assert _load_cached(source, distorted, OPTIONS) is None


def test_cache_hit_survives_execution_only_option_changes(tmp_path_factory, subtests):
    def check(changed, tmp_path):
        source = _make_file(tmp_path / "source.mp4", 1000)
        distorted = _make_file(tmp_path / "distorted.mp4", 500)
        _store_cached(
            source, distorted, _fake_result(source, distorted),
            label="original", options=VmafOptions(),
        )

        assert _load_cached(source, distorted, changed) is not None

    for changed in [
            VmafOptions(gpu_decode=False),
            VmafOptions(gpu_vendor=GpuVendor.NVIDIA),
            VmafOptions(n_threads=12),
            VmafOptions(gpu_decode=False, gpu_vendor=GpuVendor.AMD, n_threads=4),
            # Scaled with another algorithm (or on the GPU), a comparison is the
            # same comparison (ComparisonRecipe.identity_dict).
            VmafOptions(scale_algorithm="lanczos"),
            VmafOptions(scale_algorithm="bilinear"),
        ]:
        with subtests.test(changed=changed):
            check(changed, tmp_path_factory.mktemp("case"))


def test_cache_miss_survives_score_or_output_option_changes(tmp_path_factory, subtests):
    def check(changed, tmp_path):
        source = _make_file(tmp_path / "source.mp4", 1000)
        distorted = _make_file(tmp_path / "distorted.mp4", 500)
        _store_cached(
            source, distorted, _fake_result(source, distorted),
            label="original", options=VmafOptions(),
        )

        assert _load_cached(source, distorted, changed) is None

    for changed in [
            VmafOptions(
                model="version=vmaf_4k_v0.6.1",
                model_choice="version=vmaf_4k_v0.6.1",
            ),
            VmafOptions(crop_mode=CropMode.NONE),
            VmafOptions(scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
            VmafOptions(duration_limit=2.0),
            VmafOptions(n_subsample=5),
        ]:
        with subtests.test(changed=changed):
            check(changed, tmp_path_factory.mktemp("case"))


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

    Per-metric entries are independent, so adding PSNR/SSIM/XPSNR later must
    still surface the VMAF score that was already measured for the same
    comparison recipe.
    """
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),  # VMAF only
    )

    found = _load_cached(source, distorted, asked_for)
    assert found is not None
    assert found[1] == "original"


def test_a_fuller_run_is_preferred_over_a_thinner_one(tmp_path):
    # Both could answer; the one carrying more of what was asked for wins,
    # so a second run fills in fewer gaps.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="vmaf only", options=VmafOptions(),
    )
    _store_cached(
        source, distorted, _fake_result(source, distorted), label="vmaf and psnr",
        options=VmafOptions(extra_features=["name=psnr"]),
    )

    found = _load_cached(
        source, distorted,
        VmafOptions(extra_features=["name=psnr", "name=float_ssim"]),
    )
    assert found is not None and found[1] == "vmaf and psnr"


def test_reuse_across_metrics_still_respects_how_frames_were_compared(tmp_path):
    # The relaxation is ONLY about which metrics were recorded. A run that
    # cropped differently, or sampled different frames, measured different
    # pictures and must never be offered for a different setting.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(n_subsample=1),
    )

    asked = VmafOptions(
        n_subsample=5, extra_features=["name=psnr"], compute_xpsnr=True
    )
    assert _load_cached(source, distorted, asked) is None


AUTO, V061, V4K = "__auto__", "version=vmaf_v0.6.1", "version=vmaf_4k_v0.6.1"
