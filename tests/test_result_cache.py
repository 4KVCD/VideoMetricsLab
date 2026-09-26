from pathlib import Path

import pytest

from vmaf_app.core import result_cache
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    supplemental_metric_specs,
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
        supplemental_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(
        source, distorted, result, label, _cache_request(options), directory
    )


def _clear_cached(source, distorted, options, directory=None):
    return result_cache.clear(
        source, distorted, _cache_request(options), directory,
        supplemental_metric_specs(options),
    )

OPTIONS = VmafOptions()


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)


def test_default_cache_is_stable_under_the_user_profile(monkeypatch, tmp_path):
    """The default must not depend on Qt's launcher/package identity."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert result_cache.default_cache_dir() == (
        tmp_path / ".videometricslab" / "results_cache"
    )


def test_settings_and_cache_share_the_same_stable_data_root(monkeypatch, tmp_path):
    from vmaf_app.core.app_paths import settings_file

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert settings_file().parent == result_cache.default_cache_dir().parent


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


def test_metric_feature_order_does_not_change_cache_identity(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    normal = VmafOptions(extra_features=["name=psnr", "name=float_ssim"])
    reversed_order = VmafOptions(extra_features=["name=float_ssim", "name=psnr"])

    assert _cache_key(source, distorted, normal) == _cache_key(
        source, distorted, reversed_order
    )


def test_cache_miss_when_distorted_file_size_differs(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted_v1 = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(source, distorted_v1, _fake_result(source, distorted_v1), label="v1", options=OPTIONS)

    # Same name, different size (e.g. the file was re-encoded) -- must miss.
    distorted_v2 = _make_file(tmp_path / "distorted.mp4", 999)
    assert _load_cached(source, distorted_v2, OPTIONS) is None


def test_cache_miss_when_filename_differs_even_with_same_size(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "a.mp4", 500)
    b = _make_file(tmp_path / "b.mp4", 500)
    _store_cached(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert _load_cached(source, b, OPTIONS) is None


def test_cache_miss_for_same_filename_and_size_in_a_different_directory(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    a = _make_file(tmp_path / "encode-a" / "movie.mp4", 500)
    b = _make_file(tmp_path / "encode-b" / "movie.mp4", 500)
    _store_cached(source, a, _fake_result(source, a), label="a", options=OPTIONS)

    assert _load_cached(source, b, OPTIONS) is None


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


def test_clear_on_nonexistent_entry_does_not_raise(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _clear_cached(source, distorted, OPTIONS)  # never stored -- must not raise


def test_cache_miss_when_calculation_options_change(tmp_path):
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    original = VmafOptions(n_subsample=1, extra_features=["name=psnr"])
    changed = VmafOptions(n_subsample=5, extra_features=["name=psnr"])
    _store_cached(
        source, distorted, _fake_result(source, distorted), label="original", options=original
    )

    assert _load_cached(source, distorted, changed) is None


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
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),
    )

    assert _load_cached(source, distorted, changed) is not None


@pytest.mark.parametrize(
    "changed",
    [
        VmafOptions(
            model="version=vmaf_4k_v0.6.1",
            model_choice="version=vmaf_4k_v0.6.1",
        ),
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
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="original", options=VmafOptions(),
    )

    assert _load_cached(source, distorted, changed) is None


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


def test_clearing_forgets_runs_recorded_with_other_metric_sets(tmp_path):
    """"Ignore cached results" has to mean every run a lookup could return.

    A row can display compatible supplemental metrics in addition to the ones
    currently requested, so clearing must remove every metric identity that
    lookup could return for this scientific request.
    """
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    all_four = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True
    )
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="older vmaf-only run", options=VmafOptions(),
    )
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="today's run", options=all_four,
    )
    # Same pair, but a setting that changes what was measured.
    elsewhere = VmafOptions(
        extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True,
        n_subsample=5,
    )
    _store_cached(
        source, distorted, _fake_result(source, distorted),
        label="a different recipe", options=elsewhere,
    )

    _clear_cached(source, distorted, all_four)

    assert _load_cached(source, distorted, all_four) is None
    # ...but a run that measured different pictures was never in scope.
    survivor = _load_cached(source, distorted, elsewhere)
    assert survivor is not None and survivor[1] == "a different recipe"


def test_asking_for_fewer_metrics_finds_the_fuller_run(tmp_path):
    # A run holding everything asked for and more is a complete answer.
    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    _store_cached(
        source, distorted, _fake_result(source, distorted), label="all four",
        options=VmafOptions(
            extra_features=["name=psnr", "name=float_ssim"], compute_xpsnr=True
        ),
    )

    assert _load_cached(source, distorted, VmafOptions()) is not None


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
    _store_cached(
        source, distorted, _fake_result(source, distorted), label="four", options=four,
    )
    _store_cached(
        source, distorted, _fake_result(source, distorted), label="four and NEG", options=five,
    )

    found = _load_cached(source, distorted, four)
    assert found is not None and found[1] == "four and NEG"

    # Still found under its own name, and forgotten together with the rest.
    assert _load_cached(source, distorted, five)[1] == "four and NEG"
    _clear_cached(source, distorted, four)
    assert _load_cached(source, distorted, five) is None


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


def test_other_cvvdp_scores_are_those_of_other_displays_of_the_same_comparison(tmp_path):
    from vmaf_app.core.cvvdp import BUILTIN_PRESETS, CvvdpSettings
    from vmaf_app.core.metric_results import MetricProvenance, MetricResultSet, SequenceMetricResult

    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    other = _make_file(tmp_path / "other.mp4", 400)
    scored, unscored = BUILTIN_PRESETS[3].settings, BUILTIN_PRESETS[0].settings

    def request(settings):
        return analysis_request_from_vmaf_options(OPTIONS, ("vmaf", "cvvdp"), cvvdp=settings)

    provenance = MetricProvenance("Vship/cvvdp", "", "gpu", "cvvdp-vship-gpu-v1", dict(scored.spec_parameters()))
    result = _fake_result(source, distorted)
    result.merge_metric_results(MetricResultSet([SequenceMetricResult("cvvdp", 9.25, provenance)]))
    result_cache.store(source, distorted, result, "run", request(scored))

    parameters, found = result_cache.other_cvvdp_scores(source, distorted, request(unscored))
    assert parameters == tuple(unscored.spec_parameters())
    assert [(settings.same_as(scored), score) for settings, score in found] == [(True, 9.25)]
    assert result_cache.other_cvvdp_scores(source, distorted, request(scored))[1] == []  # its own is no "other"
    assert result_cache.other_cvvdp_scores(source, other, request(unscored))[1] == []  # another video's
    resized = CvvdpSettings(scored.display, resize_to_display=not scored.resize_to_display)
    assert [s.same_as(scored) for s, _ in result_cache.other_cvvdp_scores(source, distorted, request(resized))[1]] \
        == [True]
    assert result_cache.other_cvvdp_scores(
        source, distorted, analysis_request_from_vmaf_options(OPTIONS, ("vmaf",))) == ((), [])


@pytest.mark.parametrize("parameters", ["abc", 5, None, [1, 2]])
def test_a_damaged_saved_cvvdp_score_is_skipped_when_listing_other_displays(tmp_path, parameters):
    """Its parameters were read outside the error handling: one damaged file
    raised out of the background lookup, and the videos after it lost their
    saved scores."""
    import json

    import numpy as np

    from vmaf_app.core import metric_cache
    from vmaf_app.core.cvvdp import BUILTIN_PRESETS

    source = _make_file(tmp_path / "source.mp4", 1000)
    distorted = _make_file(tmp_path / "distorted.mp4", 500)
    request = analysis_request_from_vmaf_options(OPTIONS, ("vmaf", "cvvdp"), cvvdp=BUILTIN_PRESETS[0].settings)
    spec = next(s for s in request.metrics if s.key == "cvvdp")
    directory = metric_cache.recipe_directory(result_cache.cache_dir(), source, distorted, request.recipe)
    directory.mkdir(parents=True)
    identity = {**spec.identity_dict(), "parameters": parameters}
    metadata = {"format_version": metric_cache.METRIC_CACHE_FORMAT_VERSION, "kind": "sequence",
                "key": "cvvdp", "request": identity, "provenance": {}}
    np.savez(directory / "cvvdp_damaged.npz", metadata=np.array(json.dumps(metadata)), score=np.array(9.0))
    assert result_cache.other_cvvdp_scores(source, distorted, request)[1] == []


def _vmaf_run(source, distorted, size, model, score):
    """A result whose VMAF was calculated with `model`, at `size`."""
    from dataclasses import replace as _replace

    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet

    result = _fake_result(source, distorted)
    info = _replace(result.distorted_info, width=size[0], height=size[1])
    result.source_info, result.distorted_info = _replace(info, path=source), info
    provenance = MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1", {"model": model})
    result.merge_metric_results(MetricResultSet([FrameMetricResult("vmaf", [0], [0.0], [score], provenance)]))
    return result


def _vmaf_request(choice):
    return analysis_request_from_vmaf_options(VmafOptions(model_choice=choice), ("vmaf",))


def _vmaf_found(source, distorted, choice):
    loaded = result_cache.load_cached(source, distorted, _vmaf_request(choice))
    metric = loaded[0].frame_metric("vmaf") if loaded else None
    return None if metric is None else round(float(metric.values[0]), 2)


AUTO, V061, V4K = "__auto__", "version=vmaf_v0.6.1", "version=vmaf_4k_v0.6.1"


@pytest.mark.parametrize(("size", "found"), [((3840, 2160), 94.06), ((1920, 1080), None)])
def test_auto_finds_a_score_saved_with_the_model_it_picks_chosen_by_name(tmp_path, size, found):
    """A video scored with "VMAF 4K v0.6.1" chosen showed no VMAF on a row
    set to Auto, though Auto picks that model for a 4K comparison. It must
    still not be shown where Auto picks the other model."""
    source = _make_file(tmp_path / "source.mkv", 1000)
    distorted = _make_file(tmp_path / "test.mkv", 500)
    result_cache.store(source, distorted, _vmaf_run(source, distorted, size, V4K, 94.06), "run", _vmaf_request(V4K))
    assert _vmaf_found(source, distorted, AUTO) == found
    assert _vmaf_found(source, distorted, V4K) == 94.06


def test_a_score_saved_on_auto_is_found_by_its_model_chosen_by_name_and_no_other(tmp_path):
    source = _make_file(tmp_path / "source.mkv", 1000)
    distorted = _make_file(tmp_path / "test.mkv", 500)
    result_cache.store(source, distorted, _vmaf_run(source, distorted, (3840, 2160), V4K, 96.21), "run",
                       _vmaf_request(AUTO))
    assert _vmaf_found(source, distorted, AUTO) == 96.21
    assert _vmaf_found(source, distorted, V4K) == 96.21
    assert _vmaf_found(source, distorted, V061) is None


def test_old_keys_with_a_leftover_model_are_found_but_never_trusted_for_another_model(tmp_path):
    """Scores were keyed by the row's leftover model field: an Auto row saved
    4K-model scores as "vmaf_v0.6.1". The same choice finds them again; an
    explicit "VMAF v0.6.1" row must not take them for its own."""
    from vmaf_app.core import metric_cache
    from vmaf_app.core.analysis_request import MetricRequestSpec
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance

    source = _make_file(tmp_path / "source.mkv", 1000)
    distorted = _make_file(tmp_path / "test.mkv", 500)
    result_cache.store(source, distorted, _vmaf_run(source, distorted, (3840, 2160), V4K, 1.0), "run",
                       _vmaf_request(AUTO))  # writes the context
    request = _vmaf_request(AUTO)
    spec = request.metrics[0]
    directory = metric_cache.recipe_directory(result_cache.cache_dir(), source, distorted, request.recipe)
    metric_cache.metric_path(directory, spec).unlink()  # the context stays, as in a real cache
    old = MetricRequestSpec(spec.key, spec.backend_id,
                            (("model", V061), ("model_choice", AUTO), ("custom_model", "")),
                            spec.coverage, spec.implementation_compatibility_id)
    legacy = MetricProvenance("legacy", "", "unknown", "legacy-v1", {})
    metric_cache.store_metric(directory, FrameMetricResult("vmaf", [0], [0.0], [94.06], legacy), old)
    assert _vmaf_found(source, distorted, AUTO) == 94.06
    assert _vmaf_found(source, distorted, V061) is None
    assert _vmaf_found(source, distorted, V4K) is None  # its model is not certain


def test_recalculating_vmaf_clears_the_equivalent_scores_too(tmp_path):
    source = _make_file(tmp_path / "source.mkv", 1000)
    distorted = _make_file(tmp_path / "test.mkv", 500)
    result_cache.store(source, distorted, _vmaf_run(source, distorted, (3840, 2160), V4K, 94.06), "run", _vmaf_request(V4K))
    result_cache.clear(source, distorted, _vmaf_request(AUTO))
    assert _vmaf_found(source, distorted, V4K) is None

