"""Direct v2 metric-cache and generic-result regression coverage."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from vmaf_app.core import result_cache
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.execution import (
    ExecutionPreferences,
    FrameCoverage,
    MetricRequestSpec,
    build_execution_plan,
    metric_request_specs,
)
from vmaf_app.core.metric_cache import (
    clear_recipe,
    load_metric,
    load_metrics,
    metric_path,
    recipe_directory,
    store_metric,
)
from vmaf_app.core.metric_results import (
    FrameMetricResult,
    MetricProvenance,
    MetricResultSet,
    SequenceMetricResult,
    frame_scores_from_results,
    merge_metric_results,
)
from vmaf_app.core.models import CropMode, FrameScores, GpuVendor, VideoInfo, VmafOptions, VmafRunResult

PROVENANCE = MetricProvenance("test", "1.0", "cpu", "test-v1", {"window": 7})


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    source, test = tmp_path / "source.mkv", tmp_path / "test.mkv"
    source.write_bytes(b"source")
    test.write_bytes(b"test")
    return source, test


def _spec(key="test_frame_metric", *, step=1, compatibility="test-v1"):
    return MetricRequestSpec(key, (("setting", "value"),), FrameCoverage("full" if step == 1 else "sampled", step), compatibility)


def _frame(key="test_frame_metric"):
    return FrameMetricResult(key, [0, 4, 8], [0.0, 1 / 6, 1 / 3], [1.0, np.nan, np.inf], PROVENANCE)


def _run(source: Path, test: Path) -> VmafRunResult:
    info = VideoInfo(source, 16, 16, 24.0, 0.1, 3, "h264")
    test_info = VideoInfo(test, 16, 16, 24.0, 0.1, 3, "h264")
    return VmafRunResult(
        source, test, FrameScores([0, 1, 2], [0.0, 1 / 24, 2 / 24], [90, 91, 92]), 24.0,
        "version=vmaf_v0.6.1", None, None, info, test_info,
    )


def test_frame_and_sequence_metric_round_trip_with_special_values(tmp_path):
    source, test = _paths(tmp_path)
    recipe = ComparisonRecipe.from_vmaf_options(VmafOptions())
    directory = recipe_directory(tmp_path, source, test, recipe)
    frame_spec, sequence_spec = _spec(), _spec("test_sequence_metric")
    store_metric(directory, _frame(), frame_spec)
    store_metric(directory, SequenceMetricResult("test_sequence_metric", float("-inf"), PROVENANCE), sequence_spec)

    frame = load_metric(directory, frame_spec)
    sequence = load_metric(directory, sequence_spec)
    assert isinstance(frame, FrameMetricResult)
    assert frame.values.dtype == np.float32 and np.isnan(frame.values[1]) and np.isposinf(frame.values[2])
    assert frame.provenance == PROVENANCE
    assert isinstance(sequence, SequenceMetricResult) and np.isneginf(sequence.score)
    assert sequence.provenance == PROVENANCE


def test_direct_lookup_keeps_other_metrics_when_one_artifact_is_corrupt(tmp_path):
    source, test = _paths(tmp_path)
    directory = recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(VmafOptions()))
    first, second = _spec("test_frame_metric"), _spec("other_frame_metric")
    store_metric(directory, _frame(), first)
    store_metric(directory, _frame("other_frame_metric"), second)
    metric_path(directory, first).write_bytes(b"not an npz")
    found = load_metrics(directory, (first, second))
    assert not found.has("test_frame_metric")
    assert found.has("other_frame_metric")


def test_recipe_and_metric_identity_keep_scientific_choices_separate(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions(n_threads=1, gpu_decode=True, gpu_vendor=GpuVendor.NVIDIA)
    recipe = ComparisonRecipe.from_vmaf_options(options)
    baseline = recipe_directory(tmp_path, source, test, recipe)
    changed_execution = VmafOptions(n_threads=12, gpu_decode=False, gpu_vendor=GpuVendor.INTEL)
    assert recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(changed_execution)) == baseline
    assert recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(VmafOptions(crop_mode=CropMode.NONE))) != baseline
    assert recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(VmafOptions(scale_algorithm="lanczos"))) != baseline
    assert recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(VmafOptions(duration_limit=2.0))) != baseline
    source.write_bytes(b"source changed")
    assert recipe_directory(tmp_path, source, test, recipe) != baseline


def test_coverage_and_compatibility_id_produce_independent_direct_entries(tmp_path):
    source, test = _paths(tmp_path)
    directory = recipe_directory(tmp_path, source, test, ComparisonRecipe.from_vmaf_options(VmafOptions()))
    full, sampled = _spec("xpsnr", step=1), _spec("xpsnr", step=3)
    other_impl = _spec("xpsnr", compatibility="other-v1")
    store_metric(directory, _frame("xpsnr"), full)
    assert load_metric(directory, sampled) is None
    assert load_metric(directory, other_impl) is None
    assert metric_request_specs(VmafOptions(model="version=vmaf_v0.6.1"))[0] != metric_request_specs(VmafOptions(model="version=vmaf_4k_v0.6.1"))[0]


def test_clear_recipe_is_scoped(tmp_path):
    source, first = _paths(tmp_path)
    second = tmp_path / "second.mkv"
    second.write_bytes(b"second")
    recipe = ComparisonRecipe.from_vmaf_options(VmafOptions())
    one, two = recipe_directory(tmp_path, source, first, recipe), recipe_directory(tmp_path, source, second, recipe)
    store_metric(one, _frame(), _spec())
    store_metric(two, _frame(), _spec())
    assert clear_recipe(tmp_path, source, first, recipe) > 0
    assert not one.exists() and two.exists()


def test_generic_results_can_have_independent_axes_without_corrupting_legacy_view():
    vmaf = FrameMetricResult("vmaf", [0, 2], [0.0, 0.1], [90, 91], PROVENANCE)
    arbitrary = FrameMetricResult("test_frame_metric", [0, 5], [0.0, 0.25], [1, 2], PROVENANCE)
    results = MetricResultSet([vmaf, arbitrary, SequenceMetricResult("test_sequence_metric", 8.75, PROVENANCE)])
    assert results.frame("test_frame_metric").frame.tolist() == [0, 5]
    assert results.sequence("test_sequence_metric").score == 8.75
    legacy = frame_scores_from_results(results)
    assert legacy.vmaf.tolist() == [90, 91]
    assert not legacy.has("test_frame_metric")
    merged = merge_metric_results(MetricResultSet([vmaf]), MetricResultSet([arbitrary]))
    assert merged.has("vmaf") and merged.has("test_frame_metric")


def test_planning_retains_grouping_and_special_xpsnr_coverage():
    mixed = VmafOptions(compute_xpsnr=True, n_subsample=3)
    full_xpsnr = VmafOptions(compute_vmaf=False, compute_xpsnr=True, n_subsample=3)
    assert len(build_execution_plan(mixed).tasks) == 1
    assert len(build_execution_plan(mixed, MetricResultSet([_frame("vmaf")])).tasks) == 1
    assert build_execution_plan(mixed, MetricResultSet([_frame("vmaf"), _frame("xpsnr")])).tasks == ()
    assert next(spec for spec in metric_request_specs(mixed) if spec.key == "xpsnr").coverage == FrameCoverage("sampled", 3)
    assert metric_request_specs(full_xpsnr)[0].coverage == FrameCoverage("full", 1)
    assert ExecutionPreferences.from_vmaf_options(mixed) == ExecutionPreferences.from_vmaf_options(full_xpsnr)


def test_facade_writes_v2_and_lazily_promotes_a_legacy_hit(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions(gpu_decode=False)
    result_cache.store(source, test, _run(source, test), "label", options, tmp_path)
    v2_root = tmp_path / "v2"
    assert list(v2_root.rglob("vmaf_*.npz"))
    legacy = list(tmp_path.glob("*.metrics.json"))
    assert len(legacy) == 1
    shutil.rmtree(v2_root)
    loaded = result_cache.load_cached(source, test, options, tmp_path)
    assert loaded is not None and loaded[1] == "label"
    assert legacy[0].exists(), "promotion must not remove the legacy cache"
    assert list(v2_root.rglob("vmaf_*.npz"))
