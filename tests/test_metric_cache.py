"""Direct v2 metric-cache and generic-result regression coverage."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vmaf_app.core import result_cache
from vmaf_app.core.analysis_request import (
    AnalysisRequest,
    ExecutionPreferences,
    FrameCoverage,
    MetricRequestSpec,
)
from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    comparison_recipe_from_vmaf_options,
    metric_request_specs,
    supplemental_metric_specs,
)
from vmaf_app.core.metric_cache import (
    clear_metrics,
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
from vmaf_app.core.models import ComparisonResult, CropMode, FrameScores, GpuVendor, VideoInfo, VmafOptions


def _request(options: VmafOptions) -> AnalysisRequest:
    return analysis_request_from_vmaf_options(options)


def _load_cached(source, distorted, options, directory=None):
    return result_cache.load_cached(
        source, distorted, _request(options), directory,
        supplemental_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(source, distorted, result, label, _request(options), directory)


def _clear_cached(source, distorted, options, directory=None):
    return result_cache.clear(
        source, distorted, _request(options), directory, supplemental_metric_specs(options)
    )

PROVENANCE = MetricProvenance("test", "1.0", "cpu", "test-v1", {"window": 7})


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    source, test = tmp_path / "source.mkv", tmp_path / "test.mkv"
    source.write_bytes(b"source")
    test.write_bytes(b"test")
    return source, test


def _spec(key="test_frame_metric", *, step=1, compatibility="test-v1", backend="test"):
    return MetricRequestSpec(key, backend, (("setting", "value"),), FrameCoverage("full" if step == 1 else "sampled", step), compatibility)


def _frame(key="test_frame_metric"):
    return FrameMetricResult(key, [0, 4, 8], [0.0, 1 / 6, 1 / 3], [1.0, np.nan, np.inf], PROVENANCE)


def _run(source: Path, test: Path) -> ComparisonResult:
    info = VideoInfo(source, 16, 16, 24.0, 0.1, 3, "h264")
    test_info = VideoInfo(test, 16, 16, 24.0, 0.1, 3, "h264")
    return ComparisonResult(
        source, test, FrameScores([0, 1, 2], [0.0, 1 / 24, 2 / 24], [90, 91, 92]), 24.0,
        "version=vmaf_v0.6.1", None, None, info, test_info,
    )


def test_frame_and_sequence_metric_round_trip_with_special_values(tmp_path):
    source, test = _paths(tmp_path)
    recipe = comparison_recipe_from_vmaf_options(VmafOptions())
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
    directory = recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions()))
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
    recipe = comparison_recipe_from_vmaf_options(options)
    baseline = recipe_directory(tmp_path, source, test, recipe)
    changed_execution = VmafOptions(n_threads=12, gpu_decode=False, gpu_vendor=GpuVendor.INTEL)
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(changed_execution)) == baseline
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions(crop_mode=CropMode.NONE))) != baseline
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions(scale_algorithm="lanczos"))) != baseline
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions(duration_limit=2.0))) != baseline
    source.write_bytes(b"source changed")
    assert recipe_directory(tmp_path, source, test, recipe) != baseline


def test_coverage_and_compatibility_id_produce_independent_direct_entries(tmp_path):
    source, test = _paths(tmp_path)
    directory = recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions()))
    full, sampled = _spec("xpsnr", step=1), _spec("xpsnr", step=3)
    other_impl = _spec("xpsnr", compatibility="other-v1")
    store_metric(directory, _frame("xpsnr"), full)
    assert load_metric(directory, sampled) is None
    assert load_metric(directory, other_impl) is None
    assert metric_request_specs(VmafOptions(model="version=vmaf_v0.6.1"))[0] != metric_request_specs(VmafOptions(model="version=vmaf_4k_v0.6.1"))[0]
    neg_a = metric_request_specs(VmafOptions(compute_vmaf=False, compute_vmaf_neg=True, model="version=vmaf_v0.6.1"))[0]
    neg_b = metric_request_specs(VmafOptions(compute_vmaf=False, compute_vmaf_neg=True, model="version=vmaf_4k_v0.6.1"))[0]
    assert neg_a == neg_b, "standard VMAF model choice must not invalidate fixed-model NEG"


def test_clear_recipe_is_scoped(tmp_path):
    source, first = _paths(tmp_path)
    second = tmp_path / "second.mkv"
    second.write_bytes(b"second")
    recipe = comparison_recipe_from_vmaf_options(VmafOptions())
    one, two = recipe_directory(tmp_path, source, first, recipe), recipe_directory(tmp_path, source, second, recipe)
    store_metric(one, _frame(), _spec())
    store_metric(two, _frame(), _spec())
    assert clear_recipe(tmp_path, source, first, recipe) > 0
    assert not one.exists() and two.exists()


def test_generic_results_can_have_independent_axes_without_corrupting_shared_frame_view():
    vmaf = FrameMetricResult("vmaf", [0, 2], [0.0, 0.1], [90, 91], PROVENANCE)
    arbitrary = FrameMetricResult("test_frame_metric", [0, 5], [0.0, 0.25], [1, 2], PROVENANCE)
    results = MetricResultSet([vmaf, arbitrary, SequenceMetricResult("test_sequence_metric", 8.75, PROVENANCE)])
    assert results.frame("test_frame_metric").frame.tolist() == [0, 5]
    assert results.sequence("test_sequence_metric").score == 8.75
    frame_view = frame_scores_from_results(results)
    assert frame_view.vmaf.tolist() == [90, 91]
    assert not frame_view.has("test_frame_metric")
    merged = merge_metric_results(MetricResultSet([vmaf]), MetricResultSet([arbitrary]))
    assert merged.has("vmaf") and merged.has("test_frame_metric")


def test_independent_axis_registered_metric_does_not_blank_the_shared_view():
    vmaf = FrameMetricResult("vmaf", [0, 2], [0.0, 0.1], [90, 91], PROVENANCE)
    psnr = FrameMetricResult("psnr", [0, 5], [0.0, 0.25], [40, 41], PROVENANCE)
    results = MetricResultSet([vmaf, psnr])

    frame_view = frame_scores_from_results(results)

    assert frame_view.vmaf.tolist() == [90, 91]
    assert frame_view.psnr is None
    assert results.frame("psnr").frame.tolist() == [0, 5]


def test_planner_groups_arbitrary_metric_backends_without_vmaf_options():
    recipe = comparison_recipe_from_vmaf_options(VmafOptions())
    request = AnalysisRequest(
        recipe=recipe,
        metrics=(
            _spec("metric_a", backend="cpu"),
            _spec("metric_b", backend="gpu"),
            _spec("metric_c", backend="cpu"),
        ),
        execution=ExecutionPreferences(True, GpuVendor.AUTO, 0),
    )

    plan = build_execution_plan(request)

    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("cpu", ("metric_a", "metric_c")),
        ("gpu", ("metric_b",)),
    ]

    cached = MetricResultSet([_frame("metric_a"), _frame("metric_c")])
    plan = build_execution_plan(request, cached)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("gpu", ("metric_b",)),
    ]


def test_backend_routing_is_not_part_of_metric_cache_identity():
    cpu = _spec("same", backend="cpu")
    gpu = _spec("same", backend="gpu")

    assert cpu.identity_dict() == gpu.identity_dict()
    assert metric_path(Path("cache"), cpu) == metric_path(Path("cache"), gpu)


def test_auto_perceptual_cache_keeps_gpu_and_cpu_scores_separate(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions()
    recipe = comparison_recipe_from_vmaf_options(options)
    directory = recipe_directory(tmp_path, source, test, recipe)
    spec = next(spec for spec in metric_request_specs(options, ("ssimulacra2",)) if spec.key == "ssimulacra2")
    cpu_id = "ssimulacra2-libjxl-0.12.0-cpu-v1"
    gpu_id = "ssimulacra2-vship-4.0.2-gpu-v1"
    sampled = MetricRequestSpec(
        spec.key, spec.backend_id, spec.parameters, FrameCoverage("sampled", 2), spec.implementation_compatibility_id,
    )
    cpu_provenance = MetricProvenance("SSIMULACRA2", "libjxl 0.12.0", "cpu", cpu_id)
    gpu_provenance = MetricProvenance("Vship/SSIMULACRA2", "Vship 4.0.2", "gpu", gpu_id)
    store_metric(directory, FrameMetricResult(spec.key, [0], [0.0], [82.0], cpu_provenance), spec)
    assert load_metric(directory, spec).provenance.compute_backend == "cpu"
    store_metric(directory, FrameMetricResult(spec.key, [0], [0.0], [91.0], gpu_provenance), spec)
    store_metric(directory, FrameMetricResult(spec.key, [0], [0.0], [80.0], cpu_provenance), sampled)

    loaded = load_metric(directory, spec)
    assert loaded.values.tolist() == [91.0]
    assert loaded.provenance.compute_backend == "gpu"
    assert len(list(directory.glob("ssimulacra2_*.npz"))) == 3

    assert clear_metrics(tmp_path, source, test, recipe, (spec,)) == 2
    assert len(list(directory.glob("ssimulacra2_*.npz"))) == 1
    assert load_metric(directory, sampled).values.tolist() == [80.0]


def test_planning_retains_grouping_and_special_xpsnr_coverage():
    mixed = VmafOptions(compute_xpsnr=True, n_subsample=3)
    full_xpsnr = VmafOptions(compute_vmaf=False, compute_xpsnr=True, n_subsample=3)
    assert len(build_execution_plan(analysis_request_from_vmaf_options(mixed)).tasks) == 1
    assert len(build_execution_plan(analysis_request_from_vmaf_options(mixed), MetricResultSet([_frame("vmaf")])).tasks) == 1
    assert build_execution_plan(analysis_request_from_vmaf_options(mixed), MetricResultSet([_frame("vmaf"), _frame("xpsnr")])).tasks == ()
    assert next(spec for spec in metric_request_specs(mixed) if spec.key == "xpsnr").coverage == FrameCoverage("sampled", 3)
    assert metric_request_specs(full_xpsnr)[0].coverage == FrameCoverage("full", 1)
    assert analysis_request_from_vmaf_options(mixed).execution == analysis_request_from_vmaf_options(full_xpsnr).execution


def test_facade_uses_only_the_per_metric_cache(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions(gpu_decode=False)
    _store_cached(source, test, _run(source, test), "label", options, tmp_path)

    assert list((tmp_path / "v2").rglob("vmaf_*.npz"))
    assert not list(tmp_path.glob("*.metrics.json"))
    loaded = _load_cached(source, test, options, tmp_path)
    assert loaded is not None and loaded[1] == "label"


def test_facade_returns_partial_v2_results_for_a_larger_request(tmp_path):
    source, test = _paths(tmp_path)
    _store_cached(source, test, _run(source, test), "vmaf only", VmafOptions(), tmp_path)

    requested = VmafOptions(extra_features=["name=psnr"])
    loaded = _load_cached(source, test, requested, tmp_path)

    assert loaded is not None and loaded[1] == "vmaf only"
    assert loaded[0].frames.has("vmaf")
    assert not loaded[0].frames.has("psnr")


def test_cache_summary_counts_comparisons_not_metric_artifacts(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions(extra_features=["name=psnr"], compute_xpsnr=True)
    run = _run(source, test)
    run.frames = FrameScores(
        run.frames.frame, run.frames.time, run.frames.vmaf,
        psnr=[40, 41, 42], xpsnr=[38, 39, 40],
    )
    run.metric_results = MetricResultSet()
    run.__post_init__()
    _store_cached(source, test, run, "multi", options, tmp_path)

    count, size_bytes = result_cache.cache_summary(tmp_path)
    assert count == 1
    assert size_bytes > 0
