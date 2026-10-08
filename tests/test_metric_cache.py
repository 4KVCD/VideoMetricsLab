"""Direct v2 metric-cache and generic-result regression coverage."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from vmaf_app.core import metric_cache, result_cache
from vmaf_app.core.analysis_request import (
    AnalysisRequest,
    FrameCoverage,
    MetricRequestSpec,
)
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    comparison_recipe_from_vmaf_options,
    displayable_metric_specs,
    metric_request_specs,
)
from vmaf_app.core.metric_cache import (
    VSHIP_COLOR_TAGS,
    clear_metrics,
    load_metric,
    load_metrics,
    metric_path,
    recipe_directory,
    store_metric,
)
from vmaf_app.core.metric_results import (
    UNSPECIFIED_PROVENANCE,
    FrameMetricResult,
    MetricProvenance,
    MetricResultSet,
    SequenceMetricResult,
    frame_scores_from_results,
)
from vmaf_app.core.models import ComparisonResult, CropMode, FrameScores, GpuVendor, VideoInfo, VmafOptions


def _request(options: VmafOptions) -> AnalysisRequest:
    return analysis_request_from_vmaf_options(options)


def _load_cached(source, distorted, options, directory=None):
    return result_cache.load_cached(
        source, distorted, _request(options), directory,
        displayable_metric_specs(options),
    )


def _store_cached(source, distorted, result, label, options, directory=None):
    return result_cache.store(source, distorted, result, label, _request(options), directory)


def _clear_cached(source, distorted, options, directory=None):
    # The ticked metrics only, as the window clears them for a recalculation.
    return result_cache.clear(source, distorted, _request(options), directory)

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
    cached_provenance = MetricProvenance(
        PROVENANCE.implementation, "", PROVENANCE.compute_backend,
        PROVENANCE.implementation_compatibility_id, PROVENANCE.parameters,
    )
    assert frame.provenance == cached_provenance
    assert isinstance(sequence, SequenceMetricResult) and np.isneginf(sequence.score)
    assert sequence.provenance == cached_provenance


def _xpsnr(compatibility: str) -> FrameMetricResult:
    return FrameMetricResult("xpsnr", [0, 1, 2], [0.0, 1 / 24, 2 / 24], [40.0, 41.0, 42.0],
                             MetricProvenance("ffmpeg/xpsnr", "ffmpeg 9.0", "cpu", compatibility))


def test_a_score_answers_a_request_only_from_a_compatible_implementation_and_not_stale():
    options = VmafOptions(compute_xpsnr=True)
    specs = {spec.key: spec for spec in metric_request_specs(
        options, ("vmaf", "xpsnr", "ssimulacra2", "cvvdp"))}
    # XPSNR weighted by the encode (v1.4) answers no request of v1.5's.
    assert specs["xpsnr"].implementation_compatibility_id == "ffmpeg-xpsnr-v2"
    assert metric_cache.answers(specs["xpsnr"], _xpsnr("ffmpeg-xpsnr-v2"))
    assert not metric_cache.answers(specs["xpsnr"], _xpsnr("ffmpeg-xpsnr-v1"))
    # Saved before scores had provenance: libvmaf's, unchanged since, answer; XPSNR's do not.
    unversioned = FrameMetricResult("vmaf", [0], [0.0], [90.0], UNSPECIFIED_PROVENANCE)
    assert metric_cache.answers(specs["vmaf"], unversioned)
    assert not metric_cache.answers(specs["xpsnr"], _xpsnr("unversioned"))
    # A CPU SSIMULACRA2 from before the CPU tools read colours as Vship does.
    cpu = MetricProvenance("ssimulacra2", "", "cpu", "ssimulacra2-libjxl-cpu-v1")
    assert not metric_cache.answers(specs["ssimulacra2"], FrameMetricResult("ssimulacra2", [0], [0.0], [50.0], cpu))
    current = replace(cpu, parameters={"color_tags": metric_cache.CPU_COLOR_TAGS})
    assert metric_cache.answers(specs["ssimulacra2"], FrameMetricResult("ssimulacra2", [0], [0.0], [50.0], current))
    vship = MetricProvenance("Vship/cvvdp", "Vship 5.1.2", "gpu", "cvvdp-vship-gpu-v1",
                             {"color_tags": VSHIP_COLOR_TAGS})
    assert metric_cache.answers(specs["cvvdp"], SequenceMetricResult("cvvdp", 9.1, vship))


def test_a_score_is_kept_when_it_has_a_value_made_where_its_metric_is_set_and_answering_its_request():
    """metric_cache.reusable: the window's Run button and the command line
    keep a score by this one rule."""
    specs = metric_request_specs(VmafOptions(compute_xpsnr=True), ("vmaf", "xpsnr", "ssimulacra2", "cvvdp"))
    gpu = MetricProvenance("Vship/ssimulacra2", "Vship 5.1.2", "gpu", "ssimulacra2-vship-gpu-v1",
                           {"color_tags": VSHIP_COLOR_TAGS})
    cpu = MetricProvenance("ssimulacra2", "", "cpu", "ssimulacra2-libjxl-cpu-v1",
                           {"color_tags": metric_cache.CPU_COLOR_TAGS})
    cvvdp = MetricProvenance("Vship/cvvdp", "Vship 5.1.2", "gpu", "cvvdp-vship-gpu-v1", {"color_tags": VSHIP_COLOR_TAGS})

    def kept(*metrics, backends=None, gpu_can_score=lambda key: True):
        result = _run(Path("s.mkv"), Path("t.mkv"))
        result.merge_metric_results(MetricResultSet(metrics))
        return set(metric_cache.reusable(result, specs, backends or {}, gpu_can_score).keys())

    assert kept() == {"vmaf"}  # _run's VMAF; nothing else saved
    assert kept(_xpsnr("ffmpeg-xpsnr-v2")) == {"vmaf", "xpsnr"}
    assert kept(_xpsnr("ffmpeg-xpsnr-v1")) == {"vmaf"}  # another implementation's
    nan = FrameMetricResult("ssimulacra2", [0, 1], [0.0, 1 / 24], [np.nan, np.nan], gpu)
    assert "ssimulacra2" not in kept(nan)  # no score at all
    assert "cvvdp" not in kept(SequenceMetricResult("cvvdp", float("nan"), cvvdp))
    assert "cvvdp" in kept(SequenceMetricResult("cvvdp", 9.1, cvvdp))
    on_gpu = FrameMetricResult("ssimulacra2", [0], [0.0], [80.0], gpu)
    on_cpu = FrameMetricResult("ssimulacra2", [0], [0.0], [80.0], cpu)
    assert "ssimulacra2" not in kept(on_gpu, backends={"ssimulacra2": "cpu"})
    assert "ssimulacra2" in kept(on_cpu, backends={"ssimulacra2": "cpu"})
    assert "ssimulacra2" in kept(on_gpu, backends={"ssimulacra2": "gpu"})
    # A CPU score for a GPU choice: only where the GPU cannot calculate it.
    assert "ssimulacra2" not in kept(on_cpu, backends={"ssimulacra2": "gpu"})
    assert "ssimulacra2" in kept(on_cpu, backends={"ssimulacra2": "gpu"}, gpu_can_score=lambda key: False)


def test_cache_omits_library_versions_except_for_vmaf(tmp_path):
    source, test = _paths(tmp_path)
    options = VmafOptions(model_choice="version=vmaf_v0.6.1")
    directory = recipe_directory(
        tmp_path, source, test, comparison_recipe_from_vmaf_options(options)
    )
    specs = metric_request_specs(options, ("vmaf", "vmaf_neg", "psnr"))
    specs_by_key = {spec.key: spec for spec in specs}
    for key, score in (("vmaf", 97.0), ("vmaf_neg", 96.0), ("psnr", 42.0)):
        store_metric(directory, FrameMetricResult(
            key, [0], [0.0], [score],
            MetricProvenance("FFmpeg/libvmaf", "FFmpeg 9.0", "cpu", "ffmpeg-libvmaf-v1"),
        ), specs_by_key[key])

    metadata = {}
    for key, spec in specs_by_key.items():
        with np.load(metric_path(directory, spec), allow_pickle=False) as data:
            metadata[key] = json.loads(str(data["metadata"].item()))

    assert metadata["vmaf"]["provenance"]["implementation_version"] == "FFmpeg 9.0"
    assert metadata["vmaf_neg"]["provenance"]["implementation_version"] == ""
    assert metadata["psnr"]["provenance"]["implementation_version"] == ""
    assert dict(metadata["vmaf"]["request"]["parameters"])["model"] == "version=vmaf_v0.6.1"


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
    # The scaling algorithm is not what the comparison is (on the CPU or the
    # GPU, any algorithm): scores saved with one are found with another.
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions(scale_algorithm="lanczos"))) == baseline
    assert recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions(duration_limit=2.0))) != baseline
    source.write_bytes(b"source changed")
    assert recipe_directory(tmp_path, source, test, recipe) != baseline


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
    cpu_id = "ssimulacra2-libjxl-cpu-v1"
    gpu_id = "ssimulacra2-vship-gpu-v1"
    sampled = MetricRequestSpec(
        spec.key, spec.backend_id, spec.parameters, FrameCoverage("sampled", 2), spec.implementation_compatibility_id,
    )
    cpu_provenance = MetricProvenance("SSIMULACRA2", "libjxl 0.12.0", "cpu", cpu_id)
    gpu_provenance = MetricProvenance("Vship/SSIMULACRA2", "Vship 5.1.1", "gpu", gpu_id)
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


def _stored_and_reloaded(tmp_path, options, metrics, compared_frame_count):
    """Store `metrics` as one run of a 24 fps comparison, then load it back."""
    source, test = _paths(tmp_path)
    info = VideoInfo(test, 64, 48, 24.0, 1.0, 24, "hevc")
    result = ComparisonResult(
        source=source, distorted=test, frames=FrameScores.empty(), fps=24.0, model="",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
        compared_frame_count=compared_frame_count, metric_results=MetricResultSet(metrics),
    )
    request = analysis_request_from_vmaf_options(options, tuple(m.key for m in metrics))
    result_cache.store(source, test, result, "run", request, tmp_path)
    loaded = result_cache.load_cached(source, test, request, tmp_path)
    return None if loaded is None else loaded[0]


def test_subsampled_perceptual_scores_survive_a_reload(tmp_path):
    """24 frames scored every fourth frame are six scores, not 24."""
    frames = list(range(0, 24, 4))
    gpu = MetricProvenance("Vship/ssimulacra2", "5.1.1", "gpu", "ssimulacra2-vship-gpu-v1")
    metric = FrameMetricResult("ssimulacra2", frames, [f / 24 for f in frames], [80.0] * 6, gpu)

    loaded = _stored_and_reloaded(tmp_path, VmafOptions(n_subsample=4), [metric], 24)

    assert loaded is not None and loaded.has_metric("ssimulacra2")
    assert list(loaded.metric("ssimulacra2").frame) == frames


def test_metrics_on_the_same_frames_share_the_frame_view_despite_rounded_times():
    """Times are frame / fps, computed from different frame rates by
    different backends, so the same frames can carry times a rounding error
    apart. The shared view matched times exactly and so dropped every metric
    but VMAF from a real cached film (differences up to 3.3e-7 s)."""
    frames = np.arange(4, dtype=np.int32)
    vmaf = FrameMetricResult("vmaf", frames, frames / 23.976023976023978, [90, 91, 92, 93], PROVENANCE)
    psnr = FrameMetricResult("psnr", frames, frames / 23.976023976023978 + 3.3e-7, [40, 41, 42, 43], PROVENANCE)
    ssimulacra2 = FrameMetricResult("ssimulacra2", frames, frames / 23.976, [70, 71, 72, 73], PROVENANCE)

    frame_view = frame_scores_from_results(MetricResultSet([vmaf, psnr, ssimulacra2]))

    assert frame_view.psnr.tolist() == [40, 41, 42, 43]
    assert frame_view.values("ssimulacra2").tolist() == [70, 71, 72, 73]
    np.testing.assert_array_equal(frame_view.time, vmaf.time)
