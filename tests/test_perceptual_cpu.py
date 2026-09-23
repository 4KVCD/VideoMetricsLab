from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core.analysis_request import AnalysisRequest, ExecutionPreferences, FrameCoverage, MetricRequestSpec
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.execution import build_execution_plan
from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
from vmaf_app.core.metric_cache import load_metric, load_metrics, store_metric
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.metrics import METRIC_BY_KEY, MetricDirection, MetricKind
from vmaf_app.core.models import CropMode, GpuVendor, ScaleDirection, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualRunError, parse_score, run_perceptual_task


def _info(path: str) -> VideoInfo:
    return VideoInfo(Path(path), 64, 48, 24.0, 1.0, 24, "h264")


def _request(*keys: str) -> AnalysisRequest:
    return AnalysisRequest(
        recipe=ComparisonRecipe(CropMode.NONE, None, None, "bicubic", ScaleDirection.SOURCE_TO_DISTORTED, 0.0, None),
        metrics=tuple(MetricRequestSpec(key, "perceptual", (), FrameCoverage("full"), f"{key}-reference-cli-v1") for key in keys),
        execution=ExecutionPreferences(False, GpuVendor.NONE, 1),
    )


def test_perceptual_metrics_are_registered_without_ffmpeg_bindings():
    assert METRIC_BY_KEY["ssimulacra2"].kind is MetricKind.FRAME
    assert METRIC_BY_KEY["ssimulacra2"].direction is MetricDirection.HIGHER_IS_BETTER
    assert METRIC_BY_KEY["butteraugli"].direction is MetricDirection.LOWER_IS_BETTER
    assert METRIC_BY_KEY["ssimulacra2"].ffmpeg_binding is None
    assert METRIC_BY_KEY["butteraugli"].ffmpeg_binding is None


def test_mixed_request_groups_perceptual_metrics_separately():
    options = VmafOptions(extra_features=["name=psnr"], compute_vmaf=True)
    request = analysis_request_from_vmaf_options(options, ("vmaf", "psnr", "ssimulacra2", "butteraugli"))
    plan = build_execution_plan(request)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("ffmpeg", ("vmaf", "psnr")),
        ("perceptual", ("ssimulacra2", "butteraugli")),
    ]


@pytest.mark.parametrize(("text", "expected"), [
    ("SSIMULACRA2: 91.75", 91.75),
    ("butteraugli score = 0.2345", 0.2345),
])
def test_perceptual_output_parser_uses_final_scalar(text, expected):
    assert parse_score("ssimulacra2", text) == expected


def test_perceptual_output_parser_rejects_malformed_output():
    with pytest.raises(PerceptualRunError, match="numeric score"):
        parse_score("butteraugli", "comparison failed")


def test_backend_returns_independent_frame_results_without_real_tools(tmp_path, monkeypatch):
    request = _request("ssimulacra2", "butteraugli")
    references = [tmp_path / f"r-{i}.png" for i in range(3)]
    tests = [tmp_path / f"t-{i}.png" for i in range(3)]
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.find_metric_executable", lambda key: key)
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu._extract_png_pairs", lambda *args: (references, tests))
    monkeypatch.setattr("vmaf_app.core.perceptual_cpu._tool_version", lambda executable: "test-tool 1")
    monkeypatch.setattr(
        "vmaf_app.core.perceptual_cpu._run_metric",
        lambda executable, key, reference, test: 90.0 if key == "ssimulacra2" else 0.25,
    )
    output = run_perceptual_task(_info("source.mp4"), _info("test.mp4"), request, request.metrics)
    assert output.metrics.keys() == ("ssimulacra2", "butteraugli")
    assert np.array_equal(output.metrics.frame("ssimulacra2").frame, np.array([0, 1, 2]))
    assert output.metrics.frame("butteraugli").aggregate == pytest.approx(0.25)
    assert output.metrics.frame("ssimulacra2").provenance.compute_backend == "cpu"


def test_cpu_frame_extraction_applies_duration_limit_to_both_outputs(tmp_path, monkeypatch):
    from dataclasses import replace

    from vmaf_app.core.perceptual_cpu import _extract_png_pairs

    request = _request()
    recipe = replace(request.recipe, duration_limit=1.0)
    command = []

    class FinishedProcess:
        pid = 123
        returncode = 0

        @staticmethod
        def poll():
            return 0

    def fake_popen(args, **_kwargs):
        command.extend(args)
        (tmp_path / "test-00000001.png").touch()
        (tmp_path / "reference-00000001.png").touch()
        return FinishedProcess()

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.proc_util.popen", fake_popen)

    _extract_png_pairs(
        _info("source.mp4"), _info("test.mp4"), recipe, None, None, 1,
        tmp_path, None, None,
    )

    assert command.count("-t") == 2
    for pattern in ("test-%08d.png", "reference-%08d.png"):
        output_index = next(i for i, value in enumerate(command) if value.endswith(pattern))
        assert command[output_index - 6:output_index] == [
            "-t", "1.000", "-fps_mode", "passthrough", "-pix_fmt", "rgb48le",
        ]


def test_auto_crop_detection_uses_full_video_not_score_duration(monkeypatch):
    from dataclasses import replace

    from vmaf_app.core.perceptual_cpu import _resolve_crops

    request = _request()
    recipe = replace(request.recipe, crop_mode=CropMode.AUTO, duration_limit=1.0)
    calls = []

    def fake_detect(info, **kwargs):
        calls.append((info.path.name, kwargs))
        return None

    monkeypatch.setattr("vmaf_app.core.perceptual_cpu.detect_crop", fake_detect)

    _resolve_crops(_info("source.mp4"), _info("test.mp4"), recipe, None, None, None)

    assert [name for name, _kwargs in calls] == ["source.mp4", "test.mp4"]
    assert all("duration_limit" not in kwargs for _name, kwargs in calls)


def test_cached_backend_does_not_suppress_missing_backend():
    options = VmafOptions(compute_vmaf=True)
    request = analysis_request_from_vmaf_options(options, ("vmaf", "ssimulacra2"))
    cached = MetricResultSet([
        FrameMetricResult("vmaf", np.array([0]), np.array([0.0]), np.array([90.0]),
                          MetricProvenance("test", "1", "cpu", "test")),
    ])
    plan = build_execution_plan(request, cached)
    assert [(task.backend_id, task.metric_keys) for task in plan.tasks] == [
        ("perceptual", ("ssimulacra2",)),
    ]


def test_perceptual_cache_entries_are_independent_and_coverage_specific(tmp_path):
    full, sampled = (
        MetricRequestSpec("ssimulacra2", "perceptual", (), FrameCoverage(mode, step), "ssimulacra2-reference-cli-v1")
        for mode, step in (("full", 1), ("sampled", 2))
    )
    butter = MetricRequestSpec("butteraugli", "perceptual", (), FrameCoverage("full", 1), "butteraugli-reference-cli-v1")
    provenance = MetricProvenance("test", "1", "cpu", "ssimulacra2-reference-cli-v1")
    store_metric(tmp_path, FrameMetricResult("ssimulacra2", [0], [0.0], [90.0], provenance), full)
    store_metric(tmp_path, FrameMetricResult("butteraugli", [0], [0.0], [0.2], provenance), butter)
    loaded = load_metrics(tmp_path, (full, sampled, butter))
    assert loaded.has("ssimulacra2") and loaded.has("butteraugli")
    # The sampled identity has a different filename/key and cannot borrow a
    # full-coverage score accidentally.
    assert len(list(tmp_path.glob("ssimulacra2_*.npz"))) == 1
    assert load_metric(tmp_path, sampled) is None


def test_ui_selects_cpu_metric_without_extending_vmaf_options(tmp_path):
    from PySide6.QtWidgets import QApplication

    from vmaf_app.ui.main_window import COL_SSIMULACRA2, MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    row = window._add_table_row(tmp_path / "test.mp4")
    window._apply_metric_selection([row], COL_SSIMULACRA2, True, set_default=False)
    assert "ssimulacra2" in window._requested_metrics(window._rows[row])
    assert not hasattr(window._rows[row].options, "compute_ssimulacra2")
