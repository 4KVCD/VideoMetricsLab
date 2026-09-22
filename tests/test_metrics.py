"""Compatibility tests for the headless metric registry and packed scores."""
from __future__ import annotations

import numpy as np
import pytest

from vmaf_app.core.metrics import (
    FRAME_METRICS,
    METRIC_BY_KEY,
    METRICS,
    MetricAggregation,
    metric_definition,
)
from vmaf_app.core.models import FrameScores, VmafOptions


def test_registry_has_the_established_logical_order_and_metadata():
    assert tuple(metric.key for metric in METRICS) == (
        "vmaf", "vmaf_neg", "psnr", "ssim", "xpsnr",
    )
    assert FRAME_METRICS == METRICS
    assert metric_definition("xpsnr").aggregation is MetricAggregation.SQUARE_MEAN_ROOT_DB
    assert metric_definition("ssim").value_format == "{:.4f}"
    assert metric_definition("vmaf").fixed_y_max == 100.0
    with pytest.raises(TypeError):
        METRIC_BY_KEY["new"] = metric_definition("vmaf")  # type: ignore[index]


def test_options_registry_mapping_preserves_unknown_feature_order():
    options = VmafOptions(extra_features=["name=custom", "name=psnr"])
    assert options.requested_metrics() == ("vmaf", "psnr")
    options.set_metric_enabled("ssim", True)
    assert options.extra_features == ["name=custom", "name=psnr", "name=float_ssim"]
    options.set_metric_enabled("psnr", False)
    assert options.extra_features == ["name=custom", "name=float_ssim"]
    assert options.requested_metrics() == ("vmaf", "ssim")


def test_frame_scores_accept_and_preserve_unknown_metrics():
    scores = FrameScores(
        [0, 1], [0.0, 1 / 24], None,
        metrics={"psnr": [40.0, np.nan], "future_metric": [1.0, 2.0]},
    )
    assert scores.metric_keys == ("psnr", "future_metric")
    assert scores.vmaf is None
    assert scores.psnr is not None and scores.psnr[0] == 40.0
    assert scores.has("future_metric")
    sliced = scores[:1]
    assert sliced.metric_keys == scores.metric_keys
    assert sliced.values("future_metric").tolist() == [1.0]
    changed = scores.with_values("future_metric", None)
    assert not changed.has("future_metric")
    assert changed.has("psnr")
    assert scores.nbytes() == sum(array.nbytes for array in (scores.frame, scores.time, scores.psnr, scores.values("future_metric")))


def test_frame_scores_equality_checks_key_sets_and_nan_values():
    one = FrameScores([0], [0], None, metrics={"future": [np.nan]})
    same = FrameScores([0], [0], None, metrics={"future": [np.nan]})
    different = FrameScores([0], [0], None, metrics={"other": [np.nan]})
    assert one == same
    assert one != different
