"""The canonical description of metrics supported by VideoMetricsLab.

This module deliberately has no Qt, ffmpeg, or result-model imports.  The
registry is consequently safe to use from the calculation pipeline, saved
result adapters, and every UI surface without creating a dependency cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np


class MetricKind(str, Enum):
    FRAME = "frame"
    SEQUENCE = "sequence"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricAggregation(str, Enum):
    ARITHMETIC = "arithmetic"
    SQUARE_MEAN_ROOT_DB = "square_mean_root"


@dataclass(frozen=True, slots=True)
class FfmpegMetricBinding:
    """How an established metric maps onto the current FFmpeg/UI options."""

    bool_option: str | None = None
    libvmaf_feature: str | None = None


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    short_label: str
    table_header: str
    axis_label: str
    value_format: str
    value_suffix: str
    kind: MetricKind
    direction: MetricDirection
    aggregation: MetricAggregation
    fixed_y_max: float | None
    thresholds: tuple[tuple[str, float], ...]
    ffmpeg_binding: FfmpegMetricBinding | None = None
    # Optional because a registry entry can be display/import-only before an
    # executable backend ships. FFmpeg metrics retain their established
    # binding; standalone metrics name their backend explicitly.
    backend_id: str | None = None

    # These small presentation helpers keep all precision and infinity rules
    # together with the metric metadata rather than duplicated in Qt panels.
    def format_value(self, value: float | None) -> str:
        if value is None or not np.isfinite(value):
            if value is not None and np.isposinf(value):
                return "∞"
            if value is not None and np.isneginf(value):
                return "−∞"
            return "—"
        return self.value_format.format(float(value))

    def format_delta(self, value: float | None) -> str:
        if value is None or np.isnan(value):
            return "—"
        if np.isposinf(value):
            return "+∞"
        if np.isneginf(value):
            return "−∞"
        return self.value_format.replace("{:", "{:+").format(float(value))


# Kept as tuples: caller code must not be able to mutate the canonical
# registry or threshold bands at runtime.
_VMAF_THRESHOLDS = ((">", 95.0), (">", 90.0), (">", 85.0), ("<", 85.0), ("<", 80.0), ("<", 70.0))
_PSNR_THRESHOLDS = ((">", 41.0), (">", 38.0), (">", 35.0), ("<", 35.0), ("<", 34.0), ("<", 32.0))
_SSIM_THRESHOLDS = ((">", 0.99), (">", 0.98), (">", 0.97), ("<", 0.97), ("<", 0.96), ("<", 0.95))
_XPSNR_THRESHOLDS = ((">", 38.0), (">", 35.0), (">", 33.0), ("<", 33.0), ("<", 30.0), ("<", 27.0))
_SSIMULACRA2_THRESHOLDS = ((">", 90.0), (">", 80.0), (">", 70.0), ("<", 70.0), ("<", 50.0), ("<", 30.0))


# This is the logical/display order.  The version-1 result-file row order is
# intentionally represented separately in run_io, where it remains frozen.
METRICS = (
    MetricDefinition("vmaf", "VMAF", "VMAF", "VMAF", "VMAF", "{:.2f}", "", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, 100.0,
                     _VMAF_THRESHOLDS, FfmpegMetricBinding(bool_option="compute_vmaf")),
    MetricDefinition("vmaf_neg", "VMAF NEG", "VMAF NEG", "VMAF NEG", "VMAF NEG", "{:.2f}", "", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, 100.0,
                     _VMAF_THRESHOLDS, FfmpegMetricBinding(bool_option="compute_vmaf_neg")),
    MetricDefinition("psnr", "PSNR", "PSNR", "PSNR (dB)", "PSNR (dB)", "{:.2f}", " dB", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, None,
                     _PSNR_THRESHOLDS, FfmpegMetricBinding(libvmaf_feature="name=psnr")),
    MetricDefinition("ssim", "SSIM", "SSIM", "SSIM", "SSIM", "{:.4f}", "", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, None,
                     _SSIM_THRESHOLDS, FfmpegMetricBinding(libvmaf_feature="name=float_ssim")),
    MetricDefinition("xpsnr", "XPSNR", "XPSNR", "XPSNR (dB)", "XPSNR (dB)", "{:.2f}", " dB", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.SQUARE_MEAN_ROOT_DB, None,
                     _XPSNR_THRESHOLDS, FfmpegMetricBinding(bool_option="compute_xpsnr")),
    MetricDefinition("ssimulacra2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", "SSIMULACRA2", "{:.2f}", "", MetricKind.FRAME,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, 100.0,
                     _SSIMULACRA2_THRESHOLDS, backend_id="perceptual"),
    MetricDefinition("butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", "Butteraugli", "{:.4f}", "", MetricKind.FRAME,
                     MetricDirection.LOWER_IS_BETTER, MetricAggregation.ARITHMETIC, None,
                     (), backend_id="perceptual"),
    # ColorVideoVDP: one score per video in JOD (just-objectionable
    # differences; 10 = no visible difference). Its per-second curve is a
    # timeline on the sequence result, not per-frame scores. GPU only.
    MetricDefinition("cvvdp", "CVVDP", "CVVDP", "CVVDP", "CVVDP (JOD)", "{:.3f}", "", MetricKind.SEQUENCE,
                     MetricDirection.HIGHER_IS_BETTER, MetricAggregation.ARITHMETIC, 10.0,
                     (), backend_id="perceptual"),
)

METRIC_BY_KEY = MappingProxyType({metric.key: metric for metric in METRICS})
FRAME_METRICS = tuple(metric for metric in METRICS if metric.kind is MetricKind.FRAME)
SEQUENCE_METRICS = tuple(metric for metric in METRICS if metric.kind is MetricKind.SEQUENCE)


def metric_definition(key: str) -> MetricDefinition:
    """Return a metric definition or raise KeyError for an unknown key."""
    return METRIC_BY_KEY[key]
