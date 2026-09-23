"""Per-frame metric statistics and metric-specific sequence aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from vmaf_app.core.metrics import METRIC_BY_KEY, MetricAggregation
from vmaf_app.core.models import ComparisonResult

# Default threshold breakdown requested: >95, >90, >85, <85, <80, <70
DEFAULT_THRESHOLDS = METRIC_BY_KEY["vmaf"].thresholds

HISTOGRAM_BIN_EDGES: list[float] = [0, 70, 80, 85, 90, 95, 100]


@dataclass
class ThresholdStat:
    comparison: str  # ">" or "<"
    threshold: float
    count: int
    percentage: float

    @property
    def label(self) -> str:
        return f"{self.comparison} {self.threshold:g}"


@dataclass
class HistogramBin:
    low: float
    high: float
    count: int
    percentage: float


@dataclass
class VmafStats:
    count: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float
    percentile_10: float  # "10% low" -- the score below which the worst 10% of frames fall
    percentile_5: float  # "5% low"
    percentile_1: float  # "1% low"
    percentile_0_1: float  # "0.1% low" -- needs a lot of frames to be meaningful
    #: Frames scoring +inf: mathematically identical to the reference. Kept
    #: in the sequence aggregate and counted here too -- see
    #: compute_stats. They still appear per-frame and in the threshold
    #: counts, where "better than X" is exactly what they are.
    identical: int = 0
    thresholds: list[ThresholdStat] = field(default_factory=list)
    histogram: list[HistogramBin] = field(default_factory=list)

    @property
    def values(self) -> list[tuple[str, float]]:
        """(label, value) pairs. This is the single place that defines what
        shows up in the graph's stats table -- add a stat here (and compute
        it in compute_stats()) and it appears there automatically, no UI code
        changes needed.

        Unformatted, because the right precision depends on the metric: 2dp
        suits VMAF's 0-100 and PSNR's dB, and destroys SSIM, whose entire
        range is 0-1. The caller knows which metric it is asking about; this
        does not.
        """
        return [
            ("Mean", self.mean),
            ("Median", self.median),
            ("StDev", self.stdev),
            ("Min", self.minimum),
            ("Max", self.maximum),
            ("10% Low", self.percentile_10),
            ("5% Low", self.percentile_5),
            ("1% Low", self.percentile_1),
            ("0.1% Low", self.percentile_0_1),
        ]

    def summary(self, value_format: str = "{:.2f}") -> list[tuple[str, str]]:
        """`values`, rendered at the metric's own precision."""
        def formatted(value: float) -> str:
            if np.isnan(value):
                return "—"
            if np.isposinf(value):
                return "∞"
            if np.isneginf(value):
                return "−∞"
            return value_format.format(value)

        return [(label, formatted(value)) for label, value in self.values]


#: How a metric's per-frame scores combine into one number for the run.
#:
#: "arithmetic" is the plain mean, correct for VMAF and SSIM (both bounded)
#: and for libvmaf's PSNR, which clamps a perfect frame to its bit depth's
#: ceiling -- 60 dB at 8-bit, 72 dB at 10-bit -- rather than reporting
#: infinity. Verified over 4.2M frames of real results: VMAF, PSNR and SSIM
#: never produced a single infinite value.
#:
#: "square_mean_root" is XPSNR's own sequence average, and XPSNR is the one
#: metric here that does report infinity for an identical frame. FFmpeg's
#: vf_xpsnr accumulates sqrt(wsse) per frame and reports
#:
#:     10*log10(W*H*max_error / (sum_sqrt_wsse / N)^2)
#:
#: Since a frame's own value is xpsnr = 10*log10(W*H*max_error / wsse),
#: sqrt(wsse) = sqrt(W*H*max_error) * 10^(-xpsnr/20), and the constant
#: cancels when it is substituted back, leaving
#:
#:     -20 * log10( mean( 10^(-xpsnr_i/20) ) )
#:
#: which needs nothing but the per-frame values. An identical frame has
#: xpsnr = inf, so it contributes 0 to the sum and 1 to N -- exactly what
#: ffmpeg's own accumulator does with sqrt(0). Checked against ffmpeg's
#: printed average on the same clip: 54.9459 over 72 frames and 45.0831 over
#: 480, matching to four decimal places both times.
ARITHMETIC = MetricAggregation.ARITHMETIC
SQUARE_MEAN_ROOT = MetricAggregation.SQUARE_MEAN_ROOT_DB

#: Only XPSNR differs, and only because only XPSNR reports infinity.
AGGREGATE_BY_METRIC = {key: definition.aggregation for key, definition in METRIC_BY_KEY.items()}


def _square_mean_root_db(data: np.ndarray) -> float:
    """XPSNR's sequence average over per-frame decibels. See SQUARE_MEAN_ROOT."""
    with np.errstate(over="ignore"):
        distortion = np.power(10.0, -data / 20.0)
    mean = float(distortion.mean())
    # Every frame identical: no error to average, and infinity is the answer
    # ffmpeg gives too.
    return float("inf") if mean <= 0.0 else float(-20.0 * np.log10(mean))


def aggregate_scores(values, aggregate: MetricAggregation | str = ARITHMETIC) -> float | None:
    """One number for a whole run, by the metric's own convention.

    None when there is nothing to combine. NaN frames -- ones the metric was
    not computed for -- are dropped first; infinities are not, because for
    XPSNR they are meaningful and this is what handles them.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return None
    if aggregate == SQUARE_MEAN_ROOT:
        return _square_mean_root_db(data)
    return float(data.mean())


def compute_stats(
    values,
    thresholds: list[tuple[str, float]] | None = None,
    aggregate: MetricAggregation | str = ARITHMETIC,
) -> VmafStats:
    """Despite the name (kept for the VMAF-specific callers/tests that exist
    already), this works over any sequence of per-frame float scores -- PSNR,
    SSIM and XPSNR reuse it with their own threshold bands and aggregation.
    Pass an empty threshold list to omit bands entirely.

    Accepts a numpy array or a plain list. Computed vectorised: a run is
    hundreds of thousands of frames and this is called once per metric per
    series, so a Python-level sort + several passes was real, avoidable time.
    NaN entries (a metric present for only some frames) are ignored rather
    than poisoning every statistic.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    data = np.asarray(values, dtype=np.float64)
    data = data[~np.isnan(data)]
    n = int(data.size)
    if n == 0:
        return VmafStats(
            count=0, mean=0, median=0, stdev=0, minimum=0, maximum=0,
            percentile_10=0, percentile_5=0, percentile_1=0, percentile_0_1=0,
            thresholds=[], histogram=[],
        )

    # A frame identical to the reference scores +inf, which XPSNR reports
    # outright. Counted so the display can say so; the mean handles them by
    # using the metric's own aggregation (see aggregate_scores), and the
    # threshold tallies below see them too, because "better than 38 dB" is
    # precisely what a perfect frame is.
    identical = int(np.isposinf(data).sum())
    finite = data[np.isfinite(data)]

    # One sort, then every percentile is a lookup into it. Order statistics
    # are well defined with infinities present -- and the percentiles that
    # may also fall in the infinite tail for mostly identical material.
    ordered = np.sort(data)
    if np.isposinf(ordered).all():
        p10 = p5 = p1 = p01 = float("inf")
    elif np.isneginf(ordered).all():
        p10 = p5 = p1 = p01 = float("-inf")
    else:
        def percentile(q: float) -> float:
            position = (len(ordered) - 1) * q / 100.0
            lower = int(np.floor(position))
            upper = int(np.ceil(position))
            a, b = ordered[lower], ordered[upper]
            if lower == upper or a == b:
                return float(a)
            # Extended-real linear interpolation: finite-to-infinite is
            # infinite at any interior point. Opposite infinities are undefined.
            if np.isinf(a) or np.isinf(b):
                if np.isneginf(a) and np.isposinf(b):
                    return float("nan")
                return float(a if np.isinf(a) else b)
            return float(a + (b - a) * (position - lower))

        p10, p5, p1, p01 = (percentile(q) for q in (10, 5, 1, 0.1))

    threshold_stats = []
    for cmp_op, thresh in thresholds:
        count = int(np.count_nonzero(data > thresh if cmp_op == ">" else data < thresh))
        threshold_stats.append(ThresholdStat(cmp_op, thresh, count, 100.0 * count / n))

    histogram = []
    edges = HISTOGRAM_BIN_EDGES
    for lo, hi in pairwise(edges):
        in_bin = (data >= lo) & (data <= hi if hi == edges[-1] else data < hi)
        count = int(np.count_nonzero(in_bin))
        histogram.append(HistogramBin(lo, hi, count, 100.0 * count / n))

    mean = aggregate_scores(data, aggregate)
    # Standard deviation over the finite frames only: it is undefined for a
    # set containing infinity (numpy returns nan), and a spread of "nan"
    # whenever one frame happened to be identical says less than a spread of
    # the frames that actually differ.
    with np.errstate(invalid="ignore"):
        stdev = float(finite.std()) if finite.size else float("nan")

    return VmafStats(
        count=n,
        identical=identical,
        mean=mean,
        median=float(np.median(ordered)),
        stdev=stdev,  # population stdev; undefined for an infinite population
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
        percentile_10=p10,
        percentile_5=p5,
        percentile_1=p1,
        percentile_0_1=p01,
        thresholds=threshold_stats,
        histogram=histogram,
    )


def stats_for_run(result: ComparisonResult, thresholds: list[tuple[str, float]] | None = None) -> VmafStats:
    return compute_stats(result.frames.vmaf if result.frames.vmaf is not None else [], thresholds)
