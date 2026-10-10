"""Statistics of a metric's per-frame scores (compute_stats), and the one
number for a whole run, by each metric's own way of averaging
(aggregate_scores)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vmaf_app.core.metrics import METRIC_BY_KEY, MetricAggregation, MetricDirection

# The default score bands, VMAF's: >95, >90, >85, <85, <80, <70.
DEFAULT_THRESHOLDS = METRIC_BY_KEY["vmaf"].thresholds

@dataclass
class ThresholdStat:
    """How many frames score above (">") or below ("<") a threshold."""

    comparison: str  # ">" or "<"
    threshold: float
    count: int
    percentage: float

    @property
    def label(self) -> str:
        """The band as shown, such as "> 95"."""
        return f"{self.comparison} {self.threshold:g}"


@dataclass
class VmafStats:
    """The statistics of one metric's per-frame scores (for any metric,
    despite the name)."""

    count: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float
    # The worst frames' tail. For a higher-is-better metric, "10% low" is the
    # score below which the worst 10% of frames fall. For a lower-is-better
    # metric (Butteraugli) the worst frames are the high ones, so these hold
    # the 90th, 95th, 99th and 99.9th percentiles instead (see worst_is_high).
    percentile_10: float
    percentile_5: float
    percentile_1: float
    percentile_0_1: float  # needs a lot of frames to mean anything
    #: Frames scoring +inf: identical to the source. They count in the run's
    #: average and are counted here too (see compute_stats). They also appear
    #: per frame and in the threshold counts, where "better than X" is
    #: exactly what they are.
    identical: int = 0
    thresholds: list[ThresholdStat] = field(default_factory=list)
    #: True for a lower-is-better metric: the percentile_* fields are taken
    #: from the top, and labelled "High" rather than "Low".
    worst_is_high: bool = False

    @staticmethod
    def tail_labels(worst_is_high: bool = False) -> list[str]:
        """The tail fields' labels, such as "1% Low"."""
        side = "High" if worst_is_high else "Low"
        return [f"{share} {side}" for share in ("10%", "5%", "1%", "0.1%")]

    @property
    def values(self) -> list[tuple[str, float]]:
        """The statistics as (label, value) pairs. This is the one place
        that defines what the graph's statistics table shows: a statistic
        added here (and calculated in compute_stats()) appears there with
        no change to the window's code.

        The values are not formatted, because the right precision depends on
        the metric: 2 decimals suit VMAF's 0-100 and PSNR's decibels, but
        ruin SSIM, whose whole range is 0-1. The caller knows the metric;
        this does not.
        """
        return [
            ("Mean", self.mean),
            ("Median", self.median),
            ("StDev", self.stdev),
            ("Min", self.minimum),
            ("Max", self.maximum),
            *zip(
                self.tail_labels(self.worst_is_high),
                (self.percentile_10, self.percentile_5, self.percentile_1, self.percentile_0_1),
                strict=True,
            ),
        ]

    def summary(self, value_format: str = "{:.2f}") -> list[tuple[str, str]]:
        """`values` as text, at the metric's own precision."""
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
#: "arithmetic" is the plain mean, which is right for VMAF and SSIM (both
#: bounded) and for libvmaf's PSNR, which caps a perfect frame at its bit
#: depth's ceiling (60 dB at 8 bits, 72 dB at 10 bits) rather than reporting
#: infinity. Over 4.2 million frames of real results, VMAF, PSNR and SSIM
#: never gave an infinite value.
#:
#: "square_mean_root" is XPSNR's own average over the video, and XPSNR is
#: the one metric here that does report infinity for an identical frame.
#: FFmpeg's vf_xpsnr adds up sqrt(wsse) per frame and reports
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
#: xpsnr = inf, so it adds 0 to the sum and 1 to N, exactly as FFmpeg's own
#: sum does with sqrt(0). Checked against FFmpeg's printed average on the
#: same clip: 54.9459 over 72 frames and 45.0831 over 480, matching to four
#: decimal places both times.
ARITHMETIC = MetricAggregation.ARITHMETIC
SQUARE_MEAN_ROOT = MetricAggregation.SQUARE_MEAN_ROOT_DB

#: Each metric's way of averaging. Only XPSNR's differs, because only XPSNR
#: reports infinity.
AGGREGATE_BY_METRIC = {key: definition.aggregation for key, definition in METRIC_BY_KEY.items()}


def _square_mean_root_db(data: np.ndarray) -> float:
    """XPSNR's average over the video from per-frame decibels (see
    SQUARE_MEAN_ROOT)."""
    with np.errstate(over="ignore"):
        distortion = np.power(10.0, -data / 20.0)
    mean = float(distortion.mean())
    # When every frame is identical there is no error to average, and
    # infinity is FFmpeg's answer too.
    return float("inf") if mean <= 0.0 else float(-20.0 * np.log10(mean))


def aggregate_scores(values, aggregate: MetricAggregation | str = ARITHMETIC) -> float | None:
    """Returns one number for a whole run, by the metric's own way of
    averaging, or None when there is nothing to average.

    NaN frames (those the metric was not calculated for) are dropped first.
    Infinities are not, because for XPSNR they mean something, and this is
    what handles them.
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
    direction: MetricDirection | str = MetricDirection.HIGHER_IS_BETTER,
) -> VmafStats:
    """Returns the statistics of any metric's per-frame scores, with its own
    score bands (`thresholds`; an empty list leaves the bands out) and its
    own way of averaging (`aggregate`).

    `direction` decides which end is the worst frames' tail: the low end for
    VMAF and most metrics, the high end for Butteraugli, where 0 means
    identical and a bigger number a more visible difference.

    `values` can be a numpy array or a list. The work is done with numpy,
    because a run has hundreds of thousands of frames and this is called
    once per metric per graph line. NaN values (a metric missing for some
    frames) are ignored, rather than spoiling every statistic.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    worst_is_high = MetricDirection(direction) is MetricDirection.LOWER_IS_BETTER
    data = np.asarray(values, dtype=np.float64)
    data = data[~np.isnan(data)]
    n = int(data.size)
    if n == 0:
        return VmafStats(
            count=0, mean=0, median=0, stdev=0, minimum=0, maximum=0,
            percentile_10=0, percentile_5=0, percentile_1=0, percentile_0_1=0,
            thresholds=[], worst_is_high=worst_is_high,
        )

    # A frame identical to the source scores +inf, which XPSNR reports as it
    # is. Such frames are counted, so that the window can say so. The mean
    # handles them through the metric's own way of averaging (see
    # aggregate_scores), and the threshold counts below include them,
    # because "better than 38 dB" is exactly what a perfect frame is.
    identical = int(np.isposinf(data).sum())
    finite = data[np.isfinite(data)]

    # One sort, then every percentile is a lookup into it. Order statistics
    # are well defined with infinities present, and for mostly identical
    # video the percentiles may fall among the infinities too.
    ordered = np.sort(data)
    if np.isposinf(ordered).all():
        p10 = p5 = p1 = p01 = float("inf")
    elif np.isneginf(ordered).all():
        p10 = p5 = p1 = p01 = float("-inf")
    else:
        def percentile(q: float) -> float:
            """The q-th percentile, interpolated linearly as numpy does."""
            position = (len(ordered) - 1) * q / 100.0
            lower = int(np.floor(position))
            upper = int(np.ceil(position))
            a, b = ordered[lower], ordered[upper]
            if lower == upper or a == b:
                return float(a)
            # Between a finite value and an infinity, every point in between
            # is infinite. Between opposite infinities, it is undefined.
            if np.isinf(a) or np.isinf(b):
                if np.isneginf(a) and np.isposinf(b):
                    return float("nan")
                return float(a if np.isinf(a) else b)
            return float(a + (b - a) * (position - lower))

        tails = (10, 5, 1, 0.1)
        p10, p5, p1, p01 = (percentile(100 - q if worst_is_high else q) for q in tails)

    threshold_stats = []
    for cmp_op, thresh in thresholds:
        count = int(np.count_nonzero(data > thresh if cmp_op == ">" else data < thresh))
        threshold_stats.append(ThresholdStat(cmp_op, thresh, count, 100.0 * count / n))

    mean = aggregate_scores(data, aggregate)
    # The standard deviation is taken over the finite frames only. It is
    # undefined for a set containing infinity (numpy returns NaN), and a
    # spread of "NaN" whenever one frame happens to be identical says less
    # than the spread of the frames that do differ.
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
        worst_is_high=worst_is_high,
    )
