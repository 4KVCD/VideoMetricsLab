"""Summary statistics over a VMAF per-frame run."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from vmaf_app.core.models import VmafRunResult

# Default threshold breakdown requested: >95, >90, >85, <85, <80, <70
DEFAULT_THRESHOLDS: list[tuple[str, float]] = [
    (">", 95.0),
    (">", 90.0),
    (">", 85.0),
    ("<", 85.0),
    ("<", 80.0),
    ("<", 70.0),
]

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
    #: out of every summary above and counted here instead -- see
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


def mean_of_measurable(values) -> float | None:
    """The mean of a metric column, or None if there is nothing to average.

    Frames identical to the reference score +inf (XPSNR reports it; libvmaf
    clamps PSNR instead), and one of them turns an ordinary mean into inf --
    which is how a film opening on black came to report its whole encode's
    XPSNR as "inf". Those frames are excluded, and NaN (a frame the metric
    was not computed for) with them. If every frame is identical, inf is the
    honest answer and is returned as such.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return None
    finite = data[np.isfinite(data)]
    if finite.size:
        return float(finite.mean())
    return float(data[0])  # all identical, or all -inf: report it rather than hide it


def compute_stats(
    values,
    thresholds: list[tuple[str, float]] | None = None,
) -> VmafStats:
    """Despite the name (kept for the VMAF-specific callers/tests that exist
    already), this works over any sequence of per-frame float scores -- PSNR,
    SSIM and XPSNR reuse it for their own stats tables in the graph window,
    just with VMAF-specific `thresholds` left empty since ">95"-style bands
    only make sense on VMAF's fixed 0-100 scale.

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

    # A frame identical to the reference scores +inf -- XPSNR reports it
    # outright, where libvmaf instead clamps PSNR to its bit depth's ceiling.
    # A single such frame makes the mean inf and the standard deviation nan,
    # so a film that opens on a few seconds of black reported "inf" as the
    # XPSNR of the entire encode. They are held out of the summary below and
    # counted separately; the threshold tallies still see them, because
    # "better than 38 dB" is precisely what a perfect frame is.
    finite = data[np.isfinite(data)]
    identical = int(np.isposinf(data).sum())
    # Unless there is nothing else: an encode that really is identical
    # throughout has no finite frames to describe, and inf is then the
    # honest answer rather than a missing one.
    summarised = finite if finite.size else data

    # One sort, then every percentile is a lookup into it.
    ordered = np.sort(summarised)
    if np.isposinf(ordered).all():
        p10 = p5 = p1 = p01 = float("inf")
    elif np.isneginf(ordered).all():
        p10 = p5 = p1 = p01 = float("-inf")
    else:
        with np.errstate(invalid="ignore"):
            p10, p5, p1, p01 = (
                float(v) for v in np.percentile(
                    ordered, [10, 5, 1, 0.1], method="linear"
                )
            )

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

    with np.errstate(invalid="ignore"):
        mean = float(summarised.mean())
        stdev = float(summarised.std())

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


def stats_for_run(result: VmafRunResult, thresholds: list[tuple[str, float]] | None = None) -> VmafStats:
    return compute_stats(result.frames.vmaf if result.frames.vmaf is not None else [], thresholds)
