"""Summary statistics over a VMAF per-frame run."""
from __future__ import annotations

from dataclasses import dataclass, field

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
    thresholds: list[ThresholdStat] = field(default_factory=list)
    histogram: list[HistogramBin] = field(default_factory=list)

    @property
    def summary(self) -> list[tuple[str, str]]:
        """(label, formatted value) pairs for display. This is the single
        place that defines what shows up in the graph window's stats table --
        add a stat here (and compute it in compute_stats()) and it appears
        there automatically, no UI code changes needed."""
        return [
            ("Mean", f"{self.mean:.2f}"),
            ("Median", f"{self.median:.2f}"),
            ("StDev", f"{self.stdev:.2f}"),
            ("Min", f"{self.minimum:.2f}"),
            ("Max", f"{self.maximum:.2f}"),
            ("10% Low", f"{self.percentile_10:.2f}"),
            ("5% Low", f"{self.percentile_5:.2f}"),
            ("1% Low", f"{self.percentile_1:.2f}"),
            ("0.1% Low", f"{self.percentile_0_1:.2f}"),
        ]


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_stats(
    values: list[float],
    thresholds: list[tuple[str, float]] | None = None,
) -> VmafStats:
    """Despite the name (kept for the VMAF-specific callers/tests that exist
    already), this works over any list of per-frame float scores -- PSNR,
    SSIM and XPSNR reuse it for their own stats tables in the graph window,
    just with VMAF-specific `thresholds` left empty since ">95"-style bands
    only make sense on VMAF's fixed 0-100 scale.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    n = len(values)
    if n == 0:
        return VmafStats(
            count=0, mean=0, median=0, stdev=0, minimum=0, maximum=0,
            percentile_10=0, percentile_5=0, percentile_1=0, percentile_0_1=0,
            thresholds=[], histogram=[],
        )

    sorted_values = sorted(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    stdev = variance ** 0.5
    median = _percentile(sorted_values, 50)
    p10 = _percentile(sorted_values, 10)
    p5 = _percentile(sorted_values, 5)
    p1 = _percentile(sorted_values, 1)
    p01 = _percentile(sorted_values, 0.1)

    threshold_stats = []
    for cmp_op, thresh in thresholds:
        if cmp_op == ">":
            count = sum(1 for v in values if v > thresh)
        else:
            count = sum(1 for v in values if v < thresh)
        threshold_stats.append(ThresholdStat(cmp_op, thresh, count, 100.0 * count / n))

    histogram = []
    edges = HISTOGRAM_BIN_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_last = hi == edges[-1]
        if is_last:
            count = sum(1 for v in values if lo <= v <= hi)
        else:
            count = sum(1 for v in values if lo <= v < hi)
        histogram.append(HistogramBin(lo, hi, count, 100.0 * count / n))

    return VmafStats(
        count=n,
        mean=mean,
        median=median,
        stdev=stdev,
        minimum=sorted_values[0],
        maximum=sorted_values[-1],
        percentile_10=p10,
        percentile_5=p5,
        percentile_1=p1,
        percentile_0_1=p01,
        thresholds=threshold_stats,
        histogram=histogram,
    )


def stats_for_run(result: VmafRunResult, thresholds: list[tuple[str, float]] | None = None) -> VmafStats:
    return compute_stats([f.vmaf for f in result.frames], thresholds)
