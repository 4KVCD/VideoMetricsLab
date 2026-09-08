import math

import pytest

from vmaf_app.core.stats import compute_stats, mean_of_measurable


def test_basic_stats():
    stats = compute_stats([90, 92, 94, 96, 98])
    assert stats.count == 5
    assert stats.mean == 94.0
    assert stats.minimum == 90.0
    assert stats.maximum == 98.0
    assert stats.median == 94.0


def test_threshold_breakdown_matches_requested_bands():
    values = [96, 91, 86, 84, 79, 65, 100, 95.5, 90.1, 85.1]
    stats = compute_stats(values)
    by_label = {t.label: t for t in stats.thresholds}

    assert by_label["> 95"].count == sum(1 for v in values if v > 95)
    assert by_label["> 90"].count == sum(1 for v in values if v > 90)
    assert by_label["> 85"].count == sum(1 for v in values if v > 85)
    assert by_label["< 85"].count == sum(1 for v in values if v < 85)
    assert by_label["< 80"].count == sum(1 for v in values if v < 80)
    assert by_label["< 70"].count == sum(1 for v in values if v < 70)

    total = len(values)
    for t in stats.thresholds:
        assert t.percentage == pytest.approx(100.0 * t.count / total)


def test_empty_frames_returns_zeroed_stats():
    stats = compute_stats([])
    assert stats.count == 0
    assert stats.thresholds == []
    assert stats.histogram == []
    assert dict(stats.summary())["Mean"] == "0.00"  # summary must format, not raise


def test_histogram_bins_cover_all_frames():
    values = [i for i in range(0, 101, 5)]  # 0,5,...,100
    stats = compute_stats(values)
    assert sum(b.count for b in stats.histogram) == len(values)


def test_percentile_1_and_0_1_low_with_a_large_sample():
    # 1000 frames: 990 at 95, the worst 10 (1%) ramping down to a floor of 50.
    values = [95.0] * 990 + [50.0 + i for i in range(10)]
    stats = compute_stats(values)
    # 1% low should sit near the worst ~1% of frames, well below the bulk at 95.
    assert stats.percentile_1 < 95.0
    assert stats.percentile_0_1 <= stats.percentile_1  # 0.1% low is at least as extreme


def test_summary_reflects_the_computed_values():
    stats = compute_stats([90, 92, 94, 96, 98])
    summary = dict(stats.summary())
    assert summary["Mean"] == "94.00"
    assert summary["Min"] == "90.00"
    assert summary["Max"] == "98.00"
    assert "10% Low" in summary
    assert "5% Low" in summary
    assert "1% Low" in summary
    assert "0.1% Low" in summary


def test_low_percentiles_are_monotonically_non_increasing():
    # 10% low >= 5% low >= 1% low >= 0.1% low, always -- each is a stricter
    # (smaller) worst-case slice of the same distribution.
    values = [95.0] * 900 + [50.0 + i * 0.1 for i in range(100)]
    stats = compute_stats(values)
    assert stats.percentile_10 >= stats.percentile_5
    assert stats.percentile_5 >= stats.percentile_1
    assert stats.percentile_1 >= stats.percentile_0_1


def test_perfect_infinite_metric_has_mean_and_percentiles_without_warnings():
    stats = compute_stats([float("inf")] * 5, thresholds=[])

    assert stats.count == 5
    assert stats.mean == float("inf")
    assert stats.minimum == float("inf")
    assert stats.percentile_1 == float("inf")
    summary = dict(stats.summary())
    assert summary["Mean"] == "∞"
    assert summary["StDev"] == "—"

# ------------------------------------------------- frames identical to source


def test_identical_frames_do_not_turn_the_summary_into_infinity():
    """A film opening on black reported "inf" as its whole XPSNR.

    XPSNR scores a frame identical to the reference as +inf, where libvmaf
    instead clamps PSNR to its bit depth's ceiling. One such frame made the
    mean inf and the standard deviation nan -- and the opening seconds of a
    feature are routinely pixel-identical black. Reproduced on a real 4K
    pair: 40 of the first 73 frames were inf.
    """
    values = [float("inf")] * 40 + [78.78, 16.82, 12.56, 7.63, 4.60, 2.88, 3.28, 4.17, 5.87]

    stats = compute_stats(values, thresholds=[(">", 38.0), ("<", 33.0)])

    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.stdev)
    assert math.isfinite(stats.median)
    assert math.isfinite(stats.percentile_1)
    assert math.isfinite(stats.maximum)
    assert stats.mean == pytest.approx(15.177, abs=0.01)
    # Counted rather than quietly dropped: the mean describes fewer frames
    # than the run measured, and that has to be visible.
    assert stats.identical == 40
    assert stats.count == 49


def test_identical_frames_still_count_towards_the_bands():
    # A perfect frame is emphatically "better than 38 dB". Excluding it from
    # the mean must not also exclude it from the tally.
    values = [float("inf")] * 3 + [40.0, 10.0]

    stats = compute_stats(values, thresholds=[(">", 38.0), ("<", 33.0)])

    above = next(t for t in stats.thresholds if t.label == "> 38")
    below = next(t for t in stats.thresholds if t.label == "< 33")
    assert above.count == 4 and above.percentage == pytest.approx(80.0)
    assert below.count == 1 and below.percentage == pytest.approx(20.0)


def test_an_entirely_identical_encode_still_reports_infinity():
    # Nothing to average, and inf is then the honest answer rather than a
    # number invented to avoid it.
    stats = compute_stats([float("inf")] * 5)

    assert math.isinf(stats.mean)
    assert stats.identical == 5


def test_mean_of_measurable_matches_the_summary():
    assert mean_of_measurable([float("inf"), 10.0, 20.0]) == pytest.approx(15.0)
    assert mean_of_measurable([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert mean_of_measurable([float("nan")] * 3) is None
    assert mean_of_measurable([]) is None
    assert math.isinf(mean_of_measurable([float("inf")] * 2))
