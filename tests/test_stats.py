import math

import pytest

from vmaf_app.core.stats import SQUARE_MEAN_ROOT, aggregate_scores, compute_stats


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

    XPSNR scores a frame identical to the reference as +inf, and numpy's mean
    of any array containing inf is inf. The opening seconds of a feature are
    routinely pixel-identical black: across six real encodes of one film, 72
    to 175 frames of 151,919 were identical -- about 0.1%, and enough to make
    every XPSNR cell read "inf".
    """
    values = [float("inf")] * 40 + [78.78, 16.82, 12.56, 7.63, 4.60, 2.88, 3.28, 4.17, 5.87]

    stats = compute_stats(values, thresholds=[(">", 38.0), ("<", 33.0)],
                          aggregate=SQUARE_MEAN_ROOT)

    assert math.isfinite(stats.mean)
    assert stats.identical == 40
    assert stats.count == 49


def test_xpsnr_aggregates_the_way_ffmpeg_does():
    """XPSNR's sequence average is a square-mean-root, not a mean of decibels.

    ffmpeg's vf_xpsnr sums sqrt(wsse) per frame and derives one value from
    that total, so an identical frame contributes no error while still
    counting towards the frame total. Checked against ffmpeg's own printed
    average on the same clip, which it matches to four decimal places.
    """
    # Two frames, one perfect. sqrt-domain mean of 10^(-x/20) is
    # (0 + 10^(-40/20)) / 2 = 0.005, so -20*log10(0.005) = 46.0206 dB.
    assert aggregate_scores([float("inf"), 40.0], SQUARE_MEAN_ROOT) == pytest.approx(46.0206, abs=1e-4)
    # ...against 40.0 if the perfect frame were simply dropped, and inf if it
    # were included in an ordinary mean. Neither is what ffmpeg reports.
    assert aggregate_scores([float("inf"), 40.0]) == float("inf")

    # Without any infinities it is still a square-mean-root, which leans
    # towards the worse frame rather than treating decibels as linear: 50 and
    # 30 dB give 35.19, not their arithmetic 40.
    assert aggregate_scores([50.0, 30.0], SQUARE_MEAN_ROOT) == pytest.approx(35.1927, abs=1e-4)
    assert aggregate_scores([50.0, 30.0]) == pytest.approx(40.0)


def test_the_other_three_metrics_use_a_plain_mean():
    # VMAF and SSIM are bounded, and libvmaf clamps PSNR to its bit depth's
    # ceiling rather than reporting infinity -- verified over 4.2M frames of
    # real results, none of which contained one. Their means are unchanged.
    assert aggregate_scores([90.0, 92.0, 94.0]) == pytest.approx(92.0)
    assert aggregate_scores([0.99, 0.97]) == pytest.approx(0.98)


def test_identical_frames_still_count_towards_the_bands():
    # A perfect frame is emphatically "better than 38 dB". Aggregating around
    # it must not also exclude it from the tally.
    values = [float("inf")] * 3 + [40.0, 10.0]

    stats = compute_stats(values, thresholds=[(">", 38.0), ("<", 33.0)],
                          aggregate=SQUARE_MEAN_ROOT)

    above = next(t for t in stats.thresholds if t.label == "> 38")
    below = next(t for t in stats.thresholds if t.label == "< 33")
    assert above.count == 4 and above.percentage == pytest.approx(80.0)
    assert below.count == 1 and below.percentage == pytest.approx(20.0)


def test_an_entirely_identical_encode_still_reports_infinity():
    # No error to average. ffmpeg returns infinity here too.
    stats = compute_stats([float("inf")] * 5, aggregate=SQUARE_MEAN_ROOT)

    assert math.isinf(stats.mean)
    assert stats.identical == 5


def test_a_spread_is_still_reported_when_a_frame_was_identical():
    # Standard deviation is undefined over a set containing infinity (numpy
    # gives nan). The frames that actually differ still have a spread.
    stats = compute_stats([float("inf"), 40.0, 30.0], aggregate=SQUARE_MEAN_ROOT)

    assert math.isfinite(stats.stdev)
    assert stats.minimum == 30.0


def test_aggregate_scores_edge_cases():
    assert aggregate_scores([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert aggregate_scores([float("nan")] * 3) is None
    assert aggregate_scores([]) is None


def test_a_lower_is_better_metric_takes_its_worst_frames_from_the_top():
    from vmaf_app.core.metrics import MetricDirection

    values = list(range(1001))
    low = compute_stats(values, [])
    high = compute_stats(values, [], direction=MetricDirection.LOWER_IS_BETTER)
    assert (low.percentile_10, low.percentile_0_1) == (100.0, 1.0)
    assert (high.percentile_10, high.percentile_5, high.percentile_1, high.percentile_0_1) == (900.0, 950.0, 990.0, 999.0)
    assert [label for label, _ in high.values][5:] == ["10% High", "5% High", "1% High", "0.1% High"]
    assert [label for label, _ in low.values][5:] == ["10% Low", "5% Low", "1% Low", "0.1% Low"]
