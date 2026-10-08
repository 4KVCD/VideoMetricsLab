
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


def test_percentile_1_and_0_1_low_with_a_large_sample():
    # 1000 frames: 990 at 95, the worst 10 (1%) ramping down to a floor of 50.
    values = [95.0] * 990 + [50.0 + i for i in range(10)]
    stats = compute_stats(values)
    # 1% low should sit near the worst ~1% of frames, well below the bulk at 95.
    assert stats.percentile_1 < 95.0
    assert stats.percentile_0_1 <= stats.percentile_1  # 0.1% low is at least as extreme


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


def test_a_lower_is_better_metric_takes_its_worst_frames_from_the_top():
    from vmaf_app.core.metrics import MetricDirection

    values = list(range(1001))
    low = compute_stats(values, [])
    high = compute_stats(values, [], direction=MetricDirection.LOWER_IS_BETTER)
    assert (low.percentile_10, low.percentile_0_1) == (100.0, 1.0)
    assert (high.percentile_10, high.percentile_5, high.percentile_1, high.percentile_0_1) == (900.0, 950.0, 990.0, 999.0)
    assert [label for label, _ in high.values][5:] == ["10% High", "5% High", "1% High", "0.1% High"]
    assert [label for label, _ in low.values][5:] == ["10% Low", "5% Low", "1% Low", "0.1% Low"]
