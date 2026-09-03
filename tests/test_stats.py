import pytest

from vmaf_app.core.stats import compute_stats


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


