from vmaf_app.core.time_format import format_hms


def test_whole_seconds_no_decimals():
    assert format_hms(0) == "0:00:00"
    assert format_hms(65) == "0:01:05"
    assert format_hms(3725) == "1:02:05"


def test_with_decimals():
    assert format_hms(2.31, decimals=2) == "0:00:02.31"
    assert format_hms(3725.4, decimals=1) == "1:02:05.4"


def test_negative_clamped_to_zero():
    assert format_hms(-5) == "0:00:00"


def test_hours_always_shown_even_when_zero():
    assert format_hms(30) == "0:00:30"


def test_large_duration_hours_not_truncated():
    assert format_hms(3600 * 25 + 61) == "25:01:01"
