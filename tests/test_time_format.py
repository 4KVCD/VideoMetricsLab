from vmaf_app.core.time_format import format_hms


def test_whole_seconds_no_decimals():
    assert format_hms(0) == "0:00:00"
    assert format_hms(65) == "0:01:05"
    assert format_hms(3725) == "1:02:05"


def test_with_decimals():
    assert format_hms(2.31, decimals=2) == "0:00:02.31"
    assert format_hms(3725.4, decimals=1) == "1:02:05.4"


def test_rounding_carries_into_the_minutes_and_hours():
    assert format_hms(59.96, decimals=1) == "0:01:00.0"
    assert format_hms(3599.999, decimals=2) == "1:00:00.00"
    assert format_hms(59.94, decimals=1) == "0:00:59.9"
