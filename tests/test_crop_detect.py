from vmaf_app.core.crop_detect import _SAMPLE_WINDOW_SECONDS, _sample_offsets


def test_crop_samples_never_start_beyond_the_last_full_window():
    for duration in (1.0, 3.0, 5.0, 10.0, 60.0):
        latest_valid_start = max(0.0, duration - _SAMPLE_WINDOW_SECONDS)
        offsets = _sample_offsets(duration)
        assert offsets
        assert all(0.0 <= offset <= latest_valid_start for offset in offsets)


def test_very_short_clip_is_sampled_once_from_the_start():
    assert _sample_offsets(2.0) == [0.0]
