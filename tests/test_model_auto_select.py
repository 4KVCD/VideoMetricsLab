from vmaf_app.ui.main_window import _model_for_resolution


def test_1080p_uses_default_model():
    assert _model_for_resolution(1920, 1080) == "version=vmaf_v0.6.1"


def test_exact_4k_width_uses_4k_model():
    assert _model_for_resolution(3840, 2160) == "version=vmaf_4k_v0.6.1"


def test_ultrawide_4k_height_uses_4k_model():
    # e.g. a 2.35:1 UHD master cropped to content: width < 3840 but height hits 2160-class content
    assert _model_for_resolution(3840, 1634) == "version=vmaf_4k_v0.6.1"


def test_below_4k_threshold_uses_default_model():
    assert _model_for_resolution(2560, 1440) == "version=vmaf_v0.6.1"


def test_above_4k_uses_4k_model():
    assert _model_for_resolution(7680, 4320) == "version=vmaf_4k_v0.6.1"
