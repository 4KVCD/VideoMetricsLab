"""Model selection: turning a row's model *choice* into the concrete ffmpeg
`model=` value, including 4K auto-selection. Pure logic, no Qt -- these used
to live in test_main_window.py and reach into the UI module for it."""
from pathlib import Path

import pytest

from vmaf_app.core.model_select import (
    AUTO_MODEL_CHOICE,
    CUSTOM_MODEL_CHOICE,
    DEFAULT_MODEL,
    UHD_MODEL,
    model_for_resolution,
    resolve_model,
)
from vmaf_app.core.models import ScaleDirection, VideoInfo, VmafOptions, clone_options


def _fake_video_info(name: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(
        path=Path(name), width=width, height=height, fps=30.0, duration=5.0,
        nb_frames=150, codec_name="h264",
    )


# ------------------------------------------------------- resolution -> model

def test_1080p_uses_default_model():
    assert model_for_resolution(1920, 1080) == "version=vmaf_v0.6.1"


def test_exact_4k_width_uses_4k_model():
    assert model_for_resolution(3840, 2160) == "version=vmaf_4k_v0.6.1"


def test_ultrawide_4k_height_uses_4k_model():
    # e.g. a 2.35:1 UHD master cropped to content: width < 3840 but height hits 2160-class content
    assert model_for_resolution(3840, 1634) == "version=vmaf_4k_v0.6.1"


def test_below_4k_threshold_uses_default_model():
    assert model_for_resolution(2560, 1440) == "version=vmaf_v0.6.1"


def test_above_4k_uses_4k_model():
    assert model_for_resolution(7680, 4320) == "version=vmaf_4k_v0.6.1"


# ------------------------------------------------- choice -> ffmpeg model=

def test_resolve_model_auto_picks_4k_for_uhd_distorted():
    opts = VmafOptions(model_choice="__auto__")
    assert resolve_model(opts, 1920, 1080) == "version=vmaf_v0.6.1"

    assert resolve_model(opts, 3840, 2160) == "version=vmaf_4k_v0.6.1"


def test_resolve_model_fixed_choice_passes_through():
    opts = VmafOptions(model_choice="version=vmaf_4k_v0.6.1")
    assert resolve_model(opts, 1920, 1080) == "version=vmaf_4k_v0.6.1"


def test_resolve_model_custom_requires_a_path():
    opts = VmafOptions(model_choice="__custom__", custom_model_path=None)
    with pytest.raises(ValueError):
        resolve_model(opts, 1920, 1080)

    opts2 = VmafOptions(model_choice="__custom__", custom_model_path="C:/models/mine.json")
    assert resolve_model(opts2, 1920, 1080) == "path=C:/models/mine.json"


def test_clone_options_is_an_independent_copy():
    original = VmafOptions(extra_features=["name=psnr"])
    copy = clone_options(original)
    copy.extra_features.append("name=float_ssim")
    copy.n_threads = 99

    assert original.extra_features == ["name=psnr"]
    assert original.n_threads == 0


# ------------------------------------------------- analysis-resolution model

def _sized(name, w, h):
    return VideoInfo(
        path=Path(name), width=w, height=h, fps=30.0, duration=10.0,
        nb_frames=300, codec_name="h264", pix_fmt="yuv420p",
    )


@pytest.mark.parametrize(
    ("source_wh", "distorted_wh", "direction", "expected_size", "expected_model"),
    [
        # One side is scaled to the other before libvmaf sees it, so the
        # model must follow the comparison, not either input's own size.
        ((3840, 2160), (1920, 1080), ScaleDirection.DISTORTED_TO_SOURCE, (3840, 2160), UHD_MODEL),
        ((3840, 2160), (1920, 1080), ScaleDirection.SOURCE_TO_DISTORTED, (1920, 1080), DEFAULT_MODEL),
        ((1920, 1080), (3840, 2160), ScaleDirection.SOURCE_TO_DISTORTED, (3840, 2160), UHD_MODEL),
        ((1920, 1080), (3840, 2160), ScaleDirection.DISTORTED_TO_SOURCE, (1920, 1080), DEFAULT_MODEL),
        # Same size on both sides: nothing is scaled.
        ((3840, 2160), (3840, 2160), ScaleDirection.SOURCE_TO_DISTORTED, (3840, 2160), UHD_MODEL),
    ],
)
def test_auto_follows_the_resolution_frames_are_compared_at(
    source_wh, distorted_wh, direction, expected_size, expected_model
):
    from vmaf_app.core.vmaf_runner import analysis_dimensions

    options = VmafOptions(model_choice=AUTO_MODEL_CHOICE, scale_direction=direction)
    source = _sized("source.mkv", *source_wh)
    distorted = _sized("encode.mkv", *distorted_wh)

    size = analysis_dimensions(source, distorted, options)

    assert size == expected_size
    assert resolve_model(options, *size) == expected_model


def test_cropping_is_part_of_the_analysis_size():
    from vmaf_app.core.models import CropBox
    from vmaf_app.core.vmaf_runner import analysis_dimensions

    options = VmafOptions(
        model_choice=AUTO_MODEL_CHOICE,
        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
    )
    source = _sized("source.mkv", 3840, 2160)
    distorted = _sized("encode.mkv", 1920, 800)

    # Letterbox cropped off the 4K master: still 4K wide, so still the 4K
    # model -- the width threshold is what carries this case.
    size = analysis_dimensions(
        source, distorted, options, CropBox(w=3840, h=1600, x=0, y=280), None
    )
    assert size == (3840, 1600)
    assert resolve_model(options, *size) == UHD_MODEL


def test_a_round_trip_test_is_analysed_at_the_sources_own_size():
    from vmaf_app.core.models import CropBox
    from vmaf_app.core.vmaf_runner import resample_analysis_dimensions

    source = _sized("master.mkv", 3840, 2160)
    options = VmafOptions(model_choice=AUTO_MODEL_CHOICE)

    # The downscale is undone before comparison, so a 1080p round-trip test
    # of a 4K master is still a 4K comparison.
    assert resample_analysis_dimensions(source) == (3840, 2160)
    assert resolve_model(options, *resample_analysis_dimensions(source)) == UHD_MODEL
    assert resample_analysis_dimensions(source, CropBox(w=3840, h=1600, x=0, y=280)) == (3840, 1600)


def test_an_explicit_or_custom_model_is_never_second_guessed():
    from vmaf_app.core.vmaf_runner import _auto_model_or

    explicit = VmafOptions(model_choice=DEFAULT_MODEL, model=DEFAULT_MODEL)
    assert _auto_model_or(explicit, (3840, 2160)) == DEFAULT_MODEL

    custom = VmafOptions(model_choice=CUSTOM_MODEL_CHOICE, model="path=mine.json")
    assert _auto_model_or(custom, (3840, 2160)) == "path=mine.json"


def test_a_run_records_the_model_it_actually_used(monkeypatch):
    # The result carries the model for display and for reloading, so it must
    # be the one that ran, not the provisional one the UI guessed.
    from vmaf_app.core import vmaf_runner

    monkeypatch.setattr(vmaf_runner, "_resolve_crops", lambda *a, **k: (None, None))
    monkeypatch.setattr(vmaf_runner, "validate_display_geometry", lambda *a, **k: None)
    monkeypatch.setattr(
        vmaf_runner, "_execute_run", lambda *a, **k: vmaf_runner.FrameScores.empty()
    )

    options = VmafOptions(
        model_choice=AUTO_MODEL_CHOICE, model=DEFAULT_MODEL,
        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
    )
    result = vmaf_runner.run_vmaf(
        _sized("source.mkv", 3840, 2160), _sized("encode.mkv", 1920, 1080), options
    )

    assert result.model == UHD_MODEL, "the result claims a model the run did not use"
