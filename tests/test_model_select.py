"""Model selection: turning a row's model *choice* into the concrete ffmpeg
`model=` value, including 4K auto-selection. Pure logic, no Qt -- these used
to live in test_main_window.py and reach into the UI module for it."""
from pathlib import Path

import pytest

from vmaf_app.core.model_select import model_for_resolution, resolve_model
from vmaf_app.core.models import VideoInfo, VmafOptions, clone_options


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
    assert resolve_model(opts, _fake_video_info("d.mp4")) == "version=vmaf_v0.6.1"

    uhd_info = VideoInfo(path=Path("d.mp4"), width=3840, height=2160, fps=30.0, duration=5.0, nb_frames=150, codec_name="hevc")
    assert resolve_model(opts, uhd_info) == "version=vmaf_4k_v0.6.1"


def test_resolve_model_fixed_choice_passes_through():
    opts = VmafOptions(model_choice="version=vmaf_v0.6.1neg")
    assert resolve_model(opts, _fake_video_info("d.mp4")) == "version=vmaf_v0.6.1neg"


def test_resolve_model_custom_requires_a_path():
    opts = VmafOptions(model_choice="__custom__", custom_model_path=None)
    with pytest.raises(ValueError):
        resolve_model(opts, _fake_video_info("d.mp4"))

    opts2 = VmafOptions(model_choice="__custom__", custom_model_path="C:/models/mine.json")
    assert resolve_model(opts2, _fake_video_info("d.mp4")) == "path=C:/models/mine.json"


def test_clone_options_is_an_independent_copy():
    original = VmafOptions(extra_features=["name=psnr"])
    copy = clone_options(original)
    copy.extra_features.append("name=float_ssim")
    copy.n_threads = 99

    assert original.extra_features == ["name=psnr"]
    assert original.n_threads == 0
