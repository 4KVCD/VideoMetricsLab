"""Model selection: turning a row's model *choice* into the concrete ffmpeg
`model=` value, including 4K auto-selection. Pure logic, no Qt -- these used
to live in test_main_window.py and reach into the UI module for it."""
from pathlib import Path

from vmaf_app.core.builtin_models import builtin_choice, builtin_model_path
from vmaf_app.core.model_select import (
    AUTO_MODEL_CHOICE,
    CUSTOM_MODEL_CHOICE,
    DEFAULT_MODEL,
    UHD_MODEL,
    resolve_model,
)
from vmaf_app.core.models import ScaleDirection, VideoInfo, VmafOptions


def _fake_video_info(name: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(
        path=Path(name), width=width, height=height, fps=30.0, duration=5.0,
        nb_frames=150, codec_name="h264",
    )


# ------------------------------------------------- choice -> ffmpeg model=


def test_bundled_vmaf_v1_model_resolves_to_an_existing_path():
    choice = builtin_choice("vmaf_v1_3d0h")
    resolved = resolve_model(VmafOptions(model_choice=choice), 1920, 1080)
    assert resolved == f"path={builtin_model_path('vmaf_v1_3d0h')}"


# ------------------------------------------------- analysis-resolution model

def _sized(name, w, h):
    return VideoInfo(
        path=Path(name), width=w, height=h, fps=30.0, duration=10.0,
        nb_frames=300, codec_name="h264", pix_fmt="yuv420p",
    )


def test_auto_follows_the_resolution_frames_are_compared_at(subtests):
    def check(source_wh, distorted_wh, direction, expected_size, expected_model):
        from vmaf_app.core.geometry import analysis_dimensions

        options = VmafOptions(model_choice=AUTO_MODEL_CHOICE, scale_direction=direction)
        source = _sized("source.mkv", *source_wh)
        distorted = _sized("encode.mkv", *distorted_wh)

        size = analysis_dimensions(source, distorted, options)

        assert size == expected_size
        assert resolve_model(options, *size) == expected_model

    for source_wh, distorted_wh, direction, expected_size, expected_model in [
            # One side is scaled to the other before libvmaf sees it, so the
            # model must follow the comparison, not either input's own size.
            ((3840, 2160), (1920, 1080), ScaleDirection.DISTORTED_TO_SOURCE, (3840, 2160), UHD_MODEL),
            ((3840, 2160), (1920, 1080), ScaleDirection.SOURCE_TO_DISTORTED, (1920, 1080), DEFAULT_MODEL),
            ((1920, 1080), (3840, 2160), ScaleDirection.SOURCE_TO_DISTORTED, (3840, 2160), UHD_MODEL),
            ((1920, 1080), (3840, 2160), ScaleDirection.DISTORTED_TO_SOURCE, (1920, 1080), DEFAULT_MODEL),
            # Same size on both sides: nothing is scaled.
            ((3840, 2160), (3840, 2160), ScaleDirection.SOURCE_TO_DISTORTED, (3840, 2160), UHD_MODEL),
        ]:
        with subtests.test(source_wh=source_wh, distorted_wh=distorted_wh, direction=direction, expected_size=expected_size, expected_model=expected_model):
            check(source_wh, distorted_wh, direction, expected_size, expected_model)


def test_cropping_is_part_of_the_analysis_size():
    from vmaf_app.core.geometry import analysis_dimensions
    from vmaf_app.core.models import CropBox

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
    monkeypatch.setattr(vmaf_runner, "short_comparison", lambda *a, **k: None)  # no frames: a fake

    options = VmafOptions(
        model_choice=AUTO_MODEL_CHOICE, model=DEFAULT_MODEL,
        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
    )
    result = vmaf_runner.run_vmaf(
        _sized("source.mkv", 3840, 2160), _sized("encode.mkv", 1920, 1080), options
    )

    assert result.model == UHD_MODEL, "the result claims a model the run did not use"
