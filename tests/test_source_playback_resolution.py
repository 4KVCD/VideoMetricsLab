from dataclasses import replace
from pathlib import Path

from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings, comparison_dimensions
from vmaf_app.core.gstreamer_playback import output_caps_string
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import source_playback_comparison


def comparison():
    source = VideoInfo(path=Path("source.mkv"), width=3840, height=2160, fps=24,
                       duration=10, nb_frames=240, codec_name="hevc")
    encoded = replace(source, path=Path("encode.mkv"), width=1920, height=804)
    return FrameComparison(source, encoded, source_crop=CropBox(3840, 1608, 0, 276), fps=24)


def test_native_source_retains_cropped_4k_without_mutating_comparison():
    item = comparison()
    recipe = source_playback_comparison(item, True)
    assert comparison_dimensions(recipe) == (3840, 1608)
    assert comparison_dimensions(item) == (1920, 804)
    assert recipe.source_crop == item.source_crop
    assert recipe.source_info is item.source_info
    caps = output_caps_string(recipe, PreviewColorSettings(), "source")
    assert "width=3840,height=1608" in caps


def test_match_source_uses_cropped_encoded_dimensions():
    item = comparison()
    item = replace(item, distorted_info=replace(item.distorted_info, height=1080),
                   distorted_crop=CropBox(1920, 804, 0, 138))
    recipe = source_playback_comparison(item, False)
    assert comparison_dimensions(recipe) == (1920, 804)
    assert recipe.source_crop == CropBox(3840, 1608, 0, 276)


def test_match_preserves_aspect_ratio_and_never_upscales():
    item = comparison()
    taller_encode = replace(item, distorted_info=replace(item.distorted_info, height=1080))
    assert comparison_dimensions(source_playback_comparison(taller_encode, False)) == (1920, 804)
    bigger_encode = replace(item, distorted_info=replace(item.distorted_info, width=7680, height=4320))
    assert comparison_dimensions(source_playback_comparison(bigger_encode, False)) == (3840, 1608)


def test_matching_tracks_selected_encode_while_native_remains_constant():
    a = comparison()
    b = replace(a, distorted_info=replace(a.distorted_info, width=1280, height=536))
    assert comparison_dimensions(source_playback_comparison(a, False)) == (1920, 804)
    assert comparison_dimensions(source_playback_comparison(b, False)) == (1280, 536)
    assert comparison_dimensions(source_playback_comparison(a, True)) == comparison_dimensions(source_playback_comparison(b, True))
