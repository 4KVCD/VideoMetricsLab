"""A comparison that stopped long before the videos' lengths say: a file cut short."""
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.frame_coverage import short_comparison
from vmaf_app.core.models import CropMode, FrameScores, VideoInfo, VmafOptions


def _info(path: str) -> VideoInfo:
    return VideoInfo(Path(path), 1920, 1080, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_a_frame_or_two_short_is_the_whole_video():
    assert short_comparison(240, 240, 24.0) is None
    assert short_comparison(240, 239, 24.0) is None  # one file a frame shorter: libvmaf's shortest=1
    assert short_comparison(240, 228, 24.0) is None  # within half a second
    assert short_comparison(240, 232, 24.0, step=5) is None  # every 5th frame scored


def test_a_file_cut_short_is_named_with_where_its_pictures_end():
    message = short_comparison(240, 5, 24.0)
    assert message.startswith("Only 5 of the 240 frames expected could be compared")
    assert "ends after 0:00:00.2" in message and "cut short" in message
    assert short_comparison(172_000, 171_000, 24.0) is not None  # a film missing its last 40 seconds


def test_a_run_whose_test_video_ends_early_fails_instead_of_scoring_what_it_had(monkeypatch):
    """A truncated 10-second encode, five frames of picture, was given VMAF 98."""
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: FrameScores(
        np.arange(5), np.arange(5) / 24.0, vmaf=np.full(5, 98.0)))
    with pytest.raises(vr.VmafRunError, match="Only 5 of the 240 frames"):
        vr.run_vmaf(_info("s.mkv"), _info("d.mkv"),
                    VmafOptions(crop_mode=CropMode.NONE, gpu_decode=False, vmaf_on_gpu=False))


def test_a_cut_short_vship_pass_is_not_retried_on_the_cpu(monkeypatch):
    """The file's fault, not the GPU's: the CPU would stop just as short."""
    from tests import test_perceptual_vship as vship_tests
    from vmaf_app.core import perceptual_vship
    from vmaf_app.core.perceptual_cpu import ComparisonCutShortError

    monkeypatch.setattr(perceptual_vship, "short_comparison", short_comparison)
    with pytest.raises(ComparisonCutShortError, match="Only 2 of the 24 frames"):
        vship_tests._run(monkeypatch, metrics=("ssimulacra2",),
                         children=vship_tests._both(vship_tests._frames_command(2, vship_tests._FRAME_BYTES)))
