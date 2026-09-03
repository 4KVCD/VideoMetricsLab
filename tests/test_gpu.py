"""Which hardware decoder gets chosen, for each input independently."""
from __future__ import annotations

import pytest

from vmaf_app.core import gpu
from vmaf_app.core.gpu import HwAccelPlan, plan_hwaccel
from vmaf_app.core.models import GpuVendor


@pytest.fixture
def nvidia_only(monkeypatch):
    """An imaginary machine with an NVIDIA card and a cuda-capable ffmpeg."""
    monkeypatch.setattr(gpu, "available_hwaccels", lambda: {"cuda", "d3d11va"})
    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [GpuVendor.NVIDIA])


def test_both_inputs_are_accelerated_when_both_codecs_are_supported(nvidia_only):
    plan = plan_hwaccel(GpuVendor.NVIDIA, "hevc", "hevc")
    assert plan == HwAccelPlan(source="cuda", distorted="cuda")


def test_an_unsupported_distorted_codec_falls_back_to_cpu_on_its_own(nvidia_only):
    # THE case this exists for: the source is a decodable HEVC master and
    # the encode under test is in a format the GPU can't handle. The source
    # must keep its hardware decode rather than the whole run dropping to
    # software because of the other file.
    plan = plan_hwaccel(GpuVendor.NVIDIA, "hevc", "prores")

    assert plan.source == "cuda"
    assert plan.distorted is None


def test_an_unsupported_source_codec_does_not_disable_the_distorted_input(nvidia_only):
    plan = plan_hwaccel(GpuVendor.NVIDIA, "prores", "h264")

    assert plan.source is None
    assert plan.distorted == "cuda"


def test_neither_input_is_accelerated_when_gpu_decoding_is_off(nvidia_only):
    plan = plan_hwaccel(GpuVendor.NONE, "hevc", "hevc")
    assert plan == HwAccelPlan()
    assert not plan.uses_gpu


def test_a_round_trip_test_has_no_distorted_side_to_decide(nvidia_only):
    # run_resample_test decodes one file and splits it in the filtergraph,
    # so there is no second input to plan for.
    plan = plan_hwaccel(GpuVendor.NVIDIA, "hevc")
    assert plan == HwAccelPlan(source="cuda", distorted=None)


def test_a_codec_the_installed_ffmpeg_cannot_accelerate_is_not_selected(monkeypatch):
    # The vendor's preferred hwaccel has to actually be built into this
    # ffmpeg. Asking for one that isn't makes ffmpeg fail to launch at all,
    # which the run-level fallback would then have to absorb.
    monkeypatch.setattr(gpu, "available_hwaccels", lambda: set())
    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [GpuVendor.NVIDIA])

    assert plan_hwaccel(GpuVendor.NVIDIA, "hevc", "hevc") == HwAccelPlan()


def test_auto_picks_the_detected_vendor_for_both_inputs(nvidia_only):
    assert plan_hwaccel(GpuVendor.AUTO, "h264", "av1") == HwAccelPlan(
        source="cuda", distorted="cuda"
    )


def test_the_description_names_which_input_got_hardware_decode():
    # A per-input fallback is silent otherwise: the run just gets slower,
    # and looks the same as one that never attempted the GPU.
    assert HwAccelPlan().describe() == "off"
    assert "distorted cpu" in HwAccelPlan(source="cuda").describe()
    assert "source cpu" in HwAccelPlan(distorted="qsv").describe()
