"""Which hardware decoder gets chosen, for each input independently."""
from __future__ import annotations

import sys

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


def test_the_hwaccel_list_follows_the_ffmpeg_in_use(monkeypatch):
    """The list was kept for the session from whichever FFmpeg answered
    first, through a change of FFmpeg folder."""
    asked = []

    def run(cmd, **_kwargs):
        asked.append(cmd[0])
        listing = {"a/ffmpeg": "cuda", "b/ffmpeg": "qsv"}[cmd[0]]
        return type("Done", (), {"stdout": f"Hardware acceleration methods:\n{listing}\n"})()

    gpu._hwaccels_of.cache_clear()
    monkeypatch.setattr(gpu.proc_util, "run", run)
    monkeypatch.setattr(gpu, "ffmpeg_path", lambda: "a/ffmpeg")
    assert gpu.available_hwaccels() == {"cuda"}
    assert gpu.available_hwaccels() == {"cuda"}
    monkeypatch.setattr(gpu, "ffmpeg_path", lambda: "b/ffmpeg")
    assert gpu.available_hwaccels() == {"qsv"}
    assert asked == ["a/ffmpeg", "b/ffmpeg"]
    gpu._hwaccels_of.cache_clear()


def test_gpu_makers_come_from_directx_in_the_order_auto_tries_them(monkeypatch):
    """Intel's integrated GPU first in DirectX's list, a software adapter
    (Microsoft's, 0x1414) last: NVIDIA's decoder is still tried first."""
    gpu.detected_gpu_vendors.cache_clear()
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu, "_dxgi_vendor_ids", lambda: [0x8086, 0x10DE, 0x1414])
    try:
        assert gpu.detected_gpu_vendors() == [GpuVendor.NVIDIA, GpuVendor.INTEL]
    finally:
        gpu.detected_gpu_vendors.cache_clear()


def test_no_directx_means_no_gpu_maker(monkeypatch):
    def unavailable():
        raise OSError("CreateDXGIFactory1 failed")

    gpu.detected_gpu_vendors.cache_clear()
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu, "_dxgi_vendor_ids", unavailable)
    try:
        assert gpu.detected_gpu_vendors() == []
    finally:
        gpu.detected_gpu_vendors.cache_clear()


@pytest.mark.skipif(sys.platform != "win32", reason="DirectX")
def test_directx_lists_this_machines_adapters():
    vendor_ids = gpu._dxgi_vendor_ids()
    assert all(isinstance(vendor_id, int) and 0 < vendor_id < 0x10000 for vendor_id in vendor_ids)


@pytest.mark.parametrize(("pix_fmt", "downloads"), [
    ("yuv420p", True), ("yuvj420p", True), ("yuv420p10le", True), ("nv12", True), ("p010le", True), ("", True),
    ("yuv420p12le", False), ("yuv422p", False), ("yuv422p10le", False), ("yuv444p", False),
    ("yuv444p10le", False), ("gbrp", False),
])
def test_only_4_2_0_at_8_or_10_bits_comes_back_from_ffmpegs_hardware_decode(pix_fmt, downloads):
    """Checked with real FFmpeg on an RTX 5090: every other format failed in
    hwdownload, and the run started again in software, every run."""
    assert gpu.downloads_from_gpu(pix_fmt) is downloads


@pytest.mark.parametrize(("size", "downloads"), [
    ((854, 480), True), ((0, 0), True), ((853, 480), False), ((854, 479), False), ((853, 479), False),
])
def test_only_an_even_sized_video_comes_back_from_ffmpegs_hardware_decode_as_it_is(size, downloads):
    """Checked with real FFmpeg 9.0.1 on an RTX 5090, AV1 and VP9: an odd
    width or height came back padded to even (854x480 for 853x479), and
    with an odd height the chroma a row out -- 26 dB PSNR from the software
    decode, where the luma was identical."""
    assert gpu.downloads_from_gpu("yuv420p", *size) is downloads


def test_an_odd_sized_video_is_planned_in_software(monkeypatch):
    monkeypatch.setattr(gpu, "available_hwaccels", lambda: {"cuda"})
    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [GpuVendor.NVIDIA])

    plan = plan_hwaccel(GpuVendor.AUTO, "av1", "av1", source_pix_fmt="yuv420p", distorted_pix_fmt="yuv420p",
                        source_size=(1920, 1080), distorted_size=(1920, 803))

    assert plan == HwAccelPlan(source="cuda", distorted=None)


def test_a_format_ffmpegs_hardware_decode_cannot_give_is_planned_in_software(monkeypatch):
    monkeypatch.setattr(gpu, "available_hwaccels", lambda: {"cuda"})
    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: [GpuVendor.NVIDIA])

    plan = plan_hwaccel(GpuVendor.AUTO, "hevc", "hevc", source_pix_fmt="yuv444p10le", distorted_pix_fmt="yuv420p10le")

    assert plan == HwAccelPlan(source=None, distorted="cuda")
