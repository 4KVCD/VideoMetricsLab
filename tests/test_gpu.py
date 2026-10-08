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


def test_only_4_2_0_at_8_or_10_bits_comes_back_from_ffmpegs_hardware_decode(subtests):
    """Checked with real FFmpeg on an RTX 5090: every other format failed in
    hwdownload, and the run started again in software, every run."""
    def check(pix_fmt, downloads):
        assert gpu.downloads_from_gpu(pix_fmt) is downloads

    for pix_fmt, downloads in [
        ("yuv420p", True), ("yuvj420p", True), ("yuv420p10le", True), ("nv12", True), ("p010le", True), ("", True),
        ("yuv420p12le", False), ("yuv422p", False), ("yuv422p10le", False), ("yuv444p", False),
        ("yuv444p10le", False), ("gbrp", False),
    ]:
        with subtests.test(pix_fmt=pix_fmt, downloads=downloads):
            check(pix_fmt, downloads)


def test_only_an_even_sized_video_comes_back_from_ffmpegs_hardware_decode_as_it_is(subtests):
    """Checked with real FFmpeg 9.0.1 on an RTX 5090, AV1 and VP9: an odd
    width or height came back padded to even (854x480 for 853x479), and
    with an odd height the chroma a row out -- 26 dB PSNR from the software
    decode, where the luma was identical."""
    def check(size, downloads):
        assert gpu.downloads_from_gpu("yuv420p", *size) is downloads

    for size, downloads in [
        ((854, 480), True), ((0, 0), True), ((853, 480), False), ((854, 479), False), ((853, 479), False),
    ]:
        with subtests.test(size=size, downloads=downloads):
            check(size, downloads)
