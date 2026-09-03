import os
import subprocess
from pathlib import Path

import pytest

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import (
    CropBox,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
    synthetic_resample_distorted_path,
)
from vmaf_app.core.vmaf_runner import (
    _bit_depth,
    _build_ffmpeg_cmd,
    _build_filtergraph,
    _build_resample_cmd,
    _build_resample_test_filtergraph,
    _fallback_ladder,
    _hw_native_format,
    analysis_pix_fmt,
)


def _info(path, w, h, pix_fmt="yuv420p") -> VideoInfo:
    return VideoInfo(
        path=Path(path), width=w, height=h, fps=30.0, duration=10.0, nb_frames=300,
        codec_name="h264", pix_fmt=pix_fmt,
    )


def test_scales_reference_to_distorted_resolution_when_they_differ():
    source_info = _info("source.mov", 3840, 2160)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options,
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
        log_path=Path("log.json"),
    )

    assert "scale=1920:1080" in graph
    assert "[main][ref]libvmaf=" in graph


def test_source_is_downscaled_not_distorted_upscaled_when_distorted_is_lower_res():
    # The reference/source chain is [1:v]...[ref]; the distorted/main chain
    # is [0:v]...[main]. When distorted is lower-res, the scale filter must
    # land in the [ref] (source) chain -- scaling the source DOWN to match --
    # never in the [main] (distorted) chain, which would upscale distorted
    # instead and inflate the score by comparing against a blurrier source
    # than what was actually delivered.
    source_info = _info("source.mov", 3840, 2160)  # 4K source
    distorted_info = _info("distorted.mp4", 1280, 720)  # 720p distorted -- much lower res

    graph = _build_filtergraph(
        source_info, distorted_info, VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    main_chain, ref_chain, _ = graph.split(";")
    assert main_chain.startswith("[0:v]")
    assert ref_chain.startswith("[1:v]")
    assert "scale=1280:720" not in main_chain  # distorted is NOT upscaled
    assert "scale=1280:720" in ref_chain  # source IS downscaled to match distorted


def test_upscale_distorted_mode_scales_distorted_up_to_source_resolution():
    source_info = _info("source.mov", 3840, 2160)  # 4K source
    distorted_info = _info("distorted.mp4", 1920, 1080)  # 1080p distorted

    graph = _build_filtergraph(
        source_info, distorted_info,
        VmafOptions(model="version=vmaf_v0.6.1", scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    main_chain, ref_chain, _ = graph.split(";")
    assert "scale=3840:2160" in main_chain  # distorted IS upscaled to the source's resolution
    assert "scale=" not in ref_chain  # source is left untouched


def test_upscale_distorted_mode_is_a_noop_when_resolutions_already_match():
    info_a = _info("source.mov", 1920, 1080)
    info_b = _info("distorted.mp4", 1920, 1080)

    graph = _build_filtergraph(
        info_a, info_b, VmafOptions(model="version=vmaf_v0.6.1", scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    assert "scale=" not in graph


def test_no_scale_filter_when_resolutions_already_match():
    info_a = _info("source.mov", 1920, 1080)
    info_b = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        info_a, info_b, options, source_crop=None, distorted_crop=None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    assert "scale=" not in graph


def test_crop_filters_applied_and_scale_targets_cropped_distorted_dims():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    # Source is 16:9 with 2.35:1 content letterboxed; distorted has been cropped to 2.35:1.
    source_crop = CropBox(w=1920, h=817, x=0, y=131)
    distorted_crop = CropBox(w=1920, h=817, x=0, y=0)

    graph = _build_filtergraph(
        source_info, distorted_info, options, source_crop, distorted_crop,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    assert "crop=1920:817:0:131" in graph  # source crop
    assert "crop=1920:817:0:0" in graph  # distorted crop
    # both crops already match -> no rescale needed
    assert "scale=" not in graph


def test_noop_crop_is_skipped():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    full_frame_crop = CropBox(w=1920, h=1080, x=0, y=0)

    graph = _build_filtergraph(
        source_info, distorted_info, options, full_frame_crop, full_frame_crop,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    assert "crop=" not in graph


def test_hwdownload_inserted_when_gpu_decode_used():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(source="cuda"), log_path=Path("log.json"),
    )

    assert "[1:v]hwdownload,format=nv12,format=yuv420p" in graph


def test_hwdownload_uses_p010_for_10bit_source():
    # UHD/HDR masters are almost always 10-bit HEVC; NVDEC decodes these to a
    # p010 surface, not nv12 -- forcing nv12 here previously broke GPU decode
    # for exactly this common case.
    source_info = _info("source.mov", 3840, 2160, pix_fmt="yuv420p10le")
    distorted_info = _info("distorted.mp4", 3840, 2160)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(source="cuda"), log_path=Path("log.json"),
    )

    assert "[1:v]hwdownload,format=p010le,format=yuv420p" in graph


def test_default_n_threads_resolves_to_cpu_count_not_omitted():
    # libvmaf 2.0+ defaults to n_threads=1 (single-threaded) when this option
    # is left unset, so "Auto" (n_threads<=0 in our options) must still emit
    # an explicit value -- omitting it silently serializes the whole run.
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", n_threads=0)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    libvmaf_part = graph.split("libvmaf=", 1)[1]
    assert f"n_threads={os.cpu_count() or 1}" in libvmaf_part


def test_duration_limit_adds_output_side_t_flag():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf", hwaccel=HwAccelPlan(), duration_limit=30.0,
    )
    # -t must come after -lavfi (an output option, bounding the whole
    # filtered output) not before either -i (which would just be misplaced).
    lavfi_idx = cmd.index("-lavfi")
    t_idx = cmd.index("-t")
    assert t_idx > lavfi_idx
    assert cmd[t_idx + 1] == "30.000"


def test_no_duration_limit_omits_t_flag_by_default():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf", hwaccel=HwAccelPlan(),
    )
    assert "-t" not in cmd


def test_hw_native_format():
    assert _hw_native_format("yuv420p") == "nv12"
    assert _hw_native_format("yuv420p10le") == "p010le"
    assert _hw_native_format("yuv420p12le") == "p010le"
    assert _hw_native_format("") == "nv12"


def test_libvmaf_options_include_model_threads_subsample_and_features():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(
        model="version=vmaf_4k_v0.6.1", n_threads=8, n_subsample=2,
        extra_features=["name=psnr", "name=float_ssim"],
    )

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    libvmaf_part = graph.split("libvmaf=", 1)[1]
    assert "model=version=vmaf_4k_v0.6.1" in libvmaf_part
    assert "n_threads=8" in libvmaf_part
    assert "n_subsample=2" in libvmaf_part
    assert "feature=name=psnr|name=float_ssim" in libvmaf_part


# ------------------------------------------------------------------ resolution round-trip test

def test_synthetic_resample_path_is_unique_per_target_resolution():
    source = Path("C:/videos/MyMovie.mkv")
    p1080 = synthetic_resample_distorted_path(source, ResampleTarget(width=1920, label="1080p"))
    p720 = synthetic_resample_distorted_path(source, ResampleTarget(width=1280, label="720p"))

    assert p1080 != p720
    assert "1080p" in p1080.name
    assert "720p" in p720.name
    assert p1080.suffix == ".mkv"
    # same source + same target -> same path every time, so caching/dedup by
    # this identity is stable and repeatable.
    assert p1080 == synthetic_resample_distorted_path(source, ResampleTarget(width=1920, label="1080p"))


def test_resample_filtergraph_is_single_input_split_into_two_branches():
    source_info = _info("source.mkv", 3840, 2160)
    options = VmafOptions(model="version=vmaf_v0.6.1", resample_test=ResampleTarget(width=1920, label="1080p"))

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    assert "[1:v]" not in graph  # only one input -- everything derives from [0:v]
    assert "[0:v]" in graph
    assert "split=2" in graph
    assert "[main][ref]libvmaf=" in graph


def test_resample_downscales_then_upscales_back_to_source_resolution():
    source_info = _info("source.mkv", 3840, 2160)
    options = VmafOptions(
        model="version=vmaf_v0.6.1", scale_algorithm="lanczos",
        resample_test=ResampleTarget(width=1920, label="1080p"),
    )

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    dist_chain = graph.split(";")[3]  # base;split;ref;dist;libvmaf
    assert dist_chain.startswith("[dist_src]")
    # down to 1920x1080 (preserving the 3840x2160 source's 16:9 AR), then
    # back up to the source's original 3840x2160 -- never straight to 1080p.
    assert "scale=1920:1080:flags=lanczos" in dist_chain
    assert "scale=3840:2160:flags=lanczos" in dist_chain
    assert dist_chain.index("scale=1920:1080") < dist_chain.index("scale=3840:2160")


def test_resample_downscale_height_preserves_non_16_9_aspect_ratio():
    # A 2.35:1 source (already cropped, e.g. via source_crop) -- the
    # downscale height must preserve THIS aspect ratio, not assume 16:9.
    source_info = _info("source.mkv", 3840, 1634)  # ~2.35:1
    options = VmafOptions(model="version=vmaf_v0.6.1", resample_test=ResampleTarget(width=1920, label="1080p"))

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    dist_chain = graph.split(";")[3]
    # 1634 * (1920/3840) = 817, rounded to even -> 816 or 818
    assert "scale=1920:816" in dist_chain or "scale=1920:818" in dist_chain


def test_resample_applies_source_crop_before_the_split():
    source_info = _info("source.mkv", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", resample_test=ResampleTarget(width=960, label="480p"))
    crop = CropBox(w=1920, h=816, x=0, y=132)

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=crop, hwaccel_used=None, log_path=Path("log.json"),
    )

    base_chain = graph.split(";")[0]
    assert "crop=1920:816:0:132" in base_chain
    # the downscale/upscale target dimensions are based on the CROPPED
    # content (1920x816), not the raw 1920x1080 frame.
    dist_chain = graph.split(";")[3]
    assert "scale=1920:816" in dist_chain


def test_resample_cmd_has_a_single_input_video():
    cmd = _build_resample_cmd(Path("source.mkv"), "[0:v]...", hwaccel=None)
    assert cmd.count("-i") == 1


# ------------------------------------------------------------------ XPSNR

def test_xpsnr_not_requested_by_default():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr.txt"),
    )

    assert "xpsnr" not in graph
    assert "[main][ref]libvmaf=" in graph


def test_xpsnr_stage_sits_between_decode_and_libvmaf():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=True)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )

    assert "[main][ref_xpsnr]xpsnr=stats_file=xpsnr_log.txt[xmain]" in graph
    assert "[xmain][ref_vmaf]libvmaf=" in graph  # libvmaf consumes xpsnr's passthrough output, not [main] directly


def test_xpsnr_splits_the_reference_so_libvmaf_still_gets_its_own_copy():
    # Regression test for a silent, severe correctness bug: xpsnr consumes
    # [ref], and a filtergraph label can only be consumed once. Reusing
    # [ref] for libvmaf too made ffmpeg wire libvmaf up to the wrong stream
    # -- it compared the distorted video against ITSELF and reported a
    # perfect VMAF 100 / PSNR 60 / SSIM 1.0 for every frame, without any
    # error, no matter how bad the encode really was.
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=True)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )

    assert "[ref]split=2[ref_xpsnr][ref_vmaf]" in graph
    # The bare [ref] label must be consumed exactly once (by the split), and
    # never handed to two filters.
    assert graph.count("[ref]") == 2  # once produced by the decode chain, once consumed by the split
    assert "[main][ref]xpsnr" not in graph
    assert "[xmain][ref]libvmaf" not in graph


def test_xpsnr_reference_split_also_applies_to_resample_tests():
    source_info = _info("source.mov", 1920, 1080)
    options = VmafOptions(
        model="version=vmaf_v0.6.1", compute_xpsnr=True, resample_test=ResampleTarget(width=960, label="480p"),
    )

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None,
        log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )

    assert "[ref]split=2[ref_xpsnr][ref_vmaf]" in graph
    assert "[xmain][ref_vmaf]libvmaf=" in graph
    assert "xpsnr=stats_file=xpsnr_log.txt" in graph


def test_xpsnr_requested_but_no_log_path_is_a_noop():
    # Defensive: compute_xpsnr=True with no path given (shouldn't happen via
    # the UI, but the filtergraph builder must not silently reference a
    # nonexistent file) skips the xpsnr stage rather than erroring.
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=True)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"), xpsnr_log_path=None,
    )

    assert "xpsnr" not in graph



def test_parse_xpsnr_log_converts_1_indexed_to_0_indexed_frames(tmp_path):
    from vmaf_app.core.vmaf_runner import _parse_xpsnr_log

    log_path = tmp_path / "xpsnr_log.txt"
    log_path.write_text(
        "n:    1  XPSNR y: 17.3121  XPSNR u: 18.5356  XPSNR v: 18.8916\n"
        "n:    2  XPSNR y: -0.8422  XPSNR u: 3.0559  XPSNR v: 2.5208\n",
        encoding="utf-8",
    )

    result = _parse_xpsnr_log(log_path)

    assert result[0] == 17.3121  # xpsnr's n=1 -> our frame 0
    assert result[1] == -0.8422  # xpsnr's n=2 -> our frame 1 (also confirms negative values parse)


def test_parse_xpsnr_log_preserves_infinity_for_a_perfect_frame(tmp_path):
    from vmaf_app.core.vmaf_runner import _parse_xpsnr_log

    log_path = tmp_path / "xpsnr.txt"
    log_path.write_text(
        "n:    1  XPSNR y: inf  XPSNR u: inf  XPSNR v: inf\n",
        encoding="utf-8",
    )

    assert _parse_xpsnr_log(log_path) == {0: float("inf")}


def test_parse_xpsnr_log_missing_file_returns_empty_dict(tmp_path):
    from vmaf_app.core.vmaf_runner import _parse_xpsnr_log

    assert _parse_xpsnr_log(tmp_path / "does_not_exist.txt") == {}


def test_progress_frame_estimate_is_not_divided_by_libvmaf_subsampling():
    from vmaf_app.core.vmaf_runner import estimate_total_frames

    info = VideoInfo(
        path=Path("movie.mp4"), width=1920, height=1080, fps=30.0,
        duration=10.0, nb_frames=300, codec_name="h264",
    )

    assert estimate_total_frames(info, VmafOptions(n_subsample=10)) == 300


def test_crop_detection_receives_run_cancel_and_process_controls(monkeypatch):
    import threading

    from vmaf_app.core import vmaf_runner
    from vmaf_app.core.crop_detect import CropDetectCancelled
    from vmaf_app.core.process_control import ProcessHandle
    from vmaf_app.core.vmaf_runner import Cancelled

    source = VideoInfo(Path("s.mp4"), 1920, 1080, 30.0, 10.0, 300, "h264")
    distorted = VideoInfo(Path("d.mp4"), 1920, 1080, 30.0, 10.0, 300, "h264")
    cancel = threading.Event()
    handle = ProcessHandle()
    received = []

    def cancelled_crop(info, **kwargs):
        received.append((kwargs["cancel_event"], kwargs["process_handle"]))
        raise CropDetectCancelled("cancelled")

    monkeypatch.setattr(vmaf_runner, "detect_crop", cancelled_crop)

    with pytest.raises(Cancelled):
        vmaf_runner.run_vmaf(
            source, distorted, VmafOptions(),
            cancel_event=cancel, process_handle=handle,
        )
    assert received == [(cancel, handle)]


@pytest.mark.parametrize(
    ("source_changes", "distorted_changes", "message"),
    [
        ({"fps": 24.0}, {"fps": 30.0}, "Frame rates"),
        ({"duration": 10.0}, {"duration": 12.0}, "Durations"),
        ({"sar": "1:1"}, {"sar": "4:3"}, "aspect ratios"),
        ({"nominal_fps": 60.0}, {"nominal_fps": 30.0}, "Variable-frame-rate"),
    ],
)
def test_incompatible_video_timelines_are_rejected(
    source_changes, distorted_changes, message
):
    from dataclasses import replace

    from vmaf_app.core.vmaf_runner import VmafRunError, validate_video_pair

    base_source = VideoInfo(Path("s.mp4"), 1920, 1080, 30.0, 10.0, 300, "h264")
    base_distorted = VideoInfo(Path("d.mp4"), 1920, 1080, 30.0, 10.0, 300, "h264")

    with pytest.raises(VmafRunError, match=message):
        validate_video_pair(
            replace(base_source, **source_changes),
            replace(base_distorted, **distorted_changes),
            VmafOptions(),
        )


def test_parse_log_keeps_a_genuine_zero_psnr_or_ssim(tmp_path):
    # libvmaf reports a real 0.0 for badly degraded frames. Reading these
    # with `metrics.get("psnr_y") or metrics.get("psnr")` discarded the 0.0
    # and fell through, losing a legitimate score.
    import json

    from vmaf_app.core.vmaf_runner import _parse_log

    log_path = tmp_path / "vmaf_log.json"
    log_path.write_text(json.dumps({"frames": [
        {"frameNum": 0, "metrics": {"vmaf": 0.0, "psnr_y": 0.0, "float_ssim": 0.0}},
        {"frameNum": 1, "metrics": {"vmaf": 50.0, "psnr_y": 25.5, "float_ssim": 0.5}},
    ]}), encoding="utf-8")

    frames = _parse_log(log_path, fps=30.0)

    assert frames[0].psnr == 0.0
    assert frames[0].ssim == 0.0
    assert frames[1].psnr == 25.5


# --------------------------------------------------- analysis bit depth

@pytest.mark.parametrize(("pix_fmt", "expected"), [
    ("yuv420p", 8), ("nv12", 8), ("nv21", 8), ("rgb24", 8), ("yuyv422", 8), ("", 8),
    ("yuv420p10le", 10), ("yuv422p10le", 10), ("p010le", 10),
    ("yuv420p12le", 12), ("gbrp12be", 12),
    ("yuv444p16le", 16), ("p016le", 16), ("gray10le", 10),
])
def test_bit_depth_is_read_from_the_pixel_format_name(pix_fmt, expected):
    # rgb24 is the trap: the 24 is bits per *pixel*, not per component, so a
    # "any digits in the name" rule would call an 8-bit format 24-bit.
    assert _bit_depth(pix_fmt) == expected


@pytest.mark.parametrize(("formats", "expected"), [
    (("yuv420p", "yuv420p"), "yuv420p"),
    (("yuv420p10le", "yuv420p10le"), "yuv420p10le"),
    (("yuv420p12le", "yuv420p12le"), "yuv420p12le"),
    # Mixed depths promote the shallower side rather than truncating the
    # deeper one -- a 10-bit master must not be measured through an 8-bit
    # pipe just because the encode under test is 8-bit.
    (("yuv420p10le", "yuv420p"), "yuv420p10le"),
    (("yuv420p", "yuv420p10le"), "yuv420p10le"),
    (("yuv420p12le", "yuv420p10le"), "yuv420p12le"),
    # libvmaf tops out at 12-bit, so deeper intermediates analyse at 12.
    (("yuv444p16le", "yuv420p"), "yuv420p12le"),
])
def test_analysis_format_takes_the_deeper_of_the_two_inputs(formats, expected):
    assert analysis_pix_fmt(*formats) == expected


@pytest.mark.parametrize(("source_fmt", "distorted_fmt", "expected"), [
    ("yuv420p", "yuv420p", "yuv420p"),
    ("yuv420p10le", "yuv420p10le", "yuv420p10le"),
    ("yuv420p10le", "yuv420p", "yuv420p10le"),
    ("yuv420p12le", "yuv420p10le", "yuv420p12le"),
])
def test_both_branches_are_converted_to_the_same_analysis_format(
    source_fmt, distorted_fmt, expected
):
    # Both chains must name the SAME format: libvmaf compares two streams
    # and a mismatch either errors out or silently inserts a conversion
    # nobody chose.
    graph = _build_filtergraph(
        _info("source.mov", 1920, 1080, pix_fmt=source_fmt),
        _info("distorted.mp4", 1920, 1080, pix_fmt=distorted_fmt),
        VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
        log_path=Path("log.json"),
    )
    main_chain, ref_chain, _ = graph.split(";")

    assert f"format={expected}" in main_chain
    assert f"format={expected}" in ref_chain
    if expected != "yuv420p":
        assert "format=yuv420p," not in graph and "format=yuv420p[" not in graph


def test_a_ten_bit_source_is_not_analysed_at_eight_bits():
    # The regression this guards: every comparison used to end with a
    # hard-coded format=yuv420p, so a 10-bit master and a 10-bit encode were
    # both truncated to 8-bit before a single metric was computed.
    graph = _build_filtergraph(
        _info("master.mkv", 3840, 2160, pix_fmt="yuv420p10le"),
        _info("encode.mkv", 3840, 2160, pix_fmt="yuv420p10le"),
        VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
        log_path=Path("log.json"),
    )
    assert "format=yuv420p10le" in graph
    assert "format=yuv420p," not in graph


def test_a_ten_bit_resample_test_stays_ten_bit():
    graph = _build_resample_test_filtergraph(
        _info("master.mkv", 3840, 2160, pix_fmt="yuv420p10le"),
        VmafOptions(model="version=vmaf_v0.6.1", resample_test=ResampleTarget(width=1920, label="1080p")),
        source_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )
    assert "format=yuv420p10le" in graph


def test_gpu_download_feeds_the_analysis_format_rather_than_replacing_it():
    # hwdownload can only emit the surface's native format, so the chain has
    # to be hwdownload -> p010le -> the analysis format. Dropping that last
    # step leaves libvmaf comparing semi-planar p010 against planar yuv.
    graph = _build_filtergraph(
        _info("master.mkv", 3840, 2160, pix_fmt="yuv420p10le"),
        _info("encode.mkv", 3840, 2160, pix_fmt="yuv420p10le"),
        VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(source="cuda"),
        log_path=Path("log.json"),
    )
    _, ref_chain, _ = graph.split(";")
    assert "hwdownload,format=p010le,format=yuv420p10le" in ref_chain


# ------------------------------------------------- subprocess reaping

class _FakePipe:
    """A pipe whose iteration can be made to raise, and that records being
    closed."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def __iter__(self):
        yield from self._lines

    def close(self):
        self.closed = True


class _FakeProcess:
    """Stands in for a Popen that ignores terminate() until killed, which is
    what a wedged hardware decoder actually does."""

    def __init__(self, stdout_lines, *, ignores_terminate=False):
        self.pid = 4242
        self.stdout = _FakePipe(stdout_lines)
        self.stderr = _FakePipe(["ffmpeg stderr\n"])
        self.terminated = False
        self.killed = False
        self.waited = False
        self.returncode = None
        self._ignores_terminate = ignores_terminate
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        if not self._ignores_terminate:
            self._alive = False
            self.returncode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        if self._alive:
            if timeout is None:
                self._alive = False
                self.returncode = 0
            else:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return self.returncode


def _run_with_fake(monkeypatch, proc, tmp_path, on_progress=None):
    from vmaf_app.core import vmaf_runner

    monkeypatch.setattr(vmaf_runner.proc_util, "popen", lambda *a, **k: proc)
    return vmaf_runner._run_ffmpeg(
        ["ffmpeg"], total_frames=100, on_progress=on_progress,
        cancel_event=None, cwd=tmp_path,
    )


def test_a_raising_progress_callback_does_not_leave_ffmpeg_running(monkeypatch, tmp_path):
    # The leak this guards: the exception escapes the stdout loop, and an
    # ffmpeg left running holds the run's temp dir open, so on Windows the
    # enclosing TemporaryDirectory silently fails to delete.
    proc = _FakeProcess(["frame=1\n", "frame=2\n"])

    def explode(current, total, fps):
        raise RuntimeError("the UI went away")

    with pytest.raises(RuntimeError, match="the UI went away"):
        _run_with_fake(monkeypatch, proc, tmp_path, on_progress=explode)

    assert proc.terminated, "ffmpeg was left running"
    assert proc.waited, "the process was never reaped, so it stays a zombie"
    assert proc.stdout.closed and proc.stderr.closed, "pipes were left open"


def test_a_process_that_ignores_terminate_is_killed(monkeypatch, tmp_path):
    proc = _FakeProcess(["frame=1\n"], ignores_terminate=True)

    def explode(current, total, fps):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _run_with_fake(monkeypatch, proc, tmp_path, on_progress=explode)

    assert proc.terminated and proc.killed


def test_the_process_handle_is_detached_even_when_the_callback_raises(monkeypatch, tmp_path):
    from vmaf_app.core import vmaf_runner
    from vmaf_app.core.process_control import ProcessHandle

    proc = _FakeProcess(["frame=1\n"])
    handle = ProcessHandle()
    monkeypatch.setattr(vmaf_runner.proc_util, "popen", lambda *a, **k: proc)

    def explode(current, total, fps):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        vmaf_runner._run_ffmpeg(
            ["ffmpeg"], total_frames=100, on_progress=explode,
            cancel_event=None, cwd=tmp_path, process_handle=handle,
        )

    assert handle._pid is None, "a detached handle must not still address a dead pid"


def test_a_normal_run_still_returns_its_stderr_and_exit_code(monkeypatch, tmp_path):
    proc = _FakeProcess(["frame=1\n", "fps= 24.0\n", "frame=2\n"])
    seen = []

    result = _run_with_fake(
        monkeypatch, proc, tmp_path,
        on_progress=lambda c, t, f: seen.append((c, t, f)),
    )

    assert result.returncode == 0
    assert "ffmpeg stderr" in result.stderr
    assert seen == [(1, 100, 0.0), (2, 100, 24.0)]


# --------------------------------------------- per-input hardware decode

def test_each_input_gets_its_own_hwaccel_options():
    # -hwaccel is a per-input option in ffmpeg: it applies to the next -i on
    # the command line. That is what lets the two inputs decode differently,
    # and it is why the options must sit immediately before their own -i.
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf",
        hwaccel=HwAccelPlan(source="cuda", distorted="d3d11va"),
    )
    distorted_at = cmd.index(str(Path("distorted.mp4").resolve()))
    source_at = cmd.index(str(Path("source.mp4").resolve()))

    # Each input reads as: -hwaccel X -hwaccel_output_format X -i <path>
    assert cmd[distorted_at - 5:distorted_at] == [
        "-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11va", "-i",
    ]
    assert cmd[source_at - 5:source_at] == [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i",
    ]


def test_only_the_accelerated_input_gets_hwaccel_options():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf",
        hwaccel=HwAccelPlan(source="cuda", distorted=None),
    )
    distorted_at = cmd.index(str(Path("distorted.mp4").resolve()))
    source_at = cmd.index(str(Path("source.mp4").resolve()))

    assert cmd[distorted_at - 2] != "-hwaccel_output_format",         "the software-decoded input got hwaccel options"
    assert cmd[source_at - 5:source_at] == [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i",
    ]
    assert cmd.count("-hwaccel") == 1


def test_a_gpu_decoded_distorted_input_is_downloaded_before_filtering():
    # Hardware frames are surfaces, not pixels: crop and format can't touch
    # them. Without the hwdownload the filtergraph fails to configure.
    graph = _build_filtergraph(
        _info("source.mov", 1920, 1080), _info("distorted.mp4", 1920, 1080),
        VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=CropBox(x=0, y=20, w=1920, h=1040),
        hwaccel=HwAccelPlan(distorted="cuda"), log_path=Path("log.json"),
    )
    main_chain = graph.split(";")[0]

    assert main_chain.startswith("[0:v]hwdownload,format=nv12,crop=")
    assert "hwdownload" not in graph.split(";")[1], "the source was not GPU-decoded"


def test_a_ten_bit_distorted_input_downloads_through_p010():
    graph = _build_filtergraph(
        _info("source.mov", 1920, 1080, pix_fmt="yuv420p10le"),
        _info("distorted.mp4", 1920, 1080, pix_fmt="yuv420p10le"),
        VmafOptions(model="version=vmaf_v0.6.1"),
        source_crop=None, distorted_crop=None,
        hwaccel=HwAccelPlan(source="cuda", distorted="cuda"), log_path=Path("log.json"),
    )
    main_chain, ref_chain, _ = graph.split(";")

    assert "hwdownload,format=p010le,format=yuv420p10le" in main_chain
    assert "hwdownload,format=p010le,format=yuv420p10le" in ref_chain


def test_each_input_gets_an_independent_single_gpu_fallback():
    # The distorted file is the arbitrary one -- whatever encoder settings
    # are under test -- while the source is usually a known-good master, so
    # it is the first suspect when hardware decode fails.
    ladder = _fallback_ladder(HwAccelPlan(source="cuda", distorted="cuda"))

    assert ladder == [
        HwAccelPlan(source="cuda", distorted="cuda"),
        HwAccelPlan(source="cuda", distorted=None),
        HwAccelPlan(source=None, distorted="cuda"),
        HwAccelPlan(),
    ]


@pytest.mark.parametrize(("plan", "expected"), [
    (HwAccelPlan(), [HwAccelPlan()]),
    (HwAccelPlan(source="cuda"), [HwAccelPlan(source="cuda"), HwAccelPlan()]),
    (HwAccelPlan(distorted="qsv"), [HwAccelPlan(distorted="qsv"), HwAccelPlan()]),
])
def test_every_ladder_ends_at_software_decode_without_repeating_a_plan(plan, expected):
    ladder = _fallback_ladder(plan)
    assert ladder == expected
    assert ladder[-1] == HwAccelPlan(), "the last resort must be all-CPU"
    assert len(set(ladder)) == len(ladder), "a plan that already failed is retried"


def test_a_run_retries_down_the_ladder_until_one_succeeds(monkeypatch, tmp_path):
    from vmaf_app.core import vmaf_runner

    attempts = []
    statuses = []

    def fake_run_ffmpeg(cmd, total_frames, on_progress, cancel_event, cwd, process_handle=None):
        plan = cmd[0]
        attempts.append(plan)
        # Only all-software decode works on this imaginary machine.
        code = 0 if not plan.uses_gpu else 1
        if code == 0:
            (Path(cwd) / "vmaf_log.json").write_text('{"frames": []}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, code, "", "decoder error")

    monkeypatch.setattr(vmaf_runner, "_run_ffmpeg", fake_run_ffmpeg)

    vmaf_runner._execute_run(
        lambda plan, model, log_path, xpsnr_log_path: [plan],
        options=VmafOptions(), fps=30.0, total_frames=10,
        hwaccel=HwAccelPlan(source="cuda", distorted="cuda"),
        tmp_prefix="test_", on_progress=None, on_status=statuses.append,
        cancel_event=None, process_handle=None,
    )

    assert attempts == _fallback_ladder(HwAccelPlan(source="cuda", distorted="cuda"))
    # The status line names which input actually got hardware decode --
    # otherwise a silent per-input fallback looks like a run that never tried.
    assert "source cuda, distorted cuda" in statuses[0]
    assert "source cuda, distorted cpu" in statuses[1]
    assert "source cpu, distorted cuda" in statuses[2]
    assert "off" in statuses[3]


def test_a_failed_attempt_does_not_leave_a_log_for_the_retry_to_parse(monkeypatch, tmp_path):
    # ffmpeg can write a partial log before a decoder gives up. If the retry
    # then fails to produce one, that stale file would be parsed as though it
    # were the retry's own output -- a truncated run reported as a complete one.
    from vmaf_app.core import vmaf_runner

    seen_existing_log = []

    def fake_run_ffmpeg(cmd, total_frames, on_progress, cancel_event, cwd, process_handle=None):
        log = Path(cwd) / "vmaf_log.json"
        seen_existing_log.append(log.exists())
        log.write_text('{"frames": []}', encoding="utf-8")  # a partial log
        return subprocess.CompletedProcess(cmd, 1, "", "decoder error")

    monkeypatch.setattr(vmaf_runner, "_run_ffmpeg", fake_run_ffmpeg)

    with pytest.raises(vmaf_runner.VmafRunError):
        vmaf_runner._execute_run(
            lambda plan, model, log_path, xpsnr_log_path: [plan],
            options=VmafOptions(), fps=30.0, total_frames=10,
            hwaccel=HwAccelPlan(source="cuda", distorted="cuda"),
            tmp_prefix="test_", on_progress=None, on_status=None,
            cancel_event=None, process_handle=None,
        )

    assert seen_existing_log == [False, False, False, False]
