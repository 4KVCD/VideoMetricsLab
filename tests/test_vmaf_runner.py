import os
from pathlib import Path

from vmaf_app.core.models import (
    CropBox, ResampleTarget, ScaleDirection, VideoInfo, VmafOptions, synthetic_resample_distorted_path,
)
from vmaf_app.core.vmaf_runner import (
    _build_ffmpeg_cmd, _build_filtergraph, _build_resample_cmd, _build_resample_test_filtergraph, _hw_native_format,
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
        source_crop=None, distorted_crop=None, hwaccel_used=None,
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
        source_crop=None, distorted_crop=None, hwaccel_used=None, log_path=Path("log.json"),
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
        source_crop=None, distorted_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    main_chain, ref_chain, _ = graph.split(";")
    assert "scale=3840:2160" in main_chain  # distorted IS upscaled to the source's resolution
    assert "scale=" not in ref_chain  # source is left untouched


def test_upscale_distorted_mode_is_a_noop_when_resolutions_already_match():
    info_a = _info("source.mov", 1920, 1080)
    info_b = _info("distorted.mp4", 1920, 1080)

    graph = _build_filtergraph(
        info_a, info_b, VmafOptions(model="version=vmaf_v0.6.1", scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        source_crop=None, distorted_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    assert "scale=" not in graph


def test_no_scale_filter_when_resolutions_already_match():
    info_a = _info("source.mov", 1920, 1080)
    info_b = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        info_a, info_b, options, source_crop=None, distorted_crop=None,
        hwaccel_used=None, log_path=Path("log.json"),
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
        hwaccel_used=None, log_path=Path("log.json"),
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
        hwaccel_used=None, log_path=Path("log.json"),
    )

    assert "crop=" not in graph


def test_hwdownload_inserted_when_gpu_decode_used():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel_used="cuda", log_path=Path("log.json"),
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
        hwaccel_used="cuda", log_path=Path("log.json"),
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
        hwaccel_used=None, log_path=Path("log.json"),
    )

    libvmaf_part = graph.split("libvmaf=", 1)[1]
    assert f"n_threads={os.cpu_count() or 1}" in libvmaf_part


def test_duration_limit_adds_output_side_t_flag():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf", hwaccel=None, duration_limit=30.0,
    )
    # -t must come after -lavfi (an output option, bounding the whole
    # filtered output) not before either -i (which would just be misplaced).
    lavfi_idx = cmd.index("-lavfi")
    t_idx = cmd.index("-t")
    assert t_idx > lavfi_idx
    assert cmd[t_idx + 1] == "30.000"


def test_no_duration_limit_omits_t_flag_by_default():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf", hwaccel=None,
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
        hwaccel_used=None, log_path=Path("log.json"),
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
        hwaccel_used=None, log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr.txt"),
    )

    assert "xpsnr" not in graph
    assert "[main][ref]libvmaf=" in graph


def test_xpsnr_stage_sits_between_decode_and_libvmaf():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=True)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel_used=None, log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
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
        hwaccel_used=None, log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
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


def test_xpsnr_requested_but_no_log_path_is_a_noop():
    # Defensive: compute_xpsnr=True with no path given (shouldn't happen via
    # the UI, but the filtergraph builder must not silently reference a
    # nonexistent file) skips the xpsnr stage rather than erroring.
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=True)

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel_used=None, log_path=Path("log.json"), xpsnr_log_path=None,
    )

    assert "xpsnr" not in graph


def test_xpsnr_stage_also_available_for_resample_tests():
    source_info = _info("source.mov", 1920, 1080)
    options = VmafOptions(
        model="version=vmaf_v0.6.1", compute_xpsnr=True, resample_test=ResampleTarget(width=960, label="480p"),
    )

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None,
        log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )

    assert "xpsnr=stats_file=xpsnr_log.txt" in graph


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


def test_parse_xpsnr_log_missing_file_returns_empty_dict(tmp_path):
    from vmaf_app.core.vmaf_runner import _parse_xpsnr_log

    assert _parse_xpsnr_log(tmp_path / "does_not_exist.txt") == {}


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
