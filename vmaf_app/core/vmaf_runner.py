"""Builds and runs the ffmpeg + libvmaf command, streaming progress and
parsing the resulting per-frame JSON log.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from vmaf_app.core import gpu_frames, vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan
from vmaf_app.core import proc as proc_util
from vmaf_app.core.crop_detect import CropDetectCancelled, common_picture, detect_crop, detect_pair
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, check_tools, ffmpeg_path, format_version
from vmaf_app.core.frame_coverage import short_comparison
from vmaf_app.core.frame_sync import FRAMESYNC_OPTS
from vmaf_app.core.geometry import (
    analysis_dimensions,
    content_size,
    display_aspect_ratio,
    pair_problem,
    resample_analysis_dimensions,
)
from vmaf_app.core.gpu import (
    GPU_PASS,
    GPU_WAIT_MESSAGE,
    HwAccelPlan,
    analysis_pix_fmt,
    plan_hwaccel,
)
from vmaf_app.core.gpu import (
    bit_depth as _bit_depth,
)
from vmaf_app.core.gpu import (
    hw_native_format as _hw_native_format,
)
from vmaf_app.core.gpu import (
    hwaccel_args as _hwaccel_args,
)
from vmaf_app.core.isolated import run_isolated
from vmaf_app.core.metric_results import current_ffmpeg_provenance, results_from_frame_scores
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, model_for_resolution, resolve_v1_model
from vmaf_app.core.models import (
    ComparisonResult,
    CropBox,
    CropMode,
    FrameScores,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
    synthetic_resample_distorted_path,
)
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.status import GPU_VMAF_FAILED, GPU_WAIT, STARTING, Status

_log = logging.getLogger(__name__)


def _command_text(command) -> str:
    """A command as a copy-pasteable line for the log; never raises -- a log
    line must not be able to stop a run."""
    try:
        return subprocess.list2cmdline(command)
    except TypeError:
        return repr(command)

ProgressCallback = Callable[[int, int, float], None]  # (current_frame, total_frames, fps)


def _metric_results_for_current_run(frames: FrameScores, model: str, model_v1: str = "",
                                    gpu_keys: set[str] | None = None, gpu_backend: str = "cuda",
                                    cpu_keys: set[str] | None = None):
    """Adapt one FFmpeg parse into generic results without re-parsing it.
    `gpu_keys`: VMAF and NEG scored on the GPU (vmaf_cuda) -- the same
    request identity as FFmpeg's libvmaf (their scores agree to within
    4e-5), recorded as GPU scores of the bundled build, libvmaf's CUDA code
    or its Vulkan port (`gpu_backend`). `cpu_keys`: PSNR and SSIM scored by
    the bundled libvmaf's CPU extractors in the app (vmaf_cuda.CpuScorer),
    which give FFmpeg's libvmaf's scores: the same identity too."""
    status = check_tools()
    version = format_version(status.ffmpeg.version) if status.ffmpeg.runnable else "unknown"

    def provenance(key: str):
        parameters = ({"model": "version=vmaf_v0.6.1neg"} if key == "vmaf_neg"
                      else {"model": model} if key == "vmaf"
                      else {"model": model_v1} if key == "vmaf_v1" else None)
        made = current_ffmpeg_provenance(key, version, parameters)
        if key in (cpu_keys or ()):
            made = replace(made, implementation="libvmaf", implementation_version=vmaf_cuda.CPU_BUILD)
        if key in (gpu_keys or ()):
            # VMAF v1's GPU half is Vulkan's on every GPU (vmaf_v1_gpu).
            backend = "vulkan" if key == "vmaf_v1" else gpu_backend
            made = replace(made, implementation=f"libvmaf/{backend}", compute_backend="gpu",
                           implementation_version=vmaf_v1_gpu.LIBRARY_BUILD if key == "vmaf_v1"
                           else _gpu_build(gpu_backend))
        return made

    return results_from_frame_scores(frames, {key: provenance(key) for key in frames.metric_keys})


def _gpu_build(backend: str) -> str:
    """What scores VMAF on the GPU with `backend`."""
    return vmaf_vulkan.LIBRARY_BUILD if backend == "vulkan" else vmaf_cuda.LIBRARY_BUILD


class VmafRunError(RuntimeError):
    def __init__(self, message: str, stderr_tail: str = ""):
        super().__init__(message)
        self.stderr_tail = stderr_tail


class Cancelled(RuntimeError):  # noqa: N818 - a cancellation, not an error condition
    """Raised when the user cancels a run mid-flight. Deliberately not named
    `CancelledError`: nothing went wrong, and callers treat it as an expected
    outcome rather than a failure to report."""


def validate_video_pair(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions
) -> None:
    """Rejects comparisons whose timelines are ambiguous (geometry.pair_problem)."""
    if problem := pair_problem(source_info, distorted_info, options.duration_limit):
        raise VmafRunError(problem)


#: Two shapes count as the same if they agree to within this fraction. Wide
#: enough for the rounding that even dimensions force (1918x1080 against
#: 1920x1080 differs by 0.1%), far tighter than any real mismatch: 4:3
#: against 16:9 is 33% apart, and the letterbox case below is 32%.
_ASPECT_TOLERANCE = 0.01


def validate_display_geometry(
    source_info: VideoInfo, distorted_info: VideoInfo,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
) -> None:
    """Rejects a pair whose pictures are different shapes after cropping.

    The filtergraph scales one side to the other's exact width and height,
    which silently stretches a mismatched shape until libvmaf accepts it.
    The result is a real number computed from a geometrically wrong
    comparison, and it looks like any other score: a 1920x1080 source whose
    content is a letterboxed 1920x816, compared with crop off against an
    already-cropped 960x408 encode of it, scored 0.4977 -- against 87.14
    for the same pair cropped correctly.

    Run after crops are resolved, because cropping is exactly what makes a
    letterboxed source and a cropped encode comparable. Before it they
    legitimately differ.
    """
    source_dar = display_aspect_ratio(source_info, source_crop)
    distorted_dar = display_aspect_ratio(distorted_info, distorted_crop)
    if source_dar <= 0 or distorted_dar <= 0:
        return  # degenerate metadata; nothing meaningful to compare
    if abs(source_dar - distorted_dar) <= _ASPECT_TOLERANCE * max(source_dar, distorted_dar):
        return

    def describe(info: VideoInfo, crop: CropBox | None, dar: float) -> str:
        shape = f"{crop.w}x{crop.h} (cropped)" if crop else f"{info.width}x{info.height}"
        return f"{shape}, {dar:.3f}:1"

    raise VmafRunError(
        "The two videos are different shapes after cropping: "
        f"source {describe(source_info, source_crop, source_dar)} versus "
        f"distorted {describe(distorted_info, distorted_crop, distorted_dar)}. "
        "Scoring them would stretch one to fit the other and the result would "
        "be meaningless. If one is letterboxed, set black-bar handling to "
        "'Auto-detect' so the bars are removed before comparison."
    )


def _resolve_crops(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    status_callback: Callable[[str], None] | None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    hwaccel: HwAccelPlan | None = None,
) -> tuple[CropBox | None, CropBox | None]:
    """`hwaccel` is the run's own decode plan; each input's detection windows
    decode the same way the run will. Speed only -- the box is the same."""
    if options.crop_mode == CropMode.NONE:
        return None, None

    plan = hwaccel or HwAccelPlan()
    if status_callback:
        status_callback("Detecting black bars in source and distorted...")
    try:
        boxes = detect_pair(
            lambda: detect_crop(
                source_info, cancel_event=cancel_event, process_handle=process_handle,
                hwaccel=plan.source,
            ),
            lambda: detect_crop(
                distorted_info, cancel_event=cancel_event, process_handle=process_handle,
                hwaccel=plan.distorted,
            ),
        )
    except CropDetectCancelled as e:
        raise Cancelled("Cancelled by user") from e
    return common_picture(source_info, distorted_info, *boxes)


def auto_threads(concurrent_jobs: int = 1) -> int:
    """How many threads "Auto" (n_threads <= 0) means for libvmaf.

    Every logical core for a video scored on its own. When several are
    scored at once each gets an equal share, so two jobs on a 24-core
    machine ask for 12 threads each rather than 24 each: twice as many
    threads as cores cannot do more work than exactly as many, they only
    take turns on the same cores and pay for the switching.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, concurrent_jobs))


def _build_libvmaf_opts(options: VmafOptions, log_path: Path, model: str | None = None) -> list[str]:
    # ffmpeg's filtergraph option parser can't reliably handle an absolute
    # Windows path (drive-letter colon) as an option value, even escaped or
    # quoted -- so ffmpeg is always launched with cwd=log_path.parent and we
    # reference the log (and any custom model file) by bare filename here.
    model_value = model if model is not None else options.model
    v1_file = _v1_model_file(options)
    if options.compute_vmaf_neg or v1_file is not None:
        # Explicit names keep the output score arrays independent. The v1
        # model file is copied next to the log by _execute_run.
        models = ([model_value + r"\\:name=vmaf"] if options.compute_vmaf else [])
        if v1_file is not None:
            models.append(f"path={v1_file.name}" + r"\\:name=vmaf_v1")
        if options.compute_vmaf_neg:
            models.append(r"version=vmaf_v0.6.1neg\\:name=vmaf_neg")
        model_value = "|".join(models)
    opts = [
        f"log_path={log_path.name}",
        "log_fmt=json",
        f"model={model_value}" if _uses_vmaf_model(options) else "model=''",
    ]
    # libvmaf 2.0+ defaults to single-threaded (n_threads=1) unless told
    # otherwise -- omitting this option here does NOT mean "use all cores",
    # so "Auto" (n_threads <= 0) is resolved to the actual core count instead
    # of leaving it unset. A job that shares the machine with another has
    # already had its share filled in by the worker (VmafWorker._share_cores).
    resolved_threads = options.n_threads if options.n_threads > 0 else auto_threads()
    opts.append(f"n_threads={resolved_threads}")
    if options.n_subsample > 1:
        opts.append(f"n_subsample={options.n_subsample}")
    if options.extra_features:
        opts.append("feature=" + "|".join(options.extra_features))
    opts += FRAMESYNC_OPTS
    return opts


def _uses_vmaf_model(options: VmafOptions) -> bool:
    return options.compute_vmaf or options.compute_vmaf_neg or options.compute_vmaf_v1


def _v1_model_file(options: VmafOptions) -> Path | None:
    """The VMAF v1 model file a run uses, once the run has resolved it."""
    if options.compute_vmaf_v1 and options.model_v1.startswith("path="):
        return Path(options.model_v1.removeprefix("path="))
    return None


#: overlay's name for each analysis format it holds unchanged. It has none
#: for 12-bit, which is therefore scored on the CPU (vmaf_cuda.scores_on_gpu).
_OVERLAY_FORMAT = {"yuv420p": "yuv420", "yuv420p10le": "yuv420p10"}


def _gpu_pairs_stage(analysis_format: str, width: int, height: int, main_label: str, ref_label: str) -> str:
    """The frame pairs FFmpeg's libvmaf filter compares, as the raw outputs
    [vmaf_dist] and [vmaf_ref] that VMAF on the GPU reads: the two streams
    are synchronized by overlay with libvmaf's own frame sync options
    (FRAMESYNC_OPTS), side by side in one frame, and cut apart again,
    every pixel unchanged.

    The two outputs used to come straight from the two streams, paired by
    position, where libvmaf pairs them by timestamp: on a test video whose
    timestamps were a fraction of a millisecond from the source's, libvmaf
    on the CPU compared 239 frames and the GPU 240 -- and with VMAF v1,
    PSNR, SSIM or XPSNR beside it, VMAF on the GPU was refused and the
    video calculated again on the CPU.

    For an even width and height only (vmaf_cuda.scores_on_gpu sends an odd
    comparison to the CPU): pad gives a 4:2:0 picture an even size and
    blacks out an odd one's last column and row, and crop cuts on whole
    chroma samples. An odd width was once given an even offset here, and
    the run still failed in FFmpeg, every time."""
    if width & 1 or height & 1:
        raise ValueError(f"the GPU's frame pairs need an even size, not {width}x{height}")
    sync = ":".join(FRAMESYNC_OPTS)
    return (f"[{main_label}]pad={2 * width}:{height}[vmaf_canvas];"
            f"[vmaf_canvas][{ref_label}]overlay=x={width}:y=0:eval=init:"
            f"format={_OVERLAY_FORMAT[analysis_format]}:{sync},split=2[vmaf_left][vmaf_right];"
            f"[vmaf_left]crop={width}:{height}:0:0[vmaf_dist];"
            f"[vmaf_right]crop={width}:{height}:{width}:0[vmaf_ref]")


def analysis_bit_depth(source_info: VideoInfo, distorted_info: VideoInfo) -> int:
    """The bit depth two videos are compared at (analysis_pix_fmt)."""
    return _bit_depth(analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt))


#: Between the filter graphs of a run that has more than one
#: (_build_filtergraph with _split_graphs): each becomes a -filter_complex of
#: its own, ending in [graph0], [graph1].
_GRAPH_SEPARATOR = "\n"


def _split_graphs(options: VmafOptions, xpsnr_log_path: Path | None) -> bool:
    """Whether XPSNR and libvmaf's metrics (VMAF, PSNR, SSIM...) get a filter
    graph each. FFmpeg runs a filter graph on one thread: chained in one,
    XPSNR's work and the libvmaf filter's -- which allocates, zeroes and
    copies both pictures of every pair before its own threads see them --
    took turns. PSNR + SSIM + XPSNR on 4K ran at 31 fps where PSNR + SSIM
    alone ran at 46 and XPSNR alone at 62, with most cores idle. Each on a
    thread of its own, the three run at 44, PSNR + SSIM's speed."""
    return bool(options.compute_xpsnr and xpsnr_log_path is not None
                and (_uses_vmaf_model(options) or options.extra_features))


def _xpsnr_filter(source_label: str, test_label: str, xpsnr_log_path: Path) -> str:
    """FFmpeg's xpsnr filter on the two videos, the source first: it weights
    each block's error by the activity of its first input -- the original's,
    as XPSNR is defined -- and takes the frame rate that picks its temporal
    activity from its second (_xpsnr_frame_rate). It passes its first input
    on. With the test video first, as before October 2026, the weights came
    from the encode: XPSNR 1.2 dB lower on a 4K film's 3 Mb/s encode, 0.9 dB
    on its 12 Mb/s one, 1.8 dB on a 1080p test pattern's."""
    return (f"[{source_label}][{test_label}]xpsnr=stats_file={xpsnr_log_path.name}:"
            + ":".join(FRAMESYNC_OPTS))


def _build_libvmaf_stage(
    options: VmafOptions, log_path: Path, model: str | None, xpsnr_log_path: Path | None,
    main_label: str = "main", ref_label: str = "ref", output_label: str = "",
) -> str:
    """The XPSNR + libvmaf tail shared by both filtergraph builders. XPSNR
    isn't a libvmaf "feature" like PSNR/SSIM -- it's a fully separate ffmpeg
    filter with its own stats file -- so when requested it sits between
    decode and libvmaf, passing [ref] through under a new label.
    `output_label` names its output (the GPU VMAF graph maps it explicitly).
    """
    if not options.requested_metrics():
        raise VmafRunError("Select at least one metric to calculate.")
    output = f"[{output_label}]" if output_label else ""
    # XPSNR-only needs no libvmaf filter or model at all.
    if not _uses_vmaf_model(options) and not options.extra_features:
        assert xpsnr_log_path is not None
        return _xpsnr_filter(ref_label, main_label, xpsnr_log_path) + output
    libvmaf_opts = _build_libvmaf_opts(options, log_path, model)
    chains = []
    if options.compute_xpsnr and xpsnr_log_path is not None:
        # xpsnr consumes [main], and libvmaf needs it too -- but a filtergraph
        # label can only be consumed once. Without this explicit split,
        # ffmpeg silently wires libvmaf up to the wrong stream and it ends up
        # comparing one video against itself, reporting a perfect VMAF 100 /
        # PSNR 60 / SSIM 1.0 for every frame no matter how bad the encode
        # actually is. It does NOT error out, so the scores just come back
        # quietly, plausibly wrong. libvmaf takes the source from xpsnr,
        # which passes its first input on.
        chains.append(f"[{main_label}]split=2[main_xpsnr][main_vmaf]")
        chains.append(_xpsnr_filter(ref_label, "main_xpsnr", xpsnr_log_path) + "[xref]")
        main_label = "main_vmaf"
        ref_label = "xref"
    chains.append(f"[{main_label}][{ref_label}]libvmaf=" + ":".join(libvmaf_opts) + output)
    return ";".join(chains)


def _build_filtergraph(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
    hwaccel: HwAccelPlan, log_path: Path, model: str | None = None,
    xpsnr_log_path: Path | None = None, gpu_vmaf: bool = False, gpu_paired: bool = True,
) -> str:
    """`gpu_vmaf`: VMAF and NEG are scored on the GPU (vmaf_cuda), from the
    compared frames as two raw outputs, [vmaf_dist] and [vmaf_ref], and are
    all the run scores: FFmpeg's own filters score nothing. `gpu_paired`:
    FFmpeg pairs the frames (_gpu_pairs_stage). Else the GPU's side does
    (vmaf_cuda._StreamReader, frame_sync), from each video as it would reach
    libvmaf's filter, and two graphs come back, a line each -- the test
    video's and the source's, each for an FFmpeg of its own
    (_build_stream_cmds): the video's frames ([vmaf_dist], [vmaf_ref]) and a
    copy that costs nothing, which FFmpeg lists the timestamps of ([.._ts])."""
    dist_content_w, dist_content_h = content_size(distorted_info, distorted_crop)
    ref_content_w, ref_content_h = content_size(source_info, source_crop)
    resolutions_differ = (ref_content_w, ref_content_h) != (dist_content_w, dist_content_h)
    upscale_distorted = resolutions_differ and options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE

    # Both branches have to reach libvmaf in the same pixel format, and that
    # format is chosen from the deeper of the two inputs -- see
    # analysis_pix_fmt for why it is not simply yuv420p.
    analysis_format = analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt)

    # XPSNR beside libvmaf's metrics: each in a filter graph of its own,
    # which FFmpeg runs on a thread of its own (_split_graphs).
    split = not gpu_vmaf and _split_graphs(options, xpsnr_log_path)

    # --- distorted (main, input 0) chain ---
    main_ops = []
    if hwaccel.distorted and not (split and _decoder_downloads(hwaccel.distorted)):
        # Same shape as the reference chain below: frames arrive as hardware
        # surfaces and have to come back to system memory before any filter
        # that isn't hardware-aware -- including the crop -- can touch them.
        main_ops.append("hwdownload")
        main_ops.append(f"format={_hw_native_format(distorted_info.pix_fmt)}")
    if distorted_crop and not distorted_crop.is_noop(distorted_info.width, distorted_info.height):
        main_ops.append(distorted_crop.as_filter())
    main_ops.append(f"format={analysis_format}")
    if upscale_distorted:
        # Scale the distorted video UP to the source's resolution instead of
        # the default (scaling the source down to the distorted video's
        # resolution) -- see ScaleDirection.
        main_ops.append(f"scale={ref_content_w}:{ref_content_h}:flags={options.scale_algorithm}")
    main_ops.append("setpts=PTS-STARTPTS")
    if gpu_vmaf and not gpu_paired:
        main_ops.append("split=2[vmaf_dist][vmaf_dist_ts]")
    main_chain = f"[0:{VIDEO_STREAM}]{','.join(main_ops)}[main]"

    # --- source / reference (input 1) chain ---
    ref_ops = []
    if hwaccel.source and not (split and _decoder_downloads(hwaccel.source)):
        # hwdownload can only emit the hw surface's native format -- nv12 for
        # 8-bit cuda decode, p010le for 10-bit (common for UHD/HDR masters) --
        # it can't itself target the analysis format, so that conversion
        # needs its own separate format filter afterwards.
        ref_ops.append("hwdownload")
        ref_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    # Cropped before the format conversion, as the distorted chain is (and
    # Vship's, the CPU tools' and the GPU decoders'): converted first, a
    # 4:2:2 or 4:4:4 source's chroma at the crop's edges was filtered with
    # samples of the bars cut off.
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        ref_ops.append(source_crop.as_filter())
    ref_ops.append(f"format={analysis_format}")

    if resolutions_differ and not upscale_distorted:
        ref_ops.append(f"scale={dist_content_w}:{dist_content_h}:flags={options.scale_algorithm}")

    ref_ops.append("setpts=PTS-STARTPTS")
    if gpu_vmaf and not gpu_paired:
        return (f"[0:{VIDEO_STREAM}]{','.join(main_ops)}\n"
                f"[0:{VIDEO_STREAM}]{','.join([*ref_ops, 'split=2[vmaf_ref][vmaf_ref_ts]'])}")
    ref_chain = f"[1:{VIDEO_STREAM}]{','.join(ref_ops)}[ref]"

    compared_w, compared_h = (
        (ref_content_w, ref_content_h) if upscale_distorted else (dist_content_w, dist_content_h))
    if split:
        # Each graph decodes nothing itself: FFmpeg hands every graph that
        # reads an input the same decoded frames (with CUDA or D3D11VA, the
        # decoder downloads them once: _split_hwaccel_args). Each converts
        # and crops them on its own thread.
        def chains(suffix: str) -> str:
            return (f"[0:{VIDEO_STREAM}]{','.join(main_ops)}[main_{suffix}];"
                    f"[1:{VIDEO_STREAM}]{','.join(ref_ops)}[ref_{suffix}]")

        assert xpsnr_log_path is not None
        return _GRAPH_SEPARATOR.join([
            chains("v") + ";[main_v][ref_v]libvmaf=" + ":".join(_build_libvmaf_opts(options, log_path, model))
            + "[graph0]",
            chains("x") + ";" + _xpsnr_filter("ref_x", "main_x", xpsnr_log_path) + "[graph1]",
        ])
    if not gpu_vmaf:
        tail = _build_libvmaf_stage(options, log_path, model, xpsnr_log_path)
    else:
        tail = _gpu_pairs_stage(analysis_format, compared_w, compared_h, "main", "ref")
    return ";".join([main_chain, ref_chain, tail])


def _build_resample_test_filtergraph(
    source_info: VideoInfo, options: VmafOptions, source_crop: CropBox | None,
    hwaccel_used: str | None, log_path: Path, model: str | None = None,
    xpsnr_log_path: Path | None = None,
) -> str:
    """A single-input filtergraph for a resolution round-trip test: the
    source is decoded once and split into an untouched reference branch and
    a "distorted" branch that's scaled down to the target width (preserving
    the source's own aspect ratio) and back up to the source's original
    resolution -- there's no second file, both branches come from the one input.
    """
    target = options.resample_test
    assert target is not None

    orig_w, orig_h = source_info.width, source_info.height
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        orig_w, orig_h = source_crop.w, source_crop.h

    down_w = target.width
    down_h = max(2, round(down_w * orig_h / orig_w / 2) * 2)  # even, preserves the source's own aspect ratio

    # A round-trip test has one input, so the analysis format comes from
    # the source alone -- see analysis_pix_fmt.
    analysis_format = analysis_pix_fmt(source_info.pix_fmt)

    base_ops = []
    if hwaccel_used:
        # See _build_filtergraph's identical comment: hwdownload can only
        # emit the hw surface's native format, not the analysis format.
        base_ops.append("hwdownload")
        base_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        base_ops.append(source_crop.as_filter())  # before the conversion, as _build_filtergraph
    base_ops.append(f"format={analysis_format}")
    base_chain = f"[0:{VIDEO_STREAM}]{','.join(base_ops)}[base]"

    split_chain = "[base]split=2[ref_src][dist_src]"
    ref_chain = "[ref_src]setpts=PTS-STARTPTS[ref]"
    dist_chain = (
        f"[dist_src]scale={down_w}:{down_h}:flags={options.scale_algorithm},"
        f"scale={orig_w}:{orig_h}:flags={options.scale_algorithm},setpts=PTS-STARTPTS[main]"
    )

    tail = _build_libvmaf_stage(options, log_path, model, xpsnr_log_path)
    return ";".join([base_chain, split_chain, ref_chain, dist_chain, tail])


_PROGRESS_FRAME_RE = re.compile(r"frame=(\d+)")
_PROGRESS_FPS_RE = re.compile(r"fps=\s*([\d.]+)")


def _build_ffmpeg_cmd(
    distorted_path: Path, source_path: Path, filtergraph: str,
    hwaccel: HwAccelPlan, duration_limit: float = 0.0,
    gpu_outputs: list[str] | None = None,
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    # -i paths are plain argv (not filtergraph syntax) so absolute Windows
    # paths are fine here even though they aren't inside the filtergraph --
    # but they must be made absolute first, since ffmpeg's cwd is set to a
    # temp dir below (see _build_filtergraph's log_path/model comment).
    decode = _split_hwaccel_args if _GRAPH_SEPARATOR in filtergraph else _hwaccel_args
    cmd += decode(hwaccel.distorted)
    cmd += ["-i", str(Path(distorted_path).resolve())]
    cmd += decode(hwaccel.source)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit, gpu_outputs)
    return cmd


#: -hwaccel decoders that hand frames over in system memory when no
#: -hwaccel_output_format is given. QSV's still hands over its own surfaces:
#: "Impossible to convert between the formats ... src: qsv".
_DECODERS_THAT_DOWNLOAD = frozenset({"cuda", "d3d11va"})


def _decoder_downloads(hwaccel: str | None) -> bool:
    return hwaccel in _DECODERS_THAT_DOWNLOAD


def _split_hwaccel_args(hwaccel: str | None) -> list[str]:
    """-hwaccel for one input of a run with several filter graphs. Where the
    decoder can (_decoder_downloads), it downloads each picture to system
    memory itself, once, and every graph is handed it; kept on the GPU,
    each graph would hwdownload it again: PSNR + SSIM + XPSNR on 4K ran at
    38 fps that way, 44 this way. QSV's frames stay on the GPU, and each
    graph downloads them (_build_filtergraph)."""
    return ["-hwaccel", hwaccel] if _decoder_downloads(hwaccel) else _hwaccel_args(hwaccel)


def _build_stream_cmds(
    distorted_path: Path, source_path: Path, graphs: str, hwaccel: HwAccelPlan,
    outputs: tuple[list[str], list[str]],
) -> list[list[str]]:
    """An FFmpeg for each video (the test video's first), for VMAF on the GPU
    where it pairs the frames itself: `graphs` as _build_filtergraph gives
    them for that, `outputs` as vmaf_cuda.GpuAttempt.output_args does."""
    commands = []
    for path, accel, graph, output in zip((distorted_path, source_path), (hwaccel.distorted, hwaccel.source),
                                          graphs.split("\n"), outputs, strict=True):
        commands.append([ffmpeg_path(), "-nostdin", "-hide_banner", "-y", *_hwaccel_args(accel),
                         "-i", str(Path(path).resolve()), "-lavfi", graph, "-progress", "pipe:1", "-nostats",
                         *output])
    return commands


def _run_ffmpeg_pair(
    commands: list[list[str]], total_frames: int, on_progress: ProgressCallback | None,
    cancel_event: threading.Event | None, cwd: Path, process_handle: ProcessHandle | None = None,
) -> subprocess.CompletedProcess:
    """_run_ffmpeg for the two FFmpegs of _build_stream_cmds, side by side:
    the first reports the progress; one that fails ends the other. The
    result is the failed one's, or the first's."""
    stop = threading.Event()
    results: list[subprocess.CompletedProcess | None] = [None] * len(commands)
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            results[index] = _run_ffmpeg(commands[index], total_frames, on_progress if index == 0 else None, stop,
                                         cwd=cwd, process_handle=process_handle)
            if results[index].returncode != 0:
                stop.set()
        except BaseException as error:  # raised below, in the caller's thread
            errors.append(error)
            stop.set()

    threads = [threading.Thread(target=run, args=(index,), name=f"ffmpeg-{index}", daemon=True)
               for index in range(len(commands))]
    for thread in threads:
        thread.start()
    for thread in threads:  # each in turn: a join on one that has ended returns at once
        while thread.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                stop.set()
            thread.join(timeout=0.1)
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    failed = next((result for result in results if result is not None and result.returncode != 0), None)
    if failed is not None:
        return failed
    for error in errors:
        if not isinstance(error, Cancelled):  # the one ended because the other failed
            raise error
    if errors:
        raise errors[0]
    return results[0]


def _build_resample_cmd(
    source_path: Path, filtergraph: str, hwaccel: str | None, duration_limit: float = 0.0,
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    cmd += _hwaccel_args(hwaccel)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit)
    return cmd


def _build_ffmpeg_output_args(
    filtergraph: str, duration_limit: float, gpu_outputs: list[str] | None = None,
) -> list[str]:
    """`gpu_outputs`: GPU VMAF's raw outputs (vmaf_cuda.GpuAttempt), each
    with its own -t, and the run's only outputs. Several filter graphs
    (_GRAPH_SEPARATOR): a -filter_complex each, and a null output each for
    its [graphN], each with the -t."""
    graphs = filtergraph.split(_GRAPH_SEPARATOR)
    if len(graphs) > 1 and gpu_outputs is None:
        args = [part for graph in graphs for part in ("-filter_complex", graph)] + ["-progress", "pipe:1", "-nostats"]
        for index in range(len(graphs)):
            args += ["-map", f"[graph{index}]"]
            if duration_limit > 0:
                args += ["-t", f"{duration_limit:.3f}"]
            args += ["-f", "null", "-"]
        return args
    args = ["-lavfi", filtergraph, "-progress", "pipe:1", "-nostats"]
    if gpu_outputs is not None:
        return args + gpu_outputs
    if duration_limit > 0:
        # An output-side -t caps how much of the filtered output is produced
        # (and so how many frames reach libvmaf), regardless of any length
        # mismatch between the two inputs -- simpler than trying to bound
        # each input separately.
        args += ["-t", f"{duration_limit:.3f}"]
    return [*args, "-f", "null", "-"]


def _cancelled(cancel_event: threading.Event | None, process_handle: ProcessHandle | None) -> bool:
    """Whether the run was cancelled: its event set, or its processes ended
    by Cancel (ProcessHandle.terminate). XPSNR's FFmpeg beside the app's
    metrics waits on an event of its own, set only once the app's scorer
    has stopped: an FFmpeg Cancel had ended was taken for a failure there,
    and the fallback started its next attempt."""
    return ((cancel_event is not None and cancel_event.is_set())
            or (process_handle is not None and process_handle.was_terminated))


def _run_ffmpeg(
    cmd: list[str], total_frames: int,
    on_progress: ProgressCallback | None, cancel_event: threading.Event | None,
    cwd: Path, process_handle: ProcessHandle | None = None,
) -> subprocess.CompletedProcess:
    # Checked before spawning, not only inside the read loop: cancelling
    # during crop detection or between the fallback attempts would otherwise
    # start one more ffmpeg that then had to be hunted down and killed.
    if _cancelled(cancel_event, process_handle):
        raise Cancelled("Cancelled by user")
    # UTF-8, not the Windows code page: ffmpeg's stderr starts with the
    # inputs' paths and tags, and a curly quote (”) in either is a byte cp1252
    # cannot decode. That killed the drain thread, losing ffmpeg's error
    # message and leaving nothing to empty the pipe.
    proc = proc_util.popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=str(cwd),
    )
    if process_handle is not None:
        process_handle.attach(proc.pid)
    # Bound before the try so the finally can always reach it, even if the
    # thread never got as far as being created.
    stderr_thread: threading.Thread | None = None
    try:
        stderr_lines: list[str] = []

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # ffmpeg's -progress output emits several key=value lines per update
        # block (frame=, fps=, ..., progress=continue/end) rather than one
        # combined line -- fps only changes once per block, so the latest
        # value seen is carried forward and reported alongside every frame=
        # update rather than waiting for both to land on the same line.
        last_fps = 0.0
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc_util.terminate(proc)
                break
            fps_match = _PROGRESS_FPS_RE.match(line)
            if fps_match:
                last_fps = float(fps_match.group(1))
                continue
            m = _PROGRESS_FRAME_RE.match(line)
            if m and on_progress:
                on_progress(int(m.group(1)), total_frames, last_fps)

        proc.wait()
        stderr_thread.join(timeout=5)
        # Checked again here (not just inside the loop above) because a
        # paused process produces no more output for the loop to see -- it's
        # only killed via ProcessHandle.terminate() from outside, which ends
        # the loop through EOF rather than the in-loop check ever firing.
        if _cancelled(cancel_event, process_handle):
            raise Cancelled("Cancelled by user")
        return subprocess.CompletedProcess(cmd, proc.returncode, "", "".join(stderr_lines))
    finally:
        # Anything can leave the block above early -- a cancellation, or an
        # on_progress callback raising from inside the stdout loop -- and an
        # ffmpeg left running holds its pipes and the log files inside the
        # run's temp dir open. On Windows that makes the enclosing
        # TemporaryDirectory fail to delete, so the leak is a visible one:
        # files pile up in %TEMP% for the rest of the session.
        _reap(proc, stderr_thread)
        if process_handle is not None:
            process_handle.detach(proc.pid)


def _reap(proc: subprocess.Popen, drain_thread: threading.Thread | None) -> None:
    """Ends `proc` if it is still running, then joins its reader and closes
    its pipes -- in that order, so the drain thread sees a clean EOF rather
    than having the file object closed underneath it.

    Deliberately swallows its own errors: this runs in a finally block, and
    the exception that sent us there (a Cancelled, or whatever a progress
    callback raised) is the one the caller needs to see -- a secondary
    failure while tidying up must not replace it.
    """
    with contextlib.suppress(Exception):  # see the docstring
        if proc.poll() is None:
            proc_util.terminate(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate() is a polite request that a wedged decoder can
                # ignore; kill() is not refusable.
                proc_util.kill(proc)
                proc.wait(timeout=5)
    if drain_thread is not None:
        drain_thread.join(timeout=5)
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(Exception):  # see the docstring
                pipe.close()


def estimate_total_frames(
    reference_info: VideoInfo, options: VmafOptions, other_info: VideoInfo | None = None
) -> int:
    """The number of frames ffmpeg will process: `reference_info` is
    whichever video drives the output timeline (the distorted video for a
    normal run, the source for a resolution round-trip test), bounded by
    duration_limit. libvmaf's n_subsample reduces how many frames receive a
    score, but ffmpeg's progress counter still reports every decoded/output
    frame, so applying n_subsample here made progress exceed 100% and broke
    both ETAs. Used to size progress and estimate the queued work.

    `other_info` is the second input of a two-input comparison. The graph now
    stops at whichever input ends first (see FRAMESYNC_OPTS), so a distorted
    file longer than its source produces fewer frames than its own length
    suggests -- without this the progress bar would stop short of 100% and
    the ETA would never be reached.
    """
    frame_count = reference_info.estimated_frame_count
    if other_info is not None:
        frame_count = min(frame_count, other_info.estimated_frame_count)
    if options.duration_limit > 0:
        frame_count = min(frame_count, round(options.duration_limit * reference_info.fps))
    return frame_count


def _resolve_model_for_cwd(model: str, tmpdir: Path) -> str:
    """If `model` points at a custom model file (model="path=<file>"), copy
    it into tmpdir and rewrite the option to reference it by bare filename,
    for the same reason log_path is kept relative -- see _build_filtergraph.
    """
    if not model.startswith("path="):
        return model
    src = Path(model[len("path="):])
    dest = tmpdir / src.name
    shutil.copyfile(src, dest)
    return f"path={dest.name}"


def _parse_log(log_path: Path, fps: float, xpsnr_log_path: Path | None = None) -> FrameScores:
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)

    xpsnr_by_frame = _parse_xpsnr_log(xpsnr_log_path) if xpsnr_log_path is not None else {}

    # Accumulated as plain lists and packed into arrays at the end, rather
    # than one FrameScore object per frame: a feature-length run is hundreds
    # of thousands of frames, and those objects would be built only to be
    # thrown away here.
    frame_nums: list[int] = []
    vmafs: list[float | None] = []
    negs: list[float | None] = []
    v1s: list[float | None] = []
    psnrs: list[float | None] = []
    ssims: list[float | None] = []
    xpsnrs: list[float | None] = []

    for fr in data.get("frames", []):
        metrics = fr.get("metrics", {})
        frame_num = int(fr.get("frameNum", len(frame_nums)))
        vmaf = metrics.get("vmaf")
        if not any(k in metrics for k in ("vmaf", "vmaf_neg", "vmaf_v1", "psnr_y", "psnr", "float_ssim", "ssim")):
            continue
        # `a if a is not None else b`, not `a or b`: libvmaf reports a real
        # 0.0 for badly degraded frames, and `or` would discard it and fall
        # through to the other key (or to None).
        psnr = metrics.get("psnr_y")
        if psnr is None:
            psnr = metrics.get("psnr")
        ssim = metrics.get("float_ssim")
        if ssim is None:
            ssim = metrics.get("ssim")
        frame_nums.append(frame_num)
        vmafs.append(None if vmaf is None else float(vmaf))
        negs.append(metrics.get("vmaf_neg"))
        v1s.append(metrics.get("vmaf_v1"))
        psnrs.append(psnr)
        ssims.append(ssim)
        xpsnrs.append(xpsnr_by_frame.get(frame_num))

    if not frame_nums:
        return FrameScores.empty()

    frame_arr = np.array(frame_nums, dtype=np.int32)
    time_arr = frame_arr / fps if fps > 0 else np.zeros(len(frame_nums), dtype=np.float64)

    def column(values: list[float | None]) -> np.ndarray | None:
        if all(v is None for v in values):
            return None  # metric wasn't requested for this run at all
        return np.array([np.nan if v is None else v for v in values], dtype=np.float32)

    return FrameScores(
        frame=frame_arr,
        time=time_arr,
        vmaf=column(vmafs),
        vmaf_neg=column(negs),
        psnr=column(psnrs), ssim=column(ssims), xpsnr=column(xpsnrs),
        metrics={"vmaf_v1": column(v1s)},
    )


_XPSNR_NUMBER = r"[+-]?(?:inf|nan|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
_XPSNR_LINE_RE = re.compile(
    rf"n:\s*(\d+)\s+XPSNR y:\s*({_XPSNR_NUMBER})", re.IGNORECASE
)


def _parse_xpsnr_log(xpsnr_log_path: Path) -> dict[int, float]:
    """Maps 0-indexed frame number -> XPSNR Y value. The xpsnr filter's own
    stats file numbers frames from 1, while the rest of this app numbers
    them from 0 (matching libvmaf's own frameNum) -- converted here so
    callers never have to think about the mismatch.
    """
    if not xpsnr_log_path.exists():
        return {}
    result: dict[int, float] = {}
    with open(xpsnr_log_path, encoding="utf-8") as f:
        for line in f:
            m = _XPSNR_LINE_RE.match(line)
            if m:
                result[int(m.group(1)) - 1] = float(m.group(2))
    return result


#: (hwaccel plan, model resolved relative to the run's temp dir, libvmaf
#: log path, xpsnr log path or None) -> the ffmpeg argv to run. The two run
#: flavours differ only in this, so _execute_run takes it as a parameter
#: rather than duplicating the whole pipeline around it.
CommandBuilder = Callable[..., list[str]]


@dataclass(frozen=True)
class _GpuPlan:
    """VMAF and NEG scored on the GPU (vmaf_cuda), the run's only metrics:
    libvmaf's models for them, and the size and depth frames are compared at.
    Or, with `backend` "cpu", PSNR and SSIM by the bundled libvmaf's CPU
    extractors (vmaf_cuda.CpuScorer): `models` maps their keys to
    themselves, `threads` is libvmaf's."""
    models: dict[str, str]
    width: int
    height: int
    bit_depth: int
    #: "cuda" (libvmaf), "vulkan" (vmaf_vulkan, on Vulkan's GPU `device`)
    #: or "cpu".
    backend: str = "cuda"
    device: int | None = None
    threads: int = 0
    #: XPSNR's, "cpu" (_xpsnr_frame_rate).
    frame_rate: int = 0


def _gpu_frame_scores(gpu_scores, fps: float) -> FrameScores:
    """The GPU's VMAF and NEG as the run's frame scores; VmafGpuError when it
    scored no frame."""
    numbers, scores = gpu_scores
    if not len(numbers):
        raise vmaf_cuda.VmafGpuError("GPU VMAF scored no frames")
    time = numbers / fps if fps > 0 else np.zeros(len(numbers), dtype=np.float64)
    v1 = scores.get("vmaf_v1")

    def column(key: str):  # float32, as _parse_log keeps FFmpeg's
        return None if scores.get(key) is None else np.asarray(scores[key], dtype=np.float32)

    return FrameScores(numbers, time, vmaf=scores.get("vmaf"), vmaf_neg=scores.get("vmaf_neg"),
                       psnr=column("psnr"), ssim=column("ssim"), xpsnr=column("xpsnr"),
                       metrics=None if v1 is None else {"vmaf_v1": np.asarray(v1, dtype=np.float32)})


def _fallback_ladder(plan: HwAccelPlan) -> list[HwAccelPlan]:
    """The plans to try, in order, until one of them runs.

    Hardware decode can fail for reasons no capability table predicts: a
    profile the fixed-function decoder does not implement, a driver that
    reports the codec but rejects the specific bitstream, an exhausted
    decode session. The distorted file is tried-then-dropped first because
    it is the arbitrary one -- the source is usually a known-good master
    while the distorted side is whatever encoder settings are under test.
    """
    ladder = [plan]
    if plan.source is not None and plan.distorted is not None:
        # The first single-input retry keeps the usually-known-good source
        # accelerated. If that is actually the failing side, the symmetric
        # retry still preserves acceleration for the distorted input.
        ladder.append(HwAccelPlan(source=plan.source))
        ladder.append(HwAccelPlan(distorted=plan.distorted))
    if plan.uses_gpu:
        ladder.append(HwAccelPlan())
    return ladder


def _with_v1_model(options: VmafOptions, dimensions: tuple[int, int]) -> VmafOptions:
    """The options with VMAF v1's model resolved, Auto from the size frames
    are compared at -- known only after black bars are detected, as for
    VMAF v0.6.1's Auto (_auto_model_or)."""
    if not options.compute_vmaf_v1:
        return options
    return replace(options, model_v1=resolve_v1_model(options, *dimensions))


def _auto_model_or(options: VmafOptions, dimensions: tuple[int, int]) -> str:
    """The model to actually run with. Only Auto is re-decided here; an
    explicit or custom choice is the user's and is left alone."""
    if not options.compute_vmaf:
        return ""
    if options.model_choice != AUTO_MODEL_CHOICE:
        return options.model
    return model_for_resolution(*dimensions)


def _execute_run(
    build_command: CommandBuilder,
    *,
    options: VmafOptions,
    model: str | None = None,
    fps: float,
    total_frames: int,
    hwaccel: HwAccelPlan,
    tmp_prefix: str,
    on_progress: ProgressCallback | None,
    on_status: Callable[[str], None] | None,
    cancel_event: threading.Event | None,
    process_handle: ProcessHandle | None,
    gpu: _GpuPlan | None = None,
) -> FrameScores:
    """Runs one ffmpeg invocation to completion and parses its logs.

    With `gpu`, VMAF and NEG -- the run's only metrics -- are scored on the
    GPU from FFmpeg's raw outputs (vmaf_cuda.GpuAttempt, one per attempt);
    build_command then also takes those outputs' arguments.

    Shared by run_vmaf and run_resample_test, which previously carried
    byte-identical copies of the temp-dir setup, the GPU-decode fallback, the
    exit-code/missing-log checks and the log parsing -- four places a fix had
    to be remembered in, and one of them would eventually be missed.
    """
    if not options.requested_metrics():
        raise VmafRunError("Select at least one metric to calculate.")
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        log_path = tmpdir / "vmaf_log.json"
        xpsnr_log_path = tmpdir / "xpsnr_log.txt" if options.compute_xpsnr else None
        resolved_model = _resolve_model_for_cwd(
            (model if model is not None else options.model) if options.compute_vmaf and gpu is None else "", tmpdir
        )
        gpu_scores = None
        if (v1_file := _v1_model_file(options)) is not None:
            shutil.copyfile(v1_file, tmpdir / v1_file.name)  # referenced by bare name, as above

        def run_with(plan: HwAccelPlan):
            nonlocal gpu_scores
            if gpu is None:
                cmd = build_command(plan, resolved_model, log_path, xpsnr_log_path)
                _log.info("FFmpeg: %s", _command_text(cmd))
                return _run_ffmpeg(
                    cmd, total_frames, on_progress, cancel_event,
                    cwd=tmpdir, process_handle=process_handle,
                )
            # A libvmaf context and pipes of its own for each attempt: a
            # failed attempt's are spent.
            attempt = vmaf_cuda.GpuAttempt(gpu.width, gpu.height, gpu.bit_depth, gpu.models, options.n_subsample,
                                           gpu.backend, gpu.device, threads=gpu.threads, frame_rate=gpu.frame_rate)
            try:
                # One frame more than the limit: FFmpeg's libvmaf filter scores
                # the first frame at or past it (stamped 30.03 s for a 30 s
                # limit at 23.976 fps) before the null output stops there, and
                # the raw outputs are to carry the same frames.
                limit = options.duration_limit + 1 / fps if options.duration_limit > 0 and fps > 0 else 0.0
                cmd = build_command(plan, resolved_model, log_path, xpsnr_log_path, attempt.output_args(limit))
                if isinstance(cmd[0], list):  # an FFmpeg for each video
                    for one in cmd:
                        _log.info("FFmpeg: %s", _command_text(one))
                    result = _run_ffmpeg_pair(cmd, total_frames, on_progress, cancel_event,
                                              cwd=tmpdir, process_handle=process_handle)
                else:
                    _log.info("FFmpeg: %s", _command_text(cmd))
                    result = _run_ffmpeg(
                        cmd, total_frames, on_progress, cancel_event,
                        cwd=tmpdir, process_handle=process_handle,
                    )
            except BaseException:
                with contextlib.suppress(Exception):
                    attempt.finish(False)
                raise
            # Raises when libvmaf failed: FFmpeg failing then is its doing,
            # and no decode retry would help.
            gpu_scores = attempt.finish(result.returncode == 0)
            return result

        ladder = _fallback_ladder(hwaccel)
        result = None
        for attempt, plan in enumerate(ladder):
            if on_status:
                if attempt == 0:
                    doing = ("" if gpu is None else f", {_cpu_metric_names(gpu.models)} in the app"
                             if gpu.backend == "cpu" else ", VMAF on the GPU")
                    on_status(Status.decoding(f"Running ffmpeg{doing}", plan, ending="...", kind=STARTING))
                else:
                    on_status(Status.decoding("GPU decode failed, retrying", plan, ending="..."))
            result = run_with(plan)
            if result.returncode == 0:
                break
            _log.warning("FFmpeg exited with code %d (GPU decode: %s)%s. Last output:\n%s", result.returncode,
                         plan.describe(), "; retrying" if attempt + 1 < len(ladder) else "",
                         "\n".join(result.stderr.splitlines()[-25:]))
            # A stale log from the failed attempt would otherwise be parsed
            # as if the retry had produced it -- ffmpeg can write a partial
            # log before the decoder gives up.
            log_path.unlink(missing_ok=True)
            if xpsnr_log_path is not None:
                xpsnr_log_path.unlink(missing_ok=True)

        assert result is not None  # the ladder always has at least one plan
        if result.returncode != 0:
            tail = "\n".join(result.stderr.splitlines()[-25:])
            raise VmafRunError(f"ffmpeg exited with code {result.returncode}", stderr_tail=tail)

        if gpu is not None:
            frames = _gpu_frame_scores(gpu_scores, fps)
        elif not _uses_vmaf_model(options) and not options.extra_features:
            values = _parse_xpsnr_log(xpsnr_log_path)
            numbers = np.array(sorted(values), dtype=np.int32)
            frames = FrameScores(numbers, numbers / fps, None,
                                 xpsnr=np.array([values[n] for n in numbers], dtype=np.float32))
        else:
            if not log_path.exists():
                raise VmafRunError("ffmpeg finished but no metric log was produced.", stderr_tail=result.stderr[-2000:])
            frames = _parse_log(log_path, fps, xpsnr_log_path)
        missing = [m for m in options.requested_metrics() if not frames.has(m)]
        if not frames or missing:
            raise VmafRunError("No results for requested metrics: " + ", ".join(missing or options.requested_metrics()))
        return frames


def run_vmaf(
    source_info: VideoInfo,
    distorted_info: VideoInfo,
    options: VmafOptions,
    on_progress: ProgressCallback | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    result_distorted_path: Path | None = None,
) -> ComparisonResult:
    """result_distorted_path overrides the returned result's `distorted`
    identity (defaulting to distorted_info.path). It doesn't affect which
    file is actually decoded -- only what identity the result carries for
    caching/graphing -- so a caller running the *same* physical file twice
    under different options (e.g. both ScaleDirection values) can give each
    run a distinct identity instead of one colliding with/overwriting the
    other, the same way a resample test's synthetic path already does.
    """
    validate_video_pair(source_info, distorted_info, options)

    # Planned before crop detection rather than after, so the detection
    # windows can decode the way the run will. The plan depends only on the
    # codecs and the vendor, never on the crops.
    hwaccel = HwAccelPlan()
    if options.gpu_decode:
        hwaccel = plan_hwaccel(
            options.gpu_vendor, source_info.codec_name, distorted_info.codec_name,
            source_pix_fmt=source_info.pix_fmt, distorted_pix_fmt=distorted_info.pix_fmt,
            source_size=(source_info.width, source_info.height),
            distorted_size=(distorted_info.width, distorted_info.height),
        )

    source_crop, distorted_crop = _resolve_crops(
        source_info, distorted_info, options, on_status,
        cancel_event=cancel_event, process_handle=process_handle, hwaccel=hwaccel,
    )
    # After cropping, not before: removing a letterbox is precisely what
    # makes a padded source and an already-cropped encode the same shape.
    validate_display_geometry(source_info, distorted_info, source_crop, distorted_crop)

    # Auto picks its model from the size frames are compared at, which is
    # only known now: it depends on the scale direction and on crops that
    # were detected a moment ago, not on either input's own resolution.
    dimensions = analysis_dimensions(source_info, distorted_info, options, source_crop, distorted_crop)
    effective_model = _auto_model_or(options, dimensions)
    options = _with_v1_model(options, dimensions)

    def build_command(plan, model, log_path, xpsnr_log_path):
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, plan, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_ffmpeg_cmd(
            distorted_info.path, source_info.path, filtergraph, plan, options.duration_limit,
        )

    total_frames = estimate_total_frames(distorted_info, options, source_info)
    frames = None
    # On the GPU only when VMAF and NEG are all the run scores: the window's
    # runs split them from FFmpeg's other metrics (worker, GPU_VMAF). One
    # FFmpeg feeding both, which nothing used any more, could see its GPU
    # result refused for a frame count FFmpeg's filters disagreed with, and
    # the whole run made again on the CPU.
    gpu_models = None
    gpu_backend = "cuda"
    requested = set(options.requested_metrics())
    if not requested - {"vmaf", "vmaf_neg", "vmaf_v1"}:
        gpu_models = vmaf_cuda.scores_on_gpu(options.compute_vmaf, options.compute_vmaf_neg, effective_model,
                                             options.vmaf_on_gpu, analysis_bit_depth(source_info, distorted_info),
                                             size=dimensions, compute_vmaf_v1=options.compute_vmaf_v1,
                                             model_v1=options.model_v1)
        if gpu_models is not None and set(gpu_models) != requested:
            gpu_models = None  # one of them is the CPU's: FFmpeg's libvmaf calculates them together
    if gpu_models is not None:
        plan = _GpuPlan(gpu_models, *dimensions, analysis_bit_depth(source_info, distorted_info),
                        *vmaf_cuda.gpu_vmaf_backend())
        gpu_backend = plan.backend
        frames = _run_on_gpu(
            plan, source_info, distorted_info, options, source_crop, distorted_crop, effective_model, hwaccel,
            total_frames, on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle,
        )
        if frames is None:
            gpu_models = None
    cpu_keys: set[str] = set()
    if frames is None and (cpu := _cpu_metrics_plan(options, dimensions, source_info, distorted_info, hwaccel,
                                                     source_crop, distorted_crop)) is not None:
        frames = _run_cpu_metrics(
            cpu, source_info, distorted_info, options, source_crop, distorted_crop, hwaccel, total_frames,
            model=effective_model, on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
            process_handle=process_handle,
        )
        if frames is not None:
            cpu_keys = set(cpu.models)
    if frames is None:
        frames = _execute_run(
            build_command,
            options=options,
            model=effective_model,
            fps=distorted_info.fps,
            total_frames=total_frames,
            hwaccel=hwaccel,
            tmp_prefix="vmaf_run_",
            on_progress=on_progress,
            on_status=on_status,
            cancel_event=cancel_event,
            process_handle=process_handle,
        )
    compared = int(frames.frame[-1]) + 1 if len(frames) else 0
    if (short := short_comparison(total_frames, compared, distorted_info.fps, options.n_subsample)) is not None:
        raise VmafRunError(short)

    return ComparisonResult(
        source=source_info.path,
        distorted=result_distorted_path or distorted_info.path,
        frames=frames,
        fps=distorted_info.fps,
        model=effective_model,
        source_crop=source_crop,
        distorted_crop=distorted_crop,
        source_info=source_info,
        distorted_info=distorted_info,
        scale_direction=options.scale_direction,
        scale_algorithm=options.scale_algorithm,
        compared_frame_count=total_frames,
        model_choice=options.model_choice,
        model_v1=options.model_v1,
        model_choice_v1=options.model_choice_v1 if options.compute_vmaf_v1 else None,
        metric_results=_metric_results_for_current_run(
            frames, effective_model, options.model_v1, gpu_keys=set(gpu_models or ()) & set(frames.metric_keys),
            gpu_backend=gpu_backend, cpu_keys=cpu_keys & set(frames.metric_keys),
        ),
    )


#: How the status line a run sends when VMAF on the GPU fails begins: FFmpeg's
#: libvmaf calculates it from then on. The worker recognises it, so the run
#: line stops showing that video's VMAF as GPU work.
VMAF_GPU_FAILED = "VMAF on the GPU failed"


def _run_on_gpu(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, model: str, hwaccel: HwAccelPlan,
    total_frames: int, *, on_progress, on_status, cancel_event, process_handle,
) -> FrameScores | None:
    """The run with VMAF and NEG on the GPU (vmaf_cuda), in a process of its
    own: a crash in libvmaf or the NVIDIA driver ends that process, not the
    app. None when it fails for any reason but Cancel: the run is then made
    again with VMAF on the CPU, as on a PC without the GPU.

    One GPU pass at a time with Vship's (gpu.GPU_PASS): VMAF on the GPU does
    not run beside another video's GPU metrics."""
    _log.info("VMAF on the GPU (%s): %s", ", ".join(plan.models.values()), _gpu_build(plan.backend))
    if not GPU_PASS.acquire(blocking=False):
        if on_status:
            on_status(Status(GPU_WAIT_MESSAGE, kind=GPU_WAIT))
        while not GPU_PASS.acquire(timeout=0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled("Cancelled by user")
    try:
        return run_isolated(
            _score_on_gpu, plan, source_info, distorted_info, options, source_crop, distorted_crop, model, hwaccel,
            total_frames, what="libvmaf", callbacks=("on_progress", "on_status"), on_progress=on_progress,
            on_status=on_status, cancel_event=cancel_event, process_handle=process_handle, cancelled=Cancelled,
        )
    except Cancelled:
        raise
    except Exception as error:
        _log.error("VMAF on the GPU failed; calculating it on the CPU: %s", error, exc_info=error)
        if on_status:
            on_status(Status(f"{VMAF_GPU_FAILED} ({error}); calculating it on the CPU…", kind=GPU_VMAF_FAILED))
        return None
    finally:
        GPU_PASS.release()


def _score_on_gpu(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, model: str, hwaccel: HwAccelPlan,
    total_frames: int, *, on_progress=None, on_status=None, cancel_event=None, process_handle=None,
) -> FrameScores:
    """Run by _run_on_gpu in its own process: FFmpeg decodes and pairs the
    frames as on the CPU, and they are fed to libvmaf.

    When the GPU's own decoder decodes both videos, they are decoded in this
    process instead (vmaf_cuda.score_decoded, vmaf_vulkan.score_decoded):
    the same frames, without FFmpeg's decode and the CPU copies and pipes
    behind it. libvmaf's CUDA code takes NVIDIA's decoder's pictures on the
    GPU; the Vulkan scorer takes the decoder FFmpeg would have used for both
    videos -- NVIDIA's, Intel's or AMD's (_DECODED_HERE) -- whose pictures
    it uploads. If that decoding fails after it has started, the run is
    made again with FFmpeg's, as before."""
    decoder = _DECODED_HERE.get(hwaccel.source or "") if hwaccel.source == hwaccel.distorted else None
    if decoder is not None and ("vmaf_v1" in plan.models or plan.backend == "vulkan" or decoder == "nvidia"):
        try:
            return _score_decoded_on_gpu(
                plan, source_info, distorted_info, options, source_crop, distorted_crop, hwaccel, total_frames,
                on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
                process_handle=process_handle)
        except gpu_frames.GpuDecodeUnavailableError as error:
            _log.info("VMAF on the GPU: the videos are decoded by FFmpeg (%s)", error)
        except gpu_frames.GpuDecodeFailedError as error:
            _log.warning("GPU decoding for VMAF on the GPU failed; decoding through FFmpeg instead: %s", error)
            if on_status:
                on_status(f"GPU decoding failed ({error}); decoding through FFmpeg instead…")

    def build_command(hw, resolved_model, log_path, xpsnr_log_path, gpu_outputs):
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, hw, log_path,
            model=resolved_model, xpsnr_log_path=xpsnr_log_path, gpu_vmaf=True,
            gpu_paired=not isinstance(gpu_outputs, tuple),
        )
        if isinstance(gpu_outputs, tuple):  # an FFmpeg for each video: the GPU's side pairs the frames
            return _build_stream_cmds(distorted_info.path, source_info.path, filtergraph, hw, gpu_outputs)
        return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                 options.duration_limit, gpu_outputs)

    return _execute_run(
        build_command, options=options, model=model, fps=distorted_info.fps, total_frames=total_frames,
        hwaccel=hwaccel, tmp_prefix="vmaf_gpu_run_", on_progress=on_progress, on_status=on_status,
        cancel_event=cancel_event, process_handle=process_handle, gpu=plan,
    )


#: Set to "ffmpeg", PSNR and SSIM are left to FFmpeg's libvmaf filter, as
#: before _run_cpu_metrics (for comparing the two).
CPU_METRICS_VARIABLE = "VML_CPU_METRICS"
#: FFmpeg's libvmaf features -> the metrics vmaf_cuda.CpuScorer scores.
_CPU_METRIC_OF_FEATURE = {"name=psnr": "psnr", "name=float_ssim": "ssim"}


_CPU_METRIC_NAMES = {"psnr": "PSNR", "ssim": "SSIM", "xpsnr": "XPSNR"}


def _cpu_metric_names(metrics) -> str:
    """"PSNR, SSIM and XPSNR", for the statuses and the log."""
    names = [_CPU_METRIC_NAMES.get(metric, metric) for metric in metrics]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _xpsnr_frame_rate(info: VideoInfo) -> int | None:
    """XPSNR's frame rate as FFmpeg's xpsnr filter has it from its second
    input (the test video: _xpsnr_filter), in whole frames a second: below
    32 its temporal activity is first-order, else second. FFmpeg gives the input
    av_guess_frame_rate() (fftools/ffmpeg_demux.c): r_frame_rate where the
    average rate is known and within a tenth of it. Otherwise it may be the
    codec's own (for a codec with fields, H.264) or the average: None, and
    FFmpeg's filter scores XPSNR."""
    average, nominal = info.average_fps, info.nominal_fps
    if average <= 0 or nominal <= 0 or abs(1.0 - average / nominal) >= 0.09:
        return None
    return int(nominal)


def _xpsnr_in_app(width: int, height: int, bit_depth: int) -> bool:
    """Where libvmaf-fast's port of FFmpeg's xpsnr filter gives its score to
    the last bit: up to 12 bits, and an even size above 2048x1152 (where
    FFmpeg's filter reads beyond the picture of an odd one)."""
    if bit_depth > 12 or (width * height > 2048 * 1152 and (width & 1 or height & 1)):
        return False
    return vmaf_cuda.cpu_scores_xpsnr()


def _decoded_in_app(source_info: VideoInfo, distorted_info: VideoInfo, source_crop: CropBox | None,
                    distorted_crop: CropBox | None, size: tuple[int, int], hwaccel: HwAccelPlan) -> bool:
    """Whether the scoring process decodes both videos itself
    (vmaf_cuda.score_decoded_cpu), as far as is known before it asks the
    decoder, rather than taking FFmpeg's frames through pipes."""
    if hwaccel.source != hwaccel.distorted or hwaccel.source not in _DECODED_HERE:
        return False
    try:
        return not any(gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=size).scaled
                       for info, crop in ((source_info, source_crop), (distorted_info, distorted_crop)))
    except gpu_frames.GpuDecodeUnavailableError:
        return False


def _cpu_metrics_plan(options: VmafOptions, dimensions: tuple[int, int], source_info: VideoInfo,
                      distorted_info: VideoInfo, hwaccel: HwAccelPlan, source_crop: CropBox | None = None,
                      distorted_crop: CropBox | None = None) -> _GpuPlan | None:
    """PSNR, SSIM and XPSNR scored by the bundled libvmaf in the app
    (_run_cpu_metrics) instead of by FFmpeg's libvmaf and xpsnr filters:
    where they are the run's only metrics (VMAF on the CPU keeps them in
    FFmpeg's filters, which it runs anyway), and frames can come to the app
    as for VMAF on the GPU. XPSNR stays FFmpeg's where libvmaf-fast's would
    not be its score to the bit (_xpsnr_frame_rate, _xpsnr_in_app), beside
    PSNR and SSIM in the app. None: FFmpeg's filters, as before."""
    if os.environ.get(CPU_METRICS_VARIABLE, "").casefold() == "ffmpeg":
        return None
    if _uses_vmaf_model(options) or options.resample_test is not None:
        return None
    if any(feature not in _CPU_METRIC_OF_FEATURE for feature in options.extra_features):
        return None
    if not vmaf_cuda.LIBRARY_PATH.is_file():
        return None
    width, height = dimensions
    bit_depth = analysis_bit_depth(source_info, distorted_info)
    # FFmpeg's pairing (an FFmpeg older than 6.1) takes an even size and the
    # formats overlay holds unchanged (_gpu_pairs_stage).
    if not vmaf_cuda.pairs_in_app() and (width & 1 or height & 1 or bit_depth > 10):
        return None
    metrics = list(dict.fromkeys(_CPU_METRIC_OF_FEATURE[feature] for feature in options.extra_features))
    frame_rate = _xpsnr_frame_rate(distorted_info) if options.compute_xpsnr else None
    if frame_rate is not None and _xpsnr_in_app(width, height, bit_depth):
        metrics.append("xpsnr")
    elif options.compute_xpsnr and not (hwaccel.source and hwaccel.distorted):
        # XPSNR's FFmpeg beside them decodes the videos again, and a video
        # the CPU decodes is then decoded twice on it -- 70% more CPU a
        # frame for a 4K film against its encode, and a 4K film against a
        # 1080p encode ran slower (97 fps, 112 in FFmpeg's one run) -- where
        # FFmpeg's one run decodes each once for its libvmaf and XPSNR.
        return None
    if not metrics:
        return None
    if metrics == ["xpsnr"] and not _decoded_in_app(source_info, distorted_info, source_crop, distorted_crop,
                                                     dimensions, hwaccel):
        # XPSNR alone from FFmpeg's pipes: FFmpeg's filter keeps up with its
        # decoding there, in far less -- a 4K film against a 1080p encode,
        # NVIDIA decoding: 128 fps, 159 in the app, with twice the CPU and
        # 825 MB against 453; decoded on the CPU, 77 fps and 81, 1.8 GB and
        # 2.7.
        return None
    threads = options.n_threads if options.n_threads > 0 else auto_threads()
    return _GpuPlan({metric: metric for metric in metrics}, width, height, bit_depth, backend="cpu",
                    threads=threads, frame_rate=frame_rate or 0)


class _Progress:
    """Two runs' progress as one: the frames of the one that is behind."""

    def __init__(self, on_progress: ProgressCallback | None, parts: int):
        self._on_progress = on_progress
        self._latest: list[tuple[int, int, float] | None] = [None] * parts
        self._lock = threading.Lock()

    def part(self, index: int) -> ProgressCallback | None:
        if self._on_progress is None:
            return None

        def report(done: int, total: int, fps: float) -> None:
            with self._lock:
                self._latest[index] = (done, total, fps)
                if any(latest is None for latest in self._latest):
                    return
                behind = min(self._latest, key=lambda latest: latest[0])
            self._on_progress(*behind)

        return report


def _run_cpu_metrics(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, hwaccel: HwAccelPlan, total_frames: int, *,
    model: str, on_progress, on_status, cancel_event, process_handle,
) -> FrameScores | None:
    """PSNR, SSIM and XPSNR by the bundled libvmaf's CPU extractors, in a
    process of their own (a crash ends that, not the app), from the videos'
    frames as they would reach FFmpeg's filters -- decoded there by the
    GPU's decoder, or written by each video's FFmpeg (_score_cpu_metrics)
    -- and paired as those filters pair them (frame_sync), as for VMAF on
    the GPU. XPSNR the plan leaves to FFmpeg is scored in an FFmpeg of its
    own beside them, its scores taken for the frames PSNR and SSIM were
    scored for (as _parse_log takes them). XPSNR alone is scored for every
    frame, as FFmpeg's filter scores it whatever n_subsample.

    FFmpeg's libvmaf filter allocates, zeroes and copies two new pictures
    for every pair on its one filter thread before libvmaf's threads see
    them. None when the run fails for any reason but Cancel: FFmpeg's
    filter then calculates them, as before."""
    names = _cpu_metric_names(plan.models)
    _log.info("%s in the app: %s, %d threads", names, vmaf_cuda.CPU_BUILD, plan.threads)
    beside = options.compute_xpsnr and "xpsnr" not in plan.models
    xpsnr_frames: list[FrameScores] = []
    xpsnr_error: list[BaseException] = []
    progress = _Progress(on_progress, 2 if beside else 1)
    stop = threading.Event()
    xpsnr_thread = None
    if beside:
        xpsnr_options = replace(options, extra_features=[])

        def build_command(hw, resolved_model, log_path, xpsnr_log_path):
            # XPSNR's alone: run_vmaf's command would score PSNR and SSIM
            # there again, at FFmpeg's speed.
            filtergraph = _build_filtergraph(
                source_info, distorted_info, xpsnr_options, source_crop, distorted_crop, hw, log_path,
                model=resolved_model, xpsnr_log_path=xpsnr_log_path,
            )
            return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                     xpsnr_options.duration_limit)

        def xpsnr() -> None:
            try:
                xpsnr_frames.append(_execute_run(
                    build_command, options=xpsnr_options, model=model, fps=distorted_info.fps,
                    total_frames=total_frames, hwaccel=hwaccel, tmp_prefix="vmaf_xpsnr_run_",
                    on_progress=progress.part(1), on_status=None, cancel_event=stop,
                    process_handle=process_handle))
            except BaseException as error:  # raised in the caller's thread
                xpsnr_error.append(error)

        xpsnr_thread = threading.Thread(target=xpsnr, name="vmaf-xpsnr", daemon=True)
        xpsnr_thread.start()

    def wait_for_xpsnr() -> None:
        if xpsnr_thread is None:
            return
        while xpsnr_thread.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                stop.set()
            xpsnr_thread.join(timeout=0.1)

    if beside:
        scored = replace(options, compute_xpsnr=False)
    elif not options.extra_features:  # XPSNR alone
        scored = replace(options, n_subsample=1)
    else:
        scored = options
    try:
        frames = run_isolated(
            _score_cpu_metrics, plan, source_info, distorted_info, scored,
            source_crop, distorted_crop, hwaccel, total_frames, what="libvmaf",
            callbacks=("on_progress", "on_status"), on_progress=progress.part(0), on_status=on_status,
            cancel_event=cancel_event, process_handle=process_handle, cancelled=Cancelled,
        )
    except Cancelled:
        stop.set()
        wait_for_xpsnr()
        raise
    except Exception as error:
        stop.set()
        wait_for_xpsnr()
        _log.error("%s in the app failed; FFmpeg calculates them: %s", names, error, exc_info=error)
        if on_status:
            on_status(f"{names} in the app failed ({error}); calculating them with FFmpeg…")
        return None
    wait_for_xpsnr()
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    if xpsnr_error:
        raise xpsnr_error[0]
    if xpsnr_frames:
        found = xpsnr_frames[0]
        by_frame = dict(zip(found.frame.tolist(), found.values("xpsnr").tolist(), strict=True))
        xpsnr = np.array([by_frame.get(number, np.nan) for number in frames.frame.tolist()], dtype=np.float32)
        frames = FrameScores(frames.frame, frames.time, None, psnr=frames.values("psnr") if frames.has("psnr")
                             else None, ssim=frames.values("ssim") if frames.has("ssim") else None, xpsnr=xpsnr)
    return frames


def _score_cpu_metrics(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, hwaccel: HwAccelPlan, total_frames: int, *,
    on_progress=None, on_status=None, cancel_event=None, process_handle=None,
) -> FrameScores:
    """Run by _run_cpu_metrics in its own process: FFmpeg decodes, crops,
    scales and converts the videos as for its libvmaf filter, and the app
    pairs and scores them (vmaf_cuda.GpuAttempt with a CpuScorer). Where the
    GPU's own decoder decodes both videos, and nothing is scaled, they are
    decoded in this process instead (vmaf_cuda.score_decoded_cpu): the same
    frames, without FFmpeg, its conversions and the pipes."""
    if hwaccel.source == hwaccel.distorted and hwaccel.source in _DECODED_HERE:
        try:
            return _score_decoded_on_gpu(
                plan, source_info, distorted_info, options, source_crop, distorted_crop, hwaccel, total_frames,
                on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
                process_handle=process_handle)
        except gpu_frames.GpuDecodeUnavailableError as error:
            _log.info("%s in the app: the videos are decoded by FFmpeg (%s)", _cpu_metric_names(plan.models), error)
        except gpu_frames.GpuDecodeFailedError as error:
            _log.warning("GPU decoding for %s failed; decoding through FFmpeg instead: %s",
                         _cpu_metric_names(plan.models), error)
            if on_status:
                on_status(f"GPU decoding failed ({error}); decoding through FFmpeg instead…")

    def build_command(hw, resolved_model, log_path, xpsnr_log_path, gpu_outputs):
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, hw, log_path,
            model=resolved_model, xpsnr_log_path=xpsnr_log_path, gpu_vmaf=True,
            gpu_paired=not isinstance(gpu_outputs, tuple),
        )
        if isinstance(gpu_outputs, tuple):  # an FFmpeg for each video: the app pairs the frames
            return _build_stream_cmds(distorted_info.path, source_info.path, filtergraph, hw, gpu_outputs)
        return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                 options.duration_limit, gpu_outputs)

    return _execute_run(
        build_command, options=options, model="", fps=distorted_info.fps, total_frames=total_frames,
        hwaccel=hwaccel, tmp_prefix="vmaf_cpu_run_", on_progress=on_progress, on_status=on_status,
        cancel_event=cancel_event, process_handle=process_handle, gpu=plan,
    )


#: The decoder in the scoring process for what FFmpeg would decode with each
#: -hwaccel (as perceptual_vship's _GPU_DECODERS).
_DECODED_HERE = {"cuda": "nvidia", "qsv": "intel", "d3d11va": "amd"}


def _score_decoded_on_gpu(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, hwaccel: HwAccelPlan, total_frames: int, *,
    on_progress=None, on_status=None, cancel_event=None, process_handle=None,
) -> FrameScores:
    """VMAF and NEG from videos decoded in this process (vmaf_cuda.score_decoded,
    or vmaf_vulkan.score_decoded for the Vulkan backend), over the frames
    _execute_run's FFmpeg would give libvmaf on the GPU; PSNR and SSIM for the
    "cpu" backend (vmaf_cuda.score_decoded_cpu, for _score_cpu_metrics)."""
    fps = distorted_info.fps
    # As _execute_run: one frame more than the limit, which FFmpeg's libvmaf
    # filter scores before its output stops.
    limit = options.duration_limit + 1 / fps if options.duration_limit > 0 and fps > 0 else 0.0
    if on_status:
        on_status(Status.decoding(f"Running {_cpu_metric_names(plan.models)} in the app" if plan.backend == "cpu"
                                  else "Running VMAF on the GPU", hwaccel, ending="..."))

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("Cancelled by user")

    if plan.backend == "cpu":
        score_decoded = functools.partial(vmaf_cuda.score_decoded_cpu, threads=plan.threads,
                                          frame_rate=plan.frame_rate, decoder=_DECODED_HERE[hwaccel.source])
    elif "vmaf_v1" in plan.models:  # whole frames, for its scorers and any of VMAF v0.6.1's beside them
        score_decoded = functools.partial(vmaf_v1_gpu.score_decoded, backend=plan.backend, device=plan.device,
                                          decoder=_DECODED_HERE[hwaccel.source])
    elif plan.backend == "vulkan":
        score_decoded = functools.partial(vmaf_vulkan.score_decoded, device=plan.device,
                                          decoder=_DECODED_HERE[hwaccel.source])
    else:
        score_decoded = vmaf_cuda.score_decoded
    scores = score_decoded(
        source_info, distorted_info, source_crop, distorted_crop, width=plan.width, height=plan.height,
        bit_depth=plan.bit_depth, models=plan.models, n_subsample=options.n_subsample,
        duration_limit=f"{limit:.3f}" if limit > 0 else None, total_frames=total_frames,
        scale_algorithm=options.scale_algorithm,
        on_progress=on_progress, check_cancel=check_cancel, process_handle=process_handle)
    frames = _gpu_frame_scores(scores, fps)
    missing = [m for m in options.requested_metrics() if not frames.has(m)]
    if not frames or missing:
        raise VmafRunError("No results for requested metrics: " + ", ".join(missing or options.requested_metrics()))
    return frames


def run_resample_test(
    source_info: VideoInfo,
    options: VmafOptions,
    on_progress: ProgressCallback | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> ComparisonResult:
    """Runs a resolution round-trip test (see VmafOptions.resample_test):
    downscales the source to a target width, scales it back up to the
    source's original resolution, and computes VMAF against the untouched
    source -- from a single input file, not a second already-encoded one.
    """
    assert options.resample_test is not None

    hwaccel = HwAccelPlan()
    if options.gpu_decode:
        # One input file, so there is no distorted side to decide.
        hwaccel = plan_hwaccel(options.gpu_vendor, source_info.codec_name, source_pix_fmt=source_info.pix_fmt,
                               source_size=(source_info.width, source_info.height))

    source_crop: CropBox | None = None
    if options.crop_mode == CropMode.AUTO:
        if on_status:
            on_status("Detecting black bars in source...")
        try:
            source_crop = detect_crop(
                source_info, cancel_event=cancel_event, process_handle=process_handle,
                hwaccel=hwaccel.source,
            )
        except CropDetectCancelled as e:
            raise Cancelled("Cancelled by user") from e

    dimensions = resample_analysis_dimensions(source_info, source_crop)
    effective_model = _auto_model_or(options, dimensions)
    options = _with_v1_model(options, dimensions)

    def build_command(plan, model, log_path, xpsnr_log_path):
        filtergraph = _build_resample_test_filtergraph(
            source_info, options, source_crop, plan.source, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_resample_cmd(source_info.path, filtergraph, plan.source, options.duration_limit)

    total_frames = estimate_total_frames(source_info, options)
    frames = _execute_run(
        build_command,
        options=options,
        model=effective_model,
        fps=source_info.fps,
        total_frames=total_frames,
        hwaccel=hwaccel,
        tmp_prefix="vmaf_resample_",
        on_progress=on_progress,
        on_status=on_status,
        cancel_event=cancel_event,
        process_handle=process_handle,
    )

    distorted_path = synthetic_resample_distorted_path(source_info.path, options.resample_test)
    return ComparisonResult(
        source=source_info.path,
        distorted=distorted_path,
        frames=frames,
        fps=source_info.fps,
        model=effective_model,
        source_crop=source_crop,
        distorted_crop=source_crop,  # same crop applies to both branches, since both come from the same source
        source_info=source_info,
        distorted_info=source_info,  # after the round trip it's back at the source's own resolution
        scale_algorithm=options.scale_algorithm,
        resample_target=options.resample_test,
        compared_frame_count=total_frames,
        model_choice=options.model_choice,
        model_v1=options.model_v1,
        model_choice_v1=options.model_choice_v1 if options.compute_vmaf_v1 else None,
        metric_results=_metric_results_for_current_run(frames, effective_model, options.model_v1),
    )
