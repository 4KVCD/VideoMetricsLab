"""Runs a comparison's FFmpeg-based metrics: VMAF, VMAF NEG, VMAF v1, PSNR,
SSIM and XPSNR.

run_vmaf builds the FFmpeg command (the filter graph that crops, scales and
pairs the two videos, and libvmaf's and xpsnr's options), runs it while
reporting progress, and parses the per-frame scores from the logs. When the
GPU can score VMAF, run_vmaf has FFmpeg decode and pair the frames and
scores them on the GPU (vmaf_gpu, vmaf_vulkan, vmaf_v1_gpu), or decodes the
videos in a process of its own without FFmpeg. It falls back to the CPU when
the GPU fails.
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

from vmaf_app.core import gpu_frames, libvmaf, libvmaf_cpu, vmaf_gpu, vmaf_v1_gpu, vmaf_vulkan
from vmaf_app.core import proc as proc_util
from vmaf_app.core.crop_detect import CropDetectCancelled, common_picture, detect_crop, detect_pair
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, check_tools, ffmpeg_path, format_version
from vmaf_app.core.frame_coverage import short_comparison
from vmaf_app.core.frame_sync import by_timestamp, duration_args, frame_rate, framesync_opts, pairing_filters
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
    """A command as one line for the log, ready to copy and paste. Never
    raises, because a log line must not be able to stop a run."""
    try:
        return subprocess.list2cmdline(command)
    except TypeError:
        return repr(command)

ProgressCallback = Callable[[int, int, float], None]  # (current_frame, total_frames, fps)


def _metric_results_for_current_run(frames: FrameScores, model: str, model_v1: str = "",
                                    gpu_keys: set[str] | None = None, gpu_backend: str = "cuda",
                                    cpu_keys: set[str] | None = None):
    """Turns one run's parsed scores into metric results, each with its
    provenance (what calculated it).

    `gpu_keys` are the metrics scored on the GPU (vmaf_gpu): VMAF and NEG
    keep the same request identity as FFmpeg's libvmaf, because their scores
    agree to within 4e-5, but are recorded as GPU scores of the bundled
    build, made by libvmaf's CUDA code or its Vulkan port (`gpu_backend`).
    `cpu_keys` are the metrics (PSNR, SSIM, XPSNR) scored by the bundled
    libvmaf's CPU extractors in the app (libvmaf_cpu.CpuScorer), which give
    FFmpeg's libvmaf's scores and so keep the same identity too."""
    status = check_tools()
    version = format_version(status.ffmpeg.version) if status.ffmpeg.runnable else "unknown"

    def provenance(key: str):
        parameters = ({"model": "version=vmaf_v0.6.1neg"} if key == "vmaf_neg"
                      else {"model": model} if key == "vmaf"
                      else {"model": model_v1} if key == "vmaf_v1" else None)
        made = current_ffmpeg_provenance(key, version, parameters)
        if key in (cpu_keys or ()):
            made = replace(made, implementation="libvmaf", implementation_version=libvmaf_cpu.CPU_BUILD)
        if key in (gpu_keys or ()):
            # VMAF v1's GPU part runs on Vulkan on every GPU (vmaf_v1_gpu).
            backend = "vulkan" if key == "vmaf_v1" else gpu_backend
            made = replace(made, implementation=f"libvmaf/{backend}", compute_backend="gpu",
                           implementation_version=vmaf_v1_gpu.LIBRARY_BUILD if key == "vmaf_v1"
                           else _gpu_build(gpu_backend))
        return made

    return results_from_frame_scores(frames, {key: provenance(key) for key in frames.metric_keys})


def _gpu_build(backend: str) -> str:
    """The library that scores VMAF on the GPU with `backend`, as provenance
    records it."""
    return vmaf_vulkan.LIBRARY_BUILD if backend == "vulkan" else vmaf_gpu.LIBRARY_BUILD


class VmafRunError(RuntimeError):
    """A run failed. `stderr_tail` holds the end of FFmpeg's error output,
    when there is one."""

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
    """Rejects a pair whose timelines do not allow a comparison (see
    geometry.pair_problem)."""
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
    """Detects each video's black bars when the options ask for it, and
    returns the crops that leave the same picture of both (or None, None).

    `hwaccel` is the run's own decode plan, so the detection decodes each
    video the same way the run will. That only affects speed: the crop found
    is the same."""
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

    A video scored on its own gets every logical core. When several are
    scored at once, each gets an equal share, so two jobs on a 24-core PC
    ask for 12 threads each rather than 24 each. Twice as many threads as
    cores cannot do more work than exactly as many: they only take turns on
    the same cores and pay for the switching.
    """
    cores = os.cpu_count() or 1
    return max(1, cores // max(1, concurrent_jobs))


def _build_libvmaf_opts(options: VmafOptions, log_path: Path, model: str | None = None) -> list[str]:
    """The libvmaf filter's options: its log, the models, threads,
    subsampling, extra features (PSNR, SSIM) and frame sync."""
    # FFmpeg's filter graph option parser cannot reliably take an absolute
    # Windows path (with the drive letter's colon) as an option value, even
    # escaped or quoted. So FFmpeg is always started in log_path's folder,
    # and the log (and any custom model file) is named by its bare file name.
    model_value = model if model is not None else options.model
    v1_file = _v1_model_file(options)
    if options.compute_vmaf_neg or v1_file is not None:
        # Each model gets an explicit name, so each has its own scores in the
        # log. _execute_run copies the v1 model file next to the log.
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
    # libvmaf 2.0 and later run on one thread unless told otherwise:
    # leaving n_threads out does not mean "use all cores". So "Auto"
    # (n_threads <= 0) is resolved to the actual core count here. A job that
    # shares the PC with others already has its share filled in by the
    # scheduler (JobScheduler._share_cores).
    resolved_threads = options.n_threads if options.n_threads > 0 else auto_threads()
    opts.append(f"n_threads={resolved_threads}")
    if options.n_subsample > 1:
        opts.append(f"n_subsample={options.n_subsample}")
    if options.extra_features:
        opts.append("feature=" + "|".join(options.extra_features))
    opts += framesync_opts()
    return opts


def _uses_vmaf_model(options: VmafOptions) -> bool:
    """Whether the run calculates any VMAF model (VMAF, NEG or v1)."""
    return options.compute_vmaf or options.compute_vmaf_neg or options.compute_vmaf_v1


def _v1_model_file(options: VmafOptions) -> Path | None:
    """The VMAF v1 model file a run uses, once the run has resolved it."""
    if options.compute_vmaf_v1 and options.model_v1.startswith("path="):
        return Path(options.model_v1.removeprefix("path="))
    return None


#: The overlay filter's name for each analysis format it keeps unchanged. It
#: has none for 12-bit, so 12-bit comparisons are scored on the CPU
#: (vmaf_gpu.scores_on_gpu).
_OVERLAY_FORMAT = {"yuv420p": "yuv420", "yuv420p10le": "yuv420p10"}


def _gpu_pairs_stage(analysis_format: str, width: int, height: int, main_label: str, ref_label: str) -> str:
    """The filters that give VMAF on the GPU the same frame pairs FFmpeg's
    libvmaf filter compares (frame_sync), as the raw outputs [vmaf_dist] and
    [vmaf_ref], when FFmpeg pairs them (see vmaf_gpu.pairs_in_app).

    The overlay filter pairs the two streams as libvmaf does
    (frame_sync.framesync_opts), placing them side by side in one frame,
    and the frame is cut apart again, every pixel unchanged.

    This works only for an even width and height (vmaf_gpu.scores_on_gpu
    sends an odd comparison to the CPU): pad gives a 4:2:0 picture an even
    size and blacks out an odd picture's last column and row, and crop cuts
    only on whole chroma samples."""
    if width & 1 or height & 1:
        raise ValueError(f"the GPU's frame pairs need an even size, not {width}x{height}")
    sync = ":".join(framesync_opts())
    return (f"[{main_label}]pad={2 * width}:{height}[vmaf_canvas];"
            f"[vmaf_canvas][{ref_label}]overlay=x={width}:y=0:eval=init:"
            f"format={_OVERLAY_FORMAT[analysis_format]}:{sync},split=2[vmaf_left][vmaf_right];"
            f"[vmaf_left]crop={width}:{height}:0:0[vmaf_dist];"
            f"[vmaf_right]crop={width}:{height}:{width}:0[vmaf_ref]")


def analysis_bit_depth(source_info: VideoInfo, distorted_info: VideoInfo) -> int:
    """The bit depth two videos are compared at (see analysis_pix_fmt)."""
    return _bit_depth(analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt))


#: Separates the filter graphs of a run that has more than one
#: (_build_filtergraph with _split_graphs). Each becomes a -filter_complex of
#: its own, ending in [graph0], [graph1].
_GRAPH_SEPARATOR = "\n"


def _split_graphs(options: VmafOptions, xpsnr_log_path: Path | None) -> bool:
    """Whether XPSNR and libvmaf's metrics (VMAF, PSNR, SSIM...) each get a
    filter graph of their own.

    FFmpeg runs a filter graph on one thread. Chained in one graph, XPSNR's
    work and the libvmaf filter's (which allocates, zeroes and copies both
    pictures of every pair before its own threads see them) take turns: on
    4K, PSNR + SSIM + XPSNR ran at 31 fps, where PSNR + SSIM alone ran at 46
    and XPSNR alone at 62, with most cores idle. Each in a graph, on a
    thread of its own, the three run at 44 fps, the speed of PSNR + SSIM."""
    return bool(options.compute_xpsnr and xpsnr_log_path is not None
                and (_uses_vmaf_model(options) or options.extra_features))


def _xpsnr_filter(source_label: str, test_label: str, xpsnr_log_path: Path) -> str:
    """FFmpeg's xpsnr filter on the two videos, with the source first.

    The filter weights each block's error by the activity of its first
    input, which must be the original, as XPSNR is defined. It takes the
    frame rate that decides its temporal activity from its second input (see
    _xpsnr_frame_rate), and passes its first input on. With the test video
    first, the weights would come from the encode, and XPSNR would read
    0.9-1.8 dB too low."""
    return (f"[{source_label}][{test_label}]xpsnr=stats_file={xpsnr_log_path.name}:"
            + ":".join(framesync_opts()))


def _build_libvmaf_stage(
    options: VmafOptions, log_path: Path, model: str | None, xpsnr_log_path: Path | None,
    main_label: str = "main", ref_label: str = "ref", output_label: str = "",
) -> str:
    """The end of the filter graph that both graph builders share: XPSNR
    and libvmaf.

    XPSNR is not a libvmaf feature like PSNR and SSIM, but a separate FFmpeg
    filter with its own stats file. When it is requested, it sits between
    the decoded videos and libvmaf, passing [ref] on under a new label.
    `output_label` names the graph's output (for graphs that map it
    explicitly).
    """
    if not options.requested_metrics():
        raise VmafRunError("Select at least one metric to calculate.")
    output = f"[{output_label}]" if output_label else ""
    # XPSNR alone needs no libvmaf filter or model.
    if not _uses_vmaf_model(options) and not options.extra_features:
        assert xpsnr_log_path is not None
        return _xpsnr_filter(ref_label, main_label, xpsnr_log_path) + output
    libvmaf_opts = _build_libvmaf_opts(options, log_path, model)
    chains = []
    if options.compute_xpsnr and xpsnr_log_path is not None:
        # xpsnr consumes [main], and libvmaf needs it too, but a filter graph
        # label can be consumed only once. Without this explicit split,
        # FFmpeg silently connects libvmaf to the wrong stream, which then
        # compares one video with itself and reports a perfect VMAF 100,
        # PSNR 60 and SSIM 1.0 for every frame, however bad the encode is,
        # without any error. libvmaf takes the source from xpsnr, which
        # passes its first input on.
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
    """The FFmpeg filter graph of a comparison of two videos.

    Each video is downloaded from the GPU decoder if needed, cropped,
    converted to the analysis format and scaled to the compared size. Then
    either libvmaf and xpsnr score the pair, or the frames go out to be
    scored on the GPU.

    With `gpu_vmaf`, VMAF and NEG are scored on the GPU (vmaf_gpu) from the
    compared frames, given as two raw outputs, [vmaf_dist] and [vmaf_ref],
    and FFmpeg's own filters score nothing. With `gpu_paired`, FFmpeg pairs
    the frames (_gpu_pairs_stage). Otherwise the GPU side pairs them
    (vmaf_gpu._StreamReader, frame_sync), and two graphs are returned, one
    per line: the test video's and the source's, each for an FFmpeg of its
    own (_build_stream_cmds), giving the video's frames ([vmaf_dist] or
    [vmaf_ref]). When pairing by timestamp, each graph also gives a copy of
    the frames, which costs nothing, for FFmpeg to list their timestamps
    ([.._ts]).

    Both videos' frames are numbered (frame_sync.pairing_filters), so that
    libvmaf, xpsnr and overlay pair frame N with frame N. When pairing by
    timestamp, each video's times start at 0 instead."""
    dist_content_w, dist_content_h = content_size(distorted_info, distorted_crop)
    ref_content_w, ref_content_h = content_size(source_info, source_crop)
    resolutions_differ = (ref_content_w, ref_content_h) != (dist_content_w, dist_content_h)
    upscale_distorted = resolutions_differ and options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE

    # Both videos must reach libvmaf in the same pixel format, chosen from
    # the deeper of the two (see analysis_pix_fmt for why it is not simply
    # yuv420p).
    analysis_format = analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt)

    # With XPSNR beside libvmaf's metrics, each gets a filter graph of its
    # own, which FFmpeg runs on a thread of its own (_split_graphs).
    split = not gpu_vmaf and _split_graphs(options, xpsnr_log_path)

    # Frame N of each video stamped N frame times of the test video (or,
    # pairing by timestamp, each video's times started at 0).
    numbering = pairing_filters(frame_rate(distorted_info))

    # --- distorted (main, input 0) chain ---
    main_ops = []
    if hwaccel.distorted and not (split and _decoder_downloads(hwaccel.distorted)):
        # As in the source's chain below: frames arrive as hardware surfaces
        # and must come back to system memory before any filter that is not
        # hardware-aware (the crop included) can use them.
        main_ops.append("hwdownload")
        main_ops.append(f"format={_hw_native_format(distorted_info.pix_fmt)}")
    if distorted_crop and not distorted_crop.is_noop(distorted_info.width, distorted_info.height):
        main_ops.append(distorted_crop.as_filter())
    main_ops.append(f"format={analysis_format}")
    if upscale_distorted:
        # Scale the test video up to the source's resolution, instead of the
        # default of scaling the source down to the test video's resolution
        # (see ScaleDirection).
        main_ops.append(f"scale={ref_content_w}:{ref_content_h}:flags={options.scale_algorithm}")
    main_ops += numbering
    main_chain = f"[0:{VIDEO_STREAM}]{','.join(main_ops)}[main]"

    # --- source / reference (input 1) chain ---
    ref_ops = []
    if hwaccel.source and not (split and _decoder_downloads(hwaccel.source)):
        # hwdownload can only output the hardware surface's own format (nv12
        # for 8-bit CUDA decoding, p010le for 10-bit, common in UHD and HDR
        # masters), not the analysis format, so a separate format filter
        # converts it afterwards.
        ref_ops.append("hwdownload")
        ref_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    # Crop before the format conversion, as the test video's chain does (and
    # Vship's, the CPU tools' and the GPU decoders'). Converted first, a
    # 4:2:2 or 4:4:4 source's chroma at the crop's edges would be filtered
    # with samples from the black bars that are cut off.
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        ref_ops.append(source_crop.as_filter())
    ref_ops.append(f"format={analysis_format}")

    if resolutions_differ and not upscale_distorted:
        ref_ops.append(f"scale={dist_content_w}:{dist_content_h}:flags={options.scale_algorithm}")

    ref_ops += numbering
    if gpu_vmaf and not gpu_paired:
        if by_timestamp():
            return (f"[0:{VIDEO_STREAM}]{','.join([*main_ops, 'split=2[vmaf_dist][vmaf_dist_ts]'])}\n"
                    f"[0:{VIDEO_STREAM}]{','.join([*ref_ops, 'split=2[vmaf_ref][vmaf_ref_ts]'])}")
        return (f"[0:{VIDEO_STREAM}]{','.join(main_ops)}[vmaf_dist]\n"
                f"[0:{VIDEO_STREAM}]{','.join(ref_ops)}[vmaf_ref]")
    ref_chain = f"[1:{VIDEO_STREAM}]{','.join(ref_ops)}[ref]"

    compared_w, compared_h = (
        (ref_content_w, ref_content_h) if upscale_distorted else (dist_content_w, dist_content_h))
    if split:
        # The graphs do not decode anything themselves: FFmpeg gives every
        # graph that reads an input the same decoded frames (with CUDA or
        # D3D11VA, the decoder downloads them once, see _split_hwaccel_args).
        # Each graph converts and crops them on its own thread.
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
    """The filter graph of a resolution round-trip test, which has a single
    input: the source is decoded once and split into an untouched reference
    branch and a "test" branch, which is scaled down to the target width
    (keeping the source's aspect ratio) and back up to the source's own
    resolution. There is no second file: both branches come from the source.
    """
    target = options.resample_test
    assert target is not None

    orig_w, orig_h = source_info.width, source_info.height
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        orig_w, orig_h = source_crop.w, source_crop.h

    down_w = target.width
    down_h = max(2, round(down_w * orig_h / orig_w / 2) * 2)  # even, preserves the source's own aspect ratio

    # A round-trip test has one input, so the analysis format comes from the
    # source alone (see analysis_pix_fmt).
    analysis_format = analysis_pix_fmt(source_info.pix_fmt)

    base_ops = []
    if hwaccel_used:
        # As in _build_filtergraph: hwdownload can only output the hardware
        # surface's own format, not the analysis format.
        base_ops.append("hwdownload")
        base_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        base_ops.append(source_crop.as_filter())  # before the conversion, as _build_filtergraph
    base_ops.append(f"format={analysis_format}")
    base_chain = f"[0:{VIDEO_STREAM}]{','.join(base_ops)}[base]"

    split_chain = "[base]split=2[ref_src][dist_src]"
    numbering = ",".join(pairing_filters(frame_rate(source_info)))  # as _build_filtergraph's
    ref_chain = f"[ref_src]{numbering}[ref]"
    dist_chain = (
        f"[dist_src]scale={down_w}:{down_h}:flags={options.scale_algorithm},"
        f"scale={orig_w}:{orig_h}:flags={options.scale_algorithm},{numbering}[main]"
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
    """The FFmpeg command of a comparison of two videos."""
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    # The -i paths are plain arguments, not filter graph syntax, so absolute
    # Windows paths work here. They must be made absolute, because FFmpeg
    # runs in a temporary folder (see _build_libvmaf_opts).
    decode = _split_hwaccel_args if _GRAPH_SEPARATOR in filtergraph else _hwaccel_args
    cmd += decode(hwaccel.distorted)
    cmd += ["-i", str(Path(distorted_path).resolve())]
    cmd += decode(hwaccel.source)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit, gpu_outputs)
    return cmd


#: The -hwaccel decoders that hand frames over in system memory when no
#: -hwaccel_output_format is given. QSV's still hands over its own surfaces
#: ("Impossible to convert between the formats ... src: qsv").
_DECODERS_THAT_DOWNLOAD = frozenset({"cuda", "d3d11va"})


def _decoder_downloads(hwaccel: str | None) -> bool:
    """Whether the decoder hands frames over in system memory by itself."""
    return hwaccel in _DECODERS_THAT_DOWNLOAD


def _split_hwaccel_args(hwaccel: str | None) -> list[str]:
    """The -hwaccel options for one input of a run with several filter graphs.

    Where the decoder can (see _decoder_downloads), it downloads each picture
    to system memory itself, once, and every graph gets that copy. If the
    pictures stayed on the GPU, each graph would download them again: PSNR +
    SSIM + XPSNR on 4K ran at 38 fps that way, and at 44 this way. QSV's
    frames stay on the GPU, and each graph downloads them
    (_build_filtergraph)."""
    return ["-hwaccel", hwaccel] if _decoder_downloads(hwaccel) else _hwaccel_args(hwaccel)


def _build_stream_cmds(
    distorted_path: Path, source_path: Path, graphs: str, hwaccel: HwAccelPlan,
    outputs: tuple[list[str], list[str]],
) -> list[list[str]]:
    """One FFmpeg command per video (the test video's first), for VMAF on the
    GPU when the app pairs the frames itself. `graphs` are as
    _build_filtergraph gives them for that, and `outputs` as
    vmaf_gpu.GpuAttempt.output_args gives them."""
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
    """Runs the two FFmpegs of _build_stream_cmds side by side (as
    _run_ffmpeg runs one). The first reports the progress, and one that
    fails ends the other. Returns the failed one's result, or the first's."""
    stop = threading.Event()
    results: list[subprocess.CompletedProcess | None] = [None] * len(commands)
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            results[index] = _run_ffmpeg(commands[index], total_frames, on_progress if index == 0 else None, stop,
                                         cwd=cwd, process_handle=process_handle)
            if results[index].returncode != 0:
                stop.set()
        except BaseException as error:  # raised below, on the caller's thread
            errors.append(error)
            stop.set()

    threads = [threading.Thread(target=run, args=(index,), name=f"ffmpeg-{index}", daemon=True)
               for index in range(len(commands))]
    for thread in threads:
        thread.start()
    for thread in threads:  # one at a time: joining a finished thread returns at once
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
        if not isinstance(error, Cancelled):  # a Cancelled one was ended because the other failed
            raise error
    if errors:
        raise errors[0]
    return results[0]


def _build_resample_cmd(
    source_path: Path, filtergraph: str, hwaccel: str | None, duration_limit: float = 0.0,
) -> list[str]:
    """The FFmpeg command of a resolution round-trip test."""
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    cmd += _hwaccel_args(hwaccel)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit)
    return cmd


def _build_ffmpeg_output_args(
    filtergraph: str, duration_limit: float, gpu_outputs: list[str] | None = None,
) -> list[str]:
    """The FFmpeg options after the inputs: the filter graph and the outputs.

    `gpu_outputs` are GPU VMAF's raw outputs (vmaf_gpu.GpuAttempt), each
    with its own -t; they are then the run's only outputs. Several filter
    graphs (separated by _GRAPH_SEPARATOR) get a -filter_complex each, and a
    null output each for their [graphN], each with the -t.

    A -t on the output side (frame_sync.duration_args) limits how much of
    the filtered output is produced, and so how many frames reach libvmaf,
    whatever the two inputs' lengths."""
    cut = duration_args(f"{duration_limit:.3f}" if duration_limit > 0 else None)
    graphs = filtergraph.split(_GRAPH_SEPARATOR)
    if len(graphs) > 1 and gpu_outputs is None:
        args = [part for graph in graphs for part in ("-filter_complex", graph)] + ["-progress", "pipe:1", "-nostats"]
        for index in range(len(graphs)):
            args += ["-map", f"[graph{index}]", *cut, "-f", "null", "-"]
        return args
    args = ["-lavfi", filtergraph, "-progress", "pipe:1", "-nostats"]
    if gpu_outputs is not None:
        return args + gpu_outputs
    return [*args, *cut, "-f", "null", "-"]


def _cancelled(cancel_event: threading.Event | None, process_handle: ProcessHandle | None) -> bool:
    """Whether the run was cancelled: its event is set, or Cancel ended its
    processes (ProcessHandle.terminate). The second matters for XPSNR's
    FFmpeg beside the app's metrics, which waits on an event of its own that
    is set only once the app's scorer has stopped: without it, an FFmpeg
    ended by Cancel would be taken for a failure, and the fallback would
    start its next attempt."""
    return ((cancel_event is not None and cancel_event.is_set())
            or (process_handle is not None and process_handle.was_terminated))


def _run_ffmpeg(
    cmd: list[str], total_frames: int,
    on_progress: ProgressCallback | None, cancel_event: threading.Event | None,
    cwd: Path, process_handle: ProcessHandle | None = None,
) -> subprocess.CompletedProcess:
    """Runs one FFmpeg command, reporting its progress, and returns its
    result. Raises Cancelled when the run is cancelled."""
    # Checked before starting FFmpeg, not only while reading its output:
    # otherwise a Cancel during crop detection or between fallback attempts
    # would start one more FFmpeg, which would then have to be found and
    # killed.
    if _cancelled(cancel_event, process_handle):
        raise Cancelled("Cancelled by user")
    # Decoded as UTF-8, not the Windows code page: FFmpeg's error output
    # starts with the inputs' paths and tags, and a curly quote (”) in either
    # is a byte cp1252 cannot decode. That would kill the thread that drains
    # the output, losing FFmpeg's error message and leaving the pipe full.
    proc = proc_util.popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=str(cwd),
    )
    if process_handle is not None:
        process_handle.attach(proc.pid)
    # Set before the try, so the finally block can always read it, even if
    # the thread was never created.
    stderr_thread: threading.Thread | None = None
    try:
        stderr_lines: list[str] = []

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # FFmpeg's -progress output writes each update as several key=value
        # lines (frame=, fps=, ..., progress=continue or end), not one line.
        # The latest fps seen is kept and reported with every frame= line.
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
        # Checked again here, not only inside the loop: a paused process
        # writes no more output for the loop to see, and Cancel kills it
        # from outside (ProcessHandle.terminate), which ends the loop at the
        # end of the output instead of through the loop's own check.
        if _cancelled(cancel_event, process_handle):
            raise Cancelled("Cancelled by user")
        return subprocess.CompletedProcess(cmd, proc.returncode, "", "".join(stderr_lines))
    finally:
        # Anything can leave the block above early (a cancellation, or an
        # on_progress callback raising inside the loop), and an FFmpeg left
        # running keeps its pipes and the log files in the run's temporary
        # folder open. On Windows the folder then cannot be deleted, and
        # files pile up in %TEMP% for the rest of the session.
        _reap(proc, stderr_thread)
        if process_handle is not None:
            process_handle.detach(proc.pid)


def _reap(proc: subprocess.Popen, drain_thread: threading.Thread | None) -> None:
    """Ends `proc` if it is still running, then waits for its output reader
    and closes its pipes, in that order, so that the reader sees the end of
    the output rather than having the pipe closed under it.

    It swallows its own errors on purpose: it runs in a finally block, and
    the exception that led there (a Cancelled, or whatever a progress
    callback raised) is the one the caller needs to see. A second failure
    while tidying up must not replace it.
    """
    with contextlib.suppress(Exception):  # see the docstring
        if proc.poll() is None:
            proc_util.terminate(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate() is a request that a stuck decoder can ignore;
                # kill() cannot be refused.
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
    """The number of frames FFmpeg will process, which sizes the progress
    bar and the estimate of the queued work.

    `reference_info` is the video that drives the output timeline (the test
    video for a normal run, the source for a resolution round-trip test).
    The count is bounded by the duration limit. libvmaf's n_subsample
    reduces how many frames get a score, but FFmpeg's progress counter still
    counts every frame, so n_subsample is not applied here (applied, it made
    progress pass 100%).

    `other_info` is the second input of a two-input comparison. The graph
    stops when either input ends (see frame_sync.FRAMESYNC_OPTS), so a test
    video longer than its source gives fewer frames than its own length
    suggests; without this, the progress bar would stop short of 100%.
    """
    frame_count = reference_info.estimated_frame_count
    if other_info is not None:
        frame_count = min(frame_count, other_info.estimated_frame_count)
    if options.duration_limit > 0:
        frame_count = min(frame_count, round(options.duration_limit * reference_info.fps))
    return frame_count


def _resolve_model_for_cwd(model: str, tmpdir: Path) -> str:
    """When `model` names a custom model file ("path=<file>"), copies the
    file into `tmpdir` and returns the option naming it by its bare file
    name, for the reason given in _build_libvmaf_opts.
    """
    if not model.startswith("path="):
        return model
    src = Path(model[len("path="):])
    dest = tmpdir / src.name
    shutil.copyfile(src, dest)
    return f"path={dest.name}"


def _parse_log(log_path: Path, fps: float, xpsnr_log_path: Path | None = None) -> FrameScores:
    """Reads the per-frame scores from libvmaf's JSON log, and XPSNR's from
    its stats file."""
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)

    xpsnr_by_frame = _parse_xpsnr_log(xpsnr_log_path) if xpsnr_log_path is not None else {}

    # The scores are gathered in plain lists and packed into arrays at the
    # end, rather than as one object per frame: a feature-length run has
    # hundreds of thousands of frames.
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
        # Checked with "is None", not "or": libvmaf reports a real 0.0 for
        # badly degraded frames, and "or" would discard it and fall through
        # to the other key (or to None).
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
        """A metric's scores as an array (NaN where a frame has none), or
        None for a metric the run did not calculate."""
        if all(v is None for v in values):
            return None  # the run did not ask for this metric
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
    """Reads XPSNR's luma score per frame from the xpsnr filter's stats
    file, keyed by frame number counted from 0. The stats file counts frames
    from 1, while the rest of the app counts from 0 (as libvmaf's frameNum
    does), so the numbers are converted here.
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


#: Builds the FFmpeg command of a run from (decode plan, model relative to the
#: run's temporary folder, libvmaf log path, xpsnr log path or None). The two
#: kinds of run (a comparison and a resolution round-trip test) differ only
#: in this, so _execute_run takes it as a parameter.
CommandBuilder = Callable[..., list[str]]


@dataclass(frozen=True)
class _GpuPlan:
    """The metrics a run scores outside FFmpeg's filters, and how.

    Usually VMAF and NEG on the GPU (vmaf_gpu), as the run's only metrics:
    `models` holds libvmaf's models for them, and `width`, `height` and
    `bit_depth` the size and depth frames are compared at. With `backend`
    "cpu", it is PSNR, SSIM and XPSNR by the bundled libvmaf's CPU
    extractors (libvmaf_cpu.CpuScorer): `models` then maps their keys to
    themselves, and `threads` is libvmaf's thread count."""
    models: dict[str, str]
    width: int
    height: int
    bit_depth: int
    #: "cuda" (libvmaf), "vulkan" (vmaf_vulkan, on Vulkan's GPU `device`)
    #: or "cpu".
    backend: str = "cuda"
    device: int | None = None
    threads: int = 0
    #: XPSNR's frame rate, with backend "cpu" (see _xpsnr_frame_rate).
    frame_rate: int = 0


def _gpu_frame_scores(gpu_scores, fps: float) -> FrameScores:
    """The scores made outside FFmpeg (VMAF and NEG on the GPU, or the CPU
    metrics) as the run's frame scores. Raises LibvmafError when no frame
    was scored."""
    numbers, scores = gpu_scores
    if not len(numbers):
        raise libvmaf.LibvmafError("GPU VMAF scored no frames")
    time = numbers / fps if fps > 0 else np.zeros(len(numbers), dtype=np.float64)
    v1 = scores.get("vmaf_v1")

    def column(key: str):  # float32, as _parse_log keeps FFmpeg's scores
        return None if scores.get(key) is None else np.asarray(scores[key], dtype=np.float32)

    return FrameScores(numbers, time, vmaf=scores.get("vmaf"), vmaf_neg=scores.get("vmaf_neg"),
                       psnr=column("psnr"), ssim=column("ssim"), xpsnr=column("xpsnr"),
                       metrics=None if v1 is None else {"vmaf_v1": np.asarray(v1, dtype=np.float32)})


def _fallback_ladder(plan: HwAccelPlan) -> list[HwAccelPlan]:
    """The decode plans to try, in order, until one of them runs.

    Hardware decoding can fail for reasons no capability table predicts: a
    profile the decoder hardware does not implement, a driver that reports
    the codec but rejects the particular stream, or no decode sessions
    left. The test video loses hardware decoding first, because it is the
    unknown one: the source is usually a known-good master, while the test
    video is whatever encoder settings are being tried.
    """
    ladder = [plan]
    if plan.source is not None and plan.distorted is not None:
        # The first retry keeps the (usually known-good) source on the GPU.
        # If the source is the one failing, the next retry keeps the test
        # video on the GPU instead.
        ladder.append(HwAccelPlan(source=plan.source))
        ladder.append(HwAccelPlan(distorted=plan.distorted))
    if plan.uses_gpu:
        ladder.append(HwAccelPlan())
    return ladder


def _with_v1_model(options: VmafOptions, dimensions: tuple[int, int]) -> VmafOptions:
    """The options with VMAF v1's model resolved. Auto picks it from the size
    frames are compared at, which is known only after black bars are
    detected, as for VMAF v0.6.1's Auto (_auto_model_or)."""
    if not options.compute_vmaf_v1:
        return options
    return replace(options, model_v1=resolve_v1_model(options, *dimensions))


def _auto_model_or(options: VmafOptions, dimensions: tuple[int, int]) -> str:
    """The VMAF model to run with. Only Auto is decided here, from the
    compared size; an explicit or custom choice is the user's and is kept."""
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
    """Runs a run's FFmpeg command to completion and parses its logs, falling
    back to other decode plans when hardware decoding fails.

    With `gpu`, the run's metrics (VMAF and NEG, or the CPU metrics) are
    scored outside FFmpeg from its raw outputs (vmaf_gpu.GpuAttempt, one
    per attempt), and build_command then also takes those outputs' options.

    Both run_vmaf and run_resample_test use it, so the temporary folder, the
    decode fallback, the exit code and log checks and the log parsing are
    written once.
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
            shutil.copyfile(v1_file, tmpdir / v1_file.name)  # named by its bare file name, as above

        def run_with(plan: HwAccelPlan):
            """One attempt with decode plan `plan`; returns FFmpeg's result."""
            nonlocal gpu_scores
            if gpu is None:
                cmd = build_command(plan, resolved_model, log_path, xpsnr_log_path)
                _log.info("FFmpeg: %s", _command_text(cmd))
                return _run_ffmpeg(
                    cmd, total_frames, on_progress, cancel_event,
                    cwd=tmpdir, process_handle=process_handle,
                )
            # Each attempt has its own libvmaf context and pipes: a failed
            # attempt's cannot be reused.
            attempt = vmaf_gpu.GpuAttempt(gpu.width, gpu.height, gpu.bit_depth, gpu.models, options.n_subsample,
                                           gpu.backend, gpu.device, threads=gpu.threads, frame_rate=gpu.frame_rate)
            try:
                # One frame more than the limit: FFmpeg's libvmaf filter also
                # scores the first frame at or past the limit (stamped 30.03 s
                # for a 30 s limit at 23.976 fps) before the null output stops
                # there, and the raw outputs must carry the same frames.
                limit = options.duration_limit + 1 / fps if options.duration_limit > 0 and fps > 0 else 0.0
                cmd = build_command(plan, resolved_model, log_path, xpsnr_log_path, attempt.output_args(limit))
                if isinstance(cmd[0], list):  # one FFmpeg per video
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
            # This raises when the scoring failed: FFmpeg's failure is then
            # caused by it, and no decode retry would help.
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
            # Delete the failed attempt's logs: FFmpeg can write a partial log
            # before the decoder gives up, and it would otherwise be parsed as
            # if the retry had written it.
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
    """Compares two videos and returns the result, with every requested
    metric's per-frame scores. Raises VmafRunError when the run fails, and
    Cancelled when it is cancelled.

    `result_distorted_path` replaces the result's `distorted` path (by
    default distorted_info.path). It does not change which file is decoded,
    only the identity the result carries for the cache and the graphs, so a
    caller comparing the same file twice with different options (both scale
    directions, say) can keep the two results apart, as a resolution
    round-trip test's made-up path does.
    """
    validate_video_pair(source_info, distorted_info, options)

    # The decode plan is made before black-bar detection, so the detection
    # decodes the way the run will. The plan depends only on the codecs and
    # the GPU maker, never on the crops.
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
    # Checked after cropping, not before: removing a letterbox is exactly
    # what makes a letterboxed source and an already-cropped encode the same
    # shape.
    validate_display_geometry(source_info, distorted_info, source_crop, distorted_crop)

    # Auto picks its model from the size frames are compared at, which is
    # known only now: it depends on the scale direction and on the crops just
    # detected, not on either video's own resolution.
    dimensions = analysis_dimensions(source_info, distorted_info, options, source_crop, distorted_crop)
    effective_model = _auto_model_or(options, dimensions)
    options = _with_v1_model(options, dimensions)

    def build_command(plan, model, log_path, xpsnr_log_path):
        """The FFmpeg command of this comparison (a CommandBuilder)."""
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, plan, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_ffmpeg_cmd(
            distorted_info.path, source_info.path, filtergraph, plan, options.duration_limit,
        )

    total_frames = estimate_total_frames(distorted_info, options, source_info)
    frames = None
    # VMAF is scored on the GPU only when VMAF, NEG and VMAF v1 are all the
    # run scores: the window's runs put them in a run of their own, apart
    # from FFmpeg's other metrics (job_runner.GPU_VMAF).
    gpu_models = None
    gpu_backend = "cuda"
    requested = set(options.requested_metrics())
    if not requested - {"vmaf", "vmaf_neg", "vmaf_v1"}:
        gpu_models = vmaf_gpu.scores_on_gpu(options.compute_vmaf, options.compute_vmaf_neg, effective_model,
                                             options.vmaf_on_gpu, analysis_bit_depth(source_info, distorted_info),
                                             size=dimensions, compute_vmaf_v1=options.compute_vmaf_v1,
                                             model_v1=options.model_v1)
        if gpu_models is not None and set(gpu_models) != requested:
            gpu_models = None  # one must be calculated on the CPU: FFmpeg's libvmaf calculates them all together
    if gpu_models is not None:
        plan = _GpuPlan(gpu_models, *dimensions, analysis_bit_depth(source_info, distorted_info),
                        *vmaf_gpu.gpu_vmaf_backend())
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


#: The start of the status a run sends when VMAF on the GPU fails, after
#: which FFmpeg's libvmaf calculates it.
VMAF_GPU_FAILED = "VMAF on the GPU failed"


def _run_on_gpu(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, model: str, hwaccel: HwAccelPlan,
    total_frames: int, *, on_progress, on_status, cancel_event, process_handle,
) -> FrameScores | None:
    """Runs the comparison with VMAF and NEG on the GPU (vmaf_gpu), in a
    process of its own, so a crash in libvmaf or the NVIDIA driver ends that
    process and not the app. Returns None when it fails for any reason other
    than Cancel; the run is then made again with VMAF on the CPU, as on a PC
    without a suitable GPU.

    Only one GPU pass runs at a time, shared with Vship's (gpu.GPU_PASS), so
    VMAF on the GPU never runs beside another video's GPU metrics."""
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
    """The GPU run, which _run_on_gpu calls in a process of its own: FFmpeg
    decodes and pairs the frames as it would for the CPU, and they are fed
    to libvmaf on the GPU.

    When this process has a decoder for each video (_decoders_here), the
    videos are decoded here instead (vmaf_gpu.score_decoded,
    vmaf_vulkan.score_decoded). The frames are the same, without FFmpeg's
    decoding and the CPU copies and pipes behind it. Each video is decoded
    by the decoder FFmpeg would have used (NVIDIA's, Intel's or AMD's), or
    by the software decoder for a video FFmpeg would decode in software.
    libvmaf's CUDA code takes NVIDIA's pictures on the GPU and uploads the
    software decoder's; the Vulkan scorer uploads any but NVIDIA's. If that
    decoding fails after it has started, the run is made again with FFmpeg
    decoding."""
    decoders = _decoders_here(hwaccel)
    if decoders is not None and ("vmaf_v1" in plan.models or plan.backend == "vulkan"
                                 or set(decoders) <= {"nvidia", gpu_frames.SOFTWARE}):
        try:
            return _score_decoded_on_gpu(
                plan, source_info, distorted_info, options, source_crop, distorted_crop, hwaccel, total_frames,
                on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
                process_handle=process_handle)
        except gpu_frames.GpuDecodeUnavailableError as error:
            _log.info("VMAF on the GPU: the videos are decoded by FFmpeg (%s)", error)
        except gpu_frames.GpuDecodeFailedError as error:
            _log.warning("Decoding for VMAF on the GPU failed; decoding through FFmpeg instead: %s", error)
            if on_status:
                on_status(gpu_frames.decoding_failed_status(error))

    def build_command(hw, resolved_model, log_path, xpsnr_log_path, gpu_outputs):
        """The FFmpeg command (or one per video) of the GPU run."""
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, hw, log_path,
            model=resolved_model, xpsnr_log_path=xpsnr_log_path, gpu_vmaf=True,
            gpu_paired=not isinstance(gpu_outputs, tuple),
        )
        if isinstance(gpu_outputs, tuple):  # one FFmpeg per video: the GPU side pairs the frames
            return _build_stream_cmds(distorted_info.path, source_info.path, filtergraph, hw, gpu_outputs)
        return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                 options.duration_limit, gpu_outputs)

    return _execute_run(
        build_command, options=options, model=model, fps=distorted_info.fps, total_frames=total_frames,
        hwaccel=hwaccel, tmp_prefix="vmaf_gpu_run_", on_progress=on_progress, on_status=on_status,
        cancel_event=cancel_event, process_handle=process_handle, gpu=plan,
    )


#: The environment variable that, set to "ffmpeg", leaves PSNR and SSIM to
#: FFmpeg's libvmaf filter instead of _run_cpu_metrics, to compare the two.
CPU_METRICS_VARIABLE = "VML_CPU_METRICS"
#: FFmpeg's libvmaf features, mapped to the metrics libvmaf_cpu.CpuScorer scores.
_CPU_METRIC_OF_FEATURE = {"name=psnr": "psnr", "name=float_ssim": "ssim"}


_CPU_METRIC_NAMES = {"psnr": "PSNR", "ssim": "SSIM", "xpsnr": "XPSNR"}


def _cpu_metric_names(metrics) -> str:
    """The metrics' names as a phrase ("PSNR, SSIM and XPSNR"), for the
    statuses and the log."""
    names = [_CPU_METRIC_NAMES.get(metric, metric) for metric in metrics]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _xpsnr_frame_rate(info: VideoInfo) -> int | None:
    """XPSNR's frame rate, in whole frames per second, as FFmpeg's xpsnr
    filter takes it from its second input (the test video, see
    _xpsnr_filter). Below 32 fps XPSNR's temporal activity is first-order,
    otherwise second-order.

    FFmpeg gives the input the rate av_guess_frame_rate() returns
    (fftools/ffmpeg_demux.c): r_frame_rate when the average rate is known
    and within a tenth of it. Otherwise that rate may be the codec's own
    (H.264's, for a codec with fields) or the average, which cannot be
    predicted here: then this returns None, and FFmpeg's filter scores
    XPSNR."""
    average, nominal = info.average_fps, info.nominal_fps
    if average <= 0 or nominal <= 0 or abs(1.0 - average / nominal) >= 0.09:
        return None
    return int(nominal)


def _xpsnr_in_app(width: int, height: int, bit_depth: int) -> bool:
    """Whether libvmaf-fast's port of FFmpeg's xpsnr filter gives exactly
    FFmpeg's score for this comparison: up to 12 bits, and an even size above
    2048x1152 (FFmpeg's filter reads beyond the picture of an odd one)."""
    if bit_depth > 12 or (width * height > 2048 * 1152 and (width & 1 or height & 1)):
        return False
    return libvmaf_cpu.cpu_scores_xpsnr()


def _decoded_in_app(source_info: VideoInfo, distorted_info: VideoInfo, source_crop: CropBox | None,
                    distorted_crop: CropBox | None, size: tuple[int, int], hwaccel: HwAccelPlan) -> bool:
    """Whether the scoring process will decode both videos itself
    (libvmaf_cpu.score_decoded_cpu) rather than taking FFmpeg's frames through
    pipes, as far as can be known before it asks the decoder."""
    decoders = _decoders_here(hwaccel)
    if decoders is None:
        return False
    if any(not gpu_frames.decodes_codec((info.codec_name or "").casefold(), decoder)
           for info, decoder in ((source_info, decoders[0]), (distorted_info, decoders[1]))):
        return False
    try:
        return not any(gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=size).scaled
                       for info, crop in ((source_info, source_crop), (distorted_info, distorted_crop)))
    except gpu_frames.GpuDecodeUnavailableError:
        return False


def cpu_metrics_in_app() -> bool:
    """Whether the app may score PSNR, SSIM and XPSNR itself: the bundled
    libvmaf is there, and VML_CPU_METRICS does not send them to FFmpeg."""
    return os.environ.get(CPU_METRICS_VARIABLE, "").casefold() != "ffmpeg" and libvmaf.LIBRARY_PATH.is_file()


def _cpu_metrics_plan(options: VmafOptions, dimensions: tuple[int, int], source_info: VideoInfo,
                      distorted_info: VideoInfo, hwaccel: HwAccelPlan, source_crop: CropBox | None = None,
                      distorted_crop: CropBox | None = None) -> _GpuPlan | None:
    """The plan for scoring PSNR, SSIM and XPSNR with the bundled libvmaf in
    the app (_run_cpu_metrics) instead of with FFmpeg's libvmaf and xpsnr
    filters, or None to leave them to FFmpeg's filters.

    The app scores them when they are the run's only metrics (VMAF on the
    CPU keeps them in FFmpeg's filters, which it runs anyway), and when the
    frames can come to the app as they do for VMAF on the GPU. XPSNR stays
    with FFmpeg where libvmaf-fast's score would not match FFmpeg's exactly
    (see _xpsnr_frame_rate and _xpsnr_in_app), while PSNR and SSIM are
    still scored in the app."""
    if not cpu_metrics_in_app():
        return None
    if _uses_vmaf_model(options) or options.resample_test is not None:
        return None
    if any(feature not in _CPU_METRIC_OF_FEATURE for feature in options.extra_features):
        return None
    width, height = dimensions
    bit_depth = analysis_bit_depth(source_info, distorted_info)
    # When FFmpeg pairs the frames (VML_VMAF_PIPES=canvas), the size must be
    # even and the format one that overlay keeps unchanged (_gpu_pairs_stage).
    if not vmaf_gpu.pairs_in_app() and (width & 1 or height & 1 or bit_depth > 10):
        return None
    metrics = list(dict.fromkeys(_CPU_METRIC_OF_FEATURE[feature] for feature in options.extra_features))
    frame_rate = _xpsnr_frame_rate(distorted_info) if options.compute_xpsnr else None
    if frame_rate is not None and _xpsnr_in_app(width, height, bit_depth):
        metrics.append("xpsnr")
    elif options.compute_xpsnr and not (hwaccel.source and hwaccel.distorted):
        # XPSNR's FFmpeg beside them would decode the videos again, and a
        # video the CPU decodes would then be decoded twice on the CPU: 70%
        # more CPU per frame for a 4K film against its encode, and slower for
        # a 4K film against a 1080p encode (97 fps against 112). FFmpeg's
        # single run decodes each video once for both libvmaf and XPSNR.
        return None
    if not metrics:
        return None
    if metrics == ["xpsnr"] and not _decoded_in_app(source_info, distorted_info, source_crop, distorted_crop,
                                                     dimensions, hwaccel):
        # XPSNR alone, with its frames from FFmpeg's pipes: FFmpeg's own
        # filter keeps up with FFmpeg's decoding and uses far less. For a 4K
        # film against a 1080p encode with NVIDIA decoding, FFmpeg ran at
        # 128 fps and the app at 159, but with twice the CPU and 825 MB of
        # memory against 453; decoded on the CPU, 77 fps against 81, and
        # 1.8 GB against 2.7.
        return None
    threads = options.n_threads if options.n_threads > 0 else auto_threads()
    return _GpuPlan({metric: metric for metric in metrics}, width, height, bit_depth, backend="cpu",
                    threads=threads, frame_rate=frame_rate or 0)


class _Progress:
    """Reports two runs' progress as one: the progress of whichever is
    behind."""

    def __init__(self, on_progress: ProgressCallback | None, parts: int):
        self._on_progress = on_progress
        self._latest: list[tuple[int, int, float] | None] = [None] * parts
        self._lock = threading.Lock()

    def part(self, index: int) -> ProgressCallback | None:
        """The progress callback for run number `index`."""
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
    """Scores PSNR, SSIM and XPSNR with the bundled libvmaf's CPU extractors,
    in a process of their own (so a crash ends that process, not the app).

    The frames are the videos' frames as they would reach FFmpeg's filters,
    either decoded in that process by the GPU's decoder or written by each
    video's FFmpeg (see _score_cpu_metrics), and are paired as those filters
    pair them (frame_sync), as for VMAF on the GPU. This is faster than
    FFmpeg's libvmaf filter, which allocates, zeroes and copies two new
    pictures for every pair on its single filter thread before libvmaf's
    threads see them.

    XPSNR that the plan leaves to FFmpeg is scored by an FFmpeg of its own
    alongside, and its scores are taken for the frames PSNR and SSIM were
    scored for (as _parse_log takes them). XPSNR alone is scored for every
    frame, as FFmpeg's filter scores it whatever n_subsample is.

    Returns None when the run fails for any reason other than Cancel;
    FFmpeg's filters then calculate the metrics."""
    names = _cpu_metric_names(plan.models)
    _log.info("%s in the app: %s, %d threads", names, libvmaf_cpu.CPU_BUILD, plan.threads)
    beside = options.compute_xpsnr and "xpsnr" not in plan.models
    xpsnr_frames: list[FrameScores] = []
    xpsnr_error: list[BaseException] = []
    progress = _Progress(on_progress, 2 if beside else 1)
    stop = threading.Event()
    xpsnr_thread = None
    if beside:
        xpsnr_options = replace(options, extra_features=[])

        def build_command(hw, resolved_model, log_path, xpsnr_log_path):
            """The FFmpeg command for XPSNR alone."""
            # Only XPSNR: run_vmaf's command would score PSNR and SSIM there
            # again, at FFmpeg's speed.
            filtergraph = _build_filtergraph(
                source_info, distorted_info, xpsnr_options, source_crop, distorted_crop, hw, log_path,
                model=resolved_model, xpsnr_log_path=xpsnr_log_path,
            )
            return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                     xpsnr_options.duration_limit)

        def xpsnr() -> None:
            """XPSNR's FFmpeg run, on a thread of its own."""
            try:
                xpsnr_frames.append(_execute_run(
                    build_command, options=xpsnr_options, model=model, fps=distorted_info.fps,
                    total_frames=total_frames, hwaccel=hwaccel, tmp_prefix="vmaf_xpsnr_run_",
                    on_progress=progress.part(1), on_status=None, cancel_event=stop,
                    process_handle=process_handle))
            except BaseException as error:  # raised on the caller's thread
                xpsnr_error.append(error)

        xpsnr_thread = threading.Thread(target=xpsnr, name="vmaf-xpsnr", daemon=True)
        xpsnr_thread.start()

    def wait_for_xpsnr() -> None:
        """Waits for XPSNR's run, passing on a Cancel."""
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
    """The CPU metrics' run, which _run_cpu_metrics calls in a process of its
    own: FFmpeg decodes, crops, scales and converts the videos as it would
    for its libvmaf filter, and the app pairs and scores them
    (vmaf_gpu.GpuAttempt with a CpuScorer).

    Where this process has a decoder for each video (_decoders_here: the
    GPU's, or the software decoder) and nothing is scaled, the videos are
    decoded here instead (libvmaf_cpu.score_decoded_cpu): the same frames,
    without FFmpeg, its conversions and the pipes."""
    decoders = _decoders_here(hwaccel)
    if decoders is not None:
        try:
            return _score_decoded_on_gpu(
                plan, source_info, distorted_info, options, source_crop, distorted_crop, hwaccel, total_frames,
                on_progress=on_progress, on_status=on_status, cancel_event=cancel_event,
                process_handle=process_handle)
        except gpu_frames.GpuDecodeUnavailableError as error:
            _log.info("%s in the app: the videos are decoded by FFmpeg (%s)", _cpu_metric_names(plan.models), error)
        except gpu_frames.GpuDecodeFailedError as error:
            _log.warning("Decoding for %s failed; decoding through FFmpeg instead: %s",
                         _cpu_metric_names(plan.models), error)
            if on_status:
                on_status(gpu_frames.decoding_failed_status(error))

    def build_command(hw, resolved_model, log_path, xpsnr_log_path, gpu_outputs):
        """The FFmpeg command (or one per video) of the CPU metrics' run."""
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, hw, log_path,
            model=resolved_model, xpsnr_log_path=xpsnr_log_path, gpu_vmaf=True,
            gpu_paired=not isinstance(gpu_outputs, tuple),
        )
        if isinstance(gpu_outputs, tuple):  # one FFmpeg per video: the app pairs the frames
            return _build_stream_cmds(distorted_info.path, source_info.path, filtergraph, hw, gpu_outputs)
        return _build_ffmpeg_cmd(distorted_info.path, source_info.path, filtergraph, hw,
                                 options.duration_limit, gpu_outputs)

    return _execute_run(
        build_command, options=options, model="", fps=distorted_info.fps, total_frames=total_frames,
        hwaccel=hwaccel, tmp_prefix="vmaf_cpu_run_", on_progress=on_progress, on_status=on_status,
        cancel_event=cancel_event, process_handle=process_handle, gpu=plan,
    )


#: The scoring process's own decoder for each FFmpeg -hwaccel (as
#: perceptual_vship's _GPU_DECODERS).
_DECODED_HERE = {"cuda": "nvidia", "qsv": "intel", "d3d11va": "amd"}


def _decoders_here(hwaccel: HwAccelPlan) -> tuple[str, str] | None:
    """The decoders (gpu_frames' libraries) the scoring process would decode
    the source and the test video with itself, or None when FFmpeg must
    decode them.

    Each video is decoded by the decoder of the GPU FFmpeg would use
    (_DECODED_HERE), or by the software decoder where FFmpeg would decode in
    software; the two videos can differ (an HEVC source on NVIDIA's
    decoder, a VVC test video on the CPU). None means FFmpeg decodes: a
    -hwaccel with no decoder here, no software decoder bundled, or two GPU
    makers' decoders, which are never used together."""
    decoders = []
    for accel in (hwaccel.source, hwaccel.distorted):
        decoder = _DECODED_HERE.get(accel) if accel else gpu_frames.SOFTWARE if gpu_frames.software_bundled() else None
        if decoder is None:
            return None
        decoders.append(decoder)
    if len(set(decoders) - {gpu_frames.SOFTWARE}) > 1:
        return None
    return decoders[0], decoders[1]



def _score_decoded_on_gpu(
    plan: _GpuPlan, source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None, hwaccel: HwAccelPlan, total_frames: int, *,
    on_progress=None, on_status=None, cancel_event=None, process_handle=None,
) -> FrameScores:
    """Scores the plan's metrics with both videos decoded in this process,
    over the same frames _execute_run's FFmpeg would give the scorer.

    VMAF and NEG are scored by vmaf_gpu.score_decoded, or
    vmaf_vulkan.score_decoded for the Vulkan backend, and VMAF v1 by
    vmaf_v1_gpu.score_decoded. With the "cpu" backend, PSNR, SSIM and XPSNR
    are scored by libvmaf_cpu.score_decoded_cpu (for _score_cpu_metrics)."""
    fps = distorted_info.fps
    # As in _execute_run: one frame more than the limit, which FFmpeg's
    # libvmaf filter scores before its output stops.
    limit = options.duration_limit + 1 / fps if options.duration_limit > 0 and fps > 0 else 0.0
    decoders = _decoders_here(hwaccel)
    if decoders is None:
        raise gpu_frames.GpuDecodeUnavailableError("FFmpeg decodes the videos")
    if on_status:
        on_status(Status.decoding(f"Running {_cpu_metric_names(plan.models)} in the app" if plan.backend == "cpu"
                                  else "Running VMAF on the GPU", hwaccel, ending="..."))

    def check_cancel() -> None:
        """Raises Cancelled once the run is cancelled."""
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("Cancelled by user")
    if plan.backend == "cpu":
        score_decoded = functools.partial(libvmaf_cpu.score_decoded_cpu, threads=plan.threads,
                                          frame_rate=plan.frame_rate, decoder=decoders)
    elif "vmaf_v1" in plan.models:  # whole frames, for VMAF v1 and any VMAF v0.6.1 beside it
        score_decoded = functools.partial(vmaf_v1_gpu.score_decoded, backend=plan.backend, device=plan.device,
                                          decoder=decoders)
    elif plan.backend == "vulkan":
        score_decoded = functools.partial(vmaf_vulkan.score_decoded, device=plan.device, decoder=decoders)
    else:
        score_decoded = functools.partial(vmaf_gpu.score_decoded, decoder=decoders)
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
    """Runs a resolution round-trip test (see VmafOptions.resample_test): the
    source is scaled down to a target width and back up to its own
    resolution, and the metrics compare that with the untouched source. It
    needs only the source file, not a second, already-encoded one.
    """
    assert options.resample_test is not None

    hwaccel = HwAccelPlan()
    if options.gpu_decode:
        # One input file, so there is no test video to plan for.
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
        """The FFmpeg command of the round-trip test (a CommandBuilder)."""
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
