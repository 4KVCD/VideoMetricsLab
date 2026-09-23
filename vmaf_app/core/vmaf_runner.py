"""Builds and runs the ffmpeg + libvmaf command, streaming progress and
parsing the resulting per-frame JSON log.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vmaf_app.core import proc as proc_util
from vmaf_app.core.crop_detect import CropDetectCancelled, detect_crop
from vmaf_app.core.ffmpeg_locate import check_tools, ffmpeg_path, format_version
from vmaf_app.core.gpu import (
    HwAccelPlan,
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
from vmaf_app.core.metric_results import current_ffmpeg_provenance, results_from_frame_scores
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, model_for_resolution
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

ProgressCallback = Callable[[int, int, float], None]  # (current_frame, total_frames, fps)


def _metric_results_for_current_run(frames: FrameScores, model: str):
    """Adapt one FFmpeg parse into generic results without re-parsing it."""
    status = check_tools()
    version = format_version(status.ffmpeg.version) if status.ffmpeg.runnable else "unknown"
    return results_from_frame_scores(
        frames,
        {
            key: current_ffmpeg_provenance(
                key, version,
                ({"model": "version=vmaf_v0.6.1neg"} if key == "vmaf_neg"
                 else {"model": model} if key == "vmaf" else None),
            )
            for key in frames.metric_keys
        },
    )


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
    """Rejects comparisons whose timelines/display geometry are ambiguous."""
    if source_info.is_variable_frame_rate or distorted_info.is_variable_frame_rate:
        raise VmafRunError(
            "Variable-frame-rate video is not supported safely yet. Convert both videos "
            "to the same constant frame rate before comparing them."
        )
    fps_tolerance = max(0.01, max(source_info.fps, distorted_info.fps) * 0.001)
    if abs(source_info.fps - distorted_info.fps) > fps_tolerance:
        raise VmafRunError(
            f"Frame rates do not match ({source_info.fps:.3f} vs "
            f"{distorted_info.fps:.3f} fps)."
        )
    if source_info.duration > 0 and distorted_info.duration > 0:
        compared_limit = options.duration_limit
        if compared_limit <= 0:
            frame_duration = 1.0 / max(source_info.fps, distorted_info.fps, 1.0)
            if abs(source_info.duration - distorted_info.duration) > max(0.1, 2 * frame_duration):
                raise VmafRunError(
                    f"Durations do not match ({source_info.duration:.3f} vs "
                    f"{distorted_info.duration:.3f} seconds). Set a duration limit within "
                    "both files if comparing only their common opening segment."
                )
        elif min(source_info.duration, distorted_info.duration) + 0.1 < compared_limit:
            raise VmafRunError(
                "The duration limit extends beyond the end of one of the videos."
            )


#: Two shapes count as the same if they agree to within this fraction. Wide
#: enough for the rounding that even dimensions force (1918x1080 against
#: 1920x1080 differs by 0.1%), far tighter than any real mismatch: 4:3
#: against 16:9 is 33% apart, and the letterbox case below is 32%.
_ASPECT_TOLERANCE = 0.01


def _sar_fraction(sar: str) -> tuple[int, int]:
    """A sample aspect ratio as a fraction. Unknown/unset means square."""
    if sar in {"", "N/A", "0:1"}:
        return 1, 1
    try:
        num, den = (int(part) for part in sar.split(":", 1))
    except ValueError:
        return 1, 1
    if num <= 0 or den <= 0:
        return 1, 1
    return num, den


def display_aspect_ratio(info: VideoInfo, crop: CropBox | None = None) -> float:
    """The shape of the picture as displayed, after cropping.

    Storage dimensions alone are not the shape: non-square pixels stretch
    them, and a crop changes them. This is what has to match between two
    videos, not the raw SAR string -- 1920x1080 SAR 1:1 and 1440x1080 SAR
    4:3 are the same 16:9 picture stored two ways.
    """
    width = crop.w if crop else info.width
    height = crop.h if crop else info.height
    if height <= 0:
        return 0.0
    num, den = _sar_fraction(info.sar)
    return (width * num) / (height * den)


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

    if options.crop_mode == CropMode.MANUAL:
        return options.manual_source_crop, options.manual_distorted_crop

    plan = hwaccel or HwAccelPlan()
    if status_callback:
        status_callback("Detecting black bars in source...")
    try:
        src_crop = detect_crop(
            source_info, cancel_event=cancel_event, process_handle=process_handle,
            hwaccel=plan.source,
        )
    except CropDetectCancelled as e:
        raise Cancelled("Cancelled by user") from e
    if status_callback:
        status_callback("Detecting black bars in distorted...")
    try:
        dist_crop = detect_crop(
            distorted_info, cancel_event=cancel_event, process_handle=process_handle,
            hwaccel=plan.distorted,
        )
    except CropDetectCancelled as e:
        raise Cancelled("Cancelled by user") from e
    return src_crop, dist_crop


#: Analysis bit depth -> the planar 4:2:0 format both branches are converted
#: to before they meet. libvmaf compares two streams that must agree on
#: format, so one has to be picked for the pair.
_ANALYSIS_FORMAT_BY_DEPTH = {8: "yuv420p", 10: "yuv420p10le", 12: "yuv420p12le"}


def analysis_pix_fmt(*pix_fmts: str) -> str:
    """The common format the inputs are converted to before comparison.

    Takes the *deepest* of the inputs, so a 10-bit master compared against
    an 8-bit encode promotes the encode rather than truncating the master.
    Everything used to be forced to 8-bit yuv420p, which quietly discarded
    two bits of both sides on any HDR/10-bit comparison and put a floor
    under PSNR/XPSNR that had nothing to do with the encode being measured.
    """
    depth = max((_bit_depth(f) for f in pix_fmts), default=8)
    if depth <= 8:
        return _ANALYSIS_FORMAT_BY_DEPTH[8]
    if depth <= 10:
        return _ANALYSIS_FORMAT_BY_DEPTH[10]
    # libvmaf accepts up to 12-bit; deeper sources (16-bit intermediates)
    # are analysed at 12 rather than being dropped back to 8.
    return _ANALYSIS_FORMAT_BY_DEPTH[12]


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
    if options.compute_vmaf_neg:
        # Explicit names keep the two output score arrays independent.
        models = ([model_value + r"\\:name=vmaf"] if options.compute_vmaf else [])
        models.append(r"version=vmaf_v0.6.1neg\\:name=vmaf_neg")
        model_value = "|".join(models)
    opts = [
        f"log_path={log_path.name}",
        "log_fmt=json",
        f"model={model_value}" if options.compute_vmaf or options.compute_vmaf_neg else "model=''",
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
    opts += _FRAMESYNC_OPTS
    return opts


#: Both libvmaf and xpsnr are framesync filters, and framesync's defaults are
#: wrong for measurement: repeatlast=true extends the last frame of the
#: secondary input past its EOF, and eof_action=repeat keeps the comparison
#: going. A distorted file two frames longer than the source -- routine
#: encoder padding, and well inside the duration tolerance -- therefore got
#: two extra "scores" comparing real distorted frames against a frozen copy
#: of the source's final frame. Those frames score terribly (48 and 31 on a
#: 30-frame fixture that is otherwise ~100) and drag the aggregate down, so
#: the run silently reports a worse encode than was delivered.
_FRAMESYNC_OPTS = ["shortest=1", "repeatlast=0"]


def _build_libvmaf_stage(
    options: VmafOptions, log_path: Path, model: str | None, xpsnr_log_path: Path | None,
) -> str:
    """The XPSNR + libvmaf tail shared by both filtergraph builders. XPSNR
    isn't a libvmaf "feature" like PSNR/SSIM -- it's a fully separate ffmpeg
    filter with its own stats file -- so when requested it sits between
    decode and libvmaf, passing [main] through under a new label.
    """
    if not options.requested_metrics():
        raise VmafRunError("Select at least one metric to calculate.")
    # XPSNR-only needs no libvmaf filter or model at all.
    if not options.compute_vmaf and not options.compute_vmaf_neg and not options.extra_features:
        assert xpsnr_log_path is not None
        return f"[main][ref]xpsnr=stats_file={xpsnr_log_path.name}:" + ":".join(_FRAMESYNC_OPTS)
    libvmaf_opts = _build_libvmaf_opts(options, log_path, model)
    chains = []
    main_label = "main"
    ref_label = "ref"
    if options.compute_xpsnr and xpsnr_log_path is not None:
        # xpsnr consumes [ref], and libvmaf needs it too -- but a filtergraph
        # label can only be consumed once. Without this explicit split,
        # ffmpeg silently wires libvmaf up to the wrong stream and it ends up
        # comparing the distorted video against itself, reporting a perfect
        # VMAF 100 / PSNR 60 / SSIM 1.0 for every frame no matter how bad the
        # encode actually is. It does NOT error out, so the scores just come
        # back quietly, plausibly wrong.
        chains.append("[ref]split=2[ref_xpsnr][ref_vmaf]")
        chains.append(
            f"[main][ref_xpsnr]xpsnr=stats_file={xpsnr_log_path.name}:"
            + ":".join(_FRAMESYNC_OPTS) + "[xmain]"
        )
        main_label = "xmain"
        ref_label = "ref_vmaf"
    chains.append(f"[{main_label}][{ref_label}]libvmaf=" + ":".join(libvmaf_opts))
    return ";".join(chains)


def _content_size(info: VideoInfo, crop: CropBox | None) -> tuple[int, int]:
    return (crop.w, crop.h) if crop else (info.width, info.height)


def analysis_dimensions(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None = None, distorted_crop: CropBox | None = None,
) -> tuple[int, int]:
    """The size frames are actually compared at.

    One side is scaled to the other before they reach libvmaf, so neither
    input's own resolution need be the analysis resolution: a 1080p encode
    measured with "upscale distorted to source" against a 4K master is
    compared at 4K. Cropping moves it too. Shared with _build_filtergraph so
    the two cannot disagree about what the run does.
    """
    dist_content = _content_size(distorted_info, distorted_crop)
    ref_content = _content_size(source_info, source_crop)
    if ref_content == dist_content:
        return dist_content
    if options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE:
        return ref_content
    return dist_content


def resample_analysis_dimensions(
    source_info: VideoInfo, source_crop: CropBox | None = None
) -> tuple[int, int]:
    """A round-trip test compares two branches of one input at the source's
    own (cropped) size -- the downscale is undone before comparison."""
    return _content_size(source_info, source_crop)


def _build_filtergraph(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
    hwaccel: HwAccelPlan, log_path: Path, model: str | None = None,
    xpsnr_log_path: Path | None = None,
) -> str:
    dist_content_w, dist_content_h = _content_size(distorted_info, distorted_crop)
    ref_content_w, ref_content_h = _content_size(source_info, source_crop)
    resolutions_differ = (ref_content_w, ref_content_h) != (dist_content_w, dist_content_h)
    upscale_distorted = resolutions_differ and options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE

    # Both branches have to reach libvmaf in the same pixel format, and that
    # format is chosen from the deeper of the two inputs -- see
    # analysis_pix_fmt for why it is not simply yuv420p.
    analysis_format = analysis_pix_fmt(source_info.pix_fmt, distorted_info.pix_fmt)

    # --- distorted (main, input 0) chain ---
    main_ops = []
    if hwaccel.distorted:
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
    main_chain = f"[0:v]{','.join(main_ops)}[main]"

    # --- source / reference (input 1) chain ---
    ref_ops = []
    if hwaccel.source:
        # hwdownload can only emit the hw surface's native format -- nv12 for
        # 8-bit cuda decode, p010le for 10-bit (common for UHD/HDR masters) --
        # it can't itself target the analysis format, so that conversion
        # needs its own separate format filter afterwards.
        ref_ops.append("hwdownload")
        ref_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    ref_ops.append(f"format={analysis_format}")
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        ref_ops.append(source_crop.as_filter())

    if resolutions_differ and not upscale_distorted:
        ref_ops.append(f"scale={dist_content_w}:{dist_content_h}:flags={options.scale_algorithm}")

    ref_ops.append("setpts=PTS-STARTPTS")
    ref_chain = f"[1:v]{','.join(ref_ops)}[ref]"

    tail = _build_libvmaf_stage(options, log_path, model, xpsnr_log_path)
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
    resolution -- there's no second file, both branches come from [0:v].
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
    base_ops.append(f"format={analysis_format}")
    if source_crop and not source_crop.is_noop(source_info.width, source_info.height):
        base_ops.append(source_crop.as_filter())
    base_chain = f"[0:v]{','.join(base_ops)}[base]"

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
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    # -i paths are plain argv (not filtergraph syntax) so absolute Windows
    # paths are fine here even though they aren't inside the filtergraph --
    # but they must be made absolute first, since ffmpeg's cwd is set to a
    # temp dir below (see _build_filtergraph's log_path/model comment).
    cmd += _hwaccel_args(hwaccel.distorted)
    cmd += ["-i", str(Path(distorted_path).resolve())]
    cmd += _hwaccel_args(hwaccel.source)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit)
    return cmd


def _build_resample_cmd(
    source_path: Path, filtergraph: str, hwaccel: str | None, duration_limit: float = 0.0,
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    cmd += _hwaccel_args(hwaccel)
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit)
    return cmd


def _build_ffmpeg_output_args(filtergraph: str, duration_limit: float) -> list[str]:
    args = ["-lavfi", filtergraph, "-progress", "pipe:1", "-nostats"]
    if duration_limit > 0:
        # An output-side -t caps how much of the filtered output is produced
        # (and so how many frames reach libvmaf), regardless of any length
        # mismatch between the two inputs -- simpler than trying to bound
        # each input separately.
        args += ["-t", f"{duration_limit:.3f}"]
    args += ["-f", "null", "-"]
    return args


def _run_ffmpeg(
    cmd: list[str], total_frames: int,
    on_progress: ProgressCallback | None, cancel_event: threading.Event | None,
    cwd: Path, process_handle: ProcessHandle | None = None,
) -> subprocess.CompletedProcess:
    # Checked before spawning, not only inside the read loop: cancelling
    # during crop detection or between the fallback attempts would otherwise
    # start one more ffmpeg that then had to be hunted down and killed.
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    proc = proc_util.popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=str(cwd),
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
                proc.terminate()
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
        if cancel_event is not None and cancel_event.is_set():
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
            process_handle.detach()


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
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate() is a polite request that a wedged decoder can
                # ignore; kill() is not refusable.
                proc.kill()
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
    stops at whichever input ends first (see _FRAMESYNC_OPTS), so a distorted
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
    psnrs: list[float | None] = []
    ssims: list[float | None] = []
    xpsnrs: list[float | None] = []

    for fr in data.get("frames", []):
        metrics = fr.get("metrics", {})
        frame_num = int(fr.get("frameNum", len(frame_nums)))
        vmaf = metrics.get("vmaf")
        if not any(k in metrics for k in ("vmaf", "vmaf_neg", "psnr_y", "psnr", "float_ssim", "ssim")):
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
CommandBuilder = Callable[[HwAccelPlan, str | None, Path, Path | None], list[str]]


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
) -> FrameScores:
    """Runs one ffmpeg invocation to completion and parses its logs.

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
            (model if model is not None else options.model) if options.compute_vmaf else "", tmpdir
        )

        def run_with(plan: HwAccelPlan):
            cmd = build_command(plan, resolved_model, log_path, xpsnr_log_path)
            return _run_ffmpeg(
                cmd, total_frames, on_progress, cancel_event,
                cwd=tmpdir, process_handle=process_handle,
            )

        ladder = _fallback_ladder(hwaccel)
        result = None
        for attempt, plan in enumerate(ladder):
            if on_status:
                if attempt == 0:
                    on_status(f"Running ffmpeg (GPU decode: {plan.describe()})...")
                else:
                    on_status(
                        f"GPU decode failed, retrying (GPU decode: {plan.describe()})..."
                    )
            result = run_with(plan)
            if result.returncode == 0:
                break
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

        if not options.compute_vmaf and not options.compute_vmaf_neg and not options.extra_features:
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
            options.gpu_vendor, source_info.codec_name, distorted_info.codec_name
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
    effective_model = _auto_model_or(
        options,
        analysis_dimensions(
            source_info, distorted_info, options, source_crop, distorted_crop
        ),
    )

    def build_command(plan, model, log_path, xpsnr_log_path):
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, plan, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_ffmpeg_cmd(
            distorted_info.path, source_info.path, filtergraph, plan, options.duration_limit,
        )

    total_frames = estimate_total_frames(distorted_info, options, source_info)
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
        metric_results=_metric_results_for_current_run(frames, effective_model),
    )


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
        hwaccel = plan_hwaccel(options.gpu_vendor, source_info.codec_name)

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
    elif options.crop_mode == CropMode.MANUAL:
        source_crop = options.manual_source_crop

    effective_model = _auto_model_or(
        options, resample_analysis_dimensions(source_info, source_crop)
    )

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
        metric_results=_metric_results_for_current_run(frames, effective_model),
    )
