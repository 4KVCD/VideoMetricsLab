"""Builds and runs the ffmpeg + libvmaf command, streaming progress and
parsing the resulting per-frame JSON log.
"""
from __future__ import annotations

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

from vmaf_app.core.crop_detect import detect_crop
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.gpu import pick_hwaccel
from vmaf_app.core.models import (
    CropBox,
    CropMode,
    FrameScores,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
    VmafRunResult,
    synthetic_resample_distorted_path,
)
from vmaf_app.core.process_control import ProcessHandle

ProgressCallback = Callable[[int, int, float], None]  # (current_frame, total_frames, fps)


class VmafRunError(RuntimeError):
    def __init__(self, message: str, stderr_tail: str = ""):
        super().__init__(message)
        self.stderr_tail = stderr_tail


class Cancelled(RuntimeError):  # noqa: N818 - a cancellation, not an error condition
    """Raised when the user cancels a run mid-flight. Deliberately not named
    `CancelledError`: nothing went wrong, and callers treat it as an expected
    outcome rather than a failure to report."""


def _resolve_crops(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    status_callback: Callable[[str], None] | None,
) -> tuple[CropBox | None, CropBox | None]:
    if options.crop_mode == CropMode.NONE:
        return None, None

    if options.crop_mode == CropMode.MANUAL:
        return options.manual_source_crop, options.manual_distorted_crop

    if status_callback:
        status_callback("Detecting black bars in source...")
    src_crop = detect_crop(source_info)
    if status_callback:
        status_callback("Detecting black bars in distorted...")
    dist_crop = detect_crop(distorted_info)
    return src_crop, dist_crop


def _hw_native_format(source_pix_fmt: str) -> str:
    """The system-memory pixel format a cuda/qsv/d3d11va hw surface downloads
    to, based on the source's bit depth. 10/12-bit 4:2:0 sources decode to a
    p010-family surface; everything else (including 8-bit) decodes to nv12."""
    if "10le" in source_pix_fmt or "12le" in source_pix_fmt or "p010" in source_pix_fmt:
        return "p010le"
    return "nv12"


def _build_libvmaf_opts(options: VmafOptions, log_path: Path, model: str | None = None) -> list[str]:
    # ffmpeg's filtergraph option parser can't reliably handle an absolute
    # Windows path (drive-letter colon) as an option value, even escaped or
    # quoted -- so ffmpeg is always launched with cwd=log_path.parent and we
    # reference the log (and any custom model file) by bare filename here.
    opts = [
        f"log_path={log_path.name}",
        "log_fmt=json",
        f"model={model if model is not None else options.model}",
    ]
    # libvmaf 2.0+ defaults to single-threaded (n_threads=1) unless told
    # otherwise -- omitting this option here does NOT mean "use all cores",
    # so "Auto" (n_threads <= 0) is resolved to the actual core count instead
    # of leaving it unset.
    resolved_threads = options.n_threads if options.n_threads > 0 else (os.cpu_count() or 1)
    opts.append(f"n_threads={resolved_threads}")
    if options.n_subsample > 1:
        opts.append(f"n_subsample={options.n_subsample}")
    if options.extra_features:
        opts.append("feature=" + "|".join(options.extra_features))
    return opts


def _build_libvmaf_stage(
    options: VmafOptions, log_path: Path, model: str | None, xpsnr_log_path: Path | None,
) -> str:
    """The XPSNR + libvmaf tail shared by both filtergraph builders. XPSNR
    isn't a libvmaf "feature" like PSNR/SSIM -- it's a fully separate ffmpeg
    filter with its own stats file -- so when requested it sits between
    decode and libvmaf, passing [main] through under a new label.
    """
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
        chains.append(f"[main][ref_xpsnr]xpsnr=stats_file={xpsnr_log_path.name}[xmain]")
        main_label = "xmain"
        ref_label = "ref_vmaf"
    chains.append(f"[{main_label}][{ref_label}]libvmaf=" + ":".join(libvmaf_opts))
    return ";".join(chains)


def _build_filtergraph(
    source_info: VideoInfo, distorted_info: VideoInfo, options: VmafOptions,
    source_crop: CropBox | None, distorted_crop: CropBox | None,
    hwaccel_used: str | None, log_path: Path, model: str | None = None,
    xpsnr_log_path: Path | None = None,
) -> str:
    dist_content_w = distorted_crop.w if distorted_crop else distorted_info.width
    dist_content_h = distorted_crop.h if distorted_crop else distorted_info.height
    ref_content_w = source_crop.w if source_crop else source_info.width
    ref_content_h = source_crop.h if source_crop else source_info.height
    resolutions_differ = (ref_content_w, ref_content_h) != (dist_content_w, dist_content_h)
    upscale_distorted = resolutions_differ and options.scale_direction == ScaleDirection.DISTORTED_TO_SOURCE

    # --- distorted (main, input 0) chain ---
    main_ops = []
    if distorted_crop and not distorted_crop.is_noop(distorted_info.width, distorted_info.height):
        main_ops.append(distorted_crop.as_filter())
    main_ops.append("format=yuv420p")
    if upscale_distorted:
        # Scale the distorted video UP to the source's resolution instead of
        # the default (scaling the source down to the distorted video's
        # resolution) -- see ScaleDirection.
        main_ops.append(f"scale={ref_content_w}:{ref_content_h}:flags={options.scale_algorithm}")
    main_ops.append("setpts=PTS-STARTPTS")
    main_chain = f"[0:v]{','.join(main_ops)}[main]"

    # --- source / reference (input 1) chain ---
    ref_ops = []
    if hwaccel_used:
        # hwdownload can only emit the hw surface's native format -- nv12 for
        # 8-bit cuda decode, p010le for 10-bit (common for UHD/HDR masters) --
        # it can't itself target yuv420p, so that conversion needs its own
        # separate format filter afterwards.
        ref_ops.append("hwdownload")
        ref_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    ref_ops.append("format=yuv420p")
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

    base_ops = []
    if hwaccel_used:
        # See _build_filtergraph's identical comment: hwdownload can only
        # emit the hw surface's native format, not yuv420p directly.
        base_ops.append("hwdownload")
        base_ops.append(f"format={_hw_native_format(source_info.pix_fmt)}")
    base_ops.append("format=yuv420p")
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
    hwaccel: str | None, duration_limit: float = 0.0,
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    # -i paths are plain argv (not filtergraph syntax) so absolute Windows
    # paths are fine here even though they aren't inside the filtergraph --
    # but they must be made absolute first, since ffmpeg's cwd is set to a
    # temp dir below (see _build_filtergraph's log_path/model comment).
    cmd += ["-i", str(Path(distorted_path).resolve())]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel]
    cmd += ["-i", str(Path(source_path).resolve())]
    cmd += _build_ffmpeg_output_args(filtergraph, duration_limit)
    return cmd


def _build_resample_cmd(
    source_path: Path, filtergraph: str, hwaccel: str | None, duration_limit: float = 0.0,
) -> list[str]:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel]
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
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=str(cwd),
    )
    if process_handle is not None:
        process_handle.attach(proc.pid)
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
        if process_handle is not None:
            process_handle.detach()


def estimate_total_frames(reference_info: VideoInfo, options: VmafOptions) -> int:
    """The number of frames libvmaf will actually score: `reference_info` is
    whichever video drives the output timeline (the distorted video for a
    normal run, the source for a resolution round-trip test), bounded by
    duration_limit and thinned by n_subsample. Used both to size the
    progress bar during a run and, upfront, to estimate total queue time
    across jobs that haven't started yet.
    """
    frame_count = reference_info.estimated_frame_count
    if options.duration_limit > 0:
        frame_count = min(frame_count, round(options.duration_limit * reference_info.fps))
    return frame_count // max(1, options.n_subsample)


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
    vmafs: list[float] = []
    psnrs: list[float | None] = []
    ssims: list[float | None] = []
    xpsnrs: list[float | None] = []

    for fr in data.get("frames", []):
        metrics = fr.get("metrics", {})
        frame_num = int(fr.get("frameNum", len(frame_nums)))
        vmaf = metrics.get("vmaf")
        if vmaf is None:
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
        vmafs.append(float(vmaf))
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
        vmaf=np.array(vmafs, dtype=np.float32),
        psnr=column(psnrs), ssim=column(ssims), xpsnr=column(xpsnrs),
    )


_XPSNR_LINE_RE = re.compile(r"n:\s*(\d+)\s+XPSNR y:\s*(-?[\d.]+)")


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


#: (hwaccel or None for software, model resolved relative to the run's temp
#: dir, libvmaf log path, xpsnr log path or None) -> the ffmpeg argv to run.
#: The two run flavours differ only in this, so _execute_run takes it as a
#: parameter rather than duplicating the whole pipeline around it.
CommandBuilder = Callable[[str | None, str | None, Path, Path | None], list[str]]


def _execute_run(
    build_command: CommandBuilder,
    *,
    options: VmafOptions,
    fps: float,
    total_frames: int,
    hwaccel: str | None,
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
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        log_path = tmpdir / "vmaf_log.json"
        xpsnr_log_path = tmpdir / "xpsnr_log.txt" if options.compute_xpsnr else None
        resolved_model = _resolve_model_for_cwd(options.model, tmpdir)

        def run_with(accel: str | None):
            cmd = build_command(accel, resolved_model, log_path, xpsnr_log_path)
            return _run_ffmpeg(
                cmd, total_frames, on_progress, cancel_event,
                cwd=tmpdir, process_handle=process_handle,
            )

        if on_status:
            on_status(f"Running ffmpeg (GPU decode: {hwaccel or 'off'})...")
        result = run_with(hwaccel)

        if result.returncode != 0 and hwaccel is not None:
            # GPU decode path failed to launch/decode -- retry on CPU.
            if on_status:
                on_status("GPU decode failed, retrying with software decode...")
            result = run_with(None)

        if result.returncode != 0:
            tail = "\n".join(result.stderr.splitlines()[-25:])
            raise VmafRunError(f"ffmpeg exited with code {result.returncode}", stderr_tail=tail)

        if not log_path.exists():
            raise VmafRunError("ffmpeg finished but no VMAF log was produced.", stderr_tail=result.stderr[-2000:])

        return _parse_log(log_path, fps, xpsnr_log_path)


def run_vmaf(
    source_info: VideoInfo,
    distorted_info: VideoInfo,
    options: VmafOptions,
    on_progress: ProgressCallback | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
    result_distorted_path: Path | None = None,
) -> VmafRunResult:
    """result_distorted_path overrides the returned result's `distorted`
    identity (defaulting to distorted_info.path). It doesn't affect which
    file is actually decoded -- only what identity the result carries for
    caching/graphing -- so a caller running the *same* physical file twice
    under different options (e.g. both ScaleDirection values) can give each
    run a distinct identity instead of one colliding with/overwriting the
    other, the same way a resample test's synthetic path already does.
    """
    source_crop, distorted_crop = _resolve_crops(source_info, distorted_info, options, on_status)

    hwaccel = None
    if options.gpu_decode_source:
        hwaccel = pick_hwaccel(options.gpu_vendor, source_info.codec_name)

    def build_command(accel, model, log_path, xpsnr_log_path):
        filtergraph = _build_filtergraph(
            source_info, distorted_info, options, source_crop, distorted_crop, accel, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_ffmpeg_cmd(
            distorted_info.path, source_info.path, filtergraph, accel, options.duration_limit,
        )

    frames = _execute_run(
        build_command,
        options=options,
        fps=distorted_info.fps,
        total_frames=estimate_total_frames(distorted_info, options),
        hwaccel=hwaccel,
        tmp_prefix="vmaf_run_",
        on_progress=on_progress,
        on_status=on_status,
        cancel_event=cancel_event,
        process_handle=process_handle,
    )

    return VmafRunResult(
        source=source_info.path,
        distorted=result_distorted_path or distorted_info.path,
        frames=frames,
        fps=distorted_info.fps,
        model=options.model,
        source_crop=source_crop,
        distorted_crop=distorted_crop,
        source_info=source_info,
        distorted_info=distorted_info,
        scale_direction=options.scale_direction,
    )


def run_resample_test(
    source_info: VideoInfo,
    options: VmafOptions,
    on_progress: ProgressCallback | None = None,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    process_handle: ProcessHandle | None = None,
) -> VmafRunResult:
    """Runs a resolution round-trip test (see VmafOptions.resample_test):
    downscales the source to a target width, scales it back up to the
    source's original resolution, and computes VMAF against the untouched
    source -- from a single input file, not a second already-encoded one.
    """
    assert options.resample_test is not None

    source_crop: CropBox | None = None
    if options.crop_mode == CropMode.AUTO:
        if on_status:
            on_status("Detecting black bars in source...")
        source_crop = detect_crop(source_info)
    elif options.crop_mode == CropMode.MANUAL:
        source_crop = options.manual_source_crop

    hwaccel = None
    if options.gpu_decode_source:
        hwaccel = pick_hwaccel(options.gpu_vendor, source_info.codec_name)

    def build_command(accel, model, log_path, xpsnr_log_path):
        filtergraph = _build_resample_test_filtergraph(
            source_info, options, source_crop, accel, log_path,
            model=model, xpsnr_log_path=xpsnr_log_path,
        )
        return _build_resample_cmd(source_info.path, filtergraph, accel, options.duration_limit)

    frames = _execute_run(
        build_command,
        options=options,
        fps=source_info.fps,
        total_frames=estimate_total_frames(source_info, options),
        hwaccel=hwaccel,
        tmp_prefix="vmaf_resample_",
        on_progress=on_progress,
        on_status=on_status,
        cancel_event=cancel_event,
        process_handle=process_handle,
    )

    distorted_path = synthetic_resample_distorted_path(source_info.path, options.resample_test)
    return VmafRunResult(
        source=source_info.path,
        distorted=distorted_path,
        frames=frames,
        fps=source_info.fps,
        model=options.model,
        source_crop=source_crop,
        distorted_crop=source_crop,  # same crop applies to both branches, since both come from the same source
        source_info=source_info,
        distorted_info=source_info,  # after the round trip it's back at the source's own resolution
    )
