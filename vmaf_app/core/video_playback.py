"""ffmpeg commands for frame-locked source/distorted video playback."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.frame_extract import (
    FrameComparison,
    PreviewColorMode,
    PreviewColorSettings,
    comparison_dimensions,
    frame_filter,
    frame_input_path,
    frame_video_info,
    hdr_kind,
)
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import ScaleDirection
from vmaf_app.core.vmaf_runner import _hw_native_format


def source_playback_comparison(comparison, native=True):
    """Independent source-only geometry; never mutate metric recipes.

    Matching an encode only downsizes the source, preserving its aspect ratio.
    Crops are applied before comparing sizes. Smaller sources are not enlarged.
    """
    source, crop = comparison.source_info, comparison.source_crop
    width, height = (crop.w, crop.h) if crop else (source.width, source.height)
    if not native:
        encoded, encoded_crop = comparison.distorted_info, comparison.distorted_crop
        ew, eh = (encoded_crop.w, encoded_crop.h) if encoded_crop else (encoded.width, encoded.height)
        factor = min(1.0, ew / width, eh / height)
        if factor < 1:
            width, height = max(2, int(width * factor) // 2 * 2), max(2, int(height * factor) // 2 * 2)
    return replace(comparison, distorted_info=replace(source, width=width, height=height),
                   distorted_crop=None, scale_direction=ScaleDirection.SOURCE_TO_DISTORTED,
                   resample_target=None)


def neighbour_indices(count: int, selected: int) -> tuple[int, ...]:
    """Current, left and right, following the panel's wraparound navigation."""
    if count <= 0 or not 0 <= selected < count:
        return ()
    return tuple(dict.fromkeys((selected, (selected - 1) % count, (selected + 1) % count)))


def series_layout(comparisons, maximum=None):
    """Deduplicate identical views (usually the source), preserving pair mapping.

    Each recipe retains its own geometry. The FFmpeg transport pads these to
    equal tiles; selection crops the padding away again without copying pixels.
    """
    recipes, pairs, keys = [], [], {}
    for comparison in comparisons:
        pair = []
        size = playback_dimensions(comparison, maximum)
        for side in ("source", "distorted"):
            info = frame_video_info(comparison, side)
            crop = comparison.source_crop if side == "source" else comparison.distorted_crop
            key = (info, crop, size, comparison.auto_crop_pending, comparison.scale_algorithm)
            # VideoInfo is mutable; equality, not object identity, identifies a
            # reusable view. Do not hash it or assume all source crops agree.
            index = next((i for i, existing in keys.items() if existing == key), None)
            if index is None:
                index = len(recipes)
                keys[index] = key
                recipes.append((comparison, side, size))
            pair.append(index)
        pairs.append(tuple(pair))
    return recipes, pairs


def build_video_series_command(
    comparisons, start_frame, settings, plans, maximum=None, *, realtime,
    processing="vulkan", side=None, paced=True,
):
    """One clocked RGBA atlas containing every view, not just the selected pair.

    Vulkan decode keeps supported streams on the processing GPU. The transfer
    retry retains CUDA decode with an explicit host bridge; the software-decode
    retry still uses GPU processing. CPU processing is the final safe fallback.
    """
    if not comparisons or len(plans) != len(comparisons):
        raise ValueError("a decode plan is required for every comparison")
    fps = comparisons[0].fps
    if fps <= 0 or start_frame < 0:
        raise ValueError("invalid playback frame rate or start frame")
    if processing not in {"vulkan", "transfer", "software", "cpu"}:
        raise ValueError("unknown playback processing mode")
    recipes, _pairs = series_layout(comparisons, maximum)
    if side is not None:
        if len(comparisons) != 1 or side not in {"source", "distorted"}:
            raise ValueError("a single stream requires one comparison and a valid side")
        recipes = [(comparisons[0], side, playback_dimensions(comparisons[0], maximum))]
    tile_w = max(size[0] for _, _, size in recipes)
    tile_h = max(size[1] for _, _, size in recipes)
    timestamp = max(0, (start_frame - 0.125) / fps)
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    gpu = processing != "cpu"
    if gpu:
        cmd += ["-init_hw_device", "vulkan=preview", "-filter_hw_device", "preview"]
    # A single demuxer/decoder per file, even with different crops of its source.
    inputs, recipe_inputs = [], []
    for comparison, side, _ in recipes:
        path = frame_input_path(comparison, side).resolve()
        index = next((i for i, (p, _) in enumerate(inputs) if p == path), None)
        if index is None:
            plan = plans[comparisons.index(comparison)]
            accel = plan.source if side == "source" else plan.distorted
            if processing == "vulkan" and accel:
                accel = "vulkan"
            elif processing in {"software", "cpu"}:
                accel = None
            index = len(inputs)
            inputs.append((path, accel))
            args = _input_args(path, timestamp, accel, realtime and paced)
            if accel == "vulkan":
                args[-2:-2] = ["-hwaccel_device", "preview"]
            cmd += args
        recipe_inputs.append(index)
    graph = []
    for index in range(len(inputs)):
        outputs = [f"[in{n}]" for n, i in enumerate(recipe_inputs) if i == index]
        graph.append(f"[{index}:v:0]split={len(outputs)}" + "".join(outputs))
    for n, (comparison, side, size) in enumerate(recipes):
        info = frame_video_info(comparison, side)
        crop = comparison.source_crop if side == "source" else comparison.distorted_crop
        accel = inputs[recipe_inputs[n]][1]
        ops = []
        if gpu:
            # setparams only repairs missing HDR descriptors; never overwrites
            # explicitly declared SDR transfer characteristics.
            kind = hdr_kind(info)
            assumed_hdr = settings.mode == PreviewColorMode.HDR_TO_SDR and not info.color_transfer
            if kind or assumed_hdr:
                unknown = {"", "unknown", "unspecified", "reserved"}
                params = []
                for name, value, default in (
                    ("color_primaries", info.color_primaries, "bt2020"),
                    ("color_trc", info.color_transfer, "smpte2084"),
                    ("colorspace", info.color_space, "bt2020nc"),
                    ("range", info.color_range, "limited"),
                ):
                    if value.casefold() in unknown:
                        params.append(f"{name}={default}")
                if params:
                    ops.append("setparams=" + ":".join(params))
            if accel and accel != "vulkan":
                ops += ["hwdownload", f"format={_hw_native_format(info.pix_fmt)}"]
            if accel != "vulkan":
                ops.append("hwupload")
            options = [f"w={size[0]}", f"h={size[1]}", "format=rgba", "colorspace=gbr", "range=pc"]
            if crop is not None:
                options += [f"crop_x={crop.x}", f"crop_y={crop.y}", f"crop_w={crop.w}", f"crop_h={crop.h}"]
            if comparison.auto_crop_pending:
                options += ["normalize_sar=1", "fit_mode=contain"]
            if settings.mode != PreviewColorMode.UNMANAGED:
                options += ["color_primaries=bt709", "color_trc=iec61966-2-1", "tonemapping=bt.2390"]
                # A stable mapping on both sides: independent scene peak
                # detection would change the rendering when the encode differs.
                options += ["peak_detect=0"]
            ops += ["libplacebo=" + ":".join(options), "hwdownload", "format=rgba"]
        else:
            ops += [frame_filter(comparison, side, settings, output_size=size), "format=rgba"]
        ops += [f"fps={fps:.12g}", f"setpts=N/({fps:.12g}*TB)", "setsar=1"]
        if size != (tile_w, tile_h):
            ops.append(f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black")
        graph.append(f"[in{n}]" + ",".join(ops) + f"[tile{n}]")
    labels = "".join(f"[tile{n}]" for n in range(len(recipes)))
    if len(recipes) == 1:
        graph.append(labels + "null[out]")
    else:
        graph.append(labels + f"hstack=inputs={len(recipes)}:shortest=1[out]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[out]", "-an", "-sn", "-dn",
            "-pix_fmt", "rgba", "-fps_mode", "passthrough"]
    if not realtime:
        cmd += ["-frames:v", "1"]
    cmd += ["-f", "rawvideo", "pipe:1"]
    return cmd


def playback_dimensions(
    comparison: FrameComparison,
    maximum: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Largest even preview size that fits the physical display, if supplied.

    There is deliberately no built-in 1920-pixel budget.  Native GPU playback
    does not materialize this image in Python at all, and the FFmpeg fallback
    is limited by the actual monitor rather than an arbitrary resolution.
    """
    width, height = comparison_dimensions(comparison)
    if width <= 0 or height <= 0:
        raise ValueError("comparison has no usable display dimensions")
    if maximum is None:
        return max(2, width // 2 * 2), max(2, height // 2 * 2)
    max_width, max_height = maximum
    scale = min(1.0, max_width / width, max_height / height)
    out_w = max(2, int(width * scale) // 2 * 2)
    out_h = max(2, int(height * scale) // 2 * 2)
    return out_w, out_h


def _hwaccel_args(hwaccel: str | None) -> list[str]:
    if not hwaccel:
        return []
    return ["-hwaccel", hwaccel, "-hwaccel_output_format", hwaccel]


def _input_args(
    path: Path,
    timestamp: float,
    hwaccel: str | None,
    realtime: bool,
) -> list[str]:
    args: list[str] = []
    if realtime:
        args += ["-readrate", "1"]
    args += ["-ss", f"{timestamp:.9f}"]
    args += _hwaccel_args(hwaccel)
    args += ["-i", str(path.resolve())]
    return args


def _side_chain(
    comparison: FrameComparison,
    side: str,
    input_index: int,
    hwaccel: str | None,
    color_settings: PreviewColorSettings,
    output_size: tuple[int, int],
) -> str:
    prefix = ""
    if hwaccel:
        info = (
            comparison.source_info if side == "source"
            else comparison.distorted_info
        )
        prefix = f"hwdownload,format={_hw_native_format(info.pix_fmt)},"
    filters = frame_filter(
        comparison, side, color_settings, output_size=output_size
    )
    # Both streams are converted to the comparison's declared frame rate and
    # rebased to frame zero. hstack then emits one indivisible pair per tick:
    # the UI can flip halves without consulting two independent media clocks.
    fps = f"{comparison.fps:.12g}"
    return (
        f"[{input_index}:v]{prefix}{filters},fps={fps},"
        f"setpts=N/({fps}*TB)[{side}]"
    )


def build_video_pair_command(
    comparison: FrameComparison,
    start_frame: int,
    color_settings: PreviewColorSettings,
    hwaccel: HwAccelPlan,
    output_size: tuple[int, int],
    *,
    realtime: bool,
) -> list[str]:
    """Decode a stream of horizontally packed, frame-exact comparison pairs."""
    if comparison.fps <= 0:
        raise ValueError("comparison has no usable frame rate")
    if start_frame < 0:
        raise ValueError("start frame must be non-negative")
    timestamp = max(0.0, (start_frame - 0.125) / comparison.fps)
    source = frame_input_path(comparison, "source")
    distorted = frame_input_path(comparison, "distorted")
    source_chain = _side_chain(
        comparison, "source", 0, hwaccel.source,
        color_settings, output_size,
    )
    distorted_chain = _side_chain(
        comparison, "distorted", 1, hwaccel.distorted,
        color_settings, output_size,
    )
    graph = ";".join([
        source_chain,
        distorted_chain,
        "[source][distorted]hstack=inputs=2:shortest=1[out]",
    ])
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    cmd += _input_args(source, timestamp, hwaccel.source, realtime)
    cmd += _input_args(distorted, timestamp, hwaccel.distorted, realtime)
    cmd += [
        "-filter_complex", graph,
        "-map", "[out]",
        "-an", "-sn", "-dn",
        "-pix_fmt", "rgb24",
        "-fps_mode", "passthrough",
    ]
    if not realtime:
        cmd += ["-frames:v", "1"]
    cmd += ["-f", "rawvideo", "pipe:1"]
    return cmd


def ffplay_path() -> Path | None:
    candidate = Path(ffmpeg_path()).with_name(
        "ffplay.exe" if Path(ffmpeg_path()).suffix.lower() == ".exe" else "ffplay"
    )
    return candidate if candidate.is_file() else None


def build_audio_command(comparison: FrameComparison, start_frame: int) -> list[str] | None:
    """Play only the distorted audio through ffplay's native audio output."""
    player = ffplay_path()
    if player is None or comparison.fps <= 0:
        return None
    timestamp = max(0.0, start_frame / comparison.fps)
    return [
        str(player), "-nodisp", "-autoexit", "-loglevel", "error",
        "-ss", f"{timestamp:.9f}",
        "-i", str(frame_input_path(comparison, "distorted").resolve()),
        "-vn", "-sn",
    ]
