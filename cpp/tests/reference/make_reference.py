"""Writes the C++ tests' expected values, made by the Python app's own code:
its statistics (vmaf_app/core/stats.py), its metric registry
(vmaf_app/core/metrics.py), Python's own number formatting, its reading of
ffprobe's output (ffprobe.py), of settings.json (settings.py) and of
FFmpeg's version (ffmpeg_locate.py).

The ffprobe cases use ffprobe's real output on the golden clips
(cpp/tests/golden; golden.py clips makes them), kept in the file, so the
tests need neither the clips nor ffprobe.

Run it with the app's Python from the repository's root, after changing any
of those, and commit what it writes:

    .venv\\Scripts\\python.exe cpp\\tests\\reference\\make_reference.py

Numbers are written with float.hex() ("0x1.7a00000000000p+6", "inf", "nan"),
so that they reach the C++ tests exactly. Inputs come from fixed seeds, so the
files only change when the code that makes them does.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repository's root, for vmaf_app

import os
import subprocess
import tempfile

from vmaf_app.core import ffprobe as ffprobe_module
from vmaf_app.core import settings as settings_module
from vmaf_app.core.ffmpeg_locate import format_version, parse_version
from vmaf_app.core.metrics import METRICS
from vmaf_app.core.settings import Settings
from vmaf_app.core.stats import aggregate_scores, compute_stats

GOLDEN_CLIPS = Path(os.environ.get("VML_GOLDEN_CLIPS") or Path.home() / "Documents" / "VideoMetricsLab-golden" / "clips")


def exact(value: float | None) -> str | None:
    """A number as text that reads back exactly: float.hex(), or None."""
    return None if value is None else float(value).hex()


def write(name: str, data: dict) -> None:
    path = HERE / name
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ------------------------------------------------------------ number format

def number_format_cases() -> dict:
    """repr() and "{:.Nf}" of numbers chosen for their edges (where repr
    switches to exponent form, halfway cases, zeros, the extremes) and of
    random numbers of every size."""
    rng = random.Random(20261013)
    values = [
        0.0, -0.0, 1.0, -1.0, 90.0, 94.5, 0.99, 45.35198465983073, 9.108460618689008, 100.0, 0.1, 0.2, 0.3,
        1e-4, 1.5e-4, 9.999e-5, 1e-5, 1.2345e-5, 1e15, 1e16, 9.999999999999998e15, 1.2345678901234567e16,
        123456789012345.6, 1e22, 1e100, 1e-100, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
        0.125, 0.375, 2.675, 1.005, 0.5, 1.5, 2.5, -0.001, -0.0049, 0.0049, 0.005, 72.0, 60.0, 1 / 3,
        2 / 3, math.pi, math.e,
    ]
    for _ in range(200):
        values.append(rng.uniform(-1, 1) * 10.0 ** rng.randint(-320, 307))
    for _ in range(200):
        values.append(rng.uniform(0, 110))  # scores' own range
    values += [math.inf, -math.inf, math.nan]
    fixed = [{"value": exact(v), "decimals": d, "text": f"{v:.{d}f}"}
             for v in values if abs(v) < 1e30 or not math.isfinite(v) for d in (0, 1, 2, 3, 4, 6)]
    return {"repr": [{"value": exact(v), "text": repr(v)} for v in values], "fixed": fixed}


# ------------------------------------------------------------------ metrics

def metric_cases() -> dict:
    """The registry, and format_value() of numbers and of no number."""
    probe_values = [None, math.nan, math.inf, -math.inf, 0.0, -0.0, 94.5, 0.98765, 45.351984, -1.25, 1e-9, 12345.678]
    metrics = []
    for metric in METRICS:
        metrics.append({
            "key": metric.key, "label": metric.label, "short_label": metric.short_label,
            "table_header": metric.table_header, "axis_label": metric.axis_label,
            "value_format": metric.value_format, "value_suffix": metric.value_suffix, "kind": metric.kind.value,
            "direction": metric.direction.value, "aggregation": metric.aggregation.value,
            "fixed_y_max": exact(metric.fixed_y_max),
            "thresholds": [[comparison, exact(value)] for comparison, value in metric.thresholds],
            "bool_option": metric.ffmpeg_binding.bool_option if metric.ffmpeg_binding else None,
            "libvmaf_feature": metric.ffmpeg_binding.libvmaf_feature if metric.ffmpeg_binding else None,
            "backend_id": metric.backend_id,
            "format_value": [{"value": exact(v), "text": metric.format_value(v)} for v in probe_values],
        })
    return {"metrics": metrics}


# -------------------------------------------------------------------- stats

def stats_cases() -> dict:
    """compute_stats() and aggregate_scores() over the shapes of data the app
    meets: every length where numpy's pairwise sum changes its steps, NaN
    frames, identical frames (+inf), all-infinite runs and ties."""
    rng = random.Random(1013)

    def scores(n: int, low: float = 60.0, high: float = 100.0) -> list[float]:
        return [rng.uniform(low, high) for _ in range(n)]

    shapes: list[tuple[str, list[float]]] = [("empty", []), ("one", [93.25]), ("two", [93.25, 88.5])]
    for n in (3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 130, 240, 257, 1000, 2001):
        shapes.append((f"random {n}", scores(n)))
    with_nan = scores(300)
    for i in range(0, 300, 7):
        with_nan[i] = math.nan
    shapes.append(("NaN frames", with_nan))
    with_identical = scores(200, 30.0, 50.0)
    for i in range(0, 200, 9):
        with_identical[i] = math.inf
    shapes.append(("identical frames", with_identical))
    mostly_identical = [math.inf] * 150 + scores(10, 40.0, 45.0)
    rng.shuffle(mostly_identical)
    shapes.append(("mostly identical frames", mostly_identical))
    shapes.append(("all identical", [math.inf] * 50))
    shapes.append(("all minus infinity", [-math.inf] * 5))
    shapes.append(("both infinities", [-math.inf, 1.0, 2.0, math.inf]))
    shapes.append(("ties", [90.0] * 40 + [80.0] * 25 + [95.0] * 35))
    shapes.append(("all NaN", [math.nan] * 4))
    shapes.append(("butteraugli-like", scores(500, 0.2, 4.0)))
    shapes.append(("ssim-like", scores(400, 0.9, 1.0)))

    by_key = {metric.key: metric for metric in METRICS}
    cases = []
    for name, values in shapes:
        for key in ("vmaf", "xpsnr", "butteraugli", "ssim"):
            metric = by_key[key]
            for bands in ("metric's", "none"):
                thresholds = list(metric.thresholds) if bands == "metric's" else []
                stats = compute_stats(values, thresholds, metric.aggregation, metric.direction)
                cases.append({
                    "name": f"{name}, {key}, {bands} bands",
                    "shape": name,
                    "thresholds": [[comparison, exact(value)] for comparison, value in thresholds],
                    "aggregation": metric.aggregation.value, "direction": metric.direction.value,
                    "aggregate": exact(aggregate_scores(values, metric.aggregation)),
                    "expected": {
                        "count": stats.count, "identical": stats.identical, "mean": exact(stats.mean),
                        "median": exact(stats.median), "stdev": exact(stats.stdev),
                        "minimum": exact(stats.minimum), "maximum": exact(stats.maximum),
                        "worst_10_percent": exact(stats.percentile_10),
                        "worst_5_percent": exact(stats.percentile_5),
                        "worst_1_percent": exact(stats.percentile_1),
                        "worst_0_1_percent": exact(stats.percentile_0_1),
                        "worst_is_high": stats.worst_is_high,
                        "thresholds": [{"count": t.count, "percentage": exact(t.percentage)} for t in stats.thresholds],
                    },
                })
    # Each shape's values once; the cases name theirs.
    return {"shapes": {name: [exact(v) for v in values] for name, values in shapes}, "cases": cases}


# ------------------------------------------------------------------ ffprobe

class _FakeProbe:
    """ffprobe as probe_video starts it, giving `output`."""

    pid = 0
    returncode = 0

    def __init__(self, output: str):
        self.output = output

    def communicate(self, timeout=None):
        return self.output, ""


def _python_probe(path: str, output: str) -> dict:
    """probe_video's answer when ffprobe gives `output`: the details, or the
    error's message."""
    real_popen = ffprobe_module.proc_util.popen
    ffprobe_module.proc_util.popen = lambda command, **kwargs: _FakeProbe(output)
    try:
        info = ffprobe_module.probe_video(Path(path))
    except ffprobe_module.ProbeError as error:
        return {"error": str(error)}
    finally:
        ffprobe_module.proc_util.popen = real_popen
    return {
        "path": str(info.path), "width": info.width, "height": info.height, "fps": exact(info.fps),
        "duration": exact(info.duration), "nb_frames": info.nb_frames, "codec_name": info.codec_name,
        "sar": info.sar, "pix_fmt": info.pix_fmt, "bit_rate": info.bit_rate,
        "bit_rate_whole_file": info.bit_rate_whole_file, "nominal_fps": exact(info.nominal_fps),
        "average_fps": exact(info.average_fps), "color_range": info.color_range, "color_space": info.color_space,
        "color_transfer": info.color_transfer, "color_primaries": info.color_primaries,
        "chroma_location": info.chroma_location, "format_name": info.format_name,
        "start_offset": exact(info.start_offset),
    }


def _stream(**fields) -> dict:
    """A video stream as ffprobe lists one, with `fields` changed."""
    stream = {"index": 0, "codec_name": "hevc", "codec_type": "video", "width": 1920, "height": 1080,
              "pix_fmt": "yuv420p10le", "sample_aspect_ratio": "1:1", "r_frame_rate": "24000/1001",
              "avg_frame_rate": "24000/1001", "disposition": {"attached_pic": 0, "timed_thumbnails": 0}}
    stream.update(fields)
    return stream


def probe_cases() -> dict:
    """probe_video over ffprobe's real output on the golden clips, and over
    output written for the rules the clips do not reach: Matroska's tags
    (DURATION, BPS, NUMBER_OF_BYTES), a soundtrack's bitrate, cover art
    before the video, missing or odd values, no video, and output that is
    not JSON."""
    cases = []
    for clip in sorted(GOLDEN_CLIPS.glob("*.*")):
        command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(clip)]
        data = json.loads(subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                         check=True).stdout)
        data["format"]["filename"] = clip.name  # no folder of this PC in the file
        output = json.dumps(data, indent=4)
        cases.append({"name": clip.name, "path": clip.name, "output": output,
                      "expected": _python_probe(clip.name, output)})

    def written(name: str, data) -> None:
        output = data if isinstance(data, str) else json.dumps(data)
        cases.append({"name": name, "path": f"{name}.mkv", "output": output,
                      "expected": _python_probe(f"{name}.mkv", output)})

    matroska = {"format_name": "matroska,webm", "duration": "120.500000", "size": "100000000",
                "bit_rate": "6639004", "start_time": "0.000000"}
    written("matroska tags", {"streams": [
        _stream(tags={"BPS": "5000000", "NUMBER_OF_BYTES": "75000000", "DURATION": "00:02:00.000000000"}),
        {"codec_type": "audio", "tags": {"bps": "640000", "DURATION": "00:02:00.500000000"}}], "format": matroska})
    written("bytes over duration", {"streams": [
        _stream(tags={"number_of_bytes": "60000000", "DURATION": "00:02:00.000000000"})], "format": matroska})
    written("stale tags", {"streams": [
        _stream(tags={"BPS": "90000000", "DURATION": "01:45:36.289000000"})], "format": matroska})
    written("soundtrack known", {"streams": [
        _stream(), {"codec_type": "audio", "bit_rate": "640000"}], "format": matroska})
    written("soundtrack unknown", {"streams": [_stream(), {"codec_type": "audio"}], "format": matroska})
    written("no soundtrack", {"streams": [_stream()], "format": matroska})
    written("cover art first", {"streams": [
        _stream(codec_name="mjpeg", width=600, height=600, disposition={"attached_pic": 1}),
        _stream(index=1, duration="10.010000", nb_frames="240")],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.050000", "start_time": "0.000000"}})
    written("rates", {"streams": [
        _stream(avg_frame_rate="0/0", r_frame_rate="25/1", nb_frames="N/A", sample_aspect_ratio="",
                start_time="1.500000", color_transfer="smpte2084", color_range="tv")],
        "format": {"format_name": "mpegts", "duration": "N/A", "start_time": "1.400000"}})
    written("rates missing", {"streams": [_stream(avg_frame_rate="", r_frame_rate="N/A")],
                              "format": {"format_name": "hevc"}})
    written("no video", {"streams": [{"codec_type": "audio"}], "format": matroska})
    written("not JSON", "this is not ffprobe's output")
    return {"cases": cases}


# ----------------------------------------------------------------- settings

def settings_cases() -> dict:
    """Settings.load over files: right, with values of the wrong type (each
    keeps its default), with unknown keys, not an object, not JSON, none.
    parallel_jobs is null where it is the PC's default, which depends on the
    PC's cores."""
    files = {
        "good": json.dumps({"ffmpeg_dir": "D:/tools/ffmpeg", "default_compute_xpsnr": False, "parallel_jobs": 2,
                            "gpu_backend": "vulkan", "hidden_metrics": ["psnr", "ssim"],
                            "cvvdp_presets": [{"name": "Mine", "settings": {"display": {}}}],
                            "cvvdp_default_preset": "Mine", "window_width": 1600}),
        "wrong types": json.dumps({"parallel_jobs": "2", "hidden_metrics": "psnr", "use_cache": 1,
                                   "window_width": 1600.0, "default_vmaf_on_gpu": False, "language": 5,
                                   "cvvdp_presets": ["not an object"], "remembered_tests": ["a.mkv", 3]}),
        "unknown keys": json.dumps({"a_future_setting": True, "theme": "dark"}),
        "not an object": json.dumps(["a", "list"]),
        "not JSON": "{ this is not json",
        "missing": None,
    }
    cases = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "settings.json"
        real_file = settings_module.settings_file
        settings_module.settings_file = lambda: path
        try:
            for name, text in files.items():
                path.unlink(missing_ok=True)
                if text is not None:
                    path.write_text(text, encoding="utf-8")
                values = dict(vars(Settings.load()))
                if text is None or '"parallel_jobs": 2' not in text:
                    values["parallel_jobs"] = None  # the PC's default
                values["cvvdp_presets"] = json.dumps(values["cvvdp_presets"], separators=(",", ":"))
                cases.append({"name": name, "file": text, "expected": values})
        finally:
            settings_module.settings_file = real_file
    return {"cases": cases}


# ------------------------------------------------------------------ version

def version_cases() -> dict:
    """parse_version and format_version over what FFmpeg's builds print."""
    outputs = [
        "ffmpeg version 9.0.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers",
        "ffprobe version n7.1 Copyright (c) 2007-2024 the FFmpeg developers",
        "ffmpeg version 8.0 Copyright", "ffmpeg version 10.2.3.4 Copyright",
        "ffmpeg version N-113411-g1234567 Copyright", "not a version at all", "",
        "xffmpeg version 9.0", "FFMPEG VERSION 9.0", "ffmpeg version 9.",
    ]
    return {"cases": [{"output": output, "version": parse_version(output),
                       "text": format_version(parse_version(output))} for output in outputs]}


# ---------------------------------------------------------------- decisions

def decision_cases() -> dict:
    """The run's pure decisions, made by the Python app's own functions:
    pixel formats and GPU decoding (gpu.py, with given -hwaccels and GPU
    makers), frame rates as fractions and frame counts under a limit
    (frame_sync.py, both ways of pairing), why two videos cannot be compared
    and the shapes they are compared at (geometry.py), and the VMAF model
    (model_select.py)."""
    from fractions import Fraction

    from vmaf_app.core import frame_sync, gpu
    from vmaf_app.core.geometry import compared_dimensions, display_aspect_ratio, pair_problem
    from vmaf_app.core.model_select import model_for_resolution, resolve_model, resolve_v1_model
    from vmaf_app.core.models import GpuVendor, ScaleDirection, VideoInfo, VmafOptions

    formats = ["", "yuv420p", "yuv420p10le", "yuv420p12le", "yuv420p16le", "yuvj420p", "yuv422p10le",
               "yuv444p", "nv12", "p010le", "p016le", "p012be", "gray", "gray10le", "ya16be", "rgb24", "gbrp12be",
               "bgr0", "YUV420P10LE", "x2rgb10le"]
    pixel_formats = [{"format": f, "bit_depth": gpu.bit_depth(f), "hw_native": gpu.hw_native_format(f),
                      "downloads": [gpu.downloads_from_gpu(f, w, h) for w, h in ((1920, 1080), (853, 480), (0, 0))]}
                     for f in formats]
    analysis = [{"formats": pair, "format": gpu.analysis_pix_fmt(*pair)}
                for pair in (["yuv420p", "yuv420p"], ["yuv420p", "yuv420p10le"], ["p010le", "nv12"],
                             ["yuv444p16le", "yuv420p"], [], ["gray12le"])]

    real_hwaccels, real_vendors = gpu.available_hwaccels, gpu.detected_gpu_vendors
    decode = []
    try:
        for hwaccels in (["cuda", "d3d11va", "qsv"], ["d3d11va"], ["qsv", "d3d11va"], []):
            for vendors in ([GpuVendor.NVIDIA], [GpuVendor.INTEL, GpuVendor.AMD], [GpuVendor.AMD], []):
                gpu.available_hwaccels = lambda h=hwaccels: set(h)
                gpu.detected_gpu_vendors = lambda v=vendors: list(v)
                for vendor in GpuVendor:
                    for codec, pix_fmt, size in (("hevc", "yuv420p10le", (3840, 2160)), ("h264", "yuv420p10le", (1920, 1080)),
                                                 ("h264", "yuv420p", (1920, 1080)), ("av1", "yuv420p10le", (853, 479)),
                                                 ("vvc", "yuv420p10le", (1920, 1080)), ("HEVC", "yuv422p10le", (1920, 1080)),
                                                 ("vp9", "", (0, 0)), ("mpeg2video", "yuv420p", (720, 576))):
                        decode.append({"hwaccels": hwaccels, "vendors": [v.value for v in vendors],
                                       "vendor": vendor.value, "codec": codec, "pix_fmt": pix_fmt, "size": list(size),
                                       "hwaccel": gpu.pick_decode(vendor, codec, pix_fmt, *size)})
    finally:
        gpu.available_hwaccels, gpu.detected_gpu_vendors = real_hwaccels, real_vendors
    hwaccel_args = [{"hwaccel": h, "args": gpu.hwaccel_args(h)} for h in (None, "cuda", "qsv", "d3d11va", "vulkan")]

    rng = random.Random(613)
    rates = [120.0, 24000 / 1001, 30000 / 1001, 60000 / 1001, 2997 / 125, 25.0, 23.976, 59.94, 0.5, 1e-3, 1000.0,
             119.88011988011988, 10000000 / 417083, 1 / 3, math.pi, 47.952047952047955, 0.0]
    rates += [rng.uniform(1, 240) for _ in range(40)]
    fractions = [{"value": exact(r), "fraction": [Fraction(r).limit_denominator(1_000_000).numerator,
                                                  Fraction(r).limit_denominator(1_000_000).denominator]}
                 for r in rates]
    real_pairing = os.environ.pop(frame_sync.PAIRING_VARIABLE, None)
    pairing = {}
    try:
        for mode in ("", "timestamp", " TimeStamp "):
            os.environ[frame_sync.PAIRING_VARIABLE] = mode
            rate = Fraction(24000, 1001)
            pairing[mode] = {"framesync": frame_sync.framesync_opts(), "pairing": frame_sync.pairing_filters(rate),
                             "numbering": frame_sync.numbering_filters(rate),
                             "duration": frame_sync.duration_args("30.042"), "no_duration": frame_sync.duration_args(None)}
    finally:
        os.environ.pop(frame_sync.PAIRING_VARIABLE, None)
        if real_pairing is not None:
            os.environ[frame_sync.PAIRING_VARIABLE] = real_pairing
    within = [{"limit": limit, "rate": [rate.numerator, rate.denominator],
               "frames": frame_sync.frames_within(limit, rate)}
              for limit in ("0.271", "1.000", "30.042", "0.000", "0.0005", "86400.123456789", "2", "1.9999999")
              for rate in (Fraction(120), Fraction(24000, 1001), Fraction(119999999, 1000000), Fraction(25))]
    within.append({"limit": None, "rate": [120, 1], "frames": frame_sync.frames_within(None, Fraction(120))})

    def video(**fields) -> VideoInfo:
        base = {"path": Path("video.mkv"), "width": 1920, "height": 1080, "fps": 24000 / 1001, "duration": 60.0,
                "nb_frames": 0, "codec_name": "hevc"}
        base.update(fields)
        return VideoInfo(**base)

    pairs = []
    for name, source, test, limit in (
            ("same", video(), video(), 0.0),
            ("ntsc against whole", video(fps=24.0), video(), 0.0),
            ("two containers", video(fps=2997 / 125), video(), 0.0),
            ("variable", video(nominal_fps=30.0, fps=24000 / 1001), video(), 0.0),
            ("durations", video(duration=60.0), video(duration=30.0), 0.0),
            ("durations, limit", video(duration=60.0), video(duration=30.0), 20.0),
            ("limit past the end", video(duration=60.0), video(duration=30.0), 40.0),
            ("two frames apart", video(duration=60.0), video(duration=60.08), 0.0),
            ("raw stream", video(format_name="hevc", codec_name="hevc", fps=25.0, path=Path("raw.hevc")),
             video(duration=30.0), 0.0),
            ("raw h264", video(), video(format_name="h264", codec_name="h264", fps=25.0, path=Path("c.264")), 0.0),
            ("unknown durations", video(duration=0.0), video(duration=30.0), 0.0)):
        pairs.append({"name": name, "problem": pair_problem(source, test, limit),
                      "source": {"fps": exact(source.fps), "duration": exact(source.duration), "nominal_fps":
                                 exact(source.nominal_fps), "format_name": source.format_name,
                                 "codec_name": source.codec_name, "path": str(source.path)},
                      "test": {"fps": exact(test.fps), "duration": exact(test.duration), "nominal_fps":
                               exact(test.nominal_fps), "format_name": test.format_name,
                               "codec_name": test.codec_name, "path": str(test.path)},
                      "limit": exact(limit)})
    shapes = [{"width": w, "height": h, "sar": sar, "aspect": exact(display_aspect_ratio(video(width=w, height=h, sar=sar)))}
              for w, h, sar in ((1920, 1080, "1:1"), (1440, 1080, "4:3"), (720, 576, "64:45"), (1920, 1080, "N/A"),
                                (1920, 1080, "0:1"), (1920, 1080, "bad"), (1920, 0, "1:1"), (853, 480, "1:1"))]
    compared = [{"source": list(s), "test": list(t), "direction": d.value,
                 "size": list(compared_dimensions(video(width=s[0], height=s[1]), video(width=t[0], height=t[1]), d))}
                for s, t in (((3840, 2160), (1920, 1080)), ((1920, 1080), (1920, 1080)), ((1920, 800), (3840, 1600)))
                for d in ScaleDirection]
    root = Path(__file__).resolve().parents[3] / "vmaf_app"

    def model(choice: str, choice_v1: str, size: tuple[int, int]) -> dict:
        options = VmafOptions(model_choice=choice, model_choice_v1=choice_v1, custom_model_path="C:/m/custom.json")
        made = {"choice": choice, "choice_v1": choice_v1, "size": list(size)}
        for key, resolve in (("model", resolve_model), ("model_v1", resolve_v1_model)):
            try:
                value = resolve(options, *size)
                if value.startswith("path=") and Path(value[5:]).is_relative_to(root):
                    value = "path=<resources>/" + Path(value[5:]).relative_to(root).as_posix()
                made[key] = value
            except (ValueError, FileNotFoundError) as error:
                made[key] = {"error": str(error)}
        return made

    models = [model(choice, choice_v1, size)
              for choice in ("__auto__", "version=vmaf_v0.6.1", "__custom__", "__builtin:vmaf_v1_5d0h", "__builtin:nope")
              for choice_v1 in ("__auto__", "__builtin:vmaf_v1_hfr_3d0h", "version=vmaf_v0.6.1")
              for size in ((1920, 1080), (3840, 2160), (3840, 1600), (2560, 2160))]
    resolutions = [{"size": [w, h], "model": model_for_resolution(w, h)}
                   for w, h in ((3839, 2159), (3840, 1), (1, 2160), (1920, 1080))]
    return {"pixel_formats": pixel_formats, "analysis": analysis, "decode": decode, "hwaccel_args": hwaccel_args,
            "fractions": fractions, "pairing": pairing, "within": within, "pairs": pairs, "shapes": shapes,
            "compared": compared, "models": models, "resolutions": resolutions}


if __name__ == "__main__":
    write("decisions.json", decision_cases())
    write("number_format.json", number_format_cases())
    write("metrics.json", metric_cases())
    write("stats.json", stats_cases())
    write("probe.json", probe_cases())
    write("settings.json", settings_cases())
    write("versions.json", version_cases())
