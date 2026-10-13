"""Writes the C++ tests' expected values, made by the Python app's own code:
its statistics (vmaf_app/core/stats.py), its metric registry
(vmaf_app/core/metrics.py), Python's own number formatting, its reading of
ffprobe's output (ffprobe.py), of settings.json (settings.py) and of
FFmpeg's version (ffmpeg_locate.py), a run's decisions (gpu.py,
frame_sync.py, geometry.py, model_select.py), FFmpeg's commands for a run
(vmaf_runner.py, vmaf_gpu.py) and the reading of its logs, the decoding of
videos in the app (gpu_frames.py, vmaf_runner.py), VMAF predicted from
features by the bundled libvmaf and scored with the Vulkan engine
(vmaf_vulkan.py), Vship's view of a video and its colours
(perceptual_vship.py, colour.py), the line that says how far a video is
(ui/run_line.py), its command line's reading of arguments and patterns
(cli.py), and Python's json.dumps. The Vulkan engine's cases need a GPU
that Vulkan scores VMAF on.

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
    # The clips golden.py makes, and nothing else the folder may hold (an old
    # cut kept beside them, say).
    listed = json.loads((HERE.parent / "golden" / "clips.json").read_text(encoding="utf-8"))["clips"]
    clips = sorted(GOLDEN_CLIPS / entry["name"] for entry in listed)
    if missing := [clip.name for clip in clips if not clip.is_file()]:
        raise SystemExit(f"{', '.join(missing)} not in {GOLDEN_CLIPS}: make the clips first (golden.py clips).")
    for clip in clips:
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
    from vmaf_app.core.crop_detect import common_picture
    from vmaf_app.core.models import CropBox

    def box(values):
        return None if values is None else CropBox(*values)

    shared = []
    for source_size, test_size, source_box, test_box in (
            ((1920, 1080), (1920, 1080), (1920, 800, 0, 140), (1920, 802, 0, 138)),
            ((3840, 2160), (1920, 1080), (3840, 1600, 0, 280), (1920, 798, 0, 142)),
            ((3840, 2160), (1920, 1080), (3840, 1608, 0, 276), (1920, 804, 0, 138)),
            ((1920, 1080), (1440, 1080), (1920, 800, 0, 140), (1440, 800, 0, 140)),
            ((1920, 1080), (1920, 1080), (1918, 1080, 2, 0), (1916, 1078, 0, 2)),
            ((1920, 1080), (1920, 1080), (800, 1080, 0, 0), (800, 1080, 1100, 0)),
            ((1920, 1080), (1920, 1080), None, (1920, 800, 0, 140)),
            ((853, 480), (1706, 960), (851, 357, 1, 61), (1702, 714, 2, 122))):
        source = video(width=source_size[0], height=source_size[1])
        test = video(width=test_size[0], height=test_size[1])
        made = common_picture(source, test, box(source_box), box(test_box))
        shared.append({"source": list(source_size), "test": list(test_size), "source_box": source_box,
                       "test_box": test_box,
                       "shared": [None if b is None else [b.w, b.h, b.x, b.y] for b in made]})
    return {"pixel_formats": pixel_formats, "analysis": analysis, "decode": decode, "hwaccel_args": hwaccel_args,
            "fractions": fractions, "pairing": pairing, "within": within, "pairs": pairs, "shapes": shapes,
            "compared": compared, "models": models, "resolutions": resolutions, "common_picture": shared}


# --------------------------------------------------------------- FFmpeg runs

def run_cases() -> dict:
    """FFmpeg's filter graph and command for a comparison on the CPU
    (vmaf_runner._build_filtergraph and _build_ffmpeg_cmd), over video
    pairs, crops, decode plans, metrics, scaling, limits and both ways of
    pairing; the two FFmpeg commands of VMAF on the GPU (_build_stream_cmds
    with vmaf_gpu.GpuAttempt's outputs); and libvmaf's log and XPSNR's stats
    file read back (_parse_log). The commands' program and input paths are
    written as <ffmpeg>, <test> and <source>, which differ between PCs."""
    from fractions import Fraction  # noqa: F401  (frame_sync needs it imported)
    from types import SimpleNamespace

    from vmaf_app.core import frame_sync, vmaf_gpu, vmaf_runner
    from vmaf_app.core.gpu import HwAccelPlan
    from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo, VmafOptions

    def video(name: str, width: int, height: int, pix_fmt: str, fps: float) -> VideoInfo:
        return VideoInfo(path=Path(name), width=width, height=height, fps=fps, duration=10.0, nb_frames=0,
                         codec_name="hevc", pix_fmt=pix_fmt)

    pairs = {
        "4k against 1080p": (video("source.mp4", 3840, 2160, "yuv420p10le", 120.0),
                             video("test.mkv", 1920, 1080, "yuv420p10le", 120.0)),
        "same size, 8 against 10 bits": (video("source.mkv", 1920, 1080, "yuv420p", 24000 / 1001),
                                          video("test.mkv", 1920, 1080, "yuv420p10le", 24000 / 1001)),
    }
    crops = {"none": (None, None), "letterbox": (CropBox(3840, 1608, 0, 276), CropBox(1920, 804, 0, 138))}
    plans = {"software": HwAccelPlan(), "cuda": HwAccelPlan("cuda", "cuda"), "cuda source": HwAccelPlan(source="cuda"),
             "qsv": HwAccelPlan("qsv", "qsv"), "d3d11va test": HwAccelPlan(distorted="d3d11va")}
    v1_model = Path(__file__).resolve().parents[3] / "vmaf_app" / "models" / "vmaf_v1.0.16" / "vmaf_v1.0.16_3d0h.json"
    choices = {
        "vmaf": dict(compute_vmaf=True),
        "vmaf, neg": dict(compute_vmaf=True, compute_vmaf_neg=True),
        "vmaf, v1": dict(compute_vmaf=True, compute_vmaf_v1=True, model_v1=f"path={v1_model}"),
        "vmaf, psnr, ssim": dict(compute_vmaf=True, extra_features=["name=psnr", "name=float_ssim"]),
        "vmaf, xpsnr": dict(compute_vmaf=True, compute_xpsnr=True),
        "all": dict(compute_vmaf=True, compute_vmaf_neg=True, compute_xpsnr=True,
                    extra_features=["name=psnr", "name=float_ssim"]),
        "psnr, xpsnr": dict(compute_vmaf=False, compute_xpsnr=True, extra_features=["name=psnr"]),
        "xpsnr": dict(compute_vmaf=False, compute_xpsnr=True),
        "neg alone": dict(compute_vmaf=False, compute_vmaf_neg=True),
    }
    settings = {
        "plain": dict(n_threads=12),
        "subsample, lanczos, upscale": dict(n_threads=8, n_subsample=3, scale_algorithm="lanczos",
                                            scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        "limit": dict(n_threads=24, duration_limit=1.0),
        "auto threads": dict(n_threads=0),
    }
    cases = []
    gpu_cases = []
    real_pairing = os.environ.pop(frame_sync.PAIRING_VARIABLE, None)
    try:
        for pairing in ("", "timestamp"):
            os.environ[frame_sync.PAIRING_VARIABLE] = pairing
            for pair_name, (source, test) in pairs.items():
                for crop_name, (source_crop, test_crop) in crops.items():
                    if crop_name == "letterbox" and pair_name != "4k against 1080p":
                        continue
                    for plan_name, plan in plans.items():
                        for choice_name, choice in choices.items():
                            for setting_name, setting in settings.items():
                                if pairing and (plan_name != "cuda" or setting_name == "auto threads"):
                                    continue  # the other way of pairing changes only the frame filters and -t
                                options = VmafOptions(**choice, **setting)
                                xpsnr_log = Path("xpsnr_log.txt") if options.compute_xpsnr else None
                                model = "version=vmaf_v0.6.1" if options.compute_vmaf else ""
                                graph = vmaf_runner._build_filtergraph(
                                    source, test, options, source_crop, test_crop, plan, Path("vmaf_log.json"),
                                    model=model, xpsnr_log_path=xpsnr_log)
                                command = vmaf_runner._build_ffmpeg_cmd(test.path, source.path, graph, plan,
                                                                       options.duration_limit)
                                command[0] = "<ffmpeg>"
                                command = ["<test>" if part == str(test.path.resolve()) else
                                           "<source>" if part == str(source.path.resolve()) else part
                                           for part in command]
                                cases.append({
                                    "name": f"{pairing or 'frame number'}; {pair_name}; {crop_name}; {plan_name}; "
                                            f"{choice_name}; {setting_name}",
                                    "pairing": pairing, "pair": pair_name, "crop": crop_name, "plan": plan_name,
                                    "choice": choice_name, "setting": setting_name, "model": model,
                                    "cores": os.cpu_count(), "command": command})
        # VMAF on the GPU, each video decoded by an FFmpeg of its own whose
        # frames the app pairs (vmaf_gpu.GpuAttempt): the two commands, the
        # outputs' pipes written as <test pipe> and <source pipe>.
        os.environ.pop(frame_sync.PAIRING_VARIABLE, None)
        attempt = SimpleNamespace(paired=False, _by_time=False, distorted=SimpleNamespace(path="<test pipe>"),
                                  reference=SimpleNamespace(path="<source pipe>"))
        for pair_name, (source, test) in pairs.items():
            for crop_name, (source_crop, test_crop) in crops.items():
                if crop_name == "letterbox" and pair_name != "4k against 1080p":
                    continue
                for plan_name, plan in plans.items():
                    for setting_name in ("plain", "subsample, lanczos, upscale", "limit"):
                        options = VmafOptions(compute_vmaf=True, compute_vmaf_neg=True, **settings[setting_name])
                        graphs = vmaf_runner._build_filtergraph(
                            source, test, options, source_crop, test_crop, plan, Path("vmaf_log.json"),
                            gpu_vmaf=True, gpu_paired=False)
                        # As vmaf_runner._execute_run: one frame past the limit.
                        limit = options.duration_limit + 1 / test.fps if options.duration_limit > 0 else 0.0
                        commands = vmaf_runner._build_stream_cmds(
                            test.path, source.path, graphs, plan, vmaf_gpu.GpuAttempt.output_args(attempt, limit))
                        commands = [["<ffmpeg>"] + ["<test>" if part == str(test.path.resolve()) else
                                                    "<source>" if part == str(source.path.resolve()) else part
                                                    for part in command[1:]] for command in commands]
                        gpu_cases.append({"name": f"{pair_name}; {crop_name}; {plan_name}; {setting_name}",
                                          "pair": pair_name, "crop": crop_name, "plan": plan_name,
                                          "setting": setting_name, "limit": exact(limit), "commands": commands})
    finally:
        os.environ.pop(frame_sync.PAIRING_VARIABLE, None)
        if real_pairing is not None:
            os.environ[frame_sync.PAIRING_VARIABLE] = real_pairing

    def frame(number: int, **metrics) -> dict:
        return {"frameNum": number, "metrics": metrics}

    logs = {
        "every metric": {"frames": [frame(0, vmaf=99.356133, vmaf_neg=94.537025, vmaf_v1=97.85, psnr_y=56.146637,
                                          float_ssim=0.999746),
                                    frame(1, vmaf=0.0, vmaf_neg=93.1, vmaf_v1=94.8, psnr_y=0.0, float_ssim=0.0),
                                    frame(2, vmaf=98.5, psnr=50.25, ssim=0.98)]},
        "holes": {"frames": [frame(0, vmaf=99.0), frame(3, integer_motion=1.0), {"metrics": {"vmaf": 97.0}},
                             frame(7, vmaf=96.25, psnr_y=44.5)]},
        "none": {"frames": []},
        # libvmaf's own layout: the frames between other keys, and pooled
        # scores after them under the same names, which are not frames.
        "libvmaf's layout": {
            "version": "3.0.0", "fps": 23.98,
            "frames": [frame(0, integer_adm2=0.98, integer_motion2=0.0, vmaf=97.25, psnr_y=48.5),
                       frame(1, integer_adm2=0.97, integer_motion2=1.5, vmaf=96.5, psnr_y=47.75)],
            "pooled_metrics": {"vmaf": {"min": 96.5, "max": 97.25, "mean": 96.875},
                               "psnr_y": {"min": 47.75, "max": 48.5, "mean": 48.125}},
            "aggregate_metrics": {}},
        # No number where a metric is named.
        "nulls": {"frames": [frame(0, vmaf=None, psnr_y=None, psnr=40.0), frame(1, float_ssim=0.5)]},
    }
    xpsnr_logs = {"none": None,
                  "some": "n:    1  XPSNR y: 61.6992  XPSNR u: 63.1  XPSNR v: 64.2\n"
                          "n:    2  XPSNR y: inf  XPSNR u: inf  XPSNR v: inf\n"
                          "n:    4  XPSNR Y: 45.25\nnot a line\n"}
    parsed = []
    with tempfile.TemporaryDirectory() as folder:
        for log_name, log in logs.items():
            for xpsnr_name, xpsnr_text in xpsnr_logs.items():
                log_path = Path(folder) / "vmaf_log.json"
                log_path.write_text(json.dumps(log), encoding="utf-8")
                xpsnr_path = None
                if xpsnr_text is not None:
                    xpsnr_path = Path(folder) / "xpsnr_log.txt"
                    xpsnr_path.write_text(xpsnr_text, encoding="utf-8")
                frames = vmaf_runner._parse_log(log_path, 24000 / 1001, xpsnr_path)
                parsed.append({
                    "name": f"{log_name}, xpsnr {xpsnr_name}", "log": json.dumps(log), "xpsnr_log": xpsnr_text,
                    "frames": [int(n) for n in frames.frame], "times": [exact(t) for t in frames.time],
                    "metrics": {key: [exact(v) for v in frames.values(key)] for key in frames.metric_keys}})
    return {"commands": cases, "gpu_commands": gpu_cases, "logs": parsed, "log_fps": exact(24000 / 1001)}


# ----------------------------------------------------------------- run line

def run_line_cases() -> dict:
    """A video's run line (ui/run_line.py's metric_line) over snapshots of
    its tasks as job_runner's JobRun makes them, for FFmpeg's task, VMAF's
    on the GPU and Vship's, in one or more passes, with SSIMULACRA2 and
    Butteraugli on the CPU; and step_text and numbers_text."""
    from vmaf_app.core.gpu import GPU_WAIT_MESSAGE, HwAccelPlan
    from vmaf_app.core.status import GPU_PASS, GPU_VMAF_FAILED, GPU_WAIT, STARTING, Status
    from vmaf_app.ui import run_line

    steps = {
        "none": "",
        "ffmpeg starting": Status.decoding("Running ffmpeg", HwAccelPlan("cuda", None), ending="...", kind=STARTING),
        "in the app": Status.decoding("Running PSNR, SSIM and XPSNR in the app", HwAccelPlan("cuda", "cuda"),
                                      ending="..."),
        "retrying": Status.decoding("GPU decode failed, retrying", HwAccelPlan(None, None), ending="..."),
        "decoding failed": "GPU decoding failed (no decoder for the distorted video); decoding through FFmpeg "
                           "instead…",
        "gpu wait": Status(GPU_WAIT_MESSAGE, kind=GPU_WAIT),
        "gpu failed": Status("VMAF on the GPU failed (device lost); calculating it on the CPU…",
                             kind=GPU_VMAF_FAILED),
        "spaces and dots": "  Detecting black bars in distorted . …..  ",
        "only dots": "...",
        "gpu pass": Status("GPU metric 2/3: Butteraugli", kind=GPU_PASS),
        "vship": Status.decoding("Vship GPU (NVIDIA GeForce RTX 5090): calculating SSIMULACRA2",
                                 HwAccelPlan("cuda", None)),
    }
    kinds = {"": "step", STARTING: "starting", GPU_PASS: "gpu_pass", GPU_WAIT: "gpu_wait",
             GPU_VMAF_FAILED: "gpu_vmaf_failed"}

    def status(message: str) -> dict:
        brief = getattr(message, "brief", message)
        return {"text": str(message), "kind": kinds[getattr(message, "kind", "")],
                "brief": "" if brief == str(message) else brief}

    def task(backend: str, keys: tuple[str, ...], state: str, *, lane: str | None = None, current: int = 0,
             total: int = 0, fps: float = 0.0, done: tuple[str, ...] = (), cpu: tuple[str, ...] = (),
             waiting: str | None = None, step: str = "none", passes: tuple[tuple[str, ...], ...] | None = None,
             phase: tuple[int, int, int] | None = None) -> dict:
        """A task's snapshot, as JobRun.task_snapshots makes it."""
        return {"backend": backend, "metric_keys": keys,
                "lane": lane or ("gpu" if backend in ("vmaf_gpu", "perceptual") else "cpu"),
                "passes": passes or (keys,), "cpu_keys": cpu, "done_keys": done, "current": current, "total": total,
                "fps": fps, "state": state, "waiting_for": waiting, "step": steps[step], "decode": "", "phase": phase}

    every = ("vmaf", "vmaf_neg", "psnr", "ssim", "xpsnr")
    cpu, gpu = ("psnr", "ssim", "xpsnr"), ("vmaf", "vmaf_neg")
    perceptual = ("ssimulacra2", "butteraugli", "cvvdp")
    one_each = (("ssimulacra2",), ("butteraugli",), ("cvvdp",))
    lines = {
        "one task starting": [task("ffmpeg", every, "starting")],
        "one task, FFmpeg starting": [task("ffmpeg", every, "starting", step="ffmpeg starting")],
        "one task, a step": [task("ffmpeg", cpu, "starting", step="in the app")],
        "one task, a step in the test video": [task("ffmpeg", cpu, "starting", step="spaces and dots")],
        "one task running": [task("ffmpeg", every, "running", current=120, total=515, fps=9.75)],
        "one task running fast": [task("ffmpeg", cpu, "running", current=5, total=100000, fps=120.04999)],
        "no speed yet": [task("ffmpeg", every, "running", current=3, total=515)],
        "no frame count": [task("ffmpeg", every, "running", current=3, fps=24.0)],
        "almost done": [task("ffmpeg", ("psnr",), "running", current=9999, total=10000, fps=0.05)],
        "past the end": [task("ffmpeg", ("psnr",), "running", current=10001, total=10000, fps=50.0)],
        "paused": [task("ffmpeg", every, "running", current=120, total=515, fps=9.75)],
        "waiting for a CPU slot": [task("ffmpeg", cpu, "waiting", waiting="CPU"),
                                   task("vmaf_gpu", gpu, "starting", step="ffmpeg starting")],
        "both running": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0),
                         task("vmaf_gpu", gpu, "running", current=75, total=100, fps=480.5)],
        "waiting for the GPU": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0),
                                task("vmaf_gpu", gpu, "waiting", waiting="GPU", step="gpu wait")],
        "GPU waits, one task": [task("vmaf_gpu", gpu, "starting", step="gpu wait")],
        "the CPU took over": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0),
                              task("vmaf_gpu", gpu, "starting", cpu=gpu, step="gpu failed")],
        "the CPU took over, running": [task("ffmpeg", cpu, "done", done=cpu),
                                       task("vmaf_gpu", gpu, "running", current=10, total=100, fps=8.0, cpu=gpu)],
        "the CPU took over, done": [task("ffmpeg", cpu, "done", done=cpu),
                                    task("vmaf_gpu", gpu, "done", done=gpu, cpu=gpu)],
        "CPU done, GPU running": [task("ffmpeg", cpu, "done", done=cpu),
                                  task("vmaf_gpu", gpu, "running", current=30, total=100, fps=60.0)],
        "CPU failed, GPU running": [task("ffmpeg", cpu, "failed"),
                                    task("vmaf_gpu", gpu, "running", current=30, total=100, fps=60.0)],
        "both done": [task("ffmpeg", cpu, "done", done=cpu), task("vmaf_gpu", gpu, "done", done=gpu)],
        "a metric missing": [task("ffmpeg", cpu, "done", done=("psnr", "ssim")),
                             task("vmaf_gpu", gpu, "starting", step="retrying")],
        "GPU failed": [task("ffmpeg", cpu, "running", current=1, total=2, fps=1.0), task("vmaf_gpu", gpu, "failed")],
        "this video holds the GPU": [task("vmaf_gpu", ("vmaf",), "running", current=10, total=100, fps=8.0,
                                          cpu=("vmaf",)),
                                     task("perceptual", ("ssimulacra2",), "waiting", lane="gpu", waiting="GPU",
                                          step="gpu wait")],
        "a step after decoding failed": [task("ffmpeg", every, "starting", step="decoding failed")],
        # Vship's task: its passes, one at a time, sharing one progress count.
        "Vship's first pass": [task("perceptual", perceptual, "running", current=50, total=300, fps=30.0,
                                    passes=one_each, phase=(1, 3, 0), step="gpu pass")],
        "Vship's second pass": [task("perceptual", perceptual, "running", current=150, total=300, fps=25.0,
                                     passes=one_each, phase=(2, 3, 100))],
        "Vship's last pass, no speed yet": [task("perceptual", perceptual, "running", current=200, total=300,
                                                 passes=one_each, phase=(3, 3, 200))],
        "Vship starting": [task("perceptual", perceptual, "starting", passes=one_each, step="vship")],
        "Vship before its first pass": [task("perceptual", perceptual, "running", current=0, total=300,
                                             passes=one_each)],
        "Vship waiting": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0),
                          task("perceptual", perceptual, "waiting", waiting="GPU", passes=one_each,
                               step="gpu wait")],
        "Vship done, one failed": [task("perceptual", perceptual, "done", done=("ssimulacra2", "cvvdp"),
                                        passes=one_each)],
        "Vship failed": [task("ffmpeg", cpu, "done", done=cpu),
                         task("perceptual", perceptual, "failed", passes=one_each)],
        "a shared pass, then a retry": [task("perceptual", ("ssimulacra2", "butteraugli"), "running", current=150,
                                             total=200, fps=40.0,
                                             passes=(("ssimulacra2", "butteraugli"), ("butteraugli",)),
                                             phase=(2, 2, 100))],
        "one pass with all three": [task("perceptual", perceptual, "running", current=10, total=100, fps=12.5,
                                         passes=(perceptual,), phase=(1, 1, 0))],
        "the CPU took over from Vship": [task("perceptual", perceptual, "running", current=5, total=100, fps=2.0,
                                              cpu=("ssimulacra2", "butteraugli"), passes=(("cvvdp",),),
                                              done=("cvvdp",))],
        "Vship and the CPU's perceptual task": [
            task("vmaf_gpu", gpu, "done", done=gpu),
            task("perceptual", ("cvvdp",), "running", current=40, total=100, fps=20.0, passes=(("cvvdp",),),
                 phase=(1, 1, 0)),
            task("perceptual_cpu", ("ssimulacra2", "butteraugli"), "running", current=3, total=100, fps=0.5)],
    }
    return {
        "lines": [{"name": name, "paused": name == "paused",
                   "tasks": [{**{key: value for key, value in snapshot.items()
                                 if key not in ("decode", "step", "waiting_for", "fps")},
                              "fps": exact(snapshot["fps"]), "waiting_for": snapshot["waiting_for"] or "",
                              "step": status(snapshot["step"])} for snapshot in snapshots],
                   "line": run_line.metric_line(snapshots, name == "paused")[0]}
                  for name, snapshots in lines.items()],
        "steps": [{"status": status(message), "text": run_line.step_text(message)} for message in steps.values()],
        "numbers": [{"numbers": numbers, "text": run_line.numbers_text(numbers)}
                    for numbers in ([3], [1, 2], [1, 3], [2, 1, 3], [5, 6, 7, 9], [4, 5, 6])],
    }


# ------------------------------------------------------------- command line

#: Command lines for argument_cases: right ones, and each way of being wrong.
ARGUMENT_CASES = [
    [], ["-h"], ["--help"], ["--version"], ["--vers"], ["--version", "compare"], ["frobnicate"], ["-x"],
    ["devices"], ["devices", "extra"], ["devices", "-h"],
    ["compare"], ["compare", "-h"], ["compare", "ref.mkv"], ["compare", "ref.mkv", "a.mkv"],
    ["compare", "ref.mkv", "a.mkv", "b.mkv", "-m", "vmaf,psnr"],
    ["compare", "ref.mkv", "a.mkv", "b.mkv", "-m", " SSIM, vmaf ,ssim", "--model", "4k", "--black-bars", "none",
     "--scale-to", "reference", "--scaler", "lanczos", "--duration", "12.5", "--colour", "pq-bt2020",
     "--cvvdp-display", "6-inch phone", "--cvvdp-display", "30-inch 4K monitor, office", "--subsample", "3",
     "--threads", "8", "--vmaf-on", "cpu", "--ssimulacra2-on", "gpu", "--butteraugli-on", "cpu", "--cpu",
     "--no-gpu-decode", "--parallel", "2", "--recalculate", "--csv", "out", "--json", "-", "-q", "-v"],
    ["compare", "ref.mkv", "a.mkv", "-m", "all"],
    ["compare", "ref.mkv", "a.mkv", "-m", "foo,vmaf,bar"],
    ["compare", "ref.mkv", "a.mkv", "-m", ","],
    ["compare", "ref.mkv", "a.mkv", "-mvmaf,ssim"],
    ["compare", "ref.mkv", "a.mkv", "-m=vmaf"],
    ["compare", "ref.mkv", "a.mkv", "--metrics=psnr"],
    ["compare", "ref.mkv", "a.mkv", "--met", "psnr"],
    ["compare", "ref.mkv", "a.mkv", "--s", "2"],
    ["compare", "ref.mkv", "a.mkv", "--colo", "bt709"],
    ["compare", "ref.mkv", "a.mkv", "--color", "bt709"],
    ["compare", "ref.mkv", "a.mkv", "--colour", "rec2020"],
    ["compare", "ref.mkv", "a.mkv", "--vmaf-on", "GPU"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "33"],
    ["compare", "ref.mkv", "a.mkv", "--frames=0"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "-5"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "x"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "1_000"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "5", "--frames", "6"],
    ["compare", "ref.mkv", "a.mkv", "--duration", "nan"],
    ["compare", "ref.mkv", "a.mkv", "--duration", "inf"],
    ["compare", "ref.mkv", "a.mkv", "--duration", " 1e-3 "],
    ["compare", "ref.mkv", "a.mkv", "--duration", "2", "--frames", "3"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "3", "--duration", "2"],
    ["compare", "ref.mkv", "a.mkv", "--duration", "2", "--frames", "x"],
    ["compare", "ref.mkv", "a.mkv", "--frames"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "--cpu"],
    ["compare", "ref.mkv", "a.mkv", "-m"],
    ["compare", "ref.mkv", "a.mkv", "--subsample", "0"],
    ["compare", "ref.mkv", "a.mkv", "--threads", "+4"],
    ["compare", "ref.mkv", "a.mkv", "--parallel", "3"],
    ["compare", "ref.mkv", "a.mkv", "--parallel", "02"],
    ["compare", "ref.mkv", "a.mkv", "--parallel", "two"],
    ["compare", "ref.mkv", "a.mkv", "-qv"],
    ["compare", "ref.mkv", "a.mkv", "-qmvmaf"],
    ["compare", "ref.mkv", "a.mkv", "-qm", "psnr"],
    ["compare", "ref.mkv", "a.mkv", "-qx"],
    ["compare", "ref.mkv", "a.mkv", "-q=v"],
    ["compare", "ref.mkv", "a.mkv", "--cpu=yes"],
    ["compare", "ref.mkv", "a.mkv", "--no-gpu"],
    ["compare", "ref.mkv", "a.mkv", "--bogus"],
    ["compare", "ref.mkv", "a.mkv", "--version"],
    ["compare", "-m", "vmaf", "ref.mkv", "a.mkv", "b.mkv"],
    ["compare", "ref.mkv", "--frames", "5", "a.mkv", "b.mkv"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "5", "b.mkv"],
    ["compare", "--", "-ref.mkv", "-a.mkv"],
    ["compare", "ref.mkv", "--", "a.mkv", "--cpu"],
    ["compare", "ref.mkv", "a.mkv", "--", "--", "b.mkv"],
    ["compare", "-5.mkv", "a.mkv"],
    ["compare", "ref.mkv", "a.mkv", "-"],
    ["compare", "", "a.mkv"],
    ["compare", "ref.mkv", "a.mkv", "--json", "-"],
    ["compare", "ref.mkv", "a.mkv", "-h", "--frames", "x"],
    ["compare", "ref.mkv", "a.mkv", "--frames", "x", "-h"],
    ["compare", "ref.mkv", "a.mkv", "--he"],
    ["compare", "ref.mkv", "a.mkv", "--model", "my model.json"],
    ["compare", "ref.mkv", "a.mkv", "--cvvdp-display", "-phone"],
    ["compare", "ref.mkv", "a.mkv", "--cvvdp-display", "-6-inch phone"],
    ["compare", "ref.mkv", "a.mkv", "--scaler", "it's"],
    ["compare", "ref.mkv", "a.mkv", "--scaler", "tab\there\x01"],
    ["compare", "réf.mkv", "ä.mkv", "--cvvdp-display", "écran"],
]


def _plain(value):
    """A parsed value as JSON holds it: paths as text, numbers exactly."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return exact(value)
    return value


def argument_cases() -> list[dict]:
    """What the command line does with each of ARGUMENT_CASES before it runs
    a command (vmaf_app/cli.py: its argparse parser, and main() for no
    command): the values it reads, or what it prints and its exit code.
    Help is as wide as where the output is not a terminal (80 columns). The
    app's name and version is written as {version}, as the C++ app has its
    own."""
    import contextlib
    import io

    from vmaf_app import APP_NAME, __version__
    from vmaf_app.cli import build_parser

    version = f"{APP_NAME} {__version__}"
    saved = {name: os.environ.get(name) for name in ("COLUMNS", "NO_COLOR")}
    os.environ.update(COLUMNS="80", NO_COLOR="1")
    cases = []
    try:
        for argv in ARGUMENT_CASES:
            parser = build_parser()
            out, err = io.StringIO(), io.StringIO()
            values = None
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    args = parser.parse_args(argv)
                except SystemExit as stop:
                    code = stop.code
                else:
                    code = 0
                    if args.command is None:  # main() prints the help
                        parser.print_help()
                        code = 2
                    else:
                        values = {key: _plain(value) for key, value in vars(args).items()}
            cases.append({"argv": argv, "exit": code, "stdout": out.getvalue().replace(version, "{version}"),
                          "stderr": err.getvalue().replace(version, "{version}"), "values": values})
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return cases


#: The files of pattern_cases' folder ("empty/" is a folder), and the
#: patterns: lists of TEST arguments, with ref.mkv the reference. "windows"
#: marks a case that only Windows answers so (names ignore case there).
PATTERN_FILES = ["ref.mkv", "a.mkv", "b.mkv", "c.mp4", ".hidden.mkv", "x[1].mkv", "notes.txt",
                 "enc/d.mkv", "enc/e.mkv", "enc a/f.mkv", "empty/"]
PATTERN_CASES = [
    {"tests": ["*.mkv"]}, {"tests": ["a.mkv", "*.mkv"]}, {"tests": ["*.mkv", "a.mkv"]},
    {"tests": ["ref.mkv"]}, {"tests": ["missing.mkv"]}, {"tests": ["nothing*.mkv"]},
    {"tests": ["?.mkv"]}, {"tests": ["[ab].mkv"]}, {"tests": ["[!a].mkv"]}, {"tests": ["x[1].mkv"]},
    {"tests": ["enc*/*.mkv"]}, {"tests": ["*"]}, {"tests": [".*.mkv"]},
    {"tests": ["enc/../a.mkv", "a.mkv"]}, {"tests": ["empty/*"]},
    {"tests": ["*.MKV"], "windows": True},
]


def pattern_cases() -> dict:
    """The test videos expand_tests finds for each of PATTERN_CASES, in a
    folder holding PATTERN_FILES: paths within the folder, with "/"."""
    from vmaf_app.cli import expand_tests

    cases = []
    with tempfile.TemporaryDirectory() as made:
        folder = Path(made)
        for name in PATTERN_FILES:
            if name.endswith("/"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            else:
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(b"")
        for case in PATTERN_CASES:
            found = expand_tests([str(folder / test) for test in case["tests"]], folder / "ref.mkv")
            cases.append({**case, "found": [path.relative_to(folder).as_posix() for path in found]})
    return {"files": PATTERN_FILES, "cases": cases}


def json_cases() -> list[dict]:
    """json.dumps(value, indent=2), as --json writes the results, over values
    that show each of its rules: escapes, characters past ASCII, bytes that
    are not UTF-8 (a POSIX file name can hold any), numbers, empty and
    nested containers. Each value is given as JSON text, keeping its keys'
    order; in it {"$float": hex} is a number (JSON has no NaN) and
    {"$bytes": hex} a string of those bytes, read as Python reads a file
    name's."""
    def decoded(value):
        if isinstance(value, dict):
            if list(value) == ["$float"]:
                return float.fromhex(value["$float"])
            if list(value) == ["$bytes"]:
                return bytes.fromhex(value["$bytes"]).decode("utf-8", "surrogateescape")
            return {key: decoded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decoded(item) for item in value]
        return value

    floats = (0.1, 1e16, 1e-5, 123.456, 1e22, -0.0, 5e-324, math.inf, -math.inf, math.nan, 2.5, 100.0)
    values = [
        {"plain": "text", "quotes": 'a "b" \\ c', "controls": "\t\n\r\b\f\x01\x1f\x7f", "accents": "réf ä",
         "cjk": "動画", "emoji": "🎬"},
        # Not UTF-8: a lone byte, a lone continuation, overlong forms, a
        # surrogate, past U+10FFFF, a cut sequence; then good ones.
        {"path": {"$bytes": "66ff80c0afe08080eda080f4908080e2826261c3a9f09f8eac"}},
        [1, -2, 0, 12345678901234],
        [{"$float": exact(number)} for number in floats],
        {"empty object": {}, "empty list": [], "nested": {"a": [{"b": None, "c": True, "d": False}]}},
        [], {}, "a string alone", None,
    ]
    return [{"value": json.dumps(value, ensure_ascii=False), "text": json.dumps(decoded(value), indent=2)}
            for value in values]


# --------------------------------------------------------------- libvmaf

def predict_cases() -> list[dict]:
    """VMAF's scores predicted by the bundled libvmaf from features, as the
    Python app has it predict them from the Vulkan engine's
    (vmaf_vulkan.predict): for VMAF, its 4K model and VMAF NEG, over frames
    with features across their ranges and at their ends, so that some
    scores are clipped at 0 and one at 100. The features are given under
    the names libvmaf's extractors use."""
    import numpy as np

    from vmaf_app.core import vmaf_vulkan

    values = random.Random(11)
    frames = np.array([0, 1, 2, 5, 9, 10, 11, 1000])
    cases = []
    for version in ("vmaf_v0.6.1", "vmaf_4k_v0.6.1", "vmaf_v0.6.1neg"):
        rows = np.zeros((len(frames), vmaf_vulkan.FEATURE_COUNT))
        for row in rows[:-2]:
            row[0:4] = [values.uniform(0.0, 1.0) for _ in range(4)]  # VIF's scales
            row[4] = values.uniform(0.3, 1.0)  # ADM2
            row[5] = values.uniform(0.0, 40.0)  # motion2
            row[6:10] = [values.uniform(0.0, 1.0) for _ in range(4)]  # NEG's VIF
            row[10] = values.uniform(0.3, 1.0)  # NEG's ADM2
            row[11] = values.uniform(0.0, 40.0)  # motion
        rows[-2] = 0.0  # every feature at its lowest
        rows[-1, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]] = 1.0  # VIF and ADM at their highest, no motion
        scores = vmaf_vulkan.predict(version, frames, rows)
        columns = vmaf_vulkan._MODEL_FEATURES[version in vmaf_vulkan._NEG_MODELS]
        cases.append({"version": version, "frames": [int(frame) for frame in frames],
                      "features": {name: [exact(row[column]) for row in rows] for name, column in columns.items()},
                      "scores": [exact(score) for score in scores]})
    return cases


def vulkan_cases() -> dict:
    """The Vulkan engine's features and VMAF's scores, as the Python app's
    VulkanScorer gives them on this PC's GPU, for the probe's frames
    (vmaf_vulkan.probe_frames, kept as each frame's SHA-256): VMAF and NEG
    at 8 bits for every frame and at 10 bits for every second frame, and the
    4K models at an odd size. With them, the SHA-256 of the sums the probe
    expects. The engine gives the same numbers on every GPU its probe
    accepts, so the file is the same whichever PC makes it, but making it
    needs such a GPU."""
    import hashlib

    from vmaf_app.core import vmaf_vulkan

    available, device, text = vmaf_vulkan.probe()
    if not available:
        raise SystemExit(f"vulkan.json needs a GPU that Vulkan scores VMAF on: {text}")
    cases = []
    for width, height, bits, count, n_subsample, models in (
        (320, 192, 8, 3, 1, {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}),
        (320, 192, 10, 5, 2, {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}),
        (321, 193, 8, 4, 1, {"vmaf_4k": "vmaf_4k_v0.6.1", "vmaf_4k_neg": "vmaf_4k_v0.6.1neg"}),
    ):
        probe_size = vmaf_vulkan._PROBE_SIZE
        vmaf_vulkan._PROBE_SIZE = (width, height)  # the size probe_frames makes
        try:
            reference, distorted = vmaf_vulkan.probe_frames(bits, count)
        finally:
            vmaf_vulkan._PROBE_SIZE = probe_size
        scorer = vmaf_vulkan.VulkanScorer(width, height, bits, models, n_subsample, device=device)
        try:
            for ref, dis in zip(reference, distorted, strict=True):
                scorer.add(ref, dis)
            frames, scores = scorer.finish()
            _frames, rows = scorer.features()
        finally:
            scorer.close()
        cases.append({"width": width, "height": height, "bits": bits, "count": count, "n_subsample": n_subsample,
                      "models": models,
                      "reference": [hashlib.sha256(frame).hexdigest() for frame in reference],
                      "distorted": [hashlib.sha256(frame).hexdigest() for frame in distorted],
                      "frames": [int(frame) for frame in frames],
                      "rows": [[exact(value) for value in row] for row in rows],
                      "scores": {name: [exact(score) for score in values] for name, values in scores.items()}})
    return {"probe_sums": list(vmaf_vulkan._PROBE_SUMS), "cases": cases}


def decoding_cases() -> dict:
    """How the app plans to decode a video itself (gpu_frames.plan_decode),
    over sizes, crops, depths and codecs, and what it leaves to FFmpeg;
    FFmpeg's command that copies a video's packets out of its container for
    the decoders (gpu_frames._PacketReader), its program, video and
    listing's pipe written as <ffmpeg>, <video> and <listing>; and the
    decoders the videos of a run are decoded with (vmaf_runner._decoders_here),
    with VML_SOFTWARE_DECODE unset and set to ffmpeg."""
    from vmaf_app.core import gpu_frames, vmaf_runner
    from vmaf_app.core.gpu import HwAccelPlan
    from vmaf_app.core.models import CropBox, VideoInfo

    def video(**fields) -> VideoInfo:
        base = {"path": Path("video.mkv"), "width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0,
                "nb_frames": 240, "codec_name": "hevc", "pix_fmt": "yuv420p10le"}
        base.update(fields)
        return VideoInfo(**base)

    plans = []
    for name, info, crop, options in (
            ("whole", video(), None, {}),
            ("a crop rounded as FFmpeg's crop filter rounds it", video(), CropBox(1917, 1077, 3, 1), {}),
            ("a crop that keeps the whole picture", video(), CropBox(1920, 1080, 0, 0), {}),
            ("a letterbox scaled, luma only", video(width=3840, height=2160), CropBox(3840, 1608, 0, 276),
             {"shift": 6, "luma_only": True, "size": (1920, 804), "algorithm": "lanczos"}),
            ("8-bit, with no shift", video(codec_name="h264", pix_fmt="yuv420p"), None, {"shift": 6}),
            ("full range, codec in capitals", video(codec_name="H264", pix_fmt="yuvj420p"), None, {}),
            ("10-bit big-endian", video(codec_name="av1", pix_fmt="yuv420p10be"), None, {"shift": 6}),
            ("an algorithm the decoders do not have", video(codec_name="vvc"), None,
             {"size": (1280, 720), "algorithm": "area"}),
            ("scaled up, odd size", video(codec_name="mpeg2video", pix_fmt="yuv420p", width=720, height=576), None,
             {"size": (1023, 767), "algorithm": "spline"}),
            ("ffv1, bilinear", video(codec_name="ffv1", pix_fmt="yuv420p"), None, {"algorithm": "bilinear"}),
            ("another codec", video(codec_name="mpeg4"), None, {}),
            ("no codec", video(codec_name=""), None, {}),
            ("4:2:2", video(pix_fmt="yuv422p10le"), None, {}),
            ("12-bit", video(pix_fmt="yuv420p12le"), None, {}),
            ("no pixel format", video(pix_fmt=""), None, {}),
            ("no size", video(width=0), None, {}),
            ("an odd width", video(width=1919), None, {}),
            ("an odd height", video(height=1079), None, {}),
            ("a crop past the picture", video(), CropBox(1920, 1080, 2, 0), {}),
            ("a crop one column wide", video(), CropBox(1, 1080, 0, 0), {}),
            ("scaled to nothing", video(), None, {"size": (0, 720)})):
        made = {"name": name, "codec_name": info.codec_name, "pix_fmt": info.pix_fmt, "width": info.width,
                "height": info.height, "crop": None if crop is None else [crop.w, crop.h, crop.x, crop.y],
                "shift": options.get("shift", 0), "luma_only": options.get("luma_only", False),
                "size": list(options["size"]) if "size" in options else None,
                "algorithm": options.get("algorithm", "bicubic")}
        try:
            plan = gpu_frames.plan_decode(info, crop, **options)
            made["plan"] = {"codec": plan.codec, "bit_depth": plan.bit_depth, "width": plan.width,
                            "height": plan.height, "crop": [plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h],
                            "shift": plan.shift, "luma_only": plan.luma_only, "out": [plan.out_w, plan.out_h],
                            "scaler": plan.scaler, "scaled": plan.scaled, "frame_bytes": plan.frame_bytes}
        except gpu_frames.GpuDecodeUnavailableError as error:
            made["error"] = str(error)
        plans.append(made)

    commands = {}
    for codec in gpu_frames._CODECS:
        reader = gpu_frames._PacketReader(Path("video.mkv"), codec)
        try:
            command = list(reader.command)
        finally:
            reader.close()
        command[0] = "<ffmpeg>"
        commands[codec] = ["<video>" if part == str(Path("video.mkv").resolve()) else
                           "<listing>" if part == reader.pipe_path else part for part in command]

    decoders = []
    real_software = os.environ.pop(gpu_frames.SOFTWARE_VARIABLE, None)
    try:
        for software in ("", "ffmpeg"):
            os.environ[gpu_frames.SOFTWARE_VARIABLE] = software
            bundled = gpu_frames.software_bundled()
            for source in (None, "cuda", "qsv", "d3d11va", "vulkan"):
                for test in (None, "cuda", "qsv", "d3d11va"):
                    here = vmaf_runner._decoders_here(HwAccelPlan(source, test))
                    decoders.append({"software_decode": software, "bundled": bundled, "source": source,
                                     "test": test, "decoders": None if here is None else list(here)})
    finally:
        os.environ.pop(gpu_frames.SOFTWARE_VARIABLE, None)
        if real_software is not None:
            os.environ[gpu_frames.SOFTWARE_VARIABLE] = real_software
    return {"plans": plans, "packet_commands": commands, "decoders": decoders}


def vship_cases() -> dict:
    """Vship's view of a video (perceptual_vship.py), with colour.py's reading
    of its tags: the layouts FFmpeg sends it in (_image_format,
    _passthrough_format) and their frame layouts, the colour spaces Vship is
    given (_vship_colorspace), the filter chains and output arguments
    (_filter_chain, _piped_as), the sizes compared at (_scaled_sizes), and
    the GPUs and sizes a metric is refused at (size_refusal)."""
    from fractions import Fraction

    from vmaf_app.core import perceptual_vship as vship
    from vmaf_app.core.models import CropBox, GpuVendor, ScaleDirection, VideoInfo

    def video(**fields) -> VideoInfo:
        base = {"path": Path("video.mkv"), "width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0,
                "nb_frames": 240, "codec_name": "hevc", "pix_fmt": "yuv420p"}
        base.update(fields)
        return VideoInfo(**base)

    def described(info: VideoInfo) -> dict:
        return {"width": info.width, "height": info.height, "pix_fmt": info.pix_fmt,
                "color_range": info.color_range, "color_space": info.color_space,
                "color_transfer": info.color_transfer, "color_primaries": info.color_primaries,
                "chroma_location": info.chroma_location}

    def image_json(image) -> dict:
        return {"pixel_format": image.pixel_format, "family": image.family, "sample": image.sample,
                "subw": image.subw, "subh": image.subh, "full_range": image.full_range,
                "plane_order": list(image.plane_order)}

    formats = []
    for pix_fmt in ("yuv420p", "yuv420p10le", "yuv420p10be", "yuvj420p", "yuva420p", "yuva444p12le", "yuv410p",
                    "yuv411p", "yuv422p12le", "yuv440p", "yuv444p16le", "yuv420p9le", "yuv420p14le", "yuv420p10",
                    "nv12", "nv21", "p010le", "p016be", "nv16", "p210le", "nv24", "p416le", "gray", "gray10le",
                    "gray12be", "gbrp", "gbrap", "gbrp10le", "gbrp16be", "rgb24", "bgra", "0bgr", "rgb48le",
                    "bgra64be", "xyz12le", "", " YUV420P "):
        for version in (None, "5.1.1", "5.1.2", "5.2", "6.0.0", "5.1"):
            made = {"pix_fmt": pix_fmt, "version": version}
            try:
                image = vship._image_format(video(pix_fmt=pix_fmt), version)
                made["image"] = image_json(image)
                for width, height in ((1920, 1080), (321, 193), (64, 32)):
                    total, sizes, strides, _order = image.frame_layout(width, height)
                    made.setdefault("layouts", []).append(
                        {"width": width, "height": height, "frame_bytes": total, "plane_bytes": list(sizes),
                         "strides": list(strides)})
            except vship.VshipUnavailableError as error:
                made["error"] = str(error)
            formats.append(made)

    passthrough = []
    for pix_fmt, color_range, hwaccel in (
            ("yuv420p", "", "cuda"), ("yuv420p", "pc", "cuda"), ("yuvj420p", "", "qsv"), ("yuv420p10le", "tv", "cuda"),
            ("yuv420p10le", "", "d3d11va"), ("yuv420p10le", "pc", "cuda"), ("yuv420p10le", "jpeg", "cuda"),
            ("yuv420p10le", "tv", None), ("yuv420p12le", "tv", "cuda"), ("yuv422p10le", "tv", "cuda"),
            ("yuv444p", "", "cuda"), ("xyz12le", "", "cuda"), ("xyz12le", "", None)):
        made = {"pix_fmt": pix_fmt, "color_range": color_range, "hwaccel": hwaccel}
        try:
            found = vship._passthrough_format(video(pix_fmt=pix_fmt, color_range=color_range), hwaccel)
            made["passthrough"] = None if found is None else {"image": image_json(found[0]), "piped": found[1]}
        except vship.VshipUnavailableError as error:
            made["error"] = str(error)
        passthrough.append(made)

    colours = []
    for name, info in (
            ("untagged 1080p", video()),
            ("untagged 576p", video(width=720, height=576)),
            ("untagged 651 lines", video(height=651)),
            ("BT.2020 matrix only", video(color_space="bt2020nc")),
            ("ICtCp matrix only", video(color_space="ictcp")),
            ("HDR10", video(pix_fmt="yuv420p10le", color_space="bt2020nc", color_transfer="smpte2084",
                            color_primaries="bt2020", color_range="tv", chroma_location="topleft")),
            ("HLG", video(pix_fmt="yuv420p10le", color_space="bt2020nc", color_transfer="arib-std-b67",
                          color_primaries="bt2020")),
            ("BT.709 full range", video(color_space="bt709", color_transfer="bt709", color_primaries="bt709",
                                        color_range="pc")),
            ("untagged yuvj", video(pix_fmt="yuvj420p")),
            ("yuvj tagged limited", video(pix_fmt="yuvj420p", color_range="tv")),
            ("BT.470BG tagged matrix", video(color_space="bt470bg")),
            ("SMPTE 170M", video(color_space="smpte170m", color_transfer="smpte170m", color_primaries="smpte170m")),
            ("upper-case tags", video(color_space="BT709", color_transfer="BT709", color_primaries="BT709")),
            ("unspecified tags", video(color_space="unknown", color_transfer="unspecified", color_primaries="reserved",
                                       color_range="unknown")),
            ("BT.2020 10-bit curve", video(color_transfer="bt2020-10", color_primaries="bt2020")),
            ("sRGB", video(color_transfer="iec61966-2-1")),
            ("P3", video(color_primaries="smpte432")),
            ("planar RGB untagged", video(pix_fmt="gbrp")),
            ("planar RGB tagged limited", video(pix_fmt="gbrp10le", color_range="tv")),
            ("4:2:2 centre siting", video(pix_fmt="yuv422p10le", chroma_location="center")),
            ("top siting", video(chroma_location="top")),
            ("top-left siting with a dash", video(chroma_location="top-left")),
            ("unspecified siting", video(chroma_location="unspecified")),
            ("bottom siting", video(chroma_location="bottom")),
            ("an unknown matrix", video(color_space="fcc")),
            ("an unknown transfer", video(color_transfer="log100")),
            ("an unknown primaries", video(color_primaries="film")),
            ("an unknown range", video(color_range="studio")),
            ("4:1:0", video(pix_fmt="yuv410p"))):
        made = {"name": name, "info": described(info)}
        try:
            image = vship._image_format(info, "5.1.2")
            space = vship._vship_colorspace(info, image, 1280, 720)
            made["colorspace"] = {
                "width": space.width, "height": space.height, "target_width": space.target_width,
                "target_height": space.target_height, "sample": space.sample, "range": space.range,
                "subsampling": [space.subsampling.subw, space.subsampling.subh],
                "chroma_location": space.chromaLocation, "family": space.colorFamily, "matrix": space.YUVMatrix,
                "transfer": space.transferFunction, "primaries": space.primaries,
                "crop": [space.crop.top, space.crop.bottom, space.crop.left, space.crop.right]}
        except vship.VshipUnavailableError as error:
            made["error"] = str(error)
        colours.append(made)

    chains = []
    for name, info, crop, target, pixel_format, step, algorithm, hwaccel, numbered_at in (
            ("as it is", video(), None, (1920, 1080), "yuv420p", 1, "bicubic", None, Fraction(24000, 1001)),
            ("every third, from the GPU, cropped and scaled", video(pix_fmt="yuv420p10le", width=3840, height=2160),
             CropBox(3840, 1608, 0, 276), (1920, 804), "p010le", 3, "lanczos", "cuda", Fraction(120)),
            ("a crop that keeps it all", video(), CropBox(1920, 1080, 0, 0), (1920, 1080), "yuv444p16le", 1,
             "bicubic", "qsv", Fraction(25)),
            ("paired by timestamp", video(), None, (1280, 720), "yuv420p", 2, "bilinear", None, None)):
        chains.append({"name": name, "info": described(info),
                       "crop": None if crop is None else [crop.w, crop.h, crop.x, crop.y],
                       "target": list(target), "pixel_format": pixel_format, "step": step, "algorithm": algorithm,
                       "hwaccel": hwaccel,
                       "numbered_at": None if numbered_at is None else [numbered_at.numerator,
                                                                        numbered_at.denominator],
                       "chain": vship._filter_chain(info, crop, target, pixel_format, step, algorithm, hwaccel,
                                                    numbered_at)})

    piped = {pixel_format: vship._piped_as(pixel_format)
             for pixel_format in ("yuv420p", "yuv420p10le", "yuv410p", "yuv440p", "yuv411p", "yuv444p16le",
                                  "yuv420p16le", "gbrp", "gbrp10le", "nv12", "p010le", "yuv422p12le", "yuv420p10be")}

    sizes = []
    for source_size, source_crop, test_size, test_crop in (
            ((1920, 1080), None, (1920, 1080), None),
            ((3840, 2160), None, (1920, 1080), None),
            ((3840, 2160), CropBox(3840, 1608, 0, 276), (1920, 804), None),
            ((1920, 1080), None, (1280, 720), CropBox(1280, 536, 0, 92))):
        for direction in (ScaleDirection.SOURCE_TO_DISTORTED, ScaleDirection.DISTORTED_TO_SOURCE):
            source, test = video(width=source_size[0], height=source_size[1]), video(width=test_size[0],
                                                                                    height=test_size[1])
            recipe = type("Recipe", (), {"scale_direction": direction})()
            found = vship._scaled_sizes(source, test, recipe, source_crop, test_crop)
            sizes.append({"source": list(source_size), "test": list(test_size),
                          "source_crop": None if source_crop is None else [source_crop.w, source_crop.h,
                                                                           source_crop.x, source_crop.y],
                          "test_crop": None if test_crop is None else [test_crop.w, test_crop.h, test_crop.x,
                                                                       test_crop.y],
                          "to_source": direction is ScaleDirection.DISTORTED_TO_SOURCE,
                          "sizes": [list(found[0]), list(found[1])]})

    refusals = []
    vendors = {"nvidia": GpuVendor.NVIDIA, "intel": GpuVendor.INTEL, "amd": GpuVendor.AMD, "none": GpuVendor.NONE,
               "": None}
    for backend, vendor in (("vulkan", "intel"), ("vulkan", ""), ("vulkan", "nvidia"), ("cuda", "nvidia"),
                            ("vulkan", "none")):
        device = vship.VshipDevice(backend, "a GPU", 0, "5.1.2", None, vendors[vendor])
        for key in ("cvvdp", "ssimulacra2"):
            for size, computed in (((1920, 1080), None), ((3840, 2160), None), ((1920, 1080), (3840, 2160)),
                                   ((1920, 1088), (1920, 1088)), ((3840, 2160), (1920, 1080))):
                refusals.append({"backend": backend, "vendor": vendor, "key": key, "size": list(size),
                                 "computed": None if computed is None else list(computed),
                                 "refusal": vship.size_refusal(device, key, size, computed)})
    return {"formats": formats, "passthrough": passthrough, "colours": colours, "chains": chains, "piped": piped,
            "sizes": sizes, "refusals": refusals}


if __name__ == "__main__":
    write("vship.json", vship_cases())
    write("decoding.json", decoding_cases())
    write("vulkan.json", vulkan_cases())
    write("predict.json", predict_cases())
    write("python_json.json", json_cases())
    write("cli.json", {"arguments": argument_cases(), "patterns": pattern_cases()})
    write("runs.json", run_cases())
    write("run_line.json", run_line_cases())
    write("decisions.json", decision_cases())
    write("number_format.json", number_format_cases())
    write("metrics.json", metric_cases())
    write("stats.json", stats_cases())
    write("probe.json", probe_cases())
    write("settings.json", settings_cases())
    write("versions.json", version_cases())
