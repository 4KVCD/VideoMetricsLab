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
(perceptual_vship.py, colour.py), CVVDP's displays and the pooling of its
seconds (cvvdp.py, perceptual_vship.py), the pairing of frames by
timestamp (frame_sync_by_time.py), SSIMULACRA2 and Butteraugli on the CPU
(perceptual_cpu.py, colour.py), the line that says how far a video is
(ui/run_line.py), its command line's reading of arguments and patterns and
its devices command (cli.py), Python's json.dumps, the saved scores' names,
files and rules (metric_cache.py), with a comparison folder of them it saves
in metric_cache/, and the window's text in each language (i18n.py). The
Vulkan engine's cases need a GPU that Vulkan scores VMAF on.

The ffprobe cases use ffprobe's real output on the golden clips
(cpp/tests/golden; golden.py clips makes them), kept in the file, so the
tests need neither the clips nor ffprobe.

The Python app is a checkout of its last release (the release/v2.1 branch),
in the folder VML_PYTHON_APP names; the window's text is this repository's
catalogs (resources/translations) as the Python app reads them. Run it,
with a Python that has the Python app's requirements (that checkout's
requirements.txt), from the repository's root after changing any of those,
and commit what it writes:

    set VML_PYTHON_APP=<the checkout>
    .venv\\Scripts\\python.exe cpp\\tests\\reference\\make_reference.py

Numbers are written with float.hex() ("0x1.7a00000000000p+6", "inf", "nan"),
so that they reach the C++ tests exactly. Inputs come from fixed seeds, so the
files only change when the code that makes them does.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
#: The Python app, whose answers these are: a checkout of its last release
#: (the release/v2.1 branch), in the folder VML_PYTHON_APP names.
PYTHON_APP = Path(os.environ.get("VML_PYTHON_APP") or "")
if not (PYTHON_APP / "vmaf_app" / "__init__.py").is_file():
    sys.exit("name a checkout of the Python app (the release/v2.1 branch) with VML_PYTHON_APP")
sys.path.insert(0, str(PYTHON_APP))  # for vmaf_app
sys.path.insert(0, str(REPOSITORY))  # first, for this repository's scripts

import struct
import subprocess
import tempfile
import warnings

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
    """The registry, format_value() of numbers and of no number, and
    best_and_worst() of the videos' scores."""
    probe_values = [None, math.nan, math.inf, -math.inf, 0.0, -0.0, 94.5, 0.98765, 45.351984, -1.25, 1e-9, 12345.678]
    # Ordered, reversed, with gaps, equal as shown at some precisions and not
    # at others, infinite, too few, and equal.
    score_lists = [
        [90.0, 95.0, None, 85.0], [1.2, 0.9, 2.0], [95.001, 94.999, 80.0], [90.001, 89.999],
        [90.0, None, math.nan], [math.inf, 40.0, 30.0, math.inf], [-math.inf, 1.0, 2.0], [0.0, -0.0],
        [0.5004, 0.5006, 0.5005, 0.9], [], [5.0], [7.0, 7.0, 7.0], [3.0, 1.0, 3.0, 1.0, 2.0],
    ]
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
            "best_and_worst": [{"scores": [exact(v) for v in scores], "marks": metric.best_and_worst(scores)}
                               for scores in score_lists],
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
    PC's cores. And what Settings.save writes for the settings read, where
    parallel_jobs is not the PC's default."""
    files = {
        "good": json.dumps({"ffmpeg_dir": "D:/tools/ffmpeg", "default_compute_xpsnr": False, "parallel_jobs": 2,
                            "gpu_backend": "vulkan", "hidden_metrics": ["psnr", "ssim"],
                            "cvvdp_presets": [{"name": "Mine", "settings": {"display": {"z": 1, "a": 2}}}],
                            "cvvdp_default_preset": "Mine", "window_width": 1600,
                            "remembered_reference": "D:/vidéos/référence 動画.mkv",
                            "remembered_tests": ["a.mkv", "b.mkv"]}),
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
                loaded = Settings.load()
                values = dict(vars(loaded))
                saved = None
                if text is None or '"parallel_jobs": 2' not in text:
                    values["parallel_jobs"] = None  # the PC's default
                else:
                    assert loaded.save() is None
                    saved = path.read_text(encoding="utf-8")
                values["cvvdp_presets"] = json.dumps(values["cvvdp_presets"], separators=(",", ":"))
                cases.append({"name": name, "file": text, "expected": values, "saved": saved})
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
    root = PYTHON_APP.resolve() / "vmaf_app"

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
    from vmaf_app.core.models import CropBox, ResampleTarget, ScaleDirection, VideoInfo, VmafOptions

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
    v1_model = PYTHON_APP.resolve() / "vmaf_app" / "models" / "vmaf_v1.0.16" / "vmaf_v1.0.16_3d0h.json"
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
    resample_cases = []
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
        # outputs' pipes written as <test pipe> and <source pipe>, and when
        # pairing by timestamp the listings' as <test pipe>-times and
        # <source pipe>-times.
        for pairing in ("", "timestamp"):
            os.environ[frame_sync.PAIRING_VARIABLE] = pairing
            attempt = SimpleNamespace(
                paired=False, _by_time=bool(pairing),
                distorted=SimpleNamespace(path="<test pipe>", listing_path="<test pipe>-times"),
                reference=SimpleNamespace(path="<source pipe>", listing_path="<source pipe>-times"))
            attempt._timed_output_args = lambda limit, attempt=attempt: vmaf_gpu.GpuAttempt._timed_output_args(
                attempt, limit)
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
                            gpu_cases.append({
                                "name": f"{pairing or 'frame number'}; {pair_name}; {crop_name}; {plan_name}; "
                                        f"{setting_name}",
                                "pairing": pairing, "pair": pair_name, "crop": crop_name, "plan": plan_name,
                                "setting": setting_name, "limit": exact(limit), "commands": commands})
        # A resolution test: one input, the source, split into an untouched
        # reference and a test video scaled down to the target and back up
        # (_build_resample_test_filtergraph and _build_resample_cmd). The
        # pair's source, crop and decode plan are its.
        targets = {"1080p": ResampleTarget(1920, "1080p"), "720p": ResampleTarget(1280, "720p"),
                   "480p": ResampleTarget(854, "480p")}
        for pairing in ("", "timestamp"):
            os.environ[frame_sync.PAIRING_VARIABLE] = pairing
            for pair_name, (source, _test) in pairs.items():
                for crop_name, (source_crop, _test_crop) in crops.items():
                    if crop_name == "letterbox" and pair_name != "4k against 1080p":
                        continue
                    for plan_name in ("software", "cuda", "qsv"):
                        hwaccel = plans[plan_name].source
                        for target_name, target in targets.items():
                            for choice_name in ("vmaf", "vmaf, v1", "all", "xpsnr"):
                                for setting_name in ("plain", "subsample, lanczos, upscale", "limit"):
                                    if pairing and (plan_name != "cuda" or target_name != "720p"):
                                        continue  # the other way of pairing changes only the frame filters
                                    options = VmafOptions(**choices[choice_name], **settings[setting_name],
                                                          resample_test=target)
                                    xpsnr_log = Path("xpsnr_log.txt") if options.compute_xpsnr else None
                                    model = "version=vmaf_v0.6.1" if options.compute_vmaf else ""
                                    graph = vmaf_runner._build_resample_test_filtergraph(
                                        source, options, source_crop, hwaccel, Path("vmaf_log.json"), model=model,
                                        xpsnr_log_path=xpsnr_log)
                                    command = vmaf_runner._build_resample_cmd(source.path, graph, hwaccel,
                                                                              options.duration_limit)
                                    command[0] = "<ffmpeg>"
                                    command = ["<source>" if part == str(source.path.resolve()) else part
                                               for part in command]
                                    resample_cases.append({
                                        "name": f"{pairing or 'frame number'}; {pair_name}; {crop_name}; "
                                                f"{plan_name}; {target_name}; {choice_name}; {setting_name}",
                                        "pairing": pairing, "pair": pair_name, "crop": crop_name, "plan": plan_name,
                                        "target": target_name, "choice": choice_name, "setting": setting_name,
                                        "model": model, "cores": os.cpu_count(), "command": command})
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
    return {"commands": cases, "gpu_commands": gpu_cases, "resample_commands": resample_cases, "logs": parsed,
            "log_fps": exact(24000 / 1001)}


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
        "black bars": "Detecting black bars in source and distorted...",
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
             phase: tuple[int, int, int] | None = None, decode: HwAccelPlan | None = None) -> dict:
        """A task's snapshot, as JobRun.task_snapshots makes it."""
        return {"backend": backend, "metric_keys": keys,
                "lane": lane or ("gpu" if backend in ("vmaf_gpu", "perceptual") else "cpu"),
                "passes": passes or (keys,), "cpu_keys": cpu, "done_keys": done, "current": current, "total": total,
                "fps": fps, "state": state, "waiting_for": waiting, "step": steps[step], "decode": decode or "",
                "phase": phase}

    def plan(decode) -> dict | None:
        """A decode plan as the C++ snapshot holds it."""
        return {"source": decode.source, "distorted": decode.distorted} if decode else None

    on_gpu, gpu_source, on_cpu = HwAccelPlan("cuda", "cuda"), HwAccelPlan("cuda", None), HwAccelPlan(None, None)

    every = ("vmaf", "vmaf_neg", "psnr", "ssim", "xpsnr")
    cpu, gpu = ("psnr", "ssim", "xpsnr"), ("vmaf", "vmaf_neg")
    perceptual = ("ssimulacra2", "butteraugli", "cvvdp")
    one_each = (("ssimulacra2",), ("butteraugli",), ("cvvdp",))
    lines = {
        "one task starting": [task("ffmpeg", every, "starting")],
        "one task, FFmpeg starting": [task("ffmpeg", every, "starting", step="ffmpeg starting", decode=gpu_source)],
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
        "both running": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0, decode=on_gpu),
                         task("vmaf_gpu", gpu, "running", current=75, total=100, fps=480.5, decode=on_gpu)],
        "waiting for the GPU": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0),
                                task("vmaf_gpu", gpu, "waiting", waiting="GPU", step="gpu wait")],
        "GPU waits, one task": [task("vmaf_gpu", gpu, "starting", step="gpu wait")],
        "the CPU took over": [task("ffmpeg", cpu, "running", current=50, total=100, fps=30.0, decode=gpu_source),
                              task("vmaf_gpu", gpu, "starting", cpu=gpu, step="gpu failed", decode=on_cpu)],
        "the CPU took over, running": [task("ffmpeg", cpu, "done", done=cpu),
                                       task("vmaf_gpu", gpu, "running", current=10, total=100, fps=8.0, cpu=gpu)],
        "the CPU took over, done": [task("ffmpeg", cpu, "done", done=cpu),
                                    task("vmaf_gpu", gpu, "done", done=gpu, cpu=gpu)],
        "CPU done, GPU running": [task("ffmpeg", cpu, "done", done=cpu, decode=on_gpu),
                                  task("vmaf_gpu", gpu, "running", current=30, total=100, fps=60.0,
                                       decode=gpu_source)],
        "CPU failed, GPU running": [task("ffmpeg", cpu, "failed"),
                                    task("vmaf_gpu", gpu, "running", current=30, total=100, fps=60.0)],
        "both done": [task("ffmpeg", cpu, "done", done=cpu, decode=on_gpu),
                      task("vmaf_gpu", gpu, "done", done=gpu, decode=gpu_source)],
        "a metric missing": [task("ffmpeg", cpu, "done", done=("psnr", "ssim")),
                             task("vmaf_gpu", gpu, "starting", step="retrying")],
        "GPU failed": [task("ffmpeg", cpu, "running", current=1, total=2, fps=1.0), task("vmaf_gpu", gpu, "failed")],
        "this video holds the GPU": [task("vmaf_gpu", ("vmaf",), "running", current=10, total=100, fps=8.0,
                                          cpu=("vmaf",)),
                                     task("perceptual", ("ssimulacra2",), "waiting", lane="gpu", waiting="GPU",
                                          step="gpu wait")],
        "a step after decoding failed": [task("ffmpeg", every, "starting", step="decoding failed")],
        # Both places waiting for the black bars they share: said once.
        "both detecting black bars": [task("ffmpeg", cpu, "starting", step="black bars"),
                                      task("vmaf_gpu", gpu, "starting", step="black bars")],
        "each on a step of its own": [task("ffmpeg", cpu, "starting", step="in the app"),
                                      task("vmaf_gpu", gpu, "starting", step="black bars")],
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
            task("vmaf_gpu", gpu, "done", done=gpu, decode=on_gpu),
            task("perceptual", ("cvvdp",), "running", current=40, total=100, fps=20.0, passes=(("cvvdp",),),
                 phase=(1, 1, 0), decode=on_gpu),
            task("perceptual_cpu", ("ssimulacra2", "butteraugli"), "running", current=3, total=100, fps=0.5,
                 decode=on_cpu)],
    }

    def line_case(name: str, snapshots: list[dict]) -> dict:
        """The line, its tooltip, and the decoder text the window shows with
        it (for a test video, and for a resolution test)."""
        parts, tooltip, plans = run_line.metric_line(snapshots, name == "paused")
        return {"name": name, "paused": name == "paused",
                "tasks": [{**{key: value for key, value in snapshot.items()
                              if key not in ("decode", "step", "waiting_for", "fps")},
                           "fps": exact(snapshot["fps"]), "waiting_for": snapshot["waiting_for"] or "",
                           "step": status(snapshot["step"]), "decode": plan(snapshot["decode"])}
                          for snapshot in snapshots],
                "line": parts, "tooltip": tooltip,
                "decoder": run_line.decoder_text(plans, source_only=False) if plans else "",
                "decoder_source_only": run_line.decoder_text(plans, source_only=True) if plans else ""}

    return {
        "lines": [line_case(name, snapshots) for name, snapshots in lines.items()],
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
    ["compare", "ref.mkv", "a.266", "--fps", "24000/1001"],
    ["compare", "ref.mkv", "a.266", "--fps", "23.976"],
    ["compare", "ref.mkv", "a.266", "--fps=60"],
    ["compare", "ref.mkv", "a.266", "--fps", " 30000/1001 "],
    ["compare", "ref.mkv", "a.266", "--fps", "1e2"],
    ["compare", "ref.mkv", "a.266", "--fps", "0120/5"],
    ["compare", "ref.mkv", "a.266", "--fps", "0"],
    ["compare", "ref.mkv", "a.266", "--fps", "0/5"],
    ["compare", "ref.mkv", "a.266", "--fps", "-24"],
    ["compare", "ref.mkv", "a.266", "--fps", "inf"],
    ["compare", "ref.mkv", "a.266", "--fps", "nan"],
    ["compare", "ref.mkv", "a.266", "--fps", "24/0"],
    ["compare", "ref.mkv", "a.266", "--fps", "-24/1"],
    ["compare", "ref.mkv", "a.266", "--fps", "30000 / 1001"],
    ["compare", "ref.mkv", "a.266", "--fps", "24000/1001/1"],
    ["compare", "ref.mkv", "a.266", "--fps", "9007199254740992/1"],
    ["compare", "ref.mkv", "a.266", "--fps", "x"],
    ["compare", "ref.mkv", "a.266", "--fps"],
]


def _frame_rate():
    """--fps's type: a fraction of whole numbers below 2^53 ("24000/1001", as
    many as a double holds exactly) or a number ("23.976"), above 0 and
    finite. Its function is named parse, as the Python app's own number types
    are, so that argparse says "invalid parse value"."""
    def parse(text: str) -> float:
        """Reads `text` as a frame rate."""
        fraction = re.fullmatch(r"([0-9]+)/([0-9]+)", text.strip())
        if fraction:
            numerator, denominator = int(fraction[1]), int(fraction[2])
            if denominator == 0 or numerator >= 2**53 or denominator >= 2**53:
                raise ValueError(text)
            value = numerator / denominator
        else:
            value = float(text)
        if not value > 0 or not math.isfinite(value):
            raise argparse.ArgumentTypeError("must be above 0")
        return value
    return parse


#: The C++ app's own --fps, which the Python app never had, and its example.
FPS_HELP = ("the frame rate of a raw video stream (a .264, .265, .266 or .obu file, with no container), which "
            "records none, such as 24000/1001, 23.976 or 60: required for one. These scores are neither read from "
            "nor saved to the saved scores")
FPS_EXAMPLE = ("  VideoMetricsLab-cli compare reference.y4m encode.266 --fps 24000/1001\n"
               "      a raw VVC stream from an encoder, at 23.976 frames a second\n")


def _add_cpp_options(parser: argparse.ArgumentParser) -> None:
    """Adds the C++ app's own options to the Python app's parser where the
    C++ app has them, after --colour, and their example to the examples, so
    that the help and the errors are what the C++ app's must be."""
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    compare = commands.choices["compare"]
    fps = compare.add_argument("--fps", type=_frame_rate(), metavar="RATE", help=FPS_HELP)
    for actions in (compare._actions, compare._optionals._group_actions):
        actions.remove(fps)
        after = next(index for index, action in enumerate(actions) if "--colour" in action.option_strings)
        actions.insert(after + 1, fps)
    anchor = "      HDR10 video in files with no colour tags, the first 33 frames\n"
    for owner in (parser, compare):
        assert owner.epilog.count(anchor) == 1
        owner.epilog = owner.epilog.replace(anchor, anchor + FPS_EXAMPLE)


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
            _add_cpp_options(parser)
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
        # Keys out of order, for the saved scores' sorted form.
        {"b": 1, "a": {"d": [3, {"z": 1, "y": 2}], "c": {"$float": exact(2.5)}}, "B": True, "é": "x"},
    ]

    def compact(value) -> str | None:
        """The saved scores' form (metric_cache._canonical), or None where
        Python refuses the value (NaN and infinities)."""
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except ValueError:
            return None

    return [{"value": json.dumps(value, ensure_ascii=False), "text": json.dumps(decoded(value), indent=2),
             "compact": compact(decoded(value))}
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


def cvvdp_cases() -> dict:
    """CVVDP's displays (cvvdp.py): the built-in presets and the user's as
    Settings holds them (malformed ones skipped), each display's identity
    and its JSON for Vship, a request's settings read back, the size CVVDP
    is calculated at (perceptual_vship.cvvdp_computed_size), and the whole
    video's JOD pooled from its seconds (pool_cvvdp_windows), and the user's
    presets saved and deleted."""
    from dataclasses import asdict

    from vmaf_app.core import cvvdp
    from vmaf_app.core.perceptual_vship import cvvdp_computed_size, pool_cvvdp_windows

    def described(preset) -> dict:
        return {"name": preset.name, "builtin": preset.builtin, "description": preset.description,
                "identity": preset.settings.display.identity(),
                "resize_to_display": preset.settings.resize_to_display,
                "vship_json": cvvdp.vship_display_json(preset.settings.display)}

    tv = {"name": "Living room TV", "settings": {
        "display": {"width": 3840, "height": 2160, "diagonal_inches": 55.5, "viewing_distance_m": 2.25,
                    "peak_luminance": 750, "contrast": 5000, "ambient_lux": 30.123456, "reflectivity": 0.0123456789,
                    "exposure": 1.5, "hdr": True},
        "resize_to_display": True}}
    user_lists = [
        [],
        [tv],
        [{"name": "defaults", "settings": {}},
         {"name": 7, "settings": {"display": {"hdr": 1}}},
         {"name": 2.5, "settings": {"display": {"width": 1920.7, "height": 1080, "unknown": "left out"}}},
         {"name": None, "settings": {"display": {"hdr": "yes"}, "resize_to_display": "no"}},
         {"name": True, "settings": {"display": {"hdr": 0}, "resize_to_display": []}},
         {"name": "empty texts", "settings": {"display": {"hdr": ""}, "resize_to_display": ""}},
         {"name": "no settings"},
         {"settings": {}},
         {"name": "text width", "settings": {"display": {"width": "3840"}}},
         {"name": "true width", "settings": {"display": {"width": True}}},
         {"name": "null width", "settings": {"display": {"width": None}}},
         {"name": "tiny", "settings": {"display": {"width": 8}}},
         {"name": "no distance", "settings": {"display": {"viewing_distance_m": 0}}},
         {"name": "negative light", "settings": {"display": {"ambient_lux": -1}}},
         {"name": "a mirror", "settings": {"display": {"reflectivity": 1}}},
         {"name": "list settings", "settings": [1]},
         {"name": "null display", "settings": {"display": None}},
         {"name": "30-inch 4K monitor, office", "settings": {"display": {"peak_luminance": 400}}}],
    ]
    names = ("30-inch 4K monitor, office", "6-inch phone", "Living room TV", "defaults", "7", "2.5", "None",
             "True", "tiny", "missing", "")
    preset_lists = [{"saved": json.dumps(saved),
                     "presets": [described(preset) for preset in cvvdp.presets(saved)],
                     "named": {name: None if (preset := cvvdp.preset_named(name, saved)) is None
                               else described(preset) for name in names}}
                    for saved in user_lists]

    read_back = []
    for preset in cvvdp.presets([tv]):
        for resize in (False, True):
            parameters = cvvdp.CvvdpSettings(preset.settings.display, resize).spec_parameters()
            back = cvvdp.CvvdpSettings.from_spec_parameters(parameters)
            read_back.append({"parameters": dict(parameters), "identity": back.display.identity(),
                              "resize_to_display": back.resize_to_display,
                              "vship_json": cvvdp.vship_display_json(back.display)})

    sizes = []
    for size in ((1920, 1080), (3840, 2160), (1280, 536), (720, 576), (4096, 2160), (7680, 4320), (4, 3), (4, 5)):
        for width, height in ((3840, 2160), (1920, 1080), (2400, 1080), (18, 100)):
            for resize in (False, True):
                settings = cvvdp.CvvdpSettings(cvvdp.CvvdpDisplay(width=width, height=height), resize)
                sizes.append({"size": list(size), "display": [width, height], "resize_to_display": resize,
                              "computed": list(cvvdp_computed_size(size, settings))})

    rng = random.Random(20261013)
    pools = [{"windows": [[0, 1, exact(9.5)]], "jod": exact(pool_cvvdp_windows([(0, 1, 9.5)]))}]
    for case in range(60):
        windows, first = [], 0
        for _ in range(rng.randint(1, 8)):
            frames = rng.choice((1, 1, 2, 24, 25, 30, 60, 120)) if case % 5 else 1
            jod = rng.choice((10.0, 9.9999, rng.uniform(0, 10), rng.uniform(9.5, 10), rng.uniform(-2, 10.5)))
            windows.append((first, frames, jod))
            first += frames
        pools.append({"windows": [[start, frames, exact(jod)] for start, frames, jod in windows],
                      "jod": exact(pool_cvvdp_windows(windows))})

    # The user's presets saved and deleted (with_user_preset,
    # without_user_preset): added, put in place of one of the same name, the
    # name's spaces dropped, and refused for no name, a built-in's name or a
    # display CVVDP cannot model. The displays are as the app's editor gives
    # them: whole-number resolutions, the rest numbers with a fraction.
    def edited(**values) -> cvvdp.CvvdpDisplay:
        display = {"width": 2560, "height": 1440, "diagonal_inches": 27.0, "viewing_distance_m": 0.65,
                   "peak_luminance": 350.0, "contrast": 1200.0, "ambient_lux": 120.5, "reflectivity": 0.0075,
                   "exposure": 1.25, "hdr": False}
        return cvvdp.CvvdpDisplay(**{**display, **values})

    saved_lists = ([], [tv], [tv, {"name": 7, "settings": {}}, {"name": "Desk", "settings": {}}])
    edits = []
    for saved in saved_lists:
        for name, display in (("Desk", edited()), ("  Desk  ", edited(hdr=True)), ("Living room TV", edited()),
                              ("", edited()), ("   ", edited()), ("30-inch 4K monitor, office", edited()),
                              ("Tiny", edited(width=8)), ("Dark", edited(peak_luminance=-1.0))):
            try:
                result, error = cvvdp.with_user_preset(saved, name, cvvdp.CvvdpSettings(display, True)), None
            except ValueError as problem:
                result, error = None, str(problem)
            edits.append({"saved": json.dumps(saved), "name": name, "display": asdict(display),
                          "text": None if result is None else json.dumps(result, indent=2), "error": error})
        for name in ("Desk", "Living room TV", "7", "missing"):
            edits.append({"saved": json.dumps(saved), "deleted": name,
                          "text": json.dumps(cvvdp.without_user_preset(saved, name), indent=2)})
    return {"preset_lists": preset_lists, "read_back": read_back, "sizes": sizes, "pools": pools,
            "user_presets": edits}


def timed_pairing_cases() -> dict:
    """Pairing by timestamp (frame_sync_by_time.py, which Python's tests hold
    to FFmpeg's own pairing): av_rescale_q's rounding (rescale), -t in a time
    base (duration_in), and for many patterns of timestamps each pair and
    each frame's release, in order, as decoded_pairs takes them."""
    from fractions import Fraction

    from vmaf_app.core import frame_sync_by_time as by_time

    def base(value: Fraction) -> list[int]:
        return [value.numerator, value.denominator]

    bases = (Fraction(1, 1000), Fraction(1, 90000), Fraction(1, 24000), Fraction(1001, 24000), Fraction(1),
             Fraction(1, 2), Fraction(1, 1_000_000), Fraction(1, 120), Fraction(1, 12800), Fraction(1, 2**31 - 1),
             Fraction(2**31 - 1, 7))
    rng = random.Random(20261013)
    values = [0, 1, -1, 3, -3, 5, 41, 1001, 2**40, -(2**40), 2**62, -(2**62), 9 * 10**15,
              *(rng.randrange(-(2**62), 2**62) for _ in range(6))]
    rescales = []
    for value in values:
        for source in bases:
            for target in bases:
                result = by_time.rescale(value, source, target)
                if -(2**63) <= result < 2**63:  # the C++ app's numbers are 64-bit
                    rescales.append([value, base(source), base(target), result])

    durations = [[text, base(time_base), by_time.duration_in(text, time_base)]
                 for text in ("1.000", "1.008", "1.043", "30.042", "0", "0.5", "7200", "1.0000004", "86400.123456789",
                              "3")
                 for time_base in bases[:9]]

    def stamps(count: int, at) -> list[int]:
        return [int(at(n)) for n in range(count)]

    ms, tb90k, tb24k = Fraction(1, 1000), Fraction(1, 90000), Fraction(1, 24000)
    patterns = [
        ("same rate", stamps(48, lambda n: n * 42), ms, stamps(48, lambda n: n * 42), ms, None),
        ("test video shorter", stamps(47, lambda n: n * 42), ms, stamps(48, lambda n: n * 42), ms, None),
        ("source shorter", stamps(48, lambda n: n * 42), ms, stamps(46, lambda n: n * 42), ms, None),
        ("test frames a millisecond early", stamps(48, lambda n: n * 42 - (n % 3 == 1)), ms,
         stamps(48, lambda n: n * 42), ms, None),
        ("test frames a millisecond late", stamps(48, lambda n: n * 42 + (n % 4 == 1)), ms,
         stamps(48, lambda n: n * 42), ms, None),
        ("24 against 25 fps", stamps(48, lambda n: n * 1000 / 24), ms, stamps(50, lambda n: n * 40), ms, None),
        ("25 against 24 fps", stamps(50, lambda n: n * 40), ms, stamps(48, lambda n: n * 1000 / 24), ms, None),
        ("30 against 60 fps", stamps(30, lambda n: n * 1000 / 30), ms, stamps(60, lambda n: n * 1000 / 60), ms, None),
        ("60 against 30 fps", stamps(60, lambda n: n * 1000 / 60), ms, stamps(30, lambda n: n * 1000 / 30), ms, None),
        ("ties halfway", stamps(30, lambda n: n * 40 + 20), ms, stamps(31, lambda n: n * 40), ms, None),
        ("test video has gaps", stamps(40, lambda n: n * 42 + 84 * (n > 10) + 42 * (n > 25)), ms,
         stamps(48, lambda n: n * 42), ms, None),
        ("source has gaps", stamps(48, lambda n: n * 42), ms, stamps(44, lambda n: n * 42 + 126 * (n > 5)), ms, None),
        ("variable rate both", stamps(45, lambda n: n * 40 + (n * 7) % 13), ms,
         stamps(45, lambda n: n * 41 + (n * 5) % 11), ms, None),
        ("mp4 source, other time base", stamps(48, lambda n: n * 42), ms, stamps(48, lambda n: n * 3754), tb90k, None),
        ("mp4 test video, 1/24000", stamps(48, lambda n: n * 1001), tb24k, stamps(48, lambda n: n * 42), ms, None),
        ("23.976 fps from two clocks", stamps(240, lambda n: n * 1001), tb24k,
         stamps(240, lambda n: round(n * 1001 / 24)), ms, None),
        ("23.976 fps from two clocks, source first", stamps(240, lambda n: round(n * 1001 / 24)), ms,
         stamps(240, lambda n: n * 1001), tb24k, None),
        ("limit on a frame", stamps(48, lambda n: n * 42), ms, stamps(48, lambda n: n * 42), ms, "1.008"),
        ("limit between frames", stamps(48, lambda n: n * 42), ms, stamps(48, lambda n: n * 42), ms, "1.000"),
        ("limit, other time base", stamps(48, lambda n: n * 1001), tb24k, stamps(48, lambda n: n * 42), ms, "1.043"),
        ("timestamps not starting at 0", stamps(48, lambda n: 1000 + n * 42), ms,
         stamps(48, lambda n: 7_000_000 + n * 3754), tb90k, None),
        ("no source frames", stamps(10, lambda n: n * 42), ms, [], ms, None),
        ("no test frames", [], ms, stamps(10, lambda n: n * 42), ms, None),
        ("one source frame", stamps(10, lambda n: n * 42), ms, [0], ms, None),
        ("release order", [0, 42, 83, 125], ms, [0, 30, 50, 84], ms, None),
    ]
    for case in range(16):  # timestamps with jitter, at rates of their own
        test_step, source_step = rng.choice((40, 41, 42, 33, 17)), rng.choice((40, 41, 42, 33, 17, 84))
        test = sorted({n * test_step + rng.randint(-2, 2) for n in range(rng.randint(20, 60))})
        source = sorted({n * source_step + rng.randint(-2, 2) for n in range(rng.randint(20, 60))})
        limit = rng.choice((None, None, "0.5", "1.0004"))
        patterns.append((f"jitter {case}", test, ms, source, rng.choice((ms, Fraction(1, 2000))), limit))

    pairings = []
    for name, test_pts, test_base, source_pts, source_base, limit in patterns:
        events = []
        test_frames, source_frames = iter(enumerate(test_pts)), iter(enumerate(source_pts))
        stop = by_time.duration_in(limit, test_base) if limit else None
        pairs = by_time.frame_pairs(lambda frames=test_frames: next(frames, None),
                                    lambda frames=source_frames: next(frames, None), test_base, source_base,
                                    lambda frame, events=events: events.append(["release test", frame]),
                                    lambda frame, events=events: events.append(["release source", frame]))
        try:
            for test, source, when in pairs:
                if stop is not None and when >= stop:
                    break
                events.append(["pair", test, source, when])
        finally:
            pairs.close()
        pairings.append({"name": name, "test": test_pts, "test_base": base(test_base), "source": source_pts,
                         "source_base": base(source_base), "limit": limit, "events": events})
    return {"rescales": rescales, "durations": durations, "pairings": pairings}


def perceptual_cpu_cases() -> dict:
    """SSIMULACRA2 and Butteraugli on the CPU (perceptual_cpu.py, with
    colour.py's PNG chunks): the filter graphs and the FFmpeg commands that
    make the pictures (_image_filtergraph, _png_pairs), with frames paired
    by number and by timestamp; the scores read from the tools' output
    (parse_score); how many pairs are scored at a time (scoring_workers);
    Butteraugli's 3-norm of a distortion map (_butteraugli_norm3); and the
    colour chunks the pictures are given (png_colour_chunks, describe_png)."""
    from vmaf_app.core import colour, perceptual_cpu
    from vmaf_app.core.comparison_recipe import ComparisonRecipe
    from vmaf_app.core.models import CropBox, CropMode, ScaleDirection, VideoInfo

    def video(**fields) -> VideoInfo:
        base = {"path": Path("video.mkv"), "width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0,
                "nb_frames": 240, "codec_name": "hevc", "pix_fmt": "yuv420p"}
        base.update(fields)
        return VideoInfo(**base)

    def described(info: VideoInfo) -> dict:
        return {"width": info.width, "height": info.height, "fps": exact(info.fps), "pix_fmt": info.pix_fmt,
                "color_range": info.color_range, "color_space": info.color_space,
                "color_transfer": info.color_transfer, "color_primaries": info.color_primaries,
                "chroma_location": info.chroma_location}

    def box(crop: CropBox | None) -> list[int] | None:
        return None if crop is None else [crop.w, crop.h, crop.x, crop.y]

    def recipe(direction: ScaleDirection = ScaleDirection.SOURCE_TO_DISTORTED, algorithm: str = "bicubic",
               limit: float = 0.0) -> ComparisonRecipe:
        return ComparisonRecipe(CropMode.NONE, algorithm, direction, limit, None)

    hd, uhd = video(), video(width=3840, height=2160, pix_fmt="yuv420p10le", fps=120.0)
    hdr10 = {"color_space": "bt2020nc", "color_transfer": "smpte2084", "color_primaries": "bt2020",
             "color_range": "tv", "chroma_location": "topleft"}
    to_source = ScaleDirection.DISTORTED_TO_SOURCE
    graph_cases = (
        ("the same size, untagged", hd, video(fps=24000 / 1001), None, None, 1, recipe(), False),
        ("4K source scaled to the test video's 1080p, every third", uhd, hd, None, None, 3,
         recipe(algorithm="lanczos"), False),
        ("the test video scaled to the 4K source", uhd, hd, None, None, 1, recipe(to_source), False),
        ("black bars cropped from both", video(width=3840, height=2160, **hdr10),
         video(width=1920, height=1080, pix_fmt="yuv420p10le", **hdr10), CropBox(3840, 1608, 0, 276),
         CropBox(1920, 804, 0, 138), 2, recipe(), False),
        ("a crop that keeps it all", hd, hd, CropBox(1920, 1080, 0, 0), None, 1, recipe(), False),
        ("full-range yuvj beside HD", video(width=640, height=480, pix_fmt="yuvj420p", codec_name="mjpeg"), hd,
         None, None, 1, recipe(), False),
        ("planar RGB and an unknown matrix", video(pix_fmt="gbrp"), video(color_space="fcc"), None, None, 1,
         recipe(), False),
        ("by timestamp, blended", video(pix_fmt="yuv420p10le", **hdr10), video(pix_fmt="yuv420p10le", **hdr10),
         None, None, 1, recipe(), True),
        ("by timestamp, odd-sized, every third", hd, video(width=853, height=480), None, None, 3, recipe(), True),
        ("by timestamp, scaled to the source and cropped", uhd, hd, CropBox(3840, 1600, 0, 280), None, 2,
         recipe(to_source, "bilinear"), True),
        ("by timestamp, an NV12 source by position", video(pix_fmt="nv12"), hd, None, None, 2, recipe(), True),
        ("by timestamp, a siting setparams has no name for", video(chroma_location="top-left"), hd, None, None, 1,
         recipe(), True),
        ("by timestamp, a full-range yuvj source by position", video(pix_fmt="yuvj420p"), hd, None, None, 1,
         recipe(), True),
    )
    graphs = []
    for name, source, test, source_crop, test_crop, step, made_with, by_time in graph_cases:
        if by_time:
            os.environ["VML_FRAME_PAIRING"] = "timestamp"
        try:
            graph = perceptual_cpu._image_filtergraph(source, test, made_with, source_crop, test_crop, step)
        finally:
            os.environ.pop("VML_FRAME_PAIRING", None)
        graphs.append({"name": name, "source": described(source), "test": described(test),
                       "source_crop": box(source_crop), "test_crop": box(test_crop), "step": step,
                       "to_source": made_with.scale_direction is to_source, "algorithm": made_with.scale_algorithm,
                       "by_time": by_time, "graph": graph})

    class StartedError(Exception):
        """Stops _png_pairs once FFmpeg's command is known."""

    commands = []
    seen: list[list[str]] = []

    def popen(command, **_kwargs):
        seen.append(list(command))
        raise StartedError

    real_popen, real_ffmpeg = perceptual_cpu.proc_util.popen, perceptual_cpu.ffmpeg_path
    perceptual_cpu.proc_util.popen, perceptual_cpu.ffmpeg_path = popen, lambda: "ffmpeg"
    try:
        with tempfile.TemporaryDirectory() as folder:
            for limit, step in ((0.0, 1), (1.5, 3), (10.0, 1), (0.0417, 2)):
                seen.clear()
                source, test = video(path=Path("source.mkv")), video(path=Path("test.mkv"))
                with contextlib.suppress(StartedError):
                    next(perceptual_cpu._png_pairs(source, test, recipe(limit=limit), None, None, step,
                                                   Path(folder), None, None))
                places = {str(source.path.resolve()): "<source>", str(test.path.resolve()): "<test>"}
                command = [places.get(argument, argument.replace(folder + os.sep, "<folder>/"))
                           for argument in seen[0]]
                commands.append({"limit": exact(limit), "step": step, "command": command})
    finally:
        perceptual_cpu.proc_util.popen, perceptual_cpu.ffmpeg_path = real_popen, real_ffmpeg

    score_texts = ("SSIMULACRA2: 91.75", "butteraugli score = 0.2345", "  87.12345678\n", "3-norm: 1.234\n",
                   "a-5", "1.5.3", "5e", "x1e3 2E-2", ".5", "+.5e+2", "value_7", "", "no number here", "nan",
                   "1e999", "-0", "étape 5", "é5", "12. ", "-.25e-3x", "87.5\n\nwarning: 2 of 3\n",
                   "word " * 80, "1.0e+", "Infinity 4", "0x1F")
    scores = []
    for text in score_texts:
        try:
            scores.append({"text": text, "score": exact(perceptual_cpu.parse_score("ssimulacra2", text))})
        except perceptual_cpu.PerceptualRunError as error:
            scores.append({"text": text, "error": str(error)})

    uhd_pixels, gigabyte = 3840 * 2160, 1_000_000_000
    workers = []
    for pixels, metrics, concurrent, cores, free in (
            (uhd_pixels, ("ssimulacra2",), 1, 24, 64 * gigabyte),
            (uhd_pixels, ("ssimulacra2", "butteraugli"), 1, 24, 64 * gigabyte),
            (uhd_pixels, ("butteraugli",), 2, 24, 64 * gigabyte),
            (uhd_pixels, ("ssimulacra2",), 1, 24, 4 * gigabyte),
            (uhd_pixels, ("butteraugli",), 2, 24, 5 * gigabyte),
            (1920 * 1080, ("ssimulacra2",), 3, 16, 64 * gigabyte),
            (uhd_pixels, ("ssimulacra2",), 1, 1, 64 * gigabyte),
            (uhd_pixels, ("ssimulacra2",), 1, 24, 0),
            (0, ("butteraugli",), 1, 128, 64 * gigabyte),
            (64 * 48, ("ssimulacra2", "butteraugli"), 0, 24, 2 ** 40)):
        workers.append({"pixels": pixels, "metrics": list(metrics), "concurrent": concurrent, "cores": cores,
                        "free_memory": free,
                        "workers": perceptual_cpu.scoring_workers(pixels, metrics, concurrent, cores=cores,
                                                                  free_memory=free)})

    randoms = random.Random(20261013)

    def pfm(header: bytes, values: list[float], byte_order: str = "<") -> bytes:
        return header + struct.pack(f"{byte_order}{len(values)}f", *values)

    maps = [
        ("four values", pfm(b"Pf\n2 2\n-1.0\n", [1.0, 2.0, 3.0, 4.0])),
        ("big-endian", pfm(b"Pf\n3 1\n1.0\n", [0.5, 1.5, 2.5], ">")),
        ("no values", b"Pf\n0 0\n-1.0\n"),
        ("negative, zero and tiny values", pfm(b"Pf\n4 2\n-1.0\n", [-2.0, 0.0, -0.0, 1e-30, 3.25, 7.0, 1e-3, 1e4])),
        ("64x48 at random", pfm(b"Pf\n64 48\n-1.0\n", [randoms.uniform(0.0, 10.0) for _ in range(64 * 48)])),
        ("1000x7 at random, with spaces", pfm(b"Pf \n 1000  7 \n-1.000000\n",
                                              [randoms.uniform(0.0, 3.0) for _ in range(7000)])),
        ("Windows line ends", pfm(b"Pf\r\n5 1\r\n-1.0\r\n", [1.0, 1.0, 2.0, 3.0, 5.0])),
        ("more data than the size", pfm(b"Pf\n2 1\n-1.0\n", [1.0, 2.0, 99.0])),
        ("colour", pfm(b"PF\n1 1\n-1.0\n", [1.0, 1.0, 1.0])),
        ("cut short", pfm(b"Pf\n4 4\n-1.0\n", [1.0, 2.0])),
        ("no header", b"Pf\n2 2"),
        ("a size of one number", pfm(b"Pf\n4\n-1.0\n", [1.0] * 4)),
        ("a scale that is no number", pfm(b"Pf\n1 1\nminus\n", [1.0])),
    ]
    norms = []
    with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # numpy's, for the mean of no values
        for name, data in maps:
            path = Path(folder) / "map.pfm"
            path.write_bytes(data)
            made = {"name": name, "bytes": data.hex()}
            try:
                made["norm"] = exact(perceptual_cpu._butteraugli_norm3(path))
            except perceptual_cpu.PerceptualRunError as error:
                made["error"] = str(error)
            except ValueError:
                made["error"] = "unreadable"  # _run_metric says the tool wrote no readable distortion map
            norms.append(made)

    chunks = []
    for transfer in (1, 4, 5, 6, 7, 8, 13, 16, 17, 18):
        for primaries in (1, 4, 5, 6, 7, 9, 12):
            chunks.append({"transfer": transfer, "primaries": primaries,
                           "chunks": colour.png_colour_chunks(colour.VideoColour(1, transfer, primaries, False)).hex()})

    chunk = colour._png_chunk
    signature = b"\x89PNG\r\n\x1a\n"
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 2, 16, 2, 0, 0, 0))
    pictures = [
        ("FFmpeg's chunks replaced", signature + header + chunk(b"gAMA", b"\x00\x00\xb1\x8f")
         + chunk(b"cHRM", bytes(32)) + chunk(b"pHYs", bytes(9)) + chunk(b"IDAT", bytes(range(40)))
         + chunk(b"cICP", b"\x09\x10\x00\x01") + chunk(b"IEND", b"")),
        ("no colour chunks", signature + header + chunk(b"IDAT", b"\x01\x02") + chunk(b"IEND", b"")),
        ("shorter than a signature", bytes([0x89]) + b"PNG"),
        ("a chunk longer than the file", signature + header + struct.pack(">I", 1000) + b"IDAT" + bytes(10)),
    ]
    described_pictures = []
    with tempfile.TemporaryDirectory() as folder:
        for name, data in pictures:
            for picture_colour in (colour.VideoColour(9, 16, 9, False), colour.VideoColour(1, 1, 1, False)):
                path = Path(folder) / "picture.png"
                path.write_bytes(data)
                colour.describe_png(path, picture_colour)
                described_pictures.append({"name": name, "transfer": picture_colour.transfer,
                                           "primaries": picture_colour.primaries, "bytes": data.hex(),
                                           "described": path.read_bytes().hex()})
    return {"graphs": graphs, "commands": commands, "scores": scores, "workers": workers, "norms": norms,
            "chunks": chunks, "pictures": described_pictures}


def devices_cases() -> list[dict]:
    """The devices command's lines (cli.devices) for what its checks find,
    which stand in for the PC's: FFmpeg usable, missing or too old; a GPU
    for Vship, or none; VMAF on the GPU with Vulkan or CUDA, or on the CPU;
    VMAF v1 with the GPU or not; PSNR, SSIM and XPSNR in the app or in
    FFmpeg. The first line, the app's name and version, is left out: the
    apps' versions differ."""
    import io
    from types import SimpleNamespace

    from vmaf_app import cli
    from vmaf_app.core import ffmpeg_locate, libvmaf, perceptual_vship, vmaf_gpu, vmaf_runner, vmaf_v1_gpu
    from vmaf_app.core.models import GpuVendor

    real_settings = cli.Settings

    def tool(name: str, version: tuple[int, ...] | None = (9, 0, 2), error: str = "") -> ffmpeg_locate.ToolStatus:
        return ffmpeg_locate.ToolStatus(name, name, not error, None if error else version, error)

    def described_tool(status: ffmpeg_locate.ToolStatus) -> dict:
        return {"name": status.name, "path": status.path, "runnable": status.runnable,
                "version": None if status.version is None else list(status.version), "error": status.error}

    nvidia = perceptual_vship.VshipDevice("cuda", "NVIDIA GeForce RTX 5090", 0, "5.1.2", None, GpuVendor.NVIDIA)
    intel = perceptual_vship.VshipDevice("vulkan", "Intel(R) Graphics", 1, "5.1.2", None, GpuVendor.INTEL)
    usable = (tool("ffmpeg"), tool("ffprobe"))
    cases = []
    for name, tools, backend, vship, vmaf, v1, in_app in (
            ("every metric with the GPU", usable, "auto", (nvidia, ""),
             (True, "Vulkan on NVIDIA GeForce RTX 5090", ("vulkan", 0)), (True, "NVIDIA GeForce RTX 5090"), True),
            ("VMAF with CUDA, Vship with Vulkan", usable, "cuda", (intel, ""),
             (True, "NVIDIA GeForce RTX 5090 with CUDA", ("cuda", None)), (False, "the GPU's shaders failed"), False),
            ("no GPU", usable, "vulkan", (None, "no supported GPU"),
             (False, "no GPU that scores VMAF here", None), (False, "no Vulkan GPU"), True),
            ("FFmpeg missing", (tool("ffmpeg", error="not found"), tool("ffprobe")), "auto",
             (None, "no supported GPU"), (False, "no GPU", None), (False, "no GPU"), False),
            ("FFmpeg too old", (tool("ffmpeg", (8, 1)), tool("ffprobe", (8, 1))), "auto",
             (nvidia, ""), (False, "no GPU", None), (False, "no GPU"), True)):
        version = "v1-4-g994b1814"
        patches = [(cli, "Settings", SimpleNamespace(load=lambda backend=backend: real_settings(gpu_backend=backend))),
                   (cli, "_apply_settings", lambda settings: None),
                   (cli, "check_tools", lambda tools=tools: ffmpeg_locate.ToolsStatus(*tools)),
                   (perceptual_vship, "detect_vship_device", lambda vship=vship: vship),
                   (vmaf_gpu, "gpu_vmaf_available", lambda vmaf=vmaf: vmaf[:2]),
                   (vmaf_gpu, "gpu_vmaf_backend", lambda vmaf=vmaf: vmaf[2]),
                   (vmaf_v1_gpu, "available", lambda v1=v1: v1),
                   (vmaf_runner, "cpu_metrics_in_app", lambda in_app=in_app: in_app),
                   (libvmaf, "LIBVMAF_FAST_VERSION", version)]
        saved = [(owner, attribute, getattr(owner, attribute)) for owner, attribute, _value in patches]
        for owner, attribute, value in patches:
            setattr(owner, attribute, value)
        try:
            out = io.StringIO()
            cli.devices(None, out=out)
        finally:
            for owner, attribute, value in saved:
                setattr(owner, attribute, value)
        device = vship[0]
        cases.append({
            "name": name, "tools": [described_tool(status) for status in tools], "gpu_backend": backend,
            "vship": None if device is None else {"backend": device.backend, "name": device.name,
                                                  "gpu_id": device.gpu_id, "version": device.version},
            "no_vship": vship[1],
            "vmaf": {"backend": vmaf[2][0] if vmaf[0] else "", "device": vmaf[2][1] if vmaf[0] else None,
                     "text": vmaf[1]},
            "v1": {"available": v1[0], "text": v1[1]}, "cpu_metrics_in_app": in_app,
            "libvmaf_fast_version": version, "lines": out.getvalue().splitlines()[1:]})
    return cases


# ---------------------------------------------------------------- saved scores

#: The comparison folder metric_cache_cases saves, from which the C++ test
#: loads scores as the Python app does.
SAVED_SCORES = HERE / "metric_cache"


def _metric_json(metric) -> dict:
    """A metric's result, as the C++ tests read it (reference.h's
    result_from): numbers by float.hex()."""
    from vmaf_app.core.metric_results import SequenceMetricResult, provenance_to_dict

    data = {"key": metric.key, "provenance": provenance_to_dict(metric.provenance)}
    if isinstance(metric, SequenceMetricResult):
        data.update(kind="sequence", score=exact(metric.score))
        if not metric.has_timeline:
            return data
    else:
        data["kind"] = "frame"
    data.update(frame=[int(f) for f in metric.frame], time=[exact(float(t)) for t in metric.time],
                values=[exact(float(v)) for v in metric.values])
    return data


def _result_json(result) -> dict:
    """A comparison's result, as the C++ tests read it (reference.h's
    result_from)."""
    def video(details) -> dict:
        return {"path": str(details.path), "width": details.width, "height": details.height,
                "fps": exact(details.fps), "duration": exact(details.duration), "nb_frames": details.nb_frames,
                "codec_name": details.codec_name, "sar": details.sar, "pix_fmt": details.pix_fmt,
                "bit_rate": details.bit_rate, "bit_rate_whole_file": details.bit_rate_whole_file,
                "nominal_fps": exact(details.nominal_fps), "average_fps": exact(details.average_fps),
                "color_range": details.color_range, "color_space": details.color_space,
                "color_transfer": details.color_transfer, "color_primaries": details.color_primaries,
                "chroma_location": details.chroma_location}

    def crop(box) -> list | None:
        return None if box is None else [box.w, box.h, box.x, box.y]

    return {"source": str(result.source), "distorted": str(result.distorted), "fps": exact(result.fps),
            "model": result.model, "source_crop": crop(result.source_crop),
            "distorted_crop": crop(result.distorted_crop), "source_info": video(result.source_info),
            "distorted_info": video(result.distorted_info), "scale_direction": result.scale_direction.value,
            "scale_algorithm": result.scale_algorithm, "compared_frame_count": result.compared_frame_count,
            "model_choice": result.model_choice, "model_v1": result.model_v1,
            "model_choice_v1": result.model_choice_v1,
            "resample_target": None if result.resample_target is None else {
                "width": result.resample_target.width, "label": result.resample_target.label},
            "metrics": [_metric_json(result.metric_results.get(key)) for key in result.metric_results]}


#: Marks a field to take out of a file (run_file_cases).
_REMOVE = object()


def run_file_cases() -> dict:
    """Saved runs (run_io.py): the text save_run writes for results of every
    kind of score, and what load_run makes of files: the result, as save_run
    writes it again, or the error it gives."""
    import tempfile

    import numpy as np
    from vmaf_app.core import run_io
    from vmaf_app.core.builtin_models import builtin_choice
    from vmaf_app.core.metric_results import (
        FrameMetricResult,
        MetricProvenance,
        MetricResultSet,
        SequenceMetricResult,
    )
    from vmaf_app.core.models import ComparisonResult, CropBox, ResampleTarget, ScaleDirection, VideoInfo

    fps = 24000 / 1001
    source_path = "C:\\Videos\\K\u00e4se \u2013 source.mkv"
    test_path = "D:\\Encodes\\take 1.mkv"

    def info(path: str, width: int = 3840, height: int = 2160) -> VideoInfo:
        return VideoInfo(path=Path(path), width=width, height=height, fps=fps, duration=10.01, nb_frames=240,
                         codec_name="hevc", sar="1:1", pix_fmt="yuv420p10le", bit_rate=45_000_000,
                         bit_rate_whole_file=True, nominal_fps=fps, average_fps=fps, color_range="tv",
                         color_space="bt2020nc", color_transfer="smpte2084", color_primaries="bt2020",
                         chroma_location="topleft")

    frames = np.arange(6, dtype=np.int32)
    times = frames / fps
    libvmaf = MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1",
                               {"model": "version=vmaf_v0.6.1"})
    xpsnr = MetricProvenance("ffmpeg/xpsnr", "ffmpeg 9.0.1", "cpu", "ffmpeg-xpsnr-v2")
    # Parameters with their keys in order: the C++ app keeps a score's
    # parameters sorted (run_file.h), so its files differ in that order alone.
    cvvdp = MetricProvenance("Vship/cvvdp", "Vship 5.1.2", "gpu", "cvvdp-vship-gpu-v1",
                             {"display": {"hdr": True, "height": 2160, "width": 3840}, "resize": False})
    whole = MetricProvenance("unknown", "", "unknown", "unversioned")
    v1_choice = builtin_choice("vmaf_v1_3d0h_2160")

    def result(**fields) -> ComparisonResult:
        values = {"source": Path(source_path), "distorted": Path(test_path), "frames": [], "fps": fps,
                  "model": "version=vmaf_v0.6.1", "source_crop": None, "distorted_crop": None,
                  "source_info": info(source_path), "distorted_info": info(test_path, 1920, 1080),
                  "compared_frame_count": 6, "model_choice": "__auto__", **fields}
        return ComparisonResult(**values)

    every_kind = MetricResultSet([
        FrameMetricResult("vmaf", frames, times, np.array([95.5, 96.25, np.nan, 94.123456, 100.0, 0.0],
                                                           dtype=np.float32), libvmaf),
        FrameMetricResult("xpsnr", frames, times, np.array([40.5, np.inf, 38.25, np.inf, 41.0, -np.inf],
                                                            dtype=np.float32), xpsnr),
        SequenceMetricResult("cvvdp", 8.75, cvvdp, frames[:2], times[:2], np.array([8.5, 9.0], dtype=np.float32)),
        SequenceMetricResult("whole", float("inf"), whole),
    ])
    saved_results = [
        ("every kind of score", "K\u00e4se \u2013 take 1", result(
            metric_results=every_kind, source_crop=CropBox(3840, 1600, 0, 280),
            distorted_crop=CropBox(1920, 800, 0, 140), scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
            scale_algorithm="lanczos", model_v1="path=C:\\models\\vmaf_v1.json", model_choice_v1=v1_choice)),
        # Without a label, a file is named after its test video: plain file
        # names, which every system's paths read alike (a Windows path's
        # backslashes are no separators elsewhere).
        ("no label: the test video's name", "", result(
            metric_results=MetricResultSet([every_kind.get("vmaf")]), model_choice=None,
            source=Path("source.mkv"), distorted=Path("take 1.mkv"), source_info=info("source.mkv"),
            distorted_info=info("take 1.mkv", 1920, 1080))),
        ("a resolution test", "", result(
            metric_results=MetricResultSet([every_kind.get("vmaf")]), source=Path("source.mkv"),
            distorted=Path("source [downscale-720p-upscale].mkv"), source_info=info("source.mkv"),
            distorted_info=info("source.mkv"), resample_target=ResampleTarget(1280, "720p"))),
    ]
    saved = []
    texts = {}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.metrics.json"
        for name, label, saved_result in saved_results:
            run_io.save_run(saved_result, path, label=label or None)
            texts[name] = path.read_text(encoding="utf-8")
            saved.append({"name": name, "label": label, "result": _result_json(saved_result), "text": texts[name]})

        good = json.loads(texts["every kind of score"])

        def changed(**fields) -> str:
            data = json.loads(texts["every kind of score"])
            for key, value in fields.items():
                if value is _REMOVE:
                    del data[key]
                else:
                    data[key] = value
            return json.dumps(data)

        def changed_metric(index: int, **fields) -> str:
            data = json.loads(texts["every kind of score"])
            for key, value in fields.items():
                if value is _REMOVE:
                    del data["metric_results"][index][key]
                else:
                    data["metric_results"][index][key] = value
            return json.dumps(data)

        provenance = good["metric_results"][0]["provenance"]
        # As saved while VMAF v1 was a model of the one VMAF column.
        old_v1 = json.loads(texts["no label: the test video's name"])
        old_v1.update(model="path=C:\\vmaf_app\\models\\vmaf_v1.0.16_3d0h_2160.json", model_choice=v1_choice)
        files = [
            ("every kind of score", texts["every kind of score"]),
            ("a resolution test", texts["a resolution test"]),
            ("format version 1", changed(format_version=1)),
            ("format version as text", changed(format_version="2")),
            ("format version as a decimal", changed(format_version=2.0)),
            ("no format version", changed(format_version=_REMOVE)),
            ("not an object", json.dumps([good])),
            ("scores not a list", changed(metric_results={"vmaf": []})),
            ("no scores", changed(metric_results=_REMOVE)),
            ("a score not an object", changed(metric_results=[1])),
            ("a score without a key", changed_metric(0, key="")),
            ("a score without provenance", changed_metric(0, provenance=_REMOVE)),
            ("provenance not an object", changed_metric(0, provenance="libvmaf")),
            ("provenance parameters not an object", changed_metric(0, provenance={**provenance, "parameters": []})),
            ("provenance without its backend", changed_metric(
                0, provenance={key: value for key, value in provenance.items() if key != "compute_backend"})),
            ("a score of another kind", changed_metric(0, kind="histogram")),
            ("a score of no kind", changed_metric(0, kind=_REMOVE)),
            ("frame arrays not lists", changed_metric(0, values="95.5")),
            ("frame arrays of other lengths", changed_metric(0, time=[0.0])),
            ("a missing frame array", changed_metric(0, time=_REMOVE)),
            ("timeline of other lengths", changed_metric(2, timeline={"frame": [0], "time": [0.0, 1.0],
                                                                      "values": [8.5]})),
            ("scores as text", changed_metric(0, values=["95.5", " 96.25 ", "nan", "-inf", "Infinity", None])),
            ("no source", changed(source=_REMOVE)),
            ("no source details", changed(source_info=_REMOVE)),
            ("source details without a path", changed(
                source_info={key: value for key, value in good["source_info"].items() if key != "path"})),
            ("source details of an older file", changed(source_info={key: good["source_info"][key] for key in (
                "path", "width", "height", "fps", "duration", "nb_frames", "codec_name")})),
            ("another scale direction", changed(scale_direction="sideways")),
            ("no resolution test field", changed(resample_target=_REMOVE)),
            ("no model choice field", changed(model_choice=_REMOVE)),
            ("no VMAF v1 fields", changed(model_v1=_REMOVE, model_choice_v1=_REMOVE)),
            ("no label", changed(label=_REMOVE)),
            ("no crops", changed(source_crop=_REMOVE, distorted_crop=_REMOVE)),
            ("no frames counted", changed(compared_frame_count=0)),
            ("VMAF v1 scores in the VMAF column", json.dumps(old_v1)),
            ("not JSON", "{"),
        ]
        loads = []
        for name, text in files:
            case = {"name": name, "text": text}
            path.write_text(text, encoding="utf-8")
            try:
                loaded, label = run_io.load_run(path)
            except Exception as error:  # every error is the answer
                # Python's JSON parser words its errors its own way.
                case["error"] = "" if name == "not JSON" else str(error)
            else:
                run_io.save_run(loaded, path, label=label)
                case.update(label=label, resaved=path.read_text(encoding="utf-8"))
            loads.append(case)
    return {"saved": saved, "loads": loads}


def metric_cache_cases() -> dict:
    """The saved scores as the Python app names, writes and reads them
    (metric_cache.py): each request's identity and hash and the recipe's
    (ffmpeg_request.py, comparison_recipe.py); the context.json and score
    files a stored result makes; which scores answer a request (answers,
    made_as_chosen, reusable); and the scores requests load from a
    comparison folder this saves in metric_cache/, which the C++ test lays
    under the folder name its own hashing gives."""
    import shutil
    from types import SimpleNamespace

    import numpy as np
    from vmaf_app.core import metric_cache
    from vmaf_app.core.builtin_models import builtin_choice
    from vmaf_app.core.cvvdp import CvvdpDisplay, CvvdpSettings
    from vmaf_app.core.ffmpeg_request import analysis_request_from_vmaf_options
    from vmaf_app.core.metric_results import (
        FrameMetricResult,
        MetricProvenance,
        MetricResultSet,
        SequenceMetricResult,
        provenance_to_dict,
    )
    from vmaf_app.core.models import (
        ComparisonResult,
        CropBox,
        CropMode,
        ResampleTarget,
        ScaleDirection,
        VideoInfo,
        VmafOptions,
    )

    def options_of(fields: dict) -> VmafOptions:
        options = VmafOptions()
        for name, value in fields.items():
            setattr(options, name, CropMode(value) if name == "crop_mode"
                    else ScaleDirection(value) if name == "scale_direction" else value)
        return options

    def options_json(options: VmafOptions) -> dict:
        return {"n_subsample": options.n_subsample, "model_choice": options.model_choice,
                "model_choice_v1": options.model_choice_v1, "crop_mode": options.crop_mode.value,
                "scale_direction": options.scale_direction.value, "scale_algorithm": options.scale_algorithm,
                "duration_limit": exact(options.duration_limit),
                "resample_test": None if options.resample_test is None else {
                    "width": options.resample_test.width, "label": options.resample_test.label}}

    hdr_display = CvvdpSettings(CvvdpDisplay(width=1920, height=1080, diagonal_inches=55.0, viewing_distance_m=2.1,
                                             peak_luminance=1000.0, contrast=100000.0, ambient_lux=10.0,
                                             reflectivity=0.01, exposure=1.0, hdr=True), resize_to_display=True)

    def request_json(name: str, fields: dict, metrics: tuple, backends: dict, cvvdp) -> tuple[dict, object]:
        options = options_of(fields)
        request = analysis_request_from_vmaf_options(options, metrics, backends or None, cvvdp)
        return {"name": name, "options": options_json(options), "metrics": list(metrics), "backends": backends,
                "cvvdp": None if cvvdp is None else dict(cvvdp.spec_parameters())}, request

    requests = []
    for name, fields, metrics, backends, cvvdp in (
            ("VMAF on Auto", {}, ("vmaf",), {}, None),
            ("libvmaf's metrics subsampled", {"n_subsample": 3}, ("vmaf", "vmaf_neg", "psnr", "ssim", "xpsnr"), {}, None),
            ("XPSNR alone, subsampled", {"n_subsample": 3}, ("xpsnr",), {}, None),
            ("a chosen model", {"model_choice": "version=vmaf_4k_v0.6.1"}, ("vmaf", "vmaf_v1"), {}, None),
            ("a bundled VMAF v1 model", {"model_choice_v1": builtin_choice("vmaf_v1_1d5h_2160")},
             ("vmaf_v1", "psnr"), {}, None),
            ("perceptual, subsampled, SSIMULACRA2 on the CPU", {"n_subsample": 2}, ("ssimulacra2", "butteraugli"),
             {"ssimulacra2": "cpu"}, None),
            ("CVVDP on an HDR display it fills", {}, ("cvvdp", "vmaf"), {}, hdr_display),
            ("a resolution test", {"resample_test": ResampleTarget(1280, "720p")}, ("vmaf", "psnr"), {}, None),
            ("another recipe", {"crop_mode": "none", "scale_direction": "distorted_to_source",
                                "scale_algorithm": "lanczos", "duration_limit": 30.5},
             ("vmaf", "ssimulacra2"), {"butteraugli": "cpu"}, None)):
        case, request = request_json(name, fields, metrics, backends, cvvdp)
        identity = request.recipe.identity_dict()
        case.update(perceptual_backends=dict(request.execution.perceptual_backends),
                    recipe=metric_cache._canonical(identity), recipe_hash=metric_cache._hash("S", "D", identity),
                    specs=[{"identity": metric_cache._canonical(spec.identity_dict()),
                            "hash": metric_cache.metric_identity_hash(spec)} for spec in request.metrics])
        requests.append(case)

    fps = 24000 / 1001

    def info(name: str, pix_fmt: str = "yuv420p", transfer: str = "") -> VideoInfo:
        return VideoInfo(path=Path(name), width=1920, height=1080, fps=fps, duration=10.01, nb_frames=240,
                         codec_name="hevc", sar="1:1", pix_fmt=pix_fmt, bit_rate=4_500_000, nominal_fps=fps,
                         average_fps=fps, color_range="tv", color_space="bt709", color_transfer=transfer,
                         color_primaries="bt709", chroma_location="left")

    def frame_metric(key: str, values: list[float], provenance: MetricProvenance, count: int = 0):
        frames = np.arange(count or len(values), dtype=np.int32)
        return FrameMetricResult(key, frames, frames / fps, np.array(values, dtype=np.float32), provenance)

    metric_json, result_json = _metric_json, _result_json

    def result_of(metrics: list, source_info: VideoInfo, distorted_info: VideoInfo, **fields) -> ComparisonResult:
        results = MetricResultSet()
        for metric in metrics:
            results.add(metric)
        values = {"source_crop": None, "distorted_crop": None, "model_choice": "__auto__", **fields}
        return ComparisonResult(source=Path("s.mp4"), distorted=Path("t.mp4"), frames=[], fps=fps,
                                model="version=vmaf_v0.6.1", source_info=source_info, distorted_info=distorted_info,
                                compared_frame_count=240, metric_results=results, **values)

    gpu_vmaf = MetricProvenance("libvmaf/vulkan", "libvmaf-fast v1", "gpu", "ffmpeg-libvmaf-v1",
                                {"model": "version=vmaf_v0.6.1"})
    cpu_libvmaf = MetricProvenance("libvmaf", "libvmaf-fast v1", "cpu", "ffmpeg-libvmaf-v1")
    vship = MetricProvenance("Vship/ssimulacra2", "Vship 5.1.2", "gpu", "ssimulacra2-vship-gpu-v1",
                             {"coverage_step": 1, "color_tags": metric_cache.VSHIP_COLOR_TAGS})
    libjxl = MetricProvenance("ssimulacra2", "libjxl 0.11.1", "cpu", "ssimulacra2-libjxl-cpu-v1",
                              {"intermediate": "png/rgb48le", "coverage_step": 1,
                               "color_tags": metric_cache.CPU_COLOR_TAGS})
    cvvdp_made = MetricProvenance("Vship/cvvdp", "Vship 5.1.2", "gpu", "cvvdp-vship-gpu-v1",
                                  {"display": CvvdpSettings().display.identity(), "resize_to_display": False,
                                   "color_tags": metric_cache.VSHIP_COLOR_TAGS})

    # What storing a result writes: its context.json and each score's file.
    stored = result_of(
        [frame_metric("vmaf", [91.5, 92.25, 90.125, 93.0], gpu_vmaf),
         frame_metric("psnr", [41.25, 40.5, 42.0, 39.75], cpu_libvmaf),
         frame_metric("ssimulacra2", [80.5, 81.25, 79.0, 82.75], vship),
         SequenceMetricResult("cvvdp", 8.75, cvvdp_made, np.array([0, 24], np.int32), np.array([0.0, 24 / fps]),
                              np.array([8.5, 9.0], np.float32))],
        info("s.mp4", "yuv420p10le", "smpte2084"), info("t.mp4", "yuv420p10le", "smpte2084"),
        source_crop=CropBox(1920, 800, 0, 140), model_choice_v1=None)
    store_case, store_request = request_json("stored", {}, ("vmaf", "psnr", "ssimulacra2", "cvvdp"), {}, None)
    with tempfile.TemporaryDirectory() as folder:
        source, distorted = Path(folder) / "s.mp4", Path(folder) / "t.mp4"
        source.write_bytes(b"s")
        distorted.write_bytes(b"t")
        directory = metric_cache.store_result(Path(folder) / "cache", source, distorted, store_request.recipe,
                                              stored, "t", store_request.metrics)
        store_case.update(result=result_json(stored), label="t",
                          context=(directory / "context.json").read_text(encoding="utf-8"),
                          files={path.name: str(np.load(path)["metadata"].item())
                                 for path in sorted(directory.glob("*.npz"))})

    # A resolution test's: its target in context.json, and in the recipe the
    # folder is named by.
    libvmaf = MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1",
                               {"model": "version=vmaf_v0.6.1"})
    stored_resample = result_of([frame_metric("vmaf", [95.5, 96.25, 94.75, 97.0], libvmaf)], info("s.mp4"),
                                info("s.mp4"), resample_target=ResampleTarget(1280, "720p"), model_choice_v1=None)
    resample_case, resample_request = request_json(
        "stored resolution test", {"resample_test": ResampleTarget(1280, "720p")}, ("vmaf",), {}, None)
    with tempfile.TemporaryDirectory() as folder:
        source, distorted = Path(folder) / "s.mp4", Path(folder) / "t.mp4"
        source.write_bytes(b"s")
        distorted.write_bytes(b"t")
        directory = metric_cache.store_result(Path(folder) / "cache", source, distorted, resample_request.recipe,
                                              stored_resample, "t", resample_request.metrics)
        resample_case.update(result=result_json(stored_resample), label="t",
                             context=(directory / "context.json").read_text(encoding="utf-8"),
                             files={path.name: str(np.load(path)["metadata"].item())
                                    for path in sorted(directory.glob("*.npz"))})

    # Which scores answer a request.
    rgb, yuv = info("s.mp4", "gbrp10le", ""), info("t.mp4", "yuv420p", "bt709")
    perceptual_request = request_json("perceptual", {}, ("ssimulacra2", "xpsnr", "vmaf"), {}, None)[1]
    old_vship = MetricProvenance("Vship/ssimulacra2", "Vship 5.1.0", "gpu", "ssimulacra2-vship-5.1.0-gpu-v1")
    answer_cases = []
    for key, provenance, infos in (
            ("vmaf", MetricProvenance("unknown", "", "unknown", "unversioned"), ()),
            ("xpsnr", MetricProvenance("unknown", "", "unknown", "unversioned"), ()),
            ("xpsnr", MetricProvenance("ffmpeg/xpsnr", "ffmpeg 8.0", "cpu", "ffmpeg-xpsnr-v1"), ()),
            ("xpsnr", MetricProvenance("ffmpeg/xpsnr", "ffmpeg 9.0", "cpu", "ffmpeg-xpsnr-v2"), ()),
            ("ssimulacra2", vship, ()),
            ("ssimulacra2", libjxl, ()),
            ("ssimulacra2", MetricProvenance("ssimulacra2", "libjxl 0.11.1", "cpu", "ssimulacra2-libjxl-cpu-v1"), ()),
            ("ssimulacra2", old_vship, (rgb, yuv)),
            ("ssimulacra2", old_vship, (yuv, yuv)),
            ("ssimulacra2", MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0", "cpu", "ffmpeg-libvmaf-v1"), ())):
        spec = perceptual_request.metric(key)
        tags = metric_cache.video_infos(*infos)
        answer_cases.append({"key": key, "provenance": provenance_to_dict(provenance), "infos": list(tags),
                             "answers": metric_cache.answers(spec, SimpleNamespace(provenance=provenance), tags)})
    chosen_cases = [{"key": key, "choice": choice, "produced": produced, "gpu_can_score": gpu_can,
                     "counts": metric_cache.made_as_chosen(key, choice, produced, lambda _key, can=gpu_can: can)}
                    for key in ("ssimulacra2",) for choice in ("gpu", "cpu", None)
                    for produced in ("gpu", "cpu") for gpu_can in (True, False)]
    reused = result_of(
        [frame_metric("vmaf", [math.nan, math.nan], gpu_vmaf),
         frame_metric("psnr", [math.nan, 40.0], cpu_libvmaf),
         frame_metric("ssimulacra2", [70.0, 71.0], libjxl),
         frame_metric("butteraugli", [1.5, 1.25], MetricProvenance("butteraugli", "libjxl 0.11.1", "cpu",
                                                                     "butteraugli-libjxl-cpu-v1",
                                                                     {"color_tags": metric_cache.CPU_COLOR_TAGS})),
         SequenceMetricResult("cvvdp", math.nan, cvvdp_made)],
        info("s.mp4"), info("t.mp4"))
    reuse_cases = []
    for backends, gpu_can in (({"ssimulacra2": "gpu", "butteraugli": "gpu"}, True),
                              ({"ssimulacra2": "gpu", "butteraugli": "cpu"}, False)):
        case, request = request_json("reused", {}, ("vmaf", "psnr", "ssimulacra2", "butteraugli", "cvvdp"),
                                     backends, None)
        kept = metric_cache.reusable(reused, request.metrics, backends, lambda _key, can=gpu_can: can,
                                     metric_cache.video_infos(reused.source_info, reused.distorted_info))
        case.update(gpu_can_score=gpu_can, kept=list(kept))
        reuse_cases.append(case)

    # A comparison folder of saved scores, and what requests load from it.
    shutil.rmtree(SAVED_SCORES, ignore_errors=True)
    SAVED_SCORES.mkdir()
    saved = result_of([], info("s.mp4"), info("t.mp4"))
    metric_cache._atomic_json(SAVED_SCORES / "context.json", metric_cache._context_from_result(
        saved, "t", request_json("saved", {}, ("vmaf",), {}, None)[1].recipe))

    def save(fields: dict, key: str, metric, cvvdp=None) -> None:
        spec = request_json("saved", fields, (key,), {}, cvvdp)[1].metric(key)
        metric_cache.store_metric(SAVED_SCORES, metric, spec)

    save({}, "vmaf", frame_metric("vmaf", [90.5, 91.0, 89.25, 92.5, 90.0, 91.75], gpu_vmaf))
    save({}, "psnr", frame_metric("psnr", [40.5, 41.0, 39.25, 42.5, 40.0, 41.75], cpu_libvmaf))
    save({"n_subsample": 2}, "psnr", frame_metric("psnr", [30.5, 31.0, 29.25, 32.5, 30.0, 31.75], cpu_libvmaf))
    save({}, "xpsnr", frame_metric("xpsnr", [44.0, 45.5, 43.25, 44.75, 45.0, 46.0],
                                   MetricProvenance("ffmpeg/xpsnr", "ffmpeg 9.0", "cpu", "ffmpeg-xpsnr-v2")))
    save({}, "ssimulacra2", frame_metric("ssimulacra2", [80.0, 81.5, 79.25, 82.0, 80.75, 81.0], vship))
    save({}, "ssimulacra2", frame_metric("ssimulacra2", [78.0, 79.5, 77.25, 80.0, 78.75, 79.0], libjxl))
    save({}, "butteraugli", frame_metric("butteraugli", [1.5, 1.25, 1.75, 1.0, 1.5, 1.25],
                                         MetricProvenance("butteraugli", "libjxl 0.11.1", "cpu",
                                                          "butteraugli-libjxl-cpu-v1")))
    save({}, "cvvdp", SequenceMetricResult("cvvdp", 9.25, cvvdp_made, np.array([0, 24], np.int32),
                                           np.array([0.0, 24 / fps]), np.array([9.0, 9.5], np.float32)))
    save({}, "cvvdp", SequenceMetricResult("cvvdp", 7.5, cvvdp_made), hdr_display)
    old_v1 = builtin_choice("vmaf_v1_3d0h")
    save({"model_choice": old_v1}, "vmaf", frame_metric("vmaf", [88.0, 89.5, 87.25, 90.0, 88.75, 89.0], cpu_libvmaf))

    loads = []
    for name, fields, metrics, backends, cvvdp in (
            ("everything, on Auto", {}, ("vmaf", "vmaf_v1", "psnr", "xpsnr", "ssimulacra2", "butteraugli", "cvvdp"),
             {}, None),
            ("the model Auto ran, chosen", {"model_choice": "version=vmaf_v0.6.1"}, ("vmaf",), {}, None),
            ("another model chosen", {"model_choice": "version=vmaf_4k_v0.6.1"}, ("vmaf",), {}, None),
            ("SSIMULACRA2 on the CPU", {}, ("ssimulacra2",), {"ssimulacra2": "cpu"}, None),
            ("subsampled", {"n_subsample": 2}, ("psnr", "ssimulacra2"), {}, None),
            ("CVVDP on another display", {}, ("cvvdp",), {}, hdr_display)):
        case, request = request_json(name, fields, metrics, backends, cvvdp)
        loaded = metric_cache.load_metrics(SAVED_SCORES, request.metrics, dict(request.execution.perceptual_backends))
        case["loaded"] = [metric_json(loaded.get(key)) for key in loaded]
        loads.append(case)

    # The folder laid where a comparison of two files finds it.
    from vmaf_app.core import result_cache
    from vmaf_app.core.ffmpeg_request import displayable_metric_specs

    def laid(folder: Path, recipe) -> tuple[Path, Path, Path]:
        base, source, distorted = folder / "cache", folder / "s.mp4", folder / "t.mp4"
        source.write_bytes(b"s")
        distorted.write_bytes(b"t")
        directory = metric_cache.recipe_directory(base, source, distorted, recipe)
        directory.mkdir(parents=True, exist_ok=True)
        for path in SAVED_SCORES.iterdir():
            shutil.copy(path, directory / path.name)
        return base, source, distorted

    # What the window finds for a row: the scores it asks for and every other
    # score it can show (displayable_metric_specs), and CVVDP's scores saved
    # for other displays.
    window_loads = []
    with tempfile.TemporaryDirectory() as folder:
        base, source, distorted = laid(Path(folder), request_json("saved", {}, ("vmaf",), {}, None)[1].recipe)
        for name, fields, metrics, backends, cvvdp in (
                ("everything, on Auto", {}, ("vmaf", "psnr", "cvvdp"), {}, None),
                ("one metric ticked", {}, ("ssim",), {}, None),
                ("subsampled, SSIMULACRA2 on the CPU", {"n_subsample": 2}, ("psnr",), {"ssimulacra2": "cpu"}, None),
                ("another display", {}, ("vmaf",), {}, hdr_display)):
            case, request = request_json(name, fields, metrics, backends, cvvdp)
            supplemental = displayable_metric_specs(options_of(fields), cvvdp)
            found = result_cache.load_cached(source, distorted, request, base, supplemental)
            parameters, others = result_cache.other_cvvdp_scores(source, distorted, request, base, supplemental)
            case.update(
                displayable=[metric_cache._canonical(spec.identity_dict()) for spec in supplemental],
                loaded=[] if found is None else [metric_json(found[0].metric_results.get(key))
                                                 for key in found[0].metric_results],
                other_cvvdp=None if not parameters else {
                    "parameters": dict(parameters),
                    "scores": [{"parameters": dict(settings.spec_parameters()), "score": exact(score)}
                               for settings, score in others]})
            window_loads.append(case)

    # What Recalculate leaves of the folder: each request's scores go, with
    # those that would answer it too.
    clears = []
    for name, fields, metrics, backends, cvvdp in (
            ("everything, on Auto", {}, ("vmaf", "vmaf_v1", "psnr", "xpsnr", "ssimulacra2", "butteraugli", "cvvdp"),
             {}, None),
            ("a model Auto ran, chosen", {"model_choice": "version=vmaf_v0.6.1"}, ("vmaf",), {}, None),
            ("SSIMULACRA2, on either device", {}, ("ssimulacra2",), {"ssimulacra2": "cpu"}, None),
            ("VMAF v1 saved under VMAF's key", {"model_choice_v1": old_v1}, ("vmaf_v1",), {}, None),
            ("CVVDP on another display", {}, ("cvvdp",), {}, hdr_display)):
        case, request = request_json(name, fields, metrics, backends, cvvdp)
        with tempfile.TemporaryDirectory() as folder:
            base, source, distorted = laid(Path(folder), request.recipe)
            removed = metric_cache.clear_metrics(base, source, distorted, request.recipe, request.metrics)
            directory = metric_cache.recipe_directory(base, source, distorted, request.recipe)
            left = sorted(path.name for path in directory.iterdir()) if directory.exists() else []
        case.update(removed=removed, left=left)
        clears.append(case)

    return {"requests": requests, "stored": store_case, "stored_resolution_test": resample_case,
            "answers": answer_cases, "made_as_chosen": chosen_cases,
            "reusable": {"result": result_json(reused), "cases": reuse_cases},
            "saved": {"folder": SAVED_SCORES.name, "label": "t", "loads": loads, "window_loads": window_loads,
                      "clears": clears}}


# ---------------------------------------------------------------------- i18n

#: Language names as Windows gives them (QLocale's uiLanguages).
I18N_WINDOWS_NAMES = (
    "de-DE", "de", "fr-CA", "pt-BR", "pt-PT", "es-MX", "zh-Hans-CN", "zh-CN", "zh-SG", "zh-Hant-TW", "zh-TW",
    "zh-HK", "zh-MO", "zh_TW", "ar-SA", "en-GB", "en-US", "sv-SE", "", "  ", "-", "JA-jp", "ko-KR", "ru-RU",
    "uk-UA", "pl-PL", "cs-CZ", "hu-HU", "tr-TR", "nl-NL", "it-IT", "id-ID", "vi-VN", "th-TH", "nb-NO",
)

#: str.format over the format specs the catalogs use, and their edges.
I18N_FORMATS = (
    ("{count} done", {"count": 3}),
    ("{count:,} frames", {"count": 1234567}),
    ("{count:,}", {"count": 999}),
    ("{count:,}", {"count": -1234567}),
    ("{count:,}", {"count": 0}),
    ("{size:,}", {"size": 12345678901}),
    ("{fps:.2f} fps", {"fps": 23.976023976023978}),
    ("{fps:.3f}", {"fps": 59.94005994005994}),
    ("{x:.1f}", {"x": 0.05}),
    ("{x:.1f}", {"x": 0.25}),
    ("{x:.1f}", {"x": -0.04}),
    ("{x:.0f}", {"x": 2.5}),
    ("{x:.2f}", {"x": 1e21}),
    ("{x:.1f}", {"x": 3}),
    ("{x:g}", {"x": 2.5}),
    ("{x:g}", {"x": 0.0001}),
    ("{x:g}", {"x": 0.00001}),
    ("{x:g}", {"x": 123456.0}),
    ("{x:g}", {"x": 1234567.0}),
    ("{x:g}", {"x": 100.0}),
    ("{x:g}", {"x": 0.0}),
    ("{x:g}", {"x": -1.5e-7}),
    ("{x:g}", {"x": 3}),
    ("{x:,.1f}", {"x": 1234567.891}),
    ("{x:,.2f}", {"x": -9876.545}),
    ("{n:>6}", {"n": 42}),
    ("{n:>6}", {"n": 1234567}),
    ("{n:>6,}", {"n": 12345}),
    ("{n:<5}|", {"n": 42}),
    ("{name:>8}|", {"name": "ab"}),
    ("{name:<8}|", {"name": "ab"}),
    ("{name:>3}|", {"name": "abcdef"}),
    ("{x}", {"x": 2.5}),
    ("{x}", {"x": 1 / 3}),
    ("{x}", {"x": 1e16}),
    ("{x}", {"x": 100.0}),
    ("{x}", {"x": 1e-7}),
    ("{x}", {"x": 12}),
    ("{{literal}} {x}", {"x": 1}),
    ("{a} and {b}", {"a": "x", "b": "y"}),
    ("{a}{a}", {"a": "z"}),
    ("Ünïcödé {name} 動画 }}", {"name": "ок"}),
)

#: The counts each counted text is shown for.
I18N_COUNTS = (0, 1, 2, 3, 5, 11, 21, 22, 101, 1001)

#: The catalogs whose every text is checked, one of each kind: two, three
#: and six plural forms, right to left, ideographic full stops, no spaces.
I18N_LANGUAGES = ("de", "ru", "ar", "ja", "zh_CN", "th")


def _i18n_values(text: str) -> dict:
    """A value for each of `text`'s fields, of the kind its format spec
    takes: a number with a fraction for "f" and "g", a whole number for
    other specs, and text where there is none."""
    import string

    values = {}
    for _, field, spec, _ in string.Formatter().parse(text):
        if not field or field in values:
            continue
        values[field] = 1234.5678 if spec and spec[-1] in "fg" else 1234567 if spec else f"<{field}>"
    return values


def _i18n_message(template: str, values: dict | None = None) -> str:
    """`template` with each {field} filled in, as the engine writes it."""
    return re.sub(r"\{(\w+)\}", lambda match: (values or {}).get(match[1], f"V{match[1]}"), template)


def graph_cases() -> dict:
    """Metric Graphs (ui/graph_panel.py), from the Python app's own panel:
    three runs (every kind of score: a dip, XPSNR's infinity, a subsampled
    PSNR, Butteraugli's high tail, CVVDP's seconds, a short run) added in
    turn; then, on each metric's page, the table's headings and cells, the Go
    to frame box's range, the readout at set points (the charts 800 pixels
    wide) and at set frames, with all three lines shown and with one hidden."""
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from vmaf_app.core.metric_results import (
        FrameMetricResult,
        MetricProvenance,
        MetricResultSet,
        SequenceMetricResult,
    )
    from vmaf_app.core.metrics import METRICS
    from vmaf_app.core.models import ComparisonResult, VideoInfo
    from vmaf_app.ui.graph_panel import GraphPanel

    app = QApplication.instance() or QApplication(["make_reference", "-platform", "offscreen"])
    fps = 30.0
    made = MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1")

    def frame_metric(key, values, frames=None):
        frames = np.arange(len(values), dtype=np.int32) if frames is None else np.asarray(frames, dtype=np.int32)
        return FrameMetricResult(key, frames, frames / fps, np.asarray(values, dtype=np.float32), made)

    def result(name, metrics):
        info = VideoInfo(path=Path(name), width=1920, height=1080, fps=fps, duration=10 / fps, nb_frames=10,
                         codec_name="h264")
        return ComparisonResult(source=Path("source.mkv"), distorted=Path(name), frames=[], fps=fps,
                                model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None,
                                source_info=info, distorted_info=info, metric_results=MetricResultSet(metrics))

    inf = float("inf")
    runs = [
        ("a", result("a.mkv", [
            frame_metric("vmaf", [95, 95, 95, 95, 95, 30, 95, 95, 95, 95]),
            frame_metric("psnr", [45, 45.5, 44, 43, 42, 31.25, 44, 45, 46, 47]),
            frame_metric("ssim", [0.99, 0.991, 0.989, 0.98, 0.97, 0.9, 0.98, 0.99, 0.995, 0.99]),
            frame_metric("xpsnr", [40, inf, 38, inf, 41, 39, 40, 110, float("nan"), 42]),
            frame_metric("butteraugli", [float(v) for v in range(10)]),
            SequenceMetricResult("cvvdp", 8.75, made, np.array([0, 3, 6], dtype=np.int32),
                                 np.array([0.0, 0.1, 0.2]), np.array([8.5, 9.0, 8.0], dtype=np.float32)),
        ])),
        ("b", result("b.mkv", [
            frame_metric("vmaf", [95, 95, 95, 95, 95, 95, 95, 95, 40, 95]),
            frame_metric("psnr", [44, 40, 39, 41, 42], frames=[0, 2, 4, 6, 8]),
        ])),
        ("short", result("short.mkv", [frame_metric("vmaf", [80, 81, 82])])),
    ]
    panel = GraphPanel()
    for label, run in runs:
        panel.add_run(run, label)
    ids = {entry.label: sid for sid, entry in panel._entries.items()}

    def table():
        headers = [panel.stats_table.horizontalHeaderItem(c).text() for c in range(panel.stats_table.columnCount())]
        cells = [[(panel.stats_table.item(r, c).text() if panel.stats_table.item(r, c) is not None else None)
                  for c in range(panel.stats_table.columnCount())] for r in range(panel.stats_table.rowCount())]
        return {"headers": headers, "cells": cells}

    pages = []
    hovers = [(5 / fps, 70.0), (5 / fps, 99.0), (8 / fps, 60.0), (0.0, 0.0), (0.25, 50.0), (2.0, 50.0)]
    for index, metric in enumerate(METRICS):
        panel.tabs.setCurrentIndex(index)
        page = panel._pages[metric.key]
        page.chart.resize(800, 400)
        case = {"key": metric.key, "table": table(), "frame_range": [panel.frame_spin.minimum(),
                                                                     panel.frame_spin.maximum()]}
        readouts = []
        for hidden in ([], ["b"]):
            for label in hidden:
                panel.set_series_visible(ids[label], False)
            for x, y in hovers:
                page.on_hover(x, y, panel._entries)
                readouts.append({"hidden": hidden, "x": exact(x), "y": exact(y), "text": page.hover_label.text()})
            for frame in (1, 5, 7, 20):
                page.show_frame(frame, panel._entries)
                readouts.append({"hidden": hidden, "frame": frame, "text": page.hover_label.text()})
            for label in hidden:
                panel.set_series_visible(ids[label], True)
        case["readouts"] = readouts
        pages.append(case)
    panel.deleteLater()
    del app
    return {"runs": [{"label": label, "result": _result_json(run)} for label, run in runs], "pages": pages}


def chart_cases() -> dict:
    """The graph's chart (ui/chart.py), from the Python app's own widget at
    800x400: the tick spacings for spans of every size, and for lines of
    each kind (a bounded scale, one above its ceiling, decibels, an inverted
    scale, all NaN, two hours, a single point) the value axis's range, the
    ticks' labels and rows, and the mapping between pixels, times and
    values."""
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from vmaf_app.ui import chart

    app = QApplication.instance() or QApplication(["make_reference", "-platform", "offscreen"])
    spans = [0.0, -1.0, 0.004, 0.5, 1.0, 3.7, 8.0, 10.0, 59.9, 60.0, 61.0, 600.0, 1234.5, 7200.0, 14400.0,
             115200.0, 130000.0, 500000.0, 1e-7, 0.0123, 0.3, 2.5, 7.9, 8.1, 16.0, 39.99, 40.0, 100.0, 777.0]
    steps = [{"span": exact(span), "time": exact(chart._nice_time_step(span)),
              "value": exact(chart._nice_value_step(span))} for span in spans]

    fps = 24000 / 1001
    frames = np.arange(300)
    lines = {
        "VMAF": (100.0, False, frames / fps, 80 + 20 * np.sin(frames / 9.0)),
        "VMAF v1 above 100": (100.0, False, frames[:3] / fps, [99.0, 105.0, 110.0]),
        "PSNR": (None, False, frames / fps, 37.5 + 7.5 * np.cos(frames / 13.0)),
        "a flat line": (None, False, frames / fps, np.full(300, 42.0)),
        "Butteraugli, inverted": (None, True, frames[:3] / fps, [0.0, 1.5, 3.0]),
        "all NaN": (100.0, False, frames / fps, np.full(300, np.nan)),
        "two hours": (100.0, False, np.arange(172_800) / 24.0, np.full(172_800, 95.0)),
        "a single point": (None, False, np.array([4.0]), [12.5]),
    }
    charts = []
    for name, (fixed_y_max, inverted, times, values) in lines.items():
        widget = chart.ChartWidget(y_axis_label="axis", fixed_y_max=fixed_y_max, invert_y=inverted)
        widget.resize(800, 400)
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float32)
        widget.set_series(1, chart.ChartSeries(times=times, values=values, color="#4C72B0"))
        rect = widget.plot_rect()
        y0, y1 = widget.y_range()
        x0, x1 = widget.x_range()
        charts.append({
            "name": name, "fixed_y_max": fixed_y_max, "inverted": inverted,
            # The longest line by its pattern, the others in full.
            "times": {"count": len(times), "per_second": 24.0} if name == "two hours"
            else [exact(float(t)) for t in times],
            "values": {"count": len(values), "value": 95.0} if name == "two hours"
            else [exact(float(v)) for v in values],
            "plot_rect": [rect.left(), rect.top(), rect.width(), rect.height()],
            "y_range": [exact(y0), exact(y1)], "x_range": [exact(x0), exact(x1)],
            "time_labels": widget.time_tick_labels(),
            "value_ticks": [[row, label] for row, label in widget.value_ticks()],
            "pixels_for_times": [[exact(t), widget.pixel_for_time(t)]
                                 for t in (x0, x1, (x0 + x1) / 2, x0 + (x1 - x0) / 3)],
            "times_at": [[px, exact(widget.time_at(px))] for px in (rect.left(), rect.right(), 400, 123)],
            "values_at": [[py, exact(widget.value_at(py))] for py in (rect.top(), rect.bottom(), 200, 77)],
            "seconds_per_pixel": exact(widget.seconds_per_pixel()),
        })
        widget.deleteLater()
    del app
    return {"steps": steps, "charts": charts}


def i18n_cases() -> dict:
    """The window's text as vmaf_app/i18n.py gives it: the translation for
    Windows' language names, each language's plural forms, str.format over
    the format specs the catalogs use, and in I18N_LANGUAGES the texts with
    fields, the counted texts, and the engine's messages (one of each
    template the C++ window has, and the ways the window meets them:
    trimmed, with "..." for "…", one inside another, several lines)."""
    from vmaf_app import i18n

    from scripts.i18n_catalog import TRANSLATIONS_DIR, keys
    from scripts.i18n_catalog import templates as window_templates

    i18n.TRANSLATIONS_DIR = TRANSLATIONS_DIR  # this repository's catalogs, not the checkout's

    strings, plurals = keys()
    counts = [*range(230), 1000, 1001, 1002, 1011, 1012, 1021, 1100, 1111, 10**6, 10**6 + 1, 2**31 + 1]
    fielded = sorted(key for key in strings if "{" in key)
    counted = sorted(plurals)
    # The Python app's templates the C++ window has, in the Python app's order:
    # its playback's GStreamer messages, for one, the C++ app never says.
    shapes = set(window_templates())
    assert shapes <= set(i18n.MESSAGE_TEMPLATES), shapes - set(i18n.MESSAGE_TEMPLATES)
    templates = [template for template in i18n.MESSAGE_TEMPLATES if template in shapes]
    plain = next(template for template in templates if "{" not in template)
    messages = [_i18n_message(template) for template in templates]
    messages += [i18n._bare(message) for message in messages if i18n._bare(message) != message]
    messages += [message[:-1] + "..." for message in messages if message.endswith("…")]
    messages += [_i18n_message(template, {field: plain for field in ("error", "reason", "detail")})
                 for template in templates if re.search(r"\{(error|reason|detail)\}", template)]
    messages += ["\n".join([messages[0], "", messages[1], "  ", "Not a message the window knows."]),
                 "Ready.", "", "Not a message the window knows."]

    languages = {}
    for code in I18N_LANGUAGES:
        assert i18n.set_language(code) == code
        languages[code] = {
            "fielded": [i18n.tr(key, **_i18n_values(key)) for key in fielded],
            "counted": [[i18n.ntr(key, plurals[key], count, **{field: value for field, value in
                                                                  _i18n_values(key).items() if field != "count"})
                         for count in I18N_COUNTS] for key in counted],
            "messages": [i18n.tr_message(message) for message in messages],
        }
    i18n.set_language("en")
    return {
        "windows": {name: i18n.language_for(name) for name in I18N_WINDOWS_NAMES},
        "counts": counts,
        # The form for each count, one digit each.
        "plural": {code: "".join(str(i18n.plural_index(code, count)) for count in counts) for code in i18n.LANGUAGES},
        "formats": [{"text": text, "values": values, "formatted": text.format(**values)}
                    for text, values in I18N_FORMATS],
        "fielded": [{"text": key, "values": _i18n_values(key)} for key in fielded],
        "counted": [{"singular": key, "plural": plurals[key],
                     "values": {field: value for field, value in _i18n_values(key).items() if field != "count"}}
                    for key in counted],
        "counted_counts": list(I18N_COUNTS),
        "messages": messages,
        "languages": languages,
    }


def app_log_cases() -> dict:
    """Copy log's text (core/app_log.py log_text_to_share), from the Python
    app's own function, for logs as its files hold them: sessions with and
    without a run, a run in an older session or an older file, a log cut to
    the limit (with characters past ASCII, which the limit counts as one
    each, and a carriage return inside a line), and no log at all."""
    import tempfile

    from vmaf_app.core import app_log

    def line(time: str, level: str, thread: str, name: str, text: str) -> str:
        return f"2026-10-14 {time}.123 {level:<7} [{thread}] {name}: {text}"

    def session(time: str, *texts: str) -> list[str]:
        lines = [line(time, "INFO", "MainThread", "vmaf_app.main", f"==== VideoMetricsLab{app_log.SESSION_START}"),
                 line(time, "INFO", "MainThread", "vmaf_app.main", "VideoMetricsLab 2.1.0 (from source)")]
        return lines + [line(time, "INFO", "vmaf-cpu-0", "vmaf_app.core.job_runner", text) for text in texts]

    run = f"{app_log.RUN_START} 2 video(s), CPU metrics for up to 1 at once, GPU metrics one video at a time"
    first = session("09:00:00", run, "Video 1 'a': started", "Video 1 'a' finished: VMAF", "Run ended")
    second = session("10:00:00", "Run status: nothing to run")
    long_session = session("11:00:00", run, *[f"Video {n} 'Bühne – 4K {n}.mkv': VMAF done in 0:00:{n:02d}"
                                              for n in range(1, 40)], "FFmpeg: frame=1\rframe=2", "Run ended")
    cases = []

    def case(name: str, files: dict[str, list[str]], limit: int = app_log.SHARE_LIMIT) -> None:
        contents = {file: "\r\n".join(lines) + "\r\n" for file, lines in files.items()}
        with tempfile.TemporaryDirectory() as folder:
            for file, text in contents.items():
                (Path(folder) / file).write_bytes(text.encode("utf-8"))
            shared = app_log.log_text_to_share(Path(folder), limit)
        cases.append({"name": name, "files": contents, "limit": limit,
                      "text": None if shared is None else shared[0],
                      "shortened": None if shared is None else shared[1]})

    case("no log", {})
    case("a session with a run", {"VideoMetricsLab.log": first})
    case("the run in an older session", {"VideoMetricsLab.log": first + second})
    case("no run", {"VideoMetricsLab.log": second + second})
    case("the run in an older file",
         {"VideoMetricsLab.log.2": first, "VideoMetricsLab.log.1": second, "VideoMetricsLab.log": second})
    case("cut to the limit", {"VideoMetricsLab.log": first + long_session}, limit=1500)
    case("a head longer than a quarter of the limit", {"VideoMetricsLab.log": long_session}, limit=400)
    # The limit counts characters: between this session's length in
    # characters and in UTF-8 bytes, it is shared whole.
    japanese = session("12:00:00", run, *[f"Video {n} '映像のエンコード {n}.mkv': finished" for n in range(1, 9)])
    joined = "\n".join(japanese)
    case("characters past ASCII count one each", {"VideoMetricsLab.log": japanese},
         limit=(len(joined) + len(joined.encode("utf-8"))) // 2)
    return {"cases": cases}


def update_check_cases() -> dict:
    """The update check (core/update_check.py), from the Python app's own
    functions: versions compared as numbers, and GitHub's answers read into
    a release, or refused (bodies that are not JSON, name no version, or a
    tag that is not one)."""
    import json as json_module

    from vmaf_app.core import update_check

    versions = [("v1.3", "1.2.1"), ("v1.10", "1.9"), ("v1.2.1", "1.2.1"), ("v1.3.0", "1.3"), ("v1.2", "1.3"),
                ("nightly", "1.3"), ("2.0", "3.0.0-dev"), ("v3.0", "2.0"), (" v1.0 ", "1"), ("v0", "0.0.0"),
                ("v1..2", "1"), ("", "1"), ("v1.2.3.4", "1.2.3"), ("V1.4", "1.3"), ("v1.3a", "1.2"),
                ("v-1", "0"), ("v+1", "0"), ("1.03", "1.2"), ("vv1", "0")]
    version_cases = [{"candidate": candidate, "current": current,
                      "newer": update_check.is_newer(candidate, current),
                      "tuple": None if (found := update_check.version_tuple(candidate)) is None else list(found)}
                     for candidate, current in versions]

    class _Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def read(self) -> bytes:
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    bodies = [
        {"tag_name": "v1.4", "html_url": "https://github.com/x/releases/tag/v1.4", "body": "- Added a thing."},
        {"tag_name": "v2.0", "body": None},
        {"tag_name": " v1.3.0 ", "html_url": "", "body": "notes"},
        {"tag_name": "v1.5", "body": "Neu: B\u00fchne\r\n- und mehr"},
        {"tag_name": 5},
        {"tag_name": "v1", "html_url": 5},
        {"tag_name": "nightly"},
        {},
        [],
        {"tag_name": "V1.2"},
        {"name": "v1.2"},
    ]
    texts = [json_module.dumps(body) for body in bodies] + ["not json", "", "{\"tag_name\": \"v1.2\""]
    answer_cases = []
    for body in texts:
        try:
            release = update_check.latest_release(opener=lambda request, timeout, b=body: _Response(b.encode()))
            answer_cases.append({"body": body, "release": {"version": release.version, "page_url": release.page_url,
                                                           "notes": release.notes}})
        except update_check.UpdateCheckError:
            answer_cases.append({"body": body, "release": None})
    return {"versions": version_cases, "answers": answer_cases, "url": update_check.RELEASES_URL,
            "timeout": update_check.TIMEOUT_SECONDS}


def frame_extract_cases() -> dict:
    """Video Compare's still frames (core/frame_extract.py), from the Python
    app's own functions: each side's filter chain and FFmpeg command for
    pairs with crops, either scale direction, resolution tests and black bars
    still to be detected, in every colour mode; the seek times; the compared
    size; the HDR family of each video; the SDR white level used."""
    from dataclasses import replace as dc_replace

    from vmaf_app.core import frame_extract as fe
    from vmaf_app.core.models import CropBox, ResampleTarget, ScaleDirection, VideoInfo

    def video(name: str, width: int, height: int, pix_fmt: str, *, fps: float = 24000 / 1001, sar: str = "1:1",
              primaries: str = "", space: str = "", transfer: str = "", color_range: str = "",
              start_offset: float = 0.0) -> VideoInfo:
        return VideoInfo(path=Path("C:/clips") / name, width=width, height=height, fps=fps, duration=60.0,
                         nb_frames=1438, codec_name="hevc", sar=sar, pix_fmt=pix_fmt, color_primaries=primaries,
                         color_space=space, color_transfer=transfer, color_range=color_range,
                         start_offset=start_offset)

    videos = {
        "source_hdr": video("source hdr.mkv", 3840, 2160, "yuv420p10le", primaries="bt2020", space="bt2020nc",
                            transfer="smpte2084", color_range="tv", start_offset=0.032),
        "encode_sdr": video("encode.mp4", 1920, 1080, "yuv420p", primaries="bt709", space="bt709",
                            transfer="bt709", color_range="tv"),
        "encode_hdr_lost_tags": video("lost.mkv", 3840, 2160, "yuv420p10le", primaries="unknown", space="",
                                      transfer=" SMPTE2084 ", color_range="unspecified"),
        "encode_hlg": video("hlg.mkv", 1920, 1080, "yuv420p10le", primaries="bt2020", space="bt2020nc",
                            transfer="arib-std-b67", color_range="pc"),
        "encode_untagged_10bit": video("untagged.mkv", 2560, 1440, "yuv420p10le"),
        "encode_anamorphic": video("anamorphic.mkv", 1440, 1080, "yuv420p", sar="4:3", transfer="bt709"),
        "source_letterboxed": video("letterboxed.mkv", 1920, 1080, "yuv420p", transfer="bt709"),
        "encode_scope": video("scope.mkv", 1920, 816, "yuv420p", transfer="bt709"),
        "encode_flat": video("flat.mkv", 1920, 1038, "yuv420p", transfer="bt709"),
    }
    comparisons = {
        "hdr source, sdr encode": fe.FrameComparison(videos["source_hdr"], videos["encode_sdr"],
                                                     fps=videos["source_hdr"].fps, frame_count=1438),
        "test scaled to the source": fe.FrameComparison(videos["source_hdr"], videos["encode_sdr"],
                                                        scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
                                                        scale_algorithm="lanczos", fps=24000 / 1001,
                                                        frame_count=1438),
        "cropped both": fe.FrameComparison(videos["source_hdr"], videos["encode_hdr_lost_tags"],
                                           source_crop=CropBox(3840, 1600, 0, 280),
                                           distorted_crop=CropBox(3838, 1600, 2, 280), fps=24000 / 1001,
                                           frame_count=1438),
        "a crop that keeps everything": fe.FrameComparison(videos["encode_hlg"], videos["encode_hlg"],
                                                           source_crop=CropBox(1920, 1080, 0, 0), fps=25.0,
                                                           frame_count=100),
        "resolution test": fe.FrameComparison(videos["source_hdr"], videos["source_hdr"],
                                              source_crop=CropBox(3840, 1600, 0, 280),
                                              resample_target=ResampleTarget(1280, 720), fps=24000 / 1001,
                                              frame_count=1438),
        "untagged and anamorphic": fe.FrameComparison(videos["encode_untagged_10bit"],
                                                      videos["encode_anamorphic"], fps=60.0, frame_count=600),
        "black bars still to detect": fe.FrameComparison(videos["source_letterboxed"], videos["encode_scope"],
                                                         fps=24.0, frame_count=240, auto_crop_pending=True),
        "black bars still to detect, same shape": fe.FrameComparison(videos["encode_sdr"], videos["source_letterboxed"],
                                                                     fps=24.0, frame_count=240, auto_crop_pending=True),
        # 16:9 against 1.85:1, shapes 4% apart: fitted, not stretched.
        "black bars still to detect, nearly the same shape": fe.FrameComparison(
            videos["source_letterboxed"], videos["encode_flat"], fps=24.0, frame_count=240, auto_crop_pending=True),
    }
    colours = {
        "display-aware, Windows HDR on at 203 nits": fe.PreviewColorSettings(fe.PreviewColorMode.DISPLAY_AWARE, True, 203.0),
        "display-aware, Windows HDR on at 30 nits": fe.PreviewColorSettings(fe.PreviewColorMode.DISPLAY_AWARE, True, 30.0),
        "display-aware, Windows HDR off": fe.PreviewColorSettings(fe.PreviewColorMode.DISPLAY_AWARE, False, 240.0),
        "display-aware, unknown": fe.PreviewColorSettings(),
        "HDR to SDR": fe.PreviewColorSettings(fe.PreviewColorMode.HDR_TO_SDR, True, 203.0),
        "unmanaged": fe.PreviewColorSettings(fe.PreviewColorMode.UNMANAGED, True, 203.0),
    }

    def info_json(info: VideoInfo) -> dict:
        return {"path": str(info.path), "width": info.width, "height": info.height, "fps": exact(info.fps),
                "sar": info.sar, "pix_fmt": info.pix_fmt, "color_primaries": info.color_primaries,
                "color_space": info.color_space, "color_transfer": info.color_transfer,
                "color_range": info.color_range, "start_offset": exact(info.start_offset)}

    def crop_json(crop):
        return None if crop is None else [crop.w, crop.h, crop.x, crop.y]

    real_ffmpeg = fe.ffmpeg_path
    fe.ffmpeg_path = lambda: "<ffmpeg>"
    try:
        comparison_cases = []
        for name, comparison in comparisons.items():
            sides = []
            for side in ("source", "distorted"):
                filters = {colour: fe.frame_filter(comparison, side, settings) for colour, settings in colours.items()}
                commands = {str(frame): fe.build_frame_command(comparison, side, frame, colours["display-aware, unknown"])
                            for frame in (0, 1, 1000)}
                sides.append({"side": side, "filters": filters, "commands": commands})
            comparison_cases.append({
                "name": name, "source": info_json(comparison.source_info),
                "distorted": info_json(comparison.distorted_info),
                "source_crop": crop_json(comparison.source_crop), "distorted_crop": crop_json(comparison.distorted_crop),
                "scale_direction": comparison.scale_direction.value, "scale_algorithm": comparison.scale_algorithm,
                "resample_width": None if comparison.resample_target is None else comparison.resample_target.width,
                "fps": exact(comparison.fps), "frame_count": comparison.frame_count,
                "auto_crop_pending": comparison.auto_crop_pending,
                "dimensions": list(fe.comparison_dimensions(comparison)), "sides": sides,
            })
        no_rate = dc_replace(comparisons["hdr source, sdr encode"], fps=0.0)
        try:
            fe.build_frame_command(no_rate, "source", 0)
            no_rate_error = None
        except fe.FrameExtractError as error:
            no_rate_error = str(error)
    finally:
        fe.ffmpeg_path = real_ffmpeg
    return {
        "comparisons": comparison_cases,
        "colours": [{"name": name, "mode": settings.mode.value, "hdr": settings.display_hdr_enabled,
                     "white": settings.display_sdr_white_nits, "target_nits": exact(settings.target_nits)}
                    for name, settings in colours.items()],
        "hdr_kinds": [{"video": info_json(info), "kind": fe.hdr_kind(info)} for info in videos.values()],
        "seeks": [{"video": info_json(info), "frame": frame, "fps": exact(fps),
                   "seconds": exact(fe.seek_seconds(info, frame, fps))}
                  for info in (videos["source_hdr"], videos["encode_sdr"])
                  for frame, fps in ((0, 24000 / 1001), (1, 24000 / 1001), (3, 120.0), (86399, 24.0))],
        "no_frame_rate": no_rate_error,
    }


# ------------------------------------------------------------------ playback

def playback_cases() -> dict:
    """Video Compare's playback with FFmpeg (core/video_playback.py), from the
    Python app's own functions: each video's decoding command, as a decoder
    worker runs it (one video, real time, unpaced), for pairs with HDR, crops,
    black bars still to detect and untagged video, both sides and the source
    as it is played (native and scaled down), in every processing mode and
    colour mode, fitted to a view and at full size; the sizes decoded at; the
    test videos decoded beside the one shown; and the soundtrack's command."""
    from vmaf_app.core import frame_extract as fe
    from vmaf_app.core import video_playback as vp
    from vmaf_app.core.gpu import HwAccelPlan
    from vmaf_app.core.models import CropBox, ScaleDirection, VideoInfo

    def video(name: str, width: int, height: int, pix_fmt: str, codec: str, *, fps: float = 24000 / 1001,
              primaries: str = "", space: str = "", transfer: str = "", color_range: str = "",
              start_offset: float = 0.0) -> VideoInfo:
        return VideoInfo(path=Path("C:/clips") / name, width=width, height=height, fps=fps, duration=60.0,
                         nb_frames=1438, codec_name=codec, pix_fmt=pix_fmt, color_primaries=primaries,
                         color_space=space, color_transfer=transfer, color_range=color_range,
                         start_offset=start_offset)

    hdr = video("source hdr.mkv", 3840, 2160, "yuv420p10le", "hevc", primaries="bt2020", space="bt2020nc",
                transfer="smpte2084", color_range="tv", start_offset=0.032)
    sdr = video("encode.mp4", 1920, 1080, "yuv420p", "h264", primaries="bt709", space="bt709", transfer="bt709",
                color_range="tv")
    lost = video("lost tags.mkv", 3840, 2160, "yuv420p10le", "av1", primaries="unknown", transfer=" SMPTE2084 ",
                 color_range="unspecified")
    untagged = video("untagged.mkv", 2560, 1440, "yuv420p10le", "hevc")
    letterboxed = video("letterboxed.mkv", 1920, 1080, "yuv420p", "h264", transfer="bt709")
    scope = video("scope.mkv", 1920, 816, "yuv420p", "h264", transfer="bt709")
    comparisons = {
        "hdr source, sdr encode": fe.FrameComparison(hdr, sdr, fps=hdr.fps, frame_count=1438),
        "test scaled to the source": fe.FrameComparison(hdr, sdr, scale_direction=ScaleDirection.DISTORTED_TO_SOURCE,
                                                        scale_algorithm="lanczos", fps=24000 / 1001, frame_count=1438),
        "cropped both": fe.FrameComparison(hdr, lost, source_crop=CropBox(3840, 1600, 0, 280),
                                           distorted_crop=CropBox(3838, 1600, 2, 280), fps=24000 / 1001,
                                           frame_count=1438),
        "untagged": fe.FrameComparison(untagged, sdr, fps=60.0, frame_count=600),
        "black bars still to detect": fe.FrameComparison(letterboxed, scope, fps=24.0, frame_count=240,
                                                         auto_crop_pending=True),
    }
    colours = {
        "display-aware, unknown": fe.PreviewColorSettings(),
        "HDR to SDR": fe.PreviewColorSettings(fe.PreviewColorMode.HDR_TO_SDR, True, 203.0),
        "unmanaged": fe.PreviewColorSettings(fe.PreviewColorMode.UNMANAGED, True, 203.0),
    }
    plans = {"cuda, d3d11va": HwAccelPlan("cuda", "d3d11va"), "qsv, none": HwAccelPlan("qsv", None)}

    def info_json(info: VideoInfo) -> dict:
        return {"path": str(info.path), "width": info.width, "height": info.height, "fps": exact(info.fps),
                "codec_name": info.codec_name, "pix_fmt": info.pix_fmt, "color_primaries": info.color_primaries,
                "color_space": info.color_space, "color_transfer": info.color_transfer,
                "color_range": info.color_range, "start_offset": exact(info.start_offset)}

    def comparison_json(comparison) -> dict:
        def crop(box):
            return None if box is None else [box.w, box.h, box.x, box.y]
        return {"source": info_json(comparison.source_info), "distorted": info_json(comparison.distorted_info),
                "source_crop": crop(comparison.source_crop), "distorted_crop": crop(comparison.distorted_crop),
                "scale_direction": comparison.scale_direction.value, "scale_algorithm": comparison.scale_algorithm,
                "fps": exact(comparison.fps), "frame_count": comparison.frame_count,
                "auto_crop_pending": comparison.auto_crop_pending}

    real_ffmpeg, real_ffplay = vp.ffmpeg_path, vp.ffplay_path
    vp.ffmpeg_path = lambda: "<ffmpeg>"
    vp.ffplay_path = lambda: Path("C:/tools/ffplay.exe")
    try:
        commands = []
        for name, comparison in comparisons.items():
            # How the source is played is given with the comparison: the port
            # makes it with its own source_playback_comparison, as its view does.
            played = {"source as played": (vp.source_playback_comparison(comparison), "source", "native"),
                      "source scaled down": (vp.source_playback_comparison(comparison, False), "source",
                                             "scaled down"),
                      "test video": (comparison, "distorted", None)}
            for which, (recipe, side, source_played) in played.items():
                # The second plan only where the plan is used; the view's size
                # only in the usual colour mode.
                runs = [(plan_name, mode, colour_name, maximum)
                        for plan_name, modes in (("cuda, d3d11va", ("interop", "transfer", "software", "cpu")),
                                                 ("qsv, none", ("transfer",)))
                        for mode in modes
                        for colour_name, maximum in (("display-aware, unknown", None),
                                                     ("display-aware, unknown", (1280, 720)),
                                                     ("HDR to SDR", None), ("unmanaged", None))]
                for plan_name, mode, colour_name, maximum in runs:
                    plan = plans[plan_name]
                    command = vp.build_video_series_command(
                        [recipe], 7, colours[colour_name], [plan], maximum, realtime=True, processing=mode,
                        side=side, paced=False)
                    commands.append({
                        "name": f"{name}: {which}, {plan_name}, {mode}, {colour_name}, {maximum}",
                        "comparison": comparison_json(comparison), "played": source_played,
                        "side": side, "start_frame": 7,
                        "plan": [plan.source, plan.distorted], "processing": mode,
                        "colour": colour_name, "maximum": None if maximum is None else list(maximum),
                        "command": command, "size": list(vp.playback_dimensions(recipe, maximum))})
        audio = [{"comparison": comparison_json(comparison), "frame": frame,
                  "command": vp.build_audio_command(comparison, frame)}
                 for comparison in comparisons.values() for frame in (0, 1001)]
    finally:
        vp.ffmpeg_path, vp.ffplay_path = real_ffmpeg, real_ffplay
    neighbours = [{"count": count, "selected": selected, "width": width,
                   "indices": list(vp.neighbour_indices(count, selected, width))}
                  for count in (0, 1, 2, 5) for selected in (-1, 0, 1, 4) for width in (0, 1, 3, 9)]
    return {"colours": [{"name": name, "mode": settings.mode.value, "hdr": settings.display_hdr_enabled,
                         "white": settings.display_sdr_white_nits} for name, settings in colours.items()],
            "commands": commands, "audio": audio, "neighbours": neighbours}


# ------------------------------------------------------------ native playback

def native_cases() -> dict:
    """Video Compare's native playback decisions (core/gstreamer_playback.py,
    locked_presentation.py, d3d11_tonemap.py), from the Python app's own
    functions: which comparisons play natively and why not, what the
    presenter's shader does to each video's colours and with which
    parameters, and the YUV-to-RGB rows for each matrix, range and depth.
    Floats are given as the presenter takes them (C floats)."""
    import ctypes

    from vmaf_app.core import d3d11_tonemap
    from vmaf_app.core import frame_extract as fe
    from vmaf_app.core import gstreamer_playback as gp
    from vmaf_app.core.locked_presentation import yuv_to_rgb
    from vmaf_app.core.models import VideoInfo

    def floats(values) -> list:
        return [exact(value) for value in (ctypes.c_float * len(values))(*values)]

    def video(name: str, primaries: str, transfer: str, space: str, pix_fmt: str = "yuv420p10le") -> VideoInfo:
        return VideoInfo(path=Path("C:/clips") / name, width=3840, height=2160, fps=24.0, duration=10.0,
                         nb_frames=240, codec_name="hevc", pix_fmt=pix_fmt, color_primaries=primaries,
                         color_space=space, color_transfer=transfer)

    videos = {
        "sdr": video("sdr.mkv", "bt709", "bt709", "bt709", "yuv420p"),
        "untagged": video("untagged.mkv", "", "", ""),
        "transfer unspecified": video("unspecified.mkv", "bt709", "unspecified", "bt709", "yuv420p"),
        "hdr10": video("hdr10.mkv", "bt2020", "smpte2084", "bt2020nc"),
        "hdr10, primaries untagged": video("hdr10 untagged.mkv", "", "smpte2084", ""),
        "hdr10, primaries unknown": video("hdr10 unknown.mkv", "unknown", "smpte2084", "bt2020nc"),
        "hlg, primaries unspecified": video("hlg unspecified.mkv", "unspecified", "arib-std-b67", "bt2020nc"),
        "hdr10 in display p3": video("p3.mkv", "smpte432", "smpte2084", "bt2020nc"),
        "hdr10 in dci-p3": video("dci.mkv", "smpte431", "smpte2084", "bt2020nc"),
        "hdr10 in bt709": video("hdr709.mkv", "bt709", "smpte2084", "bt709"),
        "hlg": video("hlg.mkv", "bt2020", "arib-std-b67", "bt2020nc"),
        "hlg in bt709": video("hlg709.mkv", "bt709", "arib-std-b67", "bt709"),
    }
    settings = {
        "display-aware, unknown": fe.PreviewColorSettings(),
        "display-aware, HDR on": fe.PreviewColorSettings(fe.PreviewColorMode.DISPLAY_AWARE, True, 203.0),
        "display-aware, HDR off": fe.PreviewColorSettings(fe.PreviewColorMode.DISPLAY_AWARE, False, 80.0),
        "HDR to SDR": fe.PreviewColorSettings(fe.PreviewColorMode.HDR_TO_SDR, True, 203.0),
        "unmanaged": fe.PreviewColorSettings(fe.PreviewColorMode.UNMANAGED, True, 203.0),
    }

    def info_json(info: VideoInfo) -> dict:
        return {"name": info.path.name, "pix_fmt": info.pix_fmt, "color_primaries": info.color_primaries,
                "color_space": info.color_space, "color_transfer": info.color_transfer}

    def settings_json(value) -> dict:
        return {"mode": value.mode.value, "hdr": value.display_hdr_enabled, "white": value.display_sdr_white_nits}

    shadings = [{"video": info_json(info), "colours": settings_json(value), "shading": gp._shading(info, value)}
                for info in videos.values() for value in settings.values()]
    # As where GStreamer and the presenter are there: native playback is then
    # refused only for the comparison's colours.
    real_available, real_presents = gp.gstreamer_available, d3d11_tonemap.presents
    gp.gstreamer_available = lambda: (True, "")
    d3d11_tonemap.presents = lambda: True
    try:
        problems = []
        for source_name, distorted_name in [("hdr10", name) for name in videos] + [(name, "sdr") for name in videos]:
            comparison = fe.FrameComparison(videos[source_name], videos[distorted_name], fps=24.0, frame_count=240)
            for value in settings.values():
                usable, reason = gp.uses_native_gstreamer(comparison, value)
                problems.append({"source": info_json(comparison.source_info),
                                 "distorted": info_json(comparison.distorted_info),
                                 "colours": settings_json(value), "problem": None if usable else reason})
        parameters = [{"kind": kind, "primaries": primaries, "hdr": hdr,
                       "parameters": floats(d3d11_tonemap.shading(kind, primaries, hdr=hdr))}
                      for kind in ("HDR10 / PQ", "HLG")
                      for primaries in ("bt709", "bt2020", "smpte432", "", "unknown", "unspecified", "BT.2020")
                      for hdr in (False, True)]
    finally:
        gp.gstreamer_available, d3d11_tonemap.presents = real_available, real_presents
    rows = [{"matrix": matrix, "full_range": full_range, "ten_bit": ten_bit,
             "rows": floats(yuv_to_rgb(matrix, full_range, ten_bit))}
            for matrix in ("bt709", "bt601", "bt2020", "smpte240m", "fcc", "bt470m")
            for full_range in (False, True) for ten_bit in (False, True)]
    return {"shadings": shadings, "problems": problems, "parameters": parameters, "rows": rows}


# ------------------------------------------------------------------- bitrate

def bitrate_cases() -> dict:
    """The Bitrate tab's numbers (core/bitrate.py), from the Python app's own
    functions: ffprobe's lines of packets read by analyze_video_bitrate (fed
    through a stand-in process: B-frames in decode order, missing, odd and
    repeated values, lines that are not packets, none at all), and each
    stream's graphs per frame, second and group of pictures, its one-second
    bins and its summary, with the start moved to zero and not."""
    import io

    import numpy as np
    from vmaf_app.core import bitrate as br
    from vmaf_app.core.models import VideoInfo

    class StandIn:
        """A finished ffprobe that wrote `lines`."""

        pid = 0

        def __init__(self, lines: list[str]) -> None:
            self.stdout = iter(lines)
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0

    def read(lines: list[str], fps: float) -> dict:
        real_popen = br.proc_util.popen
        br.proc_util.popen = lambda *_args, **_kwargs: StandIn(lines)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "clip.mkv"
                path.write_bytes(b"")
                info = VideoInfo(path=path, width=1920, height=1080, fps=fps, duration=10.0, nb_frames=240,
                                 codec_name="h264")
                data = br.analyze_video_bitrate(info)
        except br.BitrateError as error:
            return {"error": str(error)}
        finally:
            br.proc_util.popen = real_popen
        return {"times": [exact(t) for t in data.times], "durations": [exact(d) for d in data.durations],
                "sizes": [int(s) for s in data.sizes], "keyframes": [bool(k) for k in data.keyframes]}

    frame = 1001 / 24000
    reads = [
        {"name": "B-frames in decode order", "fps": 24000 / 1001, "lines": [
            f"pts_time={0:.6f}|dts_time={-2 * frame:.6f}|duration_time={frame:.6f}|size=5000|pos=48|flags=K__\n",
            f"pts_time={3 * frame:.6f}|dts_time={-frame:.6f}|duration_time={frame:.6f}|size=900|pos=5048|flags=___\n",
            f"pts_time={frame:.6f}|dts_time={0:.6f}|duration_time={frame:.6f}|size=300|pos=5948|flags=___\n",
            f"pts_time={2 * frame:.6f}|dts_time={frame:.6f}|duration_time={frame:.6f}|size=310|pos=6248|flags=___\n",
            f"pts_time={4 * frame:.6f}|dts_time={2 * frame:.6f}|duration_time={frame:.6f}|size=4800|pos=6558|flags=K_\n",
        ]},
        {"name": "missing, odd and repeated values", "fps": 25.0, "lines": [
            "pts_time=N/A|dts_time=0.040000|duration_time=0.040000|size=1000|pos=100|flags=K__\n",
            "pts_time=N/A|dts_time=N/A|duration_time=N/A|size=1100|pos=N/A|flags=___\n",
            "pts_time=inf|dts_time=nan|duration_time=0|size=1200|flags=___\n",
            "pts_time=0.200000|dts_time=0.120000|duration_time=-0.5|size= 1300 |pos=400\n",
            "  pts_time=0.200000|dts_time=0.160000|duration_time=0.040000|size=1400|pos=300|flags=__ \r\n",
            "pts_time=0.200000|duration_time=0.040000|size=10|size=1500|pos=300|flags=K\n",
            "pts_time=0.300000|dts_time=0.240000|duration_time=0.040000|size=N/A|pos=600|flags=K__\n",
            "pts_time=0.300000|dts_time=0.240000|duration_time=0.040000|size=|pos=600|flags=K__\n",
            "pts_time=0.300000|dts_time=0.240000|duration_time=0.040000|size=12.5|pos=600|flags=K__\n",
            "this is not a packet\n",
            "\n",
            "pts_time= 0.32 |dts_time=0.28|duration_time=4e-2|size=1_600|pos=700|flags=___|extra=a=b\n",
        ]},
        {"name": "no frame rate, no durations", "fps": 0.0, "lines": [
            "pts_time=1.000000|dts_time=1.000000|duration_time=N/A|size=800|pos=0|flags=K__\n",
            "pts_time=1.500000|dts_time=1.500000|duration_time=N/A|size=700|pos=900|flags=___\n",
        ]},
        {"name": "no packets", "fps": 25.0, "lines": ["pts_time=0.0|dts_time=0.0\n", "nothing\n"]},
    ]

    rng = random.Random(1234)
    streams = {
        "the panel test's": ([0.0, 0.5, 1.0, 1.5], [0.5] * 4, [1000, 500, 2000, 500], [True, False, True, False]),
        "starting late, frames across seconds": (
            [1.4 + i * frame for i in range(50)], [frame] * 50,
            [rng.randint(200, 60000) for _ in range(50)], [i % 12 == 0 for i in range(50)]),
        "starting at 0.7, not on a second": (
            [0.7 + i * 0.3 for i in range(9)], [0.3] * 9, [rng.randint(1000, 9000) for _ in range(9)],
            [i in (0, 4) for i in range(9)]),
        "packets without a duration": (
            [0.0, 0.25, 0.25, 0.9, 1.0, 1.75], [0.25, 0.0, 0.0, 0.1, 0.0, 0.25], [4000, 50, 60, 700, 80, 900],
            [True, False, False, False, True, False]),
        "one packet": ([2.25], [0.04], [5000], [True]),
        "one keyframe, a long group": (
            [i / 30 for i in range(95)], [1 / 30] * 95, [rng.randint(100, 3000) for _ in range(95)],
            [i == 0 for i in range(95)]),
    }

    def plot_json(plot: br.BitratePlot) -> dict:
        return {"times": [exact(t) for t in plot.times], "values": [exact(v) for v in plot.values],
                "axis_label": plot.axis_label, "value_label": plot.value_label}

    stream_cases = []
    for name, (times, durations, sizes, keyframes) in streams.items():
        data = br.BitrateData(path=Path("C:/clips/stream.mkv"), times=np.array(times), durations=np.array(durations),
                              sizes=np.array(sizes), keyframes=np.array(keyframes))
        summary = br.bitrate_summary(data)
        case = {
            "name": name, "times": [exact(t) for t in times], "durations": [exact(d) for d in durations],
            "sizes": sizes, "keyframes": keyframes,
            "start_time": exact(data.start_time), "end_time": exact(data.end_time), "duration": exact(data.duration),
            "total_bytes": data.total_bytes, "average_kbps": exact(data.average_kbps),
            "summary": {"average_kbps": exact(summary.average_kbps), "minimum_kbps": exact(summary.minimum_kbps),
                        "maximum_kbps": exact(summary.maximum_kbps)},
            "bins": {}, "plots": {},
        }
        for adjust in (True, False):
            starts, ends, rates = br._compute_second_bins(data, adjust)
            case["bins"][str(adjust).lower()] = {"starts": [exact(s) for s in starts], "ends": [exact(e) for e in ends],
                                                 "rates": [exact(r) for r in rates]}
            for kind, function in (("frame", br.frame_plot), ("second", br.second_plot), ("gop", br.gop_plot)):
                case["plots"][f"{kind}|{str(adjust).lower()}"] = plot_json(function(data, adjust))
        stream_cases.append(case)
    return {"reads": [{**item, "fps": exact(item["fps"]), "expected": read(item["lines"], item["fps"])}
                      for item in reads],
            "streams": stream_cases}


def film_grain_cases() -> dict:
    """The film grain library's parameters for a VVC frame's film grain
    message (core/d3d11_film_grain.py: plane_sizes and native_params), from
    the Python app's own functions: the film grain test's messages
    (cpp/tests/fixtures/make_vvc_grain.py) and messages that reach each of the
    conversion's rules (model values missing, cutoffs past either end, a
    strong grain, the autoregressive model), on frames of each chroma
    subsampling, depth and an odd size. Only each plane's used intervals
    are given."""
    from vmaf_app.core.d3d11_film_grain import native_params, plane_sizes
    from vmaf_app.core.film_grain import FilmGrainH274, FrameGrain

    def message(log2_scale_factor, components, model_id=0):
        intervals = tuple(tuple((lower, upper, tuple(values)) for lower, upper, values in component or ())
                          for component in components)
        return FilmGrainH274(model_id, 0, log2_scale_factor, intervals, True)

    messages = {
        "fixture A": message(3, [[(16, 100, (60, 10, 4)), (120, 235, (36, 14, 14))], [(0, 255, (40, 4, 12))], None]),
        "fixture B": message(2, [[(0, 255, (30, 6, 6))], None, [(60, 200, (50, 12, 8))]]),
        "fixture once": message(4, [[(0, 255, (120, 8, 14))], [(0, 255, (60, 14, 2))], [(0, 255, (60, 2, 14))]]),
        "one model value": message(2, [[(0, 255, (50,))], [(0, 128, (21,))], [(129, 255, (-33,))]]),
        "two model values": message(5, [[(0, 255, (50, 9))], [(0, 255, (-7, 3))], None]),
        "cutoffs past either end": message(1, [[(0, 99, (10, 1, 20)), (100, 255, (-40, 15, 0))], None,
                                               [(0, 255, (70, 0, 7))]]),
        "strong grain": message(0, [[(0, 255, (32767, 8, 8))], [(0, 255, (-32768, 8, 8))], [(0, 255, (-1, 8, 8))]]),
        "autoregressive": message(0, [[(0, 255, (10,))], None, None], model_id=1),
    }
    frames = [(64, 48, 8, (1, 1), 3), (1920, 1080, 10, (1, 1), 3), (1919, 1079, 10, (1, 1), 3),
              (64, 48, 10, (1, 0), 3), (64, 48, 8, (0, 0), 3), (64, 48, 8, (1, 1), 1)]
    cases = []
    for name, grain in messages.items():
        for width, height, depth, chroma_shift, planes in frames:
            sizes = plane_sizes(width, height, chroma_shift, planes)
            params = native_params(FrameGrain(grain, 37), depth, sizes, chroma_shift)
            expected = None
            if params is not None:
                expected = {"depth": params.depth, "log2_scale_factor": params.log2_scale_factor, "poc": params.poc,
                            "planes": []}
                for plane in params.plane:
                    count = plane.intervals
                    expected["planes"].append({
                        "width": plane.width, "height": plane.height, "intervals": count,
                        "lower": list(plane.lower[:count]), "upper": list(plane.upper[:count]),
                        "scale": list(plane.scale[:count]), "h": list(plane.h[:count]), "v": list(plane.v[:count])})
            cases.append({
                "name": f"{name}, {width}x{height} {depth}-bit, chroma shift {chroma_shift}, {planes} planes",
                "model_id": grain.model_id, "log2_scale_factor": grain.log2_scale_factor,
                "components": [[{"lower": lower, "upper": upper, "values": list(values)}
                                for lower, upper, values in component] for component in grain.intervals],
                "seed": 37, "width": width, "height": height, "depth": depth,
                "chroma_shift": list(chroma_shift), "planes": planes,
                "sizes": [list(size) for size in sizes], "expected": expected})
    return {"cases": cases}


if __name__ == "__main__":
    # Its texts as they are, not escaped: the file is a third the size.
    (HERE / "i18n.json").write_text(json.dumps(i18n_cases(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write("metric_cache.json", metric_cache_cases())
    write("run_files.json", run_file_cases())
    write("chart.json", chart_cases())
    write("graph.json", graph_cases())
    write("app_log.json", app_log_cases())
    write("update_check.json", update_check_cases())
    write("frame_extract.json", frame_extract_cases())
    write("bitrate.json", bitrate_cases())
    write("playback.json", playback_cases())
    write("native.json", native_cases())
    write("film_grain.json", film_grain_cases())
    write("devices.json", devices_cases())
    write("perceptual_cpu.json", perceptual_cpu_cases())
    write("timed_pairing.json", timed_pairing_cases())
    write("cvvdp.json", cvvdp_cases())
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
