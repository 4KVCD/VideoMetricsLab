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


if __name__ == "__main__":
    write("number_format.json", number_format_cases())
    write("metrics.json", metric_cases())
    write("stats.json", stats_cases())
    write("probe.json", probe_cases())
    write("settings.json", settings_cases())
    write("versions.json", version_cases())
