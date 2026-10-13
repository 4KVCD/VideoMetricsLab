"""Writes the C++ tests' expected values, made by the Python app's own code:
its statistics (vmaf_app/core/stats.py), its metric registry
(vmaf_app/core/metrics.py) and Python's own number formatting.

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

from vmaf_app.core.metrics import METRICS
from vmaf_app.core.stats import aggregate_scores, compute_stats


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


if __name__ == "__main__":
    write("number_format.json", number_format_cases())
    write("metrics.json", metric_cases())
    write("stats.json", stats_cases())
