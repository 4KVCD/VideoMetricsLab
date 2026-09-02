# VMAF Calculator

A VMAF calculation app (Python + PySide6/Qt), inspired by FFMetrics, with:

- An expanded comparison graph window: VMAF-vs-time curves for multiple
  distorted files overlaid, with a hover readout (frame/time/VMAF per series).
- GPU-accelerated decoding of the reference video (NVDEC via `-hwaccel cuda`,
  auto-detected, with automatic fallback to software decode).
- Per-run statistics: mean/median/stdev/min/max, 1% low, and the percentage
  of frames above/below configurable VMAF thresholds (>95, >90, >85, <85,
  <80, <70).
- Automatic handling of resolution mismatches (the source is scaled to the
  distorted video's resolution).
- Automatic black-bar (letterbox/pillarbox) detection and cropping on both
  the source and distorted video independently, so masked black bars don't
  inflate the score.
- Automatic VMAF model selection (the 4K model is used when the distorted
  video is UHD or higher; otherwise the standard model), overridable in the UI.

## Setup

Requires [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) with `libvmaf` support
(a "full build" includes it) on PATH, and Python 3.10+.

```bash
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run

```bash
.venv\Scripts\python.exe -m vmaf_app.main
```

## Test

```bash
.venv\Scripts\python.exe -m pytest
```

`tests/smoke_run.py` is a manual end-to-end check (not part of the pytest
suite) that runs the real ffmpeg/libvmaf pipeline against the generated
fixtures in `tests/fixtures/`.
