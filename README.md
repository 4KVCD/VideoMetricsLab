# VMAF Calculator

A VMAF calculation app (Python + PySide6/Qt), inspired by FFMetrics, with:

- Four quality metrics per run: VMAF, PSNR, SSIM and XPSNR, each toggled from
  the checkbox in its own column header and scored in that column.
- An expanded comparison graph window: score-vs-time curves for multiple
  distorted files overlaid, one tab per metric, with a hover readout
  (frame/time/score per series).
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

Requires Python 3.11+ and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) **9 or
newer** with `libvmaf` support (a "full build" includes it). Both `ffmpeg` and
`ffprobe` are checked at startup; if either is missing, too old, or not on
PATH, the app prompts for its location and remembers the choice.

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

## Lint

```bash
.venv\Scripts\python.exe -m ruff check vmaf_app tests
```

Rules and their rationale live in `ruff.toml`.

## Architecture

Two layers, and the dependency only ever points one way — `ui` imports `core`,
never the reverse:

```
vmaf_app/
  core/            no Qt widgets; runnable and testable headless
    models.py        VmafOptions, VideoInfo, FrameScores, VmafRunResult
    vmaf_runner.py   builds the ffmpeg filtergraph, runs it, parses the logs
    ffprobe.py       media probing        crop_detect.py  black-bar detection
    stats.py         summary statistics   model_select.py VMAF model choice
    run_io.py        save/load/CSV        result_cache.py cached run lookup
    ffmpeg_locate.py tool discovery + version check
    gpu.py           hwaccel selection    process_control.py pause/resume/kill
  ui/
    main_window.py   the file table, per-row options, run orchestration
    graph_window.py  the comparison window (one tab per metric)
    chart.py         the plotting widget   widgets.py  reusable Qt widgets
    formatting.py    display formatting    worker.py   the run QThread
```

Two things are worth knowing before changing the internals:

- **Per-frame scores are stored as packed NumPy arrays** (`FrameScores`), not
  one object per frame — 28 bytes a frame against ~180. Indexing yields a
  throwaway read-only `FrameScore` view for convenience, but hot paths
  (plotting, stats, hover) should read the arrays directly.
- **`chart.py` is a purpose-built plotting widget**, not a wrapper around a
  charting library. It downsamples to one min/max pair per pixel column and
  repaints only the columns the crosshair touches, which is what keeps the
  graph window's CPU and memory in budget.

