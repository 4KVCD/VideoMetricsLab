# VMAF Calculator

A VMAF calculation app (Python + PySide6/Qt), inspired by FFMetrics, with:

- Four quality metrics per run: VMAF, PSNR, SSIM and XPSNR, each toggled from
  the checkbox in its own column header and scored in that column.
- A comparison graph, as a tab of the main window rather than a separate
  window: score-vs-time curves for multiple distorted files overlaid, one
  sub-tab per metric, with a hover readout (frame/time/score per series), a
  jump-to-frame control and a PNG export that carries its own title, legend
  and statistics.
- GPU-accelerated decoding of **both** the source and the distorted video,
  chosen independently (cuda / qsv / d3d11va, auto-detected). Either input
  falls back to software decode on its own -- by codec before ffmpeg is
  launched, and through both single-input retry combinations if the
  hardware path fails anyway.
- Analysis at the inputs' own bit depth: 8-bit compares as `yuv420p`, 10-bit
  as `yuv420p10le`, 12-bit as `yuv420p12le`, taking the deeper of the two
  inputs so a 10-bit master is not truncated to match an 8-bit encode.
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
- A Settings tab for the ffmpeg location, where results and exports are kept,
  and what new rows default to.

Comparisons whose timelines or geometry are ambiguous are refused rather than
scored: mismatched frame rates, mismatched durations (unless a duration limit
inside both files is set), mismatched sample aspect ratios, and
variable-frame-rate input. A wrong number here looks exactly like a right one,
so the run does not start.

While a run is in progress the source, the per-row options and the settings
are locked, because changing them mid-run would mean the finished result no
longer describes what was actually measured.

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

373 tests, all offscreen (no window appears) and none of which touch your
real settings file or results cache.

`tests/smoke_run.py` is a manual end-to-end check (not part of the pytest
suite) that runs the real ffmpeg/libvmaf pipeline against the fixtures in
`tests/fixtures/`. Run it from the repository root:

```bash
.venv\Scripts\python.exe tests\smoke_run.py
```

`--10bit` compares the 10-bit fixture instead, which is the only way to
confirm libvmaf accepts the deeper analysis format rather than the
filtergraph merely being built correctly. `--no-gpu` forces software decode,
which should produce an identical score.

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
    proc.py          subprocess launching settings.py     persisted settings
    time_format.py   duration formatting
  ui/
    main_window.py   the file table, per-row options, run orchestration
    graph_panel.py   the comparison graph tab (one sub-tab per metric)
    chart.py         the plotting widget   widgets.py  reusable Qt widgets
    formatting.py    display formatting    worker.py   the run QThread
    probe_worker.py  probing + cache lookup off the UI thread
    file_worker.py   result/export writing off the UI thread
```

Two things are worth knowing before changing the internals:

- **Per-frame scores are stored as packed NumPy arrays** (`FrameScores`), not
  one object per frame — 28 bytes a frame against ~180. Indexing yields a
  throwaway read-only `FrameScore` view for convenience, but hot paths
  (plotting, stats, hover) should read the arrays directly.
- **`chart.py` is a purpose-built plotting widget**, not a wrapper around a
  charting library. It downsamples to one min/max pair per pixel column and
  repaints only the columns the crosshair touches, which is what keeps the
  graph's CPU and memory in budget.
- **A cached result is keyed on more than the filenames.** The key covers each
  file's resolved absolute path, size and modification time, plus the
  calculation options -- so replacing a file in place, or changing what the
  run computes, is a miss rather than a stale hit. Results and exports are
  written on a background thread (`file_worker.py`): a feature-length run is
  ~9MB of JSON, and writing it inline froze the window.
- **Nothing in `core` may spawn a subprocess directly.** They all go through
  `proc.py`, which hides the console window Windows would otherwise flash up
  for every child process. `tests/test_architecture.py` enforces this, along
  with the layering above.

