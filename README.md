# Video Metrics Calculator

A desktop app for measuring and comparing video encode quality on Windows.
Point it at a reference video and any number of encodes, and it gives you
VMAF, PSNR, SSIM and XPSNR — per frame, graphed over time, with the two videos
playable side by side.

Built with Python and PySide6/Qt, driving FFmpeg and libvmaf. Inspired by
FFMetrics, but built around comparing several encodes at once rather than
scoring one at a time.

![The Videos tab](docs/screenshot-videos.png)

## Why

Deciding between encoder settings means running the same comparison many
times and holding the results side by side. That is what this is built for:

- **All four metrics come from one decode pass.** PSNR and SSIM are libvmaf
  features computed from the frame pair VMAF already has in hand, and XPSNR is
  chained into the same filtergraph. On a 10-second 1080p pair, VMAF alone took
  2.81s, VMAF+PSNR+SSIM 2.80s, and all four 3.58s — against 2.2s to fetch
  XPSNR in a second run afterwards. There is little reason not to measure
  everything, so all four are on by default.
- **Results are cached and reused.** A finished run is keyed on each file's
  path, size and modification time plus the settings that decide what gets
  measured, so re-adding a video brings its scores straight back instead of
  spending an hour recomputing them.
- **Ambiguous comparisons are refused, not scored.** Mismatched frame rates,
  mismatched durations, mismatched sample aspect ratios and variable-frame-rate
  input all stop the run. A wrong number looks exactly like a right one.

## Features

### Metrics

- **VMAF, PSNR, SSIM and XPSNR**, each ticked in its own column of the video
  table, on the row it applies to. A metric that has been measured shows its
  score in that cell instead of a tick box.
- **Automatic VMAF model selection** — the 4K model for UHD and above, the
  standard model otherwise, overridable per row.
- **Analysis at the inputs' own bit depth**: 8-bit compares as `yuv420p`,
  10-bit as `yuv420p10le`, 12-bit as `yuv420p12le`, taking the deeper of the
  two so a 10-bit master is not truncated to match an 8-bit encode.
- **Automatic black-bar detection** on both inputs independently, so masked
  letterboxing does not inflate a score. The table says whether bars were
  found; exact per-side pixel counts are on hover.
- **Automatic handling of resolution mismatches**, in either direction, or
  both at once for comparison.
- **Resolution round-trip tests** — score what downscaling to 1080p and back
  costs, with no second file needed.
- **GPU-accelerated decoding of both inputs**, chosen independently
  (CUDA / QSV / D3D11VA, auto-detected), each falling back to software on its
  own if the hardware cannot handle its format.
- **Optional parallel scoring** of two videos at once, adjustable mid-run.
  libvmaf does not keep a many-core CPU busy on its own, so a second video
  largely fills the idle capacity. Measured on a 24-core machine over four
  1080p comparisons: 22.6s sequential against 14.4s at two, 1.57×. A third
  gained nothing, which is why two is the maximum. Scores are identical either
  way.

### Metric Graphs

Score-vs-time curves for every scored encode, overlaid, one sub-tab per
metric. Hover for a frame/time/score readout across all series, jump to a
frame, or export a PNG that carries its own title, legend and statistics.

The statistics table shows every metric's mean at once — click a metric column
for its full breakdown: median, standard deviation, min/max, 10%/5%/1%/0.1%
lows, and the share of frames above or below six quality bands.

Those bands are calibrated so that ">90" means roughly the same thing on all
four metrics. The mapping was measured over 2,520 paired frames — three
sources at seven CRFs each, spanning VMAF 13 to 99.6 — by taking each metric's
median value on the frames where VMAF sat at 95/90/85/80/70:

| VMAF | PSNR (dB) | SSIM | XPSNR (dB) |
|------|-----------|-------|------------|
| 95   | 40.8      | 0.988 | 37.8       |
| 90   | 38.2      | 0.983 | 35.5       |
| 85   | 35.3      | 0.973 | 33.3       |
| 80   | 33.7      | 0.958 | 29.1       |
| 70   | 32.1      | 0.947 | 27.1       |

The calibration used synthetic sources, so real content will shift these.
Treat them as reasonable defaults rather than constants.

### Video Compare

Source and encode side by side, with or without metric results — useful before
committing to a feature-length run, to check framing or confirm two files are
even the same content.

- Hold **S** to reveal the source; left/right arrows cycle through encodes
  without losing position, zoom or pan.
- Completed runs supply their real cropped and scaled pictures plus per-frame
  metric readouts.
- On Windows, playback uses GStreamer/D3D11: GPU decode, GPU crop/scale and
  direct native-window output, with decoded pixels never passing through
  Python. One audio pipeline plays continuously across visual switches and
  paces the paired video.
- HDR (PQ/HLG) is either presented natively on a 10-bit swapchain or tone
  mapped to SDR by a small GPU shader. Still mode remains available for exact
  frame inspection.
- If the native path is unavailable, an FFmpeg preview path takes over
  automatically.

### Bitrate Viewer

Scans videos without running any metric. Three views: each encoded packet's size,
bitrate in one-second intervals, and bitrate per GOP from one keyframe to the
next. Multiple files overlay, zoom and export. Fresh runs add both inputs
automatically.

## Install

### Download a build

Grab the latest `VideoMetricsCalculator-windows.zip` from
[Releases](../../releases), unzip it anywhere, and run
`VideoMetricsCalculator.exe`. No installer, no Python, no compiler — GStreamer
and the HDR shader are inside.

You still need **FFmpeg 9 or newer with libvmaf**, which is not bundled (see
[Requirements](#requirements)). The app checks at startup and offers a file
picker if it cannot find one.

### From source

Requires **Python 3.11+** (developed on 3.14):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m vmaf_app.main
```

Use `pythonw.exe` instead of `python.exe` to run without a console window.

The GPU HDR→SDR shader is optional and not in the repository. Build it once
with MinGW-w64 `g++` on PATH; without it the app falls back to FFmpeg tone
mapping:

```powershell
./scripts/build_d3d11_tonemap.ps1
```

## Requirements

| | |
|---|---|
| **OS** | Windows (GStreamer playback and the HDR shader are Windows-only; metrics themselves are portable) |
| **FFmpeg** | 9 or newer, with `libvmaf`. A [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) "full build" includes it. Both `ffmpeg` and `ffprobe` must be present |
| **Python** | 3.11+, source installs only |
| **GPU** | Optional. NVIDIA / Intel / AMD decode is auto-detected and falls back to software |

FFmpeg is checked at startup. If it is missing, too old, or not on PATH, the
app prompts for its location and remembers the choice.

## Usage

1. **Videos** tab → pick a reference video, then **Add files…** for the
   encodes to compare against it.
2. Tick the metrics you want in their columns. All four are on by default;
   ticking one cell applies to every selected row, and the column header
   applies to every row.
3. Press **Calculate metrics**. Scores appear in the table as they finish and
   go straight onto the graph.
4. **Metric Graphs** for curves and statistics, **Video Compare** to look at
   the frames, **Bitrate Viewer** for bitrate distribution.

Results save to `.vmafrun.json`, export to CSV, and are cached automatically —
re-adding a video you have already scored brings its results straight back.

Right-click a row to force a recalculation, ignoring anything cached.

## Building a distributable

Produces a self-contained Windows folder and zip, with GStreamer and the HDR
shader inside:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
./scripts/build_release.ps1
```

Output lands in `dist/VideoMetricsCalculator/`. See
[`docs/BUILD.md`](docs/BUILD.md) for what is bundled, what is deliberately
not, and how to cut the size down.

## Development

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest        # 665 tests, all offscreen
.venv\Scripts\python.exe -m ruff check vmaf_app tests
```

Tests never touch your real settings file or results cache — `tests/conftest.py`
redirects both and fails loudly if anything tries to reach the real ones.

`tests/smoke_run.py` is a manual end-to-end check, outside the pytest suite,
that runs the real FFmpeg/libvmaf pipeline against `tests/fixtures/`:

```powershell
.venv\Scripts\python.exe tests\smoke_run.py            # --10bit, --no-gpu
```

### Architecture

Two layers, and the dependency only ever points one way — `ui` imports `core`,
never the reverse. `tests/test_architecture.py` enforces this.

```
vmaf_app/
  core/            no Qt widgets; runnable and testable headless
    models.py        VmafOptions, VideoInfo, FrameScores, VmafRunResult
    vmaf_runner.py   builds the ffmpeg filtergraph, runs it, parses the logs
    ffprobe.py       media probing        crop_detect.py  black-bar detection
    stats.py         summary statistics   model_select.py VMAF model choice
    bitrate.py       packet scan + frame/second/GOP aggregation
    run_io.py        save/load/CSV        result_cache.py cached run lookup
    frame_extract.py still-frame decode using a run's crop/scale recipe
    video_playback.py FFmpeg preview command construction
    gstreamer_playback.py GPU decode and native D3D11 presentation
    display_hdr.py   monitor HDR state    ffmpeg_locate.py tool discovery
    gpu.py           hwaccel selection    process_control.py pause/resume/kill
    proc.py          subprocess launching settings.py  persisted settings
  ui/
    main_window.py   the video table, per-row options, run orchestration
    graph_panel.py   the comparison graph tab (one sub-tab per metric)
    frame_compare_panel.py synchronized source/encode still-frame viewer
    video_compare_view.py  native synchronized playback surfaces
    rolling_video_view.py  four-stream pool and matching-frame presentation
    bitrate_panel.py independent multi-file bitrate viewer
    chart.py         the plotting widget  widgets.py  reusable Qt widgets
    worker.py        the run QThread      probe_worker.py probing off the UI thread
    file_worker.py   result/export writing off the UI thread
```

Four things are worth knowing before changing the internals:

- **Per-frame scores are packed NumPy arrays** (`FrameScores`), not one object
  per frame — 28 bytes a frame against ~180. Indexing yields a throwaway
  read-only view for convenience, but hot paths (plotting, stats, hover)
  should read the arrays directly.
- **`chart.py` is a purpose-built plotting widget**, not a wrapper around a
  charting library. It reduces each curve to one min/max pair per pixel column
  and repaints only the columns the crosshair touches. Benchmarked against
  pyqtgraph, which was 6.7× slower on the same data.
- **A cached result is keyed on more than filenames** — each file's resolved
  path, size and mtime, plus the options that decide what gets measured. Which
  metrics a run recorded is deliberately *not* part of that: a run holding a
  different set looked at the same frames, so it is reused rather than
  discarded. Cached runs live in `~/.vmaf-calculator/results_cache`.
- **Nothing in `core` may spawn a subprocess directly.** Everything goes
  through `proc.py`, which suppresses the console window Windows would
  otherwise flash for every child process.

## Compatibility

Existing `.vmafrun.json` files, cache keys for VMAF-enabled runs and the shared
user-data location are all preserved across versions. Internal package and
saved-file names are unchanged.

## Licence

Not yet chosen — all rights reserved by default. FFmpeg and libvmaf are
separate projects under their own licences and are not distributed with this
application.
