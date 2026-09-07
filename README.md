# Video Metrics Calculator

A video metrics calculation app (Python + PySide6/Qt), inspired by FFMetrics, with:

- Four independently selectable quality metrics: VMAF, PSNR, SSIM and XPSNR.
  VMAF is the default, not a requirement. Each metric is ticked in its own
  column, on the row it applies to; a metric that has been measured shows its
  score there instead. Ticking one cell applies to every selected row;
  column-header shortcuts apply to all rows. Check rows to include them when
  pressing **Calculate metrics**. A separate Status column distinguishes
  incomplete analyses from completed scores.
- A comparison graph, as a tab of the main window rather than a separate
  window: score-vs-time curves for multiple distorted files overlaid, one
  sub-tab per metric, with a hover readout (frame/time/score per series), a
  jump-to-frame control and a PNG export that carries its own title, legend
  and statistics.
- A standalone **Video Compare** tab for inspecting reference and test videos,
  with or without metric results. Completed analyses supply their cropped/scaled
  pictures and available per-frame metric readouts. Choose a frame or timestamp, hold **S** to
  reveal the source, and use the left/right arrow keys to cycle through
  distorted videos without losing the current position, zoom, or pan. Video
  playback keeps a rolling pool of **at most four videos**: source, current
  encode, and the current encode's left/right neighbours (wrapping at the ends).
  The **Source playback** selector keeps the cropped source at native resolution
  (default), or downsizes it to fit the selected encode's cropped resolution.
  Both modes preserve aspect ratio and fit the window; metric settings and
  saved results are unchanged. Changing this selector may briefly buffer.
  Switching retains the already-playing neighbour; only the new outer neighbour
  is prepared in the background. Very rapid navigation or software decoding
  can still require buffering. Windows playback now prefers GStreamer/D3D11:
  GPU decode where supported, GPU crop/scale, and direct native-window output.
  PQ/HLG HDR→SDR uses a native GPU shader with a 16-bit RGB intermediate,
  fixed extended-Reinhard luminance mapping (1000→100 nit), and BT.2020→BT.709
  conversion. Both sides use the same curve; this is not dynamic scene-based
  tone mapping or Dolby Vision processing. Decoded pixels never pass through
  Python on this path. H.266 uses `avdec_h266` software decoding followed by
  one GPU upload. Decoder support is build-specific: the installed GStreamer
  build also software-decodes High-10 H.264. Those decoders can still consume
  substantial CPU and RAM. Native HDR retains the original HDR signal for a
  10-bit swapchain. Matching-rate playback now pairs source/encode GPU samples
  by timeline frame before submitting either side to one native renderer.
  Holding S selects the other sample from that pair; it does not switch clocks
  or independent rendering windows. Queues remain bounded to three samples per
  stream plus the appsink queues. Frame matching assumes aligned, constant-rate
  inputs; different frame rates use the labelled FFmpeg fallback.
  One audio-only GStreamer pipeline plays the source's default audio track
  continuously across visual switches. Its media position paces paired video;
  a decoder stall pauses/repositions audio rather than accumulating drift.
  Unsupported or absent source audio leaves video playback silent. This is
  software timeline synchronization, not a guarantee of calibrated speaker/
  monitor hardware latency. Still mode remains available for exact inspection.
  If the native path is unavailable, the existing FFmpeg/Vulkan/libplacebo
  preview remains available with bounded Python RGBA queues and progressively
  more compatible GPU/CPU fallbacks. Unmanaged preview uses that fallback.
  Still
  mode remains available for direct frame seeking. PQ and HLG video or stills
  can be tone-mapped automatically for the monitor showing the app (including
  its Windows HDR/SDR-white setting), forced to an HDR-to-SDR preview, or
  shown unmanaged for diagnosis. Still-frame tone mapping uses a 100-nit
  target. Still-frame, native GPU and fallback playback tone operators differ,
  so their appearances should not be assumed identical.
- An independent **Bitrate Viewer** tab that can scan videos without running
  any quality metric. Its frame view plots each encoded video packet's size,
  its second view plots video bitrate in one-second intervals, and its GOP
  view groups bitrate from one keyframe to the next. Multiple files can be
  overlaid, inspected, zoomed, and exported. Fresh metric runs automatically
  add both inputs to the viewer; audio and container overhead are excluded.
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
  inflate the score. The Videos table shows whether bars were found on each
  input, with exact crop dimensions and per-side pixel counts in its tooltip.
- Automatic VMAF model selection (the 4K model is used when the distorted
  video is UHD or higher; otherwise the standard model), overridable in the UI.
- A Settings tab for the ffmpeg location, where results and exports are kept,
  and what new rows default to.
- Optional parallel scoring -- one tick box beside the Run button, adjustable
  while a run is in progress. libvmaf does not keep a many-core CPU busy on
  its own, so a second video largely fills the idle capacity rather than
  competing for it.
  Measured on a 24-core machine over four 1080p comparisons with the app's
  defaults (auto-crop, GPU decode): 22.6s one at a time against 14.4s at two,
  1.57x. A third gained nothing (14.5s), which is why two is the maximum.
  Scores are identical either way -- it changes how fast results arrive, never
  what they are, so it takes no part in cache identity.

Comparisons whose timelines or geometry are ambiguous are refused rather than
scored: mismatched frame rates, mismatched durations (unless a duration limit
inside both files is set), mismatched sample aspect ratios, and
variable-frame-rate input. A wrong number here looks exactly like a right one,
so the run does not start.

### Metric selection and compatibility

**Metric Graphs** selects an available metric and remembers your chosen metric.
Each metric keeps its own units, precision and graph scale; PNG exports are named
for the active metric. Missing frame scores are never replaced with neighbouring
scores. PSNR and SSIM use the same libvmaf feature extractors with or without
VMAF enabled, keeping the score definitions consistent. XPSNR-only runs use
FFmpeg's XPSNR filter directly without loading libvmaf or a VMAF model.

The libvmaf thread and frame-subsample settings apply to VMAF/PSNR/SSIM, not
XPSNR. XPSNR-only runs score every frame; in combined runs XPSNR values are
retained at the libvmaf sample positions. Video preparation settings affect
calculations; Video Compare's display tone mapping and playback resolution do not.

Existing `.vmafrun.json` files, cache keys for VMAF-enabled runs, and the shared
user-data location are preserved. The application rebrand does not relocate or
delete previous results. Internal Python package and saved-file names remain
unchanged for compatibility.

While a run is in progress the source, the per-row options and the settings
are locked, because changing them mid-run would mean the finished result no
longer describes what was actually measured.

## Setup

Requires Python 3.11+ and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) **9 or
newer** with `libvmaf` support (a "full build" includes it). Both `ffmpeg` and
`ffprobe` are checked at startup; if either is missing, too old, or not on
PATH, the app prompts for its location and remembers the choice. On Windows,
the Python requirements also install GStreamer 1.28.6 for hardware-decoded,
D3D11-presented video comparison. HDR displays use native 10-bit PQ/HLG
presentation. Build the small GPU HDR→SDR helper once using MinGW-w64 `g++`
on PATH (the app safely falls back to FFmpeg if the helper is absent):

```powershell
./scripts/build_d3d11_tonemap.ps1
```

The generated `vmaf_app/native/d3d11_tonemap.dll` is ignored by Git; its source
is `native/d3d11_tonemap.cpp`. No compiler is invoked during playback. Shader
initialization happens on a streaming thread, not the Qt UI thread. The
helper uses the Windows D3D11/D3DCompiler runtime and contains its C++ runtime.

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

577 tests, all offscreen (no window appears) and none of which touch your
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
    bitrate.py       video-packet scan + frame/second/GOP aggregation
    run_io.py        save/load/CSV        result_cache.py cached run lookup
    frame_extract.py exact still-frame decode using a run's crop/scale recipe
    video_playback.py display-sized GPU/CPU ffmpeg preview command construction
    gstreamer_playback.py synchronized GPU decode and native D3D11 presentation
    display_hdr.py   Windows monitor HDR state and configured SDR white level
    ffmpeg_locate.py tool discovery + version check
    gpu.py           hwaccel selection    process_control.py pause/resume/kill
    proc.py          subprocess launching settings.py     persisted settings
    time_format.py   duration formatting
  ui/
    main_window.py   the file table, per-row options, run orchestration
    graph_panel.py   the comparison graph tab (one sub-tab per metric)
    frame_compare_panel.py synchronized source/distorted still-frame viewer
    video_compare_view.py native, synchronized source/distorted playback surfaces
    rolling_video_view.py four-stream pool and matching-frame presentation
    playback_worker.py bounded FFmpeg RGBA frame queues and GPU fallback
    native_playback_pool.py shared-clock native GPU stream lifecycle
    bitrate_panel.py independent multi-file bitrate viewer tab
    bitrate_worker.py non-blocking ffprobe packet scans
    chart.py         the plotting widget   widgets.py  reusable Qt widgets
    formatting.py    display formatting    worker.py   the run QThread
    probe_worker.py  probing + cache lookup off the UI thread
    frame_extract_worker.py non-blocking preview-frame decoding
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
  run computes, is a miss rather than a stale hit. By default, cached results
  live in the launcher-independent per-user folder
  `~/.vmaf-calculator/results_cache`; the Settings tab can override it. The
  preferences file lives alongside it, so different launchers share that
  override. Results and exports are written on a background thread
  (`file_worker.py`): a feature-length run is ~9MB of JSON, and writing it
  inline froze the window.
- **Nothing in `core` may spawn a subprocess directly.** They all go through
  `proc.py`, which hides the console window Windows would otherwise flash up
  for every child process. `tests/test_architecture.py` enforces this, along
  with the layering above.

