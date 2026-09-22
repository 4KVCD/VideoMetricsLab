# Architecture

## Boundaries

`vmaf_app.main` creates the Qt application. `ui/main_window.py` owns video rows,
options and orchestration. The UI imports `core`; core does not import UI.
`tests/test_architecture.py` checks this boundary. Core is not completely
Qt-free: tool discovery uses QSettings and native playback loads GI bindings.

| Concern | Main modules |
| --- | --- |
| Media metadata and geometry | `core/ffprobe.py`, `crop_detect.py`, `model_select.py` |
| Metric execution | `core/vmaf_runner.py`, `ui/worker.py` |
| Metric registry and packed scores | `core/metrics.py`, `core/models.py` |
| Results and persistence | `core/run_io.py`, `core/result_cache.py` |
| Statistics and plotting | `core/stats.py`, `ui/graph_panel.py`, `chart.py` |
| Still comparison | `core/frame_extract.py`, `ui/frame_extract_worker.py` |
| Playback orchestration | `ui/rolling_video_view.py`, `video_compare_view.py` |
| Native playback and synchronization | `core/gstreamer_playback.py`, `locked_presentation.py`, `ui/locked_native_pool.py` |
| FFmpeg playback fallback | `core/video_playback.py`, `ui/playback_worker.py` |
| Bitrate analysis | `core/bitrate.py`, `ui/bitrate_worker.py`, `bitrate_panel.py` |
| Process lifecycle | `core/proc.py`, `process_control.py` |

## Metric lifecycle

1. Probe the reference and encodes asynchronously; validate compatibility.
2. Capture each row's options, crop and scaling recipe, and model selection.
3. Check cached results for matching media identity and calculation settings.
4. Schedule one or optionally two calculations. Build an FFmpeg filtergraph
   for the requested metrics; XPSNR alone does not require a VMAF score.
5. Parse per-frame logs into `FrameScores`, a collection of packed NumPy arrays.
6. Deliver results to the UI, update graph identities, and queue cache writes.

Do not mutate a running job's options in place. `clone_options` copies the
mutable feature list. Completion must verify that the row/source/settings
still match before attaching a result. Clearing a result must also remove its
graph identity, without removing unrelated imported curves.

## Persistence contracts

The legacy names `VmafOptions`, `VmafRunResult` and `.vmafrun.json` remain for
compatibility. Missing metrics are not zero scores: columns can be absent and
individual unavailable values can be NaN. JSON uses null for missing values
and strings for infinities. Use `run_io` rather than ad-hoc serialization.

Cache keys include resolved file paths, size, mtime and calculation options.
Execution-only controls such as GPU vendor and thread count are excluded.
Metric selection remains part of the stored key; lookup searches compatible
alternative metric sets and historical feature orders. A partial result is
not complete. With subsampling, XPSNR-only and libvmaf-backed results have
different frame coverage and must not be interchanged.

## Metric architecture

`core/metrics.py` is the sole registry for metric labels, formatting,
thresholds, sequence aggregation and legacy execution bindings. It is
headless: it must not import Qt, ffmpeg wrappers or result models. Current
logical order is VMAF, VMAF NEG, PSNR, SSIM and XPSNR. UI tables retain their
separate stable physical column mapping in `ui/main_window.py`.

`VmafOptions` deliberately keeps its legacy boolean fields and
`extra_features` list because those fields feed existing cache identities.
Use `metric_enabled`, `set_metric_enabled`, and `requested_metrics` for new
metric-aware code; do not reorder unknown feature strings.

`FrameScores` stores a dictionary of packed arrays and accepts future metric
keys without a model change. The five legacy properties remain compatibility
views. This flexibility is internal only: version-1 `.metrics.json` rows and
CSV output serialize the historical five columns in their fixed order. Do not
bump `run_io.FORMAT_VERSION` or silently add arbitrary keys to those files.

## Comparison recipe, requests and execution

`ComparisonRecipe` describes the scientifically compared pictures: crop
policy, manual crops, scaling algorithm and direction, duration limit, and a
possible resolution round-trip recipe. It intentionally excludes decode GPU,
GPU vendor, libvmaf thread count, and the application's parallel-job count.

Each requested result has a `MetricRequestSpec`: metric key, metric-specific
scientific parameters, frame coverage, and an implementation compatibility
identifier. VMAF variants therefore identify their model separately from the
common picture recipe. The current XPSNR exception remains explicit: XPSNR
alone has full-frame coverage despite `n_subsample`; a mixed libvmaf/XPSNR run
uses the sampled timeline. Those two outputs cannot share a cache entry.

`ExecutionPreferences` holds performance choices, separately from scientific
identity. `build_execution_plan` groups the current metrics into one
`legacy_ffmpeg` task, preserving the existing one-pass filtergraph. The Qt
worker constructs that plan in production before dispatching the established
runner. Multiple tasks are structurally supported, but no additional metric
backend exists yet.

## Generic metric results and provenance

`FrameMetricResult` owns a metric's own packed frame/time/value arrays;
different metrics do not need to share a sampling axis. `SequenceMetricResult`
stores its scalar score without inventing frame data. `MetricResultSet` holds
both kinds by key. `MetricProvenance` records implementation, version,
compute backend, compatibility ID, and metric parameters. Compute backend
means metric computation—not hardware video decoding.

`VmafRunResult` bridges this generic model to legacy consumers. It creates
generic results from `FrameScores` when loading old data, and only rebuilds a
legacy `FrameScores` view when established frame metrics share an exact axis.
Sequence and arbitrary future metrics remain generic rather than being
misaligned into the old container.

## Cache generations

The legacy flat cache remains a combined `.metrics.json` cache for rollback
and compatibility. The v2 internal metric cache is separate:

```text
<cache>/v2/<sha256 recipe hash>/context.json
<cache>/v2/<sha256 recipe hash>/<metric>_<sha256 request hash>.npz
```

Recipe hashes include source/distorted file identity (absolute path, size,
mtime) and `ComparisonRecipe`. Metric hashes include key, scientific
parameters, coverage, and implementation compatibility ID. Performance
preferences are deliberately excluded. Each metric is looked up directly—v2
does not enumerate metric subsets. NPZ loads disable pickle; corrupt one
artifact is a miss for that metric only.

During the transition, completed FFmpeg runs write both cache generations.
Lookup tries compatible v2 entries first, then legacy cache entries; a legacy
hit is lazily promoted using explicit `legacy-v1` provenance and is never
deleted. Cache clearing removes both generations. v2 refers only to this
internal cache architecture: portable `.metrics.json` and `.vmafrun.json`
remain format v1 because they cannot yet represent future sequence metrics.

User data lives under `~/.videometricslab/`, independent of checkout or launcher.
Existing `~/.vmaf-calculator/` data is migrated automatically on first launch.
The cache folder is configurable. FFmpeg location uses legacy QSettings keys.
Never rename storage paths as a cosmetic branding change.

## Playback and threading

The rolling pool retains the reference, current encode and adjacent encodes,
up to four streams. Native GStreamer/D3D11 presentation avoids sending decoded
pixels through Python. Frame locking and shared audio are coordinated by the
native pool; FFmpeg remains a fallback, not dead code.

Still extraction is a separate path. Display conversion is a preview decision,
not a change to the metric calculation recipe. Native HDR behavior depends on
the display, Windows HDR configuration, shader availability and codec support.

Workers produce data and signal completion; widgets are updated on the GUI
thread. File writes use a serial queue to preserve ordering. Capture cache
destinations when enqueueing, not when a delayed write eventually executes.
Shutdown cancels workers and waits for pending writes before closing.

## Performance rules

- Keep frame scores in arrays; avoid per-frame Python objects in hot loops.
- Plotting reduces data to pixel-scale min/max envelopes.
- Bound decode queues; do not preload all encodes in a series.
- Prefer per-input hardware decisions with software fallback.
- Profile CPU, memory, GPU and I/O separately before attributing a bottleneck.
- Background file writes temporarily suspend automatic cyclic collection;
  runnable cleanup and the deferred collection return to the GUI thread so
  PySide wrappers are never finalized by the writer. Reference counting stays
  active throughout. Preserve that thread-affinity rule when changing writes.
