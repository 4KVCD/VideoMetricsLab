# Architecture

## Boundaries

`vmaf_app.main` creates the Qt application. `ui/main_window.py` owns video rows,
options and orchestration. The UI imports `core`; core does not import UI.
`tests/test_architecture.py` checks this boundary. Core is not completely
Qt-free: tool discovery uses QSettings and native playback loads GI bindings.

| Concern | Main modules |
| --- | --- |
| Media metadata and geometry | `core/ffprobe.py`, `crop_detect.py`, `model_select.py` |
| Metric requests and execution | `core/analysis_request.py`, `ffmpeg_request.py`, `execution.py`, `vmaf_runner.py`, `ui/worker.py` |
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
2. Snapshot the row's mutable FFmpeg/UI options into an immutable
   `AnalysisRequest`: common `ComparisonRecipe`, metric request specs, and
   execution preferences.
3. Check cached results using only the request's scientific identity.
4. Group missing metrics by backend. The current FFmpeg backend still builds
   one efficient filtergraph for its requested metrics; XPSNR alone does not
   require a VMAF score. SSIMULACRA2 and Butteraugli use the perceptual backend,
   which tries Vship GPU scoring and falls back to the bundled libjxl CPU tools.
5. Parse per-frame logs into `FrameScores`, a collection of packed NumPy arrays.
6. Deliver results to the UI, update graph identities, and queue cache writes.

Do not mutate a running job's options in place. `clone_options` copies the
mutable feature list. Completion must verify that the row/source/settings
still match before attaching a result. Clearing a result must also remove its
graph identity, without removing unrelated imported curves.

## Persistence contracts

Portable user-saved results use the current format-v2 `.metrics.json` format.
Each metric is serialized independently with its own kind, frame/time axis when
applicable, values, and provenance, so sequence metrics and independently
sampled frame metrics round-trip without being flattened into a shared table.
Missing frame values use JSON null and infinities use explicit strings. Use
`run_io` rather than ad-hoc serialization.

Internal caching is per metric. Cache identity includes resolved file paths,
size, mtime, the common comparison recipe, metric-specific scientific
parameters, frame coverage, and implementation compatibility ID. Execution-only
controls such as GPU vendor and thread count are excluded. Partial cache hits
are valid and can be displayed while missing metrics are recomputed. With
subsampling, XPSNR-only and libvmaf-backed results have different frame
coverage and must not be interchanged.

## Metric architecture

`core/metrics.py` is the sole registry for metric labels, formatting,
thresholds and sequence aggregation. It is headless: it must not import Qt,
ffmpeg wrappers or result models. Metrics may optionally carry an
`FfmpegMetricBinding` for today's FFmpeg/UI adapter; a metric implemented by a
different backend does not need a `VmafOptions` field just to join the
registry. Current logical order is VMAF, VMAF NEG, PSNR, SSIM, XPSNR,
SSIMULACRA2 and Butteraugli. UI
tables retain their separate stable physical column mapping in
`ui/main_window.py`.

`VmafOptions` is now confined to the current FFmpeg/UI edge.
`core/ffmpeg_request.py` snapshots it into the backend-neutral request model;
generic cache and planning code must not import or inspect `VmafOptions`.

`FrameScores` is the shared-axis view used by established UI and CSV paths.
The five legacy convenience properties remain typed views over its packed arrays;
the newer independent-backend results use the generic metric result model.
It is not the persistence model: portable `.metrics.json` stores the generic
`MetricResultSet` directly. CSV intentionally remains a shared-frame export of
the current five displayed metrics.

## Comparison recipe, requests and execution

`ComparisonRecipe` describes the scientifically compared pictures: crop
policy, manual crops, scaling algorithm and direction, duration limit, and a
possible resolution round-trip recipe. It intentionally excludes decode GPU,
GPU vendor, libvmaf thread count, and the application's parallel-job count.

`AnalysisRequest` is the backend-neutral contract used by cache and planning.
It contains the common recipe, immutable `MetricRequestSpec` objects, and
`ExecutionPreferences`. Each metric spec carries its metric key, `backend_id`,
metric-specific scientific parameters, frame coverage, and an implementation
compatibility identifier. Backend routing is not cache identity: two backends
can reuse a result only when they explicitly declare the same implementation
compatibility ID and scientific parameters.

`core/ffmpeg_request.py` translates today's mutable `VmafOptions` into that
request model. VMAF variants identify their model separately from the common
picture recipe. The XPSNR exception remains explicit: XPSNR alone has
full-frame coverage despite `n_subsample`; a mixed libvmaf/XPSNR run uses the
sampled timeline. Those outputs cannot share a cache entry.

`build_execution_plan` only sees `AnalysisRequest`. It groups metrics by
`backend_id`, skips a backend group only when every metric in that group is
already cached, and otherwise keeps the group together so coupled backends can
execute efficiently. The current Qt worker dispatches the `ffmpeg` group to
the established runner; future CPU/GPU backends can create additional groups
without adding branches to cache identity or the planner.

## Generic metric results and provenance

`FrameMetricResult` owns a metric's own packed frame/time/value arrays;
different metrics do not need to share a sampling axis. `SequenceMetricResult`
stores its scalar score without inventing frame data. `MetricResultSet` holds
both kinds by key. `MetricProvenance` records implementation, version,
compute backend, compatibility ID, and metric parameters. Compute backend
means metric computation—not hardware video decoding.

`ComparisonResult` currently carries both the authoritative generic result set and
the shared-axis `FrameScores` view used by established UI/frame-oriented code.
It creates generic results from a supplied frame view when needed. When generic
frame metrics use different axes, the UI projection keeps the first registered
axis and includes only metrics aligned to it; independently sampled metrics stay
authoritative in `MetricResultSet` instead of blanking or corrupting the view.
Sequence metrics never need invented frame data.

## Metric cache

The internal cache stores each metric independently:

```text
<cache>/v2/<sha256 recipe hash>/context.json
<cache>/v2/<sha256 recipe hash>/<metric>_<sha256 request hash>.npz
```

Recipe hashes include source/distorted file identity (absolute path, size,
mtime) and `ComparisonRecipe`. Metric hashes include key, scientific
parameters, coverage, and implementation compatibility ID. Backend routing and
performance preferences are deliberately excluded. `result_cache` consumes
`AnalysisRequest` directly; FFmpeg/UI options are translated before reaching
the cache. Each metric is looked up directly—v2 does not enumerate metric
subsets. NPZ loads disable pickle; corrupt one artifact is a miss for that
metric only.

Completed runs write only this per-metric cache. Lookup probes requested
metric identities directly and may also load compatible supplemental metrics
for the current UI. Clearing a row removes only the metric identities that row
could load, leaving scientifically different coverage/model entries intact.
Portable `.metrics.json` files are a separate user-facing persistence format
and are not used as cache entries.

User data lives under `~/.videometricslab/`, independent of checkout or
launcher. The cache folder is configurable.

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
