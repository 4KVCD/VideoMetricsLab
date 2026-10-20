# Architecture

## Boundaries

The app is C++ (`cpp`), in four parts, each a CMake target:

| Part | What it holds |
| --- | --- |
| `vml_platform` (`cpp/platform`) | What differs between systems: starting programs and their pipes, loading libraries, the console, the PATH, the GPUs. Its headers are the same on every system. |
| `vml_core` (`cpp/core`) | The engine: reading videos, planning and running the metrics, the saved scores and saved runs, statistics. |
| `vml_cli` (`cpp/cli`) | The command line, `vml-cli` (`VideoMetricsLab-cli.exe` in the package). |
| `vml_app` (`cpp/app`) | The window, `VideoMetricsLab` (Qt 6), with Video Compare's native playback, which only the window uses. |

The window and the command line use the engine; the engine uses neither,
and links no Qt and no operating-system headers, so the same code runs under
both and on every system. CMake holds that boundary: `vml_core` links only
`vml_platform` and its vendored libraries (`cpp/third_party`).

The native libraries in `native` (C++ with a C API, built by the
`scripts/build_*.ps1` into `resources/native`) are loaded when first needed:
the GPU frame decoders and the software decoder (`native/gpu_frames.h`),
Video Compare's Direct3D 11 presenter (`d3d11_tonemap.dll`) and its VVC film
grain (`d3d11_film_grain.dll`). The bundled libvmaf, Vship and FFmpeg's
libraries are loaded the same way, from `resources/tools`.

| Concern | Main files (`cpp/core/src` unless named) |
| --- | --- |
| Media metadata and geometry | `ffprobe.cpp`, `video.cpp`, `geometry.cpp`, `crop_detect.cpp`, `model_select.cpp` |
| Metric requests and planning | `analysis_request.cpp`, `options.cpp`, `job_runner.cpp` (`plan_tasks`, `run_jobs`, `JobScheduler`) |
| FFmpeg's runs | `ffmpeg_run.cpp`, `ffmpeg_tools.cpp`, `hardware_decoding.cpp` |
| Metrics calculated in the app | `libvmaf.cpp` (the bundled libvmaf), `libvmaf_cpu.cpp` (PSNR, SSIM, XPSNR), `gpu_vmaf.cpp`, `vmaf_cuda.cpp`, `vmaf_vulkan.cpp`, `vmaf_v1_gpu.cpp` (VMAF on the GPU), `gpu_frames.cpp` (decoding), `isolated.cpp` (the process of its own they run in) |
| SSIMULACRA2, Butteraugli, CVVDP | `vship.cpp`, `vship_run.cpp` (on the GPU), `perceptual_cpu.cpp` (libjxl's tools), `perceptual.cpp`, `cvvdp.cpp` |
| Metric registry | `metrics.cpp` |
| Results and persistence | `results.cpp`, `metric_cache.cpp`, `run_file.cpp`, `result_json.cpp`, `export_csv.cpp`, `settings.cpp` |
| Statistics and graphs | `stats.cpp`; `app/graph_panel.cpp`, `app/chart.cpp` |
| The window's runs | `app/run_worker.cpp`, `app/main_window_run.cpp`, `run_line.cpp` |
| Still comparison | `frame_extract.cpp`, `app/frame_compare_panel.cpp` |
| Native playback | `native_playback.cpp`, `soundtrack_clock.cpp`; `app/native_pool.cpp`, `native_decoder.cpp`, `native_device.cpp`, `native_presentation.cpp`, `native_soundtrack.cpp`, `native_grain.cpp`, `native_dav1d.cpp` |
| FFmpeg playback fallback | `video_playback.cpp`, `app/playback_worker.cpp` |
| Bitrate analysis | `bitrate.cpp`, `app/bitrate_panel.cpp` |
| Processes and cancelling | `cpp/platform` (`process.h`), `cancel.cpp` |

## Metric lifecycle

1. The window reads the reference and the test videos off its thread
   (`app/probe_worker.cpp`, `ffprobe.cpp`) and checks they can be compared.
2. A run takes each row's options as they are into a `VmafJob`
   (`job_runner.h`): the two videos, the options, the metrics asked for, and
   the tasks that calculate them (`plan_tasks`). A task is the part of a
   video's metrics one program calculates in one place: FFmpeg's metrics on
   the CPU (PSNR, SSIM and XPSNR in the app where it decodes the videos
   itself), VMAF, NEG and v1 on the GPU, Vship's metrics on the GPU, and
   SSIMULACRA2 and Butteraugli set to the CPU, with libjxl's tools.
3. Saved scores are looked up by the request's identity (`metric_cache.h`):
   the comparison recipe and each metric's request, not where or how it is
   calculated. A saved score fills in its metric, and a task whose metrics
   are all saved is left out.
4. The CPU's queue runs two tasks at once at most (the parallel jobs
   setting), the GPU's one. GPU work runs in a process of its own
   (`isolated.h`): a crash in libvmaf, Vship or the GPU's driver ends that
   process, not the run, and FFmpeg's filters then calculate VMAF and the
   others on the CPU, and libjxl's tools SSIMULACRA2 and Butteraugli.
5. Each task's scores come back as a `MetricResultSet` (`results.h`): each
   metric's frame or sequence result, with its provenance.
6. The window shows each result as it arrives, saves its scores off its
   thread (`app/file_writes.cpp`), and draws the graphs.

A run works on its own copy of each row's options: a row edited while it
runs changes the next run, not this one. A finished result is attached only
to the row, source and settings it was calculated for.

## Persistence contracts

The C++ app reads and writes the Python app's files, so a user's saved
scores, saved runs and settings carry over from earlier versions, and either
app opens the other's. (SSIMULACRA2, Butteraugli and CVVDP scores saved on
the GPU before v2.2 are calculated again: Vship-fast computes them otherwise,
score_identity.h.)

Saved runs are `.metrics.json` files (format v2, `run_file.h`): each metric
serialized on its own, with its kind, its frame or time axis where it has
one, its values and its provenance, so that metrics sampled differently
round-trip without being flattened into one table. A missing frame value is
JSON null.

The saved scores are kept per metric:

```text
<cache>/v2/<sha256 recipe hash>/context.json
<cache>/v2/<sha256 recipe hash>/<metric>_<sha256 request hash>.npz
```

The recipe hash takes both files' identities (full path, size and
modification time: `file_identity`) and the comparison recipe; the request
hash the metric's key, its scientific parameters, its frame coverage and its
implementation compatibility ID. Where and how a metric is calculated (GPU
or CPU, the GPU's maker, threads) is left out, so a score calculated one way
stands for the other wherever the implementations declare themselves the
same. A damaged file is a miss for its metric only.

User data lives under `~/.videometricslab/` (settings, the saved scores by
default, the log), whatever folder the app runs from.

## Playback and threading

Video Compare plays a pair natively where it can: one decoder per video
(the source, and the selected test video with its neighbours), each test
frame paired with the source frame of the same number before the one
presenter draws either, in time with the source's soundtrack
(`app/native_pool.h`). Where native playback cannot play a pair, the frames
come from `ffmpeg.exe`. HDR shows as HDR on a display in HDR mode, and is
tone mapped to SDR on any other, or when the user chooses HDR to SDR.

Work that takes time (reading videos, runs, file writes, playback's decoding)
runs off the window's thread and reports with queued signals, so the window
sees events on its own thread, in the order they were sent. File writes run
on one thread of their own, in the order they were asked for, and closing
the window ends the run and waits for them.

## Performance rules

- Keep frame scores in arrays (32-bit floats, as libvmaf and FFmpeg give
  them), not per-frame objects.
- Graphs reduce each line to one min/max pair per pixel column.
- Bound decode queues; do not decode every encode of a series ahead.
- Decide hardware decoding per video, with software as the fallback.
- Measure CPU, memory, GPU and I/O separately before naming a bottleneck.
- A speedup must not change any score.
