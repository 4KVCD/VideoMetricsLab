# Benchmark

The speed numbers for the release notes: the app's previous release against
this one, in whole runs as the Run button makes them (decoding and scoring),
on each computer, with `scripts/bench_app.py`. libvmaf-fast's own benchmark
(the libraries alone, against official libvmaf) is in its repository:
`fast/BENCHMARK.md`.

## What it measures

Only what changed since v1.4, each cell one optimization, at 4K and 1080p:

| Cell | Metrics | Test video | What changed |
|---|---|---|---|
| nvidia | VMAF + NEG (CUDA) | VVC | VVC decoded in the scoring process, not piped from FFmpeg |
| nvidia | VMAF v1 | HEVC | on the GPU, not FFmpeg's libvmaf on the CPU |
| gpu | VMAF + NEG | HEVC | with Vulkan on the PC's other GPU, not on the CPU |
| gpu | VMAF v1 | HEVC | the same |
| cpu | PSNR + SSIM + XPSNR | HEVC | libvmaf-fast in the app, not FFmpeg's filters |
| cpu | SSIMULACRA2 + Butteraugli on the CPU | HEVC | frame pairs scored in parallel |
| idle | the window, 10 s after it opens | | one thread for numpy's OpenBLAS, not one a CPU thread |

- **gpu** is the PC's GPU other than NVIDIA's (Intel's, AMD's). Where the PC
  has an NVIDIA GPU as well, it is hidden from those runs
  (`CUDA_VISIBLE_DEVICES=-1`, `VK_LOADER_DRIVERS_DISABLE=*nv-vk64*`): the app
  then runs as on a PC with only that GPU.
- **Left out**, as nothing changed for them (the run of 2026-10-08: 0.98x
  to 1.01x): Vship's metrics on the GPU, and VMAF on CUDA from HEVC. Also
  the default run of every metric, which mixes the cells above; the CPU
  metrics again with another GPU decoding; and VVC beside the GPU metrics,
  where VVC's decoding sets the pace.

## Method

- **Videos**: `VideoQ_HDR10_UHD_120fps_4m00s.mp4` (HEVC Main 10, HDR10, 120
  fps) against `VideoQ HDR10 4K H.265 CRF 22 medium.mkv`, `VideoQ HDR10
  1080p H.265 CRF 22 medium.mkv`, `VideoQ HDR10 4K VVC QP 32 faster.mkv` and
  `VideoQ HDR10 1080p VVC QP 32 faster.mkv`, from the start.
- **A run** is a process of its own with a temporary profile (no saved
  scores, default settings) that starts as the app does, its GPU probes
  first. Its speed is the steady state: the frames from 10% to 90% of the
  run, so start-up is left out. Its CPU use is over the same frames, as Task
  Manager shows it (the share of the PC's logical processors).
- **Length**: a run is sized to about 12 s of steady state at the rate the
  RTX 5090 PC measured; one that comes out under 6 s is made again at its
  own rate. One round; the versions take turns.
- **Scores**: both versions score the same first frames, and the report
  compares them. They are identical except where a change was intended: at
  1080p, VMAF, NEG and VMAF v1, as the source has been scaled where it is
  decoded since v2.0, not by FFmpeg's scale filter (up to 0.16 apart);
  XPSNR, weighted by the source since v2.0; and at 4K, VMAF and NEG on a GPU
  where v1.4 used the CPU (up to about 1e-4).
- **A quiet PC**: each run waits until the CPU is under 15% busy and no other
  process uses more than 15% of a GPU engine or half a CPU thread, and is
  made again once if one did while it ran (another project here encodes AV1
  on the GPU). Runs get the environment the benchmark started with: the app,
  imported to find the GPUs, sets OPENBLAS_NUM_THREADS, which would give the
  older version this release's memory saving.

## Running it

1. The previous release beside this checkout:

       git worktree add ..\VideoMetricsLab-v1.4 v1.4

2. The five videos in one folder.
3. Close what you can. Do not change Windows' power or display settings.
4. Run it (by the 2026-10-08 run's rates, about 20 minutes on the RTX 5090
   PC):

       .venv\Scripts\python.exe scripts\bench_app.py bench --old ..\VideoMetricsLab-v1.4 ^
           --videos "E:\Video encodings" --out %TEMP%\vml-bench-app

   `--only` takes groups (`nvidia`, `gpu`, `cpu`, `idle`) or cells
   (`--only "gpu/VMAF v1/hevc-4k"`). The idle measurement opens each
   version's window for 10 s.
5. `report.md` in the output folder: frames a second (old → new), CPU, peak
   memory and the score check per cell; `bench.log` has every run. Runs
   another process disturbed twice are marked.
