# Release benchmark

The speed numbers for the README and the release notes: libvmaf-fast's new
build against its last release (3.2.0-fast.1) and against official libvmaf,
on each computer. The main PC (RTX 5090, Core Ultra 9 285K and its iGPU) runs
it, and so does the AMD laptop (Radeon 780M, Ryzen 7 8845HS), with the same
script on the same frames. This page is the laptop's instructions as much as
a record of the plan. Brian approved the plan on 2026-10-07.

## What it measures

Scoring speed only: each video pair is decoded once into memory, and every
implementation scores the same frames. Decoding is not timed.

| Metric | Baselines | Ours (new build) |
|---|---|---|
| VMAF + NEG on the GPU | libvmaf-fast 3.2.0-fast.1 (CUDA, Vulkan); official libvmaf CUDA | CUDA (NVIDIA), Vulkan (every GPU) |
| VMAF + NEG on the CPU | official libvmaf | new |
| PSNR, SSIM (CPU) | official libvmaf; 3.2.0-fast.1 | new |
| XPSNR (CPU) | FFmpeg's `xpsnr` filter (official libvmaf has no XPSNR) | new |
| VMAF v1 | official libvmaf on the CPU; 3.2.0-fast.1 (GPU + CPU) | new, all on the GPU |
| GPU memory | official CUDA, 4K VMAF + NEG | new (NVIDIA only) |

- **Official libvmaf**: Netflix's master at acdd9376 (2026-10-05), built with
  libvmaf-fast's own build script and compiler, from a build folder inside
  `libvmaf\` (upstream's CUDA code needs that). Release 3.2.1 does not build
  with Visual Studio; libvmaf-fast is built on that master.
- **3.2.0-fast.1**: the app as it was before the VMAF v1 merge (bfc00d4), with
  the DLLs it bundled. Its VMAF v1 left CAMBI and SpEED to the CPU.
- **New**: this branch's app with its `vmaf_vulkan.dll` (libvmaf-fast
  a31318b9) and libvmaf-fast's `fast` libvmaf (a1af96ff, the same code).
- **Frames**: `VideoQ_HDR10_UHD_120fps_4m00s.mp4` (HEVC Main 10, HDR10, 120
  fps), frames 1200-1247 at 4K against `VideoQ HDR10 4K H.265 CRF 22
  medium.mkv`, frames 1200-1295 scaled to 1080p (lanczos) against `VideoQ
  HDR10 1080p H.265 CRF 22 medium.mkv`. 10-bit. VMAF v1 with the app's
  models: `vmaf_v1.0.16_3d0h` at 1080p, `vmaf_v1.0.16_1d5h_2160` at 4K.
- **Method**: every implementation is run once as a warm-up (not counted),
  then 5 more times. The implementations take turns, and each run is a process
  of its own. Each run scores the frames once (untimed; these scores are
  checked), then again and again for at least 10 s, timed from the end of
  one pass over the frames to the end of another: a scorer takes a frame
  pair only when it has room for it, so its queue is as full at both ends.
  The result is the median of the 5 runs. Each run waits
  until no other benchmark runs, no other process uses an NVIDIA GPU, and the
  CPU is under 15% busy.
- **Score checks**: each implementation's per-frame scores against its
  baseline's. Expected: libvmaf-fast's CPU, PSNR and SSIM identical to
  official libvmaf; Vulkan VMAF + NEG identical to libvmaf's CUDA; VMAF v1 on
  the GPU identical to official libvmaf's CPU; XPSNR identical to FFmpeg's
  (4 decimals). GPU VMAF + NEG differs from the CPU's by up to about 1e-4 (the
  CUDA code's motion, libvmaf issue 1562); official CUDA may differ more (the
  CUDA fixes libvmaf-fast carries). Anything else is a bug: report it, and do
  not publish the numbers.

## Running it on the AMD laptop

Brian brings these by hand (not in git), all from the main PC's
`E:\Video encodings`:

| File | Bytes |
|---|---|
| `VideoQ_HDR10_UHD_120fps_4m00s.mp4` | 1,055,015,060 |
| `VideoQ HDR10 4K H.265 CRF 22 medium.mkv` | 425,144,502 |
| `VideoQ HDR10 1080p H.265 CRF 22 medium.mkv` | 162,407,007 |
| `bench_kit\official\libvmaf.dll` (official libvmaf acdd9376) | SHA-256 `f257c34d...79b39bff` |
| `bench_kit\new\libvmaf.dll` (libvmaf-fast a1af96ff) | SHA-256 `7b57f8b5...9aad525b` |

`bench_kit\SHA256SUMS` has the two DLLs' full SHA-256. The VVC encodes are
not used here.

1. Pull `release/v1.5` (this commit or later). Nothing needs building: the
   app's bundled `vmaf_vulkan.dll` is the new engine, and nothing is decoded on
   the GPU.
2. Make a worktree of the app before the VMAF v1 merge, for the 3.2.0-fast.1
   rows:

       git worktree add ..\VideoMetricsLab-bench-old bfc00d4

3. Decode the frames (about 4 GB in `%TEMP%\vml-bench`):

       .venv\Scripts\python.exe scripts\bench_release.py prepare ^
           --reference "D:\path\VideoQ_HDR10_UHD_120fps_4m00s.mp4" ^
           --distorted-2160 "D:\path\VideoQ HDR10 4K H.265 CRF 22 medium.mkv" ^
           --distorted-1080 "D:\path\VideoQ HDR10 1080p H.265 CRF 22 medium.mkv"

   It prints the SHA-256 of the frames. They must be these, the main PC's;
   otherwise the computers did not score the same frames:

   | Frames | SHA-256 |
   |---|---|
   | 4K reference | `cf8353818597f3f0706a5a6c055423e2d17c3d9c3ba8ed728567b19efe391d0d` |
   | 4K distorted | `45cdcdf7e1f88e8c912ed477fd4dc4dbd5975792684838c0d8b01d2aee80b085` |
   | 1080p reference | `323c9efb220e1f4e07ac9643237aae0d814b9e58c409004c9239791c2c175c92` |
   | 1080p distorted | `e7ca5ff4452752c2352b82b0d11a97cf1825ec8d53b106bc9e247b5427d8369d` |

4. Plug the charger in and close what you can. Do not change Windows' power
   or display settings: the script only records the active power plan and
   whether it ran on battery.
5. Run it (about 2 hours; a line for each run, results written after each
   one):

       .venv\Scripts\python.exe scripts\bench_release.py run --kit "D:\path\bench_kit" ^
           --old-app ..\VideoMetricsLab-bench-old --cooldown 15

   `--cooldown 15` rests 15 s after each run, so the laptop starts each one
   at a similar temperature. There is no CUDA on the laptop: those rows are
   left out on their own.
6. Check `%TEMP%\vml-bench\results\<computer>.md`: no "Failed" section, no
   runs that started before the PC was quiet, and the Scores column as
   expected above.
7. Send back `<computer>.json` and `<computer>.md`: give them to Brian, or,
   if he says so, commit them as `docs/benchmark-results/<computer>.*` on a
   branch of their own and push that.

`scripts\bench_release.py report A.json B.json --out all.md` puts several
computers' results in one file.
