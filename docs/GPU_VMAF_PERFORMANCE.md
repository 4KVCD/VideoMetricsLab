# GPU-native VMAF measurements

Measured locally on 2026-10-03: Windows 11, RTX 5090 (driver 616.92), the
user's 24-core CPU. These are performance diagnostics, not release screenshots
or full-film quality measurements. No HoneyBee files were used.

## Scope

1. Move decoding and GPU scoring into an isolated native helper. Link stock
   shared FFmpeg libraries; do **not** make a custom FFmpeg build. Match
   timestamps using tiny identity tags, then keep hardware-decoded luma on
   the GPU. Software-decoded luma is uploaded directly from the decoder.
2. Create its shared CUDA context with blocking synchronization, allowing
   CPU threads to sleep during GPU waits.

Proposal 3 (reusable pinned CPU-decode buffers) is **not implemented**.
The existing `vmaf_gpu` worker task and CPU/GPU scheduling remain unchanged.

## Method

- Compressed Beekeeper UHD HEVC source, not the large lossless HoneyBee file.
- Test files: `The.Beekeeper 4k x265 slower 2000kbps.mkv` and
  `The.Beekeeper QP 23 medium H.267 800mb.mkv` (VVC).
- Source CUDA decode in both tests; HEVC test CUDA decode, VVC test CPU decode.
- 720 matched frame pairs starting at 28:00, approximately 30 seconds at
  23.976 fps. Equal 3840x1608 crops (source/VVC y=276; HEVC y=0), 10-bit,
  VMAF 4K v0.6.1, no subsampling. This fixed centered ROI isolates the
  transfer workload; it is not a claim about optimal cropping for the film.
- Warm each path once, exclude warm-ups, then run three serial measured
  repetitions with rotating order. No simultaneous competing metric jobs.
- Sum Windows process user+kernel time and the harness's Python CPU time.
  Include Python scoring/copying for the old raw-pipe path. Measure wall time
  through final scoring, and verify equal frame counts and per-frame scores.
- External FFmpeg 9.0.1 for the old path; stock BtbN shared FFmpeg
  n9.0.2-22-g46d8f462ee-20261001 for the native helper. Both use the same
  patched libvmaf DLL (reported version `51ea95ba`). Different decoder
  versions remain a potential benchmark confound, but measured output
  scores agree exactly on these clips.

CPU milliseconds per pair measure total CPU **work**. Average cores measure
CPU work divided by elapsed time. A faster CPU-decoded run can do less work
per frame while still keeping the same number of cores busy.

## Results

Arithmetic means of the three measured runs:

| Test | Path | CPU ms/pair | Frames/s | Average CPU cores | Peak RSS, MB |
| --- | --- | ---: | ---: | ---: | ---: |
| HEVC | Old raw-pipe GPU scoring | 137.71 | 55.07 | 7.58 | 2057 |
| HEVC | Proposal 1, automatic waits | 1.65 | 264.53 | 0.44 | 384 |
| HEVC | Proposals 1+2, blocking waits | 1.16 | 263.27 | 0.31 | 393 |
| VVC | Old raw-pipe GPU scoring | 291.85 | 48.25 | 14.08 | 4502 |
| VVC | Proposal 1, automatic waits | 124.05 | 116.07 | 14.39 | 3118 |
| VVC | Proposals 1+2, blocking waits | 123.10 | 115.10 | 14.16 | 3129 |

- **Proposal 1:** 98.80% less CPU work for HEVC, 57.50% less for VVC.
- **Proposal 2, incremental:** 29.39% less CPU work on the native HEVC path.
  VVC's 0.76% difference is too small to distinguish confidently from noise.
- **Combined:** 99.15% less CPU work and 4.78x throughput for HEVC; 57.82%
  less CPU work and 2.39x throughput for VVC.
- All 720 scores match at the recorded six-decimal JSON precision in every
  run; maximum score difference is zero. Means: HEVC 94.801912, VVC 96.481535.
- RSS is sampled every 50 ms. Old-path RSS includes its FFmpeg process and
  Python scoring harness; native-path RSS is the helper's process. These
  are not whole-GUI RAM totals and do not include GPU VRAM. VVC's CPU decoder
  retains frame-thread buffers, so its RAM use remains much higher than HEVC.

Raw measurements, including excluded warm-ups:
[HEVC](performance/gpu-vmaf-hevc.json), [VVC](performance/gpu-vmaf-vvc.json).

## Qualification and remaining tradeoffs

The fast path currently qualifies Windows NVIDIA CUDA scoring, 8/10-bit
4:2:0, equal dimensions after even-aligned crops. Resizing, mixed depths,
rotation, unsupported layouts or missing decoders retain the prior path.
CPU-decoded VVC still costs CPU time; no zero-copy GPU decode is claimed
for it. VMAF v1 still uses its existing CPU implementation.

The stock shared FFmpeg runtime adds approximately **155 MB installed**.
The NVIDIA driver is needed, but users do not need the CUDA Toolkit or a
C++ compiler. Before publishing a release, supply corresponding source and
dependency materials for the bundled LGPL build as described in
[BUILD.md](BUILD.md). Nothing in these measurements publishes a release.

Self-review caught and corrected a container/stream seek-origin mismatch,
unnecessary CPU thread pools for GPU decoders, Unicode kernel/output paths,
and selecting an alternate default video track instead of `v:0`. Regression
tests cover actual CUDA and CPU decoders, unequal frame rates, subsampling,
NEG, fractional duration limits, Unicode paths, alternate video tracks and
cancellation/process cleanup. The existing fallback ladder is also tested.

## Reproduce

Use the project's environment and the actual compressed media paths:

```powershell
.venv\Scripts\python.exe scripts/benchmark_vmaf_native.py `
  --reference '<compressed UHD reference.mkv>' `
  --test '<HEVC test.mkv>' --test-decode cuda --test-height 1608 `
  --ffmpeg '<ffmpeg.exe>' --start 1680 --frames 720 --repeat 3 `
  --output '<hevc-results.json>'
```

Repeat for the VVC test with `--test-decode cpu --test-height 2160`.
The script is a fixed 3840x1608, 10-bit diagnostic, not a general metric UI.

Actual GPU integration tests:

```powershell
$env:VML_TEST_NATIVE_GPU = '1'
.venv\Scripts\python.exe -m pytest -n 0 tests/test_vmaf_native_integration.py
Remove-Item Env:\VML_TEST_NATIVE_GPU
```
