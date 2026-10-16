# The C++ port

v2.1 is VideoMetricsLab rewritten from Python into C++ with Qt 6: every
tab, the command line, every metric and Video Compare. It does what v2.0
did, scores as the Python code it was ported from does, and reads the scores
v2.0 saved. What changes is how much it takes from your PC.

| | v2.0 | v2.1 | |
| --- | ---: | ---: | --- |
| Download | 86.7 MB | 39.1 MB | **2.2x** smaller |
| Installed | 254 MB, 382 files | 140 MB, 118 files | **1.8x** smaller |
| Opening the window | 0.69 s | 0.23 s | **3x** faster |
| Memory when idle | 87 MB | 26 MB | **3.3x** less |
| A run of three 4K test videos, every metric | 86.4 s | 69.3 s | **1.25x** faster |
| CPU time of that run | 591 s | 425 s | 28% less |

## How it was measured

On one PC: Core Ultra 9 285K, GeForce RTX 5090 (driver 616.92), a 120 Hz
display at 175% scaling, the videos on a hard disk. Both apps were run the
same way on the same files:

- **Runs:** the command line with its default settings, every metric, on
  the first 30 s of The Beekeeper in 4K (x264) against its x265, AV1 and
  VVC encodes, also in 4K.
- **Opening and idle memory:** the packaged apps, the median of five
  launches each, taken in turn. Memory is the private memory 5 s after the
  window showed.
- **The window:** the same scripted steps in both apps, with the window on
  screen; v2.0 was run from its source code for these.

## Runs

- The run above is **1.25x** faster and uses 28% less CPU. It starts 102
  programs instead of 132.
- SSIMULACRA2, Butteraugli and CVVDP share one pass over each video when the
  GPU has the memory for it, so each video is decoded once for the three of
  them instead of once each. On its own that makes the run **1.1x** faster.
- With the GPU backend on Auto, VMAF v0.6.1 and VMAF NEG use Vulkan on
  NVIDIA GPUs too, **3.2x-4.4x** faster than with CUDA (libvmaf-fast's own
  benchmark).
- Started from the window, the same run takes 70.1 s instead of 84.4 s, and
  the window's own process uses 4.2 s of CPU instead of 9.6 s.

## The window

| | v2.0 | v2.1 | |
| --- | ---: | ---: | --- |
| The window's thread while the pointer moves over a graph, 1000 Hz mouse | 98% of a CPU core | 16% | **6x** less |
| Pointer moves handled in those 3 s, of about 3,000 | 1,478 | 2,943 | **2x** more |
| One pointer move over a graph | 2.4 ms | 1.2 ms | **2x** faster |
| One zoom step on a graph | 11.7 ms | 3.9 ms | **3x** faster |
| The window's thread during a 30 s run | 3.7 s | 1.8 s | **2x** less |

## Video Compare

Video Compare now plays videos with FFmpeg's libraries instead of
GStreamer, which is no longer bundled.

| A 4K source and three 4K test videos | v2.0 | v2.1 |
| --- | ---: | ---: |
| Memory 40 s after you leave Video Compare | 7.6 GB | 0.1 GB |
| GPU memory 40 s after you leave it | 1.2 GB | 0 GB |
| Memory while it plays | 7.6 GB | 5.6 GB |

v2.0 kept Video Compare's decoders until the app closed. v2.1 lets them go
30 s after you leave the tab, and loads playback again when the pointer
reaches the tab or a tab beside it opens.

## The same scores

- 80 test cases on real videos, covering every metric and the ways the app
  calculates them, give identical results in v2.1 and the Python code it was
  ported from: the same tables, per-frame CSV files, JSON results, saved
  scores and exit codes.
- v2.1 keeps its saved scores in the same files as v2.0 and reads v2.0's.
