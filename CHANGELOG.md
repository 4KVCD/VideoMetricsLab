# Changelog

## v1.2

- Added SSIMULACRA2 and Butteraugli. They are calculated on the GPU with the
  bundled Vship 5.1.1 (NVIDIA CUDA / AMD HIP), or on the CPU with the bundled
  libjxl 0.12.0 tools, and fall back to the CPU automatically when no supported
  GPU is found.
- Added a GPU/CPU choice for SSIMULACRA2 and Butteraugli in the Options panel.
  The Settings tab can tick either metric by default for new videos, and the
  GPU/CPU choice is remembered across restarts.
- GPU SSIMULACRA2/Butteraugli scoring runs at about 73-88 fps for a 4K pair,
  including VVC test videos.
- FFmpeg metrics (VMAF, PSNR, SSIM, XPSNR) and SSIMULACRA2/Butteraugli are
  calculated at the same time for each video. When two videos are calculated in
  parallel, only one GPU pass runs at a time.
- Added a warning, with an estimate of the temporary disk space needed, before
  calculating SSIMULACRA2 or Butteraugli on the CPU for videos longer than 10
  minutes.
- Added an "Add/remove metrics" button to the test-video table's corner to show
  or hide metric columns. Hidden metrics are not calculated.
- Adding a metric to a finished video calculates only that metric instead of
  recalculating the ones already saved.
- SSIMULACRA2 and Butteraugli show "n/a" on resolution round-trip rows instead
  of failing the whole row.
- Butteraugli graphs are drawn with 0 (best) at the top, the statistics table
  shows its worst frames ("10% High" ... "0.1% High"), and hovering snaps to its
  worst frames.
- The hover and Go-to-frame readout in Metric Graphs lists every calculated
  metric for that frame after a "|".
- Black-bar detection samples the whole video instead of only the duration
  limit. It uses at most two decoders across the app: about 2.4 GB of VRAM
  instead of 6.2 GB for a 4K pair on the GPU.
- Lowered the CPU used to feed GPU SSIMULACRA2/Butteraugli by about half, with
  identical scores.
- Fixed saved SSIMULACRA2/Butteraugli and VMAF scores disappearing when
  reloading a result whose metrics cover different frame counts.
- Fixed a video set to CPU showing a cached GPU SSIMULACRA2/Butteraugli score.
- Fixed GPU SSIMULACRA2/Butteraugli runs hanging when the GPU fails on the first
  frames; they now fall back to the CPU.
- Fixed the duration limit applying to only one of the two videos for CPU
  SSIMULACRA2/Butteraugli.
- Fixed the Butteraugli graph labelling its zero gridline "-0".
- Show the application version in the window title: VideoMetricsLab 1.2.

## v1.1.1

- Fixed the metric graph clipping VMAF v1 scores above 100.
- Show the application version in the window title: VideoMetricsLab 1.1.1.
- Omit calculation-library version metadata from non-VMAF cache entries

## v1.1

- Added bundled Netflix VMAF v1.0 model files, including standard, 4K, phone,
  and HFR variants.
- Refactored metric execution around a shared registry and backend plan.
- Added per-metric results, provenance, and cache identities so saved results
  remain tied to the exact metric implementation and settings.
- Improved graph/readout handling for the generalized metric architecture.
