# Changelog

## v1.2

- Added GPU support for ColorVideo VDP, SSIMULACRA2, and Butteraugli with
  Vship integration.
- Added CPU fallback for SSIMULACRA2 and Butteraugli with libjxl. ColorVideo
  VDP has no CPU fallback.
- Added CVVDP per-second scores and graphing alongside its whole-video score.
- Added an Add/remove metrics control to the Videos tab so metric columns can
  be shown or hidden, and new metrics can be added to completed analyses
  without recalculating existing results.
- Included various small under-the-hood bug fixes and reliability improvements.

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
