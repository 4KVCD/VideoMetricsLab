# Known limitations and investigation notes

This is development software. Passing unit tests does not establish identical
behavior on every GPU, display, driver or media format.

## Media and platform limits

- Windows is the primary tested platform. Native D3D11/HDR playback is Windows-only.
- Codec and GPU support depend on the installed GStreamer/FFmpeg build and driver.
  H.266 may require software decoding; real-time playback is not guaranteed.
- HDR appearance depends on metadata and display configuration. The fallback
  path does not offer all native presentation guarantees.
- Timing checks do not detect every content mismatch. Different edits/offsets
  are not automatically synchronized by scene content.
- XPSNR-only retains all frames; libvmaf-backed runs can be subsampled.
- Quality bands outside VMAF are heuristic, not standardized equivalents.
- Current dependency ranges are not a fully locked, reproducible environment.
- Unsigned builds can trigger Windows warnings. Verify provenance rather than
  bypassing warnings for downloads you do not trust.

See [release checks](RELEASING.md) for validation required before shipping.
