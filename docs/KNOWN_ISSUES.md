# Known limitations and investigation notes

This is development software. Passing unit tests does not establish identical
behavior on every GPU, display, driver or media format.

## Media and platform limits

- Windows is the primary tested platform. Video Compare shows HDR10 and HLG in
  HDR on Windows with Windows HDR on (D3D11) and on a Mac's HDR display (Metal);
  elsewhere they are tone-mapped to SDR.
- Codec and GPU support depend on the GPU, its driver and the installed FFmpeg build.
  H.266 may require software decoding; real-time playback is not guaranteed.
- HDR appearance depends on metadata and display configuration. The fallback
  path does not offer all native presentation guarantees.
- Timing checks do not detect every content mismatch. Different edits/offsets
  are not automatically synchronized by scene content.
- Another program loading the GPU very heavily can make its video decoder give
  wrong pictures without any error (seen on an RTX 5090 with NVDEC and D3D11VA).
  When the app's own decoder notices, the run is scored again with its decoding
  on the CPU, and the rest of the run decodes there too (a note says so);
  FFmpeg's GPU decoding alone does not notice. Avoid scoring while another
  program saturates the GPU, or turn GPU decoding off.
- XPSNR-only retains all frames; libvmaf-backed runs can be subsampled.
- XPSNR of videos of more than 8 bits is calculated by the app alone (FFmpeg's
  `xpsnr` filter reads such pictures wrongly at some widths). Where the app
  cannot (a frame rate that is not constant, more than 12 bits, an odd size
  above 2048x1152, a resolution test on FFmpeg's route), XPSNR fails with that
  reason while the other metrics are kept.
- Quality bands outside VMAF are heuristic, not standardized equivalents.
- Unsigned builds can trigger Windows warnings. Verify provenance rather than
  bypassing warnings for downloads you do not trust.

See [release checks](RELEASING.md) for validation required before shipping.
