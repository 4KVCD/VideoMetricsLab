# Publication and release checklist

## Before first publication

- Include the project's MIT `LICENSE` and copyright notice (4KVCD).
- Review third-party notices and media/screenshot provenance. Do not assume
  the application's license covers bundled libraries or test media.
- Inspect tracked files **and history** for secrets and private information.
  `.gitignore` does not remove already committed content. Rotate exposed secrets.
- Confirm ownership/permission for all contributions and screenshots.
- Set the GitHub description, topics and repository URL; verify relative links
  from the hosted README.
- Enable Issues, private vulnerability reporting, and branch protection as
  appropriate. Choose a monitored security contact. Enable CI after upload.
- Do not publish a passing-status badge until the hosted checks actually pass.

## Release validation

1. Select a clean commit and record its hash. Keep individual bug fixes in
   separate commits with regression tests.
2. Install development dependencies in a fresh environment. Run `pip check`,
   Ruff and pytest. Record Python, dependency and FFmpeg versions.
3. Test every metric, including SSIMULACRA2 and Butteraugli through both the
   Vship GPU path and the libjxl CPU fallback. Check saved-result round-trip,
   backend-specific cache reuse, still comparison and independent bitrate
   analysis.
4. Test playback on real hardware: H.264/H.265 plus supported software-decoded
   formats, source hold/release, neighbor switching, HDR/SDR, seeking and audio.
5. Build using [BUILD.md](BUILD.md). Review self-test output: the script may
   produce a zip even when self-test warns of failure. Do not release that zip.
6. Launch the extracted bundle from a different directory on a clean Windows
   account or machine. Verify external FFmpeg discovery and graceful fallback.
7. Audit the exact bundled components, preserve their notices, and satisfy
   applicable source/relinking obligations. See [THIRD_PARTY.md](THIRD_PARTY.md).
8. Update the changelog, choose a version/tag, and write release notes with
   known issues, requirements and tested hardware. Do not invent a release
   version from the development commit count.
9. Compute a SHA-256 checksum for the final zip using `Get-FileHash -Algorithm SHA256`.
10. Upload the zip and checksum to a GitHub Release only after approval.

The CI workflow checks code and offscreen tests, not GPU correctness or complete
binary licensing. A source upload and a redistributable Windows release are
different deliverables. Never include real user caches or settings in either.
