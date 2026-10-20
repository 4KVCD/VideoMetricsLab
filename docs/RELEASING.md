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
   separate commits.
2. Publish libvmaf-fast first. The app is developed with a build of the
   fork's latest commit (`scripts/build_libvmaf_fast_local.ps1`); a release
   takes a release of it. Release libvmaf-fast from that commit, after its
   own checks (its `fast/scripts/package.ps1`), then set the new version and
   the archive's SHA-256 in `scripts/fetch_libvmaf_fast.ps1`, run it and
   commit what it installs. `scripts/build_release.ps1` refuses a local build
   (`release` false in `resources/tools/libvmaf/libvmaf-fast.json`).
3. Build with `scripts/build_release.ps1` ([BUILD.md](BUILD.md)), without
   `-AllowLocalLibvmaf`. Its self-test must report no failure, and only the
   warnings the PC explains (another GPU maker's decoder, say).
4. Run the tests, Release and Debug (`ctest --preset windows-release`,
   `windows-debug`, from `cpp`), and the CI workflow on GitHub (Actions > CI
   > Run workflow); and, on a PC of each GPU maker, the package's self-test
   (step 3). Record FFmpeg's version and each GPU's driver.
5. Run the golden suite (`cpp/README.md`) with the new build against a run
   of the last release's package, and once more with `--seed-app` set to
   that package's `vml-cli`, so that the new build reads the scores the last
   release saved. Every case must be the same, or differ only where a
   change made on purpose says why.
6. Test every metric, including SSIMULACRA2 and Butteraugli through both the
   Vship-fast GPU path and the libjxl CPU fallback. After updating
   Vship-fast, compare its SSIMULACRA2, Butteraugli and CVVDP against the
   build it replaces and against libjxl, on an NVIDIA and an Intel GPU: a
   build that scores one wrongly is not bundled. Check saved-result round-trip,
   backend-specific cache reuse, still comparison and independent bitrate
   analysis. On an NVIDIA GPU, compare a long run (30 minutes of 4K) with
   Performance > VMAF compute set to GPU and to CPU: every metric but VMAF
   and NEG identical, those within 0.002 per frame (libvmaf's GPU and CPU
   code round ADM and motion differently; official Netflix CUDA, without
   the fixes libvmaf-fast carries, was off by up to 0.13 at cuts). After updating
   libvmaf-fast, run the golden suite's GPU VMAF cases on a PC of each GPU
   maker; changes to libvmaf, its pull
   requests or the Vulkan engine are made in libvmaf-fast, which runs its
   own checks before each release.
7. Test Video Compare's playback on a PC of each GPU maker: H.264, HEVC,
   AV1, VP9, VVC (with H.274 and AFGS1 film grain) and ProRes, a raw `.266`
   stream, HDR on an HDR display and tone mapped on an SDR one, the
   soundtrack, seeking, neighbour switching, and holding and releasing the
   source.
8. Launch the extracted zip from another folder on a PC without Visual
   Studio: it carries Microsoft's C++ runtime, and must start and pass its
   self-test there. Check that it finds an external FFmpeg, and asks for one
   when there is none.
9. Audit the exact bundled components, preserve their notices, and satisfy
   applicable source and relinking obligations ([THIRD_PARTY.md](THIRD_PARTY.md)):
   Qt 6.12.0 under the LGPL 3 and its translations under the BSD 3-Clause
   licence (`licenses/`), Microsoft's C++ runtime, FFmpeg's libraries,
   dav1d, libvmaf-fast, Vship, libjxl's tools and the film grain library's
   VTM tables.
10. Update the changelog, choose a version/tag, and write release notes with
    known issues, requirements and tested hardware. Do not invent a release
    version from the development commit count.
11. Compute a SHA-256 checksum for the final zip using `Get-FileHash -Algorithm SHA256`.
12. Upload the zip and checksum to a GitHub Release only after approval.

The CI workflow checks code and offscreen tests, not GPU correctness or complete
binary licensing. A source upload and a redistributable Windows release are
different deliverables. Never include real user caches or settings in either.
