# Changelog

Notable user-facing changes belong here.

## 1.0 - 2026-09-21

### Added

- GitHub contribution, security, community, architecture and user guides.
- Troubleshooting, known-issue and release-readiness documentation.

### Fixed

- Rejected cache reuse between incompatible subsampled and full-frame XPSNR runs.
- Restored cache lookup/clearing for both historical PSNR/SSIM selection orders.
- Removed old graph entries when recalculating a row.
- Handled mixed finite/infinite low percentiles and corrected XPSNR tooltips.
- Recovered from malformed settings-version values.

### Maintenance

- Removed unused eager VMAF statistics from completed-result wrappers.
- Documented removal of obsolete pyqtgraph installations; added Ruff to dev tools.

### Known issues

- Intermittent native garbage-collection crash under investigation; see
  [known issues](docs/KNOWN_ISSUES.md).
