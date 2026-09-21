# User guide

## Calculate metrics

1. In **Videos**, select a reference and add encoded/distorted files.
2. Select the metric checkboxes in each row. Cell edits apply to selected rows;
   header checkboxes apply across rows. All four metrics are enabled by default.
3. Review crop, scale, model, duration and subsampling settings before starting.
4. Choose **Calculate metrics**. Optional two-job parallelism can help on large
   CPUs, but increases resource use and is not always faster.
5. Inspect per-frame curves in **Metric Graphs**. Click a metric's mean column
   to show its detailed statistics. Export a graph PNG or CSV as needed.

The files must describe corresponding frames. The app rejects several timing
and geometry mismatches; this is not automatic content alignment. A source with
a different cut, opening sequence or frame offset must be aligned first.

## Interpreting scores

VMAF, PSNR, SSIM and XPSNR are different measurements, not interchangeable
quality percentages. Higher values generally indicate closer agreement for
the same comparison recipe. Change the crop, scale or model and you change the
question being measured. Use visual inspection alongside scores.

The graph's non-VMAF threshold bands are heuristic defaults from synthetic
calibration, not universal equivalents of VMAF thresholds. SSIM is displayed
with more decimal places than the dB metrics. XPSNR's sequence aggregate uses
its distortion-based convention, not an arithmetic average of dB values.
Identical XPSNR frames score infinity and contribute zero distortion while
remaining in the aggregate's frame count. An em dash indicates unavailable data.

Subsampling reduces libvmaf-backed scored frames. XPSNR-only currently keeps
every frame; do not interpret different coverage as identical measurements.

## Compare without calculating

Load a reference and encodes, then open **Video Compare**. Metric scores are
optional. Use still mode for frame inspection or playback mode for motion.

- Hold **S** to show the source; release it to return to the selected encode.
- Left/right switches encodes; Space toggles playback.
- Use the frame/timestamp controls to select a position.
- Review HDR preview and source-resolution options for your comparison.

This is an A/B switching view, not two permanently side-by-side players.
Crop geometry is most reliable when supplied by a completed analysis. Read
the preview status when an automatic crop has not yet been determined.
Keyboard shortcuts depend on focus; editing a text field may consume a key.

## Bitrate without calculating

Use **Bitrate Viewer** to add and analyze files independently. Metric runs also
add inputs for bitrate analysis. The three views show packet/frame size,
one-second bitrate and GOP-based bitrate. Only the first video stream is
counted: audio, subtitles and container overhead are excluded. Overall video
bitrate is video packet bytes × 8 divided by the measured video duration.

## Saved results and caches

Save portable results as `.vmafrun.json` or export CSV. Cached results are reused
when files and relevant settings match. A compatible subset can load as
partially calculated; request calculation to fill missing metrics.

Settings and the default cache are under `~/.videometricslab/`. Back up that
folder before maintenance; changing checkout should not require clearing it.
File moves or replacements can cause cache misses. Use explicit saved-result
loading when you need to inspect a portable result. Right-click recalculation
forgets matching cached results, so do not use it just to refresh the display.

Result files may contain absolute media paths. Review them before sharing.
See [troubleshooting](TROUBLESHOOTING.md) for missing results or playback errors.
