# User guide

## Calculate metrics

1. In **Videos**, select a reference and add encoded/distorted files.
2. Select the metric checkboxes in each row. Cell edits apply to selected rows;
   header checkboxes apply across rows. VMAF, PSNR, SSIM and XPSNR are enabled
   by default; VMAF NEG, SSIMULACRA2 and Butteraugli can be selected separately.
3. Review crop, scale, model, duration and subsampling settings before starting.

The model selector includes Netflix VMAF v0 models and the bundled VMAF v1
models (1080p, phone, 4K, and high-frame-rate variants). VMAF v1 requires an
FFmpeg build linked against a libvmaf version that supports the v1 feature set;
older FFmpeg packages will report a clear model/feature error and should use a
v0 model or be upgraded. For SDR, VMAF v1 is best measured at 10-bit precision.
4. Choose **Calculate metrics**. Optional two-job parallelism can help on large
   CPUs, but increases resource use and is not always faster.
5. Inspect per-frame curves in **Metric Graphs**. Click a metric's mean column
   to show its detailed statistics. Export a graph PNG or CSV as needed.

SSIMULACRA2, Butteraugli and CVVDP are calculated on the GPU with Vship. Settings
> GPU metrics > GPU backend chooses Vship's build: Auto (the default) uses CUDA
on NVIDIA, HIP on AMD and Vulkan on other GPUs such as Intel's; Vulkan can also
be chosen on any GPU. If no GPU can be used, the input
format is unsupported, or GPU scoring fails, SSIMULACRA2 and Butteraugli fall
back to the bundled libjxl CPU implementation; CVVDP runs on the GPU only. GPU and CPU results are cached separately
because their implementations can produce different scores. These two metrics
are opt-in and do not add processing to runs where they are unchecked.

VMAF v0.6.1 and VMAF NEG are calculated on the GPU too: with a bundled libvmaf
built with CUDA on an NVIDIA GPU (GeForce GTX 16 and RTX 20 series or newer),
and with a Vulkan version of the same calculation on any other GPU, such as
Intel's and AMD's, or on every GPU when Settings > GPU metrics > GPU backend is
Vulkan. The two give the same scores to the last decimal. The choice is each
video's Performance > VMAF v0.6.1 and NEG compute, GPU by default and greyed out
at CPU without a GPU. A small integrated GPU can be slower at it than a fast
CPU: choose CPU there. Like
SSIMULACRA2's and Butteraugli's, the last choice is what newly added videos
start with. The GPU's scores agree with FFmpeg's libvmaf on the CPU to within a
thousandth of a point on every frame, so unlike those two, a saved VMAF score is
reused whichever is chosen. VMAF v1, PSNR, SSIM and XPSNR have no GPU code and
are still calculated by FFmpeg on the CPU, in a run of their own beside the
GPU's, so they do not slow VMAF down. A video's VMAF on the GPU waits for the
GPU like its other GPU metrics, one video at a time, and goes before them. If the GPU calculation fails, or crashes, that
video's VMAF is calculated on the CPU instead, as it is when the GPU's driver
fails the self-test the app runs on it when it starts. Without a GPU, FFmpeg's
libvmaf calculates everything.

The files must describe corresponding frames. The app rejects several timing
and geometry mismatches; this is not automatic content alignment. A source with
a different cut, opening sequence or frame offset must be aligned first.

## Interpreting scores

VMAF, VMAF NEG, PSNR, SSIM, XPSNR, SSIMULACRA2 and Butteraugli are different
measurements, not interchangeable quality percentages. Higher values generally
indicate closer agreement for the same comparison recipe, except Butteraugli,
where lower is better. Change the crop, scale or model and you change the
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

Save portable results as `.metrics.json` or export CSV. Cached results are reused
when files and relevant settings match. A compatible subset can load as
partially calculated; request calculation to fill missing metrics.

Settings and the default cache are under `~/.videometricslab/`. Back up that
folder before maintenance; changing checkout should not require clearing it.
File moves or replacements can cause cache misses. Use explicit saved-result
loading when you need to inspect a portable result. Right-click recalculation
forgets matching cached results, so do not use it just to refresh the display.

Result files may contain absolute media paths. Review them before sharing.
See [troubleshooting](TROUBLESHOOTING.md) for missing results or playback errors.
