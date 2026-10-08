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
video's Performance > VMAF compute, GPU by default and greyed out
at CPU without a GPU. A small integrated GPU can be slower at it than a fast
CPU: choose CPU there. Like
SSIMULACRA2's and Butteraugli's, the last choice is what newly added videos
start with. The GPU's scores agree with FFmpeg's libvmaf on the CPU to within a
thousandth of a point on every frame, so unlike those two, a saved VMAF score is
reused whichever is chosen. PSNR, SSIM and XPSNR have no GPU code and
are still calculated by FFmpeg on the CPU, in a run of their own beside the
GPU's, so they do not slow VMAF down.

VMAF v1 follows the same choice, with any GPU that has a Vulkan driver. Two of
its four features, detail loss (ADM) and motion, are calculated on the GPU with
Vulkan -- on an NVIDIA GPU too, whatever the GPU backend setting -- and give
libvmaf's CPU values to the last bit. The other two, banding (CAMBI) and colour
(SpEED), are calculated by the bundled libvmaf on the CPU, and libvmaf's own
model turns the four into the score, so VMAF v1 with the GPU is the score
FFmpeg's libvmaf gives. Because half of it stays on the CPU it gains less than
VMAF v0.6.1 does: about 5 times the CPU's speed on 4K video with a fast NVIDIA
GPU that also decodes both videos -- their pictures then stay on the GPU, and
the planes the CPU's half reads are written into its memory by the GPU itself
-- and about twice where a video is decoded by the CPU (VVC, for one), or with
another maker's GPU. A video too small for SpEED
(under about 160 pixels a side, more with the models that scale it down) is
calculated on the CPU. A video's VMAF on the GPU waits for the
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

## Command line

`VideoMetricsLab-cli.exe`, beside the app, calculates metrics without the
window (from source: `python -m vmaf_app.cli`):

```powershell
.\VideoMetricsLab-cli.exe compare reference.mkv encode1.mkv encode2.mkv
```

It runs what **Calculate metrics** runs and prints each video's scores, and
for several videos a summary side by side. The app's saved settings are its
defaults (FFmpeg's folder, the cache, the GPU backend, the metrics a new video
starts with); options override them for one run and nothing is written to the
settings. Scores go to the same cache, so a comparison made on the command
line opens in the app already calculated, and scores the app has saved are
not calculated again.

More examples:

```powershell
# Two metrics for every .mkv in a folder (the reference is left out of what a pattern finds)
.\VideoMetricsLab-cli.exe compare reference.mkv encodes\*.mkv -m vmaf,ssimulacra2

# Every metric, with each video's per-frame scores as results\<video>.csv
.\VideoMetricsLab-cli.exe compare reference.mkv a.mkv b.mkv -m all --csv results

# The first 30 seconds only, everything calculated on the CPU
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv --duration 30 --cpu

# A 1080p encode against a 4K reference, compared at 4K
.\VideoMetricsLab-cli.exe compare reference_4k.mkv encode_1080p.mkv --scale-to reference

# In a script: read the score from JSON
$run = .\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv -m vmaf --json - -q | ConvertFrom-Json
$run.videos[0].metrics.vmaf.score

# What would calculate each metric on this PC
.\VideoMetricsLab-cli.exe devices
```

| Option | Meaning |
| --- | --- |
| `-m`, `--metrics LIST` | Comma-separated: `vmaf`, `vmaf_neg`, `vmaf_v1`, `psnr`, `ssim`, `xpsnr`, `ssimulacra2`, `butteraugli`, `cvvdp`, or `all`. Left out: the metrics a video added to the app starts with. |
| `--model auto\|standard\|4k\|FILE` | VMAF v0.6.1's model. `auto` uses the 4K model for a 4K comparison. |
| `--black-bars auto\|none` | Detect and cut black bars, or compare whole pictures. |
| `--scale-to test\|reference` | Which video's size the other is scaled to when they differ. |
| `--scaler bicubic\|bilinear\|lanczos\|spline` | Scaling algorithm. |
| `--duration SECONDS` | Compare only the start of each video. |
| `--subsample N` | Score every N-th frame. CVVDP needs every frame. |
| `--cpu` | Calculate on the CPU every metric that can be (CVVDP is GPU only). |
| `--vmaf-on`, `--ssimulacra2-on`, `--butteraugli-on` `gpu\|cpu` | Where that one metric is calculated (`--vmaf-on` is VMAF v0.6.1, VMAF NEG and VMAF v1). Left out: as the app's Options panel was last set. |
| `--no-gpu-decode` | Decode the videos on the CPU. |
| `--threads N` | libvmaf threads. Left out: automatic. |
| `--parallel 1\|2` | Videos calculated at once on the CPU. |
| `--recalculate` | Ignore saved scores. |
| `--csv FOLDER` | Write each video's per-frame scores as `<video>.csv`, as **Export CSV** does. |
| `--json FILE` | Also write the results as JSON. `--json -` prints the JSON instead of the tables, for a script to read. |
| `-q`, `--quiet` | No progress, only the results. |
| `-v`, `--verbose` | The app's log as well. |

`compare --help` lists the same with examples. The last column of each
video's table says where a score was calculated: a metric asked for on the
GPU is calculated on the CPU where no GPU can.

Progress is written to the error stream and results to the output stream, so
`--json -` can be piped. The exit code is 0 when every video got every metric
asked for that it can have (one it cannot, such as CVVDP with `--subsample`,
is left out with a note, as in the app), 1 when a video failed or lost a
metric, 2 for a wrong command, when FFmpeg cannot be used or when there is no
test video, and 130 after Ctrl+C. Ctrl+C stops the run as
**Cancel** does; scores that had finished are saved.

Unlike the app, the command line does not ask before a long SSIMULACRA2 or
Butteraugli run on the CPU. Text is English whatever the app's language.

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
