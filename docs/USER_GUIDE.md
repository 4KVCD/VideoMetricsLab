# User guide

## Calculate metrics

1. In **Videos**, select a reference and add encoded/distorted files: drop
   them on the test list (a file dropped on the reference box replaces the
   reference), paste a reference path, or use Browse. Drag rows to reorder
   them; click a column heading to sort by it, and again to reverse.
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

**Highlight best/worst results**, under the video table, colours each metric's
best score green and its worst red among the test videos that have one.

**Settings > Window** chooses the theme (Same as Windows, Light or Dark) and
whether the app reopens the videos it had open when it was last closed (off by
default).

SSIMULACRA2, Butteraugli and CVVDP are calculated on the GPU with Vship-fast, a
faster Vship whose metrics compute as libjxl's and the reference CVVDP's do,
with Vulkan on any GPU (NVIDIA's, AMD's and Intel's). If no GPU can be used,
the input format is unsupported, or GPU scoring fails, SSIMULACRA2 and
Butteraugli are calculated with the bundled libjxl on the CPU; CVVDP runs on
the GPU only. The GPU and the CPU convert the video to RGB differently, so
their scores differ: on a PC whose GPU can calculate them, a score calculated
on the CPU is marked "(CPU)" and the next run calculates it again on the GPU.
These metrics are opt-in and do not add processing to runs where they are
unchecked.

Each video's Performance > Vship compute chooses the Vship: GPU - Vship-fast,
or GPU - Vship, stock Vship 5.1.2 (upstream's release). Settings > GPU metrics
sets the one newly added videos start with, and the command line's default.
The two give slightly different scores, and each one's are saved apart:
switching a video shows the other one's saved scores, or calculates them on
the next run. Where Vship-fast fails on the GPU, Vship 5.1.2 calculates its
metrics instead, and such a score is marked "(Vship)" and calculated again
with Vship-fast on the next run. On a PC whose GPU Vship-fast cannot use at
all but Vship 5.1.2 can, a Vship-fast video's scores are Vship 5.1.2's, kept
and reused as its own. A video set to GPU - Vship never uses
Vship-fast: where Vship 5.1.2 fails, SSIMULACRA2 and Butteraugli are
calculated on the CPU, and CVVDP is not.

VMAF v0.6.1 and VMAF NEG are calculated on the GPU too, with a Vulkan version
of libvmaf's calculation on any GPU (NVIDIA's, AMD's and Intel's). An NVIDIA
GPU (GeForce GTX 16 and RTX 20 series or newer) whose Vulkan cannot calculate
them uses a bundled libvmaf built with CUDA instead. The two give the same
scores to the last decimal. The choice is each
video's Performance > VMAF compute, GPU by default and greyed out
at CPU without a GPU. A small integrated GPU can be slower at it than a fast
CPU: choose CPU there. The last choice is what newly added videos start with.
The GPU's scores agree with FFmpeg's libvmaf on the CPU to within a thousandth
of a point on every frame, so a saved VMAF score is reused whichever is
chosen. PSNR, SSIM and XPSNR have no GPU code: the app
calculates them on the CPU with the bundled libvmaf-fast, beside the GPU's run,
so they do not slow VMAF down. With VMAF on the CPU, FFmpeg calculates them in
the same run.

VMAF v1 follows the same choice, with any GPU that has a Vulkan driver. Its
four features, detail loss (ADM), motion, banding (CAMBI) and colour (SpEED),
are calculated on the GPU with Vulkan -- on an NVIDIA GPU too -- and give libvmaf's CPU values to the last bit, and libvmaf's
own model turns them into the score, so VMAF v1 with the GPU is the score
FFmpeg's libvmaf gives. It is about 17 times the CPU's speed on 4K video with
an RTX 5090 that also decodes both videos, 2.5 times with a VVC test video
(decoded by the CPU, which then sets the pace), and 1.4 to 1.7 times on an
Intel integrated GPU. A video too small for SpEED
(under about 160 pixels a side, more with the models that scale it down) is
calculated on the CPU. A video's VMAF on the GPU waits for the
GPU like its other GPU metrics, one video at a time, and goes before them. If the GPU calculation fails, or crashes, that
video's VMAF is calculated on the CPU instead, as it is when the GPU's driver
fails the self-test the app runs on it when it starts. Without a GPU, FFmpeg's
libvmaf calculates everything.

The files must describe corresponding frames. The app rejects several timing
and geometry mismatches; this is not automatic content alignment. A source with
a different cut, opening sequence or frame offset must be aligned first.

### AV2 videos

AV2, AOM's codec after AV1 (not yet final), is decoded by dav2d, VideoLAN's
AV2 decoder, on the CPU: FFmpeg knows no AV2 and no GPU decodes it, so the
app reads AV2 files itself. It reads them as AOM's AV2 encoder (AVM's
`avmenc`) writes them: WebM (its default), IVF (`--ivf`) and raw OBU streams
(`--obu`, `.obu`; on the command line with `--fps`, as other raw streams).
Their film grain can be scored with it or without it, as AV1's. dav2d's
pictures are AVM's own decoder's, checked at 8 and 10 bits, with and without
film grain. A WebM or IVF file's frame rate is found from its timestamps.

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

PSNR, SSIM and XPSNR are of the luma (Y) plane alone: PSNR is what an
encoder reports as PSNR-Y, not a weighted average of Y, U and V, and its
score is the mean of the frames' PSNR. XPSNR takes the reference as the
original, which FFmpeg's `xpsnr` filter wants as its first input whatever
its inputs are called, so it differs from FFmpeg commands that give the test
video first (`[dist][ref]xpsnr`).

Subsampling reduces libvmaf-backed scored frames. XPSNR-only currently keeps
every frame; do not interpret different coverage as identical measurements.

## Compare without calculating

Load a reference and encodes, then open **Video Compare**. Metric scores are
optional. Use still mode for frame inspection or playback mode for motion.

- Hold **S** to show the source; release it to return to the selected encode.
- Left/right switches encodes; Space toggles playback.
- Use the frame/timestamp controls to select a position.
- **Zoom** shows the frame fitted to the window, or at a size you choose or type in:
  100% is the video's own pixels. Drag a zoomed frame to move it.
- A video whose stream asks the player to add film grain (VVC's H.274
  grain, such as VVenC's film grain analysis writes, AV1's, AFGS1) is shown
  as it is scored: with its grain, or without it if it is scored without
  (Film grain in the Options panel). VVC's H.274 grain is added by VVdeC,
  at any bit depth.
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
window (from a source build: `cpp\build\windows\cli\Release\vml-cli.exe`):

```powershell
.\VideoMetricsLab-cli.exe compare reference.mkv encode1.mkv encode2.mkv
```

It runs what **Calculate metrics** runs and prints each video's scores, and
for several videos a summary side by side. The app's saved settings are its
defaults (FFmpeg's folder, the cache, the metrics a new video starts
with); options override them for one run and nothing is written to the
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

# HDR10 in files with no colour tags (y4m), the first 33 frames (CVVDP models an HDR display for HDR video)
.\VideoMetricsLab-cli.exe compare reference.y4m encode.y4m -m cvvdp --colour pq-bt2020 --frames 33

# CVVDP on two displays of your choice: a 30-inch HDR monitor and a 65-inch HDR TV
.\VideoMetricsLab-cli.exe compare reference.mkv encode.mkv -m cvvdp --cvvdp-display "30-inch 4K HDR monitor, dim room" --cvvdp-display "65-inch 4K HDR TV, 1000 nits, living room"

# A raw VVC stream straight from an encoder (no container), at 23.976 frames a second
.\VideoMetricsLab-cli.exe compare reference.y4m encode.266 --fps 24000/1001

# A raw YUV reference (pictures only: 1080p, 10-bit 4:2:0) and a raw VVC stream
.\VideoMetricsLab-cli.exe compare reference.yuv encode.266 --size 1920x1080 --pix-fmt yuv420p10le --fps 24000/1001

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
| `--model-v1 MODEL` | VMAF v1's model, as the Options panel's **VMAF v1 model**: `auto` (the one for the size frames are compared at), or a bundled one by its name: `vmaf_v1_3d0h`, `vmaf_v1_5d0h` (1080p), `vmaf_v1_1d5h_2160`, `vmaf_v1_3d0h_2160` (4K), or their high frame rate models `vmaf_v1_hfr_3d0h`, `vmaf_v1_hfr_5d0h`, `vmaf_v1_hfr_1d5h_2160`, `vmaf_v1_hfr_3d0h_2160`. |
| `--black-bars auto\|none` | Detect and cut black bars, or compare whole pictures. |
| `--scale-to test\|reference` | Which video's size the other is scaled to when they differ. |
| `--scaler bicubic\|bilinear\|lanczos\|spline` | Scaling algorithm. |
| `--duration SECONDS` | Compare only the start of each video. |
| `--frames N` | Compare only the first N frames of each video, such as a short encode of a longer reference. |
| `--colour pq-bt2020\|hlg-bt2020\|bt709` | The colours of a video with no colour tags (y4m, say); a tagged video keeps its own. Left out, such a video takes the other video's tags when that one has them (a y4m encode of a tagged HDR10 source is read as HDR10), and is otherwise taken as SDR, which changes CVVDP, SSIMULACRA2 and Butteraugli. Scores with colours given or taken are not saved. |
| `--film-grain with\|without` | Score with the film grain a video's stream asks the player to add, as players show it, or without it, the coded pictures only. Left out: as Settings' **Film grain** default says; with, where it asks. |
| `--fps RATE` | The frame rate of a raw video stream or raw YUV file (below), such as `24000/1001`, `23.976` or `60`. Required when the reference or a test video is one. These scores are not saved. |
| `--size WIDTHxHEIGHT`, `--pix-fmt FORMAT` | The picture size and pixel format (as FFmpeg names it: `yuv420p`, `yuv420p10le`...) of a raw YUV file (below). Required with `--fps` for one. |
| `--cvvdp-display PRESET` | The display CVVDP models, by preset name. Repeat it for a CVVDP score on each display. Left out: the app's default display, Automatic unless Settings name a preset: the HDR display (`30-inch 4K HDR monitor, dim room`, the official `standard_hdr_pq`) for an HDR reference (PQ or HLG), the SDR one (`30-inch 4K monitor, office`, `standard_4k`) otherwise. |
| `--cvvdp-fill` | Scale the compared frames up to fill CVVDP's display, aspect kept, as the Options panel's **Scale the video to fill the display**. Left out: off, as the official tool (a 1080p video covers a quarter of a 4K display). These scores are saved apart. |
| `--subsample N` | Score every N-th frame. CVVDP needs every frame. |
| `--cpu` | Calculate on the CPU every metric that can be (CVVDP is GPU only): VMAF as with `--vmaf-on cpu`, and SSIMULACRA2 and Butteraugli with libjxl rather than on the GPU. |
| `--vmaf-on` `gpu\|cpu` | Where VMAF v0.6.1, VMAF NEG and VMAF v1 are calculated. Left out: as the app's Options panel was last set. |
| `--vship` `vship-fast\|vship` | The Vship that calculates SSIMULACRA2, Butteraugli and CVVDP on the GPU: Vship-fast, or stock Vship 5.1.2, whose scores differ slightly and are saved apart. Left out: as Settings > GPU metrics sets it. |
| `--vship-pass one\|each` | SSIMULACRA2, Butteraugli and CVVDP in one pass per video, which decodes it once (faster only on a very powerful GPU, and needs much GPU memory), or a pass for each. The scores are the same. Left out: as Settings > GPU metrics sets it. |
| `--no-gpu-decode` | Decode the videos on the CPU. |
| `--gpu-decode auto\|nvidia\|intel\|amd` | Decode the videos on the GPU, as the Options panel's **GPU decode**: `auto`, the first GPU whose decoder takes each video (NVIDIA, Intel, then AMD), or that maker's, which the PC must have. A Mac uses its own. Left out: on or off as the app's Options panel was last set, the maker automatic. |
| `--threads N` | libvmaf threads. Left out: automatic. |
| `--parallel 1\|2` | Videos calculated at once on the CPU. |
| `--recalculate` | Ignore saved scores. |
| `--csv FOLDER` | Write each video's per-frame scores as `<video>.csv`, as **Export CSV** does. |
| `--json FILE` | Also write the results as JSON. `--json -` prints the JSON instead of the tables, for a script to read. |
| `-q`, `--quiet` | No progress, only the results. |
| `-v`, `--verbose` | The app's log as well. |

`compare --help` lists the same with examples. The last column of each
video's table says where a score was calculated: a metric asked for on the
GPU is calculated on the CPU where no GPU can. When a GPU metric falls back,
to the CPU or from Vship-fast to stock Vship 5.1.2, because it failed on the
GPU during the run or because the GPU's check crashed, failed or scored
wrongly, a note says so and why (in the JSON too, with the video's other
notes). A Vship pass whose FFmpeg fails to give it the videos' frames is
first made once more on Vship-fast, with no note: only a second failure
falls back. The app lists the fallbacks under **GPU fallbacks** in the
tooltip of the video's name.

Progress is written to the error stream and results to the output stream, so
`--json -` can be piped. Piped or redirected, both are UTF-8 whatever the
system's code page, so a script reads them as UTF-8 (in Python,
`subprocess.run(..., encoding="utf-8")`); a console is sent every character
as it is, whatever its code page.

The exit code is 0 when every video got every metric asked for that it can
have (one it cannot, such as CVVDP with `--subsample`, is left out with a
note, as in the app), 1 when a video failed or lost a metric, 2 for a wrong
command (a raw stream without `--fps`, or a raw YUV file without `--size`,
`--pix-fmt` and `--fps`, among them), when FFmpeg cannot be used or when
there is no test video, and 130 after Ctrl+C.
Ctrl+C stops the run as **Cancel** does; scores that had finished are saved.
Closing the console window does the same on Windows' classic console.

Unlike the app, the command line does not ask before a long SSIMULACRA2 or
Butteraugli run on the CPU. Text is English whatever the app's language.

### Raw video streams

An encoder's output with no container (`.264`, `.265`, `.hevc`, `.266`,
`.obu`, `.m2v`, `.m4v`, `.mjpeg`...) has no timestamps and records no frame
rate, so the command line needs `--fps` for it. Its frames are counted when
the run starts (the file is read once, not decoded) and timed at that rate,
and FFmpeg decodes it. A raw stream scores exactly as the same stream in a
container: checked for H.264, HEVC, VVC, AV1 (OBU), EVC, MPEG-1, MPEG-2,
MPEG-4 Part 2, H.263, H.261, MJPEG and Dirac. FFmpeg also reads raw AVS2,
AVS3, CAVS, VC-1 and APV streams; these are not checked (FFmpeg's AVS2
decoder gave the one tried fewer frames than it holds, in a container too,
so it was refused as cut short).

A container can say what a raw stream may not: its colours and where its
chroma samples sit, which an encoder may leave out and which change
SSIMULACRA2, Butteraugli and CVVDP (`--colour` gives the colours; a note
after the results says when they were guessed), and the shape of its pixels
(H.263 and H.261 record their own). Black bars are found from bytes partway
through an H.264, HEVC, VVC, MPEG-1, MPEG-2, MJPEG or Dirac stream. Any other
stream, or one that cannot be decoded from there (its only keyframe its
first, or an H.264 or VVC stream with its parameter sets only at its start),
is read from the start up to the last part sampled, which takes a while for
a long stream (`--black-bars none` skips it). The app's window does not take
raw streams.

A raw YUV file (`.yuv`) is pictures and nothing else: `--size`, `--pix-fmt`
and `--fps` say what they are, for every `.yuv` file of the command. It is
read as FFmpeg's `rawvideo`, and scores exactly as the same pictures in a
container. It has no colour tags: it takes the other video's when that one
has them, and `--colour` gives them otherwise.

## Saved results and caches

Save portable results as `.metrics.json` or export CSV. For the scores alone,
right-click the Videos table or the statistics table on Metric Graphs and
choose **Export results as CSV** or **as JSON**: either gives the same file,
every video's score and statistics for each metric (those of every metric,
not only the one Metric Graphs shows), in the shape the command line's
`--json` writes.

Cached results are reused when files and relevant settings match. A
compatible subset can load as partially calculated; request calculation to
fill missing metrics.

Settings and the default cache are under `~/.videometricslab/`. Back up that
folder before maintenance; changing checkout should not require clearing it.
File moves or replacements can cause cache misses. Use explicit saved-result
loading when you need to inspect a portable result. Right-click recalculation
forgets matching cached results, so do not use it just to refresh the display.

Result files may contain absolute media paths. Review them before sharing.
See [troubleshooting](TROUBLESHOOTING.md) for missing results or playback errors.
