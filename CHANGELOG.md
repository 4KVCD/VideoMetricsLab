# Changelog

## v2.2

v2.2 adds a macOS version for Apple silicon and makes SSIMULACRA2, Butteraugli
and CVVDP faster with Vship-fast.

### Metric performance

- SSIMULACRA2, Butteraugli and CVVDP now use
  [Vship-fast](https://github.com/4KVCD/Vship-fast): Butteraugli
  **1.75x-2.5x**, CVVDP **1.6x-2.4x** and SSIMULACRA2 up to **1.6x** faster
  than Vship's Vulkan build.

### New features

- Added a macOS version for Apple silicon (macOS 14.4 or later), with GPU
  metrics and Video Compare playback on the Mac's GPU.
- Added an app icon.
- Video Compare now shows its keys (S, ←/→ and Space) in large type on its
  black screen.
- Command line: added `--fps` to compare raw video streams (.264, .265, .266,
  .obu and others), and `--size` and `--pix-fmt` (with `--fps`) for raw .yuv
  files.
- Command line: a video without colour tags now takes the other video's, so a
  .y4m encode of an HDR10 source is read as HDR10.

### Other changes

- Removed the GPU backend setting: the GPU metrics now always use Vulkan
  (VMAF v0.6.1 and VMAF NEG use CUDA on an NVIDIA GPU where Vulkan can't run
  them).
- Removed the GPU/CPU choice for SSIMULACRA2 and Butteraugli: they now run on
  the GPU when one can be used, and on the CPU otherwise.
- SSIMULACRA2, Butteraugli and CVVDP on the GPU now compute as libjxl and
  pycvvdp do, so scores saved on the GPU by earlier versions are calculated
  again.
- Command line: removed `--ssimulacra2-on` and `--butteraugli-on`; `--cpu`
  still calculates both on the CPU.
- Command line: a metric that falls back from the GPU to the CPU now gets a
  note in the results and the JSON.
- Command line: CVVDP's JSON entry now gives its frame count, as the other
  metrics' entries do.
- Command line: on a terminal, the progress line now updates once a second,
  as the window's does.

### Bug fixes

- Fixed wrong VMAF, PSNR, SSIM and XPSNR scores when the test video's colour
  range or colour matrix differs from the reference's.
- Fixed SSIMULACRA2, Butteraugli and CVVDP reading the two videos with
  different colour matrices when one or both had no colour tags, such as a
  540p encode of an untagged 720p source.
- Fixed SSIMULACRA2, Butteraugli and CVVDP on the GPU reading the two videos'
  colours half a pixel apart when only one gave its chroma siting, such as a
  .y4m reference made from an .mkv.
- Fixed SSIMULACRA2, Butteraugli and CVVDP on the GPU reading full-range
  greyscale videos with too little contrast.
- Fixed changing a VMAF model removing the video's other scores too.
- Fixed cancelling the custom VMAF model file dialog switching the video to
  the Auto model and removing its scores.
- Fixed the duration limit, frame subsample and threads boxes applying a value
  digit by digit as it was typed.
- Fixed videos still showing scores after Clear saved results being skipped by
  the next run.
- Fixed Metric Graphs' exported PNG drawing its title and table too small on
  high-DPI screens.
- Command line: fixed CVVDP failing, and VMAF, SSIMULACRA2 and Butteraugli
  falling back to the CPU, when the videos were given as relative paths.
- Command line: fixed very large `--threads` or `--subsample` values hanging
  the run.
- Command line: fixed a missing `--model` file being found only when VMAF
  failed during the run.
- Command line: fixed CSV write failures not being shown with `-q`.

## v2.1

v2.1 is VideoMetricsLab rewritten in C++: a smaller download, faster runs,
and a window that opens faster and uses far less memory. [The C++
port](docs/CPP_PORT.md) has the details and measurements.

### Metric performance

- VMAF v0.6.1 and VMAF NEG now use Vulkan on NVIDIA GPUs too when the GPU
  backend is Auto, **3.2x-4.4x** faster than with CUDA.
- SSIMULACRA2, Butteraugli and CVVDP now share one pass over the videos when
  the GPU has the memory for it, making runs with them **1.1x** faster.

### System performance

- The download is **39 MB** instead of 87 MB.
- Runs are **1.25x** faster with 28% less CPU (three 4K test videos, every
  metric).
- The app opens **3x** faster and idles in 26 MB of memory instead of 87 MB.
- Moving the pointer over a graph uses **6x** less CPU.
- Video Compare now frees its memory 30 s after you leave it: 0.1 GB instead
  of 7.6 GB with four 4K videos.

### New features

- Added film grain synthesis for VVC encodes in Video Compare.
- Added a colour picker for each video's line in Metric Graphs and Bitrate
  Viewer.
- Command line: added `--frames N` to compare only the first N frames, for a
  short encode of a longer reference.
- Command line: added `--colour` for videos without colour tags, such as
  HDR10 in .y4m files, and a warning when colours are guessed.
- Command line: added `--cvvdp-display` to choose the display CVVDP models;
  repeat it to score on several displays.

### Other changes

- Every metric now pairs the two videos' frames by frame number, on the CPU
  and the GPU alike.
- Video Compare now plays videos with FFmpeg instead of GStreamer.
- GPU VMAF now uses
  [libvmaf-fast v1.1](https://github.com/4KVCD/libvmaf-fast/releases/tag/libvmaf-fast-v1.1).

### Bug fixes

- Fixed a crash when closing the window with a path typed in the reference
  box.
- Fixed a possible crash of VMAF v1 on the GPU when libvmaf failed to start.
- Fixed SSIMULACRA2 and Butteraugli on the CPU sometimes failing to read a
  frame.
- Fixed long runs using too much memory: up to 2.1 GB with VMAF on the CPU and
  770 MB with VMAF on the GPU over 3 hours at 60 fps.
- Fixed runs that crashed or were ended in Task Manager leaving temporary
  files behind, up to 1.4 GB for a 4K run.
- Fixed the command line sometimes losing a video's metrics from its table
  and JSON.
- Fixed a saved VMAF score being shown with the wrong model.
- Fixed "frames compared" being one frame short with some duration limits.
- Fixed Video Compare jumping to the first test video when scores arrived.
- Fixed Video Compare showing the Videos tab for seconds at the first switch
  after black bars were found.
- Fixed Video Compare playing no frames of a test video decoded in the
  background that had fallen behind.
- Fixed Video Compare keeping old black bars and still frames after the
  reference or the scaling changed.
- Fixed Video Compare cropping a second test video with the first one's black
  bars.
- Fixed Video Compare keeping each finished test video's decoder until the
  window closed.
- Fixed loading a saved run keeping the old reference's resolution tests, and
  a loaded resolution test showing the reference's details.
- Fixed the CVVDP display editor changing values that were not edited.
- Fixed a video's colour changing between Metric Graphs and Bitrate Viewer, or
  after it was removed and added again.
- Fixed the Settings tab's rows overlapping on high-DPI screens or in a small
  window; tabs now scroll instead (#6).
- Fixed a failed graph export or settings save not being reported.
- Fixed FFmpeg showing as not usable when its folder held a folder named
  ffmpeg.exe.
- Fixed texts that stayed in English in Video Compare, Metric Graphs, Bitrate
  Viewer, the run line and the CVVDP display editor.

## v2.0

v2.0 is a major performance update. VMAF, VMAF NEG and VMAF v1 now run on
any GPU, the other metrics and Video Compare are several times faster, and the
app uses far less memory.

### Metric performance

- Added Vulkan support for VMAF v1, so it now runs on any GPU (NVIDIA, AMD or
  Intel), up to **17x** faster than on the CPU.
- Added Vulkan support for VMAF v0.6.1 and VMAF NEG, so they now run on any GPU
  (NVIDIA, AMD or Intel), **4.8x-6.4x** faster than Netflix's own CUDA code and
  up to **28x** faster than on a 24-core CPU.
- PSNR, SSIM and XPSNR are now calculated in the app with libvmaf-fast,
  **2.7x-11x** faster than with FFmpeg.
- Added software decoding in the app for videos the GPU can't decode, such as
  VVC, making VMAF **1.8x-2.7x** faster for them.
- SSIMULACRA2 and Butteraugli on the CPU are now **7x** faster.

### System performance

- Video Compare now plays 4K HDR at 120 frames a second instead of 14, using a
  fifteenth of the CPU.
- Greatly reduced memory use: the app now idles at 180 MB instead of 840 MB,
  and VMAF v1 at 4K uses 2.6 GB instead of 13.8 GB.
- Metric Graphs, Video Compare and Settings now open instantly, and the window
  stays responsive with hundreds of videos.
- Video Compare no longer uses any CPU while idle.
- Detected black bars are now remembered between sessions.

### New features

- Added a command line, `VideoMetricsLab-cli`, for running comparisons from
  scripts.
- Video Compare can now play HDR video in Display P3 colours on the GPU.
- Added the ability to zoom in Video Compare.
- Added an option to highlight each metric's best and worst results in the
  Videos tab.
- Added a dark theme, which follows Windows or can be chosen in Settings.
- Added drag and drop for videos, pasting a reference path, reordering rows by
  dragging, and sorting by any column.
- File dialogs now open in the folder your videos last came from, and the app
  can reopen your last session's videos.

### Other changes

- GPU VMAF now uses
  [libvmaf-fast v1](https://github.com/4KVCD/libvmaf-fast/releases/tag/libvmaf-fast-v1).

### Bug fixes

- Fixed XPSNR being weighted by the test video's detail and motion instead of
  the reference's, as XPSNR is defined: scores are 0.9-1.8 dB higher, and
  XPSNR saved by v1.4 is recalculated.
- Fixed wrong scores, or a run that never ended, for 10-bit H.264 decoded on
  AMD GPUs.
- CVVDP is no longer calculated on Intel GPUs above 1080p, where it hangs the
  GPU and blacks out the screen.
- Fixed partly green frames in Video Compare on some NVIDIA GPUs.
- Fixed the reference showing a few frames behind the test video after a seek,
  for some MP4 references.
- Fixed FFmpeg running on after the app crashed or was ended in Task Manager.
- Fixed SSIMULACRA2 and Butteraugli starting again on the CPU, for hours, for a
  test video cut short.
- Fixed GPU memory growing each time Video Compare moved between an HDR and an
  SDR display.
- Fixed Video Compare staying blank for a test video added again.
- Fixed 10-bit H.264 on Intel and AMD GPUs failing first with FFmpeg's GPU
  decoding.
- A raw .hevc or .264 stream is now refused with how to fix it, not as
  variable-frame-rate video, in Video Compare too.
- Fixed Video Compare sometimes playing without sound, or with a test video's
  sound instead of the reference's.
- Fixed choppy sound and slow playback in Video Compare when decoding couldn't
  keep up.
- Fixed the window freezing on a seek while a video's file was slow to read.
- Fixed a pause in Video Compare playback when black-bar detection finished.
- Fixed Video Compare saying it was playing or buffering while paused.

### How GPU VMAF works

v1.4 calculated VMAF and VMAF NEG on NVIDIA GPUs only, with its own build of
libvmaf's CUDA code. v2.0 moves that work into
[libvmaf-fast](https://github.com/4KVCD/libvmaf-fast), an open-source fork of
Netflix's libvmaf, which adds a Vulkan engine for every GPU:

- **VMAF and VMAF NEG on AMD and Intel GPUs.** libvmaf's CUDA code for VIF, ADM
  and motion is ported to Vulkan compute shaders. GPUs round floating point
  differently and some have no double precision, so the shaders use exact
  integer arithmetic and tables made on the CPU instead: every feature equals
  CUDA's bit for bit. NEG reuses everything VMAF calculates except its gain
  limit.
- **VMAF v1 on any GPU.** libvmaf has no GPU code for VMAF v1's features (ADM3,
  motion3, CAMBI and SpEED), so the engine follows its CPU code, and every
  score equals the CPU's exactly.
- **PSNR, SSIM and XPSNR** are calculated by libvmaf-fast inside the app
  instead of by FFmpeg's filters, with the same scores (XPSNR is FFmpeg's
  filter, ported).
- **NVIDIA GPUs** keep libvmaf's CUDA code for VMAF and NEG, now with 13
  upstream pull requests merged, and the fork's own fixes: 4K VMAF + NEG in
  35% less GPU memory.

Where possible, frames from the GPU's own decoder are handed over without
leaving the GPU. Each GPU is checked against known results before it is used,
and one that gets them wrong (a driver bug) is left to the CPU. As in v1.4, the
GPU code runs in a process of its own, so a crash in it or in the driver
doesn't close the app.

Checked on an RTX 5090, the Intel iGPU of a Core Ultra 9 285K and a Radeon
8060S: VMAF and NEG on Vulkan equal CUDA's in 45 cases, and VMAF v1 on the GPU
equals libvmaf's CPU code in 73 cases, at 8 and 10 bits, up to 8K.

## v1.4

- VMAF v0.6.1 and VMAF NEG can now be calculated on NVIDIA GPUs.
- GPU metrics now take frames straight from the GPU's video decoder (NVIDIA,
  Intel and AMD) instead of through FFmpeg, using much less CPU and running
  noticeably faster.
- Updated Vship's Vulkan build to 5.1.2, with SSIMULACRA2 on Intel GPUs.
- A crash in Vship, libvmaf or the GPU driver while calculating metrics no
  longer closes the app.
- Reworked the progress line, with each metric's progress and time remaining.
- VMAF NEG now sits beside VMAF v0.6.1.
- Fixed wrong scores for videos with an odd width or height when decoded on
  NVIDIA GPUs.
- Fixed SSIMULACRA2, Butteraugli and CVVDP comparing mismatched frames when a
  video has dropped or extra frames.
- Fixed CPU SSIMULACRA2 and Butteraugli scores being far from the GPU's on SDR
  video. Saved CPU scores are recalculated once.
- Fixed 24 fps videos being accepted against 23.976 fps ones (and 30 against
  29.97, 60 against 59.94).
- Fixed GPU metrics on PCs whose only GPU is Intel's.
- Fixed FFmpeg's GPU decoding not being used on AMD GPUs.

### Minor bug fixes

- A video whose pictures end long before its stated length now fails instead
  of being scored.
- Fixed XPSNR missing from the table and CSV with frame subsampling and VMAF on
  the GPU.
- Fixed MKV videos with a soundtrack longer than the picture failing with
  "Durations do not match".
- Fixed MP4 files whose cover art comes before the video comparing or playing
  the cover.
- Fixed a crash on exit on PCs with a Vulkan loader but no Vulkan driver.
- Fixed adding videos whose name or title contains curly quotes, some accented
  letters, or Chinese, Japanese or Korean text.
- Fixed the Video bitrate column including the soundtrack for MKV files (marked
  ≈ where it can't be separated).
- Fixed loading a saved run of a listed video adding a second row.
- Fixed the window opening wider than its default size in Spanish and
  Portuguese with some system fonts.
- Fixed Video Compare's status line for references without a soundtrack.
- Fixed an occasional crash when running from source, started from another
  Python program.

### How VMAF on the GPU works

libvmaf, the library that calculates VMAF, has CUDA versions of the features
VMAF is made from (VIF, ADM and motion), but they can't simply be used. No
FFmpeg build the app can rely on includes them, and as merged they score
differently from the CPU: libvmaf's issues and pull requests report randomly
low motion scores, NaN VIF scores, and ADM differing from the CPU's at the
frame edges.

So VideoMetricsLab bundles its own libvmaf with CUDA, built from libvmaf
master ([cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c))
with 12 open pull requests merged in, each pinned to the commit we tested:

- [#1477](https://github.com/Netflix/vmaf/pull/1477): a native Windows (MSVC)
  build.
- [#1573](https://github.com/Netflix/vmaf/pull/1573): CUDA build fixes, and a
  crash with pinned picture memory.
- [#1583](https://github.com/Netflix/vmaf/pull/1583): a race in the motion
  feature that gave randomly low motion scores, and a double flush at the end
  of a video.
- [#1644](https://github.com/Netflix/vmaf/pull/1644) and
  [#1612](https://github.com/Netflix/vmaf/pull/1612): motion at the frame
  edges, which now mirrors them as the CPU does, and on the first frame.
- [#1614](https://github.com/Netflix/vmaf/pull/1614): a race in VIF that gave
  NaN scores.
- [#1647](https://github.com/Netflix/vmaf/pull/1647) to
  [#1651](https://github.com/Netflix/vmaf/pull/1651): five places where ADM's
  CUDA code differed from the CPU's (contrast masking, rounding, border
  clamping, an angle constant and a denominator shift).
- [#1652](https://github.com/Netflix/vmaf/pull/1652): both frames released
  when one fails to score.

With them, VIF and ADM on the GPU are identical to the CPU's. Motion differs
by about 0.00003, because its CUDA kernel rounds its blur in a different
order ([libvmaf issue 1562](https://github.com/Netflix/vmaf/issues/1562),
not fixed yet). Every frame's VMAF is within 0.00006 of the CPU's, and VMAF
NEG within 0.0008, so a saved score is reused whichever calculated it.

`scripts/build_libvmaf_cuda.ps1` builds it from exactly these commits, and two
builds give the same file. libvmaf scores on the GPU in a process of its own,
so a crash in it or in the NVIDIA driver calculates that video's VMAF on the
CPU instead of closing the app.

Where NVIDIA's decoder can decode both videos, that process decodes them too:
the pictures are cropped, scaled and paired on the GPU and handed to libvmaf
there, without ever being copied to system memory. Otherwise FFmpeg decodes
and pairs the frames and pipes them to it. Either way the frames are paired
by timestamp exactly as FFmpeg's own libvmaf filter pairs them.

VMAF v1 (which has no CUDA version), custom models, resolution tests, 12-bit
videos and comparisons at an odd width or height stay on the CPU.

Thanks to the authors of these pull requests. We hope they are merged, so that
everyone gets accurate VMAF on the GPU.

### Known issues

- On the integrated Intel GPU of a Core Ultra 9 285K, CVVDP fails on 4K videos
  ("A GPU Call failed inside Vship"). SSIMULACRA2 and Butteraugli are not
  affected.
- On HDR (PQ) video, SSIMULACRA2 on the CPU and on the GPU still differ
  widely (35 against 47 on one film): libjxl's tool handles HDR in its own
  way. Butteraugli agrees (2.89 against 2.88). Use the GPU score for HDR.

## v1.3

- Added the window in 20 languages: Simplified and Traditional Chinese,
  Spanish, Brazilian Portuguese, German, French, Japanese, Russian, Korean,
  Italian, Polish, Turkish, Arabic, Indonesian, Vietnamese, Ukrainian, Thai,
  Czech, Hungarian and Dutch. The app opens in Windows' display language, and
  Settings > Window > Language chooses another. The log, saved results and
  exported files stay in English.
- Added Intel GPU support for the GPU metrics with Vship's Vulkan build, which
  runs on any GPU with a Vulkan driver. Settings > GPU metrics > GPU backend
  chooses the build: Auto (the default) uses CUDA on NVIDIA, HIP on AMD and
  Vulkan on other GPUs. On Vulkan, SSIMULACRA2 is calculated on the CPU,
  because Vship 5.1.1's Vulkan build scores it too high (up to 17 points at
  4K).
- Updated the GPU metrics to Vship 5.1's API and color handling. Videos tagged
  BT.2020 SDR, SMPTE 170M (NTSC/DVD), Display P3, ICtCp or YCgCo, and
  monochrome and alpha videos, are now scored on the GPU instead of falling
  back to the CPU, and CVVDP no longer fails on them. RGB videos without a
  transfer tag are now read as sRGB, as Vship's own FFVship does, and their
  saved GPU scores are recalculated.
- Added a setting to calculate SSIMULACRA2, Butteraugli and CVVDP together in
  one pass per video (Settings > GPU metrics, off by default). It decodes each
  video once instead of once per metric, which helps most with 4K VVC (decoded
  on the CPU), but needs more GPU memory.
- Each metric's score now appears and is saved as soon as it is done, instead
  of when the whole video is finished. Stopping a run or closing the app loses
  at most the metric in progress.
- Fixed test frames stamped slightly earlier than the source's being compared
  with the previous source frame, which gave clusters of near-zero VMAF NEG
  and XPSNR scores. Each test frame is now compared with the source frame
  nearest in time. Saved scores of affected videos are kept until they are
  recalculated.
- Hovering a red Failed cell now says why that metric failed.
- Added a session log under Settings > Storage > Log files, with Export log...
  (the log files as one .zip) and Copy log (the latest run) buttons.
- The app now checks GitHub for a newer release when it starts, and says so
  only if there is one. Turn it off in Settings > Window.
- Windows no longer goes to sleep while metrics are being calculated. The
  screen can still turn off.
- Included various small bug fixes and reliability improvements.

### Known issues

- With Vship's Vulkan build (Intel GPUs, or GPU backend set to Vulkan),
  Butteraugli and CVVDP are wrong for videos tagged with SMPTE 170M or 240M
  primaries (NTSC/DVD), and Butteraugli is about 2.5% off for videos tagged
  BT.470BG and for untagged SD videos. CUDA and HIP are not affected.
- 4:1:0 (yuv410p) videos cannot be scored on the GPU: SSIMULACRA2 and
  Butteraugli fall back to the CPU, and CVVDP is not calculated.

## v1.2.1

- Added a VMAF v1 column with its own model list. The existing VMAF column is
  now labelled VMAF v0.6.1, and scores saved by earlier versions still load.
  VMAF v1's Auto model is chosen by the resolution the videos are compared
  at: 4K and above uses the 4K / 1.5H model, because 1.5 screen heights is the
  standard viewing distance for 4K (VMAF v0.6.1's 4K model uses it too).
  Anything below 4K uses the 1080p / 3H model, the standard distance for HD.
- Reworked the run status: each video shows its CPU and GPU metrics progress
  separately, with elapsed time and a more accurate queue ETA, plus many
  related bug fixes.
- Cancelling a run will now keep the metrics a video had already finished.
- Cancelling a run will now stay on the Videos tab rather than jump to the
  Metric Graphs tab.
- Video Compare no longer re-checks the reference each time a different test
  video is picked.
- Fixed saved VMAF scores not loading on rows set to VMAF model Auto.
- Fixed the metrics ticked in the Videos tab column headers being unticked
  after a restart.
- Included various small bug fixes and reliability improvements.

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
