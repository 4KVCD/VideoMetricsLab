# Changelog

## v1.4

- Added VMAF v0.6.1 and VMAF NEG on NVIDIA GPUs (GeForce GTX 16 and RTX 20
  series or newer), with a bundled libvmaf built with CUDA. The GPU's scores
  agree with the CPU's to within a thousandth of a point on every frame, so
  saved scores are reused either way. Each video's Performance > VMAF v0.6.1
  and NEG compute chooses NVIDIA GPU (the default) or CPU. VMAF on the GPU runs
  apart from VMAF v1, PSNR, SSIM and XPSNR, which stay on the CPU, so they do
  not slow it down. If the GPU fails, that video's VMAF is calculated on the
  CPU.
- Updated Vship's Vulkan build (Intel GPUs, or Settings > GPU metrics > GPU
  backend set to Vulkan) to 5.1.2. SSIMULACRA2 is now calculated on the GPU
  with it: Vship 5.1.1's Vulkan build scored it up to 17 points too high on
  NVIDIA GPUs, so v1.3 calculated it on the CPU. It also fixes v1.3's known
  issues with videos tagged with SMPTE 170M or 240M primaries or BT.470BG.
- Fixed Intel GPUs getting no GPU metrics: Vship 5.1.1's Vulkan build could not
  be loaded on a PC whose only GPU is Intel's, and the app then crashed as it
  closed. Fixed the same crash on PCs with a Vulkan loader but no Vulkan
  driver, such as virtual machines.
- 4:1:0 (yuv410p) videos are now scored on the GPU.
- A crash in Vship, libvmaf or the GPU driver no longer closes the app. The GPU
  metrics run in processes of their own; if one crashes, its metrics are
  handled as for any other GPU failure, and the rest of the run goes on.
- SSIMULACRA2 and Butteraugli set to CPU now run beside the GPU metrics,
  instead of after them while holding up the GPU.
- Reworked each video's progress line. It numbers the metrics under way among
  those calculated on the CPU and on the GPU, each with its percentage, fps and
  time remaining, for example "GPU metrics 1–2 of 5: VMAF v0.6.1, VMAF NEG
  50.0% (55.0 fps, 0:00:08 remaining)"; metrics calculated in one pass share
  them. Hover over the line for every metric's state. A pause no longer slows
  the rates shown for the rest of the run, a failed metric says so at once, and
  a metric waiting for its own video's VMAF on the GPU no longer says that
  another video is using the GPU.
- Removed the queue's estimated time remaining from the status line. With the
  CPU and GPU metrics running at very different speeds in two queues, it could
  not be made accurate. Each video's line still shows its own time remaining.
- A video whose pictures end long before its length says, such as an encode
  that stopped early or a copy that did not finish, now fails instead of being
  scored on the frames it had.
- Calculate metrics now names every checked video that cannot be compared with
  the reference (a different frame rate or length, an unreadable file) at
  once, and offers to calculate the others, instead of stopping at the first.
- Fixed adding a video whose title tag or file name contains a curly quote
  (”), some accented letters, or Chinese, Japanese or Korean text, which failed
  with "the JSON object must be str, bytes or bytearray, not NoneType".
- Cancelling a run now leaves the videos it never reached as they were, instead
  of marking them Cancelled.
- Included various small bug fixes and reliability improvements.

### Known issues

- VMAF on the GPU still copies every frame from the GPU's decoder through the
  CPU and back to the GPU, which takes several CPU cores (about 6.5 at 4K and
  55 fps on a Core Ultra 9 285K). CPU metrics calculated at the same time slow
  it down, to about 42 fps beside PSNR and SSIM.
- Vship's CUDA (NVIDIA) and HIP (AMD) builds are still 5.1.1 until 5.1.2 is
  released. For them, 4:1:0 videos are converted to 4:4:4 before scoring,
  which gives slightly different scores than the Vulkan build (SSIMULACRA2
  72.82 against 72.98 on one test video).
- On the integrated Intel GPU of a Core Ultra 9 285K, CVVDP fails on 4K videos
  ("A GPU Call failed inside Vship"). SSIMULACRA2 and Butteraugli are not
  affected.

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
