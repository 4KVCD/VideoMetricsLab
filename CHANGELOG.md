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
- The GPU metrics now decode each video with the GPU's own decoder, inside the
  process that scores it, on NVIDIA, Intel and AMD GPUs (H.264, HEVC and AV1,
  8- and 10-bit 4:2:0). FFmpeg used to decode on the GPU, copy every picture
  back to memory and pipe it across, which took most of a GPU metric's CPU
  time. The scores are the same; CPU use is about 15 times lower on NVIDIA and
  about 3 times lower on Intel. On NVIDIA the pictures for VMAF never leave
  the GPU. Anything else (other codecs, 4:2:2, 4:4:4, 12-bit, interlaced or
  damaged streams) is decoded through FFmpeg as before.
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
- SSIMULACRA2, Butteraugli and CVVDP now compare each test frame with the
  source frame nearest in time, as VMAF, PSNR, SSIM and XPSNR have since v1.3.
  They were paired by position, so a frame dropped from the test video put
  every later pair one frame apart.
- SSIMULACRA2 and Butteraugli on the CPU now read a video's colors as the GPU
  does, and Butteraugli uses the same norm and display brightness. A tagged
  BT.709 film scored 40.4 SSIMULACRA2 on the CPU against 55.1 on the GPU; the
  two now agree closely on SDR video. Saved CPU scores are calculated again
  once.
- A comparison resized with any scaling algorithm is now treated as the same
  comparison, so its saved scores are found again after the algorithm is
  changed.
- Fixed wrong SSIMULACRA2, Butteraugli and CVVDP scores for AV1 and VP9 videos
  with an odd width or height when GPU decoding was on (NVIDIA): FFmpeg's
  hardware decode returned a padded picture, with wrong chroma for odd
  heights. Such videos are now decoded on the CPU.
- A video at 24 fps is no longer accepted against one at 23.976 fps (or 30
  against 29.97, 60 against 59.94). Their frames drift a frame apart every 42
  seconds, which gave scores for frames that were not each other's.
- Fixed FFmpeg's GPU decoding never being used on AMD GPUs: every run started
  with a failed attempt and decoded on the CPU.
- The Video bitrate column now shows the video stream's own bitrate. For MKV
  files it showed the whole file's, soundtrack included; where only that is
  known, it is marked with "≈".
- Fixed MP4 files with cover art: the cover could be compared, scanned or
  played instead of the video.
- Fixed a video whose soundtrack runs longer than its picture failing with
  "Durations do not match".
- Fixed a crash at startup or when opening Video Compare that affected about
  one start in five when the app was launched from another Python program.
- Fixed XPSNR being calculated for every frame, and then dropped from the
  table, when libvmaf frame subsampling was on and VMAF ran on the GPU.
- Loading a saved run of a video already in the list now replaces its row
  instead of adding a second one.
- A settings file with a wrong value no longer stops a setting from working
  later; that setting keeps its default. Settings are saved in one step, so a
  crash cannot leave half a file.
- The window fits its default width in every language with wider system
  fonts.
- VMAF NEG now sits beside VMAF v0.6.1, before VMAF v1, in the Videos table,
  Metric Graphs and exported CSV files.
- Video Compare's status line no longer fills with playback diagnostics when
  the reference has no soundtrack.
- Included various small bug fixes and reliability improvements.

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
