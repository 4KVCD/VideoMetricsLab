# The C++ port

v2.1 is VideoMetricsLab rewritten from Python into C++ with Qt 6: every
tab, the command line, every metric and Video Compare. It does what v2.0
did, scores as the Python code it was ported from does, and reads the scores
v2.0 saved. What changes is how much it takes from your PC.

| | v2.0 | v2.1 | |
| --- | ---: | ---: | --- |
| Download | 86.7 MB | 39.1 MB | **2.2x** smaller |
| Installed | 254 MB, 382 files | 140 MB, 118 files | **1.8x** smaller |
| Opening the window | 0.69 s | 0.23 s | **3x** faster |
| Memory when idle | 87 MB | 26 MB | **3.3x** less |
| A run of three 4K test videos, every metric | 86.4 s | 69.3 s | **1.25x** faster |
| CPU time of that run | 591 s | 425 s | 28% less |

## How it was measured

On one PC: Core Ultra 9 285K, GeForce RTX 5090 (driver 616.92), a 120 Hz
display at 175% scaling, the videos on a hard disk. Both apps were run the
same way on the same files:

- **Runs:** the command line with its default settings, every metric, on
  the first 30 s of The Beekeeper in 4K (x264) against its x265, AV1 and
  VVC encodes, also in 4K.
- **Opening and idle memory:** the packaged apps, the median of five
  launches each, taken in turn. Memory is the private memory 5 s after the
  window showed.
- **The window:** the same scripted steps in both apps, with the window on
  screen; v2.0 was run from its source code for these.

## Runs

- The run above is **1.25x** faster and uses 28% less CPU. It starts 102
  programs instead of 132.
- SSIMULACRA2, Butteraugli and CVVDP share one pass over each video when the
  GPU has the memory for it, so each video is decoded once for the three of
  them instead of once each. On its own that makes the run **1.1x** faster.
- With the GPU backend on Auto, VMAF v0.6.1 and VMAF NEG use Vulkan on
  NVIDIA GPUs too, **3.2x-4.4x** faster than with CUDA (libvmaf-fast's own
  benchmark).
- Started from the window, the same run takes 70.1 s instead of 84.4 s, and
  the window's own process uses 4.2 s of CPU instead of 9.6 s.

## The window

| | v2.0 | v2.1 | |
| --- | ---: | ---: | --- |
| The window's thread while the pointer moves over a graph, 1000 Hz mouse | 98% of a CPU core | 16% | **6x** less |
| Pointer moves handled in those 3 s, of about 3,000 | 1,478 | 2,943 | **2x** more |
| One pointer move over a graph | 2.4 ms | 1.2 ms | **2x** faster |
| One zoom step on a graph | 11.7 ms | 3.9 ms | **3x** faster |
| The window's thread during a 30 s run | 3.7 s | 1.8 s | **2x** less |

## Video Compare

Video Compare now plays videos with FFmpeg's libraries instead of
GStreamer, which is no longer bundled.

| A 4K source and three 4K test videos | v2.0 | v2.1 |
| --- | ---: | ---: |
| Memory 40 s after you leave Video Compare | 7.6 GB | 0.1 GB |
| GPU memory 40 s after you leave it | 1.2 GB | 0 GB |
| Memory while it plays | 7.6 GB | 5.6 GB |

v2.0 kept Video Compare's decoders until the app closed. v2.1 lets them go
30 s after you leave the tab, and loads playback again when the pointer
reaches the tab or a tab beside it opens.

## The same scores

- 80 test cases on real videos, covering every metric and the ways the app
  calculates them, give identical results in v2.1 and the Python code it was
  ported from: the same tables, per-frame CSV files, JSON results, saved
  scores and exit codes.
- v2.1 keeps its saved scores in the same files as v2.0 and reads v2.0's.

## macOS port

The command line and the window run on Apple silicon Macs (arm64), macOS
14.4 or later (Qt 6.12's earliest), as a self-contained app bundle,
`VideoMetricsLab.app` (`scripts/build_release_macos.sh`). Every metric but
CVVDP runs, on the CPU.

### What macOS needed

- **The build.** The engine and the window compiled with Apple Clang with
  four fixes for warnings MSVC does not give (an unused constant, a test's
  missing field, a copied loop variable). The `macos` preset builds for
  arm64 and macOS 14.4.
- **The app bundle.** The window is built as `VideoMetricsLab.app`
  (`cpp/packaging/macos/Info.plist.in`). Its programs are in
  `Contents/MacOS` and the app's files in `Contents/Resources/resources`.
  `packaged_resource_folder()` finds them there, and the self-test uses the
  same function.
- **FFmpeg.** The user installs it, `brew install ffmpeg`, which has
  libvmaf. An app opened from the Finder does not have Homebrew on its
  PATH, so the app also looks in Homebrew's and MacPorts' folders.
- **The bundled tools** are built from their pinned releases' source,
  because neither project publishes macOS builds: libvmaf-fast v1.1's
  `libvmaf.dylib` and libjxl 0.12.0's `ssimulacra2` and `butteraugli_main`,
  linked statically (`scripts/build_libvmaf_fast_macos.sh`,
  `scripts/build_libjxl_macos.sh`). They link only the system's libraries,
  and building them twice gives the same bytes. They are not committed.
- **Processes and signals.** A cancelled or ended program's process group
  is now signalled until the program is reaped, not only while it runs, so
  what it started ends with it. The command line ignores SIGPIPE, as
  Windows has no such signal: `VideoMetricsLab-cli ... | head` no longer
  ends with 141 after the run. Ctrl+C's pipe is closed on exec.
- **The package.** macdeployqt, then Qt's binaries thinned to arm64 (they
  are universal), the plugins the app does not use removed as on Windows,
  Qt's translations in `Contents/Resources/translations` (`qt.conf`), every
  Mach-O file signed ad hoc and then the bundle. The script checks that
  everything links only what is in the bundle or the system's, and runs the
  self-test and `devices` from the bundle in an empty profile.
  `.github/workflows/macos-package.yml` builds it on GitHub.

### What failed, and why

- The bundled SSIMULACRA2 and Butteraugli failed ("Permission denied"): the
  lookup tried `ssimulacra2.exe` first and found Windows' copy in the
  repository's resources. Elsewhere than Windows it now looks only for the
  names without `.exe`.
- The self-test spoke of `ffmpeg.exe` and warned about the GPU metrics'
  frame decoders, which are built for Windows only.
- macOS's own widget style draws its controls larger than Windows' (32 px
  buttons to 26): the Videos tab's table showed one row, the bottom row of
  buttons was cut off and the metric headings' tick boxes were not drawn.
  The window uses Qt's Fusion style on macOS, which keeps Windows'
  proportions.
- Settings offered "Same as Windows" for the language and the theme, and
  the graphs asked for Consolas, which macOS lacks (100 ms to find a
  substitute at the first graph): macOS gets "Same as macOS" in every
  language, and Menlo.
- The window's tests set the profile folder with USERPROFILE, which macOS
  does not read: run on a Mac, they wrote the user's real settings. They set
  HOME there. A test that empties the PATH to hide FFmpeg also names no
  other folders to look in (VML_FFMPEG_FOLDERS), Homebrew's included.
- A Bitrate tab test read the table before the window had taken a message
  already posted to it, and failed one run in three under load.

### The same scores as Windows

On the same frames (made by FFmpeg's filters alone, lossless FFV1, with the
same frame hashes on both systems: a 1280x720 clip of 96 frames), on the
CPU, frame by frame:

| Metric | Mac against Windows |
| --- | --- |
| PSNR, XPSNR | identical on every frame, in FFmpeg and in the app |
| SSIM | 10 frames differ, by 1e-6, in FFmpeg and in the app alike |
| VMAF v0.6.1, VMAF NEG | 2 frames differ, by 8e-6 (one float step) |
| VMAF v1 | 75 frames differ, by up to 8e-5 |
| Butteraugli | 1 frame differs, by one float step (1e-6) |
| SSIMULACRA2 | every frame differs, by up to 0.015 |

- Clang fuses a multiply and an add into one instruction on Apple's chips,
  rounding once where the Windows builds round twice; the app's own code is
  built without that for this reason (`cpp/CMakeLists.txt`). Homebrew's
  FFmpeg and libvmaf are built with it, and so is the app's libvmaf-fast,
  so that its PSNR, SSIM and XPSNR are the user's FFmpeg's to the last
  digit, as on Windows they are Gyan's FFmpeg's, and a score saved from
  either route is the same. Built without it, libvmaf-fast's SSIM was
  Windows' on every frame, and then differed from Homebrew's FFmpeg's.
- libjxl's tools are built without it: Butteraugli then differs on 1 frame
  rather than 10. SSIMULACRA2's vector code sums in another order on
  Apple's chips (Highway's NEON against AVX2).
- VMAF runs in FFmpeg's libvmaf filter, which is the libvmaf the user's
  FFmpeg links (Homebrew's), not the app's.

### Video Compare

Video Compare plays natively, as on Windows: the GPU decodes both videos,
the presenter draws each pair in time with the source's soundtrack.

| On Windows | On macOS |
| --- | --- |
| Direct3D 11 device | Metal device, with FFmpeg's VideoToolbox device beside it (`native_device_mac.mm`) |
| D3D11VA decoding into textures | VideoToolbox decoding into pixel buffers (CVPixelBuffer, NV12 and P010's layouts), which Metal reads where they are |
| Pictures decoded in software uploaded to textures | written into pooled IOSurface pixel buffers |
| The source scaled down by the video processor | scaled down by a Metal pass (each pixel the average of the source it covers) |
| `d3d11_tonemap.dll`'s presenter in a child window | a CAMetalLayer in a view of its own over the view, two drawables, one frame a refresh (`native_presentation_mac.mm`) |
| HLSL shader | its port to MSL, the same arithmetic in the same order (`native_metal.h`) |
| WASAPI | CoreAudio (an AudioQueue, its callbacks on the player's thread) |
| `dav1d.dll` | `libdav1d.dylib` (`scripts/build_dav1d_macos.sh`) |
| FFmpeg's DLLs, committed | its dylibs, built by `scripts/build_ffmpeg_decoders_macos.sh` and linked weakly by @rpath, not committed |

- **The shaders** are compiled from their source when Video Compare is
  first loaded (`newLibraryWithSource:`): the Command Line Tools have no
  offline Metal compiler. That runs off the window's thread
  (`NativePool::load`, as Windows compiles its HLSL), so it adds nothing to
  the app's start or to the window's thread: on an M1 Pro it took 718 ms,
  on a thread of its own, the first time.
- **Checked on an M1 Pro** (macOS 27.0.1), in the window: a 1080p H.264
  source with its soundtrack against a 720p H.264 and a 10-bit HEVC encode,
  both decoded by VideoToolbox, drawn by Metal in time with the sound;
  played, paused, switched between the encodes, stepped. The decoder's
  tests (`NativeDecoderTest`): each codec's frame 30 is FFmpeg's, sample for
  sample, by VideoToolbox (H.264, 10-bit H.264, HEVC, AV1 and VP9 where the
  Mac decodes them) or in software (MPEG-2 within its leeway, ProRes 4:2:2,
  AV1 by dav1d); a seek back; a raw stream; the source scaled down.
  `FramePanelTest.APairPlaysNativelyOnTheDisplay` plays a pair on the
  display (`VML_TEST_ON_SCREEN=1`).
- **Found:** the dav1d mirror took EAGAIN for 11, as on Windows and Linux;
  it is 35 on macOS, so dav1d's "more data" read as an error and AV1 in
  software gave no frame. It is the system's EAGAIN now.
- **Apple's GPUs decode 10-bit H.264**, which no Windows GPU does.

### The GPU metrics

CVVDP, SSIMULACRA2 and Butteraugli run on the Mac's GPU, through Vship's
Vulkan build on MoltenVK (Khronos' Vulkan on Metal), as they run on Windows
through Vulkan (`scripts/build_vship_macos.sh`):

- **Vulkan:** Khronos' loader with MoltenVK as its driver, bundled in
  `tools/vulkan`. MoltenVK alone is not enough: Vship asks for portability
  devices (`VK_KHR_portability_enumeration`), which the loader provides. The
  app points the loader at MoltenVK (`VK_DRIVER_FILES`) before Vulkan is
  first loaded in a process, the GPU processes included, and turns off
  MoltenVK's fast math.
- **Vship:** v5.1.2, as on Windows, with two of its shaders patched
  (`scripts/patches/vship-moltenvk-addresses.patch`): SPIRV-Cross, which
  turns SPIR-V into Metal, mistranslates a byte pointer plus an offset read
  as another type, so no pipeline could be made. The patch does that
  arithmetic on the address as an integer (the same memory). Reported to
  Vship's author with the patch:
  [Line-fr/Vship#27](https://codeberg.org/Line-fr/Vship/issues/27).
- **Scores**, against Windows' Vulkan build on an RTX 5090, the same 96
  frames: SSIMULACRA2 within 0.0006 a frame, Butteraugli 1e-5, CVVDP 7e-7.
  (Vship's SSIMULACRA2 of a video against itself is not 100 on either:
  99.67 on the Mac, 99.36 on Windows, where the score magnifies rounding.)
- **Found:** the GPU pass runs in a process of its own, which had not
  pointed the loader at MoltenVK: Vship's pinned memory could not be made
  there, and the run fell back to the CPU. The environment is set before
  any Vship library loads now.
- **Found:** two Butteraugli handlers at once (the lanes Windows runs per
  metric) lost the GPU above 720p ("Invalid Resource"), and the run fell back
  to the CPU. On macOS one handler per metric scores, as fast there (1080p:
  SSIMULACRA2 9.7 s for 480 pairs with one, 9.8 with two).

### VMAF on the GPU

VMAF v0.6.1, VMAF NEG and VMAF v1 run on the Mac's GPU through
libvmaf-fast's Vulkan engine, ported to macOS on its `macos-port` branch
(`scripts/build_vmaf_vulkan_macos.sh` builds `vmaf_vulkan.dylib` from the
commit it pins), on the same loader and MoltenVK as Vship:

- **The port:** the loader opened by its rpath, the portability devices
  asked for, and one shader change: Metal has no 64-bit subgroup sums, so
  ADM's and CAMBI's are three 32-bit sums of pieces, which give the same
  bits. Windows' SPIR-V is byte-identical.
- **Compiled as libvmaf is on the Mac**, with Clang's own floating-point
  settings (fused multiply-adds), as `build_libvmaf_fast_macos.sh` builds
  libvmaf: the engine turns the GPU's sums into features with libvmaf's
  expressions, and without the fusing ADM3 differed from libvmaf's in the
  last bit on some 10-bit frames.
- **The self-tests pass:** VMAF v0.6.1's exact sums at 8 and 10 bits, and
  VMAF v1's features and scores bit for bit against the bundled libvmaf on
  the CPU (`VmafV1Gpu.EveryGpuGivesLibvmafsCpuFeaturesAndScoresBitForBit`).
- **VMAF v1's SpEED chroma is left to libvmaf** on arm64
  (`start_speed_on_gpu`): the engine's is a port of libvmaf's x86 code,
  and libvmaf's own on arm64 is its NEON code (another filter, fused
  multiply-adds), so the self-test failed on it, and with it VMAF v1 fell
  back to the CPU.
- **Scores**, the same 96 frames: VMAF v0.6.1 and NEG on the GPU are within
  one float step (4e-6) of FFmpeg's libvmaf on the CPU, as libvmaf's CUDA
  features are of its CPU ones (the engine ports the CUDA ones). VMAF v1
  within 5e-5: its SpEED is the bundled libvmaf-fast's, the CPU's run is
  Homebrew's libvmaf 3.2.1 in FFmpeg, and the two SpEEDs differ by that.

### Ending and pausing programs

- **Found:** a Cancel or Pause did not reach the FFmpeg that a process of its
  own (the GPU and CPU passes) had started: a process group is not a job, and
  each program had a group of its own. Such a process's programs now join its
  group (`start_programs_in_this_group`), which its parent ends and stops, and
  the parent removes the FIFOs a killed child left in the temporary folder.
- **Found:** the command line ignores SIGPIPE, and programs inherit that; they
  get the default back now. It also cancels on SIGTERM and SIGHUP (a closed
  Terminal window), as Windows' console handler does for a closed console.

### Still to do

- VVC's film grain: Windows draws it with Direct3D 11 compute
  (`native/d3d11_film_grain.cpp`); on macOS FFmpeg draws 8-bit 4:2:0's, and
  10-bit VVC shows without it.
- HDR on an HDR display (EDR): the presenter's layer takes PQ in BT.2020 for
  it, but the app does not yet ask macOS whether the display shows HDR
  (`query_display_hdr`), so HDR video is mapped to SDR.

### For the Linux port

- Do the same: look for FFmpeg in the distribution's folders as well as on
  the PATH; build libvmaf-fast and libjxl's tools from their pinned source,
  linked statically; never pick a `.exe` file; ignore SIGPIPE in the command
  line; decide, library by library, whether to fuse multiply-adds (GCC
  does by default, on ARM and on x86 with FMA): as the distribution's
  FFmpeg is built, for libvmaf-fast, and not, for libjxl's tools; set the
  tests' profile folder with HOME.
- Do differently: the resources are beside the programs, as on Windows,
  and `libvulkan.so.1` is the Vulkan loader, so Vship's and VMAF's Vulkan
  code may run there.
