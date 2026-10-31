# Building the distributable

```powershell
./scripts/build_release.ps1 -QtPath C:\Qt\6.12.0\msvc2022_64
```

It needs what building the app needs (`cpp/README.md`): Visual Studio 2022
or its Build Tools with the C++ tools, CMake 3.25 or newer, and Qt 6.12.0's
MSVC 2022 build with Qt's translations (qttranslations); `-QtPath` names Qt's
folder where the build folder was not configured with it. It also needs
MinGW-w64's `g++` on the PATH, for the native libraries, and `ffmpeg.exe`
and `ffprobe.exe` on the PATH, for the self-test.

The app itself is built with MSVC. The script configures with the `windows`
preset (`cmake --preset windows`, from `cpp`), whose Visual Studio generator
finds MSVC itself, from any shell. To configure a build folder with another
generator, such as Ninja, do it from a vcvars64 prompt (x64 Native Tools
Command Prompt for VS 2022): from another shell CMake takes the first
compiler on the PATH, such as that MinGW-w64 `g++`, and stops with a message
saying so. A `vml-cli` built with MinGW's `g++` loads its C++ runtime from
the PATH, where another GCC's copy can come first, and was seen to fail
silently when it scored videos.

It builds the native libraries from the commit's own source, then the window
and the command line (Release); puts Qt and Microsoft's C++ runtime beside
them and the app's own files in `resources`; checks that everything the
programs import is in the folder or is Windows'; runs the window's self-test
and the command line's `devices` from the folder, in a profile of their own;
and zips it.

| | |
|---|---|
| Folder | ~140 MB |
| Zip | ~39 MB |
| Output | `%LOCALAPPDATA%\VideoMetricsLab-build\cpp\` |

Pass `-OutputRoot <path>` to build somewhere else, and `-AllowLocalLibvmaf`
to build with a local build of libvmaf-fast, to test; a release takes a
release of it ([RELEASING.md](RELEASING.md)).

## Why the output is not in the repository

The repository may live in a OneDrive folder. Building into it would upload
the package on every build, and OneDrive's own file handles can make the
previous build's output undeletable. The package therefore goes to
`%LOCALAPPDATA%` by default. (CMake's own build folder, `cpp/build`, is in
`cpp/.gitignore`.)

## What is bundled

- **The window and the command line**, `VideoMetricsLab.exe` and
  `VideoMetricsLab-cli.exe`: the same engine, the command line with a
  console (a windowed program cannot print to the terminal that started
  it). Each is also the process of its own that GPU work runs in.
- **Qt 6.12.0**: qtbase's Core, Gui, Widgets and Network, the Windows
  platform and style plugins, the GIF, ICO and JPEG image plugins and the
  schannel TLS plugin (the update check), put beside the window by
  windeployqt, as separate DLLs that can be replaced with others of the same
  version. Left out: the Direct3D shader compilers for Qt's own Direct3D
  drawing, which the window's widgets do not use, the software OpenGL
  renderer, the TUIO touch plugin, the network information plugin and the
  TLS backend that checks no certificate. Qt's translations of its own
  dialogs and buttons (`translations\qtbase_<code>.qm`) come for each of the
  window's languages that Qt has translated.
- **Microsoft's C++ runtime**, whichever of its DLLs anything in the folder
  imports, so the app runs on a PC without it installed.
- **`d3d11_tonemap.dll`**, Video Compare's presenter: it draws the frames
  with Direct3D 11 -- NV12 or P010 into RGB, HDR mapped to SDR or kept for
  an HDR display -- and presents them. Built from
  `native/d3d11_tonemap.cpp` by `scripts/build_d3d11_tonemap.ps1`.
- **`software_frames.dll`**, the decoder for the videos FFmpeg would decode
  in software -- no GPU decoder takes them (VVC, AV1 on an older GPU, 10-bit
  H.264), the GPUs' decoders give other pictures than FFmpeg's (MPEG-2), or
  GPU decoding is off -- so that their frames reach the GPU
  metrics and the app's PSNR, SSIM and XPSNR without FFmpeg's pipes, which
  carry 4K frames at 60-80 a second. Built from `native/software_frames.cpp`
  by `scripts/build_gpu_frames.ps1` with the GPU decoders below, against
  FFmpeg's headers in `native/ffmpeg`. It decodes H.264, HEVC, VVC, VP9,
  MPEG-2 and FFV1 with FFmpeg 9.0.2's decoders, AV1 with dav1d and AV2 with
  dav2d --
  FFmpeg's decode's pictures, sample for sample, each video's film grain
  added or left out as it is scored -- and a VVC video scored with its
  H.274 grain with VVdeC (FFmpeg's `libvvdec`), the one decoder here that
  adds that grain as it should. A video of another codec or format, one that
  must be scaled, or a damaged one goes through FFmpeg's pipes;
  `VML_SOFTWARE_DECODE=ffmpeg` sends them all that way. A VVC video with its
  H.274 grain never goes there (with `VML_SOFTWARE_DECODE=ffmpeg` its metrics
  are not calculated): this decoder scales it itself, a damaged one is not
  scored, and FFmpeg's own routes (VMAF on the CPU, its xpsnr filter, the
  resolution test, the PNG pictures of SSIMULACRA2 and Butteraugli on the
  CPU) read its pictures from this decoder, run in a process of its own and
  fed to FFmpeg through a pipe as rawvideo (`cpp/core/src/grain_feed.cpp`).
- **FFmpeg's libraries** (`resources/tools/ffmpeg`: `avcodec-63.dll`,
  `avutil-61.dll`, `swresample-7.dll`, which libavcodec imports, and
  `avformat-63.dll` and `swscale-10.dll`, with which Video Compare's
  playback also reads videos and soundtracks and converts pictures and
  sound), for the software frame decoder and Video Compare: FFmpeg 9.0.2's
  libraries with only the decoders, parsers and demuxers the script lists
  and Direct3D 11's video decoding, configured LGPL with one external
  library: Fraunhofer HHI's VVC decoder VVdeC (its master, pinned by commit,
  with 4KVCD's pull requests, `scripts/patches/vvdec`), linked in as the
  decoder `libvvdec` (`scripts/patches/ffmpeg`), which the app asks for by
  name (FFmpeg's own VVC decoder stays the default). Built from FFmpeg's
  release source by `scripts/build_ffmpeg_decoders.ps1` (MinGW-w64 GCC,
  NASM and CMake; the source is pinned by its SHA-256), which also vendors
  the headers the decoders are compiled against. Beside them,
  **dav1d 1.5.4** (`dav1d.dll`, BSD), which decodes AV1 in software, as
  FFmpeg's libdav1d decoder does: built from VideoLAN's release source by
  `scripts/build_dav1d.ps1`, and **dav2d** (`dav2d.dll`, BSD), VideoLAN's
  AV2 decoder, which decodes AV2 (FFmpeg has no AV2 decoder, and no GPU
  decodes it): dav2d has no release yet, so `scripts/build_dav2d.ps1` builds
  it from a pinned commit of VideoLAN's repository. They are committed, as
  libvmaf-fast's DLLs are:
  building the app does not build them. FFmpeg itself (`ffmpeg.exe`) is
  still the user's.
- **`nvdec_frames.dll`, `vpl_frames.dll`, `amf_frames.dll` and
  `mf_frames.dll`**, the NVIDIA, Intel and AMD decoders the GPU metrics read
  their frames from (`cpp/core/src/gpu_frames.cpp`; for VMAF v1 on AMD's
  GPUs Windows' own Media Foundation decoders first, AMF's where they
  cannot), built from `native/nvdec_frames.cpp`, `native/vpl_frames.cpp`,
  `native/amf_frames.cpp` and `native/mf_frames.cpp` by
  `scripts/build_gpu_frames.ps1` (MinGW-w64 g++). They link nothing of the
  GPU makers': each loads its maker's decoder library from the graphics
  driver at run time -- NVIDIA's `nvcuda.dll` and `nvcuvid.dll`, Intel's
  oneVPL `libvpl.dll`, AMD's AMF `amfrt64.dll` -- and NVIDIA's conversion
  kernels are PTX in the source, so no CUDA toolkit or SDK is needed to
  build them. Intel's and AMD's scale pictures with a Direct3D 11 compute
  shader (`native/d3d11_scale.h`), compiled at run time by Windows' own
  `d3dcompiler_47.dll`. They contain nv-codec-headers', oneVPL's and AMF's
  API definitions (all MIT), whose notices are packaged beside them.
  Without one, FFmpeg decodes those videos. The build makes the native
  libraries from the commit's own source every time -- they are not in
  git, so ones left from another commit would be stale -- and so needs
  MinGW-w64 g++ on the PATH; it stops without it.
- **SSIMULACRA2 and Butteraugli on the CPU**, the official libjxl 0.12.0
  static Windows command-line tools, with their notices; users do not need
  to install libjxl.
- **Vship-fast's GPU library** for SSIMULACRA2, Butteraugli and CVVDP,
  `resources/tools/vship/vulkan` (any GPU with a Vulkan driver, NVIDIA, AMD
  and Intel alike): [Vship-fast](https://github.com/4KVCD/Vship-fast), this
  project's faster fork of Vship, whose metrics compute as libjxl 0.12.0's
  and pycvvdp's do. `scripts/build_vship_vulkan.ps1` builds its Vulkan
  library from a pinned commit (MinGW-w64 g++; the same source gives the
  same file, in any folder). Vship's CUDA and HIP builds are not bundled:
  on an RTX 5090, Vship-fast's Vulkan build scores the three metrics on 4K
  10-bit video at 45 pairs a second, Vship 5.1.2's CUDA build at 36.
  Re-measure it when updating Vship-fast (docs/RELEASING.md, step 6).
  Where it cannot use the GPU, the input format is unsupported, or GPU
  processing fails, SSIMULACRA2 and Butteraugli fall back to the bundled
  CPU tools. A video the GPU's own decoder decodes is
  decoded by the frame decoders above in Vship's process, FFmpeg only
  copying its compressed stream out of the container; FFmpeg decodes the
  others.
- **libvmaf with CUDA** (`resources/tools/libvmaf/libvmaf.dll`): VMAF and
  VMAF NEG on an NVIDIA GPU whose Vulkan cannot calculate them, the libvmaf that turns every GPU scorer's
  features into its score, and the app's PSNR, SSIM and XPSNR on the CPU:
  from [libvmaf-fast](https://github.com/4KVCD/libvmaf-fast), this project's
  fork of libvmaf. The DLLs are committed, so building the app does not
  build them. While the app is developed they are a build of the fork's
  latest commit: `scripts/build_libvmaf_fast_local.ps1` builds the `fast`
  branch of its local clone (`Documents\libvmaf-fast` by default; it needs
  the fork's build tools, see its `fast/README.md`) into `resources/tools`.
  A release of the app takes a release of libvmaf-fast, published first:
  `scripts/fetch_libvmaf_fast.ps1` installs its pinned release, checked by
  SHA-256, and `scripts/build_release.ps1` refuses a local build
  ([RELEASING.md](RELEASING.md)). Either writes
  `resources/tools/libvmaf/libvmaf-fast.json`, whose version every score
  made with these libraries records. The C runtime is linked in, so it needs
  only Windows and the NVIDIA driver; no CUDA runtime ships. VMAF on the CPU
  still comes from the user's FFmpeg. libvmaf and Vship run in a process of
  their own (`cpp/core/include/vml/isolated.h`), so a crash in either, or in
  the GPU's driver, falls back to the CPU instead of ending the run.
- **VMAF's features with Vulkan** (`resources/tools/vmaf_vulkan/vmaf_vulkan.dll`),
  for VMAF and VMAF NEG on any GPU: libvmaf-fast's Vulkan engine, a port of
  libvmaf's CUDA feature extractors (VIF, ADM and motion) to Vulkan compute
  shaders, from the same build. It needs only Windows and a Vulkan driver,
  and gives the same feature values as the CUDA code bit for bit; the score
  is predicted from them by the libvmaf above, on the CPU. Its VMAF v1 mode
  follows libvmaf's CPU code for ADM3 and motion3 and gives its values bit
  for bit on any GPU; CAMBI and SpEED are calculated on the GPU too where
  the engine supports the model's options for them, else by the libvmaf
  above on the CPU.

## What is not, and why

**FFmpeg (`ffmpeg.exe`, `ffprobe.exe`).** The app finds them at run time --
in the folder Settings names, on the PATH, or where common installers put
them -- and asks for their location if they are missing. They are left out
deliberately (the bundled libraries above only decode and read files, with
no encoder, filter or program, and the frame decoders decode from the
compressed streams the user's FFmpeg copies out of the containers):

- A libvmaf-enabled FFmpeg is another ~80 MB on an already large download.
- FFmpeg licensing depends on its build configuration and linked libraries.
  Redistribution requires reviewing that exact build's obligations.
- Users comparing encoders generally already have a specific FFmpeg build
  they trust, and silently shipping a different one invites results that
  disagree with their own command line for reasons nobody can see.

## Verifying a build

The packaged window can check itself:

```powershell
.\VideoMetricsLab.exe --self-test
```

It reports FFmpeg, FFmpeg's libraries for Video Compare, the GPU decoders,
VVdeC, dav1d, the presenter, Qt and its TLS backend, each
frame decoder for the GPU metrics, the software decoder's codecs, libjxl's
tools, the process of its own GPU work runs in, Vship, and VMAF and VMAF v1
on the GPU, in a dialog and in `~\.videometricslab\self-test.txt`. `--quiet`
skips the dialog and sets the exit code instead, which is how
`build_release.ps1` uses it. The script also runs
`VideoMetricsLab-cli.exe devices`, which must find FFmpeg and say where each
metric is calculated.

A broken package is not obvious from the outside: a library that fails to
load makes the app fall back to something slower. A run notes the fallback
where it can tell a failure from a PC without that GPU (a GPU check that
crashed, errored or scored wrongly, or Vship-fast replaced by stock Vship
5.1.2), but not for every library. The self-test is what shows it.

## Releasing

Complete [RELEASING.md](RELEASING.md), including the third-party licence
review ([THIRD_PARTY.md](THIRD_PARTY.md)).

The programs are unsigned, so SmartScreen warns on first run. Signing needs
a certificate; without one, "More info â†’ Run anyway" is the path, and saying
so in the release notes saves a lot of questions.
