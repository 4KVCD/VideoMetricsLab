# Building the distributable

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
./scripts/build_release.ps1
```

About a minute. It builds the GPU shader, freezes the app with PyInstaller,
verifies the packaged GStreamer against the development installation, runs
the packaged executable's self-test, and zips the result.

| | |
|---|---|
| Folder | ~203 MB |
| Zip | ~72 MB |
| Output | `%LOCALAPPDATA%\VideoMetricsLab-build\` |

Pass `-OutputRoot <path>` to build somewhere else, and `-VerifyMedia
<file>, <file>` to also decode real videos through the packaged GStreamer as
part of the build (see [GSTREAMER_BUNDLE.md](GSTREAMER_BUNDLE.md)).

## Why the output is not in `dist/`

This repository normally lives in a OneDrive folder. Building into it uploads
~450 MB on every build, and OneDrive's own file handles make the previous
build's output undeletable — PyInstaller fails with `Access is denied` on
`_internal\...` before it can start. The build therefore goes to
`%LOCALAPPDATA%` by default, and `dist/` and `build/` are in `.gitignore` in
case anyone overrides that.

## What is bundled

- **The build environment's Python, PySide6/Qt and NumPy** — the app runs with no Python
  installed.
- **GStreamer 1.28.6**, pruned to the 40 plugins the app can reach and
  their dependencies — 53 MB of the wheels' 302 — plus the `gi` bindings.
  This is what makes hardware video comparison work on a machine that has
  never seen GStreamer. [GSTREAMER_BUNDLE.md](GSTREAMER_BUNDLE.md) says what
  is kept, why, and how the build proves it still plays everything.
- **Qt**, cut the same way to the three modules the app imports (QtCore,
  QtGui, QtWidgets), the Windows platform plugin, the Windows style, the
  common image formats and their dependencies -- 42 MB of PySide6's 114.
  PyInstaller's Qt hooks would otherwise add a software OpenGL renderer,
  QtMultimedia's FFmpeg, the QML stack, PDF, networking and 96 translations.
  `scripts/qt_bundle.py` decides; the self-test reports the platform plugin,
  style and image formats that actually loaded.
- **`d3d11_tonemap.dll`**, the GPU HDR→SDR shader, built from
  `native/d3d11_tonemap.cpp` as part of the build.
- **`nvdec_frames.dll`, `vpl_frames.dll` and `amf_frames.dll`**, the
  NVIDIA, Intel and AMD decoders the GPU metrics read their frames from
  (`vmaf_app/core/gpu_frames.py`), built from `native/nvdec_frames.cpp`,
  `native/vpl_frames.cpp` and `native/amf_frames.cpp` by
  `scripts/build_gpu_frames.ps1` (MinGW-w64 g++, as for the tone mapper) as
  part of the build, about 0.3 MB each. They link nothing of the GPU makers':
  each loads its maker's decoder library from the graphics driver at run time
  -- NVIDIA's `nvcuda.dll` and `nvcuvid.dll`, Intel's oneVPL `libvpl.dll`,
  AMD's AMF `amfrt64.dll` -- and NVIDIA's conversion kernels are PTX in the
  source, so no CUDA toolkit or SDK is needed to build them. Intel's and
  AMD's scale pictures with a Direct3D 11 compute shader
  (`native/d3d11_scale.h`), compiled at run time by Windows' own
  `d3dcompiler_47.dll`. They contain
  nv-codec-headers', oneVPL's and AMF's API definitions (all MIT), whose
  notices are packaged beside them. Without one, FFmpeg decodes those
  videos as before. `scripts/check_gpu_decoder.py` checks one of them on the
  PC it runs on: pictures against FFmpeg's decode, speed, and GPU scores.
  The build makes these four from the commit's own source every time -- they
  are not in git, so ones left from another commit would be stale -- and so
  needs MinGW-w64 g++ on PATH; it stops without it, and checks all four and
  their notices are in the package.
- **SSIMULACRA2 and Butteraugli**, the official libjxl 0.12.0 static Windows
  command-line tools. Only these two executables and their notices are copied
  into the bundle; users do not need to install libjxl or a runtime separately.
  In the current build they add about 12 MB installed and 7 MB compressed.
- **Vship GPU libraries** for SSIMULACRA2, Butteraugli and CVVDP, in three
  builds: `vmaf_app/tools/vship/nvidia` (CUDA, the 5.1.2 release), `amd` (HIP,
  the 5.1.1 release: 5.1.2 has no HIP build) and `vulkan` (any GPU with a
  Vulkan driver, NVIDIA, AMD and Intel alike). The Vulkan build is made from
  the v5.1.2 tag by `scripts/build_vship_vulkan.ps1` (MinGW-w64 g++, as for the
  tone mapper; the same source gives the same file, in any folder). Against
  the release's own Vulkan library, which is not bundled, it scores Butteraugli
  and CVVDP to the bit and SSIMULACRA2 within 0.00003 (another compiler's
  rounding). Replace the HIP build with the first HIP release of 5.1.2 or
  later; until then FFmpeg upsamples 4:1:0 video to 4:4:4 for it, as 5.1.1
  cannot read it (`perceptual_vship._READS_410_SINCE`). Settings > GPU
  metrics > GPU backend picks one; Auto, the default, uses CUDA on NVIDIA, HIP
  on AMD and Vulkan on other GPUs, and a build that cannot run hands over to the
  next. A metric a build is measured to score wrongly on a GPU maker's GPUs is
  calculated on the CPU there (`perceptual_vship.SCORED_WRONGLY`, empty now);
  re-measure every build when updating Vship. Where no build can use the GPU,
  the input format is unsupported, or GPU processing fails, SSIMULACRA2 and
  Butteraugli fall back to the bundled CPU tools. The Vship CLI and FFMS2
  decoder are not included; the user's FFmpeg reads every video. A video the
  GPU's own decoder decodes -- NVIDIA's with Vship's CUDA build, Intel's or
  AMD's with any build -- is decoded by the frame decoders above in Vship's
  process, FFmpeg only copying its compressed stream out of the container;
  other videos FFmpeg decodes. Vship and metric notices are packaged beside
  the libraries.
- **libvmaf with CUDA** (`vmaf_app/tools/libvmaf/libvmaf.dll`), for VMAF and
  VMAF NEG on NVIDIA GPUs only: from
  [libvmaf-fast](https://github.com/4KVCD/libvmaf-fast), this project's fork
  of libvmaf with the open pull requests that let it build with MSVC and fix
  its CUDA code merged. `scripts/fetch_libvmaf_fast.ps1` installs its pinned
  release, checked by SHA-256, into `vmaf_app/tools`; the DLLs are committed,
  so building the app does not run it (to build them instead, see the fork's
  `fast/README.md`). The C runtime is linked in, so it needs only Windows and
  the NVIDIA driver; no CUDA runtime ships. Where NVIDIA's decoder decodes both videos, they are
  decoded, scaled and widened on the GPU by `nvdec_frames.dll` in libvmaf's
  process, the user's FFmpeg only copying their compressed streams out of
  the containers; otherwise FFmpeg decodes them. VMAF on any other GPU or the CPU, VMAF v1,
  PSNR, SSIM and XPSNR still come from the user's FFmpeg. It adds about 3.3 MB installed,
  1 MB compressed. libvmaf and Vship each run in a process of their own
  (`vmaf_app/core/isolated.py`), so a crash in either, or in the GPU driver,
  falls back to the CPU instead of closing the app.
- **VMAF's features with Vulkan** (`vmaf_app/tools/vmaf_vulkan/vmaf_vulkan.dll`),
  for VMAF and VMAF NEG on GPUs other than NVIDIA's, or on any GPU when the
  GPU backend setting is Vulkan: libvmaf-fast's Vulkan engine (`fast/vulkan`
  there), a port of libvmaf's CUDA feature extractors (VIF, ADM and motion)
  to Vulkan compute shaders, from the same release. It needs only Windows and a Vulkan driver, and
  gives the same feature values as the CUDA code bit for bit; the score is
  predicted from them by the libvmaf above, on the CPU. It adds about 0.6 MB.
  It runs in the same process of its own as libvmaf does.
- **VMAF v1 with the GPU** (`vmaf_app/core/vmaf_v1_gpu.py`) needs no library of
  its own: the same `vmaf_vulkan.dll` has a VMAF v1 mode, which follows
  libvmaf's CPU code for ADM3 and motion3 (`integer_adm.c`,
  `integer_motion.c`) and gives its values bit for bit on any GPU, NVIDIA's
  included; CAMBI and SpEED are calculated by the CPU extractors of the
  libvmaf above, on its thread pool, and the same libvmaf predicts the score
  from the four. Check it with `python scripts/compare_vmaf_v1.py --matrix`.
- **Pictures without a CPU copy** (NVIDIA): `nvdec_frames.dll` and
  `vmaf_vulkan.dll` share GPU memory. The Vulkan library exports the buffers it
  takes each frame pair's luma planes from (`vv_export`, Vulkan's
  `VK_KHR_external_memory_win32`), the decoder's CUDA imports them
  (`nvf_import`) and copies its pictures into them on the GPU; and the planes
  libvmaf's CPU extractors read are downloaded straight into libvmaf's
  pictures, page-locked (`nvf_pin`), which the GPU then writes by itself. Both
  fall back to system memory where a driver refuses.

## What is not, and why

**FFmpeg and libvmaf.** The app finds them at runtime and prompts for their
location if they are missing. They are left out deliberately (the bundled
libvmaf above scores only VMAF and VMAF NEG on NVIDIA GPUs, and the NVIDIA
decoder above decodes from the compressed streams the user's FFmpeg copies out
of the containers):

- A libvmaf-enabled FFmpeg is another ~80 MB on an already large download.
- FFmpeg licensing depends on its build configuration and linked libraries.
  Redistribution requires reviewing that exact build's obligations; the
  presence of libvmaf alone is not a sufficient licensing classification.
- Users comparing encoders generally already have a specific FFmpeg build
  they trust, and silently shipping a different one invites results that
  disagree with their own command line for reasons nobody can see.

To bundle it anyway, drop `ffmpeg.exe` and `ffprobe.exe` beside the
executable and add them to the spec's `datas`; the tool finder checks the
application directory.

## Verifying a build

The packaged executable can check itself:

```powershell
.\VideoMetricsLab.exe --self-test
```

It reports FFmpeg, GStreamer and its plugins, and the GPU shader, in a dialog
and at `~\.videometricslab\self-test.txt`. `--quiet` skips the dialog and
sets the exit code instead, which is how `build_release.ps1` uses it.

The build has a second program, `VideoMetricsLab-cli.exe`: the same code with
a console, for the command line (`vmaf_app/cli.py`; see the user guide). A
windowed program cannot print to the terminal that started it, so it is a
program of its own; it shares everything else in the folder.
`build_release.ps1` runs its `devices` command, which must find FFmpeg and
load the GPU libraries.

This exists because a broken bundle is not obvious from the outside. GStreamer
failing to load makes playback fall back to FFmpeg silently — the app opens,
the tabs work, and nothing looks wrong until someone plays a video and
wonders why it is slow. The first build off this spec did exactly that, with
GStreamer failing on `No module named 'optparse'`.

## Notes for changing the spec

- **The GStreamer wheels are shipped as data, not frozen as modules.** Each
  computes its own plugin, typelib and scanner paths from `__file__`
  (`gstreamer_libs.environment`). Frozen into the archive, every one of those
  paths would point at a file that does not exist. They are therefore in
  `datas` and in `excludes` — file by file, chosen by
  `scripts/gstreamer_bundle.py`, so change what is shipped there rather than
  in the spec.
- **`scripts/pyi_rth_gstreamer.py` replaces the `.pth` file.** Outside a
  bundle, `gstreamer_bundle.pth` runs `setup_python_environment()` at
  interpreter startup. Frozen apps do not process `.pth` files, so the
  runtime hook does it instead — before the entry script, because GStreamer
  reads those variables when it initialises.
- **Anything `gi` imports must be a `hiddenimport`.** Excluding `gi` from
  analysis means nothing traced its imports. The stdlib modules it reaches
  for are listed in the spec; regenerate with:

  ```powershell
  .venv\Scripts\python.exe -c "import pathlib,re,sys,sysconfig; root=pathlib.Path(sysconfig.get_paths()['purelib'])/'gstreamer_python'/'Lib'/'site-packages'; mods={m.group(1).split('.')[0] for f in root.rglob('*.py') for m in re.finditer(r'^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)', f.read_text(errors='replace'), re.M)}; print(sorted(m for m in mods if m in sys.stdlib_module_names))"
  ```

- **Qt modules are excluded by name.** WebEngine, Quick, QML, Charts,
  Multimedia and the rest are never imported; leaving them in roughly doubled
  the bundle.

## Trimming it further

- GStreamer and Qt are already cut to what the app uses. The next largest
  item is NumPy's OpenBLAS at ~20 MB, which NumPy links unconditionally, then
  `python314.dll` and OpenSSL (pulled in by `asyncio`, which `gi` imports).
- `--onefile` produces a single executable instead of a folder. It is nicer
  to hand someone, but it unpacks ~150 MB to a temp directory on every
  launch, which is slow enough to be noticeable.
- UPX compression is off. It reduces the download but is a common
  false-positive trigger for antivirus, which matters more for an unsigned
  binary.

## Releasing

Complete [RELEASING.md](RELEASING.md), including the third-party license review.
The bundle ships no GPL GStreamer plugins (the GPL wheels are left out
entirely and the bundled FFmpeg is LGPL 2.1), but leaving FFmpeg external
does not remove the need to review the remaining third-party components.

The executable is unsigned, so SmartScreen will warn on first run. Signing
needs a certificate; without one, "More info → Run anyway" is the path, and
saying so in the release notes saves a lot of questions.
