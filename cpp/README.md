# VideoMetricsLab in C++

VideoMetricsLab is a C++ app from v2.1 on. It was ported from the Python
app (v2.0 and before; its last code is the `release/v2.1` branch) to
run faster and in less memory, and to come to macOS and Linux after
Windows. A comment that names a `vmaf_app/...` file or a Python test
(`tests/test_...py`) means the Python app's code the C++ code was ported
from, on that branch.

## How it is checked

When it was ported, the C++ code was held to the Python app's answers:

- **Scores from the libraries** (libvmaf, Vship, libjxl's tools) identical,
  Python against C++ on the same PC: the same libraries, fed the same
  frames, give the same numbers.
- **Numbers the app calculates itself** (the mean, median, standard
  deviation, percentiles, XPSNR's average) to 12 significant digits. On
  Windows the tests ask for them exactly, and they are: sums are made in
  numpy's order (`numpy_sum` in `core/src/stats.cpp`).

The tests' expected values in `tests/reference` are those answers, made by
the Python app's own code at the port, numbers as `float.hex()` text
(`0x1.7a00000000000p+6`, `inf`, `nan`) so that they reach the tests
exactly. The Python app is no longer run: the files are fixed data, and a
change that means to give another answer edits them, in the same commit.

On real videos, `tests/golden/golden.py` holds the command line to an
earlier run of it: it runs the cases in `cases.json` (80 runs covering the
ways the app scores videos) with a `vml-cli`, each with a fresh profile
(and the case's own settings, where it has any; some cases save scores in
it first), and compares the exit codes, the tables, the per-frame CSV files
character for character, the JSON results, and what each run saved in the
saved scores. The clips are 2-second cuts of real test videos
(`clips.json`; `golden.py clips` makes them, outside the repository). Two
runs of the same build on this PC give identical results in every case, the
GPU metrics included, so a new build is held to identical results with the
last release, but where a change made on purpose says why. It runs before
a release and in a round of bug checking, not for each change. With
`--seed-app` another `vml-cli` saves the scores first: the last release's,
say, so that the new build reads the scores that release saved. A video
the app decodes itself and scales is what FFmpeg's scale filter gives,
sample for sample (`native/scale/scale_filter.h`); the `...-by-ffmpeg`
cases cover FFmpeg's route too.

    python cpp\tests\golden\golden.py run --out before --app <the last release>\vml-cli.exe
    python cpp\tests\golden\golden.py run --out after --app cpp\build\windows\cli\Release\vml-cli.exe
    python cpp\tests\golden\golden.py compare before after

## Layout

| Folder | What it holds |
| --- | --- |
| `core` | `vml_core`, the engine. No Qt and no operating-system headers, so the same code runs on every system, under the command line and the window. |
| `platform` | `vml_platform`: what differs between the systems (starting programs, pipes, loading libraries, the console, the PATH, the GPUs), behind headers that are the same on all of them. |
| `cli` | `vml-cli`, the command line, and its code (`vml_cli`), which the tests use too. |
| `app` | `VideoMetricsLab`, the window (Qt 6), and its code (`vml_app`), which its tests use too. Its icon, in `app/icons`, is drawn by `scripts/make_icons.py`, which writes the Windows `.ico`, the macOS `.icns` and the SVGs. |
| `tests/unit` | The tests (GoogleTest). |
| `tests/app` | The window's tests, run with no screen (Qt's offscreen platform), each in a profile of its own. |
| `tests/reference` | The tests' expected values: the Python app's answers at the port, edited only on purpose since. |
| `tests/golden` | The command line's runs on real videos, held to an earlier run's. |
| `third_party` | Vendored libraries: see its README. |
| `packaging` | What the Windows package adds to the build (Qt's licence texts). |

The native libraries the app loads (Video Compare's presenter, the GPU
metrics' frame decoders) are built from the repository's
`native` folder by its scripts, into `resources/native`, beside the rest
of the app's own files (`resources`: the VMAF models, the translations,
the bundled tools). `native/vulkan_presenter.cpp`, the Python app's
presenter for macOS and Linux, is kept for those systems; the C++ app does
not use it yet.

## The command line

`vml-cli` takes the Python app's commands and options (`vml-cli compare
--help`), and reads them as its argparse parser does. So far it compares on
the CPU path, VMAF v0.6.1, NEG and v1 calculated by FFmpeg's libvmaf with
PSNR, SSIM and XPSNR beside them (`--vmaf-on cpu` or `--cpu`); scores VMAF,
NEG and v1 on the GPU, the app's default, with Vulkan or, where Vulkan
cannot, with libvmaf's CUDA code on an NVIDIA GPU; scores PSNR, SSIM and XPSNR in the app, with its libvmaf's own
extractors, as the Python app does; scores SSIMULACRA2, Butteraugli and
CVVDP on the GPU with Vship-fast's Vulkan build or, with `--vship vship` or
Settings > GPU metrics' choice, stock Vship 5.1.2's, in a pass each or (Settings > GPU metrics) in
one, CVVDP on the display Settings chooses (Automatic unless it names a
preset: the HDR display for an HDR reference, the SDR one otherwise) or on
each `--cvvdp-display`; and scores SSIMULACRA2 and Butteraugli on the CPU with
libjxl's tools with `--cpu`, or where the GPU cannot: without a GPU
Vship can use, or after their GPU pass failed in a comparison up to ten
minutes long. FFmpeg writes each frame pair as 16-bit PNG pictures, held a
few pairs ahead of the tools, which score several pairs at once.
With VMAF on the GPU, PSNR, SSIM and XPSNR are a task of their own on the
CPU, and the two run at once, as the Python app runs a video's tasks; so
are Vship's metrics, in the GPU's queue after VMAF, and SSIMULACRA2 and
Butteraugli on the CPU, in the CPU's queue. Where the app decodes a video
on the CPU (a VVC video, say), PSNR, SSIM and XPSNR are scored in VMAF's
task instead, from the frames decoded for VMAF, rather than decoding the
video a second time. For VMAF and Vship's
metrics on the GPU, and PSNR, SSIM and XPSNR in the app, it decodes the
videos itself, as the Python app does: with the GPU's own decoder (NVIDIA's,
Intel's or AMD's, through the app's libraries in `resources/native`; for VMAF
v1 on an AMD GPU, Windows' own decoder where it opens, its pictures read by
Vulkan where the decoder left them), or with
FFmpeg's decoders built into the app for a video FFmpeg would decode in
software; FFmpeg only copies the compressed stream out of its container. A
video it cannot decode is decoded by an FFmpeg of its own. (The software decoder's AV1 decoder is
dav1d, VideoLAN's, built by `scripts/build_dav1d.ps1` beside FFmpeg's
libraries in `resources/tools/ffmpeg`; its AV2 decoder dav2d, built by
`scripts/build_dav2d.ps1`, decodes AV2 videos, which the app reads itself:
FFmpeg knows no AV2.) That work runs in a process of its
own (the
program again, with `--vml-isolated`), so that a crash in libvmaf, Vship
or the GPU's driver ends that process and not the run: FFmpeg's filters
then calculate VMAF and the others on the CPU, and libjxl's tools
SSIMULACRA2 and Butteraugli, as in the Python app. For a run it cannot do
yet (FFmpeg pairing the frames with `VML_VMAF_PIPES=canvas`), it says what it
cannot do and exits with code 2, rather than giving other answers than the
Python app. It reads and saves the app's saved scores, in the same files as
the Python app: a score either app saved is not calculated again (unless
`--recalculate`), by the same rules. `vml-cli devices` says FFmpeg's
version and where each metric is calculated on the PC, as the Python app's
does.

    build\windows\cli\Release\vml-cli.exe compare reference.mkv encode.mkv -m vmaf,vmaf_neg
    build\windows\cli\Release\vml-cli.exe compare reference.mkv encode.mkv -m vmaf,psnr --vmaf-on cpu
    build\windows\cli\Release\vml-cli.exe devices

## The window

`VideoMetricsLab` is the Python app's window, all its tabs. In the Videos
tab the reference and the test videos are picked, dropped, pasted or
reopened from the last session, read off the window's thread, and dragged
or sorted into order; each row's metrics are ticked in its cells or for
every row in the headings, and hidden or shown from the metrics menu; the
Options panel sets the selected videos' own settings, CVVDP's display and
its presets among them; "Test both" adds a row scoring a video the other
way round, and a resolution test a row scoring the reference scaled down
and back up. Calculate metrics runs the ticked videos as the command line
does, with each video's progress on a line of its own, Pause, Cancel and
the videos in parallel changed while it runs, and shows their scores in the
table, each metric's best and worst in green and red if asked (Highlight
best/worst results). Scores are saved as a run goes, and a video added
again shows them; the saved scores are the Python app's and the command
line's too, and Recalculate (right-click) forgets a video's. Analysis
results are saved and loaded as the Python app's are. Metric Graphs draws
each metric's scores frame by frame, with each video's statistics, and
exports them (PNG, CSV). Video Compare shows a pair of frames, or plays the
videos locked together: decoded with FFmpeg's libraries, on the GPU where
Direct3D 11 can (else in software, AV1 by dav1d, AV2 by dav2d), each video's film grain
added or left out as it is scored (VVC's H.274 by VVdeC), drawn by the
app's presenter (HDR too) and timed
by the source's soundtrack when it plays, or, where that cannot play them,
through ffmpeg.exe.
The Bitrate Viewer plots each video's bitrate by frame, second or GOP. The
Settings tab and the log are the Python app's too. It reads and writes the
Python app's settings.json, and shows its text in the user's language from
the catalogs in `resources/translations`, with Qt's own dialogs
and buttons in Qt's translations of them, in the light or dark theme. Like
the command line, the program is also the process of its own that GPU code
runs in.

## The package

`scripts/build_release.ps1` builds the app's Windows package: the
window and the command line (`VideoMetricsLab-cli.exe`, named as the Python
app's), Qt and Microsoft's C++ runtime beside them, and the app's own files
in `resources` (the VMAF models, the translations, the bundled tools and the
native libraries, from the repository's `resources`), where the programs
find them. It builds
the native libraries and the programs first, checks that everything they
import is in the folder or is Windows', then runs the window's self-test
(`VideoMetricsLab.exe --self-test`) and `VideoMetricsLab-cli.exe devices`
from the folder, in a profile of their own, and zips it. FFmpeg itself is
not bundled. What a release is checked for is in `docs/RELEASING.md`.

    ./scripts/build_release.ps1 -QtPath C:\Qt\6.12.0\msvc2022_64

## Building and testing

Each system has a CMake preset. MSVC's Debug builds check every index and
iterator, so run them too; on Linux, the sanitizer preset does that job.
The tests that need a GPU, or a library the app does not ship for the
system yet, are skipped without it, as on GitHub's runners.

The tests take at most 100 ms each, a few seconds together, and are the
check before every commit: `ctest --preset windows-release`. Real videos
are scored by the golden suite (below); a test that would take longer than
100 ms is not written.

| System | Needs | Commands (from `cpp`) |
| --- | --- | --- |
| Windows | Visual Studio 2022 or its Build Tools (the compiler Qt 6.12 supports), CMake 3.25+ | `cmake --preset windows`, `cmake --build --preset windows-release`, `ctest --preset windows-release` |
| Linux | GCC 11+, CMake 3.25+, Ninja | the same with `linux` and `linux-release`; `linux-sanitizers` and `linux-sanitizers-debug` add AddressSanitizer and UndefinedBehaviorSanitizer |
| macOS | Xcode 15+ or its Command Line Tools, CMake 3.25+, Ninja | the same with `macos` and `macos-release`; it builds for Apple silicon (arm64) and macOS 14.4 or later, Qt 6.12's earliest |

On Windows the app is built with MSVC, and CMake stops at any other
compiler. The `windows` preset finds MSVC from any shell; another generator,
such as Ninja, needs a vcvars64 prompt (`docs/BUILD.md`).

The window and its tests are built where CMake finds Qt 6.5 or newer; name
Qt's folder when configuring, as in
`cmake --preset windows -DCMAKE_PREFIX_PATH=C:/Qt/6.12.0/msvc2022_64`. On
Windows the build puts Qt's libraries beside the window and its tests
(windeployqt), so that they run from the build folder, and Qt's translations
of its own dialogs where they are installed with Qt (qttranslations), which
the package needs. On macOS the window is an app bundle,
`build/macos/app/Release/VideoMetricsLab.app`, which runs from the build
folder with Qt's libraries where Qt is installed. The bundled tools for
macOS are built from their pinned releases' source and are not committed:
`scripts/build_libvmaf_fast_macos.sh` (libvmaf.dylib, needs meson) and
`scripts/build_libjxl_macos.sh` (ssimulacra2 and butteraugli_main), and
`scripts/build_release_macos.sh` runs both and packages the app.

GitHub builds and tests the app when asked (`.github/workflows/ci.yml`:
Actions > CI > Run workflow), at the end of each phase of the work and
before a release: on Windows, with Qt, so that the window and its tests
are built and run there too (those
that need a GPU or FFmpeg skip themselves), and with Linux's sanitizer
preset, both suites. Until the port reaches Linux and macOS, it does not
build their release presets. During the port, the unit tests run on the
development PC before each commit.

## Rules for the code

- Readable first: plain names, and comments in plain sentences that say
  why, as in the rest of the repository.
- Warnings are errors, on every compiler.
- A behaviour gets one test, through the real code, compared with the
  expected values in `tests/reference` wherever they cover it.
- On macOS, `std::to_chars` for doubles needs macOS 13.3 or later; Qt 6.12
  needs 14.4, which sets the earliest macOS the app runs on.
