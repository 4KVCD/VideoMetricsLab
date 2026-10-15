# VideoMetricsLab in C++

VideoMetricsLab is a C++ app from v3.0 on. It was ported from the Python
app (v2.1 and before; its last release is the `release/v2.1` branch) to
run faster and in less memory, and to come to macOS and Linux after
Windows. A comment that names a `vmaf_app/...` file or a Python test
(`tests/test_...py`) means the Python app's code the C++ code was ported
from, on that branch.

## How it is checked

The C++ code must give the Python app's answers:

- **Scores from the libraries** (libvmaf, Vship, libjxl's tools) must be
  identical, Python against C++ on the same PC: the same libraries, fed the
  same frames, give the same numbers.
- **Numbers the app calculates itself** (the mean, median, standard
  deviation, percentiles, XPSNR's average) must agree to 12 significant
  digits. On Windows the tests ask for them exactly, and they are: sums are
  made in numpy's order (`numpy_sum` in `core/src/stats.cpp`).

The tests read the Python app's answers from `tests/reference`, which
`tests/reference/make_reference.py` makes by running the Python app's own
code: a checkout of its last release (the `release/v2.1` branch), in the
folder `VML_PYTHON_APP` names, run with a Python that has its
requirements. After changing what it covers (its docstring lists it: the
window's catalogs, say), run it again from the repository's root and
commit what it writes. Its Vulkan cases need a GPU that Vulkan scores VMAF
on; what they write is the same on every such GPU.

    set VML_PYTHON_APP=<the checkout>
    .venv\Scripts\python.exe cpp\tests\reference\make_reference.py

On real videos, `tests/golden/golden.py` holds the command lines to each
other: it runs the cases in `cases.json` (80 runs covering the ways the app
scores videos) with either app, each with a fresh profile (and the case's
own settings, where it has any; some cases save scores in it first, with
either app), and compares the exit codes, the tables, the per-frame CSV
files character for character, the JSON results, and what each run saved in
the saved scores. The clips are 2-second cuts of real test videos
(`clips.json`; `golden.py clips` makes them, outside the repository). Two
runs of the Python app on this PC give identical results in every case, the
GPU metrics included, so the C++ app is held to identical results here too.
A video the app decodes itself and scales is, by the Python app's design,
not exactly what FFmpeg's scale filter gives (gpu_frames.py's plan_decode);
both apps scale it the same way, and the `...-by-ffmpeg` case holds them to
each other on FFmpeg's route too.

    set VML_PYTHON_APP=<a checkout of release/v2.1>
    python cpp\tests\golden\golden.py run --out python-run --cpp-app path\to\vml-cli.exe
    python cpp\tests\golden\golden.py run --out cpp-run --app path\to\vml-cli.exe
    python cpp\tests\golden\golden.py compare python-run cpp-run

## Layout

| Folder | What it holds |
| --- | --- |
| `core` | `vml_core`, the engine. No Qt and no operating-system headers, so the same code runs on every system, under the command line and the window. |
| `platform` | `vml_platform`: what differs between the systems (starting programs, pipes, loading libraries, the console, the PATH, the GPUs), behind headers that are the same on all of them. |
| `cli` | `vml-cli`, the command line, and its code (`vml_cli`), which the tests use too. |
| `app` | `VideoMetricsLab`, the window (Qt 6), and its code (`vml_app`), which its tests use too. |
| `tests/unit` | The tests (GoogleTest). |
| `tests/app` | The window's tests, run with no screen (Qt's offscreen platform), each in a profile of its own. |
| `tests/reference` | The Python app's answers, and the script that makes them. |
| `tests/fixtures` | Small videos the window's tests play, and the script that made them. |
| `tests/golden` | The comparison of the two apps' command lines on real videos. |
| `third_party` | Vendored libraries: see its README. |
| `packaging` | What the Windows package adds to the build (Qt's licence texts). |

The native libraries the app loads (Video Compare's presenter, its film
grain, the GPU metrics' frame decoders) are built from the repository's
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
NEG and v1 on the GPU, the app's default, with Vulkan or, where the GPU
backend setting says CUDA or Vulkan cannot, with libvmaf's CUDA code on an
NVIDIA GPU; scores PSNR, SSIM and XPSNR in the app, with its libvmaf's own
extractors, as the Python app does; scores SSIMULACRA2, Butteraugli and
CVVDP on the GPU with Vship, in a pass each or (Settings > GPU metrics) in
one, CVVDP on the display preset Settings chooses or on each
`--cvvdp-display`; and scores SSIMULACRA2 and Butteraugli on the CPU with
libjxl's tools where they are set to run there (`--ssimulacra2-on cpu`,
`--butteraugli-on cpu` or `--cpu`), or where the GPU cannot: without a GPU
Vship can use, or after their GPU pass failed in a comparison up to ten
minutes long. FFmpeg writes each frame pair as 16-bit PNG pictures, held a
few pairs ahead of the tools, which score several pairs at once.
With VMAF on the GPU, PSNR, SSIM and XPSNR are a task of their own on the
CPU, and the two run at once, as the Python app runs a video's tasks; so
are Vship's metrics, in the GPU's queue after VMAF, and SSIMULACRA2 and
Butteraugli set to the CPU, in the CPU's queue. For VMAF and Vship's
metrics on the GPU, and PSNR, SSIM and XPSNR in the app, it decodes the
videos itself, as the Python app does: with the GPU's own decoder (NVIDIA's,
Intel's or AMD's, through the app's libraries in `resources/native`; for VMAF
v1 on an AMD GPU, Windows' own decoder where it opens, its pictures read by
Vulkan where the decoder left them), or with
FFmpeg's decoders built into the app for a video FFmpeg would decode in
software; FFmpeg only copies the compressed stream out of its container. A
video it cannot decode is decoded by an FFmpeg of its own. (The software decoder's AV1 decoder is
dav1d, VideoLAN's, built by `scripts/build_dav1d.ps1` beside FFmpeg's
libraries in `resources/tools/ffmpeg`.) That work runs in a process of its
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
Direct3D 11 can (else in software, AV1 by dav1d), a VVC video's film grain drawn as the Python app draws it
(AV1's, AFGS1, by FFmpeg), drawn by the app's presenter (HDR too) and timed
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

The tests come in two suites. The unit tests take at most 100 ms each, a
few seconds together, and are the check before every commit:
`ctest --preset windows-release`. The integration tests decode, score and
play real videos with FFmpeg, the GPU and the window, and take longer:
`ctest --preset windows-release-integration`, or both suites with
`windows-release-all`. They run at the end of each phase of the work,
before a release, and when their code changes. A test that takes longer
than 100 ms goes on an integration list in `tests/CMakeLists.txt`.

| System | Needs | Commands (from `cpp`) |
| --- | --- | --- |
| Windows | Visual Studio 2022 or its Build Tools (the compiler Qt 6.12 supports), CMake 3.25+ | `cmake --preset windows`, `cmake --build --preset windows-release`, `ctest --preset windows-release` |
| Linux | GCC 11+, CMake 3.25+, Ninja | the same with `linux` and `linux-release`; `linux-sanitizers` and `linux-sanitizers-debug` add AddressSanitizer and UndefinedBehaviorSanitizer |
| macOS | Xcode 15+, CMake 3.25+, Ninja | the same with `macos` and `macos-release` |

The window and its tests are built where CMake finds Qt 6.5 or newer; name
Qt's folder when configuring, as in
`cmake --preset windows -DCMAKE_PREFIX_PATH=C:/Qt/6.12.0/msvc2022_64`. On
Windows the build puts Qt's libraries beside the window and its tests
(windeployqt), so that they run from the build folder, and Qt's translations
of its own dialogs where they are installed with Qt (qttranslations), which
the package needs.

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
  Python app's answer wherever there is one.
- On macOS, `std::to_chars` for doubles needs macOS 13.3 or later, which
  sets the earliest macOS the engine can run on.
