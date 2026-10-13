# VideoMetricsLab in C++

The C++ port of VideoMetricsLab, made to run faster and in less memory than
the Python app, and to come to macOS and Linux after Windows. The Python app
in the rest of this repository keeps shipping until the C++ app does
everything it does; until then, the C++ code is built and tested but not
released.

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
code. After changing the Python code it uses (its docstring lists it), run
it again from the repository's root and commit what it writes. Its Vulkan
cases need a GPU that Vulkan scores VMAF on; what they write is the same
on every such GPU.

    .venv\Scripts\python.exe cpp\tests\reference\make_reference.py

On real videos, `tests/golden/golden.py` holds the command lines to each
other: it runs the cases in `cases.json` (27 runs covering the ways the app
scores videos) with either app, each with a fresh profile, and compares the
exit codes, the tables, the per-frame CSV files character for character, and
the JSON results. The clips are 2-second cuts of real test videos
(`clips.json`; `golden.py clips` makes them, outside the repository). Two
runs of the Python app on this PC give identical results in every case, the
GPU metrics included, so the C++ app is held to identical results here too.
A video the app decodes itself and scales is, by the Python app's design,
not exactly what FFmpeg's scale filter gives (gpu_frames.py's plan_decode);
both apps scale it the same way, and the `...-by-ffmpeg` case holds them to
each other on FFmpeg's route too.

    python cpp\tests\golden\golden.py run --out python-run
    python cpp\tests\golden\golden.py run --out cpp-run --app path\to\vml-cli.exe
    python cpp\tests\golden\golden.py compare python-run cpp-run

## Layout

| Folder | What it holds |
| --- | --- |
| `core` | `vml_core`, the engine. No Qt and no operating-system headers, so the same code runs on every system, under the command line and the window. |
| `platform` | `vml_platform`: what differs between the systems (starting programs, pipes, loading libraries, the console, the PATH, the GPUs), behind headers that are the same on all of them. |
| `cli` | `vml-cli`, the command line, and its code (`vml_cli`), which the tests use too. |
| `tests/unit` | The tests (GoogleTest). |
| `tests/reference` | The Python app's answers, and the script that makes them. |
| `tests/golden` | The comparison of the two apps' command lines on real videos. |
| `third_party` | Vendored libraries: see its README. |

The presenters and the window join as they are ported.

## The command line

`vml-cli` takes the Python app's commands and options (`vml-cli compare
--help`), and reads them as its argparse parser does. So far it compares on
the CPU path, VMAF v0.6.1, NEG and v1 calculated by FFmpeg's libvmaf with
PSNR, SSIM and XPSNR beside them (`--vmaf-on cpu` or `--cpu`); scores VMAF and
NEG on the GPU with Vulkan, the app's default; and scores PSNR, SSIM and XPSNR
without VMAF in the app, with its libvmaf's own extractors, as the Python app
does. For the last two it decodes the videos itself, as the Python app does:
with the GPU's own decoder (NVIDIA's, Intel's or AMD's, through the app's
libraries in `vmaf_app/native`), or with FFmpeg's decoders built into the app
for a video FFmpeg would decode in software; FFmpeg only copies the compressed
stream out of its container. A video it cannot decode is decoded by an FFmpeg
of its own. (The Python app takes dav1d, the software decoder's AV1 decoder,
from its bundled GStreamer; the C++ app looks for it beside FFmpeg's libraries
in `vmaf_app/tools/ffmpeg`, where a development build has none, so an AV1
video FFmpeg would decode in software is decoded by FFmpeg.) That work runs in
a process of its own (the program again, with `--vml-isolated`), so that a
crash in libvmaf or the GPU's driver ends that process and not the run:
FFmpeg's filters then calculate the scores on the CPU, as in the Python app.
For a run it cannot do yet (VMAF v1 on the GPU, PSNR, SSIM and XPSNR beside
VMAF on the GPU, libvmaf's CUDA code, the perceptual metrics), it says what it
cannot do and exits with code 2, rather than giving other answers than the
Python app. It neither reads nor saves the app's saved scores yet, and
calculates every score.

    build\windows\cli\Release\vml-cli.exe compare reference.mkv encode.mkv -m vmaf,vmaf_neg
    build\windows\cli\Release\vml-cli.exe compare reference.mkv encode.mkv -m vmaf,psnr --vmaf-on cpu

## Building and testing

Each system has a CMake preset. MSVC's Debug builds check every index and
iterator, so run them too; on Linux, the sanitizer preset does that job.
The tests that need a GPU, or a library the app does not ship for the
system yet, are skipped without it, as on GitHub's runners.

| System | Needs | Commands (from `cpp`) |
| --- | --- | --- |
| Windows | Visual Studio 2022 or its Build Tools (the compiler Qt 6.12 supports), CMake 3.25+ | `cmake --preset windows`, `cmake --build --preset windows-release`, `ctest --preset windows-release` |
| Linux | GCC 11+, CMake 3.25+, Ninja | the same with `linux` and `linux-release`; `linux-sanitizers` and `linux-sanitizers-debug` add AddressSanitizer and UndefinedBehaviorSanitizer |
| macOS | Xcode 15+, CMake 3.25+, Ninja | the same with `macos` and `macos-release` |

GitHub builds and tests every push of the `cpp-port` branch on all three
(`.github/workflows/cpp.yml`).

## Rules for the code

- Readable first: plain names, and comments in plain sentences that say
  why, as in the rest of the repository.
- Warnings are errors, on every compiler.
- A behaviour gets one test, through the real code, compared with the
  Python app's answer wherever there is one.
- On macOS, `std::to_chars` for doubles needs macOS 13.3 or later, which
  sets the earliest macOS the engine can run on.
