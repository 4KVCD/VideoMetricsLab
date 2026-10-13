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
code. After changing that code (`vmaf_app/core/stats.py`, `metrics.py`),
run it again from the repository's root and commit what it writes:

    .venv\Scripts\python.exe cpp\tests\reference\make_reference.py

On real videos, `tests/golden/golden.py` holds the command lines to each
other: it runs the cases in `cases.json` (12 runs covering the ways the app
scores videos) with either app, each with a fresh profile, and compares the
exit codes, the tables, the per-frame CSV files character for character, and
the JSON results. The clips are 2-second cuts of real test videos
(`clips.json`; `golden.py clips` makes them, outside the repository). Two
runs of the Python app on this PC give identical results in every case, the
GPU metrics included, so the C++ app is held to identical results here too.

    python cpp\tests\golden\golden.py run --out python-run
    python cpp\tests\golden\golden.py run --out cpp-run --app path\to\vml-cli.exe
    python cpp\tests\golden\golden.py compare python-run cpp-run

## Layout

| Folder | What it holds |
| --- | --- |
| `core` | `vml_core`, the engine. No Qt and no operating-system headers, so the same code runs on every system, under the command line and the window. |
| `tests/unit` | The tests (GoogleTest). |
| `tests/reference` | The Python app's answers, and the script that makes them. |
| `tests/golden` | The comparison of the two apps' command lines on real videos. |
| `third_party` | Vendored libraries: see its README. |

The operating-system code (decoders, presenters), the command line and the
window join as they are ported.

## Building and testing

Each system has a CMake preset. MSVC's Debug builds check every index and
iterator, so run them too; on Linux, the sanitizer preset does that job.

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
