# Contributing

Thanks for helping improve VideoMetricsLab. Start with an issue for
large changes, new playback backends, or changes to metric definitions.

## Development setup

The app is C++ (`cpp`), built on Windows for now; macOS and Linux come
later. You need Visual Studio 2022 or its Build Tools with the C++ tools,
CMake 3.25 or newer, Qt 6.12's MSVC 2022 build with Qt's translations, and
MinGW-w64's `g++` for the native libraries. From the repository root:

```powershell
./scripts/build_gpu_frames.ps1
./scripts/build_d3d11_tonemap.ps1
./scripts/build_d3d11_film_grain.ps1
cd cpp
cmake --preset windows -DCMAKE_PREFIX_PATH=C:/Qt/6.12.0/msvc2022_64
cmake --build --preset windows-release
build\windows\app\Release\VideoMetricsLab.exe
```

[cpp/README.md](cpp/README.md) has the rest: the other systems' presets, the
layout, and how the app is checked. Install FFmpeg 9 or newer with `libvmaf`
and `xpsnr`, and put its `bin` directory on PATH, with `ffprobe` beside it.
See [known limitations](docs/KNOWN_ISSUES.md).

The repository's scripts are Python (3.11 or newer):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Tests and style

From `cpp`, after building Release and Debug:

```powershell
ctest --preset windows-release
ctest --preset windows-debug
```

These are the unit tests, at most 100 ms each. The integration tests, which
decode, score and play real videos with FFmpeg and the GPU, run with
`ctest --preset windows-release-integration`; run them when you change
their area. Tests that need a GPU, or FFmpeg, skip themselves without it.

From the repository root, after changing a script or the window's text:

```powershell
.venv\Scripts\python.exe -m ruff check scripts cpp/tests
.venv\Scripts\python.exe scripts\i18n_catalog.py check
git diff --check
```

Tests run offscreen and isolate settings and saved scores. Preserve that
isolation in new tests. Some tests run actual FFmpeg encodes; do not replace
those with mocked success merely to make CI green. Hardware playback, HDR,
audio sync and performance still need interactive testing on real hardware.

Follow `cpp/README.md`'s rules for the code and nearby code: warnings are
errors, keep per-frame work in arrays, keep expensive work off the window's
thread, and start programs through `vml_platform` (`process.h`). Read [the
architecture guide](docs/ARCHITECTURE.md).

## Pull requests

- One bug fix per commit, with its test in the same commit where a test can
  catch the bug coming back.
- Explain the symptom, cause, fix and verification in the commit body.
- Keep refactors separate from behavior changes. Avoid unrelated formatting.
- For bug fixes, check that the new test fails without the fix.
- Treat persistence/cache changes as deliberate behavior changes and document them.
- Document new settings and any changes to score interpretation.
- Include screenshots for UI changes, using generated or shareable media.
- Report skipped tests and hardware-dependent checks you could not perform.

Do not commit private media, user result caches, local settings, binaries or
secrets. A result export can contain absolute media paths. Never clear real
user caches as test setup. Do not silently upgrade a contributor's runtime.

## Reporting problems

Use the bug-report template with exact steps, expected/actual behavior,
versions and sanitized diagnostics. Prefer a small generated reproducer over
a feature film. Send suspected vulnerabilities through [SECURITY.md](SECURITY.md).
Keep discussion respectful and focused on the code; see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
