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
| Folder | ~148 MB |
| Zip | ~58 MB |
| Output | `%LOCALAPPDATA%\VideoMetricsCalculator-build\` |

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

## What is not, and why

**FFmpeg and libvmaf.** The app finds them at runtime and prompts for their
location if they are missing. They are left out deliberately:

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
.\VideoMetricsCalculator.exe --self-test
```

It reports FFmpeg, GStreamer and its plugins, and the GPU shader, in a dialog
and at `~\.vmaf-calculator\self-test.txt`. `--quiet` skips the dialog and
sets the exit code instead, which is how `build_release.ps1` uses it.

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
