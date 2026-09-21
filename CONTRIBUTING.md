# Contributing

Thanks for helping improve Video Metrics Calculator. Start with an issue for
large changes, new playback backends, or changes to metric definitions.

## Development setup

Use Windows for the native playback path. From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m vmaf_app.main
```

Install FFmpeg 9+ with `libvmaf` and `xpsnr`, and put its `bin` directory on
PATH. `ffprobe` and, for fallback audio, `ffplay` should accompany it.
Native HDR tone mapping needs the optional shader build described in
[BUILD.md](docs/BUILD.md). See [known limitations](docs/KNOWN_ISSUES.md) before
choosing a Python version; Python 3.14.0 has an unresolved crash in our tests.

## Tests and style

```powershell
.venv\Scripts\python.exe -m ruff check vmaf_app tests
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Tests run offscreen and isolate settings and result caches. Preserve that
isolation in new tests. Some tests run actual FFmpeg encodes; do not replace
those with mocked success merely to make CI green. Hardware playback, HDR,
audio sync and performance still need interactive testing on real hardware.

Follow `ruff.toml` and nearby code. Use NumPy arrays for per-frame hot paths,
keep expensive work off the UI thread, and route subprocesses through
`vmaf_app.core.proc`. Read [the architecture guide](docs/ARCHITECTURE.md).

## Pull requests

- One bug fix per commit, with its regression tests in the same commit.
- Explain the symptom, cause, fix and verification in the commit body.
- Keep refactors separate from behavior changes. Avoid unrelated formatting.
- For bug fixes, check that the new test fails without the fix when feasible.
- Preserve saved-run compatibility, cache identity and shared user-data paths.
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
