# Rules for Claude Code sessions on VideoMetricsLab

Repository: github.com/4KVCD/VideoMetricsLab. Work on the current release
branch (`release/v2.0` at the time of writing) unless told otherwise, and pull
before you start. Follow these rules for all work.

## Setup

- Python: `.venv\Scripts\python.exe` (3.14). FFmpeg 9 or newer must be
  installed.
- After pulling, rebuild the GPU decoder DLLs with
  `scripts\build_gpu_frames.ps1` if your copies are older than the pull.
- `libvmaf.dll` and `vmaf_vulkan.dll` are committed and come from libvmaf-fast
  (github.com/4KVCD/libvmaf-fast). A change to libvmaf or the Vulkan shaders
  is made in that fork, not here. During development they are a build of the
  latest commit of the fork's local `fast` branch: run
  `scripts\build_libvmaf_fast_local.ps1` after it changes, and commit what it
  installs. At release, libvmaf-fast is published first, then
  `scripts\fetch_libvmaf_fast.ps1` installs that release, then this app is
  released (`docs\RELEASING.md`).

## Correctness

- A speedup must not change any score. Prove it: same frames, identical values
  before and after.
- Find the real cause before fixing. Do not commit a guess; reproduce the
  problem, then show the fix removes it.
- A test that fails reproducibly is a real bug: fix the code, not the test.

## QA round after every task

A fixed step between "tests pass" and "commit".

- Bugs: re-read everything you changed looking for your own mistakes. Check
  callers of anything renamed or removed; edge cases (empty, failed,
  cancelled, paused, cached, 0 frames); state that is reset or left stale;
  threads; translations.
- The real path: run the change in the real app or with real FFmpeg and a real
  video on this PC's GPU, not only the unit tests.
- Performance: nothing done needlessly often (per frame, per progress tick),
  repeated, or quadratic. It does not have to be maximally fast, just not
  needlessly slow.
- Sense: look for things that do not make sense: comments, messages or docs
  that no longer match the code, leftover debug code, names that mislead, a
  claim in a comment you did not verify.
- Fix what the round finds before committing, in the same commit.
- In your report, say briefly what the round covered and what it found.

## Commits

- One clean commit per task, with its own bug fixes rolled in. Do not commit a
  change and then follow it with separate fix commits.
- If you find a bug in a commit of yours that is not yet on GitHub, fold the
  fix into that commit (amend) rather than adding a fix commit.
- If the commit is already on GitHub, do not amend or rebase it and never
  force-push: make a new commit that names the one it fixes.
- Unrelated tasks still get separate commits; do not bundle two different
  tasks into one.
- Detailed message: the problem, the cause, the change, and what you measured
  or checked (with the numbers and the hardware).
- End every commit message with the `Co-Authored-By:` line for the Claude
  model you are.
- Never use `git stash`.

## Tests

- Before every commit run both, and both must pass (set `PYTHONUTF8=1`):

      .venv\Scripts\python.exe -m ruff check --output-format concise vmaf_app tests scripts
      .venv\Scripts\python.exe -m pytest -q -p no:cacheprovider

- If the suite hangs, rerun with `-n 0 -o faulthandler_timeout=60` to find
  where.
- Never write or keep a test that depends on real elapsed time or machine
  load. Use a fake clock. Delete a flaky test rather than loosening it.
- Add a test for each bug you fix where one is possible without real hardware.

## GitHub

- Do not push, post, or edit anything on GitHub without the user's explicit
  permission, each time.
- Changelog, README and release-note text: show the text and wait for a yes
  before pushing it.
- Changelog style: short one-line main changes, then a separate bug-fix
  heading. No explanations of behaviour.

## The PC

- Kill a process only by PID, and only after checking it is yours. Other
  sessions may be running.
- Never change power or display settings.
- Do not write information about the PC's state into the app's log.
- Ask before downloading anything: say the file name, the source and the size.
- Do not change the user's real app profile or settings unless the task needs
  it. Use a temporary profile for tests.
- If you change the app and it is running, check it is idle, then relaunch it.

## Reporting

- Say plainly what you ran and what happened. If something failed or was
  skipped, say so with the output.
- When you finish, report: the commit hashes, what was verified on this PC's
  GPU (name the GPU and driver version), and anything that should be
  re-checked on the other makers' GPUs.
