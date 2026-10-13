"""Holds the C++ app's results to the Python app's, on real videos.

Three commands:

    golden.py clips                    makes the short clips cases.json uses
    golden.py run --out DIR [--app EXE] [--case NAME ...]
                                       runs every case (or those named) with
                                       an app's command line: the Python
                                       app's by default, or the C++ app's
                                       vml-cli
    golden.py compare DIR_A DIR_B [--digits N]
                                       holds two runs' results to each other

Each case runs with a fresh profile (no saved scores, and no settings but
the case's own "settings") and keeps the app's exit code, its output (the tables), its --json results and its
--csv per-frame scores. `compare` asks for the same exit codes, the same
tables, the same CSV files character for character, and the same JSON
results: exactly, or with --digits N, numbers to N significant digits (for
results made on another PC or system).

The clips live outside the repository, in the folder VML_GOLDEN_CLIPS names,
or Documents/VideoMetricsLab-golden/clips. Run with the app's Python from the
repository's root:

    .venv\\Scripts\\python.exe cpp\\tests\\golden\\golden.py run --out run1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
#: The Python app's command line, run as python -c this.
PYTHON_CLI = "import sys; from vmaf_app.cli import main; sys.exit(main(sys.argv[1:]))"


def clips_folder() -> Path:
    """Where the clips are (or will be made)."""
    named = os.environ.get("VML_GOLDEN_CLIPS")
    return Path(named) if named else Path.home() / "Documents" / "VideoMetricsLab-golden" / "clips"


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


# -------------------------------------------------------------------- clips

def make_clips() -> int:
    """Cuts each clip in clips.json from its source: the first `seconds`,
    video only, by stream copy, or encoded as the clip's "encode" says.
    Clips already made are kept."""
    spec = load("clips.json")
    folder = clips_folder()
    folder.mkdir(parents=True, exist_ok=True)
    for clip in spec["clips"]:
        target = folder / clip["name"]
        if target.exists():
            print(f"{target.name}: already made")
            continue
        command = ["ffmpeg", "-hide_banner", "-v", "error", "-y", *clip.get("input_options", []),
                   "-i", clip["source"], "-map", "0:v:0", "-t", str(spec["seconds"]),
                   *clip.get("encode", ["-c", "copy"]), str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            target.unlink(missing_ok=True)  # not a clip: made again next time
            print(f"{target.name}: FAILED\n{result.stderr}")
            return 1
        print(f"{target.name}: made from {clip['source']}")
    return 0


# ---------------------------------------------------------------------- run

def run_cases(out: Path, app: Path | None, names: list[str]) -> int:
    """Runs the cases with the Python app's command line, or with `app`'s,
    and keeps what each gives in out/<case>."""
    cases = [case for case in load("cases.json")["cases"] if not names or case["name"] in names]
    missing = set(names) - {case["name"] for case in cases}
    if missing:
        print(f"no case called {', '.join(sorted(missing))}")
        return 2
    clips = clips_folder()
    for case in cases:
        folder = out / case["name"]
        folder.mkdir(parents=True, exist_ok=True)
        videos = [str(clips / case["reference"]), *(str(clips / name) for name in case["tests"])]
        options = [*case["args"], "--json", str(folder / "results.json"), "--csv", str(folder / "csv"), "-q"]
        command = ([str(app)] if app else [sys.executable, "-c", PYTHON_CLI]) + ["compare", *videos, *options]
        with tempfile.TemporaryDirectory(prefix="vml-golden-profile-") as profile:
            # A fresh home: no saved scores, no black bars kept, and no
            # settings but the case's own, in both apps' settings.json.
            if "settings" in case:
                data = Path(profile) / ".videometricslab"
                data.mkdir()
                (data / "settings.json").write_text(json.dumps(case["settings"]), encoding="utf-8")
            environment = {**os.environ, "USERPROFILE": profile, "HOME": profile, "PYTHONUTF8": "1",
                           **case.get("environment", {})}
            started = time.perf_counter()
            result = subprocess.run(command, cwd=REPOSITORY, env=environment, capture_output=True,
                                    text=True, encoding="utf-8")
            seconds = time.perf_counter() - started
        (folder / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (folder / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        (folder / "exit_code.txt").write_text(f"{result.returncode}\n", encoding="utf-8")
        print(f"{case['name']}: exit {result.returncode}, {seconds:.1f} s")
    return 0


# ------------------------------------------------------------------ compare

def same_number(a: float, b: float, digits: int | None) -> bool:
    """Equal, NaN to NaN; with `digits`, equal to that many significant
    digits."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if a == b:
        return True
    if digits is None or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= 10.0 ** -digits * max(abs(a), abs(b))


def json_differences(a, b, digits: int | None, where: str = "") -> list[str]:
    """Where two JSON values differ, as "path: a != b" lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        lines = []
        for key in dict.fromkeys([*a, *b]):
            if where == "" and key == "version":  # the apps' own versions differ
                continue
            if key not in a or key not in b:
                lines.append(f"{where}/{key}: only in {'the second' if key not in a else 'the first'}")
            else:
                lines += json_differences(a[key], b[key], digits, f"{where}/{key}")
        if list(a) != list(b) and not lines:
            lines.append(f"{where}: the same keys in another order")
        return lines
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items != {len(b)}"]
        return [line for i, (x, y) in enumerate(zip(a, b, strict=True))
                for line in json_differences(x, y, digits, f"{where}[{i}]")]
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):
        if isinstance(a, int) and isinstance(b, int):
            return [] if a == b else [f"{where}: {a} != {b}"]
        return [] if same_number(float(a), float(b), digits) else [f"{where}: {a!r} != {b!r}"]
    return [] if a == b and type(a) is type(b) else [f"{where}: {a!r} != {b!r}"]


def text_differences(a: str, b: str, what: str) -> list[str]:
    """The first line where two texts differ."""
    if a == b:
        return []
    # Not strict: when one text is a beginning of the other, the loop ends
    # without a difference and the line after it says so.
    for number, (x, y) in enumerate(zip(a.splitlines(keepends=True), b.splitlines(keepends=True), strict=False), 1):
        if x != y:
            return [f"{what}, line {number}: {x!r} != {y!r}"]
    return [f"{what}: one ends sooner ({len(a)} != {len(b)} characters)"]


def compare_runs(first: Path, second: Path, digits: int | None) -> int:
    """Holds each case's results in `second` to those in `first`."""
    cases = sorted({path.name for path in first.iterdir() if path.is_dir()}
                   | {path.name for path in second.iterdir() if path.is_dir()})
    failed = 0
    for case in cases:
        a, b = first / case, second / case
        if not (a.is_dir() and b.is_dir()):
            print(f"{case}: only in {'the first' if a.is_dir() else 'the second'} run")
            failed += 1
            continue
        lines = text_differences(_read(a / "exit_code.txt"), _read(b / "exit_code.txt"), "exit code")
        lines += text_differences(_read(a / "stdout.txt"), _read(b / "stdout.txt"), "tables")
        if (a / "results.json").exists() or (b / "results.json").exists():
            lines += json_differences(_json(a / "results.json"), _json(b / "results.json"), digits)
        csv_names = sorted({p.name for p in (a / "csv").glob("*.csv")} | {p.name for p in (b / "csv").glob("*.csv")})
        for name in csv_names:
            lines += text_differences(_read(a / "csv" / name), _read(b / "csv" / name), f"csv/{name}")
        if lines:
            failed += 1
            print(f"{case}: DIFFERENT")
            for line in lines[:12]:
                print(f"  {line}")
            if len(lines) > 12:
                print(f"  ... and {len(lines) - 12} more")
        else:
            print(f"{case}: same")
    print(f"{len(cases) - failed} of {len(cases)} cases the same"
          + ("" if digits is None else f" (numbers to {digits} significant digits)"))
    return 1 if failed else 0


def _read(path: Path) -> str:
    # newline="": CR LF stays CR LF, so line endings are compared too.
    return path.read_text(encoding="utf-8", newline="") if path.exists() else ""


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("clips", help="make the clips")
    run = commands.add_parser("run", help="run the cases")
    run.add_argument("--out", type=Path, required=True, help="the folder for the results")
    run.add_argument("--app", type=Path, help="the C++ app's vml-cli (default: the Python app)")
    run.add_argument("--case", action="append", default=[], help="run only this case (repeatable)")
    compare = commands.add_parser("compare", help="hold two runs to each other")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--digits", type=int, help="numbers need only agree to this many significant digits")
    args = parser.parse_args()
    if args.command == "clips":
        return make_clips()
    if args.command == "run":
        return run_cases(args.out, args.app, args.case)
    return compare_runs(args.first, args.second, args.digits)


if __name__ == "__main__":
    sys.exit(main())
