"""The release benchmark (docs/benchmark.md): how fast VMAF + NEG, VMAF v1,
PSNR, SSIM and XPSNR score frames with libvmaf-fast's new build, its last
release (3.2.0-fast.1, the app before the VMAF v1 merge), official libvmaf
(Netflix's code, built as libvmaf-fast is) and FFmpeg's xpsnr filter. Every
implementation scores the same frames, decoded once into memory; each run is
a process of its own, the implementations take turns, and each one's
per-frame scores are checked against its baseline's.

    python scripts/bench_release.py prepare --reference REF --distorted-2160 D4K --distorted-1080 D1080
    python scripts/bench_release.py run --kit KIT --old-app OLD_TREE [--rounds 5] [--seconds 10]
    python scripts/bench_release.py report RESULTS.json [MORE.json ...]

prepare decodes the frames into --work (default %TEMP%\\vml-bench); run times
every implementation this PC has and writes results\\<computer>.json and .md
there; report puts several computers' results in one markdown file.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SIZES = {"1080": (1920, 1080), "2160": (3840, 2160)}
SIZE_NAMES = {"1080": "1080p", "2160": "4K"}
BITS = 10
FRAME_RATE = 120
#: The app's own VMAF v1 models for these sizes (model_select's defaults).
V1_MODELS = {"1080": "vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", "2160": "vmaf_v1.0.16/vmaf_v1.0.16_1d5h_2160.json"}
METRICS = {"vmaf_neg": "VMAF + NEG", "vmaf_v1": "VMAF v1", "psnr": "PSNR", "ssim": "SSIM", "xpsnr": "XPSNR"}
BUILD_NAMES = {"official": "official libvmaf", "release": "libvmaf-fast 3.2.0-fast.1", "new": "libvmaf-fast new",
               "ffmpeg": "FFmpeg"}
#: Other sessions' benchmarks, which this one must not time over (nor they over it).
FOREIGN = re.compile(r"bench_|vship_runs|compare_vmaf|diagnose_vmaf|clean_runs|sweep_")


def app_ffmpeg() -> str:
    """FFmpeg as the app finds it (its setting, PATH, then where installers put it)."""
    sys.path.insert(0, str(HERE))
    from vmaf_app.core.ffmpeg_locate import ffmpeg_path
    return ffmpeg_path()


def frame_bytes(size: str) -> int:
    width, height = SIZES[size]
    return (width * height + 2 * (width // 2) * (height // 2)) * 2


# ------------------------------------------------------------------ prepare

def prepare(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = args.ffmpeg or app_ffmpeg()
    meta = {"reference": Path(args.reference).name, "start": args.start, "sizes": {}}
    for size, distorted, count in (("2160", args.distorted_2160, args.frames_2160),
                                   ("1080", args.distorted_1080, args.frames_1080)):
        width, height = SIZES[size]
        pick = f"select='between(n\\,{args.start}\\,{args.start + count - 1})'"
        entry = {"distorted": Path(distorted).name, "frames": count}
        for name, path, chain in (
                ("ref", args.reference,
                 pick if size == "2160" else f"{pick},scale={width}:{height}:flags=lanczos+accurate_rnd+bitexact"),
                ("dis", distorted, pick)):
            out = work / f"{name}_{size}.yuv"
            print(f"{out.name}: frames {args.start}-{args.start + count - 1} of {Path(path).name}", flush=True)
            subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0",
                            "-vf", chain, "-fps_mode", "passthrough", "-frames:v", str(count), "-pix_fmt",
                            "yuv420p10le", "-f", "rawvideo", str(out)], check=True)
            if out.stat().st_size != count * frame_bytes(size):
                raise SystemExit(f"{out.name}: {out.stat().st_size} bytes, not {count} frames of {width}x{height}")
            digest = hashlib.sha256()
            with open(out, "rb") as file:
                while chunk := file.read(1 << 24):
                    digest.update(chunk)
            entry[name] = digest.hexdigest()
        meta["sizes"][size] = entry
    (work / "frames.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


# -------------------------------------------------------------------- child

class LibvmafCpu:
    """libvmaf's CPU code with models (built-in versions or .json paths) on
    4:2:0 pictures, as FFmpeg's libvmaf filter hands them over."""

    def __init__(self, vmaf_cuda, width: int, height: int, models: dict[str, str], threads: int):
        self._vmaf = vmaf_cuda
        self._lib = lib = vmaf_cuda._load()
        lib.vmaf_model_load_from_path.restype = ctypes.c_int
        lib.vmaf_model_load_from_path.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                  ctypes.POINTER(vmaf_cuda._ModelConfig), ctypes.c_char_p]
        self._context = ctypes.c_void_p()
        self._models: dict[str, ctypes.c_void_p] = {}
        self._count = 0
        self._width, self._height = width, height
        configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, threads, 1, 0, 0)
        vmaf_cuda._check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
        for name, model in models.items():
            handle, config = ctypes.c_void_p(), vmaf_cuda._ModelConfig(name.encode(), 0)
            if model.endswith(".json"):
                error = lib.vmaf_model_load_from_path(ctypes.byref(handle), ctypes.byref(config),
                                                      vmaf_cuda.path_bytes(model))
            else:
                error = lib.vmaf_model_load(ctypes.byref(handle), ctypes.byref(config), model.encode())
            vmaf_cuda._check(error, f"Loading {model}")
            self._models[name] = handle
            vmaf_cuda._check(lib.vmaf_use_features_from_model(self._context, handle), f"Setting up {model}")
        parameters = vmaf_cuda._PictureParameters(width, height, BITS, vmaf_cuda._VMAF_PIX_FMT_YUV420P)
        vmaf_cuda._check(lib.vmaf_preallocate_pictures(
            self._context, vmaf_cuda._PictureConfiguration(parameters, 2 * (max(1, threads) + 2))),
            "Allocating pictures")
        import numpy as np
        self._np = np
        luma = width * height * 2
        chroma = (width // 2) * (height // 2) * 2
        #: (offset, rows, row bytes) of each plane in a packed frame.
        self._planes = ((0, height, width * 2), (luma, height // 2, width), (luma + chroma, height // 2, width))

    def add(self, reference, distorted) -> None:
        lib, vmaf_cuda, np = self._lib, self._vmaf, self._np
        pictures = (vmaf_cuda._Picture(), vmaf_cuda._Picture())
        for picture, frame in zip(pictures, (reference, distorted), strict=True):
            vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(picture)),
                             "Taking a picture")
            source = np.frombuffer(frame, dtype=np.uint8)
            for plane, (offset, rows, row_bytes) in enumerate(self._planes):
                target = np.ctypeslib.as_array(ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)),
                                               shape=(rows, picture.stride[plane]))
                target[:, :row_bytes] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)
        vmaf_cuda._check(lib.vmaf_read_pictures(self._context, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]),
                                                self._count), f"Scoring frame {self._count}")
        self._count += 1

    def finish(self):
        np, lib = self._np, self._lib
        self._vmaf._check(lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        value = ctypes.c_double()
        scores = {}
        for name, model in self._models.items():
            column = np.empty(self._count)
            for frame in range(self._count):
                self._vmaf._check(lib.vmaf_score_at_index(self._context, model, ctypes.byref(value), frame),
                                  f"Reading frame {frame}")
                column[frame] = float(f"{value.value:.6f}")
            scores[name] = column
        return np.arange(self._count), scores

    def close(self) -> None:
        for model in self._models.values():
            self._lib.vmaf_model_destroy(model)
        self._models = {}
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


def load_frames(work: Path, size: str) -> tuple[list[bytearray], list[bytearray]]:
    each = frame_bytes(size)
    pairs = []
    for name in ("ref", "dis"):
        frames = []
        with open(work / f"{name}_{size}.yuv", "rb") as file:
            while True:
                frame = bytearray(each)
                if file.readinto(frame) != each:
                    break
                frames.append(frame)
        pairs.append(frames)
    return pairs[0], pairs[1]


def ffmpeg_xpsnr(ffmpeg: str, work: Path, size: str, frames: int, loops: int) -> tuple[float, list[float]]:
    """FFmpeg's xpsnr filter over the raw frames, `loops` + 1 times: the seconds FFmpeg's -benchmark gives
    for the run (rtime: from its first frame read, not the process's start), the first pass's XPSNR y."""
    width, height = SIZES[size]
    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{width}x{height}", "-r", str(FRAME_RATE)]
    with tempfile.TemporaryDirectory() as folder:
        stats = Path(folder) / "xpsnr.txt"
        # The filter's own escaping of a Windows path: forward slashes, the drive's colon escaped.
        target = str(stats).replace("\\", "/").replace(":", "\\:")
        process = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-benchmark", "-stream_loop",
                                  str(loops), *raw, "-i", str(work / f"ref_{size}.yuv"), "-stream_loop", str(loops),
                                  *raw, "-i", str(work / f"dis_{size}.yuv"), "-lavfi",
                                  f"[0:v][1:v]xpsnr=stats_file='{target}'", "-f", "null", "-"],
                                 capture_output=True, text=True, check=True)
        seconds = float(re.findall(r"rtime=([0-9.]+)s", process.stderr)[-1])
        values = [float(match) for match in re.findall(r"XPSNR y: *([0-9.]+|inf)", stats.read_text())]
    return seconds, values[:frames]


def child(args) -> int:
    """One run: the implementation's scorer fed the frames once (not timed: the
    scores checked), then again and again for --seconds, timed from the end of
    one pass over the frames to the end of another. Every scorer takes a pair
    only when it has room for it, so its queue is as full at both ends and
    the frames counted are the frames scored in that time. Prints a line of
    JSON."""
    work, size = Path(args.work), args.size
    if args.kind == "ffmpeg":
        sys.path.insert(0, str(Path(args.app)))
        from vmaf_app.core.ffmpeg_locate import ffmpeg_path
        frames = json.loads((work / "frames.json").read_text())["sizes"][size]["frames"]
        first, values = ffmpeg_xpsnr(ffmpeg_path(), work, size, frames, 0)
        loops = max(1, round(args.seconds / first)) if first > 0 else 1
        seconds, _ = ffmpeg_xpsnr(ffmpeg_path(), work, size, frames, loops)
        print(json.dumps({"fps": frames * (loops + 1) / seconds, "frames": frames * (loops + 1),
                          "seconds": seconds, "scores": {"xpsnr": values}}))
        return 0
    sys.path.insert(0, str(Path(args.app)))
    from vmaf_app.core import vmaf_cuda
    vmaf_cuda.LIBRARY_PATH = Path(args.libvmaf)
    if args.vulkan:
        from vmaf_app.core import vmaf_vulkan
        vmaf_vulkan.LIBRARY_PATH = Path(args.vulkan)
    width, height = SIZES[size]
    threads = os.cpu_count() or 1
    reference, distorted = load_frames(work, size)
    models = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
    if args.kind == "cpu" and args.metric == "vmaf_neg":
        scorer = LibvmafCpu(vmaf_cuda, width, height, models, threads)
    elif args.kind == "cpu" and args.metric == "vmaf_v1":
        model = str(Path(args.app) / "vmaf_app" / "models" / V1_MODELS[size])
        scorer = LibvmafCpu(vmaf_cuda, width, height, {"vmaf_v1": model}, threads)
    elif args.kind == "cpu":
        scorer = vmaf_cuda.CpuScorer(width, height, BITS, (args.metric,), 1, threads, FRAME_RATE)
    elif args.kind == "cuda":
        scorer = vmaf_cuda.GpuScorer(width, height, BITS, models)
    elif args.kind == "vulkan":
        from vmaf_app.core import vmaf_vulkan
        scorer = vmaf_vulkan.VulkanScorer(width, height, BITS, models, device=args.device)
    elif args.kind == "v1gpu":
        from vmaf_app.core import vmaf_v1_gpu
        scorer = vmaf_v1_gpu.V1Scorer(width, height, BITS, Path(args.app) / "vmaf_app" / "models" / V1_MODELS[size],
                                      device=args.device)
    else:
        raise SystemExit(f"no implementation {args.kind}")
    try:
        count = len(reference)
        for index in range(count):
            scorer.add(reference[index], distorted[index])
        if args.memory:
            print(f"READY {os.getpid()}", flush=True)
            sys.stdin.readline()
            return 0
        timed = 0
        started = time.perf_counter()
        while True:
            for index in range(count):
                scorer.add(reference[index], distorted[index])
            timed += count
            seconds = time.perf_counter() - started
            if seconds >= args.seconds:
                break
        _frames, scores = scorer.finish()
    finally:
        scorer.close()
    keep = ("vmaf", "vmaf_neg") if args.metric == "vmaf_neg" else None
    out = {}
    for name, column in scores.items():
        if keep is None or name in keep:
            out[name if keep else args.metric] = [float(value) for value in column[:count]]
    version = vmaf_cuda._load().vmaf_version().decode() if args.kind != "ffmpeg" else ""
    print(json.dumps({"fps": timed / seconds, "frames": timed, "seconds": seconds, "scores": out,
                      "libvmaf": version}))
    return 0


# ---------------------------------------------------------------------- run

def powershell(script: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout


def machine() -> dict:
    """What the results were measured on (read only: nothing here is changed)."""
    info = {"computer": platform.node(), "os": platform.platform(), "python": platform.python_version()}
    cpu = powershell("$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
                     "'{0}|{1}|{2}' -f $p.Name.Trim(), $p.NumberOfCores, $p.NumberOfLogicalProcessors").strip()
    if cpu.count("|") == 2:
        name, cores, logical = cpu.split("|")
        info.update(cpu=name, cores=int(cores), threads=int(logical))
    memory = powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").strip()
    if memory.isdigit():
        info["memory_gb"] = round(int(memory) / 2 ** 30)
    info["gpus"] = [line.strip() for line in powershell(
        "Get-CimInstance Win32_VideoController | ForEach-Object { '{0} (driver {1})' -f $_.Name, $_.DriverVersion }"
    ).splitlines() if line.strip()]
    with contextlib.suppress(OSError):
        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True).stdout.strip()
    info["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                          text=True).stdout.strip()
    battery = powershell("Get-CimInstance Win32_Battery | ForEach-Object { $_.BatteryStatus }").strip()
    if battery:
        info["on_ac_power"] = battery.splitlines()[0].strip() == "2"
    return info


def others_running() -> bool:
    import psutil
    me = psutil.Process()
    # This run's own processes: its children, and the venv's python.exe launcher it runs under.
    mine = {me.pid, *(child.pid for child in me.children(recursive=True)), *(parent.pid for parent in me.parents())}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (process.info["pid"] not in mine and (process.info["name"] or "").lower().startswith("python")
                    and FOREIGN.search(" ".join(process.info["cmdline"] or []))):
                return True
        except psutil.Error:
            continue
    return False


def nvidia_busy() -> float:
    """Other processes' use of NVIDIA GPUs (percent of the SMs), 0 without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True).stdout
    except OSError:
        return 0.0
    busy = 0.0
    for line in out.splitlines():
        parts = line.split()
        if line.startswith("#") or len(parts) < 4 or parts[3] == "-":
            continue
        if parts[-1].lower().startswith(("dwm", "claude")):
            continue
        try:
            busy += float(parts[3])
        except ValueError:
            continue
    return busy


def wait_quiet(cpu_limit: float, longest: float = 1800.0) -> bool:
    """Until no other benchmark runs, other processes leave the NVIDIA GPU and
    the CPU idle (three samples running): True, or False after `longest`."""
    import psutil
    started, quiet = time.monotonic(), 0
    while time.monotonic() - started < longest:
        if not others_running() and nvidia_busy() < 5 and psutil.cpu_percent(interval=1.0) < cpu_limit:
            quiet += 1
            if quiet >= 3:
                return True
        else:
            quiet = 0
            time.sleep(10)
    return False


def dedicated_gpu_memory(pid: int) -> int:
    out = powershell(f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage').CounterSamples | "
                     "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum").strip()
    try:
        return int(float(out))
    except ValueError:
        return 0


def plan_rows(new_tree: Path) -> tuple[list[dict], list[dict]]:
    """The rows this PC can run, and its Vulkan GPUs."""
    sys.path.insert(0, str(new_tree))
    from vmaf_app.core import vmaf_vulkan
    gpus = [{"index": device.index, "name": device.name, "vendor": device.vendor}
            for device in vmaf_vulkan.devices()
            if device.usable and device.kind in (vmaf_vulkan._TYPE_INTEGRATED, vmaf_vulkan._TYPE_DISCRETE)]
    nvidia = any(gpu["vendor"] == 0x10DE for gpu in gpus)
    rows = [{"metric": "vmaf_neg", "build": build, "kind": "cpu"} for build in ("official", "new")]
    if nvidia:
        rows += [{"metric": "vmaf_neg", "build": build, "kind": "cuda"} for build in ("official", "release", "new")]
    for gpu in gpus:
        rows += [{"metric": "vmaf_neg", "build": build, "kind": "vulkan", "device": gpu["index"]}
                 for build in ("release", "new")]
    rows.append({"metric": "vmaf_v1", "build": "official", "kind": "cpu"})
    for gpu in gpus:
        rows += [{"metric": "vmaf_v1", "build": build, "kind": "v1gpu", "device": gpu["index"]}
                 for build in ("release", "new")]
    for metric in ("psnr", "ssim"):
        rows += [{"metric": metric, "build": build, "kind": "cpu"} for build in ("official", "release", "new")]
    rows += [{"metric": "xpsnr", "build": "ffmpeg", "kind": "ffmpeg"}, {"metric": "xpsnr", "build": "new", "kind": "cpu"}]
    names = {gpu["index"]: gpu["name"] for gpu in gpus}
    gpus_by_index = {gpu["index"]: gpu for gpu in gpus}
    nvidia_name = next((gpu["name"] for gpu in gpus if gpu["vendor"] == 0x10DE), "NVIDIA")
    for row in rows:
        kind, device = row["kind"], row.get("device")
        if kind in ("cpu", "ffmpeg"):
            where = "CPU"
        elif kind == "cuda":
            where = f"CUDA, {nvidia_name}"
        elif kind == "vulkan":
            where = f"Vulkan, {names[device]}"
        else:  # VMAF v1: 3.2.0-fast.1 left CAMBI and SpEED to the CPU
            where = f"{'GPU + CPU' if row['build'] == 'release' else 'GPU'}, {names[device]}"
        build = BUILD_NAMES[row["build"]] + (" xpsnr filter" if kind == "ffmpeg" else "")
        row["label"] = f"{build} ({where})"
        # On the NVIDIA GPU: compared with libvmaf's CUDA there too.
        row["nvidia"] = kind == "cuda" or (device is not None and gpus_by_index[device]["vendor"] == 0x10DE)
        row["id"] = "/".join(str(part) for part in (row["metric"], row["build"], kind, "" if device is None else device))
    return rows, gpus


def child_command(args, row: dict, size: str, seconds: float, memory: bool = False) -> list[str]:
    builds = {
        "official": (HERE, Path(args.kit) / "official" / "libvmaf.dll", None),
        "release": (Path(args.old_app), Path(args.old_app) / "vmaf_app" / "tools" / "libvmaf" / "libvmaf.dll",
                    Path(args.old_app) / "vmaf_app" / "tools" / "vmaf_vulkan" / "vmaf_vulkan.dll"),
        "new": (HERE, Path(args.kit) / "new" / "libvmaf.dll",
                HERE / "vmaf_app" / "tools" / "vmaf_vulkan" / "vmaf_vulkan.dll"),
        "ffmpeg": (HERE, Path(args.kit) / "new" / "libvmaf.dll", None),
    }
    app, libvmaf, vulkan = builds[row["build"]]
    command = [sys.executable, str(Path(__file__).resolve()), "child", "--work", args.work, "--size", size,
               "--metric", row["metric"], "--kind", row["kind"], "--app", str(app), "--libvmaf", str(libvmaf),
               "--seconds", str(seconds)]
    if vulkan:
        command += ["--vulkan", str(vulkan)]
    if "device" in row:
        command += ["--device", str(row["device"])]
    if memory:
        command.append("--memory")
    return command


def run(args) -> int:
    work = Path(args.work)
    frames = json.loads((work / "frames.json").read_text(encoding="utf-8"))
    for path in (Path(args.kit) / "official" / "libvmaf.dll", Path(args.kit) / "new" / "libvmaf.dll",
                 Path(args.old_app) / "vmaf_app" / "core" / "vmaf_cuda.py"):
        if not path.exists():
            raise SystemExit(f"missing: {path}")
    rows, gpus = plan_rows(HERE)
    sizes = [size for size in ("1080", "2160") if size in frames["sizes"]]
    out_dir = work / "results"
    out_dir.mkdir(exist_ok=True)
    name = args.name or platform.node()
    results = {"machine": machine(), "gpus": gpus, "frames": frames, "rows": rows, "runs": [], "scores": {},
               "memory": {}, "settings": {"rounds": args.rounds, "seconds": args.seconds, "cooldown": args.cooldown,
                                          "threads": os.cpu_count()},
               "builds": builds_info(args)}
    path = out_dir / f"{name}.json"

    def save() -> None:
        results["summary"] = summarize(results)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1), encoding="utf-8")
        temporary.replace(path)
        (out_dir / f"{name}.md").write_text(markdown(results), encoding="utf-8")

    failed: set[tuple[str, str]] = set()
    total = (args.rounds + 1) * len(sizes) * len(rows)
    done = 0
    for round_ in range(args.rounds + 1):  # round 0: the warm-up, not counted
        for size in sizes:
            for row in rows:
                done += 1
                if (row["id"], size) in failed:
                    continue
                quiet = wait_quiet(args.cpu_quiet)
                seconds = 2.0 if round_ == 0 else args.seconds
                process = subprocess.run(child_command(args, row, size, seconds), capture_output=True, text=True)
                line = process.stdout.strip().splitlines()[-1:] if process.returncode == 0 else []
                record = {"id": row["id"], "size": size, "round": round_, "quiet": quiet}
                try:
                    record.update(json.loads(line[0]))
                except (IndexError, json.JSONDecodeError):
                    record["error"] = (process.stderr or process.stdout).strip()[-2000:]
                    failed.add((row["id"], size))
                if "scores" in record:
                    scores = record.pop("scores")
                    if round_ == 0:
                        results["scores"].setdefault(row["id"], {})[size] = scores
                results["runs"].append(record)
                state = f"{record['fps']:9.1f} fps" if "fps" in record else "FAILED"
                print(f"[{done}/{total}] round {round_} {SIZE_NAMES[size]:5} {METRICS[row['metric']]:10} "
                      f"{row['label']:58} {state}{'' if quiet else ' (not quiet)'}", flush=True)
                save()
                time.sleep(args.cooldown)
    # libvmaf CUDA's GPU memory at 4K, VMAF + NEG (the context and its pictures, frames scored).
    for row in rows:
        if row["kind"] == "cuda" and "2160" in sizes and (row["id"], "2160") not in failed:
            command = child_command(args, row, "2160", 1.0, memory=True)
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            ready = process.stdout.readline().split()
            if ready[:1] == ["READY"]:
                results["memory"][row["id"]] = dedicated_gpu_memory(int(ready[1]))
            process.stdin.write("\n")
            process.stdin.flush()
            process.wait()
    save()
    print(f"\n{path}\n{path.with_suffix('.md')}")
    return 0


def builds_info(args) -> dict:
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""

    def commit(tree: Path) -> str:
        return subprocess.run(["git", "-C", str(tree), "log", "-1", "--format=%h %s"], capture_output=True,
                              text=True).stdout.strip()[:100]

    old = Path(args.old_app)
    return {
        "official_libvmaf": sha(Path(args.kit) / "official" / "libvmaf.dll"),
        "new_libvmaf": sha(Path(args.kit) / "new" / "libvmaf.dll"),
        "new_vmaf_vulkan": sha(HERE / "vmaf_app" / "tools" / "vmaf_vulkan" / "vmaf_vulkan.dll"),
        "release_libvmaf": sha(old / "vmaf_app" / "tools" / "libvmaf" / "libvmaf.dll"),
        "release_vmaf_vulkan": sha(old / "vmaf_app" / "tools" / "vmaf_vulkan" / "vmaf_vulkan.dll"),
        "new_app": commit(HERE), "release_app": commit(old),
        "ffmpeg": subprocess.run([app_ffmpeg(), "-version"], capture_output=True,
                                 text=True).stdout.split("\n")[0][:80],
    }


# ------------------------------------------------------------ summary, report

def compare(values: dict, against: dict) -> str:
    """'identical', or the largest difference of any score, per frame."""
    worst = 0.0
    for name, column in values.items():
        other = against.get(name)
        if other is None or len(other) != len(column):
            return "not compared"
        for a, b in zip(column, other, strict=True):
            if a != b:
                worst = max(worst, math.inf if math.isinf(a) or math.isinf(b) else abs(a - b))
    return "identical" if worst == 0 else f"differs by up to {worst:.6g}"


def summarize(results: dict) -> list[dict]:
    rows = {row["id"]: row for row in results["rows"]}
    summary = []
    for size in results["frames"]["sizes"]:
        fps: dict[str, list[float]] = {}
        for record in results["runs"]:
            if record["size"] == size and record["round"] > 0 and "fps" in record:
                fps.setdefault(record["id"], []).append(record["fps"])
        medians = {key: statistics.median(values) for key, values in fps.items()}
        scores = {key: value[size] for key, value in results["scores"].items() if size in value}
        for key, row in rows.items():
            if key not in medians:
                continue
            entry = {"size": size, "id": key, "metric": row["metric"], "label": row["label"],
                     "fps": medians[key], "runs": len(fps[key]), "min": min(fps[key]), "max": max(fps[key]),
                     "speedup": {}, "scores": {}}
            for other, title in baselines(row, rows):
                if other in medians:
                    entry["speedup"][title] = medians[key] / medians[other]
                if key in scores and other in scores:
                    entry["scores"][title] = compare(scores[key], scores[other])
            summary.append(entry)
    return summary


def baselines(row: dict, rows: dict) -> list[tuple[str, str]]:
    """The rows a row is measured against: (id, title)."""
    metric, build, kind, device = row["metric"], row["build"], row["kind"], row.get("device", "")
    found = []

    def add(other_build: str, other_kind: str, other_device="", title: str = "") -> None:
        key = f"{metric}/{other_build}/{other_kind}/{other_device}"
        if key in rows and key != row["id"]:
            found.append((key, title or rows[key]["label"]))

    if build == "ffmpeg" or (build == "official" and kind == "cpu"):
        return found
    if metric == "xpsnr":
        add("ffmpeg", "ffmpeg")
    elif metric == "vmaf_v1":
        add("official", "cpu")
        if build == "new":
            add("release", kind, device)
    elif metric == "vmaf_neg":
        add("official", "cpu")
        if row.get("nvidia"):
            add("official", "cuda")
            if kind == "vulkan":
                add(build, "cuda")
        if build == "new":
            add("release", kind, device)
    else:
        add("official", "cpu")
        if build == "new":
            add("release", "cpu")
    return found


def markdown(results: dict) -> str:
    info = results["machine"]
    lines = [f"# {info.get('computer', '')}: {info.get('cpu', '')}, {', '.join(info.get('gpus', []))}", "",
             f"{info.get('cores', '?')} cores / {info.get('threads', '?')} threads, {info.get('memory_gb', '?')} GB; "
             f"{info.get('os', '')}; NVIDIA driver {info.get('nvidia_driver') or '-'}; "
             f"{info.get('power_scheme', '')}" + ("" if info.get("on_ac_power", True) else "; ON BATTERY"), "",
             f"Frames: {results['frames']['reference']} from frame {results['frames']['start']}, 10-bit; "
             f"each number the median of {results['settings']['rounds']} runs of {results['settings']['seconds']} s "
             "or more (after a warm-up run), the implementations taking turns.", ""]
    for size in results["frames"]["sizes"]:
        entries = [entry for entry in results.get("summary", []) if entry["size"] == size]
        if not entries:
            continue
        width, height = SIZES[size]
        lines += [f"## {SIZE_NAMES[size]} ({width}x{height})", "",
                  "| Metric | Implementation | Frames/s | Range | Faster than | Scores |", "|---|---|---|---|---|---|"]
        for entry in entries:
            faster = "; ".join(f"{value:.2f}x {title}" for title, value in entry["speedup"].items())
            checks = "; ".join(f"{value} ({title})" for title, value in entry["scores"].items())
            lines.append(f"| {METRICS[entry['metric']]} | {entry['label']} | {entry['fps']:.1f} | "
                         f"{entry['min']:.1f}-{entry['max']:.1f} | {faster} | {checks} |")
        lines.append("")
    if results.get("memory"):
        lines += ["## GPU memory, 4K VMAF + NEG (libvmaf CUDA, dedicated)", ""]
        rows = {row["id"]: row for row in results["rows"]}
        for key, value in results["memory"].items():
            lines.append(f"- {rows[key]['label']}: {value / 2 ** 20:.0f} MB")
        lines.append("")
    failures = [record for record in results["runs"] if "error" in record]
    if failures:
        lines += ["## Failed", ""] + [f"- {record['id']} {SIZE_NAMES[record['size']]}: "
                                      f"{record['error'].splitlines()[-1] if record['error'] else '?'}"
                                      for record in failures] + [""]
    loud = sum(1 for record in results["runs"] if not record.get("quiet", True))
    if loud:
        lines += [f"{loud} runs started before the PC was quiet (30 minutes waited).", ""]
    return "\n".join(lines)


def report(args) -> int:
    parts = []
    for path in args.results:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        parts.append(markdown(results))
    text = "\n\n".join(parts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    default_work = str(Path(tempfile.gettempdir()) / "vml-bench")
    p = commands.add_parser("prepare")
    p.add_argument("--reference", required=True)
    p.add_argument("--distorted-2160", required=True)
    p.add_argument("--distorted-1080", required=True)
    p.add_argument("--start", type=int, default=1200, help="first frame (by number) of both videos")
    p.add_argument("--frames-2160", type=int, default=48)
    p.add_argument("--frames-1080", type=int, default=96)
    p.add_argument("--work", default=default_work)
    p.add_argument("--ffmpeg")
    p = commands.add_parser("run")
    p.add_argument("--kit", required=True, help="folder with official\\libvmaf.dll and new\\libvmaf.dll")
    p.add_argument("--old-app", required=True, help="the app at the commit before the VMAF v1 merge (bfc00d4)")
    p.add_argument("--work", default=default_work)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--cooldown", type=float, default=2.0, help="seconds of rest after each run")
    p.add_argument("--cpu-quiet", type=float, default=15.0, help="CPU use (percent) a run waits to be under")
    p.add_argument("--name", help="the results' file name (default: the computer's name)")
    p = commands.add_parser("child")
    for option in ("--work", "--size", "--metric", "--kind", "--app", "--libvmaf", "--vulkan"):
        p.add_argument(option)
    p.add_argument("--device", type=int)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--memory", action="store_true")
    p = commands.add_parser("report")
    p.add_argument("results", nargs="+")
    p.add_argument("--out")
    args = parser.parse_args()
    return {"prepare": prepare, "run": run, "child": child, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
