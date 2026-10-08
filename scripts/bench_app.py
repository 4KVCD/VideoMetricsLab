"""VideoMetricsLab's benchmark (docs/benchmark.md): an older release of the app
against this one, in whole runs as the Run button makes them -- decoding and
scoring -- on this PC.

Only what changed since v1.4 is measured, each cell one optimization:

  group   metrics                     videos  what changed
  nvidia  VMAF + NEG (CUDA)           VVC     VVC decoded in the scoring process, not piped from FFmpeg
  nvidia  VMAF v1                     HEVC    on the GPU, not FFmpeg's libvmaf on the CPU
  gpu     VMAF + NEG                  HEVC    with Vulkan on the PC's other GPU, not on the CPU
  gpu     VMAF v1                     HEVC    the same
  cpu     PSNR + SSIM + XPSNR         HEVC    libvmaf-fast in the app, not FFmpeg's filters
  cpu     SSIMULACRA2 + Butteraugli   HEVC    frame pairs scored in parallel on the CPU

each at 4K and 1080p, and the window's memory once it has opened ("idle").

    python scripts/bench_app.py bench --old OLD_CHECKOUT --videos DIR --out DIR [--only nvidia,gpu,cpu,idle]
    python scripts/bench_app.py report DIR

A run is a process of its own with a temporary profile (no saved scores,
default settings); its speed is the steady state, from 10% to 90% of its
frames, so start-up is left out. Each run is sized to about 12 s at the rate
expected here (EXPECTED_FPS), or made again at its measured rate when it came
out too short. Both versions score the same first frames, which are
compared. Each run waits for a quiet PC and is made again once if another
process used a GPU or much of the CPU while it ran. One round: by the
2026-10-08 run's rates, about 20 minutes on the RTX 5090 PC.

(`run` is one run, started by `bench`.)
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
#: The environment this script started with, for the runs: vmaf_app, imported
#: here to find the GPUs, sets variables of its own (OPENBLAS_NUM_THREADS since
#: v2.0), which would otherwise reach the older version's runs too.
_ENVIRONMENT = dict(os.environ)
#: The test clip and its encodes, in the folder --videos names (docs/benchmark.md).
VIDEOS = {"reference": "VideoQ_HDR10_UHD_120fps_4m00s.mp4",
          "hevc-4k": "VideoQ HDR10 4K H.265 CRF 22 medium.mkv",
          "hevc-1080p": "VideoQ HDR10 1080p H.265 CRF 22 medium.mkv",
          "vvc-4k": "VideoQ HDR10 4K VVC QP 32 faster.mkv",
          "vvc-1080p": "VideoQ HDR10 1080p VVC QP 32 faster.mkv"}
VIDEO_FPS = 120
CELLS = (
    {"group": "nvidia", "name": "VMAF + NEG", "metrics": ("vmaf", "vmaf_neg"), "videos": ("vvc-4k", "vvc-1080p")},
    {"group": "nvidia", "name": "VMAF v1", "metrics": ("vmaf_v1",), "videos": ("hevc-4k", "hevc-1080p")},
    {"group": "gpu", "name": "VMAF + NEG", "metrics": ("vmaf", "vmaf_neg"), "videos": ("hevc-4k", "hevc-1080p")},
    {"group": "gpu", "name": "VMAF v1", "metrics": ("vmaf_v1",), "videos": ("hevc-4k", "hevc-1080p")},
    {"group": "cpu", "name": "PSNR + SSIM + XPSNR", "metrics": ("psnr", "ssim", "xpsnr"),
     "videos": ("hevc-4k", "hevc-1080p")},
    {"group": "cpu", "name": "SSIMULACRA2 + Butteraugli", "metrics": ("ssimulacra2", "butteraugli"),
     "videos": ("hevc-4k", "hevc-1080p"), "perceptual_cpu": True},
)
#: Frames a second each cell scored on the RTX 5090 PC (Core Ultra 9 285K,
#: its iGPU as "gpu"), v1.4 and v2.0 on 2026-10-08, to size a run: on another PC a run that
#: comes out too short is made again at its own rate.
EXPECTED_FPS = {
    "nvidia/VMAF + NEG/vvc-4k": (48, 85), "nvidia/VMAF + NEG/vvc-1080p": (127, 341),
    "nvidia/VMAF v1/hevc-4k": (34, 584), "nvidia/VMAF v1/hevc-1080p": (102, 604),
    "gpu/VMAF + NEG/hevc-4k": (23, 62), "gpu/VMAF + NEG/hevc-1080p": (56, 126),
    "gpu/VMAF v1/hevc-4k": (28, 40), "gpu/VMAF v1/hevc-1080p": (58, 93),
    "cpu/PSNR + SSIM + XPSNR/hevc-4k": (21, 232), "cpu/PSNR + SSIM + XPSNR/hevc-1080p": (67, 185),
    "cpu/SSIMULACRA2 + Butteraugli/hevc-4k": (0.2, 1.6), "cpu/SSIMULACRA2 + Butteraugli/hevc-1080p": (0.9, 6.2),
}
#: The steady seconds a run aims for; shorter than SHORTEST, it is made again.
TARGET_SECONDS, SHORTEST_SECONDS = 12.0, 6.0
#: The fewest frames a run scores: GPU metrics, and the CPU's slow perceptual ones.
FEWEST_FRAMES, FEWEST_CPU_FRAMES = 240, 12
#: Another process counts as busy above this much of a GPU engine, or this
#: share of one CPU thread.
BUSY_GPU_PERCENT, BUSY_CPU_SHARE = 15.0, 0.5
#: The NVIDIA GPU hidden from a run (these variables only): CUDA sees no
#: device, the Vulkan loader skips NVIDIA's driver.
HIDE_NVIDIA = {"CUDA_VISIBLE_DEVICES": "-1", "VK_LOADER_DRIVERS_DISABLE": "*nv-vk64*"}


def cell_id(cell: dict, video: str) -> str:
    return f"{cell['group']}/{cell['name']}/{video}"


def file_name(cell_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", cell_key)


# ----------------------------------------------------------------- job object

class _Accounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong), ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class Job:
    """This process and everything it starts, for CPU time and peak memory."""

    def __init__(self) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = wintypes.HANDLE
        self._k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.handle = self._k32.CreateJobObjectW(None, None)
        if not self.handle or not self._k32.AssignProcessToJobObject(
                wintypes.HANDLE(self.handle), wintypes.HANDLE(self._k32.GetCurrentProcess())):
            raise OSError(ctypes.get_last_error(), "the run could not be put in a job object")

    def _query(self, kind: int, info):
        if not self._k32.QueryInformationJobObject(wintypes.HANDLE(self.handle), kind, ctypes.byref(info),
                                                   ctypes.sizeof(info), None):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject")
        return info

    def cpu_seconds(self) -> float:
        info = self._query(1, _Accounting())
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7

    def peak_memory(self) -> int:
        return int(self._query(9, _ExtendedLimit()).PeakJobMemoryUsed)


# ------------------------------------------------------------------- one run

def run_one(args) -> None:
    """One video through the Run button's code (JobScheduler) without the
    window, the app's start-up probes first; the same interface since v1.4.
    Started by `bench`, with USERPROFILE set to an empty folder."""
    job = Job()  # first: every process the run starts is counted
    repo = Path(args.repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import vmaf_app  # first, as the app imports it: before numpy

    # isort: split
    from dataclasses import replace

    import numpy as np

    from vmaf_app.core import perceptual_vship, vmaf_cuda
    from vmaf_app.core.cvvdp import default_settings
    from vmaf_app.core.ffmpeg_locate import check_tools, format_version
    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.geometry import analysis_dimensions
    from vmaf_app.core.job_runner import JobScheduler, RunEvents, VmafJob
    from vmaf_app.core.metrics import metric_definition
    from vmaf_app.core.model_select import resolve_model
    from vmaf_app.core.models import GpuVendor, VmafOptions
    from vmaf_app.core.settings import Settings
    from vmaf_app.core.vmaf_runner import validate_video_pair

    out = {"repo": str(repo), "version": vmaf_app.__version__, "metrics": args.metrics, "seconds": args.seconds,
           "vendor": args.vendor, "logical_cpus": os.cpu_count()}
    settings = Settings.load()
    # The app's start-up (main.main): the GPU backend and its probes, done
    # by the time a video is added -- here before the clock starts.
    perceptual_vship.set_vship_backend(settings.gpu_backend)
    if hasattr(vmaf_cuda, "set_gpu_backend"):
        vmaf_cuda.set_gpu_backend(settings.gpu_backend)
    tools = check_tools()
    out["ffmpeg"] = f"{format_version(tools.ffmpeg.version)} {tools.ffmpeg.path}" if tools.ffmpeg else None
    device, reason = perceptual_vship.detect_vship_device()
    out["probes"] = {"vship": str(device) if device is not None else f"none: {reason}",
                     "gpu_vmaf": list(vmaf_cuda.gpu_vmaf_available())}
    try:  # VMAF v1 on the GPU: since v2.0
        from vmaf_app.core import vmaf_v1_gpu
        out["probes"]["vmaf_v1_gpu"] = list(vmaf_v1_gpu.available())
    except (ImportError, AttributeError):
        pass

    source, distorted = probe_video(Path(args.ref)), probe_video(Path(args.dist))
    metrics = args.metrics.split(",")
    options = VmafOptions(gpu_decode=settings.default_gpu_decode, vmaf_on_gpu=settings.default_vmaf_on_gpu,
                          duration_limit=float(args.seconds), compute_vmaf=False)
    if args.vendor != "auto":
        options.gpu_vendor = GpuVendor(args.vendor)
    for key in metrics:
        if metric_definition(key).ffmpeg_binding is not None:
            options.set_metric_enabled(key, True)
    validate_video_pair(source, distorted, options)
    size = analysis_dimensions(source, distorted, options, None, None)
    model = resolve_model(options, *size) if options.compute_vmaf else ""
    backends = {key: "cpu" if args.perceptual_cpu else getattr(settings, f"default_{key}_backend", "gpu")
                for key in ("ssimulacra2", "butteraugli")}
    cvvdp = default_settings(settings.cvvdp_presets, settings.cvvdp_default_preset)
    jobs = [VmafJob(source, distorted, replace(options, model=model), label=Path(args.dist).stem,
                    result_distorted_path=Path(args.dist), metric_keys=tuple(metrics), metric_backends=backends,
                    cvvdp=cvvdp)]
    timeline, results, failures = [], {}, []
    started = [0.0]

    def progress(_index, cur, total, _fps):
        timeline.append((time.perf_counter() - started[0], int(cur), int(total), job.cpu_seconds()))

    def finished(_index, result):
        results["result"] = result

    def failed(_index, message, tail):
        failures.append({"message": message, "stderr": tail[-2000:]})

    def partly(_index, result, message, tail, why):
        results["result"] = result
        failures.append({"message": message, "stderr": tail[-2000:], "metrics": {k: str(v) for k, v in why.items()}})

    events = RunEvents(progress=progress, job_finished=finished, job_failed=failed, job_partially_failed=partly)
    scheduler = JobScheduler(jobs, settings.parallel_jobs, events, gpu_metrics_together=settings.gpu_metrics_together)
    cpu0, started[0] = job.cpu_seconds(), time.perf_counter()
    scheduler.run()
    out["wall"], out["cpu_run"] = time.perf_counter() - started[0], job.cpu_seconds() - cpu0
    out["peak_memory"], out["timeline"], out["failures"] = job.peak_memory(), timeline, failures
    result = results.get("result")
    if result is not None:
        scores, provenance = {}, {}
        for key in result.metric_results.keys():  # noqa: SIM118 -- a MetricResultSet, not a dict
            p = result.metric_results.get(key).provenance
            provenance[key] = {"implementation": p.implementation, "backend": p.compute_backend,
                               "version": p.implementation_version}
            values = np.asarray(result.metric_results.frame(key).values, dtype=float)
            scores[key] = [None if not np.isfinite(v) else float(v) for v in values]
        out.update(scores=scores, provenance=provenance)
    Path(args.out).write_text(json.dumps(out))


def steady_state(data: dict) -> dict | None:
    """Frames a second, and CPU use as Task Manager shows it (the average
    share of the PC's logical processors), from 10% to 90% of the run's
    frames: start-up and the end of the run left out."""
    timeline = data.get("timeline") or []
    if not timeline:
        return None
    total = max(row[2] for row in timeline)
    a, b = _at(timeline, 0.1 * total), _at(timeline, 0.9 * total)
    if not a or not b or b[0] <= a[0] or b[1] <= a[1]:
        return None
    seconds = b[0] - a[0]
    return {"fps": (b[1] - a[1]) / seconds, "seconds": seconds, "frames": total,
            "cpu_percent": 100 * (b[2] - a[2]) / seconds / data.get("logical_cpus", os.cpu_count())}


def _at(timeline, frame: float):
    """(time, frame, cpu) where the run reached `frame`, between its reports;
    the first report, where that is past `frame` already."""
    previous = None
    for t, cur, _total, cpu in timeline:
        if cur >= frame:
            if previous is None:
                return t, cur, cpu
            pt, pcur, pcpu = previous
            if cur == pcur:
                return t, cur, cpu
            share = (frame - pcur) / (cur - pcur)
            return pt + share * (t - pt), frame, pcpu + share * (cpu - pcpu)
        previous = (t, cur, cpu)
    return None


# ------------------------------------------------------------- the PC's state

class _CounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _CounterValue)]


class GpuEngines:
    """Windows' GPU Engine counters: each process's busiest engine, on every
    GPU, between two samples. Read by their English names, so a localised
    Windows reads them too."""

    def __init__(self) -> None:
        pdh = ctypes.WinDLL("pdh")
        pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t,
                                              ctypes.POINTER(wintypes.HANDLE)]
        pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        for function in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData,
                         pdh.PdhGetFormattedCounterArrayW):
            function.restype = wintypes.DWORD
        self._pdh, self._query, self._counter = pdh, wintypes.HANDLE(), wintypes.HANDLE()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) or pdh.PdhAddEnglishCounterW(
                self._query, r"\GPU Engine(*)\Utilization Percentage", 0, ctypes.byref(self._counter)):
            raise OSError("Windows' GPU Engine counters cannot be read")
        pdh.PdhCollectQueryData(self._query)  # a rate takes two samples

    def sample(self) -> dict[int, float]:
        """Process id -> its busiest engine since the last sample, in %."""
        self._pdh.PdhCollectQueryData(self._query)
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        self._pdh.PdhGetFormattedCounterArrayW(self._counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return {}
        buffer = (ctypes.c_byte * size.value)()
        if self._pdh.PdhGetFormattedCounterArrayW(self._counter, 0x200, ctypes.byref(size), ctypes.byref(count),
                                                  buffer):
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
        busiest: dict[int, float] = {}
        for i in range(count.value):
            match = re.match(r"pid_(\d+)_", items[i].szName or "")
            if match and items[i].FmtValue.CStatus in (0, 1):  # valid, or new data
                pid = int(match.group(1))
                busiest[pid] = max(busiest.get(pid, 0.0), items[i].FmtValue.doubleValue)
        return busiest


def _mine() -> set[int]:
    """This process, what started it (the venv's python.exe is a launcher)
    and everything it started."""
    import psutil
    me = psutil.Process()
    return {me.pid} | {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}


def _cpu_times() -> dict[int, tuple[str, float]]:
    import psutil
    times = {}
    for p in psutil.process_iter(["name", "cpu_times"]):
        if p.info["cpu_times"] is not None:
            times[p.pid] = (p.info["name"] or "?", p.info["cpu_times"].user + p.info["cpu_times"].system)
    return times


#: Windows' own: the System process carries drivers' kernel work, the run's
#: GPU driver's included, and MemCompression compresses memory the runs use.
_KERNEL = {"system idle process", "system", "registry", "memcompression", "memory compression", "secure system"}


def others_busy(engines: GpuEngines, before: dict, seconds: float) -> tuple[list[str], dict]:
    """Other processes that used a GPU engine or a CPU thread since `before`
    (_cpu_times) -- an encoder another project runs here, say -- and the
    CPU times now, for the next call."""
    mine, now, gpu = _mine(), _cpu_times(), engines.sample()
    busy = [f"{now.get(pid, ('?',))[0]} {pid} (GPU {use:.0f}%)" for pid, use in gpu.items()
            if pid not in mine and use > BUSY_GPU_PERCENT]
    busy += [f"{name} {pid} (CPU {(spent - before[pid][1]) / seconds:.1f} threads)"
             for pid, (name, spent) in now.items()
             if pid not in mine and pid in before and name.lower() not in _KERNEL
             and (spent - before[pid][1]) / seconds > BUSY_CPU_SHARE]
    return busy, now


def wait_until_quiet(engines: GpuEngines, log) -> float:
    """Until the CPU is under 15% busy and no other process uses a GPU or a
    CPU thread: the seconds waited."""
    import psutil
    started, last_note = time.monotonic(), -60.0
    before = _cpu_times()
    while True:
        cpu = psutil.cpu_percent(interval=3)
        busy, before = others_busy(engines, before, 3)
        if not busy and cpu < 15:
            return time.monotonic() - started
        waited = time.monotonic() - started
        if waited - last_note >= 60:
            last_note = waited
            log(f"  waiting for a quiet PC ({waited:.0f} s): CPU {cpu:.0f}%; {', '.join(busy[:4])}")


class Watch:
    """Other processes' use of a GPU or the CPU while a run goes, every 5
    seconds; a process busy in two samples in a row disturbed the run."""

    def __init__(self, engines: GpuEngines) -> None:
        self._engines, self.disturbed = engines, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        before, last = _cpu_times(), set()
        while not self._stop.wait(5):
            busy, before = others_busy(self._engines, before, 5)
            names = {entry.split(" (")[0] for entry in busy}
            self.disturbed += [entry for entry in busy if entry.split(" (")[0] in last]
            last = names

    def stop(self) -> list[str]:
        self._stop.set()
        self._thread.join()
        return sorted(set(self.disturbed))


class VramSampler:
    """nvidia-smi's memory used, every 100 ms, while a run goes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = open(path, "w")  # noqa: SIM115 -- open while nvidia-smi writes to it
        self._process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=self._file, stderr=subprocess.DEVNULL)

    def stop(self) -> tuple[int, int]:
        self._process.terminate()
        self._process.wait(10)
        self._file.close()
        values = [int(x) for x in self.path.read_text().split() if x.strip().isdigit()]
        self.path.unlink(missing_ok=True)
        return (values[0], max(values)) if values else (0, 0)


def read_files(paths) -> None:
    """Each video read once (the OS keeps it), so a run reads from memory."""
    for path in paths:
        with open(path, "rb") as f:
            while f.read(16 << 20):
                pass


# -------------------------------------------------------------------- driver

def _version(checkout: Path) -> str:
    text = (checkout / "vmaf_app" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r"__version__\s*=\s*[\"']([^\"']+)", text).group(1)


def _gpus() -> dict[str, str]:
    """The groups this PC can run -> the GPU maker each decodes with: nvidia
    with NVIDIA's GPU (the app's choice where it has one), gpu with the
    other one (NVIDIA hidden from the run), cpu with the app's choice."""
    sys.path.insert(0, str(HERE))
    from vmaf_app.core.gpu import detected_gpu_vendors
    from vmaf_app.core.models import GpuVendor
    vendors = detected_gpu_vendors()
    groups = {"cpu": "auto"}
    if GpuVendor.NVIDIA in vendors:
        groups["nvidia"] = "auto"
    other = next((vendor for vendor in vendors if vendor in (GpuVendor.INTEL, GpuVendor.AMD)), None)
    if other is not None:
        groups["gpu"] = other.value
    return groups


def _chosen(key: str, group: str, only: list[str]) -> bool:
    """Whether --only names this cell: by its group, or by its id's start."""
    return not only or any(item == group or key.startswith(item) for item in only)


def _frames(cell: dict, rate: float | None) -> int:
    fewest = FEWEST_CPU_FRAMES if cell.get("perceptual_cpu") else FEWEST_FRAMES
    wanted = rate * TARGET_SECONDS / 0.8 if rate else fewest  # the steady window is 80% of the frames
    return int(min(max(wanted, fewest), 230 * VIDEO_FPS))


def bench(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logfile = open(out / "bench.log", "a", encoding="utf-8")  # noqa: SIM115 -- the whole run's log

    def log(text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    checkouts = {"old": Path(args.old).resolve(), "new": Path(args.new).resolve()}
    versions = {label: _version(path) for label, path in checkouts.items()}
    videos = {name: Path(args.videos) / file for name, file in VIDEOS.items()}
    missing = [str(path) for path in videos.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing: {', '.join(missing)}")
    groups, only = _gpus(), [item.strip() for item in args.only.split(",") if item.strip()]
    engines = GpuEngines()
    profiles = Path(tempfile.gettempdir()) / "vml-bench-profiles"
    log(f"v{versions['old']} against v{versions['new']}; groups here: {groups}")
    todo = [(cell, video) for cell in CELLS if cell["group"] in groups for video in cell["videos"]
            if _chosen(cell_id(cell, video), cell["group"], only)]
    for n, (cell, video) in enumerate(todo):
        key = cell_id(cell, video)
        expected = EXPECTED_FPS.get(key, (None, None))
        for label in (["old", "new"] if n % 2 == 0 else ["new", "old"]):
            rate = expected[0 if label == "old" else 1]
            for stale in out.glob(f"{file_name(key)}__{label}__*.json"):  # an earlier bench's, into this folder
                stale.unlink()
            for attempt in (1, 2, 3):
                frames = args.frames or _frames(cell, rate)
                seconds = round(frames / VIDEO_FPS, 3)
                read_files([videos["reference"], videos[video]])
                waited = wait_until_quiet(engines, log)
                profile = profiles / f"{file_name(key)}-{label}-{attempt}"
                shutil.rmtree(profile, ignore_errors=True)
                profile.mkdir(parents=True)
                env = dict(_ENVIRONMENT, USERPROFILE=str(profile), PYTHONUTF8="1")
                for variable in HIDE_NVIDIA:
                    env.pop(variable, None)
                vendor = groups[cell["group"]]
                if cell["group"] == "gpu" and "nvidia" in groups:
                    env.update(HIDE_NVIDIA)
                target = out / f"{file_name(key)}__{label}__{attempt}.json"
                command = [args.python, str(Path(__file__).resolve()), "run", "--repo", str(checkouts[label]),
                           "--ref", str(videos["reference"]), "--dist", str(videos[video]),
                           "--metrics", ",".join(cell["metrics"]), "--seconds", str(seconds),
                           "--vendor", vendor, "--out", str(target)]
                if cell.get("perceptual_cpu"):
                    command.append("--perceptual-cpu")
                sampler = VramSampler(out / "vram.txt") if cell["group"] == "nvidia" else None
                watch = Watch(engines)
                proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=3600)
                disturbed = watch.stop()
                vram = sampler.stop() if sampler else (0, 0)
                shutil.rmtree(profile, ignore_errors=True)
                if proc.returncode != 0 or not target.exists():
                    log(f"FAILED {key} {label} (exit {proc.returncode}): {proc.stderr[-1500:]}")
                    break
                data = json.loads(target.read_text())
                steady = steady_state(data)
                data.update(cell=key, label=label, attempt=attempt, waited=waited, disturbed=disturbed,
                            vram=vram[1] - vram[0], stderr_tail=proc.stderr[-3000:])
                target.write_text(json.dumps(data))
                where = ", ".join(sorted({f"{k}:{v['backend']}" for k, v in data.get("provenance", {}).items()}))
                log(f"{key} v{versions[label]}: " + (f"{steady['fps']:.1f} fps over {steady['seconds']:.1f} s, "
                                                     f"CPU {steady['cpu_percent']:.0f}%" if steady else "no steady part")
                    + f", peak {data['peak_memory'] / 2**20:.0f} MB ({where})"
                    + (f"; disturbed by {', '.join(disturbed)}" if disturbed else "")
                    + (f"; FAILURES {data['failures']}" if data["failures"] else ""))
                if disturbed and attempt == 1:
                    continue  # once more, on a quiet PC
                if steady and steady["seconds"] < SHORTEST_SECONDS and attempt < 3 and not args.frames:
                    rate = steady["fps"]  # too short at the rate expected: again at this one
                    continue
                break
    if not only or "idle" in only:
        idle(checkouts, versions, engines, out, log, Path(args.python).with_name("pythonw.exe"))
    log("done")
    report_text = report_of(out)
    (out / "report.md").write_text(report_text, encoding="utf-8")
    print(report_text)


def idle(checkouts: dict, versions: dict, engines: GpuEngines, out: Path, log, pythonw: Path) -> None:
    """The window's private memory 10 s after it opens, each version once."""
    import psutil
    results = {}
    for label in ("old", "new"):
        wait_until_quiet(engines, log)
        profile = Path(tempfile.gettempdir()) / "vml-bench-profiles" / f"idle-{label}"
        shutil.rmtree(profile, ignore_errors=True)
        profile.mkdir(parents=True)
        env = dict(_ENVIRONMENT, USERPROFILE=str(profile), PYTHONUTF8="1")
        proc = subprocess.Popen([str(pythonw), "-m", "vmaf_app.main"], cwd=checkouts[label], env=env)
        time.sleep(10)
        root = psutil.Process(proc.pid)
        tree = [root, *root.children(recursive=True)]
        results[label] = sum(p.memory_full_info().private for p in tree if p.is_running()) / 2**20
        log(f"idle v{versions[label]}: {results[label]:.0f} MB private")
        for p in reversed(tree):  # this window's processes, by PID
            with contextlib.suppress(psutil.Error):
                p.kill()
        proc.wait(30)
        shutil.rmtree(profile, ignore_errors=True)
    (out / "idle.json").write_text(json.dumps({"versions": versions, "private_mb": results}))


# -------------------------------------------------------------------- report

def report_of(folder: Path) -> str:
    """Each cell's last run of each version: fps old -> new, CPU as Task
    Manager shows it, peak memory, and the scores of the frames both scored."""
    runs: dict = {}
    for path in sorted(folder.glob("*__*__*.json")):
        data = json.loads(path.read_text())
        if "cell" in data:
            runs.setdefault(data["cell"], {})[data["label"]] = data  # sorted: the last attempt wins
    versions = {}
    lines = ["| Cell | fps | CPU (Task Manager) | Peak memory | Scores, same frames |", "|---|--:|--:|--:|---|"]
    for cell, pair in runs.items():
        old, new = pair.get("old"), pair.get("new")
        if not old or not new:
            lines.append(f"| {cell} | missing a version | | | |")
            continue
        versions = {"old": old["version"], "new": new["version"]}
        a, b = steady_state(old), steady_state(new)
        if not a or not b:
            lines.append(f"| {cell} | no steady part | | | |")
            continue
        notes = [f"v{d['version']} disturbed by {', '.join(d['disturbed'])}" for d in (old, new) if d.get("disturbed")]
        lines.append(f"| {cell} | {a['fps']:.1f} → {b['fps']:.1f} (**{b['fps'] / a['fps']:.1f}x**) "
                     f"| {a['cpu_percent']:.0f}% → {b['cpu_percent']:.0f}% "
                     f"| {old['peak_memory'] / 2**20:,.0f} → {new['peak_memory'] / 2**20:,.0f} MB "
                     f"| {compare_scores(old, new)}{'; ' + '; '.join(notes) if notes else ''} |")
    head = f"v{versions.get('old', '?')} → v{versions.get('new', '?')}, steady state (start-up left out)\n\n"
    idle_file = folder / "idle.json"
    if idle_file.is_file():
        memory = json.loads(idle_file.read_text())["private_mb"]
        lines.append(f"\nThe window, 10 s after it opens: {memory['old']:,.0f} → {memory['new']:,.0f} MB private")
    return head + "\n".join(lines) + "\n"


def compare_scores(old: dict, new: dict) -> str:
    """Per metric, the frames both runs scored (each from the video's
    start): identical, or how far apart."""
    parts = []
    for key in sorted(set(old.get("scores", {})) | set(new.get("scores", {}))):
        x, y = old.get("scores", {}).get(key), new.get("scores", {}).get(key)
        if not x or not y:
            parts.append(f"{key} missing")
            continue
        n = min(len(x), len(y))
        pairs = list(zip(x[:n], y[:n], strict=True))
        if all(a == b for a, b in pairs):
            parts.append(f"{key} identical ({n})")
        else:
            gaps = [abs(a - b) for a, b in pairs if a is not None and b is not None]
            parts.append(f"{key} up to {max(gaps, default=float('nan')):.2g} apart ({n})")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    many = sub.add_parser("bench", help="every cell, both versions")
    many.add_argument("--old", required=True, help="the older release's checkout, e.g. a worktree of release/v1.4")
    many.add_argument("--new", default=str(HERE), help="this release's checkout (default: this one)")
    many.add_argument("--videos", required=True, help="the folder with the test clip and its encodes")
    many.add_argument("--out", required=True)
    many.add_argument("--only", default="",
                      help="comma-separated: groups (nvidia, gpu, cpu, idle) or cells, e.g. \"gpu/VMAF v1/hevc-4k\"")
    many.add_argument("--frames", type=int, default=0,
                      help="every run this many frames, to check the benchmark itself (not for numbers)")
    many.add_argument("--python", default=sys.executable, help="the Python the runs use")
    rep = sub.add_parser("report", help="the results in a folder, as a table")
    rep.add_argument("folder")
    one = sub.add_parser("run", help="one run (bench starts these)")
    one.add_argument("--repo", required=True)
    one.add_argument("--ref", required=True)
    one.add_argument("--dist", required=True)
    one.add_argument("--metrics", required=True)
    one.add_argument("--seconds", type=float, required=True)
    one.add_argument("--vendor", default="auto")
    one.add_argument("--perceptual-cpu", action="store_true")
    one.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "bench":
        bench(args)
    elif args.command == "report":
        print(report_of(Path(args.folder)))
    else:
        run_one(args)


if __name__ == "__main__":
    main()
