"""VMAF and VMAF NEG on an NVIDIA GPU: libvmaf's CUDA feature extractors
(VIF, ADM and motion) from the bundled tools/libvmaf/libvmaf.dll, built by
scripts/build_libvmaf_cuda.ps1 (libvmaf master with the pull requests that
fix its CUDA code, listed there).

Only those two scores: VMAF v1, PSNR, SSIM and XPSNR have no GPU code, and
stay in FFmpeg's libvmaf and xpsnr filters, so they are scored exactly as
before. A run that scores VMAF on the GPU still decodes each video once:
FFmpeg's graph ends in its CPU filters as before and, beside them, in two raw
outputs, one per video, written to named pipes that this module reads and
feeds to libvmaf frame pair by frame pair (see vmaf_runner._run_on_gpu).

Scores against the CPU (vmaf.exe of the same build, frame by frame): VIF and
ADM identical; motion within about 3e-5, from the order the CUDA kernel
rounds its blur in (libvmaf issue 1562, which nobody has fixed). VMAF and
NEG within 4e-5 per frame on 8-bit 1080p and 4K clips, 5e-6 on 4K 10-bit
Beekeeper -- 95.493080 against 95.493079 over 120 frames.

libvmaf runs in a process of its own (vmaf_app.core.isolated): a crash in it
or the NVIDIA driver ends that process, and VMAF is calculated on the CPU.
"""
from __future__ import annotations

import _winapi
import ctypes
import logging
import msvcrt
import os
import queue
import threading
import uuid
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "tools" / "libvmaf" / "libvmaf.dll"
#: What a GPU score records it was calculated with (its provenance).
LIBRARY_BUILD = "libvmaf cea2b4d8 + PRs 1477 1573 1583 1644 1612 1614 1647-1652 (CUDA)"

#: The app's built-in VMAF models -> libvmaf's names for them. A custom model
#: file is calculated on the CPU.
_GPU_MODELS = {"version=vmaf_v0.6.1": "vmaf_v0.6.1", "version=vmaf_4k_v0.6.1": "vmaf_4k_v0.6.1"}
_NEG_MODEL = "vmaf_v0.6.1neg"

_VMAF_PIX_FMT_YUV420P = 1
_VMAF_LOG_LEVEL_ERROR = 1
#: Pictures libvmaf holds in its pool: enough for the feeder to stay ahead of
#: the GPU without holding more host memory than it needs (25 MB each at
#: 4K 10-bit).
_PICTURES = 8
#: Frames each reader may hold ready before the feeder takes them.
_READ_AHEAD = 3
_PIPE_BYTES = 64 * 1024 * 1024


class VmafGpuError(RuntimeError):
    """VMAF could not be calculated on the GPU; the CPU calculates it."""


# ------------------------------------------------------------------ binding

class _Configuration(ctypes.Structure):
    _fields_ = [("log_level", ctypes.c_int), ("n_threads", ctypes.c_uint), ("n_subsample", ctypes.c_uint),
                ("cpumask", ctypes.c_uint64), ("gpumask", ctypes.c_uint64)]


class _Picture(ctypes.Structure):
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class _PictureParameters(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint), ("h", ctypes.c_uint), ("bpc", ctypes.c_uint), ("pix_fmt", ctypes.c_int)]


class _PictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_cnt", ctypes.c_uint)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("flags", ctypes.c_uint64)]


class _CudaConfiguration(ctypes.Structure):
    _fields_ = [("cu_ctx", ctypes.c_void_p)]


_library: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        for name, restype, argtypes in (
            ("vmaf_version", ctypes.c_char_p, []),
            ("vmaf_init", ctypes.c_int, [pointer, _Configuration]),
            ("vmaf_cuda_state_init", ctypes.c_int, [pointer, _CudaConfiguration]),
            ("vmaf_cuda_import_state", ctypes.c_int, [handle, handle]),
            ("vmaf_model_load", ctypes.c_int, [pointer, ctypes.POINTER(_ModelConfig), ctypes.c_char_p]),
            ("vmaf_use_features_from_model", ctypes.c_int, [handle, handle]),
            ("vmaf_preallocate_pictures", ctypes.c_int, [handle, _PictureConfiguration]),
            ("vmaf_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(_Picture)]),
            ("vmaf_read_pictures", ctypes.c_int,
             [handle, ctypes.POINTER(_Picture), ctypes.POINTER(_Picture), ctypes.c_uint]),
            ("vmaf_score_at_index", ctypes.c_int, [handle, handle, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
            ("vmaf_picture_unref", ctypes.c_int, [ctypes.POINTER(_Picture)]),
            ("vmaf_model_destroy", None, [handle]),
            ("vmaf_close", ctypes.c_int, [handle]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes
        _library = lib
    return _library


def _check(error: int, what: str) -> None:
    if error:
        raise VmafGpuError(f"{what} failed (libvmaf error {error})")


class GpuScorer:
    """One libvmaf context on the GPU, given frame pairs in order: the luma
    and chroma planes of 4:2:0 frames, packed as FFmpeg's rawvideo writes
    them, at `bit_depth` bits (16-bit little-endian samples above 8)."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1):
        self._lib = lib = _load()
        self._step = max(1, n_subsample)
        self._context = ctypes.c_void_p()
        self._models: dict[str, ctypes.c_void_p] = {}
        self._count = 0
        configuration = _Configuration(_VMAF_LOG_LEVEL_ERROR, 0, self._step, 0, 0)
        _check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
        try:
            state = ctypes.c_void_p()
            _check(lib.vmaf_cuda_state_init(ctypes.byref(state), _CudaConfiguration(None)), "Starting CUDA")
            _check(lib.vmaf_cuda_import_state(self._context, state), "Starting CUDA")
            for name, version in models.items():
                model = ctypes.c_void_p()
                config = _ModelConfig(name.encode(), 0)
                _check(lib.vmaf_model_load(ctypes.byref(model), ctypes.byref(config), version.encode()),
                       f"Loading the {version} model")
                self._models[name] = model
                _check(lib.vmaf_use_features_from_model(self._context, model), f"Setting up {version}")
            pictures = _PictureConfiguration(
                _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV420P), _PICTURES)
            _check(lib.vmaf_preallocate_pictures(self._context, pictures), "Allocating pictures")
        except BaseException:
            self.close()
            raise
        sample = 1 if bit_depth <= 8 else 2
        chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
        #: (plane offset in a frame, rows, bytes per row) for Y, U and V.
        self._planes = [(0, height, width * sample)]
        offset = width * height * sample
        for _ in range(2):
            self._planes.append((offset, chroma_h, chroma_w * sample))
            offset += chroma_w * chroma_h * sample
        self.frame_bytes = offset

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before)."""
        ref, dist = _Picture(), _Picture()
        _check(self._lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            _check(self._lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            self._fill(ref, reference)
            self._fill(dist, distorted)
        except BaseException:
            # Pictures taken from the pool and never handed over keep
            # vmaf_close waiting for them.
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            self._lib.vmaf_picture_unref(ctypes.byref(dist))
            raise
        # libvmaf takes both pictures, also when it fails (pull request 1652).
        _check(self._lib.vmaf_read_pictures(self._context, ctypes.byref(ref), ctypes.byref(dist), self._count),
               f"Scoring frame {self._count}")
        self._count += 1

    def _fill(self, picture: _Picture, frame: bytearray) -> None:
        source = np.frombuffer(frame, dtype=np.uint8)
        for plane, (offset, rows, row_bytes) in enumerate(self._planes):
            stride = picture.stride[plane]
            target = np.ctypeslib.as_array(
                ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)), shape=(rows, stride))
            target[:, :row_bytes] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The frame numbers scored (every n_subsample-th) and each model's
        scores for them, as libvmaf's JSON log rounds them: six decimals.
        Nothing when no pair came (the caller refuses that)."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {name: np.zeros(0) for name in self._models}
        _check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        scores = {}
        value = ctypes.c_double()
        for name, model in self._models.items():
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                _check(self._lib.vmaf_score_at_index(self._context, model, ctypes.byref(value), int(frame)),
                       f"Reading {name} for frame {frame}")
                column[slot] = float(f"{value.value:.6f}")
            scores[name] = column
        return frames, scores

    def close(self) -> None:
        for model in self._models.values():
            self._lib.vmaf_model_destroy(model)
        self._models.clear()
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


# ------------------------------------------------------------- frame feed

class _PipeReader(threading.Thread):
    """Reads one of FFmpeg's raw outputs, a named pipe this process serves,
    frame by frame into a few reusable buffers."""

    def __init__(self, name: str, frame_bytes: int) -> None:
        super().__init__(name=f"vmaf-gpu-{name}", daemon=True)
        self.path = rf"\\.\pipe\vml-vmaf-{os.getpid()}-{uuid.uuid4().hex[:12]}-{name}"
        self._handle = _winapi.CreateNamedPipe(
            self.path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,  # byte mode: PIPE_TYPE_BYTE is 0
            1, _PIPE_BYTES, _PIPE_BYTES, 0, _winapi.NULL)
        self.frames: queue.Queue[bytearray | None] = queue.Queue()
        self.free: queue.Queue[bytearray] = queue.Queue()
        for _ in range(_READ_AHEAD):
            self.free.put(bytearray(frame_bytes))
        self.error: BaseException | None = None
        self._stopped = threading.Event()

    def run(self) -> None:
        try:
            try:
                _winapi.ConnectNamedPipe(self._handle, _winapi.NULL)
            except OSError as error:
                if error.winerror != 535:  # ERROR_PIPE_CONNECTED: FFmpeg was first
                    raise
            if self._stopped.is_set():
                return
            fd = msvcrt.open_osfhandle(self._handle, os.O_RDONLY)
            self._handle = None
            with open(fd, "rb", buffering=0) as stream:
                while not self._stopped.is_set():
                    buffer = self.free.get()
                    if buffer is None or not _read_frame(stream, buffer):
                        break
                    self.frames.put(buffer)
                # The feeder may stop early (the shorter video ended):
                # FFmpeg still writes the rest of this one, read and dropped.
                spare = bytearray(1024 * 1024)
                while not self._stopped.is_set() and stream.readinto(spare):
                    pass
        except BaseException as error:  # reported by the attempt
            self.error = error
        finally:
            if self._handle is not None:
                _winapi.CloseHandle(self._handle)
                self._handle = None
            self.frames.put(None)

    def stop(self) -> None:
        """Ends the thread, also while it waits for FFmpeg to open the pipe."""
        self._stopped.set()
        self.free.put(None)
        self.release_if_unconnected()

    def release_if_unconnected(self) -> None:
        """Once FFmpeg has ended: a pipe it never opened -- an output that got
        no frame is never opened -- would be waited for forever. Connecting
        to it here ends the wait, and the reader finds the pipe empty. A pipe
        FFmpeg did open refuses a second client, and is read to its end:
        frames can still be in it after FFmpeg has gone."""
        if self.is_alive():
            try:
                with open(self.path, "wb"):
                    pass
            except OSError:
                pass  # FFmpeg's already, or the pipe is gone


def _read_frame(stream, buffer: bytearray) -> bool:
    """Fills `buffer`; False at the end of the stream."""
    view = memoryview(buffer)
    filled = 0
    while filled < len(buffer):
        count = stream.readinto(view[filled:])
        if not count:
            if filled:
                raise VmafGpuError(f"FFmpeg's raw output ended inside a frame ({filled} of {len(buffer)} bytes)")
            return False
        filled += count
    return True


class GpuAttempt:
    """One FFmpeg run's GPU half: the two pipes FFmpeg writes the compared
    frames to, and the threads that read them and feed libvmaf."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int):
        self._scorer = GpuScorer(width, height, bit_depth, models, n_subsample)
        self.distorted = _PipeReader("distorted", self._scorer.frame_bytes)
        self.reference = _PipeReader("reference", self._scorer.frame_bytes)
        self.error: BaseException | None = None
        self._result = None
        self._feeder = threading.Thread(target=self._feed, name="vmaf-gpu-feed", daemon=True)
        self.distorted.start()
        self.reference.start()
        self._feeder.start()

    def _feed(self) -> None:
        try:
            while True:
                distorted = self.distorted.frames.get()
                reference = self.reference.frames.get()
                if distorted is None or reference is None:
                    # The shorter video has ended (FFmpeg's shortest=1).
                    for frame in (distorted, reference):
                        if frame is not None:
                            (self.distorted if frame is distorted else self.reference).free.put(frame)
                    break
                self._scorer.add(reference, distorted)
                self.distorted.free.put(distorted)
                self.reference.free.put(reference)
            for reader in (self.distorted, self.reference):
                reader.free.put(None)
                reader.join()
                if reader.error is not None:
                    raise reader.error
            self._result = self._scorer.finish()
        except BaseException as error:  # finish() raises it in the run
            self.error = error
            self.stop()

    def stop(self) -> None:
        self.distorted.stop()
        self.reference.stop()

    def finish(self, ffmpeg_succeeded: bool) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """The scores, once FFmpeg has ended. None when FFmpeg failed (the
        caller retries or gives up); VmafGpuError when libvmaf failed."""
        if not ffmpeg_succeeded:
            self.stop()
        else:
            self.distorted.release_if_unconnected()
            self.reference.release_if_unconnected()
        self._feeder.join()
        try:
            if self.error is not None:
                raise VmafGpuError(f"GPU VMAF failed: {self.error}") from self.error
            return self._result if ffmpeg_succeeded else None
        finally:
            self._scorer.close()

    def output_args(self, duration_limit: float) -> list[str]:
        """FFmpeg's two raw outputs, for the graph's [vmaf_dist] and
        [vmaf_ref]: every frame as it comes (no frame rate conversion)."""
        args = []
        for label, reader in (("vmaf_dist", self.distorted), ("vmaf_ref", self.reference)):
            args += ["-map", f"[{label}]", "-fps_mode", "passthrough"]
            if duration_limit > 0:
                args += ["-t", f"{duration_limit:.3f}"]
            args += ["-f", "rawvideo", reader.path]
        return args


# ------------------------------------------------------------ availability

def gpu_models(compute_vmaf: bool, compute_vmaf_neg: bool, model: str) -> dict[str, str] | None:
    """{log name: libvmaf model} for a run's VMAF and NEG, or None when the
    GPU cannot score them: neither is requested, or VMAF uses a custom
    model file."""
    models = {}
    if compute_vmaf:
        version = _GPU_MODELS.get(model)
        if version is None:
            return None
        models["vmaf"] = version
    if compute_vmaf_neg:
        models["vmaf_neg"] = _NEG_MODEL
    return models or None


def probe() -> tuple[bool, str]:
    """Whether libvmaf can score on this GPU: starts CUDA and scores three
    small frame pairs (the kernels load only then; a GPU older than the
    build supports fails here). Run in a process of its own."""
    if not LIBRARY_PATH.is_file():
        return False, "libvmaf with CUDA is not bundled"
    try:
        lib = _load()
        version = (lib.vmaf_version() or b"").decode()
        width, height = 256, 144
        scorer = GpuScorer(width, height, 8, {"vmaf": "vmaf_v0.6.1"})
        try:
            rng = np.random.default_rng(1)
            base = rng.integers(16, 235, scorer.frame_bytes, dtype=np.uint8)
            for shift in range(3):
                reference = bytearray(np.roll(base, shift).tobytes())
                distorted = bytearray(np.clip(np.roll(base, shift).astype(np.int16) + 3, 0, 255).astype(np.uint8))
                scorer.add(reference, distorted)
            _frames, scores = scorer.finish()
        finally:
            scorer.close()
        if not np.all(np.isfinite(scores["vmaf"])):
            return False, "libvmaf's GPU test scored nothing"
        return True, f"libvmaf {version}"
    except (OSError, VmafGpuError) as error:
        return False, str(error)


# --------------------------------------------------------- the app's switch

_enabled = True
_PROBE_LOCK = threading.Lock()
_probed: tuple[bool, str] | None = None


def set_gpu_vmaf(enabled: bool) -> None:
    """Settings > GPU metrics > "Calculate VMAF on an NVIDIA GPU"."""
    global _enabled
    _enabled = enabled


def gpu_vmaf_enabled() -> bool:
    return _enabled


def gpu_vmaf_available() -> tuple[bool, str]:
    """Whether libvmaf scores on this PC's GPU, and with what (or why not).
    Probed once, in a process of its own, and only with an NVIDIA GPU."""
    global _probed
    with _PROBE_LOCK:
        if _probed is None:
            _probed = _probe_once()
            available, text = _probed
            if available:
                _log.info("VMAF on the GPU: %s", text)
            else:
                _log.info("VMAF on the GPU unavailable: %s", text)
        return _probed


def _probe_once() -> tuple[bool, str]:
    from vmaf_app.core.gpu import detected_gpu_vendors
    from vmaf_app.core.isolated import IsolatedCrashError, run_isolated
    from vmaf_app.core.models import GpuVendor

    if GpuVendor.NVIDIA not in detected_gpu_vendors():
        return False, "no NVIDIA GPU"
    try:
        return run_isolated(probe, what="libvmaf's GPU probe")
    except IsolatedCrashError as error:
        return False, str(error)


def start_gpu_vmaf_probe() -> None:
    """At startup, in the background: done by the time a run needs it."""
    threading.Thread(target=gpu_vmaf_available, name="vmaf-gpu-probe", daemon=True).start()


def scores_on_gpu(compute_vmaf: bool, compute_vmaf_neg: bool, model: str,
                  enabled: bool | None = None) -> dict[str, str] | None:
    """The models a run scores on the GPU (gpu_models), or None when its
    VMAF is calculated on the CPU: switched off, a custom model, or no GPU
    libvmaf can use. `enabled`: the setting as a run took it when it started
    (None: as it is now)."""
    if not (_enabled if enabled is None else enabled):
        return None
    models = gpu_models(compute_vmaf, compute_vmaf_neg, model)
    if models is None or not gpu_vmaf_available()[0]:
        return None
    return models
