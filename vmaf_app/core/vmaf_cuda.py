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

Where NVIDIA's decoder decodes both videos, VMAF is scored without FFmpeg's
decode (score_decoded): the videos are decoded in libvmaf's process
(nvdec_frames), scaled and widened on the GPU to the size and depth they
are compared at, the frames paired as libvmaf's filter pairs them
(frame_sync), and each frame's luma -- all VMAF reads -- copied on the GPU
into a picture of libvmaf's on the GPU. No frame crosses to system memory;
FFmpeg only copies the compressed streams out of their containers.

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
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from vmaf_app.core import nvdec_frames
from vmaf_app.core.frame_sync import frame_pairs
from vmaf_app.core.models import CropBox, VideoInfo

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
#: The CUDA device GPU VMAF runs on, its decoders included: CUDA's first,
#: which by its default order (CUDA_DEVICE_ORDER=FASTEST_FIRST) is the
#: fastest NVIDIA GPU. It is the one FFmpeg's -hwaccel cuda takes, and the
#: one Vship's CUDA build picks (the first discrete GPU it lists;
#: perceptual_vship._probe_vship_device), so on a PC with two NVIDIA GPUs
#: every GPU metric and decode still lands on the same card. The app offers
#: no choice of GPU.
_GPU = 0
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


class _CudaPictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_prealloc_method", ctypes.c_int)]


#: VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE: libvmaf's own pictures in
#: GPU memory, which the decoded frames are copied into on the GPU.
_PREALLOCATE_ON_DEVICE = 1


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
            ("vmaf_cuda_preallocate_pictures", ctypes.c_int, [handle, _CudaPictureConfiguration]),
            ("vmaf_cuda_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(_Picture)]),
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
    them, at `bit_depth` bits (16-bit little-endian samples above 8) --
    or, with `on_device`, pictures in GPU memory that add_on_device has
    filled (their luma: VMAF reads nothing else)."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 on_device: bool = False):
        self._lib = lib = _load()
        self._on_device = on_device
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
            parameters = _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV420P)
            if on_device:
                pictures = _CudaPictureConfiguration(parameters, _PREALLOCATE_ON_DEVICE)
                _check(lib.vmaf_cuda_preallocate_pictures(self._context, pictures), "Allocating pictures")
            else:
                _check(lib.vmaf_preallocate_pictures(self._context, _PictureConfiguration(parameters, _PICTURES)),
                       "Allocating pictures")
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
        self._add(lambda picture: self._fill(picture, reference), lambda picture: self._fill(picture, distorted))

    def add_on_device(self, reference: Callable[[int, int], None], distorted: Callable[[int, int], None]) -> None:
        """Scores one more pair of pictures in GPU memory: `reference` and
        `distorted` are each given a picture's luma plane (address, pitch)
        to fill, and return once it is filled."""
        self._add(lambda picture: reference(picture.data[0], picture.stride[0]),
                  lambda picture: distorted(picture.data[0], picture.stride[0]))

    def _add(self, fill_reference, fill_distorted) -> None:
        fetch = (self._lib.vmaf_cuda_fetch_preallocated_picture if self._on_device
                 else self._lib.vmaf_fetch_preallocated_picture)
        ref, dist = _Picture(), _Picture()
        _check(fetch(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            _check(fetch(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            fill_reference(ref)
            fill_distorted(dist)
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


# ------------------------------------------------- videos decoded here

def score_decoded(
    source: VideoInfo, distorted: VideoInfo, source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int,
    duration_limit: str | None, total_frames: int, scale_algorithm: str = "bicubic",
    on_progress: Callable[[int, int, float], None] | None = None,
    check_cancel: Callable[[], None] = lambda: None,
    process_handle=None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """VMAF and NEG with both videos decoded by NVIDIA's decoder in this
    process: what GpuAttempt scores from FFmpeg's raw outputs, the same
    frames compared.

    FFmpeg's graph for those outputs -- the decoded frames cropped, scaled to
    the size compared at, converted to its depth, paired by overlay with
    libvmaf's frame sync, cut by each output's -t -- is done here:
    nvdec_frames crops as FFmpeg's crop filter does, shifts 10-bit samples and
    widens 8-bit ones as FFmpeg converts them, and scales on the GPU with the
    same filter (not to the sample: the user decided a comparison scaled any
    way is the same one); frame_sync.frame_pairs pairs as the overlay does,
    and a pair is scored while the test frame's time from the first is below
    `duration_limit` (the outputs' -t, as text) in the test video's time
    base, as FFmpeg's trim filter cuts.

    NvdecUnavailableError when the videos are not decoded here (FFmpeg then
    decodes them, as before): another decoder, a codec or size it does not
    take. NvdecFailedError when decoding failed after the start -- the run
    is then made again through FFmpeg."""
    plans = []
    for info, crop in ((distorted, distorted_crop), (source, source_crop)):
        plan = nvdec_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height),
                                        algorithm=scale_algorithm)
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=nvdec_frames.WIDEN_REPEAT if full else nvdec_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise nvdec_frames.NvdecUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = nvdec_frames.decoder_supports(_GPU, plan)
        if not supported:
            raise nvdec_frames.NvdecUnavailableError(refusal)
        plans.append(plan)
    # The decoders first: the first to start CUDA sets its waits to sleep
    # rather than spin (nvdec_frames), and libvmaf's then do too.
    test = nvdec_frames.NvdecStream(distorted, plans[0], _GPU, pool=4, process_handle=process_handle)
    try:
        ref = nvdec_frames.NvdecStream(source, plans[1], _GPU, pool=4, process_handle=process_handle)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        scorer = GpuScorer(width, height, bit_depth, models, n_subsample, on_device=True)
        test.start()
        ref.start()
        test_base, ref_base = test.wait_time_base(), ref.wait_time_base()
        stop = nvdec_frames.duration_in(duration_limit, test_base) if duration_limit else None

        def puller(stream):
            def pull():
                while True:
                    check_cancel()
                    try:
                        return stream.next(100)
                    except TimeoutError:
                        continue
            return pull

        pairs = frame_pairs(puller(test), puller(ref), test_base, ref_base, test.release, ref.release)
        count = 0
        started = reported = time.perf_counter()
        try:
            for test_slot, ref_slot, when in pairs:
                if stop is not None and when >= stop:
                    break
                if ref_slot is None:
                    raise nvdec_frames.NvdecFailedError("the source has no frame for the test video's first")
                scorer.add_on_device(lambda address, pitch, slot=ref_slot: ref.copy_luma(slot, address, pitch),
                                     lambda address, pitch, slot=test_slot: test.copy_luma(slot, address, pitch))
                count += 1
                # Four times a second, about as often as FFmpeg's -progress:
                # each report crosses to the app's process.
                now = time.perf_counter()
                if on_progress is not None and now - reported >= 0.25:
                    reported = now
                    on_progress(count, total_frames, count / (now - started))
        finally:
            pairs.close()  # gives the frames it holds back
        # The pictures decoded so far are the packets', one each (the end of
        # a video was checked when it came).
        test.verify()
        ref.verify()
        result = scorer.finish()
        if on_progress is not None and count:
            elapsed = time.perf_counter() - started
            on_progress(count, total_frames, count / elapsed if elapsed > 0 else 0.0)
        return result
    finally:
        # The decoders before libvmaf: their threads stop first.
        test.close()
        ref.close()
        if scorer is not None:
            scorer.close()


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


# --------------------------------------------------------- whether a run uses it

_PROBE_LOCK = threading.Lock()
_probed: tuple[bool, str] | None = None
_probed_at = 0.0
#: A failed probe older than this is made again before the next run
#: (forget_failed_probe), as Vship's is.
FAILED_PROBE_RETRY_SECONDS = 60.0


def gpu_vmaf_available() -> tuple[bool, str]:
    """Whether libvmaf scores on this PC's GPU, and with what (or why not).
    Probed once, in a process of its own, and only with an NVIDIA GPU."""
    global _probed, _probed_at
    with _PROBE_LOCK:
        if _probed is None:
            _probed, _probed_at = _probe_once(), time.monotonic()
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
    except Exception as error:
        # Any failure is "not on this GPU": one that escaped here left the
        # probe unanswered and failed every video's setup with it, instead
        # of calculating VMAF on the CPU.
        _log.warning("libvmaf's GPU probe failed", exc_info=error)
        return False, f"the GPU probe failed: {error}"


def forget_failed_probe() -> None:
    """Before a run, off the UI thread: makes again a probe that failed a
    while ago. The failure may have been passing -- a driver being updated
    or restarted -- and was kept for the session, calculating VMAF on the
    CPU until the app restarted."""
    global _probed
    with _PROBE_LOCK:
        if (_probed is not None and not _probed[0]
                and time.monotonic() - _probed_at >= FAILED_PROBE_RETRY_SECONDS):
            _probed = None


def start_gpu_vmaf_probe() -> None:
    """At startup, in the background: done by the time a run needs it."""
    threading.Thread(target=gpu_vmaf_available, name="vmaf-gpu-probe", daemon=True).start()


def scores_on_gpu(compute_vmaf: bool, compute_vmaf_neg: bool, model: str,
                  enabled: bool = True, bit_depth: int = 8) -> dict[str, str] | None:
    """The models a run scores on the GPU (gpu_models), or None when its
    VMAF is calculated on the CPU: the video set to CPU (`enabled`, its
    VmafOptions.vmaf_on_gpu), a custom model, a comparison deeper than 10
    bits (`bit_depth`: the frames libvmaf pairs reach the GPU through
    FFmpeg's overlay, which holds 8 and 10 bits), or no GPU libvmaf can use."""
    if not enabled or bit_depth > 10:
        return None
    models = gpu_models(compute_vmaf, compute_vmaf_neg, model)
    if models is None or not gpu_vmaf_available()[0]:
        return None
    return models
