"""VMAF and VMAF NEG on the GPU.

On every GPU, NVIDIA's included, the scores come from the port of libvmaf's
CUDA feature extractors (VIF, ADM and motion) to Vulkan (vmaf_vulkan), which
gives the same scores. When Settings > GPU metrics > GPU backend is CUDA, or
Vulkan cannot run, an NVIDIA GPU uses those CUDA extractors themselves, in
the bundled tools/libvmaf/libvmaf.dll. That library is built from
libvmaf-fast (github.com/4KVCD/libvmaf-fast), a fork of libvmaf with the
fixes to its CUDA code merged. scripts/fetch_libvmaf_fast.ps1 installs a
release of it, and scripts/build_libvmaf_fast_local.ps1 installs a build of
its latest commit while the app is being developed. What is said below about
libvmaf holds for both.

This module scores only VMAF and NEG. VMAF v1 on the GPU is vmaf_v1_gpu's.
PSNR, SSIM and XPSNR have no GPU code, so they are scored on the CPU, by
this library's extractors (libvmaf_cpu.CpuScorer) or by FFmpeg's filters
(vmaf_runner._cpu_metrics_plan).

How the frames get here: a run that scores VMAF on the GPU still decodes
each video once. One FFmpeg per video writes the video's frames, as they
would reach libvmaf's filter, to a named pipe (_StreamReader). This module
pairs them as that filter does: frame N with frame N (frame_sync), or by
timestamp with VML_FRAME_PAIRING=timestamp. It then feeds libvmaf one pair
at a time (see vmaf_runner._run_on_gpu), uploading each frame's luma into a
libvmaf picture on the GPU (_HostUpload). With VML_VMAF_PIPES=canvas, a
single FFmpeg pairs the frames itself and writes two raw outputs
(_PipeReader), as every run once did; that costs about twice the CPU for the
same frames (see pairs_in_app).

Where NVIDIA's decoder can decode both videos, VMAF is scored without
FFmpeg decoding them (score_decoded). The videos are decoded in libvmaf's
process (gpu_frames) and scaled and widened on the GPU to the size and bit
depth they are compared at. The frames are paired as libvmaf's filter pairs
them (frame_sync), and each frame's luma (all VMAF reads) is copied on the
GPU into a libvmaf picture. No frame crosses to system memory: FFmpeg only
copies the compressed streams out of their containers.

How the GPU's scores compare with the CPU's (vmaf.exe of the same build,
frame by frame): VIF and ADM are identical. Motion is within about 3e-5,
because the CUDA kernel rounds its blur in a different order (libvmaf issue
1562, still open). VMAF and NEG are within 4e-5 per frame on 8-bit 1080p
and 4K clips, and within 5e-6 on a 4K 10-bit film (95.493080 against
95.493079 over 120 frames).

libvmaf runs in a process of its own (vmaf_app.core.isolated), so a crash in
it or in the NVIDIA driver ends only that process, and VMAF is then
calculated on the CPU.
"""
from __future__ import annotations

import _winapi
import ctypes
import functools
import logging
import msvcrt
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from ctypes import wintypes
from dataclasses import replace
from fractions import Fraction

import numpy as np

from vmaf_app.core import frame_sync, frame_sync_by_time, gpu_frames, libvmaf, libvmaf_cpu
from vmaf_app.core.models import CropBox, VideoInfo

_log = logging.getLogger(__name__)
#: ConnectNamedPipe's answers when the writer has already connected (535),
#: or has connected, written and closed (232).
_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA = 535, 232

#: The library a GPU score records it was calculated with (its provenance).
LIBRARY_BUILD = f"libvmaf-fast {libvmaf.LIBVMAF_FAST_VERSION} (CUDA)"

#: The app's built-in VMAF models, mapped to libvmaf's names for them. A
#: custom model file is calculated on the CPU.
_GPU_MODELS = {"version=vmaf_v0.6.1": "vmaf_v0.6.1", "version=vmaf_4k_v0.6.1": "vmaf_4k_v0.6.1"}
_NEG_MODEL = "vmaf_v0.6.1neg"

#: How many frames each reader may hold ready before the feeder takes them.
_READ_AHEAD = 3
_PIPE_BYTES = 64 * 1024 * 1024




# ------------------------------------------------- frames from system memory

class _Copy2D(ctypes.Structure):
    """CUDA's CUDA_MEMCPY2D: a copy of a rectangle of memory."""

    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_uint),
                ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t),
                ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t), ("dstMemoryType", ctypes.c_uint),
                ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64), ("dstArray", ctypes.c_void_p),
                ("dstPitch", ctypes.c_size_t),
                ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t)]


_CU_MEMORYTYPE_HOST, _CU_MEMORYTYPE_DEVICE = 1, 2
_CU_MEMHOSTALLOC_PORTABLE = 1
_CU_STREAM_NON_BLOCKING = 1

_cuda_library: ctypes.CDLL | None = None


def _cuda() -> ctypes.CDLL:
    """Loads CUDA's driver API (nvcuda.dll, part of the NVIDIA driver, which
    libvmaf loads too) once, and declares the types of the functions used."""
    global _cuda_library
    if _cuda_library is None:
        lib = ctypes.CDLL("nvcuda.dll")
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        for name, argtypes in (
            ("cuInit", [ctypes.c_uint]),
            ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
            ("cuDevicePrimaryCtxRetain", [pointer, ctypes.c_int]),
            ("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int]),
            ("cuCtxPushCurrent_v2", [handle]),
            ("cuCtxPopCurrent_v2", [pointer]),
            ("cuMemHostAlloc", [pointer, ctypes.c_size_t, ctypes.c_uint]),
            ("cuMemFreeHost", [handle]),
            ("cuStreamCreate", [pointer, ctypes.c_uint]),
            ("cuStreamDestroy_v2", [handle]),
            ("cuMemcpy2DAsync_v2", [ctypes.POINTER(_Copy2D), handle]),
            ("cuStreamSynchronize", [handle]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = ctypes.c_int, argtypes
        _cuda_library = lib
    return _cuda_library


def _cuda_check(error: int, what: str) -> None:
    """Raises libvmaf.LibvmafError if a CUDA call returned an error."""
    if error:
        raise libvmaf.LibvmafError(f"{what} failed (CUDA error {error})")


class _HostUpload:
    """Copies frames' luma from system memory into libvmaf's pictures on the
    GPU, through one page-locked buffer per side, allocated once, on a CUDA
    stream of its own.

    This is faster than libvmaf's own handling of frames in system memory (a
    pool of CPU pictures, each uploaded from pageable memory when it is
    read): about 330 4K 10-bit pairs a second on an RTX 5090, against about
    180. libvmaf's page-locked pictures (HOST_PINNED) are no faster, because
    each fetch allocates, locks and zeroes a new one, which takes 9 ms per
    4K pair.

    It works in libvmaf's CUDA context, the primary context of its GPU,
    which is made current around each use because the frames arrive on the
    caller's thread."""

    def __init__(self, width: int, height: int, sample: int):
        self._cu = cu = _cuda()
        self._row_bytes, self._rows = width * sample, height
        self.luma_bytes = self._row_bytes * height
        self._device = ctypes.c_int()
        self._context = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._staging = [ctypes.c_void_p(), ctypes.c_void_p()]
        _cuda_check(cu.cuInit(0), "Starting CUDA")
        _cuda_check(cu.cuDeviceGet(ctypes.byref(self._device), gpu_frames.CUDA_DEVICE), "Finding the GPU")
        _cuda_check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(self._context), self._device), "Taking the GPU")
        try:
            with self:
                # Non-blocking: waited for by itself, not with libvmaf's own
                # work (which cuCtxSynchronize would wait for too).
                _cuda_check(cu.cuStreamCreate(ctypes.byref(self._stream), _CU_STREAM_NON_BLOCKING),
                            "Starting the upload")
                for buffer in self._staging:
                    _cuda_check(cu.cuMemHostAlloc(ctypes.byref(buffer), self.luma_bytes, _CU_MEMHOSTALLOC_PORTABLE),
                                "Allocating the upload's memory")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> _HostUpload:
        """Makes libvmaf's CUDA context current on this thread."""
        _cuda_check(self._cu.cuCtxPushCurrent_v2(self._context), "Taking the GPU")
        return self

    def __exit__(self, *_exc) -> None:
        self._cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))

    def send(self, side: int, frame, address: int, pitch: int) -> None:
        """Starts copying `frame`'s luma to the GPU picture at `address`, whose
        rows are `pitch` bytes apart. Call wait() before sending on the same
        side again. Must be called inside `with self`."""
        source = np.frombuffer(frame, dtype=np.uint8)
        if source.size < self.luma_bytes:
            raise libvmaf.LibvmafError(f"a frame of {source.size} bytes, where its luma alone is {self.luma_bytes}")
        staging = self._staging[side].value
        ctypes.memmove(staging, source.ctypes.data, self.luma_bytes)
        copy = _Copy2D(srcMemoryType=_CU_MEMORYTYPE_HOST, srcHost=staging, srcPitch=self._row_bytes,
                       dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=address, dstPitch=pitch,
                       WidthInBytes=self._row_bytes, Height=self._rows)
        _cuda_check(self._cu.cuMemcpy2DAsync_v2(ctypes.byref(copy), self._stream), "Uploading a frame")

    def send_filled(self, side: int, fill: Callable[[int], None], address: int, pitch: int) -> None:
        """Like send, but `fill` writes the luma itself into the side's
        page-locked buffer (rows packed), instead of it being copied from a
        frame."""
        fill(self._staging[side].value)
        copy = _Copy2D(srcMemoryType=_CU_MEMORYTYPE_HOST, srcHost=self._staging[side].value, srcPitch=self._row_bytes,
                       dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=address, dstPitch=pitch,
                       WidthInBytes=self._row_bytes, Height=self._rows)
        _cuda_check(self._cu.cuMemcpy2DAsync_v2(ctypes.byref(copy), self._stream), "Uploading a frame")

    def wait(self) -> None:
        """Waits until the copies started have finished."""
        _cuda_check(self._cu.cuStreamSynchronize(self._stream), "Uploading a frame")

    def close(self) -> None:
        """Frees the buffers and the stream, and releases the GPU context."""
        if not self._context:
            return
        cu = self._cu
        if cu.cuCtxPushCurrent_v2(self._context) == 0:
            if self._stream:
                cu.cuStreamSynchronize(self._stream)
                cu.cuStreamDestroy_v2(self._stream)
            for buffer in self._staging:
                if buffer:
                    cu.cuMemFreeHost(buffer)
            cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        self._stream = ctypes.c_void_p()
        self._staging = [ctypes.c_void_p(), ctypes.c_void_p()]
        cu.cuDevicePrimaryCtxRelease_v2(self._device)
        self._context = ctypes.c_void_p()


class GpuScorer:
    """One libvmaf context on the GPU, scoring frame pairs given in order.

    add takes 4:2:0 frames in system memory, packed as FFmpeg's rawvideo
    writes them, at `bit_depth` bits (16-bit little-endian samples above 8
    bits); only their luma is uploaded, because VMAF reads nothing else.
    add_on_device takes pictures already in GPU memory instead. With
    `on_device`, only add_on_device is used, and nothing is allocated for
    uploads."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 on_device: bool = False):
        self._lib = lib = libvmaf.load()
        self._upload: _HostUpload | None = None
        self._step = max(1, n_subsample)
        self._context = ctypes.c_void_p()
        self._models: dict[str, ctypes.c_void_p] = {}
        self._count = 0
        configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, 0, self._step, 0, 0)
        libvmaf.check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
        try:
            state = ctypes.c_void_p()
            libvmaf.check(lib.vmaf_cuda_state_init(ctypes.byref(state), libvmaf.CudaConfiguration(None)), "Starting CUDA")
            libvmaf.check(lib.vmaf_cuda_import_state(self._context, state), "Starting CUDA")
            for name, version in models.items():
                model = ctypes.c_void_p()
                config = libvmaf.ModelConfig(name.encode(), 0)
                libvmaf.check(lib.vmaf_model_load(ctypes.byref(model), ctypes.byref(config), version.encode()),
                       f"Loading the {version} model")
                self._models[name] = model
                libvmaf.check(lib.vmaf_use_features_from_model(self._context, model), f"Setting up {version}")
            # libvmaf's pictures in GPU memory, filled either by the decoders
            # or by _HostUpload from frames in system memory.
            parameters = libvmaf.PictureParameters(width, height, bit_depth, libvmaf.PIX_FMT_YUV420P)
            pictures = libvmaf.CudaPictureConfiguration(parameters, libvmaf.PREALLOCATE_ON_DEVICE)
            libvmaf.check(lib.vmaf_cuda_preallocate_pictures(self._context, pictures), "Allocating pictures")
            sample = 1 if bit_depth <= 8 else 2
            if not on_device:
                self._upload = _HostUpload(width, height, sample)
        except BaseException:
            self.close()
            raise
        #: The size of a frame as FFmpeg writes it: the luma plane, then the
        #: chroma planes, whose size FFmpeg rounds up for an odd width or height.
        self.frame_bytes = (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * sample

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair; its frame index is the number of pairs given
        before it. The frames belong to the caller again when this returns."""
        upload = self._upload
        if upload is None:
            raise libvmaf.LibvmafError("this scorer takes pictures in GPU memory only")

        def fill_distorted(picture: libvmaf.AnyPicture) -> None:
            # Both copies run at the same time. The wait also covers the
            # reference's copy when this one could not be started.
            try:
                upload.send(1, distorted, picture.data[0], picture.stride[0])
            finally:
                upload.wait()

        with upload:
            self._add(lambda picture: upload.send(0, reference, picture.data[0], picture.stride[0]), fill_distorted)

    def add_on_device(self, reference: Callable[[int, int], None], distorted: Callable[[int, int], None]) -> None:
        """Scores one more pair of pictures in GPU memory. `reference` and
        `distorted` are each given a picture's luma plane (address, pitch) to
        fill, and return once it is filled."""
        self._add(lambda picture: reference(picture.data[0], picture.stride[0]),
                  lambda picture: distorted(picture.data[0], picture.stride[0]))

    def add_sides(self, reference: tuple[bool, Callable], distorted: tuple[bool, Callable]) -> None:
        """Scores one more pair, where each side is given as (on_device,
        fill). For a side on the device, fill(address, pitch) fills the
        picture's luma plane in GPU memory itself (as in add_on_device).
        Otherwise fill(address) writes the luma, rows packed, into the
        page-locked memory it is uploaded from (as add's frames are)."""
        upload = self._upload
        if upload is None and not (reference[0] and distorted[0]):
            raise libvmaf.LibvmafError("this scorer takes pictures in GPU memory only")

        def filler(side: int, on_device: bool, fill: Callable):
            if on_device:
                return lambda picture: fill(picture.data[0], picture.stride[0])
            return lambda picture: upload.send_filled(side, fill, picture.data[0], picture.stride[0])

        fill_reference, fill_distorted = filler(0, *reference), filler(1, *distorted)
        if upload is None:
            self._add(fill_reference, fill_distorted)
            return

        def fill_distorted_and_wait(picture: libvmaf.AnyPicture) -> None:
            try:
                fill_distorted(picture)
            finally:
                upload.wait()

        with upload:
            self._add(fill_reference, fill_distorted_and_wait)

    def _add(self, fill_reference, fill_distorted) -> None:
        """Takes two pictures from libvmaf's pool, has them filled, and hands
        them to libvmaf to score."""
        fetch = self._lib.vmaf_cuda_fetch_preallocated_picture
        ref, dist = libvmaf.Picture(), libvmaf.Picture()
        libvmaf.check(fetch(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            libvmaf.check(fetch(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            fill_reference(ref)
            fill_distorted(dist)
        except BaseException:
            # Pictures taken from the pool must be given back: vmaf_close
            # would wait for them forever.
            self._lib.vmaf_picture_unref(ctypes.byref(ref))
            self._lib.vmaf_picture_unref(ctypes.byref(dist))
            raise
        # libvmaf takes both pictures, even when it fails (pull request 1652).
        libvmaf.check(self._lib.vmaf_read_pictures(self._context, ctypes.byref(ref), ctypes.byref(dist), self._count),
               f"Scoring frame {self._count}")
        self._count += 1

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the frame numbers scored (every n_subsample-th frame) and
        each model's scores for them, rounded to six decimals as libvmaf's
        JSON log rounds them. Both are empty when no pair was given (the
        caller treats that as a failure)."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {name: np.zeros(0) for name in self._models}
        libvmaf.check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        scores = {}
        value = ctypes.c_double()
        for name, model in self._models.items():
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                libvmaf.check(self._lib.vmaf_score_at_index(self._context, model, ctypes.byref(value), int(frame)),
                       f"Reading {name} for frame {frame}")
                column[slot] = float(f"{value.value:.6f}")
            scores[name] = column
        return frames, scores

    def close(self) -> None:
        """Frees the upload, the models and the libvmaf context."""
        if self._upload is not None:
            self._upload.close()
            self._upload = None
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
        # Named _pipe, not _handle: since Python 3.13 Thread has a _handle
        # of its own, which start() needs ("'handle' must be a _ThreadHandle").
        self._pipe = _winapi.CreateNamedPipe(
            self.path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,  # byte mode: PIPE_TYPE_BYTE is 0
            1, _PIPE_BYTES, _PIPE_BYTES, 0, _winapi.NULL)
        self.frames: queue.Queue[bytearray | None] = queue.Queue()
        self.free: queue.Queue[bytearray] = queue.Queue()
        for _ in range(_READ_AHEAD):
            self.free.put(bytearray(frame_bytes))
        self.error: BaseException | None = None
        self._stopped = threading.Event()

    def run(self) -> None:
        """The reading thread: waits for FFmpeg to open the pipe, then reads
        frames into the free buffers until the pipe ends."""
        try:
            try:
                _winapi.ConnectNamedPipe(self._pipe, _winapi.NULL)
            except OSError as error:
                # ERROR_PIPE_CONNECTED: FFmpeg connected first. ERROR_NO_DATA:
                # FFmpeg connected, wrote and closed before this connected;
                # what it wrote is still there to read, so neither is an
                # error (treating them as one lost all the frames of a short
                # run on a busy PC).
                if error.winerror not in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
                    raise
            if self._stopped.is_set():
                return
            fd = msvcrt.open_osfhandle(self._pipe, os.O_RDONLY)
            self._pipe = None
            with open(fd, "rb", buffering=0) as stream:
                while not self._stopped.is_set():
                    buffer = self.free.get()
                    if buffer is None or not _read_frame(stream, buffer):
                        break
                    self.frames.put(buffer)
                # The feeder may stop early (when the shorter video ends), but
                # FFmpeg still writes the rest of this video: read and drop it.
                spare = bytearray(1024 * 1024)
                while not self._stopped.is_set() and stream.readinto(spare):
                    pass
        except BaseException as error:  # reported by the attempt
            self.error = error
        finally:
            if self._pipe is not None:
                _winapi.CloseHandle(self._pipe)
                self._pipe = None
            self.frames.put(None)

    def stop(self) -> None:
        """Ends the thread, even while it waits for FFmpeg to open the pipe."""
        self._stopped.set()
        self.free.put(None)
        self.release_if_unconnected()

    def release_if_unconnected(self) -> None:
        """Call once FFmpeg has ended. FFmpeg never opens an output that got
        no frame, and the reader would wait for such a pipe forever.
        Connecting to it here ends the wait, and the reader finds the pipe
        empty. A pipe FFmpeg did open refuses a second client and is read to
        its end, because frames can still be in it after FFmpeg has gone."""
        if self.is_alive():
            try:
                with open(self.path, "wb"):
                    pass
            except OSError:
                pass  # FFmpeg's already, or the pipe is gone


def _read_frame(stream, buffer: bytearray) -> bool:
    """Fills `buffer` from the stream; returns False at the end of the stream."""
    view = memoryview(buffer)
    filled = 0
    while filled < len(buffer):
        count = stream.readinto(view[filled:])
        if not count:
            if filled:
                raise libvmaf.LibvmafError(f"FFmpeg's raw output ended inside a frame ({filled} of {len(buffer)} bytes)")
            return False
        filled += count
    return True


# ------------------------------------------- each video's frames, paired here

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.PeekNamedPipe.restype = wintypes.BOOL
_kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                    ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_kernel32.ReadFile.restype = wintypes.BOOL
_kernel32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                               ctypes.c_void_p]
#: The environment variable that moves the pairing back into FFmpeg
#: ("canvas"), to compare the two ways.
PIPES_VARIABLE = "VML_VMAF_PIPES"


@functools.cache
def pairs_in_app() -> bool:
    """Whether the frames are paired here (frame_sync) from each video's own
    frames (_StreamReader), rather than by FFmpeg's overlay on a canvas of
    both videos (vmaf_runner._gpu_pairs_stage), which costs FFmpeg about four
    times the CPU. VML_VMAF_PIPES=canvas asks for the canvas.

    When pairing here, each video has an FFmpeg of its own. (In one FFmpeg,
    the faster-decoding video would fill its pipe while its frames waited
    here for the other's, and FFmpeg's single filter thread, blocked on that
    pipe, would pass no frame of the other video: a deadlock.)"""
    return os.environ.get(PIPES_VARIABLE, "").casefold() != "canvas"


class _StreamReader:
    """One of the two videos as it leaves FFmpeg's filter chain: its frames
    as a YUV4MPEG stream on a named pipe. pull() gives the next frame to
    frame_sync.frame_pairs.

    YUV4MPEG is used rather than rawvideo because FFmpeg writes a YUV4MPEG
    frame's rows straight to the pipe, while the rawvideo encoder first
    copies each frame into a packet it allocates. Each frame is left to
    gather in the pipe and is then taken in one read: read as it arrives, a
    4K frame comes in about 300 pieces, each waking this thread, which costs
    4 ms of CPU per frame instead of 2."""

    def __init__(self, name: str, frame_bytes: int) -> None:
        self.path = rf"\\.\pipe\vml-vmaf-{os.getpid()}-{uuid.uuid4().hex[:12]}-{name}"
        self.frame_bytes = frame_bytes
        self._pipe = _winapi.CreateNamedPipe(self.path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT, 1,
                                             _PIPE_BYTES, _PIPE_BYTES, 0, _winapi.NULL)
        self.free: queue.Queue[bytearray | None] = queue.Queue()
        # Buffers for the frame being paired (frame_pairs) and those read ahead.
        for _ in range(_READ_AHEAD + 1):
            self.free.put(bytearray(frame_bytes))
        self._pixels: queue.Queue[bytearray | None] = queue.Queue()
        self.error: BaseException | None = None
        self._stopped = threading.Event()
        self._writer_gone = threading.Event()
        self._thread = threading.Thread(target=self._read_frames, name=f"vmaf-gpu-{name}", daemon=True)

    def start(self) -> None:
        """Starts the reading thread."""
        self._thread.start()

    @staticmethod
    def _connect(pipe: int) -> None:
        """Waits for FFmpeg to open the pipe."""
        try:
            _winapi.ConnectNamedPipe(pipe, _winapi.NULL)
        except OSError as error:
            # As in _PipeReader.run: FFmpeg connected first, or connected,
            # wrote and closed already. Neither is an error.
            if error.winerror not in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
                raise

    def _gather(self, count: int) -> None:
        """Waits until `count` bytes are in the pipe (or as many as it can
        hold), or until FFmpeg has gone."""
        count = min(count, _PIPE_BYTES // 2)
        available = wintypes.DWORD()
        while not self._stopped.is_set():
            if not _kernel32.PeekNamedPipe(self._pipe, None, 0, None, ctypes.byref(available), None):
                return  # closed and empty: the next read reports that
            if available.value >= count or self._writer_gone.is_set():
                return
            time.sleep(0.001)

    def _read(self, buffer: bytearray) -> bool:
        """Fills `buffer` from the pipe; returns False at the end of the stream."""
        address = ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer))
        filled, got = 0, wintypes.DWORD()
        while filled < len(buffer):
            if not _kernel32.ReadFile(self._pipe, address + filled, len(buffer) - filled, ctypes.byref(got), None) \
                    or not got.value:
                if filled:
                    raise libvmaf.LibvmafError(f"FFmpeg's frames ended inside one ({filled} of {len(buffer)} bytes)")
                return False
            filled += got.value
        return True

    def _read_frames(self) -> None:
        """The reading thread: reads the YUV4MPEG stream's frames into the
        free buffers until the stream ends."""
        try:
            self._connect(self._pipe)
            if self._stopped.is_set():
                return
            # The stream is a header line, "YUV4MPEG2 W.. H.. ...\n", then for
            # each frame a "FRAME\n" line and the frame's planes, packed.
            header, byte = bytearray(), bytearray(1)
            while byte != b"\n" and len(header) < 1024 and self._read(byte):
                header += byte
            if header and not header.startswith(b"YUV4MPEG2 "):
                raise libvmaf.LibvmafError("FFmpeg's frames are not a YUV4MPEG stream")
            marker = bytearray(6)
            while header and not self._stopped.is_set():
                self._gather(len(marker) + self.frame_bytes)
                if not self._read(marker):
                    break
                if marker != b"FRAME\n":
                    raise libvmaf.LibvmafError("FFmpeg's frames are not where they were expected")
                buffer = self.free.get()
                if buffer is None:
                    break
                if not self._read(buffer):
                    raise libvmaf.LibvmafError("FFmpeg's frames ended after a frame's start")
                self._pixels.put(buffer)
            # The comparison may end before this video does: read and drop the rest.
            spare = bytearray(4 * 1024 * 1024)
            while not self._stopped.is_set():
                self._gather(len(spare))
                try:
                    if not self._read(spare):
                        break
                except libvmaf.LibvmafError:
                    break  # the tail, shorter than the buffer
        except BaseException as error:  # raised by pull()
            self.error = error
        finally:
            _winapi.CloseHandle(self._pipe)
            self._pixels.put(None)

    def pull(self) -> bytearray | None:
        """Returns the next frame, or None at the video's end."""
        buffer = self._pixels.get()
        if buffer is None:
            self._pixels.put(None)
            if self.error is not None:
                raise self.error
            return None
        return buffer

    def release(self, buffer: bytearray) -> None:
        """Gives a frame's buffer back for the next frame."""
        self.free.put(buffer)

    def drop_rest(self) -> None:
        """Says the comparison is over: whatever FFmpeg still writes is read
        and dropped."""
        self.free.put(None)

    def ffmpeg_ended(self) -> None:
        """Says FFmpeg has gone, so the reader stops waiting for more. FFmpeg
        never opens an output that got no frame, so a pipe it never opened
        is connected to here to end the wait for it (as in
        _PipeReader.release_if_unconnected)."""
        self._writer_gone.set()
        if self._thread.is_alive():
            try:
                with open(self.path, "wb"):
                    pass
            except OSError:
                pass  # FFmpeg's already, or the pipe is gone

    def stop(self) -> None:
        """Stops reading, even while waiting for FFmpeg."""
        self._stopped.set()
        self.free.put(None)
        self.ffmpeg_ended()

    def join(self) -> None:
        """Waits for the reading thread to end, and raises its error, if any."""
        self._thread.join()
        if self.error is not None:
            raise self.error


_LISTING_TB = re.compile(r"#tb 0: (\d+)/(\d+)")


class _TimedStreamReader(_StreamReader):
    """A _StreamReader that also reads each frame's timestamp, for pairing by
    timestamp (frame_sync.by_timestamp). The timestamps come as a framecrc
    listing on a second named pipe (with -enc_time_base filter, so they are
    the filters' own timestamps, which libvmaf's frame sync pairs by), and
    pull() gives the next (frame, timestamp) to frame_sync_by_time.frame_pairs."""

    def __init__(self, name: str, frame_bytes: int) -> None:
        super().__init__(name, frame_bytes)
        self.listing_path = self.path + "-times"
        self._listing = _winapi.CreateNamedPipe(self.listing_path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,
                                                1, 65536, 65536, 0, _winapi.NULL)
        self._stamps: queue.Queue[int | None] = queue.Queue()
        self.time_base: Fraction | None = None
        self._time_base_known = threading.Event()
        self._listing_thread = threading.Thread(target=self._read_listing, name=f"vmaf-gpu-{name}-times",
                                                daemon=True)

    def start(self) -> None:
        """Starts the frame and timestamp reading threads."""
        super().start()
        self._listing_thread.start()

    def _read_listing(self) -> None:
        """The timestamp thread: reads the time base and each frame's
        timestamp from FFmpeg's framecrc listing."""
        try:
            self._connect(self._listing)
            fd = msvcrt.open_osfhandle(self._listing, os.O_RDONLY)
            self._listing = None
            with open(fd, "rb") as listing:
                for raw in listing:
                    line = raw.decode("ascii", errors="replace").strip()
                    if line.startswith("#"):
                        match = _LISTING_TB.match(line)
                        if match:
                            self.time_base = Fraction(int(match.group(1)), int(match.group(2)))
                            self._time_base_known.set()
                        continue
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) >= 6 and fields[0].isdigit():
                        self._stamps.put(int(fields[2]))
        except BaseException as error:
            self.error = self.error or error
        finally:
            if self._listing is not None:
                _winapi.CloseHandle(self._listing)
            self._time_base_known.set()
            self._stamps.put(None)

    def wait_time_base(self) -> Fraction:
        """The time base of pull()'s timestamps, once FFmpeg has reported it
        (any time base for a video without frames)."""
        self._time_base_known.wait()
        return self.time_base or Fraction(1, 1000)

    def pull(self) -> tuple[bytearray, int] | None:
        """Returns the next frame and its timestamp, or None at the video's end."""
        buffer = super().pull()
        if buffer is None:
            return None
        stamp = self._stamps.get()
        if stamp is None:
            self._stamps.put(None)
            raise libvmaf.LibvmafError("FFmpeg gave a frame without its timestamp")
        return buffer, stamp

    def ffmpeg_ended(self) -> None:
        """As _StreamReader.ffmpeg_ended, for both pipes."""
        super().ffmpeg_ended()
        if self._listing_thread.is_alive():
            try:
                with open(self.listing_path, "wb"):
                    pass
            except OSError:
                pass  # FFmpeg connected already, or the pipe is gone

    def join(self) -> None:
        """Waits for both reading threads, and raises an error either had."""
        self._listing_thread.join()
        super().join()


class GpuAttempt:
    """The GPU side of one FFmpeg run: the pipes FFmpeg writes the two
    videos' frames to, and the threads that read them and feed the scorer.

    The scorer is libvmaf on the GPU (`backend` "cuda"), the Vulkan port of
    its features ("vulkan", on Vulkan's GPU number `device`), or libvmaf's
    CPU extractors ("cpu"). With `paired`, FFmpeg pairs the frames and
    writes them to two pipes of raw video (_PipeReader). By default (see
    pairs_in_app) each video comes as it is and is paired here
    (_StreamReader, or _TimedStreamReader when pairing by timestamp)."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int,
                 backend: str = "cuda", device: int | None = None, paired: bool | None = None,
                 threads: int = 0, frame_rate: int = 0):
        """With `backend` "cpu", `models` are libvmaf_cpu.CPU_FEATURES keys, scored by
        libvmaf_cpu.CpuScorer with `threads` libvmaf threads (and `frame_rate` for
        XPSNR)."""
        self.paired = not pairs_in_app() if paired is None else paired
        if backend == "cpu":
            self._scorer = libvmaf_cpu.CpuScorer(width, height, bit_depth, tuple(models), n_subsample, threads, frame_rate)
        elif "vmaf_v1" in models:
            from vmaf_app.core import vmaf_v1_gpu

            self._scorer = vmaf_v1_gpu.MultiScorer(width, height, bit_depth, models, n_subsample, backend, device)
        elif backend == "vulkan":
            from vmaf_app.core import vmaf_vulkan

            self._scorer = vmaf_vulkan.VulkanScorer(width, height, bit_depth, models, n_subsample, device=device)
        else:
            self._scorer = GpuScorer(width, height, bit_depth, models, n_subsample)
        #: Whether frames are paired by timestamp (frame_sync.by_timestamp),
        #: decided once so that the readers and the pairing agree.
        self._by_time = frame_sync.by_timestamp()
        if self.paired:
            reader = _PipeReader
        elif self._by_time:
            reader = _TimedStreamReader
        else:
            reader = _StreamReader
        self.distorted = reader("distorted", self._scorer.frame_bytes)
        self.reference = reader("reference", self._scorer.frame_bytes)
        self.error: BaseException | None = None
        self._result = None
        self._feeder = threading.Thread(target=self._feed if self.paired else self._feed_streams,
                                        name="vmaf-gpu-feed", daemon=True)
        self.distorted.start()
        self.reference.start()
        self._feeder.start()

    def _stream_pairs(self) -> Iterator[tuple[bytearray, bytearray]]:
        """Yields the two videos' frames in pairs, as libvmaf's filter pairs
        them: frame N with frame N (frame_sync), or by timestamp
        (frame_sync_by_time)."""
        test, ref = self.distorted, self.reference
        if not self._by_time:
            yield from frame_sync.frame_pairs(test.pull, ref.pull, test.release, ref.release)
            return
        pairs = frame_sync_by_time.frame_pairs(test.pull, ref.pull, test.wait_time_base(), ref.wait_time_base(),
                                               test.release, ref.release)
        try:
            for test_frame, ref_frame, _when in pairs:
                if ref_frame is None:
                    raise libvmaf.LibvmafError("the source has no frame for the test video's first")
                yield test_frame, ref_frame
        finally:
            pairs.close()

    def _feed_streams(self) -> None:
        """Pairs the two videos' frames (_stream_pairs) and scores each pair."""
        try:
            test, ref = self.distorted, self.reference
            pairs = self._stream_pairs()
            try:
                for test_frame, ref_frame in pairs:
                    self._scorer.add(ref_frame, test_frame)
            finally:
                pairs.close()
            for reader in (test, ref):
                reader.drop_rest()
            for reader in (test, ref):
                reader.join()
            self._result = self._scorer.finish()
        except BaseException as error:  # finish() raises it on the run's thread
            self.error = error
            self.stop()

    def _feed(self) -> None:
        """Scores the pairs FFmpeg made, as they arrive on the two raw pipes."""
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
        except BaseException as error:  # finish() raises it on the run's thread
            self.error = error
            self.stop()

    def stop(self) -> None:
        """Stops both readers."""
        self.distorted.stop()
        self.reference.stop()

    def finish(self, ffmpeg_succeeded: bool) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Returns the scores, once FFmpeg has ended; None when FFmpeg failed
        (the caller then retries or gives up). Raises libvmaf.LibvmafError when the
        scoring failed."""
        if not ffmpeg_succeeded:
            self.stop()
        elif self.paired:
            self.distorted.release_if_unconnected()
            self.reference.release_if_unconnected()
        else:
            self.distorted.ffmpeg_ended()
            self.reference.ffmpeg_ended()
        self._feeder.join()
        try:
            if self.error is not None:
                raise libvmaf.LibvmafError(f"GPU VMAF failed: {self.error}") from self.error
            return self._result if ffmpeg_succeeded else None
        finally:
            self._scorer.close()

    def output_args(self, duration_limit: float) -> list[str] | tuple[list[str], list[str]]:
        """FFmpeg's output options for the graph's [vmaf_dist] and [vmaf_ref].

        Every frame is passed as it comes (no frame rate conversion), cut at
        the duration limit, which keeps the same frames of both videos
        because they are numbered (frame_sync). When FFmpeg pairs the frames,
        this is one list with two raw video outputs. When the frames are
        paired here, it is a pair of lists, (the test video's FFmpeg's, the
        source's), each writing its video's frames as YUV4MPEG (pairing by
        timestamp, see _timed_output_args)."""
        if not self.paired and self._by_time:
            return self._timed_output_args(duration_limit)
        cut = frame_sync.duration_args(f"{duration_limit:.3f}" if duration_limit > 0 else None)
        if not self.paired:
            return tuple(["-map", f"[{label}]", "-fps_mode", "passthrough", *cut, "-flush_packets", "1",
                          "-f", "yuv4mpegpipe", "-strict", "-1", reader.path]
                         for label, reader in (("vmaf_dist", self.distorted), ("vmaf_ref", self.reference)))
        args = []
        for label, reader in (("vmaf_dist", self.distorted), ("vmaf_ref", self.reference)):
            args += ["-map", f"[{label}]", "-fps_mode", "passthrough", *cut, "-f", "rawvideo", reader.path]
        return args

    def _timed_output_args(self, duration_limit: float) -> tuple[list[str], list[str]]:
        """output_args when pairing by timestamp: (the test video's FFmpeg's,
        the source's), each writing its video's frames as YUV4MPEG and, from
        their [.._ts] copy, their timestamps as a framecrc listing. The
        source runs a second past the limit, so that the source frame nearest
        the test video's last frame is among those written."""
        both = []
        for label, reader, past in (("vmaf_dist", self.distorted, 0.0), ("vmaf_ref", self.reference, 1.0)):
            cut = ["-t", f"{duration_limit + past:.3f}"] if duration_limit > 0 else []
            both.append(["-map", f"[{label}]", "-fps_mode", "passthrough", *cut, "-flush_packets", "1",
                         "-f", "yuv4mpegpipe", "-strict", "-1", reader.path,
                         "-map", f"[{label}_ts]", "-fps_mode", "passthrough", "-enc_time_base", "filter", *cut,
                         "-c:v", "wrapped_avframe", "-flush_packets", "1", "-f", "framecrc", reader.listing_path])
        return both[0], both[1]


# ------------------------------------------------- videos decoded here

def score_decoded(
    source: VideoInfo, distorted: VideoInfo, source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int,
    duration_limit: str | None, total_frames: int, scale_algorithm: str = "bicubic",
    on_progress: Callable[[int, int, float], None] | None = None,
    check_cancel: Callable[[], None] = lambda: None,
    process_handle=None, decoder: str | tuple[str, str] = "nvidia",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Scores VMAF and NEG with both videos decoded in this process, giving
    the same scores GpuAttempt gets from FFmpeg's raw outputs for the same
    frames.

    The videos are decoded by NVIDIA's decoder, whose pictures stay on the
    GPU, or by the software decoder, whose pictures are uploaded; `decoder`
    is one for both, or (the source's, the test video's).

    What FFmpeg's graph does for those outputs is done here instead.
    gpu_frames crops as FFmpeg's crop filter does, shifts 10-bit samples and
    widens 8-bit ones as FFmpeg converts them, and scales on the GPU with
    the same filter (not exactly to the sample; the user decided that is
    the same comparison). frame_sync.decoded_frame_pairs pairs frame N with
    frame N, as the overlay does (or by timestamp), and as many pairs are
    scored as FFmpeg's -t `duration_limit` (as text) keeps.

    Raises GpuDecodeUnavailableError when the videos cannot be decoded here
    (another decoder, or a codec or size the decoder does not take); FFmpeg
    then decodes them. Raises GpuDecodeFailedError when decoding fails after
    it has started; the run is then made again through FFmpeg."""
    plans = []
    source_decoder, test_decoder = gpu_frames.decoder_pair(decoder)
    if not {source_decoder, test_decoder} <= {"nvidia", gpu_frames.SOFTWARE}:
        raise gpu_frames.GpuDecodeUnavailableError("libvmaf's CUDA code takes NVIDIA's and the software decoder's")
    for info, crop, backend in ((distorted, distorted_crop, test_decoder), (source, source_crop, source_decoder)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height),
                                        algorithm=scale_algorithm)
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=gpu_frames.WIDEN_REPEAT if full else gpu_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise gpu_frames.GpuDecodeUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = gpu_frames.decoder_supports(gpu_frames.device_of(backend, gpu_frames.CUDA_DEVICE), plan, backend)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
    # The software decoder writes each luma plane into the upload's memory,
    # which has the size the videos are compared at.
    if any(plan.frame_bytes != width * height * (2 if bit_depth > 8 else 1) for plan in plans):
        raise gpu_frames.GpuDecodeUnavailableError("the decoders' frames are not the size compared at")
    # The decoders are opened first: the first to start CUDA sets CUDA's
    # waits to sleep rather than spin (see gpu_frames), and libvmaf's waits
    # then sleep too.
    test = gpu_frames.GpuFrameStream(distorted, plans[0], gpu_frames.device_of(test_decoder, gpu_frames.CUDA_DEVICE), pool=4,
                                     process_handle=process_handle, backend=test_decoder)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], gpu_frames.device_of(source_decoder, gpu_frames.CUDA_DEVICE), pool=4,
                                        process_handle=process_handle, backend=source_decoder)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        on_device = source_decoder == test_decoder == "nvidia"
        scorer = GpuScorer(width, height, bit_depth, models, n_subsample, on_device=on_device)

        def side(stream, slot: int) -> tuple[bool, Callable]:
            """A decoded picture as GpuScorer.add_sides takes it."""
            if stream.backend == "nvidia":  # copied on the GPU
                return True, lambda address, pitch: stream.copy_luma(slot, address, pitch)
            return False, lambda address: stream.download(slot, address)  # into the upload's memory

        test.start()
        ref.start()
        pairs = frame_sync.decoded_frame_pairs(test, ref, distorted, duration_limit, check_cancel,
                                               test.release, ref.release)
        count = 0
        started = reported = time.perf_counter()
        try:
            for test_slot, ref_slot in pairs:
                scorer.add_sides(side(ref, ref_slot), side(test, test_slot))
                count += 1
                # Progress is reported four times a second, about as often as
                # FFmpeg's -progress, because each report crosses to the app's
                # process.
                now = time.perf_counter()
                if on_progress is not None and now - reported >= 0.25:
                    reported = now
                    on_progress(count, total_frames, count / (now - started))
        finally:
            pairs.close()  # gives back the frames the pairing holds
        # Check that the pictures decoded so far match the packets, one each
        # (a video that ended was checked at its end).
        test.verify()
        ref.verify()
        result = scorer.finish()
        if on_progress is not None and count:
            elapsed = time.perf_counter() - started
            on_progress(count, total_frames, count / elapsed if elapsed > 0 else 0.0)
        return result
    finally:
        # The decoders are closed before libvmaf, so their threads stop first.
        test.close()
        ref.close()
        if scorer is not None:
            scorer.close()


# ------------------------------------------------------------ availability

def gpu_models(compute_vmaf: bool, compute_vmaf_neg: bool, model: str,
               compute_vmaf_v1: bool = False, model_v1: str = "") -> dict[str, str] | None:
    """The models a run scores on the GPU, as {log name: libvmaf model}, or
    None when it scores none there.

    VMAF with a custom model file is calculated on the CPU, and so is VMAF
    NEG beside it. VMAF v1 has bundled models only; its entry ("vmaf_v1") is
    its model's "path=<file>", for vmaf_v1_gpu."""
    models = {}
    if compute_vmaf or compute_vmaf_neg:
        version = _GPU_MODELS.get(model) if compute_vmaf else ""
        if version is not None:
            if compute_vmaf:
                models["vmaf"] = version
            if compute_vmaf_neg:
                models["vmaf_neg"] = _NEG_MODEL
    if compute_vmaf_v1:
        models["vmaf_v1"] = model_v1
    return models or None


def probe() -> tuple[bool, str]:
    """Whether libvmaf can score on this GPU, and its version or why not.

    It starts CUDA and scores three small frame pairs, because the CUDA
    kernels load only then (a GPU older than the build supports fails here).
    Run it in a process of its own."""
    if not libvmaf.LIBRARY_PATH.is_file():
        return False, "libvmaf with CUDA is not bundled"
    try:
        lib = libvmaf.load()
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
    except (OSError, libvmaf.LibvmafError) as error:
        return False, str(error)


# --------------------------------------------------------- whether a run uses it

_PROBE_LOCK = threading.Lock()
_probed: tuple[bool, str] | None = None
_probed_at = 0.0
#: A failed probe older than this is made again before the next run (see
#: forget_failed_probe), as Vship's is.
FAILED_PROBE_RETRY_SECONDS = 60.0
#: What the probe chose: "cuda" or "vulkan", and Vulkan's GPU number.
_backend: tuple[str, int | None] = ("cuda", None)
#: Settings > GPU metrics > GPU backend, which VMAF follows as Vship does.
#: Every choice but "cuda" tries Vulkan first, on every GPU: it gives the same
#: scores as libvmaf's CUDA code, and on an RTX 5090 scores 4K VMAF and NEG at
#: 855 frames a second against official libvmaf CUDA's 177 (libvmaf-fast's
#: benchmark, frames already in memory). "cuda" tries libvmaf's CUDA code
#: first. "hip" is Vship's build for AMD; VMAF has no such build.
_preference = "auto"


def set_gpu_backend(preference: str) -> None:
    """Follows the GPU backend setting. The GPU is probed again when the
    choice changes which backend is tried first."""
    global _preference, _probed
    with _PROBE_LOCK:
        changed = _backend_order(preference) != _backend_order(_preference)
        _preference = preference
        if changed and _probed is not None:
            _probed = None
            _log.info("VMAF on the GPU: GPU backend set to %s", preference)


def _backend_order(preference: str) -> tuple[str, ...]:
    """The backends to try, in order, for a GPU backend setting."""
    return ("cuda", "vulkan") if preference == "cuda" else ("vulkan", "cuda")


def gpu_vmaf_available() -> tuple[bool, str]:
    """Whether VMAF is scored on this PC's GPU, and with what (or why not).
    The GPU is probed once per backend setting, in a process of its own, and
    again after a failure (see forget_failed_probe)."""
    global _probed, _probed_at, _backend
    with _PROBE_LOCK:
        if _probed is None:
            name, device, text = _probe_once(_preference)
            _probed, _probed_at = (name is not None, text), time.monotonic()
            available = name is not None
            if available:
                _backend = (name, device)
                _log.info("VMAF on the GPU: %s", text)
            else:
                _log.info("VMAF on the GPU unavailable: %s", text)
        return _probed


def gpu_vmaf_backend() -> tuple[str, int | None]:
    """What a run that scores VMAF on the GPU uses: ("cuda", None) or
    ("vulkan", Vulkan's GPU number). Meaningful only when
    gpu_vmaf_available() is true."""
    gpu_vmaf_available()
    return _backend


def _probe_once(preference: str = "auto") -> tuple[str | None, int | None, str]:
    """Probes the backends in the order the setting gives, and returns
    (backend, Vulkan's GPU number, description) for the first that scores
    on this PC, or (None, None, why not) when neither does."""
    from vmaf_app.core.gpu import detected_gpu_vendors
    from vmaf_app.core.isolated import IsolatedCrashError, run_isolated
    from vmaf_app.core.models import GpuVendor

    reasons = []
    for backend in _backend_order(preference):
        try:
            if backend == "cuda":
                if GpuVendor.NVIDIA not in detected_gpu_vendors():
                    reasons.append("no NVIDIA GPU")
                    continue
                try:
                    available, text = run_isolated(probe, what="libvmaf's GPU probe")
                except IsolatedCrashError as error:
                    available, text = False, str(error)
                if available:
                    return "cuda", None, f"{text} with CUDA"
                reasons.append(f"CUDA: {text}")
            else:
                from vmaf_app.core import vmaf_vulkan

                try:
                    available, device, text = run_isolated(vmaf_vulkan.probe, what="the Vulkan VMAF probe")
                except IsolatedCrashError as error:
                    available, device, text = False, None, str(error)
                if available:
                    return "vulkan", device, text
                reasons.append(f"Vulkan: {text}")
        except Exception as error:
            # Any failure means "not on this GPU". A failure that escaped here
            # would leave the probe unanswered and fail every video's setup,
            # instead of VMAF being calculated on the CPU.
            _log.warning("The %s VMAF probe failed", backend, exc_info=error)
            reasons.append(f"{backend}: the GPU probe failed: {error}")
    return None, None, "; ".join(reasons)


def forget_failed_probe() -> None:
    """Forgets a probe that failed a while ago, so that the next run probes
    again. Call it before a run, off the UI thread. The failure may have been
    temporary (a driver being updated or restarted); without this, VMAF
    would be calculated on the CPU until the app restarted."""
    global _probed
    with _PROBE_LOCK:
        if (_probed is not None and not _probed[0]
                and time.monotonic() - _probed_at >= FAILED_PROBE_RETRY_SECONDS):
            _probed = None


def start_gpu_vmaf_probe() -> None:
    """Probes the GPU in the background at startup, so the answer is ready
    by the time a run needs it."""
    threading.Thread(target=gpu_vmaf_available, name="vmaf-gpu-probe", daemon=True).start()


def scores_on_gpu(compute_vmaf: bool, compute_vmaf_neg: bool, model: str,
                  enabled: bool = True, bit_depth: int = 8,
                  size: tuple[int, int] | None = None,
                  compute_vmaf_v1: bool = False, model_v1: str = "") -> dict[str, str] | None:
    """The models a run scores on the GPU (see gpu_models), or None when its
    VMAF is calculated on the CPU. That happens:
    - when the video is set to CPU (`enabled`, its VmafOptions.vmaf_on_gpu);
    - with a custom model;
    - for a comparison deeper than 10 bits (`bit_depth`), because the frame
      pairs can reach the GPU through FFmpeg's overlay, which holds only 8
      and 10 bits;
    - for a comparison at an odd width or height (`size`, where known). The
      pairs cross that overlay side by side on one 4:2:0 canvas
      (vmaf_runner._gpu_pairs_stage), and FFmpeg's pad, which makes the
      canvas, keeps a 4:2:0 picture on whole chroma samples: it gives an
      even size and blacks out an odd picture's last column and row, so the
      attempt would fail in FFmpeg;
    - when no GPU can score it (with CUDA or Vulkan).
    VMAF v1 (`compute_vmaf_v1`, with model `model_v1`) is included where
    vmaf_v1_gpu can calculate it. Whatever the GPU does not score is left
    out, and FFmpeg's libvmaf calculates it."""
    if not enabled or bit_depth > 10 or (size is not None and (size[0] & 1 or size[1] & 1)):
        return None
    models = gpu_models(compute_vmaf, compute_vmaf_neg, model, compute_vmaf_v1, model_v1)
    if models is None:
        return None
    if "vmaf_v1" in models:
        from vmaf_app.core import vmaf_v1_gpu

        if not vmaf_v1_gpu.scores(models["vmaf_v1"], size):
            del models["vmaf_v1"]
    if set(models) - {"vmaf_v1"} and not gpu_vmaf_available()[0]:
        models = {key: value for key, value in models.items() if key == "vmaf_v1"}
    return models or None
