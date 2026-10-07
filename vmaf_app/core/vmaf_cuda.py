"""VMAF and VMAF NEG on the GPU. On an NVIDIA GPU: libvmaf's CUDA feature
extractors (VIF, ADM and motion) from the bundled tools/libvmaf/libvmaf.dll,
libvmaf-fast's (github.com/4KVCD/libvmaf-fast, a fork of libvmaf with the pull
requests that fix its CUDA code merged; scripts/fetch_libvmaf_fast.ps1
installs its release). On any other GPU, or when Settings >
GPU metrics > GPU backend is Vulkan: the port of those extractors to Vulkan
(vmaf_vulkan), which gives the same scores. What is said below of libvmaf's
scores holds for both.

Only those two scores: VMAF v1, PSNR, SSIM and XPSNR have no GPU code, and
stay in FFmpeg's libvmaf and xpsnr filters, so they are scored exactly as
before. A run that scores VMAF on the GPU still decodes each video once:
an FFmpeg for each video writes its frames, as they would reach libvmaf's
filter, to a named pipe and their timestamps to another (_StreamReader), and
this module pairs them as that filter would (frame_sync) and feeds libvmaf
frame pair by frame pair (see vmaf_runner._run_on_gpu), each frame's luma
uploaded into a picture of libvmaf's on the GPU (_HostUpload). With an FFmpeg older
than 6.1, one FFmpeg pairs them itself and writes two raw outputs
(_PipeReader), as every run did before: about twice the CPU for the same
frames (pairs_in_app).

Where NVIDIA's decoder decodes both videos, VMAF is scored without FFmpeg's
decode (score_decoded): the videos are decoded in libvmaf's process
(gpu_frames), scaled and widened on the GPU to the size and depth they
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
import functools
import logging
import msvcrt
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from vmaf_app.core import gpu_frames
from vmaf_app.core.frame_sync import frame_pairs
from vmaf_app.core.models import CropBox, VideoInfo

_log = logging.getLogger(__name__)
#: ConnectNamedPipe's answers when the writer has already connected (535),
#: or has connected, written and closed (232).
_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA = 535, 232

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "tools" / "libvmaf" / "libvmaf.dll"
#: The libvmaf-fast release the bundled libvmaf.dll and vmaf_vulkan.dll are
#: (scripts/fetch_libvmaf_fast.ps1 installs this one).
LIBVMAF_FAST_VERSION = "3.2.0-fast.1"
#: What a GPU score records it was calculated with (its provenance).
LIBRARY_BUILD = f"libvmaf-fast {LIBVMAF_FAST_VERSION} (CUDA)"

#: The app's built-in VMAF models -> libvmaf's names for them. A custom model
#: file is calculated on the CPU.
_GPU_MODELS = {"version=vmaf_v0.6.1": "vmaf_v0.6.1", "version=vmaf_4k_v0.6.1": "vmaf_4k_v0.6.1"}
_NEG_MODEL = "vmaf_v0.6.1neg"

_VMAF_PIX_FMT_YUV420P = 1
_VMAF_PIX_FMT_YUV400P = 4
_VMAF_LOG_LEVEL_ERROR = 1
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


class _PlainPicture(ctypes.Structure):
    """libvmaf's VmafPicture, as libvmaf-fast v3.2.0-fast.1 lays it out."""
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class _ColorPicture(ctypes.Structure):
    """VmafPicture since upstream libvmaf's vmaf_picture_convert (Netflix/vmaf
    0497a0f2, in libvmaf-fast after v3.2.0-fast.1): a VmafColor (range,
    primaries, transfer, matrix: four enums) between data and ref. With the
    old layout against such a libvmaf, it writes past the app's pictures
    (vmaf_picture_alloc: heap corruption) and reads ref and priv from the
    wrong place."""
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("color", ctypes.c_int * 4), ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


#: The VmafPicture layout of the libvmaf loaded: _load sets it, and pictures
#: are made through this name after _load, never before.
_Picture: type[ctypes.Structure] = _PlainPicture


class _PictureParameters(ctypes.Structure):
    _fields_ = [("w", ctypes.c_uint), ("h", ctypes.c_uint), ("bpc", ctypes.c_uint), ("pix_fmt", ctypes.c_int)]


class _PictureConfiguration(ctypes.Structure):
    """libvmaf's pool of pictures in system memory: vmaf_v1_gpu's (VMAF v1's
    CPU features). GpuScorer's are in GPU memory."""

    _fields_ = [("pic_params", _PictureParameters), ("pic_cnt", ctypes.c_uint)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("flags", ctypes.c_uint64)]


class _CudaConfiguration(ctypes.Structure):
    _fields_ = [("cu_ctx", ctypes.c_void_p)]


class _CudaPictureConfiguration(ctypes.Structure):
    _fields_ = [("pic_params", _PictureParameters), ("pic_prealloc_method", ctypes.c_int)]


#: VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE: libvmaf's own pictures in
#: GPU memory, which frames decoded on the GPU are copied into there, and
#: frames from system memory uploaded into (_HostUpload).
_PREALLOCATE_ON_DEVICE = 1


_library: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _library, _Picture
    if _library is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        # The picture layout this libvmaf has: vmaf_picture_convert came with
        # the colour field.
        _Picture = _ColorPicture if hasattr(lib, "vmaf_picture_convert") else _PlainPicture
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
            ("vmaf_use_feature", ctypes.c_int, [handle, ctypes.c_char_p, handle]),
            ("vmaf_feature_dictionary_set", ctypes.c_int, [pointer, ctypes.c_char_p, ctypes.c_char_p]),
            ("vmaf_feature_dictionary_free", ctypes.c_int, [pointer]),
            ("vmaf_feature_score_at_index", ctypes.c_int,
             [handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes
        _library = lib
    return _library


def _check(error: int, what: str) -> None:
    if error:
        raise VmafGpuError(f"{what} failed (libvmaf error {error})")


def path_bytes(path) -> bytes:
    """An existing file's path as libvmaf opens it: with fopen(), which reads
    the name in Windows' ANSI code page, not UTF-8. UTF-8 bytes of a folder
    with an accented letter (a user's own name in their profile and its Temp
    folder) named a file that is not there. A name the code page has no
    letters for (Chinese on a western Windows) goes by its short 8.3 path
    where the drive keeps those; VmafGpuError where it does not."""
    text = str(path)
    for attempt in range(2):
        try:
            return text.encode("mbcs" if os.name == "nt" else "utf-8", errors="strict")
        except UnicodeEncodeError:
            if attempt:
                break
            short = ctypes.create_unicode_buffer(32768)
            if not ctypes.windll.kernel32.GetShortPathNameW(text, short, len(short)):
                break
            text = short.value
    raise VmafGpuError(f"libvmaf cannot open {path}: its name has letters outside this PC's code page")


# ------------------------------------------------- frames from system memory

class _Copy2D(ctypes.Structure):
    """CUDA_MEMCPY2D."""

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
    """CUDA's driver API (nvcuda.dll, the NVIDIA driver's), which libvmaf
    loads too."""
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
    if error:
        raise VmafGpuError(f"{what} failed (CUDA error {error})")


class _HostUpload:
    """Copies frames' luma from system memory into libvmaf's pictures on the
    GPU: through one page-locked buffer a side, allocated once, and a stream
    of its own.

    libvmaf's own way with frames in system memory (a pool of CPU pictures,
    each uploaded from pageable memory when it is read) scored 4K 10-bit
    pairs at about 180 a second on an RTX 5090, and this at about 330.
    Its page-locked pictures (HOST_PINNED) are no way round: each fetch
    allocates, locks and zeroes a new one, 9 ms a 4K pair.

    In libvmaf's CUDA context, the primary one of its GPU, which is made
    current around each use: the frames come on a thread of the caller's."""

    def __init__(self, width: int, height: int, sample: int):
        self._cu = cu = _cuda()
        self._row_bytes, self._rows = width * sample, height
        self.luma_bytes = self._row_bytes * height
        self._device = ctypes.c_int()
        self._context = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._staging = [ctypes.c_void_p(), ctypes.c_void_p()]
        _cuda_check(cu.cuInit(0), "Starting CUDA")
        _cuda_check(cu.cuDeviceGet(ctypes.byref(self._device), _GPU), "Finding the GPU")
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
        _cuda_check(self._cu.cuCtxPushCurrent_v2(self._context), "Taking the GPU")
        return self

    def __exit__(self, *_exc) -> None:
        self._cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))

    def send(self, side: int, frame, address: int, pitch: int) -> None:
        """Starts the copy of `frame`'s luma to the picture at `address`
        (rows `pitch` bytes apart); wait() says when `frame`'s side may be
        sent again. Within `with self`."""
        source = np.frombuffer(frame, dtype=np.uint8)
        if source.size < self.luma_bytes:
            raise VmafGpuError(f"a frame of {source.size} bytes, where its luma alone is {self.luma_bytes}")
        staging = self._staging[side].value
        ctypes.memmove(staging, source.ctypes.data, self.luma_bytes)
        copy = _Copy2D(srcMemoryType=_CU_MEMORYTYPE_HOST, srcHost=staging, srcPitch=self._row_bytes,
                       dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=address, dstPitch=pitch,
                       WidthInBytes=self._row_bytes, Height=self._rows)
        _cuda_check(self._cu.cuMemcpy2DAsync_v2(ctypes.byref(copy), self._stream), "Uploading a frame")

    def wait(self) -> None:
        _cuda_check(self._cu.cuStreamSynchronize(self._stream), "Uploading a frame")

    def close(self) -> None:
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
    """One libvmaf context on the GPU, given frame pairs in order: 4:2:0
    frames packed as FFmpeg's rawvideo writes them, at `bit_depth` bits
    (16-bit little-endian samples above 8), whose luma is uploaded (add;
    VMAF reads nothing else) -- or pictures in GPU memory that add_on_device
    has filled. `on_device`: add_on_device alone is used, and nothing is
    allocated for uploads."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 on_device: bool = False):
        self._lib = lib = _load()
        self._upload: _HostUpload | None = None
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
            # libvmaf's pictures in GPU memory, whoever fills them: the
            # decoders, or _HostUpload from frames in system memory.
            parameters = _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV420P)
            pictures = _CudaPictureConfiguration(parameters, _PREALLOCATE_ON_DEVICE)
            _check(lib.vmaf_cuda_preallocate_pictures(self._context, pictures), "Allocating pictures")
            sample = 1 if bit_depth <= 8 else 2
            if not on_device:
                self._upload = _HostUpload(width, height, sample)
        except BaseException:
            self.close()
            raise
        #: A frame as FFmpeg writes it: the luma, then the chroma planes,
        #: which FFmpeg rounds up for an odd size.
        self.frame_bytes = (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * sample

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before). The
        frames are the caller's again when it returns."""
        upload = self._upload
        if upload is None:
            raise VmafGpuError("this scorer takes pictures in GPU memory only")

        def fill_distorted(picture: _Picture) -> None:
            # Both copies are under way together; the reference's is waited
            # for too when this one could not be started.
            try:
                upload.send(1, distorted, picture.data[0], picture.stride[0])
            finally:
                upload.wait()

        with upload:
            self._add(lambda picture: upload.send(0, reference, picture.data[0], picture.stride[0]), fill_distorted)

    def add_on_device(self, reference: Callable[[int, int], None], distorted: Callable[[int, int], None]) -> None:
        """Scores one more pair of pictures in GPU memory: `reference` and
        `distorted` are each given a picture's luma plane (address, pitch)
        to fill, and return once it is filled."""
        self._add(lambda picture: reference(picture.data[0], picture.stride[0]),
                  lambda picture: distorted(picture.data[0], picture.stride[0]))

    def _add(self, fill_reference, fill_distorted) -> None:
        fetch = self._lib.vmaf_cuda_fetch_preallocated_picture
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
        if self._upload is not None:
            self._upload.close()
            self._upload = None
        for model in self._models.values():
            self._lib.vmaf_model_destroy(model)
        self._models.clear()
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


#: libvmaf's CPU feature extractors scored by CpuScorer: the app's metric key
#: -> (the extractor, the feature whose score is the metric's, the decimals
#: kept of it), as FFmpeg's libvmaf filter is asked for them ("name=psnr",
#: "name=float_ssim") and as its log names and rounds the scores the app
#: reads (vmaf_runner._parse_log); XPSNR as FFmpeg's xpsnr filter writes it
#: in its stats file ("%3.4f"), by libvmaf-fast's port of that filter.
CPU_FEATURES = {"psnr": ("psnr", "psnr_y", 6), "ssim": ("float_ssim", "float_ssim", 6),
                "xpsnr": ("xpsnr", "xpsnr_y", 4)}
#: What a CPU score records it was calculated with.
CPU_BUILD = f"libvmaf-fast {LIBVMAF_FAST_VERSION} (CPU)"
#: The most memory CpuScorer's pictures take, unless two pairs of them take
#: more (8K): page-locked, from NVIDIA's decoder. 4K 10-bit from NVIDIA's
#: decoder, PSNR + SSIM: 2 pairs 49 fps, 3 93, 4 129, 5 159, 6 (this) and 8
#: 172 fps, where the decoders' feed levels off. With XPSNR, whose pairs
#: take longer, and the pair it keeps from the frames before on top: 7
#: pairs 135 fps, 8 154, 9 169 (CPU_XPSNR_PICTURE_MEMORY), 10 171.
CPU_PICTURE_MEMORY = 200 << 20
CPU_XPSNR_PICTURE_MEMORY = 270 << 20


class CpuScorer:
    """PSNR, SSIM and XPSNR of frame pairs given in order, by libvmaf-fast's
    own CPU extractors -- the ones FFmpeg's libvmaf filter runs, which give
    the same scores to the six decimals its log keeps, and its port of
    FFmpeg's xpsnr filter (CPU_FEATURES) -- in this process, on its
    pictures (vmaf_preallocate_pictures). FFmpeg's filter allocates, zeroes
    and copies two new pictures for every pair on its one filter thread
    before libvmaf's threads see them: 4K PSNR + SSIM ran at 67 fps there,
    with most cores idle. The pictures are luma only: the app keeps PSNR's
    luma score (psnr_y) and SSIM, which read nothing else, and libvmaf's
    PSNR leaves out the chroma planes of a picture without them -- the
    same psnr_y, a third less to copy and to score. `metrics`: CPU_FEATURES
    keys. Frames as GpuScorer takes them (4:2:0, packed as FFmpeg's rawvideo
    writes them; their luma is read), or a decoder's (add_decoded)."""

    def __init__(self, width: int, height: int, bit_depth: int, metrics: tuple[str, ...], n_subsample: int = 1,
                 threads: int = 0, frame_rate: int = 0):
        """`frame_rate`: XPSNR's, as FFmpeg's filter takes it (whole frames a
        second of the reference video, vmaf_runner._xpsnr_frame_rate)."""
        self._lib = lib = _load()
        self._step = max(1, n_subsample)
        self._metrics = tuple(metrics)
        self._context = ctypes.c_void_p()
        self._count = 0
        #: Pictures' memory page-locked for NVIDIA's decoder -> the stream
        #: that did it (None: it could not).
        self._pinned: dict[int, object] = {}
        self._staging: list[bytearray] | None = None
        sample = 1 if bit_depth <= 8 else 2
        self._rows, self._row_bytes = height, width * sample
        #: A frame's luma: the size of a decoder's luma-only frame.
        self.luma_bytes = width * height * sample
        self.frame_bytes = self.luma_bytes + 2 * ((width + 1) // 2) * ((height + 1) // 2) * sample
        threads = max(0, int(threads))
        configuration = _Configuration(_VMAF_LOG_LEVEL_ERROR, threads, self._step, 0, 0)
        _check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
        try:
            # With XPSNR, PSNR comes from the squared errors it sums anyway
            # (its option "psnr"), not a pass of its own over both pictures:
            # 4K PSNR + SSIM + XPSNR 6% faster with Intel's decoder, 11% with
            # NVIDIA's.
            fused = "psnr" in self._metrics and "xpsnr" in self._metrics
            for metric in self._metrics:
                if fused and metric == "psnr":
                    continue
                extractor = CPU_FEATURES[metric][0]
                options = ctypes.c_void_p()
                if metric == "xpsnr":
                    # Weighted by the source's activity (the extractor's
                    # default), as FFmpeg's filter weights by its first input,
                    # the source in the app's commands (vmaf_runner._xpsnr_filter).
                    settings = [("frame_rate", str(int(frame_rate)))]
                    if fused:
                        settings.append(("psnr", "true"))
                    for key, value in settings:
                        if lib.vmaf_feature_dictionary_set(ctypes.byref(options), key.encode(), value.encode()):
                            lib.vmaf_feature_dictionary_free(ctypes.byref(options))
                            raise VmafGpuError(f"Setting {extractor}'s {key} failed")
                # libvmaf frees the options (one it refuses early keeps a few bytes).
                _check(lib.vmaf_use_feature(self._context, extractor.encode(), options), f"Starting {extractor}")
            pairs = cpu_pairs(width, height, bit_depth, threads, xpsnr="xpsnr" in self._metrics)
            parameters = _PictureParameters(width, height, bit_depth, _VMAF_PIX_FMT_YUV400P)
            _check(lib.vmaf_preallocate_pictures(self._context, _PictureConfiguration(parameters, 2 * pairs)),
                   "Allocating pictures")
        except BaseException:
            self.close()
            raise

    def add(self, reference, distorted) -> None:
        """Scores one more pair (frame index = how many came before). The
        frames are the caller's again when it returns."""
        ref, dist = self._take()
        try:
            self._fill(ref, reference, self.frame_bytes)
            self._fill(dist, distorted, self.frame_bytes)
        except BaseException:
            self._give_back(ref, dist)
            raise
        self._score(ref, dist)

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Scores one more pair held by gpu_frames' decoders, decoding luma
        only (luma_bytes frames). NVIDIA's copies it into libvmaf's picture
        itself, rows as far apart as the picture's (download_planes), into
        memory it page-locks the first time each of the pool's pictures
        comes round. Intel's and AMD's write it straight into the picture
        where its rows are packed, as libvmaf aligns them at most widths
        (multiples of 32 samples: 3840, 1920, 1280), else into memory it is
        copied from as add() copies a frame."""
        ref, dist = self._take()
        try:
            self._download(ref, ref_stream, ref_slot, 0)
            self._download(dist, test_stream, test_slot, 1)
        except BaseException:
            self._give_back(ref, dist)
            raise
        self._score(ref, dist)

    def _take(self) -> tuple[_Picture, _Picture]:
        lib = self._lib
        ref, dist = _Picture(), _Picture()
        _check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            _check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        return ref, dist

    def _give_back(self, ref: _Picture, dist: _Picture) -> None:
        # Pictures taken from the pool and never handed over keep vmaf_close
        # waiting for them.
        self._lib.vmaf_picture_unref(ctypes.byref(ref))
        self._lib.vmaf_picture_unref(ctypes.byref(dist))

    def _score(self, ref: _Picture, dist: _Picture) -> None:
        # libvmaf takes both pictures, also when it fails (pull request 1652).
        _check(self._lib.vmaf_read_pictures(self._context, ctypes.byref(ref), ctypes.byref(dist), self._count),
               f"Scoring frame {self._count}")
        self._count += 1

    def _download(self, picture: _Picture, stream, slot: int, which: int) -> None:
        address, pitch = picture.data[0], picture.stride[0]
        if stream.backend == "nvidia":
            if address not in self._pinned:
                self._pinned[address] = stream if stream.pin(address, pitch * picture.h[0]) else None
            stream.download_planes(slot, (address, None, None), (pitch, 0, 0))
        elif pitch == self._row_bytes:
            stream.download(slot, address)
        else:
            if self._staging is None:
                self._staging = [bytearray(self.luma_bytes), bytearray(self.luma_bytes)]
            buffer = self._staging[which]
            stream.download(slot, ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer)))
            self._fill(picture, buffer, self.luma_bytes)

    def _fill(self, picture: _Picture, frame, size: int) -> None:
        """The luma of `frame`, `size` bytes or more."""
        source = np.frombuffer(frame, dtype=np.uint8)
        if len(source) < size:
            raise VmafGpuError(f"frame {self._count} is shorter than a picture")
        target = np.ctypeslib.as_array(ctypes.cast(picture.data[0], ctypes.POINTER(ctypes.c_uint8)),
                                       shape=(self._rows, picture.stride[0]))
        target[:, :self._row_bytes] = source[:self.luma_bytes].reshape(self._rows, self._row_bytes)

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The frame numbers scored (every n_subsample-th) and each metric's
        scores for them, rounded as the files the app reads FFmpeg's from
        round them (CPU_FEATURES)."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {metric: np.zeros(0) for metric in self._metrics}
        _check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        scores = {}
        value = ctypes.c_double()
        for metric in self._metrics:
            _extractor, feature, decimals = CPU_FEATURES[metric]
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                _check(self._lib.vmaf_feature_score_at_index(self._context, feature.encode(), ctypes.byref(value),
                                                             int(frame)),
                       f"Reading {metric} for frame {frame}")
                column[slot] = float(f"{value.value:.{decimals}f}")
            scores[metric] = column
        return frames, scores

    def close(self) -> None:
        """Before the decoders are closed, where the frames came from them."""
        # Unpinned first: vmaf_close frees the pictures, and CUDA's memory is
        # unregistered before it is freed. No copy into it is under way
        # (download_planes returns once its copy is done); an extractor still
        # reading it does not mind.
        for address, stream in self._pinned.items():
            if stream is not None:
                stream.unpin(address)
        self._pinned = {}
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = ctypes.c_void_p()


@functools.cache
def cpu_scores_xpsnr() -> bool:
    """Whether the bundled libvmaf-fast has the xpsnr extractor (a port of
    FFmpeg's filter, in its builds after 3.2.0-fast.1) with its option
    "psnr", which CpuScorer uses: a pair of tiny pictures is scored, as
    libvmaf ignores an option it does not know."""
    try:
        return _scores_xpsnr_and_psnr()
    except (OSError, VmafGpuError):
        return False


def _scores_xpsnr_and_psnr() -> bool:
    lib = _load()
    context = ctypes.c_void_p()
    _check(lib.vmaf_init(ctypes.byref(context), _Configuration(_VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0)), "Starting libvmaf")
    try:
        options = ctypes.c_void_p()
        _check(lib.vmaf_feature_dictionary_set(ctypes.byref(options), b"psnr", b"true"), "Setting an option")
        if lib.vmaf_use_feature(context, b"xpsnr", options):
            return False
        parameters = _PictureParameters(64, 64, 8, _VMAF_PIX_FMT_YUV400P)
        _check(lib.vmaf_preallocate_pictures(context, _PictureConfiguration(parameters, 2)), "Allocating pictures")
        ref, dist = _Picture(), _Picture()
        _check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(ref)), "Taking a picture")
        try:  # a picture never handed back keeps vmaf_close waiting for it
            _check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        # libvmaf takes both pictures, also when it fails.
        _check(lib.vmaf_read_pictures(context, ctypes.byref(ref), ctypes.byref(dist), 0), "Scoring")
        _check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing")
        value = ctypes.c_double()
        return all(lib.vmaf_feature_score_at_index(context, name, ctypes.byref(value), 0) == 0
                   for name in (b"xpsnr_y", b"psnr_y"))
    finally:
        lib.vmaf_close(context)


def cpu_pairs(width: int, height: int, bit_depth: int, threads: int, xpsnr: bool = False) -> int:
    """How many pairs of pictures CpuScorer's pool holds: one for each of
    libvmaf's threads and one being filled, up to 8 threads' worth -- PSNR
    and SSIM of 4K pairs from memory went little faster with more (139 fps
    with 8 threads, 152 with 24) -- in at most CPU_PICTURE_MEMORY
    (CPU_XPSNR_PICTURE_MEMORY with XPSNR), but two; with XPSNR, one more for
    the two test pictures it keeps from the frames before. A pair is scored
    on one thread: each more in the pool is one more scored at a time."""
    picture = ((width + 31) & ~31) * height * (1 if bit_depth <= 8 else 2)
    memory = CPU_XPSNR_PICTURE_MEMORY if xpsnr else CPU_PICTURE_MEMORY
    return max(2, min(min(8, max(1, threads)) + 1, memory // (2 * picture))) + xpsnr


def score_decoded_cpu(
    source: VideoInfo, distorted: VideoInfo, source_crop: CropBox | None, distorted_crop: CropBox | None, *,
    width: int, height: int, bit_depth: int, models: dict[str, str] | tuple[str, ...], n_subsample: int,
    duration_limit: str | None, total_frames: int, decoder: str, threads: int = 0, frame_rate: int = 0,
    scale_algorithm: str = "bicubic",
    on_progress: Callable[[int, int, float], None] | None = None,
    check_cancel: Callable[[], None] = lambda: None,
    process_handle=None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """PSNR and SSIM (CpuScorer; `models`: CPU_FEATURES keys) with both videos
    decoded in this process by the GPU's own decoder (gpu_frames: NVIDIA's,
    Intel's or AMD's), luma only, as score_decoded does for VMAF: the frames
    FFmpeg's outputs carry -- the decoders crop as FFmpeg's crop filter does
    and widen 8-bit samples as FFmpeg converts them -- paired as libvmaf's
    filter pairs them (frame_sync) and cut where FFmpeg's -t would cut them.
    Not for a comparison scaled to a size: the decoders do not scale as
    FFmpeg does to the sample, and PSNR and SSIM, unlike VMAF, are scored
    here to be FFmpeg's (GpuDecodeUnavailableError, and FFmpeg decodes, as
    for any video they do not take). GpuDecodeFailedError when decoding
    failed after the start."""
    plans = []
    device = _GPU if decoder == "nvidia" else 0
    for info, crop in ((distorted, distorted_crop), (source, source_crop)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height),
                                      algorithm=scale_algorithm)
        if plan.scaled:
            raise gpu_frames.GpuDecodeUnavailableError("the comparison is scaled, by FFmpeg")
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=gpu_frames.WIDEN_REPEAT if full else gpu_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise gpu_frames.GpuDecodeUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = gpu_frames.decoder_supports(device, plan, decoder)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
    test = gpu_frames.GpuFrameStream(distorted, plans[0], device, pool=4, process_handle=process_handle,
                                     backend=decoder)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], device, pool=4, process_handle=process_handle,
                                        backend=decoder)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        scorer = CpuScorer(width, height, bit_depth, tuple(models), n_subsample, threads, frame_rate)
        if test.frame_bytes != scorer.luma_bytes or ref.frame_bytes != scorer.luma_bytes:
            raise gpu_frames.GpuDecodeUnavailableError("the decoders' frames are not the size compared at")
        test.start()
        ref.start()
        test_base, ref_base = test.wait_time_base(), ref.wait_time_base()
        stop = gpu_frames.duration_in(duration_limit, test_base) if duration_limit else None

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
                    raise gpu_frames.GpuDecodeFailedError("the source has no frame for the test video's first")
                scorer.add_decoded(ref, ref_slot, test, test_slot)
                count += 1
                now = time.perf_counter()
                if on_progress is not None and now - reported >= 0.25:
                    reported = now
                    on_progress(count, total_frames, count / (now - started))
        finally:
            pairs.close()
        test.verify()
        ref.verify()
        return scorer.finish()
    finally:
        if scorer is not None:  # first: it gives back the memory the decoders page-locked
            scorer.close()
        test.close()
        ref.close()


# ------------------------------------------------------------- frame feed

class _PipeReader(threading.Thread):
    """Reads one of FFmpeg's raw outputs, a named pipe this process serves,
    frame by frame into a few reusable buffers."""

    def __init__(self, name: str, frame_bytes: int) -> None:
        super().__init__(name=f"vmaf-gpu-{name}", daemon=True)
        self.path = rf"\\.\pipe\vml-vmaf-{os.getpid()}-{uuid.uuid4().hex[:12]}-{name}"
        # _pipe, not _handle: Thread has a _handle of its own since Python
        # 3.13, which start() needs ("'handle' must be a _ThreadHandle").
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
        try:
            try:
                _winapi.ConnectNamedPipe(self._pipe, _winapi.NULL)
            except OSError as error:
                # ERROR_PIPE_CONNECTED: FFmpeg was first. ERROR_NO_DATA: it
                # came, wrote and went before this connected -- what it
                # wrote is still there to read. Taken for an error, a short
                # run on a busy machine lost all of it.
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
                # The feeder may stop early (the shorter video ended):
                # FFmpeg still writes the rest of this one, read and dropped.
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


# ------------------------------------------- frames with their timestamps

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.PeekNamedPipe.restype = wintypes.BOOL
_kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                    ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_kernel32.ReadFile.restype = wintypes.BOOL
_kernel32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                               ctypes.c_void_p]
#: The variable that takes the pairing back into FFmpeg ("canvas"), for
#: comparing the two ways.
PIPES_VARIABLE = "VML_VMAF_PIPES"
_LISTING_TB = re.compile(r"#tb 0: (\d+)/(\d+)")


@functools.cache
def pairs_in_app() -> bool:
    """Whether the frame pairs are worked out here (frame_sync) from each
    video's own frames and timestamps -- _StreamReader -- and not by
    FFmpeg's overlay on a canvas of both (vmaf_runner._gpu_pairs_stage),
    which cost FFmpeg about four times the CPU. Each video then has an
    FFmpeg of its own: in one, the video that decodes faster fills its pipe
    while its frames wait here for the other's, and FFmpeg's one filter
    thread, held by that pipe, passes no frame of the other's -- neither
    side moves again. It takes an FFmpeg that keeps the filters' timestamps
    in an output (-enc_time_base filter, FFmpeg 6.1), which is asked once."""
    if os.environ.get(PIPES_VARIABLE, "").casefold() == "canvas":
        return False
    from vmaf_app.core import proc as proc_util
    from vmaf_app.core.ffmpeg_locate import ffmpeg_path

    command = [ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
               "color=s=16x16:r=1:d=1", "-frames:v", "1", "-fps_mode", "passthrough", "-enc_time_base", "filter",
               "-c:v", "wrapped_avframe", "-f", "framecrc", "-"]
    try:
        done = proc_util.run(command, capture_output=True, timeout=30)
    except Exception:  # no FFmpeg, or one that does not answer: FFmpeg pairs them, as before
        return False
    return done.returncode == 0 and b"#tb 0:" in done.stdout


class _StreamReader:
    """One of the two videos as it leaves FFmpeg's filter chain: its frames,
    a YUV4MPEG stream on one named pipe, and their timestamps, a framecrc
    listing on another (-enc_time_base filter: the filters' own time base
    and timestamps, which are what libvmaf's frame sync pairs by). pull()
    gives the next (frame, timestamp) to frame_sync.frame_pairs.

    YUV4MPEG, not rawvideo: FFmpeg writes its frames' rows straight to the
    pipe, where the rawvideo encoder first copies each frame into a packet
    it allocates. Each frame is left to gather in the pipe and then taken
    in one read: read as it comes, it arrives in about 300 pieces, each a
    wake-up of this thread (2 ms of CPU a 4K frame against 4)."""

    def __init__(self, name: str, frame_bytes: int) -> None:
        stem = rf"\\.\pipe\vml-vmaf-{os.getpid()}-{uuid.uuid4().hex[:12]}-{name}"
        self.path, self.listing_path = stem, stem + "-times"
        self.frame_bytes = frame_bytes
        self._pipe = _winapi.CreateNamedPipe(self.path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT, 1,
                                             _PIPE_BYTES, _PIPE_BYTES, 0, _winapi.NULL)
        self._listing = _winapi.CreateNamedPipe(self.listing_path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,
                                                1, 65536, 65536, 0, _winapi.NULL)
        self.free: queue.Queue[bytearray | None] = queue.Queue()
        # A frame held and the one after it (frame_pairs), one read ahead and one being read.
        for _ in range(_READ_AHEAD + 1):
            self.free.put(bytearray(frame_bytes))
        self._pixels: queue.Queue[bytearray | None] = queue.Queue()
        self._stamps: queue.Queue[int | None] = queue.Queue()
        self.time_base: Fraction | None = None
        self._time_base_known = threading.Event()
        self.error: BaseException | None = None
        self._stopped = threading.Event()
        self._writer_gone = threading.Event()
        self._threads = [threading.Thread(target=self._read_frames, name=f"vmaf-gpu-{name}", daemon=True),
                         threading.Thread(target=self._read_listing, name=f"vmaf-gpu-{name}-times", daemon=True)]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _connect(pipe: int) -> None:
        try:
            _winapi.ConnectNamedPipe(pipe, _winapi.NULL)
        except OSError as error:
            # As _PipeReader: FFmpeg was first, or came, wrote and went.
            if error.winerror not in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
                raise

    def _gather(self, count: int) -> None:
        """Waits until `count` bytes are in the pipe (or all it holds), or
        FFmpeg has gone."""
        count = min(count, _PIPE_BYTES // 2)
        available = wintypes.DWORD()
        while not self._stopped.is_set():
            if not _kernel32.PeekNamedPipe(self._pipe, None, 0, None, ctypes.byref(available), None):
                return  # closed and empty: the read says so
            if available.value >= count or self._writer_gone.is_set():
                return
            time.sleep(0.001)

    def _read(self, buffer: bytearray) -> bool:
        """Fills `buffer`; False at the end of the stream."""
        address = ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer))
        filled, got = 0, wintypes.DWORD()
        while filled < len(buffer):
            if not _kernel32.ReadFile(self._pipe, address + filled, len(buffer) - filled, ctypes.byref(got), None) \
                    or not got.value:
                if filled:
                    raise VmafGpuError(f"FFmpeg's frames ended inside one ({filled} of {len(buffer)} bytes)")
                return False
            filled += got.value
        return True

    def _read_frames(self) -> None:
        try:
            self._connect(self._pipe)
            if self._stopped.is_set():
                return
            # "YUV4MPEG2 W.. H.. ...\n", then "FRAME\n" and the planes, packed, for each frame.
            header, byte = bytearray(), bytearray(1)
            while byte != b"\n" and len(header) < 1024 and self._read(byte):
                header += byte
            if header and not header.startswith(b"YUV4MPEG2 "):
                raise VmafGpuError("FFmpeg's frames are not a YUV4MPEG stream")
            marker = bytearray(6)
            while header and not self._stopped.is_set():
                self._gather(len(marker) + self.frame_bytes)
                if not self._read(marker):
                    break
                if marker != b"FRAME\n":
                    raise VmafGpuError("FFmpeg's frames are not where they were expected")
                buffer = self.free.get()
                if buffer is None:
                    break
                if not self._read(buffer):
                    raise VmafGpuError("FFmpeg's frames ended after a frame's start")
                self._pixels.put(buffer)
            # The comparison may end before this video does: the rest is read and dropped.
            spare = bytearray(4 * 1024 * 1024)
            while not self._stopped.is_set():
                self._gather(len(spare))
                try:
                    if not self._read(spare):
                        break
                except VmafGpuError:
                    break  # the tail, shorter than the buffer
        except BaseException as error:  # raised by pull()
            self.error = error
        finally:
            _winapi.CloseHandle(self._pipe)
            self._pixels.put(None)

    def _read_listing(self) -> None:
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
        """The time base of pull()'s timestamps; any, for a video without a frame."""
        self._time_base_known.wait()
        return self.time_base or Fraction(1, 1000)

    def pull(self) -> tuple[bytearray, int] | None:
        """The next frame and its timestamp; None at the video's end."""
        buffer = self._pixels.get()
        if buffer is None:
            self._pixels.put(None)
            if self.error is not None:
                raise self.error
            return None
        stamp = self._stamps.get()
        if stamp is None:
            self._stamps.put(None)
            raise VmafGpuError("FFmpeg gave a frame without its timestamp")
        return buffer, stamp

    def release(self, buffer: bytearray) -> None:
        self.free.put(buffer)

    def drop_rest(self) -> None:
        """The comparison is over: what FFmpeg still writes is read and dropped."""
        self.free.put(None)

    def ffmpeg_ended(self) -> None:
        """FFmpeg has gone: no wait for more, and a pipe it never opened --
        an output that got no frame is never opened -- is connected to here,
        which ends the wait for it (see _PipeReader.release_if_unconnected)."""
        self._writer_gone.set()
        for thread, path in zip(self._threads, (self.path, self.listing_path), strict=True):
            if thread.is_alive():
                try:
                    with open(path, "wb"):
                        pass
                except OSError:
                    pass  # FFmpeg's already, or the pipe is gone

    def stop(self) -> None:
        self._stopped.set()
        self.free.put(None)
        self.ffmpeg_ended()

    def join(self) -> None:
        for thread in self._threads:
            thread.join()
        if self.error is not None:
            raise self.error


class GpuAttempt:
    """One FFmpeg run's GPU half: the pipes FFmpeg writes the two videos'
    frames to, and the threads that read them and feed libvmaf (`backend`
    "cuda") or the Vulkan port of its features ("vulkan", on Vulkan's GPU
    number `device`). `paired`: the frames come paired by FFmpeg, on two
    pipes of raw video (_PipeReader); by default (pairs_in_app) each video
    comes as it is, with its timestamps, and is paired here (_StreamReader)."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int,
                 backend: str = "cuda", device: int | None = None, paired: bool | None = None,
                 threads: int = 0, frame_rate: int = 0):
        """`backend` "cpu": `models` are CPU_FEATURES keys, scored by
        CpuScorer with `threads` of libvmaf's (XPSNR's `frame_rate`)."""
        self.paired = not pairs_in_app() if paired is None else paired
        if backend == "cpu":
            self._scorer = CpuScorer(width, height, bit_depth, tuple(models), n_subsample, threads, frame_rate)
        elif "vmaf_v1" in models:
            from vmaf_app.core import vmaf_v1_gpu

            self._scorer = vmaf_v1_gpu.MultiScorer(width, height, bit_depth, models, n_subsample, backend, device)
        elif backend == "vulkan":
            from vmaf_app.core import vmaf_vulkan

            self._scorer = vmaf_vulkan.VulkanScorer(width, height, bit_depth, models, n_subsample, device=device)
        else:
            self._scorer = GpuScorer(width, height, bit_depth, models, n_subsample)
        reader = _PipeReader if self.paired else _StreamReader
        self.distorted = reader("distorted", self._scorer.frame_bytes)
        self.reference = reader("reference", self._scorer.frame_bytes)
        self.error: BaseException | None = None
        self._result = None
        self._feeder = threading.Thread(target=self._feed if self.paired else self._feed_streams,
                                        name="vmaf-gpu-feed", daemon=True)
        self.distorted.start()
        self.reference.start()
        self._feeder.start()

    def _feed_streams(self) -> None:
        """Pairs the two videos' frames as libvmaf's filter would (frame_sync)
        and scores each pair."""
        try:
            test, ref = self.distorted, self.reference
            pairs = frame_pairs(test.pull, ref.pull, test.wait_time_base(), ref.wait_time_base(),
                                test.release, ref.release)
            try:
                for test_frame, ref_frame, _when in pairs:
                    if ref_frame is None:
                        raise VmafGpuError("the source has no frame for the test video's first")
                    self._scorer.add(ref_frame, test_frame)
            finally:
                pairs.close()
            for reader in (test, ref):
                reader.drop_rest()
            for reader in (test, ref):
                reader.join()
            self._result = self._scorer.finish()
        except BaseException as error:  # finish() raises it in the run
            self.error = error
            self.stop()

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
        elif self.paired:
            self.distorted.release_if_unconnected()
            self.reference.release_if_unconnected()
        else:
            self.distorted.ffmpeg_ended()
            self.reference.ffmpeg_ended()
        self._feeder.join()
        try:
            if self.error is not None:
                raise VmafGpuError(f"GPU VMAF failed: {self.error}") from self.error
            return self._result if ffmpeg_succeeded else None
        finally:
            self._scorer.close()

    def output_args(self, duration_limit: float) -> list[str]:
        """FFmpeg's outputs for the graph's [vmaf_dist] and [vmaf_ref]: every
        frame as it comes (no frame rate conversion). Paired by FFmpeg: two
        of raw video. Paired here: (the test video's FFmpeg's, the
        source's), each its frames as YUV4MPEG and, from their [.._ts] copy,
        their timestamps as a listing; the source runs a second past the
        limit, for the frame nearest the test video's last to be among them."""
        args = []
        if not self.paired:
            both = []
            for label, reader, past in (("vmaf_dist", self.distorted, 0.0), ("vmaf_ref", self.reference, 1.0)):
                cut = ["-t", f"{duration_limit + past:.3f}"] if duration_limit > 0 else []
                both.append(["-map", f"[{label}]", "-fps_mode", "passthrough", *cut, "-flush_packets", "1",
                             "-f", "yuv4mpegpipe", "-strict", "-1", reader.path,
                             "-map", f"[{label}_ts]", "-fps_mode", "passthrough", "-enc_time_base", "filter", *cut,
                             "-c:v", "wrapped_avframe", "-flush_packets", "1", "-f", "framecrc", reader.listing_path])
            return tuple(both)
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
    gpu_frames crops as FFmpeg's crop filter does, shifts 10-bit samples and
    widens 8-bit ones as FFmpeg converts them, and scales on the GPU with the
    same filter (not to the sample: the user decided a comparison scaled any
    way is the same one); frame_sync.frame_pairs pairs as the overlay does,
    and a pair is scored while the test frame's time from the first is below
    `duration_limit` (the outputs' -t, as text) in the test video's time
    base, as FFmpeg's trim filter cuts.

    GpuDecodeUnavailableError when the videos are not decoded here (FFmpeg then
    decodes them, as before): another decoder, a codec or size it does not
    take. GpuDecodeFailedError when decoding failed after the start -- the run
    is then made again through FFmpeg."""
    plans = []
    for info, crop in ((distorted, distorted_crop), (source, source_crop)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height),
                                        algorithm=scale_algorithm)
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=gpu_frames.WIDEN_REPEAT if full else gpu_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise gpu_frames.GpuDecodeUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = gpu_frames.decoder_supports(_GPU, plan)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
    # The decoders first: the first to start CUDA sets its waits to sleep
    # rather than spin (gpu_frames), and libvmaf's then do too.
    test = gpu_frames.GpuFrameStream(distorted, plans[0], _GPU, pool=4, process_handle=process_handle)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], _GPU, pool=4, process_handle=process_handle)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        scorer = GpuScorer(width, height, bit_depth, models, n_subsample, on_device=True)
        test.start()
        ref.start()
        test_base, ref_base = test.wait_time_base(), ref.wait_time_base()
        stop = gpu_frames.duration_in(duration_limit, test_base) if duration_limit else None

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
                    raise gpu_frames.GpuDecodeFailedError("the source has no frame for the test video's first")
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

def gpu_models(compute_vmaf: bool, compute_vmaf_neg: bool, model: str,
               compute_vmaf_v1: bool = False, model_v1: str = "") -> dict[str, str] | None:
    """{log name: libvmaf model} for a run's VMAF, NEG and VMAF v1 on the
    GPU, or None when it scores none of them. VMAF with a custom model file
    is calculated on the CPU, and VMAF NEG with it; VMAF v1 ("vmaf_v1": its
    model's "path=<file>", for vmaf_v1_gpu) has bundled models only."""
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
#: What the probe chose: "cuda" or "vulkan", and Vulkan's GPU number.
_backend: tuple[str, int | None] = ("cuda", None)
#: Settings > GPU metrics > GPU backend, which VMAF follows as Vship does:
#: "cuda" and "auto" take libvmaf's CUDA code where it runs (an NVIDIA GPU)
#: and Vulkan elsewhere; "vulkan" takes Vulkan on every GPU. "hip" is Vship's
#: build for AMD: VMAF has none, and takes Vulkan there as with "auto".
_preference = "auto"


def set_gpu_backend(preference: str) -> None:
    """Follows the GPU backend setting; the GPU is probed again when the
    choice changes what is tried first."""
    global _preference, _probed
    with _PROBE_LOCK:
        changed = _backend_order(preference) != _backend_order(_preference)
        _preference = preference
        if changed and _probed is not None:
            _probed = None
            _log.info("VMAF on the GPU: GPU backend set to %s", preference)


def _backend_order(preference: str) -> tuple[str, ...]:
    return ("vulkan", "cuda") if preference == "vulkan" else ("cuda", "vulkan")


def gpu_vmaf_available() -> tuple[bool, str]:
    """Whether VMAF is scored on this PC's GPU, and with what (or why not).
    Probed once per backend choice, in a process of its own (and again
    after a failure, see forget_failed_probe)."""
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
    """("cuda", None) or ("vulkan", Vulkan's GPU number): what a run that
    scores VMAF on the GPU uses. Meaningful when gpu_vmaf_available()."""
    gpu_vmaf_available()
    return _backend


def _probe_once(preference: str = "auto") -> tuple[str | None, int | None, str]:
    """(backend, Vulkan's GPU number, what it is) for the first backend, in
    the order the setting gives, that scores here; (None, None, why not)
    when neither does."""
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
            # Any failure is "not on this GPU": one that escaped here left the
            # probe unanswered and failed every video's setup with it, instead
            # of calculating VMAF on the CPU.
            _log.warning("The %s VMAF probe failed", backend, exc_info=error)
            reasons.append(f"{backend}: the GPU probe failed: {error}")
    return None, None, "; ".join(reasons)


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
                  enabled: bool = True, bit_depth: int = 8,
                  size: tuple[int, int] | None = None,
                  compute_vmaf_v1: bool = False, model_v1: str = "") -> dict[str, str] | None:
    """The models a run scores on the GPU (gpu_models), or None when its
    VMAF is calculated on the CPU:
    - the video set to CPU (`enabled`, its VmafOptions.vmaf_on_gpu);
    - a custom model;
    - a comparison deeper than 10 bits (`bit_depth`: the frames libvmaf
      pairs reach the GPU through FFmpeg's overlay, which holds 8 and 10
      bits);
    - a comparison at an odd width or height (`size`, where it is known):
      the pairs cross that overlay side by side on one 4:2:0 canvas
      (vmaf_runner._gpu_pairs_stage), and FFmpeg's pad, which makes it,
      keeps a 4:2:0 picture on whole chroma samples -- it gives an even
      size and blacks out an odd picture's last column and row. The
      attempt used to be made, fail in FFmpeg, and VMAF be calculated
      again on the CPU after a "VMAF on the GPU failed";
    - no GPU that scores it (CUDA or Vulkan).
    VMAF v1 (`compute_vmaf_v1`, its model `model_v1`) is among them where
    vmaf_v1_gpu calculates it here; what the GPU does not score is left out,
    and FFmpeg's libvmaf calculates it."""
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
