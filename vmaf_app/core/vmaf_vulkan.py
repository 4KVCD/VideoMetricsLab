"""VMAF and VMAF NEG on any GPU with Vulkan: the features VMAF is predicted
from (VIF, ADM and motion) are calculated by tools/vmaf_vulkan/vmaf_vulkan.dll
(libvmaf-fast's fast/vulkan, from its release like libvmaf.dll), a port of
libvmaf's CUDA feature extractors to Vulkan compute shaders.

The port gives the same feature values as libvmaf's CUDA code (vmaf_cuda), to
the last bit, on any GPU: its shaders use integer arithmetic only. The score
is then predicted from them by libvmaf itself (the bundled libvmaf.dll, on the
CPU: vmaf_import_feature_score and vmaf_score_at_index), so a score is the
same number whichever of the two calculated the features.

VulkanScorer takes the frames GpuScorer takes and returns what it returns.
VMAF and VMAF NEG together cost little more than one of them: the library
calculates what the two share once.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vmaf_app.core import vmaf_cuda

_log = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "tools" / "vmaf_vulkan" / "vmaf_vulkan.dll"
#: What a score records it was calculated with (its provenance).
LIBRARY_BUILD = f"libvmaf-fast {vmaf_cuda.LIBVMAF_FAST_VERSION} (Vulkan)"

#: Positions in the library's feature rows (vmaf_vulkan.cpp).
_VIF, _ADM2, _MOTION2, _VIF_NEG, _ADM2_NEG, _MOTION = 0, 4, 5, 6, 10, 11
FEATURE_COUNT = 12
#: libvmaf's names for a model's features -> their position in a row.
_MODEL_FEATURES = {
    False: {"VMAF_integer_feature_adm2_score": _ADM2, "VMAF_integer_feature_motion2_score": _MOTION2,
            **{f"VMAF_integer_feature_vif_scale{scale}_score": _VIF + scale for scale in range(4)}},
    # VMAF NEG's features carry their enhancement gain limit in the name.
    True: {"integer_adm2_egl_1": _ADM2_NEG, "VMAF_integer_feature_motion2_score": _MOTION2,
           **{f"integer_vif_scale{scale}_egl_1": _VIF_NEG + scale for scale in range(4)}},
}
_NEG_MODELS = ("vmaf_v0.6.1neg", "vmaf_4k_v0.6.1neg")

#: Vulkan's device types: the others are virtual GPUs and CPU rasterisers.
_TYPE_INTEGRATED, _TYPE_DISCRETE = 1, 2


class VmafVulkanError(vmaf_cuda.VmafGpuError):
    """VMAF could not be calculated with Vulkan; the CPU calculates it."""


@dataclass(frozen=True)
class VulkanDevice:
    index: int
    name: str
    vendor: int
    kind: int
    #: Has what the shaders need (64-bit integers).
    usable: bool
    has_double: bool


_library: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _library
    if _library is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        unsigned = ctypes.POINTER(ctypes.c_uint32)
        for name, restype, argtypes in (
            ("vv_device", ctypes.c_int, [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, unsigned, unsigned, unsigned]),
            ("vv_create", ctypes.c_int, [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
            ("vv_destroy", None, [handle]),
            ("vv_error", ctypes.c_char_p, []),
            ("vv_submit", ctypes.c_int,
             [handle, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]),
            ("vv_staging", ctypes.c_int, [handle, pointer, pointer, unsigned]),
            ("vv_commit", ctypes.c_int, [handle, ctypes.c_int]),
            ("vv_shared_next", ctypes.c_int, [handle, ctypes.POINTER(ctypes.c_int), unsigned, unsigned]),
            ("vv_export", ctypes.c_int, [handle, ctypes.c_int, pointer, ctypes.POINTER(ctypes.c_uint64)]),
            ("vv_flush", ctypes.c_int, [handle]),
            ("vv_features", ctypes.c_int, [handle, ctypes.c_uint, ctypes.POINTER(ctypes.c_double)]),
            ("vv_sums", ctypes.c_int, [handle, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]),
        ):
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes
        # libvmaf-fast 3.2.0-fast.1's has none: AMD's decoder then does not share its memory.
        if hasattr(lib, "vv_shared_device"):
            lib.vv_shared_device.restype = ctypes.c_int
            lib.vv_shared_device.argtypes = [handle, ctypes.c_char_p, ctypes.c_char_p, unsigned]
        _library = lib
    return _library


def _check(lib: ctypes.CDLL, error: int, what: str) -> None:
    if error < 0:
        raise VmafVulkanError(f"{what} failed: {(lib.vv_error() or b'').decode(errors='replace')}")


def devices() -> list[VulkanDevice]:
    """The GPUs Vulkan lists, in its order."""
    lib = _load()
    found = []
    name = ctypes.create_string_buffer(256)
    vendor, kind, flags = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    count = lib.vv_device(-1, None, 0, None, None, None)
    _check(lib, count, "Listing the GPUs")
    for index in range(count):
        lib.vv_device(index, name, len(name), ctypes.byref(vendor), ctypes.byref(kind), ctypes.byref(flags))
        found.append(VulkanDevice(index, name.value.decode(errors="replace"), vendor.value, kind.value,
                                  bool(flags.value & 1), bool(flags.value & 2)))
    return found


def best_device(found: list[VulkanDevice] | None = None) -> VulkanDevice | None:
    """The GPU to score on: a discrete one before an integrated one."""
    usable = [device for device in (devices() if found is None else found)
              if device.usable and device.kind in (_TYPE_DISCRETE, _TYPE_INTEGRATED)]
    usable.sort(key=lambda device: (device.kind != _TYPE_DISCRETE, device.index))
    return usable[0] if usable else None


#: vv_create's flag for a context whose frames come from GPU memory (SharedLumas).
SHARED_FLAG = 1 << 19


@dataclass(frozen=True)
class SharedDevice:
    """What a decoder importing a context's shared buffers on its own Vulkan
    device must match (vv_shared_device): the GPU and the driver -- Vulkan
    imports the memory only where both are the exporter's -- and the memory
    type to allocate it as."""

    device_uuid: bytes
    driver_uuid: bytes
    memory_type: int


class SharedLumas:
    """The buffers a Vulkan context made with SHARED_FLAG copies each frame
    pair's luma planes from, as GPU memory a decoder on the same GPU writes
    them into (gpu_frames' copy_luma): the planes then never leave the GPU,
    where they used to be copied to system memory and back by the CPU.
    `stream`: the decoder that imports them (GpuFrameStream) -- with CUDA
    (NVIDIA's), or into a Vulkan device of its own (AMD's), which takes the
    SharedDevice vv_shared_device names: import_memory(handle, size,
    exporter), exporter None from an engine without vv_shared_device.
    VmafVulkanError when the two cannot share memory -- another GPU, an old
    driver -- and the caller then scores from system memory as before."""

    def __init__(self, lib: ctypes.CDLL, context: ctypes.c_void_p, stream) -> None:
        self._lib, self._context, self._stream = lib, context, stream
        self._imports: list[tuple[int, int]] = []  # per slot: (address, what unimport takes)
        handle, size = ctypes.c_void_p(), ctypes.c_uint64()
        try:
            exporter = None  # AMD's decoder imports nothing without it (GpuFrameStream.import_memory)
            if hasattr(lib, "vv_shared_device"):
                device, driver = ctypes.create_string_buffer(16), ctypes.create_string_buffer(16)
                memory_type = ctypes.c_uint32()
                _check(lib, lib.vv_shared_device(context, device, driver, ctypes.byref(memory_type)),
                       "Sharing Vulkan's memory")
                exporter = SharedDevice(device.raw, driver.raw, memory_type.value)
            while lib.vv_export(context, len(self._imports), ctypes.byref(handle), ctypes.byref(size)) == 0:
                try:
                    imported = stream.import_memory(handle.value, size.value, exporter)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
                if imported is None:
                    raise VmafVulkanError("the decoder cannot write into Vulkan's memory")
                self._imports.append(imported)
            if not self._imports:
                raise VmafVulkanError(f"Vulkan's memory is not shared: {(lib.vv_error() or b'').decode(errors='replace')}")
            #: The engine's slots: a frame begun (next) waits for the one this
            #: many before it to be done.
            self.slots = len(self._imports)
        except BaseException:
            self.close()
            raise

    def next(self) -> tuple[int, int, int]:
        """Where the next pair's planes go: (the reference's address, the
        distorted's, the bytes from one row to the next). vv_commit scores
        them once both are written."""
        slot, stride, plane = ctypes.c_int(), ctypes.c_uint32(), ctypes.c_uint32()
        _check(self._lib, self._lib.vv_shared_next(self._context, ctypes.byref(slot), ctypes.byref(stride),
                                                   ctypes.byref(plane)), "Scoring a frame")
        address = self._imports[slot.value][0]
        return address, address + plane.value, stride.value

    def close(self) -> None:
        """Before the decoder and the context are closed."""
        for _address, memory in self._imports:
            self._stream.unimport(memory)
        self._imports = []


class VulkanScorer:
    """VMAF's features of frame pairs given in order, calculated with Vulkan:
    the luma and chroma planes of 4:2:0 frames, packed as FFmpeg's rawvideo
    writes them, at `bit_depth` bits (16-bit little-endian samples above 8).
    Only the luma is used, as by VMAF. `shared`: the frames come from GPU
    memory instead (add_shared), from the decoder `shared` is."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 device: int | None = None, native_double: bool = False, in_flight: int = 3, skip: int = 0, pass_limit: int = 0,
                 decouple_variant: int = 0, shared=None):
        self._shared: SharedLumas | None = None
        self._lib = lib = _load()
        self._vmaf = vmaf_cuda._load()
        self._models = dict(models)
        self._step = max(1, n_subsample)
        self._count = 0
        self._context = ctypes.c_void_p()
        if device is None:
            chosen = best_device()
            if chosen is None:
                raise VmafVulkanError("no GPU that Vulkan can calculate VMAF on")
            device = chosen.index
        flags = ((1 if native_double else 0) | (max(1, min(16, in_flight)) << 8) | ((skip & 7) << 16)
                 | ((pass_limit & 0xFF) << 20) | ((decouple_variant & 7) << 28))
        if shared is not None:
            flags |= SHARED_FLAG
        _check(lib, lib.vv_create(ctypes.byref(self._context), device, width, height, bit_depth, flags),
               "Starting Vulkan")
        if shared is not None:
            try:
                self._shared = SharedLumas(lib, self._context, shared)
            except BaseException:
                self.close()
                raise
        sample = 1 if bit_depth <= 8 else 2
        self._luma_stride = width * sample
        self._luma_bytes = width * height * sample
        chroma = ((width + 1) // 2) * ((height + 1) // 2) * sample
        self.frame_bytes = width * height * sample + 2 * chroma

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before)."""
        score = self._count % self._step == 0
        if min(len(reference), len(distorted)) < self._luma_bytes:  # the library reads that much of each
            raise VmafVulkanError(f"frame {self._count} is shorter than its luma plane")
        ref = (ctypes.c_char * len(reference)).from_buffer(reference)
        dist = (ctypes.c_char * len(distorted)).from_buffer(distorted)
        _check(self._lib, self._lib.vv_submit(self._context, ctypes.addressof(ref), self._luma_stride,
                                              ctypes.addressof(dist), self._luma_stride, int(score)),
               f"Scoring frame {self._count}")
        self._count += 1

    def add_decoded(self, reference, distorted) -> None:
        """Scores one more pair that a decoder writes itself: `reference` and
        `distorted` are each given the address to write a luma plane to,
        rows packed (gpu_frames' download of a luma-only plan), and return
        once it is written. The planes go straight into the memory the GPU
        copies them from; a frame that is not scored (n_subsample) needs the
        reference alone, for motion."""
        score = self._count % self._step == 0
        ref, dist, stride = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_uint32()
        _check(self._lib, self._lib.vv_staging(self._context, ctypes.byref(ref), ctypes.byref(dist),
                                               ctypes.byref(stride)), f"Scoring frame {self._count}")
        for fill, address in ((reference, ref.value), (distorted, dist.value)) if score else ((reference, ref.value),):
            if stride.value == self._luma_stride:
                fill(address)
            else:  # rows padded to four bytes: an odd width
                packed = np.empty(self._luma_bytes, dtype=np.uint8)
                fill(packed.ctypes.data)
                rows = self._luma_bytes // self._luma_stride
                target = np.ctypeslib.as_array(ctypes.cast(address, ctypes.POINTER(ctypes.c_uint8)),
                                               shape=(rows, stride.value))
                target[:, :self._luma_stride] = packed.reshape(rows, self._luma_stride)
        _check(self._lib, self._lib.vv_commit(self._context, int(score)), f"Scoring frame {self._count}")
        self._count += 1

    def add_shared(self, reference, distorted) -> None:
        """Scores one more pair a decoder copies on the GPU: `reference` and
        `distorted` are each given (the GPU address to copy a luma plane to,
        the bytes between its rows), and return once it is there."""
        score = self._count % self._step == 0
        ref, dist, pitch = self._shared.next()
        reference(ref, pitch)
        if score:
            distorted(dist, pitch)
        _check(self._lib, self._lib.vv_commit(self._context, int(score)), f"Scoring frame {self._count}")
        self._count += 1

    def features(self) -> tuple[np.ndarray, np.ndarray]:
        """The frame numbers scored and their feature rows (FEATURE_COUNT
        doubles each), once every frame is in."""
        _check(self._lib, self._lib.vv_flush(self._context), "Scoring")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        rows = np.zeros((len(frames), FEATURE_COUNT), dtype=np.float64)
        for slot, frame in enumerate(frames):
            row = rows[slot].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _check(self._lib, self._lib.vv_features(self._context, int(frame), row), f"Reading frame {frame}")
        return frames, rows

    def sums(self, frame: int) -> np.ndarray:
        """A frame's raw sums, for tests."""
        out = np.zeros(128, dtype=np.uint64)
        count = self._lib.vv_sums(self._context, frame, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), 128)
        _check(self._lib, count, f"Reading frame {frame}")
        return out[:count]

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """As GpuScorer.finish: the frame numbers scored and each model's
        scores for them, rounded as libvmaf's JSON log rounds them."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {name: np.zeros(0) for name in self._models}
        frames, rows = self.features()
        return frames, {name: predict(version, frames, rows) for name, version in self._models.items()}

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        if self._context:
            self._lib.vv_destroy(self._context)
            self._context = ctypes.c_void_p()


def predict(version: str, frames: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """A libvmaf model's scores for feature rows: libvmaf's own prediction,
    from the features under the names its extractors would give them."""
    lib = vmaf_cuda._load()
    lib.vmaf_import_feature_score.restype = ctypes.c_int
    lib.vmaf_import_feature_score.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint]
    context, model = ctypes.c_void_p(), ctypes.c_void_p()
    configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0)
    vmaf_cuda._check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
    try:
        config = vmaf_cuda._ModelConfig(b"vmaf", 0)
        vmaf_cuda._check(lib.vmaf_model_load(ctypes.byref(model), ctypes.byref(config), version.encode()),
                         f"Loading the {version} model")
        columns = _MODEL_FEATURES[version in _NEG_MODELS]
        for name, column in columns.items():
            encoded = name.encode()
            for frame, row in zip(frames, rows, strict=True):
                vmaf_cuda._check(lib.vmaf_import_feature_score(context, encoded, float(row[column]), int(frame)),
                                 f"Giving libvmaf frame {frame}")
        scores = np.empty(len(frames), dtype=np.float64)
        value = ctypes.c_double()
        for slot, frame in enumerate(frames):
            vmaf_cuda._check(lib.vmaf_score_at_index(context, model, ctypes.byref(value), int(frame)),
                             f"Predicting frame {frame}")
            scores[slot] = float(f"{value.value:.6f}")
        return scores
    finally:
        if model:
            lib.vmaf_model_destroy(model)
        lib.vmaf_close(context)


# ------------------------------------------------------------ availability

#: Vulkan's number of the GPU to score on, instead of the one best_device()
#: picks: for testing an integrated GPU on a PC that also has a discrete one.
DEVICE_VARIABLE = "VML_VMAF_VULKAN_DEVICE"
#: SHA-256 of the sums the probe's frames give (8-bit, then 10-bit), which
#: are the sums behind feature values identical to libvmaf's CUDA code's
#: (tests/test_vmaf_vulkan.py compares them where CUDA runs).
#: VMAF v0.6.1's sums in the engine's accumulator, which the probe checks:
#: motion's, VIF's (4 scales of 9), ADM's denominators (4 x 3) and its
#: numerators (2 limits x 4 scales x 3); VMAF v1's CAMBI follows them.
_PROBE_SLOTS = 1 + 4 * 9 + 4 * 3 + 2 * 4 * 3
_PROBE_SUMS = ("33668b74708340de30734d996e3b60ea025c4c55a5d977707ea6482467f85dee",
               "76f6f75034298e831c97642952e3682d896b85713efddf8e32d65a19c09bc3b1")
_PROBE_SIZE = (320, 192)


def probe_frames(bits: int, count: int = 3) -> tuple[list[bytearray], list[bytearray]]:
    """The probe's frame pairs: flat, smooth and textured areas, blurred,
    noisy and sharpened in turn, so every branch of the shaders is taken.
    Made with integer arithmetic only: the same bytes on every PC."""
    width, height = _PROBE_SIZE
    peak = (1 << bits) - 1
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    chroma = np.full(2 * (width // 2) * (height // 2), peak // 2, dtype=np.uint8 if bits == 8 else "<u2").tobytes()
    reference, distorted = [], []
    for index in range(count):
        noise = ((x * 2654435761 + y * 40503 + index * 69069) * 1103515245 >> 16) & 0xFF
        ramp = (x * 3 + y * 2 + index * 5) % 256
        blocks = ((x // 16 + y // 16 + index) % 2) * 160 + 40
        ref = np.where(x < width // 3, blocks, np.where(x < 2 * width // 3, ramp, (ramp + noise) // 2))
        blurred = ref.copy()
        blurred[:, 1:-1] = (ref[:, :-2] + 2 * ref[:, 1:-1] + ref[:, 2:]) // 4
        dis = (blurred + (noise % 7) - 3 if index % 3 == 0
               else 2 * ref - blurred if index % 3 == 1  # sharpened: a gain above 1, which VMAF NEG limits
               else (blurred * 7 + noise) // 8)
        for frames, picture in ((reference, ref), (distorted, dis)):
            samples = np.clip(picture, 0, 255) * peak // 255
            frames.append(bytearray(samples.astype(np.uint8 if bits == 8 else "<u2").tobytes() + chroma))
    return reference, distorted


def probe_sums(device: int, bits: int) -> str:
    """The SHA-256 of the sums a GPU gives for the probe's frames."""
    reference, distorted = probe_frames(bits)
    scorer = VulkanScorer(*_PROBE_SIZE, bits, {}, device=device)
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        scorer.features()
        digest = hashlib.sha256()
        for frame in range(len(reference)):
            # VMAF v0.6.1's sums: the engine keeps VMAF v1's after them.
            digest.update(scorer.sums(frame)[:_PROBE_SLOTS].astype("<u8").tobytes())
        return digest.hexdigest()
    finally:
        scorer.close()


def probe() -> tuple[bool, int | None, str]:
    """(whether Vulkan scores VMAF on this PC, on which GPU, what it is or
    why not). Scores the probe's frames at 8 and 10 bits and accepts the GPU
    only if its sums are exactly the known ones: a driver that compiles a
    shader wrongly (Intel's and AMD's did, see libvmaf-fast's README) gives wrong
    scores, not an error. Run in a process of its own."""
    if not LIBRARY_PATH.is_file():
        return False, None, "vmaf_vulkan.dll is not bundled"
    if not vmaf_cuda.LIBRARY_PATH.is_file():
        return False, None, "libvmaf, which predicts the score, is not bundled"
    try:
        found = devices()
        override = os.environ.get(DEVICE_VARIABLE, "")
        device = (next((d for d in found if str(d.index) == override.strip()), None) if override.strip()
                  else best_device(found))
        if device is None or not device.usable:
            return False, None, "no GPU that Vulkan can calculate VMAF on"
        for bits, expected in zip((8, 10), _PROBE_SUMS, strict=True):
            if probe_sums(device.index, bits) != expected:
                return False, None, f"{device.name}'s driver calculates VMAF wrongly ({bits}-bit self-test)"
        return True, device.index, f"Vulkan on {device.name}"
    except (OSError, vmaf_cuda.VmafGpuError) as error:
        return False, None, str(error)


# ------------------------------------------------- videos decoded here

#: gpu_frames' decoder library for a GPU, by its maker (Vulkan's vendor ID).
_DECODERS = {0x10DE: "nvidia", 0x8086: "intel", 0x1002: "amd"}


def score_decoded(
    source, distorted, source_crop=None, distorted_crop=None, *, width: int, height: int, bit_depth: int,
    models: dict[str, str], n_subsample: int, duration_limit: str | None, total_frames: int,
    scale_algorithm: str = "bicubic", on_progress=None, check_cancel=lambda: None, process_handle=None,
    device: int | None = None, decoder: str | None = None,
):
    """vmaf_cuda.score_decoded with the Vulkan scorer: both videos decoded
    in this process by the GPU's own decoder (gpu_frames), cropped, scaled
    and widened there, paired as libvmaf's filter pairs them (frame_sync),
    and each frame's luma written by the decoder straight into the memory
    Vulkan copies it to the GPU from. FFmpeg only copies the compressed
    streams out of their containers. `decoder`: gpu_frames' library
    ("nvidia", "intel", "amd", "software"), one for both videos or (the
    source's, the test video's); by default the one of the GPU scored on."""
    import time
    from dataclasses import replace

    from vmaf_app.core import gpu_frames
    from vmaf_app.core.frame_sync import frame_pairs

    found = devices()
    chosen = best_device(found) if device is None else next((d for d in found if d.index == device), None)
    if chosen is None:
        raise VmafVulkanError("no GPU that Vulkan can calculate VMAF on")
    source_backend, test_backend = gpu_frames.decoder_pair(decoder or _DECODERS.get(chosen.vendor) or "")
    for library in (source_backend, test_backend):
        if not library or not gpu_frames.available(library):
            raise gpu_frames.GpuDecodeUnavailableError(f"no decoder library for {chosen.name}")
    plans = []
    for info, crop, library in ((distorted, distorted_crop, test_backend), (source, source_crop, source_backend)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, luma_only=True, size=(width, height),
                                      algorithm=scale_algorithm)
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=gpu_frames.WIDEN_REPEAT if full else gpu_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise gpu_frames.GpuDecodeUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = gpu_frames.decoder_supports(0, plan, library)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
    # The decoders write each luma plane into memory of the size compared at.
    luma_bytes = width * height * (2 if bit_depth > 8 else 1)
    if any(plan.frame_bytes != luma_bytes for plan in plans):
        raise gpu_frames.GpuDecodeUnavailableError("the decoders' frames are not the size compared at")
    test = gpu_frames.GpuFrameStream(distorted, plans[0], 0, pool=4, process_handle=process_handle,
                                     backend=test_backend)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], 0, pool=4, process_handle=process_handle,
                                        backend=source_backend)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        # The planes stay on the GPU where its decoder and Vulkan share memory: both NVIDIA's.
        if source_backend == test_backend == "nvidia":
            try:
                scorer = VulkanScorer(width, height, bit_depth, models, n_subsample, device=chosen.index, shared=test)
            except VmafVulkanError as error:
                _log.info("VMAF with Vulkan: the frames go through system memory (%s)", error)
        shared = scorer is not None
        if scorer is None:
            scorer = VulkanScorer(width, height, bit_depth, models, n_subsample, device=chosen.index)
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
                if shared:
                    scorer.add_shared(lambda address, pitch, slot=ref_slot: ref.copy_luma(slot, address, pitch),
                                      lambda address, pitch, slot=test_slot: test.copy_luma(slot, address, pitch))
                else:
                    scorer.add_decoded(lambda address, slot=ref_slot: ref.download(slot, address),
                                       lambda address, slot=test_slot: test.download(slot, address))
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
        if scorer is not None:  # first: it gives the decoder's hold on its memory back
            scorer.close()
        test.close()
        ref.close()
