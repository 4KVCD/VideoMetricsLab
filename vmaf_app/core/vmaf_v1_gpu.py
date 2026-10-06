"""VMAF v1 with the GPU: the score FFmpeg's libvmaf gives for a VMAF v1 model
(vmaf_v1.0.16 and its variants), calculated several times faster.

VMAF v1 is predicted from four features. Where each is calculated here:

- ADM3 and motion3, about two thirds of the CPU's work: on the GPU, by
  vmaf_vulkan.dll's VMAF v1 mode (libvmaf-fast's fast/vulkan), which follows
  libvmaf's CPU code (integer_adm.c, integer_motion.c) in integer arithmetic
  and gives its values bit for bit, on any GPU.
- CAMBI and SpEED chroma: on the GPU too where the engine calculates the
  model's options (vv_v1_cambi, vv_v1_speed), bit for bit as libvmaf's CPU
  code: CAMBI wholly; SpEED's filtering of the chroma planes, and what is
  left of it (its 25x25 covariance in double, eigenvalues, CRT log2) by
  libvmaf's own lines compiled into the engine. Otherwise by libvmaf's own
  CPU extractors, from the bundled libvmaf.dll, on a pool of threads beside
  the GPU: the same code on the same frames.

The score is then predicted by libvmaf itself from the four (as vmaf_vulkan
predicts VMAF v0.6.1), so it is libvmaf's CPU score, not an approximation of
it.

V1Scorer takes the frames vmaf_cuda.GpuScorer takes -- 4:2:0 frames packed as
FFmpeg's rawvideo writes them, here with their chroma, which SpEED reads --
and returns the frame numbers scored and their scores.

From NVIDIA's decoder it takes them without a CPU copy (add_decoded): the
lumas are copied on the GPU into memory the decoder and Vulkan share
(vmaf_vulkan.SharedLumas), and the planes CAMBI and SpEED read are downloaded
straight into libvmaf's pictures, page-locked, which the GPU writes by itself.
Through system memory, as before, the one thread that feeds both halves spent
9.7 ms a 4K frame pair copying -- all of its time at 98 frames a second; now
0.6 ms, and the CPU's half sets the pace (about 145 frames a second).
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from vmaf_app.core import vmaf_cuda, vmaf_vulkan

_log = logging.getLogger(__name__)

#: What a score records it was calculated with (its provenance).
LIBRARY_BUILD = f"libvmaf-fast {vmaf_cuda.LIBVMAF_FAST_VERSION} (Vulkan: ADM3, motion3, CAMBI, SpEED)"

_ADM3, _MOTION3 = "VMAF_integer_feature_adm3_score", "VMAF_integer_feature_motion3_score"
_CAMBI, _SPEED = "Cambi_feature_cambi_score", "Speed_chroma_feature_speed_chroma_uv_score"
#: The model's feature -> (libvmaf's extractor, the start of the name libvmaf
#: gives its score).
_CPU_FEATURES = {_CAMBI: ("cambi", "cambi"), _SPEED: ("speed_chroma", "speed_chroma_uv")}
_GPU_FEATURES = {_ADM3: "integer_adm3", _MOTION3: "integer_motion3"}
_VMAF_OUTPUT_FORMAT_JSON = 2
#: Large enough for SpEED with the smallest prescale a model has (see
#: speed_too_small): libvmaf crashes on less.
_PROBE_SIZE = (960, 720)

_bound = False
_BIND_LOCK = threading.Lock()


class VmafV1Error(vmaf_cuda.VmafGpuError):
    """VMAF v1 could not be calculated with the GPU; the CPU calculates it."""


def _libvmaf() -> ctypes.CDLL:
    global _bound
    lib = vmaf_cuda._load()
    with _BIND_LOCK:
        if not _bound:
            handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            for name, argtypes in (
                ("vmaf_use_feature", [handle, ctypes.c_char_p, handle]),
                ("vmaf_feature_dictionary_set", [pointer, ctypes.c_char_p, ctypes.c_char_p]),
                ("vmaf_write_output", [handle, ctypes.c_char_p, ctypes.c_int]),
                ("vmaf_model_load_from_path", [pointer, ctypes.POINTER(vmaf_cuda._ModelConfig), ctypes.c_char_p]),
                ("vmaf_feature_score_at_index", [handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
                ("vmaf_import_feature_score", [handle, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint]),
                ("vmaf_picture_alloc", [ctypes.POINTER(vmaf_cuda._Picture), ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_uint]),
            ):
                function = getattr(lib, name)
                function.restype, function.argtypes = ctypes.c_int, argtypes
            _bound = True
    return lib


def _vulkan() -> ctypes.CDLL:
    lib = vmaf_vulkan._load()
    lib.vv_create_v1.restype = ctypes.c_int
    lib.vv_create_v1.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.vv_features_v1.restype = ctypes.c_int
    lib.vv_features_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
    if hasattr(lib, "vv_v1_cambi"):  # CAMBI on the GPU
        lib.vv_v1_cambi.restype = ctypes.c_int
        lib.vv_v1_cambi.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        lib.vv_v1_cambi_scores.restype = ctypes.c_int
        lib.vv_v1_cambi_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
    if hasattr(lib, "vv_v1_speed"):  # SpEED chroma on the GPU
        lib.vv_v1_speed.restype = ctypes.c_int
        lib.vv_v1_speed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        lib.vv_submit_v1.restype = ctypes.c_int
        lib.vv_submit_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_ssize_t), ctypes.c_int]
        lib.vv_chroma_staging.restype = ctypes.c_int
        lib.vv_chroma_staging.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_uint32)]
        lib.vv_v1_speed_scores.restype = ctypes.c_int
        lib.vv_v1_speed_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
    if hasattr(lib, "vv_shared_chroma"):  # the chroma in the shared buffers, for a decoder to copy there
        lib.vv_shared_chroma.restype = ctypes.c_int
        lib.vv_shared_chroma.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                                         ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    if hasattr(lib, "vv_import_timeline"):  # a decoder's copies the GPU waits for (vv_commit_after)
        lib.vv_import_timeline.restype = ctypes.c_int
        lib.vv_import_timeline.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.vv_commit_after.restype = ctypes.c_int
        lib.vv_commit_after.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    return lib


#: CAMBI's options the GPU takes (vv_v1_cambi), with cambi.c's defaults.
_CAMBI_DEFAULTS = {"cambi_high_res_speedup": 0, "cambi_vis_lum_threshold": 0.0, "cambi_max_val": 1000.0,
                   "topk": 0.6, "cambi_topk": 0.6, "window_size": 65, "tvi_threshold": 0.019,
                   "max_log_contrast": 2, "eotf": "bt1886", "cambi_eotf": "bt1886"}


def _cambi_gpu_options(options: dict) -> ctypes.Array | None:
    """vv_v1_cambi's nine doubles for the model's CAMBI options, None when it
    sets one the GPU does not take (full_ref, heatmaps, enc_* and src_* sizes,
    another max_log_contrast or EOTF): libvmaf calculates it then."""
    if set(options) - set(_CAMBI_DEFAULTS):
        return None
    o = {**_CAMBI_DEFAULTS, **options}
    # cambi.c: topk and eotf if set to other than their default, else cambi_topk and cambi_eotf.
    topk = float(o["topk"]) if float(o["topk"]) != 0.6 else float(o["cambi_topk"])
    eotf = str(o["cambi_eotf"]) if str(o["cambi_eotf"]) != "bt1886" else str(o["eotf"])
    if eotf not in ("bt1886", "pq") or int(o["max_log_contrast"]) != 2:
        return None
    return (ctypes.c_double * 9)(float(o["cambi_high_res_speedup"]), float(o["cambi_vis_lum_threshold"]),
                                 float(o["cambi_max_val"]), topk, float(o["window_size"]), float(o["tvi_threshold"]),
                                 float(o["max_log_contrast"]), 0.0 if eotf == "bt1886" else 1.0,
                                 float((1 if os.environ.get("VMAF_FAST_TEST_CAMBI_POOL_ON_CPU") else 0)
                                       | (2 if os.environ.get("VMAF_FAST_TEST_CAMBI_BRUTE_FORCE") else 0)))


#: SpEED chroma's options the GPU takes (vv_v1_speed), with speed.c's
#: defaults, and the aliases libvmaf also reads them by.
_SPEED_DEFAULTS = {"speed_kernelscale": 1.0, "speed_prescale": 1.0, "speed_prescale_method": "nearest",
                   "speed_sigma_nn": 0.29, "speed_nn_floor": 0.0, "speed_weight_var_mode": 0,
                   "speed_max_val": 1000.0}
_SPEED_ALIASES = {"ks": "speed_kernelscale", "ps": "speed_prescale", "psm": "speed_prescale_method",
                  "snn": "speed_sigma_nn", "nnf": "speed_nn_floor", "wvm": "speed_weight_var_mode",
                  "mxv": "speed_max_val"}


def _speed_gpu_options(options: dict) -> ctypes.Array | None:
    """vv_v1_speed's seven doubles for the model's SpEED options, None when it
    sets one the GPU does not take: libvmaf calculates it then."""
    named = {_SPEED_ALIASES.get(key, key): value for key, value in options.items()}
    if len(named) != len(options) or set(named) - set(_SPEED_DEFAULTS):
        return None
    o = {**_SPEED_DEFAULTS, **named}

    def number(key: str) -> float:  # as libvmaf parses the option's text
        return float(_text(o[key]))

    return (ctypes.c_double * 7)(number("speed_kernelscale"), number("speed_prescale"),
                                 1.0 if _text(o["speed_prescale_method"]) == "bilinear" else 0.0,
                                 number("speed_sigma_nn"), number("speed_nn_floor"),
                                 float(int(_text(o["speed_weight_var_mode"]))), number("speed_max_val"))


def model_options(model_path: Path) -> dict[str, dict]:
    """A model's features and their options; VmafV1Error unless they are VMAF
    v1's four."""
    try:
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))["model_dict"]
        options = dict(zip(model["feature_names"], model["feature_opts_dicts"], strict=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise VmafV1Error(f"{Path(model_path).name} is not a VMAF model file: {error}") from error
    if set(options) != {_ADM3, _MOTION3, _CAMBI, _SPEED}:
        raise VmafV1Error(f"{Path(model_path).name} is not a VMAF v1 model")
    return options


def speed_too_small(options: dict[str, dict], width: int, height: int) -> bool:
    """Whether SpEED has no block to score at this size: it works on the
    chroma, scaled by the model's speed_prescale and then down by 16, in
    blocks of 5 (speed_init_dimensions). libvmaf reports that and then reads
    past its buffers, on the CPU as here, so such a video is refused."""
    prescale = float(options[_SPEED].get("speed_prescale", 1.0))
    return min(int((size // 2) * prescale + 0.5) >> 4 for size in (width, height)) < 5


def _text(value) -> str:
    """An option's value as libvmaf's option parser reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _use_cpu_features(lib: ctypes.CDLL, context: ctypes.c_void_p, options: dict[str, dict],
                      on_gpu: frozenset = frozenset()) -> None:
    for feature, (extractor, _prefix) in _CPU_FEATURES.items():
        if feature in on_gpu:
            continue
        dictionary = ctypes.c_void_p()
        for key, value in options[feature].items():
            vmaf_cuda._check(lib.vmaf_feature_dictionary_set(ctypes.byref(dictionary), key.encode(),
                                                             _text(value).encode()), f"Setting {key}")
        vmaf_cuda._check(lib.vmaf_use_feature(context, extractor.encode(), dictionary), f"Starting {extractor}")


def _output_names(lib: ctypes.CDLL, context: ctypes.c_void_p) -> list[str]:
    """The names of the features a context has scores for (libvmaf has no
    call that lists them: they are read from its JSON output)."""
    handle, path = tempfile.mkstemp(prefix="vml-vmaf-v1-", suffix=".json")
    os.close(handle)
    try:
        vmaf_cuda._check(lib.vmaf_write_output(context, path.encode(), _VMAF_OUTPUT_FORMAT_JSON), "Listing features")
        return list(json.loads(Path(path).read_text(encoding="utf-8"))["frames"][-1]["metrics"])
    finally:
        Path(path).unlink(missing_ok=True)


def _probe_pictures(lib: ctypes.CDLL, context: ctypes.c_void_p) -> None:
    """Three small grey frame pairs through a context."""
    width, height = _PROBE_SIZE
    for index in range(3):
        pictures = vmaf_cuda._Picture(), vmaf_cuda._Picture()
        for picture in pictures:
            vmaf_cuda._check(lib.vmaf_picture_alloc(ctypes.byref(picture), vmaf_cuda._VMAF_PIX_FMT_YUV420P, 8,
                                                    width, height), "Allocating a picture")
            for plane in range(3):
                ctypes.memset(picture.data[plane], 100 + index, picture.stride[plane] * picture.h[plane])
        vmaf_cuda._check(lib.vmaf_read_pictures(context, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]), index),
                         "Scoring a probe frame")
    vmaf_cuda._check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing the probe")


_NAMES: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
_NAMES_LOCK = threading.Lock()


def feature_names(model_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """({feature: the name the model's prediction reads it under}, {CAMBI and
    SpEED: the name libvmaf's extractor, given the model's options, stores it
    under}). libvmaf builds both from the options; they are found by running
    it on three small frames, once per model."""
    key = str(Path(model_path).resolve())
    with _NAMES_LOCK:
        if key in _NAMES:
            return _NAMES[key]
    lib = _libvmaf()
    options = model_options(model_path)
    configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0)
    found = []
    for with_model in (True, False):
        context, model = ctypes.c_void_p(), ctypes.c_void_p()
        vmaf_cuda._check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
        try:
            if with_model:
                config = vmaf_cuda._ModelConfig(b"vmaf", 0)
                vmaf_cuda._check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                               str(model_path).encode()), "Loading the model")
                vmaf_cuda._check(lib.vmaf_use_features_from_model(context, model), "Setting up the model")
            else:
                _use_cpu_features(lib, context, options)
            _probe_pictures(lib, context)
            names = _output_names(lib, context)
        finally:
            if model:
                lib.vmaf_model_destroy(model)
            lib.vmaf_close(context)
        wanted = {**_GPU_FEATURES, **{feature: prefix for feature, (_e, prefix) in _CPU_FEATURES.items()}}
        if not with_model:
            wanted = {feature: prefix for feature, (_e, prefix) in _CPU_FEATURES.items()}
        named = {}
        for feature, prefix in wanted.items():
            matches = [name for name in names if name == prefix or name.startswith(prefix + "_")]
            if len(matches) != 1:
                raise VmafV1Error(f"libvmaf names {feature} in a way this build does not know: {matches or names}")
            named[feature] = matches[0]
        found.append(named)
    with _NAMES_LOCK:
        _NAMES[key] = (found[0], found[1])
    return _NAMES[key]


class V1Scorer:
    """VMAF v1 of frame pairs given in order: 4:2:0 frames packed as FFmpeg's
    rawvideo writes them, at `bit_depth` bits (16-bit little-endian samples
    above 8). `threads`: libvmaf's for CAMBI and SpEED."""

    def __init__(self, width: int, height: int, bit_depth: int, model_path: Path, n_subsample: int = 1,
                 device: int | None = None, threads: int | None = None, name: str = "vmaf_v1", shared=None,
                 gpu_cambi: bool = True, gpu_speed: bool = True):
        """`gpu_cambi`, `gpu_speed`: CAMBI, SpEED chroma on the GPU where the
        engine calculates it (False: libvmaf's CPU code). `shared`: the decoder (GpuFrameStream) the frames come from without
        a CPU copy (add_decoded): its lumas copied on the GPU into Vulkan's
        memory, the planes libvmaf's extractors read downloaded by the GPU
        into their pictures."""
        self._shared: vmaf_vulkan.SharedLumas | None = None
        self._pinned: dict[int, object] = {}
        self._lib = lib = _libvmaf()
        self._vulkan = vulkan = _vulkan()
        self._model_path = Path(model_path)
        self._name = name
        self._step = max(1, n_subsample)
        self._count = 0
        self._width, self._height, self._bit_depth = width, height, bit_depth
        options = model_options(self._model_path)
        if speed_too_small(options, width, height):
            raise VmafV1Error(f"{width}x{height} is too small for this VMAF v1 model's SpEED feature")
        self._model_names, self._cpu_names = feature_names(self._model_path)
        if device is None:
            chosen = vmaf_vulkan.best_device()
            if chosen is None:
                raise VmafV1Error("no GPU that Vulkan can calculate VMAF on")
            device = chosen.index
        adm, motion = options[_ADM3], options[_MOTION3]
        values = (ctypes.c_double * 9)(
            adm.get("adm_norm_view_dist", 3.0), adm.get("adm_ref_display_height", 1080), adm.get("adm_csf_mode", 0),
            adm.get("adm_noise_weight", 0.03125), adm.get("adm_dlm_weight", 1.0), adm.get("adm_min_val", 0.0),
            motion.get("motion_max_val", 10000.0), float(bool(motion.get("motion_five_frame_window", False))),
            float(bool(motion.get("motion_moving_average", False))))
        unsupported = (set(adm) - {"adm_norm_view_dist", "adm_ref_display_height", "adm_csf_mode", "adm_noise_weight",
                                   "adm_dlm_weight", "adm_min_val", "adm_enhn_gain_limit"}
                       | set(motion) - {"motion_max_val", "motion_five_frame_window", "motion_moving_average"})
        if unsupported or adm.get("adm_enhn_gain_limit", 100.0) != 1.0:
            raise VmafV1Error(f"the model has ADM or motion options the GPU does not calculate: {sorted(unsupported)}")
        self._gpu = ctypes.c_void_p()
        self._cpu = ctypes.c_void_p()
        flags = (3 << 8) | (vmaf_vulkan.SHARED_FLAG if shared is not None else 0)
        vmaf_vulkan._check(vulkan, vulkan.vv_create_v1(ctypes.byref(self._gpu), device, width, height, bit_depth,
                                                       flags, values), "Starting Vulkan")
        try:
            if shared is not None:
                if width & 1 or height & 1:
                    raise VmafV1Error("an odd size is not handed over on the GPU")
                self._shared = vmaf_vulkan.SharedLumas(vulkan, self._gpu, shared)
            #: The features the GPU calculates beyond ADM3 and motion3.
            self._on_gpu: frozenset = frozenset()
            cambi = _cambi_gpu_options(options[_CAMBI]) if gpu_cambi and hasattr(vulkan, "vv_v1_cambi") else None
            if cambi is not None:
                code = vulkan.vv_v1_cambi(self._gpu, cambi)
                if code == 0:
                    self._on_gpu = self._on_gpu | {_CAMBI}
                elif code != -4:  # -4: options it does not calculate; libvmaf does then
                    vmaf_vulkan._check(vulkan, code, "Starting CAMBI")
            speed = _speed_gpu_options(options[_SPEED]) if gpu_speed and hasattr(vulkan, "vv_v1_speed") else None
            if speed is not None:
                code = vulkan.vv_v1_speed(self._gpu, speed)
                if code == 0:
                    self._on_gpu = self._on_gpu | {_SPEED}
                elif code != -4:
                    vmaf_vulkan._check(vulkan, code, "Starting SpEED")
            #: From a decoder handing over: where its chroma planes go in each
            #: shared buffer (offset, spacing, row bytes), for it to copy them
            #: there on the GPU; None: through the engine's host memory.
            self._shared_chroma: tuple[int, int, int] | None = None
            if _SPEED in self._on_gpu and self._shared is not None and hasattr(vulkan, "vv_shared_chroma") \
                    and hasattr(shared, "copy_planes"):
                offset, spacing, stride = ctypes.c_uint64(), ctypes.c_uint32(), ctypes.c_uint32()
                if vulkan.vv_shared_chroma(self._gpu, ctypes.byref(offset), ctypes.byref(spacing),
                                           ctypes.byref(stride)) == 0:
                    self._shared_chroma = (offset.value, spacing.value, stride.value)
            #: Whether the decoder's copies into the shared buffers are waited for
            #: by the GPU (its timeline semaphore, imported) instead of by this
            #: thread, which then goes on to the next pair at once.
            self._after = False
            if self._shared is not None and hasattr(vulkan, "vv_import_timeline") \
                    and hasattr(shared, "copy_luma_async"):
                timeline = shared.timeline()
                if timeline is not None:
                    try:
                        self._after = vulkan.vv_import_timeline(self._gpu, timeline) == 0
                    finally:
                        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(timeline))
            #: libvmaf's extractors are run only for what the GPU does not calculate.
            self._cpu_features = bool(set(_CPU_FEATURES) - self._on_gpu)
            if self._cpu_features:
                workers = threads if threads is not None else max(2, min(16, (os.cpu_count() or 4) - 2))
                configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, workers, 1, 0, 0)
                vmaf_cuda._check(lib.vmaf_init(ctypes.byref(self._cpu), configuration), "Starting libvmaf")
                _use_cpu_features(lib, self._cpu, options, self._on_gpu)
                pictures = vmaf_cuda._PictureConfiguration(
                    vmaf_cuda._PictureParameters(width, height, bit_depth, vmaf_cuda._VMAF_PIX_FMT_YUV420P),
                    2 * (workers + 2))
                vmaf_cuda._check(lib.vmaf_preallocate_pictures(self._cpu, pictures), "Allocating pictures")
        except BaseException:
            self.close()
            raise
        self._set_layout(width, height, bit_depth)

    def _set_layout(self, width: int, height: int, bit_depth: int) -> None:
        sample = 1 if bit_depth <= 8 else 2
        self._sample = sample
        self._luma_stride = width * sample
        chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
        #: (offset in a frame, rows, bytes per row) of Y, U and V.
        self._planes = [(0, height, width * sample)]
        offset = width * height * sample
        for _ in range(2):
            self._planes.append((offset, chroma_h, chroma_w * sample))
            offset += chroma_w * chroma_h * sample
        self.frame_bytes = offset

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair (frame index = how many came before)."""
        if min(len(reference), len(distorted)) < self.frame_bytes:
            raise VmafV1Error(f"frame {self._count} is cut short")
        score = self._count % self._step == 0
        ref = (ctypes.c_char * len(reference)).from_buffer(reference)
        dist = (ctypes.c_char * len(distorted)).from_buffer(distorted)
        if _SPEED in self._on_gpu:  # with the chroma planes
            bases = (ctypes.addressof(ref), ctypes.addressof(dist))
            planes = (ctypes.c_void_p * 6)(*(base + offset for base in bases for offset, _r, _b in self._planes))
            strides = (ctypes.c_ssize_t * 6)(*(row_bytes for _base in bases for _o, _r, row_bytes in self._planes))
            code = self._vulkan.vv_submit_v1(self._gpu, planes, strides, int(score))
        else:
            code = self._vulkan.vv_submit(self._gpu, ctypes.addressof(ref), self._luma_stride, ctypes.addressof(dist),
                                          self._luma_stride, int(score))
        vmaf_vulkan._check(self._vulkan, code, f"Scoring frame {self._count}")
        if score and self._cpu_features:
            self._to_libvmaf(reference, distorted)
        self._count += 1

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Scores one more pair held by the decoders (the made-with-`shared`
        way in): nothing of it is copied by the CPU."""
        score = self._count % self._step == 0
        ref_address, dist_address, pitch = self._shared.next()
        # Not waited for where the GPU waits for them (self._after): the
        # timeline's value once the last of them is done.
        luma = (lambda stream, *a: stream.copy_luma_async(*a)) if self._after else             (lambda stream, *a: stream.copy_luma(*a) or 0)
        planes = (lambda stream, *a: stream.copy_planes_async(*a)) if self._after else             (lambda stream, *a: stream.copy_planes(*a) or 0)
        chroma_copied = score and _SPEED in self._on_gpu and self._shared_chroma is not None
        if chroma_copied:
            # The chroma planes copied by the GPU into the slot's shared buffer,
            # each picture's with its luma (one copy on the GPU a picture).
            offset, spacing, stride = self._shared_chroma
            at = [ref_address + offset + i * spacing for i in range(4)]
            copied = max(planes(ref_stream, ref_slot, (ref_address, at[0], at[1]), (pitch, stride, stride)),
                         planes(test_stream, test_slot, (dist_address, at[2], at[3]), (pitch, stride, stride)))
        else:
            copied = luma(ref_stream, ref_slot, ref_address, pitch)
            if score:
                copied = max(copied, luma(test_stream, test_slot, dist_address, pitch))
        if copied:
            vmaf_vulkan._check(self._vulkan, self._vulkan.vv_commit_after(self._gpu, copied),
                               f"Scoring frame {self._count}")
        if score and _SPEED in self._on_gpu and not chroma_copied:
            # The chroma planes downloaded into the engine's memory for them
            # (Vulkan's, which another API is not given to pin).
            chroma, stride = (ctypes.c_void_p * 4)(), ctypes.c_uint32()
            vmaf_vulkan._check(self._vulkan, self._vulkan.vv_chroma_staging(self._gpu, chroma, ctypes.byref(stride)),
                               f"Scoring frame {self._count}")
            pitches = (0, stride.value, stride.value)
            ref_stream.download_planes(ref_slot, (None, chroma[0], chroma[1]), pitches)
            test_stream.download_planes(test_slot, (None, chroma[2], chroma[3]), pitches)
        vmaf_vulkan._check(self._vulkan, self._vulkan.vv_commit(self._gpu, int(score)),
                           f"Scoring frame {self._count}")
        if score and self._cpu_features:
            lib = self._lib
            ref, dist = vmaf_cuda._Picture(), vmaf_cuda._Picture()
            vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(ref)), "Taking a picture")
            try:
                vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(dist)),
                                 "Taking a picture")
            except BaseException:
                lib.vmaf_picture_unref(ctypes.byref(ref))
                raise
            try:
                for picture in (ref, dist):
                    self._pin(picture, test_stream)
                pitches = (dist.stride[0], dist.stride[1], dist.stride[2])
                # As _to_libvmaf: the distorted frame whole, the reference's chroma.
                test_stream.download_planes(test_slot, (dist.data[0], dist.data[1], dist.data[2]), pitches)
                ref_stream.download_planes(ref_slot, (None, ref.data[1], ref.data[2]), pitches)
            except BaseException:
                lib.vmaf_picture_unref(ctypes.byref(ref))
                lib.vmaf_picture_unref(ctypes.byref(dist))
                raise
            vmaf_cuda._check(lib.vmaf_read_pictures(self._cpu, ctypes.byref(ref), ctypes.byref(dist), self._count),
                             f"Scoring frame {self._count}")
        self._count += 1

    def _pin(self, picture, stream) -> None:
        """Page-locks a picture of libvmaf's pool the first time it comes
        (they come round again), for the GPU to write into by itself."""
        address = picture.data[0]
        if address not in self._pinned:
            size = picture.stride[0] * picture.h[0] + 2 * picture.stride[1] * picture.h[1]
            self._pinned[address] = stream if stream.pin(address, size) else None

    def _to_libvmaf(self, reference: bytearray, distorted: bytearray) -> None:
        """Gives libvmaf's extractors the pair: the distorted frame whole
        (CAMBI reads its luma) and the reference's chroma (SpEED reads both
        chromas; nothing reads the reference's luma here)."""
        lib = self._lib
        ref, dist = vmaf_cuda._Picture(), vmaf_cuda._Picture()
        vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(ref)), "Taking a picture")
        try:
            vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            self._fill(ref, reference, planes=(1, 2))
            self._fill(dist, distorted, planes=(0, 1, 2))
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            lib.vmaf_picture_unref(ctypes.byref(dist))
            raise
        vmaf_cuda._check(lib.vmaf_read_pictures(self._cpu, ctypes.byref(ref), ctypes.byref(dist), self._count),
                         f"Scoring frame {self._count}")

    def _fill(self, picture, frame: bytearray, planes: tuple[int, ...]) -> None:
        source = np.frombuffer(frame, dtype=np.uint8)
        for plane in planes:
            offset, rows, row_bytes = self._planes[plane]
            # libvmaf's chroma planes are the size rounded down, FFmpeg's up.
            rows = min(rows, picture.h[plane])
            width = min(row_bytes, picture.w[plane] * self._sample)
            stride = picture.stride[plane]
            target = np.ctypeslib.as_array(
                ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)), shape=(rows, stride))
            target[:, :width] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)[:, :width]

    def features(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The frame numbers scored and each feature's values for them, once
        every frame is in."""
        lib = self._lib
        vmaf_vulkan._check(self._vulkan, self._vulkan.vv_flush(self._gpu), "Scoring")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        rows = np.zeros((self._count, 6), dtype=np.float64)
        if self._count:
            if self._cpu_features:
                vmaf_cuda._check(lib.vmaf_read_pictures(self._cpu, None, None, 0), "Finishing")
            self._vulkan.vv_features_v1(self._gpu, rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count)
        values = {_ADM3: rows[frames, 0], _MOTION3: rows[frames, 3]}
        if _SPEED in self._on_gpu:
            speed = np.full((self._count, 3), np.nan, dtype=np.float64)
            if self._count:
                vmaf_vulkan._check(self._vulkan, self._vulkan.vv_v1_speed_scores(
                    self._gpu, speed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count), "Reading SpEED")
            values[_SPEED] = speed[frames, 2]  # speed_chroma_uv_score
        if _CAMBI in self._on_gpu:
            cambi = np.full(self._count, np.nan, dtype=np.float64)
            if self._count:
                vmaf_vulkan._check(self._vulkan, self._vulkan.vv_v1_cambi_scores(
                    self._gpu, cambi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count), "Reading CAMBI")
            values[_CAMBI] = cambi[frames]
        value = ctypes.c_double()
        for feature, name in self._cpu_names.items():
            if feature in self._on_gpu:
                continue
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                vmaf_cuda._check(lib.vmaf_feature_score_at_index(self._cpu, name.encode(), ctypes.byref(value),
                                                                 int(frame)), f"Reading frame {frame}")
                column[slot] = value.value
            values[feature] = column
        return frames, values

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """As GpuScorer.finish: the frame numbers scored and the model's
        scores for them, rounded as libvmaf's JSON log rounds them."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {self._name: np.zeros(0)}
        frames, values = self.features()
        return frames, {self._name: predict(self._model_path, frames, values, self._model_names)}

    def close(self) -> None:
        """Before the decoders are closed, where the frames came from them."""
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        if self._gpu:
            self._vulkan.vv_destroy(self._gpu)
            self._gpu = ctypes.c_void_p()
        if self._cpu:
            self._lib.vmaf_close(self._cpu)  # waits for the extractors: nothing reads the pictures after it
            self._cpu = ctypes.c_void_p()
        for address, stream in self._pinned.items():
            if stream is not None:
                stream.unpin(address)
        self._pinned = {}


def predict(model_path: Path, frames: np.ndarray, values: dict[str, np.ndarray],
            names: dict[str, str] | None = None) -> np.ndarray:
    """A VMAF v1 model's scores for its four features' values: libvmaf's own
    prediction, from the features under the names the model reads them by."""
    lib = _libvmaf()
    names = names or feature_names(model_path)[0]
    context, model = ctypes.c_void_p(), ctypes.c_void_p()
    configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, 0, 1, 0, 0)
    vmaf_cuda._check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
    try:
        config = vmaf_cuda._ModelConfig(b"vmaf", 0)
        vmaf_cuda._check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                       str(model_path).encode()), "Loading the model")
        for feature, name in names.items():
            encoded = name.encode()
            for frame, value in zip(frames, values[feature], strict=True):
                vmaf_cuda._check(lib.vmaf_import_feature_score(context, encoded, float(value), int(frame)),
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


def cpu_reference(model_path: Path, width: int, height: int, bit_depth: int, reference: list, distorted: list,
                  n_subsample: int = 1, threads: int = 0):
    """libvmaf's CPU VMAF v1 of frames in memory: (the frame numbers scored,
    {feature: values}, the scores rounded as V1Scorer's). What V1Scorer is
    checked against, by the self-test here and scripts/compare_vmaf_v1.py."""
    lib = _libvmaf()
    names, _ = feature_names(model_path)
    context, model = ctypes.c_void_p(), ctypes.c_void_p()
    configuration = vmaf_cuda._Configuration(vmaf_cuda._VMAF_LOG_LEVEL_ERROR, threads, max(1, n_subsample), 0, 0)
    vmaf_cuda._check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
    try:
        config = vmaf_cuda._ModelConfig(b"vmaf", 0)
        vmaf_cuda._check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                       str(model_path).encode()), "Loading the model")
        vmaf_cuda._check(lib.vmaf_use_features_from_model(context, model), "Setting up the model")
        pictures = vmaf_cuda._PictureConfiguration(
            vmaf_cuda._PictureParameters(width, height, bit_depth, vmaf_cuda._VMAF_PIX_FMT_YUV420P),
            2 * (max(1, threads) + 2))
        vmaf_cuda._check(lib.vmaf_preallocate_pictures(context, pictures), "Allocating pictures")
        layout = V1Scorer.__new__(V1Scorer)
        layout._set_layout(width, height, bit_depth)
        for index, frames in enumerate(zip(reference, distorted, strict=True)):
            ref, dist = vmaf_cuda._Picture(), vmaf_cuda._Picture()
            for picture, frame in zip((ref, dist), frames, strict=True):
                vmaf_cuda._check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(picture)), "Taking a picture")
                layout._fill(picture, frame, (0, 1, 2))
            vmaf_cuda._check(lib.vmaf_read_pictures(context, ctypes.byref(ref), ctypes.byref(dist), index),
                             f"Scoring frame {index}")
        vmaf_cuda._check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing")
        scored = np.arange(0, len(reference), max(1, n_subsample), dtype=np.int32)
        value = ctypes.c_double()
        values = {}
        for feature, name in names.items():
            column = np.empty(len(scored), dtype=np.float64)
            for slot, frame in enumerate(scored):
                vmaf_cuda._check(lib.vmaf_feature_score_at_index(context, name.encode(), ctypes.byref(value),
                                                                 int(frame)), f"Reading frame {frame}")
                column[slot] = value.value
            values[feature] = column
        scores = np.empty(len(scored), dtype=np.float64)
        for slot, frame in enumerate(scored):
            vmaf_cuda._check(lib.vmaf_score_at_index(context, model, ctypes.byref(value), int(frame)),
                             f"Predicting frame {frame}")
            scores[slot] = float(f"{value.value:.6f}")
        return scored, values, scores
    finally:
        if model:
            lib.vmaf_model_destroy(model)
        lib.vmaf_close(context)


# ------------------------------------------- with VMAF v0.6.1 and VMAF NEG

V1_KEY = "vmaf_v1"


class MultiScorer:
    """A run's GPU scorers fed the same frames: VMAF v0.6.1 and NEG's
    (libvmaf's CUDA code or the Vulkan port, by `backend`) and VMAF v1's.
    `models` as vmaf_cuda.gpu_models gives them: "vmaf_v1" maps to the
    model's "path=<file>"."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 backend: str = "cuda", device: int | None = None, shared=None):
        """`shared`: NVIDIA's decoder (GpuFrameStream), whose pictures the
        scorers then take on the GPU (add_decoded) instead of from system
        memory (add). VmafGpuError where they cannot."""
        models = dict(models)
        v1_model = models.pop(V1_KEY, None)
        self._scorers = []
        self._vmaf = None
        try:
            if models:
                if backend == "vulkan":
                    self._vmaf = vmaf_vulkan.VulkanScorer(width, height, bit_depth, models, n_subsample,
                                                          device=device, shared=shared)
                else:
                    self._vmaf = vmaf_cuda.GpuScorer(width, height, bit_depth, models, n_subsample,
                                                     on_device=shared is not None)
                self._scorers.append(self._vmaf)
            if v1_model is not None:
                self._scorers.append(V1Scorer(width, height, bit_depth, Path(v1_model.removeprefix("path=")),
                                              n_subsample, device=device if backend == "vulkan" else None,
                                              shared=shared))
        except BaseException:
            self.close()
            raise
        self.frame_bytes = self._scorers[0].frame_bytes

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """One more pair held by the decoders, scored without a CPU copy."""
        for scorer in self._scorers:
            if isinstance(scorer, V1Scorer):
                scorer.add_decoded(ref_stream, ref_slot, test_stream, test_slot)
            else:  # VMAF v0.6.1 and NEG: the luma planes, copied on the GPU
                (scorer.add_shared if isinstance(scorer, vmaf_vulkan.VulkanScorer) else scorer.add_on_device)(
                    lambda address, pitch: ref_stream.copy_luma(ref_slot, address, pitch),
                    lambda address, pitch: test_stream.copy_luma(test_slot, address, pitch))

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        for scorer in self._scorers:
            scorer.add(reference, distorted)

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        frames, scores = None, {}
        for scorer in self._scorers:
            numbers, values = scorer.finish()
            if frames is not None and not np.array_equal(frames, numbers):
                raise VmafV1Error("the GPU's scorers scored different frames")
            frames = numbers
            scores.update(values)
        return frames, scores

    def close(self) -> None:
        for scorer in self._scorers:
            scorer.close()
        self._scorers = []


def score_decoded(
    source, distorted, source_crop=None, distorted_crop=None, *, width: int, height: int, bit_depth: int,
    models: dict[str, str], n_subsample: int, duration_limit: str | None, total_frames: int,
    scale_algorithm: str = "bicubic", on_progress=None, check_cancel=lambda: None, process_handle=None,
    backend: str = "cuda", device: int | None = None, decoder: str = "nvidia",
):
    """vmaf_cuda.score_decoded for a run with VMAF v1: both videos decoded in
    this process by the GPU's own decoder (gpu_frames) -- whole frames, as
    SpEED reads the chroma -- cropped, scaled and widened there, paired as
    libvmaf's filter pairs them (frame_sync), and each pair given to the
    run's scorers (MultiScorer). FFmpeg only copies the compressed streams
    out of their containers."""
    import time
    from dataclasses import replace

    from vmaf_app.core import gpu_frames
    from vmaf_app.core.frame_sync import frame_pairs

    if not gpu_frames.available(decoder):
        raise gpu_frames.GpuDecodeUnavailableError(f"the {decoder} decoder library is not bundled")
    plans = []
    for info, crop in ((distorted, distorted_crop), (source, source_crop)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, size=(width, height), algorithm=scale_algorithm)
        if plan.bit_depth < bit_depth:
            full = (info.color_range or "").casefold() in {"pc", "jpeg", "full"} or (
                not info.color_range and (info.pix_fmt or "").casefold().startswith("yuvj"))
            plan = replace(plan, widen=gpu_frames.WIDEN_REPEAT if full else gpu_frames.WIDEN_SHIFT)
        if plan.bit_depth > bit_depth:
            raise gpu_frames.GpuDecodeUnavailableError(
                f"the videos are compared at {bit_depth} bits and one is {plan.bit_depth}-bit")
        supported, refusal = gpu_frames.decoder_supports(0, plan, decoder)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
    test = gpu_frames.GpuFrameStream(distorted, plans[0], 0, pool=4, process_handle=process_handle, backend=decoder,
                                     handover=True)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], 0, pool=4, process_handle=process_handle, backend=decoder,
                                        handover=True)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        # Its pictures stay on the GPU, and reach libvmaf's without a CPU copy:
        # NVIDIA's, and AMD's handing over (to Vulkan only, not libvmaf's CUDA).
        if test.handover and ref.handover and (decoder == "nvidia" or backend == "vulkan"):
            try:
                scorer = MultiScorer(width, height, bit_depth, models, n_subsample, backend, device, shared=test)
            except vmaf_cuda.VmafGpuError as error:
                _log.info("VMAF v1 with the GPU: the frames go through system memory (%s)", error)
        on_gpu = scorer is not None
        if scorer is None:
            scorer = MultiScorer(width, height, bit_depth, models, n_subsample, backend, device)
        if scorer.frame_bytes != test.frame_bytes:
            raise VmafV1Error("the decoder's frames are not the size the scorers take")
        buffers = bytearray(scorer.frame_bytes), bytearray(scorer.frame_bytes)
        addresses = [ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer)) for buffer in buffers]
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
                if on_gpu:
                    scorer.add_decoded(ref, ref_slot, test, test_slot)
                else:
                    ref.download(ref_slot, addresses[0])
                    test.download(test_slot, addresses[1])
                    scorer.add(*buffers)
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


# ------------------------------------------------------------ availability

def _self_test_frames(bits: int, count: int = 4) -> tuple[list[bytearray], list[bytearray]]:
    """Frame pairs for the self-test: textured, moving, the distorted one
    blurred and noisy. Integer arithmetic only: the same bytes on every PC."""
    width, height = _PROBE_SIZE
    peak = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    cy, cx = np.mgrid[0:height // 2, 0:width // 2].astype(np.int64)
    reference, distorted = [], []
    for index in range(count):
        noise = ((x * 2654435761 + y * 40503 + index * 69069) * 1103515245 >> 16) & 0xFF
        ref = (((x * 3 + y * 2 + index * 7) % 256) + noise) // 2
        dis = ref.copy()
        dis[:, 1:-1] = (ref[:, :-2] + 2 * ref[:, 1:-1] + ref[:, 2:]) // 4
        dis = np.clip(dis + (noise % 9) - 4, 0, 255)
        frames = []
        for luma, shift in ((ref, 0), (dis, 3)):
            planes = [luma, (cx * 2 + cy + index * 3 + shift) % 256, (cx + cy * 2 + index * 5 + 2 * shift) % 256]
            frames.append(bytearray(b"".join((plane * peak // 255).astype(dtype).tobytes() for plane in planes)))
        reference.append(frames[0])
        distorted.append(frames[1])
    return reference, distorted


def probe() -> tuple[bool, str]:
    """Whether VMAF v1 is calculated with the GPU on this PC. The GPU is
    accepted only if it gives libvmaf's CPU values exactly: vmaf_vulkan's
    self-test of the shaders it shares with VMAF v0.6.1, then ADM3 and
    motion3 of a few frames, at 8 and at 10 bits and with the five-frame
    motion window, against libvmaf's CPU extractors. A driver that compiles
    a shader wrongly gives wrong scores, not an error. Run in a process of
    its own."""
    try:
        available, device, text = vmaf_vulkan.probe()
        if not available:
            return False, text
        models = Path(__file__).resolve().parents[1] / "models"
        for model, bits in (("vmaf_v1.0.16/vmaf_v1.0.16_3d0h.json", 8),
                            ("vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_1d5h_2160.json", 10)):
            path = models / model
            if not path.is_file():
                return False, "the VMAF v1 models are not bundled"
            reference, distorted = _self_test_frames(bits)
            _frames, expected, _scores = cpu_reference(path, *_PROBE_SIZE, bits, reference, distorted)
            scorer = V1Scorer(*_PROBE_SIZE, bits, path, device=device, threads=2)
            try:
                for ref, dis in zip(reference, distorted, strict=True):
                    scorer.add(ref, dis)
                _frames, values = scorer.features()
            finally:
                scorer.close()
            for feature in (_ADM3, _MOTION3, _CAMBI, _SPEED):
                if not np.array_equal(values[feature].view(np.uint64), expected[feature].view(np.uint64)):
                    return False, f"the GPU calculates VMAF v1 wrongly ({bits}-bit self-test)"
        return True, text
    except (OSError, vmaf_cuda.VmafGpuError) as error:
        return False, str(error)


_PROBE_LOCK = threading.Lock()
_probed: tuple[bool, str] | None = None


def available() -> tuple[bool, str]:
    """Whether VMAF v1 is calculated with the GPU here, and on what (or why
    not). Probed once, in a process of its own."""
    global _probed
    with _PROBE_LOCK:
        if _probed is None:
            from vmaf_app.core.isolated import IsolatedCrashError, run_isolated

            try:
                _probed = run_isolated(probe, what="the GPU VMAF v1 probe")
            except IsolatedCrashError as error:
                _probed = (False, str(error))
            except Exception as error:
                _log.warning("The GPU VMAF v1 probe failed", exc_info=error)
                _probed = (False, f"the GPU probe failed: {error}")
            _log.info("VMAF v1 with the GPU%s: %s", "" if _probed[0] else " unavailable", _probed[1])
        return _probed


def scores(model: str, size: tuple[int, int] | None = None) -> bool:
    """Whether a run's VMAF v1 is calculated with the GPU: `model` as
    vmaf_cuda.gpu_models names it ("path=<file>"; empty while Auto is not
    resolved, when the job is planned), one of VMAF v1's, at a `size` (where
    known) SpEED has blocks to score at, on a PC whose GPU passed the
    self-test. FFmpeg's libvmaf calculates it otherwise."""
    options: dict[str, dict] = {_SPEED: {}}
    if model:
        try:
            options = model_options(Path(model.removeprefix("path=")))
        except VmafV1Error:
            return False
    if size is not None and speed_too_small(options, *size):
        return False
    return available()[0]


def start_probe() -> None:
    """At startup, in the background: done by the time a run needs it."""
    threading.Thread(target=available, name="vmaf-v1-gpu-probe", daemon=True).start()
