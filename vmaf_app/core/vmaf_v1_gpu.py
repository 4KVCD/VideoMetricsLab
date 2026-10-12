"""Calculates VMAF v1 with the GPU: the score FFmpeg's libvmaf gives for a
VMAF v1 model (vmaf_v1.0.16 and its variants), several times faster.

VMAF v1 is predicted from four features, calculated here as follows:

- ADM3 and motion3, about two thirds of the CPU's work, are calculated on
  the GPU by vmaf_vulkan.dll's VMAF v1 mode (libvmaf-fast's fast/vulkan).
  It follows libvmaf's CPU code (integer_adm.c, integer_motion.c) in
  integer arithmetic, and gives the same values bit for bit on any GPU.
- CAMBI and SpEED chroma are calculated on the GPU too when the engine
  supports the model's options for them (vv_v1_cambi, vv_v1_speed), again
  bit for bit as libvmaf's CPU code. CAMBI runs wholly on the GPU. For
  SpEED, the GPU filters the chroma planes, and the rest (its 25x25
  covariance in double precision, the eigenvalues, the C runtime's log2)
  runs libvmaf's own code compiled into the engine. Otherwise these two
  features are calculated by libvmaf's own CPU extractors from the bundled
  libvmaf.dll, on a pool of threads beside the GPU: the same code on the
  same frames.

libvmaf itself then predicts the score from the four features (as
vmaf_vulkan does for VMAF v0.6.1), so the score is libvmaf's CPU score, not
an approximation of it.

V1Scorer takes the same frames as vmaf_gpu.GpuScorer: 4:2:0 frames packed
as FFmpeg's rawvideo writes them, here with their chroma, which SpEED
reads. It returns the frame numbers scored and their scores.

Frames decoded by NVIDIA's decoder reach it without a copy through the CPU
(add_decoded). The lumas are copied on the GPU into memory that the decoder
and Vulkan share (vmaf_vulkan.SharedLumas), and the GPU downloads the planes
that CAMBI and SpEED read straight into libvmaf's page-locked pictures.
Copying through system memory instead took the thread that feeds both the
GPU and the CPU 9.7 ms per 4K frame pair, all of its time at 98 frames a
second. This way it takes 0.6 ms, and the CPU's features set the pace
(about 145 frames a second).
"""
from __future__ import annotations

import collections
import ctypes
import functools
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from vmaf_app.core import libvmaf, vmaf_gpu, vmaf_vulkan

_log = logging.getLogger(__name__)

#: The library a score records as having calculated it (its provenance).
LIBRARY_BUILD = f"libvmaf-fast {libvmaf.LIBVMAF_FAST_VERSION} (Vulkan: ADM3, motion3, CAMBI, SpEED)"

_ADM3, _MOTION3 = "VMAF_integer_feature_adm3_score", "VMAF_integer_feature_motion3_score"
_CAMBI, _SPEED = "Cambi_feature_cambi_score", "Speed_chroma_feature_speed_chroma_uv_score"
#: For each feature libvmaf can calculate on the CPU: (libvmaf's extractor,
#: the start of the name libvmaf gives its score).
_CPU_FEATURES = {_CAMBI: ("cambi", "cambi"), _SPEED: ("speed_chroma", "speed_chroma_uv")}
_GPU_FEATURES = {_ADM3: "integer_adm3", _MOTION3: "integer_motion3"}
_VMAF_OUTPUT_FORMAT_JSON = 2
#: The probe's picture size. It must be large enough for SpEED with the
#: smallest prescale a model has (see speed_too_small), because libvmaf
#: crashes on smaller pictures.
_PROBE_SIZE = (960, 720)

_bound = False
_BIND_LOCK = threading.Lock()


class VmafV1Error(libvmaf.LibvmafError):
    """VMAF v1 could not be calculated with the GPU, so the CPU calculates it."""


def _libvmaf() -> ctypes.CDLL:
    """The bundled libvmaf.dll, with the functions this module uses declared."""
    global _bound
    lib = libvmaf.load()
    with _BIND_LOCK:
        if not _bound:
            handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            for name, argtypes in (
                ("vmaf_use_feature", [handle, ctypes.c_char_p, handle]),
                ("vmaf_feature_dictionary_set", [pointer, ctypes.c_char_p, ctypes.c_char_p]),
                ("vmaf_write_output", [handle, ctypes.c_char_p, ctypes.c_int]),
                ("vmaf_model_load_from_path", [pointer, ctypes.POINTER(libvmaf.ModelConfig), ctypes.c_char_p]),
                ("vmaf_feature_score_at_index", [handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
                ("vmaf_import_feature_score", [handle, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint]),
                ("vmaf_picture_alloc", [ctypes.POINTER(libvmaf.Picture), ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_uint]),
            ):
                function = getattr(lib, name)
                function.restype, function.argtypes = ctypes.c_int, argtypes
            _bound = True
    return lib


def _vulkan() -> ctypes.CDLL:
    """vmaf_vulkan.dll, with its VMAF v1 functions declared. The optional ones
    are declared only if this build of the engine has them."""
    lib = vmaf_vulkan.load()
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
    if hasattr(lib, "vv_pictures"):  # the GPU reads a decoder's pictures where it left them (V1Scorer.pictures)
        lib.vv_pictures.restype = ctypes.c_int
        lib.vv_pictures.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                                    ctypes.c_int]
        lib.vv_pictures_mode.restype = ctypes.c_int
        lib.vv_pictures_mode.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "vv_import_timeline"):  # the GPU waits for a decoder's copies (vv_commit_after)
        lib.vv_import_timeline.restype = ctypes.c_int
        lib.vv_import_timeline.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.vv_commit_after.restype = ctypes.c_int
        lib.vv_commit_after.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    return lib


#: The CAMBI options the GPU supports (vv_v1_cambi), with cambi.c's defaults.
_CAMBI_DEFAULTS = {"cambi_high_res_speedup": 0, "cambi_vis_lum_threshold": 0.0, "cambi_max_val": 1000.0,
                   "topk": 0.6, "cambi_topk": 0.6, "window_size": 65, "tvi_threshold": 0.019,
                   "max_log_contrast": 2, "eotf": "bt1886", "cambi_eotf": "bt1886"}


def _cambi_gpu_options(options: dict) -> ctypes.Array | None:
    """Returns vv_v1_cambi's nine numbers for the model's CAMBI options, or
    None when the model sets an option the GPU does not support (full_ref,
    heatmaps, the enc_* and src_* sizes, another max_log_contrast or EOTF),
    in which case libvmaf calculates CAMBI."""
    if set(options) - set(_CAMBI_DEFAULTS):
        return None
    o = {**_CAMBI_DEFAULTS, **options}
    # As cambi.c chooses: topk if it is not the default, else cambi_topk; and
    # cambi_eotf if it is not the default, else eotf.
    topk = float(o["topk"]) if float(o["topk"]) != 0.6 else float(o["cambi_topk"])
    eotf = str(o["cambi_eotf"]) if str(o["cambi_eotf"]) != "bt1886" else str(o["eotf"])
    if eotf not in ("bt1886", "pq") or int(o["max_log_contrast"]) != 2:
        return None
    return (ctypes.c_double * 9)(float(o["cambi_high_res_speedup"]), float(o["cambi_vis_lum_threshold"]),
                                 float(o["cambi_max_val"]), topk, float(o["window_size"]), float(o["tvi_threshold"]),
                                 float(o["max_log_contrast"]), 0.0 if eotf == "bt1886" else 1.0,
                                 float((1 if os.environ.get("VMAF_FAST_TEST_CAMBI_POOL_ON_CPU") else 0)
                                       | (2 if os.environ.get("VMAF_FAST_TEST_CAMBI_BRUTE_FORCE") else 0)))


#: The SpEED chroma options the GPU supports (vv_v1_speed), with speed.c's
#: defaults, and the short names libvmaf also accepts for them.
_SPEED_DEFAULTS = {"speed_kernelscale": 1.0, "speed_prescale": 1.0, "speed_prescale_method": "nearest",
                   "speed_sigma_nn": 0.29, "speed_nn_floor": 0.0, "speed_weight_var_mode": 0,
                   "speed_max_val": 1000.0}
_SPEED_ALIASES = {"ks": "speed_kernelscale", "ps": "speed_prescale", "psm": "speed_prescale_method",
                  "snn": "speed_sigma_nn", "nnf": "speed_nn_floor", "wvm": "speed_weight_var_mode",
                  "mxv": "speed_max_val"}


def _speed_gpu_options(options: dict) -> ctypes.Array | None:
    """Returns vv_v1_speed's seven numbers for the model's SpEED options, or
    None when the model sets an option the GPU does not support, in which
    case libvmaf calculates SpEED."""
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
    """Returns a model's features and their options. Raises VmafV1Error unless
    they are VMAF v1's four features."""
    try:
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))["model_dict"]
        options = dict(zip(model["feature_names"], model["feature_opts_dicts"], strict=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise VmafV1Error(f"{Path(model_path).name} is not a VMAF model file: {error}") from error
    if set(options) != {_ADM3, _MOTION3, _CAMBI, _SPEED}:
        raise VmafV1Error(f"{Path(model_path).name} is not a VMAF v1 model")
    return options


def speed_too_small(options: dict[str, dict], width: int, height: int) -> bool:
    """Whether SpEED has no block to score at this size. SpEED works on the
    chroma planes, scaled by the model's speed_prescale and then down by 16,
    in blocks of 5 (speed_init_dimensions). On a video too small for one
    block, libvmaf reports the problem and then reads past its buffers, on
    the CPU as well as here, so such a video is refused."""
    prescale = float(options[_SPEED].get("speed_prescale", 1.0))
    return min(int((size // 2) * prescale + 0.5) >> 4 for size in (width, height)) < 5


def _text(value) -> str:
    """An option's value as libvmaf's option parser reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _use_cpu_features(lib: ctypes.CDLL, context: ctypes.c_void_p, options: dict[str, dict],
                      on_gpu: frozenset = frozenset()) -> None:
    """Starts libvmaf's CPU extractors, with the model's options, for the
    features that are not calculated on the GPU (`on_gpu`)."""
    for feature, (extractor, _prefix) in _CPU_FEATURES.items():
        if feature in on_gpu:
            continue
        dictionary = ctypes.c_void_p()
        for key, value in options[feature].items():
            libvmaf.check(lib.vmaf_feature_dictionary_set(ctypes.byref(dictionary), key.encode(),
                                                             _text(value).encode()), f"Setting {key}")
        libvmaf.check(lib.vmaf_use_feature(context, extractor.encode(), dictionary), f"Starting {extractor}")


def _output_names(lib: ctypes.CDLL, context: ctypes.c_void_p) -> list[str]:
    """Returns the names of the features a context has scores for. libvmaf
    has no call that lists them, so they are read from its JSON output."""
    handle, path = tempfile.mkstemp(prefix="vml-vmaf-v1-", suffix=".json")
    os.close(handle)
    try:
        libvmaf.check(lib.vmaf_write_output(context, libvmaf.path_bytes(path), _VMAF_OUTPUT_FORMAT_JSON),
                         "Listing features")
        return list(json.loads(Path(path).read_text(encoding="utf-8"))["frames"][-1]["metrics"])
    finally:
        Path(path).unlink(missing_ok=True)


def _probe_pictures(lib: ctypes.CDLL, context: ctypes.c_void_p) -> None:
    """Scores three small grey frame pairs with a context."""
    width, height = _PROBE_SIZE
    for index in range(3):
        pictures = libvmaf.Picture(), libvmaf.Picture()
        for picture in pictures:
            libvmaf.check(lib.vmaf_picture_alloc(ctypes.byref(picture), libvmaf.PIX_FMT_YUV420P, 8,
                                                    width, height), "Allocating a picture")
            for plane in range(3):
                ctypes.memset(picture.data[plane], 100 + index, picture.stride[plane] * picture.h[plane])
        libvmaf.check(lib.vmaf_read_pictures(context, ctypes.byref(pictures[0]), ctypes.byref(pictures[1]), index),
                         "Scoring a probe frame")
    libvmaf.check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing the probe")


_NAMES: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
_NAMES_LOCK = threading.Lock()


def feature_names(model_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Returns two dictionaries: {feature: the name the model's prediction
    reads it under}, and {CAMBI and SpEED: the name libvmaf's extractor
    stores it under, given the model's options}. libvmaf builds both names
    from the options, so they are found by running libvmaf on three small
    frames, once per model."""
    key = str(Path(model_path).resolve())
    with _NAMES_LOCK:
        if key in _NAMES:
            return _NAMES[key]
    lib = _libvmaf()
    options = model_options(model_path)
    configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, 0, 1, 0, 0)
    found = []
    for with_model in (True, False):
        context, model = ctypes.c_void_p(), ctypes.c_void_p()
        libvmaf.check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
        try:
            if with_model:
                config = libvmaf.ModelConfig(b"vmaf", 0)
                libvmaf.check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                               libvmaf.path_bytes(model_path)), "Loading the model")
                libvmaf.check(lib.vmaf_use_features_from_model(context, model), "Setting up the model")
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


#: vv_create_v1's flag for reading the frames from a decoder's textures
#: (vv_pictures).
_PICTURES_FLAG = 1 << 2


class V1Scorer:
    """Calculates VMAF v1 of frame pairs given in order: 4:2:0 frames packed
    as FFmpeg's rawvideo writes them, at `bit_depth` bits (16-bit
    little-endian samples above 8 bits). `threads` is the number of threads
    libvmaf uses for CAMBI and SpEED on the CPU."""

    def __init__(self, width: int, height: int, bit_depth: int, model_path: Path, n_subsample: int = 1,
                 device: int | None = None, threads: int | None = None, name: str = "vmaf_v1", shared=None,
                 gpu_cambi: bool = True, gpu_speed: bool = True, pictures: bool = False):
        """`gpu_cambi` and `gpu_speed` calculate CAMBI and SpEED chroma on the
        GPU when the engine supports the model's options. False uses
        libvmaf's CPU code.

        `shared` is the decoder (GpuFrameStream) that the frames come from
        without a copy through the CPU (add_decoded). Its lumas are copied on
        the GPU into Vulkan's memory, and the GPU downloads the planes that
        libvmaf's extractors read into their pictures.

        `pictures` means the decoders give their pictures as textures
        (GpuFrameStream.gives_textures). The engine then reads them where
        they are, without copying them (vv_pictures), if it can, and
        self.pictures says whether it does. Each pair's slots are then
        released through release_later."""
        self._shared: vmaf_vulkan.SharedLumas | None = None
        #: The decoders' slots handed back (release_later) that the engine may
        #: still read: (the last frame that reads the slot, stream, slot).
        self._held: collections.deque = collections.deque()
        self.pictures = False
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
        if pictures and shared is not None and hasattr(vulkan, "vv_pictures"):
            flags |= _PICTURES_FLAG
        vmaf_vulkan.check(vulkan, vulkan.vv_create_v1(ctypes.byref(self._gpu), device, width, height, bit_depth,
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
                elif code != -4:  # -4: options the GPU does not support, so libvmaf calculates CAMBI
                    vmaf_vulkan.check(vulkan, code, "Starting CAMBI")
            speed = _speed_gpu_options(options[_SPEED]) if gpu_speed and hasattr(vulkan, "vv_v1_speed") else None
            if speed is not None:
                code = vulkan.vv_v1_speed(self._gpu, speed)
                if code == 0:
                    self._on_gpu = self._on_gpu | {_SPEED}
                elif code != -4:
                    vmaf_vulkan.check(vulkan, code, "Starting SpEED")
            #: Where a decoder handing frames over puts its chroma planes in
            #: each shared buffer (offset, spacing, row bytes), copying them
            #: there on the GPU. None means they go through the engine's host
            #: memory instead.
            self._shared_chroma: tuple[int, int, int] | None = None
            if _SPEED in self._on_gpu and self._shared is not None and hasattr(vulkan, "vv_shared_chroma") \
                    and hasattr(shared, "copy_planes"):
                offset, spacing, stride = ctypes.c_uint64(), ctypes.c_uint32(), ctypes.c_uint32()
                if vulkan.vv_shared_chroma(self._gpu, ctypes.byref(offset), ctypes.byref(spacing),
                                           ctypes.byref(stride)) == 0:
                    self._shared_chroma = (offset.value, spacing.value, stride.value)
            #: Whether the GPU, rather than this thread, waits for the decoder's
            #: copies into the shared buffers (through the decoder's timeline
            #: semaphore, imported). This thread then goes on to the next pair
            #: at once.
            self._after = False
            if self._shared is not None and hasattr(vulkan, "vv_import_timeline") \
                    and hasattr(shared, "copy_luma_async"):
                timeline = shared.timeline()
                if timeline is not None:
                    try:
                        self._after = vulkan.vv_import_timeline(self._gpu, timeline) == 0
                    finally:
                        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(timeline))
            #: The engine reads the decoders' pictures where they are (vv_pictures).
            self.pictures = bool(flags & _PICTURES_FLAG) and vulkan.vv_pictures_mode(self._gpu) == 1
            #: Whether libvmaf's CPU extractors run, which they do only for the
            #: features the GPU does not calculate.
            self._cpu_features = bool(set(_CPU_FEATURES) - self._on_gpu)
            if self._cpu_features:
                workers = threads if threads is not None else max(2, min(16, (os.cpu_count() or 4) - 2))
                configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, workers, 1, 0, 0)
                libvmaf.check(lib.vmaf_init(ctypes.byref(self._cpu), configuration), "Starting libvmaf")
                _use_cpu_features(lib, self._cpu, options, self._on_gpu)
                pictures = libvmaf.PictureConfiguration(
                    libvmaf.PictureParameters(width, height, bit_depth, libvmaf.PIX_FMT_YUV420P),
                    2 * (workers + 2))
                libvmaf.check(lib.vmaf_preallocate_pictures(self._cpu, pictures), "Allocating pictures")
        except BaseException:
            self.close()
            raise
        self._set_layout(width, height, bit_depth)

    def _set_layout(self, width: int, height: int, bit_depth: int) -> None:
        """Works out where each plane lies in a packed frame of this size and
        bit depth."""
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
        """Scores one more pair. Its frame index is the number of pairs given
        before it."""
        if min(len(reference), len(distorted)) < self.frame_bytes:
            raise VmafV1Error(f"frame {self._count} is cut short")
        score = self._count % self._step == 0
        ref = (ctypes.c_char * len(reference)).from_buffer(reference)
        dist = (ctypes.c_char * len(distorted)).from_buffer(distorted)
        if _SPEED in self._on_gpu:  # the GPU needs the chroma planes too
            bases = (ctypes.addressof(ref), ctypes.addressof(dist))
            planes = (ctypes.c_void_p * 6)(*(base + offset for base in bases for offset, _r, _b in self._planes))
            strides = (ctypes.c_ssize_t * 6)(*(row_bytes for _base in bases for _o, _r, row_bytes in self._planes))
            code = self._vulkan.vv_submit_v1(self._gpu, planes, strides, int(score))
        else:
            code = self._vulkan.vv_submit(self._gpu, ctypes.addressof(ref), self._luma_stride, ctypes.addressof(dist),
                                          self._luma_stride, int(score))
        vmaf_vulkan.check(self._vulkan, code, f"Scoring frame {self._count}")
        if score and self._cpu_features:
            self._to_libvmaf(reference, distorted)
        self._count += 1

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Scores one more pair held by the decoders, for a scorer made with
        `shared`. The CPU copies none of it."""
        score = self._count % self._step == 0
        ref_address, dist_address, pitch = self._shared.next()
        if self.pictures:
            self._release_done()
            ref, dis = ref_stream.slot_texture(ref_slot), test_stream.slot_texture(test_slot)
            vmaf_vulkan.check(self._vulkan, self._vulkan.vv_pictures(self._gpu, *ref, *dis), f"Scoring frame {self._count}")
            vmaf_vulkan.check(self._vulkan, self._vulkan.vv_commit(self._gpu, int(score)), f"Scoring frame {self._count}")
            if score and self._cpu_features:
                self._to_libvmaf_decoded(ref_stream, ref_slot, test_stream, test_slot)
            self._count += 1
            return
        # The copies into the shared buffers. When the GPU waits for them
        # (self._after), this thread does not: each copy returns the value
        # the decoder's timeline reaches once the copy is done, else 0.
        luma = ((lambda stream, *a: stream.copy_luma_async(*a)) if self._after else
                (lambda stream, *a: stream.copy_luma(*a) or 0))
        planes = ((lambda stream, *a: stream.copy_planes_async(*a)) if self._after else
                  (lambda stream, *a: stream.copy_planes(*a) or 0))
        chroma_copied = score and _SPEED in self._on_gpu and self._shared_chroma is not None
        if chroma_copied:
            # The GPU copies each picture's chroma planes into the slot's shared
            # buffer along with its luma (one copy on the GPU per picture).
            offset, spacing, stride = self._shared_chroma
            at = [ref_address + offset + i * spacing for i in range(4)]
            copied = max(planes(ref_stream, ref_slot, (ref_address, at[0], at[1]), (pitch, stride, stride)),
                         planes(test_stream, test_slot, (dist_address, at[2], at[3]), (pitch, stride, stride)))
        else:
            copied = luma(ref_stream, ref_slot, ref_address, pitch)
            if score:
                copied = max(copied, luma(test_stream, test_slot, dist_address, pitch))
        if copied:
            vmaf_vulkan.check(self._vulkan, self._vulkan.vv_commit_after(self._gpu, copied),
                               f"Scoring frame {self._count}")
        if score and _SPEED in self._on_gpu and not chroma_copied:
            # The chroma planes are downloaded into the engine's memory for
            # them. That memory is Vulkan's, so unlike libvmaf's pictures it is
            # not page-locked through the decoder (see _pin).
            chroma, stride = (ctypes.c_void_p * 4)(), ctypes.c_uint32()
            vmaf_vulkan.check(self._vulkan, self._vulkan.vv_chroma_staging(self._gpu, chroma, ctypes.byref(stride)),
                               f"Scoring frame {self._count}")
            pitches = (0, stride.value, stride.value)
            ref_stream.download_planes(ref_slot, (None, chroma[0], chroma[1]), pitches)
            test_stream.download_planes(test_slot, (None, chroma[2], chroma[3]), pitches)
        vmaf_vulkan.check(self._vulkan, self._vulkan.vv_commit(self._gpu, int(score)),
                           f"Scoring frame {self._count}")
        if score and self._cpu_features:
            self._to_libvmaf_decoded(ref_stream, ref_slot, test_stream, test_slot)
        self._count += 1

    def release_later(self, stream, slot: int, reference: bool) -> None:
        """In pictures mode, releases `stream`'s `slot`, which the pairs given
        so far no longer need, once the engine has read it: after the last
        frame given, or for a source picture after the next frame too,
        because the next frame's motion feature reads it."""
        self._held.append((self._count - 1 + (1 if reference else 0), stream, slot))

    def _release_done(self) -> None:
        """Releases the slots held by release_later that the engine is done
        with. The engine has finished every frame up to one round of shared
        slots back, because starting a frame in a shared slot waits for the
        frame that used the slot before."""
        done = self._count - self._shared.slots
        while self._held and self._held[0][0] <= done:
            _last, stream, slot = self._held.popleft()
            stream.release(slot)

    def _release_all(self) -> None:
        """Releases every slot release_later holds, once the engine is done."""
        while self._held:
            _last, stream, slot = self._held.popleft()
            stream.release(slot)

    def _to_libvmaf_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Gives libvmaf's extractors the pair. The GPU downloads the planes
        they read straight into libvmaf's pictures."""
        lib = self._lib
        ref, dist = libvmaf.Picture(), libvmaf.Picture()
        libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(ref)), "Taking a picture")
        try:
            libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(dist)),
                             "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        try:
            for picture in (ref, dist):
                self._pin(picture, test_stream)
            pitches = (dist.stride[0], dist.stride[1], dist.stride[2])
            # As in _to_libvmaf: the whole test frame, and the source's chroma.
            test_stream.download_planes(test_slot, (dist.data[0], dist.data[1], dist.data[2]), pitches)
            ref_stream.download_planes(ref_slot, (None, ref.data[1], ref.data[2]), pitches)
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            lib.vmaf_picture_unref(ctypes.byref(dist))
            raise
        libvmaf.check(lib.vmaf_read_pictures(self._cpu, ctypes.byref(ref), ctypes.byref(dist), self._count),
                         f"Scoring frame {self._count}")

    def _pin(self, picture, stream) -> None:
        """Page-locks a picture from libvmaf's pool the first time it is used,
        so that the GPU can write into it by itself. The pool's pictures are
        used again and again, so each is locked only once."""
        address = picture.data[0]
        if address not in self._pinned:
            size = picture.stride[0] * picture.h[0] + 2 * picture.stride[1] * picture.h[1]
            self._pinned[address] = stream if stream.pin(address, size) else None

    def _to_libvmaf(self, reference: bytearray, distorted: bytearray) -> None:
        """Gives libvmaf's extractors the pair: the whole test frame (CAMBI
        reads its luma) and the source's chroma (SpEED reads both videos'
        chroma, and nothing here reads the source's luma)."""
        lib = self._lib
        ref, dist = libvmaf.Picture(), libvmaf.Picture()
        libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(ref)), "Taking a picture")
        try:
            libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._cpu, ctypes.byref(dist)), "Taking a picture")
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
        libvmaf.check(lib.vmaf_read_pictures(self._cpu, ctypes.byref(ref), ctypes.byref(dist), self._count),
                         f"Scoring frame {self._count}")

    def _fill(self, picture, frame: bytearray, planes: tuple[int, ...]) -> None:
        """Copies the given planes of a packed frame into a libvmaf picture."""
        source = np.frombuffer(frame, dtype=np.uint8)
        for plane in planes:
            offset, rows, row_bytes = self._planes[plane]
            # libvmaf rounds an odd size's chroma planes down, and FFmpeg up.
            rows = min(rows, picture.h[plane])
            width = min(row_bytes, picture.w[plane] * self._sample)
            stride = picture.stride[plane]
            target = np.ctypeslib.as_array(
                ctypes.cast(picture.data[plane], ctypes.POINTER(ctypes.c_uint8)), shape=(rows, stride))
            target[:, :width] = source[offset:offset + rows * row_bytes].reshape(rows, row_bytes)[:, :width]

    def features(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the frame numbers scored and each feature's values for
        them. Call it once every frame has been given."""
        lib = self._lib
        vmaf_vulkan.check(self._vulkan, self._vulkan.vv_flush(self._gpu), "Scoring")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        rows = np.zeros((self._count, 6), dtype=np.float64)
        if self._count:
            if self._cpu_features:
                libvmaf.check(lib.vmaf_read_pictures(self._cpu, None, None, 0), "Finishing")
            self._vulkan.vv_features_v1(self._gpu, rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count)
        values = {_ADM3: rows[frames, 0], _MOTION3: rows[frames, 3]}
        if _SPEED in self._on_gpu:
            speed = np.full((self._count, 3), np.nan, dtype=np.float64)
            if self._count:
                vmaf_vulkan.check(self._vulkan, self._vulkan.vv_v1_speed_scores(
                    self._gpu, speed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count), "Reading SpEED")
            values[_SPEED] = speed[frames, 2]  # speed_chroma_uv_score
        if _CAMBI in self._on_gpu:
            cambi = np.full(self._count, np.nan, dtype=np.float64)
            if self._count:
                vmaf_vulkan.check(self._vulkan, self._vulkan.vv_v1_cambi_scores(
                    self._gpu, cambi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self._count), "Reading CAMBI")
            values[_CAMBI] = cambi[frames]
        value = ctypes.c_double()
        for feature, name in self._cpu_names.items():
            if feature in self._on_gpu:
                continue
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                libvmaf.check(lib.vmaf_feature_score_at_index(self._cpu, name.encode(), ctypes.byref(value),
                                                                 int(frame)), f"Reading frame {frame}")
                column[slot] = value.value
            values[feature] = column
        return frames, values

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the frame numbers scored and the model's scores for them,
        rounded as libvmaf's JSON log rounds them (as GpuScorer.finish
        does)."""
        if not self._count:
            self._release_all()
            return np.zeros(0, dtype=np.int32), {self._name: np.zeros(0)}
        frames, values = self.features()
        self._release_all()  # the engine has finished every frame
        return frames, {self._name: predict(self._model_path, frames, values, self._model_names)}

    def close(self) -> None:
        """Frees the scorer. When the frames came from decoders, call it
        before the decoders are closed."""
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        if self._gpu:
            self._vulkan.vv_destroy(self._gpu)  # waits for its frames
            self._gpu = ctypes.c_void_p()
        self._release_all()
        if self._cpu:
            self._lib.vmaf_close(self._cpu)  # waits for the extractors, so nothing reads the pictures after it
            self._cpu = ctypes.c_void_p()
        for address, stream in self._pinned.items():
            if stream is not None:
                stream.unpin(address)
        self._pinned = {}


def predict(model_path: Path, frames: np.ndarray, values: dict[str, np.ndarray],
            names: dict[str, str] | None = None) -> np.ndarray:
    """Returns a VMAF v1 model's scores for its four features' values,
    predicted by libvmaf itself from the features given under the names the
    model reads them by."""
    lib = _libvmaf()
    names = names or feature_names(model_path)[0]
    context, model = ctypes.c_void_p(), ctypes.c_void_p()
    configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, 0, 1, 0, 0)
    libvmaf.check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
    try:
        config = libvmaf.ModelConfig(b"vmaf", 0)
        libvmaf.check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                       libvmaf.path_bytes(model_path)), "Loading the model")
        for feature, name in names.items():
            encoded = name.encode()
            for frame, value in zip(frames, values[feature], strict=True):
                libvmaf.check(lib.vmaf_import_feature_score(context, encoded, float(value), int(frame)),
                                 f"Giving libvmaf frame {frame}")
        scores = np.empty(len(frames), dtype=np.float64)
        value = ctypes.c_double()
        for slot, frame in enumerate(frames):
            libvmaf.check(lib.vmaf_score_at_index(context, model, ctypes.byref(value), int(frame)),
                             f"Predicting frame {frame}")
            scores[slot] = float(f"{value.value:.6f}")
        return scores
    finally:
        if model:
            lib.vmaf_model_destroy(model)
        lib.vmaf_close(context)


def cpu_reference(model_path: Path, width: int, height: int, bit_depth: int, reference: list, distorted: list,
                  n_subsample: int = 1, threads: int = 0):
    """Calculates VMAF v1 of frames in memory with libvmaf's CPU code, and
    returns (the frame numbers scored, {feature: values}, the scores rounded
    as V1Scorer rounds them). V1Scorer is checked against it by the
    self-test here and by scripts/compare_vmaf_v1.py."""
    lib = _libvmaf()
    names, _ = feature_names(model_path)
    context, model = ctypes.c_void_p(), ctypes.c_void_p()
    configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, threads, max(1, n_subsample), 0, 0)
    libvmaf.check(lib.vmaf_init(ctypes.byref(context), configuration), "Starting libvmaf")
    try:
        config = libvmaf.ModelConfig(b"vmaf", 0)
        libvmaf.check(lib.vmaf_model_load_from_path(ctypes.byref(model), ctypes.byref(config),
                                                       libvmaf.path_bytes(model_path)), "Loading the model")
        libvmaf.check(lib.vmaf_use_features_from_model(context, model), "Setting up the model")
        pictures = libvmaf.PictureConfiguration(
            libvmaf.PictureParameters(width, height, bit_depth, libvmaf.PIX_FMT_YUV420P),
            2 * (max(1, threads) + 2))
        libvmaf.check(lib.vmaf_preallocate_pictures(context, pictures), "Allocating pictures")
        layout = V1Scorer.__new__(V1Scorer)
        layout._set_layout(width, height, bit_depth)
        for index, frames in enumerate(zip(reference, distorted, strict=True)):
            ref, dist = libvmaf.Picture(), libvmaf.Picture()
            try:
                for picture, frame in zip((ref, dist), frames, strict=True):
                    libvmaf.check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(picture)),
                                     "Taking a picture")
                    layout._fill(picture, frame, (0, 1, 2))
            except BaseException:
                # A picture taken and never handed over keeps vmaf_close waiting for it.
                for picture in (ref, dist):
                    if picture.data[0]:
                        lib.vmaf_picture_unref(ctypes.byref(picture))
                raise
            libvmaf.check(lib.vmaf_read_pictures(context, ctypes.byref(ref), ctypes.byref(dist), index),
                             f"Scoring frame {index}")
        libvmaf.check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing")
        scored = np.arange(0, len(reference), max(1, n_subsample), dtype=np.int32)
        value = ctypes.c_double()
        values = {}
        for feature, name in names.items():
            column = np.empty(len(scored), dtype=np.float64)
            for slot, frame in enumerate(scored):
                libvmaf.check(lib.vmaf_feature_score_at_index(context, name.encode(), ctypes.byref(value),
                                                                 int(frame)), f"Reading frame {frame}")
                column[slot] = value.value
            values[feature] = column
        scores = np.empty(len(scored), dtype=np.float64)
        for slot, frame in enumerate(scored):
            libvmaf.check(lib.vmaf_score_at_index(context, model, ctypes.byref(value), int(frame)),
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
    """Feeds the same frames to a run's GPU scorers: the one for VMAF v0.6.1
    and NEG (libvmaf's CUDA code or the Vulkan port, chosen by `backend`),
    and the one for VMAF v1. `models` is as vmaf_gpu.gpu_models gives it,
    with "vmaf_v1" mapped to the model's "path=<file>"."""

    def __init__(self, width: int, height: int, bit_depth: int, models: dict[str, str], n_subsample: int = 1,
                 backend: str = "cuda", device: int | None = None, shared=None, pictures: bool = False):
        """`shared` is the decoder (GpuFrameStream) whose pictures the scorers
        then take on the GPU (add_decoded) instead of from system memory
        (add). LibvmafError is raised if they cannot.

        `pictures` makes VMAF v1's scorer read the decoders' textures (see
        V1Scorer) if the engine can, and self.pictures says whether it does.
        The decoders' slots are then given back through release_later."""
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
                    self._vmaf = vmaf_gpu.GpuScorer(width, height, bit_depth, models, n_subsample,
                                                     on_device=shared is not None)
                self._scorers.append(self._vmaf)
            if v1_model is not None:
                self._scorers.append(V1Scorer(width, height, bit_depth, Path(v1_model.removeprefix("path=")),
                                              n_subsample, device=device if backend == "vulkan" else None,
                                              shared=shared, pictures=pictures))
        except BaseException:
            self.close()
            raise
        self.frame_bytes = self._scorers[0].frame_bytes
        #: VMAF v1's scorer if it reads the decoders' pictures where they are,
        #: else None.
        self._pictures = next((scorer for scorer in self._scorers if isinstance(scorer, V1Scorer) and scorer.pictures),
                              None)
        self.pictures = self._pictures is not None

    def release_later(self, stream, slot: int, reference: bool) -> None:
        """In pictures mode, gives a decoder's slot that the pairs no longer
        need back once the GPU has read it."""
        self._pictures.release_later(stream, slot, reference)

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Scores one more pair held by the decoders, without a copy through
        the CPU."""
        for scorer in self._scorers:
            if isinstance(scorer, V1Scorer):
                scorer.add_decoded(ref_stream, ref_slot, test_stream, test_slot)
            else:  # VMAF v0.6.1 and NEG read only the luma planes, copied on the GPU
                (scorer.add_shared if isinstance(scorer, vmaf_vulkan.VulkanScorer) else scorer.add_on_device)(
                    lambda address, pitch: ref_stream.copy_luma(ref_slot, address, pitch),
                    lambda address, pitch: test_stream.copy_luma(test_slot, address, pitch))

    def add(self, reference: bytearray, distorted: bytearray) -> None:
        """Scores one more pair given in system memory."""
        for scorer in self._scorers:
            scorer.add(reference, distorted)

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the frame numbers scored and every scorer's scores for
        them."""
        frames, scores = None, {}
        for scorer in self._scorers:
            numbers, values = scorer.finish()
            if frames is not None and not np.array_equal(frames, numbers):
                raise VmafV1Error("the GPU's scorers scored different frames")
            frames = numbers
            scores.update(values)
        return frames, scores

    def close(self) -> None:
        """Frees every scorer."""
        for scorer in self._scorers:
            scorer.close()
        self._scorers = []


def score_decoded(
    source, distorted, source_crop=None, distorted_crop=None, *, width: int, height: int, bit_depth: int,
    models: dict[str, str], n_subsample: int, duration_limit: str | None, total_frames: int,
    scale_algorithm: str = "bicubic", on_progress=None, check_cancel=lambda: None, process_handle=None,
    backend: str = "cuda", device: int | None = None, decoder: str = "nvidia",
):
    """The counterpart of vmaf_gpu.score_decoded for a run with VMAF v1.

    Both videos are decoded in this process by the GPU's own decoder
    (gpu_frames), as whole frames because SpEED reads the chroma. They are
    cropped, scaled and widened there, paired as libvmaf's filter pairs them
    (frame_sync), and each pair is given to the run's scorers (MultiScorer).
    FFmpeg only copies the compressed streams out of their containers.
    `decoder` is gpu_frames' decoder library: one name for both videos, or a
    pair (the source's, the test video's)."""
    import time
    from dataclasses import replace

    from vmaf_app.core import frame_sync, gpu_frames

    source_decoder, test_decoder = gpu_frames.decoder_pair(decoder)
    for library in (source_decoder, test_decoder):
        if not gpu_frames.available(library):
            raise gpu_frames.GpuDecodeUnavailableError(f"the {library} decoder library is not bundled")
    plans = []
    for info, crop, library in ((distorted, distorted_crop, test_decoder), (source, source_crop, source_decoder)):
        plan = gpu_frames.plan_decode(info, crop, shift=6, size=(width, height), algorithm=scale_algorithm)
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
    # When the engine reads pictures where the decoder left them (V1Scorer's
    # pictures mode), they stay the decoder's until the GPU is done with them:
    # two pairs being matched up, the frames in flight (3), and the previous
    # source picture for motion. That takes six of a decoder's slots, and is
    # faster (4K HEVC: 98.2 -> 106.1 fps). It is used for two H.264 videos
    # too. Windows' H.264 decoder gives layers of a texture array, which are
    # copied into textures of their own anyway (native/mf_frames.cpp), so the
    # speed is the same (1080p: 251.0 -> 251.3 fps), but it uses less GPU
    # memory, because the engine's buffers for the planes are not made
    # (1080p: 342 -> 291 MB committed).
    pool = 6 if (backend == "vulkan" and V1_KEY in models and source_decoder == test_decoder
                 and gpu_frames.hands_over_textures(test_decoder)) else 4
    test = gpu_frames.GpuFrameStream(distorted, plans[0], 0, pool=pool, process_handle=process_handle,
                                     backend=test_decoder, handover=True)
    try:
        ref = gpu_frames.GpuFrameStream(source, plans[1], 0, pool=pool, process_handle=process_handle,
                                        backend=source_decoder, handover=True)
    except BaseException:
        test.close()
        raise
    scorer = None
    try:
        # The decoder's pictures stay on the GPU and reach the scorers without
        # a copy through the CPU. This works with NVIDIA's decoder, and with
        # AMD's when it hands over to Vulkan (not to libvmaf's CUDA code),
        # when the same decoder decodes both videos.
        if (test.handover and ref.handover and source_decoder == test_decoder
                and (test_decoder == "nvidia" or backend == "vulkan")):
            try:
                scorer = MultiScorer(width, height, bit_depth, models, n_subsample, backend, device, shared=test,
                                     pictures=pool > 4 and test.gives_textures and ref.gives_textures)
            except libvmaf.LibvmafError as error:
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
        release_test, release_ref = test.release, ref.release
        if on_gpu and scorer.pictures:  # given back once the GPU has read them
            release_test = functools.partial(scorer.release_later, test, reference=False)
            release_ref = functools.partial(scorer.release_later, ref, reference=True)
        pairs = frame_sync.decoded_frame_pairs(test, ref, distorted, duration_limit, check_cancel,
                                               release_test, release_ref)
        count = 0
        started = reported = time.perf_counter()
        try:
            for test_slot, ref_slot in pairs:
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
        if scorer is not None:  # closed first, because it holds on to the decoders' memory
            scorer.close()
        test.close()
        ref.close()


# ------------------------------------------------------------ availability

def _self_test_frames(bits: int, count: int = 4) -> tuple[list[bytearray], list[bytearray]]:
    """Makes frame pairs for the self-test: textured and moving, with the
    test frame blurred and noisy. They are made with integer arithmetic
    only, so they are the same bytes on every PC."""
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
    """Returns whether VMAF v1 can be calculated with the GPU on this PC, and
    what the GPU is or why not. The GPU is accepted only if it gives
    libvmaf's CPU values exactly: first vmaf_vulkan's self-test of the
    shaders it shares with VMAF v0.6.1, then the four features of a few
    frames, at 8 and at 10 bits and with the five-frame motion window,
    compared with libvmaf's CPU extractors. A driver that compiles a shader
    wrongly gives wrong scores, not an error. Run it in a process of its
    own."""
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
    except (OSError, libvmaf.LibvmafError) as error:
        return False, str(error)


_PROBE_LOCK = threading.Lock()
_probed: tuple[bool, str] | None = None


def available() -> tuple[bool, str]:
    """Returns whether VMAF v1 is calculated with the GPU here, and on which
    GPU or why not. The GPU is probed once, in a process of its own."""
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
    """Returns whether a run's VMAF v1 is calculated with the GPU: when
    `model` is a VMAF v1 model, `size` (if known) leaves SpEED blocks to
    score, and this PC's GPU passed the self-test. Otherwise FFmpeg's
    libvmaf calculates it. `model` is as vmaf_gpu.gpu_models names it
    ("path=<file>"), or empty while the job is planned and Auto is not yet
    resolved."""
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
