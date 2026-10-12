"""The bundled libvmaf (tools/libvmaf/libvmaf.dll): loading it, and its C
API's structures and constants, for the modules that call it in the app's
own process: vmaf_gpu (VMAF and NEG with CUDA), vmaf_vulkan and vmaf_v1_gpu
(which have libvmaf predict the scores from their features), and
libvmaf_cpu (PSNR, SSIM and XPSNR).

The library is libvmaf-fast (github.com/4KVCD/libvmaf-fast), a fork of
libvmaf. scripts/fetch_libvmaf_fast.ps1 installs a release of it, and
scripts/build_libvmaf_fast_local.ps1 a build of its latest commit while the
app is being developed.

Use the names here through the module (libvmaf.Picture, libvmaf.load()),
not imported on their own: load() sets Picture to the picture layout of
the libvmaf it loads.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "tools" / "libvmaf" / "libvmaf.dll"
#: Where the script that installed the bundled libvmaf.dll and vmaf_vulkan.dll
#: recorded what they were built from: a libvmaf-fast release
#: (scripts/fetch_libvmaf_fast.ps1) or, while the app is being developed, a
#: local build of the fork's latest commit (scripts/build_libvmaf_fast_local.ps1).
BUILD_RECORD = LIBRARY_PATH.with_name("libvmaf-fast.json")


def _bundled_version(record: Path) -> str:
    """The libvmaf-fast version the build record names ("v1" for release
    v1, "v1-2-g211f80e0" for a local build two commits after it), or
    "unknown" without a record."""
    try:
        return str(json.loads(record.read_text(encoding="utf-8"))["version"])
    except (OSError, ValueError, KeyError, TypeError):
        return "unknown"


LIBVMAF_FAST_VERSION = _bundled_version(BUILD_RECORD)

#: Values of libvmaf's enums: VmafPixelFormat's 4:2:0 and luma-only (4:0:0)
#: formats, and VmafLogLevel's errors-only level.
PIX_FMT_YUV420P = 1
PIX_FMT_YUV400P = 4
LOG_LEVEL_ERROR = 1


class LibvmafError(RuntimeError):
    """A call into the bundled libvmaf failed. Its callers then calculate
    the scores another way, with FFmpeg."""


# ------------------------------------------------------- structures and loading

class Configuration(ctypes.Structure):
    """libvmaf's VmafConfiguration."""
    _fields_ = [("log_level", ctypes.c_int), ("n_threads", ctypes.c_uint), ("n_subsample", ctypes.c_uint),
                ("cpumask", ctypes.c_uint64), ("gpumask", ctypes.c_uint64)]


class PlainPicture(ctypes.Structure):
    """libvmaf's VmafPicture, as libvmaf-fast v3.2.0-fast.1 lays it out."""
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


class ColorPicture(ctypes.Structure):
    """libvmaf's VmafPicture since upstream libvmaf added vmaf_picture_convert
    (Netflix/vmaf 0497a0f2, in libvmaf-fast after v3.2.0-fast.1): it has a
    VmafColor (range, primaries, transfer and matrix, four enums) between
    data and ref. Using the old layout with such a libvmaf makes it write
    past the app's pictures (heap corruption in vmaf_picture_alloc) and read
    ref and priv from the wrong place."""
    _fields_ = [("pix_fmt", ctypes.c_int), ("bpc", ctypes.c_uint), ("w", ctypes.c_uint * 3),
                ("h", ctypes.c_uint * 3), ("stride", ctypes.c_ssize_t * 3), ("data", ctypes.c_void_p * 3),
                ("color", ctypes.c_int * 4), ("ref", ctypes.c_void_p), ("priv", ctypes.c_void_p)]


#: The VmafPicture layout of the libvmaf loaded. load sets it, so pictures
#: are made through this name only after load.
Picture: type[PlainPicture | ColorPicture] = PlainPicture
#: A picture of either layout, for annotations: Picture is a variable, which
#: a type checker cannot use as a type. Both layouts have the fields the
#: app reads (data, stride, w, h).
AnyPicture = PlainPicture | ColorPicture


class PictureParameters(ctypes.Structure):
    """libvmaf's VmafPictureParameters: a picture's size, depth and format."""
    _fields_ = [("w", ctypes.c_uint), ("h", ctypes.c_uint), ("bpc", ctypes.c_uint), ("pix_fmt", ctypes.c_int)]


class PictureConfiguration(ctypes.Structure):
    """libvmaf's pool of pictures in system memory, used by vmaf_v1_gpu for
    VMAF v1's CPU features. GpuScorer's pictures are in GPU memory."""

    _fields_ = [("pic_params", PictureParameters), ("pic_cnt", ctypes.c_uint)]


class ModelConfig(ctypes.Structure):
    """libvmaf's VmafModelConfig."""
    _fields_ = [("name", ctypes.c_char_p), ("flags", ctypes.c_uint64)]


class CudaConfiguration(ctypes.Structure):
    """libvmaf's VmafCudaConfiguration: the CUDA context to use."""
    _fields_ = [("cu_ctx", ctypes.c_void_p)]


class CudaPictureConfiguration(ctypes.Structure):
    """libvmaf's VmafCudaPictureConfiguration."""
    _fields_ = [("pic_params", PictureParameters), ("pic_prealloc_method", ctypes.c_int)]


#: VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE: libvmaf's own pictures in
#: GPU memory. Frames decoded on the GPU are copied into them there, and
#: frames from system memory are uploaded into them (vmaf_gpu._HostUpload).
PREALLOCATE_ON_DEVICE = 1


_library: ctypes.CDLL | None = None


def load() -> ctypes.CDLL:
    """Loads the bundled libvmaf once and declares its functions' types."""
    global _library, Picture
    if _library is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        # The picture layout this libvmaf uses: the colour field came with
        # vmaf_picture_convert.
        Picture = ColorPicture if hasattr(lib, "vmaf_picture_convert") else PlainPicture
        handle, pointer = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        for name, restype, argtypes in (
            ("vmaf_version", ctypes.c_char_p, []),
            ("vmaf_init", ctypes.c_int, [pointer, Configuration]),
            ("vmaf_cuda_state_init", ctypes.c_int, [pointer, CudaConfiguration]),
            ("vmaf_cuda_import_state", ctypes.c_int, [handle, handle]),
            ("vmaf_model_load", ctypes.c_int, [pointer, ctypes.POINTER(ModelConfig), ctypes.c_char_p]),
            ("vmaf_use_features_from_model", ctypes.c_int, [handle, handle]),
            ("vmaf_preallocate_pictures", ctypes.c_int, [handle, PictureConfiguration]),
            ("vmaf_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(Picture)]),
            ("vmaf_cuda_preallocate_pictures", ctypes.c_int, [handle, CudaPictureConfiguration]),
            ("vmaf_cuda_fetch_preallocated_picture", ctypes.c_int, [handle, ctypes.POINTER(Picture)]),
            ("vmaf_read_pictures", ctypes.c_int,
             [handle, ctypes.POINTER(Picture), ctypes.POINTER(Picture), ctypes.c_uint]),
            ("vmaf_score_at_index", ctypes.c_int, [handle, handle, ctypes.POINTER(ctypes.c_double), ctypes.c_uint]),
            ("vmaf_picture_unref", ctypes.c_int, [ctypes.POINTER(Picture)]),
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


def check(error: int, what: str) -> None:
    """Raises LibvmafError if a libvmaf call returned an error."""
    if error:
        raise LibvmafError(f"{what} failed (libvmaf error {error})")


def path_bytes(path) -> bytes:
    """An existing file's path encoded as libvmaf needs it to open the file.

    libvmaf opens files with fopen(), which reads the name in Windows' ANSI
    code page, not UTF-8. (Encoded as UTF-8, a folder name with an accented
    letter, such as the user's name in their profile and Temp folder, would
    name a file that is not there.) A name with letters the code page lacks
    (Chinese on a Western Windows, say) is given by its short 8.3 path,
    where the drive keeps those; otherwise LibvmafError is raised."""
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
    raise LibvmafError(f"libvmaf cannot open {path}: its name has letters outside this PC's code page")
