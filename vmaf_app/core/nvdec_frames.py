"""Decoding a video on the GPU in the scoring process for the GPU metrics:
on NVIDIA GPUs straight into GPU memory (vmaf_app/native/nvdec_frames.dll),
on Intel's and AMD's into system memory through their makers' decoder
libraries (vpl_frames.dll with oneVPL, amf_frames.dll with AMF) -- all built
by scripts/build_gpu_frames.ps1 from native/, with one C API
(native/gpu_frames.h).

The GPU metrics used to get their frames from FFmpeg: FFmpeg decoded on the
GPU, copied each picture back to system memory, converted it and wrote it
to a pipe, which the scoring process read into memory the GPU then copied it
from again. At 4K that is several CPU copies of 25 MB per frame and per
video -- most of a GPU metric's CPU use, and at 200 frames per second more
than a pipe carries. Here FFmpeg only copies the compressed stream out of
its container (-c:v copy: a few MB per second) and the GPU's decoder decodes
it in this process. NVIDIA's pictures stay on the GPU: libvmaf reads them
there, and Vship's page-locked buffers are filled by the GPU's copy engine.
Intel's and AMD's decoders hand their pictures over in system memory, and
one CPU pass puts each into Vship's buffer (on Intel's integrated GPUs,
system memory is the GPU's own).

The pictures are the ones FFmpeg's decode gives, sample for sample: the same
decoder hardware, the stream's display area, a crop rounded as FFmpeg's crop
filter rounds it (its edges to even, for 4:2:0), and P016's 10-bit samples
either kept in the top bits (Vship reads them as 16-bit) or shifted down,
which is FFmpeg's conversion to yuv420p10le exactly -- and 8-bit samples
widened to 10 bits as FFmpeg widens them. A picture the comparison scales is
scaled here with FFmpeg's filter for the algorithm chosen (native/
scale_filter.h; by the GPU on NVIDIA's, the CPU on Intel's and AMD's): not
FFmpeg's to the sample -- within 1 of it on film -- which the user decided is
the same comparison (ComparisonRecipe.identity_dict). Which pictures come out,
and with which timestamps, is checked against the packets that went in
(NvdecStream.verify): a picture the decoder dropped or added fails the run,
and the caller makes it again through FFmpeg.

What is not decoded here -- another GPU, a codec or format the decoder
does not take, an interlaced or damaged stream -- goes the
FFmpeg way as before (NvdecUnavailableError before the first picture,
NvdecFailedError after it).
"""
from __future__ import annotations

import _winapi
import contextlib
import ctypes
import heapq
import logging
import msvcrt
import os
import queue
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import VIDEO_STREAM, ffmpeg_path, ffprobe_path
from vmaf_app.core.models import CropBox, VideoInfo

_log = logging.getLogger(__name__)

_NATIVE = Path(__file__).resolve().parents[1] / "native"
#: Each GPU maker's decoder library, all with one C API (native/gpu_frames.h):
#: NVIDIA's decoder through nvcuvid, Intel's through oneVPL, AMD's through AMF.
LIBRARIES = {"nvidia": _NATIVE / "nvdec_frames.dll", "intel": _NATIVE / "vpl_frames.dll",
             "amd": _NATIVE / "amf_frames.dll"}
LIBRARY_PATH = LIBRARIES["nvidia"]

#: FFmpeg's codec names -> NVDEC's (cudaVideoCodec), and the bitstream filters
#: that turn the container's packets into what NVDEC's parser reads: Annex B
#: start codes for H.264 and HEVC, with the parameter sets from the stream's
#: configuration before every keyframe -- mp4toannexb adds them only before
#: an IDR, and an open-GOP H.264 stream cut at a recovery point has none, so
#: NVDEC decoded nothing of it. AV1's packets are its low-overhead OBU format
#: already; its sequence header goes to the parser separately.
_CODECS = {"h264": (4, "h264_mp4toannexb,dump_extra=freq=keyframe"),
           "hevc": (8, "hevc_mp4toannexb,dump_extra=freq=keyframe"), "av1": (11, None)}
#: Packet flags (AV_PKT_FLAG_*).
_KEY, _DISCARD = 0x1, 0x4
#: FFmpeg warnings that mean the timestamps it copied may not be the ones its
#: decode would give a picture (it rewrites non-monotonic ones as it muxes).
_TIMESTAMP_WARNING = re.compile(r"timestamp|\bdts\b|\bpts\b|monoton", re.IGNORECASE)
#: Pixel formats FFmpeg decodes these to -> bit depth. 4:2:0 only: 4:2:2 and
#: 4:4:4 are not decoded here.
_DEPTHS = {"yuv420p": 8, "yuvj420p": 8, "yuv420p10le": 10, "yuv420p10be": 10}

_NVF_FRAME, _NVF_END, _NVF_ERROR, _NVF_ABORTED, _NVF_TIMEOUT = 1, 0, -1, -2, 2
_ES_PIPE_BYTES = 8 * 1024 * 1024
_NOPTS = -(1 << 63)


class NvdecUnavailableError(RuntimeError):
    """This video is not decoded here; FFmpeg decodes it, as before."""


class NvdecFailedError(RuntimeError):
    """Decoding failed after it had started; the caller decodes the video
    again through FFmpeg."""


# ------------------------------------------------------------------ binding

class _Params(ctypes.Structure):
    _fields_ = [("device", ctypes.c_int), ("codec", ctypes.c_int), ("bit_depth", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("crop_x", ctypes.c_int), ("crop_y", ctypes.c_int), ("crop_w", ctypes.c_int), ("crop_h", ctypes.c_int),
                ("shift", ctypes.c_int), ("luma_only", ctypes.c_int), ("pool", ctypes.c_int),
                ("extradata", ctypes.c_void_p), ("extradata_size", ctypes.c_int),
                ("out_w", ctypes.c_int), ("out_h", ctypes.c_int), ("scaler", ctypes.c_int), ("widen", ctypes.c_int)]


class _Info(ctypes.Structure):
    _fields_ = [("coded_width", ctypes.c_int), ("coded_height", ctypes.c_int),
                ("display_left", ctypes.c_int), ("display_top", ctypes.c_int),
                ("display_right", ctypes.c_int), ("display_bottom", ctypes.c_int),
                ("bit_depth", ctypes.c_int), ("chroma_format", ctypes.c_int), ("progressive", ctypes.c_int),
                ("decode_surfaces", ctypes.c_int),
                ("decoded", ctypes.c_longlong), ("displayed", ctypes.c_longlong), ("frame_bytes", ctypes.c_longlong)]


_libraries: dict[str, ctypes.CDLL] = {}
_library_lock = threading.Lock()


def _load(backend: str = "nvidia") -> ctypes.CDLL:
    with _library_lock:
        if backend not in _libraries:
            path = LIBRARY_PATH if backend == "nvidia" else LIBRARIES[backend]
            if not path.is_file():
                raise NvdecUnavailableError(f"the GPU frame decoder ({path.name}) is not bundled")
            try:
                lib = ctypes.CDLL(str(path))
            except OSError as error:
                raise NvdecUnavailableError(f"the GPU frame decoder could not be loaded: {error}") from error
            handle, text = ctypes.c_void_p, ctypes.c_char_p
            for name, restype, argtypes in (
                ("nvf_open", handle, [ctypes.POINTER(_Params), text, ctypes.c_int]),
                ("nvf_push", ctypes.c_int, [handle, ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]),
                ("nvf_finish", ctypes.c_int, [handle]),
                ("nvf_pop", ctypes.c_int,
                 [handle, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong)]),
                ("nvf_download", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_void_p]),
                ("nvf_copy_luma", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_longlong]),
                ("nvf_release", None, [handle, ctypes.c_int]),
                ("nvf_abort", None, [handle]),
                ("nvf_close", None, [handle]),
                ("nvf_error", ctypes.c_int, [handle, text, ctypes.c_int]),
                ("nvf_info", None, [handle, ctypes.POINTER(_Info)]),
                ("nvf_supports", ctypes.c_int,
                 [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, text, ctypes.c_int]),
            ):
                function = getattr(lib, name)
                function.restype, function.argtypes = restype, argtypes
            _libraries[backend] = lib
        return _libraries[backend]


# ------------------------------------------------------------------- plans

#: The app's scaling algorithms (VmafOptions.scale_algorithm) -> the
#: decoders' filters (native/scale_filter.h).
_SCALERS = {"bilinear": 0, "bicubic": 1, "lanczos": 2, "spline": 3}
#: How an 8-bit video is widened to the 10 bits it is compared at: as FFmpeg
#: converts it, shifted left by 2 -- full range's luma with its top bits
#: repeated in the bottom ones.
WIDEN_SHIFT, WIDEN_REPEAT = 1, 2


@dataclass(frozen=True)
class DecodePlan:
    """How one video is decoded here: its codec, depth, the rectangle handed
    on (FFmpeg's crop, rounded as FFmpeg rounds it), the size it is scaled to
    and with which filter, and the sample layout."""

    codec: str
    bit_depth: int
    width: int
    height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    #: Right shift of 16-bit samples: 6 for yuv420p10le, 0 to keep P016's
    #: layout (Vship reads it as 16-bit).
    shift: int = 0
    #: Only the luma plane (VMAF reads nothing else).
    luma_only: bool = False
    #: The size the crop is scaled to (0: not scaled), and the filter.
    out_w: int = 0
    out_h: int = 0
    scaler: str = "bicubic"
    #: 8-bit pictures handed back as 10-bit (WIDEN_SHIFT, WIDEN_REPEAT; 0: not).
    widen: int = 0

    @property
    def bytes_per_sample(self) -> int:
        """Of the pictures handed back."""
        return 2 if self.bit_depth > 8 or self.widen else 1

    @property
    def output_size(self) -> tuple[int, int]:
        return (self.out_w or self.crop_w, self.out_h or self.crop_h)

    @property
    def scaled(self) -> bool:
        return self.output_size != (self.crop_w, self.crop_h)

    @property
    def frame_bytes(self) -> int:
        width, height = self.output_size
        luma = width * height * self.bytes_per_sample
        if self.luma_only:
            return luma
        return luma + 2 * ((width + 1) // 2) * ((height + 1) // 2) * self.bytes_per_sample


def plan_decode(info: VideoInfo, crop: CropBox | None, *, shift: int = 0, luma_only: bool = False,
                size: tuple[int, int] | None = None, algorithm: str = "bicubic", widen: int = 0) -> DecodePlan:
    """The plan for decoding `info` here, cropped to `crop` and scaled to
    `size` with `algorithm`, or NvdecUnavailableError with the reason it is
    decoded by FFmpeg. Scaled here, a picture is not FFmpeg's scale filter's
    to the sample, but the user decided (2026-10-03) that a comparison scaled
    any way is the same comparison (ComparisonRecipe.identity_dict)."""
    codec = (info.codec_name or "").casefold()
    if codec not in _CODECS:
        raise NvdecUnavailableError(f"{info.codec_name or 'this codec'} is decoded by FFmpeg")
    depth = _DEPTHS.get((info.pix_fmt or "").casefold())
    if depth is None:
        raise NvdecUnavailableError(f"{info.pix_fmt or 'this pixel format'} is decoded by FFmpeg")
    if info.width <= 0 or info.height <= 0:
        raise NvdecUnavailableError("the video's size is unknown")
    if crop is None or crop.is_noop(info.width, info.height):
        x, y, w, h = 0, 0, info.width, info.height
    else:
        # FFmpeg's crop filter keeps a 4:2:0 crop on whole chroma samples:
        # its left and top edges move to the even sample at or before them,
        # and an odd width or height loses its last column or row.
        x, y, w, h = crop.x & ~1, crop.y & ~1, crop.w & ~1, crop.h & ~1
    if w <= 0 or h <= 0 or x + w > info.width or y + h > info.height:
        raise NvdecUnavailableError("the crop is outside the picture")
    out_w, out_h = size if size is not None else (w, h)
    if out_w <= 0 or out_h <= 0:
        raise NvdecUnavailableError("the size it is scaled to is empty")
    if widen and depth != 8:
        raise NvdecUnavailableError("only 8-bit video is widened")
    return DecodePlan(codec, depth, info.width, info.height, x, y, w, h,
                      shift if depth > 8 else 0, luma_only, out_w, out_h,
                      algorithm if algorithm in _SCALERS else "bicubic", widen)


def available(backend: str = "nvidia") -> bool:
    """Whether the decoder library is bundled and loads (the GPU is not asked)."""
    try:
        _load(backend)
    except NvdecUnavailableError:
        return False
    return True


def decoder_supports(device: int, plan: DecodePlan, backend: str = "nvidia") -> tuple[bool, str]:
    """Whether GPU `device`'s decoder takes the plan's codec, depth and size."""
    try:
        lib = _load(backend)
    except NvdecUnavailableError as error:
        return False, str(error)
    error = ctypes.create_string_buffer(512)
    ok = lib.nvf_supports(device, _CODECS[plan.codec][0], plan.bit_depth, plan.width, plan.height, error, len(error))
    return bool(ok), error.value.decode(errors="replace")


# ------------------------------------------------------------- the packets

def _ffmpeg_time(text: str) -> int:
    """`text`, an FFmpeg duration in seconds ("30.042"), in microseconds, as
    av_parse_time reads it: exactly, to the sixth decimal."""
    return int(Decimal(text).scaleb(6).to_integral_value(rounding="ROUND_DOWN"))


def rescale(value: int, source: Fraction, target: Fraction) -> int:
    """av_rescale_q: `value` in time base `source` in time base `target`,
    rounded to nearest, halves away from zero."""
    b = source.numerator * target.denominator
    c = target.numerator * source.denominator
    product = value * b
    if product >= 0:
        return (product + c // 2) // c
    return -((-product + c // 2) // c)


def duration_in(text: str, time_base: Fraction) -> int:
    """FFmpeg's output -t `text` in `time_base`, as its trim filter holds it:
    a picture is kept while its time from the first is below this."""
    return rescale(_ffmpeg_time(text), Fraction(1, 1_000_000), time_base)


_TB_RE = re.compile(r"#tb 0: (\d+)/(\d+)")


class _PacketReader:
    """FFmpeg copying the video's first video stream out of its container:
    the packets on its standard output (-f data), and each packet's
    timestamp and size in a framecrc listing written to a named pipe served
    here. A thread reads the listing as it comes, so neither of FFmpeg's
    outputs waits for the other."""

    def __init__(self, path: Path, codec: str, process_handle=None) -> None:
        bsf = _CODECS[codec][1]
        filters = ["-bsf:v", bsf] if bsf else []
        self.pipe_path = rf"\\.\pipe\vml-nvdec-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._pipe = _winapi.CreateNamedPipe(
            self.pipe_path, _winapi.PIPE_ACCESS_INBOUND, _winapi.PIPE_WAIT,
            1, 1024 * 1024, 1024 * 1024, 0, _winapi.NULL)
        # -copyinkf: every packet, as a decoder would be given them; the
        # first must be a keyframe (packets).
        self.command = [
            # -y: the listing's pipe exists already (this process serves it).
            ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
            "-i", str(Path(path).resolve()),
            "-map", f"0:{VIDEO_STREAM}", "-c:v", "copy", "-copyinkf", *filters, "-f", "data", "pipe:1",
            "-map", f"0:{VIDEO_STREAM}", "-c:v", "copy", "-copyinkf", *filters, "-flush_packets", "1", "-f", "framecrc",
            self.pipe_path,
        ]
        self._process_handle = process_handle
        self._listing: queue.Queue[tuple[int, int, int] | BaseException | None] = queue.Queue()
        self.time_base: Fraction | None = None
        self._time_base_known = threading.Event()
        self.process: subprocess.Popen | None = None
        self._stdout = None
        self._stderr: list[str] = []
        #: A warning of FFmpeg's about timestamps (_TIMESTAMP_WARNING).
        self.timestamp_warning: str | None = None
        self._threads: list[threading.Thread] = []
        self._closed = False

    def start(self) -> None:
        read_handle, write_handle = _winapi.CreatePipe(None, _ES_PIPE_BYTES)
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
        try:
            self.process = proc_util.popen(self.command, stdin=subprocess.DEVNULL, stdout=write_fd,
                                           stderr=subprocess.PIPE)
        except BaseException:
            os.close(write_fd)
            _winapi.CloseHandle(read_handle)
            raise
        finally:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        self._stdout = open(msvcrt.open_osfhandle(read_handle, os.O_RDONLY), "rb", buffering=0)  # noqa: SIM115
        if self._process_handle is not None:
            self._process_handle.attach(self.process.pid)
        for target, name in ((self._read_listing, "listing"), (self._drain_stderr, "stderr")):
            thread = threading.Thread(target=target, name=f"nvdec-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _read_listing(self) -> None:
        try:
            try:
                _winapi.ConnectNamedPipe(self._pipe, _winapi.NULL)
            except OSError as error:
                if error.winerror != 535:  # ERROR_PIPE_CONNECTED: FFmpeg was first
                    raise
            fd = msvcrt.open_osfhandle(self._pipe, os.O_RDONLY)
            self._pipe = None
            last_dts = None
            with open(fd, "rb") as listing:
                for raw in listing:
                    line = raw.decode("ascii", errors="replace").strip()
                    if line.startswith("#"):
                        match = _TB_RE.match(line)
                        if match:
                            self.time_base = Fraction(int(match.group(1)), int(match.group(2)))
                            self._time_base_known.set()
                        continue
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) < 6 or not fields[0].isdigit():
                        continue
                    printed = next((f for f in fields[6:] if f.startswith("F=")), None)
                    # framecrc prints a packet's flags only when they are not
                    # just "keyframe".
                    flags = _KEY if printed is None else int(printed[2:], 16)
                    # A decode time that does not go up is one FFmpeg's muxer
                    # rewrote (to the previous one, and the presentation time
                    # with it): known here with the packet, where its warning
                    # can come after a run that stopped early has used it.
                    dts = int(fields[1])
                    if dts != _NOPTS:
                        if last_dts is not None and dts <= last_dts:
                            raise NvdecFailedError("the video's decode timestamps do not go up")
                        last_dts = dts
                    self._listing.put((int(fields[2]), int(fields[4]), flags))
        except BaseException as error:  # handed to the reader of packets
            self._listing.put(error)
        finally:
            if self._pipe is not None:
                with contextlib.suppress(OSError):
                    _winapi.CloseHandle(self._pipe)
                self._pipe = None
            self._time_base_known.set()
            self._listing.put(None)

    def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        with contextlib.suppress(OSError, ValueError):
            for raw in self.process.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if len(self._stderr) < 50:
                    self._stderr.append(line)
                if self.timestamp_warning is None and _TIMESTAMP_WARNING.search(line):
                    self.timestamp_warning = line

    def _next_listing(self):
        """The next (pts, size, flags), or None at the end of the listing. An
        FFmpeg that ended without opening the listing's pipe is let through
        it, so the wait ends."""
        while True:
            try:
                return self._listing.get(timeout=0.2)
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    with contextlib.suppress(OSError), open(self.pipe_path, "wb"):
                        pass

    def packets(self):
        """(packet, pts, flags) in decode order, then FFmpeg's exit is checked."""
        first = True
        while True:
            item = self._next_listing()
            if item is None:
                break
            if isinstance(item, NvdecFailedError):
                raise item
            if isinstance(item, BaseException):
                raise NvdecFailedError(f"reading FFmpeg's packet list failed: {item}") from item
            pts, size, flags = item
            if pts == _NOPTS:
                raise NvdecFailedError("a packet has no timestamp")
            if first and not flags & _KEY:
                raise NvdecFailedError("the video does not start with a keyframe")
            if self.timestamp_warning is not None:
                raise NvdecFailedError(f"FFmpeg rewrote the video's timestamps: {self.timestamp_warning}")
            first = False
            data = bytearray(size)
            view = memoryview(data)
            filled = 0
            while filled < size:
                count = self._stdout.readinto(view[filled:])
                if not count:
                    raise NvdecFailedError("FFmpeg's packets ended before its packet list")
                filled += count
            yield data, pts, flags
        code = self.process.wait() if self.process is not None else 0
        for thread in self._threads:
            thread.join(timeout=10)  # the warnings, all of them
        if self._closed:
            return
        if self.timestamp_warning is not None:
            raise NvdecFailedError(f"FFmpeg rewrote the video's timestamps: {self.timestamp_warning}")
        if code != 0 or first:
            # No packet at all is a failure too: FFmpeg can exit with 0 when
            # it refused to open an output.
            raise NvdecFailedError(f"FFmpeg failed copying the video's packets (exit code {code})"
                                   + (f": {self.stderr_text()}" if self.stderr_text() else ""))

    def wait_time_base(self, timeout: float) -> Fraction | None:
        self._time_base_known.wait(timeout)
        return self.time_base

    def stderr_text(self) -> str:
        return "\n".join(self._stderr).strip()[-500:]

    def close(self) -> None:
        self._closed = True
        if self.process is not None:
            if self.process.poll() is None:
                with contextlib.suppress(OSError):
                    proc_util.terminate(self.process)
            with contextlib.suppress(Exception):
                self.process.wait(timeout=10)
            if self._process_handle is not None:
                self._process_handle.detach(self.process.pid)
        with contextlib.suppress(OSError), open(self.pipe_path, "wb"):
            pass  # a listing thread still waiting for FFmpeg to connect
        for thread in self._threads:
            thread.join(timeout=10)
        if self._stdout is not None:
            with contextlib.suppress(OSError):
                self._stdout.close()
        if self._pipe is not None:
            with contextlib.suppress(OSError):
                _winapi.CloseHandle(self._pipe)
            self._pipe = None


def _av1_sequence_header(path: Path) -> bytes:
    """The AV1 sequence header OBUs of `path`'s stream configuration (its
    av1C box minus the box's own 4-byte header), as FFmpeg's NVDEC decoder
    gives them to NVIDIA's parser. Empty when there is none."""
    command = [ffprobe_path(), "-v", "error", "-select_streams", VIDEO_STREAM, "-show_entries", "stream=extradata",
               "-show_data", "-of", "default=noprint_wrappers=1", str(Path(path).resolve())]
    try:
        output = proc_util.run(command, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return b""
    data = bytearray()
    for line in output.splitlines():
        match = re.match(r"^[0-9a-f]{8}: ((?:[0-9a-f]{2,4} ?)+)", line.strip())
        if match:
            data += bytes.fromhex(match.group(1).replace(" ", ""))
    if len(data) > 4 and data[0] & 0x80:
        return bytes(data[4:])
    return b""


# ----------------------------------------------------------------- streams

class NvdecStream:
    """One video decoded on GPU `device`, its pictures in a pool of
    `pool` slots in GPU memory, in display order. A thread feeds FFmpeg's
    packets to the decoder; next() hands out the pictures, each slot given
    back with release() once its picture has been copied on."""

    def __init__(self, info: VideoInfo, plan: DecodePlan, device: int = 0, *, pool: int = 4,
                 process_handle=None, backend: str = "nvidia") -> None:
        self.info = info
        self.plan = plan
        self.backend = backend
        self._lib = _load(backend)
        self._handle = None
        self._reader = _PacketReader(info.path, plan.codec, process_handle)
        self._extradata = _av1_sequence_header(info.path) if plan.codec == "av1" else b""
        self._extradata_buffer = ctypes.create_string_buffer(self._extradata) if self._extradata else None
        params = _Params(device, _CODECS[plan.codec][0], plan.bit_depth, plan.width, plan.height,
                         plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h, plan.shift, int(plan.luma_only),
                         pool, ctypes.cast(self._extradata_buffer, ctypes.c_void_p) if self._extradata_buffer else None,
                         len(self._extradata), plan.output_size[0], plan.output_size[1], _SCALERS[plan.scaler],
                         plan.widen)
        error = ctypes.create_string_buffer(1024)
        handle = self._lib.nvf_open(ctypes.byref(params), error, len(error))
        if not handle:
            self._reader.close()  # its pipe
            raise NvdecUnavailableError(error.value.decode(errors="replace") or "the GPU's decoder could not start")
        self._handle = handle
        self.frame_bytes = plan.frame_bytes
        self._feed_error: BaseException | None = None
        #: The timestamps of the packets fed whose pictures have not come out
        #: yet, smallest first (a heap), and how many were fed.
        self._waiting: list[int] = []
        self._fed_count = 0
        self._fed_lock = threading.Lock()
        #: Timestamps of packets the demuxer marked discard -- the frames an
        #: MP4 edit list cuts off: decoded (later frames may refer to them),
        #: never handed out, as FFmpeg's decode drops them.
        self._discard: set[int] = set()
        self._shown_count = 0
        self._last_shown: int | None = None
        self._finished = False
        self._closing = False
        self._feeder = threading.Thread(target=self._feed, name="nvdec-feed", daemon=True)

    def wait_time_base(self, timeout: float = 60.0) -> Fraction:
        """The stream's time base, once FFmpeg has reported it (before its
        first packet)."""
        self._reader.wait_time_base(timeout)
        return self.time_base

    @property
    def time_base(self) -> Fraction:
        time_base = self._reader.time_base
        if time_base is None:
            raise NvdecFailedError("FFmpeg did not report the stream's time base")
        return time_base

    def start(self) -> None:
        self._reader.start()
        self._feeder.start()

    def _feed(self) -> None:
        lib, handle = self._lib, self._handle
        try:
            for data, pts, flags in self._reader.packets():
                with self._fed_lock:
                    heapq.heappush(self._waiting, pts)
                    self._fed_count += 1
                    if flags & _DISCARD:
                        self._discard.add(pts)
                buffer = (ctypes.c_ubyte * len(data)).from_buffer(data)
                code = lib.nvf_push(handle, buffer, len(data), pts)
                if code != 0:
                    return  # the decoder's error, or a close: next() reports it
            lib.nvf_finish(handle)
        except BaseException as error:
            if not self._closing:
                self._feed_error = error
            lib.nvf_abort(handle)

    def _error(self) -> str:
        text = ctypes.create_string_buffer(1024)
        self._lib.nvf_error(self._handle, text, len(text))
        return text.value.decode(errors="replace")

    def next(self, timeout_ms: int = 100) -> tuple[int, int] | None:
        """The next picture's (slot, pts), None at the end of the video, or
        TimeoutError after `timeout_ms` (so the caller can look at Cancel).
        NvdecFailedError when decoding failed, or the decoder's pictures are
        not the packets' (see verify)."""
        slot, pts = ctypes.c_int(), ctypes.c_longlong()
        while True:
            code = self._lib.nvf_pop(self._handle, timeout_ms, ctypes.byref(slot), ctypes.byref(pts))
            if code != _NVF_FRAME:
                break
            try:
                discarded = self._take(pts.value)
            except NvdecFailedError:
                self.release(slot.value)
                raise
            if not discarded:
                return slot.value, pts.value
            self.release(slot.value)
        if code == _NVF_TIMEOUT:
            raise TimeoutError
        if code == _NVF_END:
            self._finished = True
            self.verify()
            return None
        if self._feed_error is not None:
            raise NvdecFailedError(str(self._feed_error)) from self._feed_error
        if code == _NVF_ABORTED:
            raise NvdecFailedError("decoding was stopped")
        raise NvdecFailedError(self._error() or "the GPU's decoder failed")

    def _take(self, pts: int) -> bool:
        """Checks the picture stamped `pts`, as it comes out, against the
        packets fed: in timestamp order, and the next packet's -- every
        packet stamped earlier has had its picture already. A picture the
        decoder dropped or made up failed the run only at its end, a whole
        pass later. Whether the picture is one the demuxer discards."""
        if self._last_shown is not None and pts <= self._last_shown:
            raise NvdecFailedError("the GPU's decoder gave pictures out of order")
        with self._fed_lock:
            # A packet stamped earlier is in the heap only if it was fed: its
            # picture is owed before this one. One fed after this picture
            # came out would come out of order, which fails above.
            if self._waiting and self._waiting[0] < pts:
                raise NvdecFailedError(
                    f"the GPU's decoder gave {self._shown_count} pictures for {self._shown_count + 1} packets")
            if not self._waiting or self._waiting[0] != pts:
                raise NvdecFailedError("the GPU's decoder gave pictures the packets do not have")
            heapq.heappop(self._waiting)
            discarded = pts in self._discard
        self._shown_count += 1
        self._last_shown = pts
        return discarded

    def verify(self) -> None:
        """NvdecFailedError unless the pictures handed out are the packets',
        one each, in timestamp order -- what FFmpeg's decode of the stream
        gives. Each picture is checked as it comes out (_take); after the
        end, no packet may be left without its picture."""
        if not self._finished:
            return
        with self._fed_lock:
            fed = self._fed_count
        if self._shown_count != fed:
            raise NvdecFailedError(f"the GPU's decoder gave {self._shown_count} pictures for {fed} packets")

    def release(self, slot: int) -> None:
        self._lib.nvf_release(self._handle, slot)

    def download(self, slot: int, address: int) -> None:
        """Copies the slot's picture, planes packed, to page-locked memory at
        `address` (frame_bytes long)."""
        if self._lib.nvf_download(self._handle, slot, address) != 0:
            raise NvdecFailedError(self._error() or "the GPU's decoder failed")

    def copy_luma(self, slot: int, address: int, pitch: int) -> None:
        """Copies the slot's luma plane to GPU memory at `address`, rows `pitch` bytes apart."""
        if self._lib.nvf_copy_luma(self._handle, slot, address, pitch) != 0:
            raise NvdecFailedError(self._error() or "the GPU's decoder failed")

    def abort(self) -> None:
        """Ends every wait in next() and in the feeding thread; from any thread."""
        if self._handle is not None:
            self._closing = True
            self._lib.nvf_abort(self._handle)

    def stats(self) -> _Info:
        info = _Info()
        self._lib.nvf_info(self._handle, ctypes.byref(info))
        return info

    def close(self) -> None:
        """Stops decoding and frees everything; safe to call twice."""
        if self._handle is None:
            return
        self._closing = True
        self._lib.nvf_abort(self._handle)
        self._reader.close()
        if self._feeder.ident is not None:
            self._feeder.join(timeout=30)
        if not self._feeder.is_alive():
            self._lib.nvf_close(self._handle)
        else:  # never seen; leaking the decoder is safer than freeing it under the thread
            _log.error("The GPU decoder's feeding thread did not stop; its decoder is left open")
        self._handle = None
