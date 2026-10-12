"""PSNR, SSIM and XPSNR calculated in the app's own process, with the
bundled libvmaf's CPU feature extractors (libvmaf.py): the same extractors
FFmpeg's libvmaf filter runs, and libvmaf-fast's port of FFmpeg's xpsnr
filter. vmaf_runner decides when a run uses them instead of FFmpeg's filters.
"""
from __future__ import annotations

import ctypes
import functools
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from vmaf_app.core import frame_sync, gpu_frames, libvmaf
from vmaf_app.core.models import CropBox, VideoInfo

#: The libvmaf CPU feature extractors CpuScorer runs, by the app's metric key:
#: (the extractor, the feature that is the metric's score, the decimals kept).
#: They match what FFmpeg's libvmaf filter is asked for ("name=psnr",
#: "name=float_ssim") and how its log names and rounds the scores the app reads
#: (vmaf_runner._parse_log). XPSNR comes from libvmaf-fast's port of FFmpeg's
#: xpsnr filter, rounded as that filter writes its stats file ("%3.4f").
CPU_FEATURES = {"psnr": ("psnr", "psnr_y", 6), "ssim": ("float_ssim", "float_ssim", 6),
                "xpsnr": ("xpsnr", "xpsnr_y", 4)}
#: The library a CPU score records it was calculated with.
CPU_BUILD = f"libvmaf-fast {libvmaf.LIBVMAF_FAST_VERSION} (CPU)"
#: The most memory CpuScorer's pictures may take (unless two pairs alone take
#: more, at 8K). The memory is page-locked when the frames come from NVIDIA's
#: decoder. Measured on 4K 10-bit video from NVIDIA's decoder, PSNR + SSIM
#: ran at 49 fps with 2 pairs, 93 with 3, 129 with 4, 159 with 5, and 172 with
#: 6 (this limit) or 8, where the decoders' feed levels off. XPSNR's pairs take
#: longer, and it keeps a pair from the previous frames, so it gets more:
#: 135 fps with 7 pairs, 154 with 8, 169 with 9 (CPU_XPSNR_PICTURE_MEMORY) and
#: 171 with 10.
CPU_PICTURE_MEMORY = 200 << 20
CPU_XPSNR_PICTURE_MEMORY = 270 << 20


class CpuScorer:
    """Scores PSNR, SSIM and XPSNR of frame pairs given in order, with
    libvmaf-fast's own CPU extractors (CPU_FEATURES), in this process, on
    libvmaf's preallocated pictures (vmaf_preallocate_pictures).

    These are the extractors FFmpeg's libvmaf filter runs, and they give the
    same scores to the six decimals its log keeps; XPSNR is libvmaf-fast's
    port of FFmpeg's xpsnr filter. Running them here is faster than in
    FFmpeg: FFmpeg's filter allocates, zeroes and copies two new pictures
    for every pair on its single filter thread before libvmaf's threads see
    them, so 4K PSNR + SSIM ran at 67 fps there, with most cores idle.

    The pictures hold luma only. The app keeps PSNR's luma score (psnr_y)
    and SSIM, which read nothing else, and libvmaf's PSNR skips the chroma
    planes of a picture without them: the same psnr_y, with a third less to
    copy and score. `metrics` are CPU_FEATURES keys. Frames are given as
    GpuScorer takes them (4:2:0, packed as FFmpeg's rawvideo writes them;
    only the luma is read), or as a decoder's pictures (add_decoded)."""

    def __init__(self, width: int, height: int, bit_depth: int, metrics: tuple[str, ...], n_subsample: int = 1,
                 threads: int = 0, frame_rate: int = 0):
        """`frame_rate` is XPSNR's frame rate, as FFmpeg's filter takes it:
        the reference video's whole frames per second
        (vmaf_runner._xpsnr_frame_rate)."""
        self._lib = lib = libvmaf.load()
        self._step = max(1, n_subsample)
        self._metrics = tuple(metrics)
        self._context = ctypes.c_void_p()
        self._count = 0
        #: The pictures' memory page-locked for NVIDIA's decoder, mapped to
        #: the decoder stream that locked it (None where it could not).
        self._pinned: dict[int, gpu_frames.GpuFrameStream | None] = {}
        self._staging: list[bytearray] | None = None
        sample = 1 if bit_depth <= 8 else 2
        self._rows, self._row_bytes = height, width * sample
        #: A frame's luma: the size of a decoder's luma-only frame.
        self.luma_bytes = width * height * sample
        self.frame_bytes = self.luma_bytes + 2 * ((width + 1) // 2) * ((height + 1) // 2) * sample
        threads = max(0, int(threads))
        configuration = libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, threads, self._step, 0, 0)
        libvmaf.check(lib.vmaf_init(ctypes.byref(self._context), configuration), "Starting libvmaf")
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
                            raise libvmaf.LibvmafError(f"Setting {extractor}'s {key} failed")
                # libvmaf frees the options (if it refuses them early, a few bytes leak).
                libvmaf.check(lib.vmaf_use_feature(self._context, extractor.encode(), options), f"Starting {extractor}")
            pairs = cpu_pairs(width, height, bit_depth, threads, xpsnr="xpsnr" in self._metrics)
            parameters = libvmaf.PictureParameters(width, height, bit_depth, libvmaf.PIX_FMT_YUV400P)
            libvmaf.check(lib.vmaf_preallocate_pictures(self._context, libvmaf.PictureConfiguration(parameters, 2 * pairs)),
                   "Allocating pictures")
        except BaseException:
            self.close()
            raise

    def add(self, reference, distorted) -> None:
        """Scores one more pair; its frame index is the number of pairs given
        before it. The frames belong to the caller again when this returns."""
        ref, dist = self._take()
        try:
            self._fill(ref, reference, self.frame_bytes)
            self._fill(dist, distorted, self.frame_bytes)
        except BaseException:
            self._give_back(ref, dist)
            raise
        self._score(ref, dist)

    def add_decoded(self, ref_stream, ref_slot: int, test_stream, test_slot: int) -> None:
        """Scores one more pair held by gpu_frames' decoders, which decode
        luma only (luma_bytes per frame).

        NVIDIA's decoder copies the luma into libvmaf's picture itself, with
        the picture's row spacing (download_planes), into memory it
        page-locks the first time each of the pool's pictures comes round.
        Intel's and AMD's write it straight into the picture when the
        picture's rows are packed, as libvmaf lays them out at most widths
        (multiples of 32 samples: 3840, 1920, 1280); otherwise they write it
        into a buffer that is then copied in, as add() copies a frame."""
        ref, dist = self._take()
        try:
            self._download(ref, ref_stream, ref_slot, 0)
            self._download(dist, test_stream, test_slot, 1)
        except BaseException:
            self._give_back(ref, dist)
            raise
        self._score(ref, dist)

    def _take(self) -> tuple[libvmaf.AnyPicture, libvmaf.AnyPicture]:
        """Takes a pair of pictures from libvmaf's pool."""
        lib = self._lib
        ref, dist = libvmaf.Picture(), libvmaf.Picture()
        libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(ref)), "Taking a picture")
        try:
            libvmaf.check(lib.vmaf_fetch_preallocated_picture(self._context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        return ref, dist

    def _give_back(self, ref: libvmaf.AnyPicture, dist: libvmaf.AnyPicture) -> None:
        """Returns a pair of pictures to the pool unscored. Pictures taken
        from the pool must be given back: vmaf_close would wait for them."""
        self._lib.vmaf_picture_unref(ctypes.byref(ref))
        self._lib.vmaf_picture_unref(ctypes.byref(dist))

    def _score(self, ref: libvmaf.AnyPicture, dist: libvmaf.AnyPicture) -> None:
        """Hands a filled pair to libvmaf to score."""
        # libvmaf takes both pictures, even when it fails (pull request 1652).
        libvmaf.check(self._lib.vmaf_read_pictures(self._context, ctypes.byref(ref), ctypes.byref(dist), self._count),
               f"Scoring frame {self._count}")
        self._count += 1

    def _download(self, picture: libvmaf.AnyPicture, stream, slot: int, which: int) -> None:
        """Fills `picture` with the luma of a decoder's picture (see
        add_decoded); `which` is 0 for the source, 1 for the test video."""
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

    def _fill(self, picture: libvmaf.AnyPicture, frame, size: int) -> None:
        """Copies the luma of `frame` (at least `size` bytes) into `picture`."""
        source = np.frombuffer(frame, dtype=np.uint8)
        if len(source) < size:
            raise libvmaf.LibvmafError(f"frame {self._count} is shorter than a picture")
        target = np.ctypeslib.as_array(ctypes.cast(picture.data[0], ctypes.POINTER(ctypes.c_uint8)),
                                       shape=(self._rows, picture.stride[0]))
        target[:, :self._row_bytes] = source[:self.luma_bytes].reshape(self._rows, self._row_bytes)

    def finish(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Returns the frame numbers scored (every n_subsample-th frame) and
        each metric's scores for them, rounded as the files the app reads
        FFmpeg's scores from round them (CPU_FEATURES)."""
        if not self._count:
            return np.zeros(0, dtype=np.int32), {metric: np.zeros(0) for metric in self._metrics}
        libvmaf.check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "Finishing")
        frames = np.arange(0, self._count, self._step, dtype=np.int32)
        scores = {}
        value = ctypes.c_double()
        for metric in self._metrics:
            _extractor, feature, decimals = CPU_FEATURES[metric]
            column = np.empty(len(frames), dtype=np.float64)
            for slot, frame in enumerate(frames):
                libvmaf.check(self._lib.vmaf_feature_score_at_index(self._context, feature.encode(), ctypes.byref(value),
                                                             int(frame)),
                       f"Reading {metric} for frame {frame}")
                column[slot] = float(f"{value.value:.{decimals}f}")
            scores[metric] = column
        return frames, scores

    def close(self) -> None:
        """Frees the libvmaf context. When the frames came from the decoders,
        call this before closing them."""
        # The page-locked memory is unregistered first, because vmaf_close
        # frees the pictures, and memory must be unregistered from CUDA
        # before it is freed. No copy into it can be under way
        # (download_planes returns once its copy is done), and an extractor
        # still reading it is not affected.
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
    FFmpeg's filter, in builds after 3.2.0-fast.1) with the "psnr" option
    CpuScorer uses. libvmaf ignores options it does not know, so the check
    scores a pair of tiny pictures and looks for both scores."""
    try:
        return _scores_xpsnr_and_psnr()
    except (OSError, libvmaf.LibvmafError):
        return False


def _scores_xpsnr_and_psnr() -> bool:
    """cpu_scores_xpsnr's check, which may raise."""
    lib = libvmaf.load()
    context = ctypes.c_void_p()
    libvmaf.check(lib.vmaf_init(ctypes.byref(context), libvmaf.Configuration(libvmaf.LOG_LEVEL_ERROR, 0, 1, 0, 0)), "Starting libvmaf")
    try:
        options = ctypes.c_void_p()
        libvmaf.check(lib.vmaf_feature_dictionary_set(ctypes.byref(options), b"psnr", b"true"), "Setting an option")
        if lib.vmaf_use_feature(context, b"xpsnr", options):
            return False
        parameters = libvmaf.PictureParameters(64, 64, 8, libvmaf.PIX_FMT_YUV400P)
        libvmaf.check(lib.vmaf_preallocate_pictures(context, libvmaf.PictureConfiguration(parameters, 2)), "Allocating pictures")
        ref, dist = libvmaf.Picture(), libvmaf.Picture()
        libvmaf.check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(ref)), "Taking a picture")
        try:  # a picture never given back would keep vmaf_close waiting for it
            libvmaf.check(lib.vmaf_fetch_preallocated_picture(context, ctypes.byref(dist)), "Taking a picture")
        except BaseException:
            lib.vmaf_picture_unref(ctypes.byref(ref))
            raise
        # libvmaf takes both pictures, even when it fails.
        libvmaf.check(lib.vmaf_read_pictures(context, ctypes.byref(ref), ctypes.byref(dist), 0), "Scoring")
        libvmaf.check(lib.vmaf_read_pictures(context, None, None, 0), "Finishing")
        value = ctypes.c_double()
        return all(lib.vmaf_feature_score_at_index(context, name, ctypes.byref(value), 0) == 0
                   for name in (b"xpsnr_y", b"psnr_y"))
    finally:
        lib.vmaf_close(context)


def cpu_pairs(width: int, height: int, bit_depth: int, threads: int, xpsnr: bool = False) -> int:
    """How many pairs of pictures CpuScorer's pool holds.

    Each pair is scored on one thread, so each extra pair in the pool is one
    more scored at a time. The pool holds one pair per libvmaf thread plus
    one being filled, counting at most 8 threads (PSNR and SSIM of 4K pairs
    went little faster with more: 139 fps with 8 threads, 152 with 24). It
    stays within CPU_PICTURE_MEMORY (CPU_XPSNR_PICTURE_MEMORY with XPSNR),
    but holds at least two pairs. XPSNR gets one more pair, for the two test
    pictures it keeps from the previous frames."""
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
    """Scores PSNR, SSIM and XPSNR (CpuScorer; `models` are CPU_FEATURES keys)
    with both videos decoded in this process, luma only, as score_decoded
    does for VMAF.

    The videos are decoded by the GPU's own decoder (gpu_frames: NVIDIA's,
    Intel's or AMD's) or by the software decoder; `decoder` is one for both,
    or (the source's, the test video's). The frames are those FFmpeg's
    outputs would carry: the decoders crop as FFmpeg's crop filter does and
    widen 8-bit samples as FFmpeg converts them. They are paired as
    libvmaf's filter pairs them (frame_sync) and cut where FFmpeg's -t
    would cut them.

    A comparison that is scaled is refused with GpuDecodeUnavailableError
    (FFmpeg then decodes, as for any video the decoders do not take): the
    decoders do not scale exactly as FFmpeg does, and unlike VMAF, these
    scores must match FFmpeg's. Raises GpuDecodeFailedError when decoding
    fails after it has started."""
    plans = []
    source_decoder, test_decoder = gpu_frames.decoder_pair(decoder)
    for info, crop, backend in ((distorted, distorted_crop, test_decoder), (source, source_crop, source_decoder)):
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
        supported, refusal = gpu_frames.decoder_supports(gpu_frames.device_of(backend, gpu_frames.CUDA_DEVICE), plan, backend)
        if not supported:
            raise gpu_frames.GpuDecodeUnavailableError(refusal)
        plans.append(plan)
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
        scorer = CpuScorer(width, height, bit_depth, tuple(models), n_subsample, threads, frame_rate)
        if test.frame_bytes != scorer.luma_bytes or ref.frame_bytes != scorer.luma_bytes:
            raise gpu_frames.GpuDecodeUnavailableError("the decoders' frames are not the size compared at")
        test.start()
        ref.start()
        pairs = frame_sync.decoded_frame_pairs(test, ref, distorted, duration_limit, check_cancel,
                                               test.release, ref.release)
        count = 0
        started = reported = time.perf_counter()
        try:
            for test_slot, ref_slot in pairs:
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
        if scorer is not None:  # first: it releases the memory the decoders page-locked
            scorer.close()
        test.close()
        ref.close()
