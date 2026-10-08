"""gpu_frames: decoding on the GPU in the scoring process for the GPU
metrics. The plan and arithmetic are checked everywhere; the decoded
pictures against FFmpeg's decode with each GPU maker's decoder the PC has
(NVIDIA's, Intel's, AMD's), and with the software decoder where it is
built (its library is not in git)."""
from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import gpu_frames as nv
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.models import CropBox, VideoInfo


def _info(**overrides) -> VideoInfo:
    fields = {"path": Path("x.mkv"), "width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0,
              "nb_frames": 240, "codec_name": "hevc", "pix_fmt": "yuv420p10le"}
    fields.update(overrides)
    return VideoInfo(**fields)


# ------------------------------------------------------------------ the plan


def test_a_crop_is_rounded_as_ffmpegs_crop_filter_rounds_it():
    """vf_crop on 4:2:0: left and top to the even sample at or before them,
    width and height down to even (checked against FFmpeg 9: crop=1917:1077:3:1
    gives 1916x1076)."""
    plan = nv.plan_decode(_info(), CropBox(1917, 1077, 3, 1))
    assert (plan.crop_x, plan.crop_y, plan.crop_w, plan.crop_h) == (2, 0, 1916, 1076)


def test_what_the_gpu_decoders_do_not_decode_is_left_to_ffmpeg(subtests):
    def check(field, value):
        with pytest.raises(nv.GpuDecodeUnavailableError):
            nv.plan_decode(_info(**{field: value}), None)

    for field, value in [("codec_name", "mpeg4"), ("codec_name", "prores"),
                                                  ("pix_fmt", "yuv422p10le"), ("pix_fmt", "yuv420p12le"),
                                                  ("pix_fmt", "yuv444p"), ("width", 0),
                                                  # An odd size: refused here, not once the pass has started.
                                                  ("width", 1919), ("height", 1079)]:
        with subtests.test(field=field, value=value):
            check(field, value)


# ------------------------------------------------------------ the arithmetic

def test_rescale_rounds_as_av_rescale_q(subtests):
    def check(value, source, target, expected):
        assert nv.rescale(value, source, target) == expected

    for value, source, target, expected in [
        (1, Fraction(1, 1000), Fraction(1, 90000), 90),
        (41, Fraction(1, 1000), Fraction(1001, 24000), 1),        # 0.98 -> 1
        (1001, Fraction(1, 24000), Fraction(1, 1000), 42),        # 41.708 -> 42
        (3, Fraction(1, 2), Fraction(1, 1), 2),                   # 1.5: halves away from zero
        (-3, Fraction(1, 2), Fraction(1, 1), -2),
        (5, Fraction(1, 2), Fraction(1, 1), 3),
    ]:
        with subtests.test(value=value, source=source, target=target, expected=expected):
            check(value, source, target, expected)


# ----------------------------------------- decoding, on the PC's GPUs

#: Each GPU maker's decoder: its tests run where the PC has one.
BACKENDS = ("nvidia", "intel", "amd", "software")


def _gpu_decodes(plan: nv.DecodePlan, backend: str = "nvidia") -> bool:
    if not nv.LIBRARIES[backend].is_file():
        return False
    return nv.decoder_supports(0, plan, backend)[0]


def _need(plan: nv.DecodePlan, backend: str) -> None:
    if not _gpu_decodes(plan, backend):
        pytest.skip(f"no {backend} decoder for this on this PC")


def _clip(path: Path, codec: str, pix_fmt: str, extra: list[str] | None = None, seconds: float = 2.0) -> Path:
    encoder = {"h264": ["-c:v", "libx264", "-preset", "veryfast", "-bf", "3"],
               "hevc": ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error"]}[codec]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s=640x360:r=24000/1001:d={seconds}", "-pix_fmt", pix_fmt, *encoder,
                    *(extra or []), str(path)], check=True)
    return path


def _decode(info: VideoInfo, plan: nv.DecodePlan, backend: str = "nvidia") -> tuple[list[str], list[int]]:
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    out = np.empty(plan.frame_bytes, dtype=np.uint8)
    sums, stamps = [], []
    try:
        stream.start()
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            sums.append(hashlib.md5(out).hexdigest())
            stamps.append(item[1])
    finally:
        stream.close()
    return sums, stamps


def _ffmpeg_decode(path: Path, plan: nv.DecodePlan) -> list[str]:
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    out = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split(",")[5].strip() for line in out.splitlines() if line and not line.startswith("#")]


def test_pictures_are_ffmpegs_decode(tmp_path_factory, subtests):
    def check(codec, pix_fmt, crop, backend, tmp_path):
        path = _clip(tmp_path / "clip.mkv", codec, pix_fmt)
        info = probe_video(path)
        plan = nv.plan_decode(info, crop, shift=6)
        _need(plan, backend)
        sums, stamps = _decode(info, plan, backend)
        assert len(sums) == round(2.0 * 24000 / 1001)
        assert sums == _ffmpeg_decode(path, plan)
        assert stamps == sorted(stamps)

    for codec, pix_fmt, crop in [
        ("h264", "yuv420p", None),
        ("h264", "yuv420p", CropBox(600, 300, 20, 30)),
        ("hevc", "yuv420p10le", CropBox(638, 358, 2, 2)),
    ]:
        for backend in BACKENDS:
            with subtests.test(codec=codec, pix_fmt=pix_fmt, crop=crop, backend=backend):
                check(codec, pix_fmt, crop, backend, tmp_path_factory.mktemp("case"))


def test_ten_bit_kept_in_the_top_bits_is_the_shifted_picture_times_64(tmp_path_factory, subtests):
    def check(backend, tmp_path):
        path = _clip(tmp_path / "clip.mkv", "hevc", "yuv420p10le", seconds=0.5)
        info = probe_video(path)
        shifted, kept = nv.plan_decode(info, None, shift=6), nv.plan_decode(info, None, shift=0)
        _need(shifted, backend)
        pictures = []
        for plan in (shifted, kept):
            stream = nv.GpuFrameStream(info, plan, backend=backend)
            out = np.empty(plan.frame_bytes // 2, dtype=np.uint16)
            try:
                stream.start()
                item = None
                while item is None:
                    try:
                        item = stream.next(1000)
                    except TimeoutError:
                        continue
                stream.download(item[0], out.ctypes.data)
                stream.release(item[0])
            finally:
                stream.close()
            pictures.append(out)
        assert np.array_equal(pictures[1], pictures[0] << 6)

    for backend in BACKENDS:
        with subtests.test(backend=backend):
            check(backend, tmp_path_factory.mktemp("case"))


def test_frames_an_mp4_edit_list_cuts_off_are_not_handed_out(tmp_path_factory, subtests):
    """A copy cut out of an MP4 starts at a keyframe before the cut, and its
    edit list marks the frames before the cut discard: FFmpeg decodes them
    (the frames after refer to them) and drops them."""
    def check(backend, tmp_path):
        whole = _clip(tmp_path / "whole.mp4", "h264", "yuv420p", ["-g", "48"], seconds=4.0)
        cut = tmp_path / "cut.mp4"
        subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-ss", "1.3", "-i", str(whole), "-c", "copy",
                        str(cut)], check=True)
        info = probe_video(cut)
        plan = nv.plan_decode(info, None)
        _need(plan, backend)
        sums, _stamps = _decode(info, plan, backend)
        expected = _ffmpeg_decode(cut, plan)
        assert sums == expected

    for backend in BACKENDS:
        with subtests.test(backend=backend):
            check(backend, tmp_path_factory.mktemp("case"))


def _ffmpeg_frames(path: Path, chain: str, depth: int) -> np.ndarray:
    raw = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", chain,
                          "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint16 if depth > 8 else np.uint8)


def _frames(info: VideoInfo, plan: nv.DecodePlan, backend: str) -> np.ndarray:
    stream = nv.GpuFrameStream(info, plan, backend=backend)
    dtype = np.uint16 if plan.bytes_per_sample == 2 else np.uint8
    pictures = []
    try:
        stream.start()
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            out = np.empty(plan.frame_bytes // plan.bytes_per_sample, dtype=dtype)
            stream.download(item[0], out.ctypes.data)
            stream.release(item[0])
            pictures.append(out)
    finally:
        stream.close()
    return np.concatenate(pictures)


def test_scaled_pictures_are_ffmpegs_but_for_rounding(tmp_path_factory, subtests):
    """Not FFmpeg's scale filter's to the sample (a comparison scaled any way
    is the same comparison), but the same filter: on this synthetic picture's
    hard edges and odd sizes, where they differ most -- the CPU scaler of
    Intel's and AMD's decoders filters down the columns first where that is
    faster, so its cap falls after the other pass -- all but 1 in 200
    samples within 1, none more than 6 (on film, every sample within 1). A
    wrong plane, siting or filter is tens to hundreds off."""
    def check(pix_fmt, size, algorithm, backend, tmp_path):
        codec = "hevc" if pix_fmt == "yuv420p10le" else "h264"
        path = _clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=0.5)
        info = probe_video(path)
        plan = nv.plan_decode(info, None, shift=6, size=size, algorithm=algorithm)
        _need(plan, backend)
        ours = _frames(info, plan, backend).astype(np.int32)
        depth = 10 if pix_fmt == "yuv420p10le" else 8
        fmt = "yuv420p10le" if depth > 8 else "yuv420p"
        want = _ffmpeg_frames(path, f"scale={size[0]}:{size[1]}:flags={algorithm},format={fmt}", depth).astype(np.int32)
        assert ours.shape == want.shape
        difference = np.abs(ours - want)
        assert difference.max() <= 6
        assert np.mean(difference > 1) < 0.005

    for pix_fmt, size, algorithm in [
        ("yuv420p10le", (320, 180), "bicubic"),
        ("yuv420p", (1280, 720), "lanczos"),
        ("yuv420p", (426, 240), "bilinear"),
        ("yuv420p10le", (960, 540), "spline"),
    ]:
        for backend in BACKENDS:
            with subtests.test(pix_fmt=pix_fmt, size=size, algorithm=algorithm, backend=backend):
                check(pix_fmt, size, algorithm, backend, tmp_path_factory.mktemp("case"))


def test_widened_eight_bit_is_ffmpegs_conversion_to_ten(tmp_path_factory, subtests):
    """Widening is exact: v << 2, as FFmpeg converts limited-range 8-bit."""
    def check(backend, tmp_path):
        path = _clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5)
        info = probe_video(path)
        plan = nv.plan_decode(info, None, widen=nv.WIDEN_SHIFT)
        _need(plan, backend)
        ours = _frames(info, plan, backend)
        assert np.array_equal(ours, _ffmpeg_frames(path, "format=yuv420p10le", 10))

    for backend in BACKENDS:
        with subtests.test(backend=backend):
            check(backend, tmp_path_factory.mktemp("case"))


def test_widened_full_range_luma_has_its_top_bits_repeated(tmp_path_factory, subtests):
    """255 becomes 1023, as FFmpeg widens full-range video; the chroma is
    shifted as ever."""
    def check(backend, tmp_path):
        path = _clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5)
        info = probe_video(path)
        plan = nv.plan_decode(info, None, widen=nv.WIDEN_REPEAT)
        _need(plan, backend)
        ours = _frames(info, plan, backend).reshape(-1, 640 * 360 * 3 // 2)
        narrow = _ffmpeg_frames(path, "format=yuv420p", 8).astype(np.uint16).reshape(ours.shape)
        luma = 640 * 360
        assert np.array_equal(ours[:, :luma], (narrow[:, :luma] << 2) | (narrow[:, :luma] >> 6))
        assert np.array_equal(ours[:, luma:], narrow[:, luma:] << 2)

    for backend in BACKENDS:
        with subtests.test(backend=backend):
            check(backend, tmp_path_factory.mktemp("case"))


# ------------------------------------------------- scaling on the GPU (Intel, AMD)

#: PCI vendor ids of the GPUs the scaling shader is checked on; the last is
#: Windows' software device, which every PC has.
_SCALE_GPUS = {"nvidia": 0x10DE, "intel": 0x8086, "amd": 0x1002, "software": 0x1414}


def _test_picture(width: int, height: int, depth: int) -> bytes:
    """NV12/P010 noise with hard edges, where a filter rings: a wrong weight,
    order of summing, cap or rounding shows."""
    rng = np.random.default_rng(7)
    top = (1 << depth) - 1
    luma = rng.integers(0, top + 1, (height, width))
    luma[: height // 3, : width // 2] = top
    luma[height // 3: height // 2, width // 3:] = 0
    chroma = rng.integers(0, top + 1, (height // 2, width))
    chroma[: height // 5] = top
    planes = np.concatenate([luma, chroma])
    return (planes << 6).astype("<u2").tobytes() if depth > 8 else planes.astype(np.uint8).tobytes()


def test_the_scaling_shader_gives_the_cpu_scalers_picture(subtests):
    """Sample for sample: the same weights, summed in the same order."""
    def check(gpu, depth, size, to, algorithm, shift, luma_only, crop):
        if not nv.available("intel"):
            pytest.skip("the decoders are not built")
        x, y, w, h = crop or (0, 0, *size)
        plan = nv.DecodePlan("hevc" if depth > 8 else "h264", depth, size[0], size[1], x, y, w, h, shift, luma_only,
                             to[0], to[1], algorithm)
        picture = _test_picture(size[0], size[1], depth)
        on_gpu = nv.scale_picture(plan, picture, _SCALE_GPUS[gpu])
        if on_gpu is None:
            pytest.skip(f"no {gpu} GPU with Direct3D 11 here")
        assert on_gpu == nv.scale_picture(plan, picture, None)

    for gpu in list(_SCALE_GPUS):
        for depth, size, to, algorithm, shift, luma_only, crop in [
            (10, (1280, 720), (640, 360), "bicubic", 6, False, None),
            (10, (1280, 720), (640, 360), "lanczos", 0, False, None),  # the samples kept in the top bits
            (8, (640, 360), (1280, 720), "bicubic", 0, False, None),  # up: along the rows first
            (8, (1280, 720), (427, 241), "spline", 0, False, None),  # odd output: its chroma is rounded up
            (10, (1280, 720), (854, 480), "bilinear", 6, True, None),
            (10, (1280, 720), (640, 300), "bicubic", 6, False, (0, 60, 1280, 600)),  # cropped: black bars cut
            (8, (640, 360), (320, 640), "lanczos", 0, False, None),  # narrower and taller at once
        ]:
            with subtests.test(gpu=gpu, depth=depth, size=size, to=to, algorithm=algorithm, shift=shift, luma_only=luma_only, crop=crop):
                check(gpu, depth, size, to, algorithm, shift, luma_only, crop)


def test_decoded_pictures_are_scaled_on_the_gpu_and_are_the_cpus(tmp_path_factory, subtests):
    def check(backend, pix_fmt, size, algorithm, tmp_path):
        from dataclasses import replace

        codec = "hevc" if pix_fmt == "yuv420p10le" else "h264"
        info = probe_video(_clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=0.5))
        plan = nv.plan_decode(info, None, shift=6, size=size, algorithm=algorithm)
        _need(plan, backend)
        stream = nv.GpuFrameStream(info, plan, backend=backend)
        try:
            stream.start()
            while stream.stats().displayed == 0:  # where the first picture was scaled
                with contextlib.suppress(TimeoutError):
                    if (item := stream.next(1000)) is None:
                        break
                    stream.release(item[0])
            assert stream.scale_note() == ""
            assert stream.stats().scaled_on_gpu == 1
        finally:
            stream.close()
        assert np.array_equal(_frames(info, plan, backend), _frames(info, replace(plan, cpu_scaling=True), backend))

    for backend in ["intel", "amd"]:
        for pix_fmt, size, algorithm in [
            ("yuv420p10le", (320, 180), "bicubic"),
            ("yuv420p", (1920, 1080), "lanczos"),
        ]:
            with subtests.test(backend=backend, pix_fmt=pix_fmt, size=size, algorithm=algorithm):
                check(backend, pix_fmt, size, algorithm, tmp_path_factory.mktemp("case"))


@pytest.mark.parametrize("backend", ["intel", "amd"])
def test_a_gpu_picture_that_is_not_the_cpus_hands_the_scaling_to_the_cpu(tmp_path, backend):
    """A driver whose shader arithmetic differed would change the scores:
    the decoders scale their first pictures on the CPU as well, and at the
    first that differs the CPU's is the one handed out, from then on."""
    from dataclasses import replace

    info = probe_video(_clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=0.5))
    plan = nv.plan_decode(info, None, size=(320, 180))
    _need(plan, backend)
    spoiled = replace(plan, cpu_scaling=2)  # the GPU's first picture is made wrong
    stream = nv.GpuFrameStream(info, spoiled, backend=backend)
    try:
        stream.start()
        while (item := _next(stream)) is not None:
            stream.release(item[0])
        assert stream.scale_note() == "the GPU's scaled picture is not the CPU's"
        assert stream.stats().scaled_on_gpu == 0
    finally:
        stream.close()
    assert np.array_equal(_frames(info, spoiled, backend), _frames(info, replace(plan, cpu_scaling=1), backend))


def _next(stream):
    while True:
        with contextlib.suppress(TimeoutError):
            return stream.next(1000)


# ------------------------------------- AMD's hand-over (native/amf_handover.h)

@pytest.fixture(params=["windows", "amf"])
def amd_decoder(request, monkeypatch):
    """AMD's hand-over through Windows' own decoder (native/mf_frames.cpp),
    which GpuFrameStream opens first, or through AMF's (native/amf_frames.cpp),
    where Windows' does not open."""
    if request.param == "amf":
        monkeypatch.setattr(nv.GpuFrameStream, "_open_media_foundation", lambda *args: None)
    else:
        opened = nv.GpuFrameStream._open_media_foundation

        def windows_only(*args):
            if (handle := opened(*args)) is None:
                pytest.skip("Windows' decoder does not decode this on this PC")
            return handle

        monkeypatch.setattr(nv.GpuFrameStream, "_open_media_foundation", windows_only)
    return request.param


def _handed_over(info: VideoInfo, plan: nv.DecodePlan, planes: bool, pin: bool) -> list[str]:
    """Each picture's MD5, from AMD's decoder handing over: downloaded whole,
    or plane by plane into rows longer than the picture's (as libvmaf's
    pictures are), into memory pinned for the GPU or not."""
    stream = nv.GpuFrameStream(info, plan, backend="amd", handover=True)
    width, height = plan.output_size
    sample = plan.bytes_per_sample
    widths, heights = (width, (width + 1) // 2, (width + 1) // 2), (height, (height + 1) // 2, (height + 1) // 2)
    pitches = tuple(w * sample + 64 for w in widths)
    count = 1 if plan.luma_only else 3
    rows = [np.zeros((h, pitch), np.uint8) for h, pitch in zip(heights[:count], pitches[:count], strict=True)]
    whole = np.empty(plan.frame_bytes, np.uint8)
    sums = []
    try:
        if not stream.handover:
            pytest.skip("AMD's decoder does not hand over on this PC")
        if pin:
            assert all(stream.pin(plane.ctypes.data, plane.nbytes) for plane in rows)
        stream.start()
        while (item := _next(stream)) is not None:
            if planes:
                addresses = [plane.ctypes.data for plane in rows] + [None] * (3 - count)
                stream.download_planes(item[0], tuple(addresses), pitches)
                picture = np.concatenate([plane[:, :w * sample].ravel() for plane, w in zip(rows, widths[:count], strict=True)])
            else:
                stream.download(item[0], whole.ctypes.data)
                picture = whole
            stream.release(item[0])
            sums.append(hashlib.md5(picture).hexdigest())
    finally:
        if pin:
            for plane in rows:
                stream.unpin(plane.ctypes.data)
        stream.close()
    return sums


def test_pictures_amd_hands_over_are_ffmpegs_decode(amd_decoder, tmp_path_factory, subtests):
    def check(codec, pix_fmt, crop, luma_only, planes, pin, tmp_path):
        path = _clip(tmp_path / "clip.mkv", codec, pix_fmt, seconds=1.0)
        info = probe_video(path)
        plan = nv.plan_decode(info, crop, shift=6, luma_only=luma_only)
        _need(plan, "amd")
        if luma_only:  # the luma of FFmpeg's pictures
            expected = [hashlib.md5(picture[:plan.frame_bytes]).hexdigest() for picture in _ffmpeg_pictures(path, plan)]
        else:
            expected = _ffmpeg_decode(path, plan)
        assert _handed_over(info, plan, planes, pin) == expected

    for codec, pix_fmt, crop, luma_only in [
        ("h264", "yuv420p", None, False),
        ("h264", "yuv420p", CropBox(600, 300, 20, 30), False),
        ("hevc", "yuv420p10le", CropBox(638, 358, 2, 2), False),
        ("hevc", "yuv420p10le", None, True),
        ("h264", "yuv420p", CropBox(600, 300, 20, 30), True),
    ]:
        for planes, pin in [(False, False), (True, False), (True, True)]:
            with subtests.test(codec=codec, pix_fmt=pix_fmt, crop=crop, luma_only=luma_only, planes=planes, pin=pin):
                check(codec, pix_fmt, crop, luma_only, planes, pin, tmp_path_factory.mktemp("case"))


def _ffmpeg_pictures(path: Path, plan: nv.DecodePlan) -> list[bytes]:
    fmt = "yuv420p10le" if plan.bit_depth > 8 else "yuv420p"
    raw = subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"crop={plan.crop_w}:{plan.crop_h}:{plan.crop_x}:{plan.crop_y},format={fmt}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    size = (plan.crop_w * plan.crop_h + 2 * ((plan.crop_w + 1) // 2) * ((plan.crop_h + 1) // 2)) * plan.bytes_per_sample
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def test_two_amd_decoders_handing_over_at_once_give_ffmpegs_pictures(tmp_path, amd_decoder):
    """Scoring decodes both videos at once, and the decoders share the
    hand-over's Vulkan device, its queue and what is imported and pinned."""
    paths = [_clip(tmp_path / "a.mkv", "hevc", "yuv420p10le"), _clip(tmp_path / "b.mkv", "h264", "yuv420p")]
    infos = [probe_video(path) for path in paths]
    plans = [nv.plan_decode(info, None, shift=6) for info in infos]
    for plan in plans:
        _need(plan, "amd")
    expected = [_ffmpeg_decode(path, plan) for path, plan in zip(paths, plans, strict=True)]
    for _round in range(3):
        results: list[list[str] | BaseException] = [[], []]

        def decode(index: int, results: list = results) -> None:
            try:
                results[index] = _handed_over(infos[index], plans[index], True, True)
            except BaseException as error:  # reported below
                results[index] = error

        threads = [threading.Thread(target=decode, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for result in results:
            if isinstance(result, BaseException):
                raise result
        assert results == expected


def _stream_fed(*timestamps):
    """An GpuFrameStream's picture bookkeeping alone, with these packets fed."""
    import heapq
    import threading

    stream = nv.GpuFrameStream.__new__(nv.GpuFrameStream)
    stream.backend = "nvidia"
    stream._waiting, stream._fed_count, stream._fed_lock = [], 0, threading.Lock()
    stream._discard, stream._shown_count, stream._last_shown, stream._finished = set(), 0, None, False
    for pts in timestamps:
        heapq.heappush(stream._waiting, pts)
        stream._fed_count += 1
    return stream


def test_pictures_in_order_pass_and_the_end_counts_them():
    stream = _stream_fed(0, 3000, 1000, 2000)
    stream._discard.add(0)
    assert [stream._take(pts) for pts in (0, 1000, 2000)] == [True, False, False]
    stream._finished = True
    with pytest.raises(nv.GpuDecodeFailedError, match="3 pictures for 4 packets"):
        stream.verify()
    stream._take(3000)
    stream.verify()
    with pytest.raises(nv.GpuDecodeFailedError, match="out of order"):
        stream._take(2500)


# ----------------------------------------------------- the software decoder

#: FFmpeg's encoders for the software decoder's codecs, as FFmpeg's command line takes them.
_SOFTWARE_ENCODERS = {
    "h264": ["-c:v", "libx264", "-preset", "veryfast", "-bf", "3"],
    "hevc": ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "bframes=4:log-level=error"],
    "vvc": ["-c:v", "libvvenc", "-preset", "faster"],
    "vp9": ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8"],
    "av1": ["-c:v", "libsvtav1", "-preset", "12"],
    "mpeg2video": ["-c:v", "mpeg2video", "-bf", "2", "-g", "12"],
    "ffv1": ["-c:v", "ffv1", "-level", "3", "-slices", "4"],  # its configuration in the container, not the packets
}


def _software_clip(path: Path, codec: str, pix_fmt: str, seconds: float = 1.0) -> Path:
    encoder = _SOFTWARE_ENCODERS[codec]
    listed = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if f" {encoder[1]} " not in listed:
        pytest.skip(f"this FFmpeg has no {encoder[1]}")
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s=320x240:r=24000/1001:d={seconds}", "-pix_fmt", pix_fmt, *encoder, str(path)],
                   check=True)
    return path


def test_the_software_decoders_pictures_are_ffmpegs_decode(tmp_path_factory, subtests):
    """Every codec it is asked for, sample for sample: whichever FFmpeg
    decodes them, the decoders' pictures are the same."""
    def check(codec, pix_fmt, crop, tmp_path):
        path = _software_clip(tmp_path / f"clip.{'webm' if codec == 'vp9' else 'mkv'}", codec, pix_fmt)
        info = probe_video(path)
        plan = nv.plan_decode(info, crop, shift=6)
        _need(plan, nv.SOFTWARE)
        sums, stamps = _decode(info, plan, nv.SOFTWARE)
        assert len(sums) == round(24000 / 1001)
        assert sums == _ffmpeg_decode(path, plan)
        assert stamps == sorted(stamps)

    for codec, pix_fmt in [
        ("h264", "yuv420p10le"),  # no GPU decoder here takes it
        ("vvc", "yuv420p10le"),   # in Matroska, whose decode times FFmpeg's copy would "repair" (_CODECS)
        ("vp9", "yuv420p10le"),
        ("av1", "yuv420p10le"),   # dav1d's
        ("av1", "yuv420p"),
        ("mpeg2video", "yuv420p"),
        ("ffv1", "yuv420p10le"),
        ("ffv1", "yuv420p"),
    ]:
        for crop in [None, CropBox(318, 236, 1, 3)]:
            with subtests.test(codec=codec, pix_fmt=pix_fmt, crop=crop):
                check(codec, pix_fmt, crop, tmp_path_factory.mktemp("case"))


def test_a_video_not_at_the_depth_planned_fails_once_it_is_seen(tmp_path):
    """The decoder checks each picture against the plan: 10-bit pictures for
    an 8-bit plan are refused, not handed out as something else."""
    path = _software_clip(tmp_path / "ten.mkv", "hevc", "yuv420p10le", seconds=0.5)
    info = replace(probe_video(path), pix_fmt="yuv420p")
    plan = nv.plan_decode(info, None)
    _need(plan, nv.SOFTWARE)
    with pytest.raises(nv.GpuDecodeFailedError, match="4:2:0 at the video's depth"):
        _decode(info, plan, nv.SOFTWARE)


def test_a_stream_closed_part_way_stops_and_lets_go(tmp_path):
    """Cancel, or the shorter video ending: the feeding thread waits for a
    slot the caller never gives back, and close ends that wait."""
    path = _software_clip(tmp_path / "clip.mkv", "h264", "yuv420p", seconds=2.0)
    info = probe_video(path)
    plan = nv.plan_decode(info, None)
    _need(plan, nv.SOFTWARE)
    stream = nv.GpuFrameStream(info, plan, pool=2, backend=nv.SOFTWARE)
    try:
        stream.start()
        taken = 0
        while taken < 2:  # the pool's two slots, never given back
            try:
                stream.next(1000)
            except TimeoutError:
                continue
            taken += 1
    finally:
        stream.close()
    assert stream._handle is None and not stream._feeder.is_alive()


@pytest.mark.parametrize(("codec", "pix_fmt", "options", "added"), [
    # H.274 grain from vvenc's analysis, in SEI: FFmpeg 9 leaves VVC's out
    # (it adds AOM's AFGS1 grain to VVC: checked on real encodes).
    ("vvc", "yuv420p10le", ["-vvenc-params", "FGA=1"], False),
    ("av1", "yuv420p10le", ["-svtav1-params", "film-grain=12"], True),  # AV1's own, dav1d's
])
def test_film_grain_is_added_as_ffmpegs_decode_adds_it(tmp_path, codec, pix_fmt, options, added):
    """The pictures with the grain a stream asks for are FFmpeg 9's, which
    7.1's VVC decoder's were not (a VVC encode with AFGS1 grain: 43 of 300
    pictures the same)."""
    encoder = _SOFTWARE_ENCODERS[codec]
    listed = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if f" {encoder[1]} " not in listed:
        pytest.skip(f"this FFmpeg has no {encoder[1]}")
    path = tmp_path / "grain.mkv"
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc2=s=320x240:r=24:d=1,noise=alls=24:allf=t", "-pix_fmt", pix_fmt, *encoder,
                    *options, str(path)], check=True)
    info = probe_video(path)
    plan = nv.plan_decode(info, None, shift=6)
    _need(plan, nv.SOFTWARE)
    without_grain = subprocess.run(
        [ffmpeg_path(), "-nostdin", "-v", "error", "-export_side_data", "film_grain", "-i", str(path),
         "-map", "0:v:0", "-vf", f"format={pix_fmt}", "-fps_mode", "passthrough", "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    sums = _decode(info, plan, nv.SOFTWARE)[0]
    expected = _ffmpeg_decode(path, plan)
    assert sums == expected
    plain = [line.split(",")[5].strip() for line in without_grain.splitlines() if line and not line.startswith("#")]
    assert len(plain) == len(expected)
    differ = sum(a != b for a, b in zip(expected, plain, strict=True))
    assert differ > len(expected) // 2 if added else differ == 0
