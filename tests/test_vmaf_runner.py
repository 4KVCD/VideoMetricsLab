import subprocess
from pathlib import Path

import pytest

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import (
    CropBox,
    ResampleTarget,
    ScaleDirection,
    VideoInfo,
    VmafOptions,
)
from vmaf_app.core.vmaf_runner import (
    _GRAPH_SEPARATOR,
    VmafRunError,
    _bit_depth,
    _build_ffmpeg_cmd,
    _build_filtergraph,
    _build_resample_test_filtergraph,
    _fallback_ladder,
    analysis_pix_fmt,
    validate_display_geometry,
)


def _info(path, w, h, pix_fmt="yuv420p") -> VideoInfo:
    return VideoInfo(
        path=Path(path), width=w, height=h, fps=30.0, duration=10.0, nb_frames=300,
        codec_name="h264", pix_fmt=pix_fmt,
    )


def test_scales_reference_to_distorted_resolution_when_they_differ():
    source_info = _info("source.mov", 3840, 2160)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    graph = _build_filtergraph(
        source_info, distorted_info, options,
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
        log_path=Path("log.json"),
    )

    assert "scale=1920:1080" in graph
    assert "[main][ref]libvmaf=" in graph


def test_upscale_distorted_mode_scales_distorted_up_to_source_resolution():
    source_info = _info("source.mov", 3840, 2160)  # 4K source
    distorted_info = _info("distorted.mp4", 1920, 1080)  # 1080p distorted

    graph = _build_filtergraph(
        source_info, distorted_info,
        VmafOptions(model="version=vmaf_v0.6.1", scale_direction=ScaleDirection.DISTORTED_TO_SOURCE),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    main_chain, ref_chain, _ = graph.split(";")
    assert "scale=3840:2160" in main_chain  # distorted IS upscaled to the source's resolution
    assert "scale=" not in ref_chain  # source is left untouched


def test_crop_filters_applied_and_scale_targets_cropped_distorted_dims():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(model="version=vmaf_v0.6.1")

    # Source is 16:9 with 2.35:1 content letterboxed; distorted has been cropped to 2.35:1.
    source_crop = CropBox(w=1920, h=817, x=0, y=131)
    distorted_crop = CropBox(w=1920, h=817, x=0, y=0)

    graph = _build_filtergraph(
        source_info, distorted_info, options, source_crop, distorted_crop,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    assert "crop=1920:817:0:131" in graph  # source crop
    assert "crop=1920:817:0:0" in graph  # distorted crop
    # both crops already match -> no rescale needed
    assert "scale=" not in graph


def test_duration_limit_adds_output_side_t_flag():
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf", hwaccel=HwAccelPlan(), duration_limit=30.0,
    )
    # -t must come after -lavfi (an output option, bounding the whole
    # filtered output) not before either -i (which would just be misplaced).
    lavfi_idx = cmd.index("-lavfi")
    t_idx = cmd.index("-t")
    assert t_idx > lavfi_idx
    assert cmd[t_idx + 1] == "30.000"


def test_libvmaf_options_include_model_threads_subsample_and_features():
    source_info = _info("source.mov", 1920, 1080)
    distorted_info = _info("distorted.mp4", 1920, 1080)
    options = VmafOptions(
        model="version=vmaf_4k_v0.6.1", n_threads=8, n_subsample=2,
        extra_features=["name=psnr", "name=float_ssim"],
    )

    graph = _build_filtergraph(
        source_info, distorted_info, options, None, None,
        hwaccel=HwAccelPlan(), log_path=Path("log.json"),
    )

    libvmaf_part = graph.split("libvmaf=", 1)[1]
    assert "model=version=vmaf_4k_v0.6.1" in libvmaf_part
    assert "n_threads=8" in libvmaf_part
    assert "n_subsample=2" in libvmaf_part
    assert "feature=name=psnr|name=float_ssim" in libvmaf_part


# ------------------------------------------------------------------ resolution round-trip test


def test_resample_downscales_then_upscales_back_to_source_resolution():
    source_info = _info("source.mkv", 3840, 2160)
    options = VmafOptions(
        model="version=vmaf_v0.6.1", scale_algorithm="lanczos",
        resample_test=ResampleTarget(width=1920, label="1080p"),
    )

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None, log_path=Path("log.json"),
    )

    dist_chain = graph.split(";")[3]  # base;split;ref;dist;libvmaf
    assert dist_chain.startswith("[dist_src]")
    # down to 1920x1080 (preserving the 3840x2160 source's 16:9 AR), then
    # back up to the source's original 3840x2160 -- never straight to 1080p.
    assert "scale=1920:1080:flags=lanczos" in dist_chain
    assert "scale=3840:2160:flags=lanczos" in dist_chain
    assert dist_chain.index("scale=1920:1080") < dist_chain.index("scale=3840:2160")


# ------------------------------------------------------------------ XPSNR


def _xpsnr_and_libvmaf_graphs(hwaccel=None, **options):
    source_info = _info("source.mov", 3840, 2160, "yuv420p10le")
    distorted_info = _info("distorted.mp4", 1920, 1080, "yuv420p10le")
    options = VmafOptions(compute_vmaf=False, compute_xpsnr=True, extra_features=["name=psnr", "name=float_ssim"],
                          **options)
    graphs = _build_filtergraph(
        source_info, distorted_info, options, CropBox(3840, 1608, 0, 276), None,
        hwaccel=hwaccel or HwAccelPlan(), log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )
    return graphs


def test_each_graph_reads_both_videos_itself_and_pairs_its_own_copies():
    """The regression the old layout's split guarded against -- one label
    consumed twice, so libvmaf compared the distorted video with itself and
    reported a perfect score -- cannot happen: each graph takes both decoded
    videos from FFmpeg, prepares them the same way, and consumes each of its
    own labels once."""
    libvmaf, xpsnr = _xpsnr_and_libvmaf_graphs().split(_GRAPH_SEPARATOR)
    for graph, suffix in ((libvmaf, "v"), (xpsnr, "x")):
        assert graph.count("[0:V:0]") == 1 and graph.count("[1:V:0]") == 1
        assert graph.count(f"[main_{suffix}]") == 2 and graph.count(f"[ref_{suffix}]") == 2  # made once, used once
    assert "[main_v][ref_v]libvmaf=" in libvmaf
    assert "[ref_x][main_x]xpsnr=" in xpsnr  # the source first: its activity weights the errors
    # The same preparation in both: crop, conversion, scaling of the source to the test video's size.
    assert libvmaf.split(";")[:2] == [chain.replace("_x]", "_v]") for chain in xpsnr.split(";")[:2]]
    assert "crop=3840:1608:0:276" in libvmaf and "scale=1920:1080" in libvmaf


def test_chained_xpsnr_takes_the_source_first_and_passes_it_on_to_libvmaf():
    """XPSNR weights each block's error by its first input's activity: the
    source's, as it is defined. It consumes the test video too, so that is
    split for libvmaf, which takes the source from XPSNR's output."""
    source_info = _info("source.mov", 1920, 1080)
    options = VmafOptions(
        model="version=vmaf_v0.6.1", compute_xpsnr=True, resample_test=ResampleTarget(width=960, label="480p"),
    )

    graph = _build_resample_test_filtergraph(
        source_info, options, source_crop=None, hwaccel_used=None,
        log_path=Path("log.json"), xpsnr_log_path=Path("xpsnr_log.txt"),
    )

    assert "[main]split=2[main_xpsnr][main_vmaf]" in graph
    assert "[ref][main_xpsnr]xpsnr=stats_file=xpsnr_log.txt:" in graph
    assert "[main_vmaf][xref]libvmaf=" in graph
    for label in ("[main_xpsnr]", "[main_vmaf]", "[xref]"):
        assert graph.count(label) == 2  # made once, used once


def test_parse_xpsnr_log_converts_1_indexed_to_0_indexed_frames(tmp_path):
    from vmaf_app.core.vmaf_runner import _parse_xpsnr_log

    log_path = tmp_path / "xpsnr_log.txt"
    log_path.write_text(
        "n:    1  XPSNR y: 17.3121  XPSNR u: 18.5356  XPSNR v: 18.8916\n"
        "n:    2  XPSNR y: -0.8422  XPSNR u: 3.0559  XPSNR v: 2.5208\n",
        encoding="utf-8",
    )

    result = _parse_xpsnr_log(log_path)

    assert result[0] == 17.3121  # xpsnr's n=1 -> our frame 0
    assert result[1] == -0.8422  # xpsnr's n=2 -> our frame 1 (also confirms negative values parse)


# --------------------------------------------------- analysis bit depth

def test_bit_depth_is_read_from_the_pixel_format_name(subtests):
    def check(pix_fmt, expected):
        # rgb24 is the trap: the 24 is bits per *pixel*, not per component, so a
        # "any digits in the name" rule would call an 8-bit format 24-bit.
        assert _bit_depth(pix_fmt) == expected

    for pix_fmt, expected in [
        ("yuv420p", 8), ("nv12", 8), ("nv21", 8), ("rgb24", 8), ("yuyv422", 8), ("", 8),
        ("yuv420p10le", 10), ("yuv422p10le", 10), ("p010le", 10),
        ("yuv420p12le", 12), ("gbrp12be", 12),
        ("yuv444p16le", 16), ("p016le", 16), ("gray10le", 10),
    ]:
        with subtests.test(pix_fmt=pix_fmt, expected=expected):
            check(pix_fmt, expected)


def test_analysis_format_takes_the_deeper_of_the_two_inputs(subtests):
    def check(formats, expected):
        assert analysis_pix_fmt(*formats) == expected

    for formats, expected in [
        (("yuv420p", "yuv420p"), "yuv420p"),
        (("yuv420p10le", "yuv420p10le"), "yuv420p10le"),
        (("yuv420p12le", "yuv420p12le"), "yuv420p12le"),
        # Mixed depths promote the shallower side rather than truncating the
        # deeper one -- a 10-bit master must not be measured through an 8-bit
        # pipe just because the encode under test is 8-bit.
        (("yuv420p10le", "yuv420p"), "yuv420p10le"),
        (("yuv420p", "yuv420p10le"), "yuv420p10le"),
        (("yuv420p12le", "yuv420p10le"), "yuv420p12le"),
        # libvmaf tops out at 12-bit, so deeper intermediates analyse at 12.
        (("yuv444p16le", "yuv420p"), "yuv420p12le"),
    ]:
        with subtests.test(formats=formats, expected=expected):
            check(formats, expected)


def test_both_branches_are_converted_to_the_same_analysis_format(subtests):
    def check(source_fmt, distorted_fmt, expected):
        # Both chains must name the SAME format: libvmaf compares two streams
        # and a mismatch either errors out or silently inserts a conversion
        # nobody chose.
        graph = _build_filtergraph(
            _info("source.mov", 1920, 1080, pix_fmt=source_fmt),
            _info("distorted.mp4", 1920, 1080, pix_fmt=distorted_fmt),
            VmafOptions(model="version=vmaf_v0.6.1"),
            source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
            log_path=Path("log.json"),
        )
        main_chain, ref_chain, _ = graph.split(";")

        assert f"format={expected}" in main_chain
        assert f"format={expected}" in ref_chain
        if expected != "yuv420p":
            assert "format=yuv420p," not in graph and "format=yuv420p[" not in graph

    for source_fmt, distorted_fmt, expected in [
        ("yuv420p", "yuv420p", "yuv420p"),
        ("yuv420p10le", "yuv420p10le", "yuv420p10le"),
        ("yuv420p10le", "yuv420p", "yuv420p10le"),
        ("yuv420p12le", "yuv420p10le", "yuv420p12le"),
    ]:
        with subtests.test(source_fmt=source_fmt, distorted_fmt=distorted_fmt, expected=expected):
            check(source_fmt, distorted_fmt, expected)


# ------------------------------------------------- subprocess reaping

class _FakePipe:
    """A pipe whose iteration can be made to raise, and that records being
    closed."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def __iter__(self):
        yield from self._lines

    def close(self):
        self.closed = True


class _FakeProcess:
    """Stands in for a Popen that ignores terminate() until killed, which is
    what a wedged hardware decoder actually does."""

    def __init__(self, stdout_lines, *, ignores_terminate=False):
        self.pid = 4242
        self.stdout = _FakePipe(stdout_lines)
        self.stderr = _FakePipe(["ffmpeg stderr\n"])
        self.terminated = False
        self.killed = False
        self.waited = False
        self.returncode = None
        self._ignores_terminate = ignores_terminate
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        if not self._ignores_terminate:
            self._alive = False
            self.returncode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        if self._alive:
            if timeout is None:
                self._alive = False
                self.returncode = 0
            else:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return self.returncode


# --------------------------------------------- per-input hardware decode

def test_each_input_gets_its_own_hwaccel_options():
    # -hwaccel is a per-input option in ffmpeg: it applies to the next -i on
    # the command line. That is what lets the two inputs decode differently,
    # and it is why the options must sit immediately before their own -i.
    cmd = _build_ffmpeg_cmd(
        Path("distorted.mp4"), Path("source.mp4"), "[0:v][1:v]libvmaf",
        hwaccel=HwAccelPlan(source="cuda", distorted="d3d11va"),
    )
    distorted_at = cmd.index(str(Path("distorted.mp4").resolve()))
    source_at = cmd.index(str(Path("source.mp4").resolve()))

    # Each input reads as: -hwaccel X -hwaccel_output_format <X's pixel format> -i <path>
    assert cmd[distorted_at - 5:distorted_at] == [
        "-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11", "-i",
    ]
    assert cmd[source_at - 5:source_at] == [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i",
    ]


@pytest.mark.parametrize(("plan", "expected"), [
    (HwAccelPlan(), [HwAccelPlan()]),
    (HwAccelPlan(source="cuda"), [HwAccelPlan(source="cuda"), HwAccelPlan()]),
    (HwAccelPlan(distorted="qsv"), [HwAccelPlan(distorted="qsv"), HwAccelPlan()]),
])
def test_every_ladder_ends_at_software_decode_without_repeating_a_plan(plan, expected):
    ladder = _fallback_ladder(plan)
    assert ladder == expected
    assert ladder[-1] == HwAccelPlan(), "the last resort must be all-CPU"
    assert len(set(ladder)) == len(ladder), "a plan that already failed is retried"


def test_a_run_retries_down_the_ladder_until_one_succeeds(monkeypatch, tmp_path):
    from vmaf_app.core import vmaf_runner

    attempts = []
    statuses = []

    def fake_run_ffmpeg(cmd, total_frames, on_progress, cancel_event, cwd, process_handle=None):
        plan = cmd[0]
        attempts.append(plan)
        # Only all-software decode works on this imaginary machine.
        code = 0 if not plan.uses_gpu else 1
        if code == 0:
            (Path(cwd) / "vmaf_log.json").write_text('{"frames": [{"frameNum": 0, "metrics": {"vmaf": 90}}]}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, code, "", "decoder error")

    monkeypatch.setattr(vmaf_runner, "_run_ffmpeg", fake_run_ffmpeg)

    vmaf_runner._execute_run(
        lambda plan, model, log_path, xpsnr_log_path: [plan],
        options=VmafOptions(), fps=30.0, total_frames=10,
        hwaccel=HwAccelPlan(source="cuda", distorted="cuda"),
        tmp_prefix="test_", on_progress=None, on_status=statuses.append,
        cancel_event=None, process_handle=None,
    )

    assert attempts == _fallback_ladder(HwAccelPlan(source="cuda", distorted="cuda"))
    # The status line names which input actually got hardware decode --
    # otherwise a silent per-input fallback looks like a run that never tried.
    assert "source cuda, distorted cuda" in statuses[0]
    assert "source cuda, distorted cpu" in statuses[1]
    assert "source cpu, distorted cuda" in statuses[2]
    assert "off" in statuses[3]


# ------------------------------------------- frame-count mismatch (framesync)


@pytest.mark.parametrize("compute_xpsnr", [False, True])
def test_the_comparison_stops_at_the_shorter_input(compute_xpsnr):
    """libvmaf and xpsnr are both framesync filters, and framesync's defaults
    extend the last frame of the secondary input past its EOF. A distorted
    file two frames longer than its source -- routine encoder padding, well
    inside the duration tolerance -- therefore scored two extra frames
    against a frozen copy of the source's final frame.

    Measured on a 30-vs-32-frame fixture: 32 scores, the last two 48.31 and
    31.27, dragging the mean from 99.64 to 95.84.
    """
    graph = _build_filtergraph(
        _info("source.mkv", 320, 180), _info("distorted.mkv", 320, 180),
        VmafOptions(model="version=vmaf_v0.6.1", compute_xpsnr=compute_xpsnr),
        source_crop=None, distorted_crop=None, hwaccel=HwAccelPlan(),
        log_path=Path("log.json"),
        xpsnr_log_path=Path("xpsnr.txt") if compute_xpsnr else None,
    )

    libvmaf_stage = graph.split("libvmaf=")[1]
    assert "shortest=1" in libvmaf_stage
    assert "repeatlast=0" in libvmaf_stage
    assert "ts_sync_mode=nearest" in libvmaf_stage
    if compute_xpsnr:
        # xpsnr sits before libvmaf and does its own framesync, so it needs
        # the same treatment or its stats file gains the phantom frames even
        # when the VMAF log does not.
        xpsnr_stage = graph.split("xpsnr=")[1].split("[xmain]")[0]
        assert "shortest=1" in xpsnr_stage
        assert "repeatlast=0" in xpsnr_stage
        assert "ts_sync_mode=nearest" in xpsnr_stage


# ------------------------------------------------- display geometry (shape)

def _shaped(name, w, h, sar="1:1"):
    return VideoInfo(
        path=Path(name), width=w, height=h, fps=30.0, duration=10.0,
        nb_frames=300, codec_name="h264", sar=sar, pix_fmt="yuv420p",
    )


def test_a_letterboxed_source_against_a_cropped_encode_is_rejected_uncropped():
    """THE case this exists for. A 1920x1080 source whose real content is a
    letterboxed 1920x816, compared with crop off against an already-cropped
    960x408 encode of it: the filtergraph scales 1920x1080 straight to
    960x408, squashing 16:9 into 2.35:1. libvmaf accepts it and returns a
    number -- 0.4977 on the real fixture, against 87.14 for the same pair
    cropped correctly. A wrong score is worse than a refused one.
    """
    with pytest.raises(VmafRunError, match="different shapes"):
        validate_display_geometry(
            _shaped("source.mkv", 1920, 1080), _shaped("encode.mkv", 960, 408),
            None, None,
        )


def test_the_same_pair_is_accepted_once_the_letterbox_is_cropped_off():
    # Cropping is exactly what makes them comparable, which is why the check
    # runs after crops are resolved rather than before.
    validate_display_geometry(
        _shaped("source.mkv", 1920, 1080), _shaped("encode.mkv", 960, 408),
        CropBox(w=1920, h=816, x=0, y=132), None,
    )


def test_a_test_video_stamped_a_millisecond_early_is_still_compared_frame_for_frame(tmp_path):
    """Two MKVs with the same frames at 23.976 fps, the test's timestamps
    1 ms earlier than the source's on every third frame (MKV rounds frame
    times to whole milliseconds, and two programs can round them apart).
    Each of those test frames was compared with the source's previous frame:
    an anime episode's VMAF NEG read 0 at scene cuts while its SSIMULACRA2,
    which pairs frames in order, read 93."""
    import shutil

    import numpy as np

    from vmaf_app.core.ffprobe import probe_video
    from vmaf_app.core.models import CropMode
    from vmaf_app.core.vmaf_runner import run_vmaf

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")

    def make(*args):
        subprocess.run([ffmpeg, "-v", "error", "-y", *args], check=True, capture_output=True, timeout=60)

    source, test, early = (tmp_path / name for name in ("source.mkv", "test.mkv", "early.mkv"))
    make("-f", "lavfi", "-i", "testsrc2=size=192x108:rate=24000/1001:duration=1", "-c:v", "ffv1", str(source))
    make("-i", str(source), "-c:v", "libx264", "-crf", "30", "-bf", "0", str(test))
    make("-i", str(test), "-c", "copy", "-bsf:v",
         "setts=pts=PTS-eq(mod(N\\,3)\\,1):dts=DTS-eq(mod(N\\,3)\\,1)", str(early))
    options = VmafOptions(compute_vmaf=True, compute_xpsnr=True, extra_features=["name=psnr"],
                          gpu_decode=False, crop_mode=CropMode.NONE, n_threads=2)
    source_info = probe_video(source)
    clean = run_vmaf(source_info, probe_video(test), options)
    jittered = run_vmaf(source_info, probe_video(early), options)
    for key in ("vmaf", "psnr", "xpsnr"):
        assert np.array_equal(np.asarray(jittered.frames.values(key)), np.asarray(clean.frames.values(key))), key
