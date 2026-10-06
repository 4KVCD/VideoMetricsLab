"""PSNR, SSIM and XPSNR by the bundled libvmaf's CPU extractors in the app
(vmaf_cuda.CpuScorer, vmaf_runner._run_cpu_metrics), in place of FFmpeg's
libvmaf and xpsnr filters: the same scores, and FFmpeg's filters again
whenever the app's way fails. The scorer runs for real -- libvmaf's CPU code
needs no GPU."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import vmaf_cuda
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import FrameScores, ResampleTarget, VideoInfo, VmafOptions

needs_libvmaf = pytest.mark.skipif(not vmaf_cuda.LIBRARY_PATH.is_file(), reason="libvmaf.dll is not bundled")
needs_xpsnr = pytest.mark.skipif(not (vmaf_cuda.LIBRARY_PATH.is_file() and vmaf_cuda.cpu_scores_xpsnr()),
                                 reason="the bundled libvmaf-fast has no XPSNR")


def _info(path: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def _rated(path: str, average: float = 24.0, nominal: float = 24.0) -> VideoInfo:
    """ffprobe's avg_frame_rate (0: none) and r_frame_rate."""
    return VideoInfo(Path(path), 1920, 1080, average or nominal, 10.0, 240, "hevc", pix_fmt="yuv420p",
                     nominal_fps=nominal, average_fps=average)


def _frames(width: int, height: int, bits: int, count: int, seed: int) -> list[bytes]:
    """4:2:0 frames as FFmpeg's rawvideo writes them: noise over gradients."""
    rng = np.random.default_rng(seed)
    peak = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    chroma = (width + 1) // 2 * ((height + 1) // 2)
    y, x = np.mgrid[0:height, 0:width]
    frames = []
    for index in range(count):
        luma = np.clip((x * 3 + y * 2 + index * 5) % 256 * peak // 255 + rng.integers(-peak // 20, peak // 20,
                                                                                         (height, width)), 0, peak)
        planes = [luma.ravel(), rng.integers(0, peak + 1, chroma), rng.integers(0, peak + 1, chroma)]
        frames.append(b"".join(plane.astype(dtype).tobytes() for plane in planes))
    return frames


def _ffmpegs_scores(tmp_path: Path, width: int, height: int, bits: int, reference, distorted) -> dict:
    """PSNR (psnr_y) and SSIM (float_ssim) of the frames by FFmpeg's libvmaf
    filter, as its log has them."""
    fmt = "yuv420p10le" if bits > 8 else "yuv420p"
    for name, frames in (("ref.yuv", reference), ("dis.yuv", distorted)):
        (tmp_path / name).write_bytes(b"".join(frames))
    raw = ["-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{width}x{height}", "-r", "24"]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", *raw, "-i", "dis.yuv", *raw, "-i", "ref.yuv",
                    "-lavfi", "[0:v][1:v]libvmaf=log_fmt=json:log_path=log.json:model=\'\':"
                              "feature=name=psnr|name=float_ssim", "-f", "null", "-"],
                   check=True, cwd=tmp_path, capture_output=True)
    frames = json.loads((tmp_path / "log.json").read_text())["frames"]
    return {"psnr": [f["metrics"]["psnr_y"] for f in frames], "ssim": [f["metrics"]["float_ssim"] for f in frames]}


@needs_libvmaf
@pytest.mark.parametrize(("width", "height", "bits"), [(320, 240, 8), (642, 362, 10), (321, 241, 8), (641, 361, 10)])
def test_the_scorers_psnr_and_ssim_are_ffmpegs_libvmafs(tmp_path, width, height, bits):
    """To the six decimals FFmpeg's log keeps, odd sizes too (whose chroma
    FFmpeg rounds up and libvmaf down)."""
    reference, distorted = _frames(width, height, bits, 5, 1), _frames(width, height, bits, 5, 2)
    scorer = vmaf_cuda.CpuScorer(width, height, bits, ("psnr", "ssim"), threads=4)
    try:
        assert scorer.frame_bytes == len(reference[0])
        for ref, dist in zip(reference, distorted, strict=True):
            scorer.add(ref, dist)
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    want = _ffmpegs_scores(tmp_path, width, height, bits, reference, distorted)
    assert frames.tolist() == list(range(5))
    assert scores["psnr"].tolist() == want["psnr"]
    assert scores["ssim"].tolist() == want["ssim"]


def _ffmpegs_xpsnr(tmp_path: Path, width: int, height: int, bits: int, rate: int, reference, distorted) -> list:
    """XPSNR (Y) of the frames by FFmpeg's xpsnr filter, the test video its
    first input as the app's commands give it, from its stats file."""
    fmt = "yuv420p10le" if bits > 8 else "yuv420p"
    for name, frames in (("ref.yuv", reference), ("dis.yuv", distorted)):
        (tmp_path / name).write_bytes(b"".join(frames))
    raw = ["-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{width}x{height}", "-r", str(rate)]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", *raw, "-i", "dis.yuv", *raw, "-i", "ref.yuv",
                    "-lavfi", "[0:v][1:v]xpsnr=stats_file=xpsnr.txt", "-f", "null", "-"],
                   check=True, cwd=tmp_path, capture_output=True)
    values = vr._parse_xpsnr_log(tmp_path / "xpsnr.txt")
    return [values[number] for number in sorted(values)]


@needs_xpsnr
@pytest.mark.parametrize(("width", "height", "bits", "rate"), [
    (320, 240, 8, 24),  # small: the weights min-smoothed
    (641, 361, 10, 60),  # odd, second-order temporal activity
    (2304, 1296, 10, 30),  # above 2048x1152: downsampled activity
])
def test_the_scorers_xpsnr_is_ffmpegs(tmp_path, width, height, bits, rate):
    """To the four decimals FFmpeg's stats file keeps, frame after frame (its
    temporal activity reads the frames before); PSNR beside it, from XPSNR's
    own squared errors, FFmpeg's libvmaf's to its six."""
    reference, distorted = _frames(width, height, bits, 5, 5), _frames(width, height, bits, 5, 6)
    scorer = vmaf_cuda.CpuScorer(width, height, bits, ("psnr", "xpsnr"), threads=4, frame_rate=rate)
    try:
        for ref, dist in zip(reference, distorted, strict=True):
            scorer.add(ref, dist)
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    assert frames.tolist() == list(range(5))
    assert scores["xpsnr"].tolist() == _ffmpegs_xpsnr(tmp_path, width, height, bits, rate, reference, distorted)
    assert scores["psnr"].tolist() == _ffmpegs_scores(tmp_path, width, height, bits, reference, distorted)["psnr"]


@needs_libvmaf
def test_every_n_subsample_th_frame_is_scored(tmp_path):
    reference, distorted = _frames(320, 240, 8, 7, 3), _frames(320, 240, 8, 7, 4)
    scorer = vmaf_cuda.CpuScorer(320, 240, 8, ("ssim",), n_subsample=3)
    try:
        for ref, dist in zip(reference, distorted, strict=True):
            scorer.add(ref, dist)
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    want = _ffmpegs_scores(tmp_path, 320, 240, 8, reference, distorted)
    assert frames.tolist() == [0, 3, 6]
    assert scores["ssim"].tolist() == [want["ssim"][i] for i in (0, 3, 6)]
    assert set(scores) == {"ssim"}


@needs_libvmaf
def test_a_frame_cut_short_is_an_error_and_the_pictures_go_back():
    """A picture taken from libvmaf's pool and never handed over would keep
    vmaf_close waiting for ever."""
    scorer = vmaf_cuda.CpuScorer(320, 240, 8, ("psnr",))
    try:
        with pytest.raises(vmaf_cuda.VmafGpuError, match="shorter"):
            scorer.add(bytes(scorer.frame_bytes), bytes(scorer.frame_bytes - 1))
        frames, scores = scorer.finish()
        assert frames.tolist() == [] and scores["psnr"].tolist() == []
    finally:
        scorer.close()


# ------------------------------------------------------------- the plan

@pytest.fixture
def app_scores(monkeypatch):
    monkeypatch.delenv(vr.CPU_METRICS_VARIABLE, raising=False)
    monkeypatch.setattr(vmaf_cuda, "pairs_in_app", lambda: True)


@needs_libvmaf
@pytest.mark.parametrize(("features", "metrics"), [
    (["name=psnr", "name=float_ssim"], {"psnr": "psnr", "ssim": "ssim"}),
    (["name=float_ssim"], {"ssim": "ssim"}),
    (["name=psnr"], {"psnr": "psnr"}),
])
def test_psnr_and_ssim_alone_are_the_apps(app_scores, features, metrics):
    plan = vr._cpu_metrics_plan(VmafOptions(compute_vmaf=False, extra_features=features, n_threads=6),
                                (1920, 1080), _info("s.mkv"), _info("d.mkv"), HwAccelPlan())
    assert plan is not None and plan.backend == "cpu" and plan.models == metrics
    assert (plan.width, plan.height, plan.bit_depth, plan.threads) == (1920, 1080, 8, 6)


@needs_libvmaf
@pytest.mark.parametrize("options", [
    VmafOptions(compute_vmaf=True, extra_features=["name=psnr"]),  # FFmpeg's filter runs for VMAF anyway
    VmafOptions(compute_vmaf=False, compute_vmaf_neg=True, extra_features=["name=psnr"]),
    VmafOptions(compute_vmaf=False, extra_features=[]),
    VmafOptions(compute_vmaf=False, extra_features=["name=psnr", "name=psnr_hvs"]),  # not one CpuScorer has
    VmafOptions(compute_vmaf=False, extra_features=["name=psnr"], resample_test=ResampleTarget(1280, "720p")),
])
def test_the_rest_stays_with_ffmpegs_filter(app_scores, options):
    assert vr._cpu_metrics_plan(options, (1920, 1080), _info("s.mkv"), _info("d.mkv"), HwAccelPlan()) is None


def test_the_variable_leaves_them_to_ffmpeg(app_scores, monkeypatch):
    monkeypatch.setenv(vr.CPU_METRICS_VARIABLE, "FFmpeg")
    options = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"])
    assert vr._cpu_metrics_plan(options, (1920, 1080), _info("s.mkv"), _info("d.mkv"), HwAccelPlan()) is None


@needs_libvmaf
def test_ffmpegs_own_pairing_takes_an_even_size_and_at_most_10_bits(monkeypatch):
    """An FFmpeg older than 6.1 pairs the frames on a canvas (_gpu_pairs_stage)."""
    monkeypatch.delenv(vr.CPU_METRICS_VARIABLE, raising=False)
    monkeypatch.setattr(vmaf_cuda, "pairs_in_app", lambda: False)
    options = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"])
    assert vr._cpu_metrics_plan(options, (1920, 1080), _info("s.mkv"), _info("d.mkv"), HwAccelPlan()) is not None
    assert vr._cpu_metrics_plan(options, (1921, 1080), _info("s.mkv"), _info("d.mkv"), HwAccelPlan()) is None


@pytest.mark.parametrize(("average", "nominal", "rate"), [
    (24000 / 1001, 24000 / 1001, 23),
    (60.0, 60.0, 60),
    (29.97, 30.0, 30),
    (0.0, 25.0, None),  # no average: FFmpeg may take the codec's rate
    (25.0, 50.0, None),  # fields: the same
    (24.0, 0.0, None),
])
def test_xpsnrs_frame_rate_is_ffmpegs_where_that_is_known(average, nominal, rate):
    assert vr._xpsnr_frame_rate(_rated("s.mkv", average, nominal)) == rate


def test_xpsnr_is_the_apps_only_where_it_is_ffmpegs_to_the_bit(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "cpu_scores_xpsnr", lambda: True)
    assert vr._xpsnr_in_app(1920, 1080, 10) and vr._xpsnr_in_app(1279, 719, 12)
    assert not vr._xpsnr_in_app(2561, 1440, 10)  # FFmpeg's filter reads beyond an odd picture
    assert not vr._xpsnr_in_app(1920, 1080, 16)
    monkeypatch.setattr(vmaf_cuda, "cpu_scores_xpsnr", lambda: False)  # a libvmaf-fast without it
    assert not vr._xpsnr_in_app(1920, 1080, 10)


@needs_libvmaf
def test_xpsnr_joins_them_in_the_app_where_it_can(app_scores, monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "cpu_scores_xpsnr", lambda: True)
    options = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"], compute_xpsnr=True)
    cuda = HwAccelPlan(source="cuda", distorted="cuda")
    plan = vr._cpu_metrics_plan(options, (1920, 1080), _rated("s.mkv", 60.0, 60.0), _rated("d.mkv"), cuda)
    assert plan.models == {"psnr": "psnr", "xpsnr": "xpsnr"} and plan.frame_rate == 60
    # A frame rate FFmpeg might not take as known here: FFmpeg's, beside them.
    plan = vr._cpu_metrics_plan(options, (1920, 1080), _rated("s.mkv", 0.0, 25.0), _rated("d.mkv"), cuda)
    assert plan.models == {"psnr": "psnr"}


@needs_libvmaf
def test_xpsnr_alone_is_the_apps_only_from_the_gpus_decoder(app_scores, monkeypatch):
    """From FFmpeg's pipes FFmpeg's filter keeps up, in less memory."""
    monkeypatch.setattr(vmaf_cuda, "cpu_scores_xpsnr", lambda: True)
    options = VmafOptions(compute_vmaf=False, compute_xpsnr=True)
    cuda = HwAccelPlan(source="cuda", distorted="cuda")
    monkeypatch.setattr(vr, "_decoded_in_app", lambda *_args: True)
    plan = vr._cpu_metrics_plan(options, (1920, 1080), _rated("s.mkv"), _rated("d.mkv"), cuda)
    assert plan.models == {"xpsnr": "xpsnr"} and plan.frame_rate == 24
    monkeypatch.setattr(vr, "_decoded_in_app", lambda *_args: False)
    assert vr._cpu_metrics_plan(options, (1920, 1080), _rated("s.mkv"), _rated("d.mkv"), cuda) is None


def test_the_app_decodes_when_one_gpu_decodes_both_and_nothing_is_scaled():
    cuda = HwAccelPlan(source="cuda", distorted="cuda")
    source, test = _info("s.mkv"), _info("d.mkv")
    assert vr._decoded_in_app(source, test, None, None, (1920, 1080), cuda)
    assert not vr._decoded_in_app(source, test, None, None, (1280, 720), cuda)  # scaled
    assert not vr._decoded_in_app(source, test, None, None, (1920, 1080), HwAccelPlan(source="cuda"))
    prores = VideoInfo(Path("p.mov"), 1920, 1080, 24.0, 10.0, 240, "prores", pix_fmt="yuv422p10le")
    assert not vr._decoded_in_app(prores, test, None, None, (1920, 1080), cuda)


@needs_libvmaf
@pytest.mark.parametrize(("hwaccel", "in_the_app"), [
    (HwAccelPlan(source="cuda", distorted="cuda"), True),
    (HwAccelPlan(source="qsv", distorted=None), False),
    (HwAccelPlan(), False),
])
def test_with_xpsnr_a_video_the_cpu_decodes_stays_with_ffmpegs_one_run(app_scores, hwaccel, in_the_app):
    """XPSNR's FFmpeg would decode it on the CPU a second time."""
    options = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"], compute_xpsnr=True)
    plan = vr._cpu_metrics_plan(options, (1920, 1080), _info("s.mkv"), _info("d.mkv"), hwaccel)
    assert (plan is not None) == in_the_app
    without_xpsnr = VmafOptions(compute_vmaf=False, extra_features=["name=psnr"])
    assert vr._cpu_metrics_plan(without_xpsnr, (1920, 1080), _info("s.mkv"), _info("d.mkv"), hwaccel) is not None


# ------------------------------------------------------------- the run

def _scores(frames, **columns) -> FrameScores:
    numbers = np.array(frames, dtype=np.int32)
    return FrameScores(numbers, numbers / 24.0, None, **{key: np.array(values, dtype=np.float32)
                                                          for key, values in columns.items()})


def _plan() -> vr._GpuPlan:
    return vr._GpuPlan({"psnr": "psnr", "ssim": "ssim"}, 1920, 1080, 8, backend="cpu", threads=4)


def _run(monkeypatch, isolated, xpsnr=None, options=None, **kwargs):
    monkeypatch.setattr(vr, "run_isolated", isolated)
    calls = []

    def execute_run(build_command, **kwargs_):
        calls.append({**kwargs_, "build_command": build_command})
        if isinstance(xpsnr, BaseException):
            raise xpsnr
        return xpsnr

    monkeypatch.setattr(vr, "_execute_run", execute_run)
    options = options or VmafOptions(compute_vmaf=False, extra_features=["name=psnr", "name=float_ssim"],
                                     compute_xpsnr=xpsnr is not None)
    return vr._run_cpu_metrics(_plan(), _info("s.mkv"), _info("d.mkv"), options, None, None, HwAccelPlan(), 240,
                               model="", on_progress=None, cancel_event=None, process_handle=None, **kwargs), calls


def test_xpsnr_is_ffmpegs_beside_them_taken_for_the_frames_they_scored(monkeypatch):
    seen = {}

    def isolated(target, plan, source, distorted, options, *args, **kwargs):
        seen["options"] = options
        return _scores([0, 2, 4], psnr=[30.0, 31.0, 32.0], ssim=[0.9, 0.91, 0.92])

    frames, calls = _run(monkeypatch, isolated, xpsnr=_scores([0, 1, 2, 3], xpsnr=[40.0, 41.0, 42.0, 43.0]),
                         on_status=None)
    assert not seen["options"].compute_xpsnr  # the app's process scores PSNR and SSIM alone
    assert calls[0]["options"].extra_features == [] and calls[0]["options"].compute_xpsnr
    assert frames.frame.tolist() == [0, 2, 4]
    assert frames.values("psnr").tolist() == [30.0, 31.0, 32.0]
    assert frames.values("xpsnr")[:2].tolist() == [40.0, 42.0] and np.isnan(frames.values("xpsnr")[2])
    command = " ".join(calls[0]["build_command"](HwAccelPlan(), "", Path("vmaf.json"), Path("xpsnr.txt")))
    assert "xpsnr=" in command and "libvmaf" not in command  # FFmpeg scored them again beside XPSNR


@pytest.mark.parametrize(("features", "subsample"), [(["name=psnr"], 3), ([], 1)])
def test_xpsnr_in_the_app_needs_no_ffmpeg_of_its_own(monkeypatch, features, subsample):
    """XPSNR alone is scored for every frame, as FFmpeg's filter scores it."""
    seen = {}

    def isolated(target, plan, source, distorted, options, *args, **kwargs):
        seen["options"] = options
        return _scores([0, 3], psnr=[30.0, 31.0], xpsnr=[40.0, 41.0])

    def execute_run(*_args, **_kwargs):
        raise AssertionError("an FFmpeg for XPSNR")

    monkeypatch.setattr(vr, "run_isolated", isolated)
    monkeypatch.setattr(vr, "_execute_run", execute_run)
    models = {"psnr": "psnr", "xpsnr": "xpsnr"} if features else {"xpsnr": "xpsnr"}
    plan = vr._GpuPlan(models, 1920, 1080, 8, backend="cpu", threads=4, frame_rate=24)
    options = VmafOptions(compute_vmaf=False, extra_features=features, compute_xpsnr=True, n_subsample=3)
    frames = vr._run_cpu_metrics(plan, _info("s.mkv"), _info("d.mkv"), options, None, None, HwAccelPlan(), 240,
                                 model="", on_progress=None, on_status=None, cancel_event=None, process_handle=None)
    assert seen["options"].compute_xpsnr and seen["options"].n_subsample == subsample
    assert frames.values("xpsnr").tolist() == [40.0, 41.0]


def test_a_failure_in_the_app_leaves_them_to_ffmpegs_filter(monkeypatch):
    statuses = []

    def isolated(*_args, **_kwargs):
        raise vmaf_cuda.VmafGpuError("libvmaf crashed")

    frames, _calls = _run(monkeypatch, isolated, on_status=statuses.append)
    assert frames is None
    assert any("PSNR and SSIM in the app failed" in status for status in statuses)


def test_a_cancel_is_not_a_failure(monkeypatch):
    def isolated(*_args, **_kwargs):
        raise vr.Cancelled("Cancelled by user")

    with pytest.raises(vr.Cancelled):
        _run(monkeypatch, isolated, xpsnr=_scores([0], xpsnr=[40.0]), on_status=None)


def test_xpsnrs_ffmpeg_failing_fails_the_run(monkeypatch):
    def isolated(*_args, **_kwargs):
        return _scores([0], psnr=[30.0], ssim=[0.9])

    with pytest.raises(vr.VmafRunError, match="ffmpeg exited"):
        _run(monkeypatch, isolated, xpsnr=vr.VmafRunError("ffmpeg exited with code 1"), on_status=None)


def test_two_runs_progress_is_the_one_behinds():
    seen = []
    progress = vr._Progress(lambda *args: seen.append(args), 2)
    progress.part(0)(10, 100, 50.0)
    assert seen == []  # until both have said something
    progress.part(1)(4, 100, 20.0)
    progress.part(0)(20, 100, 50.0)
    progress.part(1)(30, 100, 60.0)
    assert seen == [(4, 100, 20.0), (4, 100, 20.0), (20, 100, 50.0)]


def test_their_scores_say_they_are_the_apps_libvmafs_with_ffmpegs_identity():
    frames = _scores([0, 1], psnr=[30.0, 31.0], ssim=[0.9, 0.91], xpsnr=[40.0, 41.0])
    results = vr._metric_results_for_current_run(frames, "", cpu_keys={"psnr", "ssim"})
    in_the_app = vr._metric_results_for_current_run(frames, "", cpu_keys={"xpsnr"}).get("xpsnr").provenance
    assert in_the_app.implementation == "libvmaf" and in_the_app.implementation_version == vmaf_cuda.CPU_BUILD
    assert in_the_app.implementation_compatibility_id == results.get("xpsnr").provenance.implementation_compatibility_id
    for key in ("psnr", "ssim"):
        provenance = results.get(key).provenance
        assert provenance.implementation == "libvmaf" and provenance.implementation_version == vmaf_cuda.CPU_BUILD
        assert provenance.compute_backend == "cpu"
        assert provenance.implementation_compatibility_id == "ffmpeg-libvmaf-v1"  # saved scores are reused
    assert results.get("xpsnr").provenance.implementation == "ffmpeg/xpsnr"


def test_the_gpus_frame_scores_take_psnr_and_ssim():
    frames = vr._gpu_frame_scores((np.array([0, 1], dtype=np.int32), {"psnr": np.array([30.123456, 31.0]),
                                                                      "ssim": np.array([0.9, 0.95]),
                                                                      "xpsnr": np.array([40.1234, np.inf])}), 24.0)
    assert frames.values("xpsnr").dtype == np.float32 and frames.values("xpsnr")[1] == np.inf
    assert frames.values("psnr").dtype == np.float32
    assert frames.values("psnr").tolist() == np.array([30.123456, 31.0], dtype=np.float32).tolist()
    assert frames.has("ssim") and not frames.has("vmaf")
