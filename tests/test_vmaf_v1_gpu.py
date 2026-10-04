"""VMAF v1 with the GPU (vmaf_v1_gpu): which runs take it, as far as that can
be tested without a GPU -- GitHub's runner has none -- and, on a PC with one,
the scorer against libvmaf's CPU VMAF v1 on the same frames."""
import json
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import job_runner, vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan
from vmaf_app.core.models import VideoInfo, VmafOptions

MODELS = Path(vmaf_v1_gpu.__file__).resolve().parents[1] / "models"
STANDARD = MODELS / "vmaf_v1.0.16" / "vmaf_v1.0.16_3d0h.json"
HFR = MODELS / "vmaf_v1.0.16_hfr" / "vmaf_v1.0.16_hfr_1d5h_2160.json"


def _info(path: str, width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def _halves(keys, options=None):
    job = job_runner.VmafJob(_info("s.mp4"), _info("d.mp4"), options or VmafOptions(), label="d", metric_keys=keys)
    run = job_runner.JobRun(job_runner.JobScheduler([job]), 0, job)
    return [(task.backend_id, task.metric_keys, run.pool_of(task)) for task in run.plan.tasks]


# ---------------------------------------------------------- which runs take it

def test_every_bundled_v1_model_has_the_four_features_the_scorer_calculates():
    models = sorted(MODELS.glob("*/vmaf_v1.0.16_*.json"))
    assert len(models) == 8
    for model in models:
        options = vmaf_v1_gpu.model_options(model)
        assert set(options) == {vmaf_v1_gpu._ADM3, vmaf_v1_gpu._MOTION3, vmaf_v1_gpu._CAMBI, vmaf_v1_gpu._SPEED}
        assert options[vmaf_v1_gpu._ADM3]["adm_enhn_gain_limit"] == 1.0  # the only limit the shaders have


def _other_model(folder: Path) -> Path:
    """A model file with VMAF v0.6.1's features."""
    names = ["VMAF_integer_feature_adm2_score", "VMAF_integer_feature_motion2_score"]
    path = folder / "other.json"
    path.write_text(json.dumps({"model_dict": {"feature_names": names, "feature_opts_dicts": [{}, {}]}}))
    return path


def test_a_model_that_is_not_vmaf_v1s_is_refused(tmp_path):
    with pytest.raises(vmaf_v1_gpu.VmafV1Error, match="not a VMAF v1 model"):
        vmaf_v1_gpu.model_options(_other_model(tmp_path))
    with pytest.raises(vmaf_v1_gpu.VmafV1Error, match="not a VMAF model file"):
        vmaf_v1_gpu.model_options(tmp_path / "missing.json")


def test_a_video_speed_has_no_block_to_score_at_is_the_cpus():
    """libvmaf reads past its buffers there, on the CPU as with the GPU."""
    plain = {vmaf_v1_gpu._SPEED: {}}
    halved = {vmaf_v1_gpu._SPEED: {"speed_prescale": 0.5}}
    assert not vmaf_v1_gpu.speed_too_small(plain, 160, 160) and vmaf_v1_gpu.speed_too_small(plain, 1920, 158)
    assert not vmaf_v1_gpu.speed_too_small(halved, 320, 320) and vmaf_v1_gpu.speed_too_small(halved, 316, 1080)


def test_options_are_written_as_libvmafs_parser_reads_them():
    assert [vmaf_v1_gpu._text(value) for value in (True, False, 0.5, 2, "bilinear")] == [
        "true", "false", "0.5", "2", "bilinear"]


def test_vmaf_v1_goes_to_the_gpu_where_its_self_test_passed(monkeypatch, tmp_path):
    model = f"path={STANDARD}"
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", compute_vmaf_v1=True, model_v1=model) == {
        "vmaf": "vmaf_v0.6.1"}  # conftest: no GPU for VMAF v1
    monkeypatch.setattr(vmaf_v1_gpu, "_probed", (True, "Vulkan on a GPU"))
    assert vmaf_cuda.scores_on_gpu(True, True, "version=vmaf_v0.6.1", compute_vmaf_v1=True, model_v1=model) == {
        "vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg", "vmaf_v1": model}
    # Its GPU half is Vulkan's: it needs no GPU that scores VMAF v0.6.1.
    monkeypatch.setattr(vmaf_cuda, "_probed", (False, "no GPU"))
    assert vmaf_cuda.scores_on_gpu(True, False, "version=vmaf_v0.6.1", compute_vmaf_v1=True, model_v1=model) == {
        "vmaf_v1": model}
    assert vmaf_cuda.scores_on_gpu(False, False, "", compute_vmaf_v1=True, model_v1="") == {"vmaf_v1": ""}  # Auto
    # The CPU's: the video set to CPU, more than 10 bits, an odd size, too small for SpEED, another model.
    for refused in ({"enabled": False}, {"bit_depth": 12}, {"size": (1921, 1080)}, {"size": (1920, 120)},
                    {"model_v1": f"path={_other_model(tmp_path)}"}):
        arguments = {"compute_vmaf_v1": True, "model_v1": model, **refused}
        assert vmaf_cuda.scores_on_gpu(False, False, "", **arguments) is None, refused


def test_vmaf_v1_joins_the_gpus_half_of_a_job(monkeypatch):
    gpu_vmaf = job_runner.GPU_VMAF
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf"))
    assert _halves(("vmaf", "vmaf_v1", "psnr")) == [  # conftest: no GPU for VMAF v1
        ("ffmpeg", ("vmaf_v1", "psnr"), "cpu"), (gpu_vmaf, ("vmaf",), "gpu")]
    monkeypatch.setattr(vmaf_v1_gpu, "_probed", (True, "Vulkan on a GPU"))
    assert _halves(("vmaf", "vmaf_v1", "psnr")) == [
        ("ffmpeg", ("psnr",), "cpu"), (gpu_vmaf, ("vmaf", "vmaf_v1"), "gpu")]
    assert _halves(("vmaf_v1",)) == [(gpu_vmaf, ("vmaf_v1",), "gpu")]
    assert _halves(("vmaf_v1", "psnr"), VmafOptions(vmaf_on_gpu=False)) == [("ffmpeg", ("vmaf_v1", "psnr"), "cpu")]


def test_a_crashed_or_failed_probe_leaves_vmaf_v1_to_the_cpu(monkeypatch):
    from vmaf_app.core import isolated

    def crash(function, **_):
        raise isolated.IsolatedCrashError("the GPU VMAF v1 probe", 0xC0000005)

    monkeypatch.setattr(vmaf_v1_gpu, "_probed", None)
    monkeypatch.setattr(isolated, "run_isolated", crash)
    assert vmaf_v1_gpu.available()[1].startswith("the GPU VMAF v1 probe crashed")
    assert not vmaf_v1_gpu.scores(f"path={STANDARD}", (1920, 1080))


def test_the_self_test_frames_are_the_same_bytes_everywhere():
    import hashlib

    for bits in (8, 10):
        reference, distorted = vmaf_v1_gpu._self_test_frames(bits)
        again = vmaf_v1_gpu._self_test_frames(bits)
        assert reference == again[0] and distorted == again[1]
        width, height = vmaf_v1_gpu._PROBE_SIZE
        assert {len(frame) for frame in reference + distorted} == {width * height * 3 // 2 * (1 if bits == 8 else 2)}
        assert len({hashlib.sha256(bytes(frame)).digest() for frame in reference + distorted}) == 8


# ------------------------------------------------------- on a PC with a GPU

def _usable_devices():
    try:
        return [device for device in vmaf_vulkan.devices() if device.usable and device.kind in (1, 2)]
    except (OSError, vmaf_cuda.VmafGpuError):
        return []


_DEVICES = _usable_devices() if vmaf_vulkan.LIBRARY_PATH.is_file() and vmaf_cuda.LIBRARY_PATH.is_file() else []
needs_gpu = pytest.mark.skipif(not _DEVICES, reason="no GPU that Vulkan can calculate VMAF on")


def _score(model, bits, reference, distorted, step=1, device=None):
    scorer = vmaf_v1_gpu.V1Scorer(*vmaf_v1_gpu._PROBE_SIZE, bits, model, step, device=device, threads=2)
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        frames, values = scorer.features()
        return frames, values, vmaf_v1_gpu.predict(model, frames, values)
    finally:
        scorer.close()


@needs_gpu
@pytest.mark.parametrize(("model", "bits", "step"), [(STANDARD, 8, 1), (STANDARD, 10, 2), (HFR, 10, 1), (HFR, 8, 3)])
def test_every_gpu_gives_libvmafs_cpu_features_and_scores_bit_for_bit(model, bits, step):
    reference, distorted = vmaf_v1_gpu._self_test_frames(bits, 7)
    frames, expected, scores = vmaf_v1_gpu.cpu_reference(model, *vmaf_v1_gpu._PROBE_SIZE, bits, reference, distorted,
                                                         step)
    assert frames.tolist() == list(range(0, 7, step))
    assert np.all((scores > 0) & (scores < 100)) and len(set(scores.tolist())) > 1
    for device in _DEVICES:
        got_frames, values, got = _score(model, bits, reference, distorted, step, device.index)
        assert np.array_equal(got_frames, frames)
        for feature in expected:
            assert np.array_equal(values[feature].view(np.uint64), expected[feature].view(np.uint64)), (
                device.name, feature)
        assert np.array_equal(got, scores), device.name


@needs_gpu
def test_the_scorer_returns_scores_as_the_other_gpu_scorers_do():
    reference, distorted = vmaf_v1_gpu._self_test_frames(8, 3)
    scorer = vmaf_v1_gpu.V1Scorer(*vmaf_v1_gpu._PROBE_SIZE, 8, STANDARD, threads=2)
    try:
        assert scorer.frame_bytes == len(reference[0])
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        with pytest.raises(vmaf_v1_gpu.VmafV1Error, match="cut short"):
            scorer.add(reference[0][:-1], distorted[0])
        frames, scores = scorer.finish()
    finally:
        scorer.close()
        scorer.close()  # safe to call twice
    assert frames.tolist() == [0, 1, 2] and set(scores) == {"vmaf_v1"}
    assert np.array_equal(scores["vmaf_v1"], np.round(scores["vmaf_v1"], 6))  # six decimals, as libvmaf's log


@needs_gpu
def test_a_run_with_no_frames_has_no_scores():
    scorer = vmaf_v1_gpu.V1Scorer(*vmaf_v1_gpu._PROBE_SIZE, 8, STANDARD, threads=2)
    try:
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    assert len(frames) == 0 and len(scores["vmaf_v1"]) == 0


@needs_gpu
def test_a_size_too_small_for_speed_is_refused_before_libvmaf_sees_it():
    with pytest.raises(vmaf_v1_gpu.VmafV1Error, match="too small"):
        vmaf_v1_gpu.V1Scorer(128, 96, 8, STANDARD)


@needs_gpu
def test_vmaf_v1_beside_vmaf_and_neg_scores_the_same_frames():
    reference, distorted = vmaf_v1_gpu._self_test_frames(8, 4)
    alone = _score(STANDARD, 8, reference, distorted, 2)[2]
    models = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg", "vmaf_v1": f"path={STANDARD}"}
    scorer = vmaf_v1_gpu.MultiScorer(*vmaf_v1_gpu._PROBE_SIZE, 8, models, 2, backend="vulkan")
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    assert frames.tolist() == [0, 2] and set(scores) == set(models)
    assert np.array_equal(scores["vmaf_v1"], alone)
    assert np.all(scores["vmaf_neg"] <= scores["vmaf"])


@needs_gpu
def test_the_probe_accepts_this_pc_and_refuses_a_gpu_that_calculates_wrongly(monkeypatch):
    monkeypatch.delenv(vmaf_vulkan.DEVICE_VARIABLE, raising=False)
    available, text = vmaf_v1_gpu.probe()
    assert available and text.startswith("Vulkan on ")
    real = vmaf_v1_gpu.cpu_reference

    def wrong(*args, **kwargs):  # as a driver whose ADM is a bit off
        frames, values, scores = real(*args, **kwargs)
        values[vmaf_v1_gpu._ADM3] = np.nextafter(values[vmaf_v1_gpu._ADM3], 2.0)
        return frames, values, scores

    monkeypatch.setattr(vmaf_v1_gpu, "cpu_reference", wrong)
    available, text = vmaf_v1_gpu.probe()
    assert not available and "calculates VMAF v1 wrongly" in text
