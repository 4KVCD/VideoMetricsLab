"""VMAF on the GPU with Vulkan (vmaf_vulkan): which backend a run takes, as
far as it can be tested without a GPU -- GitHub's runner has none -- and, on
a PC with one, the library itself: its sums on the probe's frames, and its
features against libvmaf's CUDA code where that runs."""
import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from vmaf_app.core import vmaf_cuda, vmaf_vulkan
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import GpuVendor, VmafOptions

MODELS = {"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"}
#: Before conftest's autouse fixture replaces it for every test.
_REAL_SCORE_DECODED_ON_GPU = vr._score_decoded_on_gpu


# ------------------------------------------------------- choosing a backend

def _probes(monkeypatch, vendors, cuda=(True, "libvmaf 1"), vulkan=(True, 0, "Vulkan on a GPU")):
    """The two probes replaced: `cuda` and `vulkan` are what they return."""
    from vmaf_app.core import gpu, isolated

    monkeypatch.setattr(gpu, "detected_gpu_vendors", lambda: vendors)
    monkeypatch.setattr(isolated, "run_isolated",
                        lambda function, **_: cuda if function is vmaf_cuda.probe else vulkan)


def test_auto_takes_cuda_on_an_nvidia_gpu_and_vulkan_on_any_other(monkeypatch):
    _probes(monkeypatch, [GpuVendor.NVIDIA])
    assert vmaf_cuda._probe_once("auto") == ("cuda", None, "libvmaf 1 with CUDA")
    assert vmaf_cuda._probe_once("cuda")[0] == "cuda"
    _probes(monkeypatch, [GpuVendor.INTEL], vulkan=(True, 1, "Vulkan on Intel(R) Graphics"))
    assert vmaf_cuda._probe_once("auto") == ("vulkan", 1, "Vulkan on Intel(R) Graphics")
    assert vmaf_cuda._probe_once("hip")[0] == "vulkan"  # Vship's AMD build: VMAF has none


def test_the_other_backend_is_tried_when_the_first_cannot_score(monkeypatch):
    _probes(monkeypatch, [GpuVendor.NVIDIA], cuda=(False, "CUDA failed to start"))
    assert vmaf_cuda._probe_once("auto")[0] == "vulkan"
    _probes(monkeypatch, [GpuVendor.NVIDIA], vulkan=(False, None, "its driver calculates VMAF wrongly"))
    assert vmaf_cuda._probe_once("vulkan")[0] == "cuda"
    _probes(monkeypatch, [GpuVendor.INTEL], vulkan=(False, None, "no Vulkan driver"))
    assert vmaf_cuda._probe_once("auto") == (None, None, "no NVIDIA GPU; Vulkan: no Vulkan driver")


def test_a_run_feeds_the_backend_and_gpu_the_plan_names(monkeypatch):
    seen = {}

    class Attempt:
        def __init__(self, *args, **_kwargs):
            seen["args"] = args

        def output_args(self, limit):
            return ["RAW"]

        def finish(self, succeeded):
            return np.array([0, 1], dtype=np.int32), {"vmaf": np.array([90.0, 91.0])}

    monkeypatch.setattr(vmaf_cuda, "GpuAttempt", Attempt)
    monkeypatch.setattr(vr, "_run_ffmpeg", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "", ""))
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 64, 48, 8, "vulkan", 1)
    vr._execute_run(lambda *args: ["ffmpeg"], options=VmafOptions(), fps=24.0, total_frames=10,
                    hwaccel=HwAccelPlan(), tmp_prefix="vmaf_test_", on_progress=None, on_status=None,
                    cancel_event=None, process_handle=None, gpu=plan)
    assert seen["args"] == (64, 48, 8, {"vmaf": "vmaf_v0.6.1"}, 1, "vulkan", 1)


def test_a_gpu_run_decodes_in_its_own_process_where_the_gpus_decoder_can(subtests):
    """Where this process has a decoder for both videos -- the GPU's FFmpeg
    would decode them with, or the software decoder for a video FFmpeg
    decodes in software -- it decodes them itself; anything else goes
    through FFmpeg's pipes."""
    def check(backend, hwaccel, software, expected, monkeypatch):
        from pathlib import Path

        from vmaf_app.core.models import VideoInfo

        taken = []
        # conftest turns decoding in the scoring process off; this test is about it
        monkeypatch.setattr(vr, "_score_decoded_on_gpu", _REAL_SCORE_DECODED_ON_GPU)

        monkeypatch.setattr(vr.gpu_frames, "software_bundled", lambda: software)

        def cuda_decoded(*args, decoder=None, **kwargs):
            taken.append(f"cuda {'/'.join(decoder)}")
            return np.array([0], dtype=np.int32), {"vmaf": np.array([90.0])}

        def vulkan_decoded(*args, decoder=None, **kwargs):
            taken.append(f"vulkan {'/'.join(decoder)}")
            return np.array([0], dtype=np.int32), {"vmaf": np.array([90.0])}

        monkeypatch.setattr(vmaf_cuda, "score_decoded", cuda_decoded)
        monkeypatch.setattr(vmaf_vulkan, "score_decoded", vulkan_decoded)
        monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: taken.append("pipes") or "pipes")
        info = VideoInfo(Path("a.mkv"), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt="yuv420p")
        plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 64, 48, 8, backend, 0 if backend == "vulkan" else None)
        vr._score_on_gpu(plan, info, info, VmafOptions(compute_vmaf=True), None, None, "version=vmaf_v0.6.1",
                         HwAccelPlan(source=hwaccel[0], distorted=hwaccel[1]), 24)
        assert taken == [expected]

    for backend, hwaccel, software, expected in [
        ("cuda", ("cuda", "cuda"), False, "cuda nvidia/nvidia"),     # NVIDIA's decoder, libvmaf's CUDA code (as master)
        ("vulkan", ("cuda", "cuda"), False, "vulkan nvidia/nvidia"),  # the Vulkan setting on NVIDIA
        ("vulkan", ("qsv", "qsv"), False, "vulkan intel/intel"),     # Intel's GPU: oneVPL
        ("vulkan", ("d3d11va", "d3d11va"), False, "vulkan amd/amd"),
        ("cuda", ("qsv", "qsv"), False, "pipes"),              # libvmaf's CUDA code takes NVIDIA's pictures only
        ("vulkan", ("cuda", None), False, "pipes"),            # one video FFmpeg decodes in software (VVC, ...)
        ("vulkan", ("qsv", "cuda"), False, "pipes"),           # two GPU makers' decoders are not tried together
        ("vulkan", ("qsv", "cuda"), True, "pipes"),
        # ... decoded here by the software decoder where it is bundled:
        ("vulkan", ("cuda", None), True, "vulkan nvidia/software"),
        ("cuda", ("cuda", None), True, "cuda nvidia/software"),   # its pictures uploaded
        ("cuda", (None, None), True, "cuda software/software"),
        ("vulkan", (None, "qsv"), True, "vulkan software/intel"),
        ("cuda", ("qsv", None), True, "pipes"),                 # CUDA's code takes no Intel pictures
    ]:
        with subtests.test(backend=backend, hwaccel=hwaccel, software=software, expected=expected), pytest.MonkeyPatch.context() as case_patch:
            check(backend, hwaccel, software, expected, case_patch)


def test_neg_models_get_the_features_limited_to_a_gain_of_one():
    plain, neg = vmaf_vulkan._MODEL_FEATURES[False], vmaf_vulkan._MODEL_FEATURES[True]
    assert plain["VMAF_integer_feature_adm2_score"] == 4 and neg["integer_adm2_egl_1"] == 10
    assert [plain[f"VMAF_integer_feature_vif_scale{s}_score"] for s in range(4)] == [0, 1, 2, 3]
    assert [neg[f"integer_vif_scale{s}_egl_1"] for s in range(4)] == [6, 7, 8, 9]
    assert plain["VMAF_integer_feature_motion2_score"] == neg["VMAF_integer_feature_motion2_score"] == 5


def test_the_probe_frames_are_the_same_bytes_everywhere():
    """The probe compares a GPU's sums with known ones: its frames must not
    depend on the PC or on numpy's version."""
    for bits, expected in ((8, "6c2fdf1b"), (10, "f9f6b6d3")):
        reference, distorted = vmaf_vulkan.probe_frames(bits)
        digest = hashlib.sha256(b"".join(bytes(frame) for frame in reference + distorted)).hexdigest()
        assert digest.startswith(expected), (bits, digest)


# ------------------------------------------------------- on a PC with a GPU

def _usable_devices():
    try:
        return [device for device in vmaf_vulkan.devices() if device.usable and device.kind in (1, 2)]
    except (OSError, vmaf_cuda.VmafGpuError):
        return []


_DEVICES = _usable_devices() if vmaf_vulkan.LIBRARY_PATH.is_file() else []
needs_gpu = pytest.mark.skipif(not _DEVICES, reason="no GPU that Vulkan can calculate VMAF on")


@needs_gpu
@pytest.mark.parametrize("bits", [8, 10])
def test_every_gpu_gives_the_known_sums_for_the_probe_frames(bits):
    for device in _DEVICES:
        assert vmaf_vulkan.probe_sums(device.index, bits) == vmaf_vulkan._PROBE_SUMS[bits == 10], device.name


@needs_gpu
def test_a_frame_that_is_not_scored_still_counts_for_the_next_frames_motion():
    """libvmaf's n_subsample: motion is calculated for every frame, the rest
    for the scored ones."""
    reference, distorted = vmaf_vulkan.probe_frames(8, 4)
    rows = {}
    for step in (1, 2):
        scorer = vmaf_vulkan.VulkanScorer(*vmaf_vulkan._PROBE_SIZE, 8, {}, n_subsample=step)
        try:
            for ref, dis in zip(reference, distorted, strict=True):
                scorer.add(ref, dis)
            frames, features = scorer.features()
        finally:
            scorer.close()
        rows[step] = dict(zip(frames.tolist(), features, strict=True))
    for frame in (0, 2):
        assert np.array_equal(rows[1][frame], rows[2][frame])


def _cuda_features(reference, distorted, bits):
    scorer = vmaf_cuda.GpuScorer(*vmaf_vulkan._PROBE_SIZE, bits, MODELS)
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(bytearray(ref), bytearray(dis))
        frames, scores = scorer.finish()
        lib = vmaf_cuda._load()
        lib.vmaf_feature_score_at_index.restype = ctypes.c_int
        lib.vmaf_feature_score_at_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                    ctypes.POINTER(ctypes.c_double), ctypes.c_uint]
        names = {**vmaf_vulkan._MODEL_FEATURES[False], **vmaf_vulkan._MODEL_FEATURES[True]}
        rows = np.zeros((len(frames), vmaf_vulkan.FEATURE_COUNT))
        value = ctypes.c_double()
        for name, column in names.items():
            for slot, frame in enumerate(frames):
                assert lib.vmaf_feature_score_at_index(scorer._context, name.encode(), ctypes.byref(value),
                                                       int(frame)) == 0
                rows[slot, column] = value.value
        return rows, scores
    finally:
        scorer.close()


def _cuda_runs() -> bool:
    if not _DEVICES or not vmaf_cuda.LIBRARY_PATH.is_file() or not any(d.vendor == 0x10DE for d in _DEVICES):
        return False
    from vmaf_app.core.isolated import IsolatedCrashError, run_isolated

    try:
        return run_isolated(vmaf_cuda.probe, what="libvmaf's GPU probe")[0]
    except IsolatedCrashError:
        return False


@pytest.mark.skipif(not _cuda_runs(), reason="libvmaf's CUDA code does not run here")
@pytest.mark.parametrize("bits", [8, 10])
def test_every_gpu_gives_cudas_features_and_scores_bit_for_bit(bits):
    reference, distorted = vmaf_vulkan.probe_frames(bits, 6)
    cuda_rows, cuda_scores = _cuda_features(reference, distorted, bits)
    columns = sorted({**vmaf_vulkan._MODEL_FEATURES[False], **vmaf_vulkan._MODEL_FEATURES[True]}.values())
    for device in _DEVICES:
        scorer = vmaf_vulkan.VulkanScorer(*vmaf_vulkan._PROBE_SIZE, bits, MODELS, device=device.index)
        try:
            for ref, dis in zip(reference, distorted, strict=True):
                scorer.add(ref, dis)
            _frames, rows = scorer.features()
            _frames, scores = scorer.finish()
        finally:
            scorer.close()
        for column in columns:
            assert np.array_equal(rows[:, column].view(np.uint64), cuda_rows[:, column].view(np.uint64)), (
                device.name, column)
        for name in MODELS:
            assert np.array_equal(scores[name], cuda_scores[name]), (device.name, name)
