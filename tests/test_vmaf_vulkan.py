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


def test_the_vulkan_backend_setting_takes_vulkan_on_an_nvidia_gpu_too(monkeypatch):
    _probes(monkeypatch, [GpuVendor.NVIDIA])
    assert vmaf_cuda._probe_once("vulkan") == ("vulkan", 0, "Vulkan on a GPU")


def test_the_other_backend_is_tried_when_the_first_cannot_score(monkeypatch):
    _probes(monkeypatch, [GpuVendor.NVIDIA], cuda=(False, "CUDA failed to start"))
    assert vmaf_cuda._probe_once("auto")[0] == "vulkan"
    _probes(monkeypatch, [GpuVendor.NVIDIA], vulkan=(False, None, "its driver calculates VMAF wrongly"))
    assert vmaf_cuda._probe_once("vulkan")[0] == "cuda"
    _probes(monkeypatch, [GpuVendor.INTEL], vulkan=(False, None, "no Vulkan driver"))
    assert vmaf_cuda._probe_once("auto") == (None, None, "no NVIDIA GPU; Vulkan: no Vulkan driver")


def test_changing_the_backend_setting_probes_again_only_when_the_order_changes(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "_preference", "auto")
    monkeypatch.setattr(vmaf_cuda, "_probed", (True, "libvmaf with CUDA"))
    vmaf_cuda.set_gpu_backend("cuda")  # the same order as Auto: CUDA, then Vulkan
    assert vmaf_cuda._probed is not None
    vmaf_cuda.set_gpu_backend("vulkan")
    assert vmaf_cuda._probed is None and vmaf_cuda._preference == "vulkan"


def test_the_probe_result_says_which_backend_a_run_uses(monkeypatch):
    _probes(monkeypatch, [GpuVendor.AMD], vulkan=(True, 2, "Vulkan on a Radeon"))
    monkeypatch.setattr(vmaf_cuda, "_probed", None)
    monkeypatch.setattr(vmaf_cuda, "_backend", ("cuda", None))
    monkeypatch.setattr(vmaf_cuda, "_preference", "auto")
    assert vmaf_cuda.gpu_vmaf_available() == (True, "Vulkan on a Radeon")
    assert vmaf_cuda.gpu_vmaf_backend() == ("vulkan", 2)
    assert vmaf_cuda.scores_on_gpu(True, True, "version=vmaf_v0.6.1") == MODELS


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


@pytest.mark.parametrize(("backend", "hwaccel", "expected"), [
    ("cuda", ("cuda", "cuda"), "cuda nvidia"),     # NVIDIA's decoder, libvmaf's CUDA code (as master)
    ("vulkan", ("cuda", "cuda"), "vulkan nvidia"),  # the Vulkan setting on NVIDIA
    ("vulkan", ("qsv", "qsv"), "vulkan intel"),     # Intel's GPU: oneVPL
    ("vulkan", ("d3d11va", "d3d11va"), "vulkan amd"),
    ("cuda", ("qsv", "qsv"), "pipes"),              # libvmaf's CUDA code takes NVIDIA's pictures only
    ("vulkan", ("cuda", None), "pipes"),            # one video FFmpeg decodes in software (VVC, ...)
    ("vulkan", ("qsv", "cuda"), "pipes"),
])
def test_a_gpu_run_decodes_in_its_own_process_where_the_gpus_decoder_can(monkeypatch, backend, hwaccel, expected):
    """Where FFmpeg would decode both videos with the GPU, the scoring process
    decodes them itself; anything else goes through FFmpeg's pipes."""
    from pathlib import Path

    from vmaf_app.core.models import VideoInfo

    taken = []
    # conftest turns decoding in the scoring process off; this test is about it
    monkeypatch.setattr(vr, "_score_decoded_on_gpu", _REAL_SCORE_DECODED_ON_GPU)

    def cuda_decoded(*args, **kwargs):
        taken.append("cuda nvidia")
        return np.array([0], dtype=np.int32), {"vmaf": np.array([90.0])}

    def vulkan_decoded(*args, decoder=None, **kwargs):
        taken.append(f"vulkan {decoder}")
        return np.array([0], dtype=np.int32), {"vmaf": np.array([90.0])}

    monkeypatch.setattr(vmaf_cuda, "score_decoded", cuda_decoded)
    monkeypatch.setattr(vmaf_vulkan, "score_decoded", vulkan_decoded)
    monkeypatch.setattr(vr, "_execute_run", lambda *a, **k: taken.append("pipes") or "pipes")
    info = VideoInfo(Path("a.mkv"), 64, 48, 24.0, 1.0, 24, "hevc", pix_fmt="yuv420p")
    plan = vr._GpuPlan({"vmaf": "vmaf_v0.6.1"}, 64, 48, 8, backend, 0 if backend == "vulkan" else None)
    vr._score_on_gpu(plan, info, info, VmafOptions(compute_vmaf=True), None, None, "version=vmaf_v0.6.1",
                     HwAccelPlan(source=hwaccel[0], distorted=hwaccel[1]), 24)
    assert taken == [expected]


def test_the_attempt_scores_with_the_vulkan_scorer_for_the_vulkan_backend(monkeypatch):
    made = {}

    class Scorer:
        frame_bytes = 6

        def __init__(self, *args, **kwargs):
            made["args"], made["kwargs"] = args, kwargs

        def finish(self):
            return np.zeros(0, dtype=np.int32), {}

        def close(self):
            pass

    monkeypatch.setattr(vmaf_vulkan, "VulkanScorer", Scorer)
    attempt = vmaf_cuda.GpuAttempt(64, 48, 10, {"vmaf": "vmaf_v0.6.1"}, 2, "vulkan", 1)
    attempt.finish(False)
    assert made == {"args": (64, 48, 10, {"vmaf": "vmaf_v0.6.1"}, 2), "kwargs": {"device": 1}}


def test_a_vulkan_score_records_that_vulkan_calculated_it():
    from vmaf_app.core.models import FrameScores

    frames = FrameScores(np.arange(2, dtype=np.int32), np.zeros(2), vmaf=np.array([90.0, 91.0], dtype=np.float32),
                         psnr=np.array([40.0, 41.0], dtype=np.float32))
    results = vr._metric_results_for_current_run(frames, "version=vmaf_v0.6.1", gpu_keys={"vmaf"},
                                                 gpu_backend="vulkan")
    vmaf, psnr = results.get("vmaf").provenance, results.get("psnr").provenance
    assert (vmaf.implementation, vmaf.implementation_version, vmaf.compute_backend) == (
        "libvmaf/vulkan", vmaf_vulkan.LIBRARY_BUILD, "gpu")
    assert psnr.implementation != "libvmaf/vulkan" and psnr.compute_backend != "gpu"
    cuda = vr._metric_results_for_current_run(frames, "version=vmaf_v0.6.1", gpu_keys={"vmaf"}).get("vmaf")
    assert cuda.provenance.implementation == "libvmaf/cuda"
    # The same request identity: a saved score answers a run on either.
    assert (cuda.provenance.implementation_compatibility_id
            == results.get("vmaf").provenance.implementation_compatibility_id)


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


def test_the_best_device_is_a_discrete_gpu_before_an_integrated_one():
    device = vmaf_vulkan.VulkanDevice
    found = [device(0, "Intel", 0x8086, 1, True, False), device(1, "Radeon", 0x1002, 2, True, True),
             device(2, "llvmpipe", 0x10005, 4, True, True), device(3, "old", 0x10DE, 2, False, False)]
    assert vmaf_vulkan.best_device(found).name == "Radeon"
    assert vmaf_vulkan.best_device(found[:1]).name == "Intel"
    assert vmaf_vulkan.best_device(found[2:]) is None  # a CPU rasteriser, and a GPU without 64-bit integers


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
def test_the_probe_accepts_this_pc(monkeypatch):
    monkeypatch.delenv(vmaf_vulkan.DEVICE_VARIABLE, raising=False)
    available, device, text = vmaf_vulkan.probe()
    assert available and device == vmaf_vulkan.best_device().index and text.startswith("Vulkan on ")
    monkeypatch.setattr(vmaf_vulkan, "_PROBE_SUMS", ("0" * 64, "0" * 64))  # as a driver that calculates wrongly
    available, device, text = vmaf_vulkan.probe()
    assert not available and device is None and "calculates VMAF wrongly" in text


@needs_gpu
def test_the_scorer_returns_what_the_cuda_scorer_returns():
    reference, distorted = vmaf_vulkan.probe_frames(8, 5)
    scorer = vmaf_vulkan.VulkanScorer(*vmaf_vulkan._PROBE_SIZE, 8, MODELS, n_subsample=2)
    try:
        assert scorer.frame_bytes == len(reference[0])
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        frames, scores = scorer.finish()
    finally:
        scorer.close()
    assert frames.tolist() == [0, 2, 4] and set(scores) == {"vmaf", "vmaf_neg"}
    for values in scores.values():
        assert values.shape == (3,) and np.all((values > 20) & (values <= 100))
        assert np.array_equal(values, np.round(values, 6))  # six decimals, as libvmaf's log
    assert np.all(scores["vmaf_neg"] <= scores["vmaf"])  # NEG never rewards a gain
    assert scores["vmaf_neg"][1] < scores["vmaf"][1] - 0.01 or scores["vmaf_neg"][0] < scores["vmaf"][0]


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


@needs_gpu
def test_unsupported_sizes_and_gpus_are_refused_with_a_reason():
    with pytest.raises(vmaf_vulkan.VmafVulkanError, match="picture size"):
        vmaf_vulkan.VulkanScorer(16, 16, 8, {})
    with pytest.raises(vmaf_vulkan.VmafVulkanError, match="bit depth"):
        vmaf_vulkan.VulkanScorer(64, 64, 7, {})
    with pytest.raises(vmaf_vulkan.VmafVulkanError, match="no such GPU"):
        vmaf_vulkan.VulkanScorer(64, 64, 8, {}, device=99)
    scorer = vmaf_vulkan.VulkanScorer(64, 64, 10, {})
    try:  # a frame cut short must not be read past its end
        with pytest.raises(vmaf_vulkan.VmafVulkanError, match="shorter than its luma plane"):
            scorer.add(bytearray(64 * 64 * 2), bytearray(64 * 64 * 2 - 1))
        scorer.add(bytearray(64 * 64 * 2), bytearray(64 * 64 * 2))  # the luma alone is enough
    finally:
        scorer.close()


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


class _ExportingLibrary:
    """vmaf_vulkan.dll as SharedLumas finds it: two shared buffers to export
    (each a real handle, which SharedLumas closes), and vv_shared_device or
    not (libvmaf-fast 3.2.0-fast.1 has none)."""

    def __init__(self, names_device: bool) -> None:
        if names_device:
            self.vv_shared_device = self._shared_device

    def vv_export(self, context, slot, handle, size):
        if slot >= 2:
            return -3
        handle._obj.value = ctypes.windll.kernel32.CreateEventW(None, 0, 0, None)
        size._obj.value = 1 << 20
        return 0

    @staticmethod
    def _shared_device(context, device, driver, memory_type):
        ctypes.memmove(device, b"d" * 16, 16)
        ctypes.memmove(driver, b"r" * 16, 16)
        memory_type._obj.value = 3
        return 0

    def vv_error(self):
        return b""


class _ImportingStream:
    def __init__(self) -> None:
        self.imports, self.unimported = [], []

    def import_memory(self, handle, size, exporter=None):
        self.imports.append((size, exporter))
        return (len(self.imports) << 40, len(self.imports))

    def unimport(self, memory):
        self.unimported.append(memory)


def test_shared_lumas_names_vulkans_gpu_and_driver_to_the_importing_decoder():
    """AMD's decoder imports Vulkan's memory into a Vulkan device of its own,
    which it may do only from the same GPU and driver, into the same memory
    type: vv_shared_device names them, and every import is told."""
    stream = _ImportingStream()
    shared = vmaf_vulkan.SharedLumas(_ExportingLibrary(names_device=True), ctypes.c_void_p(), stream)
    exporter = vmaf_vulkan.SharedDevice(b"d" * 16, b"r" * 16, 3)
    assert stream.imports == [(1 << 20, exporter), (1 << 20, exporter)]
    shared.close()
    assert stream.unimported == [1, 2]


def test_shared_lumas_names_nothing_with_an_engine_that_cannot():
    """With libvmaf-fast 3.2.0-fast.1's engine NVIDIA's decoder imports as
    before, and AMD's then imports nothing: its frames go through system
    memory (GpuFrameStream.import_memory)."""
    stream = _ImportingStream()
    vmaf_vulkan.SharedLumas(_ExportingLibrary(names_device=False), ctypes.c_void_p(), stream).close()
    assert stream.imports == [(1 << 20, None), (1 << 20, None)]
