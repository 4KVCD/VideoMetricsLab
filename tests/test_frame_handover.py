"""Pictures handed from NVIDIA's decoder to the scorers without a CPU copy
(gpu_frames' pin, download_planes and import_memory; vmaf_vulkan.SharedLumas;
vmaf_v1_gpu's add_decoded): the same pictures, and the same scores, as
through system memory. On a PC with an NVIDIA GPU; GitHub's runner has none.
And AMD's pictures read where Windows' decoder left them (vv_pictures): the
same scores as from the pictures copied, on a PC with an AMD GPU."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import gpu_frames, vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan
from vmaf_app.core.ffmpeg_locate import ffmpeg_path
from vmaf_app.core.ffprobe import probe_video

MODELS = Path(vmaf_v1_gpu.__file__).resolve().parents[1] / "models"
V1 = f"path={MODELS / 'vmaf_v1.0.16' / 'vmaf_v1.0.16_3d0h.json'}"
W, H = 640, 360


def _clip(path: Path, pix_fmt: str, quality: int, codec: str = "hevc") -> Path:
    encoder = ["-c:v", "libx264", "-qp", str(quality)] if codec == "h264" else         ["-c:v", "libx265", "-x265-params", f"qp={quality}:log-level=error"]
    subprocess.run([ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s={W}x{H}:r=24000/1001:d=1", "-pix_fmt", pix_fmt, "-preset", "ultrafast",
                    *encoder, str(path)], check=True)
    return path


def _nvidia_decodes(info, bits: int) -> bool:
    try:
        if not gpu_frames.LIBRARIES["nvidia"].is_file() or not vmaf_vulkan.LIBRARY_PATH.is_file():
            return False
        plan = gpu_frames.plan_decode(info, None, shift=6 if bits > 8 else 0)
        return gpu_frames.decoder_supports(0, plan, "nvidia")[0] and any(
            device.vendor == 0x10DE and device.usable for device in vmaf_vulkan.devices())
    except (OSError, vmaf_cuda.VmafGpuError, gpu_frames.GpuDecodeUnavailableError):
        return False


@pytest.fixture(params=[("yuv420p", 8), ("yuv420p10le", 10)], ids=["8-bit", "10-bit"])
def pair(request, tmp_path):
    pix_fmt, bits = request.param
    source = probe_video(_clip(tmp_path / "source.mkv", pix_fmt, 4))
    distorted = probe_video(_clip(tmp_path / "distorted.mkv", pix_fmt, 38))
    if not _nvidia_decodes(source, bits):
        pytest.skip("no NVIDIA GPU that decodes this and shares memory with Vulkan")
    return source, distorted, bits


def test_planes_downloaded_into_pinned_pitched_memory_are_the_packed_picture(pair):
    """A libvmaf picture's planes: rows further apart than their samples,
    page-locked, each written by the GPU."""
    source, _distorted, bits = pair
    plan = gpu_frames.plan_decode(source, None, shift=6 if bits > 8 else 0)
    sample = 2 if bits > 8 else 1
    rows = (H, H // 2, H // 2)
    widths = (W * sample, W // 2 * sample, W // 2 * sample)
    pitches = (W * sample + 64, W // 2 * sample + 32, W // 2 * sample + 32)
    packed = np.empty(plan.frame_bytes, dtype=np.uint8)
    planes = [np.full(rows[i] * pitches[i], 0xEE, dtype=np.uint8) for i in range(3)]
    stream = gpu_frames.GpuFrameStream(source, plan, backend="nvidia")
    try:
        assert all(stream.pin(plane.ctypes.data, plane.nbytes) for plane in planes)
        stream.start()
        count = 0
        while True:
            try:
                item = stream.next(1000)
            except TimeoutError:
                continue
            if item is None:
                break
            stream.download(item[0], packed.ctypes.data)
            stream.download_planes(item[0], tuple(plane.ctypes.data for plane in planes), pitches)
            offset = 0
            for index in range(3):
                got = planes[index].reshape(rows[index], pitches[index])
                expected = packed[offset:offset + rows[index] * widths[index]].reshape(rows[index], widths[index])
                assert np.array_equal(got[:, :widths[index]], expected), (count, index)
                assert np.all(got[:, widths[index]:] == 0xEE)  # nothing written between the rows
                offset += rows[index] * widths[index]
            # A plane left out is left alone.
            planes[0][:] = 0x11
            stream.download_planes(item[0], (None, planes[1].ctypes.data, planes[2].ctypes.data), pitches)
            assert np.all(planes[0] == 0x11)
            planes[0][:] = 0xEE
            stream.release(item[0])
            count += 1
        assert count > 20
        for plane in planes:
            stream.unpin(plane.ctypes.data)
    finally:
        stream.close()


def _without_sharing(monkeypatch):
    """As a driver whose CUDA cannot take Vulkan's memory: the frames go
    through system memory, as before."""
    monkeypatch.setattr(gpu_frames.GpuFrameStream, "import_memory", lambda self, handle, size, exporter=None: None)


def test_vmaf_v1_scores_the_same_from_pictures_kept_on_the_gpu(pair, subtests):
    def check(models, backend, monkeypatch):
        source, distorted, bits = pair
        arguments = dict(width=W, height=H, bit_depth=bits, models=models, n_subsample=1, duration_limit=None,
                         total_frames=0, backend=backend, decoder="nvidia")
        taken = []
        real = vmaf_v1_gpu.MultiScorer.add_decoded
        monkeypatch.setattr(vmaf_v1_gpu.MultiScorer, "add_decoded",
                            lambda self, *args: (taken.append(1), real(self, *args))[1])
        frames, scores = vmaf_v1_gpu.score_decoded(source, distorted, **arguments)
        assert len(taken) == len(frames) > 20  # the way without a copy was the one taken
        _without_sharing(monkeypatch)
        old_frames, old_scores = vmaf_v1_gpu.score_decoded(source, distorted, **arguments)
        assert len(taken) == len(frames)  # and here it was not
        assert np.array_equal(frames, old_frames) and set(scores) == set(models)
        for key in models:
            assert np.array_equal(scores[key], old_scores[key]), key
            assert np.all((scores[key] > 0) & (scores[key] <= 100)) and len(set(scores[key].tolist())) > 1

    for models, backend in [
        ({"vmaf_v1": V1}, "cuda"),
        ({"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg", "vmaf_v1": V1}, "cuda"),
        ({"vmaf": "vmaf_v0.6.1", "vmaf_v1": V1}, "vulkan"),
    ]:
        with subtests.test(models=models, backend=backend), pytest.MonkeyPatch.context() as case_patch:
            check(models, backend, case_patch)


def test_vulkans_vmaf_scores_the_same_from_lumas_copied_on_the_gpu(pair, monkeypatch):
    source, distorted, bits = pair
    arguments = dict(width=W, height=H, bit_depth=bits, models={"vmaf": "vmaf_v0.6.1", "vmaf_neg": "vmaf_v0.6.1neg"},
                     n_subsample=2, duration_limit=None, total_frames=0, decoder="nvidia")
    taken = []
    real = vmaf_vulkan.VulkanScorer.add_shared
    monkeypatch.setattr(vmaf_vulkan.VulkanScorer, "add_shared",
                        lambda self, *args: (taken.append(1), real(self, *args))[1])
    frames, scores = vmaf_vulkan.score_decoded(source, distorted, **arguments)
    assert taken and frames.tolist()[:3] == [0, 2, 4]
    count = len(taken)
    _without_sharing(monkeypatch)
    old_frames, old_scores = vmaf_vulkan.score_decoded(source, distorted, **arguments)
    assert len(taken) == count
    assert np.array_equal(frames, old_frames)
    for key in scores:
        assert np.array_equal(scores[key], old_scores[key]), key


# ------------------------------------- AMD's: the pictures read where they are

def _amd_gives_textures(info, bits: int) -> bool:
    try:
        if not gpu_frames.hands_over_textures("amd") or not vmaf_vulkan.LIBRARY_PATH.is_file():
            return False
        plan = gpu_frames.plan_decode(info, None, shift=6 if bits > 8 else 0)
        return gpu_frames.decoder_supports(0, plan, "amd")[0] and any(
            device.vendor == 0x1002 and device.usable for device in vmaf_vulkan.devices())
    except (OSError, vmaf_cuda.VmafGpuError, gpu_frames.GpuDecodeUnavailableError):
        return False


def test_vmaf_v1_scores_the_same_from_amds_textures_as_from_copies(tmp_path_factory, subtests):
    """Vulkan VMAF reading the pictures where Windows' decoder left them
    (vv_pictures, their slots given back once the GPU has read them) scores
    what it scores from the pictures copied into its memory: HEVC's, and
    H.264's (layers of a texture array, copied into textures of their own)."""
    def check(models, n_subsample, codec, tmp_path, monkeypatch):
        pix_fmt, bits = ("yuv420p", 8) if codec == "h264" else ("yuv420p10le", 10)
        source = probe_video(_clip(tmp_path / "source.mkv", pix_fmt, 4, codec))
        distorted = probe_video(_clip(tmp_path / "distorted.mkv", pix_fmt, 38, codec))
        if not _amd_gives_textures(source, bits):
            pytest.skip("no AMD GPU whose Windows decoder gives its pictures as textures")
        arguments = dict(width=W, height=H, bit_depth=bits, models=models, n_subsample=n_subsample, duration_limit=None,
                         total_frames=0, backend="vulkan", decoder="amd")
        used = []
        real = vmaf_v1_gpu.V1Scorer.add_decoded
        monkeypatch.setattr(vmaf_v1_gpu.V1Scorer, "add_decoded",
                            lambda self, *args: (used.append(self.pictures), real(self, *args))[1])
        frames, scores = vmaf_v1_gpu.score_decoded(source, distorted, **arguments)
        assert used and all(used)  # read where they were
        monkeypatch.setattr(gpu_frames, "hands_over_textures", lambda backend: False)
        used.clear()
        old_frames, old_scores = vmaf_v1_gpu.score_decoded(source, distorted, **arguments)
        assert used and not any(used)  # copied
        assert np.array_equal(frames, old_frames) and set(scores) == set(models)
        for key in models:
            assert np.array_equal(scores[key], old_scores[key]), key
            assert np.all((scores[key] > 0) & (scores[key] <= 100)) and len(set(scores[key].tolist())) > 1

    for models, n_subsample, codec in [({"vmaf_v1": V1}, 1, "hevc"), ({"vmaf_v1": V1}, 2, "hevc"),
                                                                  ({"vmaf": "vmaf_v0.6.1", "vmaf_v1": V1}, 1, "hevc"),
                                                                  ({"vmaf_v1": V1}, 1, "h264")]:
        with subtests.test(models=models, n_subsample=n_subsample, codec=codec), pytest.MonkeyPatch.context() as case_patch:
            check(models, n_subsample, codec, tmp_path_factory.mktemp("case"), case_patch)
