"""The GPU VMAF libraries are libvmaf-fast's release (github.com/4KVCD/
libvmaf-fast), installed by scripts/fetch_libvmaf_fast.ps1 and committed:
the release the script installs, the one the app says its scores come from
and what is in vmaf_app/tools must agree."""
import re
from pathlib import Path

from vmaf_app.core import vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan

ROOT = Path(__file__).resolve().parents[1]


def test_the_fetched_release_is_the_one_scores_are_recorded_with():
    script = (ROOT / "scripts" / "fetch_libvmaf_fast.ps1").read_text(encoding="utf-8")
    version = re.search(r"\[string\]\$Version = '([^']+)'", script).group(1)
    assert version == vmaf_cuda.LIBVMAF_FAST_VERSION
    assert re.search(r"\[string\]\$Sha256 = '[0-9a-f]{64}'", script)  # pinned, not trusted on download
    for build in (vmaf_cuda.LIBRARY_BUILD, vmaf_vulkan.LIBRARY_BUILD, vmaf_v1_gpu.LIBRARY_BUILD):
        assert f"libvmaf-fast {version}" in build


def test_the_release_and_its_licences_are_bundled():
    tools = ROOT / "vmaf_app" / "tools"
    for name in ("libvmaf/libvmaf.dll", "libvmaf/licenses/LICENSE.libvmaf.txt",
                 "libvmaf/licenses/LICENSE.pthreads4w.txt", "libvmaf/licenses/LICENSE.nv-codec-headers.txt",
                 "vmaf_vulkan/vmaf_vulkan.dll", "vmaf_vulkan/licenses/LICENSE.libvmaf.txt"):
        assert (tools / name).is_file(), name
    assert tools / "libvmaf" / "libvmaf.dll" == vmaf_cuda.LIBRARY_PATH
    assert tools / "vmaf_vulkan" / "vmaf_vulkan.dll" == vmaf_vulkan.LIBRARY_PATH
