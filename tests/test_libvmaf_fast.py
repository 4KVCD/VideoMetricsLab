"""The GPU VMAF libraries are libvmaf-fast's release (github.com/4KVCD/
libvmaf-fast), installed by scripts/fetch_libvmaf_fast.ps1 and committed:
the release the script installs, the one the app says its scores come from
and what is in vmaf_app/tools must agree."""
import ctypes
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


class _FakeFunction:
    restype = argtypes = None


class _FakeLibrary:
    """A libvmaf with or without vmaf_picture_convert (the newer picture layout)."""

    def __init__(self, newer: bool):
        self._newer = newer

    def __getattr__(self, name):  # a function, kept (as ctypes keeps them) for its argtypes
        if name == "vmaf_picture_convert" and not self._newer:
            raise AttributeError(name)
        function = _FakeFunction()
        setattr(self, name, function)
        return function


def test_pictures_take_the_layout_of_the_libvmaf_loaded(monkeypatch):
    """libvmaf-fast after v3.2.0-fast.1 has a VmafColor between data and ref
    (upstream's vmaf_picture_convert): the app's pictures must match the
    library they go to, or libvmaf writes past them."""
    old, new = vmaf_cuda._PlainPicture, vmaf_cuda._ColorPicture
    assert ctypes.sizeof(new) == ctypes.sizeof(old) + 16
    assert new.ref.offset == old.ref.offset + 16 and new.data.offset == old.data.offset
    for newer, expected in ((False, old), (True, new)):
        monkeypatch.setattr(vmaf_cuda, "_library", None)
        monkeypatch.setattr(vmaf_cuda, "_Picture", new if newer else old)  # either may be left from before
        monkeypatch.setattr(vmaf_cuda.ctypes, "CDLL", lambda path, newer=newer: _FakeLibrary(newer))
        vmaf_cuda._load()
        assert vmaf_cuda._Picture is expected
        assert vmaf_cuda._library.vmaf_read_pictures.argtypes[1]._type_ is expected


def test_the_bundled_libvmaf_has_the_layout_its_version_has():
    bundled = ROOT / "vmaf_app" / "tools" / "libvmaf" / "libvmaf.dll"
    lib = ctypes.CDLL(str(bundled))
    assert hasattr(lib, "vmaf_picture_convert") == (vmaf_cuda.LIBVMAF_FAST_VERSION != "3.2.0-fast.1")
