"""The GPU VMAF libraries are libvmaf-fast's (github.com/4KVCD/libvmaf-fast),
committed in vmaf_app/tools: a release, installed by
scripts/fetch_libvmaf_fast.ps1, or while the app is developed a build of the
fork's latest commit, by scripts/build_libvmaf_fast_local.ps1. Either records
what it installed in libvmaf/libvmaf-fast.json: what the libraries are, what
the app says its scores come from and, for a release, the one the fetch
script pins must agree."""
import ctypes
import json
import re
from pathlib import Path

from vmaf_app.core import vmaf_cuda, vmaf_v1_gpu, vmaf_vulkan

ROOT = Path(__file__).resolve().parents[1]


def _record() -> dict:
    return json.loads(vmaf_cuda.BUILD_RECORD.read_text(encoding="utf-8"))


def test_the_bundled_build_is_the_one_scores_are_recorded_with():
    record = _record()
    assert record["version"] == vmaf_cuda.LIBVMAF_FAST_VERSION
    for build in (vmaf_cuda.LIBRARY_BUILD, vmaf_cuda.CPU_BUILD, vmaf_vulkan.LIBRARY_BUILD, vmaf_v1_gpu.LIBRARY_BUILD):
        assert build.startswith(f"libvmaf-fast {record['version']} (")
    assert re.fullmatch(r"[0-9a-f]{40}", record["commit"])
    script = (ROOT / "scripts" / "fetch_libvmaf_fast.ps1").read_text(encoding="utf-8")
    assert re.search(r"\[string\]\$Sha256 = '[0-9a-f]{64}'", script)  # pinned, not trusted on download
    if record["release"]:
        assert record["version"] == re.search(r"\[string\]\$Version = '([^']+)'", script).group(1)
    else:
        # git describe of the commit: the release it follows (v1 and on, or
        # the one before them, 3.2.0-fast.1), how far, which commit.
        follows, _, commit = re.fullmatch(r"(.+)-(\d+)-g([0-9a-f]+)", record["version"]).groups()
        assert re.fullmatch(r"v\d+|\d+\.\d+\.\d+-fast\.\d+", follows)
        assert record["commit"].startswith(commit)


def test_both_libraries_are_the_recorded_commit():
    """Each reports the commit it was built from: one left from another build
    would score without the scores saying so (release/v1.5 had a Vulkan
    engine of a31318b9 beside the release's libvmaf, both named 3.2.0-fast.1)."""
    commit = _record()["commit"]
    libvmaf = ctypes.CDLL(str(vmaf_cuda.LIBRARY_PATH))
    libvmaf.vmaf_version.restype = ctypes.c_char_p
    vulkan = ctypes.CDLL(str(vmaf_vulkan.LIBRARY_PATH))
    vulkan.vv_version.restype = ctypes.c_char_p
    for reported in (libvmaf.vmaf_version().decode(), vulkan.vv_version().decode()):
        assert reported and commit.startswith(reported), reported


def test_both_installers_write_the_record_the_app_reads():
    for name, release in (("fetch_libvmaf_fast.ps1", "true"), ("build_libvmaf_fast_local.ps1", "false")):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert f"'libvmaf/{vmaf_cuda.BUILD_RECORD.name}'" in script, name
        assert f'`"release`": {release}' in script, name
    assert vmaf_cuda.BUILD_RECORD.parent == vmaf_cuda.LIBRARY_PATH.parent  # packaged with the library's folder


def test_without_a_record_the_version_is_unknown(tmp_path):
    assert vmaf_cuda._bundled_version(tmp_path / "missing.json") == "unknown"
    for text in ("{", "[]", '{"commit": "a1af96ff"}'):
        (tmp_path / "bad.json").write_text(text, encoding="utf-8")
        assert vmaf_cuda._bundled_version(tmp_path / "bad.json") == "unknown", text


def test_the_build_and_its_licences_are_bundled():
    tools = ROOT / "vmaf_app" / "tools"
    for name in ("libvmaf/libvmaf.dll", "libvmaf/licenses/LICENSE.libvmaf.txt",
                 "libvmaf/licenses/LICENSE.pthreads4w.txt", "libvmaf/licenses/LICENSE.nv-codec-headers.txt",
                 "libvmaf/licenses/LICENSE.xpsnr.txt",  # libvmaf-fast's XPSNR, FFmpeg's filter ported: LGPL
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
