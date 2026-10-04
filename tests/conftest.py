"""Shared test fixtures.

The important one is `isolate_user_state`: without it the suite reads and
writes the *user's real* settings file and results cache. That is not just
untidy -- a test that ticks "compute PSNR by default" persisted it, which
then leaked into every later test in the run AND into the installed app.
It also keeps the suite off the machine's power state.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Set before Qt is imported anywhere: the startup ffmpeg check opens a modal
# dialog unless it sees this, and a modal in a test blocks the run forever
# rather than failing it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vmaf_app.core import crop_detect, result_cache
from vmaf_app.core.app_paths import DATA_DIR_NAME
from vmaf_app.core.settings import Settings


def pytest_addoption(parser):
    parser.addoption(
        "--packaging", action="store_true",
        help="also run the packaging checks against the installed wheels (CI and releases do)",
    )


def pytest_collection_modifyitems(config, items):
    """The packaging checks scan every DLL of the installed PySide6 and
    GStreamer wheels: 12 of the suite's 72 seconds, for code that changes
    only when the build does. Skipped unless asked for with --packaging."""
    if config.getoption("--packaging"):
        return
    skip = pytest.mark.skip(reason="packaging check; run with --packaging")
    for item in items:
        if "packaging" in item.keywords:
            item.add_marker(skip)


def _real_user_cache_dir() -> Path:
    """Where the installed app keeps results. Nothing in the suite may touch
    it, so it is resolved once here to be recognised and refused."""
    # Compute this directly rather than invoking application path helpers.
    return Path.home() / DATA_DIR_NAME / "results_cache"


@pytest.fixture(autouse=True)
def isolate_user_state(tmp_path, monkeypatch):
    """Points settings and the results cache at a per-test temp folder."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(Settings, "path", staticmethod(lambda: settings_file))

    cache = tmp_path / "results_cache"
    cache.mkdir()

    # Written as a real settings file rather than only set through the
    # override, because MainWindow's constructor re-applies whatever its
    # loaded Settings say. With cache_dir blank that is None -- "use the
    # platform folder" -- so simply building a window silently un-isolated
    # the suite and pointed it back at the user's own cache.
    #
    # parallel_jobs is pinned because its default depends on the core count
    # of the machine running the suite (see default_parallel_jobs).
    #
    # The metric ticks are pinned to VMAF, PSNR, SSIM and XPSNR, which most
    # window tests were written around. The shipped defaults tick every
    # metric; test_a_fresh_install_ticks_every_metric covers those.
    settings_file.write_text(
        json.dumps({
            "cache_dir": str(cache),
            "parallel_jobs": 1,
            "default_compute_vmaf_neg": False,
            "default_compute_ssimulacra2": False,
            "default_compute_butteraugli": False,
            "default_compute_cvvdp": False,
            "default_compute_vmaf_v1": False,
        }),
        encoding="utf-8",
    )

    real_override = result_cache.set_cache_dir_override
    real_cache = _real_user_cache_dir()

    def guarded_override(directory):
        # A backstop for the same leak arriving by another route (a test
        # that saves a fresh Settings, say). Failing loudly is the point:
        # the previous symptom was a test quietly reading and writing real
        # user data, which stays invisible until it corrupts something.
        if directory is None:
            real_override(cache)
            return
        if Path(directory) == real_cache:
            raise AssertionError(
                f"a test pointed the results cache at the user's real folder "
                f"({real_cache}); it must stay inside tmp_path"
            )
        real_override(Path(directory))

    monkeypatch.setattr(result_cache, "set_cache_dir_override", guarded_override)
    result_cache.set_cache_dir_override(cache)
    # A run holds a Windows keep-awake request (vmaf_app.core.power): from a
    # test, a real request on the machine running the suite -- 41 of them in
    # the window tests. Stubbed where Windows is called, so no route reaches
    # it: a window test starting a run, or the power module called directly.
    if sys.platform == "win32":
        import ctypes

        monkeypatch.setattr(ctypes.windll.kernel32, "SetThreadExecutionState", lambda flags: 0x80000000)
    # VMAF on the GPU is off unless a test asks for it: GitHub's runner has
    # no GPU, so the suite runs as there wherever it runs.
    from vmaf_app.core import vmaf_cuda

    monkeypatch.setattr(vmaf_cuda, "_probed", (False, "off in tests"))
    monkeypatch.setattr(vmaf_cuda, "_probed_at", float("inf"))  # never retried (forget_failed_probe)
    # Likewise decoding in the scoring process with NVIDIA's decoder
    # (gpu_frames): FFmpeg decodes unless a test asks for it, so a test
    # faking NVIDIA decode does not reach a real GPU where there is one.
    from vmaf_app.core import gpu_frames, perceptual_vship, vmaf_runner

    def ffmpeg_decodes(*_args, **_kwargs):
        raise gpu_frames.GpuDecodeUnavailableError("off in tests")

    monkeypatch.setattr(perceptual_vship, "_native_decoder", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vmaf_runner, "_score_decoded_on_gpu", ffmpeg_decodes)
    # Detected black bars are remembered per file for the life of the
    # process; a test's answer must not leak into the next one's.
    crop_detect.clear_cache()
    yield
    real_override(None)
    crop_detect.clear_cache()
