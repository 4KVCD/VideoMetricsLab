from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import FrameScore, ResampleTarget, VideoInfo, VmafOptions, VmafRunResult
from vmaf_app.ui import worker as worker_module
from vmaf_app.ui.worker import VmafJob, VmafWorker
from vmaf_app.core.vmaf_runner import Cancelled


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _info(name: str) -> VideoInfo:
    return VideoInfo(path=Path(name), width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=150, codec_name="h264")


def _fake_result(name: str) -> VmafRunResult:
    info = _info(name)
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    return VmafRunResult(
        source=Path("source.mp4"), distorted=Path(name), frames=frames, fps=30.0, model="version=vmaf_v0.6.1",
        source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )


def test_worker_dispatches_to_run_vmaf_for_a_normal_job(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(worker_module, "run_vmaf", lambda *a, **kw: calls.append("run_vmaf") or _fake_result("d.mp4"))
    monkeypatch.setattr(worker_module, "run_resample_test", lambda *a, **kw: calls.append("run_resample_test"))

    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d")
    w = VmafWorker([job])
    w.run()  # run synchronously in-test rather than as a real thread

    assert calls == ["run_vmaf"]


def test_worker_dispatches_to_run_resample_test_when_resample_target_is_set(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(worker_module, "run_vmaf", lambda *a, **kw: calls.append("run_vmaf"))
    monkeypatch.setattr(
        worker_module, "run_resample_test",
        lambda *a, **kw: calls.append("run_resample_test") or _fake_result("s [downscale-1080p-upscale].mp4"),
    )

    options = VmafOptions(resample_test=ResampleTarget(width=1920, label="1080p"))
    job = VmafJob(_info("s.mp4"), _info("s.mp4"), options, label="1080p test")
    w = VmafWorker([job])
    w.run()

    assert calls == ["run_resample_test"]


def test_worker_reports_cancellation_as_a_distinct_terminal_state(qapp, monkeypatch):
    def cancelled_run(*args, **kwargs):
        raise Cancelled("cancelled")

    monkeypatch.setattr(worker_module, "run_vmaf", cancelled_run)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d")
    worker = VmafWorker([job])
    reported = []
    worker.cancelled.connect(lambda: reported.append(True))

    worker.run()

    assert reported == [True]
