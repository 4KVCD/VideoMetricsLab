from io import BytesIO
from types import SimpleNamespace

import pytest

from vmaf_app.core.frame_extract import PreviewColorSettings
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.ui import playback_worker
from vmaf_app.ui.native_playback_pool import _StopNative


class FakeProcess:
    pid = 123

    def __init__(self, output, code, error=b""):
        self.stdout, self.stderr = BytesIO(output), BytesIO(error)
        self.returncode = code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def setup_worker(monkeypatch, attempts):
    from vmaf_app.core import video_playback

    monkeypatch.setattr(video_playback, "playback_dimensions", lambda *args: (1, 1))
    commands = []

    def command(items, frame, *args, **kwargs):
        commands.append((kwargs["processing"], frame))
        return ["fake"]

    monkeypatch.setattr(playback_worker, "build_video_series_command", command)
    processes = iter(attempts)
    monkeypatch.setattr(playback_worker.proc, "popen", lambda *args, **kwargs: next(processes))
    worker = playback_worker.StreamDecodeWorker(None, "source", 100, PreviewColorSettings(),
                                               HwAccelPlan(source="cuda"), None)
    worker._handle = SimpleNamespace(attach=lambda pid: None, detach=lambda: None, terminate=lambda: None)
    frames, failures = [], []
    worker._put = lambda number, payload: frames.append((number, payload)) or True
    worker.failed.connect(failures.append)
    return worker, commands, frames, failures


def test_device_loss_after_frames_resumes_without_replaying(monkeypatch):
    worker, commands, frames, failures = setup_worker(monkeypatch, [
        FakeProcess(b"aaaa" + b"bb", 1, b"VK_ERROR_DEVICE_LOST"),
        FakeProcess(b"ccccdddd", 0),
    ])
    worker.run()
    assert commands == [("vulkan", 100), ("transfer", 101)]
    assert frames == [(100, b"aaaa"), (101, b"cccc"), (102, b"dddd")]
    assert not failures
    assert worker.ended and not worker.error
    assert worker.attempt_errors == ["VK_ERROR_DEVICE_LOST"]


def test_midstream_failures_exhaust_bounded_ladder(monkeypatch):
    worker, commands, frames, failures = setup_worker(monkeypatch, [
        FakeProcess(b"aaaa", 1, b"device lost") for _ in range(4)
    ])
    worker.run()
    assert commands == [("vulkan", 100), ("transfer", 101), ("software", 102), ("cpu", 103)]
    assert [n for n, _ in frames] == [100, 101, 102, 103]
    assert failures == ["device lost"]
    assert not worker.ended


def test_clean_eof_does_not_retry(monkeypatch):
    worker, commands, _, failures = setup_worker(monkeypatch, [FakeProcess(b"aaaa", 0)])
    worker.run()
    assert commands == [("vulkan", 100)]
    assert worker.ended and not failures


def test_cancelled_worker_does_not_retry_or_emit_failure(monkeypatch):
    worker, commands, _, failures = setup_worker(monkeypatch, [])
    worker.cancel()
    worker.run()
    assert not commands and not failures


def test_native_teardown_accepts_repeated_cancel_without_reentering_stop():
    stops = []
    worker = _StopNative(SimpleNamespace(stop=lambda: stops.append(True)), None)
    worker.cancel()
    worker.cancel()
    assert not stops
    worker.run()
    worker.cancel()
    assert stops == [True]


def test_extraction_cancellation_does_not_cancel_playback():
    from vmaf_app.ui.frame_compare_panel import FrameComparePanel

    calls = []
    extraction = SimpleNamespace(isRunning=lambda: True, cancel=lambda: calls.append("extract"))
    panel = SimpleNamespace(_workers=[extraction], live_workers=lambda: pytest.fail("mixed ownership"))
    FrameComparePanel._cancel_workers(panel)
    assert calls == ["extract"]
