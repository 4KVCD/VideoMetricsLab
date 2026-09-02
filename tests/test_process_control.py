import psutil
import pytest

from vmaf_app.core.process_control import ProcessHandle


class FakeProcess:
    """Stands in for psutil.Process so tests don't depend on real OS
    process suspension, which is awkward to assert on deterministically."""

    instances: dict[int, "FakeProcess"] = {}

    def __init__(self, pid: int):
        if pid in FakeProcess.instances:
            return  # __init__ still runs on a cached __new__ return; don't reset .calls
        FakeProcess.instances[pid] = self
        self.pid = pid
        self.calls: list[str] = []

    def __new__(cls, pid: int):
        if pid in cls.instances:
            return cls.instances[pid]
        return super().__new__(cls)

    def suspend(self):
        self.calls.append("suspend")

    def resume(self):
        self.calls.append("resume")

    def terminate(self):
        self.calls.append("terminate")


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeProcess.instances.clear()
    yield
    FakeProcess.instances.clear()


@pytest.fixture
def patched_psutil(monkeypatch):
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    return FakeProcess


def test_pause_then_attach_applies_pause_to_new_process(patched_psutil):
    handle = ProcessHandle()
    handle.pause()  # requested before any process exists yet
    handle.attach(1234)

    assert FakeProcess.instances[1234].calls == ["suspend"]
    assert handle.is_pause_requested is True


def test_attach_then_pause_and_resume(patched_psutil):
    handle = ProcessHandle()
    handle.attach(1234)
    handle.pause()
    handle.resume()

    assert FakeProcess.instances[1234].calls == ["suspend", "resume"]
    assert handle.is_pause_requested is False


def test_terminate_works_even_while_paused(patched_psutil):
    handle = ProcessHandle()
    handle.attach(1234)
    handle.pause()
    handle.terminate()

    assert FakeProcess.instances[1234].calls == ["suspend", "terminate"]


def test_pause_state_carries_over_to_next_attached_process(patched_psutil):
    # Models the GPU-decode-failure retry: the first process dies, a new one
    # starts, and a pause requested mid-run should still apply to it.
    handle = ProcessHandle()
    handle.attach(1111)
    handle.pause()
    handle.detach()
    handle.attach(2222)

    assert FakeProcess.instances[2222].calls == ["suspend"]


def test_detach_without_pause_does_nothing_to_next_process(patched_psutil):
    handle = ProcessHandle()
    handle.attach(1111)
    handle.detach()
    handle.attach(2222)

    assert 2222 not in FakeProcess.instances


def test_pause_and_terminate_are_no_ops_before_any_process_attached(patched_psutil):
    handle = ProcessHandle()
    handle.pause()
    handle.terminate()  # no pid attached -- must not raise
    handle.resume()
