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

    def children(self, recursive=False):
        return []


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeProcess.instances.clear()
    yield
    FakeProcess.instances.clear()


@pytest.fixture
def patched_psutil(monkeypatch):
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    return FakeProcess


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


def test_a_handle_reaches_every_attached_process(monkeypatch):
    """Black-bar detection runs its sample windows as several processes at
    once. Pause or Cancel during that moment has to reach all of them -- a
    handle that remembered only the latest pid left the others running."""
    actions = []
    monkeypatch.setattr(
        ProcessHandle, "_try", staticmethod(lambda pid, action: actions.append((pid, action)))
    )
    handle = ProcessHandle()
    handle.attach(101)
    handle.attach(102)
    handle.attach(103)

    handle.pause()
    assert sorted(actions) == [(101, "suspend"), (102, "suspend"), (103, "suspend")]

    actions.clear()
    handle.detach(102)  # one window finished; the others are still running
    handle.terminate()
    assert sorted(actions) == [(101, "terminate"), (103, "terminate")]

    handle.detach()  # no pid: everything, as single-process callers expect
    actions.clear()
    handle.resume()
    assert actions == []
