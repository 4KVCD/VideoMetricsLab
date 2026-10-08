import psutil
import pytest

from tests.factories import STDLIB_PYTHON
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


# ------------------------------------------------ FFmpeg behind a launcher
#
# Chocolatey installs ffmpeg.exe as a shim: a launcher that starts the real
# ffmpeg.exe as its child. Real processes here, not the fake above.

_LAUNCHER = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"


def _launcher_with_child():
    """A launcher process and the long-running child it started, once the
    child is running: a pause landing while the launcher is still creating
    it makes Windows refuse the creation, and the child is gone."""
    import subprocess

    launcher = subprocess.Popen([STDLIB_PYTHON, "-S", "-c", _LAUNCHER, STDLIB_PYTHON, "-S", "-c",
                                 "import sys, time; print('running', flush=True); time.sleep(60)"],
                                stdout=subprocess.PIPE, text=True)
    if launcher.stdout.readline().strip() != "running":  # the child inherits the launcher's stdout
        launcher.kill()
        raise AssertionError("the launcher did not start its child")
    [child] = psutil.Process(launcher.pid).children()
    return launcher, child


def test_pause_resume_and_cancel_reach_a_process_started_by_a_launcher():
    launcher, child = _launcher_with_child()
    handle = ProcessHandle()
    try:
        handle.attach(launcher.pid)
        handle.pause()
        assert child.status() == psutil.STATUS_STOPPED, "the real process kept running while paused"
        handle.resume()
        assert child.status() != psutil.STATUS_STOPPED
        handle.terminate()
        child.wait(timeout=10)
        launcher.wait(timeout=10)
        assert not child.is_running()
    finally:
        for process in (child,):
            if process.is_running():
                process.kill()
        if launcher.poll() is None:
            launcher.kill()
        launcher.stdout.close()
