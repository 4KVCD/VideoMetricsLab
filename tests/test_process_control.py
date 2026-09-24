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



def test_a_pause_requested_before_the_process_existed_survives_a_resume_race():
    """attach() recorded the pid, released the lock and only then suspended.
    A resume arriving in that gap ran first and the stale suspend afterwards,
    leaving the process stopped with nothing left to start it again.

    The window is forced open here rather than hoped for: the suspend call
    itself blocks until resume has been attempted.
    """
    import threading

    handle = ProcessHandle()
    calls = []
    suspending = threading.Event()
    let_suspend_finish = threading.Event()

    def blocking_try(pid, action):
        if action == "suspend":
            suspending.set()
            let_suspend_finish.wait(5)
        # Recorded on COMPLETION, not on entry: what matters is which call
        # last touched the process, and the whole bug is that the suspend
        # finishes after the resume.
        calls.append(action)

    handle._try = blocking_try
    handle.pause()

    attaching = threading.Thread(target=lambda: handle.attach(4242))
    attaching.start()
    assert suspending.wait(5), "attach never tried to suspend"

    # Resume arrives while the suspend is still in flight.
    resuming = threading.Thread(target=handle.resume)
    resuming.start()
    resuming.join(timeout=1)
    let_suspend_finish.set()
    attaching.join(timeout=5)
    resuming.join(timeout=5)

    assert not handle.is_pause_requested
    assert calls[-1] == "resume", (
        f"the last thing done to the process was {calls[-1]!r}, so it stayed paused"
    )


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
    """A launcher process and the long-running child it started."""
    import subprocess
    import sys
    import time

    launcher = subprocess.Popen([sys.executable, "-c", _LAUNCHER, sys.executable, "-c",
                                 "import time; time.sleep(60)"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        children = psutil.Process(launcher.pid).children()
        if children:
            return launcher, children[0]
        time.sleep(0.05)
    launcher.kill()
    raise AssertionError("the launcher did not start its child")


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
