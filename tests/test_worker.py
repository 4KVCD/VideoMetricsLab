import contextlib
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import FrameScore, ResampleTarget, VideoInfo, VmafOptions, VmafRunResult
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError
from vmaf_app.ui import worker as worker_module
from vmaf_app.ui.worker import VmafJob, VmafWorker


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


# --------------------------------------------------- scoring several at once

def _drain(qapp) -> None:
    """Delivers queued signals.

    Lanes run on their own threads, so Qt queues their signals to the main
    thread rather than calling straight through -- which is exactly what
    keeps the real handlers on the GUI thread. Without an event loop running
    in the test, nothing arrives until the queue is pumped by hand.
    """
    for _ in range(5):
        qapp.processEvents()


def _jobs(count: int) -> list[VmafJob]:
    return [
        VmafJob(_info("s.mp4"), _info(f"d{n}.mp4"), VmafOptions(), label=f"d{n}")
        for n in range(count)
    ]


def _concurrency_probe(monkeypatch):
    """Records how many jobs were ever inside run_vmaf simultaneously."""
    import threading

    state = {"live": 0, "peak": 0, "order": [], "entered": 0}
    lock = threading.Lock()
    # Only the first two callers gate on each other: that is enough to prove
    # they overlap, and making every job wait would cost a barrier timeout
    # per job on the single-lane runs.
    gate = threading.Barrier(2, timeout=2)

    def counted(source, distorted, *a, **kw):
        with lock:
            state["live"] += 1
            state["entered"] += 1
            state["peak"] = max(state["peak"], state["live"])
            state["order"].append(distorted.path.name)
            gating = state["entered"] <= 2
        if gating:
            with contextlib.suppress(threading.BrokenBarrierError):
                gate.wait()
        with lock:
            state["live"] -= 1
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(worker_module, "run_vmaf", counted)
    return state


def test_two_jobs_really_run_at_the_same_time(qapp, monkeypatch):
    """libvmaf leaves much of a many-core CPU idle, so a second video fills
    the gap rather than competing for it -- but only if they genuinely
    overlap."""
    state = _concurrency_probe(monkeypatch)
    worker = VmafWorker(_jobs(4), parallel_jobs=2)

    worker.run()

    assert state["peak"] == 2
    assert state["live"] == 0


def test_one_at_a_time_stays_one_at_a_time(qapp, monkeypatch):
    state = _concurrency_probe(monkeypatch)
    worker = VmafWorker(_jobs(2), parallel_jobs=1)

    worker.run()

    assert state["peak"] == 1


def test_every_job_runs_exactly_once_across_the_lanes(qapp, monkeypatch):
    # Lanes pull from one shared iterator; handing each lane a fixed slice
    # would leave one idle while the other still had a queue.
    state = _concurrency_probe(monkeypatch)
    finished = []
    worker = VmafWorker(_jobs(7), parallel_jobs=2)
    worker.job_finished.connect(lambda index, result: finished.append(index))

    worker.run()
    _drain(qapp)

    assert sorted(state["order"]) == [f"d{n}.mp4" for n in range(7)]
    assert sorted(finished) == list(range(7))


def test_more_lanes_than_jobs_does_not_start_empty_lanes(qapp, monkeypatch):
    state = _concurrency_probe(monkeypatch)
    worker = VmafWorker(_jobs(1), parallel_jobs=4)

    worker.run()

    assert state["order"] == ["d0.mp4"]


def test_the_parallel_count_is_clamped_to_something_sane(qapp):
    assert VmafWorker(_jobs(1), parallel_jobs=0)._parallel_jobs == 1
    assert VmafWorker(_jobs(1), parallel_jobs=-3)._parallel_jobs == 1
    assert VmafWorker(_jobs(1), parallel_jobs=99)._parallel_jobs == worker_module.MAX_PARALLEL_JOBS


def test_pausing_reaches_every_running_job(qapp, monkeypatch):
    """One handle can only ever address one pid, so with several ffmpegs up
    a single shared handle would leave all but one running."""
    import threading

    handles = []
    started = threading.Barrier(3, timeout=10)
    release = threading.Event()

    def capture(source, distorted, *a, **kw):
        handles.append(kw["process_handle"])
        started.wait()
        release.wait(10)
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(worker_module, "run_vmaf", capture)
    worker = VmafWorker(_jobs(2), parallel_jobs=2)
    runner = threading.Thread(target=worker.run)
    runner.start()
    try:
        started.wait(timeout=10)
        worker.pause()
        assert worker.is_paused
        assert len(handles) == 2
        assert all(h.is_pause_requested for h in handles), "a running job was left unpaused"

        worker.resume()
        assert not worker.is_paused
        assert not any(h.is_pause_requested for h in handles)
    finally:
        release.set()
        runner.join(timeout=10)


def test_a_job_that_starts_while_paused_comes_up_paused(qapp, monkeypatch):
    # Otherwise pressing Pause and waiting would quietly let the next video
    # start at full speed.
    worker = VmafWorker(_jobs(1), parallel_jobs=1)
    worker.pause()

    handle = worker._claim_handle(0)

    assert handle.is_pause_requested


def test_cancelling_terminates_every_running_job(qapp, monkeypatch):
    import threading

    terminated = []
    started = threading.Barrier(3, timeout=10)
    # The jobs have to still be running when cancel() is called; without
    # this they would raise and release their handles first, and cancel
    # would find nothing to terminate whether or not it worked.
    release = threading.Event()

    def capture(source, distorted, *a, **kw):
        handle = kw["process_handle"]
        handle.terminate = lambda h=handle: terminated.append(h)
        started.wait()
        release.wait(10)
        raise Cancelled("cancelled")

    monkeypatch.setattr(worker_module, "run_vmaf", capture)
    worker = VmafWorker(_jobs(2), parallel_jobs=2)
    reported = []
    worker.cancelled.connect(lambda: reported.append(True))
    runner = threading.Thread(target=worker.run)
    runner.start()
    try:
        started.wait(timeout=10)
        worker.cancel()
        assert len(terminated) == 2, "cancel did not reach every running ffmpeg"
    finally:
        release.set()
        runner.join(timeout=10)
    _drain(qapp)

    # Both lanes raise Cancelled, but the run stopped once.
    assert reported == [True]


def test_a_failing_job_does_not_take_the_other_lane_with_it(qapp, monkeypatch):
    def sometimes_fails(source, distorted, *a, **kw):
        if distorted.path.name == "d0.mp4":
            raise VmafRunError("boom", "stderr tail")
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(worker_module, "run_vmaf", sometimes_fails)
    worker = VmafWorker(_jobs(3), parallel_jobs=2)
    failed, finished = [], []
    worker.job_failed.connect(lambda i, m, t: failed.append(i))
    worker.job_finished.connect(lambda i, r: finished.append(i))

    worker.run()
    _drain(qapp)

    assert failed == [0]
    assert sorted(finished) == [1, 2]
