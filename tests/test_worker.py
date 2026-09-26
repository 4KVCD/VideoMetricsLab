import contextlib
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import ComparisonResult, FrameScore, ResampleTarget, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualRunError, PerceptualTaskOutput
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError
from vmaf_app.ui import worker as worker_module
from vmaf_app.ui.worker import VmafJob, VmafWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _info(name: str) -> VideoInfo:
    return VideoInfo(path=Path(name), width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=150, codec_name="h264")


def _fake_result(name: str) -> ComparisonResult:
    info = _info(name)
    frames = [FrameScore(frame=i, time=i / 30.0, vmaf=90.0) for i in range(10)]
    return ComparisonResult(
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


def _perceptual_output() -> PerceptualTaskOutput:
    metric = FrameMetricResult(
        "ssimulacra2", [0], [0.0], [87.0],
        MetricProvenance("test", "1", "gpu", "test-v1"),
    )
    return PerceptualTaskOutput(MetricResultSet([metric]), None, None, 10)


def test_ffmpeg_and_vship_run_at_the_same_time_and_merge(qapp, monkeypatch):
    started = threading.Barrier(3, timeout=5)
    release = threading.Event()
    handles = []

    def ffmpeg(*args, **kwargs):
        handles.append(kwargs["process_handle"])
        started.wait()
        assert release.wait(5)
        return _fake_result("d.mp4")

    def vship(*args, **kwargs):
        handles.append(kwargs["process_handle"])
        started.wait()
        assert release.wait(5)
        return _perceptual_output()

    monkeypatch.setattr(worker_module, "run_vmaf", ffmpeg)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d",
                  metric_keys=("vmaf", "ssimulacra2"))
    worker = VmafWorker([job])
    finished = []
    worker.job_finished.connect(lambda _, result: finished.append(result))
    runner = threading.Thread(target=worker.run)
    runner.start()
    try:
        started.wait()  # both tasks must enter before either is released
        assert handles[0] is handles[1]  # one pause/cancel control covers both
    finally:
        release.set()
        runner.join(timeout=5)
    assert not runner.is_alive()
    _drain(qapp)
    assert len(finished) == 1
    assert finished[0].has_metric("vmaf")
    assert finished[0].has_metric("ssimulacra2")


def _run_one(qapp, monkeypatch, ffmpeg, vship, keys=("vmaf", "ssimulacra2")):
    monkeypatch.setattr(worker_module, "run_vmaf", ffmpeg)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    worker = VmafWorker([VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=keys)])
    events = []
    worker.job_finished.connect(lambda _, result: events.append(("finished", result)))
    worker.job_partially_failed.connect(lambda _, result, message, tail: events.append(("partly", result, message, tail)))
    worker.job_failed.connect(lambda _, message, tail: events.append(("failed", message, tail)))
    worker.cancelled.connect(lambda: events.append(("cancelled",)))
    worker.run()
    _drain(qapp)
    return events


def test_a_perceptual_failure_keeps_the_ffmpeg_metrics(qapp, monkeypatch):
    """SSIMULACRA2 failing (an unsupported input, a tool error) used to
    cancel the libvmaf pass and fail the video, discarding VMAF. The FFmpeg
    task now finishes and its metrics are the result."""
    ffmpeg_saw_cancel = []

    def ffmpeg(*args, **kwargs):
        time.sleep(0.2)  # still running when the perceptual task fails
        ffmpeg_saw_cancel.append(kwargs["cancel_event"].is_set())
        return _fake_result("d.mp4")

    def vship(*args, **kwargs):
        raise PerceptualRunError("Variable-frame-rate video is not supported safely yet.", "tail")

    events = _run_one(qapp, monkeypatch, ffmpeg, vship)
    (kind, result, message, tail), = events
    assert kind == "partly"
    assert result.has_metric("vmaf") and not result.has_metric("ssimulacra2")
    assert message == "SSIMULACRA2 failed: Variable-frame-rate video is not supported safely yet."
    assert tail == "tail"
    assert ffmpeg_saw_cancel == [False]


def test_an_ffmpeg_failure_keeps_the_perceptual_metrics(qapp, monkeypatch):
    def ffmpeg(*args, **kwargs):
        raise VmafRunError("FFmpeg failed", "stderr tail")

    events = _run_one(qapp, monkeypatch, ffmpeg, lambda *a, **k: _perceptual_output())
    (kind, result, message, tail), = events
    assert kind == "partly"
    assert result.has_metric("ssimulacra2") and not result.has_metric("vmaf")
    assert message == "VMAF failed: FFmpeg failed" and tail == "stderr tail"


def test_both_groups_failing_fails_the_video_without_cancelling_the_run(qapp, monkeypatch):
    def ffmpeg(*args, **kwargs):
        raise VmafRunError("FFmpeg failed", "stderr tail")

    def vship(*args, **kwargs):
        raise PerceptualRunError("Vship failed")

    events = _run_one(qapp, monkeypatch, ffmpeg, vship)
    # One failure for the video, reported with whichever error came first:
    # the two groups run on their own threads. This used to expect the
    # Vship error, relying on a 0.1 s sleep in the FFmpeg stand-in, and
    # failed on a busy machine that started the Vship thread later.
    assert len(events) == 1 and events[0][0] == "failed"
    assert events[0][1:] in {("Vship failed", ""), ("FFmpeg failed", "stderr tail")}


def test_user_cancel_reaches_both_concurrent_backends(qapp, monkeypatch):
    started = threading.Barrier(3, timeout=5)
    observed = []

    def ffmpeg(*args, **kwargs):
        started.wait()
        token = kwargs["cancel_event"]
        deadline = time.monotonic() + 5
        while not token.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        observed.append(("ffmpeg", token.is_set()))
        raise Cancelled("user cancelled")

    def vship(*args, **kwargs):
        started.wait()
        token = kwargs["cancel_event"]
        deadline = time.monotonic() + 5
        while not token.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        observed.append(("vship", token.is_set()))
        raise PerceptualCancelled("user cancelled")

    monkeypatch.setattr(worker_module, "run_vmaf", ffmpeg)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d",
                  metric_keys=("vmaf", "ssimulacra2"))
    worker = VmafWorker([job])
    cancelled, finished = [], []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.job_finished.connect(lambda *_: finished.append(True))
    runner = threading.Thread(target=worker.run)
    runner.start()
    try:
        started.wait()
        worker.cancel()
    finally:
        runner.join(timeout=5)
    assert not runner.is_alive()
    _drain(qapp)
    assert sorted(observed) == [("ffmpeg", True), ("vship", True)]
    assert cancelled == [True]
    assert not finished


# ---------------------------------------------------- sharing out the cores

def _threads_asked_for(monkeypatch) -> dict[str, int]:
    """Records the n_threads each job reached run_vmaf with, by file."""
    seen: dict[str, int] = {}

    def record(source, distorted, options, *a, **kw):
        seen[distorted.path.name] = options.n_threads
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(worker_module, "run_vmaf", record)
    return seen


def test_auto_threads_are_halved_when_two_videos_are_scored_at_once(qapp, monkeypatch):
    """Two libvmaf instances each asking for all 24 cores is 48 threads
    taking turns on 24; each gets half instead."""
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = _threads_asked_for(monkeypatch)

    VmafWorker(_jobs(3), parallel_jobs=2).run()

    assert seen == {"d0.mp4": 12, "d1.mp4": 12, "d2.mp4": 12}


def test_a_lone_video_keeps_every_core_however_the_box_is_ticked(qapp, monkeypatch):
    """The setting says two may run at once; with one video queued, one
    runs. Planning it for half the machine would leave the other half
    idle for the whole run."""
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = _threads_asked_for(monkeypatch)

    VmafWorker(_jobs(1), parallel_jobs=2).run()

    assert seen == {"d0.mp4": 0}  # still Auto, which the runner resolves to every core


def test_videos_scored_one_at_a_time_keep_every_core(qapp, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = _threads_asked_for(monkeypatch)

    VmafWorker(_jobs(3), parallel_jobs=1).run()

    assert seen == {"d0.mp4": 0, "d1.mp4": 0, "d2.mp4": 0}


def test_an_explicit_thread_count_is_the_users_and_is_not_shared(qapp, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = _threads_asked_for(monkeypatch)
    jobs = [
        VmafJob(_info("s.mp4"), _info(f"d{n}.mp4"), VmafOptions(n_threads=20), label=f"d{n}")
        for n in range(2)
    ]

    VmafWorker(jobs, parallel_jobs=2).run()

    assert seen == {"d0.mp4": 20, "d1.mp4": 20}


def test_a_resample_test_shares_the_cores_like_any_other_job(qapp, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = []

    def record(source, options, *a, **kw):
        seen.append(options.n_threads)
        return _fake_result("s [downscale-1080p-upscale].mp4")

    monkeypatch.setattr(worker_module, "run_resample_test", record)
    options = VmafOptions(resample_test=ResampleTarget(width=1920, label="1080p"))
    jobs = [VmafJob(_info("s.mp4"), _info("s.mp4"), options, label=f"t{n}") for n in range(2)]

    VmafWorker(jobs, parallel_jobs=2).run()

    assert seen == [12, 12]


def test_a_video_started_after_the_count_was_raised_gets_half(qapp, monkeypatch):
    """The share is decided as each video starts, not when the run does.
    The one already running keeps its threads -- ffmpeg cannot be re-told
    -- and everything that starts after the change shares the machine."""
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    first_started = threading.Event()
    release = threading.Event()
    seen: dict[str, int] = {}

    def blocking(source, distorted, options, *a, **kw):
        seen[distorted.path.name] = options.n_threads
        first_started.set()
        release.wait(10)
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(worker_module, "run_vmaf", blocking)
    worker = VmafWorker(_jobs(4), parallel_jobs=1)
    runner = threading.Thread(target=worker.run)
    runner.start()
    try:
        assert first_started.wait(5)
        deadline = time.time() + 5
        worker.set_parallel_jobs(2)
        while len(seen) < 2 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        release.set()
        runner.join(timeout=10)

    # Lanes pull their index before they wait for a slot, so which file went
    # first is a race; which *launch* went first is not (insertion order).
    launched = list(seen.values())
    assert launched[0] == 0      # launched alone: Auto, every core
    assert launched[1] == 12     # launched beside it: half


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



# ------------------------------- races between the controls and a new lane

def test_resume_cannot_be_overtaken_by_a_lane_starting_paused(qapp):
    """The reported race: _claim_handle recorded "we are paused", released
    the lock, and only then paused the handle. A Resume landing in that gap
    ran first and the stale pause ran after it, leaving that lane suspended
    for good while the UI reported the run as resumed.

    Recording and pausing now happen under the one lock that resume() also
    takes, so the two cannot interleave.
    """
    import threading

    worker = VmafWorker(_jobs(2), parallel_jobs=2)
    worker.pause()

    claimed = []
    inside = threading.Event()
    proceed = threading.Event()
    real_pause = worker_module.ProcessHandle.pause

    def slow_pause(self):
        inside.set()
        proceed.wait(5)
        real_pause(self)

    # Widen the window the race needs, then try to resume through it.
    monkey = threading.Thread(target=lambda: claimed.append(worker._claim_handle(0)))
    worker_module.ProcessHandle.pause = slow_pause
    try:
        monkey.start()
        assert inside.wait(5)
        resumed = threading.Thread(target=worker.resume)
        resumed.start()
        proceed.set()
        monkey.join(timeout=5)
        resumed.join(timeout=5)
    finally:
        worker_module.ProcessHandle.pause = real_pause

    assert not worker.is_paused
    assert claimed and claimed[0] is not None
    assert not claimed[0].is_pause_requested, "a lane was left paused after Resume"


def test_a_lane_cannot_start_a_job_after_cancel(qapp):
    """The second reported race: a lane checked the cancel flag, then cancel
    ran and found no handle to terminate, then the lane registered one and
    launched ffmpeg anyway. Claiming is now refused once cancel has been
    seen, under the same lock cancel collects handles with."""
    worker = VmafWorker(_jobs(2), parallel_jobs=2)
    worker.cancel()

    assert worker._claim_handle(0) is None


def test_cancel_terminates_handles_claimed_before_it(qapp):
    worker = VmafWorker(_jobs(2), parallel_jobs=2)
    handle = worker._claim_handle(0)
    terminated = []
    handle.terminate = lambda: terminated.append(True)

    worker.cancel()

    assert terminated == [True]


def test_a_lane_blocked_on_a_slot_is_released_by_cancel(qapp):
    # Lanes wait for room to run. Cancel has to wake them, or the worker
    # thread never joins and the window cannot close.
    import threading

    worker = VmafWorker(_jobs(4), parallel_jobs=1)
    worker._busy["cpu"] = 1  # pretend the single slot is taken
    worker._queues["cpu"].append((SimpleNamespace(started=True, admitted=set()), None))
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(worker._next_task("cpu")))
    waiter.start()
    try:
        worker.cancel()
        waiter.join(timeout=5)
    finally:
        assert not waiter.is_alive(), "a lane stayed blocked after cancel"
    assert outcome == [None]


def test_the_lane_count_can_be_raised_while_running(qapp):
    # The whole reason the control sits next to Run: a long queue is when
    # someone notices the CPU is idle.
    import threading

    started = threading.Event()
    release = threading.Event()
    live = {"count": 0, "peak": 0}
    lock = threading.Lock()

    def blocking(source, distorted, *a, **kw):
        with lock:
            live["count"] += 1
            live["peak"] = max(live["peak"], live["count"])
        started.set()
        release.wait(10)
        with lock:
            live["count"] -= 1
        return _fake_result(distorted.path.name)

    monkeypatch_target = worker_module.run_vmaf
    worker_module.run_vmaf = blocking
    try:
        worker = VmafWorker(_jobs(4), parallel_jobs=1)
        runner = threading.Thread(target=worker.run)
        runner.start()
        assert started.wait(5)
        time.sleep(0.3)
        assert live["peak"] == 1, "more than one ran before the count was raised"

        worker.set_parallel_jobs(2)
        time.sleep(0.5)
        assert live["peak"] == 2, "raising the count did not start another video"
    finally:
        release.set()
        worker_module.run_vmaf = monkeypatch_target
        runner.join(timeout=10)


def test_lowering_the_lane_count_does_not_interrupt_a_running_job(qapp):
    worker = VmafWorker(_jobs(4), parallel_jobs=2)
    worker._active = 2

    worker.set_parallel_jobs(1)

    # Nothing was cancelled; there is simply no room for another to start.
    assert worker.parallel_jobs == 1
    assert not worker._cancel_event.is_set()


def _saved_run(*metrics: FrameMetricResult) -> ComparisonResult:
    result = _fake_result("d.mp4")
    result.merge_metric_results(MetricResultSet(metrics))
    return result


def test_a_row_with_saved_vmaf_runs_only_the_new_perceptual_metric(qapp, monkeypatch):
    """Adding SSIMULACRA2 to a row that already had VMAF recalculated VMAF
    too: the planner was never told what was saved."""
    calls = []
    monkeypatch.setattr(worker_module, "run_vmaf", lambda *a, **kw: calls.append("ffmpeg") or _fake_result("d.mp4"))
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback",
                        lambda *a, **kw: calls.append("perceptual") or _perceptual_output())
    saved = _fake_result("d.mp4")
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=("vmaf", "ssimulacra2"),
                  cached_result=saved, cached_metrics=MetricResultSet([saved.frame_metric("vmaf")]))
    worker = VmafWorker([job])
    finished = []
    worker.job_finished.connect(lambda _, result: finished.append(result))
    worker.run()
    _drain(qapp)

    assert calls == ["perceptual"]
    result, = finished
    assert result.frame_metric("vmaf").values.tolist() == [90.0] * 10  # the saved scores
    assert result.frame_metric("ssimulacra2").values.tolist() == [87.0]
    assert not saved.has_metric("ssimulacra2"), "the row's own result must not change under it"


def test_a_row_with_a_saved_perceptual_score_runs_only_ffmpeg(qapp, monkeypatch):
    calls = []
    fresh = _fake_result("d.mp4")
    monkeypatch.setattr(worker_module, "run_vmaf", lambda *a, **kw: calls.append("ffmpeg") or fresh)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback",
                        lambda *a, **kw: calls.append("perceptual") or _perceptual_output())
    butteraugli = FrameMetricResult("butteraugli", [0, 1], [0.0, 0.033], [1.5, 2.5],
                                    MetricProvenance("butteraugli", "0.12", "cpu", "butteraugli-libjxl-cpu-v1"))
    saved = _saved_run(butteraugli)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=("vmaf", "butteraugli"),
                  cached_result=saved, cached_metrics=MetricResultSet([butteraugli]))
    worker = VmafWorker([job])
    finished = []
    worker.job_finished.connect(lambda _, result: finished.append(result))
    worker.run()
    _drain(qapp)

    assert calls == ["ffmpeg"]
    result, = finished
    assert result.frame_metric("butteraugli").values.tolist() == [1.5, 2.5]
    assert result.has_metric("vmaf")



def test_cvvdp_failing_beside_ssimulacra2_is_a_partial_failure(qapp, monkeypatch):
    """CVVDP runs on the GPU only. When it fails while SSIMULACRA2 in the
    same pass finishes, the video keeps VMAF and SSIMULACRA2 and says that
    CVVDP failed and why, rather than looking fully scored."""
    def vship(*args, **kwargs):
        output = _perceptual_output()
        return PerceptualTaskOutput(output.metrics, None, None, 10, {"cvvdp": "out of GPU memory"})

    events = _run_one(qapp, monkeypatch, lambda *a, **k: _fake_result("d.mp4"), vship,
                      keys=("vmaf", "ssimulacra2", "cvvdp"))
    (kind, result, message, _tail), = events
    assert kind == "partly"
    assert result.has_metric("vmaf") and result.has_metric("ssimulacra2") and not result.has_metric("cvvdp")
    assert message == "CVVDP failed: out of GPU memory"


def test_the_jobs_cvvdp_settings_reach_the_request(qapp, monkeypatch):
    from vmaf_app.core.cvvdp import DEFAULT_PRESET, with_display

    seen = []

    def vship(_source, _test, request, specs, **kwargs):
        seen.append(dict(specs[0].parameters)["display"]["peak_luminance"])
        return _perceptual_output()

    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=("cvvdp",),
                  cvvdp=with_display(DEFAULT_PRESET.settings, peak_luminance=321))
    VmafWorker([job]).run()
    _drain(qapp)
    assert seen == [321]


def test_a_half_waiting_for_the_gpu_is_reported_beside_the_running_half(qapp, monkeypatch):
    from vmaf_app.core.perceptual_vship import GPU_WAIT_MESSAGE

    ffmpeg_reported, gpu_waiting = threading.Event(), threading.Event()

    def ffmpeg(*args, on_progress=None, **kwargs):
        gpu_waiting.wait(5)
        on_progress(10, 100, 11.0)
        ffmpeg_reported.set()
        time.sleep(0.2)
        return _fake_result("d.mp4")

    def vship(*args, on_status=None, on_progress=None, **kwargs):
        on_status(GPU_WAIT_MESSAGE)
        gpu_waiting.set()
        ffmpeg_reported.wait(5)
        time.sleep(0.05)
        on_progress(50, 100, 40.0)
        return _perceptual_output()

    monkeypatch.setattr(worker_module, "run_vmaf", ffmpeg)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    worker = VmafWorker([VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d",
                                 metric_keys=("vmaf", "ssimulacra2"))])
    seen = []
    worker.halves.connect(lambda _index, halves: seen.append([(labels, state) for labels, *_x, state in halves]))
    worker.run()
    _drain(qapp)
    assert [("VMAF", "running"), ("SSIMULACRA2", "waiting")] in seen
    assert [("VMAF", "running"), ("SSIMULACRA2", "running")] in seen


def _split_job(name: str, keys=("vmaf", "ssimulacra2")) -> VmafJob:
    return VmafJob(_info("s.mp4"), _info(name), VmafOptions(), label=name, metric_keys=keys)


def test_cpu_lanes_take_the_next_cpu_work_and_the_gpu_goes_in_list_order(qapp, monkeypatch):
    """The user's queue: a video needing both halves, one whose FFmpeg
    metrics were saved (GPU only), and another needing both. A lane took a
    whole video, so the second lane went to the GPU-only video, which took
    the GPU ahead of the first video while the third video's FFmpeg metrics
    never started. Now the two CPU lanes run the first and third videos'
    FFmpeg metrics while the first video has the GPU."""
    overlap = threading.Barrier(3, timeout=5)
    gpu_order, lock = [], threading.Lock()

    def ffmpeg(source, distorted, *a, **k):
        if distorted.path.name in ("d0.mp4", "d2.mp4"):
            overlap.wait()
        return _fake_result(distorted.path.name)

    def vship(source, distorted, *a, **k):
        with lock:
            gpu_order.append(distorted.path.name)
        if distorted.path.name == "d0.mp4":
            overlap.wait()
        return _perceptual_output()

    monkeypatch.setattr(worker_module, "run_vmaf", ffmpeg)
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    worker = VmafWorker([_split_job("d0.mp4"), _split_job("d1.mp4", ("ssimulacra2",)), _split_job("d2.mp4")],
                        parallel_jobs=2)
    finished = []
    worker.job_finished.connect(lambda index, _result: finished.append(index))
    worker.run()
    _drain(qapp)
    assert not overlap.broken, "the first and third videos' FFmpeg metrics did not run beside the first's GPU half"
    assert gpu_order == ["d0.mp4", "d1.mp4", "d2.mp4"]
    assert sorted(finished) == [0, 1, 2]


def test_no_more_than_three_videos_are_in_progress_at_once(qapp, monkeypatch):
    """The CPU lanes may run ahead of a slow GPU half, but not without limit:
    each video in progress has a line in the window."""
    release = threading.Event()
    started = []

    def vship(source, distorted, *a, **k):
        if distorted.path.name == "d0.mp4":
            release.wait(5)
        return _perceptual_output()

    monkeypatch.setattr(worker_module, "run_vmaf", lambda s, d, *a, **k: _fake_result(d.path.name))
    monkeypatch.setattr(worker_module, "apply_vship_cpu_fallback", vship)
    worker = VmafWorker([_split_job(f"d{n}.mp4") for n in range(6)], parallel_jobs=2)
    worker.job_started.connect(lambda index, _label: started.append(index))
    runner = threading.Thread(target=worker.run)
    runner.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    in_progress_while_blocked = sorted(started)
    release.set()
    runner.join(10)
    _drain(qapp)
    assert in_progress_while_blocked == [0, 1, 2]
    assert sorted(started) == list(range(6))

