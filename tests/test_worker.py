import os
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from vmaf_app.core import job_runner
from vmaf_app.core.job_runner import JobRun, JobScheduler, VmafJob
from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
from vmaf_app.core.models import ComparisonResult, FrameScore, VideoInfo, VmafOptions
from vmaf_app.core.perceptual_cpu import PerceptualCancelled, PerceptualRunError, PerceptualTaskOutput
from vmaf_app.core.vmaf_runner import Cancelled, VmafRunError
from vmaf_app.ui.worker import VmafWorker


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
    monkeypatch.setattr(job_runner, "run_vmaf", lambda *a, **kw: calls.append("run_vmaf") or _fake_result("d.mp4"))
    monkeypatch.setattr(job_runner, "run_resample_test", lambda *a, **kw: calls.append("run_resample_test"))

    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d")
    w = VmafWorker([job])
    w.run()  # run synchronously in-test rather than as a real thread

    assert calls == ["run_vmaf"]


def test_worker_reports_cancellation_as_a_distinct_terminal_state(qapp, monkeypatch):
    def cancelled_run(*args, **kwargs):
        raise Cancelled("cancelled")

    monkeypatch.setattr(job_runner, "run_vmaf", cancelled_run)
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

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
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


def _run_one(qapp, monkeypatch, ffmpeg, vship, keys=("vmaf", "ssimulacra2"), on_worker=None):
    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
    worker = VmafWorker([VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=keys)])
    if on_worker is not None:
        on_worker(worker)
    events = []
    worker.job_finished.connect(lambda _, result: events.append(("finished", result)))
    worker.job_partially_failed.connect(
        lambda _, result, message, tail, _reasons: events.append(("partly", result, message, tail)))
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
    perceptual_failed = threading.Event()

    def ffmpeg(*args, **kwargs):
        assert perceptual_failed.wait(5)  # still running when the perceptual task has failed
        ffmpeg_saw_cancel.append(kwargs["cancel_event"].is_set())
        return _fake_result("d.mp4")

    def vship(*args, **kwargs):
        raise PerceptualRunError("Variable-frame-rate video is not supported safely yet.", "tail")

    def watch(worker):
        def seen(_index, snapshots):
            if any(task["backend"] == "perceptual" and task["state"] == "failed" for task in snapshots):
                perceptual_failed.set()
        worker.task_progress.connect(seen, Qt.ConnectionType.DirectConnection)

    events = _run_one(qapp, monkeypatch, ffmpeg, vship, on_worker=watch)
    (kind, result, message, tail), = events
    assert kind == "partly"
    assert result.has_metric("vmaf") and not result.has_metric("ssimulacra2")
    assert message == "SSIMULACRA2 failed: Variable-frame-rate video is not supported safely yet."
    assert tail == "tail"
    assert ffmpeg_saw_cancel == [False]


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

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
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

    monkeypatch.setattr(job_runner, "run_vmaf", record)
    return seen


def test_auto_threads_are_halved_when_two_videos_are_scored_at_once(qapp, monkeypatch):
    """Two libvmaf instances each asking for all 24 cores is 48 threads
    taking turns on 24; each gets half instead."""
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    seen = _threads_asked_for(monkeypatch)

    VmafWorker(_jobs(3), parallel_jobs=2).run()

    assert seen == {"d0.mp4": 12, "d1.mp4": 12, "d2.mp4": 12}


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


def _concurrency_probe(monkeypatch, *, gate_two: bool = False):
    """Records how many jobs were ever inside run_vmaf simultaneously, and
    in what order. With gate_two, the first two wait for each other: they
    must be in run_vmaf at the same time to go on."""
    state = {"live": 0, "peak": 0, "order": [], "entered": 0}
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def counted(source, distorted, *a, **kw):
        with lock:
            state["live"] += 1
            state["entered"] += 1
            state["peak"] = max(state["peak"], state["live"])
            state["order"].append(distorted.path.name)
            gating = gate_two and state["entered"] <= 2
        if gating:
            gate.wait()
        with lock:
            state["live"] -= 1
        return _fake_result(distorted.path.name)

    monkeypatch.setattr(job_runner, "run_vmaf", counted)
    return state


def test_two_jobs_really_run_at_the_same_time(qapp, monkeypatch):
    """libvmaf leaves much of a many-core CPU idle, so a second video fills
    the gap rather than competing for it -- but only if they genuinely
    overlap."""
    state = _concurrency_probe(monkeypatch, gate_two=True)
    worker = VmafWorker(_jobs(4), parallel_jobs=2)

    worker.run()

    assert state["peak"] == 2
    assert state["live"] == 0


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

    monkeypatch.setattr(job_runner, "run_vmaf", capture)
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

    monkeypatch.setattr(job_runner, "run_vmaf", capture)
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

    monkeypatch.setattr(job_runner, "run_vmaf", sometimes_fails)
    worker = VmafWorker(_jobs(3), parallel_jobs=2)
    failed, finished = [], []
    worker.job_failed.connect(lambda i, m, t: failed.append(i))
    worker.job_finished.connect(lambda i, r: finished.append(i))

    worker.run()
    _drain(qapp)

    assert failed == [0]
    assert sorted(finished) == [1, 2]


# ------------------------------- races between the controls and a new lane


def test_a_row_with_saved_vmaf_runs_only_the_new_perceptual_metric(qapp, monkeypatch):
    """Adding SSIMULACRA2 to a row that already had VMAF recalculated VMAF
    too: the planner was never told what was saved."""
    calls = []
    monkeypatch.setattr(job_runner, "run_vmaf", lambda *a, **kw: calls.append("ffmpeg") or _fake_result("d.mp4"))
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback",
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


def test_the_jobs_cvvdp_settings_reach_the_request(qapp, monkeypatch):
    from vmaf_app.core.cvvdp import DEFAULT_PRESET, with_display

    seen = []

    def vship(_source, _test, request, specs, **kwargs):
        seen.append(dict(specs[0].parameters)["display"]["peak_luminance"])
        return _perceptual_output()

    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", vship)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), "d", metric_keys=("cvvdp",),
                  cvvdp=with_display(DEFAULT_PRESET.settings, peak_luminance=321))
    VmafWorker([job]).run()
    _drain(qapp)
    assert seen == [321]


def _split_job(name: str, keys=("vmaf", "ssimulacra2")) -> VmafJob:
    return VmafJob(_info("s.mp4"), _info(name), VmafOptions(), label=name, metric_keys=keys)


def _halves(job, together=False):
    """Each half's backend, queue and passes, as the run line is told them."""
    run = JobRun(JobScheduler([job], gpu_metrics_together=together), 0, job)
    with run.lock:
        return [(task["backend"], task["lane"], task["passes"]) for task in run.task_snapshots()]


def test_the_cpus_and_the_gpus_perceptual_scores_make_one_result(qapp, monkeypatch):
    def output(key, backend):
        metric = FrameMetricResult(key, [0], [0.0], [5.0], MetricProvenance("test", "1", backend, "test-v1"))
        return PerceptualTaskOutput(MetricResultSet([metric]), None, None, 10)

    calls = []

    def perceptual(*args, **kwargs):
        keys = tuple(spec.key for spec in args[3])
        calls.append(keys)
        return output(keys[0], "cpu" if keys == ("ssimulacra2",) else "gpu")

    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", perceptual)
    job = VmafJob(_info("s.mp4"), _info("d.mp4"), VmafOptions(), label="d",
                  metric_keys=("ssimulacra2", "butteraugli"), metric_backends={"ssimulacra2": "cpu"})
    worker = VmafWorker([job])
    finished = []
    worker.job_finished.connect(lambda _index, result: finished.append(result))
    worker.run()
    _drain(qapp)
    assert sorted(calls) == [("butteraugli",), ("ssimulacra2",)]
    [result] = finished
    assert result.has_metric("ssimulacra2") and result.has_metric("butteraugli")


def test_a_videos_finished_half_is_sent_while_its_other_half_runs(qapp, monkeypatch):
    """A video's GPU metrics were done hours before its VMAF, but nothing of
    the video was shown or saved until both were."""
    gpu_sent = threading.Event()

    def ffmpeg(s, d, *a, **k):
        assert gpu_sent.wait(10), "the GPU half's scores were held back"
        return _fake_result(d.path.name)

    monkeypatch.setattr(job_runner, "run_vmaf", ffmpeg)
    monkeypatch.setattr(job_runner, "apply_vship_cpu_fallback", lambda *a, **k: _perceptual_output())
    worker = VmafWorker([_split_job("d.mp4")], parallel_jobs=2)
    updates, finished = [], []
    worker.result_updated.connect(lambda index, result: (updates.append(result), gpu_sent.set()),
                                  Qt.DirectConnection)
    worker.job_finished.connect(lambda index, result: finished.append(result))
    worker.run()
    _drain(qapp)
    assert updates and updates[0].has_metric("ssimulacra2") and not updates[0].has_metric("vmaf")
    assert finished and finished[0].has_metric("ssimulacra2") and finished[0].has_metric("vmaf")
