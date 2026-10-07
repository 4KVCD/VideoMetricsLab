"""The job queue without Qt: JobScheduler reports through plain callables.

The scheduling itself -- lanes, pausing, cancelling, halves -- is tested
through the window's VmafWorker in test_worker.py; these make sure the core
runs on its own, as a script or another front end would use it."""
import threading
from pathlib import Path

from vmaf_app.core import job_runner
from vmaf_app.core.job_runner import JobScheduler, RunEvents, VmafJob
from vmaf_app.core.models import ComparisonResult, FrameScore, VideoInfo, VmafOptions
from vmaf_app.core.vmaf_runner import Cancelled


def _info(name: str) -> VideoInfo:
    return VideoInfo(path=Path(name), width=1920, height=1080, fps=30.0, duration=5.0, nb_frames=150,
                     codec_name="h264")


def _result(name: str) -> ComparisonResult:
    info = _info(name)
    return ComparisonResult(
        source=Path("s.mp4"), distorted=Path(name), frames=[FrameScore(frame=0, time=0.0, vmaf=90.0)], fps=30.0,
        model="version=vmaf_v0.6.1", source_crop=None, distorted_crop=None, source_info=info, distorted_info=info,
    )


def _job(name: str) -> VmafJob:
    return VmafJob(_info("s.mp4"), _info(name), VmafOptions(), label=name)


def test_a_run_reports_each_video_through_plain_callables(monkeypatch):
    monkeypatch.setattr(job_runner, "run_vmaf", lambda s, d, *a, **k: _result(d.path.name))
    seen = []
    events = RunEvents(
        job_started=lambda index, label: seen.append(("started", index, label)),
        job_finished=lambda index, result: seen.append(("finished", index, result.distorted.name)),
        all_finished=lambda: seen.append(("all finished",)),
    )

    JobScheduler([_job("d0.mp4"), _job("d1.mp4")], parallel_jobs=2, events=events).run()

    # The two videos' lanes run side by side: only the last event's place is fixed.
    assert seen[-1] == ("all finished",)
    assert sorted(seen[:-1]) == [("finished", 0, "d0.mp4"), ("finished", 1, "d1.mp4"),
                                 ("started", 0, "d0.mp4"), ("started", 1, "d1.mp4")]


def test_a_cancelled_run_says_so_once_however_many_videos_stopped(monkeypatch):
    def cancelled(*_args, **_kwargs):
        raise Cancelled("cancelled")

    monkeypatch.setattr(job_runner, "run_vmaf", cancelled)
    seen = []

    JobScheduler([_job("d0.mp4"), _job("d1.mp4")], parallel_jobs=2,
                 events=RunEvents(cancelled=lambda: seen.append("cancelled"))).run()

    assert seen == ["cancelled"]


def test_a_run_cancelled_before_it_starts_starts_nothing(monkeypatch):
    monkeypatch.setattr(job_runner, "run_vmaf", lambda *a, **k: _result("d0.mp4"))
    seen = []
    scheduler = JobScheduler([_job("d0.mp4")], events=RunEvents(
        job_started=lambda index, label: seen.append("started"), cancelled=lambda: seen.append("cancelled")))
    scheduler.cancel()

    scheduler.run()

    assert seen == ["cancelled"]


def test_events_left_out_are_not_needed(monkeypatch):
    monkeypatch.setattr(job_runner, "run_vmaf", lambda s, d, *a, **k: _result(d.path.name))

    JobScheduler([_job("d0.mp4")]).run()


def test_a_task_cancel_token_waits_as_an_event_does():
    """The token goes where a threading.Event is taken (cancel_event): it
    answers wait() as one, for the run's cancel and for the job's own."""
    run = threading.Event()
    token = job_runner._TaskCancelToken(run)
    assert token.wait(0) is False and not token.is_set()
    run.set()
    assert token.wait(0) is True and token.wait() is True
    token = job_runner._TaskCancelToken(threading.Event())
    token.cancel_job()
    assert token.wait(0) is True and token.is_set()


def test_a_task_cancel_token_wakes_a_waiter_when_the_run_is_cancelled():
    run = threading.Event()
    token = job_runner._TaskCancelToken(run)
    woke = []
    waiter = threading.Thread(target=lambda: woke.append(token.wait(30)))
    waiter.start()
    run.set()
    waiter.join(30)
    assert woke == [True]
