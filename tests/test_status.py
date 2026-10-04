"""vmaf_app.core.status: status messages with their data beside their words."""
from __future__ import annotations

import pickle

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.isolated import run_isolated
from vmaf_app.core.status import STARTING, Status, brief_of, kind_of, plan_of


def test_a_decoding_status_names_its_plan_in_its_words_and_carries_it():
    plan = HwAccelPlan(source="cuda")
    message = Status.decoding("Running ffmpeg", plan, ending="...", kind=STARTING)
    assert message == "Running ffmpeg (GPU decode: source cuda, distorted cpu)..."
    assert (plan_of(message), kind_of(message), brief_of(message)) == (plan, STARTING, "Running ffmpeg")


def test_plain_text_has_no_plan_or_kind_and_is_its_own_brief():
    assert (plan_of("Detecting black bars"), kind_of("Detecting black bars"),
            brief_of("Detecting black bars")) == (None, "", "Detecting black bars")


def test_a_status_pickles_with_its_data():
    message = Status.decoding("Vship GPU (x): calculating SSIMULACRA2", HwAccelPlan(distorted="qsv"))
    back = pickle.loads(pickle.dumps(message))
    assert back == message and plan_of(back) == HwAccelPlan(distorted="qsv") and brief_of(back) == message.brief


def _report(on_status=None):
    on_status(Status.decoding("Running VMAF on the GPU", HwAccelPlan("cuda", "cuda"), ending="..."))


def test_a_status_from_a_child_process_arrives_with_its_plan():
    """The GPU code runs in processes of its own (core.isolated)."""
    received = []
    run_isolated(_report, what="test", callbacks=("on_status",), on_status=received.append)
    assert plan_of(received[0]) == HwAccelPlan("cuda", "cuda")
