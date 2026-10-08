"""vmaf_app.core.status: status messages with their data beside their words."""
from __future__ import annotations

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.isolated import run_isolated
from vmaf_app.core.status import Status, plan_of


def _report(on_status=None):
    on_status(Status.decoding("Running VMAF on the GPU", HwAccelPlan("cuda", "cuda"), ending="..."))


def test_a_status_from_a_child_process_arrives_with_its_plan():
    """The GPU code runs in processes of its own (core.isolated)."""
    received = []
    run_isolated(_report, what="test", callbacks=("on_status",), on_status=received.append)
    assert plan_of(received[0]) == HwAccelPlan("cuda", "cuda")
