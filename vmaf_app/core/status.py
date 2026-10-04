"""Status messages that carry, beside their words, what the app reads from them.

A run reports what it is doing as text for the window's run line and the
log. Some of it was also read by the app -- where each video is decoded,
from "(GPU decode: source cuda, distorted cpu)"; that VMAF on the GPU had
failed, from the words it began with; that a step only said FFmpeg was
starting -- by searching the English words with regular expressions and
prefixes. Reworded, or a step added, and the app quietly misread its own
messages. A Status is that text, and those facts as data.

It is a str, so whatever only shows or logs the text needs nothing new, and
it pickles with its data across the processes the GPU code runs in
(core.isolated).
"""
from __future__ import annotations

from vmaf_app.core.gpu import HwAccelPlan

#: What a status says that the app acts on (Status.kind).
STARTING = "starting"          # FFmpeg is starting: nothing to show as a step
GPU_PASS = "gpu_pass"          # one of several GPU metric passes begins
GPU_WAIT = "gpu_wait"          # waiting for another video's GPU pass to end
GPU_VMAF_FAILED = "gpu_vmaf_failed"  # VMAF on the GPU failed; the CPU takes over


class Status(str):
    """`text`, with `plan`, the decode plan the step runs with (or None),
    `kind`, one of the kinds above (or ""), and `brief`, the text without
    the plan, which the run line shows on its own."""

    plan: HwAccelPlan | None
    kind: str
    brief: str

    def __new__(cls, text: str, *, plan: HwAccelPlan | None = None, kind: str = "",
                brief: str | None = None) -> Status:
        status = super().__new__(cls, text)
        status.plan, status.kind, status.brief = plan, kind, text if brief is None else brief
        return status

    @classmethod
    def decoding(cls, brief: str, plan: HwAccelPlan, *, ending: str = "…", kind: str = "") -> Status:
        """`brief` with the decode plan it runs with: "Running ffmpeg (GPU
        decode: source cuda, distorted cpu)…"."""
        return cls(f"{brief} (GPU decode: {plan.describe()}){ending}", plan=plan, kind=kind, brief=brief)

    def __reduce__(self):
        return _rebuild, (str(self), self.plan, self.kind, self.brief)


def _rebuild(text: str, plan: HwAccelPlan | None, kind: str, brief: str) -> Status:
    return Status(text, plan=plan, kind=kind, brief=brief)


def plan_of(message: str) -> HwAccelPlan | None:
    """The decode plan a status carries, if it is a Status that has one."""
    return getattr(message, "plan", None)


def kind_of(message: str) -> str:
    return getattr(message, "kind", "")


def brief_of(message: str) -> str:
    return getattr(message, "brief", message)
