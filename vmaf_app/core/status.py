"""Status messages that carry, besides their text, the facts the app reads
from them.

A run reports what it is doing as text, for the window's run line and for
the log. The app also needs some facts from these reports: where each video
is decoded, whether VMAF on the GPU failed, and whether a step only says
FFmpeg is starting. A Status carries those facts as data next to the text,
so the app never has to parse its own English messages.

Status is a str, so code that only shows or logs the text needs nothing
new, and it keeps its data when pickled across the processes the GPU code
runs in (core.isolated).
"""
from __future__ import annotations

from vmaf_app.core.gpu import HwAccelPlan

#: The kinds of status the app acts on (Status.kind).
STARTING = "starting"          # FFmpeg is starting: nothing to show as a step
GPU_PASS = "gpu_pass"          # one of several GPU metric passes begins
GPU_WAIT = "gpu_wait"          # waiting for another video's GPU pass to end
GPU_VMAF_FAILED = "gpu_vmaf_failed"  # VMAF on the GPU failed; the CPU takes over


class Status(str):
    """A status message (`text`) with its facts: `plan`, the decode plan the
    step runs with (or None); `kind`, one of the kinds above (or ""); and
    `brief`, the text without the decode plan, which the run line shows
    separately."""

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
        """A status for a step that decodes: `brief` followed by the decode
        plan, as in "Running ffmpeg (GPU decode: source cuda, distorted
        cpu)…"."""
        return cls(f"{brief} (GPU decode: {plan.describe()}){ending}", plan=plan, kind=kind, brief=brief)

    def __reduce__(self):
        return _rebuild, (str(self), self.plan, self.kind, self.brief)


def _rebuild(text: str, plan: HwAccelPlan | None, kind: str, brief: str) -> Status:
    """Rebuilds a Status from its parts (for pickling across processes)."""
    return Status(text, plan=plan, kind=kind, brief=brief)


def plan_of(message: str) -> HwAccelPlan | None:
    """The decode plan a status carries, if it is a Status that has one."""
    return getattr(message, "plan", None)


def kind_of(message: str) -> str:
    """The kind of a status ("" for plain text)."""
    return getattr(message, "kind", "")


def brief_of(message: str) -> str:
    """A status's text without its decode plan (plain text as it is)."""
    return getattr(message, "brief", message)
