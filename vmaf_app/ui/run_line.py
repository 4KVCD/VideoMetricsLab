"""The progress line shown under the Run button while a video is running.

The line says which of the video's metrics are being calculated, where (on
the CPU or the GPU) and how far along they are, for example:

    CPU metrics 1-4 of 4: VMAF v1, PSNR, SSIM, XPSNR 23.3% (9.8 fps, 0:00:22 remaining)

It is built from snapshots of the video's tasks
(core.job_runner.JobRun.task_snapshots). This module also builds the line's
tooltip and the "Decoder: ..." text that says where each video is decoded.

Everything here is a plain function of the snapshots. MainWindow decides
which video shows which line and when the line is drawn again.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.status import GPU_PASS as GPU_PASS_STATUS
from vmaf_app.core.status import STARTING, brief_of, kind_of
from vmaf_app.core.time_format import format_hms
from vmaf_app.i18n import N_, tr, tr_message

#: How each place is named on the run line.
_PLACE_NAMES = {"CPU": N_("CPU metrics"), "GPU": N_("GPU metrics")}


@dataclass(frozen=True)
class MetricUnit:
    """A group of a video's metrics that are calculated together, in one
    pass and in one place, with how far along they are. Each unit is one
    entry on the video's run line (see metric_units)."""

    place: str  # "CPU" or "GPU": where the metrics are calculated now
    keys: tuple[str, ...]
    state: str  # "running", "starting", "waiting", "queued", "done" or "failed"
    task: dict  # the snapshot of the task the metrics belong to (core.job_runner.JobRun.task_snapshots)
    done: int = 0  # progress is done / whole; the line rounds it down
    whole: int = 0
    scale: int = 1  # whole / scale is the pass's frame count, so (whole - done) / scale / fps is the seconds left
    fps: float = 0.0


def step_text(message: str) -> str:
    """The step a task is on, from its status message, shown before any
    progress figures arrive: for example "Detecting black bars in source".

    Returns "" for a message that only says FFmpeg or a GPU pass is
    starting. The decode plan is left out of the text, because the line
    shows it separately as "Decoder: ...". Without the step, a video with
    CPU and GPU tasks said only "CPU starting" for as long as black bars
    were being looked for on a 4K source."""
    if kind_of(message) in (STARTING, GPU_PASS_STATUS):
        return ""
    text = brief_of(message).strip().rstrip(".\u2026").strip().replace("distorted", "test video")
    return tr_message(text) if text else ""


def half_place(task: dict[str, object]) -> str:
    """Where a task's metrics are calculated now: "CPU" for a task in the
    CPU's queue, and for a task in the GPU's queue whose metrics the CPU
    has taken over (its cpu_keys); "GPU" otherwise."""
    return "CPU" if task.get("lane") == "cpu" or task.get("cpu_keys") else "GPU"


def metric_units(snapshots) -> list[MetricUnit]:
    """Splits a video's metrics into the units they are calculated in, in
    the order they run. A unit is one pass of one or more metrics, in one
    place. Each metric appears in one unit only: the last one it is in, for
    example its retry.

    The snapshots are the video's tasks (VmafWorker.task_progress): FFmpeg's
    metrics on the CPU, VMAF and NEG on the GPU, Vship's metrics on the GPU
    in one or more passes, and SSIMULACRA2 and Butteraugli on the CPU. When
    the CPU takes over the metrics of a task in the GPU's queue (after a GPU
    failure), they are shown as CPU metrics, with the CPU's figures."""
    units: list[MetricUnit] = []
    for task in snapshots:
        state = str(task.get("state"))
        done_keys = set(task.get("done_keys", ()))
        cpu_keys = tuple(task.get("cpu_keys", ()))
        current, total = int(task.get("current") or 0), int(task.get("total") or 0)
        fps = float(task.get("fps") or 0.0)

        def finished(keys, place, task=task, done_keys=done_keys):
            """Adds the units for a task's finished metrics: done or failed."""
            for kept, outcome in ((tuple(key for key in keys if key in done_keys), "done"),
                                  (tuple(key for key in keys if key not in done_keys), "failed")):
                if kept:
                    units.append(MetricUnit(place, kept, outcome, task))

        def under_way(keys, place, task=task, state=state, current=current, total=total, fps=fps):
            """Adds the unit for metrics still being calculated."""
            units.append(MetricUnit(place, keys, state, task, current, total, 1, fps))

        if task.get("lane") == "cpu" or cpu_keys:
            # One pass of the task's metrics on the CPU, after any passes on the GPU.
            for group in task.get("passes", ()) if cpu_keys else ():
                finished(tuple(key for key in group if key not in cpu_keys), "GPU")
            keys = cpu_keys or tuple(task.get("metric_keys", ()))
            if state in ("done", "failed"):
                finished(keys, "CPU")
            else:
                under_way(keys, "CPU")
            continue
        passes = ([tuple(group) for group in task.get("passes", ()) if group]
                  or [tuple(task.get("metric_keys", ()))])
        if state in ("done", "failed"):
            for group in passes:
                finished(group, "GPU")
            continue
        phase = task.get("phase")
        if state != "running" or not phase:
            under_way(passes[0], "GPU")
            units.extend(MetricUnit("GPU", group, "queued", task) for group in passes[1:])
            continue
        number, count, start = phase
        for position, group in enumerate(passes, 1):
            if position < number:
                finished(group, "GPU")
            elif position == number:
                # All of the task's passes share one progress count
                # (run_vship_task). This pass counts from `start`, out of
                # `left` passes of the same length.
                left = max(1, count - number + 1)
                units.append(MetricUnit("GPU", group, "running", task, (current - start) * left,
                                         total - start, left, fps))
            else:
                units.append(MetricUnit("GPU", group, "queued", task))
    kept: list[MetricUnit] = []
    seen: set[str] = set()
    for unit in reversed(units):
        keys = tuple(key for key in unit.keys if key not in seen)
        seen.update(keys)
        if keys:
            kept.append(replace(unit, keys=keys))
    return kept[::-1]


def numbers_text(numbers: list[int]) -> str:
    """Metric numbers as the line shows them: "3", "1\u20132" or "1, 3"."""
    numbers = sorted(numbers)
    if len(numbers) > 1 and numbers[-1] - numbers[0] == len(numbers) - 1:
        return f"{numbers[0]}\u2013{numbers[-1]}"
    return ", ".join(map(str, numbers))


def metric_line(snapshots, paused: bool) -> tuple[list[str], str, dict[str, HwAccelPlan]]:
    """Builds a video's run line, its tooltip and its decode plans.

    The line lists, for the CPU and then the GPU, the metrics under way,
    each numbered among that place's metrics, with their progress, for
    example "CPU metrics 1-4 of 4: VMAF v1, PSNR, SSIM, XPSNR 23.3% (9.8 fps,
    0:00:22 remaining)" and "GPU metrics 1-2 of 5: VMAF v0.6.1, VMAF NEG
    50.0% (55.0 fps, ...)". Metrics calculated in one pass share its
    figures. The tooltip has every metric on a line of its own.

    The decode plans are each task's, keyed by the place the task is in
    (for decoder_text). When two tasks in one place decode differently, each
    key also names its metrics by number: "Source: GPU (CPU metrics 1-4) /
    CPU (CPU metrics 5)"."""
    units = metric_units(snapshots)
    order: dict[str, list[str]] = {"CPU": [], "GPU": []}
    for unit in units:
        for key in unit.keys:
            if key not in order[unit.place]:
                order[unit.place].append(key)

    def head(unit: MetricUnit) -> str:
        """The start of a unit's text, such as "GPU metric 1 of 2: SSIMULACRA2"."""
        numbers = sorted(order[unit.place].index(key) + 1 for key in unit.keys)
        labels = ", ".join(metric_definition(order[unit.place][number - 1]).label for number in numbers)
        return tr("{kind} {numbers} of {count}: {labels}", kind=tr(_PLACE_NAMES[unit.place]),
                  numbers=numbers_text(numbers), count=len(order[unit.place]), labels=labels)

    def holder(unit: MetricUnit) -> str:
        """The metrics of this video whose task holds the GPU's turn while
        `unit` waits for it. That task may be running on the CPU after a GPU
        failure; it keeps the GPU's turn until it finishes."""
        for other in units:
            if (other.task is not unit.task and other.state in ("running", "starting")
                    and other.task.get("lane") == "gpu"):
                return ", ".join(metric_definition(key).label for key in other.keys)
        return ""

    parts = []
    heads = []  # each part's head, and the step it is starting on, if any
    for place in ("CPU", "GPU"):
        if not order[place]:
            continue
        mine = [unit for unit in units if unit.place == place]
        shown = ([unit for unit in mine if unit.state in ("running", "starting")]
                 or [unit for unit in mine if unit.state == "waiting"][:1]
                 or [unit for unit in mine if unit.state == "queued"][:1]
                 or [unit for unit in mine if unit.state == "failed"])
        if not shown:
            parts.append(tr("{kind} done", kind=tr(_PLACE_NAMES[place])))
            heads.append(("", ""))
        for unit in shown:
            parts.append(_unit_text(head(unit), unit, paused, holder(unit)))
            heads.append((head(unit), step_text(unit.task.get("step") or "") if unit.state == "starting" else ""))
    # Every part starting on the same step (the CPU's and the GPU's metrics
    # waiting for the black bars they share): the step said once, first,
    # rather than after each part.
    steps = {step for _head, step in heads}
    if len(parts) > 1 and len(steps) == 1 and "" not in steps:
        parts = [steps.pop()] + [part_head for part_head, _step in heads]
    lines = []
    for place in ("CPU", "GPU"):
        for key in order[place]:
            unit = next(unit for unit in units if unit.place == place and key in unit.keys)
            line = tr("{kind} {numbers} of {count}: {labels}", kind=tr(_PLACE_NAMES[place]),
                      numbers=order[place].index(key) + 1, count=len(order[place]),
                      labels=metric_definition(key).label)
            lines.append(_unit_tooltip(line, unit, paused))
    # Each task decodes the videos itself. A finished task no longer
    # decodes, so it is left out unless every task has finished.
    decoding = ([task for task in snapshots if task.get("decode") and task.get("state") not in ("done", "failed")]
                or [task for task in snapshots if task.get("decode")])
    places = [half_place(task) for task in decoding]
    plans: dict[str, HwAccelPlan] = {}
    for task, place in zip(decoding, places, strict=True):
        name = tr(_PLACE_NAMES[place])
        if any(other is not task and where == place and other["decode"] != task["decode"]
               for other, where in zip(decoding, places, strict=True)):
            keys = task.get("cpu_keys") or task.get("metric_keys", ())
            name += " " + numbers_text([order[place].index(key) + 1 for key in keys if key in order[place]])
        plans[name] = task["decode"]
    return parts, "\n".join(lines), plans


def _unit_text(head: str, unit: MetricUnit, paused: bool, holder: str) -> str:
    """One unit's entry on the run line: `head` (for example "GPU metrics
    1-2 of 5: VMAF v0.6.1, VMAF NEG"), then its progress or what it is
    waiting for."""
    if unit.state == "waiting":
        if unit.task.get("waiting_for") == "CPU":
            return tr("{kind} queued (waiting for a free CPU slot)", kind=head)
        if holder:
            return tr("{kind} queued (waiting for this video's {other} to finish)", kind=head, other=holder)
        return tr("{kind} queued (another video is using the GPU)", kind=head)
    if unit.state in ("starting", "queued"):
        step = step_text(unit.task.get("step") or "") if unit.state == "starting" else ""
        return tr("{kind} ({step})", kind=head, step=step) if step else tr("{kind} starting", kind=head)
    if unit.state == "done":
        return tr("{kind} done", kind=head)
    if unit.state == "failed":  # the reason is in the video's tooltip once the run is over
        return tr("{kind} failed", kind=head)
    details = []
    if paused:
        # The last speed and time left would read as if the unit were still running.
        details.append(tr("paused"))
    elif unit.fps > 0 and unit.whole > 0:
        details.append(tr("{fps:.1f} fps", fps=unit.fps))
        details.append(tr("{time} remaining",
                          time=format_hms(max(0, unit.whole - unit.done) / unit.scale / unit.fps)))
    return f"{head} {_unit_percent(unit)}" + (f" ({tr(', ').join(details)})" if details else "")


def _unit_tooltip(line: str, unit: MetricUnit, paused: bool) -> str:
    """One metric's line in its video's tooltip, for example "GPU metrics 3
    of 5: SSIMULACRA2 40.0% (0:00:15 remaining)" or "... (done)"."""
    if unit.state == "done":
        return tr("{label} (done)", label=line)
    if unit.state == "failed":
        return tr("{label} (failed)", label=line)
    if unit.state == "running":
        text = f"{line} {_unit_percent(unit)}"
        if paused:
            return f"{text} ({tr('paused')})"
        if unit.fps > 0 and unit.whole > 0:
            seconds = max(0, unit.whole - unit.done) / unit.scale / unit.fps
            return f"{text} ({tr('{time} remaining', time=format_hms(seconds))})"
        return text
    if unit.state == "starting":
        return tr("{kind} starting", kind=line)
    return tr("{label} (queued)", label=line)


def _unit_percent(unit: MetricUnit) -> str:
    """The unit's progress as a percentage, rounded down, so that 99.96%
    does not read "100.0%" while work remains."""
    return f"{min(1000, 1000 * max(0, unit.done) // unit.whole) / 10 if unit.whole > 0 else 0.0:.1f}%"


def decoder_text(plans: dict[str, HwAccelPlan], *, source_only: bool) -> str:
    """Where each video is decoded, as the run line shows it: "Decoder:
    Source: GPU, test video: CPU".

    `plans` holds each task's decode plan, keyed by the task's name on the
    line ("CPU metrics"). Only whether the GPU or the CPU decodes each video
    is shown, not the decoder's name. When tasks decode a video differently
    (one fell back to the CPU, say), each is named: "test video: CPU (CPU
    metrics) / GPU (GPU metrics)".
    """
    def where(plan: HwAccelPlan) -> dict[str, str]:
        """Where each video is decoded in one plan: "GPU" or "CPU"."""
        return {"source": "GPU" if plan.source else "CPU", "distorted": "GPU" if plan.distorted else "CPU"}

    by_half = {half: where(plan) for half, plan in plans.items()}

    def side(name: str) -> str:
        """Where one video is decoded, by task when the tasks differ."""
        found = {half: sides[name] for half, sides in by_half.items()}
        if len(set(found.values())) == 1:
            return next(iter(found.values()))
        return " / ".join(f"{decoder} ({half})" for half, decoder in found.items())

    if source_only:  # a resolution test decodes only the source
        return tr("Decoder: Source: {source}", source=side("source"))
    return tr("Decoder: Source: {source}, test video: {test}", source=side("source"), test=side("distorted"))
