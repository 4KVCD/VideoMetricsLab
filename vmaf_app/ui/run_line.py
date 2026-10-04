"""A running video's line under the Run button, from its halves' snapshots
(core.job_runner.JobRun.task_snapshots): which of its metrics are under
way, where and how far -- "CPU metrics 1-4 of 4: VMAF v1, PSNR, SSIM, XPSNR
23.3% (9.8 fps, 0:00:22 remaining)" -- its tooltip, and where each video is
decoded.

Functions of the snapshots alone; MainWindow keeps which video has which
line, and when each is drawn again."""
from __future__ import annotations

from dataclasses import dataclass, replace

from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.metrics import metric_definition
from vmaf_app.core.status import GPU_PASS as GPU_PASS_STATUS
from vmaf_app.core.status import STARTING, brief_of, kind_of
from vmaf_app.core.time_format import format_hms
from vmaf_app.i18n import N_, tr, tr_message

#: Each place's name on the run line.
_PLACE_NAMES = {"CPU": N_("CPU metrics"), "GPU": N_("GPU metrics")}


@dataclass(frozen=True)
class MetricUnit:
    """Metrics of a video calculated together -- in one pass, in one place --
    and how far they are: one entry on its run line (metric_units).

    The line used to be built from the programs' halves -- FFmpeg's, Vship's
    -- and counted their passes: with Vship's metrics in one pass, VMAF and
    four GPU metrics read "GPU metrics 1 of 2"."""

    place: str  # "CPU" or "GPU": where they are calculated as the run stands
    keys: tuple[str, ...]
    state: str  # "running", "starting", "waiting", "queued", "done" or "failed"
    task: dict  # the half's snapshot (core.job_runner.JobRun.task_snapshots)
    done: int = 0  # how far: done / whole, rounded down on the line
    whole: int = 0
    scale: int = 1  # whole / scale is the pass's frames: (whole - done) / scale / fps seconds left
    fps: float = 0.0


def step_text(message: str) -> str:
    """A half's status message as its step before figures arrive, e.g.
    "Detecting black bars in source" -- "" for one that only says FFmpeg
    or a GPU pass is starting. The decode plan is left out (its brief):
    it is shown on its own, as "Decoder: ...". Before, a video with
    CPU and GPU halves said only "CPU starting" for as long as black
    bars on a 4K source were being looked for."""
    if kind_of(message) in (STARTING, GPU_PASS_STATUS):
        return ""
    text = brief_of(message).strip().rstrip(".\u2026").strip().replace("distorted", "test video")
    return tr_message(text) if text else ""


def half_place(task: dict[str, object]) -> str:
    """Where a half's metrics are calculated now: "CPU" for the CPU's
    queue, and for a half in the GPU's whose metrics the CPU has taken
    (cpu_keys); "GPU" otherwise."""
    return "CPU" if task.get("lane") == "cpu" or task.get("cpu_keys") else "GPU"


def metric_units(snapshots) -> list[MetricUnit]:
    """A video's metrics in the units they are calculated in -- a pass
    of one or more metrics, in one place -- in the order they run, each
    metric in one unit only: the last it is in (a retry, say).

    From the worker's halves (VmafWorker.task_progress): FFmpeg's
    metrics on the CPU, VMAF and NEG on the GPU, Vship's metrics on the
    GPU in one or more passes, SSIMULACRA2/Butteraugli on the CPU; a
    half in the GPU's queue whose metrics the CPU takes over -- a GPU
    failure -- has them as CPU metrics, with the CPU's figures."""
    units: list[MetricUnit] = []
    for task in snapshots:
        state = str(task.get("state"))
        done_keys = set(task.get("done_keys", ()))
        cpu_keys = tuple(task.get("cpu_keys", ()))
        current, total = int(task.get("current") or 0), int(task.get("total") or 0)
        fps = float(task.get("fps") or 0.0)

        def finished(keys, place, task=task, done_keys=done_keys):
            for kept, outcome in ((tuple(key for key in keys if key in done_keys), "done"),
                                  (tuple(key for key in keys if key not in done_keys), "failed")):
                if kept:
                    units.append(MetricUnit(place, kept, outcome, task))

        def under_way(keys, place, task=task, state=state, current=current, total=total, fps=fps):
            units.append(MetricUnit(place, keys, state, task, current, total, 1, fps))

        if task.get("lane") == "cpu" or cpu_keys:
            # A pass of its metrics on the CPU, after any on the GPU.
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
                # Its passes are one count (run_vship_task): this one
                # from `start`, out of `left` passes of its length.
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
    """Metric numbers as the line shows them: "3", "1\u20132", "1, 3"."""
    numbers = sorted(numbers)
    if len(numbers) > 1 and numbers[-1] - numbers[0] == len(numbers) - 1:
        return f"{numbers[0]}\u2013{numbers[-1]}"
    return ", ".join(map(str, numbers))


def metric_line(snapshots, paused: bool) -> tuple[list[str], str, dict[str, HwAccelPlan]]:
    """A video's run line, by metric: for the CPU and then the GPU, the
    metrics under way, each numbered among that place's metrics, with
    their figures -- "CPU metrics 1-4 of 4: VMAF v1, PSNR, SSIM, XPSNR
    23.3% (9.8 fps, 0:00:22 remaining)", "GPU metrics 1-2 of 5: VMAF
    v0.6.1, VMAF NEG 50.0% (55.0 fps, ...)". Metrics in one pass share
    its figures. Then the tooltip: every metric on a line of its own.

    Built from FFmpeg's and Vship's halves it counted passes -- "GPU
    metrics 1 of 2" for VMAF and Vship's metrics in one pass -- named
    metrics by the program instead of the place, a CPU SSIMULACRA2 in
    Vship's half among the GPU metrics, and took two halves on the CPU
    for two "CPU metrics".

    Last, each half's decode plan by the place it is in (decoder_text),
    two halves in one place that decode differently by their metrics'
    numbers: "Source: GPU (CPU metrics 1-4) / CPU (CPU metrics 5)"."""
    units = metric_units(snapshots)
    order: dict[str, list[str]] = {"CPU": [], "GPU": []}
    for unit in units:
        for key in unit.keys:
            if key not in order[unit.place]:
                order[unit.place].append(key)

    def head(unit: MetricUnit) -> str:
        numbers = sorted(order[unit.place].index(key) + 1 for key in unit.keys)
        labels = ", ".join(metric_definition(order[unit.place][number - 1]).label for number in numbers)
        return tr("{kind} {numbers} of {count}: {labels}", kind=tr(_PLACE_NAMES[unit.place]),
                  numbers=numbers_text(numbers), count=len(order[unit.place]), labels=labels)

    def holder(unit: MetricUnit) -> str:
        """The metrics of this video holding the GPU's turn while `unit`
        waits for it: on the CPU after a GPU failure, in the same turn."""
        for other in units:
            if (other.task is not unit.task and other.state in ("running", "starting")
                    and other.task.get("lane") == "gpu"):
                return ", ".join(metric_definition(key).label for key in other.keys)
        return ""

    parts = []
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
        for unit in shown:
            parts.append(_unit_text(head(unit), unit, paused, holder(unit)))
    lines = []
    for place in ("CPU", "GPU"):
        for key in order[place]:
            unit = next(unit for unit in units if unit.place == place and key in unit.keys)
            line = tr("{kind} {numbers} of {count}: {labels}", kind=tr(_PLACE_NAMES[place]),
                      numbers=order[place].index(key) + 1, count=len(order[place]),
                      labels=metric_definition(key).label)
            lines.append(_unit_tooltip(line, unit, paused))
    # Each half decodes on its own; one that is done no longer does,
    # unless none is left.
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
    """A unit on the run line: `head` ("GPU metrics 1-2 of 5: VMAF
    v0.6.1, VMAF NEG"), then how far it is or what it waits for."""
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
    if unit.state == "failed":  # why: in the video's tooltip when it is over
        return tr("{kind} failed", kind=head)
    details = []
    if paused:
        # Its last rate and time left would read as if it were running.
        details.append(tr("paused"))
    elif unit.fps > 0 and unit.whole > 0:
        details.append(tr("{fps:.1f} fps", fps=unit.fps))
        details.append(tr("{time} remaining",
                          time=format_hms(max(0, unit.whole - unit.done) / unit.scale / unit.fps)))
    return f"{head} {_unit_percent(unit)}" + (f" ({tr(', ').join(details)})" if details else "")


def _unit_tooltip(line: str, unit: MetricUnit, paused: bool) -> str:
    """A metric's line in its video's tooltip: "GPU metrics 3 of 5:
    SSIMULACRA2 40.0% (0:00:15 remaining)", "... (done)"."""
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
    """Rounded down: 99.96% must not read "100.0%" while work remains."""
    return f"{min(1000, 1000 * max(0, unit.done) // unit.whole) / 10 if unit.whole > 0 else 0.0:.1f}%"


def decoder_text(plans: dict[str, HwAccelPlan], *, source_only: bool) -> str:
    """Where each video is decoded, as the run line shows it: "Decoder:
    Source: GPU, test video: CPU".

    `plans` holds each half's decode plan by the half's name on the line
    ("CPU metrics"). The line showed it nearly as it came, "Decode:
    source cuda, test cuda": the decoder API's name where the question
    is only whether the GPU or the CPU decodes each video. Where the
    halves differ -- one fell back to software -- each is named: "test
    video: CPU (CPU metrics) / GPU (GPU metrics)".
    """
    def where(plan: HwAccelPlan) -> dict[str, str]:
        return {"source": "GPU" if plan.source else "CPU", "distorted": "GPU" if plan.distorted else "CPU"}

    by_half = {half: where(plan) for half, plan in plans.items()}

    def side(name: str) -> str:
        found = {half: sides[name] for half, sides in by_half.items()}
        if len(set(found.values())) == 1:
            return next(iter(found.values()))
        return " / ".join(f"{decoder} ({half})" for half, decoder in found.items())

    if source_only:  # a resolution test decodes only the source
        return tr("Decoder: Source: {source}", source=side("source"))
    return tr("Decoder: Source: {source}, test video: {test}", source=side("source"), test=side("distorted"))
