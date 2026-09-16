"""Packet-level video bitrate analysis and plot aggregation.

Only the first video stream is examined.  Audio, subtitles and container
overhead are deliberately excluded: this viewer answers how the encoded
video stream spends its bits, not how large the whole file is.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffprobe_path
from vmaf_app.core.models import VideoInfo
from vmaf_app.core.process_control import ProcessHandle

BitrateProgress = Callable[[int, int], None]


class BitrateError(RuntimeError):
    pass


class BitrateCancelled(BitrateError):  # noqa: N818
    """Packet scanning was deliberately cancelled."""


@dataclass(slots=True)
class BitrateData:
    path: Path
    times: np.ndarray
    durations: np.ndarray
    sizes: np.ndarray
    keyframes: np.ndarray
    # Derived once, kept for the life of the data. The per-second bins walk
    # every packet in Python -- 290 ms for a 151,919-packet encode -- and
    # the table asked for them again on every refresh, for every row: a
    # five-row table cost 1.4 s of UI thread each time it redrew. The scan
    # that produced this data took minutes; its summary is not worth
    # recomputing, and analyze_video_bitrate primes it on the worker so the
    # UI thread never computes it at all. Keyed by adjust_start, because the
    # bin edges move with it.
    _second_bins: dict = field(default_factory=dict, repr=False, compare=False)
    _summary: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.durations = np.asarray(self.durations, dtype=np.float64)
        self.sizes = np.asarray(self.sizes, dtype=np.int64)
        self.keyframes = np.asarray(self.keyframes, dtype=np.bool_)
        length = len(self.times)
        if not all(len(column) == length for column in (
            self.durations, self.sizes, self.keyframes,
        )):
            raise ValueError("Bitrate packet columns must have the same length")

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def start_time(self) -> float:
        return float(self.times[0]) if self.frame_count else 0.0

    @property
    def end_time(self) -> float:
        if not self.frame_count:
            return 0.0
        return float(np.max(self.times + self.durations))

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def total_bytes(self) -> int:
        return int(self.sizes.sum(dtype=np.int64))

    @property
    def average_kbps(self) -> float:
        return self.total_bytes * 8 / self.duration / 1000 if self.duration > 0 else 0.0

    def second_bins(self, adjust_start: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(starts, ends, kb/s) per one-second bin, computed once per variant."""
        bins = self._second_bins.get(adjust_start)
        if bins is None:
            bins = _compute_second_bins(self, adjust_start)
            self._second_bins[adjust_start] = bins
        return bins

    def summary(self) -> BitrateSummary:
        if self._summary is None:
            _starts, _ends, rates = self.second_bins(True)
            self._summary = BitrateSummary(
                average_kbps=self.average_kbps,
                minimum_kbps=float(np.min(rates)) if len(rates) else 0.0,
                maximum_kbps=float(np.max(rates)) if len(rates) else 0.0,
            )
        return self._summary

    def prime(self) -> None:
        """Computes what the table and the default plot will ask for, so the
        thread that produced the data pays for its summary as well."""
        self.summary()


@dataclass(frozen=True, slots=True)
class BitratePlot:
    times: np.ndarray
    values: np.ndarray
    axis_label: str
    value_label: str


@dataclass(frozen=True, slots=True)
class BitrateSummary:
    average_kbps: float
    minimum_kbps: float
    maximum_kbps: float


def _number(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in {None, "", "N/A"} else None
    except ValueError:
        return None
    return parsed if parsed is not None and np.isfinite(parsed) else None


def _parse_compact_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in line.strip().split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def analyze_video_bitrate(
    info: VideoInfo,
    process_handle: ProcessHandle | None = None,
    on_progress: BitrateProgress | None = None,
) -> BitrateData:
    """Scan first-video-stream packets with ffprobe, ordered by presentation time."""
    if not info.path.is_file():
        raise BitrateError(f"Video file is missing: {info.path}")
    fallback_duration = 1.0 / info.fps if info.fps > 0 else 0.0
    command = [
        ffprobe_path(), "-v", "error", "-select_streams", "v:0",
        "-show_packets",
        "-show_entries", "packet=pts_time,dts_time,duration_time,size,pos,flags",
        "-of", "compact=p=0:nk=0", str(info.path),
    ]
    handle = process_handle or ProcessHandle()
    try:
        process = proc_util.popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except (OSError, FileNotFoundError) as exc:
        raise BitrateError(f"Could not start ffprobe: {exc}") from exc

    handle.attach(process.pid)
    packets: list[tuple[float | None, float | None, float, int, int, bool]] = []
    total = max(1, info.estimated_frame_count)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            fields = _parse_compact_line(line)
            try:
                size = int(fields.get("size", ""))
            except ValueError:
                continue
            pts = _number(fields.get("pts_time"))
            dts = _number(fields.get("dts_time"))
            duration = _number(fields.get("duration_time")) or fallback_duration
            try:
                position = int(fields.get("pos", len(packets)))
            except ValueError:
                position = len(packets)
            packets.append((pts, dts, max(0.0, duration), size, position,
                            "K" in fields.get("flags", "")))
            if on_progress is not None and len(packets) % 500 == 0:
                on_progress(len(packets), total)
        return_code = process.wait()
        stderr = process.stderr.read() if process.stderr is not None else ""
    finally:
        handle.detach()

    if return_code != 0:
        if handle.was_terminated:
            raise BitrateCancelled("Bitrate analysis was cancelled.")
        raise BitrateError(stderr.strip()[-2000:] or f"ffprobe exited with code {return_code}.")
    if not packets:
        raise BitrateError("ffprobe returned no packets for the first video stream.")

    # Packets are emitted in decode order.  Plotting and GOP grouping use
    # presentation order, so B-frames need a stable re-order by PTS.
    next_time = 0.0
    resolved: list[tuple[float, float, int, int, bool]] = []
    for pts, dts, duration, size, position, keyframe in packets:
        timestamp = pts if pts is not None else dts
        if timestamp is None:
            timestamp = next_time
        resolved.append((timestamp, duration, size, position, keyframe))
        next_time = max(next_time, timestamp + duration)
    resolved.sort(key=lambda packet: (packet[0], packet[3]))

    if on_progress is not None:
        on_progress(len(resolved), total)
    return BitrateData(
        path=info.path,
        times=np.fromiter((p[0] for p in resolved), dtype=np.float64),
        durations=np.fromiter((p[1] for p in resolved), dtype=np.float64),
        sizes=np.fromiter((p[2] for p in resolved), dtype=np.int64),
        keyframes=np.fromiter((p[4] for p in resolved), dtype=np.bool_),
    )


def _step_points(starts: np.ndarray, ends: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(values):
        return np.empty(0, np.float64), np.empty(0, np.float32)
    times = np.empty(len(values) * 2, np.float64)
    plotted = np.empty(len(values) * 2, np.float32)
    times[0::2], times[1::2] = starts, ends
    plotted[0::2], plotted[1::2] = values, values
    return times, plotted


def frame_plot(data: BitrateData, adjust_start: bool = True) -> BitratePlot:
    offset = data.start_time if adjust_start else 0.0
    return BitratePlot(
        times=data.times - offset,
        values=(data.sizes.astype(np.float64) * 8 / 1000).astype(np.float32),
        axis_label="Frame size (kbit)", value_label="kbit",
    )


def _compute_second_bins(data: BitrateData, adjust_start: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The raw computation. Callers go through BitrateData.second_bins."""
    if not data.frame_count:
        empty = np.empty(0, np.float64)
        return empty, empty, np.empty(0, np.float32)
    offset = data.start_time if adjust_start else np.floor(data.start_time)
    relative_start = data.start_time - offset
    relative_end = data.end_time - offset
    first_bin = int(np.floor(relative_start))
    last_bin = int(np.ceil(relative_end))
    starts = np.arange(first_bin, last_bin, dtype=np.float64)
    ends = starts + 1.0
    byte_counts = np.zeros(len(starts), dtype=np.float64)

    for start, duration, size in zip(data.times - offset, data.durations, data.sizes, strict=True):
        end = start + duration
        if duration <= 0:
            index = min(len(starts) - 1, max(0, int(np.floor(start)) - first_bin))
            byte_counts[index] += size
            continue
        begin_index = max(0, int(np.floor(start)) - first_bin)
        end_index = min(len(starts) - 1, int(np.ceil(end)) - 1 - first_bin)
        for index in range(begin_index, end_index + 1):
            overlap = max(0.0, min(end, ends[index]) - max(start, starts[index]))
            byte_counts[index] += size * overlap / duration

    coverage = np.maximum(
        1e-12,
        np.minimum(ends, relative_end) - np.maximum(starts, relative_start),
    )
    rates = (byte_counts * 8 / coverage / 1000).astype(np.float32)
    # With start adjustment off, restore the actual timeline on the X axis.
    timeline_offset = 0.0 if adjust_start else offset
    return starts + timeline_offset, ends + timeline_offset, rates


def second_plot(data: BitrateData, adjust_start: bool = True) -> BitratePlot:
    starts, ends, rates = data.second_bins(adjust_start)
    times, values = _step_points(starts, ends, rates)
    return BitratePlot(times, values, "Video bitrate (kb/s)", "kb/s")


def gop_plot(data: BitrateData, adjust_start: bool = True) -> BitratePlot:
    if not data.frame_count:
        return BitratePlot(
            np.empty(0, np.float64), np.empty(0, np.float32),
            "GOP bitrate (kb/s)", "kb/s",
        )
    starts_at = [0]
    starts_at.extend(int(i) for i in np.flatnonzero(data.keyframes) if i > 0)
    starts: list[float] = []
    ends: list[float] = []
    rates: list[float] = []
    for group, begin in enumerate(starts_at):
        finish = starts_at[group + 1] if group + 1 < len(starts_at) else data.frame_count
        start = float(data.times[begin])
        end = float(np.max(data.times[begin:finish] + data.durations[begin:finish]))
        duration = end - start
        size = int(data.sizes[begin:finish].sum(dtype=np.int64))
        starts.append(start)
        ends.append(end)
        rates.append(size * 8 / duration / 1000 if duration > 0 else 0.0)
    offset = data.start_time if adjust_start else 0.0
    times, values = _step_points(
        np.asarray(starts) - offset,
        np.asarray(ends) - offset,
        np.asarray(rates, dtype=np.float32),
    )
    return BitratePlot(times, values, "GOP bitrate (kb/s)", "kb/s")


def bitrate_summary(data: BitrateData) -> BitrateSummary:
    return data.summary()
