"""Probing video files with ffprobe."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffprobe_path
from vmaf_app.core.models import VideoInfo
from vmaf_app.core.process_control import ProcessHandle


class ProbeError(RuntimeError):
    pass


class ProbeCancelled(RuntimeError):  # noqa: N818 - expected control flow
    """Raised when a probe is abandoned on purpose, e.g. at shutdown."""


def _parse_frame_rate(rate_str: str) -> float:
    """A rate as ffprobe prints it ("24000/1001", "25"); 0 for one it
    could not tell ("N/A", "0/0"), as for a missing one."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            num, den = float(num), float(den)
            return num / den if den else 0.0
        return float(rate_str)
    except ValueError:
        return 0.0


def _video_stream(streams: list[dict]) -> dict | None:
    """The first video stream that is a video: not cover art or a
    thumbnail, which FFmpeg lists as video streams too -- an MP4's cover
    (covr) can come before its video track. The same stream FFmpeg's
    commands read (ffmpeg_locate.VIDEO_STREAM)."""
    for stream in streams:
        disposition = stream.get("disposition") or {}
        if (stream.get("codec_type") == "video" and not disposition.get("attached_pic")
                and not disposition.get("timed_thumbnails")):
            return stream
    return None


def _tag(stream: dict, name: str) -> str | None:
    """A stream tag, by its name in any case (Matroska writers differ)."""
    for key, value in (stream.get("tags") or {}).items():
        if key.upper() == name:
            return value
    return None


def _clock_seconds(text: str | None) -> float | None:
    """ "01:45:36.289000000" in seconds, or None."""
    try:
        hours, minutes, seconds = (text or "").split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _whole(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _stream_bit_rate(stream: dict, duration: float, file_size: int) -> int:
    """One stream's own bitrate, from what the container records: its
    bit_rate, or Matroska's statistics tags (BPS, or NUMBER_OF_BYTES over
    its length). 0 when there is none.

    The statistics are used only if they fit in the file: FFmpeg copies
    them from its input into what it writes, even through a re-encode, so a
    30 s cut of a film carried the film's 55 GB and 69.8 Mb/s."""
    rate = _whole(stream.get("bit_rate"))
    if rate > 0:
        return rate
    size = _whole(_tag(stream, "NUMBER_OF_BYTES"))
    tagged = _whole(_tag(stream, "BPS")) or (round(size * 8 / duration) if size > 0 and duration > 0 else 0)
    if tagged <= 0 or file_size <= 0 or size > file_size or tagged * duration / 8 > file_size * 1.02:
        return 0
    return tagged


def _video_bit_rate(video: dict, streams: list[dict], fmt: dict, duration: float) -> tuple[int, bool]:
    """(the video stream's bitrate, whether it is the whole file's instead).

    A Matroska file records no bitrate per stream unless its writer added
    statistics tags, and the file's own rate counts the soundtrack: a film
    with a TrueHD track read 4-5 Mb/s high. Without the video's own, the
    audio tracks' rates are taken from the file's when every one is known;
    otherwise the file's rate is all there is, and says so."""
    file_size = _whole(fmt.get("size"))
    own = _stream_bit_rate(video, duration, file_size)
    if own:
        return own, False
    total = _whole(fmt.get("bit_rate"))
    if total <= 0:
        return 0, False
    try:
        container = float(fmt.get("duration") or duration)
    except ValueError:
        container = duration
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    rates = [_stream_bit_rate(stream, _stream_duration(stream, container), file_size) for stream in audio]
    if all(rates) and total > sum(rates):
        return total - sum(rates), False
    return total, True


def _stream_duration(stream: dict, container: float) -> float:
    """The video's own length. A Matroska file gives no stream duration,
    and the container's is its longest stream's -- an audio track running
    on after the picture, which then failed "Durations do not match", or
    made a whole video read as cut short. Its writers record each track's
    length as a DURATION tag; one longer than the container is stale (left
    by a remux that cut the file) and is not used."""
    if stream.get("duration"):
        try:
            return float(stream["duration"])
        except ValueError:
            pass
    tagged = _clock_seconds(_tag(stream, "DURATION"))
    if tagged is not None and tagged > 0 and (container <= 0 or tagged <= container + 0.5):
        return tagged
    return container


def probe_video(path: Path, process_handle: ProcessHandle | None = None) -> VideoInfo:
    """Reads `path`'s media info.

    `process_handle` makes the probe abandonable. ffprobe on a large file on
    a slow or network disk can take many seconds, and a plain
    subprocess.run() cannot be interrupted -- the flag a canceller sets is
    invisible to a call already blocked inside it. That is what let the
    window be torn down with an ffprobe still running underneath it.
    """
    cmd = [
        ffprobe_path(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        # ffprobe writes its JSON as UTF-8. Left to the default, Python decodes
        # it with the Windows code page (cp1252), which has no character for
        # some UTF-8 bytes: a title tag or file name with a curly quote (”)
        # killed the reader thread and left stdout as None.
        proc = proc_util.popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        raise ProbeError(
            "ffprobe was not found. Make sure ffmpeg is installed and on PATH, "
            "or set a custom ffmpeg folder in Settings."
        ) from e
    except OSError as e:
        raise ProbeError(f"Could not start ffprobe for {path}: {e}") from e

    if process_handle is not None:
        process_handle.attach(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired as e:
        proc_util.kill(proc)
        proc.communicate()
        raise ProbeError(f"ffprobe timed out while reading {path}") from e
    finally:
        if process_handle is not None:
            process_handle.detach()

    if proc.returncode != 0:
        if process_handle is not None and process_handle.was_terminated:
            raise ProbeCancelled(f"Probe of {path} was cancelled")
        raise ProbeError(f"ffprobe failed for {path}:\n{stderr.strip()}")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ProbeError(f"Could not parse ffprobe output for {path}") from e

    streams = data.get("streams", [])
    v = _video_stream(streams)
    if v is None:
        raise ProbeError(f"No video stream found in {path}")
    fmt = data.get("format", {})

    fps = _parse_frame_rate(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = _parse_frame_rate(v.get("r_frame_rate") or "0/1")
    nominal_fps = _parse_frame_rate(v.get("r_frame_rate") or "0/1")
    average_fps = _parse_frame_rate(v.get("avg_frame_rate") or "0/1")

    try:
        container = float(fmt.get("duration") or 0.0)
    except ValueError:
        container = 0.0
    duration = _stream_duration(v, container)

    nb_frames = 0
    if v.get("nb_frames"):
        try:
            nb_frames = int(v["nb_frames"])
        except ValueError:
            nb_frames = 0

    bit_rate, bit_rate_whole_file = _video_bit_rate(v, streams, fmt, duration)

    return VideoInfo(
        path=path,
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
        codec_name=v.get("codec_name", "unknown"),
        sar=v.get("sample_aspect_ratio", "1:1") or "1:1",
        pix_fmt=v.get("pix_fmt", ""),
        bit_rate=bit_rate,
        bit_rate_whole_file=bit_rate_whole_file,
        nominal_fps=nominal_fps,
        average_fps=average_fps,
        color_range=v.get("color_range", "") or "",
        color_space=v.get("color_space", "") or "",
        color_transfer=v.get("color_transfer", "") or "",
        color_primaries=v.get("color_primaries", "") or "",
        chroma_location=v.get("chroma_location", "") or "",
        format_name=fmt.get("format_name", "") or "",
    )
