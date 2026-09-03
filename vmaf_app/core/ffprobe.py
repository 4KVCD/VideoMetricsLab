"""Probing video files with ffprobe."""
from __future__ import annotations

import json
from pathlib import Path

from vmaf_app.core import proc as proc_util
from vmaf_app.core.ffmpeg_locate import ffprobe_path
from vmaf_app.core.models import VideoInfo


class ProbeError(RuntimeError):
    pass


def _parse_frame_rate(rate_str: str) -> float:
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        num, den = float(num), float(den)
        return num / den if den else 0.0
    return float(rate_str)


def probe_video(path: Path) -> VideoInfo:
    cmd = [
        ffprobe_path(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = proc_util.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise ProbeError(
            "ffprobe was not found. Make sure ffmpeg is installed and on PATH, "
            "or set a custom ffmpeg folder in Settings."
        ) from e

    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}:\n{proc.stderr.strip()}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ProbeError(f"Could not parse ffprobe output for {path}") from e

    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ProbeError(f"No video stream found in {path}")
    v = video_streams[0]
    fmt = data.get("format", {})

    fps = _parse_frame_rate(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
    if fps <= 0:
        fps = _parse_frame_rate(v.get("r_frame_rate") or "0/1")

    duration = float(v.get("duration") or fmt.get("duration") or 0.0)

    nb_frames = 0
    if v.get("nb_frames"):
        try:
            nb_frames = int(v["nb_frames"])
        except ValueError:
            nb_frames = 0

    bit_rate = 0
    for candidate in (v.get("bit_rate"), fmt.get("bit_rate")):
        if candidate:
            try:
                bit_rate = int(candidate)
                break
            except ValueError:
                continue

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
    )
