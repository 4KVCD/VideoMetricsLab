import json
import subprocess
from pathlib import Path

import pytest

from vmaf_app.core import ffprobe
from vmaf_app.core.ffprobe import ProbeError


class FakeFfprobe:
    """Stands in for the ffprobe subprocess so a probe can be held open."""

    def __init__(self, stdout="{}", returncode=0, hang=False):
        self.pid = 9191
        self._stdout = stdout
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.terminated = False
        self.communicated = 0

    def communicate(self, timeout=None):
        self.communicated += 1
        if self._hang and self.communicated == 1:
            raise subprocess.TimeoutExpired("ffprobe", timeout or 60)
        return self._stdout, ""

    def kill(self):
        self.killed = True
        self._hang = False
        self.returncode = -9

    def terminate(self):
        self.terminated = True
        self._hang = False
        self.returncode = -15


def _fake_popen(monkeypatch, proc):
    monkeypatch.setattr(ffprobe, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffprobe.proc_util, "popen", lambda *a, **kw: proc)
    return proc


def test_a_genuine_ffprobe_failure_is_still_reported_as_an_error(monkeypatch):
    # Only a termination WE asked for reads as a cancellation; an unreadable
    # file must still say so.
    from vmaf_app.core.process_control import ProcessHandle

    _fake_popen(monkeypatch, FakeFfprobe(returncode=1))

    with pytest.raises(ProbeError, match="ffprobe failed"):
        ffprobe.probe_video(Path("broken.mp4"), process_handle=ProcessHandle())


def test_probe_preserves_hdr_colour_tags(monkeypatch):
    payload = """{
      "streams": [{
        "codec_type": "video", "width": 3840, "height": 2160,
        "avg_frame_rate": "24/1", "duration": "1", "nb_frames": "24",
        "codec_name": "hevc", "pix_fmt": "yuv420p10le",
        "color_range": "tv", "color_space": "bt2020nc",
        "color_transfer": "smpte2084", "color_primaries": "bt2020"
      }],
      "format": {"duration": "1"}
    }"""
    _fake_popen(monkeypatch, FakeFfprobe(stdout=payload))

    info = ffprobe.probe_video(Path("hdr.mkv"))

    assert info.color_range == "tv"
    assert info.color_space == "bt2020nc"
    assert info.color_transfer == "smpte2084"
    assert info.color_primaries == "bt2020"


def _probe_payload(monkeypatch, streams, fmt):
    _fake_popen(monkeypatch, FakeFfprobe(stdout=json.dumps({"streams": streams, "format": fmt})))
    return ffprobe.probe_video(Path("v.mkv"))


_VIDEO = {"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160, "pix_fmt": "yuv420p10le",
          "avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001"}


def test_a_videos_length_and_start_are_its_own_not_the_soundtracks(monkeypatch):
    """Matroska gives no stream duration; the container's is the longest
    track's. An audio track running on after the picture made "Durations do
    not match" -- or a whole video read as cut short."""
    video = {**_VIDEO, "tags": {"DURATION": "01:45:36.289000000"}}
    info = _probe_payload(monkeypatch, [video], {"duration": "6340.0"})
    assert info.duration == pytest.approx(6336.289)
    # An MP4 whose soundtrack starts at 0 and video 32 ms later: seeking
    # counts from the file's start (frame_extract.seek_seconds).
    video = {**_VIDEO, "start_time": "0.032031"}
    assert _probe_payload(monkeypatch, [video], {"duration": "10", "start_time": "0.000000"}).start_offset == \
        pytest.approx(0.032031)
    assert _probe_payload(monkeypatch, [_VIDEO], {"duration": "10"}).start_offset == 0
