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


def test_ffprobe_timeout_becomes_a_readable_probe_error(monkeypatch):
    proc = _fake_popen(monkeypatch, FakeFfprobe(hang=True))

    with pytest.raises(ProbeError, match="timed out"):
        ffprobe.probe_video(Path("slow.mp4"))

    assert proc.killed, "a timed-out ffprobe was left running"


def test_a_probe_can_be_cancelled_through_its_process_handle(monkeypatch):
    """A plain subprocess.run() cannot be interrupted: a flag set by a
    canceller is invisible to a call already blocked inside it. That is what
    let the window be torn down with an ffprobe still running underneath."""
    from vmaf_app.core.process_control import ProcessHandle

    _fake_popen(monkeypatch, FakeFfprobe(returncode=-15))
    handle = ProcessHandle()
    asked = []
    monkeypatch.setattr(handle, "_try", lambda pid, action: asked.append((pid, action)))

    handle.terminate()  # as the canceller does, from another thread

    with pytest.raises(ffprobe.ProbeCancelled):
        ffprobe.probe_video(Path("slow.mp4"), process_handle=handle)
    assert handle.was_terminated


def test_a_genuine_ffprobe_failure_is_still_reported_as_an_error(monkeypatch):
    # Only a termination WE asked for reads as a cancellation; an unreadable
    # file must still say so.
    from vmaf_app.core.process_control import ProcessHandle

    _fake_popen(monkeypatch, FakeFfprobe(returncode=1))

    with pytest.raises(ProbeError, match="ffprobe failed"):
        ffprobe.probe_video(Path("broken.mp4"), process_handle=ProcessHandle())


def test_the_handle_is_detached_once_the_probe_returns(monkeypatch):
    from vmaf_app.core.process_control import ProcessHandle

    _fake_popen(monkeypatch, FakeFfprobe(returncode=1))
    handle = ProcessHandle()

    with pytest.raises(ProbeError):
        ffprobe.probe_video(Path("broken.mp4"), process_handle=handle)

    assert handle._pid is None, "a detached handle must not still address a dead pid"


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
