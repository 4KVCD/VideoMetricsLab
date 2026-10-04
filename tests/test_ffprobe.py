import json
import subprocess
from pathlib import Path

import pytest

from tests.factories import STDLIB_PYTHON
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


def test_a_curly_quote_in_ffprobes_json_is_read_as_utf8(monkeypatch):
    """Reported on Reddit against 1.3: every probe failed with "the JSON
    object must be str, bytes or bytearray, not NoneType". ffprobe writes
    UTF-8; decoded as cp1252, the 0x9D byte of ” has no character, so the
    reader thread died and stdout came back as None. A real child process
    writes the bytes here, so the app's own decoding is what is tested."""
    payload = json.dumps({
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 320, "height": 180,
                     "pix_fmt": "yuv420p", "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
                     "duration": "1.0"}],
        "format": {"duration": "1.0", "tags": {"title": "Director’s Cut “Final”"}},
    }, ensure_ascii=False).encode("utf-8")
    script = f"import sys; sys.stdout.buffer.write({payload!r})"
    real_popen = ffprobe.proc_util.popen
    monkeypatch.setattr(ffprobe, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffprobe.proc_util, "popen",
                        lambda cmd, **kw: real_popen([STDLIB_PYTHON, "-S", "-c", script], **kw))

    info = ffprobe.probe_video(Path("Director’s Cut “Final”.mkv"))

    assert (info.width, info.height) == (320, 180)


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

    assert handle._pids == set(), "a detached handle must not still address a dead pid"


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


def test_cover_art_listed_before_the_video_is_not_taken_for_it(monkeypatch):
    """An MP4's cover (covr) can be the first video stream FFmpeg lists:
    the app then described, and compared, a still picture."""
    cover = {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
             "disposition": {"attached_pic": 1}}
    info = _probe_payload(monkeypatch, [cover, {**_VIDEO, "duration": "10.0"}], {"duration": "10.0"})
    assert (info.codec_name, info.width) == ("hevc", 3840)


def test_only_cover_art_is_no_video(monkeypatch):
    cover = {"codec_type": "video", "codec_name": "png", "width": 600, "height": 600,
             "disposition": {"attached_pic": 1}}
    with pytest.raises(ProbeError, match="No video stream"):
        _probe_payload(monkeypatch, [cover], {"duration": "10.0"})


def test_a_matroska_videos_length_is_its_own_not_the_soundtracks(monkeypatch):
    """Matroska gives no stream duration; the container's is the longest
    track's. An audio track running on after the picture made "Durations do
    not match" -- or a whole video read as cut short."""
    video = {**_VIDEO, "tags": {"DURATION": "01:45:36.289000000"}}
    info = _probe_payload(monkeypatch, [video], {"duration": "6340.0"})
    assert info.duration == pytest.approx(6336.289)


def test_a_stale_duration_tag_longer_than_the_file_is_not_used(monkeypatch):
    """A remux that cut the file can keep the original's DURATION tag."""
    video = {**_VIDEO, "tags": {"DURATION": "01:45:36.289000000"}}
    info = _probe_payload(monkeypatch, [video], {"duration": "600.0"})
    assert info.duration == 600.0


def test_an_unknown_frame_rate_is_zero_not_an_exception(monkeypatch):
    video = {**_VIDEO, "avg_frame_rate": "N/A", "r_frame_rate": "N/A", "duration": "1.0"}
    info = _probe_payload(monkeypatch, [video], {"duration": "1.0"})
    assert info.fps == 0.0


_STATS = {"DURATION": "00:10:00.000000000"}


def test_a_matroska_videos_bitrate_is_its_own_not_the_files(monkeypatch):
    video = {**_VIDEO, "tags": {**_STATS, "BPS": "20000000", "NUMBER_OF_BYTES": "1500000000"}}
    audio = {"codec_type": "audio", "tags": {"BPS": "4000000"}}
    info = _probe_payload(monkeypatch, [video, audio],
                          {"duration": "600.0", "bit_rate": "24100000", "size": "1807500000"})
    assert (info.bit_rate, info.bit_rate_whole_file) == (20_000_000, False)


def test_statistics_copied_from_a_longer_file_are_not_believed(monkeypatch):
    """FFmpeg copies mkvmerge's statistics into what it writes, even through
    a re-encode: a 30 s cut of a film carried the film's 55 GB."""
    video = {**_VIDEO, "tags": {"DURATION": "00:00:30.072000000", "BPS": "69797727",
                                "NUMBER_OF_BYTES": "55282321636"}}
    info = _probe_payload(monkeypatch, [video], {"duration": "30.072", "bit_rate": "31910785", "size": "119952642"})
    assert (info.bit_rate, info.bit_rate_whole_file) == (31_910_785, False)


def test_without_the_videos_own_rate_known_audio_is_taken_off_the_files(monkeypatch):
    video = {**_VIDEO, "tags": _STATS}
    audio = {"codec_type": "audio", "bit_rate": "640000"}
    info = _probe_payload(monkeypatch, [video, audio], {"duration": "600.0", "bit_rate": "10640000"})
    assert (info.bit_rate, info.bit_rate_whole_file) == (10_000_000, False)


def test_with_nothing_but_the_files_rate_it_says_so(monkeypatch):
    from vmaf_app.ui.formatting import bitrate_note, bitrate_string

    video = {**_VIDEO, "tags": _STATS}
    audio = {"codec_type": "audio"}
    info = _probe_payload(monkeypatch, [video, audio], {"duration": "600.0", "bit_rate": "10640000"})
    assert (info.bit_rate, info.bit_rate_whole_file) == (10_640_000, True)
    assert bitrate_string(info) == "≈10.6 Mb/s"
    assert "whole file" in bitrate_note(info)
