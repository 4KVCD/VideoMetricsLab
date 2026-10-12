"""VMAF on the GPU from FFmpeg's frames, paired in the app (vmaf_gpu's
_StreamReader and GpuAttempt, vmaf_runner's FFmpeg for each video): what can
be tested without a GPU or FFmpeg, by writing to the pipes as FFmpeg does."""
import subprocess
import threading
from fractions import Fraction

import pytest

from vmaf_app.core import libvmaf, vmaf_gpu
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.vmaf_runner import Cancelled

HEADER = b"YUV4MPEG2 W2 H2 F24000:1001 Ip A1:1 C420jpeg\n"


def _write(reader, frames, tail=b""):
    """As FFmpeg: the frames to the pipe, then gone."""
    with open(reader.path, "wb") as pixels:
        pixels.write(HEADER + b"".join(b"FRAME\n" + frame for frame in frames) + tail)


def _pulled(reader):
    frames = []
    while (frame := reader.pull()) is not None:
        frames.append(bytes(frame))
        reader.release(frame)
    return frames


def test_a_video_comes_as_its_frames():
    reader = vmaf_gpu._StreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write, args=(reader, [b"abcdef", b"ghijkl", b"mnopqr"]))
    writer.start()
    assert _pulled(reader) == [b"abcdef", b"ghijkl", b"mnopqr"]
    writer.join(5)
    reader.ffmpeg_ended()
    reader.join()
    assert reader.pull() is None  # and stays ended


def test_frames_that_end_inside_one_are_an_error_not_a_short_video():
    reader = vmaf_gpu._StreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write, args=(reader, [b"abcdef"]), kwargs={"tail": b"FRAME\nxyz"})
    writer.start()
    frame = reader.pull()
    assert bytes(frame) == b"abcdef"
    reader.release(frame)
    writer.join(5)
    reader.ffmpeg_ended()
    with pytest.raises(libvmaf.LibvmafError, match="ended"):
        reader.pull()


class _Scorer:
    frame_bytes = 6

    def __init__(self, *args, **kwargs):
        self.pairs = []

    def add(self, reference, distorted):
        self.pairs.append((bytes(reference), bytes(distorted)))

    def finish(self):
        return "frames", {"vmaf": len(self.pairs)}

    def close(self):
        self.closed = True


def test_the_attempt_pairs_frame_n_of_one_video_with_frame_n_of_the_other(monkeypatch):
    """As libvmaf's filter pairs the two videos' numbered frames
    (frame_sync); the source's extra frame at the end is read and dropped."""
    monkeypatch.setattr(vmaf_gpu, "GpuScorer", _Scorer)
    attempt = vmaf_gpu.GpuAttempt(2, 2, 8, {"vmaf": "vmaf_v0.6.1"}, 1, paired=False)
    test = [b"t0....", b"t1....", b"t2...."]
    source = [b"s0....", b"s1....", b"s2....", b"s3...."]
    writers = [threading.Thread(target=_write, args=(attempt.distorted, test)),
               threading.Thread(target=_write, args=(attempt.reference, source))]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)
    assert attempt.finish(True) == ("frames", {"vmaf": 3})
    assert attempt._scorer.pairs == [(b"s0....", b"t0...."), (b"s1....", b"t1...."), (b"s2....", b"t2....")]
    assert attempt._scorer.closed


# --------------------------------------------- pairing by timestamp, as before

def _listing(time_base: str, stamps) -> bytes:
    lines = ["#software: Lavf", f"#tb 0: {time_base}", "#media_type 0: video"]
    lines += [f"0, {stamp:10d}, {stamp:10d},       41,      424, 0x28dc17fd" for stamp in stamps]
    return ("\n".join(lines) + "\n").encode()


def _write_timed(reader, frames, time_base="1/1000", stamps=None):
    """As FFmpeg when pairing by timestamp: the frames to one pipe, their
    timestamps to the other, then gone."""
    stamps = range(0, 42 * len(frames), 42) if stamps is None else stamps
    _write(reader, frames)
    with open(reader.listing_path, "wb") as listing:
        listing.write(_listing(time_base, stamps))


def test_pairing_by_timestamp_a_video_comes_as_its_frames_and_their_timestamps():
    reader = vmaf_gpu._TimedStreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write_timed,
                              args=(reader, [b"abcdef", b"ghijkl", b"mnopqr"], "1001/24000", [0, 1, 3]))
    writer.start()
    assert reader.wait_time_base() == Fraction(1001, 24000)
    pulled = []
    while (item := reader.pull()) is not None:
        pulled.append((bytes(item[0]), item[1]))
        reader.release(item[0])
    assert pulled == [(b"abcdef", 0), (b"ghijkl", 1), (b"mnopqr", 3)]
    writer.join(5)
    reader.ffmpeg_ended()
    reader.join()
    assert reader.pull() is None  # and stays ended


def test_pairing_by_timestamp_the_attempt_pairs_as_libvmafs_filter_would(monkeypatch):
    """With VML_FRAME_PAIRING=timestamp: by the nearest timestamp, each
    video's from its own first frame and in its own time base
    (frame_sync_by_time); the source's extra frame at the end is read and
    dropped."""
    monkeypatch.setenv("VML_FRAME_PAIRING", "timestamp")
    monkeypatch.setattr(vmaf_gpu, "GpuScorer", _Scorer)
    attempt = vmaf_gpu.GpuAttempt(2, 2, 8, {"vmaf": "vmaf_v0.6.1"}, 1, paired=False)
    test = [b"t0....", b"t1....", b"t2...."]
    source = [b"s0....", b"s1....", b"s2....", b"s3...."]
    writers = [threading.Thread(target=_write_timed, args=(attempt.distorted, test, "1/1000", [0, 42, 83])),
               threading.Thread(target=_write_timed, args=(attempt.reference, source, "1001/24000", [7, 8, 9, 10]))]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)
    assert attempt.finish(True) == ("frames", {"vmaf": 3})
    assert attempt._scorer.pairs == [(b"s0....", b"t0...."), (b"s1....", b"t1...."), (b"s2....", b"t2....")]
    assert attempt._scorer.closed


def _info(path, width=1920, height=1080):
    from pathlib import Path

    from vmaf_app.core.models import VideoInfo

    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_the_two_ffmpegs_run_side_by_side_and_one_that_fails_ends_the_other(monkeypatch, tmp_path):
    seen = []

    def fake(result_of):
        def run(cmd, total_frames, on_progress, cancel_event, cwd, process_handle=None):
            seen.append((cmd[0], on_progress is not None))
            code = result_of[cmd[0]]
            if code == "waits":  # as an FFmpeg still running when the other fails
                assert cancel_event.wait(10)
                raise Cancelled("Cancelled by user")
            return subprocess.CompletedProcess(cmd, code, "", f"{cmd[0]} said so")
        return run

    monkeypatch.setattr(vr, "_run_ffmpeg", fake({"test": 0, "source": 0}))
    result = vr._run_ffmpeg_pair([["test"], ["source"]], 10, lambda *a: None, None, tmp_path)
    assert result.returncode == 0 and result.args == ["test"]
    assert sorted(seen) == [("source", False), ("test", True)]  # the test video's reports the progress

    monkeypatch.setattr(vr, "_run_ffmpeg", fake({"test": "waits", "source": 1}))
    result = vr._run_ffmpeg_pair([["test"], ["source"]], 10, None, None, tmp_path)
    assert result.returncode == 1 and result.stderr == "source said so"

    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(vr, "_run_ffmpeg", fake({"test": "waits", "source": "waits"}))
    with pytest.raises(Cancelled):
        vr._run_ffmpeg_pair([["test"], ["source"]], 10, None, cancel, tmp_path)
