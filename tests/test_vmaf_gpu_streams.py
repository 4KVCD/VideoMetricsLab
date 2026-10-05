"""VMAF on the GPU from FFmpeg's frames, paired in the app (vmaf_cuda's
_StreamReader and GpuAttempt, vmaf_runner's FFmpeg for each video): what can
be tested without a GPU or FFmpeg, by writing to the pipes as FFmpeg does."""
import subprocess
import threading
from fractions import Fraction
from types import SimpleNamespace

import pytest

from vmaf_app.core import vmaf_cuda
from vmaf_app.core import vmaf_runner as vr
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.vmaf_runner import Cancelled

HEADER = b"YUV4MPEG2 W2 H2 F24000:1001 Ip A1:1 C420jpeg\n"


def _listing(time_base: str, stamps) -> bytes:
    lines = ["#software: Lavf", f"#tb 0: {time_base}", "#media_type 0: video"]
    lines += [f"0, {stamp:10d}, {stamp:10d},       41,      424, 0x28dc17fd" for stamp in stamps]
    return ("\n".join(lines) + "\n").encode()


def _write(reader, frames, time_base="1/1000", stamps=None, tail=b""):
    """As FFmpeg: the frames to one pipe, their timestamps to the other, then gone."""
    stamps = range(0, 42 * len(frames), 42) if stamps is None else stamps
    with open(reader.path, "wb") as pixels:
        pixels.write(HEADER + b"".join(b"FRAME\n" + frame for frame in frames) + tail)
    with open(reader.listing_path, "wb") as listing:
        listing.write(_listing(time_base, stamps))


def _pulled(reader):
    items = []
    while (item := reader.pull()) is not None:
        items.append((bytes(item[0]), item[1]))
        reader.release(item[0])
    return items


def test_a_video_comes_as_its_frames_and_their_timestamps():
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write, args=(reader, [b"abcdef", b"ghijkl", b"mnopqr"], "1001/24000", [0, 1, 3]))
    writer.start()
    assert reader.wait_time_base() == Fraction(1001, 24000)
    assert _pulled(reader) == [(b"abcdef", 0), (b"ghijkl", 1), (b"mnopqr", 3)]
    writer.join(5)
    reader.ffmpeg_ended()
    reader.join()
    assert reader.pull() is None  # and stays ended


def test_more_frames_than_the_reader_holds_wait_in_the_pipe_for_their_turn():
    frames = [bytes([65 + index]) * 6 for index in range(20)]
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write, args=(reader, frames))
    writer.start()
    assert _pulled(reader) == [(frame, 42 * index) for index, frame in enumerate(frames)]
    writer.join(5)
    reader.ffmpeg_ended()
    reader.join()


def test_frames_written_before_the_reader_connected_are_still_read():
    """As _PipeReader's: FFmpeg can open a pipe, write and close it before
    the reader's thread connects."""
    reader = vmaf_cuda._StreamReader("test", 6)
    _write(reader, [b"abcdef", b"ghijkl"])
    reader.start()
    assert _pulled(reader) == [(b"abcdef", 0), (b"ghijkl", 42)]
    reader.ffmpeg_ended()
    reader.join()


def test_pipes_ffmpeg_never_opened_do_not_hold_the_run():
    """An output that gets no frame is never opened."""
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    reader.ffmpeg_ended()
    assert reader.pull() is None
    reader.join()
    assert reader.wait_time_base() == Fraction(1, 1000)  # any: there is no frame to stamp


def test_frames_that_end_inside_one_are_an_error_not_a_short_video():
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    writer = threading.Thread(target=_write, args=(reader, [b"abcdef"]), kwargs={"tail": b"FRAME\nxyz"})
    writer.start()
    item = reader.pull()
    assert bytes(item[0]) == b"abcdef"
    reader.release(item[0])
    writer.join(5)
    reader.ffmpeg_ended()
    with pytest.raises(vmaf_cuda.VmafGpuError, match="ended"):
        reader.pull()


def test_a_stream_that_is_not_yuv4mpeg_is_refused():
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    with open(reader.path, "wb") as pixels:
        pixels.write(b"RIFF....not frames\n" + b"x" * 64)
    reader.ffmpeg_ended()
    with pytest.raises(vmaf_cuda.VmafGpuError, match="YUV4MPEG"):
        reader.pull()


def test_a_frame_without_its_timestamp_is_an_error():
    reader = vmaf_cuda._StreamReader("test", 6)
    reader.start()
    with open(reader.path, "wb") as pixels:
        pixels.write(HEADER + b"FRAME\nabcdef" + b"FRAME\nghijkl")
    with open(reader.listing_path, "wb") as listing:
        listing.write(_listing("1/1000", [0]))
    reader.ffmpeg_ended()
    assert bytes(reader.pull()[0]) == b"abcdef"
    with pytest.raises(vmaf_cuda.VmafGpuError, match="timestamp"):
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


def test_the_attempt_pairs_the_two_videos_frames_as_libvmafs_filter_would(monkeypatch):
    """By the nearest timestamp, each video's from its own first frame and
    in its own time base (frame_sync); the source's extra frame at the end
    is read and dropped."""
    monkeypatch.setattr(vmaf_cuda, "GpuScorer", _Scorer)
    attempt = vmaf_cuda.GpuAttempt(2, 2, 8, {"vmaf": "vmaf_v0.6.1"}, 1, paired=False)
    test = [b"t0....", b"t1....", b"t2...."]
    source = [b"s0....", b"s1....", b"s2....", b"s3...."]
    writers = [threading.Thread(target=_write, args=(attempt.distorted, test, "1/1000", [0, 42, 83])),
               threading.Thread(target=_write, args=(attempt.reference, source, "1001/24000", [7, 8, 9, 10]))]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)
    assert attempt.finish(True) == ("frames", {"vmaf": 3})
    assert attempt._scorer.pairs == [(b"s0....", b"t0...."), (b"s1....", b"t1...."), (b"s2....", b"t2....")]
    assert attempt._scorer.closed


def test_a_source_frame_nearer_in_time_is_the_one_compared(monkeypatch):
    """The test video at 12 frames a second against a source at 24: every
    other source frame, as FFmpeg's filter pairs them."""
    monkeypatch.setattr(vmaf_cuda, "GpuScorer", _Scorer)
    attempt = vmaf_cuda.GpuAttempt(2, 2, 8, {"vmaf": "vmaf_v0.6.1"}, 1, paired=False)
    writers = [threading.Thread(target=_write, args=(attempt.distorted, [b"t0....", b"t1...."], "1/1000", [0, 83])),
               threading.Thread(target=_write, args=(attempt.reference, [b"s0....", b"s1....", b"s2....", b"s3...."],
                                                    "1/1000", [0, 42, 83, 125]))]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)
    attempt.finish(True)
    assert attempt._scorer.pairs == [(b"s0....", b"t0...."), (b"s2....", b"t1....")]


def test_a_failed_ffmpeg_ends_the_attempt_without_scores(monkeypatch):
    monkeypatch.setattr(vmaf_cuda, "GpuScorer", _Scorer)
    attempt = vmaf_cuda.GpuAttempt(2, 2, 8, {"vmaf": "vmaf_v0.6.1"}, 1, paired=False)
    assert attempt.finish(False) is None  # nothing ever opened the pipes
    assert attempt._scorer.closed


def test_each_video_gets_its_frames_and_a_listing_of_their_timestamps():
    attempt = object.__new__(vmaf_cuda.GpuAttempt)
    attempt.paired = False
    attempt.distorted = SimpleNamespace(path="D", listing_path="DT")
    attempt.reference = SimpleNamespace(path="R", listing_path="RT")
    test, source = attempt.output_args(30.5)
    assert test == [
        "-map", "[vmaf_dist]", "-fps_mode", "passthrough", "-t", "30.500", "-flush_packets", "1",
        "-f", "yuv4mpegpipe", "-strict", "-1", "D",
        "-map", "[vmaf_dist_ts]", "-fps_mode", "passthrough", "-enc_time_base", "filter", "-t", "30.500",
        "-c:v", "wrapped_avframe", "-flush_packets", "1", "-f", "framecrc", "DT"]
    # The source a second longer: the frame nearest the test video's last is among them.
    assert source[4:6] == ["-t", "31.500"] and source[-1] == "RT" and "[vmaf_ref_ts]" in source
    whole, _ = attempt.output_args(0.0)
    assert "-t" not in whole


def test_the_pairing_goes_back_to_ffmpeg_when_asked_or_when_ffmpeg_is_too_old(monkeypatch):
    from vmaf_app.core import proc

    monkeypatch.setenv(vmaf_cuda.PIPES_VARIABLE, "canvas")
    vmaf_cuda.pairs_in_app.cache_clear()
    assert vmaf_cuda.pairs_in_app() is False
    monkeypatch.delenv(vmaf_cuda.PIPES_VARIABLE)
    for done, expected in ((subprocess.CompletedProcess([], 0, b"#tb 0: 1/1\n0, 0, 0, 1, 424, 0x0\n", b""), True),
                           (subprocess.CompletedProcess([], 8, b"", b"Unable to parse option value"), False)):
        vmaf_cuda.pairs_in_app.cache_clear()
        monkeypatch.setattr(proc, "run", lambda *args, done=done, **kwargs: done)
        assert vmaf_cuda.pairs_in_app() is expected
    vmaf_cuda.pairs_in_app.cache_clear()


def _info(path, width=1920, height=1080):
    from pathlib import Path

    from vmaf_app.core.models import VideoInfo

    return VideoInfo(Path(path), width, height, 24.0, 10.0, 240, "hevc", pix_fmt="yuv420p")


def test_each_video_has_a_graph_and_an_ffmpeg_of_its_own():
    """In one FFmpeg the video that decodes faster filled its pipe while its
    frames waited for the other's, and FFmpeg's one filter thread, held by
    that pipe, passed no frame of the other's: the run stood still."""
    from pathlib import Path

    from vmaf_app.core.models import CropMode, VmafOptions

    options = VmafOptions(crop_mode=CropMode.NONE)
    graphs = vr._build_filtergraph(_info("s.mkv", 3840, 2160), _info("d.mkv"), options, None, None,
                                   HwAccelPlan(source="cuda"), Path("log.json"), gpu_vmaf=True, gpu_paired=False)
    test, source = graphs.split("\n")
    assert test == "[0:V:0]format=yuv420p,setpts=PTS-STARTPTS,split=2[vmaf_dist][vmaf_dist_ts]"
    assert source == ("[0:V:0]hwdownload,format=nv12,format=yuv420p,scale=1920:1080:flags=bicubic,"
                      "setpts=PTS-STARTPTS,split=2[vmaf_ref][vmaf_ref_ts]")
    commands = vr._build_stream_cmds(Path("d.mkv"), Path("s.mkv"), graphs, HwAccelPlan(source="cuda"),
                                     (["-f", "yuv4mpegpipe", "D"], ["-f", "yuv4mpegpipe", "R"]))
    assert [command[-1] for command in commands] == ["D", "R"]
    assert "-hwaccel" not in commands[0] and "-hwaccel" in commands[1]
    assert all(command.count("-i") == 1 and "-progress" in command for command in commands)
    assert commands[0][commands[0].index("-lavfi") + 1] == test


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
