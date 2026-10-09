from io import BytesIO
from types import SimpleNamespace

from vmaf_app.core.frame_extract import PreviewColorSettings
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.ui import playback_worker


class FakeProcess:
    pid = 123

    def __init__(self, output, code, error=b""):
        self.stdout, self.stderr = BytesIO(output), BytesIO(error)
        self.returncode = code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def setup_worker(monkeypatch, attempts, decoder="cuda"):
    from vmaf_app.core import video_playback

    monkeypatch.setattr(video_playback, "playback_dimensions", lambda *args: (1, 1))
    commands = []

    def command(items, frame, *args, **kwargs):
        commands.append((kwargs["processing"], frame))
        return ["fake"]

    monkeypatch.setattr(playback_worker, "build_video_series_command", command)
    processes = iter(attempts)
    def popen_piped(*_args, **_kwargs):
        process = next(processes)
        return process, process.stdout

    monkeypatch.setattr(playback_worker.proc, "popen_piped", popen_piped)
    worker = playback_worker.StreamDecodeWorker(None, "source", 100, PreviewColorSettings(),
                                               HwAccelPlan(source=decoder), None)
    worker._handle = SimpleNamespace(attach=lambda pid: None, detach=lambda: None, terminate=lambda: None)
    frames, failures = [], []
    worker._put = lambda number, payload: frames.append((number, payload)) or True
    worker.failed.connect(failures.append)
    return worker, commands, frames, failures


def test_a_failed_decoder_resumes_down_a_bounded_ladder_without_replaying(monkeypatch, subtests):
    with subtests.test("device lost after frames"):
        worker, commands, frames, failures = setup_worker(monkeypatch, [
            FakeProcess(b"aaaa" + b"bb", 1, b"VK_ERROR_DEVICE_LOST"),
            FakeProcess(b"ccccdddd", 0),
        ])
        worker.run()
        assert commands == [("interop", 100), ("transfer", 101)]
        assert frames == [(100, b"aaaa"), (101, b"cccc"), (102, b"dddd")]
        assert not failures
        assert worker.ended and not worker.error
        assert worker.attempt_errors == ["VK_ERROR_DEVICE_LOST"]
    # NVDEC's pictures go to Vulkan on the GPU first, the other makers'
    # through system memory; never through FFmpeg's Vulkan decoder.
    for decoder, ladder in (("cuda", ["interop", "transfer", "software", "cpu"]),
                            ("d3d11va", ["transfer", "software", "cpu"]),
                            (None, ["software", "cpu"])):
        with subtests.test("every attempt fails", decoder=decoder):
            worker, commands, frames, failures = setup_worker(monkeypatch, [
                FakeProcess(b"aaaa", 1, b"device lost") for _ in ladder
            ], decoder)
            worker.run()
            assert commands == [(mode, 100 + n) for n, mode in enumerate(ladder)]
            assert [n for n, _ in frames] == [100 + n for n in range(len(ladder))]
            assert failures == ["device lost"]
            assert not worker.ended
