from dataclasses import replace

import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import build_video_series_command, neighbour_indices
from vmaf_app.ui import rolling_video_view
from vmaf_app.ui.playback_worker import StreamDecodeWorker


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def series(tmp_path, count=5):
    source = tmp_path / "source.mkv"
    source.touch()
    info = VideoInfo(path=source, width=16, height=12, fps=24, duration=10,
                     nb_frames=240, codec_name="hevc", pix_fmt="yuv420p10le",
                     color_transfer="smpte2084", color_primaries="bt2020", color_space="bt2020nc")
    result = []
    for i in range(count):
        path = tmp_path / f"{i}.mkv"
        path.touch()
        result.append(FrameComparison(info, replace(info, path=path), fps=24, frame_count=240))
    return result


def test_neighbours_are_bounded_unique_and_wrap():
    assert neighbour_indices(1, 0) == (0,)
    assert neighbour_indices(2, 0) == (0, 1)
    assert neighbour_indices(5, 0) == (0, 4, 1)
    assert neighbour_indices(5, 4) == (4, 3, 0)
    assert neighbour_indices(0, 0) == ()


def test_gpu_stream_uses_vulkan_decode_and_gpu_rgb_without_cpu_tone_mapper(tmp_path):
    item = replace(series(tmp_path, 1)[0], source_crop=CropBox(16, 8, 0, 2))
    command = build_video_series_command([item], 24, PreviewColorSettings(),
                                         [HwAccelPlan(source="cuda")], side="source",
                                         realtime=True, paced=False)
    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-i") == 1
    assert "vulkan" in command
    assert "libplacebo=" in graph
    assert "format=rgba" in graph
    assert "crop_y=2" in graph and "crop_h=8" in graph
    assert "hwdownload,format=rgba" in graph
    assert "tonemap=mobius" not in graph and "zscale=" not in graph
    assert "-readrate" not in command
    assert "peak_detect=0" in graph


def test_software_decoded_vvc_still_uses_gpu_colour_conversion(tmp_path):
    item = series(tmp_path, 1)[0]
    item = replace(item, distorted_info=replace(item.distorted_info, codec_name="vvc"))
    command = build_video_series_command([item], 0, PreviewColorSettings(),
                                         [HwAccelPlan(source="cuda")], side="distorted", realtime=True)
    graph = command[command.index("-filter_complex") + 1]
    assert "-hwaccel" not in command
    assert "hwupload,libplacebo=" in graph


class FakeWorker(QThread):
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, comparison, side, start_frame, settings, plan, maximum, parent):
        super().__init__(parent)
        self.comparison, self.side, self.start_frame = comparison, side, start_frame
        self.running, self.cancelled = False, False
        self.frames = []

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def finish(self):
        self.running = False
        self.finished.emit()

    def drain_through(self, frame):
        due = [item for item in self.frames if item[0] <= frame]
        self.frames = [item for item in self.frames if item[0] > frame]
        return due

    def latest_frame_number(self):
        return max((item[0] for item in self.frames), default=None)


def make_view(monkeypatch, tmp_path, count=5):
    monkeypatch.setattr(rolling_video_view, "StreamDecodeWorker", FakeWorker)
    monkeypatch.setattr(rolling_video_view, "plan_hwaccel", lambda *args: HwAccelPlan())
    view = rolling_video_view.RollingVideoCompareView()
    view.set_audio_enabled(False)
    items = series(tmp_path, count)
    view.load(items[0], 1000, series=items)
    return view, items


def cleanup(view):
    workers = view.live_workers()
    view.clear()
    for worker in workers:
        worker.finish()
    view.close()


def test_selection_keeps_source_and_warm_neighbours_without_fifth_decoder(qapp, monkeypatch, tmp_path):
    view, items = make_view(monkeypatch, tmp_path)
    original = dict(view._pool)
    assert len(original) == 4
    generation = view._generation
    view.load(items[1], 1000, series=items)
    assert view._generation == generation
    for key in (next(k for k in original if k[0] == "source"), ("distorted", 0), ("distorted", 1)):
        assert view._pool[key] is original[key]
    assert ("distorted", 2) not in view._pool  # old decoder still shutting down
    assert original[("distorted", 4)].cancelled
    original[("distorted", 4)].finish()
    assert ("distorted", 2) in view._pool
    assert len(view._pool) == 4
    cleanup(view)


def test_pair_presents_only_matching_frame_numbers(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 1)
    source_key = next(k for k in view._pool if k[0] == "source")
    payload = bytes(16 * 12 * 4)
    view._pool[source_key].frames = [(24, payload)]
    view._pool[("distorted", 0)].frames = [(25, payload)]
    view._tick()
    assert view._presented == -1
    view._pool[("distorted", 0)].frames.insert(0, (24, payload))
    view._tick()
    assert view._presented == 24
    assert view._source_surface._payload is payload
    assert view._distorted_surface._payload is payload
    cleanup(view)


def test_seek_replaces_all_streams_and_pause_does_not(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 2)
    old = dict(view._pool)
    view.set_playing(False)
    assert view._pool == old
    view.set_position(2000)
    assert all(w.cancelled for w in old.values())
    for worker in old.values():
        worker.finish()
    assert all(w.start_frame == 48 for w in view._pool.values())
    cleanup(view)


def test_worker_queue_keeps_future_frames_without_copying(tmp_path):
    item = series(tmp_path, 1)[0]
    worker = StreamDecodeWorker(item, "source", 0, PreviewColorSettings(), HwAccelPlan(), None)
    payload = b"frame"
    worker._put(1, payload)
    worker._put(2, payload)
    assert worker.drain_through(0) == []
    assert worker.drain_through(1) == [(1, payload)]
    assert worker.drain_through(2) == [(2, payload)]
    worker.cancel()
    assert not worker._put(3, payload)


def test_cropped_source_uses_one_hashable_pool_key(qapp, monkeypatch, tmp_path):
    view, items = make_view(monkeypatch, tmp_path)
    old = view.live_workers()
    items = [replace(item, source_crop=CropBox(16, 8, 0, 2)) for item in items]
    view.load(items[0], 1000, series=items)
    for worker in old:
        worker.finish()
    assert len([k for k in view._pool if k[0] == "source"]) == 1
    source = next(w for k, w in view._pool.items() if k[0] == "source")
    view.load(items[1], 1000, series=items)
    assert next(w for k, w in view._pool.items() if k[0] == "source") is source
    cleanup(view)


def test_fast_stream_frames_are_not_discarded_before_slow_partner_arrives(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 1)
    source = next(w for k, w in view._pool.items() if k[0] == "source")
    distorted = view._pool[("distorted", 0)]
    payload = bytes(16 * 12 * 4)
    view._clock_frame = 30
    source.frames = [(i, payload) for i in (24, 25, 26)]
    distorted.frames = [(24, payload)]
    view._tick()
    assert view._presented == 24
    assert [f for f, _ in source.frames] == [25, 26]
    distorted.frames = [(25, payload)]
    view._tick()
    assert view._presented == 25
    cleanup(view)


def test_pause_freezes_presented_frame_not_ahead_of_decode_clock(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 1)
    view._presented = 24
    view._clock_frame = 27
    view.set_playing(False)
    assert view._target_frame() == 24
    cleanup(view)


def test_monitor_resolution_change_restarts_before_reinterpreting_frame_bytes(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 1)
    monkeypatch.setattr(view, "_display_pixel_size", lambda: (800, 600))
    calls = []
    monkeypatch.setattr(view, "_restart_decoder", lambda **kwargs: calls.append(kwargs))
    view._tick()
    assert calls == [{"realtime": False}]
    cleanup(view)


def test_native_pool_retains_neighbours_and_shifts_shared_clock_after_pause(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from vmaf_app.ui import native_playback_pool as native

    class Surface:
        # This is a clock/lifecycle test. Native HWNDs are covered by the
        # Windows smoke test, not Qt's offscreen platform implementation.
        def __init__(self, parent):
            pass

        def setGeometry(self, rect):
            pass

        def setFocusProxy(self, view):
            pass

        def set_native_playback(self, enabled):
            pass

        def show(self):
            pass

        def hide(self):
            pass

        def raise_(self):
            pass

        def winId(self):
            return 1

        def deleteLater(self):
            pass

    class Finished:
        def connect(self, callback, *args):
            self.callback = callback

        def emit(self):
            self.callback()

    class Stop:
        def __init__(self, pipeline, parent):
            self.pipeline = pipeline
            self.finished = Finished()

        def start(self):
            self.pipeline.stop()
            self.finished.emit()

        def deleteLater(self):
            pass

        def isRunning(self):
            return False

    class Clock:
        now = 1_000_000_000

        def get_time(self):
            return self.now

    clock = Clock()

    class Pipeline:
        def __init__(self, *args, **kwargs):
            self._ready = False
            self.alignments = []

        def start(self, position, playing):
            self.start_position = position

        def poll(self):
            self._ready = True
            return SimpleNamespace(error=None)

        def align_clock(self, clock, base, offset):
            self.alignments.append((base, offset))

        def set_playing(self, playing):
            self.playing = playing

        def set_audio_enabled(self, enabled):
            self.audio = enabled

        def stop(self):
            pass

    monkeypatch.setattr(native, "gstreamer_available", lambda: (True, ""))
    monkeypatch.setattr(native, "_load_gstreamer", lambda: (
        SimpleNamespace(SystemClock=SimpleNamespace(obtain=lambda: clock), MSECOND=1_000_000), None))
    monkeypatch.setattr(native, "GstComparePipeline", Pipeline)
    monkeypatch.setattr(native, "_PairedFrameWidget", Surface)
    monkeypatch.setattr(native, "_StopNative", Stop)
    view = SimpleNamespace(_retired_workers=set(), _audio_enabled=False, rect=lambda: None)
    pool = native.NativePlaybackPool(view, series(tmp_path), 0, 1000, PreviewColorSettings(), True)
    assert len(pool.entries) == 4
    pool.poll()
    original = dict(pool.entries)
    clock.now += 500_000_000
    assert pool.poll() == 1500
    pool.set_playing(False)
    clock.now += 2_000_000_000
    assert pool.poll() == 1500
    pool.set_playing(True)
    assert pool.base == 3_000_000_000
    pool.sync(1)
    assert pool.entries[("distorted", 1)] is original[("distorted", 1)]
    assert pool.entries[("distorted", 0)] is original[("distorted", 0)]
    assert len(pool.entries) <= 4
    pool.stop()
    assert not view._retired_workers
