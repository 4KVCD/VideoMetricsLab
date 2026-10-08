from dataclasses import replace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread, Signal
from PySide6.QtWidgets import QApplication

from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import CropBox, VideoInfo
from vmaf_app.core.video_playback import build_video_series_command, neighbour_indices
from vmaf_app.ui import video_compare_view
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
    assert neighbour_indices(5, 0) == (0, 1, 4)
    assert neighbour_indices(5, 4) == (4, 0, 3)
    assert neighbour_indices(0, 0) == ()


def test_neighbours_are_added_right_then_left_then_further_out():
    """The Settings count: 1 is the selected video alone, 2 adds the one to
    its right, 3 the one to its left, 4 the next right, 5 the next left."""
    assert neighbour_indices(7, 3, 1) == (3,)
    assert neighbour_indices(7, 3, 2) == (3, 4)
    assert neighbour_indices(7, 3, 3) == (3, 4, 2)
    assert neighbour_indices(7, 3, 4) == (3, 4, 2, 5)
    assert neighbour_indices(7, 3, 5) == (3, 4, 2, 5, 1)
    assert neighbour_indices(7, 3, 6) == (3, 4, 2, 5, 1, 6)
    assert neighbour_indices(7, 3, 99) == (3, 4, 2, 5, 1, 6, 0)  # never more than there are
    assert neighbour_indices(5, 4, 4) == (4, 0, 3, 1)             # wraps like the arrow keys
    assert neighbour_indices(3, 0, 0) == (0,)                      # never fewer than the selected one


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
    monkeypatch.setattr(video_compare_view, "StreamDecodeWorker", FakeWorker)
    monkeypatch.setattr(video_compare_view, "plan_hwaccel", lambda *args, **kwargs: HwAccelPlan())
    view = video_compare_view.VideoCompareView()
    view.set_audio_enabled(False)
    items = series(tmp_path, count)
    view.load(items[0], 1000, series=items)
    return view, items


def cleanup(view):
    """Stops the view and deletes it, its workers first.

    The view has no parent here (in the app it always has one), so Python owns
    it, and once its pool is cleared the only thing still keeping it alive is
    the lambda on each worker's finished signal. Left to Qt, a worker's
    deferred deletion released that lambda and freed the view while Qt was
    still deleting its child: the process aborted in whichever later test
    processed the events -- now and then under pytest-xdist."""
    workers = view.live_workers()
    view.clear()
    for worker in workers:
        worker.finish()
    view.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)  # the workers, while the view is held
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


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


def test_the_decoded_video_count_applies_immediately_to_a_running_pool(qapp, monkeypatch, tmp_path):
    """Settings tab -> panel -> view -> pool, while playing. Lowering it
    retires neighbours (not the selected pair); raising it starts them."""
    view, _items = make_view(monkeypatch, tmp_path, count=6)
    assert view.decoder_limit == 4 and len(view._pool) == 4  # source + selected + right + left

    view.set_decoded_videos(1)
    kept = set(view._pool)
    source_key = next(k for k in kept if k[0] == "source")
    assert kept == {source_key, ("distorted", 0)}
    for worker in list(view._retired_workers):  # finishing one removes it from the set
        worker.finish()

    view.set_decoded_videos(5)
    distorted = sorted(k[1] for k in view._pool if k[0] == "distorted")
    assert distorted == [0, 1, 2, 4, 5]  # selected 0, right 1, left 5, second right 2, second left 4
    assert len(view._pool) == 6 == view.decoder_limit
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


def test_playback_places_both_videos_by_the_shared_zoom(qapp, monkeypatch, tmp_path):
    """Zoomed, FFmpeg's frames are drawn at the panel's zoom, and a GPU
    frame is cut to the part in view (LockedPresentation)."""
    from types import SimpleNamespace

    from vmaf_app.ui.zoom import Zoom

    view, _ = make_view(monkeypatch, tmp_path, 1)
    view.resize(20, 10)
    zoom = Zoom()
    view.set_zoom(zoom)
    assert view.native_view(None) is None  # fitted: frames go as before
    zoom.factor = 2.0  # the 16 x 12 comparison at 32 x 24, centred
    ratio = view.devicePixelRatioF()
    place = view.zoom_placement((16, 12))
    assert (place.width, place.height) == pytest.approx((32 / ratio, 24 / ratio))
    # A source decoded at twice the size is drawn at the same size; one of
    # another shape, its black bars kept, keeps its shape.
    assert view.zoom_placement((32, 24)) == place
    taller = view.zoom_placement((16, 16))
    assert (taller.width, taller.height) == pytest.approx((32 / ratio, 32 / ratio))
    caps = SimpleNamespace(get_structure=lambda _index: SimpleNamespace(
        get_value=lambda name: {"width": 16, "height": 12}[name]))
    crop, rectangle = view.native_view(SimpleNamespace(get_caps=lambda: caps))
    if ratio == 1:
        assert crop == (3, 4, 10, 5) and rectangle == (0, 0, 20, 10)
        assert view.native_area().getRect() == (0, 0, 20, 10)
    cleanup(view)


def test_a_zoom_decodes_again_only_when_the_frames_would_differ(qapp, monkeypatch, tmp_path):
    """Zoomed, FFmpeg decodes at the comparison's full size, not the
    screen's: decoded again only where the screen held the frames smaller."""
    view, _ = make_view(monkeypatch, tmp_path, 1)
    calls = []
    monkeypatch.setattr(view, "_restart_decoder", lambda **kwargs: calls.append(kwargs))
    view._zoom.factor = 2.0
    view.zoom_changed()  # 16 x 12 fits any screen: the same frames
    assert calls == [] and view._pool_maximum is None
    view._zoom.factor = None
    monkeypatch.setattr(view, "_display_pixel_size", lambda: (8, 6))
    view._pool_screen = (8, 6)
    view.zoom_changed()  # fitted to an 8 x 6 screen: smaller frames
    assert calls == [{"realtime": False}]
    cleanup(view)


def test_a_zoom_decodes_again_for_an_encode_beside_the_selected_one(qapp, monkeypatch, tmp_path):
    """The encodes decoded beside the selected one keep their workers for
    a switch: one the screen held smaller must be decoded again for a
    zoom too, or switching to it showed it scaled up, softer than it is."""
    monkeypatch.setattr(video_compare_view, "StreamDecodeWorker", FakeWorker)
    monkeypatch.setattr(video_compare_view, "plan_hwaccel", lambda *args, **kwargs: HwAccelPlan())
    items = series(tmp_path, 2)
    larger = replace(items[1], distorted_info=replace(items[1].distorted_info, width=64, height=48))
    view = video_compare_view.VideoCompareView()
    view.set_audio_enabled(False)
    monkeypatch.setattr(view, "_display_pixel_size", lambda: (32, 24))  # holds the 64 x 48 encode at 32 x 24
    view.load(items[0], 1000, series=[items[0], larger])
    calls = []
    monkeypatch.setattr(view, "_restart_decoder", lambda **kwargs: calls.append(kwargs))
    view._zoom.factor = 2.0
    view.zoom_changed()  # the selected 16 x 12 encode and the source are the same; the larger one is not
    assert calls == [{"realtime": False}]
    cleanup(view)


def test_monitor_resolution_change_restarts_before_reinterpreting_frame_bytes(qapp, monkeypatch, tmp_path):
    view, _ = make_view(monkeypatch, tmp_path, 1)
    monkeypatch.setattr(view, "_display_pixel_size", lambda: (800, 600))
    calls = []
    monkeypatch.setattr(view, "_restart_decoder", lambda **kwargs: calls.append(kwargs))
    view._tick()
    assert calls == [{"realtime": False}]
    cleanup(view)
