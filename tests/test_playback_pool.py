from dataclasses import replace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread, Signal
from PySide6.QtWidgets import QApplication

from vmaf_app.core.frame_extract import FrameComparison
from vmaf_app.core.gpu import HwAccelPlan
from vmaf_app.core.models import VideoInfo
from vmaf_app.core.video_playback import neighbour_indices
from vmaf_app.ui import video_compare_view


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
