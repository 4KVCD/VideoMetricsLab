"""Read-only real-media rolling-playback smoke test (does not load/save scores).

Example: python -m scripts.smoke_playback_pool SOURCE.mkv A.mkv B.mkv C.mkv
Optional --crop-height applies a centred crop to taller inputs for this test.
The test owns a separate hidden window, not the user's running application.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
from PySide6.QtWidgets import QApplication

from vmaf_app.core.ffprobe import probe_video
from vmaf_app.core.frame_extract import FrameComparison, PreviewColorSettings
from vmaf_app.core.models import CropBox
from vmaf_app.ui.rolling_video_view import RollingVideoCompareView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("distorted", nargs="+", type=Path)
    parser.add_argument("--crop-height", type=int)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--start", type=float, default=50)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--require-gstreamer", action="store_true")
    parser.add_argument("--exercise-seek", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--stall", action="store_true")
    args = parser.parse_args()
    if args.locked:
        from vmaf_app.ui import native_playback_pool
        from vmaf_app.ui.locked_native_pool import LockedNativePool

        native_playback_pool.NativePlaybackPool = LockedNativePool
    app = QApplication.instance() or QApplication([])
    source = probe_video(args.source)

    def crop(info):
        height = args.crop_height
        if height and height < info.height:
            return CropBox(info.width, height, 0, (info.height - height) // 2)
        return None

    series = []
    for path in args.distorted:
        info = probe_video(path)
        series.append(FrameComparison(source, info, source_crop=crop(source), distorted_crop=crop(info),
                                      fps=source.fps, frame_count=round(min(source.duration, info.duration) * source.fps)))
    view = RollingVideoCompareView()
    view.resize(1280, 720)
    if args.visible:
        view.setWindowTitle("Playback QA — separate test window")
        view.show()
        view.showMaximized()
    view.set_audio_enabled(args.audio)
    samples, switches, statuses = [], [], []
    view.position_changed.connect(lambda p: samples.append((time.monotonic(), p)))
    view.status_changed.connect(lambda text: statuses.append(text) if not statuses or statuses[-1] != text else None)
    settings = PreviewColorSettings(display_hdr_enabled=args.native)
    start = time.monotonic()
    view.load(series[0], round(args.start * 1000), playing=True, color_settings=settings, series=series)
    maximum = 0
    stall_installed = False
    av_errors = []
    process = psutil.Process()
    cpu_start = None
    measured_start = None
    peak_rss = 0
    selected = 0
    next_switch = 4.0
    sought = False
    try:
        while time.monotonic() - start < args.seconds:
            app.processEvents()
            count = len(view._pool) if view._native_pool is None else len(view._native_pool.entries)
            count += sum(w.isRunning() and getattr(w, "counts_as_decoder", True) for w in view._retired_workers)
            maximum = max(maximum, count)
            elapsed = time.monotonic() - start
            pool = view._native_pool
            if args.locked and pool is not None:
                if args.stall and not stall_installed:
                    original_pull = pool._pull_sample

                    def stalled_pull(key, sink, original=original_pull):
                        if key[0] == "distorted" and 4 < time.monotonic() - start < 6:
                            return None
                        return original(key, sink)

                    pool._pull_sample = stalled_pull
                    stall_installed = True
                if pool.audio_running:
                    audio_ms = pool.audio.poll()
                    if audio_ms is not None:
                        av_errors.append(abs(audio_ms - pool.position))
            peak_rss = max(peak_rss, process.memory_info().rss)
            if elapsed > 3 and cpu_start is None:
                cpu_start = sum(process.cpu_times()[:2])
                measured_start = time.monotonic()
            if args.exercise_seek and elapsed > 8 and not sought:
                view.set_playing(False)
                view.show_source(True)
                view.set_position(round((args.start + 30) * 1000))
                view.show_source(False)
                view.set_playing(True)
                sought = True
            if elapsed >= next_switch and len(series) > 1:
                selected = (selected + 1) % len(series)
                before = dict(view._pool if view._native_pool is None else view._native_pool.entries)
                switched = time.monotonic()
                view.load(series[selected], view.position, playing=True, color_settings=settings, series=series)
                after = view._pool if view._native_pool is None else view._native_pool.entries
                switches.append({"selected": selected, "call_ms": (time.monotonic() - switched) * 1000,
                                 "retained_workers": sum(after.get(k) is w for k, w in before.items())})
                next_switch += 3
            time.sleep(0.005)
        playing_stats = {}
        if view._native_pool is not None:
            for key, entry in view._native_pool.entries.items():
                playing_stats[str(key)] = {
                    side: sink.get_property("stats").to_string()
                    for side, sink in entry[0]._sinks.items()
                }
        view.set_playing(False)
        paused = view.position
        for _ in range(20):
            app.processEvents()
            time.sleep(0.005)
        if args.snapshot_dir and view._pool_active:
            args.snapshot_dir.mkdir(parents=True, exist_ok=True)
            for name, surface in (("source", view._source_surface), ("distorted", view._distorted_surface)):
                if not surface._image.isNull():
                    surface._image.save(str(args.snapshot_dir / f"{name}.png"))
        result = {"maximum_decoders": maximum, "position_ms": view.position,
                  "paused_position_ms": paused, "switches": switches,
                  "status": statuses[-8:], "positions": len(samples), "details": {str(k): v for k, v in view._details.items()},
                  "first_frame_seconds": samples[0][0] - start if samples else None}
        result["peak_rss_mib"] = peak_rss / 1024**2
        result["max_running_av_difference_ms"] = max(av_errors, default=None)
        if cpu_start is not None:
            result["steady_cpu_cores"] = (sum(process.cpu_times()[:2]) - cpu_start) / (time.monotonic() - measured_start)
        result["native_streams"] = {}
        if view._native_pool is not None:
            for key, entry in view._native_pool.entries.items():
                pipeline = entry[0]
                result["native_streams"][str(key)] = {
                    "caps": pipeline._negotiated_description(),
                    "sinks": playing_stats[str(key)],
                    "position": pipeline.poll().position_ms,
                }
            if args.snapshot_dir:
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                for source_side in (False, True):
                    view.show_source(source_side)
                    app.processEvents()
                    time.sleep(.1)
                    view.screen().grabWindow(int(view.winId())).save(str(
                        args.snapshot_dir / ("source.png" if source_side else "distorted.png")))
        result["attempt_errors"] = {str(k): w.attempt_errors for k, w in view._pool.items() if w.attempt_errors}
        if len(samples) > 1 and not args.exercise_seek:
            wall = samples[-1][0] - samples[0][0]
            result["media_seconds_per_wall_second"] = (samples[-1][1] - samples[0][1]) / 1000 / wall
        print(json.dumps(result, indent=2))
        if maximum > 4 or not samples:
            raise RuntimeError("playback pool smoke test failed")
        if args.require_gstreamer and view._native_pool is None:
            raise RuntimeError("Expected native GStreamer, but playback fell back")
    finally:
        view.clear()
        deadline = time.monotonic() + 15
        while view.live_workers() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if view.live_workers():
            raise RuntimeError("decoder did not shut down")
        view.close()


if __name__ == "__main__":
    main()
