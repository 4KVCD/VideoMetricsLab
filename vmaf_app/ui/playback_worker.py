"""Bounded independent FFmpeg streams for a rolling comparison neighbourhood.

Workers decode ahead by only three frames. The consumer owns one media clock;
workers never run their own playback clocks or pass image bytes through signals.
"""
from __future__ import annotations

import threading
from collections import deque

from PySide6.QtCore import QThread, Signal

from vmaf_app.core import proc
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.video_playback import build_video_series_command

#: Buffers of frames the view is done with, kept to read the next frames
#: into (StreamDecodeWorker.recycle). A worker's frames are at most three
#: queued, three in the view's history and one being read: a few spares
#: keep them all in use; more are dropped.
_SPARE_FRAMES = 4


def _read_into(pipe, buffer) -> int:
    """Fills `buffer` from `pipe`; the bytes read, fewer at its end. Read
    into it directly: BufferedReader.readinto passes what fits its own
    buffer through that buffer, a second copy of every frame."""
    view = memoryview(buffer)
    got = 0
    while got < len(view):
        read = pipe.readinto(view[got:])
        if not read:
            break
        got += read
    return got


class StreamDecodeWorker(QThread):
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, comparison, side, start_frame, settings, plan, maximum, parent=None):
        super().__init__(parent)
        self.comparison = comparison
        self.side = side
        self.start_frame = start_frame
        self.settings = settings
        self.plan = plan
        self.maximum = maximum
        self._condition = threading.Condition()
        self._frames = deque()
        self._spare = []
        self._frame_bytes = 0
        self._cancelled = False
        self._handle = ProcessHandle()
        self.ended = False
        self.error = ""
        self.attempt_errors = []

    def cancel(self):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        # Ended from a thread of its own: finding FFmpeg's process tree
        # (proc.process_tree) took 20-60 ms on the window's thread, for each
        # decoder a seek stopped -- four, when three test videos decode.
        threading.Thread(target=self._handle.terminate, name="decoder-stop", daemon=True).start()

    def drain_through(self, frame):
        """Transfer ownership of due frames; leave future frames queued."""
        with self._condition:
            result = []
            while self._frames and self._frames[0][0] <= frame:
                result.append(self._frames.popleft())
            if result:
                # Room for the decoder only now. Woken at every look (100 a
                # second), a paused one woke for nothing.
                self._condition.notify_all()
            return result

    def recycle(self, payload):
        """A frame's buffer the view no longer holds or shows, read into
        again: a new one for each frame, a 4K frame's pages were faulted in
        and zeroed by the reading thread, and freed by the window's."""
        with self._condition:
            if len(self._spare) < _SPARE_FRAMES and len(payload) == self._frame_bytes:
                self._spare.append(payload)

    def _buffer(self):
        with self._condition:
            if self._spare:
                return self._spare.pop()
        return bytearray(self._frame_bytes)

    def latest_frame_number(self):
        with self._condition:
            return self._frames[-1][0] if self._frames else None

    def _put(self, frame, payload):
        with self._condition:
            while len(self._frames) >= 3 and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                return False
            self._frames.append((frame, payload))
            return True

    def run(self):
        from vmaf_app.core.video_playback import playback_dimensions

        width, height = playback_dimensions(self.comparison, self.maximum)
        frame_bytes = self._frame_bytes = width * height * 4
        accel = self.plan.source if self.side == "source" else self.plan.distorted
        # The GPU's own decoder first, never FFmpeg's Vulkan decoder: see
        # build_video_series_command. NVDEC's pictures stay on the GPU where
        # FFmpeg can hand them to Vulkan there ("interop").
        modes = ("software", "cpu") if not accel else (
            ("interop", "transfer", "software", "cpu") if accel == "cuda" else ("transfer", "software", "cpu"))
        errors = self.attempt_errors
        next_frame = self.start_frame
        for mode in modes:
            if self._cancelled:
                return
            command = build_video_series_command(
                [self.comparison], next_frame, self.settings, [self.plan], self.maximum,
                realtime=True, processing=mode, side=self.side, paced=False,
            )
            process = None
            frames = None
            reader = None
            tail = deque(maxlen=30)
            first = True
            attempt_error = ""
            try:
                # Through the large pipe (proc.FRAME_PIPE_BYTES): with
                # subprocess's own, of 4 KB, 4K RGBA came at 30 frames a
                # second; with this, 72.
                process, frames = proc.popen_piped(command)
                self._handle.attach(process.pid)
                if self._cancelled:
                    self._handle.terminate()
                    return

                def drain(pipe=process.stderr, tail=tail):
                    for line in iter(pipe.readline, b""):
                        tail.append(line)

                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                while not self._cancelled:
                    # The large pipe takes a whole frame: a read or two each.
                    payload = self._buffer()
                    read = _read_into(frames, payload)
                    if read != frame_bytes:
                        if read:
                            attempt_error = "Decoder returned a truncated frame."
                        break
                    if first:
                        first = False
                        detail = {
                            "interop": "GPU decode + GPU tone mapping/RGB",
                            "transfer": "hardware decode + GPU tone mapping/RGB (host transfer)",
                            "software": "software decode + GPU tone mapping/RGB",
                            "cpu": "CPU fallback (GPU processing unavailable)",
                        }[mode]
                        if errors:
                            detail += " · retry: " + errors[-1].splitlines()[0][:180]
                        self.ready.emit(detail)
                    if not self._put(next_frame, payload):
                        return
                    next_frame += 1
                if not self._cancelled:
                    process.wait()
            except Exception as exc:
                attempt_error = str(exc)
            finally:
                if process is not None:
                    if process.poll() is None:
                        proc.terminate(process)
                    process.wait()
                    self._handle.detach()
                    if reader is not None:
                        reader.join()
                    if frames is not None:
                        frames.close()
                    if process.stderr is not None:
                        process.stderr.close()
            if self._cancelled:
                return
            detail = b"".join(tail).decode("utf-8", errors="replace").strip()
            if not first and not attempt_error and process is not None and process.returncode == 0:
                self.ended = True
                return
            # Retain already queued complete frames and resume at the first
            # missing frame. This also handles device loss after startup,
            # without replaying old frames or creating unbounded retry loops.
            errors.append(detail or attempt_error or "Decoder produced no frames.")
        self.error = errors[-1] if errors else "Decoder produced no frames."
        self.failed.emit(self.error[-1500:])
