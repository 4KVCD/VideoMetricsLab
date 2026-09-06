"""Bounded independent FFmpeg streams for a rolling comparison neighbourhood.

Workers decode ahead by only three frames. The consumer owns one media clock;
workers never run their own playback clocks or pass image bytes through signals.
"""
from __future__ import annotations

import subprocess
import threading
from collections import deque

from PySide6.QtCore import QThread, Signal

from vmaf_app.core import proc
from vmaf_app.core.process_control import ProcessHandle
from vmaf_app.core.video_playback import build_video_series_command


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
        self._cancelled = False
        self._handle = ProcessHandle()
        self.ended = False
        self.error = ""
        self.attempt_errors = []

    def cancel(self):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        self._handle.terminate()

    def drain_through(self, frame):
        """Transfer ownership of due frames; leave future frames queued."""
        with self._condition:
            result = []
            while self._frames and self._frames[0][0] <= frame:
                result.append(self._frames.popleft())
            self._condition.notify_all()
            return result

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
        frame_bytes = width * height * 4
        accel = self.plan.source if self.side == "source" else self.plan.distorted
        modes = ("vulkan", "transfer", "software", "cpu") if accel else ("software", "cpu")
        errors = self.attempt_errors
        for mode in modes:
            if self._cancelled:
                return
            command = build_video_series_command(
                [self.comparison], self.start_frame, self.settings, [self.plan], self.maximum,
                realtime=True, processing=mode, side=self.side, paced=False,
            )
            process = None
            reader = None
            tail = deque(maxlen=30)
            first = True
            try:
                process = proc.popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
                self._handle.attach(process.pid)
                if self._cancelled:
                    self._handle.terminate()
                    return

                def drain(pipe=process.stderr, tail=tail):
                    for line in iter(pipe.readline, b""):
                        tail.append(line)

                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                frame = self.start_frame
                while not self._cancelled:
                    # BufferedReader assembles this in C, avoiding thousands
                    # of tiny Python reads and their GIL overhead on Windows.
                    payload = process.stdout.read(frame_bytes)
                    if len(payload) != frame_bytes:
                        break
                    if first:
                        first = False
                        detail = {
                            "vulkan": "Vulkan GPU decode + GPU tone mapping/RGB",
                            "transfer": "hardware decode + GPU tone mapping/RGB (host transfer)",
                            "software": "software decode + GPU tone mapping/RGB",
                            "cpu": "CPU fallback (GPU processing unavailable)",
                        }[mode]
                        if errors:
                            detail += " · retry: " + errors[-1].splitlines()[0][:180]
                        self.ready.emit(detail)
                    if not self._put(frame, payload):
                        return
                    frame += 1
                if not self._cancelled:
                    process.wait()
            except Exception as exc:
                errors.append(str(exc))
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
                    self._handle.detach()
                    if reader is not None:
                        reader.join()
                    if process.stdout is not None:
                        process.stdout.close()
                    if process.stderr is not None:
                        process.stderr.close()
            if self._cancelled:
                return
            detail = b"".join(tail).decode("utf-8", errors="replace").strip()
            if not first:
                self.ended = True
                if process is not None and process.returncode:
                    self.error = detail or "Video decoding stopped unexpectedly."
                    self.failed.emit(self.error[-1500:])
                return
            errors.append(detail or "Decoder produced no frames.")
        self.error = errors[-1] if errors else "Decoder produced no frames."
        self.failed.emit(self.error[-1500:])
