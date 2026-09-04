"""Two synchronized video surfaces for instant source/distorted switching."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QStackedLayout, QWidget

from vmaf_app.core.frame_extract import FrameComparison, frame_input_path


class VideoCompareView(QWidget):
    """Play both physical inputs and raise either surface without reopening it.

    The distorted player is the clock and audio master. The source is muted
    and periodically nudged back to the distorted position if the independent
    media clocks drift far enough to become visually meaningful.
    """

    position_changed = Signal(int)       # milliseconds
    playing_changed = Signal(bool)
    status_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #171717;")

        self.source_video = QVideoWidget(self)
        self.distorted_video = QVideoWidget(self)
        for video in (self.source_video, self.distorted_video):
            video.setAspectRatioMode(Qt.KeepAspectRatio)
            video.setFocusPolicy(Qt.StrongFocus)

        # StackAll leaves both native video surfaces visible and decoding;
        # setCurrentWidget only raises one. A conventional stacked widget
        # hides the other surface, which can leave it without a current frame
        # when S is pressed.
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackAll)
        layout.addWidget(self.source_video)
        layout.addWidget(self.distorted_video)
        layout.setCurrentWidget(self.distorted_video)
        self._stack = layout

        self.audio_output = QAudioOutput(self)
        self.source_player = self._new_source_player()
        self.distorted_player = self._new_distorted_player()

        self._source_path: Path | None = None
        self._distorted_path: Path | None = None
        self._wanted_playing = False
        self._showing_source = False
        self._max_drift_ms = 20
        self._pending_position = 0
        self._starting_pair = False
        self._pair_started = False
        self._source_frame_us = -1
        self._distorted_frame_us = -1
        self._pending_source_us: int | None = None
        self._resume_after_pending_source = False
        self._temporary_sync_pause = False
        self._source_load_pending = False
        self._distorted_load_pending = False

        self.source_video.videoSink().videoFrameChanged.connect(
            self._on_source_frame
        )
        self.distorted_video.videoSink().videoFrameChanged.connect(
            self._on_distorted_frame
        )
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(20)
        self._sync_timer.timeout.connect(self._synchronize)

    def _new_source_player(self) -> QMediaPlayer:
        player = QMediaPlayer(self)
        player.setVideoOutput(self.source_video)
        player.mediaStatusChanged.connect(
            lambda status: self._on_media_status("source", status)
        )
        player.errorOccurred.connect(
            lambda _error, text: self._on_error("source", text)
        )
        return player

    def _new_distorted_player(self) -> QMediaPlayer:
        player = QMediaPlayer(self)
        player.setVideoOutput(self.distorted_video)
        player.setAudioOutput(self.audio_output)
        player.positionChanged.connect(self._on_master_position)
        player.playbackStateChanged.connect(self._on_playback_state)
        player.mediaStatusChanged.connect(
            lambda status: self._on_media_status("distorted", status)
        )
        player.errorOccurred.connect(
            lambda _error, text: self._on_error("distorted", text)
        )
        return player

    @staticmethod
    def _retire_player(player: QMediaPlayer) -> None:
        player.blockSignals(True)
        player.stop()
        player.setVideoOutput(None)
        player.setAudioOutput(None)
        player.deleteLater()

    @staticmethod
    def can_play(comparison: FrameComparison) -> tuple[bool, str]:
        if comparison.resample_target is not None:
            return (
                False,
                "Video playback is unavailable for synthetic resolution tests; "
                "use Still frame to inspect their processed output.",
            )
        source = frame_input_path(comparison, "source")
        distorted = frame_input_path(comparison, "distorted")
        missing = next((path for path in (source, distorted) if not path.is_file()), None)
        if missing is not None:
            return False, f"Video file is missing: {missing}"
        return True, ""

    @property
    def is_playing(self) -> bool:
        return self.distorted_player.playbackState() == QMediaPlayer.PlayingState

    @property
    def playback_requested(self) -> bool:
        return self._wanted_playing

    @property
    def position(self) -> int:
        return self.distorted_player.position()

    def load(
        self,
        comparison: FrameComparison,
        position_ms: int,
        *,
        playing: bool = False,
    ) -> bool:
        available, reason = self.can_play(comparison)
        if not available:
            self.set_playing(False)
            self.status_changed.emit(reason)
            return False

        source = frame_input_path(comparison, "source").resolve()
        distorted = frame_input_path(comparison, "distorted").resolve()
        self._wanted_playing = playing
        position_ms = max(0, int(position_ms))
        self._pending_position = position_ms
        if comparison.fps > 0:
            # Correct once the clocks differ by roughly half a displayed
            # frame. Faster material is checked more often, without polling
            # faster than 8 ms or wasting work on low-frame-rate video.
            half_frame_ms = 500 / comparison.fps
            self._max_drift_ms = max(5, min(45, round(half_frame_ms)))
            self._sync_timer.setInterval(max(8, min(50, round(half_frame_ms))))

        # Most comparisons share a source. Preserve its live decoder while
        # left/right replaces only the distorted player, avoiding a visible
        # source interruption if S is held during the switch.
        source_changed = source != self._source_path
        distorted_changed = distorted != self._distorted_path
        if source_changed or distorted_changed:
            self._pair_started = False
        if (source_changed or distorted_changed) and playing:
            # Do not let the already-loaded side run ahead while its partner
            # opens. Both restart from the same clock once both report ready.
            self.source_player.pause()
            self.distorted_player.pause()
        if source_changed:
            self._source_frame_us = -1
            self._source_load_pending = True
            self._retire_player(self.source_player)
            self.source_player = self._new_source_player()
            self.source_player.setSource(QUrl.fromLocalFile(str(source)))
            self._source_path = source
        if distorted_changed:
            self._distorted_frame_us = -1
            self._distorted_load_pending = True
            self._retire_player(self.distorted_player)
            self.distorted_player = self._new_distorted_player()
            self.distorted_player.setSource(QUrl.fromLocalFile(str(distorted)))
            self._distorted_path = distorted

        self.source_player.setPosition(position_ms)
        self.distorted_player.setPosition(position_ms)
        self.show_source(self._showing_source)
        if playing:
            # Preserve position_ms even if the backend has not delivered its
            # LoadingMedia transition yet and still exposes the old status.
            self._wanted_playing = True
            self._start_pair_if_ready()
            self.playing_changed.emit(True)
        else:
            self.set_playing(False)
        self.status_changed.emit(f"Loaded {distorted.name}")
        return True

    def clear(self) -> None:
        self._wanted_playing = False
        self._pending_position = 0
        self._sync_timer.stop()
        for player in (self.source_player, self.distorted_player):
            player.stop()
            player.setSource(QUrl())
        self._source_path = None
        self._distorted_path = None
        self._source_frame_us = -1
        self._distorted_frame_us = -1
        self._pending_source_us = None
        self._resume_after_pending_source = False
        self._temporary_sync_pause = False
        self._pair_started = False
        self._source_load_pending = False
        self._distorted_load_pending = False
        self.playing_changed.emit(False)

    def set_position(self, position_ms: int) -> None:
        position_ms = max(0, int(position_ms))
        self._pending_position = position_ms
        self.distorted_player.setPosition(position_ms)
        self.source_player.setPosition(position_ms)

    def set_playing(self, playing: bool) -> None:
        self._wanted_playing = bool(playing)
        if playing and self._distorted_path is not None:
            # A newly assigned source reports position 0 while LoadingMedia.
            # Keep load()'s requested comparison position until it is ready;
            # otherwise left/right jumps a playing comparison back to frame 0.
            if self._ready(self.distorted_player):
                self._pending_position = self.distorted_player.position()
            self._start_pair_if_ready()
        else:
            self.distorted_player.pause()
            self.source_player.pause()
            self._sync_timer.stop()
            self._pair_started = False
        self.playing_changed.emit(bool(playing))

    def show_source(self, showing: bool) -> None:
        self._showing_source = bool(showing)
        if not showing:
            if self._resume_after_pending_source:
                self._resume_synchronized_pair()
            self._pending_source_us = None
            self._stack.setCurrentWidget(self.distorted_video)
            return
        if self._presented_frames_aligned():
            self._pending_source_us = None
            self._stack.setCurrentWidget(self.source_video)
            return

        # Do not flash a wrong source frame. Ask the hidden source decoder for
        # the frame currently on the distorted surface, then raise it as soon
        # as its sink reports that timestamp. No pixel download is involved.
        self._pending_source_us = self._distorted_frame_us
        if self._distorted_frame_us >= 0:
            if self.is_playing:
                # Freeze the visible distorted frame for the fraction of a
                # frame needed by source to catch it. That is preferable to
                # flashing an adjacent source frame, and both resume from the
                # exact same timestamp as soon as the sink confirms it.
                self._temporary_sync_pause = True
                self._resume_after_pending_source = True
                self.source_player.pause()
                self.distorted_player.pause()
                self._sync_timer.stop()
            self.source_player.setPlaybackRate(1.0)
            self.source_player.setPosition(round(self._distorted_frame_us / 1000))
        else:
            self.source_player.setPosition(self.distorted_player.position())
        self._stack.setCurrentWidget(self.distorted_video)

    def set_audio_enabled(self, enabled: bool) -> None:
        self.audio_output.setMuted(not enabled)

    def _synchronize(self) -> None:
        if not self.is_playing:
            return
        if self._source_frame_us >= 0 and self._distorted_frame_us >= 0:
            drift_ms = (self._source_frame_us - self._distorted_frame_us) / 1000
        else:
            drift_ms = self.source_player.position() - self.distorted_player.position()
        if abs(drift_ms) <= self._max_drift_ms:
            if self.source_player.playbackRate() != 1.0:
                self.source_player.setPlaybackRate(1.0)
            return
        if abs(drift_ms) <= self._max_drift_ms * 3:
            # A small speed nudge avoids an expensive keyframe seek for a
            # one-frame scheduling difference. Source has no audio, so this
            # correction is inaudible and normally lasts only a few frames.
            self.source_player.setPlaybackRate(1.12 if drift_ms < 0 else 0.88)
            return
        self.source_player.setPlaybackRate(1.0)
        self.source_player.setPosition(self.distorted_player.position())

    def _presented_frames_aligned(self) -> bool:
        if self._source_frame_us < 0 or self._distorted_frame_us < 0:
            return False
        return abs(self._source_frame_us - self._distorted_frame_us) <= (
            self._max_drift_ms * 1000
        )

    def _on_source_frame(self, frame) -> None:
        timestamp = int(frame.startTime())
        if timestamp >= 0:
            self._source_frame_us = timestamp
        target = self._pending_source_us
        if not self._showing_source or target is None or timestamp < 0:
            return
        if abs(timestamp - target) <= self._max_drift_ms * 1000:
            self._pending_source_us = None
            self._stack.setCurrentWidget(self.source_video)
            if self._resume_after_pending_source:
                self._resume_synchronized_pair()

    def _on_distorted_frame(self, frame) -> None:
        timestamp = int(frame.startTime())
        if timestamp >= 0:
            self._distorted_frame_us = timestamp

    @staticmethod
    def _ready(player: QMediaPlayer) -> bool:
        return player.mediaStatus() in {
            QMediaPlayer.LoadedMedia,
            QMediaPlayer.BufferingMedia,
            QMediaPlayer.BufferedMedia,
            QMediaPlayer.StalledMedia,
        }

    def _start_pair_if_ready(self) -> None:
        if (
            not self._wanted_playing
            or self._starting_pair
            or self._source_load_pending
            or self._distorted_load_pending
            or not self._ready(self.source_player)
            or not self._ready(self.distorted_player)
        ):
            return
        if (
            self._pair_started
            and self.source_player.isPlaying()
            and self.distorted_player.isPlaying()
        ):
            self._sync_timer.start()
            return
        self._starting_pair = True
        try:
            position = self._pending_position
            # Some backends retain PlayingState briefly across setSource().
            # Force a real state transition once the replacement media is
            # loaded; otherwise play() can be treated as a no-op at position
            # zero even though the old decoder has already been discarded.
            if not self._pair_started:
                self.source_player.stop()
                self.distorted_player.stop()
            self.source_player.setPosition(position)
            self.distorted_player.setPosition(position)
            self.source_player.play()
            self.distorted_player.play()
            self._pair_started = True
            self._sync_timer.start()
        finally:
            self._starting_pair = False

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlayingState
        if not playing:
            self._sync_timer.stop()
        if self._temporary_sync_pause:
            return
        self.playing_changed.emit(playing)

    def _resume_synchronized_pair(self) -> None:
        position = max(0, round(self._distorted_frame_us / 1000))
        self._pending_position = position
        self.source_player.setPosition(position)
        self.distorted_player.setPosition(position)
        self.source_player.play()
        self.distorted_player.play()
        self._sync_timer.start()
        self._resume_after_pending_source = False
        self._temporary_sync_pause = False

    def _on_master_position(self, position: int) -> None:
        if not self._distorted_load_pending:
            self.position_changed.emit(int(position))

    def _on_media_status(self, side: str, status: QMediaPlayer.MediaStatus) -> None:
        if status in {
            QMediaPlayer.LoadedMedia,
            QMediaPlayer.BufferingMedia,
            QMediaPlayer.BufferedMedia,
            QMediaPlayer.StalledMedia,
        }:
            if side == "source":
                self._source_load_pending = False
            else:
                self._distorted_load_pending = False
        self._start_pair_if_ready()
        if status == QMediaPlayer.EndOfMedia:
            self._wanted_playing = False
            self._pair_started = False
            self.source_player.pause()
            self.distorted_player.pause()
            self._sync_timer.stop()
            self.playing_changed.emit(False)
        elif status == QMediaPlayer.InvalidMedia:
            self.set_playing(False)
            player = self.source_player if side == "source" else self.distorted_player
            self.status_changed.emit(
                player.errorString() or f"Could not play {side} video."
            )

    def _on_error(self, side: str, text: str) -> None:
        self.set_playing(False)
        self.status_changed.emit(f"Could not play {side} video: {text or 'unknown error'}")
