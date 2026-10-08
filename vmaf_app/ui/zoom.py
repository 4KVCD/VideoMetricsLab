"""Video Compare's zoom: the frame fitted to the view, or at a size of its
own, and which part of it is in view, moved by dragging it.

One Zoom serves the tab's every view -- the still frames and both kinds of
playback, FFmpeg's frames drawn here and GStreamer's on the GPU -- so a
switch between the source and an encode, or between still and playing,
keeps the same part of the picture at the same size. Sizes are the views'
logical pixels; `ratio` is the screen's device pixels to one of those.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt

#: The zooms that may be typed in, as factors: 10% to 1600%.
MINIMUM, MAXIMUM = 0.1, 16.0


@dataclass(frozen=True)
class Placement:
    """Where a frame is drawn in a view: its top-left corner and its size,
    in the view's logical pixels. Larger than the view when zoomed in."""

    x: float
    y: float
    width: float
    height: float


def _centred(centre: float, view: float, size: float) -> float:
    """`centre` (a fraction of the frame) kept so that a frame larger than
    the view covers all of it; a smaller one is centred."""
    if size <= view:
        return 0.5
    margin = view / (2 * size)
    return min(max(centre, margin), 1 - margin)


class Zoom:
    """`factor`: None to fit the frame to the view; else its size, 1.0 for
    one of its pixels on one of the screen's. `centre`: the frame's point in
    the middle of the view, as fractions of its width and height."""

    def __init__(self) -> None:
        self.factor: float | None = None
        self.centre = (0.5, 0.5)

    def placement(self, view: tuple[float, float], frame: tuple[int, int], ratio: float) -> Placement:
        """Where a `frame`-sized picture (its pixels) goes in `view`."""
        (view_width, view_height), (frame_width, frame_height) = view, frame
        scale = (min(view_width / frame_width, view_height / frame_height) if self.factor is None
                 else self.factor / ratio)
        width, height = frame_width * scale, frame_height * scale
        x = view_width / 2 - _centred(self.centre[0], view_width, width) * width
        y = view_height / 2 - _centred(self.centre[1], view_height, height) * height
        return Placement(x, y, width, height)

    def can_drag(self, view: tuple[float, float], frame: tuple[int, int], ratio: float) -> bool:
        """Whether the frame is larger than the view, so dragging moves it."""
        place = self.placement(view, frame, ratio)
        return place.width > view[0] + 0.5 or place.height > view[1] + 0.5

    def drag(self, dx: float, dy: float, view: tuple[float, float], frame: tuple[int, int], ratio: float) -> bool:
        """Moves the frame by (dx, dy) view pixels, as far as it can go;
        whether it moved."""
        if self.factor is None:
            return False
        place = self.placement(view, frame, ratio)
        current = ((view[0] / 2 - place.x) / place.width, (view[1] / 2 - place.y) / place.height)
        moved = (_centred(current[0] - dx / place.width, view[0], place.width),
                 _centred(current[1] - dy / place.height, view[1], place.height))
        if moved == current:
            return False
        self.centre = moved
        return True

    def visible(self, view: tuple[float, float], frame: tuple[int, int], ratio: float):
        """The part of the frame in view, as fractions of it (left, top,
        right, bottom), and where that part is drawn (a Placement within
        the view)."""
        place = self.placement(view, frame, ratio)
        left, top = max(0.0, place.x), max(0.0, place.y)
        right, bottom = min(view[0], place.x + place.width), min(view[1], place.y + place.height)
        fractions = ((left - place.x) / place.width, (top - place.y) / place.height,
                     (right - place.x) / place.width, (bottom - place.y) / place.height)
        return fractions, Placement(left, top, right - left, bottom - top)


def parse_percent(text: str) -> float | None:
    """A zoom as typed: "150", "150%", "150 %" or "1,5 %"... -> 1.5; None
    when it is not a number. Kept within MINIMUM and MAXIMUM."""
    cleaned = text.replace("%", "").replace(",", ".").strip()
    try:
        percent = float(cleaned)
    except ValueError:
        return None
    if percent != percent or percent in (float("inf"), float("-inf")):
        return None
    return min(max(percent / 100, MINIMUM), MAXIMUM)


def percent_text(factor: float) -> str:
    """1.5 -> "150%"; 0.125 -> "12.5%"."""
    return f"{round(factor * 100, 1):g}%"


class DragsZoomedFrame:
    """For a widget showing a frame through a Zoom: holding the left button
    down and moving drags a frame larger than the view, under a hand cursor.

    The widget gives _zoom_context() -- (zoom, view size, frame size,
    ratio), or None when it shows no frame -- and _zoom_dragged(), to show
    the frame where it has moved to."""

    _drag_from = None

    def _zoom_context(self):
        raise NotImplementedError

    def _zoom_dragged(self) -> None:
        raise NotImplementedError

    def _can_drag(self) -> bool:
        context = self._zoom_context()
        return context is not None and context[0].can_drag(*context[1:])

    def refresh_cursor(self) -> None:
        """An open hand where the frame can be dragged; else the usual one."""
        if self._drag_from is None:
            if self._can_drag():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._can_drag():
            self._drag_from = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        dx, dy = position.x() - self._drag_from.x(), position.y() - self._drag_from.y()
        self._drag_from = position
        context = self._zoom_context()
        if context is not None and context[0].drag(dx, dy, *context[1:]):
            self._zoom_dragged()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_from is not None and event.button() == Qt.LeftButton:
            self._drag_from = None
            self.refresh_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)
