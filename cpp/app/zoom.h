// Video Compare's zoom (the Python app's vmaf_app/ui/zoom.py): the frame
// fitted to the view, or at a size of its own, and which part of it is in
// view, moved by dragging it.
//
// One Zoom serves all of the tab's views (the still frames now, playback
// later), so that switching between the source and an encode keeps the same
// part of the picture at the same size. Sizes are in the views' logical
// pixels; `ratio` is the screen's device pixels per logical pixel.
#pragma once

#include <QString>

#include <array>
#include <optional>
#include <utility>

namespace vml::app {

// The zooms that may be typed in, as factors: 10% to 1600%.
inline constexpr double minimum_zoom = 0.1;
inline constexpr double maximum_zoom = 16.0;

// Where a frame is drawn in a view: its top-left corner and its size, in
// the view's logical pixels. Larger than the view when zoomed in.
struct Placement {
    double x = 0;
    double y = 0;
    double width = 0;
    double height = 0;
};

class Zoom {
public:
    using Extent = std::pair<double, double>;  // width, height

    // Nothing to fit the frame to the view; else its size, 1.0 for one of
    // its pixels on one of the screen's.
    std::optional<double> factor;
    // The frame's point in the middle of the view, as fractions of its width
    // and height.
    std::pair<double, double> centre{0.5, 0.5};

    // Where a `frame`-sized picture (its pixels) goes in `view`.
    Placement placement(Extent view, Extent frame, double ratio) const;
    // The part of the frame in view, as fractions of it (left, top, right,
    // bottom), and where that part is drawn within the view.
    struct Visible {
        std::array<double, 4> fractions{};
        Placement drawn;
    };
    Visible visible(Extent view, Extent frame, double ratio) const;
    // Whether the frame is larger than the view, so dragging moves it.
    bool can_drag(Extent view, Extent frame, double ratio) const;
    // Moves the frame by (dx, dy) view pixels, as far as it can go; whether
    // it moved.
    bool drag(double dx, double dy, Extent view, Extent frame, double ratio);
};

// A zoom as typed: "150", "150%", "150 %" or "1,5 %"... -> 1.5; nothing when
// it is not a number. Kept within minimum_zoom and maximum_zoom.
std::optional<double> parse_percent(const QString& text);

// 1.5 -> "150%"; 0.125 -> "12.5%".
QString percent_text(double factor);

}  // namespace vml::app
