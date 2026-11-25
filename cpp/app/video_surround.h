// The colour around a frame in Video Compare, and of its screen while a
// frame loads: black, as a television shows a film's letterbox or pillarbox
// bars. The bars the app cuts off (black-bar detection) are then shown as
// they were, black, with no lighter band where the picture ends. Every view
// and presenter of the tab fills with it (VideoCompareView, the native
// presenters, PairedFrameWidget, FrameView). With nothing to compare, the
// tab's screen takes the theme's colours instead (FrameView::show_empty).
#pragma once

#include <QColor>

#include <array>

namespace vml::app {

// As RGBA floats, for the presenters' clears.
inline constexpr std::array<float, 4> video_surround_rgba{0.0f, 0.0f, 0.0f, 1.0f};

inline QColor video_surround() { return QColor(0, 0, 0); }

}  // namespace vml::app
