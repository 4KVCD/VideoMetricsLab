// display_hdr.h on Windows (and Linux): the display configuration's HDR
// state; no watch, a window's moves say the changes.
#include "display_hdr.h"

#include <QWidget>

namespace vml::app {

std::optional<platform::DisplayHdrInfo> window_display_hdr(QWidget* window) {
    return platform::query_display_hdr(reinterpret_cast<void*>(window->winId()));
}

struct DisplayWatch::Impl {};

DisplayWatch::DisplayWatch(std::function<void()> /*changed*/) {}

DisplayWatch::~DisplayWatch() = default;

}  // namespace vml::app
