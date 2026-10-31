// The HDR state of the display a window is on, and a watch on the system's
// display changes, for Video Compare's display-aware preview. Windows asks
// its display configuration (platform::query_display_hdr); macOS its
// NSScreen's EDR headroom (display_hdr_mac.mm), which needs AppKit and so is
// the window's, not vml_platform's.
#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "vml/platform/system.h"

class QWidget;

namespace vml::app {

// The HDR state of the display `window` (a top-level widget) is on; nothing
// where it cannot be known yet: on macOS, a window with no native window or
// not on a screen yet, or one of a platform that is not Cocoa (the tests'
// offscreen windows, whose ids are not views).
std::optional<platform::DisplayHdrInfo> window_display_hdr(QWidget* window);

// Calls `changed`, on the window's thread, when the displays' settings may
// have changed (macOS: their parameters or a colour profile, as when HDR is
// switched on an external display); never elsewhere, where a window's moves
// between displays say it. Ends with this object.
class DisplayWatch {
public:
    explicit DisplayWatch(std::function<void()> changed);
    ~DisplayWatch();
    DisplayWatch(const DisplayWatch&) = delete;
    DisplayWatch& operator=(const DisplayWatch&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vml::app
