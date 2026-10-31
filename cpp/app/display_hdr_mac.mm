// display_hdr.h on macOS: the EDR headroom of the NSScreen a window is on
// (platform::display_hdr_from_edr), and a watch on the displays' settings.
#include "display_hdr.h"

#import <AppKit/AppKit.h>

#include <QGuiApplication>
#include <QWidget>

#include <atomic>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "vml/app_log.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.frame_compare");

// Only Cocoa's windows are views: the offscreen platform's ids (the tests')
// are counters.
bool cocoa() { return QGuiApplication::platformName() == QStringLiteral("cocoa"); }

std::string text(NSString* value) { return value != nil ? std::string(value.UTF8String) : std::string(); }

std::string format_g(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%g", value);
    return buffer;
}

}  // namespace

std::optional<platform::DisplayHdrInfo> window_display_hdr(QWidget* window) {
    // winId() would make a window native that is not yet: only one that is
    // is asked.
    if (window == nullptr || !cocoa() || window->internalWinId() == 0) {
        return std::nullopt;
    }
    NSView* view = (__bridge NSView*)reinterpret_cast<void*>(window->internalWinId());
    NSScreen* screen = view.window.screen;
    if (screen == nil) {
        return std::nullopt;  // not on a screen yet
    }
    const double potential = screen.maximumPotentialExtendedDynamicRangeColorComponentValue;
    platform::DisplayHdrInfo info = platform::display_hdr_from_edr(potential, text(screen.localizedName));
    static platform::DisplayHdrInfo logged;
    if (info != logged) {
        logged = info;
        logger.info("Display " + info.device_name + ": EDR headroom " + format_g(potential) + " (now " +
                    format_g(screen.maximumExtendedDynamicRangeColorComponentValue) + ", reference " +
                    format_g(screen.maximumReferenceExtendedDynamicRangeColorComponentValue) + ")");
    }
    return info;
}

struct DisplayWatch::Impl {
    std::shared_ptr<std::atomic<bool>> alive = std::make_shared<std::atomic<bool>>(true);
    std::vector<id> observers;
};

DisplayWatch::DisplayWatch(std::function<void()> changed) : impl_(std::make_unique<Impl>()) {
    if (!cocoa()) {
        return;
    }
    // Any window's: asking again changes nothing when nothing did. A window's
    // own moves between displays come as Qt's screen changes.
    const std::shared_ptr<std::atomic<bool>> alive = impl_->alive;
    for (NSNotificationName name :
         {NSApplicationDidChangeScreenParametersNotification, NSWindowDidChangeScreenProfileNotification}) {
        impl_->observers.push_back([[NSNotificationCenter defaultCenter]
            addObserverForName:name
                        object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification*) {
                      // One already queued when the watch ended does nothing.
                      if (alive->load()) {
                          changed();
                      }
                    }]);
    }
}

DisplayWatch::~DisplayWatch() {
    impl_->alive->store(false);
    for (id observer : impl_->observers) {
        [[NSNotificationCenter defaultCenter] removeObserver:observer];
    }
}

}  // namespace vml::app
