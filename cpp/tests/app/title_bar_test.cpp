// The title bar's RepeatedMouseFilter: with Qt::ExpandedClientAreaHint, Qt
// delivers each mouse press and release twice on Windows, so a menu a button
// opened on its press closed at once, and Qt made no double clicks (#178).
#include <QGuiApplication>
#include <QMouseEvent>
#include <QStyleHints>
#include <QWidget>
#include <QWindow>

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "title_bar.h"

namespace vml::app {
namespace {

using Events = std::vector<std::tuple<QEvent::Type, QPointF, qint64>>;

// The window's events as its widgets get them: the kinds the filter lets
// through, and those it sends itself.
struct Delivered : QObject {
    std::vector<QEvent::Type> types;
    bool eventFilter(QObject*, QEvent* event) override {
        const QEvent::Type type = event->type();
        if (type == QEvent::MouseButtonPress || type == QEvent::MouseButtonRelease ||
            type == QEvent::MouseButtonDblClick) {
            types.push_back(type);
        }
        return false;
    }
};

// Runs `events` (a kind, a place, a time in milliseconds) through the filter,
// on the window it makes double clicks on, as Qt would deliver them.
std::vector<QEvent::Type> delivered(const Events& events) {
    QWidget widget;
    widget.winId();  // its QWindow
    QWindow* window = widget.windowHandle();
    qint64 now = 0;
    RepeatedMouseFilter filter(&widget, [&now] { return now; });
    Delivered seen;
    window->installEventFilter(&seen);
    for (const auto& [type, at, time] : events) {
        now = time;
        QMouseEvent event(type, at, at, at, Qt::LeftButton,
                          type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton, Qt::NoModifier);
        if (!filter.eventFilter(window, &event)) {
            QCoreApplication::sendEvent(window, &event);  // as Qt goes on to deliver what is let through
        }
    }
    window->removeEventFilter(&seen);
    return seen.types;
}

constexpr QEvent::Type press = QEvent::MouseButtonPress;
constexpr QEvent::Type release = QEvent::MouseButtonRelease;
constexpr QEvent::Type twice = QEvent::MouseButtonDblClick;

TEST(RepeatedMouseFilter, DropsTheRepeatOfEachPressAndRelease) {
    const QPointF at(400, 415);
    // As Qt 6.12 delivers one click to such a window: press, press, release,
    // release, a few milliseconds apart.
    EXPECT_EQ(delivered({{press, at, 0}, {press, at, 12}, {release, at, 60}, {release, at, 63}}),
              (std::vector<QEvent::Type>{press, release}));
    // A press elsewhere, or the same place after 100 ms, is a press of its own.
    EXPECT_EQ(delivered({{press, at, 0}, {press, QPointF(500, 415), 5}, {press, QPointF(500, 415), 2000}}),
              (std::vector<QEvent::Type>{press, press, press}));
}

TEST(RepeatedMouseFilter, MakesTheWindowsDoubleClicks) {
    const QPointF at(400, 415);
    const int interval = QGuiApplication::styleHints()->mouseDoubleClickInterval();
    // A doubled double click, Qt's own double click in it dropped: press,
    // release, then the second press and its double click.
    EXPECT_EQ(delivered({{press, at, 0}, {press, at, 5}, {release, at, 60}, {release, at, 63},
                         {press, at, 120}, {twice, at, 121}, {press, at, 125}, {release, at, 180}, {release, at, 183}}),
              (std::vector<QEvent::Type>{press, release, press, twice, release}));
    // Slower than the system's double-click time: two clicks.
    EXPECT_EQ(delivered({{press, at, 0}, {release, at, 50}, {press, at, interval + 100}, {release, at, interval + 150}}),
              (std::vector<QEvent::Type>{press, release, press, release}));
    // A third quick press starts a new double click rather than making another.
    EXPECT_EQ(delivered({{press, at, 0}, {release, at, 30}, {press, at, 120}, {release, at, 150}, {press, at, 240},
                         {release, at, 270}}),
              (std::vector<QEvent::Type>{press, release, press, twice, release, press, release}));
}

}  // namespace
}  // namespace vml::app
