// The app's title bar's hit test (title_bar.h), where it can be checked
// without a window: which resize edge a point is on.

#ifdef _WIN32

#include <QPoint>
#include <QSize>

#include <gtest/gtest.h>

#include <windows.h>

#include "title_bar.h"

namespace vml::app {
namespace {

// The window the maintainer measured on #178 (175% scaling): a 2526 x 1589
// client area, resize metrics of 12 pixels, the Close button over its right
// edge. The left, right and bottom edges resize only on Windows' border
// outside the client area, so the Close button's right side closes; the top
// keeps a band inside, with nothing above the bar.
TEST(TitleBar, ResizeEdgesLeaveTheClientAreaToItsContents) {
    const QSize client(2526, 1589);
    const auto edge = [&client](int x, int y) { return resize_edge(QPoint(x, y), client, 12, 12); };
    // Inside the left, right and bottom edges: the contents, the Close
    // button's right side among them (6 px in, below the top band).
    EXPECT_EQ(edge(3, 800), HTNOWHERE);
    EXPECT_EQ(edge(2526 - 6, 30), HTNOWHERE);
    EXPECT_EQ(edge(2526 - 4, 800), HTNOWHERE);
    EXPECT_EQ(edge(800, 1589 - 4), HTNOWHERE);
    EXPECT_EQ(edge(800, 800), HTNOWHERE);
    // Windows' invisible border, outside the client area.
    EXPECT_EQ(edge(-4, 800), HTLEFT);
    EXPECT_EQ(edge(2526 + 3, 800), HTRIGHT);
    EXPECT_EQ(edge(800, 1589 + 3), HTBOTTOM);
    EXPECT_EQ(edge(-3, 1589 - 5), HTBOTTOMLEFT);
    EXPECT_EQ(edge(2526 + 2, 1589 + 2), HTBOTTOMRIGHT);
    EXPECT_EQ(edge(5, 1589 + 3), HTBOTTOMLEFT);
    // The top: a band inside, its corners within the side metric.
    EXPECT_EQ(edge(800, 5), HTTOP);
    EXPECT_EQ(edge(5, 5), HTTOPLEFT);
    EXPECT_EQ(edge(2526 - 6, 5), HTTOPRIGHT);
    EXPECT_EQ(edge(800, 12), HTNOWHERE);  // below the band: the bar
}

}  // namespace
}  // namespace vml::app

#endif
