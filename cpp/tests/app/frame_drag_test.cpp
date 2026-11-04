// Dragging a zoomed frame: the cursors, how far it moved, and the events it leaves alone.
#include <QMouseEvent>
#include <QPointF>
#include <QWidget>

#include <gtest/gtest.h>

#include "frame_drag.h"

namespace vml::app {
namespace {

QMouseEvent mouse(QEvent::Type type, QPointF at, Qt::MouseButton button = Qt::LeftButton) {
    return QMouseEvent(type, at, at, button, button, Qt::NoModifier);
}

TEST(FrameDrag, FollowsTheMouseWhereTheFrameCanMove) {
    QWidget widget;
    bool movable = false;
    QPointF total;
    FrameDrag drag(
        widget, [&movable] { return movable; }, [&total](QPointF moved) { total += moved; });

    QMouseEvent refused = mouse(QEvent::MouseButtonPress, {10, 10});
    EXPECT_FALSE(drag.press(&refused));
    drag.refresh_cursor();
    EXPECT_EQ(widget.cursor().shape(), Qt::ArrowCursor);

    movable = true;
    drag.refresh_cursor();
    EXPECT_EQ(widget.cursor().shape(), Qt::OpenHandCursor);
    QMouseEvent right = mouse(QEvent::MouseButtonPress, {10, 10}, Qt::RightButton);
    EXPECT_FALSE(drag.press(&right));
    QMouseEvent early = mouse(QEvent::MouseMove, {11, 11});
    EXPECT_FALSE(drag.move(&early));

    QMouseEvent press = mouse(QEvent::MouseButtonPress, {10, 10});
    EXPECT_TRUE(drag.press(&press));
    EXPECT_EQ(widget.cursor().shape(), Qt::ClosedHandCursor);
    QMouseEvent first = mouse(QEvent::MouseMove, {13, 14});
    EXPECT_TRUE(drag.move(&first));
    QMouseEvent second = mouse(QEvent::MouseMove, {15, 10});
    EXPECT_TRUE(drag.move(&second));
    EXPECT_EQ(total, QPointF(5, 0));

    drag.refresh_cursor();  // left alone during a drag
    EXPECT_EQ(widget.cursor().shape(), Qt::ClosedHandCursor);
    QMouseEvent release = mouse(QEvent::MouseButtonRelease, {15, 10});
    EXPECT_TRUE(drag.release(&release));
    EXPECT_EQ(widget.cursor().shape(), Qt::OpenHandCursor);
    EXPECT_FALSE(drag.release(&release));
}

}  // namespace
}  // namespace vml::app
