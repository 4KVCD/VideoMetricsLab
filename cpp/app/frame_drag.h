// Dragging a zoomed frame with the left mouse button, for the views that show
// one: the hand cursors, and how far each move of the mouse went.
#pragma once

#include <QPointF>

#include <functional>
#include <optional>

class QMouseEvent;
class QWidget;

namespace vml::app {

class FrameDrag {
public:
    // `can_drag` says whether the frame is larger than the view just now;
    // `moved` is given each move of the mouse during a drag. Both are called
    // with `widget` alive, which owns this.
    FrameDrag(QWidget& widget, std::function<bool()> can_drag, std::function<void(QPointF)> moved);

    // An open hand where the frame can be dragged, else the usual cursor; left
    // alone during a drag.
    void refresh_cursor();

    // The widget's mouse events: each returns whether it took the event, which
    // otherwise goes to the widget's base class.
    bool press(QMouseEvent* event);
    bool move(QMouseEvent* event);
    bool release(QMouseEvent* event);

private:
    QWidget& widget_;
    std::function<bool()> can_drag_;
    std::function<void(QPointF)> moved_;
    std::optional<QPointF> from_;
};

}  // namespace vml::app
