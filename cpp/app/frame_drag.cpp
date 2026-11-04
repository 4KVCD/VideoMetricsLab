#include "frame_drag.h"

#include <QMouseEvent>
#include <QWidget>

#include <utility>

namespace vml::app {

FrameDrag::FrameDrag(QWidget& widget, std::function<bool()> can_drag, std::function<void(QPointF)> moved)
    : widget_(widget), can_drag_(std::move(can_drag)), moved_(std::move(moved)) {}

void FrameDrag::refresh_cursor() {
    if (from_) {
        return;
    }
    if (can_drag_()) {
        widget_.setCursor(Qt::OpenHandCursor);
    } else {
        widget_.unsetCursor();
    }
}

bool FrameDrag::press(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton || !can_drag_()) {
        return false;
    }
    from_ = event->position();
    widget_.setCursor(Qt::ClosedHandCursor);
    event->accept();
    return true;
}

bool FrameDrag::move(QMouseEvent* event) {
    if (!from_) {
        return false;
    }
    const QPointF moved = event->position() - *from_;
    from_ = event->position();
    moved_(moved);
    event->accept();
    return true;
}

bool FrameDrag::release(QMouseEvent* event) {
    if (!from_ || event->button() != Qt::LeftButton) {
        return false;
    }
    from_.reset();
    refresh_cursor();
    event->accept();
    return true;
}

}  // namespace vml::app
