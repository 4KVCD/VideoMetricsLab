#include "popup_panel.h"

#include <QAbstractButton>
#include <QKeyEvent>
#include <QMouseEvent>

namespace vml::app {

PopupPanel::PopupPanel(QWidget* parent) : QFrame(parent, Qt::Popup) {}

PopupPanel::~PopupPanel() {
    // Deleted while open, with no hide first: the button is let up still.
    if (isVisible()) {
        hold_anchor_down(false);
    }
}

void PopupPanel::set_anchor(QWidget* anchor) {
    anchor_ = anchor;
    if (auto* button = qobject_cast<QAbstractButton*>(anchor)) {
        // A click() (UI Automation's Invoke) lets the button up after its
        // pressed signal has opened the panel: down again while it is open.
        connect(button, &QAbstractButton::released, this, [this] {
            if (isVisible()) {
                hold_anchor_down(true);
            }
        });
    }
}

void PopupPanel::hold_anchor_down(bool down) {
    if (auto* button = qobject_cast<QAbstractButton*>(anchor_.data())) {
        button->setDown(down);
    }
}

void PopupPanel::open_on_press(QAbstractButton* button, std::function<void()> open) {
    QObject::connect(button, &QAbstractButton::pressed, button, [open = std::move(open)] { open(); });
}

void PopupPanel::showEvent(QShowEvent* event) {
    QFrame::showEvent(event);
    hold_anchor_down(true);
}

void PopupPanel::hideEvent(QHideEvent* event) {
    QFrame::hideEvent(event);
    hold_anchor_down(false);
}

void PopupPanel::mousePressEvent(QMouseEvent* event) {
    // A popup receives every press while it is open. One outside it closes
    // it (QWidget's own handling), and Qt then replays the press to the
    // widget under the pointer; on the anchor button that is a click that
    // opens the panel again, so the button would never close it.
    if (anchor_ != nullptr && !rect().contains(event->position().toPoint())) {
        const QPoint at = anchor_->mapFromGlobal(event->globalPosition().toPoint());
        setAttribute(Qt::WA_NoMouseReplay, anchor_->isVisible() && anchor_->rect().contains(at));
    }
    QFrame::mousePressEvent(event);
}

void PopupPanel::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape) {
        close();
        return;
    }
    QFrame::keyPressEvent(event);
}

}  // namespace vml::app
