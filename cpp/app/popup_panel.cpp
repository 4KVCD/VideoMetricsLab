#include "popup_panel.h"

#include <QAbstractButton>
#include <QKeyEvent>
#include <QMouseEvent>

namespace vml::app {

PopupPanel::PopupPanel(QWidget* parent) : QFrame(parent, Qt::Popup) {}

void PopupPanel::open_on_press(QAbstractButton* button, std::function<void()> open) {
    QObject::connect(button, &QAbstractButton::pressed, button, [button, open = std::move(open)] {
        button->setDown(false);
        open();
    });
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
