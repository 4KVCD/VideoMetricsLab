#include "popup_panel.h"

#include <QAbstractButton>
#include <QKeyEvent>
#include <QMouseEvent>

namespace vml::app {

namespace {

// The time of the press on a panel's button that closed the panel, kept on
// the button: that press replayed to it must not open the panel again.
constexpr char closing_press[] = "vmlClosingPress";

// Takes from a panel's button the press that has just closed its panel,
// known by its time (a replay keeps the press's time; a new press has its
// own).
class ClosingPressFilter : public QObject {
public:
    using QObject::QObject;

    bool eventFilter(QObject* watched, QEvent* event) override {
        if (event->type() != QEvent::MouseButtonPress && event->type() != QEvent::MouseButtonDblClick) {
            return false;
        }
        const QVariant closing = watched->property(closing_press);
        if (!closing.isValid() || closing.toULongLong() != static_cast<QMouseEvent*>(event)->timestamp()) {
            return false;
        }
        watched->setProperty(closing_press, QVariant());
        return true;
    }
};

}  // namespace

PopupPanel::PopupPanel(QWidget* parent) : QFrame(parent, Qt::Popup) {}

PopupPanel::~PopupPanel() {
    // Deleted while open, with no hide first: the button is let up still.
    if (isVisible()) {
        hold_anchor_down(false);
    }
}

void PopupPanel::set_anchor(QWidget* anchor) {
    // Connected once an anchor: a panel kept and opened again (the Export
    // panel's open_under sets its anchor each time) gained a connection a
    // time.
    if (anchor == anchor_) {
        return;
    }
    disconnect(anchor_released_);
    anchor_ = anchor;
    if (auto* button = qobject_cast<QAbstractButton*>(anchor)) {
        // A click() (UI Automation's Invoke) lets the button up after its
        // pressed signal has opened the panel: down again while it is open.
        anchor_released_ = connect(button, &QAbstractButton::released, this, [this] {
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
    button->installEventFilter(new ClosingPressFilter(button));
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
    // Qt::WA_NoMouseReplay asks Qt not to, but once any popup of the window
    // has been closed with Esc, Qt replays the press anyway, later and with
    // the press's own time (Qt 6.12, Windows): the button's filter takes it
    // (open_on_press, ClosingPressFilter).
    if (anchor_ != nullptr && !rect().contains(event->position().toPoint())) {
        const QPoint at = anchor_->mapFromGlobal(event->globalPosition().toPoint());
        const bool on_anchor = anchor_->isVisible() && anchor_->rect().contains(at);
        setAttribute(Qt::WA_NoMouseReplay, on_anchor);
        if (on_anchor) {
            anchor_->setProperty(closing_press, QVariant::fromValue<qulonglong>(event->timestamp()));
        }
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
