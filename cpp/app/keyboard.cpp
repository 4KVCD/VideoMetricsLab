#include "keyboard.h"

#include <QAbstractButton>
#include <QApplication>
#include <QDialog>
#include <QKeyEvent>
#include <QMenu>
#include <QPainter>
#include <QPushButton>
#include <QTabBar>
#include <QTimer>
#include <QToolButton>

#include <vector>

#include "theme.h"

namespace vml::app::keyboard {

namespace {

// Keys that move the focus, or move within the focused control as the
// focus would: after one, the ring shows.
bool moves_focus(const QKeyEvent& key) {
    if (key.modifiers().testFlag(Qt::AltModifier)) {
        return false;
    }
    switch (key.key()) {
    case Qt::Key_Tab:
    case Qt::Key_Backtab:
    case Qt::Key_Up:
    case Qt::Key_Down:
    case Qt::Key_Left:
    case Qt::Key_Right:
    case Qt::Key_Home:
    case Qt::Key_End:
    case Qt::Key_PageUp:
    case Qt::Key_PageDown:
        return true;
    default:
        return false;
    }
}

// Whether the ring goes around `widget` when it has the focus: not in a menu
// or a combo box's open list, which mark their own current item.
bool ringed(const QWidget& widget) {
    const QWidget* window = widget.window();
    return qobject_cast<const QMenu*>(window) == nullptr && !window->inherits("QComboBoxPrivateContainer");
}

class FocusFollower : public QObject {
public:
    explicit FocusFollower(QObject* parent) : QObject(parent) {}

    bool visible = false;

    void queue() {
        if (!queued_) {
            queued_ = true;
            QTimer::singleShot(0, this, [this] {
                queued_ = false;
                update_rings();
            });
        }
    }

    // The ring where the focus is, in Modern while it is visible; every other
    // window's hidden.
    void update_rings() {
        QWidget* focus = QApplication::focusWidget();
        const bool wanted = visible && focus != nullptr && focus->isVisible() && ringed(*focus) &&
                            theme::current_interface() == theme::Interface::modern;
        std::erase_if(rings_, [](const QPointer<FocusRing>& ring) { return ring.isNull(); });
        FocusRing* shown = nullptr;
        if (wanted) {
            for (const QPointer<FocusRing>& ring : rings_) {
                if (ring->parentWidget() == focus->window()) {
                    shown = ring;
                }
            }
            if (shown == nullptr) {
                shown = new FocusRing(focus->window());
                rings_.emplace_back(shown);
            }
            shown->follow(focus);
        }
        for (const QPointer<FocusRing>& ring : rings_) {
            if (ring != shown) {
                ring->follow(nullptr);
            }
        }
    }

    FocusRing* ring_in(const QWidget* window) const {
        for (const QPointer<FocusRing>& ring : rings_) {
            if (!ring.isNull() && ring->parentWidget() == window) {
                return ring;
            }
        }
        return nullptr;
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override {
        switch (event->type()) {
        case QEvent::KeyPress: {
            const auto* key = static_cast<QKeyEvent*>(event);
            if (watched != QApplication::focusWidget()) {
                break;  // the same key on its way up to the focus's parents
            }
            if (moves_focus(*key) && !visible) {
                if (key->key() == Qt::Key_Tab || key->key() == Qt::Key_Backtab) {
                    visible = true;
                    queue();
                } else {
                    // An arrow or paging key shows the ring only if it moved
                    // the focus, seen once the key has been handled: one a
                    // panel takes for itself (Video Compare's arrows switch
                    // the test video) leaves a clicked control unringed, and
                    // Space there still plays.
                    QTimer::singleShot(0, this, [this, before = QPointer<QWidget>(QApplication::focusWidget())] {
                        if (!visible && QApplication::focusWidget() != before) {
                            visible = true;
                            queue();
                        }
                    });
                }
            }
            if (visible && press_with_enter(*key)) {
                return true;
            }
            break;
        }
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonDblClick:
        case QEvent::TouchBegin:
            if (visible) {
                visible = false;
                queue();
            }
            break;
        case QEvent::FocusIn: {
            const Qt::FocusReason reason = static_cast<QFocusEvent*>(event)->reason();
            if (reason == Qt::TabFocusReason || reason == Qt::BacktabFocusReason || reason == Qt::ShortcutFocusReason) {
                visible = true;
            }
            queue();
            break;
        }
        case QEvent::Move:
        case QEvent::Resize:
        case QEvent::Show:
        case QEvent::Hide:
        case QEvent::LayoutRequest:
            // The focused control, or what holds it (a page scrolled), moved.
            if (visible && watched->isWidgetType()) {
                QWidget* focus = QApplication::focusWidget();
                auto* widget = static_cast<QWidget*>(watched);
                if (focus != nullptr && (widget == focus || widget->isAncestorOf(focus)) &&
                    qobject_cast<FocusRing*>(widget) == nullptr) {
                    queue();
                }
            }
            break;
        default:
            break;
        }
        return QObject::eventFilter(watched, event);
    }

private:
    // Enter on a focused button presses it, as Space does. Not in a dialog,
    // where Enter is its default button's (Qt's own, as on Windows).
    static bool press_with_enter(const QKeyEvent& key) {
        if ((key.key() != Qt::Key_Return && key.key() != Qt::Key_Enter) ||
            (key.modifiers() & ~Qt::KeypadModifier) != Qt::NoModifier || key.isAutoRepeat()) {
            return false;
        }
        auto* button = qobject_cast<QAbstractButton*>(QApplication::focusWidget());
        if (button == nullptr || !button->isEnabled() ||
            (qobject_cast<QPushButton*>(button) == nullptr && qobject_cast<QToolButton*>(button) == nullptr) ||
            qobject_cast<QDialog*>(button->window()) != nullptr) {
            return false;
        }
        button->click();  // pressed, then clicked: a button opening a panel on its press opens it
        return true;
    }

    bool queued_ = false;
    std::vector<QPointer<FocusRing>> rings_;
};

FocusFollower* follower = nullptr;

}  // namespace

void install() {
    if (follower != nullptr || qApp == nullptr) {
        return;
    }
    follower = new FocusFollower(qApp);
    qApp->installEventFilter(follower);
    QObject::connect(qApp, &QApplication::focusChanged, follower, [] { follower->queue(); });
}

bool focus_visible() { return follower != nullptr && follower->visible; }

void set_focus_visible(bool visible) {
    if (follower != nullptr) {
        follower->visible = visible;
        follower->update_rings();
    }
}

FocusRing* ring_in(QWidget* window) { return follower != nullptr ? follower->ring_in(window) : nullptr; }

FocusRing::FocusRing(QWidget* window) : QWidget(window) {
    setObjectName(QStringLiteral("focusRing"));
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setFocusPolicy(Qt::NoFocus);
    hide();
}

void FocusRing::follow(QWidget* target) {
    if (target_ != target) {
        if (auto* bar = qobject_cast<QTabBar*>(target_.data())) {
            disconnect(bar, nullptr, this, nullptr);
        }
        target_ = target;
        // A tab bar's ring is around its current tab, which the arrows change.
        if (auto* bar = qobject_cast<QTabBar*>(target)) {
            connect(bar, &QTabBar::currentChanged, this, [this] { follow(target_); });
        }
    }
    QRect area;
    if (target != nullptr) {
        // Only the part shown: a control half scrolled out of its page has
        // its ring around the half on screen.
        area = target->visibleRegion().boundingRect();
        if (auto* bar = qobject_cast<QTabBar*>(target); bar != nullptr && bar->currentIndex() >= 0) {
            area = area.intersected(bar->tabRect(bar->currentIndex()));
        }
    }
    if (area.isEmpty()) {
        hide();
        return;
    }
    drawn_ = QRect(target->mapTo(parentWidget(), area.topLeft()), area.size());
    const int out = gap + width;
    setGeometry(drawn_.adjusted(-out, -out, out, out));
    raise();
    show();
    update();
}

void FocusRing::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    QPen pen(QApplication::palette().color(QPalette::Accent), width);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    // The pen centred on a line half its width inside the edge; the corners
    // round as the controls' do (6 px), widened by the gap.
    const qreal inset = width / 2.0;
    const QRectF line = QRectF(rect()).adjusted(inset, inset, -inset, -inset);
    const qreal radius = 6 + gap + inset;
    painter.drawRoundedRect(line, radius, radius);
}

}  // namespace vml::app::keyboard
