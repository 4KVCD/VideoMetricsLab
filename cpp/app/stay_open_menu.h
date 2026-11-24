// A menu of check boxes that stays open while boxes are ticked, so several
// can be switched in one visit: the Saved Results table's columns.
#pragma once

#include <QAction>
#include <QMenu>
#include <QMouseEvent>

namespace vml::app {

class StayOpenMenu : public QMenu {
public:
    using QMenu::QMenu;

protected:
    void mouseReleaseEvent(QMouseEvent* event) override {
        QAction* action = actionAt(event->position().toPoint());
        if (action != nullptr && action->isCheckable() && action->isEnabled()) {
            action->trigger();
            return;
        }
        QMenu::mouseReleaseEvent(event);
    }
};

}  // namespace vml::app
