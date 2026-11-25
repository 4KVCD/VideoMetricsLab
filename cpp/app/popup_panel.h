// A panel a button opens as a popup (Qt::Popup): Add/remove metrics'
// (MetricsPanel) and the tab bar's Export panel (ExportPanel). It closes on
// a press outside it or on Esc; a second click on its button closes it too.
// Its button is drawn pressed while it is open, as a Qt button's menu holds
// its button down (docs/ARCHITECTURE.md, "Buttons that open a panel").
#pragma once

#include <QFrame>
#include <QPointer>

#include <functional>

class QAbstractButton;
class QHideEvent;
class QKeyEvent;
class QMouseEvent;
class QShowEvent;

namespace vml::app {

class PopupPanel : public QFrame {
    Q_OBJECT

public:
    explicit PopupPanel(QWidget* parent = nullptr);
    ~PopupPanel() override;

    // The button that opens the panel: a press on it while the panel is open
    // closes the panel, and is not passed on to the button, whose click would
    // open it again at once. A press anywhere else outside the panel closes
    // it and still reaches what it lands on (a tab clicked is switched to).
    // A button anchor is held down while the panel is open, however it was
    // opened or closed.
    void set_anchor(QWidget* anchor);

    // Opens a panel (`open`) on `button`'s press, as Qt opens a button's
    // menu, not on its click: a click's release comes a whole press later
    // (80-150 ms as people click), which reads as lag. The button stays down
    // from the press, and the panel keeps it so until it closes, as the
    // popup takes the release. `pressed` alone, for a click() (UI
    // Automation's Invoke) and Space emit it too: each opens the panel once.
    static void open_on_press(QAbstractButton* button, std::function<void()> open);

protected:
    void hideEvent(QHideEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    // The anchor drawn down while the panel is open (a button's), or up.
    void hold_anchor_down(bool down);

    QPointer<QWidget> anchor_;
    QMetaObject::Connection anchor_released_;  // its released signal's, set_anchor's
};

}  // namespace vml::app
