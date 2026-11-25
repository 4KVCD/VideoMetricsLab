// A panel a button opens as a popup (Qt::Popup): Add/remove metrics'
// (MetricsPanel) and the tab bar's Export panel (ExportPanel). It closes on
// a press outside it or on Esc; a second click on its button closes it too.
#pragma once

#include <QFrame>
#include <QPointer>

#include <functional>

class QAbstractButton;
class QKeyEvent;
class QMouseEvent;

namespace vml::app {

class PopupPanel : public QFrame {
    Q_OBJECT

public:
    explicit PopupPanel(QWidget* parent = nullptr);

    // The button that opens the panel: a press on it while the panel is open
    // closes the panel, and is not passed on to the button, whose click would
    // open it again at once. A press anywhere else outside the panel closes
    // it and still reaches what it lands on (a tab clicked is switched to).
    void set_anchor(QWidget* anchor) { anchor_ = anchor; }

    // Opens a panel (`open`) on `button`'s press, as Qt opens a button's
    // menu, not on its click: a click's release comes a whole press later
    // (80-150 ms as people click), which reads as lag. The button is drawn
    // up again at once, as the popup takes the release. `pressed` alone, for
    // a click() (UI Automation's Invoke) and Space emit it too: each opens
    // the panel once.
    static void open_on_press(QAbstractButton* button, std::function<void()> open);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QPointer<QWidget> anchor_;
};

}  // namespace vml::app
