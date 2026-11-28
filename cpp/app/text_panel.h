// A panel of text a button opens (a PopupPanel): its title, the text,
// selectable with the mouse and the keyboard, and Copy, which puts all of it
// on the clipboard. Video Compare's Advanced info, which was a tooltip: it
// could not be selected or copied, and went away as the pointer moved.
#pragma once

#include <QString>

#include "popup_panel.h"

class QLabel;
class QPushButton;
class QTimer;

namespace vml::app {

class TextPanel : public PopupPanel {
    Q_OBJECT

public:
    explicit TextPanel(const QString& title, QWidget* parent = nullptr);

    // The text, plain, a line each; changed while the panel is open, it is
    // shown at once (a selection in it is lost).
    void set_text(const QString& text);
    QString text() const;

    QPushButton* copy_button() const { return copy_; }

private:
    QLabel* text_ = nullptr;
    QPushButton* copy_ = nullptr;
    QTimer* copied_ = nullptr;  // "Copied" said on the button a moment
};

}  // namespace vml::app
