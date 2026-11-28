// The window from the keyboard: where the focus is, shown while it is moved
// with the keys.
//
// VML Modern's style sheet draws its own buttons, tabs and tables, and Qt
// then draws their focus faintly or not at all: a tinted box behind a
// button's text, nothing on a tab or a table. Modern draws a ring of its
// accent around the control the keys have moved to instead, as Windows 11
// and browsers do (:focus-visible): from the first Tab, Shift+Tab, or
// arrow or shortcut that moves the focus, until a click. A click focuses with no ring, as
// Windows' own controls show none for one. VML Classic is drawn by the
// system's style, whose own focus rectangles follow the same rule.
//
// While the ring shows, Enter presses the focused button as Space does (the
// default button of a dialog keeps Enter there).
#pragma once

#include <QPointer>
#include <QWidget>

namespace vml::app::keyboard {

// Starts following the focus for the whole application, once (more calls
// do nothing). The window calls it as it is built.
void install();

// Whether the focus was moved with the keyboard since the last click: the
// focus ring shows (in Modern), and keys a control takes for itself go to it
// (Video Compare's Space presses a focused button rather than playing).
bool focus_visible();

// Shows the focus ring at once, as a key that moves the focus would (tests).
void set_focus_visible(bool visible);

// The ring drawn around the focused control, a widget of its own over the
// control's window (it takes no clicks). Exposed for tests.
class FocusRing : public QWidget {
    Q_OBJECT

public:
    explicit FocusRing(QWidget* window);
    // Around `target` (in this ring's window), or hidden with none.
    void follow(QWidget* target);
    QWidget* target() const { return target_; }
    // The ring's width and its gap from the control, in pixels.
    static constexpr int width = 2;
    static constexpr int gap = 1;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QPointer<QWidget> target_;
    QRect drawn_;  // the control's rectangle, in the window, that the ring goes around
};

// The ring in `window` now, or none (tests).
FocusRing* ring_in(QWidget* window);

}  // namespace vml::app::keyboard
