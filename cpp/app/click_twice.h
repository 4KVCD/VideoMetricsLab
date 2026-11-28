// Clicking twice to confirm, in place of a window that asks (docs/
// ARCHITECTURE.md, "Asking and telling the user"): the first click on a
// button arms it, in the error colour, with a line saying what the second
// click does (shown beside it by the owner); the second, while it is armed,
// does it, however soon it comes: a double-click is a confident click
// twice, not a slip to guard against. It lets go after a few seconds, when
// another button is armed, or when asked (another decision: a run starting,
// another button clicked).
#pragma once

#include <QObject>
#include <QPointer>
#include <QString>

class QLabel;
class QPushButton;
class QTimer;

namespace vml::app {

class ClickTwice : public QObject {
    Q_OBJECT

public:
    // `hold_ms`: how long a button stays armed.
    explicit ClickTwice(QObject* parent, int hold_ms = 6000);

    // A click on `button`: true when it confirms what the first click
    // armed; else `button` is armed, with `line`, and false.
    bool click(QPushButton* button, const QString& line);
    void disarm();

    QPushButton* armed() const { return armed_; }
    const QString& line() const { return line_; }

    // `label` shows the armed button's line, in the error colour and bold,
    // and nothing while none is armed: the line beside or under a button
    // the owner has no other use for (Delete profile's, Delete preset's).
    // The label keeps a line's height empty and never widens its row, so
    // nothing beside or under it moves as the line comes and goes.
    void show_line_in(QLabel* label);
    // The look of an armed button's line, for a label that shows other
    // text too (the Videos footer's length note).
    static void style_as_armed(QLabel* label, bool armed);

signals:
    // A button armed or let go.
    void changed();

private:
    void arm(QPushButton* button, const QString& line);

    QTimer* timer_ = nullptr;
    QPointer<QPushButton> armed_;
    QString armed_role_;  // the button's role before, given back as it lets go
    QString line_;
};

}  // namespace vml::app
