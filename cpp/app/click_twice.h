// Clicking twice to confirm, in place of a window that asks (docs/
// ARCHITECTURE.md, "Asking and telling the user"): the first click on a
// button arms it, in the error colour, with a line saying what the second
// click does (shown beside it by the owner); the second, while it is armed,
// does it. It lets go after a few seconds, when another button is armed, or
// when asked (another decision: a run starting, another button clicked).
// A second click within a double-click's time is not taken: a double-click
// is one decision, not two.
#pragma once

#include <QObject>
#include <QPointer>
#include <QString>

#include <functional>

class QPushButton;
class QTimer;

namespace vml::app {

class ClickTwice : public QObject {
    Q_OBJECT

public:
    // Milliseconds, steadily rising (the tests give their own).
    using Clock = std::function<qint64()>;

    // `clock`: QElapsedTimer's where none is given. `hold_ms`: how long a
    // button stays armed.
    explicit ClickTwice(QObject* parent, Clock clock = {}, int hold_ms = 6000);

    // A click on `button`: true when it confirms what the first click
    // armed; else `button` is armed, with `line`, and false.
    bool click(QPushButton* button, const QString& line);
    void disarm();

    QPushButton* armed() const { return armed_; }
    const QString& line() const { return line_; }

signals:
    // A button armed or let go.
    void changed();

private:
    void arm(QPushButton* button, const QString& line);

    Clock clock_;
    QTimer* timer_ = nullptr;
    QPointer<QPushButton> armed_;
    QString armed_role_;  // the button's role before, given back as it lets go
    QString line_;
    qint64 armed_at_ = 0;
};

}  // namespace vml::app
