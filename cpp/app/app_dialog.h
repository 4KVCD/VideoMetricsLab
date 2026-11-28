// The window's dialogs, one kind of each, in the app's look: what every
// dialog that waits for the user is built on (docs/ARCHITECTURE.md, "Asking
// and telling the user").
//
// AppDialog is a QDialog with, on Windows, the app's own title bar
// (TitleBar, as the main window's: the title and a close button, in the
// theme's colours) in place of the system's; elsewhere, and in a contrast
// theme, the system's, as the main window has. Its contents go in body().
//
// show_message is the window's message: an icon drawn by the app in the
// theme's colours (never the system's), the text, buttons and an optional
// tick box. Dialogs (dialogs.h) calls it; nothing else opens a QMessageBox.
#pragma once

#include <QColor>
#include <QDialog>
#include <QPixmap>
#include <QString>

#include <functional>
#include <optional>
#include <utility>
#include <vector>

class QBoxLayout;
class QLabel;
class QProgressBar;
class QPushButton;

namespace vml::app {

class TitleBar;

class AppDialog : public QDialog {
public:
    AppDialog(QWidget* parent, const QString& title);

    // Where the dialog's contents go: give it a layout.
    QWidget* body() const { return body_; }

protected:
    void showEvent(QShowEvent* event) override;
    bool nativeEvent(const QByteArray& type, void* message, qintptr* result) override;

private:
    QWidget* body_ = nullptr;
#ifdef _WIN32
    TitleBar* bar_ = nullptr;
#endif
};

// A form's OK and Cancel at the foot of `layout` (the dialog's body's),
// right-aligned: OK (`ok_text`, else "OK") the accent's and Enter's,
// accepting; Cancel rejecting. OK, to enable as the form allows.
QPushButton* add_dialog_buttons(AppDialog& dialog, QBoxLayout* layout, const QString& ok_text = QString());

// What a message is about, which its icon says: information and a question
// in the accent, a warning in amber, a failure in the error colour, and
// `danger` (what can't be undone, such as deleting every saved score) in the
// error colour, filled, with its headline in that colour.
enum class MessageKind { information, question, warning, failure, danger };

// The icon of a message of `kind`, `size` pixels square, in the theme in use.
QPixmap message_icon(MessageKind kind, int size, qreal ratio = 1.0);

struct MessageButton {
    // `primary`: the accent's (the answer most take); `danger`: the error
    // colour's (what can't be undone); `plain`: neither.
    enum class Role { plain, primary, danger };
    QString text;
    Role role = Role::plain;
};

struct Message {
    MessageKind kind = MessageKind::information;
    QString title;
    // A line above the text, bold (in the error colour for `danger`): what
    // the answer does. Empty: none.
    QString headline;
    QString text;
    std::vector<MessageButton> buttons;  // none: OK alone
    int default_button = 0;              // Enter's
    int escape_button = -1;              // Esc's and the close button's; -1: the last
    QString check;                       // a tick box's text; empty: none
    QString details;                     // more, behind Show details (an update's notes); empty: none
};

struct MessageAnswer {
    int button = -1;  // the button clicked (Message::buttons' index), or Esc's
    bool checked = false;
};

// Shows `message` over `parent` and waits for the answer.
MessageAnswer show_message(QWidget* parent, const Message& message);

// The same, not waiting: the dialog, made and not shown, which calls
// `answered` when it closes (it then deletes itself). For what the window
// offers without stopping (an update available).
AppDialog* message_dialog(QWidget* parent, const Message& message, std::function<void(MessageAnswer)> answered);

// Qt's colour picker in an AppDialog. With `reset` (its text), a button
// that closes it asking for the colour's default back.
struct ColorChoice {
    enum class Outcome { cancelled, chosen, reset };
    Outcome outcome = Outcome::cancelled;
    QColor color;  // the colour chosen
};
ColorChoice choose_color(QWidget* parent, const QString& title, const QColor& initial,
                         const QString& reset = QString());

// A line of text: the text entered, or nothing for Cancel.
std::optional<QString> ask_line(QWidget* parent, const QString& title, const QString& label, const QString& text);

// A task's progress in an AppDialog, with Cancel (rejected()): a label and a
// bar, busy (no end known) while its range is 0 to 0.
class ProgressDialog : public AppDialog {
public:
    ProgressDialog(QWidget* parent, const QString& title, const QString& label);
    void set_label(const QString& text);
    void set_range(int minimum, int maximum);
    void set_value(int value);

private:
    QLabel* label_ = nullptr;
    QProgressBar* bar_ = nullptr;
};

}  // namespace vml::app
