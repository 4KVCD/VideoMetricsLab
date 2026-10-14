// The dialogs that wait for the user (QMessageBox, QFileDialog), behind
// functions: the window's tests answer them in the user's place, as the test
// harness fails any dialog that opens.
#pragma once

#include <QString>

#include <functional>

class QWidget;

namespace vml::app {

struct Dialogs {
    // A message with an OK button: information, and a failure (critical).
    std::function<void(const QString& title, const QString& text)> inform;
    std::function<void(const QString& title, const QString& text)> alert;
    // A question answered Yes or No: whether Yes.
    std::function<bool(const QString& title, const QString& text)> ask;
    // The same with No the default button, for a question whose No keeps
    // things as they are: Enter does not undo work (whether Yes).
    std::function<bool(const QString& title, const QString& text)> ask_no_default;
    // The file dialogs; an empty answer is Cancel.
    std::function<QString(const QString& title, const QString& filter)> open_file;
    std::function<QString(const QString& title, const QString& suggested, const QString& filter)> save_file;
    // A folder, the dialog opening at `start` (none: wherever it opens).
    std::function<QString(const QString& title, const QString& start)> choose_folder;
};

// The real dialogs, over `parent`.
Dialogs standard_dialogs(QWidget* parent);

}  // namespace vml::app
