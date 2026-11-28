// The window's keyboard shortcuts for its main actions, and the list of them
// (Keyboard shortcuts, F1). One table, shortcuts(), makes both the window's
// actions (MainWindow::add_shortcuts) and the list's rows, so the list
// cannot say a key the window does not have.
#pragma once

#include <QKeySequence>
#include <QString>

#include <vector>

#include "app_dialog.h"

class QAbstractButton;
class QGridLayout;

namespace vml::app {

// What a shortcut does: most press a button of the window's.
enum class Command {
    add_videos,        // Videos: Add files
    choose_reference,  // Videos: the reference's Browse
    import_run,        // Videos: Import run file...
    calculate,         // Calculate metrics
    pause,             // Pause / Resume
    cancel,            // Cancel the run
    export_all,        // the tab bar's Export
    settings,          // the Settings tab
    shortcuts,         // this list
};

struct Shortcut {
    Command command;
    QKeySequence keys;
};

// The shortcuts anywhere in the window, in the list's order.
const std::vector<Shortcut>& shortcuts();

// What a command does, as the list says it (translated).
QString command_text(Command command);

// The key of the window's tabs, Ctrl+1 for the first, up to the eighth.
QKeySequence tab_keys(int index);

// Removes the selected test videos, in the Videos table.
QKeySequence remove_keys();

// Adds `button`'s shortcut to its tooltip ("Shortcut: F5"), after the
// tooltip it had when first called (none: the shortcut alone), or, `shown`
// false, takes it away again (the shortcuts turned off in Settings). Called
// after the tooltip is set.
void show_shortcut_in_tip(QAbstractButton* button, const QKeySequence& keys, bool shown = true);

// The list: the window's shortcuts, the Videos table's and Video Compare's
// keys (its key guide's, key_guide.h), each beside what it does, in the
// app's own dialog. Esc or Close closes it.
class ShortcutsDialog : public AppDialog {
    Q_OBJECT

public:
    explicit ShortcutsDialog(QWidget* parent = nullptr);

private:
    void add_heading(const QString& text);
    void add_row(const QString& keys, const QString& action);

    QGridLayout* grid_ = nullptr;
    int rows_ = 0;
};

}  // namespace vml::app
