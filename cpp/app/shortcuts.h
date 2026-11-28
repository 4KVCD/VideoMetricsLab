// The window's main actions that have keys, and the list of the keys
// (Keyboard shortcuts, F1). The keys themselves are key_bindings.h's: its
// one table makes the window's actions (MainWindow::add_shortcuts) and the
// list's rows, so the list cannot say a key the window does not have.
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

// Adds `button`'s shortcut to its tooltip ("Shortcut: F5"), after the
// tooltip it had when first called (none: the shortcut alone), or, `shown`
// false, takes it away again (the shortcuts turned off in Settings). Called
// after the tooltip is set.
void show_shortcut_in_tip(QAbstractButton* button, const QKeySequence& keys, bool shown = true);

// The list: every action's keys as they are now (key_bindings.h), by where
// they work, each beside what it does, in the app's own dialog. Esc or
// Close closes it.
class ShortcutsDialog : public AppDialog {
    Q_OBJECT

public:
    explicit ShortcutsDialog(QWidget* parent = nullptr);
    // The rows again, from the keys in use now.
    void refresh();

private:
    void add_heading(const QString& text);
    void add_row(const QString& keys, const QString& action);

    QGridLayout* grid_ = nullptr;
    int rows_ = 0;
};

}  // namespace vml::app
