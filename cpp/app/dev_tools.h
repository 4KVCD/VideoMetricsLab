// Tools for the app's developers and for AI sessions testing it, never for
// its users: keys that show the window's state as text, save a picture of
// it, switch its looks and languages, draw its Tab order and its controls'
// names, and fake a run's states. Off unless the environment variable
// VML_DEV is 1 as the app starts; not in the shortcut list (F1), not a
// setting, not translated. tools\vm\README.md says how sessions use them.
#pragma once

#include <QKeySequence>
#include <QPointer>
#include <QString>
#include <QTimer>
#include <QWidget>

#include <filesystem>
#include <functional>
#include <vector>

#include <nlohmann/json.hpp>

namespace vml::app::dev {

// Whether the developers' tools are on: VML_DEV=1.
bool enabled();

enum class Command {
    dump_state,      // the window's state as JSON (describe), to ui-state.json
    save_picture,    // the window alone as window.png
    next_look,       // the next of the nine looks, light and dark (VML Modern)
    tab_order,       // numbers on every control Tab stops at, on and off
    names,           // each control outlined with its object name, on and off
    next_language,   // the next language, then the long-text one: restarts
    forced_state,    // the next of a run's hard-to-reach states, faked
};

struct Key {
    Command command;
    QKeySequence keys;
    const char* what;  // for tools\vm\README.md and the log, in English
};

// Ctrl+Alt+Shift+F1 to F7: no published shortcut uses Ctrl+Alt+Shift, and
// no text box takes a function key.
const std::vector<Key>& keys();

// The folder the files go in: the profile's log folder.
std::filesystem::path output_folder();

// The application's state for someone who cannot see it: every window shown
// (the main one first, then dialogs and popups), each with its visible
// controls as a tree: class, object name, geometry in the window, enabled,
// focus, text, the current tab of each tab bar, a table's headings and
// first rows. Plus the interface, the look, the language and the device
// pixel ratio (geometry is in logical pixels).
nlohmann::json describe(QWidget* main_window);

// One control and the visible ones inside it, as describe() lists them.
nlohmann::json describe_widget(const QWidget* widget, const QWidget* window);

// The controls Tab stops at in `window`, in order, as Qt's Tab moves
// through them: visible, enabled, taking Tab focus.
std::vector<QWidget*> tab_stops(QWidget* window);

// Drawn over a window, taking no clicks: the Tab order's numbers, or each
// named control's outline and name. Redrawn a few times a second while on.
class Overlay : public QWidget {
    Q_OBJECT

public:
    enum class Mode { none, tab_order, names };
    explicit Overlay(QWidget* window);
    Mode mode() const { return mode_; }
    // Shows `mode`, or hides the overlay with none.
    void set_mode(Mode mode);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void follow_window();
    Mode mode_ = Mode::none;
    QTimer refresh_;
};

// Calls `run` for each of keys()' key presses anywhere in the application,
// dialogs and popups included (outside the event filter, a moment later).
// Once; nothing unless enabled().
void install(QObject* owner, std::function<void(Command)> run);

}  // namespace vml::app::dev
