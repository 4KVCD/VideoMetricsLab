// The window's key and mouse bindings: one table of every action that has a
// key (its id, where it works, its default keys), and the user's changes to
// them from Settings > Shortcuts (Settings.key_bindings_json and
// mouse_bindings_json). The window's shortcuts, the buttons' tooltips, the
// list of keys (F1), Video Compare's keys and mouse, its key guide and the
// Settings page all read them here, so none can say a key another doesn't
// have.
#pragma once

#include <QKeySequence>
#include <QObject>
#include <QString>

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class QKeyEvent;

namespace vml::app {

class KeyBindingsSignal : public QObject {
    Q_OBJECT

signals:
    void changed();
};

// Where an action's keys work.
enum class KeyGroup {
    window,         // anywhere in the window (its shortcuts)
    tabs,           // anywhere: go to a tab
    videos,         // the Videos table
    video_compare,  // the Video Compare tab, unless a text box has the focus
};

struct KeyAction {
    const char* id;  // stable: what settings.json names it by
    KeyGroup group;
    std::vector<QKeySequence> defaults;  // at most two
};

// Every action that has a key, in the list's order.
const std::vector<KeyAction>& key_actions();
const KeyAction* key_action(std::string_view id);

// What an action does, as the list says it (translated).
QString action_text(std::string_view id);
// A group's heading (translated).
QString group_text(KeyGroup group);

// The keys an action has now: the user's, else its defaults. Empty when the
// user has unbound it.
std::vector<QKeySequence> keys_for(std::string_view id);
// Whether a key press is one of an action's keys (modifiers as pressed; the
// keypad's left out).
bool matches(const QKeyEvent& event, std::string_view id);

// The user's changes: changed actions only, each with its keys (empty:
// unbound).
using KeyBindings = std::map<std::string, std::vector<QKeySequence>>;
// From and to settings.json's text. Unknown actions and keys Qt can't read
// are left out.
KeyBindings parse_key_bindings(const std::string& json);
std::string key_bindings_json(const KeyBindings& bindings);

// The mouse in Video Compare: what each gesture does. Gestures by id, each to
// an action id ("" for nothing).
struct MouseGesture {
    const char* id;
    const char* default_action;
};
const std::vector<MouseGesture>& mouse_gestures();
QString gesture_text(std::string_view id);
// The actions a gesture can be given, "" (nothing) first.
const std::vector<std::string>& mouse_actions();
using MouseBindings = std::map<std::string, std::string>;  // changed gestures only
MouseBindings parse_mouse_bindings(const std::string& json);
std::string mouse_bindings_json(const MouseBindings& bindings);
// The action a gesture does now ("" for nothing).
std::string action_for_gesture(std::string_view gesture);

// The bindings in use from now on; key_bindings_signal() says so.
void use_bindings(KeyBindings keys, MouseBindings mouse);
const KeyBindings& key_bindings();
const MouseBindings& mouse_bindings();
KeyBindingsSignal* key_bindings_signal();  // emits changed()

// Why `keys` can't be given to an action of `group` ("" when it can): a key
// the window keeps for itself (Tab, Esc, Copy and Paste, the developers'
// keys), or, outside Video Compare, one that would stop typing in text boxes.
QString refused_key(const QKeySequence& keys, KeyGroup group);
// The other action that has `keys` where `id`'s would work too: the same
// group, or any group against a window-wide one. Nothing if none.
std::optional<std::string> clashing_action(std::string_view id, const QKeySequence& keys,
                                           const KeyBindings& bindings);

// Keys as the system names them ("Ctrl+O, Ctrl+P", in the window's language).
QString keys_text(const std::vector<QKeySequence>& keys);
// A key as Video Compare shows it on screen: the arrows as arrows, Space by
// its name (translated), any other as the system names it.
QString key_cap_text(const QKeySequence& keys);
// Video Compare's line of its main keys ("S: Hold to see the source · ←/→:
// Switch test video · Space: Play / pause"), from the keys in use.
QString video_compare_hint();

}  // namespace vml::app
