#include "key_bindings.h"

#include <QKeyEvent>

#include <algorithm>
#include <utility>

#include <nlohmann/json.hpp>

#include "i18n.h"
#include "vml/app_log.h"
#include "zoom.h"

namespace vml::app {

namespace {

const app_log::Logger logger("vml.keys");

QKeySequence key(QKeyCombination combination) { return QKeySequence(combination); }

KeyBindings& keys_in_use() {
    static KeyBindings bindings;
    return bindings;
}

MouseBindings& mouse_in_use() {
    static MouseBindings bindings;
    return bindings;
}

// The keys an action has with `bindings` (the user's, else its defaults).
std::vector<QKeySequence> keys_with(const KeyAction& action, const KeyBindings& bindings) {
    const auto changed = bindings.find(action.id);
    return changed != bindings.end() ? changed->second : action.defaults;
}

bool window_wide(KeyGroup group) { return group == KeyGroup::window || group == KeyGroup::tabs; }

}  // namespace

const std::vector<KeyAction>& key_actions() {
    // Windows' usual keys where there is one (Ctrl+O to open, F1 for help,
    // Ctrl+, for settings), F5 to start as Visual Studio and many encoders
    // do, with Shift+F5 to stop it and F6 beside it to pause. Video Compare:
    // a player's (Space, the arrows); frame steps on , and . as mpv's.
    static const std::vector<KeyAction> all{
        {"add_videos", KeyGroup::window, {key(Qt::CTRL | Qt::Key_O)}},
        {"choose_reference", KeyGroup::window, {key(Qt::CTRL | Qt::SHIFT | Qt::Key_O)}},
        {"import_run", KeyGroup::window, {key(Qt::CTRL | Qt::Key_I)}},
        {"calculate", KeyGroup::window, {key(Qt::Key_F5)}},
        {"pause", KeyGroup::window, {key(Qt::Key_F6)}},
        {"cancel", KeyGroup::window, {key(Qt::SHIFT | Qt::Key_F5)}},
        {"export_all", KeyGroup::window, {key(Qt::CTRL | Qt::Key_E)}},
        {"settings", KeyGroup::window, {key(Qt::CTRL | Qt::Key_Comma)}},
        {"shortcuts", KeyGroup::window, {key(Qt::Key_F1)}},
        {"tab_1", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_1)}},
        {"tab_2", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_2)}},
        {"tab_3", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_3)}},
        {"tab_4", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_4)}},
        {"tab_5", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_5)}},
        {"tab_6", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_6)}},
        {"tab_7", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_7)}},
        {"tab_8", KeyGroup::tabs, {key(Qt::CTRL | Qt::Key_8)}},
        {"remove_selected", KeyGroup::videos, {QKeySequence(QKeySequence::Delete)}},
        {"play_pause", KeyGroup::video_compare, {key(Qt::Key_Space)}},
        {"show_source", KeyGroup::video_compare, {key(Qt::Key_S)}},
        {"previous_video", KeyGroup::video_compare, {key(Qt::Key_Left)}},
        {"next_video", KeyGroup::video_compare, {key(Qt::Key_Right)}},
        {"previous_frame", KeyGroup::video_compare, {key(Qt::Key_Comma)}},
        {"next_frame", KeyGroup::video_compare, {key(Qt::Key_Period)}},
        {"zoom_fit", KeyGroup::video_compare, {key(Qt::CTRL | Qt::Key_0)}},
        {"zoom_100", KeyGroup::video_compare, {}},
        {"zoom_toggle", KeyGroup::video_compare, {}},
        {"zoom_in", KeyGroup::video_compare, {}},
        {"zoom_out", KeyGroup::video_compare, {}},
    };
    return all;
}

const KeyAction* key_action(std::string_view id) {
    const auto& all = key_actions();
    const auto found = std::find_if(all.begin(), all.end(), [id](const KeyAction& action) { return action.id == id; });
    return found != all.end() ? &*found : nullptr;
}

QString action_text(std::string_view id) {
    static const std::map<std::string_view, const char*> texts{
        {"add_videos", N_("Add files")},
        {"choose_reference", N_("Choose the reference")},
        {"import_run", N_("Import run file...")},
        {"calculate", N_("Calculate metrics")},
        {"pause", N_("Pause / resume")},
        {"cancel", N_("Cancel")},
        {"export_all", N_("Export")},
        {"settings", N_("Settings")},
        {"shortcuts", N_("Keyboard shortcuts")},
        {"tab_1", N_("Videos")},
        {"tab_2", N_("Metric Graphs")},
        {"tab_3", N_("Worst Frames")},
        {"tab_4", N_("Video Compare")},
        {"tab_5", N_("Bitrate Viewer")},
        {"tab_6", N_("BD-rate")},
        {"tab_7", N_("Saved Results")},
        {"tab_8", N_("Settings")},
        {"remove_selected", N_("Remove selected")},
        {"play_pause", N_("Play / pause")},
        {"show_source", N_("Hold to see the source")},
        {"previous_video", N_("Previous test video")},
        {"next_video", N_("Next test video")},
        {"previous_frame", N_("Previous frame")},
        {"next_frame", N_("Next frame")},
        {"zoom_fit", N_("Fit to window")},
        {"zoom_toggle", N_("Fit to window or 100%")},
        {"zoom_in", N_("Zoom in")},
        {"zoom_out", N_("Zoom out")},
    };
    if (id == "zoom_100") {
        return percent_text(1.0);
    }
    if (id.empty()) {
        return app::tr("Nothing");
    }
    const auto found = texts.find(id);
    return found != texts.end() ? app::tr(found->second) : QString::fromUtf8(id.data(), static_cast<qsizetype>(id.size()));
}

QString group_text(KeyGroup group) {
    switch (group) {
    case KeyGroup::window:
        return app::tr("Anywhere in the window");
    case KeyGroup::tabs:
        return app::tr("Go to a tab");
    case KeyGroup::videos:
        return app::tr("Videos");
    case KeyGroup::video_compare:
        return app::tr("Video Compare");
    }
    return {};
}

std::vector<QKeySequence> keys_for(std::string_view id) {
    const KeyAction* action = key_action(id);
    return action != nullptr ? keys_with(*action, keys_in_use()) : std::vector<QKeySequence>{};
}

bool matches(const QKeyEvent& event, std::string_view id) {
    const QKeySequence pressed(QKeyCombination(event.modifiers() & ~Qt::KeypadModifier, Qt::Key(event.key())));
    const std::vector<QKeySequence> keys = keys_for(id);
    return std::find(keys.begin(), keys.end(), pressed) != keys.end();
}

KeyBindings parse_key_bindings(const std::string& json) {
    KeyBindings bindings;
    const nlohmann::json data = nlohmann::json::parse(json, nullptr, false);
    if (!data.is_object()) {
        return bindings;
    }
    for (auto item = data.begin(); item != data.end(); ++item) {
        if (key_action(item.key()) == nullptr || !item.value().is_array()) {
            logger.warning("Key bindings: left out " + item.key());
            continue;
        }
        std::vector<QKeySequence> keys;
        for (const nlohmann::json& text : item.value()) {
            const QKeySequence keys_read =
                text.is_string() ? QKeySequence::fromString(QString::fromStdString(text.get<std::string>()),
                                                            QKeySequence::PortableText)
                                 : QKeySequence();
            // One combination of a key Qt knows (it reads nonsense as
            // Key_unknown, or as several).
            if (keys_read.count() != 1 || keys_read[0].key() == Qt::Key_unknown || keys.size() == 2) {
                logger.warning("Key bindings: a key of " + item.key() + " left out");
                continue;
            }
            keys.push_back(keys_read);
        }
        bindings[item.key()] = std::move(keys);
    }
    return bindings;
}

std::string key_bindings_json(const KeyBindings& bindings) {
    nlohmann::json data = nlohmann::json::object();
    for (const auto& [id, keys] : bindings) {
        nlohmann::json list = nlohmann::json::array();
        for (const QKeySequence& one : keys) {
            list.push_back(one.toString(QKeySequence::PortableText).toStdString());
        }
        data[id] = std::move(list);
    }
    return data.dump();
}

const std::vector<MouseGesture>& mouse_gestures() {
    // A player's: a click plays, the wheel steps; Ctrl and the wheel zoom.
    static const std::vector<MouseGesture> all{
        {"click", "play_pause"},           {"double_click", "zoom_toggle"},  {"middle", "show_source"},
        {"wheel_up", "previous_frame"},    {"wheel_down", "next_frame"},     {"ctrl_wheel_up", "zoom_in"},
        {"ctrl_wheel_down", "zoom_out"},   {"back", "previous_video"},       {"forward", "next_video"},
    };
    return all;
}

QString gesture_text(std::string_view id) {
    static const std::map<std::string_view, const char*> texts{
        {"click", N_("Click")},
        {"double_click", N_("Double-click")},
        {"middle", N_("Middle button (held)")},
        {"wheel_up", N_("Wheel up")},
        {"wheel_down", N_("Wheel down")},
        {"ctrl_wheel_up", N_("Ctrl+wheel up")},
        {"ctrl_wheel_down", N_("Ctrl+wheel down")},
        {"back", N_("Back button")},
        {"forward", N_("Forward button")},
    };
    const auto found = texts.find(id);
    return found != texts.end() ? app::tr(found->second) : QString();
}

const std::vector<std::string>& mouse_actions() {
    static const std::vector<std::string> all{"",          "play_pause",  "show_source",    "previous_video",
                                              "next_video", "previous_frame", "next_frame", "zoom_fit",
                                              "zoom_100",  "zoom_toggle", "zoom_in",        "zoom_out"};
    return all;
}

MouseBindings parse_mouse_bindings(const std::string& json) {
    MouseBindings bindings;
    const nlohmann::json data = nlohmann::json::parse(json, nullptr, false);
    if (!data.is_object()) {
        return bindings;
    }
    const auto& gestures = mouse_gestures();
    const auto& actions = mouse_actions();
    for (auto item = data.begin(); item != data.end(); ++item) {
        const bool known_gesture = std::any_of(gestures.begin(), gestures.end(),
                                               [&](const MouseGesture& gesture) { return gesture.id == item.key(); });
        if (!known_gesture || !item.value().is_string() ||
            std::find(actions.begin(), actions.end(), item.value().get<std::string>()) == actions.end()) {
            logger.warning("Mouse bindings: left out " + item.key());
            continue;
        }
        bindings[item.key()] = item.value().get<std::string>();
    }
    return bindings;
}

std::string mouse_bindings_json(const MouseBindings& bindings) {
    nlohmann::json data = nlohmann::json::object();
    for (const auto& [gesture, action] : bindings) {
        data[gesture] = action;
    }
    return data.dump();
}

std::string action_for_gesture(std::string_view gesture) {
    const auto changed = mouse_in_use().find(std::string(gesture));
    if (changed != mouse_in_use().end()) {
        return changed->second;
    }
    for (const MouseGesture& known : mouse_gestures()) {
        if (known.id == gesture) {
            return known.default_action;
        }
    }
    return {};
}

void use_bindings(KeyBindings keys, MouseBindings mouse) {
    keys_in_use() = std::move(keys);
    mouse_in_use() = std::move(mouse);
    emit key_bindings_signal()->changed();
}

const KeyBindings& key_bindings() { return keys_in_use(); }
const MouseBindings& mouse_bindings() { return mouse_in_use(); }

KeyBindingsSignal* key_bindings_signal() {
    static KeyBindingsSignal signal;
    return &signal;
}

QString refused_key(const QKeySequence& keys, KeyGroup group) {
    if (keys.isEmpty()) {
        return {};
    }
    const QKeyCombination combination = keys[0];
    const Qt::KeyboardModifiers modifiers = combination.keyboardModifiers();
    const int code = combination.key();
    const QString named = keys.toString(QKeySequence::NativeText);
    // The window's own: moving the focus and between tabs, closing,
    // pressing, the text boxes' editing keys, and the developers' keys
    // (Ctrl+Alt+Shift+F1 to F7, a branch of their own).
    const bool own_key = code == Qt::Key_Tab || code == Qt::Key_Backtab || code == Qt::Key_Escape ||
                         code == Qt::Key_Return || code == Qt::Key_Enter ||
                         // Settings' sections, Metric Graphs' metrics (pass_ctrl_tab_up)
                         (modifiers == Qt::ControlModifier && (code == Qt::Key_PageDown || code == Qt::Key_PageUp)) ||
                         (modifiers == (Qt::ControlModifier | Qt::AltModifier | Qt::ShiftModifier) &&
                          code >= Qt::Key_F1 && code <= Qt::Key_F7);
    bool editing_key = false;
    for (const QKeySequence::StandardKey standard :
         {QKeySequence::Copy, QKeySequence::Cut, QKeySequence::Paste, QKeySequence::Undo, QKeySequence::Redo,
          QKeySequence::SelectAll}) {
        const QList<QKeySequence> bound = QKeySequence::keyBindings(standard);
        editing_key = editing_key || bound.contains(keys);
    }
    if (own_key || editing_key) {
        return app::tr("{keys} is kept for the window itself.", {{"keys", named}});
    }
    // A plain key everywhere in the window would be taken from the text
    // boxes; Video Compare's keys give way to a text box with the focus.
    const bool plain = (modifiers & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)) == 0;
    const bool function_key = (code >= Qt::Key_F1 && code <= Qt::Key_F35) || code == Qt::Key_Delete ||
                              code == Qt::Key_Insert || code == Qt::Key_Pause;
    if (group != KeyGroup::video_compare && plain && !function_key) {
        return app::tr("{keys} would stop typing in text boxes: add Ctrl or Alt.", {{"keys", named}});
    }
    return {};
}

std::optional<std::string> clashing_action(std::string_view id, const QKeySequence& keys,
                                           const KeyBindings& bindings) {
    const KeyAction* action = key_action(id);
    if (action == nullptr || keys.isEmpty()) {
        return std::nullopt;
    }
    for (const KeyAction& other : key_actions()) {
        if (other.id == id) {
            continue;
        }
        const bool same_place = other.group == action->group || window_wide(other.group) || window_wide(action->group);
        if (!same_place) {
            continue;
        }
        const std::vector<QKeySequence> theirs = keys_with(other, bindings);
        if (std::find(theirs.begin(), theirs.end(), keys) != theirs.end()) {
            return std::string(other.id);
        }
    }
    return std::nullopt;
}

QString key_cap_text(const QKeySequence& keys) {
    switch (keys[0].toCombined()) {
    case Qt::Key_Left: return QString::fromUtf8("\xE2\x86\x90");   // ←
    case Qt::Key_Right: return QString::fromUtf8("\xE2\x86\x92");  // →
    case Qt::Key_Space: return app::tr("Space");
    default: return keys.toString(QKeySequence::NativeText);
    }
}

QString video_compare_hint() {
    QStringList parts;
    const auto part = [&parts](const QString& keys, const QString& action) {
        parts.append(app::tr("{keys}: {action}", {{"keys", keys}, {"action", action}}));
    };
    if (const std::vector<QKeySequence> source = keys_for("show_source"); !source.empty()) {
        part(key_cap_text(source.front()), action_text("show_source"));
    }
    const std::vector<QKeySequence> previous = keys_for("previous_video");
    const std::vector<QKeySequence> next = keys_for("next_video");
    if (!previous.empty() && !next.empty()) {
        part(key_cap_text(previous.front()) + QStringLiteral("/") + key_cap_text(next.front()),
             app::tr("Switch test video"));
    }
    if (const std::vector<QKeySequence> play = keys_for("play_pause"); !play.empty()) {
        part(key_cap_text(play.front()), action_text("play_pause"));
    }
    return parts.join(QString::fromUtf8("   \xC2\xB7   "));  // ·
}

QString keys_text(const std::vector<QKeySequence>& keys) {
    QStringList names;
    for (const QKeySequence& one : keys) {
        names.append(one.toString(QKeySequence::NativeText));
    }
    return names.join(QStringLiteral(", "));
}

}  // namespace vml::app
