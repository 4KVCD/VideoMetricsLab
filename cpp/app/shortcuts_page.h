// Settings > Shortcuts: every action's keys, changeable, and what the mouse
// does on Video Compare's video (key_bindings.h). In the manner of
// MPC-HC's Keys and Mouse pages: one row per action, a key changed in place,
// a search, a reset per row and for all; a key another action has where
// both work, or one the window keeps for itself, is refused and said why.
#pragma once

#include <QWidget>

#include <functional>
#include <map>
#include <string>

#include "key_bindings.h"

class QCheckBox;
class QComboBox;
class QKeySequenceEdit;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;

namespace vml {
struct Settings;
}

namespace vml::app {

class ShortcutsPage : public QWidget {
    Q_OBJECT

public:
    // `top`: the controls shown above the table (the shortcuts' on/off box,
    // the list's button). `changed` saves the settings.
    ShortcutsPage(Settings& settings, std::vector<QWidget*> top, std::function<void()> changed,
                  QWidget* parent = nullptr);

    // Gives `id`'s key `slot` (0 or 1) the keys `keys` (none: removes it).
    // Refused, with the reason shown, for a key the window keeps or another
    // action has where both work. Whether it was taken (tests too).
    bool bind(const std::string& id, int slot, const QKeySequence& keys);
    // Every key and gesture back to its default.
    void reset_all();
    // The note under the table (tests).
    QString note() const;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void fill_table();
    void show_row(int row);
    void filter(const QString& text);
    void start_editing(int row, int column);
    void finish_editing(bool take);
    void reset_row(const std::string& id);
    void save();
    void say(const QString& text);
    std::string id_of(int row) const;

    Settings& settings_;
    std::function<void()> changed_;
    KeyBindings keys_;
    MouseBindings mouse_;
    QLineEdit* search_ = nullptr;
    QTableWidget* table_ = nullptr;
    QLabel* note_ = nullptr;
    QKeySequenceEdit* editor_ = nullptr;
    int editing_row_ = -1;
    int editing_column_ = -1;
    bool editor_pressed_ = false;  // a key pressed in the box since it opened
    std::map<std::string, QComboBox*> gestures_;
};

}  // namespace vml::app
