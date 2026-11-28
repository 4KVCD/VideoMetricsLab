#include "shortcuts_page.h"

#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

#include "i18n.h"
#include "theme.h"
#include "vml/settings.h"

namespace vml::app {

namespace {

enum Column { action_column, keys_column, other_column, reset_column, column_count };

// An action's keys with `bindings` (the user's, else its defaults).
std::vector<QKeySequence> keys_with(const std::string& id, const KeyBindings& bindings) {
    const auto changed = bindings.find(id);
    if (changed != bindings.end()) {
        return changed->second;
    }
    const KeyAction* action = key_action(id);
    return action != nullptr ? action->defaults : std::vector<QKeySequence>{};
}

}  // namespace

ShortcutsPage::ShortcutsPage(Settings& settings, std::vector<QWidget*> top, std::function<void()> changed,
                             QWidget* parent)
    : QWidget(parent),
      settings_(settings),
      changed_(std::move(changed)),
      keys_(parse_key_bindings(settings.key_bindings_json)),
      mouse_(parse_mouse_bindings(settings.mouse_bindings_json)) {
    setObjectName(QStringLiteral("shortcutsPage"));
    auto* layout = new QVBoxLayout(this);
    auto* top_row = new QHBoxLayout;
    for (QWidget* widget : top) {
        top_row->addWidget(widget);
    }
    top_row->addStretch(1);
    auto* reset = new QPushButton(app::tr("Reset all"));
    reset->setObjectName(QStringLiteral("resetAllShortcuts"));
    reset->setToolTip(app::tr("Every key and mouse action back to its default."));
    connect(reset, &QPushButton::clicked, this, [this] { reset_all(); });
    top_row->addWidget(reset);
    layout->addLayout(top_row);

    auto* keys_box = new QGroupBox(app::tr("Keys"));
    auto* keys_layout = new QVBoxLayout(keys_box);
    search_ = new QLineEdit;
    search_->setPlaceholderText(app::tr("Find an action or key"));
    search_->setClearButtonEnabled(true);
    connect(search_, &QLineEdit::textChanged, this, [this](const QString& text) { filter(text); });
    keys_layout->addWidget(search_);
    table_ = new QTableWidget(0, column_count);
    table_->setObjectName(QStringLiteral("shortcutsTable"));
    table_->setHorizontalHeaderLabels({app::tr("Action"), app::tr("Keys"), app::tr("Other keys"), QString()});
    table_->verticalHeader()->hide();
    table_->setSelectionBehavior(QAbstractItemView::SelectItems);
    table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);  // keys are taken by start_editing
    table_->setTabKeyNavigation(false);  // Tab leaves the table; the arrows move between its cells
    table_->horizontalHeader()->setSectionResizeMode(action_column, QHeaderView::Stretch);
    table_->horizontalHeader()->setSectionResizeMode(reset_column, QHeaderView::ResizeToContents);
    table_->setMinimumHeight(360);
    table_->installEventFilter(this);
    connect(table_, &QTableWidget::cellDoubleClicked, this, [this](int row, int column) { start_editing(row, column); });
    keys_layout->addWidget(table_);
    auto* how = new QLabel(app::tr("Double-click a key, or press Enter or F2 on it, then press the new key. Esc "
                                   "cancels; Delete or Backspace removes the key."));
    how->setWordWrap(true);
    theme::set_role(how, "hint");
    keys_layout->addWidget(how);
    note_ = new QLabel;
    note_->setObjectName(QStringLiteral("shortcutsNote"));
    note_->setWordWrap(true);
    theme::set_role(note_, "notice");
    note_->hide();
    keys_layout->addWidget(note_);
    layout->addWidget(keys_box, 1);

    auto* mouse_box = new QGroupBox(app::tr("The mouse on Video Compare's video"));
    auto* mouse_layout = new QGridLayout(mouse_box);
    int row = 0;
    for (const MouseGesture& gesture : mouse_gestures()) {
        auto* choice = new QComboBox;
        for (const std::string& action : mouse_actions()) {
            choice->addItem(action_text(action), QString::fromStdString(action));
        }
        const std::string id = gesture.id;
        connect(choice, &QComboBox::activated, this, [this, id, choice] {
            const std::string action = choice->currentData().toString().toStdString();
            const auto& gestures = mouse_gestures();
            const auto known = std::find_if(gestures.begin(), gestures.end(),
                                            [&id](const MouseGesture& one) { return one.id == id; });
            if (known != gestures.end() && action == known->default_action) {
                mouse_.erase(id);
            } else {
                mouse_[id] = action;
            }
            save();
            fill_table();  // the combo boxes' bold for changed ones too
        });
        mouse_layout->addWidget(new QLabel(gesture_text(id)), row, 0);
        mouse_layout->addWidget(choice, row, 1);
        gestures_[id] = choice;
        ++row;
    }
    mouse_layout->setColumnStretch(2, 1);
    layout->addWidget(mouse_box);
    fill_table();
}

std::string ShortcutsPage::id_of(int row) const {
    const QTableWidgetItem* item = table_->item(row, action_column);
    return item != nullptr ? item->data(Qt::UserRole).toString().toStdString() : std::string();
}

void ShortcutsPage::fill_table() {
    table_->setRowCount(0);
    std::optional<KeyGroup> group;
    for (const KeyAction& action : key_actions()) {
        if (group != action.group) {
            group = action.group;
            const int heading = table_->rowCount();
            table_->insertRow(heading);
            auto* item = new QTableWidgetItem(group_text(action.group));
            QFont bold = item->font();
            bold.setWeight(QFont::DemiBold);
            item->setFont(bold);
            item->setFlags(Qt::ItemIsEnabled);
            table_->setItem(heading, action_column, item);
            table_->setSpan(heading, action_column, 1, column_count);
        }
        const int row = table_->rowCount();
        table_->insertRow(row);
        auto* name = new QTableWidgetItem(action_text(action.id));
        name->setData(Qt::UserRole, QString::fromUtf8(action.id));
        name->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        table_->setItem(row, action_column, name);
        show_row(row);
    }
    for (const MouseGesture& gesture : mouse_gestures()) {
        QComboBox* choice = gestures_.at(gesture.id);
        const auto changed = mouse_.find(gesture.id);
        const std::string action = changed != mouse_.end() ? changed->second : gesture.default_action;
        choice->setCurrentIndex(std::max(0, choice->findData(QString::fromStdString(action))));
        QFont font = choice->font();
        font.setBold(changed != mouse_.end());
        choice->setFont(font);
        choice->setToolTip(app::tr("Default: {keys}", {{"keys", action_text(gesture.default_action)}}));
    }
    filter(search_->text());
}

void ShortcutsPage::show_row(int row) {
    const std::string id = id_of(row);
    const KeyAction* action = key_action(id);
    if (action == nullptr) {
        return;
    }
    const std::vector<QKeySequence> keys = keys_with(id, keys_);
    const bool changed = keys_.contains(id);
    const QString defaults = action->defaults.empty() ? app::tr("Nothing") : keys_text(action->defaults);
    for (const int column : {keys_column, other_column}) {
        const std::size_t slot = column == keys_column ? 0 : 1;
        auto* item = new QTableWidgetItem(slot < keys.size() ? keys[slot].toString(QKeySequence::NativeText) : QString());
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        QFont font = item->font();
        font.setBold(changed);
        item->setFont(font);
        item->setToolTip(app::tr("Default: {keys}", {{"keys", defaults}}));
        table_->setItem(row, column, item);
    }
    if (changed) {
        auto* reset = new QToolButton;
        reset->setText(QString::fromUtf8("\xE2\x86\xBA"));  // ↺
        reset->setToolTip(app::tr("Reset to the default"));
        reset->setAutoRaise(true);
        connect(reset, &QToolButton::clicked, this, [this, id] { reset_row(id); });
        table_->setCellWidget(row, reset_column, reset);
    } else {
        table_->removeCellWidget(row, reset_column);
    }
}

void ShortcutsPage::filter(const QString& text) {
    // An action shows when its name or one of its keys has the text; a
    // heading while one of its actions shows.
    const QString wanted = text.trimmed();
    int heading = -1;
    bool any = false;
    for (int row = 0; row < table_->rowCount(); ++row) {
        if (id_of(row).empty()) {
            if (heading >= 0) {
                table_->setRowHidden(heading, !any);
            }
            heading = row;
            any = false;
            continue;
        }
        bool shown = wanted.isEmpty();
        for (const int column : {action_column, keys_column, other_column}) {
            const QTableWidgetItem* item = table_->item(row, column);
            shown = shown || (item != nullptr && item->text().contains(wanted, Qt::CaseInsensitive));
        }
        table_->setRowHidden(row, !shown);
        any = any || shown;
    }
    if (heading >= 0) {
        table_->setRowHidden(heading, !any);
    }
}

bool ShortcutsPage::bind(const std::string& id, int slot, const QKeySequence& keys) {
    const KeyAction* action = key_action(id);
    if (action == nullptr || slot < 0 || slot > 1) {
        return false;
    }
    std::vector<QKeySequence> now = keys_with(id, keys_);
    if (!keys.isEmpty()) {
        if (const QString why = refused_key(keys, action->group); !why.isEmpty()) {
            say(why);
            return false;
        }
        if (std::find(now.begin(), now.end(), keys) != now.end()) {
            say(QString());
            return true;  // already one of its keys
        }
        if (const std::optional<std::string> other = clashing_action(id, keys, keys_)) {
            say(app::tr("{keys} is also {action}. Change that one first.",
                        {{"keys", keys.toString(QKeySequence::NativeText)}, {"action", action_text(*other)}}));
            return false;
        }
    }
    const auto at = static_cast<std::size_t>(slot);
    if (keys.isEmpty()) {
        if (at < now.size()) {
            now.erase(now.begin() + static_cast<std::ptrdiff_t>(at));
        }
    } else if (at < now.size()) {
        now[at] = keys;
    } else {
        now.push_back(keys);
    }
    if (now == action->defaults) {
        keys_.erase(id);
    } else {
        keys_[id] = now;
    }
    say(QString());
    save();
    for (int row = 0; row < table_->rowCount(); ++row) {
        if (id_of(row) == id) {
            show_row(row);
        }
    }
    return true;
}

void ShortcutsPage::reset_row(const std::string& id) {
    keys_.erase(id);
    save();
    fill_table();
}

void ShortcutsPage::reset_all() {
    keys_.clear();
    mouse_.clear();
    say(QString());
    save();
    fill_table();
}

void ShortcutsPage::save() {
    settings_.key_bindings_json = key_bindings_json(keys_);
    settings_.mouse_bindings_json = mouse_bindings_json(mouse_);
    use_bindings(keys_, mouse_);
    if (changed_) {
        changed_();
    }
}

void ShortcutsPage::say(const QString& text) {
    note_->setText(text);
    note_->setVisible(!text.isEmpty());
}

QString ShortcutsPage::note() const { return note_->text(); }

void ShortcutsPage::start_editing(int row, int column) {
    if ((column != keys_column && column != other_column) || id_of(row).empty()) {
        return;
    }
    finish_editing(false);
    editor_ = new QKeySequenceEdit;
    editor_->setMaximumSequenceLength(1);  // one key combination: no chords
    editor_->installEventFilter(this);
    editor_pressed_ = false;
    editing_row_ = row;
    editing_column_ = column;
    connect(editor_, &QKeySequenceEdit::editingFinished, this, [this] { finish_editing(true); });
    table_->setCellWidget(row, column, editor_);
    editor_->setFocus(Qt::OtherFocusReason);
}

void ShortcutsPage::finish_editing(bool take) {
    if (editor_ == nullptr) {
        return;
    }
    QKeySequenceEdit* editor = std::exchange(editor_, nullptr);
    const int row = std::exchange(editing_row_, -1);
    const int column = std::exchange(editing_column_, -1);
    const QKeySequence keys = editor->keySequence();
    editor->removeEventFilter(this);
    // Not deleted inside its own signal.
    table_->removeCellWidget(row, column);
    editor->hide();
    editor->deleteLater();
    table_->setCurrentCell(row, column);
    table_->setFocus(Qt::OtherFocusReason);
    if (take && !keys.isEmpty()) {
        bind(id_of(row), column == keys_column ? 0 : 1, QKeySequence(keys[0]));
    }
}

bool ShortcutsPage::eventFilter(QObject* watched, QEvent* event) {
    const auto opens_editor = [](const QKeyEvent& key) {
        return key.modifiers() == Qt::NoModifier &&
               (key.key() == Qt::Key_Return || key.key() == Qt::Key_Enter || key.key() == Qt::Key_F2);
    };
    // Enter or F2 opens the key box on its press; the box then gets the
    // key's release, and QKeySequenceEdit ends on a release whose press it
    // never had, empty: a release before the box's first press is kept from
    // it.
    if (event->type() == QEvent::KeyRelease && watched == editor_ && !editor_pressed_) {
        return true;
    }
    if (event->type() != QEvent::KeyPress) {
        return QWidget::eventFilter(watched, event);
    }
    const auto* key = static_cast<QKeyEvent*>(event);
    const bool plain = key->modifiers() == Qt::NoModifier;
    if (watched == editor_) {
        // A lock key alone is no key to bind (QKeySequenceEdit would take
        // NumLock, as some keyboard tools send around each key).
        if (key->key() == Qt::Key_NumLock || key->key() == Qt::Key_CapsLock || key->key() == Qt::Key_ScrollLock) {
            return true;
        }
        editor_pressed_ = true;
        // In the key box: Esc cancels, Delete or Backspace removes the key.
        if (plain && key->key() == Qt::Key_Escape) {
            finish_editing(false);
            return true;
        }
        if (plain && (key->key() == Qt::Key_Delete || key->key() == Qt::Key_Backspace)) {
            const int row = editing_row_;
            const int column = editing_column_;
            finish_editing(false);
            bind(id_of(row), column == keys_column ? 0 : 1, QKeySequence());
            return true;
        }
        return false;
    }
    if (watched == table_ && plain) {
        const int row = table_->currentRow();
        const int column = table_->currentColumn();
        if (opens_editor(*key)) {
            if (!key->isAutoRepeat()) {
                start_editing(row, column);
            }
            return true;
        }
        if ((key->key() == Qt::Key_Delete || key->key() == Qt::Key_Backspace) &&
            (column == keys_column || column == other_column) && !id_of(row).empty()) {
            bind(id_of(row), column == keys_column ? 0 : 1, QKeySequence());
            return true;
        }
    }
    return QWidget::eventFilter(watched, event);
}

}  // namespace vml::app
