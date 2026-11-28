#include "shortcuts.h"

#include <QAbstractButton>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <optional>
#include <string_view>

#include "i18n.h"
#include "key_bindings.h"
#include "theme.h"

namespace vml::app {

namespace {

// A key as the system names it ("Ctrl+O", in the window's language).
QString key_text(const QKeySequence& keys) { return keys.toString(QKeySequence::NativeText); }

}  // namespace

void show_shortcut_in_tip(QAbstractButton* button, const QKeySequence& keys, bool shown) {
    // The tooltip as it was before any shortcut line, kept on the button.
    const char* const kept = "vmlTipWithoutShortcut";
    if (!button->property(kept).isValid()) {
        button->setProperty(kept, button->toolTip());
    }
    const QString before = button->property(kept).toString();
    const QString line = app::tr("Shortcut: {keys}", {{"keys", key_text(keys)}});
    button->setToolTip(!shown              ? before
                       : before.isEmpty() ? line
                                          : before + QStringLiteral("\n\n") + line);
}

ShortcutsDialog::ShortcutsDialog(QWidget* parent) : AppDialog(parent, app::tr("Keyboard shortcuts")) {
    setObjectName(QStringLiteral("shortcutsDialog"));
    auto* layout = new QVBoxLayout(body());
    grid_ = new QGridLayout;
    grid_->setHorizontalSpacing(24);
    grid_->setColumnStretch(1, 1);
    layout->addLayout(grid_);
    auto* close = new QPushButton(app::tr("Close"));
    connect(close, &QPushButton::clicked, this, &QDialog::reject);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch(1);
    buttons->addWidget(close);
    layout->addSpacing(8);
    layout->addLayout(buttons);
    refresh();
}

void ShortcutsDialog::refresh() {
    while (QLayoutItem* item = grid_->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    rows_ = 0;
    std::optional<KeyGroup> group;
    for (const KeyAction& action : key_actions()) {
        const std::vector<QKeySequence> keys = keys_for(action.id);
        if (keys.empty()) {
            continue;  // unbound: nothing to press
        }
        if (group != action.group) {
            group = action.group;
            add_heading(group_text(action.group));
        }
        add_row(keys_text(keys), action_text(action.id));
        if (action.id == std::string_view("shortcuts")) {
            // Fixed: Qt's own tab keys.
            add_row(key_text(QKeySequence(Qt::CTRL | Qt::Key_Tab)) + QStringLiteral(", ") +
                        key_text(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Tab)),
                    app::tr("Next or previous tab"));
            add_row(key_text(QKeySequence(Qt::CTRL | Qt::Key_PageDown)) + QStringLiteral(", ") +
                        key_text(QKeySequence(Qt::CTRL | Qt::Key_PageUp)),
                    app::tr("Next or previous section of Settings, or metric of Metric Graphs"));
        }
    }
}

void ShortcutsDialog::add_heading(const QString& text) {
    auto* heading = new QLabel(text);
    QFont bold = heading->font();
    bold.setWeight(QFont::DemiBold);  // in both interfaces; Modern's title role is the same weight
    heading->setFont(bold);
    theme::set_role(heading, "title");
    if (rows_ > 0) {
        grid_->setRowMinimumHeight(rows_++, 8);  // a gap above each heading but the first
    }
    grid_->addWidget(heading, rows_++, 0, 1, 2);
}

void ShortcutsDialog::add_row(const QString& keys, const QString& action) {
    auto* key = new QLabel(keys);
    key->setTextInteractionFlags(Qt::TextSelectableByMouse);
    grid_->addWidget(key, rows_, 0);
    grid_->addWidget(new QLabel(action), rows_, 1);
    ++rows_;
}

}  // namespace vml::app
