#include "shortcuts.h"

#include <QAbstractButton>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include "i18n.h"
#include "key_guide.h"
#include "theme.h"

namespace vml::app {

namespace {

// A key as the system names it ("Ctrl+O", in the window's language).
QString key_text(const QKeySequence& keys) { return keys.toString(QKeySequence::NativeText); }

}  // namespace

const std::vector<Shortcut>& shortcuts() {
    // Windows' usual keys where there is one (Ctrl+O to open, F1 for help,
    // Ctrl+, for settings), and F5 to start as Visual Studio and many
    // encoders do, with Shift+F5 to stop it and F6 beside it to pause.
    static const std::vector<Shortcut> list{
        {Command::add_videos, QKeySequence(Qt::CTRL | Qt::Key_O)},
        {Command::choose_reference, QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_O)},
        {Command::import_run, QKeySequence(Qt::CTRL | Qt::Key_I)},
        {Command::calculate, QKeySequence(Qt::Key_F5)},
        {Command::pause, QKeySequence(Qt::Key_F6)},
        {Command::cancel, QKeySequence(Qt::SHIFT | Qt::Key_F5)},
        {Command::export_all, QKeySequence(Qt::CTRL | Qt::Key_E)},
        {Command::settings, QKeySequence(Qt::CTRL | Qt::Key_Comma)},
        {Command::shortcuts, QKeySequence(Qt::Key_F1)},
    };
    return list;
}

QString command_text(Command command) {
    switch (command) {
    case Command::add_videos:
        return app::tr("Add files");
    case Command::choose_reference:
        return app::tr("Choose the reference");
    case Command::import_run:
        return app::tr("Import run file...");
    case Command::calculate:
        return app::tr("Calculate metrics");
    case Command::pause:
        return app::tr("Pause / resume");
    case Command::cancel:
        return app::tr("Cancel");
    case Command::export_all:
        return app::tr("Export");
    case Command::settings:
        return app::tr("Settings");
    case Command::shortcuts:
        return app::tr("Keyboard shortcuts");
    }
    return {};
}

QKeySequence tab_keys(int index) { return QKeySequence(Qt::CTRL | static_cast<Qt::Key>(Qt::Key_1 + index)); }

QKeySequence remove_keys() { return QKeySequence(QKeySequence::Delete); }

void show_shortcut_in_tip(QAbstractButton* button, const QKeySequence& keys) {
    const QString line = app::tr("Shortcut: {keys}", {{"keys", key_text(keys)}});
    button->setToolTip(button->toolTip().isEmpty() ? line : button->toolTip() + QStringLiteral("\n\n") + line);
}

ShortcutsDialog::ShortcutsDialog(QWidget* parent) : AppDialog(parent, app::tr("Keyboard shortcuts")) {
    setObjectName(QStringLiteral("shortcutsDialog"));
    auto* layout = new QVBoxLayout(body());
    grid_ = new QGridLayout;
    grid_->setHorizontalSpacing(24);
    grid_->setColumnStretch(1, 1);
    layout->addLayout(grid_);

    add_heading(app::tr("Anywhere in the window"));
    for (const Shortcut& shortcut : shortcuts()) {
        add_row(key_text(shortcut.keys), command_text(shortcut.command));
    }
    add_row(key_text(QKeySequence(Qt::CTRL | Qt::Key_Tab)) + QStringLiteral(", ") +
                key_text(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Tab)),
            app::tr("Next or previous tab"));
    add_row(key_text(tab_keys(0)) + QString::fromUtf8(" \xE2\x80\xA6 ") + key_text(tab_keys(7)),  // Ctrl+1 … Ctrl+8
            app::tr("Go to a tab"));

    add_heading(app::tr("Videos"));
    add_row(key_text(remove_keys()), app::tr("Remove selected"));

    add_heading(app::tr("Video Compare"));
    for (const KeyRow& row : video_compare_keys()) {
        add_row(row.keys.size() == 2 ? row.keys[0] + QStringLiteral(" ") + row.keys[1] : row.keys.front(), row.action);
    }

    auto* close = new QPushButton(app::tr("Close"));
    connect(close, &QPushButton::clicked, this, &QDialog::reject);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch(1);
    buttons->addWidget(close);
    layout->addSpacing(8);
    layout->addLayout(buttons);
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
