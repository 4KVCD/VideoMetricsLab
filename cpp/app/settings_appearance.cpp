// Settings > Appearance: the accent (the system's, a preset, or a colour of
// one's own), and light or dark and their looks (theme.h).
#include <QAbstractButton>
#include <QApplication>
#include <QButtonGroup>
#include <QColor>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPalette>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QStyle>
#include <QVBoxLayout>

#include <optional>
#include <utility>

#include "i18n.h"
#include "app_dialog.h"
#include "settings_panel.h"
#include "theme.h"

namespace vml::app {

namespace {

// Modern's accent presets: their names and colours.
constexpr std::pair<const char*, const char*> accent_presets[] = {
    {N_("Lilac"), "#c38fe8"}, {N_("Blue"), "#4f8fe8"},  {N_("Teal"), "#2fa9b2"},
    {N_("Green"), "#3fa66b"}, {N_("Amber"), "#d98a2b"}, {N_("Rose"), "#e0607e"},
};

}  // namespace

QWidget* SettingsPanel::build_appearance_page() {
    auto* page = new QWidget;
    auto* page_layout = new QVBoxLayout(page);


    // In a contrast theme, VML Modern takes its colors alone.
    auto* contrast_note = new QLabel(app::tr("Windows' contrast theme sets the colors while it is on. The accent, "
                                             "theme and looks chosen here apply once it is off."));
    contrast_note->setObjectName("contrastNote");
    contrast_note->setWordWrap(true);
    theme::set_role(contrast_note, "notice");
    theme::contrast_only(contrast_note);
    page_layout->addWidget(contrast_note);

    // The accent, and the theme: a row each, named on the left.
    auto* look_box = new QGroupBox;
    auto* grid = new QGridLayout(look_box);
    grid->setHorizontalSpacing(20);
    grid->setVerticalSpacing(14);
    grid->setColumnMinimumWidth(0, 180);
    grid->setColumnStretch(1, 1);
    const auto row_name = [](const QString& name, const QString& note) {
        auto* names = new QWidget;
        auto* layout = new QVBoxLayout(names);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(2);
        auto* title = new QLabel(name);
        QFont bold = title->font();
        bold.setWeight(QFont::DemiBold);
        title->setFont(bold);
        auto* under = new QLabel(note);
        theme::style(under, "color: {muted};");
        layout->addWidget(title);
        layout->addWidget(under);
        return names;
    };
    grid->addWidget(row_name(app::tr("Accent"), QString()), 0, 0);
    accent_controls = new QWidget;
    auto* accent_row = new QHBoxLayout(accent_controls);
    accent_row->setContentsMargins(0, 0, 0, 0);
    accent_row->setSpacing(10);
    accent_buttons = new QButtonGroup(this);
#ifdef __APPLE__
    auto* system_accent = new QRadioButton(app::tr("Same as macOS"));
#else
    auto* system_accent = new QRadioButton(app::tr("Same as Windows"));
#endif
    system_accent->setObjectName("systemAccent");
    accent_buttons->addButton(system_accent);
    accent_row->addWidget(system_accent);
    for (const auto& [name, hex] : accent_presets) {
        auto* swatch = new QPushButton;
        swatch->setCheckable(true);
        swatch->setFixedSize(theme::scaled(28), theme::scaled(28));
        swatch->setToolTip(app::tr(name));
        swatch->setAccessibleName(app::tr(name));
        swatch->setProperty("hex", QString::fromLatin1(hex));
        swatch->setCursor(Qt::PointingHandCursor);
        // A disc of its colour, ringed when chosen.
        theme::style(swatch, QStringLiteral("QPushButton {{ background: %1; border: 2px solid transparent; "
                                            "border-radius: 14px; padding: 0; min-height: 0; }} "
                                            "QPushButton:checked {{ border-color: palette(window-text); }} "
                                            "QPushButton:disabled {{ background: {faint}; }}")
                                 .arg(QString::fromLatin1(hex)));
        accent_buttons->addButton(swatch);
        accent_row->addWidget(swatch);
    }
    accent_row->addWidget(theme::separator());
    auto* custom = new QPushButton(app::tr("Custom..."));
    custom->setObjectName("customAccent");
    connect(custom, &QPushButton::clicked, this, [this] { choose_custom_accent(); });
    accent_row->addWidget(custom);
    accent_hex = new QLineEdit;
    accent_hex->setObjectName("accentHex");
    accent_hex->setAccessibleName(app::tr("Accent color"));
    accent_hex->setToolTip(app::tr("The accent's color as #RRGGBB: type one of your own."));
    accent_hex->setValidator(
        new QRegularExpressionValidator(QRegularExpression(QStringLiteral("#?[0-9A-Fa-f]{0,6}")), accent_hex));
    accent_hex->setFixedWidth(accent_hex->fontMetrics().horizontalAdvance(QStringLiteral("#MMMMMM")) + 24);
    connect(accent_hex, &QLineEdit::editingFinished, this, [this] {
        QString typed = accent_hex->text().trimmed().toLower();
        if (!typed.startsWith('#')) {
            typed.prepend('#');
        }
        if (theme::accent_of(typed.toStdString())) {
            set_accent(typed);
            on_edited();
        } else {
            set_accent(accent_value);  // not a colour: what it was
        }
    });
    accent_row->addWidget(accent_hex);
    accent_row->addStretch(1);
    connect(accent_buttons, &QButtonGroup::buttonClicked, this, [this](QAbstractButton* button) {
        set_accent(button->property("hex").toString());
        on_edited();
    });
    grid->addWidget(accent_controls, 0, 1);

    // The theme: light, dark or the system's, each drawn in VML Modern in
    // the look picked for it below.
    struct Choice {
        QString name;
        QString value;
        QString tip;
    };
    // A row of segments, one per choice, the one `current` names ticked
    // (else the first).
    const auto segments = [this](const char* object_name, QButtonGroup*& group, const std::vector<Choice>& choices,
                                 const std::string& current, bool swatches) {
        auto* row = new QWidget;
        row->setObjectName(object_name);
        theme::set_role(row, "segments");
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(3, 3, 3, 3);
        layout->setSpacing(2);
        group = new QButtonGroup(this);
        for (const Choice& choice : choices) {
            auto* button = new QPushButton(choice.name);
            button->setCheckable(true);
            button->setProperty("value", choice.value);
            button->setChecked(QString::fromStdString(current) == choice.value);
            button->setToolTip(choice.tip);
            if (swatches) {
                const QPixmap swatch = theme::look_swatch(choice.value, 18);
                button->setIcon(QIcon(swatch));
                button->setIconSize(swatch.deviceIndependentSize().toSize());
            }
            group->addButton(button);
            layout->addWidget(button);
        }
        if (group->checkedButton() == nullptr) {
            group->buttons().front()->setChecked(true);  // an unknown setting: the first
        }
        auto* line = new QHBoxLayout;
        line->addWidget(row);
        line->addStretch(1);
        return line;
    };
    grid->addWidget(row_name(app::tr("Theme"), QString()), 1, 0);
    grid->addLayout(segments("themeModes", theme_buttons,
                             {
#ifdef __APPLE__
                                 {app::tr("Same as macOS"), "", QString()},
#else
                                 {app::tr("Same as Windows"), "", QString()},
#endif
                                 {app::tr("Light"), "light", QString()},
                                 {app::tr("Dark"), "dark", QString()},
                             },
                             settings_.theme, false),
                    1, 1);
    look_controls = new QWidget;
    auto* looks_grid = new QGridLayout(look_controls);
    looks_grid->setContentsMargins(0, 0, 0, 0);
    looks_grid->setHorizontalSpacing(20);
    looks_grid->setVerticalSpacing(14);
    looks_grid->setColumnMinimumWidth(0, 180);
    looks_grid->setColumnStretch(1, 1);
    looks_grid->addWidget(row_name(app::tr("Light look"), QString()), 0, 0);
    // Each look's description, for its button's tooltip.
    const auto look_tip = [](const QString& look) {
        return look == "dim"     ? app::tr("Light, a step darker throughout, for less glare.")
               : look == "paper" ? app::tr("Dim, warmed a little, like unbleached paper.")
               : look == "desert"
                   ? app::tr("Warm cream with dark text and edges and a burnt-orange accent, after Windows' Desert "
                             "contrast theme.")
               : look == "mocha" ? app::tr("Dark chocolate browns with an amber accent.")
               : look == "joker" ? app::tr("Plum purples with a vivid green accent.")
               : look == "nitrate"
                   ? app::tr("Green-black shadows with cream text and a gold accent, like an old film print.")
               : look == "phosphor"
                   ? app::tr("Near-black with umber edges, bright cream text and a gold accent, for the most "
                             "contrast.")
                   : QString();
    };
    const auto looks_of = [&look_tip](bool dark) {
        std::vector<Choice> choices;
        for (const auto& [name, value] : theme::look_choices(dark)) {
            choices.push_back({name, value, look_tip(value)});
        }
        return choices;
    };
    looks_grid->addLayout(segments("lightLooks", light_look_buttons, looks_of(false), settings_.light_look, true), 0,
                          1);
    looks_grid->addWidget(row_name(app::tr("Dark look"), QString()), 1, 0);
    looks_grid->addLayout(segments("darkLooks", dark_look_buttons, looks_of(true), settings_.dark_look, true), 1, 1);
    grid->addWidget(look_controls, 2, 0, 1, 2);
    // The accent's first choice names what no accent gives: the system's,
    // or, in a look with an accent of its own (Mocha's amber, Joker's green,
    // Nitrate's and Phosphor's gold, Desert's burnt orange), the look's.
    // The look is the one the theme picked shows; for
    // the system's, the one the window shows now.
    const auto name_system_accent = [this, system_accent] {
        const QString mode = theme_buttons->checkedButton()->property("value").toString();
        const bool dark = mode == "dark" || (mode.isEmpty() && theme::is_dark());
        const QString look = (dark ? dark_look_buttons : light_look_buttons)->checkedButton()->property("value").toString();
        if (theme::look_accent(look)) {
            system_accent->setText(app::tr("Theme default"));
            system_accent->setToolTip(look == "mocha"   ? app::tr("The theme's own accent: Mocha's amber.")
                                      : look == "joker" ? app::tr("The theme's own accent: Joker's green.")
                                      : look == "nitrate" ? app::tr("The theme's own accent: Nitrate's gold.")
                                      : look == "phosphor" ? app::tr("The theme's own accent: Phosphor's gold.")
                                                        : app::tr("The theme's own accent: Desert's burnt orange."));
        } else {
#ifdef __APPLE__
            system_accent->setText(app::tr("Same as macOS"));
#else
            system_accent->setText(app::tr("Same as Windows"));
#endif
            system_accent->setToolTip(QString());
        }
    };
    name_system_accent();
    for (QButtonGroup* group : {theme_buttons, light_look_buttons, dark_look_buttons}) {
        connect(group, &QButtonGroup::buttonClicked, this, [this, name_system_accent] {
            name_system_accent();
            on_edited();
        });
    }

    auto* footnote = new QLabel(app::tr("Text on an accent button is dark or light, whichever reads on it. Exported "
                                        "graphs keep their own colors (Storage > Exported graphs)."));
    footnote->setWordWrap(true);
    theme::style(footnote, "color: {muted};");
    grid->addWidget(footnote, 3, 1);
    page_layout->addWidget(look_box);
    page_layout->addStretch(1);

    set_accent(QString::fromStdString(settings_.accent_colour).toLower());
    return page;
}

void SettingsPanel::choose_theme(const QString& value) {
    for (QAbstractButton* button : theme_buttons->buttons()) {
        if (button->property("value").toString() == value) {
            // A click, as choose_look's: setChecked alone emits no
            // buttonClicked, which saves, applies and names the accent's
            // first choice (Theme default for Mocha's amber, say).
            button->click();
            return;
        }
    }
}

void SettingsPanel::choose_look(bool dark, const QString& value) {
    for (QAbstractButton* button : (dark ? dark_look_buttons : light_look_buttons)->buttons()) {
        if (button->property("value").toString() == value) {
            button->click();  // as a click in Settings: saved, applied, the accent's first choice named
            return;
        }
    }
}

void SettingsPanel::set_accent(const QString& hex) {
    // The system's (""), a preset's colour, or one of one's own: the
    // matching choice ticked, none for one's own, and its code shown.
    accent_value = theme::accent_of(hex.toStdString()) ? hex.toLower() : QString();
    QAbstractButton* match = nullptr;
    for (QAbstractButton* button : accent_buttons->buttons()) {
        if (button->property("hex").toString() == accent_value) {
            match = button;
        }
    }
    accent_buttons->setExclusive(false);
    for (QAbstractButton* button : accent_buttons->buttons()) {
        button->setChecked(button == match);
    }
    accent_buttons->setExclusive(true);
    const std::optional<QColor> colour = theme::accent_of(accent_value.toStdString());
    // None chosen: the one the window shows, the look's or the system's,
    // even before the palette has followed a change between light and dark.
    accent_hex->setText(colour ? accent_value.toUpper() : theme::accent_shown().name().toUpper());
    accent_hex->setPlaceholderText(QString());
}

void SettingsPanel::choose_custom_accent() {
    // Starts at the accent in use: the chosen one, else the one shown.
    const std::optional<QColor> current = theme::accent_of(accent_value.toStdString());
    const ColorChoice choice = choose_color(this, app::tr("Accent color"), current.value_or(theme::accent_shown()));
    if (choice.outcome != ColorChoice::Outcome::chosen) {
        return;
    }
    set_accent(choice.color.name());
    on_edited();
}

}  // namespace vml::app
