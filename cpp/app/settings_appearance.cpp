// Settings > Appearance: the interface (VML Classic or VML Modern), each a
// card with a picture of it, Modern's accent (the system's, a preset, or a
// colour of one's own), and light or dark (theme.h).
#include <QAbstractButton>
#include <QApplication>
#include <QButtonGroup>
#include <QColor>
#include <QColorDialog>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
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
#include "settings_panel.h"
#include "theme.h"

namespace vml::app {

namespace {

// Modern's accent presets: their names and colours.
constexpr std::pair<const char*, const char*> accent_presets[] = {
    {N_("Lilac"), "#c38fe8"}, {N_("Blue"), "#4f8fe8"},  {N_("Teal"), "#2fa9b2"},
    {N_("Green"), "#3fa66b"}, {N_("Amber"), "#d98a2b"}, {N_("Rose"), "#e0607e"},
};

// A small picture of an interface, drawn in the colours in use: Classic's
// framed box and flat buttons, or Modern's card and accent button.
class InterfacePreview : public QWidget {
public:
    InterfacePreview(bool modern, QWidget* parent = nullptr) : QWidget(parent), modern_(modern) {
        setFixedSize(180, 104);
    }
    // The accent Modern's picture shows: none for the system's.
    void set_accent(std::optional<QColor> accent) {
        accent_ = accent;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        const bool dark = theme::is_dark();
        const QRectF frame = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        if (!modern_) {
            // Classic: square corners, every edge a line, every button alike.
            const QColor ground = dark ? QColor("#202020") : QColor("#f3f3f3");
            const QColor line = dark ? QColor("#3a3a3a") : QColor("#c8c8c8");
            const QColor button = dark ? QColor("#2d2d2d") : QColor("#e6e6e6");
            painter.fillRect(rect(), ground);
            painter.setPen(line);
            painter.drawRect(frame);
            painter.fillRect(QRectF(8, 8, 112, 8), line);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(QRectF(8.5, 22.5, 163, 54));
            painter.setBrush(button);
            painter.drawRect(QRectF(8.5, 84.5, 34, 10));
            painter.drawRect(QRectF(46.5, 84.5, 34, 10));
            return;
        }
        // Modern: a card on the window, a primary button in the accent.
        const QColor accent = accent_.value_or(QApplication::palette().color(QPalette::Accent));
        QPainterPath outline;
        outline.addRoundedRect(frame, 6, 6);
        painter.fillPath(outline, theme::modern_colour("window", dark));
        painter.setPen(Qt::NoPen);
        painter.setBrush(theme::modern_colour("border_strong", dark));
        painter.drawRoundedRect(QRectF(8, 8, 72, 8), 2, 2);
        painter.setPen(theme::modern_colour("border", dark));
        painter.setBrush(theme::modern_colour("card", dark));
        painter.drawRoundedRect(QRectF(8.5, 22.5, 163, 54), 5, 5);
        painter.setPen(Qt::NoPen);
        painter.setBrush(accent);
        painter.drawRoundedRect(QRectF(8, 84, 44, 12), 3, 3);
        painter.setPen(theme::modern_colour("border_strong", dark));
        painter.setBrush(theme::modern_colour("raised", dark));
        painter.drawRoundedRect(QRectF(56.5, 84.5, 34, 11), 3, 3);
    }

private:
    bool modern_;
    std::optional<QColor> accent_;
};

// An interface's card: its radio button, its picture, its name and what it
// is. A click anywhere on it picks it.
class InterfaceCard : public QFrame {
public:
    InterfaceCard(QRadioButton* radio, InterfacePreview* preview, const QString& name, const QString& about)
        : radio_(radio) {
        setObjectName("interfaceCard");
        setFrameShape(QFrame::StyledPanel);  // Classic's frame; Modern's sheet draws a card
        theme::set_role(this, "choice");
        setCursor(Qt::PointingHandCursor);
        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(12, 12, 12, 12);
        layout->setSpacing(14);
        layout->addWidget(radio);
        layout->addWidget(preview);
        auto* words = new QVBoxLayout;
        words->setSpacing(4);
        auto* title = new QLabel(name);
        QFont bold = title->font();
        bold.setWeight(QFont::DemiBold);
        title->setFont(bold);
        auto* text = new QLabel(about);
        text->setWordWrap(true);
        theme::style(text, "color: {muted};");
        words->addStretch(1);
        words->addWidget(title);
        words->addWidget(text);
        words->addStretch(1);
        layout->addLayout(words, 1);
        connect(radio, &QRadioButton::toggled, this, [this](bool on) { show_checked(on); });
        show_checked(radio->isChecked());
    }

protected:
    void mouseReleaseEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton && rect().contains(event->position().toPoint())) {
            radio_->click();
        }
        QFrame::mouseReleaseEvent(event);
    }

private:
    // Modern's sheet draws the card picked in the accent.
    void show_checked(bool on) {
        setProperty("checked", on);
        style()->unpolish(this);
        style()->polish(this);
    }
    QRadioButton* radio_;
};

}  // namespace

QWidget* SettingsPanel::build_appearance_page() {
    auto* page = new QWidget;
    auto* page_layout = new QVBoxLayout(page);

    // The interface: a card each.
    auto* interface_box = new QGroupBox(app::tr("Interface"));
    auto* interface_layout = new QVBoxLayout(interface_box);
    auto* interface_hint = new QLabel(app::tr("VML Modern shows at once, VML Classic when the app is next started. "
                                              "Your settings, results and layout stay as they are."));
    interface_hint->setWordWrap(true);
    theme::style(interface_hint, "color: {muted};");
    interface_layout->addWidget(interface_hint);
    auto* cards = new QHBoxLayout;
    cards->setSpacing(12);
    interface_buttons = new QButtonGroup(this);
    const bool modern = theme::interface_of(settings_.interface_style) == theme::Interface::modern;
    auto* classic_radio = new QRadioButton;
    classic_radio->setAccessibleName(app::tr("VML Classic"));
    classic_radio->setChecked(!modern);
    auto* modern_radio = new QRadioButton;
    modern_radio->setAccessibleName(app::tr("VML Modern"));
    modern_radio->setChecked(modern);
    interface_buttons->addButton(classic_radio, 0);
    interface_buttons->addButton(modern_radio, 1);
    auto* modern_picture = new InterfacePreview(true);
    modern_preview = modern_picture;
    // VML Modern first: the default.
    cards->addWidget(new InterfaceCard(modern_radio, modern_picture, app::tr("VML Modern"),
                                       app::tr("Cards, grouped toolbars, one accent button per tab, scores that read "
                                               "first.")),
                     1);
    cards->addWidget(new InterfaceCard(classic_radio, new InterfacePreview(false), app::tr("VML Classic"),
                                       app::tr("The window as it has been: the system's own controls, framed boxes, "
                                               "every button alike.")),
                     1);
    interface_layout->addLayout(cards);
    page_layout->addWidget(interface_box);

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
    grid->addWidget(row_name(app::tr("Accent"), app::tr("VML Modern only")), 0, 0);
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
        swatch->setFixedSize(28, 28);
        swatch->setToolTip(app::tr(name));
        swatch->setAccessibleName(app::tr(name));
        swatch->setProperty("hex", QString::fromLatin1(hex));
        swatch->setCursor(Qt::PointingHandCursor);
        // A disc of its colour, ringed when chosen, in either interface.
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

    grid->addWidget(row_name(app::tr("Theme"), app::tr("Mocha: VML Modern only")), 1, 0);
    auto* modes = new QWidget;
    modes->setObjectName("themeModes");
    theme::set_role(modes, "segments");
    auto* modes_row = new QHBoxLayout(modes);
    modes_row->setContentsMargins(3, 3, 3, 3);
    modes_row->setSpacing(2);
    theme_buttons = new QButtonGroup(this);
    const std::pair<QString, QString> choices[] = {
#ifdef __APPLE__
        {app::tr("Same as macOS"), ""},
#else
        {app::tr("Same as Windows"), ""},
#endif
        {app::tr("Light"), "light"},
        {app::tr("Dark"), "dark"},
        {app::tr("Mocha"), "mocha"},
    };
    for (const auto& [name, value] : choices) {
        auto* mode = new QPushButton(name);
        mode->setCheckable(true);
        mode->setProperty("value", value);
        mode->setChecked(QString::fromStdString(settings_.theme) == value);
        if (value == "mocha") {
            mode->setToolTip(app::tr("Dark chocolate browns with an amber accent. VML Classic shows Dark."));
        }
        theme_buttons->addButton(mode);
        modes_row->addWidget(mode);
    }
    if (theme_buttons->checkedButton() == nullptr) {
        theme_buttons->buttons().front()->setChecked(true);  // an unknown setting: the system's
    }
    connect(theme_buttons, &QButtonGroup::buttonClicked, this, [this] { on_edited(); });
    auto* modes_line = new QHBoxLayout;
    modes_line->addWidget(modes);
    modes_line->addStretch(1);
    grid->addLayout(modes_line, 1, 1);

    auto* footnote = new QLabel(app::tr("Text on an accent button is dark or light, whichever reads on it. Exported "
                                        "graphs keep their own colors (Storage > Exported graphs)."));
    footnote->setWordWrap(true);
    theme::style(footnote, "color: {muted};");
    grid->addWidget(footnote, 2, 1);
    page_layout->addWidget(look_box);
    page_layout->addStretch(1);

    // The accent's controls, Modern's alone (theme::apply_palette): usable
    // while it is picked.
    set_accent(QString::fromStdString(settings_.accent_colour).toLower());
    accent_controls->setEnabled(modern);
    connect(interface_buttons, &QButtonGroup::idClicked, this, [this](int id) {
        accent_controls->setEnabled(id == 1);
        on_edited();
    });
    return page;
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
    accent_hex->setText(colour ? accent_value.toUpper()
                               : QApplication::palette().color(QPalette::Accent).name().toUpper());
    accent_hex->setPlaceholderText(QString());
    static_cast<InterfacePreview*>(modern_preview)->set_accent(colour);
}

void SettingsPanel::choose_custom_accent() {
    // Starts at the accent in use: the chosen one, else the palette's.
    const std::optional<QColor> current = theme::accent_of(accent_value.toStdString());
    QColorDialog dialog(current.value_or(QApplication::palette().color(QPalette::Accent)), this);
    dialog.setWindowTitle(app::tr("Accent color"));
    dialog.setOption(QColorDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    set_accent(dialog.currentColor().name());
    on_edited();
}

}  // namespace vml::app
