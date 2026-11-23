#include "theme.h"

#include <QApplication>
#include <QBrush>
#include <QGroupBox>
#include <QFrame>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHash>
#include <QPalette>
#include <QMargins>
#include <QPainter>
#include <QPixmap>
#include <QPainterPath>
#include <QPointer>
#include <QProxyStyle>
#include <QStyle>
#include <QStyleFactory>
#include <QStyleOption>
#include <QStyleHints>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "i18n.h"

namespace vml::app::theme {

namespace {

// name -> (light, dark)
const std::map<QString, std::pair<QString, QString>> colours = {
    {"muted", {"#666666", "#a6a6a6"}},      // hints and secondary text
    {"faint", {"#888888", "#8c8c8c"}},      // n/a cells
    {"dimmed", {"#999999", "#7a7a7a"}},     // a row's cells while it is not the one shown
    {"run_line", {"#444444", "#c8c8c8"}},   // the per-video progress lines
    {"failed", {"#a03030", "#ff8a80"}},
    {"stale", {"#8a6d00", "#e6c35c"}},
    {"good", {"#207020", "#81c784"}},
    {"error_border", {"#cc3333", "#ef6a6a"}},
    {"banner_background", {"#fff3cd", "#4d4012"}},
    {"banner_border", {"#ffe08a", "#8f7623"}},
    {"banner_text", {"#3d3000", "#fff0c2"}},
    {"link", {"#2a5db0", "#8ab4f8"}},       // a graph table heading that sorts
    {"link_selected", {"#12327a", "#c2d7ff"}},
    // Behind the graph statistics' means, and the selected metric's.
    {"mean_tint", {"#f4f6fa", "#262a31"}},
    {"mean_selected", {"#cfe0fa", "#27406a"}},
    // Behind a metric's best and worst score in the video table
    // (Highlight best/worst results).
    {"best", {"#c8e6c9", "#1e4d2b"}},
    {"worst", {"#ffcdd2", "#5c2328"}},
};

// name -> (light, dark): the VML Modern interface's colours, its palette's
// and its style sheet's (modern_style_sheet). The accent is the palette's.
const std::map<QString, std::pair<QString, QString>> modern_colours = {
    {"window", {"#f3f3f5", "#141416"}},         // the window behind the cards
    {"card", {"#ffffff", "#252528"}},           // a group box, a table, a menu
    {"card_alt", {"#f6f6f8", "#29292d"}},       // a table's every other row
    {"inset", {"#f5f5f7", "#2a2a2e"}},          // a box inside a card, without an edge
    {"raised", {"#ececf0", "#2e2e33"}},         // a button, a table's header, a toolbar
    {"hover", {"#e2e2e8", "#37373d"}},
    {"pressed", {"#d6d6dd", "#2a2a2f"}},
    {"field", {"#ffffff", "#1f1f22"}},          // a text box, a list to pick from
    {"border", {"#dedee3", "#34343a"}},
    {"border_strong", {"#c5c5cd", "#45454c"}},  // a control's own edge
    {"text", {"#1d1d21", "#e8e8ea"}},
    {"text_soft", {"#3c3c44", "#c4c4cc"}},      // headings of a table
    {"text_muted", {"#5f5f69", "#a9a9b0"}},     // a tab not shown
    {"disabled", {"#9a9aa3", "#75757d"}},
    {"scroll", {"#c0c0c8", "#4a4a52"}},
    {"danger", {"#b3261e", "#ff9a8f"}},
    {"danger_border", {"#e3a9a3", "#6b3530"}},
};

// A look: VML Modern's colours for Light or Dark, in part replaced. Light's
// looks are Light itself, Dim, Paper and Desert; Dark's, Dark and Mocha
// (Settings' light_look and dark_look). `modern` replaces modern_colours',
// `window` some of `colours`', `chart` the chart's; `accent` is the look's
// own, used while no accent is chosen.
struct Look {
    std::map<QString, QString> modern;
    std::map<QString, QString> window;
    std::optional<ChartColours> chart;
    std::optional<QColor> accent;
};

// Mocha, in place of Modern's dark colours: a dark chocolate
// brown behind lighter brown cards, cream text, after the Solar Dusk
// palette (shadcn), whose amber is Mocha's accent (mocha_accent). The window
// is a deep brown, not near-black, so that it reads as brown.
const std::map<QString, QString> mocha_colours = {
    {"window", "#1b130e"},
    {"card", "#251c16"},
    {"card_alt", "#291f18"},
    {"inset", "#2b2019"},
    {"raised", "#32261e"},
    {"hover", "#3f3128"},
    {"pressed", "#211812"},
    {"field", "#16100c"},
    {"border", "#382c24"},
    {"border_strong", "#4f4034"},
    {"text", "#fbf4ec"},
    {"text_soft", "#e6d8cb"},
    {"text_muted", "#b8a596"},
    {"disabled", "#877668"},
    {"scroll", "#67564a"},
    {"danger", "#f87171"},
    {"danger_border", "#7f2d2a"},
};

// name -> colour: Mocha's in place of a few of `colours`' dark ones, whose
// blue would clash with its amber: Metric Graphs' statistics and their
// sorting headings.
const std::map<QString, QString> mocha_window_colours = {
    {"mean_tint", "#30241c"},
    {"mean_selected", "#6a4419"},
    {"link", "#fbbf24"},
    {"link_selected", "#fde68a"},
};

// Dim: every one of Light's surfaces the same step darker (0.08 of OKLab's
// lightness), for less glare; its text as Light's.
const std::map<QString, QString> dim_colours = {
    {"window", "#d9d9db"}, {"card", "#e4e4e4"},  {"card_alt", "#dcdcde"}, {"inset", "#dbdbdd"},
    {"raised", "#d2d2d6"}, {"hover", "#c8c8ce"}, {"pressed", "#bcbcc3"},  {"field", "#e4e4e4"},
    {"border", "#c4c4c9"}, {"border_strong", "#acacb3"}, {"scroll", "#a7a7af"},
};
const std::map<QString, QString> dim_window_colours = {
    {"mean_tint", "#dadcdf"}, {"mean_selected", "#b5c6df"},     {"best", "#aeccb0"},
    {"worst", "#e4b3b8"},     {"banner_background", "#e4d9b3"}, {"banner_border", "#e4c670"},
};

// Paper: Dim's step, warmed a little, as unbleached paper is.
const std::map<QString, QString> paper_colours = {
    {"window", "#ddd8d2"}, {"card", "#e9e4dc"},  {"card_alt", "#e0dbd5"}, {"inset", "#dfdad4"},
    {"raised", "#d6d1cd"}, {"hover", "#ccc8c5"}, {"pressed", "#c0bcbb"},  {"field", "#e9e4dc"},
    {"border", "#c8c4c1"}, {"border_strong", "#b0abab"}, {"scroll", "#aba6a7"},
};
const std::map<QString, QString> paper_window_colours = {
    {"mean_tint", "#dedbd7"}, {"mean_selected", "#bac6d7"},     {"best", "#b3cba7"},
    {"worst", "#e8b3b0"},     {"banner_background", "#e9d8aa"}, {"banner_border", "#e8c564"},
};

// Desert: Windows 11's Desert contrast theme (hcwhite.theme), its
// background, text, gray text and link, with dark edges as contrast themes
// draw them, and its selection's burnt orange as the accent.
const std::map<QString, QString> desert_colours = {
    {"window", "#fffaef"},     {"card", "#fffaef"},          {"card_alt", "#f7efde"}, {"inset", "#faf3e4"},
    {"raised", "#fffaef"},     {"hover", "#f2e6cc"},         {"pressed", "#e8d8b6"},  {"field", "#ffffff"},
    {"border", "#808080"},     {"border_strong", "#3d3d3d"}, {"text", "#202020"},     {"text_soft", "#3d3d3d"},
    {"text_muted", "#676767"}, {"disabled", "#676767"},      {"scroll", "#808080"},   {"danger", "#a4262c"},
    {"danger_border", "#a4262c"},
};
const std::map<QString, QString> desert_window_colours = {
    {"muted", "#676767"},         {"faint", "#676767"},         {"dimmed", "#808080"},
    {"run_line", "#3d3d3d"},      {"link", "#1c5e75"},          {"link_selected", "#0e3442"},
    {"mean_tint", "#f7efde"},     {"mean_selected", "#f1d6c2"}, {"banner_background", "#fff0c8"},
    {"banner_border", "#903909"}, {"banner_text", "#202020"},
};

// Joker: plum surfaces, lavender-white text, a vivid green accent and
// purple links, after the Dracula palette (MIT), warmed toward plum.
const std::map<QString, QString> joker_colours = {
    {"window", "#1c1724"},        {"card", "#251e30"},       {"card_alt", "#2a2236"}, {"inset", "#2b2338"},
    {"raised", "#322940"},        {"hover", "#403552"},      {"pressed", "#201a29"},  {"field", "#18131f"},
    {"border", "#362c45"},        {"border_strong", "#4d405f"}, {"text", "#f6f2fa"},  {"text_soft", "#ddd4e8"},
    {"text_muted", "#ab9fbf"},    {"disabled", "#6f6385"},   {"scroll", "#57496b"},   {"danger", "#ff6b81"},
    {"danger_border", "#73304a"},
};
const std::map<QString, QString> joker_window_colours = {
    {"mean_tint", "#2d2539"}, {"mean_selected", "#3f3157"}, {"link", "#bd93f9"},
    {"link_selected", "#e0ccff"}, {"muted", "#ab9fbf"},     {"faint", "#ab9fbf"},
};

// Nitrate and Phosphor: sepia film noir, from one palette (cream #f8dbaf,
// gold #c2a269, khaki #b39966, umber #584625, green-black #0c130b). Nitrate's
// surfaces are the green-tinged black, cold shadow under warm cream text, as
// an old print; Phosphor's a neutral near-black with umber edges and brighter
// cream, for the most contrast. Both take the gold as their accent.
const std::map<QString, QString> nitrate_colours = {
    {"window", "#0a0f09"},        {"card", "#121810"},       {"card_alt", "#161c13"}, {"inset", "#171d14"},
    {"raised", "#1e2419"},        {"hover", "#2a2f22"},      {"pressed", "#0e130c"},  {"field", "#080c07"},
    {"border", "#232a1e"},        {"border_strong", "#3a3f2c"}, {"text", "#f8dbaf"},  {"text_soft", "#e6c896"},
    {"text_muted", "#b39966"},    {"disabled", "#6e6448"},   {"scroll", "#474530"},   {"danger", "#e07a5f"},
    {"danger_border", "#6a3324"},
};
const std::map<QString, QString> nitrate_window_colours = {
    {"mean_tint", "#1b2117"}, {"mean_selected", "#3b3420"}, {"link", "#c2a269"},
    {"link_selected", "#f8dbaf"}, {"muted", "#b39966"},     {"faint", "#b39966"},
};
const std::map<QString, QString> phosphor_colours = {
    {"window", "#080808"},        {"card", "#111110"},       {"card_alt", "#151513"}, {"inset", "#161614"},
    {"raised", "#1c1b18"},        {"hover", "#2a2720"},      {"pressed", "#0c0c0b"},  {"field", "#050505"},
    {"border", "#584625"},        {"border_strong", "#b39966"}, {"text", "#fbe6c6"},  {"text_soft", "#f8dbaf"},
    {"text_muted", "#c2a269"},    {"disabled", "#7a6b50"},   {"scroll", "#584625"},   {"danger", "#ff8a70"},
    {"danger_border", "#8a3b2a"},
};
const std::map<QString, QString> phosphor_window_colours = {
    {"mean_tint", "#1a1814"}, {"mean_selected", "#3f331e"}, {"link", "#f8dbaf"},
    {"link_selected", "#ffffff"}, {"muted", "#c2a269"},     {"faint", "#c2a269"},
};

// Light's chart on `card`: its lines and text as Light's.
ChartColours light_chart_on(const QColor& card) {
    return {card, QColor(40, 40, 40), QColor(90, 90, 90), QColor(0, 0, 0, 40), QColor(120, 120, 120),
            QColor(120, 60, 200), QColor(40, 110, 230, 40)};
}

// The looks by name; Light and Dark are none of them (Modern's own colours).
// Light's: Dim, Paper, Desert; Dark's: Mocha, Joker, Nitrate, Phosphor
// (apply_theme, is_dark_look).
const std::map<QString, Look>& looks() {
    static const std::map<QString, Look> all = {
        {"dim", {dim_colours, dim_window_colours, light_chart_on(QColor("#e4e4e4")), std::nullopt}},
        {"paper", {paper_colours, paper_window_colours, light_chart_on(QColor("#e9e4dc")), std::nullopt}},
        {"desert",
         {desert_colours, desert_window_colours,
          ChartColours{QColor("#fffaef"), QColor("#3d3d3d"), QColor("#676767"), QColor(0, 0, 0, 40),
                       QColor("#676767"), QColor(120, 60, 200), QColor(144, 57, 9, 40)},
          QColor("#903909")}},
        {"mocha",
         {mocha_colours, mocha_window_colours,
          // its card and cream text
          ChartColours{QColor("#251c16"), QColor("#ece0d4"), QColor("#b8a596"), QColor(255, 247, 237, 34),
                       QColor("#b8a596"), QColor(253, 186, 116), QColor(245, 158, 11, 48)},
          QColor("#d97706")}},
        {"joker",
         {joker_colours, joker_window_colours,
          ChartColours{QColor("#251e30"), QColor("#f6f2fa"), QColor("#ab9fbf"), QColor(246, 242, 250, 28),
                       QColor("#ab9fbf"), QColor("#ff79c6"), QColor(80, 250, 123, 40)},
          QColor("#50fa7b")}},
        {"nitrate",
         {nitrate_colours, nitrate_window_colours,
          ChartColours{QColor("#121810"), QColor("#f8dbaf"), QColor("#b39966"), QColor(248, 219, 175, 26),
                       QColor("#b39966"), QColor("#f8dbaf"), QColor(194, 162, 105, 44)},
          QColor("#c2a269")}},
        {"phosphor",
         {phosphor_colours, phosphor_window_colours,
          ChartColours{QColor("#111110"), QColor("#fbe6c6"), QColor("#c2a269"), QColor(248, 219, 175, 34),
                       QColor("#c2a269"), QColor("#f8dbaf"), QColor(194, 162, 105, 50)},
          QColor("#c2a269")}},
    };
    return all;
}

// Whether `name` is one of Dark's looks (not Dark itself).
bool is_dark_look(const QString& name) {
    return name == "mocha" || name == "joker" || name == "nitrate" || name == "phosphor";
}

// The looks Settings asks for (apply_theme).
struct LooksAsked {
    QString light = QStringLiteral("light");
    QString dark = QStringLiteral("dark");
};
LooksAsked& looks_asked() {
    static LooksAsked asked;
    return asked;
}

// VML Modern's style: Fusion, with its tick boxes and radio buttons drawn in
// Modern's colours. Fusion draws their edge darker than the window, which
// all but vanishes on Modern's dark cards; a style sheet that drew them
// would draw them without their tick. The rest is Fusion's, under the
// application's style sheet.
class ModernStyle : public QProxyStyle {
public:
    ModernStyle() : QProxyStyle(QStyleFactory::create(QStringLiteral("fusion"))) {}

    // A group box without a title is marked (vmlUntitled), so the sheet
    // leaves it no room for one: a style sheet's [title=""] matches none.
    void polish(QWidget* widget) override {
        QProxyStyle::polish(widget);
        if (auto* box = qobject_cast<QGroupBox*>(widget)) {
            box->setProperty("vmlUntitled", box->title().isEmpty());
        }
    }
    using QProxyStyle::polish;

    void drawPrimitive(PrimitiveElement element, const QStyleOption* option, QPainter* painter,
                       const QWidget* widget) const override {
        const bool check = element == PE_IndicatorCheckBox || element == PE_IndicatorItemViewItemCheck;
        if (!check && element != PE_IndicatorRadioButton) {
            QProxyStyle::drawPrimitive(element, option, painter, widget);
            return;
        }
        const bool dark = is_dark();
        const bool enabled = (option->state & State_Enabled) != 0;
        const bool on = (option->state & State_On) != 0;
        const bool partial = (option->state & State_NoChange) != 0;
        const QColor accent = option->palette.color(QPalette::Accent);
        const QColor edge = modern_colour(enabled ? "border_strong" : "border", dark);
        const QColor field = modern_colour("field", dark);
        QRectF box = QRectF(option->rect).adjusted(0.5, 0.5, -0.5, -0.5);
        const qreal side = std::min(box.width(), box.height());
        box = QRectF(box.center().x() - side / 2, box.center().y() - side / 2, side, side);
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        const bool filled = (on || partial) && enabled;
        painter->setPen(QPen(filled ? accent : edge, 1.0));
        painter->setBrush(filled ? accent : field);
        if (check) {
            painter->drawRoundedRect(box, 3, 3);
            if (on || partial) {
                const QColor mark = enabled ? text_on(accent) : modern_colour("disabled", dark);
                painter->setPen(QPen(mark, std::max<qreal>(1.6, side / 8), Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
                painter->setBrush(Qt::NoBrush);
                if (partial) {
                    painter->drawLine(QPointF(box.left() + side * 0.28, box.center().y()),
                                      QPointF(box.right() - side * 0.28, box.center().y()));
                } else {
                    QPainterPath tick;
                    tick.moveTo(box.left() + side * 0.24, box.top() + side * 0.52);
                    tick.lineTo(box.left() + side * 0.43, box.top() + side * 0.70);
                    tick.lineTo(box.left() + side * 0.77, box.top() + side * 0.32);
                    painter->drawPath(tick);
                }
            }
        } else {
            painter->setBrush(field);
            painter->setPen(QPen(on && enabled ? accent : edge, on ? 1.6 : 1.0));
            painter->drawEllipse(box);
            if (on) {
                painter->setPen(Qt::NoPen);
                painter->setBrush(enabled ? accent : modern_colour("disabled", dark));
                const qreal dot = side * 0.5;
                painter->drawEllipse(QRectF(box.center().x() - dot / 2, box.center().y() - dot / 2, dot, dot));
            }
        }
        painter->restore();
    }
};

// What apply_appearance applied: the interface, the accent, the style the
// application had before (Classic's), and the Modern sheet and palette last
// set, so that refresh sets them again only when they change.
struct Appearance {
    Interface ui = Interface::classic;
    std::optional<QColor> accent;
    QString classic_style;
    bool applied = false;
    QString sheet;
    std::optional<bool> dark;
    std::optional<QString> looks;  // looks_key()'s, when the palette was set
    // The system's accent, as the palette had it before the app set one:
    // put back when a look's own accent is no longer in use.
    std::optional<QColor> system_accent;
};

Appearance& appearance() {
    static Appearance applied;
    return applied;
}

// `colour` at `alpha` (0 to 1) as a style sheet writes it.
QString rgba(const QColor& colour, double alpha) {
    return QStringLiteral("rgba(%1, %2, %3, %4)")
        .arg(colour.red())
        .arg(colour.green())
        .arg(colour.blue())
        .arg(alpha, 0, 'f', 2);
}

// The look in use for `dark` (Settings' dark or light look), in VML Modern;
// none in Classic, or for Light and Dark themselves.
const Look* look_for(bool dark) {
    if (appearance().ui != Interface::modern) {
        return nullptr;
    }
    const auto found = looks().find(dark ? looks_asked().dark : looks_asked().light);
    return found == looks().end() ? nullptr : &found->second;
}

// The looks asked for, as one text: a change of either is a change.
QString looks_key() { return looks_asked().light + '/' + looks_asked().dark; }

// The accent chosen, else the look's own (Mocha's amber, Desert's burnt
// orange) while it is in use; none: the system's.
std::optional<QColor> accent_chosen(bool dark) {
    if (appearance().accent) {
        return appearance().accent;
    }
    if (const Look* look = look_for(dark); look != nullptr && look->accent) {
        return look->accent;
    }
    return std::nullopt;
}

// The accent in use: the one chosen, else the palette's (the system's).
QColor accent_in_use(bool dark) {
    return accent_chosen(dark).value_or(QGuiApplication::palette().color(QPalette::Accent));
}

// The application's palette for the interface and accent applied: in
// Modern, its colours for `dark` and the accent where one is chosen. Classic
// sets none: Windows 11's style draws its own colours only while the
// application has no palette of its own (with any, even the accent alone,
// it fell back to a grey window), so Classic is the system's, its accent
// too. Set only when it changes.
void apply_palette(bool dark) {
    const Appearance& applied = appearance();
    if (applied.ui == Interface::classic) {
        return;
    }
    QPalette palette;  // nothing set: what is not set below is the style's
    {
        const auto colour = [dark](const char* name) { return modern_colour(name, dark); };
        palette.setColor(QPalette::Window, colour("window"));
        palette.setColor(QPalette::WindowText, colour("text"));
        palette.setColor(QPalette::Base, colour("card"));
        palette.setColor(QPalette::AlternateBase, colour("card_alt"));
        palette.setColor(QPalette::Button, colour("raised"));
        palette.setColor(QPalette::ButtonText, colour("text"));
        palette.setColor(QPalette::Text, colour("text"));
        palette.setColor(QPalette::BrightText, colour("danger"));
        palette.setColor(QPalette::PlaceholderText, colour("disabled"));
        palette.setColor(QPalette::ToolTipBase, colour("raised"));
        palette.setColor(QPalette::ToolTipText, colour("text"));
        palette.setColor(QPalette::Light, colour("hover"));
        palette.setColor(QPalette::Midlight, colour("raised"));
        palette.setColor(QPalette::Mid, colour("border_strong"));
        palette.setColor(QPalette::Dark, colour("border"));
        palette.setColor(QPalette::Shadow, dark ? QColor(0, 0, 0) : QColor(160, 160, 168));
        palette.setColor(QPalette::Link, dark ? QColor("#8ab4f8") : QColor("#2a5db0"));
        for (const QPalette::ColorRole role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
            palette.setColor(QPalette::Disabled, role, colour("disabled"));
        }
    }
    // The accent chosen or the look's; else the system's, set again only
    // where the palette (copied from the application's) still has another,
    // such as Mocha's amber after a switch to Dark.
    std::optional<QColor> accent = accent_chosen(dark);
    if (!accent && applied.system_accent && palette.color(QPalette::Accent) != *applied.system_accent) {
        accent = applied.system_accent;
    }
    if (accent) {
        palette.setColor(QPalette::Accent, *accent);
        palette.setColor(QPalette::Highlight, *accent);
        palette.setColor(QPalette::HighlightedText, text_on(*accent));
    }
    if (palette != QApplication::palette() || palette.resolveMask() != QApplication::palette().resolveMask()) {
        QApplication::setPalette(palette);
    }
}

// The Modern sheet for `dark`, set on the application if it changed;
// Classic's none.
// How many SheetHeld are alive.
int& sheet_holds() {
    static int holds = 0;
    return holds;
}

void apply_sheet(bool dark) {
    Appearance& applied = appearance();
    const QString sheet =
        applied.ui == Interface::modern ? modern_style_sheet(dark, accent_in_use(dark)) : QString();
    if (sheet_holds() > 0) {
        applied.sheet = sheet;  // set when the last SheetHeld goes
        return;
    }
    if (sheet != applied.sheet || qApp->styleSheet() != sheet) {
        applied.sheet = sheet;
        qApp->setStyleSheet(sheet);
    }
}

// Whether the system asks for dark colours; else, with none asked, whether
// the style's own window is dark (Modern's palette is set from this, so it
// cannot be asked).
bool scheme_is_dark() {
    const Qt::ColorScheme scheme = QGuiApplication::styleHints()->colorScheme();
    if (scheme != Qt::ColorScheme::Unknown) {
        return scheme == Qt::ColorScheme::Dark;
    }
    return QApplication::style()->standardPalette().window().color().lightness() < 128;
}

// The widgets style() styled, and their templates. A widget leaves when it
// is destroyed.
QHash<QWidget*, QString>& styled() {
    static QHash<QWidget*, QString> widgets;
    return widgets;
}

// What starts a template only Classic draws (style_classic).
const QString classic_only_mark = QStringLiteral("{classic_only}");

// `template_text` with each {name} the theme's colour of that name, as
// str.format(**colours) fills it: a doubled brace is a brace. {hint_style}
// is a hint's font-style for the interface in use; a template that starts
// with classic_only_mark is none in Modern.
QString filled(const QString& template_text) {
    const bool modern = appearance().ui == Interface::modern;
    if (template_text.startsWith(classic_only_mark)) {
        // VML Modern's style sheet draws the widget instead.
        return modern ? QString() : filled(template_text.mid(classic_only_mark.size()));
    }
    const bool dark = is_dark();
    std::vector<std::pair<QString, QString>> values;
    for (const auto& [name, pair] : colours) {
        values.emplace_back(name, dark ? pair.second : pair.first);
    }
    // A hint's slant: Classic's italic; Modern's hints are muted, upright.
    values.emplace_back("hint_style", modern ? "normal" : "italic");
    QString result;
    qsizetype at = 0;
    while (at < template_text.size()) {
        const QChar c = template_text[at];
        if ((c == '{' || c == '}') && at + 1 < template_text.size() && template_text[at + 1] == c) {
            result += c;
            at += 2;
            continue;
        }
        if (c == '{') {
            const qsizetype close = template_text.indexOf('}', at + 1);
            const QString name = close < 0 ? QString() : template_text.mid(at + 1, close - at - 1);
            const auto found = std::find_if(values.begin(), values.end(),
                                            [&name](const auto& value) { return value.first == name; });
            if (found != values.end()) {
                result += found->second;
                at = close + 1;
                continue;
            }
        }
        result += c;
        ++at;
    }
    return result;
}

}  // namespace

bool is_dark() {
    if (QGuiApplication::instance() == nullptr) {
        return false;
    }
    const Qt::ColorScheme scheme = QGuiApplication::styleHints()->colorScheme();
    if (scheme == Qt::ColorScheme::Dark) {
        return true;
    }
    if (scheme == Qt::ColorScheme::Light) {
        return false;
    }
    return QGuiApplication::palette().window().color().lightness() < 128;
}

QString look() {
    const bool dark = is_dark();
    if (appearance().ui != Interface::modern) {
        return dark ? QStringLiteral("dark") : QStringLiteral("light");
    }
    return dark ? looks_asked().dark : looks_asked().light;
}

bool is_mocha() { return look() == "mocha"; }

QColor mocha_accent() { return *looks().at("mocha").accent; }

std::optional<QColor> look_accent(const QString& name) {
    const auto found = looks().find(name);
    return found == looks().end() ? std::nullopt : found->second.accent;
}

QPixmap look_swatch(const QString& name, int size) {
    const bool dark = name == "dark" || is_dark_look(name);
    const auto found = looks().find(name);
    const auto colour = [&](const char* key) {
        if (found != looks().end()) {
            if (const auto own = found->second.modern.find(QString::fromLatin1(key)); own != found->second.modern.end()) {
                return QColor(own->second);
            }
        }
        const auto& pair = modern_colours.at(QString::fromLatin1(key));
        return QColor(dark ? pair.second : pair.first);
    };
    // The square at the left, a gap after it for the text beside it.
    const int gap = std::max(4, size / 3);
    const qreal ratio = qApp != nullptr ? qApp->devicePixelRatio() : 1.0;
    QPixmap swatch(QSize(size + gap, size) * ratio);
    swatch.setDevicePixelRatio(ratio);
    swatch.fill(Qt::transparent);
    QPainter painter(&swatch);
    painter.setRenderHint(QPainter::Antialiasing);
    const QRectF card(0.5, 0.5, size - 1.0, size - 1.0);
    painter.setPen(QPen(colour("border_strong"), 1.0));
    painter.setBrush(colour("card"));
    painter.drawRoundedRect(card, 3.0, 3.0);
    painter.setPen(Qt::NoPen);
    painter.setBrush(colour("text"));
    const qreal bar = std::max(2.0, size / 8.0);
    painter.drawRoundedRect(QRectF(size * 0.22, size * 0.28, size * 0.56, bar), bar / 2, bar / 2);
    painter.setBrush(look_accent(name).value_or(colour("text_muted")));
    painter.drawRoundedRect(QRectF(size * 0.22, size * 0.58, size * 0.34, bar), bar / 2, bar / 2);
    return swatch;
}

QColor color(const char* name, std::optional<bool> dark) {
    const QString key = QString::fromUtf8(name);
    if (!dark) {  // the window's own colours, in its look
        if (const Look* look = look_for(is_dark()); look != nullptr) {
            if (const auto found = look->window.find(key); found != look->window.end()) {
                return QColor(found->second);
            }
        }
    }
    const auto& pair = colours.at(key);
    return QColor(dark.value_or(is_dark()) ? pair.second : pair.first);
}

QColor readable(const QColor& series_colour) { return is_dark() ? series_colour.lighter(150) : series_colour; }

ChartColours chart_colours(std::optional<bool> dark) {
    if (!dark) {  // the window's own colours, in its look
        if (const Look* look = look_for(is_dark()); look != nullptr && look->chart) {
            return *look->chart;
        }
    }
    if (dark.value_or(is_dark())) {
        return {QColor(30, 30, 30), QColor(220, 220, 220), QColor(150, 150, 150), QColor(255, 255, 255, 38),
                QColor(170, 170, 170), QColor(200, 160, 255), QColor(120, 170, 255, 48)};
    }
    return light_chart_on(QColor("white"));
}

ChartColours export_chart_colours(const std::string& palette, const std::string& background, const std::string& text) {
    ChartColours exported = chart_colours(palette == "dark"     ? std::optional<bool>(true)
                                          : palette == "window" ? std::nullopt
                                                                : std::optional<bool>(false));
    const QColor chosen_background(QString::fromStdString(background));
    const QColor chosen_text(QString::fromStdString(text));
    if (!chosen_background.isValid() && !chosen_text.isValid()) {
        return exported;
    }
    if (chosen_background.isValid()) {
        exported.background = chosen_background;
    }
    if (chosen_text.isValid()) {
        exported.text = chosen_text;
    }
    // The axes part way from the background to the text, as the palettes'
    // are; the grid the text, faint, as theirs is.
    const auto mix = [](const QColor& from, const QColor& to, double share) {
        return QColor::fromRgbF(static_cast<float>(from.redF() + (to.redF() - from.redF()) * share),
                                static_cast<float>(from.greenF() + (to.greenF() - from.greenF()) * share),
                                static_cast<float>(from.blueF() + (to.blueF() - from.blueF()) * share));
    };
    exported.axis = mix(exported.background, exported.text, 0.6);
    exported.grid = exported.text;
    exported.grid.setAlpha(40);
    return exported;
}

void style_classic(QWidget* widget, const QString& template_text) {
    style(widget, classic_only_mark + template_text);
}

void style(QWidget* widget, const QString& template_text) {
    QHash<QWidget*, QString>& widgets = styled();
    if (!widgets.contains(widget)) {
        QObject::connect(widget, &QObject::destroyed, [widget] { styled().remove(widget); });
    }
    widgets.insert(widget, template_text);
    widget->setStyleSheet(filled(template_text));
}

void foreground(QTableWidgetItem* item, std::optional<QColor> colour) {
    item->setData(Qt::ForegroundRole, colour ? QVariant(*colour) : QVariant());
}

void recolour(QTableWidgetItem* item) {
    const QVariant data = item->data(Qt::ForegroundRole);
    if (!data.isValid()) {
        return;
    }
    const QColor current = data.canConvert<QBrush>() ? data.value<QBrush>().color() : data.value<QColor>();
    const QString name = current.name().toLower();
    for (const auto& [key, pair] : colours) {
        const bool look_match = std::any_of(looks().begin(), looks().end(), [&key, &name](const auto& look) {
            const auto found = look.second.window.find(key);
            return found != look.second.window.end() && found->second.toLower() == name;
        });
        if (pair.first.toLower() == name || pair.second.toLower() == name || look_match) {
            item->setData(Qt::ForegroundRole, color(key.toUtf8().constData()));
            return;
        }
    }
}

void refresh() {
    // Modern's palette and sheet for the colours now asked for (light or
    // dark), and the accent; set only where they change.
    if (appearance().applied && QApplication::instance() != nullptr) {
        const bool dark = scheme_is_dark();
        if (appearance().dark != dark || appearance().looks != looks_key()) {
            appearance().dark = dark;
            appearance().looks = looks_key();
            apply_palette(dark);
        }
        apply_sheet(dark);
    }
    const QWidgetList widgets = QApplication::allWidgets();
    // A palette left from before the switch follows the application's again:
    // a widget whose child has a style sheet can miss Qt's update. The sheets
    // below set what they need anew; the app sets no palette of its own but
    // for a widget that says it keeps one.
    const QColor window = QApplication::palette().window().color();
    for (QWidget* widget : widgets) {
        if (widget->property("keeps_its_palette").toBool()) {
            continue;
        }
        if (widget->testAttribute(Qt::WA_SetPalette) || widget->palette().window().color() != window) {
            widget->setPalette(QPalette());
        }
    }
    const QHash<QWidget*, QString>& templates = styled();
    for (auto item = templates.constBegin(); item != templates.constEnd(); ++item) {
        item.key()->setStyleSheet(filled(item.value()));
    }
    for (QWidget* widget : widgets) {
        if (!templates.contains(widget) && !widget->styleSheet().isEmpty()) {
            widget->setStyleSheet(widget->styleSheet());
        }
    }
}

Interface interface_of(const std::string& name) { return name == "modern" ? Interface::modern : Interface::classic; }

std::optional<QColor> accent_of(const std::string& colour) {
    const QColor chosen(QString::fromStdString(colour));
    if (colour.size() != 7 || colour.front() != '#' || !chosen.isValid()) {
        return std::nullopt;
    }
    return chosen;
}

namespace {
void show_for_interface();  // below, with the widgets it shows
}  // namespace

void apply_appearance(Interface ui, std::optional<QColor> accent) {
    if (QApplication::instance() == nullptr) {
        return;
    }
    Appearance& applied = appearance();
    if (!applied.applied) {
        applied.classic_style = QApplication::style()->name();
        applied.system_accent = QGuiApplication::palette().color(QPalette::Accent);
        applied.applied = true;
    }
    applied.ui = ui;
    applied.accent = accent;
    // The style first: setting it sets the style's own palette, which the
    // app's is laid over.
    const bool modern_now = dynamic_cast<ModernStyle*>(QApplication::style()) != nullptr;
    if (ui == Interface::modern && !modern_now) {
        QApplication::setStyle(new ModernStyle);  // the application owns it
    } else if (ui == Interface::classic && modern_now) {
        // Not while the app runs: Windows 11's style, set again after an
        // application palette, drew a grey window and an older Windows' tabs,
        // and kept Modern's accent. Classic comes back at the next start.
        return;
    }
    const bool dark = scheme_is_dark();
    applied.dark = dark;
    applied.looks = looks_key();
    apply_palette(dark);
    apply_sheet(dark);
    // Templates that differ by interface ({hint_style}, style_classic).
    const QHash<QWidget*, QString>& templates = styled();
    for (auto item = templates.constBegin(); item != templates.constEnd(); ++item) {
        item.key()->setStyleSheet(filled(item.value()));
    }
    show_for_interface();
}

Interface current_interface() { return appearance().ui; }

SheetHeld::SheetHeld() {
    // A sheet set already is taken off meanwhile: before a window is built
    // there are few widgets to restyle.
    if (++sheet_holds() == 1 && QApplication::instance() != nullptr && !qApp->styleSheet().isEmpty()) {
        qApp->setStyleSheet(QString());
    }
}

SheetHeld::~SheetHeld() {
    if (--sheet_holds() == 0 && appearance().applied && QApplication::instance() != nullptr &&
        qApp->styleSheet() != appearance().sheet) {
        qApp->setStyleSheet(appearance().sheet);
    }
}

void set_role(QWidget* widget, const char* role) {
    widget->setProperty("vmlRole", QString::fromUtf8(role));
    // A role set after the widget was polished is drawn as it once it is
    // polished again.
    widget->style()->unpolish(widget);
    widget->style()->polish(widget);
}

namespace {

// Widgets shown in one interface only (modern_only, classic_only), group
// boxes titled in Classic only (classic_box), and containers' margins in
// Modern (container).
struct InterfaceOnly {
    QPointer<QWidget> widget;
    bool modern;
};
std::vector<InterfaceOnly>& interface_only() {
    static std::vector<InterfaceOnly> widgets;
    return widgets;
}
std::vector<std::pair<QPointer<QGroupBox>, QString>>& classic_boxes() {
    static std::vector<std::pair<QPointer<QGroupBox>, QString>> boxes;
    return boxes;
}
std::vector<std::pair<QPointer<QWidget>, QMargins>>& containers() {
    static std::vector<std::pair<QPointer<QWidget>, QMargins>> widgets;
    return widgets;
}

// Each widget shown, box titled and container's room set as the interface
// in use has it.
void show_for_interface() {
    const bool modern = appearance().ui == Interface::modern;
    auto& widgets = interface_only();
    std::erase_if(widgets, [](const InterfaceOnly& item) { return item.widget.isNull(); });
    for (const InterfaceOnly& item : widgets) {
        item.widget->setVisible(item.modern == modern);
    }
    auto& boxes = classic_boxes();
    std::erase_if(boxes, [](const auto& item) { return item.first.isNull(); });
    for (const auto& [box, title] : boxes) {
        box->setTitle(modern ? QString() : title);
    }
    auto& padded = containers();
    std::erase_if(padded, [](const auto& item) { return item.first.isNull(); });
    for (const auto& [widget, margins] : padded) {
        widget->layout()->setContentsMargins(modern ? margins : QMargins());
    }
}

void add_interface_only(QWidget* widget, bool modern) {
    interface_only().push_back({widget, modern});
    // Hidden where it is another interface's, and otherwise left as it is:
    // shown before it has a parent, as it is when built, a widget is made a
    // window of its own, and unmade when it is added to one. That cost about
    // 115 ms of the window's build in VML Modern, 20 ms in Classic.
    if (modern != (appearance().ui == Interface::modern)) {
        widget->hide();
    }
}

}  // namespace

void modern_only(QWidget* widget) { add_interface_only(widget, true); }

void classic_only(QWidget* widget) { add_interface_only(widget, false); }

void classic_box(QGroupBox* box, const QString& title) {
    set_role(box, "unboxed");
    classic_boxes().emplace_back(box, title);
    box->setTitle(appearance().ui == Interface::modern ? QString() : title);
}

QWidget* container(const char* role, bool horizontal, QBoxLayout** layout) {
    auto* widget = new QWidget;
    set_role(widget, role);
    auto* box = horizontal ? static_cast<QBoxLayout*>(new QHBoxLayout(widget))
                           : static_cast<QBoxLayout*>(new QVBoxLayout(widget));
    // Modern's room inside it, by role; Classic's none (show_for_interface).
    const QString name = QString::fromUtf8(role);
    const QMargins margins = name == "card" || name == "padding" ? QMargins(14, 12, 14, 12)
                             : name == "footer"                  ? QMargins(14, 8, 14, 8)
                             : name == "toolbar"                 ? QMargins(10, 6, 10, 6)
                                                                 : QMargins();
    containers().emplace_back(widget, margins);
    box->setContentsMargins(appearance().ui == Interface::modern ? margins : QMargins());
    if (layout != nullptr) {
        *layout = box;
    }
    return widget;
}

QFrame* separator(int classic_gap) {
    // Classic: an empty frame the gap's width. Modern's sheet draws it a
    // line and sets its width (min-width and max-width outrank this).
    auto* line = new QFrame;
    line->setFrameShape(QFrame::NoFrame);
    line->setFixedWidth(classic_gap);
    set_role(line, "separator");
    if (classic_gap == 0) {
        // No gap in Classic: not there at all, so the layout's spacing is
        // not doubled around it.
        modern_only(line);
    }
    return line;
}

QColor text_on(const QColor& colour) {
    // WCAG's relative luminance; the text that contrasts more.
    const auto linear = [](float channel) {
        return channel <= 0.04045F ? channel / 12.92F : std::pow((channel + 0.055F) / 1.055F, 2.4F);
    };
    const float luminance = 0.2126F * linear(colour.redF()) + 0.7152F * linear(colour.greenF()) +
                            0.0722F * linear(colour.blueF());
    const float with_black = (luminance + 0.05F) / 0.05F;
    const float with_white = 1.05F / (luminance + 0.05F);
    return with_black >= with_white ? QColor("#141416") : QColor("#ffffff");
}

QColor modern_colour(const char* name, bool dark) {
    const QString key = QString::fromUtf8(name);
    if (const Look* look = look_for(dark); look != nullptr) {
        if (const auto found = look->modern.find(key); found != look->modern.end()) {
            return QColor(found->second);
        }
    }
    const auto& pair = modern_colours.at(key);
    return QColor(dark ? pair.second : pair.first);
}

QString modern_style_sheet(bool dark, const QColor& accent) {
    QString sheet = QStringLiteral(R"(
QGroupBox { background: {card}; border: 1px solid {border}; border-radius: 10px; margin-top: 0; padding: 36px 12px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 14px; top: 11px; padding: 0; color: {text}; }
QGroupBox[vmlUntitled="true"] { padding-top: 12px; }
QGroupBox QGroupBox { background: {inset}; border: 0; border-radius: 8px; padding: 32px 10px 10px 10px; }
QGroupBox QGroupBox::title { left: 10px; top: 9px; color: {text_soft}; }
QGroupBox QGroupBox[vmlUntitled="true"] { padding-top: 10px; }
QTabBar[vmlRole="segmented"] { qproperty-drawBase: 0; }
QTabBar[vmlRole="segmented"] { outline: 0; }
QTabBar[vmlRole="segmented"]::tab { background: transparent; color: {text_soft}; border: 0; border-radius: 6px; padding: 5px 12px; margin: 2px 2px 6px 0; min-height: 18px; }
QTabBar[vmlRole="segmented"]::tab:hover:!selected { color: {text}; background: {hover}; }
QTabBar[vmlRole="segmented"]::tab:selected { background: {accent}; color: {accent_text}; font-weight: 600; }
QTabWidget[vmlRole="segmented"]::pane { border: 0; top: 0; }
QTabWidget[vmlRole="segmented"]::tab-bar { left: 0; }
QFrame[vmlRole="separator"] { background: {border_strong}; border: 0; min-width: 1px; max-width: 1px; margin: 4px 6px; }
QWidget[vmlRole="padding"] { background: transparent; }
QGroupBox[vmlRole="unboxed"] { background: transparent; border: 0; border-radius: 0; margin: 0; padding: 0; }
QWidget[vmlRole="footer"] { background: {card_alt}; border: 0; border-top: 1px solid {border}; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; }
QLabel[vmlRole="title"] { font-weight: 600; color: {text}; }
QLabel[vmlRole="hint"] { color: {text_muted}; }
QFrame[vmlRole="choice"] { background: {raised}; border: 1px solid {border_strong}; border-radius: 10px; }
QFrame[vmlRole="choice"][checked="true"] { background: {accent_tint}; border: 2px solid {accent}; }
QWidget[vmlRole="segments"] { background: {field}; border: 1px solid {border}; border-radius: 8px; }
QWidget[vmlRole="segments"] QPushButton { background: transparent; color: {text_soft}; border: 0; border-radius: 5px; padding: 4px 12px; min-height: 18px; }
QWidget[vmlRole="segments"] QPushButton:hover:!checked { background: {hover}; color: {text}; }
QWidget[vmlRole="segments"] QPushButton:checked { background: {hover}; color: {text}; font-weight: 600; }
QLabel[vmlRole="notice"] { background: {accent_tint}; color: {text}; border: 1px solid {border}; border-radius: 8px; padding: 8px 12px; }
QPushButton { background: {raised}; color: {text}; border: 1px solid {border_strong}; border-radius: 6px; padding: 4px 12px; min-height: 18px; }
QPushButton:hover { background: {hover}; }
QPushButton:pressed { background: {pressed}; }
QPushButton:checked { background: {accent_tint}; border-color: {accent}; }
QPushButton:disabled { background: {card}; color: {disabled}; border-color: {border}; }
QPushButton[vmlRole="primary"] { background: {accent}; color: {accent_text}; border: 1px solid {accent}; font-weight: 600; padding: 4px 16px; }
QPushButton[vmlRole="primary"]:hover { background: {accent_hover}; border-color: {accent_hover}; }
QPushButton[vmlRole="primary"]:pressed { background: {accent_pressed}; }
QPushButton[vmlRole="primary"]:disabled { background: {raised}; color: {disabled}; border-color: {border}; font-weight: 400; }
QPushButton[vmlRole="danger"] { background: transparent; color: {danger}; border-color: {danger_border}; }
QPushButton[vmlRole="danger"]:hover { background: {danger_tint}; }
QPushButton[vmlRole="danger"]:disabled { color: {disabled}; border-color: {border}; }
QPushButton[vmlRole="quiet"] { background: transparent; border-color: transparent; }
QPushButton[vmlRole="quiet"]:hover { background: {hover}; }
QPushButton[vmlRole="quiet"]:disabled { color: {disabled}; }
QPushButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: right center; right: 4px; }
QLineEdit, QComboBox { background: {field}; color: {text}; border: 1px solid {border_strong}; border-radius: 5px; padding: 2px 6px; min-height: 18px; selection-background-color: {accent}; selection-color: {accent_text}; }
QLineEdit:focus, QComboBox:focus { border-color: {accent}; }
QLineEdit:disabled, QComboBox:disabled { color: {disabled}; background: {card}; border-color: {border}; }
QAbstractSpinBox > QLineEdit, QComboBox > QLineEdit { border: 0; background: transparent; padding: 0; min-height: 0; }
QComboBox QAbstractItemView { background: {card}; color: {text}; border: 1px solid {border_strong}; selection-background-color: {accent_tint}; selection-color: {text}; outline: 0; }
QTabWidget::pane { border: 0; border-top: 1px solid {border}; top: -1px; }
QTabBar::tab { background: transparent; color: {text_muted}; border: 0; border-bottom: 2px solid transparent; padding: 6px 12px; margin-right: 2px; }
QTabBar::tab:selected { color: {text}; font-weight: 600; border-bottom: 2px solid {accent}; }
QTabBar::tab:hover:!selected { color: {text}; border-bottom: 2px solid {border_strong}; }
QTabBar::tab:disabled { color: {disabled}; }
QHeaderView::section { background: {raised}; color: {text_soft}; border: 0; border-right: 1px solid {border}; border-bottom: 1px solid {border}; padding: 4px 6px; font-weight: 600; }
QTableView, QTableWidget, QListView, QTreeView { background: {card}; alternate-background-color: {card_alt}; color: {text}; border: 1px solid {border}; border-radius: 6px; gridline-color: {border}; selection-background-color: {accent_tint}; selection-color: {text}; }
QTableCornerButton::section { background: {raised}; border: 0; }
QMenu { background: {card}; color: {text}; border: 1px solid {border_strong}; border-radius: 6px; padding: 4px; }
QMenu::item { padding: 5px 24px 5px 20px; border-radius: 4px; }
QMenu::item:selected { background: {accent_tint}; }
QMenu::item:disabled { color: {disabled}; }
QMenu::separator { height: 1px; background: {border}; margin: 4px 6px; }
QToolTip { background: {raised}; color: {text}; border: 1px solid {border_strong}; padding: 4px; }
QProgressBar { background: {field}; border: 1px solid {border}; border-radius: 4px; text-align: center; color: {text}; }
QProgressBar::chunk { background: {accent}; border-radius: 3px; }
QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle { background: {scroll}; border-radius: 4px; min-height: 28px; min-width: 28px; margin: 2px; }
QScrollBar::handle:hover { background: {text_muted}; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QWidget[vmlRole="card"] { background: {card}; border: 1px solid {border}; border-radius: 8px; }
QFrame[vmlRole="separator"] { background: {border_strong}; border: 0; min-width: 1px; max-width: 1px; margin: 4px 6px; }
QWidget[vmlRole="tile"] { background: {raised}; border: 0; border-radius: 8px; }
QWidget[vmlRole="toolbar"] { background: {raised}; border: 1px solid {border}; border-radius: 8px; }
)");
    const QColor accent_hover = dark ? accent.lighter(112) : accent.darker(108);
    const QColor accent_pressed = dark ? accent.darker(108) : accent.darker(118);
    std::vector<std::pair<QString, QString>> values{
        {"accent", accent.name()},
        {"accent_text", text_on(accent).name()},
        {"accent_hover", accent_hover.name()},
        {"accent_pressed", accent_pressed.name()},
        {"accent_tint", rgba(accent, dark ? 0.26 : 0.20)},
        {"danger_tint", rgba(modern_colour("danger", dark), 0.12)},
    };
    for (const auto& [name, pair] : modern_colours) {
        values.emplace_back(name, modern_colour(name.toUtf8().constData(), dark).name());
    }
    // Longest names first: {card_alt} before {card}.
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (const auto& [name, value] : values) {
        sheet.replace("{" + name + "}", value);
    }
    return sheet.trimmed();
}

std::vector<std::pair<QString, QString>> theme_choices() {
    return {
#ifdef __APPLE__
        {app::tr("Same as macOS"), ""},
#else
        {app::tr("Same as Windows"), ""},
#endif
        {app::tr("Light"), "light"},
        {app::tr("Dark"), "dark"},
    };
}

std::vector<std::pair<QString, QString>> look_choices(bool dark) {
    if (dark) {
        return {{app::tr("Dark"), "dark"},         {app::tr("Mocha"), "mocha"},       {app::tr("Joker"), "joker"},
                {app::tr("Nitrate"), "nitrate"}, {app::tr("Phosphor"), "phosphor"}};
    }
    return {{app::tr("Light"), "light"}, {app::tr("Dim"), "dim"}, {app::tr("Paper"), "paper"}, {app::tr("Desert"), "desert"}};
}

void apply_theme(const QString& theme, const QString& light_look, const QString& dark_look) {
    if (QGuiApplication::instance() == nullptr) {
        return;
    }
    // "mocha", from before there were looks, is Dark in Mocha's look.
    const Qt::ColorScheme scheme = theme == "light"                     ? Qt::ColorScheme::Light
                                   : theme == "dark" || theme == "mocha" ? Qt::ColorScheme::Dark
                                                                        : Qt::ColorScheme::Unknown;
    const QString looks_before = looks_key();
    // Each side's own looks only: a dark look for light would draw light
    // in dark colours.
    const bool light_known = light_look == "dim" || light_look == "paper" || light_look == "desert";
    looks_asked().light = light_known ? light_look : QStringLiteral("light");
    looks_asked().dark = theme == "mocha"          ? QStringLiteral("mocha")
                         : is_dark_look(dark_look) ? dark_look
                                                   : QStringLiteral("dark");
    QGuiApplication::styleHints()->setColorScheme(scheme);
    // A look for another is no change of scheme, so nothing tells the
    // window: its palette and sheet are set here.
    if (looks_key() != looks_before) {
        refresh();
    }
}

}  // namespace vml::app::theme
