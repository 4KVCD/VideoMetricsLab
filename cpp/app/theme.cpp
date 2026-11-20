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
    {"window", {"#f3f3f5", "#1b1b1d"}},         // the window behind the cards
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

// The accent in use: the one chosen, else the palette's (the system's).
QColor accent_in_use() {
    if (appearance().accent) {
        return *appearance().accent;
    }
    return QGuiApplication::palette().color(QPalette::Accent);
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
    if (applied.accent) {
        palette.setColor(QPalette::Accent, *applied.accent);
        palette.setColor(QPalette::Highlight, *applied.accent);
        palette.setColor(QPalette::HighlightedText, text_on(*applied.accent));
    }
    if (palette != QApplication::palette() || palette.resolveMask() != QApplication::palette().resolveMask()) {
        QApplication::setPalette(palette);
    }
}

// The Modern sheet for `dark`, set on the application if it changed;
// Classic's none.
void apply_sheet(bool dark) {
    Appearance& applied = appearance();
    const QString sheet =
        applied.ui == Interface::modern ? modern_style_sheet(dark, accent_in_use()) : QString();
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

QColor color(const char* name, std::optional<bool> dark) {
    const auto& pair = colours.at(QString::fromUtf8(name));
    return QColor(dark.value_or(is_dark()) ? pair.second : pair.first);
}

QColor readable(const QColor& series_colour) { return is_dark() ? series_colour.lighter(150) : series_colour; }

ChartColours chart_colours(std::optional<bool> dark) {
    if (dark.value_or(is_dark())) {
        return {QColor(30, 30, 30), QColor(220, 220, 220), QColor(150, 150, 150), QColor(255, 255, 255, 38),
                QColor(170, 170, 170), QColor(200, 160, 255), QColor(120, 170, 255, 48)};
    }
    return {QColor("white"), QColor(40, 40, 40), QColor(90, 90, 90), QColor(0, 0, 0, 40), QColor(120, 120, 120),
            QColor(120, 60, 200), QColor(40, 110, 230, 40)};
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
    const bool dark = is_dark();
    for (const auto& [_, pair] : colours) {
        if (pair.first.toLower() == name || pair.second.toLower() == name) {
            item->setData(Qt::ForegroundRole, QColor(dark ? pair.second : pair.first));
            return;
        }
    }
}

void refresh() {
    // Modern's palette and sheet for the colours now asked for (light or
    // dark), and the accent; set only where they change.
    if (appearance().applied && QApplication::instance() != nullptr) {
        const bool dark = scheme_is_dark();
        if (appearance().dark != dark) {
            appearance().dark = dark;
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
    widget->setVisible(modern == (appearance().ui == Interface::modern));
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
    const auto& pair = modern_colours.at(QString::fromUtf8(name));
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
        values.emplace_back(name, dark ? pair.second : pair.first);
    }
    // Longest names first: {card_alt} before {card}.
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (const auto& [name, value] : values) {
        sheet.replace("{" + name + "}", value);
    }
    return sheet.trimmed();
}

void apply_theme(const QString& theme) {
    if (QGuiApplication::instance() == nullptr) {
        return;
    }
    const Qt::ColorScheme scheme = theme == "light"  ? Qt::ColorScheme::Light
                                   : theme == "dark" ? Qt::ColorScheme::Dark
                                                     : Qt::ColorScheme::Unknown;
    QGuiApplication::styleHints()->setColorScheme(scheme);
}

}  // namespace vml::app::theme
