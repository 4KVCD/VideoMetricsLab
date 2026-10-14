#include "theme.h"

#include <QApplication>
#include <QBrush>
#include <QGuiApplication>
#include <QHash>
#include <QPalette>
#include <QPointer>
#include <QStyleHints>
#include <QTableWidgetItem>
#include <QWidget>

#include <algorithm>
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

// The widgets style() styled, and their templates. A widget leaves when it
// is destroyed.
QHash<QWidget*, QString>& styled() {
    static QHash<QWidget*, QString> widgets;
    return widgets;
}

// `template_text` with each {name} the theme's colour of that name, as
// str.format(**colours) fills it: a doubled brace is a brace.
QString filled(const QString& template_text) {
    const bool dark = is_dark();
    std::vector<std::pair<QString, QString>> values;
    for (const auto& [name, pair] : colours) {
        values.emplace_back(name, dark ? pair.second : pair.first);
    }
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

const QStringList& themes() {
    static const QStringList all = {"", "light", "dark"};
    return all;
}

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
                QColor(170, 170, 170)};
    }
    return {QColor("white"), QColor(40, 40, 40), QColor(90, 90, 90), QColor(0, 0, 0, 40), QColor(120, 120, 120)};
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
    const QWidgetList widgets = QApplication::allWidgets();
    // A palette left from before the switch follows the application's again:
    // a widget whose child has a style sheet can miss Qt's update. The sheets
    // below set what they need anew; the app sets no palette of its own.
    const QColor window = QApplication::palette().window().color();
    for (QWidget* widget : widgets) {
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
