// The window's colours, for a light or a dark theme (the Python app's
// vmaf_app/ui/theme.py).
//
// Qt's windows11 style draws the window in Windows' light or dark colours
// (or the ones Settings > Window > Theme asks for: apply_theme), and every
// widget that takes its colours from the palette follows. What the app
// colours itself (a muted hint, a failed score, a chart) is named here once,
// with a value for each theme.
//
// color(name) is the colour for the theme in use. A widget styled with
// style(widget, "color: {muted};") is styled again when the theme changes
// (refresh), as the palette's own colours are. Painted things (the charts,
// the table's cells) ask for their colours when they draw. Files the app
// writes (an exported graph) keep the light colours whatever the window
// shows.
#pragma once

#include <QColor>
#include <QString>
#include <QStringList>

#include <optional>

class QTableWidgetItem;
class QWidget;

namespace vml::app::theme {

// Settings.theme's values: "" follows Windows.
const QStringList& themes();

// The colours a chart is drawn in.
struct ChartColours {
    QColor background;
    QColor text;
    QColor axis;
    QColor grid;
    QColor crosshair;
};

// Whether the window is drawn in dark colours now.
bool is_dark();

// A named colour for the theme in use (or for dark or light, if given).
QColor color(const char* name, std::optional<bool> dark = std::nullopt);

// A series' colour as text: lighter on a dark window, where the palette's
// darker blues and reds read poorly; as it is on a light one.
QColor readable(const QColor& series_colour);

// The chart colours for the theme in use (or for dark or light, if given).
ChartColours chart_colours(std::optional<bool> dark = std::nullopt);

// widget->setStyleSheet(template) with {name}s filled in from this theme's
// colours, done again by refresh() when the theme changes.
void style(QWidget* widget, const QString& template_text);

// A table cell's text colour; none for the table's own, which follows the
// palette as the theme changes.
void foreground(QTableWidgetItem* item, std::optional<QColor> colour);

// A table cell coloured from this module, in its colour's value for the
// theme now in use; a cell without a colour of its own is left alone.
void recolour(QTableWidgetItem* item);

// Restyles every widget style() styled, for the theme now in use, and
// applies every other style sheet again: Qt resolves a sheet's palette()
// colours when the sheet is applied.
void refresh();

// Asks Qt for Settings.theme's colours: "light", "dark", or "" for Windows'
// own.
void apply_theme(const QString& theme);

}  // namespace vml::app::theme
