// The window's colours, for a light or a dark theme (the Python app's
// vmaf_app/ui/theme.py).
//
// Qt's windows11 style draws the window in Windows' light or dark colours
// (or the ones Settings > Window > Theme asks for: apply_theme), or, in the
// VML Modern interface, Fusion in the app's own (apply_appearance), and every
// widget that takes its colours from the palette follows. What the app
// colours itself (a muted hint, a failed score, a chart) is named here once,
// with a value for each theme.
//
// color(name) is the colour for the theme in use. A widget styled with
// style(widget, "color: {muted};") is styled again when the theme changes
// (refresh), as the palette's own colours are. Painted things (the charts,
// the table's cells) ask for their colours when they draw. An exported
// graph is drawn in the colours Settings' Exported graphs asks for
// (export_chart_colours): light unless the user chose otherwise.
#pragma once

#include <QColor>
#include <QString>

#include <optional>
#include <string>
#include <utility>
#include <vector>

class QBoxLayout;
class QFrame;
class QGroupBox;
class QTableWidgetItem;
class QWidget;

namespace vml::app::theme {

// The colours a chart is drawn in.
struct ChartColours {
    QColor background;
    QColor text;
    QColor axis;
    QColor grid;
    QColor crosshair;
    QColor difference;  // Metric Graphs' difference of two lines (ChartWidget::set_band)
    QColor highlight;   // Metric Graphs' frame range, shaded under the lines (ChartWidget::set_highlight)
};

// Whether the window is drawn in dark colours now (Mocha's too).
bool is_dark();

// Whether the window is drawn in Mocha's colours now: Settings.theme "mocha",
// in VML Modern (in Classic, Mocha is Dark).
bool is_mocha();

// Mocha's accent, used while no accent is chosen (Solar Dusk's amber).
QColor mocha_accent();

// A named colour for the theme in use (or for dark or light, if given).
QColor color(const char* name, std::optional<bool> dark = std::nullopt);

// A series' colour as text: lighter on a dark window, where the palette's
// darker blues and reds read poorly; as it is on a light one.
QColor readable(const QColor& series_colour);

// The chart colours for the theme in use (or for dark or light, if given).
ChartColours chart_colours(std::optional<bool> dark = std::nullopt);

// The colours an exported graph is drawn in (Settings' Exported graphs):
// `palette` "dark" or "window" (the theme in use) for those charts' colours,
// anything else the light ones; then `background` and `text` ("#rrggbb"),
// where given, in place of the palette's. With either given, the axes and
// the grid are worked out from the two, so that they read on any
// background; the lines, the difference and the frame range keep theirs.
ChartColours export_chart_colours(const std::string& palette, const std::string& background, const std::string& text);

// widget->setStyleSheet(template) with {name}s filled in from this theme's
// colours, done again by refresh() when the theme changes. {hint_style} is
// a hint's font-style: italic in Classic, upright in VML Modern.
void style(QWidget* widget, const QString& template_text);

// style(), in VML Classic only: in Modern the widget has no sheet of its
// own, and Modern's draws it (a role, say).
void style_classic(QWidget* widget, const QString& template_text);

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

// Asks Qt for Settings.theme's colours: "light", "dark", "mocha" (dark, in
// VML Modern's chocolate browns and amber, after the Solar Dusk palette), or
// "" for Windows' own.
void apply_theme(const QString& theme);

// The themes to choose from, in order, each as its name (translated) and its
// Settings.theme value: the system's (""), "light", "dark" and "mocha".
// Settings > Appearance and the title bar's theme menu both offer these.
std::vector<std::pair<QString, QString>> theme_choices();

// The window's interface (Settings > Window > Interface).
//
// Classic is the window as it has always been: the system's style
// (windows11 on Windows, Fusion on macOS), its palette, no style sheet of
// the app's own.
//
// VML Modern is Qt's Fusion style, which draws every control from the
// palette and takes a style sheet as written, with the palette and one
// application style sheet made from modern_colours: cards for group boxes,
// a strip for a toolbar, and buttons by their role (set_role). Both follow
// light and dark (refresh applies Modern's again). Modern takes the accent
// chosen; Classic keeps the system's (apply_palette says why).
enum class Interface { classic, modern };

// Settings.interface's interface: "modern" is Modern, anything else Classic.
Interface interface_of(const std::string& name);

// The accent colour: Settings.accent_colour ("#rrggbb"), or none for the
// system's, as the palette has it.
std::optional<QColor> accent_of(const std::string& colour);

// Applies `ui` and, in Modern, `accent` (none: the system's) to the
// application, at once: the style, the palette and the style sheet. Classic
// is the window as it was before either existed; asked for while Modern is
// shown, it is left for the next start (the system's style does not come
// back whole once the app has set a palette).
void apply_appearance(Interface ui, std::optional<QColor> accent);

// The interface applied last (Classic until apply_appearance).
Interface current_interface();

// While one is alive, VML Modern's style sheet is not set on the
// application (one set already is taken off); when the last goes, the sheet
// of the interface and colours applied meanwhile is. Not for while a window
// is shown, which would be drawn without it meanwhile.
//
// main holds it while the window is built. With the sheet set, Qt matched
// each of the window's ~1100 widgets against it about four times as they
// were built and added to their parents (a new parent polishes a polished
// widget's subtree again); set once the window is built, it matches each
// once. Sizes taken from size hints while the window is built are taken
// again then (MainWindow::fit_to_style).
class SheetHeld {
public:
    SheetHeld();
    ~SheetHeld();
    SheetHeld(const SheetHeld&) = delete;
    SheetHeld& operator=(const SheetHeld&) = delete;
};

// A widget's role in the Modern interface's style sheet: "primary" (the
// one action a tab is for, in the accent), "danger" (one that removes
// something), "quiet" (a lesser one, as text), "card" (a box of its own),
// "tile" (a figure in a card, such as a saved score) or "toolbar" (a strip
// of controls). On a tab bar, "segmented": its tabs a row of pills, the
// current one in the accent (and its QTabWidget's too: no frame). On a
// label, "notice": a note set apart, a tinted strip. Classic draws every
// role alike.
void set_role(QWidget* widget, const char* role);

// A gap between a toolbar's groups of controls: in Classic `classic_gap`
// pixels of nothing, as the layout had (none: not there at all); in VML
// Modern a thin upright line of its border colour (the "separator" role).
QFrame* separator(int classic_gap = 12);

// Layout that is VML Modern's alone, Classic staying as it has always been.
//
// A widget shown in one interface only (a card's title, a label one layout
// has and the other has not): its visibility follows the interface applied.
// Only for widgets nothing else shows or hides.
void modern_only(QWidget* widget);
void classic_only(QWidget* widget);
// A group box Classic frames, titled `title`, and Modern leaves unboxed and
// untitled (role "unboxed"): the cards inside it are Modern's boxes.
void classic_box(QGroupBox* box, const QString& title);
// A widget for a role (a "card", a "toolbar", a "footer", or "padding": room
// alone) that groups controls in Modern: its layout's margins are Modern's
// room inside the role, none in Classic, where the role is not drawn, so the
// controls sit as they did; they follow the interface applied. Its layout
// runs across (horizontal) or down.
QWidget* container(const char* role, bool horizontal, QBoxLayout** layout = nullptr);

// Black or white, whichever reads better on `colour` (text on an accent
// button).
QColor text_on(const QColor& colour);

// The Modern interface's colours by name, for dark or light: the palette's
// and the style sheet's.
QColor modern_colour(const char* name, bool dark);

// The Modern interface's style sheet for dark or light and `accent`.
QString modern_style_sheet(bool dark, const QColor& accent);

}  // namespace vml::app::theme
