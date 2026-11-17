// The image a graph's Export PNG saves (Metric Graphs, the Bitrate tab): the
// bare chart is a set of unlabelled lines, which says nothing once the file
// leaves the app, so a title, a table keyed by the lines' colours, and the
// table's figures go around it.
#pragma once

#include <QColor>
#include <QFont>
#include <QPixmap>
#include <QString>

#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "theme.h"

namespace vml::app {

class ChartWidget;

// The statistics' cells, the readout and the exported table: a monospaced
// face, so that digits share a width; the system's own (asking Qt for one
// the system lacks costs 100 ms the first time, finding a substitute).
#ifdef __APPLE__
inline constexpr const char* number_font_family = "Menlo";
#else
inline constexpr const char* number_font_family = "Consolas";
#endif

// The table under the chart: its headings, and (label, colour, cells) for
// each line drawn, the cells one per heading after the first.
using ExportTable = std::pair<std::vector<QString>, std::vector<std::tuple<QString, QColor, std::vector<QString>>>>;

// A coloured word after the title, for marks the table does not key (the
// Bitrate tab's frame types).
struct ExportKey {
    QString text;
    QColor color;
};

// `chart` as shown, with `title` above it (then `keys`) and `table` under
// it (none without rows), in `font` for the title; all of it in `colours`
// (theme::export_chart_colours). At the chart's pixel ratio, so that the
// text is the size of its labels.
QPixmap export_image(const ChartWidget& chart, const QString& title, const ExportTable& table, const QFont& font,
                     const theme::ChartColours& colours, std::span<const ExportKey> keys = {});

}  // namespace vml::app
