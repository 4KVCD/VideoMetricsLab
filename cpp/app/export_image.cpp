#include "export_image.h"

#include <QFontMetrics>
#include <QPainter>

#include <algorithm>
#include <cmath>

#include "chart.h"

namespace vml::app {

namespace {

// The image's layout.
constexpr int export_margin = 16;
constexpr int export_gap = 8;
constexpr int export_swatch = 12;
constexpr int export_row_padding = 6;

}  // namespace

QPixmap export_image(const ChartWidget& chart_widget, const QString& title, const ExportTable& table,
                     const QFont& font) {
    // The chart comes in the screen's pixels (twice its size on a Retina
    // display); the title, legend and table are drawn at the same scale, in
    // the same units, or their text would be half the size of its labels.
    QPixmap chart = chart_widget.render_to_pixmap();
    const double ratio = std::max(1.0, chart_widget.devicePixelRatioF());
    chart.setDevicePixelRatio(ratio);
    const int chart_width = static_cast<int>(std::lround(chart.width() / ratio));
    const int chart_height = static_cast<int>(std::lround(chart.height() / ratio));
    const auto& [headers, rows] = table;

    QFont title_font(font);
    title_font.setBold(true);
    title_font.setPointSize(std::max(10, title_font.pointSize() + 3));
    const QFontMetrics title_metrics(title_font);
    QFont cell_font(QString::fromLatin1(number_font_family));
    cell_font.setStyleHint(QFont::Monospace);
    const QFontMetrics cell_metrics(cell_font);
    const int row_height = cell_metrics.lineSpacing() + export_row_padding;

    // The first column carries the colour square too.
    std::vector<int> widths;
    for (std::size_t column = 0; column < headers.size(); ++column) {
        int width = cell_metrics.horizontalAdvance(headers[column]);
        for (const auto& [label, color, cells] : rows) {
            width = std::max(width, cell_metrics.horizontalAdvance(column == 0 ? label : cells[column - 1]));
        }
        if (column == 0) {
            width += export_swatch + export_gap;
        }
        widths.push_back(width + 2 * export_gap);
    }
    int table_width = 0;
    for (const int width : widths) {
        table_width += width;
    }
    const int table_height = rows.empty() ? 0 : static_cast<int>(rows.size() + 1) * row_height;
    const int content_width = std::max(chart_width, table_width);
    const int height = export_margin + title_metrics.height() + export_gap + chart_height +
                       (rows.empty() ? 0 : export_gap + table_height) + export_margin;

    QPixmap image(static_cast<int>(std::lround((content_width + 2 * export_margin) * ratio)),
                  static_cast<int>(std::lround(height * ratio)));
    image.setDevicePixelRatio(ratio);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setPen(QColor(QStringLiteral("#111111")));
    painter.setFont(title_font);
    painter.drawText(export_margin, export_margin + title_metrics.ascent(), title);
    int y = export_margin + title_metrics.height() + export_gap;
    painter.drawPixmap(export_margin, y, chart);
    y += chart_height + export_gap;
    if (!rows.empty()) {
        QFont header_font(cell_font);
        header_font.setBold(true);
        int baseline = y + cell_metrics.ascent() + export_row_padding / 2;
        painter.setFont(header_font);
        painter.setPen(QColor(QStringLiteral("#111111")));
        int x = export_margin;
        for (std::size_t column = 0; column < headers.size(); ++column) {
            painter.drawText(x + export_gap, baseline, headers[column]);
            x += widths[column];
        }
        // The plain cell font again, not the headings' bold.
        painter.setFont(cell_font);
        int index = 1;
        for (const auto& [label, color, cells] : rows) {
            baseline = y + index * row_height + cell_metrics.ascent() + export_row_padding / 2;
            x = export_margin;
            // The square ties the row to its line on the chart above.
            painter.fillRect(x + export_gap, baseline - export_swatch, export_swatch, export_swatch, color);
            painter.setPen(QColor(QStringLiteral("#111111")));
            painter.drawText(x + export_gap + export_swatch + export_gap, baseline, label);
            x += widths[0];
            for (std::size_t column = 0; column < cells.size(); ++column) {
                painter.drawText(x + export_gap, baseline, cells[column]);
                x += widths[column + 1];
            }
            ++index;
        }
    }
    painter.end();
    return image;
}

}  // namespace vml::app
