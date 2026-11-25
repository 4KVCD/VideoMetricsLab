#include "graph_legend.h"

#include <QGridLayout>
#include <QLabel>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QScreen>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>

#include "i18n.h"
#include "theme.h"

namespace vml::app {
namespace {

constexpr int swatch_width = 32;
constexpr int swatch_height = 16;

// A swatch on the graph's background, drawn by `draw` at `ratio`.
QPixmap swatch(qreal ratio, const QColor& background, const std::function<void(QPainter&)>& draw) {
    QPixmap pixmap(QSize(swatch_width, swatch_height) * ratio);
    pixmap.setDevicePixelRatio(ratio);
    pixmap.fill(background);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    draw(painter);
    return pixmap;
}

// The scores' line, as the graph draws a stretch of it.
void draw_line(QPainter& painter, const QColor& colour) {
    painter.setPen(QPen(colour, 1.5));
    const QPointF points[] = {{2, 11}, {9, 5}, {15, 9}, {22, 4}, {30, 7}};
    painter.drawPolyline(points, static_cast<int>(std::size(points)));
}

}  // namespace

GraphLegend::GraphLegend(const QString& metric, bool lower_is_better, bool per_second, QWidget* parent)
    : PopupPanel(parent) {
    setObjectName(QStringLiteral("graphLegend"));
    setAttribute(Qt::WA_DeleteOnClose);
    setAccessibleName(app::tr("Graph legend"));
    auto* layout = new QVBoxLayout(this);
    auto* title = new QLabel(app::tr("What the graph shows"));
    theme::set_role(title, "title");
    layout->addWidget(title);
    auto* rows = new QGridLayout;
    rows->setHorizontalSpacing(10);
    rows->setVerticalSpacing(6);
    layout->addLayout(rows);

    // The graph's own colours (ChartWidget::draw): its background, the
    // line's link colour, the dots' highlight, the shading's and the
    // cursor's.
    const theme::ChartColours colours = theme::chart_colours();
    const QColor line = palette().color(QPalette::Link);
    const QColor dot = palette().color(QPalette::Highlight);
    const qreal ratio = devicePixelRatioF();
    const auto add = [&](const QPixmap& pixmap, const QString& text) {
        auto* picture = new QLabel;
        picture->setPixmap(pixmap);
        auto* words = new QLabel(text);
        words->setWordWrap(true);
        words->setObjectName(QStringLiteral("legendText"));
        const int row = rows->rowCount();
        rows->addWidget(picture, row, 0, Qt::AlignTop);
        rows->addWidget(words, row, 1);
    };
    add(swatch(ratio, colours.background, [&](QPainter& p) { draw_line(p, line); }),
        per_second ? app::tr("{metric} of each second of the video.", {{"metric", metric}})
                   : app::tr("{metric} of each frame of the video.", {{"metric", metric}}));
    add(swatch(ratio, colours.background,
               [&](QPainter& p) {
                   draw_line(p, line);
                   p.setPen(Qt::NoPen);
                   p.setBrush(dot);
                   p.drawEllipse(QPointF(22, 4), 3.0, 3.0);
               }),
        app::tr("A dot: the worst frame of a dip, as Worst frames lists them."));
    add(swatch(ratio, colours.background,
               [&](QPainter& p) {
                   p.fillRect(QRectF(8, 0, 16, swatch_height), colours.highlight);
                   draw_line(p, line);
               }),
        app::tr("Shaded: a weak segment, where the conditions hold."));
    add(swatch(ratio, colours.background,
               [&](QPainter& p) {
                   p.fillRect(QRectF(8, 0, 16, swatch_height), colours.highlight);
                   p.fillRect(QRectF(8, 0, 16, swatch_height), colours.highlight);  // shaded twice, as the graph does
                   draw_line(p, line);
               }),
        app::tr("Shaded twice: the weak segment selected in Weak segments."));
    add(swatch(ratio, colours.background,
               [&](QPainter& p) {
                   draw_line(p, line);
                   p.setPen(QPen(colours.crosshair, 1));
                   p.drawLine(QPointF(16.5, 0), QPointF(16.5, swatch_height));
               }),
        app::tr("A line across: the frame selected in Worst frames, or the one under the pointer."));
    if (lower_is_better) {
        auto* note = new QLabel(
            app::tr("Lower is better for {metric}: its graph is drawn upside down, so worse is still lower.",
                    {{"metric", metric}}));
        note->setWordWrap(true);
        note->setObjectName(QStringLiteral("legendText"));
        theme::style(note, "color: {muted};");
        layout->addWidget(note);
    }
    setMaximumWidth(fontMetrics().averageCharWidth() * 56);  // the lines wrapped, not one long row
}

void GraphLegend::open_under(QWidget* anchor) {
    set_anchor(anchor);
    adjustSize();
    QPoint at = anchor->mapToGlobal(QPoint(anchor->width() - width(), anchor->height() + 2));
    if (const QScreen* screen = anchor->screen()) {
        const QRect room = screen->availableGeometry();
        at.setX(std::clamp(at.x(), room.left(), std::max(room.left(), room.right() - width())));
    }
    move(at);
    show();
}

QStringList GraphLegend::texts() const {
    QStringList texts;
    for (const QLabel* label : findChildren<QLabel*>(QStringLiteral("legendText"))) {
        texts << label->text();
    }
    return texts;
}

}  // namespace vml::app
