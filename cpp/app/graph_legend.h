// What Worst Frames' graph shows, opened by the ⓘ beside its save button: a
// row for each mark it draws (the scores' line, a worst frame's dot, a weak
// segment's shading, the segment selected, the frame selected), each with
// a swatch drawn in the colours the graph draws it in now, so that it reads
// the same in every look.
#pragma once

#include <QColor>
#include <QString>
#include <QStringList>

#include "popup_panel.h"

class QWidget;

namespace vml::app {

class GraphLegend : public PopupPanel {
    Q_OBJECT

public:
    // `metric`: the metric shown, by its label; `lower_is_better`: its graph
    // drawn upside down (said in a last row); `per_second`: scored each
    // second (CVVDP), not each frame.
    GraphLegend(const QString& metric, bool lower_is_better, bool per_second, QWidget* parent = nullptr);

    // Opens it under `anchor`, its right edge on the anchor's, kept on screen.
    void open_under(QWidget* anchor);

    // The rows' texts, in order, for the tests.
    QStringList texts() const;
};

}  // namespace vml::app
