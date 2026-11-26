// The order every list of metrics in the window shows them in: the Videos
// table's columns, Metric Graphs' tabs and readouts, Worst Frames, Video
// Compare, BD-rate, Saved results, notifications and Settings' notification
// fields and weak-segment conditions. The registry's order (Add/remove
// metrics') until the user drags a column of the Videos table to another
// place (Settings::metric_order); the window keeps one order for all of
// them. Exported files keep the registry's order, as the command line
// writes them, for the scripts that read them.
#pragma once

#include <QObject>

#include <string>
#include <vector>

#include "vml/metrics.h"

namespace vml::app {

// Every metric in the order shown (ordered_metrics of the order set), each
// plane after its metric. For the window's thread.
const std::vector<const MetricDefinition*>& shown_metrics();

// `keys` in the order shown; keys of no metric last, in the order they were.
std::vector<std::string> in_order_shown(std::vector<std::string> keys);

// The order the user set: keys of metrics that are not planes, empty for
// the registry's. Tells metric_order_signal() when the order changes.
void set_metric_order(const std::vector<std::string>& order);

// Sends changed() when the order shown changes, for each list to put its
// metrics in the new order.
class MetricOrderSignal : public QObject {
    Q_OBJECT

signals:
    void changed();
};
MetricOrderSignal& metric_order_signal();

}  // namespace vml::app
