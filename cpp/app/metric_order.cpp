#include "metric_order.h"

#include <algorithm>

namespace vml::app {
namespace {

std::vector<const MetricDefinition*>& order_shown() {
    static std::vector<const MetricDefinition*> shown = ordered_metrics({});
    return shown;
}

}  // namespace

const std::vector<const MetricDefinition*>& shown_metrics() { return order_shown(); }

std::vector<std::string> in_order_shown(std::vector<std::string> keys) {
    const std::vector<const MetricDefinition*>& shown = shown_metrics();
    const auto place = [&shown](const std::string& key) {
        const auto found = std::find_if(shown.begin(), shown.end(),
                                        [&key](const MetricDefinition* metric) { return metric->key == key; });
        return found - shown.begin();  // shown.size() for a key of no metric
    };
    std::stable_sort(keys.begin(), keys.end(),
                     [&place](const std::string& a, const std::string& b) { return place(a) < place(b); });
    return keys;
}

void set_metric_order(const std::vector<std::string>& order) {
    std::vector<const MetricDefinition*> shown = ordered_metrics(order);
    if (shown == order_shown()) {
        return;
    }
    order_shown() = std::move(shown);
    emit metric_order_signal().changed();
}

MetricOrderSignal& metric_order_signal() {
    static MetricOrderSignal signal;
    return signal;
}

}  // namespace vml::app
