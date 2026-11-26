// The metrics in their two sections, General purpose and Specialized
// (metrics_in, from metrics_hidden_at_first), each with a few words beside
// its name and a tooltip of what it measures (metric_guide): Add/remove
// metrics' panel, where a tick shows a column, and Settings > Metrics'
// Default metrics, where it is what new videos start with. One widget, so
// the two places list the metrics alike.
#pragma once

#include <QFrame>
#include <QPoint>
#include <QString>
#include <QWidget>

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "popup_panel.h"
#include "vml/metrics.h"

class QCheckBox;
class QGridLayout;
class QLabel;
class QPushButton;
class QToolButton;

class QEvent;
class QScrollArea;

namespace vml::app {

// A metric's tooltip: what it measures, its unit and which way is better,
// then its broad ranges where they have a source (metric_guide). A plane's
// (a Y, U or V column) says whose plane it is, then its metric's text; it
// has no ranges, as its metric's bands are not its own.
QString metric_tip(const std::string& key);

// Add/remove metrics' count, "9 of 17 shown".
QString shown_count_text(long long shown, long long total);

// The ticks after a section's All or None: every metric of `section`
// ticked, or none. None keeps at least `keep` metrics ticked in all: the
// section's first ticked ones, in the registry's order, where the other
// section has too few.
std::set<std::string> section_ticks(const std::set<std::string>& ticked, MetricSection section, bool all,
                                    std::size_t keep);

class MetricList : public QWidget {
    Q_OBJECT

public:
    // Where the list is: Add/remove metrics' panel (one column, Specialized
    // folded unless one of its metrics is ticked, the planes' switch, at
    // least one metric ticked) or Settings > Metrics (the sections side by
    // side, Specialized open, any number ticked).
    enum class Place { panel, settings };

    explicit MetricList(Place place, QWidget* parent = nullptr);

    // The keys of the metrics ticked.
    std::set<std::string> ticked() const;
    // Ticks these metrics and unticks the others, without ticks_changed.
    void set_ticked(const std::set<std::string>& keys);
    // One metric's tick, without ticks_changed.
    void set_ticked(const std::string& key, bool ticked);

    // Whether Specialized's metrics are shown, not folded away.
    bool specialized_open() const;
    void set_specialized_open(bool open);

    // The panel's switch for the planes' columns (Settings::show_yuv_planes),
    // set without planes_toggled; none in Settings.
    void set_planes_shown(bool shown);
    QCheckBox* planes_box() const { return planes_box_; }

    // A metric's tick box, for tests; nullptr for a key not listed.
    QCheckBox* box(const std::string& key) const;

signals:
    // A metric was ticked or unticked, or a section's All or None clicked.
    void ticks_changed();
    // The planes' switch was clicked.
    void planes_toggled(bool shown);

private:
    QWidget* build_section(MetricSection section);
    void on_section_link(MetricSection section, bool all);
    // The section heads, the count and which boxes can be unticked, after
    // the ticks changed.
    void refresh();

    Place place_;
    std::map<std::string, QCheckBox*> boxes_;  // by metric key
    QLabel* count_ = nullptr;                  // the panel's "9 of 17 shown"
    std::vector<QGridLayout*> grids_;  // the sections' rows
    QLabel* general_head_ = nullptr;
    QToolButton* fold_ = nullptr;      // Specialized's heading, which folds it
    QWidget* specialized_body_ = nullptr;
    QCheckBox* planes_box_ = nullptr;
};

// Add/remove metrics: the MetricList of the panel, in a popup that stays
// open while metrics are ticked and closes on a click outside, a second click
// on its button (set_anchor) or Esc.
class MetricsPanel : public PopupPanel {
    Q_OBJECT

public:
    explicit MetricsPanel(QWidget* parent);
    MetricList* list() const { return list_; }
    // Shows the panel with its top-right corner at `top_right` (a button's
    // bottom right), or its top left at `top_left`, kept on the screen.
    void show_at(const QPoint& top_left, std::optional<QPoint> top_right = std::nullopt);
    // The list's scroll area: it scrolls only where the list is taller than
    // the screen's room (a small screen, a large scaling).
    QScrollArea* scroll() const { return scroll_; }
    // The panel at most `height` high, the list scrolled within it where it
    // is taller (show_at: its screen's available height; 0: no bound).
    void fit_height(int height);
    // Under the list: the metrics put back in the registry's order, after
    // the user dragged the Videos table's columns (Settings::metric_order).
    QPushButton* default_order_button() const { return default_order_; }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    MetricList* list_;
    QScrollArea* scroll_ = nullptr;
    QWidget* order_row_ = nullptr;  // how to order the metrics, and default_order_
    QPushButton* default_order_ = nullptr;
    int room_ = 0;  // the most height the list may take on its screen (show_at); 0: no bound
};

}  // namespace vml::app
