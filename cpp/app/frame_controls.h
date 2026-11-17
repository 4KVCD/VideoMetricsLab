// The frame controls under a graph of frames, shared by Metric Graphs and the
// Bitrate Viewer: Go to frame, and a frame range (issue #17) in two boxes of
// its own, "Highlight range from frame: [ ] to frame: [ ]", with Clear range.
//
// The widget keeps the range and its boxes; the tab it is in turns frames
// into times on its own graph, shades the range and gives its statistics.
#pragma once

#include <QWidget>

#include <optional>
#include <utility>

class QPushButton;
class QSpinBox;

namespace vml::app {

class FrameControls : public QWidget {
    Q_OBJECT

public:
    explicit FrameControls(QWidget* parent = nullptr);

    // The frames chosen, in order (the boxes' two ends either way round), or
    // nothing until both boxes hold a frame: the whole video.
    std::optional<std::pair<int, int>> selected_frames() const;
    // The frames the graph shows, lowest and highest: Go to frame cannot ask
    // for one outside them, and the range boxes go as far, or to their own
    // ends where those lie further out.
    void set_bounds(int lo, int hi);
    // A range dragged across the graph, from frame `first` to `last`;
    // `finished` once the drag ends (range_changed).
    void set_dragged_range(int first, int last, bool finished);
    // No range, the boxes back to "—": clear() says so (range_changed),
    // forget_range() does not (the tab has no videos left).
    void clear();
    void forget_range();

    // The boxes and the button, for the tests.
    QSpinBox* frame_spin = nullptr;
    QSpinBox* range_start_spin = nullptr;
    QSpinBox* range_end_spin = nullptr;
    QPushButton* clear_range_button = nullptr;

signals:
    // Go to frame: the frame asked for.
    void frame_requested(int frame);
    // The range changed (selected_frames); `finished` is false while a drag
    // lasts, when a tab shades it but leaves its statistics until it ends.
    void range_changed(bool finished);

private:
    // A range box edited: the range is its two boxes' frames once both hold one.
    void on_range_box_changed();
    // Both range boxes back to "—" (no end), their signals held.
    void reset_range_boxes();
    // How refresh_range_boxes sets the boxes' bounds: exactly the graph's
    // frames and the range's ends, or, after a box's own step, left as they
    // are where they hold those already.
    enum class BoxBounds { exact, keep_wider };
    // The range boxes bounded by the graph's frames and the range's ends.
    void refresh_range_boxes(BoxBounds bounds = BoxBounds::exact);

    // Nothing (no range, the whole video) until both boxes hold a frame.
    // Each box's lowest value, shown as "—", is no end.
    std::optional<std::pair<int, int>> range_;
    int lo_ = 0;
    int hi_ = 0;
};

}  // namespace vml::app
