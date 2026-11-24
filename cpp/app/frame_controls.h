// The frame controls under a graph of frames, shared by Metric Graphs and the
// Bitrate Viewer: Go to frame, and a frame range (issue #17) in two boxes of
// its own, "Highlight range from frame: [ ] to frame: [ ]", with Clear range.
// Where the tab gives its timeline (set_timeline: Metric Graphs), the same
// in times under them, to the millisecond: "Go to time: [ ]" and "Highlight
// range from time: [ ] to time: [ ]", kept in step with the frames.
//
// The widget keeps the range and its boxes; the tab it is in turns frames
// into times on its own graph, shades the range and gives its statistics.
#pragma once

#include <QString>
#include <QWidget>

#include <functional>
#include <optional>
#include <utility>
#include <vector>

class QLineEdit;
class QPushButton;
class QSpinBox;

namespace vml::app {

class FrameControls : public QWidget {
    Q_OBJECT

public:
    explicit FrameControls(QWidget* parent = nullptr);

    // A graph's frames in time, for the time boxes: a frame's time, and the
    // frame at a time, the first at it or after (`at_or_after`) or the last
    // at it or before. A time typed to the millisecond is a frame's when it
    // is within half a millisecond of it, as its time is shown.
    struct Timeline {
        std::function<std::optional<double>(int frame)> time_of;
        std::function<std::optional<int>(double seconds, bool at_or_after)> frame_at;
    };
    // Shows the time boxes, using `timeline` from then on.
    void set_timeline(Timeline timeline);

    // A time typed in a time box, in seconds: [[h:]m:]s[.fff] ("1:02.5",
    // "0:01:02.500", "62.5"; a comma for the point too); nothing for other
    // text, or none typed.
    static std::optional<double> parse_time(const QString& text);
    // A time as the boxes show it, to the millisecond: 0:01:02.500.
    static QString time_text(double seconds);

    // The frames chosen, in order (the boxes' two ends either way round), or
    // nothing until both boxes hold a frame: the whole video. Nothing while
    // the range is in seconds.
    std::optional<std::pair<int, int>> selected_frames() const;
    // The range in seconds instead of frames, for lines whose frames are not
    // the same times (videos of different frame rates, where frame 1000 is a
    // moment of its own in each): the frame range boxes greyed with `why` as
    // their tooltip, the time boxes and a drag (set_dragged_seconds) giving
    // the range. A change of mode clears the range.
    void set_range_in_seconds(bool on, const QString& why = {});
    bool range_in_seconds() const { return in_seconds_; }
    // The range in seconds chosen, in order, while range_in_seconds; nothing
    // until both time boxes hold a time.
    std::optional<std::pair<double, double>> selected_seconds() const;
    // A range dragged across the graph, in seconds (range_in_seconds).
    void set_dragged_seconds(double first, double last, bool finished);
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
    // The time boxes, hidden until set_timeline.
    QLineEdit* time_edit = nullptr;
    QPushButton* time_go_button = nullptr;
    QLineEdit* range_start_time = nullptr;
    QLineEdit* range_end_time = nullptr;

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
    // Go to time: the frame shown at the time typed (the last at it or
    // before, else the first after it).
    void on_time_requested();
    // A range time box edited: the frames of its two times, from the first
    // at the first time or after to the last at the second or before, once
    // both hold a time.
    void on_range_time_edited();
    // The time boxes showing the range's frames' times, and Go to time the
    // frame's, as the frames are chosen another way.
    void refresh_range_times();
    void show_frame_time(int frame);

    // Nothing (no range, the whole video) until both boxes hold a frame.
    // Each box's lowest value, shown as "—", is no end.
    std::optional<std::pair<int, int>> range_;
    // The range while range_in_seconds, as range_ is otherwise.
    bool in_seconds_ = false;
    std::optional<std::pair<double, double>> seconds_;
    QString frame_range_tip_;  // the frame range boxes' own tooltip, given back after seconds
    int lo_ = 0;
    int hi_ = 0;
    std::optional<Timeline> timeline_;
    std::vector<QWidget*> time_row_;  // the time boxes and their labels
};

}  // namespace vml::app
