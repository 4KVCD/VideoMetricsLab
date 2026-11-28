// Video Compare's keys in large type, on its black screen under the message
// (FrameView, frame_compare_panel.h): each key drawn as a key, next to what
// it does.
#pragma once

#include <QFrame>
#include <QSize>
#include <QString>

#include <vector>

class QGridLayout;
class QLabel;

namespace vml::app {

// One of Video Compare's keys (or a pair, ← →) and what it does.
struct KeyRow {
    std::vector<QString> keys;
    QString action;
};

// Video Compare's main keys as its guide shows them, from the keys in use
// (key_bindings.h): the source, the test video, play and pause. An action
// with no key is left out.
std::vector<KeyRow> video_compare_keys();

class KeyGuide : public QFrame {
    Q_OBJECT

public:
    explicit KeyGuide(QWidget* parent = nullptr);
    // Takes the largest of its text sizes at which it fits in `room`.
    void fit(QSize room);
    // In the theme's colours (the tab's screen with nothing to compare), or
    // light on black (the default: on the video's black screen).
    void set_themed(bool themed);

protected:
    void changeEvent(QEvent* event) override;

private:
    // The rows from the keys in use (again, when Settings changes them).
    void build_rows();
    void add_row(int row, const std::vector<QString>& keys, const QString& action);
    void set_scale(double scale);
    void show_look();

    QGridLayout* grid_ = nullptr;
    std::vector<QLabel*> keys_;
    std::vector<QLabel*> actions_;
    QSize room_;
    double scale_ = 0;  // the text's size, as a multiple of the window's
    bool themed_ = false;
    bool restyling_ = false;  // in show_look: its own palette changes
};

}  // namespace vml::app
