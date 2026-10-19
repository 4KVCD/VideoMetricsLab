// Video Compare's keys in large type, on its black screen under the message
// (FrameView, frame_compare_panel.h): each key drawn as a key, next to what
// it does.
#pragma once

#include <QFrame>
#include <QSize>

#include <vector>

class QGridLayout;
class QLabel;

namespace vml::app {

class KeyGuide : public QFrame {
    Q_OBJECT

public:
    explicit KeyGuide(QWidget* parent = nullptr);
    // Takes the largest of its text sizes at which it fits in `room`.
    void fit(QSize room);

protected:
    void changeEvent(QEvent* event) override;

private:
    void add_row(int row, const std::vector<QString>& keys, const QString& action);
    void set_scale(double scale);

    QGridLayout* grid_ = nullptr;
    std::vector<QLabel*> keys_;
    std::vector<QLabel*> actions_;
    QSize room_;
    double scale_ = 0;  // the text's size, as a multiple of the window's
};

}  // namespace vml::app
