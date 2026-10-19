#include "key_guide.h"

#include <QEvent>
#include <QFontMetrics>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>

#include <algorithm>
#include <array>
#include <cmath>

#include "i18n.h"

namespace vml::app {

namespace {

// The text sizes tried, largest first, as multiples of the window's font.
constexpr std::array scales{1.8, 1.5, 1.25, 1.0, 0.8};

QFont scaled(const QFont& font, double factor) {
    QFont result(font);
    if (font.pointSizeF() > 0) {
        result.setPointSizeF(font.pointSizeF() * factor);
    } else {
        result.setPixelSize(std::max(1, static_cast<int>(std::lround(font.pixelSize() * factor))));
    }
    return result;
}

}  // namespace

KeyGuide::KeyGuide(QWidget* parent) : QFrame(parent) {
    setObjectName(QStringLiteral("keyGuide"));
    setStyleSheet(QStringLiteral("#keyGuide { background: #0a0a0a; border: 1px solid #555; border-radius: 12px; }"));
    grid_ = new QGridLayout(this);
    add_row(0, {QStringLiteral("S")}, app::tr("Hold to see the source"));
    add_row(1, {QString::fromUtf8("\xE2\x86\x90"), QString::fromUtf8("\xE2\x86\x92")},  // ← →
            app::tr("Switch test video"));
    add_row(2, {app::tr("Space")}, app::tr("Play / pause"));
    set_scale(scales.front());
}

// A row: its keys, drawn as keys, next to what they do.
void KeyGuide::add_row(int row, const std::vector<QString>& keys, const QString& action) {
    auto* caps = new QHBoxLayout;
    caps->setSpacing(8);
    caps->addStretch(1);  // the keys next to what they do, in either direction
    // In the keyboard's order whichever way the text runs: ← left of →.
    std::vector<QString> in_order = keys;
    if (QGuiApplication::isRightToLeft()) {
        std::reverse(in_order.begin(), in_order.end());
    }
    for (const QString& key : in_order) {
        auto* cap = new QLabel(key);
        cap->setAlignment(Qt::AlignCenter);
        cap->setStyleSheet(QStringLiteral("color: #fff; background: #262626; border: 2px solid #8a8a8a; "
                                          "border-bottom-width: 4px; border-radius: 8px; padding: 0 8px;"));
        caps->addWidget(cap);
        keys_.push_back(cap);
    }
    grid_->addLayout(caps, row, 0);
    auto* label = new QLabel(action);
    label->setStyleSheet(QStringLiteral("color: #eee; background: transparent;"));
    grid_->addWidget(label, row, 1);
    actions_.push_back(label);
}

void KeyGuide::fit(QSize room) {
    if (room == room_ && scale_ != 0) {
        return;  // as fitted: the messages change as the frames are stepped through
    }
    room_ = room;
    for (const double scale : scales) {
        if (scale != scale_) {
            set_scale(scale);
        }
        const QSize wanted = sizeHint();
        if (wanted.width() <= room.width() && wanted.height() <= room.height()) {
            return;
        }
    }
}

void KeyGuide::changeEvent(QEvent* event) {
    QFrame::changeEvent(event);
    if (event->type() == QEvent::FontChange) {
        scale_ = 0;  // the window's text size changed: every size again
        fit(room_);
    }
}

void KeyGuide::set_scale(double scale) {
    scale_ = scale;
    QFont key_font = scaled(font(), scale);
    key_font.setWeight(QFont::Medium);
    const QFont action_font = scaled(font(), scale);
    // The keys square, and Space three keys wide; the gaps in step with the
    // text.
    const int side = static_cast<int>(std::lround(QFontMetrics(key_font).height() * 1.4)) + 6;
    for (QLabel* key : keys_) {
        key->setFont(key_font);
        key->setMinimumSize(key->text().size() > 1 ? 3 * side : side, side);
    }
    for (QLabel* action : actions_) {
        action->setFont(action_font);
    }
    const int line = QFontMetrics(action_font).height();
    grid_->setHorizontalSpacing(line * 3 / 4);
    grid_->setVerticalSpacing(line / 2);
    grid_->setContentsMargins(line * 6 / 5, line * 4 / 5, line * 6 / 5, line * 4 / 5);
    grid_->activate();
}

}  // namespace vml::app
