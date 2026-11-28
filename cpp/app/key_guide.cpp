#include "key_guide.h"

#include <QEvent>
#include <QFontMetrics>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QScopedValueRollback>

#include <algorithm>
#include <array>
#include <cmath>

#include "i18n.h"
#include "key_bindings.h"

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

std::vector<KeyRow> video_compare_keys() {
    const auto cap = [](const QKeySequence& keys) { return key_cap_text(keys); };
    std::vector<KeyRow> rows;
    if (const std::vector<QKeySequence> source = keys_for("show_source"); !source.empty()) {
        rows.push_back({{cap(source.front())}, app::tr("Hold to see the source")});
    }
    const std::vector<QKeySequence> previous = keys_for("previous_video");
    const std::vector<QKeySequence> next = keys_for("next_video");
    if (!previous.empty() && !next.empty()) {
        rows.push_back({{cap(previous.front()), cap(next.front())}, app::tr("Switch test video")});
    }
    if (const std::vector<QKeySequence> play = keys_for("play_pause"); !play.empty()) {
        rows.push_back({{cap(play.front())}, app::tr("Play / pause")});
    }
    return rows;
}

KeyGuide::KeyGuide(QWidget* parent) : QFrame(parent) {
    setObjectName(QStringLiteral("keyGuide"));
    grid_ = new QGridLayout(this);
    build_rows();
    set_scale(scales.front());
    show_look();
    connect(key_bindings_signal(), &KeyBindingsSignal::changed, this, [this] {
        build_rows();
        set_scale(scale_ != 0 ? scale_ : scales.front());
        show_look();
        if (room_.isValid()) {
            scale_ = 0;  // fitted again to the room it has
            fit(room_);
        }
    });
}

void KeyGuide::build_rows() {
    // Clearing a layout deletes only its own items: each row's caps are a
    // layout of their own, emptied too.
    while (QLayoutItem* item = grid_->takeAt(0)) {
        if (QLayout* inner = item->layout()) {
            while (QLayoutItem* cap = inner->takeAt(0)) {
                delete cap->widget();
                delete cap;
            }
        }
        delete item->widget();
        delete item;
    }
    keys_.clear();
    actions_.clear();
    int row = 0;
    for (const KeyRow& keys : video_compare_keys()) {
        add_row(row++, keys.keys, keys.action);
    }
}

void KeyGuide::set_themed(bool themed) {
    if (themed != themed_) {
        themed_ = themed;
        show_look();
    }
}

void KeyGuide::show_look() {
    // A style sheet set here changes the palette too: that change is this
    // one's, not the theme's (changeEvent), and would come back here forever.
    const QScopedValueRollback restyling(restyling_, true);
    // On the video's black screen: light on black. In the theme's colours:
    // the palette's (re-read as it changes, changeEvent), so that the guide
    // looks like the window's other controls in each interface and theme.
    if (themed_) {
        setStyleSheet(QStringLiteral("#keyGuide { background: palette(base); border: 1px solid palette(mid); "
                                     "border-radius: 12px; }"));
    } else {
        setStyleSheet(QStringLiteral("#keyGuide { background: #0a0a0a; border: 1px solid #555; border-radius: 12px; }"));
    }
    const QString cap = themed_ ? QStringLiteral("color: palette(button-text); background: palette(button); "
                                                 "border: 2px solid palette(mid); ")
                                : QStringLiteral("color: #fff; background: #262626; border: 2px solid #8a8a8a; ");
    for (QLabel* key : keys_) {
        key->setStyleSheet(cap + QStringLiteral("border-bottom-width: 4px; border-radius: 8px; padding: 0 8px;"));
    }
    for (QLabel* action : actions_) {
        action->setStyleSheet(themed_ ? QStringLiteral("background: transparent;")
                                      : QStringLiteral("color: #eee; background: transparent;"));
    }
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
        caps->addWidget(cap);
        keys_.push_back(cap);
    }
    grid_->addLayout(caps, row, 0);
    auto* label = new QLabel(action);
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
    } else if (event->type() == QEvent::PaletteChange && themed_ && !restyling_) {
        show_look();  // the theme changed: its colours again
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
