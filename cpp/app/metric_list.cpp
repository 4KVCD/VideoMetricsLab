#include "metric_list.h"

#include <QCheckBox>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>
#include <array>
#include <vector>

#include "i18n.h"
#include "theme.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

constexpr std::array<MetricSection, 2> sections{MetricSection::general_purpose, MetricSection::specialized};

long long listed_count() {
    return static_cast<long long>(metrics_in(MetricSection::general_purpose).size() +
                                  metrics_in(MetricSection::specialized).size());
}

// A muted line of text that wraps.
QLabel* muted_line(const QString& text) {
    auto* label = new QLabel(text);
    label->setWordWrap(true);
    theme::style(label, "color: {muted};");
    return label;
}

}  // namespace

QString metric_tip(const std::string& key) {
    const MetricDefinition& metric = metric_definition(key);
    const MetricGuide guide = metric_guide(key);
    const QString measures = app::tr(std::string(guide.measures).c_str());
    if (metric.is_plane()) {
        return app::tr("{name}: the {plane} plane alone, calculated with {metric}.",
                       {{"name", qt_text(metric.label)},
                        {"plane", qt_text(metric.table_header)},
                        {"metric", qt_text(metric_definition(metric.plane_of).label)}}) +
               "\n\n" + measures;
    }
    if (guide.ranges.empty()) {
        return measures;
    }
    return measures + "\n\n" + app::tr(std::string(guide.ranges).c_str());
}

QString shown_count_text(long long shown, long long total) {
    return app::ntr("{shown} of {count} shown", "{shown} of {count} shown", total, {{"shown", shown}});
}

std::set<std::string> section_ticks(const std::set<std::string>& ticked, MetricSection section, bool all,
                                    std::size_t keep) {
    std::set<std::string> result = ticked;
    const std::vector<std::string_view> keys = metrics_in(section);
    for (const std::string_view key : keys) {
        if (all) {
            result.emplace(key);
        } else {
            result.erase(std::string(key));
        }
    }
    // None: the section's first ticked metrics stay, as many as the other
    // section leaves short of `keep`.
    for (const std::string_view key : keys) {
        if (result.size() >= keep) {
            break;
        }
        if (ticked.contains(std::string(key))) {
            result.emplace(key);
        }
    }
    return result;
}

// ------------------------------------------------------------------ the list

MetricList::MetricList(Place place, QWidget* parent) : QWidget(parent), place_(place) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    if (place_ == Place::panel) {
        auto* head = new QHBoxLayout;
        auto* title = new QLabel(app::tr("Metrics"));
        QFont bold = title->font();
        bold.setBold(true);
        title->setFont(bold);
        head->addWidget(title);
        head->addStretch(1);
        count_ = new QLabel;
        theme::style(count_, "color: {muted};");
        head->addWidget(count_);
        layout->addLayout(head);
        layout->addWidget(muted_line(app::tr("A metric ticked here is shown as a column; each row calculates the "
                                             "metrics shown that are ticked in its own cells. Hover one for what it "
                                             "measures.")));
        for (const MetricSection section : sections) {
            layout->addWidget(build_section(section));
        }
    } else {
        layout->addWidget(muted_line(app::tr("What new videos start with ticked. Hover one for what it measures.")));
        auto* columns = new QHBoxLayout;
        columns->setSpacing(24);
        for (const MetricSection section : sections) {
            QWidget* column = build_section(section);
            columns->addWidget(column, 1, Qt::AlignTop);
        }
        layout->addLayout(columns);
    }
    // The line on grain and clean content: one number does not mean the same
    // quality in every video.
    auto* rule = new QFrame;
    rule->setFrameShape(QFrame::HLine);
    rule->setFrameShadow(QFrame::Sunken);
    layout->addWidget(rule);
    layout->addWidget(muted_line(app::tr("Scores depend on the content: grainy film scores lower than clean CGI or "
                                         "animation at the same visible quality. Compare encodes of the same video, "
                                         "not videos with each other.")));
    // The two headings of one height, so that side by side they line up.
    general_head_->setMinimumHeight(fold_->sizeHint().height());
    // In one column, the sections' few words line up.
    if (place_ == Place::panel) {
        int names = 0;
        for (const auto& [key, box] : boxes_) {
            names = std::max(names, box->sizeHint().width());
        }
        for (QGridLayout* grid : grids_) {
            grid->setColumnMinimumWidth(0, names);
        }
    }
    // Open in both places, so that every metric is in sight: folded, the
    // specialized ones (MS-SSIM, PSNR-HVS-M, CAMBI, CIEDE2000...) were found
    // only by those who knew to unfold them (#251). Folding still works.
    set_specialized_open(true);
    refresh();
}

QWidget* MetricList::build_section(MetricSection section) {
    auto* widget = new QWidget;
    auto* layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);
    const std::vector<std::string_view> keys = metrics_in(section);
    const bool specialized = section == MetricSection::specialized;

    auto* head = new QHBoxLayout;
    head->setSpacing(4);
    QFont bold = font();
    bold.setBold(true);
    if (specialized) {
        // The heading folds and opens the section: its arrow and its text.
        fold_ = new QToolButton;
        fold_->setAutoRaise(true);
        fold_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        fold_->setFont(bold);
        fold_->setToolTip(app::tr("Show or fold away the specialized metrics."));
        connect(fold_, &QToolButton::clicked, this, [this] { set_specialized_open(!specialized_open()); });
        head->addWidget(fold_);
    } else {
        auto* title =
            new QLabel(app::tr("General purpose ({count})", {{"count", static_cast<long long>(keys.size())}}));
        title->setFont(bold);
        general_head_ = title;
        head->addWidget(title);
    }
    head->addStretch(1);
    // All · None, as links: they tick or untick the section's metrics.
    auto* links = new QLabel(QStringLiteral("<a href=\"all\">%1</a> · <a href=\"none\">%2</a>")
                                 .arg(app::tr("All").toHtmlEscaped(), app::tr("None").toHtmlEscaped()));
    links->setTextInteractionFlags(Qt::LinksAccessibleByMouse | Qt::LinksAccessibleByKeyboard);
    connect(links, &QLabel::linkActivated, this,
            [this, section](const QString& link) { on_section_link(section, link == QLatin1String("all")); });
    head->addWidget(links);
    layout->addLayout(head);

    if (specialized) {
        auto* about = muted_line(app::tr("Codec-comparison weightings, color and banding: for comparing with "
                                         "published results, or a specific check."));
        layout->addWidget(about);
    }

    auto* body = new QWidget;
    auto* grid = new QGridLayout(body);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(16);
    grid->setVerticalSpacing(2);
    int row = 0;
    for (const std::string_view key : keys) {
        const std::string name(key);
        const QString tip = metric_tip(name);
        auto* box = new QCheckBox(qt_text(metric_definition(key).label));
        box->setToolTip(tip);
        connect(box, &QCheckBox::clicked, this, [this] {
            refresh();
            emit ticks_changed();
        });
        auto* brief = new QLabel(app::tr(std::string(metric_guide(key).brief).c_str()));
        theme::style(brief, "color: {muted};");
        brief->setToolTip(tip);
        grid->addWidget(box, row, 0);
        grid->addWidget(brief, row, 1);
        boxes_.emplace(name, box);
        ++row;
    }
    grid->setColumnStretch(1, 1);
    grids_.push_back(grid);
    layout->addWidget(body);
    if (specialized) {
        specialized_body_ = body;
        if (place_ == Place::panel) {
            // The planes of the YUV metrics: shown with their metric, so
            // under the five that have them, with no tick of their own.
            planes_box_ = new QCheckBox(app::tr("Y, U and V of the YUV metrics"));
            planes_box_->setToolTip(app::tr(
                "A column for each of the Y, U and V planes beside PSNR YUV avg. dB, SSIM YUV, MS-SSIM YUV, PSNR-HVS-M "
                "and "
                "VMAF YUV, "
                "calculated and saved with them at no extra cost, shown or not. Videos scored before version 2.7 while "
                "this was off need Calculate again to fill them."));
            connect(planes_box_, &QCheckBox::clicked, this, [this](bool shown) { emit planes_toggled(shown); });
            grid->addWidget(planes_box_, row, 0, 1, 2, Qt::AlignBottom);
            grid->setRowMinimumHeight(row, planes_box_->sizeHint().height() + 6);
        }
    }
    return widget;
}

std::set<std::string> MetricList::ticked() const {
    std::set<std::string> keys;
    for (const auto& [key, box] : boxes_) {
        if (box->isChecked()) {
            keys.insert(key);
        }
    }
    return keys;
}

void MetricList::set_ticked(const std::set<std::string>& keys) {
    for (const auto& [key, box] : boxes_) {
        box->setChecked(keys.contains(key));
    }
    refresh();
}

void MetricList::set_ticked(const std::string& key, bool ticked) {
    if (QCheckBox* found = box(key)) {
        found->setChecked(ticked);
        refresh();
    }
}

bool MetricList::specialized_open() const { return specialized_body_->isVisibleTo(this); }

void MetricList::set_specialized_open(bool open) {
    specialized_body_->setVisible(open);
    fold_->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
    refresh();
}

void MetricList::set_planes_shown(bool shown) {
    if (planes_box_ != nullptr) {
        const QSignalBlocker blocker(planes_box_);
        planes_box_->setChecked(shown);
    }
}

QCheckBox* MetricList::box(const std::string& key) const {
    const auto found = boxes_.find(key);
    return found == boxes_.end() ? nullptr : found->second;
}

void MetricList::on_section_link(MetricSection section, bool all) {
    const std::set<std::string> before = ticked();
    const std::set<std::string> after = section_ticks(before, section, all, place_ == Place::panel ? 1 : 0);
    if (after == before) {
        return;
    }
    set_ticked(after);
    emit ticks_changed();
}

void MetricList::refresh() {
    const std::set<std::string> keys = ticked();
    if (count_ != nullptr) {
        count_->setText(shown_count_text(static_cast<long long>(keys.size()), listed_count()));
    }
    if (fold_ != nullptr) {
        const std::vector<std::string_view> specialized = metrics_in(MetricSection::specialized);
        QString head = app::tr("Specialized ({count})", {{"count", static_cast<long long>(specialized.size())}});
        if (!specialized_open()) {
            // Folded: how many of them are ticked, which cannot be seen.
            const long long ticked_here = std::count_if(specialized.begin(), specialized.end(), [&keys](auto key) {
                return keys.contains(std::string(key));
            });
            head += QStringLiteral(" · ") +
                    (place_ == Place::panel
                         ? app::ntr("{count} shown", "{count} shown", ticked_here)
                         : app::ntr("{count} ticked", "{count} ticked", ticked_here));
        }
        fold_->setText(head);
    }
    if (place_ == Place::panel) {
        // The last metric shown cannot be hidden: a table with no metrics
        // could calculate nothing.
        for (const auto& [key, box] : boxes_) {
            const bool last = keys.size() == 1 && box->isChecked();
            box->setEnabled(!last);
            box->setToolTip(last ? app::tr("At least one metric must stay shown.") : metric_tip(key));
        }
    }
}

// ----------------------------------------------------------------- the panel

namespace {

// The panel's scroll area: as large as the list, up to `room` high, then
// scrolled. QScrollArea's own size hint is at most 24 lines of text high,
// which would scroll a list that fits.
class FittingScroll : public QScrollArea {
public:
    explicit FittingScroll(std::function<int()> room) : room_(std::move(room)) {}
    QSize sizeHint() const override {
        // The list's own size, its minimum width included (72 characters):
        // narrower, the panel cut its right side off.
        const QWidget* list = widget();
        QSize hint = list != nullptr ? list->sizeHint().expandedTo(list->minimumSize()) : QSize();
        const int room = room_();
        if (room > 0 && hint.height() > room) {
            hint.setHeight(room);
            hint.rwidth() += verticalScrollBar()->sizeHint().width();  // beside the list, not over it
        }
        return hint + QSize(2 * frameWidth(), 2 * frameWidth());
    }
    QSize minimumSizeHint() const override { return sizeHint(); }

private:
    std::function<int()> room_;
};

}  // namespace

MetricsPanel::MetricsPanel(QWidget* parent) : PopupPanel(parent) {
    setAttribute(Qt::WA_DeleteOnClose);
    setFrameShape(QFrame::StyledPanel);
    setFrameShadow(QFrame::Raised);
    setAutoFillBackground(true);
    theme::set_role(this, "card");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(12, 10, 12, 10);
    // The popup is always its content's size: it grows as Specialized opens
    // and shrinks back as it folds. (An adjustSize as it folded came before
    // the layout had taken the change in, and kept the open height, the room
    // spread as gaps between the lines.)
    layout->setSizeConstraint(QLayout::SetFixedSize);
    list_ = new MetricList(MetricList::Place::panel);
    // Wide enough for each metric's few words on one line, and its lines of
    // text on two or three.
    list_->setMinimumWidth(list_->fontMetrics().averageCharWidth() * 72);
    // Scrolled where it is taller than its screen's room: with Specialized
    // open (#251), the panel is about 610 pixels at 100 %.
    scroll_ = new FittingScroll([this] { return room_; });
    scroll_->setFrameShape(QFrame::NoFrame);
    scroll_->setWidgetResizable(true);
    scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll_->setWidget(list_);
    scroll_->viewport()->setAutoFillBackground(false);  // the card's own background
    list_->setAutoFillBackground(false);
    list_->installEventFilter(this);  // its layout changed (Specialized folded): the panel sized again
    layout->addWidget(scroll_);
    // The order every list shows the metrics in: changed by dragging a
    // metric's heading in the Videos table, put back here.
    order_row_ = new QWidget;
    auto* order = new QHBoxLayout(order_row_);
    order->setContentsMargins(0, 0, 0, 0);
    auto* how = new QLabel(app::tr("Drag a heading in the Videos table to move its metric in every list."));
    how->setWordWrap(true);
    theme::set_role(how, "hint");
    order->addWidget(how, 1);
    default_order_ = new QPushButton(app::tr("Default order"));
    default_order_->setToolTip(app::tr("The metrics in the order this list shows them, in every list."));
    order->addWidget(default_order_, 0, Qt::AlignVCenter);
    layout->addWidget(order_row_);
}

void MetricsPanel::fit_height(int height) {
    // The list's room: the panel's less its margins and frame.
    const QMargins margins = layout()->contentsMargins();
    const int below = order_row_->sizeHint().height() + layout()->spacing();  // the order's row under the list
    room_ = height > 0 ? std::max(1, height - margins.top() - margins.bottom() - 2 * frameWidth() - below) : 0;
    scroll_->updateGeometry();
    adjustSize();
}

bool MetricsPanel::eventFilter(QObject* watched, QEvent* event) {
    if (watched == list_ && event->type() == QEvent::LayoutRequest) {
        scroll_->updateGeometry();  // the list's size changed: the scroll area's hint with it
    }
    return PopupPanel::eventFilter(watched, event);
}

void MetricsPanel::show_at(const QPoint& top_left, std::optional<QPoint> top_right) {
    // The panel as high as the screen has room for.
    if (const QScreen* screen = QGuiApplication::screenAt(top_right.value_or(top_left)); screen != nullptr) {
        fit_height(screen->availableGeometry().height());
    }
    adjustSize();
    QPoint place = top_right ? *top_right - QPoint(width(), 0) : top_left;
    if (const QScreen* screen = QGuiApplication::screenAt(top_right.value_or(top_left)); screen != nullptr) {
        const QRect room = screen->availableGeometry();
        place.setX(std::clamp(place.x(), room.left(), std::max(room.left(), room.right() - width())));
        place.setY(std::clamp(place.y(), room.top(), std::max(room.top(), room.bottom() - height())));
    }
    move(place);
    show();
}

}  // namespace vml::app
