#include "saved_score_details.h"

#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QScreen>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>

#include "i18n.h"
#include "quick_save.h"
#include "theme.h"
#include "window_helpers.h"
#include "vml/metrics.h"
#include "vml/score_summary.h"
#include "vml/number_format.h"

namespace vml::app {

namespace {

// The worst frames' tails, as ScoreStats has them.
struct Tail {
    const char* key;
    const char* share;
    double ScoreStats::*value;
};
constexpr Tail tails[] = {{"worst_10", "10%", &ScoreStats::worst_10_percent},
                          {"worst_5", "5%", &ScoreStats::worst_5_percent},
                          {"worst_1", "1%", &ScoreStats::worst_1_percent},
                          {"worst_0_1", "0.1%", &ScoreStats::worst_0_1_percent}};

std::optional<double ScoreStats::*> stat_of(const std::string& key) {
    if (key == "median") {
        return &ScoreStats::median;
    }
    if (key == "stdev") {
        return &ScoreStats::stdev;
    }
    if (key == "min") {
        return &ScoreStats::minimum;
    }
    if (key == "max") {
        return &ScoreStats::maximum;
    }
    for (const Tail& tail : tails) {
        if (key == tail.key) {
            return tail.value;
        }
    }
    return std::nullopt;
}

QString metric_name(const std::string& key) {
    const MetricDefinition* metric = find_metric(key);
    return metric != nullptr ? qt_text(metric->short_label) : qt_text(key);
}

}  // namespace

const std::vector<std::string>& saved_details_column_keys() {
    return score_summary_column_keys();  // one list, the file's and the panel's
}

QString saved_details_heading(const std::string& key) {
    for (const Tail& tail : tails) {
        if (key == tail.key) {
            return app::tr("Worst {share}", {{"share", tail.share}});
        }
    }
    return key == "median"   ? app::tr("Median")
           : key == "stdev"  ? app::tr("StDev")
           : key == "min"    ? app::tr("Min")
           : key == "max"    ? app::tr("Max")
           : key == "bands"  ? app::tr("Score bands")
           : key == "saved"  ? app::tr("Saved")
           : key == "device" ? app::tr("CPU or GPU")
           : key == "made"   ? app::tr("Made with")
           : key == "model"  ? app::tr("Model or display")
                             : qt_text(key);
}

QString saved_details_tip(const std::string& key) {
    for (const Tail& tail : tails) {
        if (key == tail.key) {
            return app::tr("The score {share} of the frames are worse than: the low end where a higher score is "
                           "better, the high end where a lower one is (Metric Graphs' {share} Low or High).",
                           {{"share", tail.share}});
        }
    }
    return key == "median"   ? app::tr("The middle frame's score.")
           : key == "stdev"  ? app::tr("How far the frames' scores spread from their mean.")
           : key == "min"    ? app::tr("The lowest frame's score.")
           : key == "max"    ? app::tr("The highest frame's score.")
           : key == "bands"  ? app::tr("The share of the frames in each of the metric's score bands, as Metric "
                                       "Graphs' table has them (\u2265 95, < 70, ...).")
           : key == "saved"  ? app::tr("When the score was saved.")
           : key == "device" ? app::tr("Whether the score was calculated on the CPU or the GPU.")
           : key == "made"   ? app::tr("What calculated the score, and its version.")
           : key == "model"  ? app::tr("VMAF's model, or the display CVVDP modelled.")
                             : QString();
}

SavedScoresPanel::SavedScoresPanel(const QString& title, std::vector<SavedScoreRow> rows,
                                   const std::vector<std::string>& columns, QWidget* parent)
    : PopupPanel(parent), rows_(std::move(rows)) {
    setObjectName(QStringLiteral("savedScoresDetails"));
    auto* layout = new QVBoxLayout(this);
    auto* heading = new QLabel(title);
    theme::set_role(heading, "title");
    heading->setTextFormat(Qt::PlainText);
    auto* head = new QHBoxLayout;
    head->addWidget(heading, 1);
    save_button_ = new SaveButton(app::tr("Save summary"), SaveKind::data);  // the window says what it saves (SavedResultsPanel::open_details)
    head->addWidget(save_button_);
    layout->addLayout(head);

    std::vector<std::string> shown;  // in the keys' own order
    for (const std::string& key : saved_details_column_keys()) {
        if (std::find(columns.begin(), columns.end(), key) != columns.end()) {
            shown.push_back(key);
        }
    }
    const bool several = std::any_of(rows_.begin(), rows_.end(), [](const SavedScoreRow& row) {
        return row.versions > 1;
    });
    QStringList headings{app::tr("Metric"), app::tr("Score")};
    QStringList tips{QString(), app::tr("The whole video's score, as Metric Graphs shows it.")};
    for (const std::string& key : shown) {
        headings << saved_details_heading(key);
        tips << saved_details_tip(key);
    }
    if (several) {
        used_column_ = static_cast<int>(headings.size());
        headings << app::tr("Used");
        tips << app::tr("The score Open in Metric Graphs and the exports take, where a metric has several. Click "
                        "another of its rows to use that one.");
    }
    table_ = new QTableWidget(static_cast<int>(rows_.size()), static_cast<int>(headings.size()));
    table_->setHorizontalHeaderLabels(headings);
    for (int column = 0; column < headings.size(); ++column) {
        table_->horizontalHeaderItem(column)->setToolTip(tips[column]);
    }
    table_->verticalHeader()->setVisible(false);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setSelectionMode(QAbstractItemView::NoSelection);
    table_->setFocusPolicy(Qt::NoFocus);
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setStretchLastSection(false);
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const SavedScoreRow& score = rows_[static_cast<std::size_t>(row)];
        const MetricDefinition* metric = find_metric(score.key);
        const bool again = row > 0 && rows_[static_cast<std::size_t>(row) - 1].key == score.key;
        QStringList cells{again ? QString() : metric_name(score.key), score.score};
        for (const std::string& key : shown) {
            if (key == "bands") {
                QStringList bands;  // as Metric Graphs' columns head them, with each one's share
                if (score.stats && score.stats->count > 0) {
                    for (const ThresholdCount& band : score.stats->thresholds) {
                        bands << QString(QChar::fromLatin1(band.threshold.comparison)) + " " +
                                     qt_text(format_general(band.threshold.value)) + ": " +
                                     qt_text(format_fixed(band.percentage, 1)) + "%";
                    }
                }
                cells << (bands.isEmpty() ? QStringLiteral("\u2014") : bands.join(QStringLiteral(" \u00B7 ")));
            } else if (const std::optional<double ScoreStats::*> stat = stat_of(key)) {
                cells << (score.stats && score.stats->count > 0 && metric != nullptr
                              ? qt_text(metric->format_value((*score.stats).*(*stat)))
                              : QStringLiteral("—"));
            } else {
                cells << (key == "saved"    ? score.saved
                          : key == "device" ? score.device
                          : key == "made"   ? score.made
                                            : (score.model.isEmpty() ? QStringLiteral("—") : score.model));
            }
        }
        if (several) {
            cells << QString();
        }
        const QString tip = !score.note.isEmpty()   ? score.note
                            : score.versions > 1 ? app::tr("Click to use this score.")
                                                 : QString();
        for (int column = 0; column < cells.size(); ++column) {
            auto* item = new QTableWidgetItem(cells[column]);
            if (column == 1 || (column >= 2 && column < 2 + static_cast<int>(shown.size()) &&
                                stat_of(shown[static_cast<std::size_t>(column - 2)]))) {
                item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            }
            if (column == used_column_) {
                item->setTextAlignment(Qt::AlignCenter);
            }
            item->setToolTip(tip);
            if (!score.note.isEmpty()) {
                item->setForeground(theme::color("muted"));
            }
            table_->setItem(row, column, item);
        }
    }
    show_used();
    connect(table_, &QTableWidget::cellClicked, this, [this](int row, int) { pick(row); });
    layout->addWidget(table_);

    auto* settings_link = new QLabel(QStringLiteral("<a href=\"settings\">") +
                                     app::tr("Choose columns…").toHtmlEscaped() + QStringLiteral("</a>"));
    settings_link->setObjectName(QStringLiteral("savedDetailsSettings"));
    settings_link->setTextFormat(Qt::RichText);
    settings_link->setToolTip(app::tr("Settings > Saved Results"));
    connect(settings_link, &QLabel::linkActivated, this, [this] {
        close();
        emit settings_requested();
    });
    layout->addWidget(settings_link, 0, Qt::AlignRight);

    // As wide and tall as the scores need, up to three quarters of the
    // screen; it scrolls past that.
    table_->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);  // its width
}

void SavedScoresPanel::pick(int row) {
    if (row < 0 || row >= static_cast<int>(rows_.size())) {
        return;
    }
    SavedScoreRow& chosen = rows_[static_cast<std::size_t>(row)];
    if (chosen.versions < 2 || chosen.used || !chosen.note.isEmpty()) {
        return;  // nothing to choose, or the one used, or one that can't be
    }
    for (SavedScoreRow& other : rows_) {
        if (other.key == chosen.key) {
            other.used = &other == &chosen;
        }
    }
    show_used();
    if (picked) {
        picked(chosen.key, chosen.file);
    }
}

void SavedScoresPanel::show_used() {
    if (used_column_ < 0) {
        return;
    }
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        const SavedScoreRow& score = rows_[static_cast<std::size_t>(row)];
        if (QTableWidgetItem* item = table_->item(row, used_column_)) {
            item->setText(score.versions > 1 && score.used ? QStringLiteral("●") : QString());
        }
    }
}

void SavedScoresPanel::show_under(QWidget* anchor) {
    // Its height exactly its heading's and rows', measured once styled
    // (AdjustToContents left a band of empty rows under them), up to three
    // quarters of the screen.
    ensurePolished();
    table_->ensurePolished();
    table_->resizeRowsToContents();
    const QScreen* anchor_screen = QGuiApplication::screenAt(anchor->mapToGlobal(QPoint(0, 0)));
    const int tallest = (anchor_screen != nullptr ? anchor_screen->availableGeometry().height() : 1000) * 3 / 4;
    table_->setFixedHeight(std::min(tallest, table_->horizontalHeader()->sizeHint().height() +
                                              table_->verticalHeader()->length() + 2 * table_->frameWidth()));
    adjustSize();
    QPoint place = anchor->mapToGlobal(anchor->rect().bottomLeft());
    if (const QScreen* screen = QGuiApplication::screenAt(place); screen != nullptr) {
        const QRect room = screen->availableGeometry();
        place.setX(std::clamp(place.x(), room.left(), std::max(room.left(), room.right() - width())));
        place.setY(std::clamp(place.y(), room.top(), std::max(room.top(), room.bottom() - height())));
    }
    move(place);
    show();
}

}  // namespace vml::app
