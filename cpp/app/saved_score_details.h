// Saved Results' details panel (its ⓘ button): every saved score of the
// comparison selected, a row a score file, with the statistics and the facts
// of how it was made that Settings > Saved Results chooses
// (Settings::saved_details_columns). Where a metric has several saved scores,
// a click on one of its rows makes it the one used. A PopupPanel: it closes on
// Esc, on a press outside it, or on a second click of its button.
#pragma once

#include <QString>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "popup_panel.h"
#include "vml/stats.h"

class QLabel;

namespace vml::app {
class SaveButton;
}
class QTableWidget;

namespace vml::app {

// The columns the panel can show after the metric and its score, as
// Settings keeps them, in the order shown: the statistics of the frames'
// scores, as Metric Graphs' table has them ("median", "stdev", "min", "max",
// "worst_10", "worst_5", "worst_1", "worst_0_1", and "bands": the share in
// each of the metric's score bands, in one column as each metric has its
// own), then how the score was made ("model", "made", "device"), then when
// it was saved ("saved").
const std::vector<std::string>& saved_details_column_keys();
// A column's heading and its tooltip, translated.
QString saved_details_heading(const std::string& key);
QString saved_details_tip(const std::string& key);

// A saved score as the panel shows it: one score file of a metric.
struct SavedScoreRow {
    std::string key;
    std::filesystem::path file;
    QString score;                     // its one number, as Metric Graphs shows it
    std::optional<ScoreStats> stats;   // its frames' (or seconds'); none where it can't be read
    QString saved;                     // when it was saved
    QString device;                    // "GPU" or "CPU"
    QString made;                      // the implementation and its version
    QString model;                     // VMAF's model, CVVDP's display; empty for none
    QString note;                      // why it isn't used ("out of date..."); empty where usable
    bool used = false;                 // the one Open and the exports take
    int versions = 1;                  // how many of its metric are saved
};

class SavedScoresPanel : public PopupPanel {
    Q_OBJECT

public:
    // `title` names the comparison; `columns` are Settings' keys, shown in
    // saved_details_column_keys()'s order, unknown ones left out.
    SavedScoresPanel(const QString& title, std::vector<SavedScoreRow> rows, const std::vector<std::string>& columns,
                     QWidget* parent = nullptr);

    // A metric's other score clicked: it is to be the one used.
    std::function<void(const std::string& key, const std::filesystem::path& file)> picked;

    // The save button at its top right: the window sets what it saves.
    SaveButton* save_button() const { return save_button_; }

    // Opens under `anchor`, its left edge on the anchor's, kept on screen.
    void show_under(QWidget* anchor);

    // For the tests.
    QTableWidget* table() const { return table_; }
    const std::vector<SavedScoreRow>& rows() const { return rows_; }

signals:
    // The line under the table: Settings > Saved Results to be shown.
    void settings_requested();

private:
    void pick(int row);
    void show_used();

    std::vector<SavedScoreRow> rows_;
    QTableWidget* table_ = nullptr;
    int used_column_ = -1;
    SaveButton* save_button_ = nullptr;  // the column marking the score used, where a metric has several; else -1
};

}  // namespace vml::app
