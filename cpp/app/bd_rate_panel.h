// The BD-rate tab: how much bitrate a test encoder saves (negative) or costs
// (positive) against an anchor for the same score, for each sequence and
// each of the nine metrics, with their mean and median; and one sequence's
// rate curves. Each side's encodes come from scored videos (the session's
// runs, as Metric Graphs lists them) or CSV files in the command line's
// bdrate format; the numbers are the command line's (vml/bd_rate_report.h).
#pragma once

#include <QString>
#include <QStringList>
#include <QWidget>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dialogs.h"
#include "graph_panel.h"
#include "vml/bd_rate_report.h"

class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QVBoxLayout;

namespace vml::app {

class RateCurveChart;

class BdRatePanel : public QWidget {
    Q_OBJECT

public:
    // `runs`: the session's scored comparisons, asked for when videos are
    // added.
    explicit BdRatePanel(std::function<std::vector<GraphPanel::Run>()> runs, QWidget* parent = nullptr);

    // The results table's colours again, for the theme now in use.
    void refresh_theme();
    // The scored videos added take the scores of their lines as they are now:
    // a line scored again has others. A video no longer listed keeps those it
    // had, which are still its: a list is often cleared between the anchor's
    // videos and the test's.
    void bring_videos_up_to_date();

    // Its questions and messages; the tests answer them.
    Dialogs dialogs;
    // Which of `items` (the scored videos) to add: their places in the list;
    // none for Cancel.
    std::function<std::vector<int>(const QStringList& items)> choose_videos;
    // Puts text on the clipboard.
    std::function<void(const QString& text)> copy_text;

    // One encoder: the anchor or the test.
    struct Side {
        QGroupBox* box = nullptr;
        QLineEdit* name_edit = nullptr;
        QLabel* count_label = nullptr;
        QTableWidget* sequences = nullptr;
        QPushButton* add_videos_button = nullptr;
        QPushButton* import_button = nullptr;
        QPushButton* clear_button = nullptr;
        // Its encodes by where they came from: a CSV file ("csv:" and its
        // path) or a scored video ("video:" and its line's identity). Adding
        // one again replaces it.
        std::vector<std::pair<QString, RateTable>> sources;
        // Its scored videos whose bitrate is the whole file's, soundtrack
        // included (VideoInfo::bit_rate_whole_file), by source key.
        std::map<QString, QString> whole_file_rates;
    };
    Side anchor;
    Side test;
    QComboBox* group_combo = nullptr;
    QPushButton* warnings_button = nullptr;
    QPushButton* export_button = nullptr;
    QPushButton* copy_button = nullptr;
    QLabel* heading_label = nullptr;
    QTableWidget* results = nullptr;
    QComboBox* metric_combo = nullptr;
    QLabel* sequence_label = nullptr;
    RateCurveChart* chart = nullptr;

    // Adds a CSV file's encodes to `side`; says why not where it cannot.
    void import_csv(Side& side, const QString& path);
    // The report shown; none while there is none to show.
    const std::optional<BdRateReport>& report() const { return report_; }
    // The warnings the warnings button lists: the report's and the tab's own.
    const QStringList& warnings() const { return warnings_; }

protected:
    void showEvent(QShowEvent* event) override;

private:
    QGroupBox* build_side(Side& side, const QString& title);
    void add_options_row(QVBoxLayout* root);
    void add_results(QVBoxLayout* root);
    void add_chart(QVBoxLayout* root);

    void on_add_videos(Side& side);
    // bring_videos_up_to_date for a side: whether anything changed, and the
    // names of the videos that no longer have a bitrate, which leave.
    bool update_videos_of(Side& side, const std::vector<GraphPanel::Run>& runs, QStringList& without_rate);
    void on_import(Side& side);
    void on_clear(Side& side);
    void on_export();
    void on_copy();
    void on_show_warnings();
    // A side's sources changed: its table, the group choices and the report.
    void side_changed(Side& side);
    void refresh_side_table(Side& side);
    void refresh_group_choices();
    // The report worked out again and shown.
    void recalculate();
    void on_name_changed();
    void show_heading();  // of the report shown
    void fill_results();
    void on_results_row(int row);
    void refresh_chart();

    QString side_name(const Side& side) const;
    static RateTable merged(const Side& side);

    std::function<std::vector<GraphPanel::Run>()> runs_;
    std::optional<BdRateReport> report_;
    QStringList warnings_;
    // The sequence the chart shows, by name; none for no sequence row chosen.
    std::optional<std::string> chosen_sequence_;
    // The rows of the results table that are sequences, by row: their names.
    std::map<int, std::string> sequence_rows_;
    bool filling_ = false;  // while fill_results sets the table's selection
};

}  // namespace vml::app
