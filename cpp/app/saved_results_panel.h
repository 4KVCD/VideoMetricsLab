// The Saved results tab: the comparisons saved in the results cache
// (saved_results.h), listed, opened in Metric Graphs, exported (results,
// per-frame CSV, graphs, weak segments) and deleted. Where a metric has
// several saved scores, the newest is used unless another is picked.
#pragma once

#include <QString>
#include <QWidget>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dialogs.h"
#include "vml/results.h"
#include "vml/saved_results.h"
#include "vml/settings.h"
#include "vml/weak_frames.h"

class QAction;
class QComboBox;
class QFormLayout;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QPushButton;
class QTableWidget;
template <typename T>
class QFutureWatcher;

namespace vml::app {

// A saved comparison read back: its result, label and folder.
struct LoadedSavedResult {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::filesystem::path folder;
};

class SavedResultsPanel : public QWidget {
    Q_OBJECT

public:
    // `contents`: what Settings' Exports include (the right-click menu's).
    enum class Export { contents, results_json, results_csv, frames_csv, graphs, segments };

    // `settings` are the window's (the saved results' folder, the weak
    // segment profiles); `dialogs` the window's, which outlive the panel.
    SavedResultsPanel(const Settings& settings, const Dialogs& dialogs, QWidget* parent = nullptr);
    ~SavedResultsPanel() override;

    // Lists the saved comparisons again, read on a thread of their own.
    void refresh();
    // Called with the list read (by refresh, or by the tests directly).
    void show_list(std::vector<SavedComparison> list);
    // While metrics are calculated, Delete is locked: the run writes there.
    void set_run_active(bool active);
    // The weak segment rules offered: the profile in use, every profile and
    // every preset (filled again when the tab is shown).
    void fill_segment_rules();

    // The selected comparisons, read with the scores picked.
    std::vector<LoadedSavedResult> selected_results() const;
    // The rules the weak segments export uses: the one picked.
    SegmentRules segment_rules() const;

    // For the tests.
    QTableWidget* table() const { return table_; }
    QLineEdit* filter() const { return filter_; }
    // The list a metric's saved score is picked from: none where it has one.
    QComboBox* score_choice(const std::string& key) const;
    QComboBox* rules_choice() const { return rules_combo_; }
    void delete_selected();
    // The table's right-click menu: the selected comparisons' export, of
    // what Settings' Exports include and named after it ("this video's"
    // with one, "the selected videos'" with several); none with none. The
    // action added, or nullptr.
    QAction* fill_menu(QMenu& menu) const;

signals:
    void open_requested(const std::vector<vml::app::LoadedSavedResult>& results);
    void export_requested(vml::app::SavedResultsPanel::Export kind,
                          const std::vector<vml::app::LoadedSavedResult>& results);
    // Saved results were deleted: Settings' count is out of date.
    void deleted();

private:
    std::filesystem::path base() const;
    void apply_filter();
    void show_details();
    std::vector<const SavedComparison*> selected() const;
    void update_buttons();

    const Settings& settings_;
    const Dialogs& dialogs_;
    std::vector<SavedComparison> list_;
    // The score picked for a metric of a comparison (its folder): else its newest.
    std::map<std::filesystem::path, std::map<std::string, std::filesystem::path>> picked_;
    QFutureWatcher<std::vector<SavedComparison>>* reading_ = nullptr;
    bool run_active_ = false;
    QLabel* folder_label_ = nullptr;
    QLineEdit* filter_ = nullptr;
    QTableWidget* table_ = nullptr;
    QGroupBox* details_ = nullptr;
    QFormLayout* details_form_ = nullptr;
    std::map<std::string, QComboBox*> score_combos_;
    QComboBox* rules_combo_ = nullptr;
    QPushButton* open_button_ = nullptr;
    QPushButton* results_button_ = nullptr;
    QPushButton* frames_button_ = nullptr;
    QPushButton* graphs_button_ = nullptr;
    QPushButton* segments_button_ = nullptr;
    QPushButton* delete_button_ = nullptr;
};

}  // namespace vml::app
