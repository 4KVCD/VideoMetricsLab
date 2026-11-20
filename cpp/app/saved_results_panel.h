// The Saved results tab: the comparisons saved in the results cache
// (saved_results.h), listed, opened in Metric Graphs, exported (results,
// per-frame scores, graphs, weak segments) and deleted. Where a metric has
// several saved scores, the newest is used unless another is picked.
#pragma once

#include <QSet>
#include <QString>
#include <QWidget>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dialogs.h"
#include "vml/results.h"
#include "vml/saved_results.h"
#include "vml/settings.h"
#include "vml/weak_frames.h"

class QAction;
class QComboBox;
class QGridLayout;
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
    enum class Export { contents, results_json, results_csv, frames_json, frames_csv, graphs, segments };

    // `settings` are the window's (the saved results' folder, the weak
    // segment profiles); `dialogs` the window's, which outlive the panel.
    SavedResultsPanel(const Settings& settings, const Dialogs& dialogs, QWidget* parent = nullptr);
    ~SavedResultsPanel() override;

    // Lists the saved comparisons again, read on a thread of their own.
    void refresh();
    // Called with the list read (by refresh, or by the tests directly).
    void show_list(std::vector<SavedComparison> list);
    // While metrics are calculated, Delete is locked for the comparisons of
    // the run's pairs (`pairs`, each its reference and a test video): the
    // run reads and saves their scores. Others can still be deleted, those of
    // the same reference with another test video too: each comparison is a
    // folder of its own. Until the run's folders are known
    // (add_run_folders), every comparison is locked. Inactive: none is.
    void set_run_active(bool active,
                        const std::vector<std::pair<std::filesystem::path, std::filesystem::path>>& pairs = {});
    // Folders the run reads and saves scores in (metric_cache.h's
    // recipe_directory), locked too: a file moved, renamed or copied since it
    // was scored has its scores in a folder whose context.json names its old
    // paths until the run saves there. Added to those known; the first call
    // unlocks the comparisons of neither. Ignored when no run is on.
    void add_run_folders(const std::vector<std::filesystem::path>& folders);
    // Whether Delete is locked for `saved`: a run is on, and its folders are
    // not known yet, or its reference and test video are one of the run's
    // pairs, or its folder one of the run's.
    bool locked_by_run(const SavedComparison& saved) const;
    // The selected comparisons, read with the scores picked.
    std::vector<LoadedSavedResult> selected_results() const;
    // The weak segment rules offered under Export > Weak segments, by their
    // choice ("in-use", "profile:<name>", "preset:<key>") with their text:
    // the profile in use, every profile and every preset, as they are now.
    std::vector<std::pair<QString, QString>> segment_rule_choices() const;
    // Picks the rules by their choice, and the one picked (the profile in
    // use until another is picked, or after the one picked is gone).
    void pick_segment_rules(const QString& choice);
    QString segment_rule_choice() const;
    // The rules the weak segments are found by, in both of the tab's
    // exports: the ones picked.
    SegmentRules segment_rules() const;

    // For the tests.
    QTableWidget* table() const { return table_; }
    QLineEdit* filter() const { return filter_; }
    // The list a metric's saved score is picked from: none where it has one.
    QComboBox* score_choice(const std::string& key) const;
    // The Export button's menu, made as it opens: the right-click menu's
    // export, then results, per-frame scores, graphs and weak segments each
    // on its own.
    QMenu* export_menu() const { return export_menu_; }
    void fill_export_menu(QMenu& menu);
    void delete_selected();
    // The table's right-click menu: the selected comparisons' export, of
    // what Settings' Exports include and named after it ("this video's"
    // with one, "the selected videos'" with several); none with none. The
    // action added, or nullptr.
    QAction* fill_menu(QMenu& menu) const;
    // The table's right-click menu, made as it opens: Open in Metric Graphs,
    // then fill_menu's export; nothing with nothing selected.
    void fill_context_menu(QMenu& menu);

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
    // The run's pairs and folders, each as locked_by_run compares them
    // (run_key), and whether its folders are known yet.
    QSet<QString> run_pairs_;
    QSet<QString> run_folders_;
    bool run_folders_known_ = false;
    QLabel* folder_label_ = nullptr;
    QLineEdit* filter_ = nullptr;
    QTableWidget* table_ = nullptr;
    QGroupBox* details_ = nullptr;
    QGridLayout* details_grid_ = nullptr;
    std::map<std::string, QComboBox*> score_combos_;
    QString rules_choice_ = QStringLiteral("in-use");
    QPushButton* open_button_ = nullptr;
    QPushButton* export_button_ = nullptr;
    QMenu* export_menu_ = nullptr;
    QPushButton* delete_button_ = nullptr;
};

}  // namespace vml::app
