// The Saved results tab: the comparisons saved in the results cache
// (saved_results.h), listed, opened in Metric Graphs, exported (results,
// per-frame scores, graphs, weak segments) and deleted. The selected
// comparison's scores are in a column on the table's left, a card each (a
// row each in VML Classic), and every detail of them in a panel its ⓘ
// button opens (SavedScoresPanel). Where a metric has several saved scores,
// the newest usable one is used unless another is picked. The table's
// columns are resized by dragging and hidden from the header's right-click
// menu; both are kept in Settings.
#pragma once

#include <QSet>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "quick_save.h"
#include "saved_score_details.h"
#include "dialogs.h"
#include "file_writes.h"
#include "vml/metric_cache.h"
#include "vml/results.h"
#include "vml/saved_results.h"
#include "vml/settings.h"
#include "vml/weak_frames.h"

class QAction;
class QComboBox;
class QGridLayout;
class QGroupBox;
class QLabel;
class QLayout;
class QLineEdit;
class QMenu;
class QPushButton;
class QScrollArea;
class QTableWidget;
class QToolButton;
template <typename T>
class QFutureWatcher;

namespace vml::app {

class ElidedLabel;

// A saved comparison read back: its result, label and folder.
struct LoadedSavedResult {
    std::shared_ptr<const ComparisonResult> result;
    QString label;
    std::filesystem::path folder;
    // When each metric's score used was saved, by its key: the score
    // details' "saved" column.
    std::map<std::string, std::chrono::system_clock::time_point> saved{};
};

class SavedResultsPanel : public QWidget {
    Q_OBJECT

public:
    // `contents`: what Settings' Exports include (the right-click menu's).
    enum class Export {
        contents,
        results_json,
        results_csv,
        frames_json,
        frames_csv,
        score_summary_json,
        score_summary_csv,
        graphs,
        segments,
        run_file
    };

    // The quick grabs the Videos table's and Saved Results' menus share
    // (docs/ARCHITECTURE.md, "Exports"), after their archive item: results
    // and per-frame scores as JSON or CSV, the graphs as a .zip, the weak
    // segments, then the run file; each calling `chosen` with its kind,
    // usable where `enabled`. One in `refused` is greyed, its reason its
    // tooltip: said before the click, not in a window after it.
    static void add_quick_grabs(QMenu& menu, bool enabled, const std::function<void(Export)>& chosen,
                                const std::map<Export, QString>& refused = {});

    // `settings` are the window's (the saved results' folder, the weak
    // segment profiles); `dialogs` the window's, which outlive the panel;
    // `file_writes` the window's queue of writes, which Delete goes through.
    SavedResultsPanel(const Settings& settings, const Dialogs& dialogs, FileWrites& file_writes,
                      QWidget* parent = nullptr);

    // The list's save icon (issue #224): the selected comparisons' results,
    // saved at one click by the window (`save_results`, in Settings' Quick
    // saves `formats`); Save as... on a right-click is the Export menu's.
    std::function<void(const std::vector<LoadedSavedResult>& results)> save_results;
    std::function<std::vector<std::string>()> quick_formats;
    // Whether the Videos tab, with the settings as they are, would use the
    // saved score `choice` of `key` for `saved`'s videos (the run's own
    // check: metric_cache.h's run_takes): Settings > Saved Results' "the one
    // the Videos tab would use". Unset: the newest.
    std::function<bool(const SavedComparison& saved, const std::string& key, const SavedScoreChoice& choice)>
        videos_would_use;
    // The details panel's save icon: the selected comparison's score
    // details, saved at one click in the quick saves' formats.
    std::function<void(const std::vector<LoadedSavedResult>& results)> save_score_summary;
    SaveButton* save_button() const { return save_button_; }
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
    // paths until the run saves there. Added to those known; `all` (the
    // folders found at the run's start, not one a save adds) unlocks the
    // comparisons of neither. Ignored when no run is on.
    void add_run_folders(const std::vector<std::filesystem::path>& folders, bool all);
    // Whether Delete is locked for `saved`: a run is on, and its folders are
    // not known yet, or its reference and test video are one of the run's
    // pairs, or its folder one of the run's.
    bool locked_by_run(const SavedComparison& saved) const;
    // The selected comparisons, read with the scores picked.
    std::vector<LoadedSavedResult> selected_results() const;
    // The weak segment rules the tab bar's Export panel offers, by their
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
    // The score of `key` the selected comparison uses (the one picked, else
    // its newest usable one), or none with no comparison shown or no score of
    // `key`. pick_score makes `file` the one used, as a click in the details
    // panel or on a card does.
    std::optional<std::filesystem::path> used_score(const std::string& key) const;
    void pick_score(const std::string& key, const std::filesystem::path& file);
    // The ⓘ button over the scores, and the panel it opens (nullptr with
    // no comparison's scores shown).
    QToolButton* details_button() const { return details_button_; }
    SavedScoresPanel* open_details();
    // Waits for the selected comparison's scores to be read, and shows them.
    void wait_for_details();
    // The cards a row in the scores' column.
    static constexpr int tiles_a_row = 3;
    // Settings > Saved Results changed how the scores are shown (a card for
    // the score used, or for each): shown again.
    void scores_settings_changed();
    // The Saved scores box as tiles (VML Modern's) or rows (Classic's), not
    // as the interface in use has it; shown again. For tests, which run in
    // Classic.
    void set_tiles(bool tiles);
    // The score read for `file` (details_read_), or nothing where it could
    // not be read.
    const SavedScore* read_score(const std::filesystem::path& file) const;
    void delete_selected();
    // The table's right-click menu: the selected comparisons' export, of
    // what Settings' Exports include and named after it ("this video's"
    // with one, "the selected videos'" with several); none with none. The
    // action added, or nullptr.
    QAction* fill_menu(QMenu& menu) const;
    // The table's right-click menu, made as it opens: Open in Metric Graphs,
    // then the Export menu's items, then Delete... (as delete_button_);
    // nothing with nothing selected.
    void fill_context_menu(QMenu& menu);
    // The archive of the selection, then the quick grabs
    // (add_quick_items), added to `menu`: the table's right-click's.
    void add_export_items(QMenu& menu);
    // The quick grabs of the selection (add_quick_grabs): the save icon's
    // Save as... items.
    void add_quick_items(QMenu& menu);
    // The header's right-click menu: every column with a tick, unticked
    // hiding it (the test video's always shown), then Reset column widths.
    void fill_column_menu(QMenu& menu);
    // The columns as Settings keeps them: their widths, one a column, or
    // none while they are fitted to the tab (never resized, or reset); and
    // the hidden ones by their key ("reference", "saved", "metrics", "size").
    std::vector<int> column_widths() const;
    std::vector<std::string> hidden_columns() const;

signals:
    void open_requested(const std::vector<vml::app::LoadedSavedResult>& results);
    void export_requested(vml::app::SavedResultsPanel::Export kind,
                          const std::vector<vml::app::LoadedSavedResult>& results);
    // Saved results were deleted: Settings' count is out of date.
    void deleted();
    // A column was hidden, shown or resized (once a drag ends):
    // column_widths and hidden_columns are to be saved.
    void columns_changed();
    // The details panel's link: Settings > Saved Results to be shown.
    void scores_settings_requested();
    // The list was read again: what was known of its comparisons may be gone.
    void listed();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void changeEvent(QEvent* event) override;

private:
    std::filesystem::path base() const;
    void fit_columns();
    void keep_widths();
    void show_column(int column, bool shown);
    void apply_filter();
    void show_details();
    // Deletes every item of `layout`, the widgets of layouts within it too.
    static void clear_layout(QLayout* layout);
    // A score's card: `name` (a label, or the radio button that picks it),
    // `badge` beside it (how many are saved; none: nothing), its score
    // large, and `line` under it (CPU or GPU, or when it was saved).
    QWidget* make_tile(const std::string& key, QWidget* name, const QString& score, const QString& line,
                       const QString& badge, const QString& badge_tip) const;
    // The badge saying how many of a metric's scores are saved.
    static QLabel* saved_badge(const std::string& key, const QString& text, const QString& tip);
    // Whether the Saved scores box shows tiles: VML Modern's, unless a test
    // asked (set_tiles).
    bool showing_tiles() const;
    // The selected comparison's scores, as cards (VML Modern's) or rows
    // (Classic's): one for each metric's score used, or for each score
    // (Settings::saved_score_versions).
    void show_scores(const SavedComparison& saved);
    // How a saved score was made, as its file records it: CPU or GPU, the
    // implementation and its version, the model or CVVDP's display.
    struct Made {
        QString device;
        QString made;
        QString model;
    };
    Made made_of(const std::variant<FrameMetricResult, SequenceMetricResult>& score) const;
    // Why a score file is not used ("out of date..."), or empty where it is usable.
    QString unusable_note(const std::filesystem::path& file) const;
    // The score files read for a comparison other than the one shown: a file
    // read to choose among a metric's several scores is not read again to
    // load it.
    using ScoreReads = std::map<std::filesystem::path, std::optional<SavedScore>>;
    // The file of `key` `saved` uses: the one picked, else the one Settings'
    // rule picks (pick_saved_score). The files it reads to choose are kept in
    // `reads` where given.
    std::optional<std::filesystem::path> used_file(const SavedComparison& saved, const std::string& key,
                                                   ScoreReads* reads = nullptr) const;
    // The details panel's rows for `saved`, a score file each.
    std::vector<SavedScoreRow> detail_rows(const SavedComparison& saved) const;
    // A card's height, the same for every card.
    int tile_height() const;
    // The ⓘ button's icon, drawn in the text's colour.
    void draw_details_icon();
    // The scores' column's width: three cards of the widest metric name or score.
    int scores_column_width() const;
    std::vector<const SavedComparison*> selected() const;
    SaveButton* save_button_ = nullptr;
    // Whether the selected comparisons are of more than one reference, which
    // graphs and weak segments are not made for.
    bool several_references() const;
    void update_buttons();

    const Settings& settings_;
    const Dialogs& dialogs_;
    FileWrites& file_writes_;
    std::vector<SavedComparison> list_;
    // The score picked for a metric of a comparison (its folder): else its newest.
    std::map<std::filesystem::path, std::map<std::string, std::filesystem::path>> picked_;
    QFutureWatcher<std::vector<SavedComparison>>* reading_ = nullptr;
    // Set to stop the listing running: a newer one has started, or the tab
    // is going (it can take minutes on a slow share).
    std::shared_ptr<std::atomic<bool>> listing_stopped_;
    bool run_active_ = false;
    // The run's pairs and folders, each as locked_by_run compares them
    // (run_key), and whether its folders are known yet.
    QSet<QString> run_pairs_;
    QSet<QString> run_folders_;
    bool run_folders_known_ = false;
    QLabel* folder_label_ = nullptr;
    QLineEdit* filter_ = nullptr;
    QTableWidget* table_ = nullptr;
    // The columns' widths, a hidden one's as it was before it was hidden;
    // whether they are the user's (else fitted to the tab, as the table
    // grows and shrinks); whether the header is pressed, and a column was
    // resized while it was (saved as it is let go); whether the columns are
    // being fitted.
    std::vector<int> widths_;
    bool own_widths_ = false;
    bool header_pressed_ = false;
    bool resized_by_hand_ = false;
    bool fitting_ = false;
    QGroupBox* details_ = nullptr;
    QGridLayout* details_grid_ = nullptr;
    QScrollArea* details_scroll_ = nullptr;  // the column's cards or rows, scrolled
    QToolButton* details_button_ = nullptr;
    ElidedLabel* details_name_ = nullptr;  // the comparison whose scores are shown
    std::optional<bool> tiles_;  // set_tiles' choice; none: the interface's
    QString rules_choice_ = QStringLiteral("in-use");
    QPushButton* open_button_ = nullptr;
    QPushButton* delete_button_ = nullptr;
    // A Delete queued on file_writes_: set by it as it ends (done last),
    // read once the queue says it is idle (on_writes_idle).
    struct Deleting {
        std::atomic<bool> done = false;
        int failed = 0;
    };
    std::shared_ptr<Deleting> deleting_;
    void on_writes_idle();
    // The score files of the comparison the box shows, as read on a thread
    // of their own (show_details): none where one can't be read. Open and
    // the exports take them from here rather than read them again.
    struct ReadScores {
        std::filesystem::path folder;
        std::map<std::filesystem::path, std::optional<SavedScore>> scores;
        std::map<std::filesystem::path, ScoreStats> stats;  // each score's frames', for the details panel
        std::uint64_t listing = 0;  // the listing it was read for (listing_)
        bool complete = false;      // every file read, not stopped
    };
    QFutureWatcher<ReadScores>* details_reading_ = nullptr;
    ReadScores details_read_;
    // Counts the listings shown (show_list): a read made for an earlier one
    // may have missed files the new one lists, and is not used.
    std::uint64_t listing_ = 0;
    // Set to stop the read running: one no longer wanted (another row
    // chosen, a new listing) stops at its next file rather than read the
    // rest of a comparison no one will see.
    std::shared_ptr<std::atomic<bool>> details_stop_;
    std::filesystem::path details_reading_folder_;  // what the read running reads
    void stop_details_read();
    void fill_details(const SavedComparison& saved);
};

}  // namespace vml::app
