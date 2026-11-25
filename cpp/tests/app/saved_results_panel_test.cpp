// The Saved Results tab (saved_results_panel.h): the comparisons of the
// results cache listed and filtered, a comparison's saved scores shown, the
// selection read back for Metric Graphs and the exports, the weak segment
// rules picked, and Delete (locked during a run).
#include <QAction>
#include <QComboBox>
#include <QGroupBox>
#include <QCoreApplication>
#include <QLabel>
#include <QGridLayout>
#include <QLayout>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QRadioButton>
#include <QToolButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTest>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

#include "dialogs.h"
#include "saved_results_panel.h"
#include "saved_score_details.h"
#include "test_videos.h"
#include "widgets.h"
#include "vml/analysis_request.h"
#include "vml/metric_cache.h"
#include "vml/metrics.h"
#include "vml/segment_presets.h"

namespace vml::app {
namespace {

FrameMetricResult frame_metric(const std::string& key, std::vector<float> values) {
    FrameMetricResult metric;
    metric.key = key;
    for (int frame = 0; frame < static_cast<int>(values.size()); ++frame) {
        metric.frame.push_back(frame);
        metric.time.push_back(frame / 24.0);
    }
    metric.values = std::move(values);
    metric.provenance = current_ffmpeg_provenance(key, "9.0.2");
    return metric;
}

TEST(SavedResultsPanel, ListsReadsAndDeletesTheSavedComparisons) {
    const test::Folder folder;
    std::ofstream(folder / "s.mp4") << "s";
    std::ofstream(folder / "t.mp4") << "t";
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    ComparisonResult result;
    result.source = folder / "s.mp4";
    result.distorted = folder / "t.mp4";
    result.fps = 24.0;
    result.metric_results.add(frame_metric("vmaf", {95.0F, 60.0F, 95.0F}));
    result.metric_results.add(frame_metric("psnr", {40.0F, 30.0F, 40.0F}));
    const AnalysisRequest request =
        analysis_request(VmafOptions{}, {"vmaf", "psnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const std::filesystem::path saved =
        store_result(folder / "cache", result.source, result.distorted, request.recipe, result, "t", request.metrics);

    Dialogs dialogs = standard_dialogs(nullptr);
    dialogs.message = [](const Message&) { return MessageAnswer{0, false}; };  // Delete saved results
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list(list_saved_comparisons(folder / "cache"));
    QTableWidget& table = *panel.table();
    ASSERT_EQ(table.rowCount(), 1);
    EXPECT_EQ(table.item(0, 0)->text(), "t");
    EXPECT_EQ(table.item(0, 1)->text(), "s.mp4");
    EXPECT_TRUE(table.item(0, 3)->text().contains("PSNR"));

    panel.filter()->setText("nothing like it");
    EXPECT_TRUE(table.isRowHidden(0));
    panel.filter()->setText("T.MP4");  // the full path, any case
    EXPECT_FALSE(table.isRowHidden(0));

    // Its scores, shown when it's selected, leave the table's height as it
    // was: the row clicked stays under the pointer.
    panel.resize(900, 700);
    panel.show();
    panel.layout()->activate();
    const int height = table.height();
    table.selectRow(0);
    panel.wait_for_details();
    panel.layout()->activate();
    EXPECT_EQ(table.height(), height);
    EXPECT_TRUE(panel.used_score("vmaf"));
    EXPECT_EQ(panel.findChild<QRadioButton*>("pick_vmaf"), nullptr);  // one score: nothing to pick
    // The scores are in a column on the table's left.
    const QWidget* column = panel.findChild<QWidget*>("savedScoresColumn");
    ASSERT_NE(column, nullptr);
    EXPECT_LT(column->geometry().right(), table.geometry().left());
    // In VML Modern a tile a metric: its score itself, large, as Metric
    // Graphs shows it, and how it was made under it.
    panel.set_tiles(true);
    for (const std::string key : {"vmaf", "psnr"}) {
        const QString name = QString::fromStdString(key);
        const auto* score = panel.findChild<QLabel*>("score_" + name);
        const auto* made = panel.findChild<ElidedLabel*>("made_" + name);
        ASSERT_NE(score, nullptr) << key;
        ASSERT_NE(made, nullptr) << key;
        EXPECT_EQ(score->text().toStdString(),
                  find_metric(key)->format_value(result.metric_results.frame(key)->aggregate()));
        EXPECT_GT(score->font().pointSizeF(), made->font().pointSizeF());
        EXPECT_TRUE(made->toolTip().isEmpty());  // nothing more to say of one score
        EXPECT_TRUE(made->full_text().contains("CPU") || made->full_text().contains("GPU"))
            << made->full_text().toStdString();
        EXPECT_TRUE(panel.findChild<QWidget*>("tile_" + name)->isAncestorOf(score));
    }
    // In VML Classic a row a metric: its score itself, then how it was made.
    panel.set_tiles(false);
    for (const std::string key : {"vmaf", "psnr"}) {
        const std::string score =
            find_metric(key)->format_value(result.metric_results.frame(key)->aggregate()) + " \xC2\xB7 ";
        bool found = false;
        for (const QLabel* label : panel.findChildren<QLabel*>()) {
            found = found || label->text().startsWith(QString::fromStdString(score));
        }
        std::string seen;
        for (const QLabel* label : panel.findChildren<QLabel*>()) {
            seen += "[" + label->text().toStdString() + "] ";
        }
        EXPECT_TRUE(found) << key << ": no line starts with " << score << " in " << seen;
        EXPECT_EQ(panel.findChild<QWidget*>("tile_" + QString::fromStdString(key)), nullptr);
    }
    // Shown again (another row, the same row clicked again): the rows before
    // are gone, not left under the new ones (each row is a layout of its
    // own, whose labels deleting it left behind).
    panel.scores_settings_changed();
    panel.scores_settings_changed();
    {
        const std::string score =
            find_metric("vmaf")->format_value(result.metric_results.frame("vmaf")->aggregate()) + " \xC2\xB7 ";
        int lines = 0;
        for (const QLabel* label : panel.findChildren<QLabel*>()) {
            lines += label->text().startsWith(QString::fromStdString(score)) ? 1 : 0;
        }
        EXPECT_EQ(lines, 1);
    }
    const std::vector<LoadedSavedResult> read = panel.selected_results();
    ASSERT_EQ(read.size(), 1U);
    EXPECT_EQ(read[0].label, "t");
    EXPECT_EQ(read[0].folder, saved);
    EXPECT_EQ(read[0].result->metric_results.frame("vmaf")->values, (std::vector<float>{95.0F, 60.0F, 95.0F}));

    // The right-click menu: the selection's export, named after what Settings'
    // Exports include.
    {
        QMenu menu;
        const QAction* contents = panel.fill_menu(menu);
        ASSERT_NE(contents, nullptr);
        EXPECT_EQ(contents->text(), "Export this video's results, graphs, worst frames and weak segments...");
        EXPECT_TRUE(contents->isEnabled());
    }

    // A double-click opens it in Metric Graphs, as does the right-click
    // menu's first entry, above the export.
    QSignalSpy opened(&panel, &SavedResultsPanel::open_requested);
    emit table.cellDoubleClicked(0, 0);
    EXPECT_EQ(opened.count(), 1);
    {
        QMenu menu;
        panel.fill_context_menu(menu);
        QStringList texts;
        for (const QAction* action : menu.actions()) {
            if (!action->isSeparator()) {
                texts << action->text();
            }
        }
        // Open, then the archive of the selection and the quick grabs.
        EXPECT_EQ(texts.size(), 9);
        EXPECT_EQ(texts.mid(0, 2), (QStringList{"Open in Metric Graphs",
                                                "Export this video's results, graphs, worst frames and weak segments..."}));
        EXPECT_EQ(texts.last(), "Run file (.metrics.json)...");
        menu.actions().first()->trigger();
        EXPECT_EQ(opened.count(), 2);
        ASSERT_EQ(opened.at(1).at(0).value<std::vector<LoadedSavedResult>>().size(), 1U);
    }

    // The save icon's Save as...: each kind on its own as the Videos tab's
    // has them, the run file last. The rules are the tab bar's Export
    // panel's choice (segment_rule_choices, pick_segment_rules).
    QMenu menu;
    panel.save_button()->save_as(menu);
    QStringList texts;
    for (const QAction* action : menu.actions()) {
        if (!action->isSeparator()) {
            texts << action->text();
        }
    }
    EXPECT_EQ(texts, (QStringList{"Results as JSON...", "Results as CSV...", "Per-frame scores as JSON...",
                                  "Per-frame scores as CSV...", "Graphs as ZIP...", "Weak segments as JSON or CSV...",
                                  "Run file (.metrics.json)..."}));
    QSignalSpy exports(&panel, &SavedResultsPanel::export_requested);
    const auto item = [&menu](const QString& text) -> QAction* {
        for (QAction* action : menu.actions()) {
            if (action->text() == text) {
                return action;
            }
        }
        return nullptr;
    };
    ASSERT_NE(item("Per-frame scores as JSON..."), nullptr);
    item("Per-frame scores as JSON...")->trigger();
    EXPECT_EQ(exports.at(0).at(0).value<SavedResultsPanel::Export>(), SavedResultsPanel::Export::frames_json);
    item("Run file (.metrics.json)...")->trigger();  // as the Videos tab's
    EXPECT_EQ(exports.at(1).at(0).value<SavedResultsPanel::Export>(), SavedResultsPanel::Export::run_file);
    EXPECT_EQ(panel.segment_rule_choice(), "in-use");
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions), settings.weak_segment_conditions);
    panel.pick_segment_rules("preset:animation");
    EXPECT_EQ(panel.segment_rule_choice(), "preset:animation");
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions),
              condition_texts(segment_preset("animation")->rules.conditions));

    // Delete: locked during a run of its pair, or of files the run finds in
    // its folder (moved, renamed or copied since), as the run reads and saves
    // its scores there, and of any until the run's folders are known; open
    // during a run of other pairs, of its reference or its test video with
    // another too; else, asked and done.
    QPushButton* remove = nullptr;
    for (QPushButton* button : panel.findChildren<QPushButton*>()) {
        if (button->text() == "Delete...") {
            remove = button;
        }
    }
    ASSERT_NE(remove, nullptr);
    const std::filesystem::path elsewhere = saved.parent_path() / "another";  // another comparison's folder
    panel.set_run_active(true, {{folder / "s.mp4", folder / "other.mp4"}, {folder / "s.mp4", folder / "t.mp4"}});
    panel.add_run_folders({elsewhere}, true);
    EXPECT_FALSE(remove->isEnabled());
    EXPECT_FALSE(remove->toolTip().isEmpty());
    panel.delete_selected();
    EXPECT_TRUE(std::filesystem::exists(saved));
    panel.set_run_active(true, {{folder / "s.mp4", folder / "sub" / ".." / "t.mp4"}});  // the same file
    panel.add_run_folders({elsewhere}, true);
    EXPECT_FALSE(remove->isEnabled());
#ifdef _WIN32
    panel.set_run_active(true, {{folder / "S.MP4", folder / "T.MP4"}});  // any case
    panel.add_run_folders({elsewhere}, true);
    EXPECT_FALSE(remove->isEnabled());
#endif
    // Its test video moved: the run's pair is another, its folder this one.
    panel.set_run_active(true, {{folder / "s.mp4", folder / "moved" / "t.mp4"}});
    EXPECT_FALSE(remove->isEnabled());  // its folders not known yet
    panel.add_run_folders({elsewhere, saved.parent_path() / "." / saved.filename()}, true);
    EXPECT_FALSE(remove->isEnabled());
    // Its reference with another test video, its test video against another
    // reference: other folders, deletable once the run's are known.
    panel.set_run_active(true, {{folder / "s.mp4", folder / "other.mp4"}, {folder / "other.mp4", folder / "t.mp4"}});
    EXPECT_FALSE(remove->isEnabled());
    panel.add_run_folders({elsewhere}, false);  // a save's folder: the start's not known yet
    EXPECT_FALSE(remove->isEnabled());
    panel.add_run_folders({elsewhere}, true);
    EXPECT_TRUE(remove->isEnabled());
    EXPECT_TRUE(remove->toolTip().isEmpty());
    QSignalSpy deleted(&panel, &SavedResultsPanel::deleted);
    panel.delete_selected();  // during a run, of other videos
    EXPECT_FALSE(remove->isEnabled());  // until the queue has run it
    // Said once the queue says it is idle, just after it is.
    EXPECT_TRUE(deleted.count() == 1 || deleted.wait(30000));
    EXPECT_EQ(deleted.count(), 1);
    EXPECT_TRUE(remove->isEnabled() || table.rowCount() == 0);  // the listing after it may be done
    EXPECT_FALSE(std::filesystem::exists(saved));
    panel.set_run_active(false);
}

// Issue #133: the tab took each metric's newest saved score without the
// checks the Videos tab's lookup makes of any: one it refuses and scores
// again (here FFmpeg's XPSNR of a pair compared at 10 bits) was shown,
// opened and exported as valid. It is marked, and left out.
TEST(SavedResultsPanel, ScoresTheVideosTabRefusesAreMarkedAndLeftOut) {
    const test::Folder folder;
    std::ofstream(folder / "s.mkv") << "s";
    std::ofstream(folder / "t.mkv") << "t";
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    ComparisonResult result;
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    result.source_info.pix_fmt = "yuv420p10le";
    result.distorted_info.pix_fmt = "yuv420p10le";
    result.fps = 24.0;
    result.metric_results.add(frame_metric("vmaf", {95.0F, 60.0F, 95.0F}));
    result.metric_results.add(frame_metric("xpsnr", {40.0F, 30.0F, 40.0F}));
    const AnalysisRequest request =
        analysis_request(VmafOptions{}, {"vmaf", "xpsnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    store_result(folder / "cache", result.source, result.distorted, request.recipe, result, "t", request.metrics);

    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list(list_saved_comparisons(folder / "cache"));
    ASSERT_EQ(panel.table()->rowCount(), 1);
    ASSERT_TRUE(panel.table()->item(0, 3)->text().contains("XPSNR"));  // saved
    panel.table()->selectRow(0);
    panel.wait_for_details();
    int marked = 0;
    for (const QLabel* label : panel.findChildren<QLabel*>()) {
        marked += label->text().contains("out of date") ? 1 : 0;
    }
    EXPECT_EQ(marked, 1);
    const std::vector<LoadedSavedResult> read = panel.selected_results();
    ASSERT_EQ(read.size(), 1U);
    EXPECT_TRUE(read[0].result->metric_results.frame("vmaf"));
    EXPECT_FALSE(read[0].result->metric_results.frame("xpsnr"));
}

// Issue #133: every row clicked read all of its comparison's score files in
// full, per-frame scores too, on the window's thread, and Open read them
// again. They are read on a thread of their own, the box saying so
// meanwhile, and what was read is what Open takes.
TEST(SavedResultsPanel, ScoresAreReadOffTheWindowsThreadAndKept) {
    const test::Folder folder;
    std::ofstream(folder / "s.mkv") << "s";
    std::ofstream(folder / "t.mkv") << "t";
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    ComparisonResult result;
    result.source = folder / "s.mkv";
    result.distorted = folder / "t.mkv";
    result.fps = 24.0;
    result.metric_results.add(frame_metric("vmaf", {95.0F, 60.0F, 95.0F}));
    const AnalysisRequest request =
        analysis_request(VmafOptions{}, {"vmaf"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    const std::filesystem::path saved =
        store_result(folder / "cache", result.source, result.distorted, request.recipe, result, "t", request.metrics);
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list(list_saved_comparisons(folder / "cache"));
    const auto texts = [&panel] {
        QStringList found;
        for (const QLabel* label : panel.findChildren<QLabel*>()) {
            found << label->text();
        }
        return found.join(QStringLiteral(" | "));
    };
    panel.table()->selectRow(0);
    EXPECT_TRUE(texts().contains("Reading..."));  // until the thread is done
    panel.wait_for_details();
    EXPECT_FALSE(texts().contains("Reading..."));
    EXPECT_TRUE(texts().contains("VMAF"));

    // Its files gone since: Open takes the scores read.
    for (const auto& file : std::filesystem::directory_iterator(saved)) {
        if (file.path().extension() == ".npz") {
            std::filesystem::remove(file.path());
        }
    }
    const std::vector<LoadedSavedResult> read = panel.selected_results();
    ASSERT_EQ(read.size(), 1U);
    ASSERT_TRUE(read[0].result->metric_results.frame("vmaf"));
    EXPECT_EQ(read[0].result->metric_results.frame("vmaf")->values, (std::vector<float>{95.0F, 60.0F, 95.0F}));
}

// #205: the filter hid a selected row and left it selected: the box kept its
// scores and Open, Export and Delete stayed enabled, acting on nothing (or on
// the rows still shown only). A row the filter hides is no longer selected.
TEST(SavedResultsPanel, ARowTheFilterHidesIsNoLongerSelected) {
    const test::Folder folder;
    std::ofstream(folder / "s.mkv") << "s";
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    const AnalysisRequest request =
        analysis_request(VmafOptions{}, {"vmaf"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
    for (const char* name : {"first.mkv", "second.mkv"}) {
        std::ofstream(folder / name) << name;
        ComparisonResult result;
        result.source = folder / "s.mkv";
        result.distorted = folder / name;
        result.fps = 24.0;
        result.metric_results.add(frame_metric("vmaf", {95.0F, 60.0F}));
        store_result(folder / "cache", result.source, result.distorted, request.recipe, result, name,
                     request.metrics);
    }
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list(list_saved_comparisons(folder / "cache"));
    QTableWidget& table = *panel.table();
    ASSERT_EQ(table.rowCount(), 2);
    const auto row_of = [&table](const QString& name) {
        for (int row = 0; row < table.rowCount(); ++row) {
            if (table.item(row, 0)->text() == name) {
                return row;
            }
        }
        return -1;
    };
    const auto button = [&panel](const QString& text) {
        for (QPushButton* found : panel.findChildren<QPushButton*>()) {
            if (found->text() == text) {
                return found;
            }
        }
        return static_cast<QPushButton*>(nullptr);
    };
    // The comparison the scores' column names, over its cards.
    const auto title = [&panel] {
        QStringList names;
        for (const ElidedLabel* name : panel.findChildren<ElidedLabel*>()) {
            names << "Saved scores of " + name->full_text();
        }
        return names.join(QStringLiteral(" | "));
    };
    ASSERT_NE(button("Open in Metric Graphs"), nullptr);

    table.selectRow(row_of("first.mkv"));
    panel.wait_for_details();
    EXPECT_TRUE(title().contains("Saved scores of first.mkv"));
    EXPECT_TRUE(button("Open in Metric Graphs")->isEnabled());
    panel.filter()->setText("second");
    EXPECT_FALSE(title().contains("Saved scores of first.mkv"));
    EXPECT_FALSE(button("Open in Metric Graphs")->isEnabled());
    EXPECT_FALSE(button("Delete...")->isEnabled());
    EXPECT_TRUE(panel.selected_results().empty());
    // Shown again, it comes back unselected: nothing chosen that isn't seen.
    panel.filter()->clear();
    EXPECT_TRUE(table.selectionModel()->selectedRows().empty());

    // Both selected, one hidden: the one left is the selection, shown in the box.
    table.selectAll();
    panel.filter()->setText("second");
    panel.wait_for_details();
    EXPECT_EQ(table.selectionModel()->selectedRows().size(), 1);
    EXPECT_TRUE(title().contains("Saved scores of second.mkv"));

    // A reference not found: the filter takes its path, not its cell's
    // tooltip, whose sentence ("Not found there: ...") every such row shares.
    std::vector<SavedComparison> list = list_saved_comparisons(folder / "cache");
    for (SavedComparison& saved : list) {
        saved.reference_found = false;
    }
    panel.show_list(list);
    panel.filter()->setText("found");
    EXPECT_TRUE(table.isRowHidden(0));
    EXPECT_TRUE(table.isRowHidden(1));
    panel.filter()->setText("S.MKV");
    EXPECT_FALSE(table.isRowHidden(0));
}

// Issue #133: Delete kept pointers into the list across its question, whose
// event loop a listing finishing in the meantime replaces it in: the folders
// deleted were read from freed memory (a crash, or the wrong folder).
TEST(SavedResultsPanel, DeleteDeletesWhatWasSelectedWhenTheListChangesUnderItsQuestion) {
    const test::Folder folder;
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    const std::filesystem::path v2 = folder / "cache" / "v2";
    std::vector<SavedComparison> list(2);
    for (std::size_t i = 0; i < list.size(); ++i) {
        list[i].folder = v2 / ("comparison" + std::to_string(i));
        list[i].label = "t" + std::to_string(i);
        std::filesystem::create_directories(list[i].folder);
        std::ofstream(list[i].folder / "context.json") << "{}";
    }
    Dialogs dialogs = standard_dialogs(nullptr);
    SavedResultsPanel* shown = nullptr;
    dialogs.message = [&shown, &list](const Message&) {
        shown->show_list({list[1], list[0]});  // the listing done while asked
        return MessageAnswer{0, false};
    };
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    shown = &panel;
    panel.show_list({list[0]});
    panel.table()->selectRow(0);
    panel.delete_selected();
    ASSERT_TRUE(writes.wait_until_idle(std::chrono::seconds(30)));
    EXPECT_FALSE(std::filesystem::exists(list[0].folder));
    EXPECT_TRUE(std::filesystem::exists(list[1].folder));
}

// Issue #133: Delete removed the folders at once, while a run's saves could
// still be queued to be written: a save then brought a deleted folder back.
// Delete goes through the same queue, after them.
TEST(SavedResultsPanel, DeleteWaitsForTheSavesQueuedBeforeIt) {
    const test::Folder folder;
    Settings settings;
    settings.cache_dir = (folder / "cache").string();
    SavedComparison saved;
    saved.folder = folder / "cache" / "v2" / "comparison";
    saved.label = "t";
    std::filesystem::create_directories(saved.folder);
    std::ofstream(saved.folder / "context.json") << "{}";
    Dialogs dialogs = standard_dialogs(nullptr);
    dialogs.message = [](const Message&) { return MessageAnswer{0, false}; };
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list({saved});
    panel.table()->selectRow(0);
    // A run's save, still queued: it writes once let go.
    std::promise<void> let_go;
    writes.submit(QStringLiteral("save"), [held = let_go.get_future().share(), into = saved.folder] {
        held.wait();
        std::filesystem::create_directories(into);
        std::ofstream(into / "vmaf_0.npz") << "score";
    });
    QSignalSpy deleted(&panel, &SavedResultsPanel::deleted);
    panel.delete_selected();
    EXPECT_TRUE(std::filesystem::exists(saved.folder / "context.json"));  // not yet
    let_go.set_value();
    EXPECT_TRUE(deleted.count() == 1 || deleted.wait(30000));  // once the queue is idle
    EXPECT_FALSE(std::filesystem::exists(saved.folder));
    EXPECT_EQ(deleted.count(), 1);
}

TEST(SavedResultsPanel, ClickingARowKeepsTheListWhereItWas) {
    // A long list scrolled to its end: clicking one of its last rows shows
    // its scores without the table shrinking under the pointer and
    // scrolling away from the row clicked.
    std::vector<SavedComparison> list(40);
    for (std::size_t i = 0; i < list.size(); ++i) {
        list[i].folder = std::filesystem::path("saved") / std::to_string(i);
        list[i].label = "t" + std::to_string(i);
        for (const MetricDefinition& metric : metrics()) {
            for (int copy = 0; copy < 2; ++copy) {  // two of each: a list to pick from
                list[i].scores.push_back({list[i].folder / (std::string(metric.key) + std::to_string(copy)),
                                          std::string(metric.key), {}, 0});
            }
        }
    }
    Settings settings;
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list(list);
    panel.resize(900, 700);
    panel.show();
    panel.layout()->activate();
    QTableWidget& table = *panel.table();
    QScrollBar& bar = *table.verticalScrollBar();
    bar.setValue(bar.maximum());
    const int height = table.height();
    const int scrolled = bar.value();
    const int last = table.rowAt(table.viewport()->height() - 2);
    ASSERT_GE(last, 0);
    QTest::mouseClick(table.viewport(), Qt::LeftButton, {}, table.visualItemRect(table.item(last, 0)).center());
    panel.wait_for_details();
    panel.layout()->activate();
    QCoreApplication::processEvents();
    EXPECT_TRUE(panel.used_score("vmaf"));
    EXPECT_EQ(table.height(), height);
    EXPECT_EQ(bar.value(), scrolled);
    EXPECT_EQ(table.rowAt(table.visualItemRect(table.item(last, 0)).center().y()), last);
}

// Two saved scores of VMAF (two models), and one of PSNR, of one comparison.
struct TwoOfVmaf {
    test::Folder folder;
    Settings settings;
    std::filesystem::path older;  // VMAF's first score
    std::filesystem::path newer;  // its second, saved later: the one used unless another is picked

    TwoOfVmaf() {
        std::ofstream(folder / "s.mkv") << "s";
        std::ofstream(folder / "t.mkv") << "t";
        settings.cache_dir = (folder / "cache").string();
        // Saved twice, the first VMAF score kept as a file of its own, saved
        // a day earlier (a score made another way has a file of its own).
        for (const bool first : {true, false}) {
            ComparisonResult result;
            result.source = folder / "s.mkv";
            result.distorted = folder / "t.mkv";
            result.fps = 24.0;
            result.metric_results.add(
                frame_metric("vmaf", first ? std::vector<float>{90.0F, 80.0F} : std::vector<float>{70.0F, 60.0F}));
            result.metric_results.add(frame_metric("psnr", {40.0F, 30.0F}));
            const AnalysisRequest request =
                analysis_request(VmafOptions{}, {"vmaf", "psnr"}, {}, CvvdpSettings{}, VideoInfo{}, VideoInfo{});
            const std::filesystem::path stored = store_result(folder / "cache", result.source, result.distorted,
                                                              request.recipe, result, "t", request.metrics);
            if (first) {
                for (const auto& file : std::filesystem::directory_iterator(stored)) {
                    const std::string name = file.path().filename().string();
                    if (name.starts_with("vmaf_") && name.ends_with(".npz")) {
                        const std::filesystem::path copy = stored / "vmaf_0000000000000000.npz";
                        std::filesystem::copy_file(file.path(), copy);
                        const auto saved_at = std::filesystem::last_write_time(file.path());
                        std::filesystem::last_write_time(copy, saved_at - std::chrono::hours(24));
                    }
                }
            }
        }
        const std::vector<SavedComparison> list = list_saved_comparisons(folder / "cache");
        EXPECT_EQ(list.size(), 1U);
        std::vector<std::filesystem::path> vmaf;
        for (const SavedScoreFile& file : list.at(0).scores) {
            if (file.key == "vmaf") {
                vmaf.push_back(file.file);  // newest first
            }
        }
        EXPECT_EQ(vmaf.size(), 2U);
        if (vmaf.size() == 2) {
            newer = vmaf[0];
            older = vmaf[1];
        }
    }
};

// Where a metric has several saved scores, its card says how many and uses
// the newest; Settings' "each" shows a card for each, whose radio buttons
// pick the one used. The details panel lists every score file with its
// statistics, marks the one used, and a click on another row picks it.
TEST(SavedResultsPanel, SeveralScoresOfAMetricAreCountedAndPicked) {
    TwoOfVmaf saved;
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(saved.settings, dialogs, writes);
    panel.set_tiles(true);
    panel.show_list(list_saved_comparisons(saved.folder / "cache"));
    panel.resize(1200, 800);
    panel.show();
    panel.table()->selectRow(0);
    panel.wait_for_details();
    EXPECT_EQ(panel.used_score("vmaf"), saved.newer);
    const auto* badge = panel.findChild<QLabel*>("saved_vmaf");  // in the link's colour, bold
    ASSERT_NE(badge, nullptr);
    EXPECT_EQ(badge->text(), "2 saved");
    EXPECT_EQ(panel.findChild<QLabel*>("saved_psnr"), nullptr);  // one score: no badge
    EXPECT_EQ(panel.findChild<QRadioButton*>("pick_vmaf"), nullptr);
    EXPECT_TRUE(panel.details_button()->isEnabled());

    // The details: a row a file, the statistics Settings chooses, the one
    // used marked, a click on the other picking it.
    SavedScoresPanel* details = panel.open_details();
    ASSERT_NE(details, nullptr);
    QTableWidget& rows = *details->table();
    ASSERT_EQ(rows.rowCount(), 3);  // VMAF's two, PSNR's one
    QStringList headings;
    for (int column = 0; column < rows.columnCount(); ++column) {
        headings << rows.horizontalHeaderItem(column)->text();
    }
    EXPECT_EQ(headings, (QStringList{"Metric", "Score", "Min", "Max", "Worst 1%", "Model or display", "Made with",
                                     "CPU or GPU", "Saved", "Used"}));
    EXPECT_EQ(rows.item(0, 0)->text().toStdString(), find_metric("vmaf")->short_label);
    EXPECT_EQ(rows.item(1, 0)->text(), "");  // its second score, under it
    EXPECT_EQ(rows.item(0, 2)->text().toStdString(), find_metric("vmaf")->format_value(60.0));  // the newer's minimum
    EXPECT_EQ(rows.item(0, 9)->text(), "\u25CF");
    EXPECT_EQ(rows.item(1, 9)->text(), "");
    emit rows.cellClicked(1, 0);
    EXPECT_EQ(panel.used_score("vmaf"), saved.older);
    EXPECT_EQ(rows.item(1, 9)->text(), "\u25CF");
    const std::vector<LoadedSavedResult> read = panel.selected_results();
    ASSERT_EQ(read.size(), 1U);
    EXPECT_EQ(read[0].result->metric_results.frame("vmaf")->values, (std::vector<float>{90.0F, 80.0F}));
    details->close();

    // "each": a card for each, the one used ticked; the other ticked uses it.
    saved.settings.saved_score_versions = "each";
    panel.scores_settings_changed();
    const QList<QRadioButton*> picks = panel.findChildren<QRadioButton*>("pick_vmaf");
    ASSERT_EQ(picks.size(), 2);
    EXPECT_TRUE(picks[1]->isChecked());  // the older, picked above
    picks[0]->setChecked(true);
    EXPECT_EQ(panel.used_score("vmaf"), saved.newer);
    QCoreApplication::processEvents();  // shown again: the newer ticked
    const QList<QRadioButton*> again = panel.findChildren<QRadioButton*>("pick_vmaf");
    ASSERT_EQ(again.size(), 2);
    EXPECT_TRUE(again[0]->isChecked());
}

// The column is as wide as three cards and keeps every score, however many:
// a comparison of many metrics scrolls in it, and the table beside it keeps
// its height whatever is selected.
TEST(SavedResultsPanel, TheScoresColumnHoldsEveryScoreThreeCardsARow) {
    SavedComparison many;
    many.folder = std::filesystem::path("saved") / "many";
    many.label = "many";
    for (const MetricDefinition& metric : metrics()) {
        if (!metric.is_plane()) {
            many.scores.push_back({many.folder / std::string(metric.key), std::string(metric.key), {}, 0});
        }
    }
    Settings settings;
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.set_tiles(true);
    panel.show_list({many});
    panel.resize(1200, 700);
    panel.show();
    panel.layout()->activate();
    const int height = panel.table()->height();
    panel.table()->selectRow(0);
    panel.wait_for_details();
    panel.layout()->activate();
    QCoreApplication::processEvents();
    EXPECT_EQ(panel.table()->height(), height);
    const QWidget* first = panel.findChild<QWidget*>("tile_vmaf");
    ASSERT_NE(first, nullptr);
    EXPECT_LE(first->geometry().top(), 2);  // at the column's top
    int in_first_row = 0;
    for (const MetricDefinition& metric : metrics()) {
        if (metric.is_plane()) {
            continue;
        }
        const QWidget* tile = panel.findChild<QWidget*>("tile_" + QString::fromStdString(std::string(metric.key)));
        ASSERT_NE(tile, nullptr) << metric.key;  // none left out
        in_first_row += tile->geometry().top() == first->geometry().top() ? 1 : 0;
    }
    EXPECT_EQ(in_first_row, SavedResultsPanel::tiles_a_row);
    for (const QLabel* name : first->findChildren<QLabel*>()) {
        EXPECT_LE(name->sizeHint().width(), first->contentsRect().width()) << name->text().toStdString();
    }
}

// The header's menu: every column ticked, unticking one hides it and ticking
// shows it again, each change told to be saved; the test video's can't be
// unticked.
TEST(SavedResultsPanel, TheHeaderMenuHidesAndShowsColumns) {
    Settings settings;
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    QSignalSpy changed(&panel, &SavedResultsPanel::columns_changed);
    QTableWidget& table = *panel.table();
    QMenu menu;
    panel.fill_column_menu(menu);
    std::vector<QAction*> columns;
    for (QAction* action : menu.actions()) {
        if (action->isCheckable()) {
            columns.push_back(action);
        }
    }
    ASSERT_EQ(columns.size(), static_cast<std::size_t>(table.columnCount()));
    for (int column = 0; column < table.columnCount(); ++column) {
        EXPECT_TRUE(columns[static_cast<std::size_t>(column)]->isChecked());
        EXPECT_FALSE(table.isColumnHidden(column));
    }

    columns[3]->trigger();  // Metrics
    EXPECT_TRUE(table.isColumnHidden(3));
    EXPECT_EQ(panel.hidden_columns(), std::vector<std::string>{"metrics"});
    EXPECT_EQ(changed.count(), 1);
    columns[3]->trigger();
    EXPECT_FALSE(table.isColumnHidden(3));
    EXPECT_TRUE(panel.hidden_columns().empty());
    EXPECT_EQ(changed.count(), 2);

    EXPECT_FALSE(columns[0]->isEnabled());  // the test video's
    columns[0]->trigger();
    EXPECT_FALSE(table.isColumnHidden(0));
    EXPECT_EQ(changed.count(), 2);
}

// The columns hidden and their widths, saved in Settings and read back: the
// same columns hidden and the same widths in a new tab. Settings without
// them (older ones) show every column, fitted to the tab.
TEST(SavedResultsPanel, TheColumnsAreKeptInSettings) {
    const test::Folder folder;
    Settings settings;
    settings.saved_results_hidden_columns = {"reference", "size", "test"};  // "test" can't be hidden
    settings.saved_results_column_widths = {310, 250, 140, 230, 90};
    ASSERT_EQ(settings.save_to(folder / "settings.json"), std::nullopt);
    const Settings loaded = Settings::load_from(folder / "settings.json");
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(loaded, dialogs, writes);
    QTableWidget& table = *panel.table();
    EXPECT_FALSE(table.isColumnHidden(0));
    EXPECT_TRUE(table.isColumnHidden(1));
    EXPECT_FALSE(table.isColumnHidden(2));
    EXPECT_TRUE(table.isColumnHidden(4));
    EXPECT_EQ(table.columnWidth(0), 310);
    EXPECT_EQ(table.columnWidth(2), 140);
    EXPECT_EQ(panel.hidden_columns(), (std::vector<std::string>{"reference", "size"}));
    EXPECT_EQ(panel.column_widths(), settings.saved_results_column_widths);

    // A hidden column shown again at its width.
    QMenu menu;
    panel.fill_column_menu(menu);
    menu.actions()[1]->trigger();
    EXPECT_FALSE(table.isColumnHidden(1));
    EXPECT_EQ(table.columnWidth(1), 250);

    const Settings old = Settings::load_from(folder / "missing.json");
    SavedResultsPanel fitted(old, dialogs, writes);
    EXPECT_TRUE(fitted.hidden_columns().empty());
    EXPECT_TRUE(fitted.column_widths().empty());
    for (int column = 0; column < fitted.table()->columnCount(); ++column) {
        EXPECT_FALSE(fitted.table()->isColumnHidden(column));
    }
}

}  // namespace
}  // namespace vml::app
