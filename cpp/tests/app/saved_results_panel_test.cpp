// The Saved Results tab (saved_results_panel.h): the comparisons of the
// results cache listed and filtered, a comparison's saved scores shown, the
// selection read back for Metric Graphs and the exports, the weak segment
// rules picked, and Delete (locked during a run).
#include <QAction>
#include <QComboBox>
#include <QCoreApplication>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
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
#include "test_videos.h"
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
    dialogs.ask_no_default = [](const QString&, const QString&) { return true; };
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
    EXPECT_EQ(panel.score_choice("vmaf"), nullptr);  // one score: nothing to pick
    // Each line starts with the score itself, as Metric Graphs shows it,
    // before how it was made.
    for (const std::string key : {"vmaf", "psnr"}) {
        const std::string score =
            find_metric(key)->format_value(result.metric_results.frame(key)->aggregate()) + " \xC2\xB7 ";
        bool found = false;
        for (const QLabel* label : panel.findChildren<QLabel*>()) {
            found = found || label->text().startsWith(QString::fromStdString(score));
        }
        EXPECT_TRUE(found) << key << ": no line starts with " << score;
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
        EXPECT_EQ(contents->text(), "Export this video's results, graphs and weak segments...");
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
        EXPECT_EQ(texts, (QStringList{"Open in Metric Graphs",
                                      "Export this video's results, graphs and weak segments..."}));
        menu.actions().first()->trigger();
        EXPECT_EQ(opened.count(), 2);
        ASSERT_EQ(opened.at(1).at(0).value<std::vector<LoadedSavedResult>>().size(), 1U);
    }

    // The Export button's menu: the right-click menu's export, then each
    // kind on its own; under Weak segments, the rules, the one picked ticked.
    QMenu& menu = *panel.export_menu();
    panel.fill_export_menu(menu);
    QStringList texts;
    for (const QAction* action : menu.actions()) {
        if (!action->isSeparator()) {
            texts << action->text();
        }
    }
    EXPECT_EQ(texts, (QStringList{"Export this video's results, graphs and weak segments...", "Results",
                                  "Per-frame scores", "Graphs as ZIP...", "Weak segments"}));
    QSignalSpy exports(&panel, &SavedResultsPanel::export_requested);
    menu.actions().first()->trigger();
    ASSERT_EQ(exports.count(), 1);
    EXPECT_EQ(exports.at(0).at(0).value<SavedResultsPanel::Export>(), SavedResultsPanel::Export::contents);
    QMenu* frames = menu.actions().at(3)->menu();
    ASSERT_NE(frames, nullptr);
    frames->actions().first()->trigger();  // As JSON...
    EXPECT_EQ(exports.at(1).at(0).value<SavedResultsPanel::Export>(), SavedResultsPanel::Export::frames_json);
    QMenu* segments = menu.actions().last()->menu();
    ASSERT_NE(segments, nullptr);
    QStringList ticked;
    QAction* animation = nullptr;
    for (QAction* action : segments->actions()) {
        if (action->isChecked()) {
            ticked << action->text();
        }
        if (action->text().contains("Animation")) {
            animation = action;
        }
    }
    EXPECT_EQ(ticked, QStringList{"Profile in use (Default)"});
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions), settings.weak_segment_conditions);
    ASSERT_NE(animation, nullptr);
    animation->trigger();
    EXPECT_EQ(panel.segment_rule_choice(), "preset:animation");
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions),
              condition_texts(segment_preset("animation")->rules.conditions));
    panel.fill_export_menu(menu);  // made again as it opens: the same entries, no menu left behind
    EXPECT_EQ(menu.findChildren<QMenu*>(Qt::FindDirectChildrenOnly).size(), 3);

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
    panel.add_run_folders({elsewhere});
    EXPECT_FALSE(remove->isEnabled());
    EXPECT_FALSE(remove->toolTip().isEmpty());
    panel.delete_selected();
    EXPECT_TRUE(std::filesystem::exists(saved));
    panel.set_run_active(true, {{folder / "s.mp4", folder / "sub" / ".." / "t.mp4"}});  // the same file
    panel.add_run_folders({elsewhere});
    EXPECT_FALSE(remove->isEnabled());
#ifdef _WIN32
    panel.set_run_active(true, {{folder / "S.MP4", folder / "T.MP4"}});  // any case
    panel.add_run_folders({elsewhere});
    EXPECT_FALSE(remove->isEnabled());
#endif
    // Its test video moved: the run's pair is another, its folder this one.
    panel.set_run_active(true, {{folder / "s.mp4", folder / "moved" / "t.mp4"}});
    EXPECT_FALSE(remove->isEnabled());  // its folders not known yet
    panel.add_run_folders({elsewhere, saved.parent_path() / "." / saved.filename()});
    EXPECT_FALSE(remove->isEnabled());
    // Its reference with another test video, its test video against another
    // reference: other folders, deletable once the run's are known.
    panel.set_run_active(true, {{folder / "s.mp4", folder / "other.mp4"}, {folder / "other.mp4", folder / "t.mp4"}});
    EXPECT_FALSE(remove->isEnabled());
    panel.add_run_folders({elsewhere});
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
    dialogs.ask_no_default = [&shown, &list](const QString&, const QString&) {
        shown->show_list({list[1], list[0]});  // the listing done while asked
        return true;
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
    dialogs.ask_no_default = [](const QString&, const QString&) { return true; };
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
    EXPECT_NE(panel.score_choice("vmaf"), nullptr);
    EXPECT_EQ(table.height(), height);
    EXPECT_EQ(bar.value(), scrolled);
    EXPECT_EQ(table.rowAt(table.visualItemRect(table.item(last, 0)).center().y()), last);
}

TEST(SavedResultsPanel, AFewScoresSitAtTheTopOfTheBox) {
    // The box is as tall as every metric needs: a comparison of one metric
    // has its row at the top, not spread down the spare height.
    SavedComparison one;
    one.folder = std::filesystem::path("saved") / "one";
    one.label = "one";
    for (int copy = 0; copy < 2; ++copy) {
        one.scores.push_back({one.folder / ("vmaf" + std::to_string(copy)), "vmaf", {}, 0});
    }
    Settings settings;
    const Dialogs dialogs = standard_dialogs(nullptr);
    FileWrites writes;
    SavedResultsPanel panel(settings, dialogs, writes);
    panel.show_list({one});
    panel.resize(900, 700);
    panel.show();
    panel.table()->selectRow(0);
    panel.wait_for_details();
    panel.layout()->activate();
    QCoreApplication::processEvents();
    QComboBox* combo = panel.score_choice("vmaf");
    ASSERT_NE(combo, nullptr);
    EXPECT_LE(combo->geometry().top(), 2);
}

// The header's menu: every column ticked, unticking one hides it and ticking
// shows it again, each change told to be saved; the test video's can't be
// unticked.
TEST(SavedResultsPanel, TheHeaderMenuHidesAndShowsColumns) {
    Settings settings;
    const Dialogs dialogs = standard_dialogs(nullptr);
    SavedResultsPanel panel(settings, dialogs);
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
    SavedResultsPanel panel(loaded, dialogs);
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
    SavedResultsPanel fitted(old, dialogs);
    EXPECT_TRUE(fitted.hidden_columns().empty());
    EXPECT_TRUE(fitted.column_widths().empty());
    for (int column = 0; column < fitted.table()->columnCount(); ++column) {
        EXPECT_FALSE(fitted.table()->isColumnHidden(column));
    }
}

}  // namespace
}  // namespace vml::app
