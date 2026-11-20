// The Saved results tab (saved_results_panel.h): the comparisons of the
// results cache listed and filtered, a comparison's saved scores shown, the
// selection read back for Metric Graphs and the exports, the weak segment
// rules picked, and Delete (locked during a run).
#include <QAction>
#include <QComboBox>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QSignalSpy>
#include <QTableWidget>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "dialogs.h"
#include "saved_results_panel.h"
#include "test_videos.h"
#include "vml/analysis_request.h"
#include "vml/metric_cache.h"
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
    SavedResultsPanel panel(settings, dialogs);
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

    table.selectRow(0);
    EXPECT_EQ(panel.score_choice("vmaf"), nullptr);  // one score: nothing to pick
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

    // A double-click opens it in Metric Graphs.
    QSignalSpy opened(&panel, &SavedResultsPanel::open_requested);
    emit table.cellDoubleClicked(0, 0);
    EXPECT_EQ(opened.count(), 1);

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

    // Delete: locked during a run; else, asked and done.
    QPushButton* remove = nullptr;
    for (QPushButton* button : panel.findChildren<QPushButton*>()) {
        if (button->text() == "Delete...") {
            remove = button;
        }
    }
    ASSERT_NE(remove, nullptr);
    panel.set_run_active(true);
    EXPECT_FALSE(remove->isEnabled());
    panel.delete_selected();
    EXPECT_TRUE(std::filesystem::exists(saved));
    panel.set_run_active(false);
    EXPECT_TRUE(remove->isEnabled());
    QSignalSpy deleted(&panel, &SavedResultsPanel::deleted);
    panel.delete_selected();
    EXPECT_EQ(deleted.count(), 1);
    EXPECT_FALSE(std::filesystem::exists(saved));
}

}  // namespace
}  // namespace vml::app
