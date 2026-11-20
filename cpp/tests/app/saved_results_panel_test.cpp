// The Saved results tab (saved_results_panel.h): the comparisons of the
// results cache listed and filtered, a comparison's saved scores shown, the
// selection read back for Metric Graphs and the exports, the weak segment
// rules picked, and Delete (locked during a run).
#include <QComboBox>
#include <QCoreApplication>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTest>

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

    // Its scores, shown when it's selected, leave the table's height as it
    // was: the row clicked stays under the pointer.
    panel.resize(900, 700);
    panel.show();
    panel.layout()->activate();
    const int height = table.height();
    table.selectRow(0);
    panel.layout()->activate();
    EXPECT_EQ(table.height(), height);
    EXPECT_EQ(panel.score_choice("vmaf"), nullptr);  // one score: nothing to pick
    const std::vector<LoadedSavedResult> read = panel.selected_results();
    ASSERT_EQ(read.size(), 1U);
    EXPECT_EQ(read[0].label, "t");
    EXPECT_EQ(read[0].folder, saved);
    EXPECT_EQ(read[0].result->metric_results.frame("vmaf")->values, (std::vector<float>{95.0F, 60.0F, 95.0F}));

    // A double-click opens it in Metric Graphs.
    QSignalSpy opened(&panel, &SavedResultsPanel::open_requested);
    emit table.cellDoubleClicked(0, 0);
    EXPECT_EQ(opened.count(), 1);

    // The weak segment rules: the profile in use, else the one picked.
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions), settings.weak_segment_conditions);
    panel.rules_choice()->setCurrentIndex(panel.rules_choice()->findData("preset:animation"));
    EXPECT_EQ(condition_texts(panel.segment_rules().conditions),
              condition_texts(segment_preset("animation")->rules.conditions));

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
    SavedResultsPanel panel(settings, dialogs);
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
    SavedResultsPanel panel(settings, dialogs);
    panel.show_list({one});
    panel.resize(900, 700);
    panel.show();
    panel.table()->selectRow(0);
    panel.layout()->activate();
    QCoreApplication::processEvents();
    QComboBox* combo = panel.score_choice("vmaf");
    ASSERT_NE(combo, nullptr);
    EXPECT_LE(combo->geometry().top(), 2);
}

}  // namespace
}  // namespace vml::app
