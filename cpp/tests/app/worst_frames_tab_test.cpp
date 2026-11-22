// The Worst Frames tab (issue #24): a video's worst frames and weak
// segments, with the conditions, the profiles and the presets that Metric
// Graphs' Weak segments dialog had.
#include <QAction>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QTabBar>
#include <QTableWidget>

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chart.h"
#include "worst_frames_tab.h"
#include "vml/results.h"
#include "vml/segment_profiles.h"
#include "vml/settings.h"

namespace vml::app {
namespace {

// A result with VMAF's `values`, a frame each at 30 a second.
std::shared_ptr<const ComparisonResult> vmaf_result(const std::string& name, const std::vector<float>& values) {
    ComparisonResult result;
    result.source = "source.mkv";
    result.distorted = name;
    result.fps = 30.0;
    FrameMetricResult vmaf{"vmaf", {}, {}, values, current_ffmpeg_provenance("vmaf", "9.0.1")};
    for (std::size_t i = 0; i < values.size(); ++i) {
        vmaf.frame.push_back(static_cast<int>(i));
        vmaf.time.push_back(static_cast<double>(i) / 30.0);
    }
    result.metric_results.add(std::move(vmaf));
    return std::make_shared<const ComparisonResult>(std::move(result));
}

QString header(const QTableWidget& table, int column) { return table.horizontalHeaderItem(column)->text(); }

// The tab with `settings`' profiles, handed back after each save as the
// window does, and `answers` for the dialogs it asks.
struct Bench {
    explicit Bench(Settings& kept, Dialogs answers = {}) : settings(kept), dialogs(std::move(answers)) {
        hand_back();
        QObject::connect(&tab, &WorstFramesTab::rules_saved, [this](const std::string& profile, const SegmentRules& rules) {
            saved.emplace_back(profile, rules);
            store_segment_profile(settings, profile, rules);
            hand_back();
        });
    }
    void hand_back() { tab.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name); }
    QDoubleSpinBox* threshold() const { return tab.editor()->findChild<QDoubleSpinBox*>(); }
    QAction* preset(const QString& name) const {
        for (QAction* action : tab.preset_button()->menu()->actions()) {
            if (action->text() == name) {
                return action;
            }
        }
        return nullptr;
    }

    Settings& settings;
    Dialogs dialogs;
    WorstFramesTab tab{dialogs};
    std::vector<std::pair<std::string, SegmentRules>> saved;
};

TEST(WorstFramesTab, ListsTheDipsAndTheSegmentsAndOpensThemInVideoCompare) {
    // The worst frame of each dip, within the range; the weak segments by
    // the metric's first band of worse frames (no profile has conditions);
    // a double-click asks for that frame of that video in Video Compare.
    std::vector<float> values(90, 95.0F);
    values[9] = 60.0F;
    values[10] = 40.0F;
    values[70] = 50.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    ASSERT_NE(tab.chosen_video(), nullptr);
    ASSERT_NE(tab.metric(), nullptr);
    EXPECT_EQ(std::string(tab.metric()->key), "vmaf");

    QTableWidget& list = *tab.worst_table();
    ASSERT_EQ(list.rowCount(), 2);
    EXPECT_EQ(list.item(0, 0)->text(), "10");
    EXPECT_EQ(list.item(1, 0)->text(), "70");
    EXPECT_EQ(list.item(0, 2)->text(), "40.00");
    EXPECT_EQ(tab.rules().conditions.front().text(), "vmaf<85");
    ASSERT_EQ(tab.segments().size(), 2U);
    EXPECT_EQ(tab.segments()[0].first_frame, 9);
    EXPECT_EQ(tab.segments()[0].last_frame, 10);
    EXPECT_EQ(tab.segments()[1].first_frame, 70);
    EXPECT_EQ(tab.segments_table()->rowCount(), 2);
    // The videos' list: each one's segments and worst score.
    EXPECT_EQ(tab.videos_table()->item(0, 1)->text(), "2");
    EXPECT_EQ(tab.videos_table()->item(0, 2)->text(), "40.00");
    // Each condition's frames, said beside the conditions; a "Weak by"
    // column only with several conditions, any of which makes a frame weak.
    EXPECT_TRUE(tab.conditions_note()->text().startsWith("VMAF v0.6.1 < 85.00: "))
        << tab.conditions_note()->text().toStdString();
    for (int column = 0; column < tab.segments_table()->columnCount(); ++column) {
        EXPECT_NE(header(*tab.segments_table(), column), "Weak by");
    }
    // The graph: the segments shaded.
    ASSERT_NE(tab.chart(), nullptr);
    EXPECT_EQ(tab.chart()->shaded().size(), 2U);

    std::pair<std::string, int> asked;
    WorstFrameList handed;
    QObject::connect(&tab, &WorstFramesTab::open_requested,
                     [&](const std::string& identity, int frame, const WorstFrameList& stops) {
                         asked = {identity, frame};
                         handed = stops;
                     });
    list.selectRow(1);
    emit list.cellDoubleClicked(1, 0);
    const std::pair<std::string, int> wanted{"row:a", 70};
    EXPECT_EQ(asked, wanted);
    // The list goes with it, for Video Compare's Previous and Next worst.
    EXPECT_EQ(handed.identity, "row:a");
    ASSERT_EQ(handed.stops.size(), 2U);
    EXPECT_EQ(handed.stops[0].frame, 10);
    EXPECT_EQ(handed.stops[0].score, "40.00");
    // A segment: its first frame, without a list.
    tab.segments_table()->selectRow(1);
    emit tab.segments_table()->cellDoubleClicked(1, 0);
    EXPECT_EQ(asked, (std::pair<std::string, int>{"row:a", 70}));
    EXPECT_TRUE(handed.stops.empty());
    ASSERT_TRUE(tab.chart()->highlight());  // the segment picked, shaded again

    tab.set_range(std::pair(50, 89));
    ASSERT_EQ(list.rowCount(), 1);
    EXPECT_EQ(list.item(0, 0)->text(), "70");
    EXPECT_EQ(tab.segments().size(), 1U);

    // The exports are named after the metrics they cover.
    QStringList suggested;
    Dialogs dialogs;
    dialogs.save_file = [&suggested](const QString&, const QString& name, const QString&) {
        suggested.append(name);
        return QString();  // Cancel
    };
    Bench exporting(settings, dialogs);
    exporting.tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    for (QPushButton* button : exporting.tab.findChildren<QPushButton*>()) {
        if (button->text() == "Export") {
            for (QAction* action : button->menu()->actions()) {
                if (action->text().startsWith("Export ") && action->text() != "Export all videos") {
                    action->trigger();
                }
            }
        }
    }
    EXPECT_EQ(suggested, (QStringList{"a.vmaf.segments.json", "a.vmaf.segments.csv"}));
}

TEST(WorstFramesTab, DetailsTellsTheFrameUnderThePointerElseTheRowSelected) {
    // Details: the frame the pointer is on, with why it is weak; a worst
    // frame's dot a few pixels away taken instead (a frame further off is
    // not); once the pointer leaves, the row selected in either list.
    std::vector<float> values(90, 95.0F);
    values[10] = 40.0F;
    values[40] = 80.0F;
    values[70] = 50.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.resize(1400, 900);
    tab.show();
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    ASSERT_NE(tab.chart(), nullptr);
    EXPECT_TRUE(tab.details()->text().isEmpty());  // the hint shown instead

    // Frame 40: weak, in a segment of its own, not among the worst listed.
    emit tab.chart()->hovered(40.0 / 30.0, 80.0);
    QString text = tab.details()->text();
    EXPECT_TRUE(text.startsWith("Frame 40   0:00:01.333")) << text.toStdString();
    EXPECT_TRUE(text.contains("In weak segment 40 to 40"));
    EXPECT_TRUE(text.contains("Weak by VMAF v0.6.1 < 85.00"));
    EXPECT_FALSE(text.contains("Worst frame"));
    // A frame away from 70's dot is itself; a pixel or two away, the dot's.
    emit tab.chart()->hovered(68.0 / 30.0, 95.0);
    EXPECT_TRUE(tab.details()->text().startsWith("Frame 68 ")) << tab.details()->text().toStdString();
    EXPECT_TRUE(tab.details()->text().contains("No condition holds"));
    const int dot = tab.chart()->pixel_for_time(70.0 / 30.0);
    emit tab.chart()->hovered((70.0 - 0.1) / 30.0, 95.0);
    ASSERT_LE(std::abs(tab.chart()->pixel_for_time((70.0 - 0.1) / 30.0) - dot), 6);
    text = tab.details()->text();
    EXPECT_TRUE(text.startsWith("Frame 70 ")) << text.toStdString();
    EXPECT_TRUE(text.contains("Worst frame 2 of 2"));

    // Left: nothing selected, nothing told; then the rows selected.
    emit tab.chart()->left();
    EXPECT_TRUE(tab.details()->text().isEmpty());
    tab.worst_table()->selectRow(0);
    EXPECT_TRUE(tab.details()->text().startsWith("Frame 10 ")) << tab.details()->text().toStdString();
    EXPECT_TRUE(tab.details()->text().contains("Worst frame 1 of 2"));
    tab.segments_table()->selectRow(1);  // 40 to 40
    EXPECT_TRUE(tab.details()->text().startsWith("Segment 40 to 40")) << tab.details()->text().toStdString();
}

TEST(WorstFramesTab, StartsFromAProfileAndSavesOnlyWhenAsked) {
    // The profile in use (Settings > Weak segments) is where the conditions
    // start; another can be picked. What is changed here holds until Save to
    // profile (the profile picked: its conditions on metrics this video has
    // none of left as they were) or Save as new profile.
    std::vector<float> values(90, 95.0F);
    values[20] = 75.0F;
    values[40] = 60.0F;
    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    settings.weak_segments_all = true;
    settings.weak_segment_merge_frames = 3;
    settings.weak_segment_profiles_json = R"([{"name": "Strict", "conditions": ["vmaf<92"]}])";
    QStringList names{"Mine", "strict"};
    QString alerted;
    Dialogs dialogs;
    dialogs.ask_text = [&names](const QString&, const QString&, const QString&) -> std::optional<QString> {
        return names.takeFirst();
    };
    dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    Bench bench(settings, dialogs);
    WorstFramesTab* shown = &bench.tab;
    shown->set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});

    EXPECT_EQ(shown->profile_combo()->currentText(), "Default");
    const SegmentRules start = shown->rules();
    EXPECT_EQ(condition_texts(start.conditions), std::vector<std::string>{"vmaf<70"});
    EXPECT_TRUE(start.all);
    EXPECT_EQ(start.merge_frames, 3);
    ASSERT_EQ(shown->segments().size(), 1U);  // frame 40
    EXPECT_FALSE(shown->save_button()->isEnabled());  // as saved

    QDoubleSpinBox* threshold = shown->editor()->findChild<QDoubleSpinBox*>();
    ASSERT_NE(threshold, nullptr);
    threshold->setValue(80.0);
    EXPECT_EQ(shown->segments().size(), 2U);  // frames 20 and 40, at once
    EXPECT_TRUE(bench.saved.empty());  // not saved by changing it
    ASSERT_TRUE(shown->save_button()->isEnabled());
    shown->save_button()->click();
    ASSERT_EQ(bench.saved.size(), 1U);
    EXPECT_EQ(bench.saved[0].first, "");
    EXPECT_EQ(condition_texts(bench.saved[0].second.conditions),
              (std::vector<std::string>{"vmaf<80", "ssimulacra2<60"}));
    EXPECT_TRUE(bench.saved[0].second.ranges.empty());
    EXPECT_FALSE(shown->save_button()->isEnabled());

    // Another profile picked: its rules, as saved.
    shown->profile_combo()->setCurrentIndex(shown->profile_combo()->findText("Strict"));
    EXPECT_EQ(condition_texts(shown->rules().conditions), std::vector<std::string>{"vmaf<92"});
    EXPECT_FALSE(shown->rules().all);
    EXPECT_FALSE(shown->save_button()->isEnabled());
    // Saved as a new profile, which is then the one picked; a name taken
    // (any case) is refused.
    shown->save_new_button()->click();
    ASSERT_EQ(bench.saved.size(), 2U);
    EXPECT_EQ(bench.saved[1].first, "Mine");
    EXPECT_EQ(shown->profile_combo()->currentText(), "Mine");
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Mine")->rules.conditions),
              std::vector<std::string>{"vmaf<92"});
    shown->save_new_button()->click();
    EXPECT_EQ(bench.saved.size(), 2U);
    EXPECT_EQ(alerted, "Profile not saved");

    // Export all videos asks the window for its report of every video, by
    // the rules shown: unsaved ones too.
    threshold = shown->editor()->findChild<QDoubleSpinBox*>();
    threshold->setValue(85.0);
    std::vector<SegmentRules> reports;
    QObject::connect(shown, &WorstFramesTab::report_requested,
                     [&reports](const SegmentRules& rules) { reports.push_back(rules); });
    for (QPushButton* button : shown->findChildren<QPushButton*>()) {
        if (button->text() == "Export") {
            for (QAction* action : button->menu()->actions()) {
                if (action->text() == "Export all videos") {
                    action->trigger();
                }
            }
        }
    }
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(condition_texts(reports[0].conditions), std::vector<std::string>{"vmaf<85"});
}

TEST(WorstFramesTab, TakesProfilesChangedInSettings) {
    // Issue #135: a profile made in Settings meanwhile is not replaced by Save
    // as new profile under its name, and Save to profile on one deleted there
    // does not bring it back.
    std::vector<float> values(90, 95.0F);
    values[40] = 60.0F;
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<70"};
    settings.weak_segment_profiles_json = R"([{"name": "Strict", "conditions": ["vmaf<92"]}])";
    Dialogs dialogs;
    QString alerted;
    dialogs.ask_text = [](const QString&, const QString&, const QString&) -> std::optional<QString> {
        return QStringLiteral("film");
    };
    dialogs.alert = [&alerted](const QString& title, const QString&) { alerted = title; };
    WorstFramesTab tab(dialogs);
    const auto hand_back = [&] {
        tab.set_segment_profiles(segment_profiles(settings), segment_profile_in_use(settings).name);
    };
    hand_back();
    QObject::connect(&tab, &WorstFramesTab::rules_saved, [&](const std::string& profile, const SegmentRules& rules) {
        store_segment_profile(settings, profile, rules);
        hand_back();
    });
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    tab.profile_combo()->setCurrentIndex(tab.profile_combo()->findText("Strict"));

    // Meanwhile, in Settings: Film made, Strict deleted.
    store_segment_profiles(settings,
                           {segment_profiles(settings).front(), {"Film", SegmentRules{{{"vmaf", '<', 80.0}}}}});
    hand_back();
    EXPECT_EQ(tab.profile_combo()->findText("Strict"), -1);
    EXPECT_EQ(tab.profile_combo()->currentText(), "Default");  // the one in use
    EXPECT_EQ(condition_texts(tab.rules().conditions), std::vector<std::string>{"vmaf<92"});  // kept

    tab.save_new_button()->click();
    EXPECT_EQ(alerted, "Profile not saved");
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "Film")->rules.conditions),
              std::vector<std::string>{"vmaf<80"});
    ASSERT_TRUE(tab.save_button()->isEnabled());
    tab.save_button()->click();
    EXPECT_FALSE(segment_profile_named(settings, "Strict"));
    EXPECT_EQ(condition_texts(segment_profile_named(settings, "")->rules.conditions),
              std::vector<std::string>{"vmaf<92"});
}

TEST(WorstFramesTab, LoadsAPresetWithoutSavingIt) {
    // From preset loads a built-in preset's rules in place of those shown:
    // the segments follow at once, nothing is saved until Save to profile,
    // which then keeps the preset's conditions on metrics this video has
    // none of too.
    std::vector<float> values(90, 95.0F);
    for (int frame = 20; frame < 30; ++frame) {
        values[static_cast<std::size_t>(frame)] = 80.0F;  // ten frames below 85
    }
    values[50] = 80.0F;  // two, shorter than Grainy film's six
    values[51] = 80.0F;
    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    EXPECT_TRUE(tab.segments().empty());  // nothing below 70
    for (QAction* action : tab.preset_button()->menu()->actions()) {
        // HDR's one condition is on CVVDP, which this video has no scores of.
        EXPECT_EQ(action->isEnabled(), action->text() != "HDR") << action->text().toStdString();
    }
    QAction* grainy = bench.preset("Grainy film");
    ASSERT_NE(grainy, nullptr);
    grainy->trigger();
    const SegmentRules shown = tab.rules();
    EXPECT_EQ(condition_texts(shown.conditions), std::vector<std::string>{"vmaf<85"});
    EXPECT_EQ(shown.min_frames, 6);
    EXPECT_EQ(shown.merge_frames, 12);
    ASSERT_EQ(tab.segments().size(), 1U);
    EXPECT_EQ(tab.segments()[0].first_frame, 20);
    EXPECT_EQ(tab.segments()[0].last_frame, 29);
    EXPECT_TRUE(bench.saved.empty());  // not saved by loading it
    ASSERT_TRUE(tab.save_button()->isEnabled());

    tab.save_button()->click();
    ASSERT_EQ(bench.saved.size(), 1U);
    EXPECT_EQ(condition_texts(bench.saved[0].second.conditions),
              (std::vector<std::string>{"vmaf<85", "ssimulacra2<55"}));
    EXPECT_EQ(bench.saved[0].second.min_frames, 6);
    EXPECT_FALSE(tab.save_button()->isEnabled());
}

TEST(WorstFramesTab, SavesTheHiddenConditionsSettingsHasNow) {
    // Save to profile keeps the profile's conditions on metrics this video
    // has none of as Settings has them now: one removed there meanwhile is
    // not put back. A preset loaded keeps its own.
    Settings settings;
    settings.weak_segment_conditions = {"ssimulacra2<60", "vmaf<70"};
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", std::vector<float>(90, 95.0F))}});
    EXPECT_FALSE(tab.save_button()->isEnabled());  // as saved

    settings.weak_segment_conditions = {"vmaf<70"};  // in Settings: SSIMULACRA2's removed
    bench.hand_back();
    EXPECT_FALSE(tab.save_button()->isEnabled());  // still as saved: nothing to put back

    bench.preset("Grainy film")->trigger();
    settings.weak_segment_conditions = {"vmaf<75"};  // changed in Settings again
    bench.hand_back();
    ASSERT_TRUE(tab.save_button()->isEnabled());
    tab.save_button()->click();
    ASSERT_EQ(bench.saved.size(), 1U);
    EXPECT_EQ(condition_texts(bench.saved[0].second.conditions),
              (std::vector<std::string>{"vmaf<85", "ssimulacra2<55"}));
}

TEST(WorstFramesTab, KeepsTheChosenVideoWhileItIsRecalculated) {
    // Recalculating a video takes its line away until its new scores come;
    // the tab shows another video meanwhile, then the chosen one again.
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    const WeakFramesVideo a{"row:a", "a", vmaf_result("a.mkv", {95, 40, 95})};
    tab.set_videos({a, {"row:b", "b", vmaf_result("b.mkv", {95, 50, 95})}});
    tab.videos_table()->setCurrentCell(1, 0);
    ASSERT_EQ(tab.chosen_video()->identity, "row:b");

    tab.set_videos({a});
    EXPECT_EQ(tab.chosen_video()->identity, "row:a");
    tab.set_videos({a, {"row:b", "b", vmaf_result("b.mkv", {95, 55, 95})}});
    EXPECT_EQ(tab.chosen_video()->identity, "row:b");
    EXPECT_EQ(tab.worst_table()->item(0, 2)->text(), "55.00");
}

TEST(WorstFramesTab, NamesTheVideosAsMetricGraphsDoes) {
    // The videos list takes Metric Graphs' shown names (the Video names
    // setting), the whole name in the tooltip; without one, the whole name.
    Settings settings;
    Bench bench(settings);
    const std::vector<float> values(90, 95.0F);
    bench.tab.set_videos({{"row:a", "Example Show - S01E03 - CRF 20", vmaf_result("a.mkv", values), "…CRF 20"},
                          {"row:b", "b", vmaf_result("b.mkv", values)}});
    const QTableWidget* videos = bench.tab.videos_table();
    EXPECT_EQ(videos->item(0, 0)->text(), "…CRF 20");
    EXPECT_EQ(videos->item(0, 0)->toolTip(), "Example Show - S01E03 - CRF 20");
    EXPECT_EQ(videos->item(1, 0)->text(), "b");
}

}  // namespace
}  // namespace vml::app
