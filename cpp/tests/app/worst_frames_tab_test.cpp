// The Worst Frames tab (issue #24): a video's worst frames and weak
// segments, with the conditions, the profiles and the presets that Metric
// Graphs' Weak segments dialog had.
#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QTabBar>
#include <QTableWidget>
#include <QToolButton>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chart.h"
#include "graph_legend.h"
#include "worst_frames_tab.h"

#include "quick_save.h"
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

// Every Export menu's items, as each shows them when opened.
QList<QAction*> export_items(const WorstFramesTab& tab) {
    // Every card's save icon's Save as... items.
    QList<QAction*> items;
    for (SaveButton* button : tab.findChildren<SaveButton*>()) {
        auto* menu = new QMenu(button);
        button->save_as(*menu);
        items += menu->actions();
    }
    return items;
}

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
    ASSERT_NE(tab.shown_metric(), nullptr);
    EXPECT_EQ(std::string(tab.shown_metric()->key), "vmaf");

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
    // The two lists, of up to 1000 rows, are sized once a fill ends
    // (fit_list), not measured again for each cell set: a long video's
    // segments took over a minute on the window's thread each time the tab
    // was shown again. Their columns still fit what they hold.
    for (QTableWidget* table : {tab.worst_table(), tab.segments_table()}) {
        EXPECT_EQ(table->horizontalHeader()->sectionResizeMode(0), QHeaderView::Interactive);
        EXPECT_GE(table->horizontalHeader()->sectionSize(0),
                  table->fontMetrics().horizontalAdvance(table->horizontalHeaderItem(0)->text()));
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

    // The exports are named after the metrics they cover; each is on a
    // right-click on its card's save icon.
    QStringList suggested;
    Dialogs dialogs;
    dialogs.save_file = [&suggested](const QString&, const QString& name, const QString&) {
        suggested.append(name);
        return QString();  // Cancel
    };
    dialogs.ask_text = [](const QString&, const QString&, const QString& text) { return std::optional<QString>(text); };
    Bench exporting(settings, dialogs);
    exporting.tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    QStringList labels;
    for (QAction* action : export_items(exporting.tab)) {
        labels.append(action->text());
        if (action->isEnabled()) {
            action->trigger();
        }
    }
    labels.sort();
    EXPECT_EQ(labels, (QStringList{"Graph as PNG...", "Weak segments as CSV...", "Weak segments as JSON...",
                                   "Worst frames as CSV...", "Worst frames as JSON..."}));
    suggested.sort();
    EXPECT_EQ(suggested, (QStringList{"a.vmaf.segments.csv", "a.vmaf.segments.json", "a.vmaf.worst.csv",
                                      "a.vmaf.worst.json", "a_vmaf.png"}));
}

// A click on a list's heading sorts it by that column, by number (120 after
// 70, where text would put it before): ascending, descending, then the
// list's own order. A row keeps its frame wherever it is sorted to: the one
// selected stays selected, a double-click opens that frame, and Video
// Compare steps through the list as it is shown. A segment's row opens its
// own segment.
TEST(WorstFramesTab, SortsItsListsAndEachRowKeepsItsFrame) {
    std::vector<float> values(150, 95.0F);
    values[10] = 40.0F;
    values[70] = 50.0F;
    values[120] = 45.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    QTableWidget& list = *tab.worst_table();
    const auto frames = [&list] {
        QStringList shown;
        for (int row = 0; row < list.rowCount(); ++row) {
            shown << list.item(row, 0)->text();
        }
        return shown;
    };
    ASSERT_EQ(frames(), (QStringList{"10", "120", "70"}));  // worst first
    QHeaderView* heading = list.horizontalHeader();

    emit heading->sectionClicked(0);  // Frame, ascending
    EXPECT_EQ(frames(), (QStringList{"10", "70", "120"}));
    EXPECT_EQ(heading->sortIndicatorSection(), 0);  // its arrow on Frame
    list.selectRow(2);  // frame 120
    emit heading->sectionClicked(0);  // descending
    EXPECT_EQ(frames(), (QStringList{"120", "70", "10"}));
    ASSERT_EQ(list.selectedItems().size(), list.columnCount());
    EXPECT_EQ(list.selectedItems().front()->row(), 0);  // still frame 120

    std::pair<std::string, int> asked;
    WorstFrameList handed;
    QObject::connect(&tab, &WorstFramesTab::open_requested,
                     [&](const std::string& identity, int frame, const WorstFrameList& stops) {
                         asked = {identity, frame};
                         handed = stops;
                     });
    emit list.cellDoubleClicked(0, 0);
    EXPECT_EQ(asked, (std::pair<std::string, int>{"row:a", 120}));
    ASSERT_EQ(handed.stops.size(), 3U);
    EXPECT_EQ(handed.stops[0].frame, 120);  // stepped through as shown
    EXPECT_EQ(handed.stops[2].frame, 10);

    emit heading->sectionClicked(0);  // the list's own order again
    EXPECT_EQ(frames(), (QStringList{"10", "120", "70"}));
    EXPECT_EQ(heading->sortIndicatorSection(), -1);  // no arrow

    // Weak segments, by First descending: the last segment first, and its
    // row opens it.
    QTableWidget& segments = *tab.segments_table();
    ASSERT_EQ(tab.segments().size(), 3U);
    emit segments.horizontalHeader()->sectionClicked(0);
    emit segments.horizontalHeader()->sectionClicked(0);
    EXPECT_EQ(segments.item(0, 0)->text(), QString::number(tab.segments()[2].first_frame));
    segments.selectRow(0);
    emit segments.cellDoubleClicked(0, 0);
    EXPECT_EQ(asked, (std::pair<std::string, int>{"row:a", tab.segments()[2].first_frame}));
    EXPECT_TRUE(handed.stops.empty());
}

// Two quick clicks on a heading are two clicks: Windows sends the second as a
// double-click (press, release, double-click, release), which the heading
// took as no click at all, so a list clicked twice quickly stayed ascending
// (the user, 2026-11-26: "eating mouse clicks or lagging").
TEST(WorstFramesTab, AQuickSecondClickOnAHeadingSortsAgain) {
    std::vector<float> values(150, 95.0F);
    values[10] = 40.0F;
    values[70] = 50.0F;
    values[120] = 45.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    QTableWidget& list = *tab.worst_table();
    list.resize(600, 300);
    list.show();
    QHeaderView* heading = list.horizontalHeader();
    QWidget* surface = heading->viewport();
    const QPoint at(heading->sectionViewportPosition(0) + heading->sectionSize(0) / 2, heading->height() / 2);
    const auto send = [&](QEvent::Type type) {
        QMouseEvent event(type, at, surface->mapToGlobal(at), Qt::LeftButton,
                          type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(surface, &event);
    };
    const auto frames = [&list] {
        QStringList shown;
        for (int row = 0; row < list.rowCount(); ++row) {
            shown << list.item(row, 0)->text();
        }
        return shown;
    };
    send(QEvent::MouseButtonPress);
    send(QEvent::MouseButtonRelease);
    EXPECT_EQ(frames(), (QStringList{"10", "70", "120"}));  // the first click: ascending
    send(QEvent::MouseButtonDblClick);  // the second, as Windows sends it within its double-click time
    send(QEvent::MouseButtonRelease);
    EXPECT_EQ(frames(), (QStringList{"120", "70", "10"}));  // descending
    send(QEvent::MouseButtonPress);  // a third, slower
    send(QEvent::MouseButtonRelease);
    EXPECT_EQ(frames(), (QStringList{"10", "120", "70"}));  // the list's own order
}

// A click on a heading sorts the list where it is: no column moves or
// changes width, so the pointer stays on the heading it clicked (the user,
// 2026-11-26: the sort arrow appearing widened the columns and shifted
// them under the pointer).
TEST(WorstFramesTab, SortingMovesNoColumn) {
    std::vector<float> values(150, 95.0F);
    values[10] = 40.0F;
    values[70] = 50.0F;
    values[120] = 45.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.resize(1400, 900);
    tab.show();
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    for (QTableWidget* list : {tab.worst_table(), tab.segments_table()}) {
        QHeaderView* heading = list->horizontalHeader();
        const auto edges = [heading] {
            std::vector<std::pair<int, int>> shown;
            for (int c = 0; c < heading->count(); ++c) {
                shown.emplace_back(heading->sectionViewportPosition(c), heading->sectionSize(c));
            }
            return shown;
        };
        const auto before = edges();
        for (int click = 0; click < 3; ++click) {  // ascending, descending, its own order
            for (int column = 0; column < heading->count(); ++column) {
                emit heading->sectionClicked(column);
                EXPECT_EQ(edges(), before) << "column " << column << ", click " << click;
                // and not by hiding the arrow: shown, on the column just clicked
                // (each click is on another column than the last: ascending)
                EXPECT_TRUE(heading->isSortIndicatorShown());
                EXPECT_EQ(heading->sortIndicatorSection(), column);
            }
        }
    }
}

// Beside 10, 25 and 50, a number of the user's own: Custom... and its number
// box, the count kept in Settings and shown as Custom again when it is no
// preset; a preset picked hides the box.
TEST(WorstFramesTab, ListsACustomNumberOfDips) {
    std::vector<float> values(300, 95.0F);
    for (int dip = 0; dip < 8; ++dip) {
        values[static_cast<std::size_t>(5 + dip * 35)] = 40.0F + static_cast<float>(dip);  // 8 dips, 35 frames apart
    }
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    std::vector<int> told;
    QObject::connect(&tab, &WorstFramesTab::count_changed, [&told](int count) { told.push_back(count); });
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    ASSERT_EQ(tab.worst_table()->rowCount(), 8);  // 10 asked, 8 there
    EXPECT_TRUE(tab.count_spin()->isHidden());

    tab.set_worst_count(3);  // Settings' count, no preset
    EXPECT_EQ(tab.count_combo()->currentText(), "Custom...");
    EXPECT_FALSE(tab.count_spin()->isHidden());
    EXPECT_EQ(tab.count_spin()->value(), 3);
    EXPECT_EQ(tab.worst_table()->rowCount(), 3);
    EXPECT_TRUE(told.empty());  // Settings' own count: not told back

    tab.count_spin()->setValue(5);
    EXPECT_EQ(tab.worst_table()->rowCount(), 5);
    EXPECT_EQ(told, std::vector<int>{5});

    tab.count_combo()->setCurrentIndex(0);  // 10 dips
    EXPECT_TRUE(tab.count_spin()->isHidden());
    EXPECT_EQ(tab.worst_table()->rowCount(), 8);
    EXPECT_EQ(told, (std::vector<int>{5, 10}));

    tab.count_combo()->setCurrentIndex(tab.count_combo()->count() - 1);  // Custom..., from 10
    EXPECT_FALSE(tab.count_spin()->isHidden());
    EXPECT_EQ(tab.count_spin()->value(), 10);
    EXPECT_EQ(told.size(), 2U);  // nothing changed until a number is given

    tab.set_worst_count(5000);  // past the most
    EXPECT_EQ(tab.count_spin()->value(), most_worst_frames);
}

// The graph's ⓘ opens what it shows: the metric's line, a worst frame's
// dot, a weak segment's shading, the segment selected, the frame selected;
// for a metric where lower is better, that its graph is upside down.
TEST(WorstFramesTab, TheGraphsLegendSaysWhatItShows) {
    std::vector<float> values(90, 95.0F);
    values[10] = 40.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    ASSERT_NE(tab.legend_button(), nullptr);
    EXPECT_FALSE(tab.legend_button()->icon().isNull());
    tab.legend_button()->click();  // opened on its press, as a button's menu is
    auto* legend = tab.findChild<GraphLegend*>();
    ASSERT_NE(legend, nullptr);
    EXPECT_TRUE(legend->isVisible());
    const QStringList texts = legend->texts();
    ASSERT_EQ(texts.size(), 5);
    EXPECT_EQ(texts[0], "VMAF v0.6.1 of each frame of the video.");
    EXPECT_TRUE(texts[1].startsWith("A dot"));
    legend->close();

    const GraphLegend upside_down("Butteraugli", true, false);
    ASSERT_EQ(upside_down.texts().size(), 6);
    EXPECT_TRUE(upside_down.texts().back().startsWith("Lower is better for Butteraugli"));
    EXPECT_EQ(GraphLegend("CVVDP", false, true).texts().front(), "CVVDP of each second of the video.");
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

    // Another number of dips lists them again, and the row selected is
    // selected again (#225): Details tells it, as before. A row no longer
    // listed is checked in DetailsForgetsARowNoLongerListed.
    tab.segments_table()->clearSelection();
    tab.worst_table()->selectRow(0);
    ASSERT_TRUE(tab.details()->text().startsWith("Frame 10 "));
    QComboBox* dips = nullptr;
    for (QComboBox* combo : tab.findChildren<QComboBox*>()) {
        if (combo->count() > 0 && combo->itemData(0).toInt() == worst_frame_counts.front()) {
            dips = combo;
        }
    }
    ASSERT_NE(dips, nullptr);
    dips->setCurrentIndex(1);
    EXPECT_TRUE(tab.details()->text().startsWith("Frame 10 ")) << tab.details()->text().toStdString();
    tab.set_worst_count(worst_frame_counts.back());
    EXPECT_TRUE(tab.details()->text().startsWith("Frame 10 ")) << tab.details()->text().toStdString();
}

TEST(WorstFramesTab, DetailsForgetsARowNoLongerListed) {
    // Twelve dips, a second apart: the 12th worst is listed among 25, not
    // among 10. Selected, then the number of dips lowered (the box, or
    // set_worst_count), it's gone, and Details no longer tells it.
    std::vector<float> values(400, 95.0F);
    for (int i = 0; i < 12; ++i) {
        values[static_cast<std::size_t>(5 + i * 32)] = 30.0F + static_cast<float>(i);
    }
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", values)}});
    QComboBox* dips = nullptr;
    for (QComboBox* combo : tab.findChildren<QComboBox*>()) {
        if (combo->count() > 0 && combo->itemData(0).toInt() == worst_frame_counts.front()) {
            dips = combo;
        }
    }
    ASSERT_NE(dips, nullptr);
    QTableWidget& worst = *tab.worst_table();
    dips->setCurrentIndex(1);  // 25
    ASSERT_EQ(worst.rowCount(), 12);
    worst.selectRow(11);  // frame 357, the 12th worst
    ASSERT_TRUE(tab.details()->text().startsWith("Frame 357 ")) << tab.details()->text().toStdString();
    dips->setCurrentIndex(0);  // 10
    ASSERT_EQ(worst.rowCount(), 10);
    EXPECT_TRUE(worst.selectedItems().isEmpty());
    EXPECT_TRUE(tab.details()->text().isEmpty()) << tab.details()->text().toStdString();

    tab.set_worst_count(worst_frame_counts[1]);
    worst.selectRow(11);
    ASSERT_TRUE(tab.details()->text().startsWith("Frame 357 ")) << tab.details()->text().toStdString();
    tab.set_worst_count(worst_frame_counts.front());
    EXPECT_TRUE(worst.selectedItems().isEmpty());
    EXPECT_TRUE(tab.details()->text().isEmpty()) << tab.details()->text().toStdString();
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

TEST(WorstFramesTab, TakesTheProfilesConditionsOnTheMetricsOfTheVideoPicked) {
    // The profile's condition on a metric the first video lacks is set
    // aside for it; picked next, a video that has the metric is judged by
    // it too, as the exports and the command line judge it, and saving
    // keeps it once.
    Settings settings;
    settings.weak_segment_conditions = {"vmaf<90", "ssimulacra2<65"};
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    auto b = std::make_shared<ComparisonResult>(*vmaf_result("b.mkv", std::vector<float>(30, 95.0F)));
    FrameMetricResult ssimulacra2{"ssimulacra2", {}, {}, {}, current_ffmpeg_provenance("ssimulacra2", "9.0.1")};
    for (int i = 0; i < 30; ++i) {
        ssimulacra2.frame.push_back(i);
        ssimulacra2.time.push_back(i / 30.0);
        ssimulacra2.values.push_back(i >= 10 && i < 15 ? 50.0F : 80.0F);
    }
    b->metric_results.add(ssimulacra2);
    tab.set_videos({{"row:a", "a", vmaf_result("a.mkv", std::vector<float>(30, 95.0F))}, {"row:b", "b", b}});
    EXPECT_EQ(condition_texts(tab.rules().conditions), (std::vector<std::string>{"vmaf<90"}));

    tab.videos_table()->setCurrentCell(1, 0);
    std::vector<std::string> shown = condition_texts(tab.rules().conditions);
    std::ranges::sort(shown);
    EXPECT_EQ(shown, (std::vector<std::string>{"ssimulacra2<65", "vmaf<90"}));
    ASSERT_EQ(tab.segments().size(), 1U);
    EXPECT_EQ(tab.segments()[0].first_frame, 10);
    EXPECT_FALSE(tab.save_button()->isEnabled());  // as saved

    bench.threshold()->setValue(bench.threshold()->value() - 1);  // a change, then saved
    ASSERT_TRUE(tab.save_button()->isEnabled());
    tab.save_button()->click();
    ASSERT_EQ(bench.saved.size(), 1U);
    EXPECT_EQ(bench.saved[0].second.conditions.size(), 2U);  // each condition once
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

TEST(WorstFramesTab, KeepsTheChosenVideoWhenItIsScoredAgain) {
    // A video scored again comes back as a new line (a new identity, its
    // file the same): it stays chosen, and the metrics it now has are in
    // the bar. So does a run arriving in pieces, which keeps its identity.
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    const WeakFramesVideo a{"run:1", "a", vmaf_result("a.mkv", {95, 40, 95})};
    tab.set_videos({a, {"run:2", "b", vmaf_result("b.mkv", {95, 50, 95})}});
    tab.videos_table()->setCurrentCell(1, 0);
    ASSERT_EQ(tab.chosen_video()->identity, "run:2");

    // Scored again with SSIMULACRA2 too: a new line of the same file.
    auto again = std::make_shared<ComparisonResult>(*vmaf_result("b.mkv", {95, 55, 95}));
    FrameMetricResult ssimulacra2{"ssimulacra2", {0, 1, 2}, {0.0, 1.0 / 30, 2.0 / 30}, {80, 30, 80},
                                  current_ffmpeg_provenance("ssimulacra2", "9.0.1")};
    again->metric_results.add(ssimulacra2);
    tab.set_videos({a, {"run:3", "b", again}});
    ASSERT_NE(tab.chosen_video(), nullptr);
    EXPECT_EQ(tab.chosen_video()->identity, "run:3");
    EXPECT_EQ(tab.videos_table()->currentRow(), 1);
    EXPECT_EQ(tab.metric_bar()->count(), 2);
    EXPECT_EQ(tab.worst_table()->item(0, 2)->text(), "55.00");

    // The same line with a metric dropped: the bar loses it.
    tab.metric_bar()->setCurrentIndex(1);  // SSIMULACRA2
    ASSERT_EQ(std::string(tab.shown_metric()->key), "ssimulacra2");
    tab.set_videos({a, {"run:3", "b", vmaf_result("b.mkv", {95, 55, 95})}});
    EXPECT_EQ(tab.metric_bar()->count(), 1);
    ASSERT_NE(tab.shown_metric(), nullptr);
    EXPECT_EQ(std::string(tab.shown_metric()->key), "vmaf");
    EXPECT_EQ(tab.worst_table()->item(0, 2)->text(), "55.00");

    // A line of the same file and reference beside it (a CVVDP display row
    // of its) is not taken for it: not while its line is away being scored
    // again, and not once it is back (#221's review: the sibling was).
    const WeakFramesVideo sibling{"run:5", "b (display)", vmaf_result("b.mkv", {95, 70, 95})};
    tab.set_videos({a, {"run:3", "b", vmaf_result("b.mkv", {95, 55, 95})}, sibling});
    tab.videos_table()->setCurrentCell(1, 0);
    ASSERT_EQ(tab.chosen_video()->identity, "run:3");
    tab.set_videos({a, sibling});  // being scored again
    ASSERT_NE(tab.chosen_video(), nullptr);
    EXPECT_NE(tab.chosen_video()->identity, "run:5");
    tab.set_videos({a, {"run:6", "b", vmaf_result("b.mkv", {95, 65, 95})}, sibling});
    ASSERT_NE(tab.chosen_video(), nullptr);
    EXPECT_EQ(tab.chosen_video()->identity, "run:6");
}

// The row opened in Video Compare is the one selected on the way back,
// though the window fills the lists again whenever the tab is opened; after
// Previous / Next worst there, the frame Video Compare stepped to.
TEST(WorstFramesTab, KeepsTheRowOpenedInVideoCompare) {
    std::vector<float> values(90, 95.0F);
    values[10] = 40.0F;
    values[50] = 45.0F;
    values[89] = 50.0F;
    Settings settings;
    Bench bench(settings);
    WorstFramesTab& tab = bench.tab;
    const auto selected = [](const QTableWidget& table) {
        const QList<QTableWidgetItem*> items = table.selectedItems();
        return items.isEmpty() ? -1 : items.front()->row();
    };
    const WeakFramesVideo a{"row:a", "a", vmaf_result("a.mkv", values)};
    tab.set_videos({a});
    QTableWidget& worst = *tab.worst_table();
    ASSERT_EQ(worst.rowCount(), 3);
    worst.selectRow(1);
    emit worst.cellDoubleClicked(1, 0);  // frame 50
    tab.set_videos({a});  // back on the tab
    EXPECT_EQ(selected(worst), 1);

    // Next worst in Video Compare: frame 89, its row.
    const QString vmaf = tab.shown_list().metric;
    tab.keep_worst_frame("row:a", vmaf, 89);
    EXPECT_EQ(selected(worst), 2);
    tab.set_videos({a});
    EXPECT_EQ(selected(worst), 2);
    // Another video's list, or another metric's, leaves it as it is.
    tab.keep_worst_frame("row:b", vmaf, 10);
    tab.keep_worst_frame("row:a", "PSNR", 10);
    EXPECT_EQ(selected(worst), 2);
    tab.set_videos({a});
    EXPECT_EQ(selected(worst), 2);

    // A segment: its row, and none in Worst frames.
    QTableWidget& segments = *tab.segments_table();
    segments.selectRow(1);
    tab.set_videos({a});
    EXPECT_EQ(selected(segments), 1);
    EXPECT_EQ(selected(worst), -1);

    // Nothing selected by the user: nothing selected again.
    segments.clearSelection();
    tab.set_videos({a});
    EXPECT_EQ(selected(segments), -1);
    EXPECT_EQ(selected(worst), -1);
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
