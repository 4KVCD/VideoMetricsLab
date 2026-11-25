// The BD-rate tab (bd_rate_panel.h): its encodes from CSV files and scored
// videos, its table, chart and exports. The numbers themselves are the
// command line's, tested there (cli_test.cpp, bd_rate_test.cpp).
#include "bd_rate_panel.h"

#include <QComboBox>
#include <QFile>
#include <QLineEdit>
#include <QMenu>
#include <QPushButton>
#include <QTableWidget>
#include <QTemporaryDir>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "rate_curve_chart.h"
#include "theme.h"
#include "vml/platform/system.h"

namespace vml::app {
namespace {

// A CSV file in `folder` with `text`; its path.
QString write_csv(const QTemporaryDir& folder, const QString& name, const QByteArray& text) {
    const QString path = folder.filePath(name);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    file.write(text);
    return path;
}

// A scored comparison of a video at `bits_per_second` against `reference`.
GraphPanel::Run scored_run(const std::string& reference, long long bits_per_second, bool whole_file,
                           const std::string& identity) {
    auto result = std::make_shared<ComparisonResult>();
    result->source = reference;
    result->distorted = identity + ".mkv";
    result->distorted_info.bit_rate = bits_per_second;
    result->distorted_info.bit_rate_whole_file = whole_file;
    result->metric_results.add(FrameMetricResult{"vmaf", {0}, {0.0}, {90.0F}, {}});
    return {result, QString(), identity};
}

// The anchor's and the test's CSV files: s1's test at half the rate (-50 %),
// s2 the same but for a PSNR not scored (n/a); every other metric has no
// column (n/a); the table, its colours and tooltips, the chart of s1 (the
// first sequence), Copy as markdown and Export CSV as the command line's
// text; then scored videos added (one without a bitrate refused, one with
// a whole file's rate warned of), and Clear.
// VML Modern's roles: the options a toolbar, the results and the chart each
// a card, the heading the results' title, each side's Clear quiet.
TEST(BdRatePanel, ItsPartsHaveTheirRoles) {
    BdRatePanel panel([] { return std::vector<GraphPanel::Run>{}; });
    const auto role = [](const QWidget* widget) { return widget->property("vmlRole").toString(); };
    EXPECT_EQ(role(panel.group_combo->parentWidget()), "toolbar");
    EXPECT_EQ(role(panel.results->parentWidget()), "card");
    EXPECT_EQ(role(panel.heading_label->parentWidget()), "card");
    EXPECT_EQ(role(panel.chart->parentWidget()), "card");
    EXPECT_EQ(role(panel.anchor.clear_button), "quiet");
    EXPECT_EQ(role(panel.test.clear_button), "quiet");
    EXPECT_EQ(role(panel.heading_label), "title");
}

TEST(BdRatePanel, ShowsTheBdRatesOfTwoEncoders) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    std::vector<GraphPanel::Run> runs{scored_run("refs/baseball.y4m", 6551340, false, "q27"),
                                      scored_run("refs/baseball.y4m", 0, false, "q32"),
                                      scored_run("refs/baseball.y4m", 2000000, true, "q37")};
    BdRatePanel panel([&runs] { return runs; });
    QStringList informed;
    panel.dialogs.inform = [&informed](const QString& title, const QString&) { informed.append(title); };
    panel.dialogs.alert = [](const QString& title, const QString& text) { ADD_FAILURE() << title.toStdString() << text.toStdString(); };
    QString copied;
    panel.copy_text = [&copied](const QString& text) { copied = text; };
    EXPECT_FALSE(panel.report());
    EXPECT_FALSE(panel.copy_button->isEnabled());

    panel.import_file(panel.anchor, write_csv(folder, "fast.csv",
                                             "sequence,kbps,vmaf,psnr\ns1,1000,30,30\ns1,2000,33,33\ns1,4000,36,36\n"
                                             "s1,8000,39,39\ns2,1000,30,30\ns2,2000,33,33\ns2,4000,36,36\n"
                                             "s2,8000,39,39\n"));
    panel.import_file(panel.test, write_csv(folder, "slow.csv",
                                           "sequence,kbps,vmaf,psnr\ns1,500,30,30\ns1,1000,33,33\ns1,2000,36,36\n"
                                           "s1,4000,39,39\ns2,1000,30,30\ns2,2000,33,\ns2,4000,36,36\n"
                                           "s2,8000,39,39\n"));
    ASSERT_TRUE(panel.report());
    EXPECT_EQ(panel.anchor.name_edit->text(), "fast");
    EXPECT_EQ(panel.anchor.sequences->rowCount(), 2);
    EXPECT_EQ(panel.heading_label->text(),
              "slow against fast. Sequences in both: 2. Negative: less bitrate for the same score.");
    // A column per metric, the planes none of these scores (hidden); mean,
    // median, s1, s2
    int shown_columns = 0;
    for (int column = 0; column < panel.results->columnCount(); ++column) {
        shown_columns += panel.results->isColumnHidden(column) ? 0 : 1;
    }
    ASSERT_EQ(shown_columns, 20);
    EXPECT_EQ(panel.results->horizontalHeaderItem(1)->text(), "VMAF v0.6.1");
    ASSERT_EQ(panel.results->rowCount(), 4);
    const QTableWidgetItem* s1_vmaf = panel.results->item(2, 1);
    EXPECT_EQ(panel.results->item(2, 0)->text(), "s1");
    EXPECT_EQ(s1_vmaf->text(), QString::fromUtf8("\xE2\x88\x92" "50.0%"));
    EXPECT_EQ(s1_vmaf->foreground().color(), theme::color("good"));
    EXPECT_EQ(panel.results->item(3, 4)->text(), "n/a");  // s2's PSNR
    EXPECT_EQ(panel.results->item(3, 4)->toolTip(), "not scored at every encode");
    EXPECT_EQ(panel.results->item(0, 2)->toolTip(), "No sequence has a BD-rate for this metric.");  // VMAF NEG
    EXPECT_EQ(panel.results->item(1, 1)->text(), QString::fromUtf8("\xE2\x88\x92" "25.0%"));  // the median

    // The chart: s1, VMAF, the anchor's four encodes and the test's, the
    // scores both cover shaded
    ASSERT_TRUE(panel.chart->plot());
    EXPECT_EQ(panel.chart->plot()->anchor.points.size(), 4u);
    EXPECT_EQ(panel.chart->plot()->test.name, "slow");
    EXPECT_EQ(panel.chart->plot()->label, QString::fromUtf8("VMAF v0.6.1: \xE2\x88\x92" "50.0%"));
    EXPECT_EQ(panel.chart->shared_scores(), (std::pair{30.0, 39.0}));
    panel.results->setCurrentCell(3, 0);
    panel.metric_combo->setCurrentIndex(panel.metric_combo->findData("psnr"));
    EXPECT_EQ(panel.chart->plot()->label, "PSNR: n/a (not scored at every encode)");

    panel.test.name_edit->setText("slower");  // the heading, the legend and the copy follow
    EXPECT_TRUE(panel.heading_label->text().startsWith("slower against fast."));
    EXPECT_EQ(panel.chart->plot()->test.name, "slower");
    panel.copy_button->click();
    EXPECT_EQ(copied.toStdString(), bd_rate_table(*panel.report(), "fast", "slower"));
    const QString exported = folder.filePath("bd.csv");
    panel.dialogs.save_file = [&exported](const QString&, const QString&, const QString&) { return exported; };
    // The table card's save icon: Save as... on a right-click, JSON first.
    QMenu items;
    panel.table_save->save_as(items);
    ASSERT_EQ(items.actions().size(), 2);
    EXPECT_EQ(items.actions()[0]->text().toStdString(), "Table as JSON...");
    EXPECT_EQ(items.actions()[1]->text().toStdString(), "Table as CSV...");
    items.actions()[1]->trigger();
    QFile file(exported);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll().toStdString(), bd_rate_csv(*panel.report()));
    // At one click, in the export folder: the table as JSON (the command
    // line's bdrate --json, the encoders named) and CSV; the chart as PNG.
    // No background runner: the files are written at once.
    panel.quick_saves = {[&folder] { return folder.path(); }, [] { return std::vector<std::string>{"json", "csv"}; },
                         QuickSaveRunner{}};
    std::vector<ExportOutcome> told;
    panel.export_report = [&told](const ExportOutcome& outcome) { told.push_back(outcome); };
    panel.table_save->refresh();  // as the pointer coming to it does
    panel.table_save->click();
    panel.chart_save->click();
    ASSERT_EQ(told.size(), 2U);
    EXPECT_EQ(told[0].files.size(), 2) << told[0].error.toStdString();
    EXPECT_EQ(told[1].files.size(), 1) << told[1].error.toStdString();
    QFile json(folder.filePath("slower vs fast - BD-rate.json"));
    ASSERT_TRUE(json.open(QIODevice::ReadOnly));
    const auto document = nlohmann::json::parse(json.readAll().toStdString());
    EXPECT_EQ(document["format"], "bdrate-1");
    EXPECT_EQ(document["anchor"], "fast");
    EXPECT_EQ(document["test"], "slower");
    EXPECT_TRUE(QFile::exists(folder.filePath("slower vs fast - BD-rate.csv")));
    EXPECT_TRUE(QFile::exists(folder.filePath("slower vs fast - BD-rate PSNR.png")));

    // Scored videos: the one without a bitrate refused, the whole file's
    // rate among the warnings
    panel.choose_videos = [](const QStringList& items) {
        EXPECT_EQ(items.size(), 3);
        EXPECT_EQ(items[0], "q27.mkv (reference baseball.y4m, 6551 kb/s)");
        return std::vector<int>{0, 1, 2};
    };
    panel.anchor.add_videos_button->click();
    EXPECT_EQ(informed, QStringList{"Bitrate unknown"});
    EXPECT_EQ(panel.anchor.sources.size(), 3u);
    EXPECT_EQ(panel.anchor.sources.back().second.encodes.front().sequence, "baseball");
    EXPECT_EQ(panel.anchor.sources.back().second.encodes.front().kbps, 2000);
    EXPECT_TRUE(panel.warnings().contains("q37.mkv: its bitrate is the whole file's, soundtrack included"));
    EXPECT_TRUE(panel.warnings().contains("baseball: only in the anchor, left out"));
    panel.anchor.add_videos_button->click();  // the same videos again: replaced, not added twice
    EXPECT_EQ(panel.anchor.sources.size(), 3u);

    panel.test.clear_button->click();
    EXPECT_FALSE(panel.report());
    EXPECT_EQ(panel.results->rowCount(), 0);
    EXPECT_FALSE(panel.chart->plot());
    EXPECT_TRUE(panel.test.name_edit->text().isEmpty());  // Clear forgot the name too
}

// A line scored again gives the tab its new scores; a line gone from the list
// keeps those it had; one that lost its bitrate leaves, and that is said.
TEST(BdRatePanel, ScoredVideosFollowTheirLines) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    std::vector<GraphPanel::Run> runs{scored_run("refs/clip.y4m", 2000000, false, "q27"),
                                      scored_run("refs/clip.y4m", 1000000, false, "q32")};
    BdRatePanel panel([&runs] { return runs; });
    panel.choose_videos = [](const QStringList& items) {
        std::vector<int> all(static_cast<std::size_t>(items.size()));
        std::iota(all.begin(), all.end(), 0);
        return all;
    };
    QStringList informed;
    panel.dialogs.inform = [&informed](const QString& title, const QString&) { informed.append(title); };
    panel.anchor.add_videos_button->click();
    const auto rate_of = [&panel](const std::size_t source) {
        return panel.anchor.sources.at(source).second.encodes.front().kbps;
    };
    ASSERT_EQ(panel.anchor.sources.size(), 2u);
    EXPECT_EQ(rate_of(0), 2000);

    runs[0] = scored_run("refs/clip.y4m", 3000000, false, "q27");
    panel.bring_videos_up_to_date();
    EXPECT_EQ(rate_of(0), 3000);

    runs.erase(runs.begin());  // q27 no longer listed
    runs[0] = scored_run("refs/clip.y4m", 0, false, "q32");
    panel.bring_videos_up_to_date();
    ASSERT_EQ(panel.anchor.sources.size(), 1u);
    EXPECT_EQ(rate_of(0), 3000);
    EXPECT_EQ(informed, QStringList{"Bitrate unknown"});
}

// #205: a video scored again gets a line of another identity; the side kept
// the old line's scores, as for a video taken out of the session, and Add
// videos then added the file a second time (two points at one bitrate). The
// video added follows the one new line of its file.
TEST(BdRatePanel, AVideoScoredAgainKeepsItsPlace) {
    const auto run_of = [](const std::string& identity, long long bits_per_second) {
        GraphPanel::Run run = scored_run("refs/clip.y4m", bits_per_second, false, identity);
        std::const_pointer_cast<ComparisonResult>(run.result)->distorted = "encodes/q27.mkv";
        return run;
    };
    std::vector<GraphPanel::Run> runs{run_of("run:1", 2000000)};
    BdRatePanel panel([&runs] { return runs; });
    panel.choose_videos = [](const QStringList& items) {
        std::vector<int> all(static_cast<std::size_t>(items.size()));
        std::iota(all.begin(), all.end(), 0);
        return all;
    };
    panel.anchor.add_videos_button->click();
    ASSERT_EQ(panel.anchor.sources.size(), 1u);

    runs[0] = run_of("run:2", 3000000);  // scored again
    panel.bring_videos_up_to_date();
    ASSERT_EQ(panel.anchor.sources.size(), 1u);
    EXPECT_EQ(panel.anchor.sources.front().second.encodes.front().kbps, 3000);
    panel.anchor.add_videos_button->click();
    EXPECT_EQ(panel.anchor.sources.size(), 1u);

    // Two new lines of its file (scored again with other settings, as two
    // rows): which one it was is not known, and it stays as it was.
    runs = {run_of("run:3", 4000000), run_of("run:4", 5000000)};
    panel.bring_videos_up_to_date();
    ASSERT_EQ(panel.anchor.sources.size(), 1u);
    EXPECT_EQ(panel.anchor.sources.front().second.encodes.front().kbps, 3000);
}

// Sequences go by the reference's file name: two files of one name are said.
TEST(BdRatePanel, WarnsOfTwoReferencesOfOneName) {
    std::vector<GraphPanel::Run> runs{scored_run("one/clip.y4m", 2000000, false, "a"),
                                      scored_run("two/clip.y4m", 1000000, false, "b")};
    int asked = 0;
    BdRatePanel panel([&runs] { return runs; });
    panel.choose_videos = [&asked](const QStringList&) { return std::vector<int>{asked++}; };
    panel.anchor.add_videos_button->click();
    panel.test.add_videos_button->click();
    const QString warning = "clip: its encodes have different references, " +
                            QString::fromStdString(platform::path_text(std::filesystem::path("one/clip.y4m"))) + ", " +
                            QString::fromStdString(platform::path_text(std::filesystem::path("two/clip.y4m")));
    EXPECT_TRUE(panel.warnings().contains(warning)) << panel.warnings().join('|').toStdString();
}

// The chart with encodes the BD-rate could not use: a rate of 0 is not
// drawn, and a side with an encode not drawn gets dots but no curve (nor a
// shaded range); it is painted all the same. Then no sequence in common:
// the warnings still name those left out.
TEST(BdRatePanel, DrawsOnlyWhatTheBdRateUsed) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    BdRatePanel panel([] { return std::vector<GraphPanel::Run>{}; });
    panel.dialogs.alert = [](const QString& title, const QString& text) {
        ADD_FAILURE() << title.toStdString() << text.toStdString();
    };
    panel.import_file(panel.anchor, write_csv(folder, "a.csv",
                                             "sequence,kbps,vmaf,psnr\ns1,1000,30,30\ns1,2000,33,\ns1,0,36,36\n"
                                             "s1,8000,39,39\n"));
    panel.import_file(panel.test, write_csv(folder, "b.csv",
                                           "sequence,kbps,vmaf,psnr\ns1,500,30,30\ns1,1000,33,33\ns1,2000,36,36\n"
                                           "s1,4000,39,39\n"));
    ASSERT_TRUE(panel.chart->plot());
    EXPECT_EQ(panel.chart->plot()->anchor.points.size(), 3u);  // not the rate of 0
    EXPECT_TRUE(panel.chart->plot()->anchor.line.empty());
    EXPECT_EQ(panel.chart->plot()->test.line.size(), 96u);
    EXPECT_FALSE(panel.chart->shared_scores());
    EXPECT_EQ(panel.chart->plot()->label, "VMAF v0.6.1: n/a (a rate not above 0, or a number not finite)");
    panel.chart->resize(640, 300);
    EXPECT_FALSE(panel.chart->grab().isNull());
    panel.metric_combo->setCurrentIndex(panel.metric_combo->findData("xpsnr"));  // no encode has one
    EXPECT_TRUE(panel.chart->plot()->anchor.points.empty());
    EXPECT_FALSE(panel.chart->grab().isNull());

    panel.test.clear_button->click();
    panel.import_file(panel.test, write_csv(folder, "c.csv", "sequence,kbps,vmaf\nt1,500,30\n"));
    EXPECT_EQ(panel.test.name_edit->text(), "c");  // the new file's name, not the cleared one's
    EXPECT_FALSE(panel.report());
    EXPECT_EQ(panel.heading_label->text(), "the anchor and the test have no sequence in common");
    EXPECT_TRUE(panel.warnings().contains("s1: only in the anchor, left out"));
    EXPECT_TRUE(panel.warnings().contains("t1: only in the test, left out"));
}

// A side's encodes from a JSON file too: results as the command line's
// --json writes them, under the button's name for both.
TEST(BdRatePanel, ImportsEncodesFromJson) {
    QTemporaryDir folder;
    ASSERT_TRUE(folder.isValid());
    BdRatePanel panel([] { return std::vector<GraphPanel::Run>{}; });
    panel.dialogs.alert = [](const QString& title, const QString& text) {
        ADD_FAILURE() << title.toStdString() << text.toStdString();
    };
    EXPECT_EQ(panel.anchor.import_button->text().toStdString(), "Import CSV/JSON\xE2\x80\xA6");
    QString filter;
    panel.dialogs.open_file = [&filter](const QString&, const QString& asked) {
        filter = asked;
        return QString();
    };
    panel.anchor.import_button->click();
    EXPECT_EQ(filter.toStdString(), "CSV or JSON file (*.csv *.json)");
    panel.import_file(panel.anchor,
                      write_csv(folder, "x264.json",
                                R"({"format": 2, "videos": [
                                    {"reference": "Y:/clips/s1.y4m", "bit_rate": 1000000, "metrics": {"vmaf": {"score": 30}}},
                                    {"reference": "Y:/clips/s1.y4m", "bit_rate": 2000000, "metrics": {"vmaf": {"score": 33}}}]})"));
    EXPECT_EQ(panel.anchor.name_edit->text(), "x264");
    ASSERT_EQ(panel.anchor.sequences->rowCount(), 1);
    EXPECT_EQ(panel.anchor.sequences->item(0, 0)->text(), "s1");
}

}  // namespace
}  // namespace vml::app
