// The BD-rate tab (bd_rate_panel.h): its encodes from CSV files and scored
// videos, its table, chart and exports. The numbers themselves are the
// command line's, tested there (cli_test.cpp, bd_rate_test.cpp).
#include "bd_rate_panel.h"

#include <QComboBox>
#include <QFile>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rate_curve_chart.h"
#include "theme.h"

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

    panel.import_csv(panel.anchor, write_csv(folder, "fast.csv",
                                             "sequence,kbps,vmaf,psnr\ns1,1000,30,30\ns1,2000,33,33\ns1,4000,36,36\n"
                                             "s1,8000,39,39\ns2,1000,30,30\ns2,2000,33,33\ns2,4000,36,36\n"
                                             "s2,8000,39,39\n"));
    panel.import_csv(panel.test, write_csv(folder, "slow.csv",
                                           "sequence,kbps,vmaf,psnr\ns1,500,30,30\ns1,1000,33,33\ns1,2000,36,36\n"
                                           "s1,4000,39,39\ns2,1000,30,30\ns2,2000,33,\ns2,4000,36,36\n"
                                           "s2,8000,39,39\n"));
    ASSERT_TRUE(panel.report());
    EXPECT_EQ(panel.anchor.name_edit->text(), "fast");
    EXPECT_EQ(panel.anchor.sequences->rowCount(), 2);
    EXPECT_EQ(panel.heading_label->text(),
              "slow against fast. Sequences in both: 2. Negative: less bitrate for the same score.");
    // A column per metric; mean, median, s1, s2
    ASSERT_EQ(panel.results->columnCount(), 10);
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
    panel.export_button->click();
    QFile file(exported);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.readAll().toStdString(), bd_rate_csv(*panel.report()));

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
    panel.import_csv(panel.anchor, write_csv(folder, "a.csv",
                                             "sequence,kbps,vmaf,psnr\ns1,1000,30,30\ns1,2000,33,\ns1,0,36,36\n"
                                             "s1,8000,39,39\n"));
    panel.import_csv(panel.test, write_csv(folder, "b.csv",
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
    panel.import_csv(panel.test, write_csv(folder, "c.csv", "sequence,kbps,vmaf\nt1,500,30\n"));
    EXPECT_EQ(panel.test.name_edit->text(), "c");  // the new file's name, not the cleared one's
    EXPECT_FALSE(panel.report());
    EXPECT_EQ(panel.heading_label->text(), "the anchor and the test have no sequence in common");
    EXPECT_TRUE(panel.warnings().contains("s1: only in the anchor, left out"));
    EXPECT_TRUE(panel.warnings().contains("t1: only in the test, left out"));
}

}  // namespace
}  // namespace vml::app
